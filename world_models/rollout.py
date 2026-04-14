"""Autoregressive rollout over a frozen I2V model.

state = image, action = text prompt, next_state = last frame of generated clip.
Generates N steps, saves each clip and its last frame, concatenates all clips
into a single rollout.mp4 for inspection.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import requests
from PIL import Image

HERE = Path(__file__).parent
DEFAULT_START = HERE / "assets" / "start_frame.png"
RUNS = HERE / "runs"

DEFAULT_ACTIONS = [
    "the red ball rolls slowly to the right across the beige table",
    "the red ball continues rolling to the right across the beige table",
    "the red ball slows down and comes to rest on the beige table",
    "the camera pushes in slowly toward the red ball on the beige table",
    "the red ball rolls forward toward the camera across the beige table",
]


def post_video(base_url: str, prompt: str, image_path: Path, seed: int) -> str:
    with image_path.open("rb") as f:
        r = requests.post(
            f"{base_url}/v1/videos",
            files={"input_reference": (image_path.name, f, "image/png")},
            data={
                "prompt": prompt,
                "size": "832x480",
                "num_frames": "33",
                "fps": "24",
                "num_inference_steps": "30",
                "guidance_scale": "6.0",
                "flow_shift": "5.0",
                "seed": str(seed),
            },
            timeout=60,
        )
    r.raise_for_status()
    return r.json()["id"]


def wait_for_video(base_url: str, video_id: str, poll_s: float = 3.0) -> dict:
    url = f"{base_url}/v1/videos/{video_id}"
    while True:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        body = r.json()
        status = body["status"]
        if status == "completed":
            return body
        if status == "failed":
            raise RuntimeError(f"video {video_id} failed: {body.get('error')}")
        time.sleep(poll_s)


def download_video(base_url: str, video_id: str, out_path: Path) -> None:
    r = requests.get(f"{base_url}/v1/videos/{video_id}/content", timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def extract_last_frame(video_path: Path, out_path: Path) -> None:
    frames = iio.imread(video_path, plugin="FFMPEG")
    last = frames[-1]
    Image.fromarray(last).save(out_path)


def concat_clips(clip_paths: list[Path], out_path: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    list_file = out_path.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in clip_paths)
    )
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(out_path)],
        check=True, capture_output=True,
    )
    list_file.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8099")
    ap.add_argument("--start-frame", type=Path, default=DEFAULT_START)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--actions", nargs="+", default=None,
                    help="Override action sequence (one prompt per step)")
    args = ap.parse_args()

    actions = args.actions or DEFAULT_ACTIONS
    if len(actions) < args.steps:
        raise SystemExit(
            f"need at least {args.steps} actions, got {len(actions)}"
        )
    actions = actions[: args.steps]

    run_dir = RUNS / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = run_dir / "frames"
    clips_dir = run_dir / "clips"
    frames_dir.mkdir()
    clips_dir.mkdir()

    initial = frames_dir / "frame_000.png"
    shutil.copy(args.start_frame, initial)
    current_frame = initial
    clip_paths: list[Path] = []
    log: list[dict] = []

    print(f"[rollout] run dir: {run_dir}")
    for step, prompt in enumerate(actions):
        print(f"[rollout] step {step}: {prompt!r}")
        t0 = time.time()
        video_id = post_video(args.base_url, prompt, current_frame, args.seed + step)
        body = wait_for_video(args.base_url, video_id)
        clip_path = clips_dir / f"step_{step:02d}.mp4"
        download_video(args.base_url, video_id, clip_path)
        next_frame = frames_dir / f"frame_{step + 1:03d}.png"
        extract_last_frame(clip_path, next_frame)
        dt = time.time() - t0
        print(
            f"[rollout]   done in {dt:.1f}s "
            f"(inference {body['inference_time_s']:.1f}s, "
            f"peak {body['peak_memory_mb']:.0f} MB)"
        )
        log.append({
            "step": step,
            "prompt": prompt,
            "video_id": video_id,
            "clip": str(clip_path.relative_to(run_dir)),
            "next_frame": str(next_frame.relative_to(run_dir)),
            "wall_s": round(dt, 2),
            "inference_s": round(body["inference_time_s"], 2),
            "peak_memory_mb": body["peak_memory_mb"],
        })
        clip_paths.append(clip_path)
        current_frame = next_frame

    rollout_path = run_dir / "rollout.mp4"
    concat_clips(clip_paths, rollout_path)
    (run_dir / "metadata.json").write_text(json.dumps({
        "steps": args.steps,
        "seed": args.seed,
        "start_frame": str(args.start_frame),
        "actions": actions,
        "log": log,
    }, indent=2))
    print(f"[rollout] wrote {rollout_path}")


if __name__ == "__main__":
    main()
