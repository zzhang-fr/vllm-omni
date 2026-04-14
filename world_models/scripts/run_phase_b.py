"""Phase B: operating-point sweep.

Vary num_inference_steps and num_frames independently, one axis at a time,
measuring per-call inference_time_s and per-stage breakdowns (scraped from
container logs). The goal is to chart the cost curve between our offline
baseline (33 frames, 30 steps) and the streaming regime that
StreamDiffusionV2 / DreamZero target (4 frames, 1-4 steps).

Writes world_models/profiles/phase_b_sweep.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import run_one  # noqa: E402
from _log_profile import group_runs, read_profile_lines  # noqa: E402

HERE = Path(__file__).parent.parent
START = HERE / "assets" / "start_frame.png"
PROFILES = HERE / "profiles"
CONTAINER = "vllm-omni-wm-server"

PROMPT = "The red ball rolls slowly to the right across the beige table."


def snapshot_latest_run(baseline_count: int) -> dict | None:
    runs = group_runs(read_profile_lines(CONTAINER))
    if len(runs) <= baseline_count:
        return None
    r = runs[-1]
    return {
        "pipeline": r.pipeline,
        "durations": {k: round(v, 4) for k, v in r.durations.items()},
        "counts": r.counts,
    }


def sweep_axis(
    axis_name: str,
    values: list[int],
    fixed: dict,
) -> list[dict]:
    results = []
    for v in values:
        kwargs = dict(fixed, **{axis_name: v})
        pre_runs = len(group_runs(read_profile_lines(CONTAINER)))
        print(f"[phase_b] {axis_name}={v}  (fixed={fixed})")
        t0 = time.time()
        body = run_one(PROMPT, START, **kwargs)
        wall = time.time() - t0
        # Wait briefly so the final profiler log line lands in the docker buffer
        time.sleep(1.0)
        stages = snapshot_latest_run(pre_runs)
        results.append({
            "axis": axis_name,
            "value": v,
            "fixed": fixed,
            "client_wall_s": round(wall, 3),
            "inference_time_s": body.get("inference_time_s"),
            "peak_memory_mb": body.get("peak_memory_mb"),
            "stages": stages,
        })
        durs = (stages or {}).get("durations", {})
        fwd = durs.get("forward", 0)
        dec = durs.get("vae.decode", 0)
        enc = durs.get("vae.encode", 0)
        txt = durs.get("text_encoder.forward", 0)
        dit = fwd - enc - dec - txt
        print(
            f"  inference={body.get('inference_time_s'):.2f}s  "
            f"DiT={dit:.2f}s  VAE-dec={dec:.2f}s  peak={body.get('peak_memory_mb')}MB"
        )
    return results


def main() -> None:
    PROFILES.mkdir(exist_ok=True)

    # Sweep 1: denoising steps (hold frames=33 = baseline)
    steps_results = sweep_axis(
        "num_inference_steps",
        [30, 16, 8, 4, 2, 1],
        fixed={"num_frames": 33},
    )

    # Sweep 2: frame count (hold steps=30 = baseline)
    frames_results = sweep_axis(
        "num_frames",
        [33, 16, 8, 4],
        fixed={"num_inference_steps": 30},
    )

    out = {
        "phase": "B",
        "model": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
        "config_constant": {
            "size": "832x480",
            "guidance_scale": 6.0,
            "flow_shift": 5.0,
            "seed": 42,
            "tp": 1,
        },
        "sweep_steps": steps_results,
        "sweep_frames": frames_results,
    }
    (PROFILES / "phase_b_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"[phase_b] wrote {PROFILES / 'phase_b_sweep.json'}")


if __name__ == "__main__":
    main()
