"""Shared helpers for phase scripts: talking to the vllm-omni video API."""
from __future__ import annotations

import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8099"


def submit_video(
    prompt: str,
    image_path: Path,
    *,
    base_url: str = BASE_URL,
    size: str = "832x480",
    num_frames: int = 33,
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
    flow_shift: float = 5.0,
    seed: int = 42,
) -> str:
    with image_path.open("rb") as f:
        r = requests.post(
            f"{base_url}/v1/videos",
            files={"input_reference": (image_path.name, f, "image/png")},
            data={
                "prompt": prompt,
                "size": size,
                "num_frames": str(num_frames),
                "fps": "24",
                "num_inference_steps": str(num_inference_steps),
                "guidance_scale": str(guidance_scale),
                "flow_shift": str(flow_shift),
                "seed": str(seed),
            },
            timeout=60,
        )
    r.raise_for_status()
    return r.json()["id"]


def poll_until_done(video_id: str, base_url: str = BASE_URL, interval_s: float = 2.0) -> dict:
    url = f"{base_url}/v1/videos/{video_id}"
    while True:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        body = r.json()
        if body["status"] == "completed":
            return body
        if body["status"] == "failed":
            raise RuntimeError(f"video {video_id} failed: {body.get('error')}")
        time.sleep(interval_s)


def run_one(
    prompt: str,
    image_path: Path,
    **kwargs,
) -> dict:
    t0 = time.time()
    vid = submit_video(prompt, image_path, **kwargs)
    body = poll_until_done(vid)
    body["_client_wall_s"] = round(time.time() - t0, 3)
    return body
