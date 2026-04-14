"""Phase A: baseline compute breakdown.

Runs one inference at the current baseline config, and for every successful
job it *also* scrapes the matching per-stage timings out of the Docker
container logs (workaround for the empty stage_durations API field).

Writes world_models/profiles/phase_a_<model>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import run_one  # noqa: E402
from _log_profile import latest_run  # noqa: E402

HERE = Path(__file__).parent.parent
START = HERE / "assets" / "start_frame.png"
PROFILES = HERE / "profiles"
CONTAINER = "vllm-omni-wm-server"


def main() -> None:
    PROFILES.mkdir(exist_ok=True)
    prompt = "The red ball rolls slowly to the right across the beige table."
    print("[phase_a] baseline run: 832x480, 33 frames, 30 steps")
    body = run_one(prompt, START)
    run = latest_run(CONTAINER)
    stages = {}
    if run is not None:
        stages = {
            "pipeline": run.pipeline,
            "durations_s": {k: round(v, 4) for k, v in run.durations.items()},
            "counts": run.counts,
        }
    body["stage_durations_from_logs"] = stages
    model = body.get("model", "unknown").split("/")[-1]
    out = PROFILES / f"phase_a_{model}.json"
    out.write_text(json.dumps(body, indent=2, default=str))
    print(f"[phase_a] wrote {out}")
    print(f"[phase_a] inference_s={body.get('inference_time_s'):.3f}")
    print(f"[phase_a] peak_memory_mb={body.get('peak_memory_mb')}")
    if stages:
        durs = stages["durations_s"]
        total = body.get("inference_time_s", 0.0)
        print("[phase_a] per-stage (from logs):")
        accounted = 0.0
        for k in ("text_encoder.forward", "vae.encode", "forward", "vae.decode"):
            v = durs.get(k)
            if v is None:
                continue
            pct = 100 * v / total if total else 0
            print(f"  {k:28s} {v:8.3f}s  ({pct:5.1f}%)")
            if k != "forward":  # forward is the outer wrapper; don't double-count
                accounted += v
        dit_only = durs.get("forward", 0) - accounted
        if dit_only > 0:
            print(f"  {'DiT denoising (derived)':28s} {dit_only:8.3f}s  ({100*dit_only/total:5.1f}%)")
        other = total - durs.get("forward", 0)
        print(f"  {'framework (derived)':28s} {other:8.3f}s  ({100*other/total:5.1f}%)")


if __name__ == "__main__":
    main()
