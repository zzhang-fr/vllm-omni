"""Plot Phase A breakdown and Phase B sweep results.

Reads phase_a_*.json and phase_b_sweep.json from world_models/profiles/
and emits PNGs next to them.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.parent
PROFILES = HERE / "profiles"


def plot_phase_a() -> None:
    p = PROFILES / "phase_a_hunyuanvideo_1_5_i2v.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    stages = data["stages_s"]
    labels = [
        ("DiT denoising", stages["dit_denoising_loop_derived"]),
        ("VAE decode", stages["vae_decode"]),
        ("VAE encode", stages["vae_encode"]),
        ("Text encoder", stages["text_encoder_forward_total"]),
        ("Framework", stages["framework_overhead_derived"]),
    ]
    total = data["total_inference_s"]
    fig, ax = plt.subplots(figsize=(9, 3))
    left = 0
    colors = ["#3b7dd8", "#d8973b", "#8a3bd8", "#d83b7d", "#7d7d7d"]
    for (label, val), c in zip(labels, colors):
        pct = 100 * val / total
        ax.barh([0], [val], left=left, color=c, label=f"{label}: {val:.2f}s ({pct:.1f}%)")
        left += val
    ax.set_yticks([])
    ax.set_xlabel("seconds")
    ax.set_xlim(0, total * 1.01)
    ax.set_title(
        f"HunyuanVideo-1.5 I2V — per-stage time (832x480, 33 frames, 30 steps, 1x A100)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=8)
    fig.tight_layout()
    out = PROFILES / "phase_a_breakdown.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def _dit_from_stages(stages: dict | None) -> float:
    if stages is None:
        return 0.0
    d = stages.get("durations") or {}
    return (
        d.get("forward", 0)
        - d.get("vae.encode", 0)
        - d.get("vae.decode", 0)
        - d.get("text_encoder.forward", 0)
    )


def plot_phase_b() -> None:
    p = PROFILES / "phase_b_sweep.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())

    for axis_key, outfile, xlabel in [
        ("sweep_steps", "phase_b_steps.png", "num_inference_steps"),
        ("sweep_frames", "phase_b_frames.png", "num_frames"),
    ]:
        rows = data[axis_key]
        xs = [r["value"] for r in rows]
        inf = [r.get("inference_time_s") or 0 for r in rows]
        dit = [_dit_from_stages(r.get("stages")) for r in rows]
        vae = [(r.get("stages", {}) or {}).get("durations", {}).get("vae.decode", 0) for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(xs, inf, "-o", label="total inference_time_s", color="#222")
        ax.plot(xs, dit, "-s", label="DiT denoising (derived)", color="#3b7dd8")
        ax.plot(xs, vae, "-^", label="VAE decode", color="#d8973b")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("seconds")
        ax.set_title(f"Phase B — latency vs {xlabel} (HunyuanVideo-1.5 I2V, 1x A100)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PROFILES / outfile, dpi=120)
        plt.close(fig)
        print(f"wrote {PROFILES / outfile}")


if __name__ == "__main__":
    plot_phase_a()
    plot_phase_b()
