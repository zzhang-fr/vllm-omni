# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Wan 2.2 video generation with sparse attention acceleration.

Sparse attention reduces computation in DiT self-attention layers by
attending only to a subset of tokens via a plugin function (e.g. SpargeAttn).
The SparseAttentionBackend dispatcher resolves the plugin at init time and
handles cross-attention fallback, tensor layout, and config loading.

Usage:
    # SpargeAttn with default topk (0.5)
    python wan2_2_sparse.py --attn-backend spargeattn

    # SpargeAttn with custom topk via JSON config (flat form)
    python wan2_2_sparse.py --attn-backend '{"backend":"spargeattn","topk":0.3}'

    # Auto-detect installed plugin
    python wan2_2_sparse.py --attn-backend auto

    # Dense baseline (no sparse attention)
    python wan2_2_sparse.py --attn-backend dense

    # With sequence parallelism
    python wan2_2_sparse.py --attn-backend spargeattn --ring-degree 2
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, DiffusionSparseAttnConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan 2.2 video generation with sparse attention.")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Model ID or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="A serene lakeside sunrise with mist over the water.",
        help="Text prompt.",
    )
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--num-inference-steps", type=int, default=40, help="Sampling steps.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG scale.")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS.")
    parser.add_argument("--output", type=str, default="wan22_sparse_output.mp4", help="Output path.")

    # Attention backend: simple name or JSON with config
    parser.add_argument(
        "--attn-backend",
        type=str,
        default=None,
        help="Sparse attention plugin. "
        "Simple name: spargeattn, auto, dense. "
        'JSON: \'{"backend":"spargeattn","topk":0.3}\'. '
        "Import path: 'module.path:func_name'.",
    )

    # Parallelism
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses SP degree.")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring SP degree.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="TP size.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2], help="CFG parallel size.")

    # Other
    parser.add_argument("--flow-shift", type=float, default=None, help="Scheduler flow_shift.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    return parser.parse_args()


def main():
    args = parse_args()

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    # Build sparse attention config from --attn-backend
    attn_backend_name = args.attn_backend
    sparse_attn = None
    if attn_backend_name:
        if attn_backend_name.strip().startswith("{"):
            import json

            sparse_attn = DiffusionSparseAttnConfig.from_dict(json.loads(attn_backend_name))
        else:
            sparse_attn = DiffusionSparseAttnConfig(backend=attn_backend_name)

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    omni_kwargs = dict(
        model=args.model,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
    )
    if sparse_attn is not None:
        omni_kwargs["sparse_attn"] = sparse_attn
    if args.flow_shift is not None:
        omni_kwargs["flow_shift"] = args.flow_shift

    backend_str = sparse_attn.backend if sparse_attn else "default"
    params_str = str(sparse_attn.params) if sparse_attn else "N/A"
    print(f"\n{'=' * 60}")
    print("Wan 2.2 Attention Configuration:")
    print(f"  Backend: {backend_str}")
    print(f"  Params: {params_str}")
    print(f"  Video: {args.width}x{args.height}, {args.num_frames} frames")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"{'=' * 60}\n")

    omni = Omni(**omni_kwargs)

    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    generation_start = time.perf_counter()
    result = omni.generate(
        prompt_dict,
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
        ),
    )
    generation_time = time.perf_counter() - generation_start

    print(f"\nGeneration time: {generation_time:.2f}s ({generation_time * 1000:.0f}ms)")

    # Extract frames from result
    # Pipeline returns list[OmniRequestOutput]; each .images is
    # [ndarray(batch, frames, H, W, C)] with values in [0, 1].
    frames = result
    if isinstance(frames, list):
        frames = frames[0] if frames else None
    if isinstance(frames, OmniRequestOutput) and frames.images:
        frames = frames.images

    if frames is None:
        raise ValueError("No video frames in output.")

    # Unwrap: list[ndarray] -> ndarray
    if isinstance(frames, list) and len(frames) == 1:
        frames = frames[0]

    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for export_to_video.")

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().float().numpy()

    if isinstance(frames, np.ndarray):
        # Remove batch dim: (B, F, H, W, C) -> (F, H, W, C)
        while frames.ndim > 4:
            frames = frames[0]
        # export_to_video expects float [0,1] arrays — it handles uint8 conversion
        if frames.dtype in (np.float32, np.float64):
            frames = np.clip(frames, 0, 1)
        elif np.issubdtype(frames.dtype, np.integer):
            frames = frames.astype(np.float32) / 255.0
        frames = list(frames)  # list of (H, W, C) arrays

    export_to_video(frames, str(output_path), fps=args.fps)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
