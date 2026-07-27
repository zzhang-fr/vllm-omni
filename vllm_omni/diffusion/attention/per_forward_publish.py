# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generic, model-agnostic publication of per-forward attention state.

Step-aware and structure-aware sparse attention kernels need two things the flat
q/k/v tensors do not carry: how far along the denoise loop we are, and how to map
the flat token sequence back to ``(frame, patch)`` coordinates. Both are already
determined by state the framework can see:

* the raw video geometry primitives — the post-VAE pre-patch grid ``(T, H, W)``
  and the patch size ``(p_t, p_h, p_w)`` — follow from the transformer's forward
  input (the ``(B, C, T, H, W)`` latent) plus the model's patch configuration;
* the denoise loop length is the diffusers-convention ``pipeline._num_timesteps``
  (exposed as the ``num_timesteps`` property), set by every in-tree pipeline
  right after ``scheduler.set_timesteps`` and before its timestep loop.

So instead of asking every DiT / pipeline to publish this from inside its own
``forward`` (an ``O(models)`` cost that couples core model code to the
sparse-attention feature), the framework registers a single ``forward_pre_hook``
on the transformer module(s) at load time and publishes onto
:class:`~vllm_omni.diffusion.forward_context.ForwardContext`. No model code is
touched — adding a new model needs no per-forward-state edits at all.

Everything downstream is unchanged: the attention bridge surfaces the state onto
:class:`PerForwardState`, and the SPARSE_ATTN dispatcher's default resolver (or a
plugin ``geometry_fn``) derives ``total_latent_frames`` / ``patches_per_frame``.

The hook is best-effort. Geometry no-ops for non-patchified models (no
``patch_size``, or a non-5D input such as image/audio latents); the step total
no-ops for pipelines that do not follow the ``_num_timesteps`` convention. The
field is simply left unset and a kernel falls back to its own path — it never
breaks a run. A pipeline with a non-standard loop can still publish explicitly
via the ``set_forward_context_*`` setters; an explicitly set value is never
overwritten by the hook.
"""

from __future__ import annotations

import weakref

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context_total_denoise_steps,
    set_forward_context_video_geometry,
)

logger = init_logger(__name__)

_PER_FORWARD_HOOK_FLAG = "_vllm_omni_per_forward_hook"


def _resolve_patch_size(module: torch.nn.Module) -> tuple[int, int, int] | None:
    """Normalize a transformer's patch size to ``(p_t, p_h, p_w)``.

    Handles the two conventions in-tree:
      * a single 3-tuple (e.g. Wan ``config.patch_size == (p_t, p_h, p_w)``);
      * a scalar spatial patch plus a separate temporal patch (e.g. HunyuanVideo
        ``patch_size`` + ``patch_size_t``).
    A scalar ``patch_size`` with no ``patch_size_t`` defaults to ``p_t = 1``:
    video DiTs overwhelmingly do not patch time, so an unpatched temporal axis is
    the safer prior than assuming an isotropic patch (which would silently halve
    the published frame count for a model with ``patch_size=2``).
    Reads from ``module.config`` first, then falls back to module attributes, so
    it works whether or not the transformer carries a diffusers-style ``config``.
    Returns ``None`` when no usable patch size is found.
    """
    cfg = getattr(module, "config", None)

    def _get(name: str):
        val = getattr(cfg, name, None) if cfg is not None else None
        return val if val is not None else getattr(module, name, None)

    ps = _get("patch_size")
    pt = _get("patch_size_t")

    if isinstance(ps, (tuple, list)) and len(ps) == 3:
        return int(ps[0]), int(ps[1]), int(ps[2])
    if isinstance(ps, int):
        return int(pt) if pt is not None else 1, int(ps), int(ps)
    return None


def _resolve_total_denoise_steps(pipeline) -> int | None:
    """Read the denoise loop length off a pipeline, or ``None``.

    Uses the diffusers convention ``_num_timesteps`` (``len(timesteps)``, set
    before the timestep loop), preferring the public ``num_timesteps`` property
    when the pipeline exposes one. Note this is the length of the loop currently
    being run: a chunked pipeline that re-sets it per chunk (e.g. Helios) yields
    the per-chunk total, which is what a progress fraction wants anyway.
    """
    total = getattr(pipeline, "num_timesteps", None)
    if total is None:
        total = getattr(pipeline, "_num_timesteps", None)
    if isinstance(total, int) and not isinstance(total, bool) and total > 0:
        return total
    return None


def _publish_video_geometry(module: torch.nn.Module, args, kwargs) -> None:
    """Publish raw geometry primitives from the transformer's latent input."""
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None and args:
        hidden_states = args[0]
    # Only patchified video latents (B, C, T, H, W) carry a 3D grid.
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() < 5:
        return
    patch_size = _resolve_patch_size(module)
    if patch_size is None:
        return
    t, h, w = (int(s) for s in hidden_states.shape[-3:])
    set_forward_context_video_geometry(latent_shape=(t, h, w), patch_size=patch_size)


def _publish_denoise_total(pipeline) -> None:
    """Publish the denoise loop length, unless already set explicitly."""
    if pipeline is None or not is_forward_context_available():
        return
    if get_forward_context().total_denoise_steps is not None:
        return  # an explicit publisher wins over the framework default
    total = _resolve_total_denoise_steps(pipeline)
    if total is not None:
        set_forward_context_total_denoise_steps(total)


def _make_per_forward_pre_hook(pipeline_ref):
    """Build the pre-hook closure; ``pipeline_ref`` is a zero-arg getter."""

    def _per_forward_pre_hook(module: torch.nn.Module, args, kwargs) -> None:
        try:
            _publish_video_geometry(module, args, kwargs)
            _publish_denoise_total(pipeline_ref())
        except Exception:  # best-effort: state stays unset, never break a forward
            logger.debug("per-forward pre-hook failed; leaving state unset", exc_info=True)

    return _per_forward_pre_hook


def register_per_forward_hooks(pipeline) -> None:
    """Register the per-forward pre-hook on a pipeline's transformer module(s).

    Idempotent (guarded by a flag attribute) and safe to call after
    ``torch.compile`` has replaced the transformer attributes — the pre-hook fires
    before the compiled forward and only reads shapes and a Python attribute, so
    it cannot cause a graph break. A no-op for pipelines without a transformer.

    The pipeline is captured weakly where possible so the hook (owned by a module
    the pipeline itself owns) does not create a reference cycle.
    """
    if pipeline is None:
        return
    try:
        pipeline_ref = weakref.ref(pipeline)
    except TypeError:  # not weak-referenceable; hold it directly
        pipeline_ref = lambda: pipeline  # noqa: E731

    for attr in ("transformer", "transformer_2"):
        module = getattr(pipeline, attr, None)
        if not isinstance(module, torch.nn.Module):
            continue
        if getattr(module, _PER_FORWARD_HOOK_FLAG, False):
            continue
        module.register_forward_pre_hook(_make_per_forward_pre_hook(pipeline_ref), with_kwargs=True)
        setattr(module, _PER_FORWARD_HOOK_FLAG, True)
        logger.debug("Registered per-forward state pre-hook on pipeline.%s", attr)
