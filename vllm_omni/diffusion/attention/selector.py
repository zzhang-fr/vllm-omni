# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion attention backend selector.

RFC #2632 P2: per-role dispatch.

Lookup precedence per call:
    config.per_role[role] → config.default → DIFFUSION_ATTENTION_BACKEND env var
    → platform default

The platform layer (`current_omni_platform.get_diffusion_attn_backend_cls`)
encapsulates GPU/arch-specific defaults and validates the selected backend.

We intentionally do **not** wrap this function in `@cache`. The selector only
runs during model construction (per `Attention.__init__`), not on the forward
path; caching adds compound-key bookkeeping for `AttentionConfig` (which is
mutable) without measurable benefit. See `tests/diffusion/attention/
test_selector_benchmark.py` for the latency baseline.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend
from vllm_omni.diffusion.attention.role import AttentionRole

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

logger = init_logger(__name__)


def _load_backend_cls(cls_path: str) -> type[AttentionBackend]:
    module_path, _, class_name = cls_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid backend class path: {cls_path!r}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}") from e
    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in {module_path}: {e}") from e


def get_attn_backend(
    role: AttentionRole | str | None = None,
    head_size: int = -1,
    config: AttentionConfig | None = None,
) -> tuple[type[AttentionBackend], AttentionSpec]:
    """Resolve the attention backend for a given (role, config).

    Args:
        role: Role of the attention layer (self / cross / joint / other).
            ``None`` is treated as "no role-specific lookup" — equivalent to
            asking only for the default backend.
        head_size: Head size hint for the platform selector. Use ``-1`` when
            unknown (e.g. inside a generic helper) — the platform selector
            already tolerates that today.
        config: Per-role attention config; usually
            ``forward_context.omni_diffusion_config.attention``. ``None`` is
            allowed and means "fall through to platform default".

    Returns:
        ``(backend_cls, spec)``: the resolved backend class plus the
        ``AttentionSpec`` that produced it. Callers feed ``spec.extra`` into
        ``AttentionImpl(..., backend_kwargs=spec.extra)``. When falling
        through to the platform default, ``spec`` is
        ``AttentionSpec(backend="auto", extra={})`` — a sentinel telling
        callers there is no static spec to forward.
    """
    # Lazy import: avoids a circular module dependency between data.py
    # (which defines AttentionConfig) and the platform layer.
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
    from vllm_omni.platforms import current_omni_platform

    role_key: str | None
    if role is None:
        role_key = None
    elif isinstance(role, AttentionRole):
        role_key = role.value
    else:
        role_key = AttentionRole.coerce(role).value

    spec: AttentionSpec | None = None
    if isinstance(config, AttentionConfig):
        spec = config.get(role_key)
    elif config is not None:
        raise TypeError(f"get_attn_backend: config must be AttentionConfig or None, got {type(config).__name__}")

    selected_backend = spec.backend if spec is not None else None
    if selected_backend is None:
        # Fall through to env var (lowest-priority default), preserving legacy behavior.
        selected_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")

    backend_cls_path = current_omni_platform.get_diffusion_attn_backend_cls(
        selected_backend=selected_backend,
        head_size=head_size,
    )
    backend_cls = _load_backend_cls(backend_cls_path)

    if spec is None:
        spec = AttentionSpec(backend="auto", extra={})
    return backend_cls, spec
