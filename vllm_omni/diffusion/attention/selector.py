# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusion attention backend selector.

This module provides per-role attention backend selection for diffusion models.
The selector supports:
1. Per-role dispatch via AttentionConfig (config.per_role[role] -> config.default)
2. Environment variable override (DIFFUSION_ATTENTION_BACKEND)
3. Platform-specific defaults

Usage:
    from vllm_omni.diffusion.attention.selector import get_attn_backend

    # Legacy usage (returns backend class only, for backward compat)
    backend_cls = get_attn_backend(head_size=64)

    # Per-role usage with config (returns tuple)
    backend_cls, spec = get_attn_backend(head_size=128, role="self", config=attn_config)
"""

import importlib
import os
from typing import TYPE_CHECKING, overload

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

logger = init_logger(__name__)


def _get_plugin_backend_cls(name: str) -> "type[AttentionBackend] | None":
    """Check vllm_omni.attn_backend entry_points for a plugin with this name.

    For full AttentionBackend plugins (e.g. custom flash_attn replacement).
    Sparse attention plugins should use the vllm_omni.sparse_attn entry
    point group instead -- they are resolved by the sparse_attention dispatcher.

    Plugin packages declare themselves in pyproject.toml:
        [project.entry-points."vllm_omni.attn_backend"]
        my_backend = "my_package.backend:MyAttentionBackend"
    """
    try:
        import importlib.metadata

        for ep in importlib.metadata.entry_points(group="vllm_omni.attn_backend"):
            if ep.name.lower() == name.lower():
                logger.info(
                    "Loading attention backend plugin '%s' from %s",
                    name,
                    ep.value,
                )
                return ep.load()
    except Exception as e:
        logger.debug("Plugin backend lookup failed for '%s': %s", name, e)
    return None


def _load_backend_cls(cls_path: str) -> type[AttentionBackend]:
    """Load a backend class from its fully qualified path.

    Args:
        cls_path: Fully qualified class path (e.g.,
            "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend")

    Returns:
        The loaded backend class
    """
    module_path, class_name = cls_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
        return backend_class
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in module {module_path}: {e}") from e


def _resolve_backend(
    head_size: int = -1,
    role: str | None = None,
    config: "AttentionConfig | None" = None,
) -> "tuple[type[AttentionBackend], AttentionSpec]":
    """Internal: resolve backend class and spec.

    Lookup order:
    1. config.get(role) -> per_role then default
    2. DIFFUSION_ATTENTION_BACKEND env var
    3. Platform default
    """
    from vllm_omni.diffusion.data import AttentionSpec
    from vllm_omni.platforms import current_omni_platform

    selected_backend: str | None = None
    spec: AttentionSpec | None = None

    # 1. Config lookup (per_role -> default)
    if config is not None:
        spec = config.get(role)
        if spec is not None and spec.backend:
            selected_backend = spec.backend

    # 2. Env var fallback
    if selected_backend is None:
        selected_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")

    # 3. Check plugins first for non-enum names
    if selected_backend:
        plugin_cls = _get_plugin_backend_cls(selected_backend)
        if plugin_cls is not None:
            if spec is None:
                spec = AttentionSpec(backend=selected_backend)
            return plugin_cls, spec

    # 4. Platform-based selection (handles both named and default backends)
    backend_cls_path = current_omni_platform.get_diffusion_attn_backend_cls(
        selected_backend=selected_backend,
        head_size=head_size,
    )

    backend_cls = _load_backend_cls(backend_cls_path)
    resolved_name = backend_cls.get_name() if hasattr(backend_cls, "get_name") else (selected_backend or "")

    if spec is None:
        spec = AttentionSpec(backend=resolved_name)

    return backend_cls, spec


def get_attn_backend(
    head_size: int = -1,
    role: str | None = None,
    config: "AttentionConfig | None" = None,
) -> "type[AttentionBackend] | tuple[type[AttentionBackend], AttentionSpec]":
    """Get the attention backend for diffusion models.

    When called with only ``head_size`` (the legacy pattern), returns just the
    backend class for backward compatibility.  When ``role`` or ``config`` is
    provided, returns ``(backend_class, AttentionSpec)``.

    Lookup order:
    1. ``config.get(role)`` -- per-role, then config default
    2. ``DIFFUSION_ATTENTION_BACKEND`` env var
    3. Platform default

    Args:
        head_size: Head size for attention (may affect backend selection).
        role: Attention role name (e.g. "self", "cross", "joint").
        config: Optional :class:`AttentionConfig` with per-role specs.

    Returns:
        ``type[AttentionBackend]`` (legacy) **or**
        ``tuple[type[AttentionBackend], AttentionSpec]`` (new API).
    """
    backend_cls, spec = _resolve_backend(head_size=head_size, role=role, config=config)

    # Backward compatibility: when called without role/config, return just the class
    if role is None and config is None:
        return backend_cls

    return backend_cls, spec
