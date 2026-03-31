# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusion attention backend selector.

This module provides the interface for selecting diffusion attention backends.
The actual backend selection logic is delegated to the platform layer
(vllm_omni.platforms), similar to how vLLM handles attention backend selection.

Usage:
    from vllm_omni.diffusion.attention.selector import get_attn_backend

    # Get the appropriate backend for current platform
    backend_cls = get_attn_backend(head_size=64)

    # Or override via environment variable
    # export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
"""

import importlib
import os
from functools import cache

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
)

logger = init_logger(__name__)


def _get_plugin_backend_cls(name: str) -> "type[AttentionBackend] | None":
    """Check vllm_omni.attn_backend entry_points for a plugin with this name.

    Plugin packages declare themselves in pyproject.toml:
        [project.entry-points."vllm_omni.attn_backend"]
        spargeattn = "sparge_vllm_omni.backend:SpargeBackend"

    Users then use:  --attn-backend spargeattn
    or equivalently: DIFFUSION_ATTENTION_BACKEND=spargeattn
    """
    try:
        import importlib.metadata
        for ep in importlib.metadata.entry_points(
            group="vllm_omni.attn_backend"
        ):
            if ep.name.lower() == name.lower():
                logger.info(
                    "Loading attention backend plugin '%s' from %s",
                    name, ep.value,
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
        raise ImportError(f"Failed to import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in module: {e}")


@cache
def get_attn_backend(head_size: int) -> type[AttentionBackend]:
    """
    Get attention backend for diffusion models.

    The backend selection is delegated to the current platform
    (vllm_omni.platforms.current_omni_platform), which selects the
    appropriate backend based on:
    1. User override via DIFFUSION_ATTENTION_BACKEND environment variable
    2. Platform-specific defaults and capabilities

    This is similar to how vLLM's get_attn_backend_cls works, where the
    platform layer decides which backend to use based on hardware capabilities.

    Args:
        head_size: Head size for attention computation (may affect backend selection)

    Returns:
        The selected attention backend class
    """
    from vllm_omni.platforms import current_omni_platform

    # Check environment variable for user override
    selected_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")

    # Check entry_points plugins before enum lookup
    if selected_backend:
        plugin_cls = _get_plugin_backend_cls(selected_backend)
        if plugin_cls is not None:
            return plugin_cls

    # Platform-based selection via enum
    backend_cls_path = current_omni_platform.get_diffusion_attn_backend_cls(
        selected_backend=selected_backend,
        head_size=head_size,
    )

    return _load_backend_cls(backend_cls_path)
