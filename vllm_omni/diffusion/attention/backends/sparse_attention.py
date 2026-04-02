# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Built-in dispatcher backend for DiT sparse attention.

Selected via --attn-backend sparse_attention.
Reads is_self_attention (from Attention.__init__ extra_impl_args):
  - False -> SDPA fallback, zero overhead on cross-attention
  - True  -> look up sparse attention function from plugin

Plugin resolution in self-attention path:
  1. sparse_attn.backend == "auto" -> first available in vllm_omni.sparse_attn EP group
  2. sparse_attn.backend == short name -> match in vllm_omni.sparse_attn EP group
  3. sparse_attn.backend == "module.path:func_name" -> import directly
  4. None found -> dense SDPA fallback with warning

Plugins register a function with signature:
    fn(q, k, v, params, is_causal) -> Tensor
    q/k/v: (B, H, S, D) tensors
    params: dict of plugin-specific parameters
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionSparseAttnConfig

logger = init_logger(__name__)

# Type alias for sparse attention function plugins
SparseAttnFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, dict, bool],
    torch.Tensor,
]


class SparseAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "sparse_attention"

    @staticmethod
    def get_impl_cls():
        return _SparseAttentionImpl

    @staticmethod
    def get_metadata_cls():
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []


class _SparseAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self._num_heads = num_heads
        self._head_size = head_size
        self._softmax_scale = softmax_scale
        self._causal = causal
        self._is_self_attention: bool = extra_impl_args.get("is_self_attention", True)

        # Dense fallback (used for cross-attention, or when no plugin found)
        self._dense = SDPABackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
        )

        # Sparse attention function plugin (None = use dense)
        self._sparse_fn: SparseAttnFn | None = None
        self._params: dict = {}

        if not self._is_self_attention:
            return  # cross-attention: always dense, done

        cfg = self._read_sparse_cfg()
        if cfg is None or cfg.backend in ("none", "dense"):
            return  # dense: use SDPA fallback, no plugin needed

        plugin = self._resolve_plugin(cfg)
        if plugin is None:
            logger.warning(
                "No sparse attention plugin found for backend='%s'. Falling back to dense.",
                cfg.backend,
            )
            return

        self._sparse_fn = plugin
        self._params = cfg.params
        logger.info(
            "Using sparse attention plugin '%s' (params=%s)",
            cfg.backend,
            cfg.params,
        )

    # ------------------------------------------------------------------
    # AttentionImpl interface
    # ------------------------------------------------------------------

    def _forward_sparse(self, query, key, value, attn_metadata=None):
        """Sparse plugin path for self-attention on supported platforms."""
        # query/key/value: (B, S, H, D) -> (B, H, S, D)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        out = self._sparse_fn(q, k, v, self._params, self._causal)

        # (B, H, S, D) -> (B, S, H, D)
        return out.transpose(1, 2)

    def forward_cuda(self, query, key, value, attn_metadata=None):
        if self._sparse_fn is None:
            return self._dense.forward_cuda(query, key, value, attn_metadata)
        return self._forward_sparse(query, key, value, attn_metadata)

    def forward_hip(self, query, key, value, attn_metadata=None):
        return self._dense.forward_hip(query, key, value, attn_metadata)

    def forward_npu(self, query, key, value, attn_metadata=None):
        return self._dense.forward_npu(query, key, value, attn_metadata)

    def forward_xpu(self, query, key, value, attn_metadata=None):
        return self._dense.forward_xpu(query, key, value, attn_metadata)

    def forward_musa(self, query, key, value, attn_metadata=None):
        return self._dense.forward_musa(query, key, value, attn_metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_sparse_cfg() -> DiffusionSparseAttnConfig | None:
        try:
            from vllm_omni.diffusion.forward_context import get_forward_context

            return get_forward_context().omni_diffusion_config.sparse_attn
        except (RuntimeError, LookupError, AssertionError):
            # RuntimeError / AssertionError: forward context not yet initialised
            # LookupError: config key missing during early init
            return None
        except AttributeError:
            logger.debug(
                "sparse_attn config not available on omni_diffusion_config",
                exc_info=True,
            )
            return None

    @staticmethod
    def _resolve_plugin(cfg: DiffusionSparseAttnConfig) -> SparseAttnFn | None:
        """Resolve a sparse attention function from config.

        Returns a callable fn(q, k, v, params, is_causal) -> Tensor,
        or None if no plugin found.
        """
        name = cfg.backend
        if not name or name == "auto":
            return _SparseAttentionImpl._first_available_plugin()
        if ":" in name or ("." in name and not name.startswith(".")):
            # Fully qualified "module.path:func_name" or "module.path.func"
            try:
                return _import_function(name)
            except Exception as e:
                logger.warning("Failed to load plugin '%s': %s", name, e)
                return None
        # Short name -> vllm_omni.sparse_attn EP group
        try:
            for ep in importlib.metadata.entry_points(group="vllm_omni.sparse_attn"):
                if ep.name.lower() == name.lower():
                    try:
                        return ep.load()
                    except Exception:
                        logger.warning(
                            "Sparse plugin '%s' (%s) found but failed to load",
                            ep.name,
                            ep.value,
                            exc_info=True,
                        )
                        return None
        except Exception:
            logger.debug("Failed to query entry_points for sparse_attn", exc_info=True)
        return None

    @staticmethod
    def _first_available_plugin() -> SparseAttnFn | None:
        try:
            for ep in importlib.metadata.entry_points(group="vllm_omni.sparse_attn"):
                try:
                    fn = ep.load()
                    logger.info("Auto-selected sparse plugin: %s", ep.name)
                    return fn
                except ImportError:
                    logger.debug("Sparse plugin '%s' not importable, skipping", ep.name)
                    continue
                except Exception:
                    logger.warning(
                        "Sparse plugin '%s' found but failed to load",
                        ep.name,
                        exc_info=True,
                    )
                    continue
        except Exception:
            logger.debug("Failed to query entry_points for sparse_attn", exc_info=True)
        return None


def _import_function(path: str) -> SparseAttnFn:
    """Import a function from 'module.path:func' or 'module.path.func'."""
    if ":" in path:
        module_path, func_name = path.rsplit(":", 1)
    else:
        module_path, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)
