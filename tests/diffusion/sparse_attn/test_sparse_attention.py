# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SparseAttentionBackend and the function-based plugin system.

Tests:
- SparseAttentionBackend inherits AttentionBackend
- _SparseAttentionImpl cross-attention path (is_self_attention=False -> dense)
- _SparseAttentionImpl self-attention with no sparse config -> dense
- Function-based plugin resolution (entry points, import path)
- SPARSE_ATTENTION enum registration
- is_self_attention parameter in Attention layer
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend
from vllm_omni.diffusion.attention.backends.sparse_attention import (
    SparseAttentionBackend,
    _SparseAttentionImpl,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestSparseAttentionBackendInheritance:
    """Test SparseAttentionBackend implements AttentionBackend."""

    def test_inherits_attention_backend(self):
        assert issubclass(SparseAttentionBackend, AttentionBackend)

    def test_get_name(self):
        assert SparseAttentionBackend.get_name() == "sparse_attention"

    def test_get_impl_cls(self):
        cls = SparseAttentionBackend.get_impl_cls()
        assert cls is _SparseAttentionImpl

    def test_get_metadata_cls(self):
        from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

        assert SparseAttentionBackend.get_metadata_cls() is AttentionMetadata

    def test_get_builder_cls_none(self):
        assert SparseAttentionBackend.get_builder_cls() is None

    def test_get_supported_head_sizes(self):
        assert SparseAttentionBackend.get_supported_head_sizes() == []


class TestSparseAttentionImplCrossAttention:
    """Test _SparseAttentionImpl with is_self_attention=False always uses dense."""

    def test_cross_attn_uses_dense(self):
        impl = _SparseAttentionImpl(
            num_heads=8,
            head_size=64,
            softmax_scale=0.125,
            causal=False,
            is_self_attention=False,
        )
        assert impl._is_self_attention is False
        assert impl._sparse_fn is None
        assert impl._dense is not None

    def test_self_attn_default_true(self):
        """Default is_self_attention=True."""
        impl = _SparseAttentionImpl(
            num_heads=8,
            head_size=64,
            softmax_scale=0.125,
            causal=False,
        )
        assert impl._is_self_attention is True

    def test_self_attn_no_config_falls_back_to_dense(self):
        """When no sparse config is available, self-attention also uses dense."""
        impl = _SparseAttentionImpl(
            num_heads=8,
            head_size=64,
            softmax_scale=0.125,
            causal=False,
            is_self_attention=True,
        )
        # Without forward context, _read_sparse_cfg returns None -> no plugin loaded
        assert impl._sparse_fn is None


class TestEnumRegistration:
    """Test SPARSE_ATTENTION is in the registry enum."""

    def test_sparse_attention_in_enum(self):
        from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum

        assert hasattr(DiffusionAttentionBackendEnum, "SPARSE_ATTENTION")

    def test_sparse_attention_class_path(self):
        from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum

        backend = DiffusionAttentionBackendEnum.SPARSE_ATTENTION
        assert "sparse_attention.SparseAttentionBackend" in backend.value


class TestEntryPointsPluginLookup:
    """Test _get_plugin_backend_cls in selector.py."""

    def test_no_plugin_returns_none(self):
        from vllm_omni.diffusion.attention.selector import _get_plugin_backend_cls

        result = _get_plugin_backend_cls("nonexistent_backend_xyz")
        assert result is None

    def test_plugin_found_via_entry_point(self):
        """Mock an entry_point and verify it's loaded."""
        from vllm_omni.diffusion.attention.selector import _get_plugin_backend_cls

        mock_ep = MagicMock()
        mock_ep.name = "testplugin"
        mock_ep.value = "test.module:TestBackend"
        mock_backend_cls = type("TestBackend", (AttentionBackend,), {})
        mock_ep.load.return_value = mock_backend_cls

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = _get_plugin_backend_cls("testplugin")

        assert result is mock_backend_cls

    def test_plugin_case_insensitive(self):
        """Plugin lookup should be case-insensitive."""
        from vllm_omni.diffusion.attention.selector import _get_plugin_backend_cls

        mock_ep = MagicMock()
        mock_ep.name = "SpargeAttn"
        mock_ep.value = "test.module:SpargeBackend"
        mock_backend_cls = type("SpargeBackend", (AttentionBackend,), {})
        mock_ep.load.return_value = mock_backend_cls

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = _get_plugin_backend_cls("spargeattn")

        assert result is mock_backend_cls


class TestIsSelfattentionInAttentionLayer:
    """Test that Attention layer passes is_self_attention to impl."""

    def _mock_distributed(self):
        from contextlib import ExitStack

        stack = ExitStack()

        mock_tp = MagicMock()
        mock_tp.world_size = 1
        mock_tp.rank_in_group = 0
        mock_tp.local_rank = 0
        import vllm.distributed.parallel_state as ps

        stack.enter_context(patch.object(ps, "_TP", mock_tp))

        mock_config = MagicMock()
        mock_config.attention_backend = None
        mock_config.parallel_config.ring_degree = 1
        mock_ctx = stack.enter_context(patch("vllm_omni.diffusion.attention.layer.get_forward_context"))
        mock_ctx.return_value.omni_diffusion_config = mock_config
        stack.enter_context(
            patch(
                "vllm_omni.diffusion.attention.layer.is_forward_context_available",
                return_value=True,
            )
        )
        return stack

    def test_default_is_self_attention_true(self):
        """By default, Attention sets is_self_attention=True."""
        with self._mock_distributed():
            from vllm_omni.diffusion.attention.layer import Attention

            attn = Attention(
                num_heads=8,
                head_size=64,
                causal=False,
                softmax_scale=0.125,
            )
            # The impl was constructed — no crash means it accepted the kwarg
            assert attn.attention is not None

    def test_is_self_attention_false_accepted(self):
        """Attention(is_self_attention=False) works."""
        with self._mock_distributed():
            from vllm_omni.diffusion.attention.layer import Attention

            attn = Attention(
                num_heads=8,
                head_size=64,
                causal=False,
                softmax_scale=0.125,
                is_self_attention=False,
            )
            assert attn.attention is not None


class TestSparseAttentionImplResolvePlugin:
    """Test _SparseAttentionImpl._resolve_plugin with function-based plugins."""

    def test_resolve_auto_no_plugins(self):
        """Auto backend returns None when no plugins installed."""
        from vllm_omni.diffusion.data import DiffusionSparseAttnConfig

        cfg = DiffusionSparseAttnConfig(backend="auto")
        with patch("importlib.metadata.entry_points", return_value=[]):
            result = _SparseAttentionImpl._resolve_plugin(cfg)
            assert result is None

    def test_resolve_import_path(self):
        """Full import path 'module:func' resolves to a callable."""
        from vllm_omni.diffusion.data import DiffusionSparseAttnConfig

        # Use a known function as test target
        cfg = DiffusionSparseAttnConfig(backend="torch.nn.functional:scaled_dot_product_attention")
        result = _SparseAttentionImpl._resolve_plugin(cfg)
        import torch.nn.functional as F

        assert result is F.scaled_dot_product_attention

    def test_resolve_short_name_via_entry_point(self):
        """Short name resolved via vllm_omni.sparse_attn entry point group."""
        from vllm_omni.diffusion.data import DiffusionSparseAttnConfig

        mock_fn = MagicMock()
        mock_ep = MagicMock()
        mock_ep.name = "testbackend"
        mock_ep.load.return_value = mock_fn

        cfg = DiffusionSparseAttnConfig(backend="testbackend")
        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = _SparseAttentionImpl._resolve_plugin(cfg)

        assert result is mock_fn

    def test_resolve_auto_picks_first(self):
        """Auto backend picks the first available plugin function."""
        mock_fn = MagicMock()
        mock_ep = MagicMock()
        mock_ep.name = "myplugin"
        mock_ep.load.return_value = mock_fn

        result_fn = None
        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result_fn = _SparseAttentionImpl._first_available_plugin()

        assert result_fn is mock_fn
