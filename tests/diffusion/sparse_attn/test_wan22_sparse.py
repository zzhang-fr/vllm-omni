# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Wan 2.2 sparse attention integration after refactor.

Tests:
- WanSelfAttention no longer has sparse_attn attribute
- WanCrossAttention passes role=AttentionRole.CROSS
- Sparse attention is now handled via the attention backend layer
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _mock_distributed():
    """Context manager to mock distributed and forward context for unit tests."""
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
    mock_config.attention = None
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


class TestWanSelfAttentionNoSparseAttr:
    """Test that WanSelfAttention no longer has sparse_attn attribute."""

    def _make_self_attn(self):
        with _mock_distributed():
            from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
                WanSelfAttention,
            )

            return WanSelfAttention(dim=512, num_heads=8, head_dim=64)

    def test_no_sparse_attn_attribute(self):
        attn = self._make_self_attn()
        assert not hasattr(attn, "sparse_attn")


class TestWanCrossAttentionRole:
    """Test that WanCrossAttention passes role=AttentionRole.CROSS."""

    def _make_cross_attn(self):
        with _mock_distributed():
            from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
                WanCrossAttention,
            )

            return WanCrossAttention(dim=512, num_heads=8, head_dim=64)

    def test_cross_attn_role_is_cross(self):
        """Verify the Attention layer in WanCrossAttention was constructed
        with role=AttentionRole.CROSS."""
        from vllm_omni.diffusion.attention.role import AttentionRole

        cross_attn = self._make_cross_attn()
        # The Attention layer should have role=AttentionRole.CROSS
        assert cross_attn.attn is not None
        assert cross_attn.attn.role == AttentionRole.CROSS


class TestNoEnableSparseAttention:
    """Test that WanTransformer3DModel no longer has enable_sparse_attention."""

    def _make_transformer(self):
        from vllm.config import VllmConfig, set_current_vllm_config

        with _mock_distributed(), set_current_vllm_config(VllmConfig()):
            from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
                WanTransformer3DModel,
            )

            return WanTransformer3DModel(
                num_attention_heads=4,
                attention_head_dim=64,
                in_channels=16,
                out_channels=16,
                text_dim=512,
                freq_dim=256,
                ffn_dim=1024,
                num_layers=2,
            )

    def test_no_enable_sparse_attention_method(self):
        transformer = self._make_transformer()
        assert not hasattr(transformer, "enable_sparse_attention")

    def test_self_attn_has_no_sparse_attn(self):
        transformer = self._make_transformer()
        for block in transformer.blocks:
            assert not hasattr(block.attn1, "sparse_attn")
