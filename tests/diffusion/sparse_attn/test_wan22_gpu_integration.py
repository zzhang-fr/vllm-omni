# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU integration tests for SparseAttentionBackend with Wan 2.2.

Tests the production path: SparseAttentionBackend -> _SparseAttentionImpl -> plugin fn.
Skipped when GPU is not available.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionSparseAttnConfig

_has_cuda = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(not _has_cuda, reason="Requires CUDA GPU")


@requires_gpu
class TestSparseAttentionImplGPU:
    """Test _SparseAttentionImpl forward on GPU with plugin detection."""

    def _make_impl(self, sparse_cfg=None, is_self_attention=True):
        """Create a _SparseAttentionImpl with mocked forward context."""
        from vllm_omni.diffusion.attention.backends.sparse_attention import _SparseAttentionImpl

        with patch(
            "vllm_omni.diffusion.attention.backends.sparse_attention._SparseAttentionImpl._read_sparse_cfg",
            return_value=sparse_cfg,
        ):
            return _SparseAttentionImpl(
                num_heads=4,
                head_size=64,
                softmax_scale=64**-0.5,
                causal=False,
                num_kv_heads=4,
                is_self_attention=is_self_attention,
            )

    def test_dense_fallback_forward(self):
        """No sparse config -> SDPA dense forward works on GPU."""
        impl = self._make_impl(sparse_cfg=None)
        q = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
        out = impl.forward(q, k, v)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    def test_cross_attention_always_dense(self):
        """Cross-attention (is_self_attention=False) ignores sparse config."""
        cfg = DiffusionSparseAttnConfig(backend="auto", params={"topk": 0.3})
        impl = self._make_impl(sparse_cfg=cfg, is_self_attention=False)
        q = torch.randn(1, 64, 4, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 64, 4, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 64, 4, 64, dtype=torch.bfloat16, device="cuda")
        out = impl.forward(q, k, v)
        assert out.shape == q.shape

    def test_mock_sparse_fn_forward(self):
        """Self-attention with a mock sparse function."""

        # Create impl with no sparse config (dense)
        impl = self._make_impl(sparse_cfg=None, is_self_attention=True)

        # Inject a mock sparse function that uses SDPA
        def fake_sparse_fn(q, k, v, params, is_causal):
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
            )

        impl._sparse_fn = fake_sparse_fn
        impl._params = {"topk": 0.5}

        q = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
        out = impl.forward(q, k, v)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()
