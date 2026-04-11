# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU end-to-end test for per-role attention dispatch (RFC #2632 P3).

Constructs `Attention(role=...)` layers under a real `OmniDiffusionConfig`
forward context, runs a tiny CUDA forward through both, and verifies that:

  1. Each role lands on the configured backend impl.
  2. The forward path produces finite outputs.

Skipped on hosts without CUDA. Does NOT load any model weights.
"""

import pytest
import torch

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.role import AttentionRole
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    override_forward_context,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _has_flash_attn() -> bool:
    try:
        from vllm_omni.diffusion.attention.backends.utils.fa import HAS_FLASH_ATTN

        return bool(HAS_FLASH_ATTN)
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip the env var so the test only sees the AttentionConfig path."""
    monkeypatch.delenv("DIFFUSION_ATTENTION_BACKEND", raising=False)


def test_real_per_role_dispatch_self_sdpa_cross_sdpa():
    omni = OmniDiffusionConfig(attention={"per_role": {"self": "TORCH_SDPA", "cross": "TORCH_SDPA"}})
    with override_forward_context(ForwardContext(omni_diffusion_config=omni)):
        self_layer = _build_attn(AttentionRole.SELF)
        cross_layer = _build_attn(AttentionRole.CROSS)

    assert type(self_layer.attention).__name__ == "SDPAImpl"
    assert type(cross_layer.attention).__name__ == "SDPAImpl"
    _round_trip(self_layer)
    _round_trip(cross_layer)


@pytest.mark.skipif(not _has_flash_attn(), reason="flash-attn not installed")
def test_real_per_role_dispatch_self_sdpa_cross_flashattn():
    """Mixed backends per role: SDPA for self, FLASH_ATTN for cross."""
    omni = OmniDiffusionConfig(attention={"per_role": {"self": "TORCH_SDPA", "cross": "FLASH_ATTN"}})
    with _ctx(omni):
        self_layer = _build_attn(AttentionRole.SELF)
        cross_layer = _build_attn(AttentionRole.CROSS)

    assert type(self_layer.attention).__name__ == "SDPAImpl"
    assert type(cross_layer.attention).__name__ == "FlashAttentionImpl"
    _round_trip(self_layer)
    _round_trip(cross_layer)


def test_legacy_attention_backend_real_dispatch():
    """Legacy `attention_backend` field still drives backend selection on GPU."""
    omni = _legacy_omni("TORCH_SDPA")
    with _ctx(omni):
        layer = _build_attn(AttentionRole.SELF)
    assert type(layer.attention).__name__ == "SDPAImpl"
    _round_trip(layer)


# ----- helpers -------------------------------------------------------------- #


from contextlib import contextmanager  # noqa: E402


def _legacy_omni(backend: str) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(attention_backend=backend)


@contextmanager
def _ctx(omni: "OmniDiffusionConfig"):
    with override_forward_context(ForwardContext(omni_diffusion_config=omni)):
        yield


def _build_attn(role) -> Attention:
    layer = Attention(
        num_heads=8,
        head_size=64,
        causal=False,
        softmax_scale=0.125,
        num_kv_heads=8,
        role=role,
    )
    return layer.to("cuda")


def _round_trip(layer: Attention):
    B, S, H, D = 2, 16, 8, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    out = layer(q, k, v)
    assert out.shape == (B, S, H, D)
    assert torch.isfinite(out).all()
