# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end role → config → backend integration test (RFC #2632 P3).

Verifies the full plumbing without loading model weights:

  AttentionConfig (in OmniDiffusionConfig)
        │
   forward_context.omni_diffusion_config
        │
   Attention(role=...).__init__
        │
   get_attn_backend(role, head_size, config)
        │
   AttentionImpl with the right backend_kwargs

Uses the SDPA backend (no GPU/library deps) so this runs in CI on any host.
The Wan2.2-weights inference path is in `test_e2e_per_role.py`
(gated on the `VLLM_OMNI_E2E_MODEL` env var).
"""

from contextlib import contextmanager

from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.role import AttentionRole
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    override_forward_context,
)


@contextmanager
def _forward_ctx(omni: OmniDiffusionConfig):
    ctx = ForwardContext(omni_diffusion_config=omni)
    with override_forward_context(ctx):
        yield


def _make_attn(role) -> Attention:
    return Attention(
        num_heads=4,
        head_size=64,
        causal=False,
        softmax_scale=1.0 / 8.0,
        num_kv_heads=4,
        role=role,
    )


# --------------------------------------------------------------------------- #


def test_per_role_dispatch_through_forward_context():
    omni = OmniDiffusionConfig(
        attention={
            "per_role": {
                "self": {"backend": "TORCH_SDPA"},
                "cross": {"backend": "TORCH_SDPA"},
            }
        }
    )
    with _forward_ctx(omni):
        self_layer = _make_attn(AttentionRole.SELF)
        cross_layer = _make_attn(AttentionRole.CROSS)
    assert isinstance(self_layer.attention, SDPAImpl)
    assert isinstance(cross_layer.attention, SDPAImpl)
    assert self_layer.role is AttentionRole.SELF
    assert cross_layer.role is AttentionRole.CROSS


def test_default_attention_spec_used_for_all_roles():
    omni = OmniDiffusionConfig(attention={"default": "TORCH_SDPA"})
    with _forward_ctx(omni):
        layer = _make_attn(AttentionRole.SELF)
    assert isinstance(layer.attention, SDPAImpl)


def test_legacy_attention_backend_string_migrated():
    omni = OmniDiffusionConfig(attention_backend="TORCH_SDPA")
    with _forward_ctx(omni):
        layer = _make_attn(AttentionRole.SELF)
    assert isinstance(layer.attention, SDPAImpl)
    assert omni.attention.default.backend == "TORCH_SDPA"


def test_no_attention_config_fallback_succeeds():
    omni = OmniDiffusionConfig()
    with _forward_ctx(omni):
        layer = _make_attn(AttentionRole.CROSS)
    assert layer.role is AttentionRole.CROSS
    assert layer.attention is not None


def test_backend_kwargs_propagate_to_impl():
    omni = OmniDiffusionConfig(
        attention={
            "per_role": {
                "self": {
                    "backend": "TORCH_SDPA",
                    "extra": {"custom_flag": True, "n": 7},
                },
            }
        }
    )
    with _forward_ctx(omni):
        layer = _make_attn(AttentionRole.SELF)
    assert isinstance(layer.attention, SDPAImpl)
    assert layer.attention._backend_kwargs == {"custom_flag": True, "n": 7}


def test_string_attention_field_parses_cli_form():
    omni = OmniDiffusionConfig(attention="self=TORCH_SDPA,cross=TORCH_SDPA")
    with _forward_ctx(omni):
        s = _make_attn(AttentionRole.SELF)
        c = _make_attn(AttentionRole.CROSS)
    assert isinstance(s.attention, SDPAImpl)
    assert isinstance(c.attention, SDPAImpl)


def test_attention_layer_default_role_self():
    """When constructed without role= and without forward context, it defaults to SELF."""
    layer = Attention(
        num_heads=4,
        head_size=64,
        causal=False,
        softmax_scale=1.0 / 8.0,
        num_kv_heads=4,
    )
    assert layer.role is AttentionRole.SELF
