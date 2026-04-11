# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify role-tagging on the Attention layer and Wan2.2's call sites (RFC #2632 P3).

We don't try to instantiate the full Wan2.2 model here — that requires the
vLLM TP groups. Instead we:
  1. Cover the `Attention(role=...)` plumbing with a direct unit test (the
     layer is what actually consumes the role).
  2. Statically verify that Wan2.2's self/cross attention modules pass the
     correct role kwarg (so we'll fail fast if someone removes them).
"""

from pathlib import Path

import pytest

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.role import AttentionRole

# ---------------- Direct attention plumbing ---------------------------------


def _make_attn(role) -> Attention:
    return Attention(
        num_heads=4,
        head_size=64,
        causal=False,
        softmax_scale=1.0 / 8.0,
        num_kv_heads=4,
        role=role,
    )


def test_default_role_is_self():
    layer = Attention(num_heads=4, head_size=64, causal=False, softmax_scale=1.0 / 8.0, num_kv_heads=4)
    assert layer.role is AttentionRole.SELF


def test_role_enum_passthrough():
    layer = _make_attn(AttentionRole.CROSS)
    assert layer.role is AttentionRole.CROSS


def test_role_string_coerced():
    layer = _make_attn("joint")
    assert layer.role is AttentionRole.JOINT


def test_invalid_role_raises():
    with pytest.raises((ValueError, TypeError)):
        _make_attn("not_a_role")


# ---------------- Wan2.2 source-level verification --------------------------
# Static check: parse the Wan2.2 transformer file and assert each *Attention
# class constructs `Attention(...)` with the correct `role=` keyword. This
# avoids needing TP groups while still catching accidental removal.


WAN22_FILE = Path(__file__).resolve().parents[3] / ("vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py")


def _attention_role_kwarg_in_class(class_name: str) -> str | None:
    """Return the dotted-name role passed to `Attention(...)` inside the named class.

    Returns None if no `Attention(...)` call is found or no `role=` kwarg.
    """
    src = WAN22_FILE.read_text()
    import ast as _ast

    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == class_name:
            for sub in _ast.walk(node):
                if isinstance(sub, _ast.Call) and (isinstance(sub.func, _ast.Name) and sub.func.id == "Attention"):
                    for kw in sub.keywords:
                        if kw.arg == "role":
                            return _ast.unparse(kw.value)
    return None


def test_wan22_self_attention_passes_self_role():
    role_expr = _attention_role_kwarg_in_class("WanSelfAttention")
    assert role_expr is not None, "WanSelfAttention must pass role= to Attention(...)"
    assert role_expr.endswith(".SELF") or role_expr == "AttentionRole.SELF"


def test_wan22_cross_attention_passes_cross_role():
    role_expr = _attention_role_kwarg_in_class("WanCrossAttention")
    assert role_expr is not None, "WanCrossAttention must pass role= kwarg"
    assert role_expr.endswith(".CROSS") or role_expr == "AttentionRole.CROSS"


def test_wan22_imports_attention_role():
    src = WAN22_FILE.read_text()
    assert "from vllm_omni.diffusion.attention.role import AttentionRole" in src
