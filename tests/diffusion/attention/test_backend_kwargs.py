# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that backend_kwargs flow through AttentionImpl base + subclasses (RFC #2632 P1)."""

import pytest

from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

# FlashAttention may not be available on every machine — only import lazily.
try:
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    _HAVE_FLASH = True
except Exception:
    _HAVE_FLASH = False


@pytest.fixture
def common_kwargs():
    return dict(num_heads=4, head_size=64, softmax_scale=0.125)


def test_sdpa_stores_backend_kwargs(common_kwargs):
    impl = SDPAImpl(**common_kwargs, backend_kwargs={"quant": "int8"})
    assert impl._backend_kwargs == {"quant": "int8"}


def test_sdpa_default_backend_kwargs_is_empty_dict(common_kwargs):
    impl = SDPAImpl(**common_kwargs)
    assert impl._backend_kwargs == {}
    # Ensure each instance gets its own dict (no aliasing of class attribute)
    impl2 = SDPAImpl(**common_kwargs)
    impl._backend_kwargs["x"] = 1
    assert "x" not in impl2._backend_kwargs


def test_sdpa_unknown_extra_kwargs_silently_accepted(common_kwargs):
    """Backends should accept unknown init kwargs without error (forward-compat)."""
    SDPAImpl(**common_kwargs, foo="bar", random=123)


def test_sdpa_stores_common_attrs(common_kwargs):
    impl = SDPAImpl(**common_kwargs, num_kv_heads=2)
    assert impl.num_heads == 4
    assert impl.head_size == 64
    assert impl.num_kv_heads == 2
    assert impl.softmax_scale == 0.125
    assert impl_causal(impl) is False


def impl_causal(impl) -> bool:
    return impl.causal


def test_flash_backend_kwargs(common_kwargs):
    if not _HAVE_FLASH:
        pytest.skip("flash-attn impl not importable")
    impl = FlashAttentionImpl(**common_kwargs, backend_kwargs={"sparse_pattern": "block"})
    assert impl._backend_kwargs == {"sparse_pattern": "block"}


def test_base_class_attribute_default_is_empty():
    """The class-level _backend_kwargs default must not leak across instances."""
    # AttentionImpl is abstract via subclasses; SDPAImpl is the concrete sentinel.
    a = SDPAImpl(num_heads=2, head_size=64, softmax_scale=1.0)
    b = SDPAImpl(num_heads=2, head_size=64, softmax_scale=1.0)
    assert a._backend_kwargs == {}
    assert b._backend_kwargs == {}
    a._backend_kwargs["k"] = 1
    assert "k" not in b._backend_kwargs


def test_backend_kwargs_preserved_through_constructor():
    bk = {"a": 1, "b": [1, 2]}
    impl = SDPAImpl(num_heads=4, head_size=64, softmax_scale=1.0, backend_kwargs=bk)
    assert impl._backend_kwargs == {"a": 1, "b": [1, 2]}
    # Defensive top-level copy: adding new keys to bk should not affect impl state.
    bk["new"] = 999
    assert "new" not in impl._backend_kwargs
