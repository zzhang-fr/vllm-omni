# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the role-aware get_attn_backend selector (RFC #2632 P2)."""

import pytest

from vllm_omni.diffusion.attention.role import AttentionRole
from vllm_omni.diffusion.attention.selector import get_attn_backend
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec


class _DummyBackend:
    """Marker class returned by the patched loader."""

    @staticmethod
    def get_name() -> str:
        return "DUMMY"

    @staticmethod
    def get_impl_cls():
        return None


def _patch(monkeypatch, recorder: list[str | None]):
    """Replace platform + backend loader so the selector is fully isolated.

    `recorder` collects every `selected_backend` value the platform sees.
    """

    class _StubPlatform:
        @staticmethod
        def get_diffusion_attn_backend_cls(*, selected_backend, head_size):
            recorder.append(selected_backend)
            return "tests.diffusion.attention.test_selector_per_role._DummyBackend"

    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform", _StubPlatform, raising=True)
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.selector._load_backend_cls",
        lambda path: _DummyBackend,
    )


# --------------------------------------------------------------------------- #


def test_per_role_dispatch(monkeypatch):
    seen: list[str | None] = []
    _patch(monkeypatch, seen)

    cfg = AttentionConfig(
        per_role={
            "self": AttentionSpec(backend="FLASH_ATTN"),
            "cross": AttentionSpec(backend="TORCH_SDPA"),
        }
    )
    cls_self, spec_self = get_attn_backend(AttentionRole.SELF, head_size=64, config=cfg)
    cls_cross, spec_cross = get_attn_backend(AttentionRole.CROSS, head_size=64, config=cfg)

    assert cls_self is _DummyBackend
    assert cls_cross is _DummyBackend
    assert spec_self.backend == "FLASH_ATTN"
    assert spec_cross.backend == "TORCH_SDPA"
    assert seen == ["FLASH_ATTN", "TORCH_SDPA"]


def test_fallback_to_default(monkeypatch):
    seen: list[str | None] = []
    _patch(monkeypatch, seen)

    cfg = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))
    _, spec_self = get_attn_backend(AttentionRole.SELF, head_size=64, config=cfg)
    _, spec_cross = get_attn_backend(AttentionRole.CROSS, head_size=64, config=cfg)

    assert spec_self.backend == "FLASH_ATTN"
    assert spec_cross.backend == "FLASH_ATTN"
    assert seen == ["FLASH_ATTN", "FLASH_ATTN"]


def test_fallback_to_platform_default(monkeypatch):
    monkeypatch.delenv("DIFFUSION_ATTENTION_BACKEND", raising=False)
    seen: list = []
    _patch(monkeypatch, seen)

    cls, spec = get_attn_backend(AttentionRole.SELF, head_size=64, config=AttentionConfig())
    assert cls is _DummyBackend
    assert spec.backend == "auto"
    assert seen == [None]


def test_env_var_used_when_no_config(monkeypatch):
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    seen: list = []
    _patch(monkeypatch, seen)

    get_attn_backend(AttentionRole.SELF, head_size=64, config=AttentionConfig())
    assert seen == ["TORCH_SDPA"]


def test_config_overrides_env(monkeypatch):
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    seen: list = []
    _patch(monkeypatch, seen)

    cfg = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))
    get_attn_backend(AttentionRole.SELF, head_size=64, config=cfg)
    assert seen == ["FLASH_ATTN"]


def test_role_accepts_string(monkeypatch):
    seen: list = []
    _patch(monkeypatch, seen)

    cfg = AttentionConfig(per_role={"cross": AttentionSpec(backend="TORCH_SDPA")})
    get_attn_backend("cross", head_size=64, config=cfg)
    assert seen == ["TORCH_SDPA"]


def test_no_cache_role_distinguished(monkeypatch):
    """No @cache: switching role on identical (head_size, config) must re-dispatch."""
    seen: list = []
    _patch(monkeypatch, seen)

    cfg = AttentionConfig(
        per_role={
            "self": AttentionSpec(backend="FLASH_ATTN"),
            "cross": AttentionSpec(backend="TORCH_SDPA"),
        }
    )
    get_attn_backend(AttentionRole.SELF, head_size=64, config=cfg)
    get_attn_backend(AttentionRole.CROSS, head_size=64, config=cfg)
    get_attn_backend(AttentionRole.SELF, head_size=64, config=cfg)
    assert seen == ["FLASH_ATTN", "TORCH_SDPA", "FLASH_ATTN"]


def test_no_config_role_none(monkeypatch):
    """role=None and config=None -> platform default fallthrough."""
    seen: list = []
    _patch(monkeypatch, seen)
    monkeypatch.delenv("DIFFUSION_ATTENTION_BACKEND", raising=False)

    cls, spec = get_attn_backend(role=None, head_size=64, config=None)
    assert cls is _DummyBackend
    assert spec.backend == "auto"
    assert seen == [None]


def test_invalid_config_type_raises(monkeypatch):
    seen: list = []
    _patch(monkeypatch, seen)

    with pytest.raises(TypeError):
        get_attn_backend(AttentionRole.SELF, head_size=64, config={"oops": True})  # type: ignore[arg-type]


def test_extra_dict_propagates_in_spec(monkeypatch):
    seen: list = []
    _patch(monkeypatch, seen)

    cfg = AttentionConfig(per_role={"self": AttentionSpec(backend="FLASH_ATTN", extra={"custom": 1})})
    _, spec = get_attn_backend(AttentionRole.SELF, head_size=64, config=cfg)
    assert spec.extra == {"custom": 1}
