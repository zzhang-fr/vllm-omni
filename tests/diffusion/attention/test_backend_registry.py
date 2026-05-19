# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the open-registry path on `DiffusionAttentionBackendEnum`.

Covers the `CUSTOM_ATTN` slot and the `register_diffusion_backend(name, cls)`
mechanism for third-party plugin backends, plus selector integration
(``per_role.<role>.backend=<plugin-name>`` round-trip).
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.registry import (
    _DIFFUSION_ATTN_OVERRIDES,
    _DIFFUSION_ATTN_REGISTRY,
    DiffusionAttentionBackendEnum,
    register_diffusion_backend,
    resolve_registered_backend,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# -- Test doubles -----------------------------------------------------------


class _FakeImpl(AttentionImpl):
    def __init__(self, *args, **kwargs):
        pass


class _FakeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "_FAKE_BACKEND"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return _FakeImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []


class _SecondFakeBackend(_FakeBackend):
    @staticmethod
    def get_name() -> str:
        return "_SECOND_FAKE_BACKEND"


class _NotABackend:
    pass


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot/restore the registry around each test so they don't bleed."""
    overrides_snapshot = dict(_DIFFUSION_ATTN_OVERRIDES)
    registry_snapshot = dict(_DIFFUSION_ATTN_REGISTRY)
    yield
    _DIFFUSION_ATTN_OVERRIDES.clear()
    _DIFFUSION_ATTN_OVERRIDES.update(overrides_snapshot)
    _DIFFUSION_ATTN_REGISTRY.clear()
    _DIFFUSION_ATTN_REGISTRY.update(registry_snapshot)


# -- CUSTOM_ATTN slot -------------------------------------------------------


class TestCustomAttnSlot:
    def test_custom_attn_member_exists(self):
        assert hasattr(DiffusionAttentionBackendEnum, "CUSTOM_ATTN")

    def test_custom_attn_default_path_empty(self):
        assert DiffusionAttentionBackendEnum.CUSTOM_ATTN.value == ""

    def test_custom_attn_get_path_unregistered_raises(self):
        with pytest.raises(ValueError, match="must be registered"):
            DiffusionAttentionBackendEnum.CUSTOM_ATTN.get_path()

    def test_custom_attn_override_then_resolve(self):
        register_diffusion_backend(
            DiffusionAttentionBackendEnum.CUSTOM_ATTN, _FakeBackend
        )
        assert (
            DiffusionAttentionBackendEnum.CUSTOM_ATTN.get_class() is _FakeBackend
        )


# -- Named-plugin registration ---------------------------------------------


class TestNamedRegistration:
    def test_register_by_class(self):
        register_diffusion_backend("my_kernel", _FakeBackend)
        assert resolve_registered_backend("my_kernel") == (
            f"{_FakeBackend.__module__}.{_FakeBackend.__qualname__}"
        )

    def test_register_by_path_string(self):
        path = f"{_FakeBackend.__module__}.{_FakeBackend.__qualname__}"
        register_diffusion_backend("my_kernel", path)
        assert resolve_registered_backend("my_kernel") == path

    def test_register_as_decorator(self):
        decorator = register_diffusion_backend("my_kernel")

        @decorator
        class _LocalBackend(_FakeBackend):
            @staticmethod
            def get_name() -> str:
                return "_LOCAL"

        assert resolve_registered_backend("my_kernel") == (
            f"{_LocalBackend.__module__}.{_LocalBackend.__qualname__}"
        )

    def test_lookup_is_case_insensitive(self):
        register_diffusion_backend("My_Kernel", _FakeBackend)
        assert resolve_registered_backend("my_kernel") is not None
        assert resolve_registered_backend("MY_KERNEL") is not None
        assert resolve_registered_backend("My_Kernel") is not None

    def test_idempotent_same_path(self):
        # Registering the same class twice is a no-op.
        register_diffusion_backend("my_kernel", _FakeBackend)
        register_diffusion_backend("my_kernel", _FakeBackend)
        assert resolve_registered_backend("my_kernel") == (
            f"{_FakeBackend.__module__}.{_FakeBackend.__qualname__}"
        )

    def test_duplicate_keeps_first(self):
        # vllm.logger isn't captured by pytest caplog, so we assert the
        # outcome (first-wins) rather than the warning text.
        register_diffusion_backend("my_kernel", _FakeBackend)
        register_diffusion_backend("my_kernel", _SecondFakeBackend)
        # First registration kept.
        assert resolve_registered_backend("my_kernel") == (
            f"{_FakeBackend.__module__}.{_FakeBackend.__qualname__}"
        )


class TestNameShadowing:
    @pytest.mark.parametrize("builtin", ["FLASH_ATTN", "TORCH_SDPA", "SAGE_ATTN", "CUSTOM_ATTN"])
    def test_cannot_shadow_builtin(self, builtin):
        with pytest.raises(ValueError, match="shadows built-in"):
            register_diffusion_backend(builtin, _FakeBackend)

    def test_cannot_shadow_builtin_lowercase(self):
        with pytest.raises(ValueError, match="shadows built-in"):
            register_diffusion_backend("flash_attn", _FakeBackend)

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            register_diffusion_backend("", _FakeBackend)


class TestRegistrationValidation:
    def test_non_backend_class_rejected(self):
        with pytest.raises(TypeError, match="not a subclass of AttentionBackend"):
            register_diffusion_backend("my_kernel", _NotABackend)

    def test_skip_validation_when_class_unimportable(self):
        # Path that can't be imported — should pass with validate=False
        register_diffusion_backend(
            "deferred_kernel", "definitely.not.a.real.Module", validate=False
        )
        assert resolve_registered_backend("deferred_kernel") == "definitely.not.a.real.Module"

    def test_validation_runs_on_path_string(self):
        with pytest.raises(ValueError, match="failed to import"):
            register_diffusion_backend("bad_path", "definitely.not.a.real.Module")

    def test_invalid_first_arg_type(self):
        with pytest.raises(TypeError, match="must be a DiffusionAttentionBackendEnum or str"):
            register_diffusion_backend(123, _FakeBackend)  # type: ignore[arg-type]


# -- Selector integration --------------------------------------------------


class TestSelectorIntegration:
    def test_selector_resolves_registered_name(self):
        from vllm_omni.diffusion.attention.selector import _cached_get_backend_cls

        # Cache must be cleared between tests since the registry changes.
        _cached_get_backend_cls.cache_clear()

        register_diffusion_backend("integration_kernel", _FakeBackend)
        cls = _cached_get_backend_cls("integration_kernel", 64)
        assert cls is _FakeBackend

    def test_selector_per_role_resolves_plugin(self):
        from vllm_omni.diffusion.attention.selector import (
            _cached_get_backend_cls,
            get_attn_backend_for_role,
        )
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

        _cached_get_backend_cls.cache_clear()
        register_diffusion_backend("per_role_kernel", _FakeBackend)

        config = AttentionConfig(
            per_role={"self": AttentionSpec(backend="per_role_kernel")}
        )
        backend_cls, spec = get_attn_backend_for_role(
            role="self", head_size=64, attention_config=config
        )
        assert backend_cls is _FakeBackend
        assert spec is not None
        assert spec.backend == "per_role_kernel"

    def test_builtin_still_works(self):
        from vllm_omni.diffusion.attention.selector import _cached_get_backend_cls

        _cached_get_backend_cls.cache_clear()

        # SDPA is always available on every platform — safe to resolve.
        cls = _cached_get_backend_cls("TORCH_SDPA", 64)
        assert issubclass(cls, AttentionBackend)


# -- Enum metaclass error message includes registered names ---------------


class TestEnumErrorIncludesRegistered:
    def test_unknown_name_lists_registered(self):
        register_diffusion_backend("listed_plugin", _FakeBackend)
        with pytest.raises(ValueError, match="listed_plugin"):
            DiffusionAttentionBackendEnum["DEFINITELY_NOT_A_BACKEND"]
