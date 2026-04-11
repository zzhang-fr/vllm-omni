# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``attention=`` kwarg normalization in AsyncOmniEngine
(RFC #2632).

Covers the YAML-stage-config override path that the real-weights E2E tests
cannot reach without loading a model: we verify that an ``AttentionConfig``
dataclass coming from ``Omni(attention=...)`` can be assigned into an
OmegaConf ``DictConfig`` stage cfg without raising.
"""

from omegaconf import OmegaConf

from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.engine.async_omni_engine import _normalize_attention_kwarg


def test_normalize_none_passthrough():
    assert _normalize_attention_kwarg(None) is None


def test_normalize_str_passthrough():
    assert _normalize_attention_kwarg("self=TORCH_SDPA") == "self=TORCH_SDPA"


def test_normalize_dict_passthrough():
    d = {"per_role": {"self": "TORCH_SDPA"}}
    assert _normalize_attention_kwarg(d) is d


def test_normalize_attention_config_flattens_to_plain_dict():
    cfg = AttentionConfig_factory(default="FLASH_ATTN", cross=("TORCH_SDPA", {"k": 1}))
    flat = _normalize_attention_kwarg(cfg)

    assert isinstance(flat, dict)
    assert flat["default"] == {"backend": "FLASH_ATTN", "extra": {}}
    assert flat["per_role"]["cross"] == {"backend": "TORCH_SDPA", "extra": {"k": 1}}
    # every nested value must be a plain Python primitive
    _assert_plain_dict(flat)


def test_normalized_attention_is_assignable_to_omegaconf():
    """The whole point of normalization: OmegaConf must accept the value
    without raising. This is the exact thing that `_resolve_stage_configs`
    does in the YAML-override branch.
    """
    cfg = OmegaConf.create({"engine_args": {"attention": None}})
    spec = AttentionConfig(per_role={"joint": AttentionSpec(backend="TORCH_SDPA", extra={"flag": True})})
    flat = _normalize_attention_kwarg(spec)
    cfg.engine_args.attention = flat  # must not raise
    assert cfg.engine_args.attention.per_role.joint.backend == "TORCH_SDPA"
    assert cfg.engine_args.attention.per_role.joint.extra.flag is True


def test_normalize_str_roundtrip_via_omegaconf():
    """Plain string form (`"self=FLASH_ATTN,cross=TORCH_SDPA"`) is also
    accepted — the downstream OmniDiffusionConfig parses the CLI string."""
    cfg = OmegaConf.create({"engine_args": {"attention": None}})
    flat = _normalize_attention_kwarg("self=FLASH_ATTN,cross=TORCH_SDPA")
    cfg.engine_args.attention = flat
    assert cfg.engine_args.attention == "self=FLASH_ATTN,cross=TORCH_SDPA"


# --- helpers ---------------------------------------------------------------


def AttentionConfig_factory(*, default=None, **per_role) -> AttentionConfig:  # noqa: N802
    """Small factory to keep the test bodies noise-free."""
    return AttentionConfig(
        default=(AttentionSpec(backend=default) if default else None),
        per_role={
            role: (
                AttentionSpec(backend=spec[0], extra=spec[1])
                if isinstance(spec, tuple)
                else AttentionSpec(backend=spec)
            )
            for role, spec in per_role.items()
        },
    )


def _assert_plain_dict(obj):
    """Recursively assert that every leaf value is a primitive or dict/list."""
    allowed = (int, float, str, bool, type(None))
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_plain_dict(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_plain_dict(v)
    else:
        assert isinstance(obj, allowed), f"non-primitive leaf: {type(obj).__name__}"
