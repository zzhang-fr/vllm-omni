# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AttentionSpec / AttentionConfig (RFC #2632 P1)."""

import pytest

from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

# ----- AttentionSpec ----------------------------------------------------------


def test_spec_basic():
    spec = AttentionSpec(backend="FLASH_ATTN")
    assert spec.backend == "FLASH_ATTN"
    assert spec.extra == {}


def test_spec_with_extra():
    spec = AttentionSpec(backend="X", extra={"k": 1})
    assert spec.extra == {"k": 1}


def test_spec_coerce_from_string():
    spec = AttentionSpec.coerce("FLASH_ATTN")
    assert spec.backend == "FLASH_ATTN"
    assert spec.extra == {}


def test_spec_coerce_from_dict():
    spec = AttentionSpec.coerce({"backend": "TORCH_SDPA", "extra": {"quant": "fp8"}})
    assert spec.backend == "TORCH_SDPA"
    assert spec.extra == {"quant": "fp8"}


def test_spec_coerce_dict_missing_backend_raises():
    with pytest.raises(ValueError):
        AttentionSpec.coerce({"extra": {"k": 1}})


def test_spec_coerce_extra_must_be_mapping():
    with pytest.raises(TypeError):
        AttentionSpec.coerce({"backend": "X", "extra": [1, 2, 3]})


def test_spec_coerce_passthrough_existing_spec():
    s1 = AttentionSpec(backend="A")
    assert AttentionSpec.coerce(s1) is s1


# ----- AttentionConfig --------------------------------------------------------


def test_empty_config():
    cfg = AttentionConfig()
    assert cfg.is_empty()
    assert cfg.get("self") is None
    assert cfg.default is None
    assert cfg.per_role == {}


def test_config_default_only():
    cfg = AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN"))
    assert not cfg.is_empty()
    assert cfg.get("self").backend == "FLASH_ATTN"
    assert cfg.get("cross").backend == "FLASH_ATTN"
    assert cfg_get_unknown_role(cfg) == "FLASH_ATTN"


def cfg_get_unknown_role(cfg: AttentionConfig) -> str:
    spec = cfg.get("nonsense")
    assert spec is not None
    return spec.backend


def test_config_per_role_overrides_default():
    cfg = AttentionConfig(
        default=AttentionSpec(backend="TORCH_SDPA"),
        per_role={"self": AttentionSpec(backend="FLASH_ATTN")},
    )
    assert cfg.get("self").backend == "FLASH_ATTN"
    assert cfg.get("cross").backend == "TORCH_SDPA"


def test_config_from_dict_full():
    cfg = AttentionConfig.from_dict(
        {
            "default": "FLASH_ATTN",
            "per_role": {
                "self": {"backend": "SAGE_ATTN"},
                "cross": {"backend": "TORCH_SDPA", "extra": {"flag": True}},
            },
        }
    )
    assert cfg.default.backend == "FLASH_ATTN"
    assert cfg.get("self").backend == "SAGE_ATTN"
    cross = cfg.get("cross")
    assert cross.backend == "TORCH_SDPA"
    assert cross.extra == {"flag": True}


def test_config_from_dict_empty():
    assert AttentionConfig.from_dict(None).is_empty()
    assert AttentionConfig.from_dict({}).is_empty()


def test_config_from_dict_invalid_per_role_type():
    with pytest.raises(TypeError):
        AttentionConfig.from_dict({"per_role": [("self", "FLASH_ATTN")]})


def test_config_from_legacy():
    cfg = AttentionConfig.from_legacy("FLASH_ATTN")
    assert cfg.default.backend == "FLASH_ATTN"
    assert cfg.per_role == {}
    # Empty / None legacy returns empty config
    assert AttentionConfig.from_legacy(None).is_empty()
    assert AttentionConfig.from_legacy("").is_empty()


def test_parse_cli_single_backend():
    cfg = AttentionConfig.parse_cli("FLASH_ATTN")
    assert cfg.default.backend == "FLASH_ATTN"
    assert cfg.per_role == {}


def test_parse_cli_per_role():
    cfg = AttentionConfig.parse_cli("self=SAGE_ATTN,cross=TORCH_SDPA")
    assert cfg.default is None
    assert cfg.get("self").backend == "SAGE_ATTN"
    assert cfg.get("cross").backend == "TORCH_SDPA"


def test_parse_cli_mixed():
    cfg = AttentionConfig.parse_cli("FLASH_ATTN,cross=TORCH_SDPA")
    assert cfg.default.backend == "FLASH_ATTN"
    assert cfg.get("self").backend == "FLASH_ATTN"  # falls through to default
    assert cfg.get("cross").backend == "TORCH_SDPA"


def test_parse_cli_empty_returns_empty():
    assert AttentionConfig.parse_cli(None).is_empty()
    assert AttentionConfig.parse_cli("").is_empty()
    assert AttentionConfig.parse_cli("   ").is_empty()


def test_parse_cli_unknown_role_raises():
    with pytest.raises(ValueError):
        AttentionConfig.parse_cli("bogus=FLASH_ATTN")


def test_parse_cli_two_defaults_raises():
    with pytest.raises(ValueError):
        AttentionConfig.parse_cli("FLASH_ATTN,TORCH_SDPA")


# ----- OmniDiffusionConfig integration --------------------------------------


def test_omni_config_legacy_attention_backend_migrates(monkeypatch):
    # Avoid heavy port-allocation logic in __post_init__ for this unit test.
    monkeypatch.setattr(
        "vllm_omni.diffusion.utils.network_utils.find_unused_port",
        lambda *a, **k: 0,
        raising=False,
    )
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    cfg = OmniDiffusionConfig(attention_backend="FLASH_ATTN")
    assert isinstance(cfg.attention, AttentionConfig)
    assert cfg.attention.default.backend == "FLASH_ATTN"
    # mirrored back to attention_backend (legacy alias)
    assert cfg.attention_backend == "FLASH_ATTN"


def test_omni_config_attention_dict_replaces_legacy():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    cfg = OmniDiffusionConfig(attention={"per_role": {"self": "SAGE_ATTN", "cross": "TORCH_SDPA"}})
    assert isinstance(cfg.attention, AttentionConfig)
    assert cfg.attention.get("self").backend == "SAGE_ATTN"
    assert cfg.attention.get("cross").backend == "TORCH_SDPA"


def test_omni_config_no_attention_is_empty():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    cfg = OmniDiffusionConfig()
    assert isinstance(cfg.attention, AttentionConfig)
    assert cfg.attention.is_empty()


def test_omni_config_attention_str_parses_cli_form():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    cfg = OmniDiffusionConfig(attention="self=FLASH_ATTN,cross=TORCH_SDPA")
    assert cfg.attention.get("self").backend == "FLASH_ATTN"
    assert cfg.attention.get("cross").backend == "TORCH_SDPA"
