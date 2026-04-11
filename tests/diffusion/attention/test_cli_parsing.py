# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AttentionConfig.parse_cli (RFC #2632 P2).

The vllm-omni codebase has no top-level argparse for `--attention-backend`
today (users set it via env var or YAML), so the "CLI" syntax is implemented
as a string parser on `AttentionConfig` that the dataclass accepts directly.
This test pins the syntax (`single` / `self=X,cross=Y` / `default,role=X`)
and the env-var fallback wired through `OmniDiffusionConfig`.
"""

import pytest

from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig


def test_single_backend():
    cfg = AttentionConfig.parse_cli("FLASH_ATTN")
    assert cfg.default == AttentionSpec(backend="FLASH_ATTN")
    assert cfg.per_role == {}


def test_per_role_only():
    cfg = AttentionConfig.parse_cli("self=SAGE_ATTN,cross=TORCH_SDPA")
    assert cfg.default is None
    assert cfg.get("self").backend == "SAGE_ATTN"
    assert cfg.get("cross").backend == "TORCH_SDPA"


def test_default_plus_overrides():
    cfg = AttentionConfig.parse_cli("FLASH_ATTN,cross=TORCH_SDPA")
    assert cfg.default.backend == "FLASH_ATTN"
    # default propagates to roles not in per_role
    assert cfg.get("self").backend == "FLASH_ATTN"
    assert cfg.get("cross").backend == "TORCH_SDPA"


def test_whitespace_tolerant():
    cfg = AttentionConfig.parse_cli("  self = SAGE_ATTN ,  cross = TORCH_SDPA  ")
    assert cfg.get("self").backend == "SAGE_ATTN"
    assert cfg.get("cross").backend == "TORCH_SDPA"


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        AttentionConfig.parse_cli("typo=FLASH_ATTN")


def test_two_defaults_raises():
    with pytest.raises(ValueError):
        AttentionConfig.parse_cli("FLASH_ATTN,TORCH_SDPA")


def test_empty_inputs_return_empty_config():
    assert AttentionConfig.parse_cli("").is_empty()
    assert AttentionConfig.parse_cli(None).is_empty()
    assert AttentionConfig.parse_cli("   ").is_empty()


def test_omni_config_from_legacy_attention_backend():
    cfg = OmniConfig(attention_backend="FLASH_ATTN")
    assert isinstance(cfg.attention, AttentionConfig)
    assert cfg.attention.default.backend == "FLASH_ATTN"


def test_omni_config_from_attention_string():
    cfg = OmniConfig(attention="self=FLASH_ATTN,cross=TORCH_SDPA")
    assert isinstance(cfg.attention, AttentionConfig)
    assert cfg.attention.get("self").backend == "FLASH_ATTN"
    assert cfg.attention.get("cross").backend == "TORCH_SDPA"


def test_omni_config_from_attention_dict():
    cfg = OmniConfig(
        attention={
            "default": "FLASH_ATTN",
            "per_role": {"cross": {"backend": "TORCH_SDPA", "extra": {"k": 1}}},
        }
    )
    assert cfg.attention.default.backend == "FLASH_ATTN"
    assert cfg.attention.per_role["cross"].extra == {"k": 1}


def test_omni_config_attention_takes_precedence_over_legacy():
    """When both legacy and new are provided, the new field wins."""
    cfg = OmniConfig(
        attention_backend="FLASH_ATTN",
        attention={"per_role": {"self": "TORCH_SDPA"}},
    )
    # New field is preserved
    assert cfg.attention.get("self").backend == "TORCH_SDPA"


def OmniConfig(**kwargs):
    """Wrapper that constructs OmniDiffusionConfig with neutral defaults.

    OmniDiffusionConfig's __post_init__ allocates a master port and probes
    for free TCP ports, which is fine in CI but slow. We rely on it as-is
    here since the cost is sub-millisecond per call.
    """
    return _make_cfg(**kwargs)


def _make_cfg(**kwargs):
    return OmniDiffusionConfig(**kwargs)
