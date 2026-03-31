# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for sparse attention config parsing.

Tests:
- DiffusionSparseAttnConfig dataclass and from_dict factory
- OmniDiffusionConfig integration with sparse_attn
"""

import pytest

from vllm_omni.diffusion.data import DiffusionSparseAttnConfig, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestDiffusionSparseAttnConfig:
    """Test DiffusionSparseAttnConfig dataclass."""

    def test_defaults(self):
        config = DiffusionSparseAttnConfig()
        assert config.backend == "auto"
        assert config.topk_ratio == 0.5

    def test_from_dict(self):
        config = DiffusionSparseAttnConfig.from_dict(
            {
                "backend": "flashinfer",
                "topk_ratio": 0.3,
            }
        )
        assert config.backend == "flashinfer"
        assert config.topk_ratio == 0.3

    def test_from_dict_ignores_unknown(self):
        config = DiffusionSparseAttnConfig.from_dict(
            {
                "backend": "dense",
                "unknown_field": 42,
            }
        )
        assert config.backend == "dense"

    def test_from_dict_empty(self):
        config = DiffusionSparseAttnConfig.from_dict({})
        assert config.backend == "auto"


class TestOmniDiffusionConfigSparseAttn:
    """Test sparse_attn integration in OmniDiffusionConfig."""

    def test_sparse_attn_none_by_default(self):
        config = OmniDiffusionConfig()
        assert config.sparse_attn is None

    def test_sparse_attn_from_dict(self):
        config = OmniDiffusionConfig(sparse_attn={"backend": "flashinfer", "topk_ratio": 0.4})
        assert isinstance(config.sparse_attn, DiffusionSparseAttnConfig)
        assert config.sparse_attn.backend == "flashinfer"
        assert config.sparse_attn.topk_ratio == 0.4


class TestOmniDiffusionConfigFromKwargs:
    """Test OmniDiffusionConfig.from_kwargs with sparse_attn fields."""

    def test_from_kwargs_with_sparse_attn_dict(self):
        config = OmniDiffusionConfig.from_kwargs(sparse_attn={"backend": "flashinfer", "topk_ratio": 0.3})
        assert isinstance(config.sparse_attn, DiffusionSparseAttnConfig)
        assert config.sparse_attn.backend == "flashinfer"
        assert config.sparse_attn.topk_ratio == 0.3

    def test_from_kwargs_no_sparse_attn(self):
        config = OmniDiffusionConfig.from_kwargs()
        assert config.sparse_attn is None
