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
        assert config.params == {}

    def test_from_dict_flat(self):
        """Flat JSON: unknown keys auto-routed into params."""
        config = DiffusionSparseAttnConfig.from_dict(
            {
                "backend": "flashinfer",
                "topk": 0.3,
                "block_size": 128,
            }
        )
        assert config.backend == "flashinfer"
        assert config.params == {"topk": 0.3, "block_size": 128}

    def test_from_dict_nested(self):
        """Nested JSON: explicit params key."""
        config = DiffusionSparseAttnConfig.from_dict(
            {
                "backend": "spargeattn",
                "params": {"topk": 0.5},
            }
        )
        assert config.backend == "spargeattn"
        assert config.params == {"topk": 0.5}

    def test_from_dict_ignores_backend_in_params(self):
        config = DiffusionSparseAttnConfig.from_dict(
            {
                "backend": "dense",
                "extra_key": 42,
            }
        )
        assert config.backend == "dense"
        assert config.params == {"extra_key": 42}

    def test_from_dict_empty(self):
        config = DiffusionSparseAttnConfig.from_dict({})
        assert config.backend == "auto"
        assert config.params == {}


class TestOmniDiffusionConfigSparseAttn:
    """Test sparse_attn integration in OmniDiffusionConfig."""

    def test_sparse_attn_none_by_default(self):
        config = OmniDiffusionConfig()
        assert config.sparse_attn is None

    def test_sparse_attn_from_dict(self):
        config = OmniDiffusionConfig(sparse_attn={"backend": "flashinfer", "topk": 0.4})
        assert isinstance(config.sparse_attn, DiffusionSparseAttnConfig)
        assert config.sparse_attn.backend == "flashinfer"
        assert config.sparse_attn.params == {"topk": 0.4}


class TestOmniDiffusionConfigFromKwargs:
    """Test OmniDiffusionConfig.from_kwargs with sparse_attn fields."""

    def test_from_kwargs_with_sparse_attn_dict(self):
        config = OmniDiffusionConfig.from_kwargs(sparse_attn={"backend": "flashinfer", "topk": 0.3})
        assert isinstance(config.sparse_attn, DiffusionSparseAttnConfig)
        assert config.sparse_attn.backend == "flashinfer"
        assert config.sparse_attn.params == {"topk": 0.3}

    def test_from_kwargs_no_sparse_attn(self):
        config = OmniDiffusionConfig.from_kwargs()
        assert config.sparse_attn is None
