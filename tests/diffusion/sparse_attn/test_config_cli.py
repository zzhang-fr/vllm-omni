# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for sparse attention config wiring.

Tests:
- OmniDiffusionConfig.from_kwargs with sparse_attn dict
- DiffusionSparseAttnConfig construction with params dict
- sparse_attn_backend field removed
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.data import DiffusionSparseAttnConfig, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestFromKwargsWithSparseAttn:
    """Test OmniDiffusionConfig.from_kwargs with sparse attn config."""

    def test_sparse_attn_dict_passthrough(self):
        config = OmniDiffusionConfig.from_kwargs(
            sparse_attn={"backend": "flashinfer", "topk": 0.4},
        )
        assert isinstance(config.sparse_attn, DiffusionSparseAttnConfig)
        assert config.sparse_attn.backend == "flashinfer"
        assert config.sparse_attn.params == {"topk": 0.4}

    def test_no_sparse_attn(self):
        config = OmniDiffusionConfig.from_kwargs()
        assert config.sparse_attn is None

    def test_sparse_attn_backend_field_removed(self):
        """sparse_attn_backend field no longer exists on OmniDiffusionConfig."""
        assert not hasattr(OmniDiffusionConfig, "sparse_attn_backend")
        config = OmniDiffusionConfig()
        assert not hasattr(config, "sparse_attn_backend")

    def test_sparse_attn_config_from_dict(self):
        config = OmniDiffusionConfig(
            sparse_attn={"backend": "dense", "topk": 0.3},
        )
        assert isinstance(config.sparse_attn, DiffusionSparseAttnConfig)
        assert config.sparse_attn.backend == "dense"
        assert config.sparse_attn.params == {"topk": 0.3}


class TestEndToEndConfigFlow:
    """Test the full config flow for sparse attention."""

    def test_sparse_attn_via_dict(self):
        """Simulate: --sparse-attn '{"backend":"spargeattn","topk":0.5}'."""
        sparse_config = {"backend": "spargeattn", "topk": 0.5}
        config = OmniDiffusionConfig.from_kwargs(sparse_attn=sparse_config)

        assert isinstance(config.sparse_attn, DiffusionSparseAttnConfig)
        assert config.sparse_attn.backend == "spargeattn"
        assert config.sparse_attn.params == {"topk": 0.5}

    def test_no_sparse_args_preserves_none(self):
        config = OmniDiffusionConfig.from_kwargs()
        assert config.sparse_attn is None
