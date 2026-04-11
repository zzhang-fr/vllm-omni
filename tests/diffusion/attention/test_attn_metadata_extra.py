# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test AttentionMetadata.extra dict (RFC #2632 P1)."""

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


def test_default_extra_is_empty_dict():
    md = AttentionMetadata()
    assert md.extra == {}


def test_each_instance_has_independent_extra_dict():
    a = AttentionMetadata()
    b = AttentionMetadata()
    a.extra["k"] = 1
    assert "k" not in b.extra


def test_extra_accepts_arbitrary_values():
    block_mask = torch.zeros(2, 2)
    md = AttentionMetadata(extra={"block_mask": block_mask, "extra_int": 5})
    assert md.extra["extra_int"] == 5
    assert torch.equal(md.extra["block_mask"], block_mask)


def test_extra_passthrough_into_existing_fields_unaffected():
    md = AttentionMetadata(attn_mask=torch.ones(1), extra={"k": 1})
    assert md.attn_mask is not None
    assert md.extra == {"k": 1}
