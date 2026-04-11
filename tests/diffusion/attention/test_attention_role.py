# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the AttentionRole enum (RFC #2632 P1)."""

import pytest

from vllm_omni.diffusion.attention.role import AttentionRole


def test_enum_values():
    assert AttentionRole.SELF.value == "self"
    assert AttentionRole.CROSS.value == "cross"
    assert AttentionRole.JOINT.value == "joint"
    assert AttentionRole.OTHER.value == "other"


def test_string_inheritance():
    # str-Enum allows direct equality with plain strings
    assert AttentionRole.SELF == "self"
    assert AttentionRole("self") is AttentionRole.SELF


def test_unknown_value_raises():
    with pytest.raises(ValueError):
        AttentionRole("not_a_role")


def test_coerce_accepts_member_and_string():
    assert AttentionRole.coerce("CROSS") is AttentionRole.CROSS
    assert AttentionRole.coerce("self") is AttentionRole.SELF
    assert AttentionRole.coerce(AttentionRole.JOINT) is AttentionRole.JOINT


def test_coerce_rejects_other_types():
    with pytest.raises(TypeError):
        AttentionRole.coerce(0)  # type: ignore[arg-type]
