# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention role enumeration for per-role backend dispatch."""

from enum import Enum


class AttentionRole(str, Enum):
    """Semantic role of an attention layer in a DiT block."""

    SELF = "self"
    CROSS = "cross"
    JOINT = "joint"
    OTHER = "other"

    @classmethod
    def coerce(cls, value) -> "AttentionRole":
        """Accept an AttentionRole, a string (case-insensitive), or raise ValueError."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                valid = ", ".join(r.value for r in cls)
                raise ValueError(
                    f"Invalid attention role: {value!r}. Valid roles: {valid}"
                ) from None
        raise ValueError(
            f"Cannot coerce {type(value).__name__} to AttentionRole"
        )
