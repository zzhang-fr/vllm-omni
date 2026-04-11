# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention role enum for per-role attention backend configuration.

See RFC #2632. ``AttentionRole`` lets a model tag each ``Attention`` layer with
its semantic role (self / cross / joint) so the selector can dispatch to a
different backend per role at construction time.

Inheriting from ``str`` keeps the enum YAML/JSON friendly and lets it be
compared directly with plain strings (``role == "self"``).
"""

from enum import Enum


class AttentionRole(str, Enum):
    """Semantic role of an attention layer.

    Values are the lowercase strings that appear in user-facing config
    (`per_role: {self: ..., cross: ...}`).
    """

    SELF = "self"
    CROSS = "cross"
    JOINT = "joint"
    OTHER = "other"

    @classmethod
    def coerce(cls, value: "AttentionRole | str") -> "AttentionRole":
        """Convert a string or AttentionRole into an AttentionRole.

        Accepts either an existing enum member or a case-insensitive string.
        Raises ``ValueError`` for unknown roles so misconfigurations fail loudly.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value.lower())
        raise TypeError(f"Cannot coerce {type(value).__name__} to AttentionRole")
