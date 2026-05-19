# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the typed per-forward metadata layer (RFC #3715 Proposal B).

Covers ``DiffusionAttentionMetadata`` defaults, subclass identity,
``ForwardContext.total_denoise_steps`` plumbing, and ``Attention._with_diffusion_step_state``
auto-upgrade behaviour.
"""

from __future__ import annotations

import torch
import pytest

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    DiffusionAttentionMetadata,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# -- Dataclass shape --------------------------------------------------------


class TestDataclassShape:
    def test_is_subclass(self):
        assert issubclass(DiffusionAttentionMetadata, AttentionMetadata)

    def test_defaults_none(self):
        m = DiffusionAttentionMetadata()
        assert m.denoising_step is None
        assert m.total_steps is None
        assert m.total_latent_frames is None
        assert m.patches_per_frame is None
        assert m.encoder_seq_len is None

    def test_parent_fields_preserved(self):
        m = DiffusionAttentionMetadata()
        # Inherited from AttentionMetadata
        assert m.attn_mask is None
        assert m.joint_strategy == "front"
        assert m.extra == {}

    def test_explicit_construction(self):
        m = DiffusionAttentionMetadata(
            denoising_step=7,
            total_steps=50,
            total_latent_frames=33,
            patches_per_frame=256,
            encoder_seq_len=512,
            extra={"topk": 0.3},
        )
        assert m.denoising_step == 7
        assert m.total_steps == 50
        assert m.total_latent_frames == 33
        assert m.patches_per_frame == 256
        assert m.encoder_seq_len == 512
        assert m.extra == {"topk": 0.3}


# -- ForwardContext plumbing ------------------------------------------------


class TestForwardContextPlumbing:
    def test_total_denoise_steps_default(self):
        from vllm_omni.diffusion.forward_context import ForwardContext

        ctx = ForwardContext()
        assert ctx.total_denoise_steps is None

    def test_set_total_denoise_steps_helper(self):
        from vllm_omni.diffusion.forward_context import (
            ForwardContext,
            override_forward_context,
            set_forward_context_total_denoise_steps,
        )

        with override_forward_context(ForwardContext()):
            set_forward_context_total_denoise_steps(42)
            from vllm_omni.diffusion.forward_context import get_forward_context

            assert get_forward_context().total_denoise_steps == 42

    def test_set_total_denoise_steps_no_context_safe(self):
        """Setter is a no-op when no forward context is active."""
        from vllm_omni.diffusion.forward_context import (
            set_forward_context_total_denoise_steps,
        )

        # Must not raise even without an active context.
        set_forward_context_total_denoise_steps(10)

    def test_create_forward_context_accepts_total_steps(self):
        from vllm_omni.diffusion.forward_context import create_forward_context

        ctx = create_forward_context(total_denoise_steps=99)
        assert ctx.total_denoise_steps == 99


# -- Attention._with_diffusion_step_state auto-upgrade ---------------------


def _patched_context(step_idx=None, total=None):
    """Context manager that activates a ForwardContext with given step state."""
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
    )

    return override_forward_context(
        ForwardContext(denoise_step_idx=step_idx, total_denoise_steps=total)
    )


class TestStepStateUpgrade:
    def _call(self, attn_metadata, step_idx=None, total=None):
        from vllm_omni.diffusion.attention.layer import Attention

        with _patched_context(step_idx, total):
            return Attention._with_diffusion_step_state(attn_metadata)

    def test_no_context_returns_metadata_unchanged(self):
        # No forward context → passthrough.
        from vllm_omni.diffusion.attention.layer import Attention

        meta = AttentionMetadata(extra={"foo": "bar"})
        assert Attention._with_diffusion_step_state(meta) is meta

    def test_context_without_step_state_passthrough(self):
        meta = AttentionMetadata(extra={"foo": "bar"})
        out = self._call(meta, step_idx=None, total=None)
        assert out is meta

    def test_none_input_with_step_state_creates_diffusion_meta(self):
        out = self._call(None, step_idx=5, total=50)
        assert isinstance(out, DiffusionAttentionMetadata)
        assert out.denoising_step == 5
        assert out.total_steps == 50

    def test_attention_metadata_upgraded_preserves_parent_fields(self):
        mask = torch.zeros(2, 4)
        meta = AttentionMetadata(
            attn_mask=mask,
            joint_strategy="rear",
            extra={"topk": 0.3},
            full_attn_spans=[[(0, 4)]],
        )
        out = self._call(meta, step_idx=3, total=20)
        assert isinstance(out, DiffusionAttentionMetadata)
        # New typed fields populated
        assert out.denoising_step == 3
        assert out.total_steps == 20
        # Geometry fields untouched (model author's responsibility)
        assert out.total_latent_frames is None
        # Parent fields fully preserved
        assert out.attn_mask is mask
        assert out.joint_strategy == "rear"
        assert out.extra == {"topk": 0.3}
        assert out.full_attn_spans == [[(0, 4)]]

    def test_diffusion_metadata_input_uses_replace(self):
        """When input is already DiffusionAttentionMetadata, dataclasses.replace path runs."""
        meta = DiffusionAttentionMetadata(
            total_latent_frames=33,
            patches_per_frame=256,
            extra={"x": 1},
        )
        out = self._call(meta, step_idx=7, total=50)
        assert isinstance(out, DiffusionAttentionMetadata)
        assert out.denoising_step == 7
        assert out.total_steps == 50
        # Geometry preserved (was set by "model author")
        assert out.total_latent_frames == 33
        assert out.patches_per_frame == 256
        assert out.extra == {"x": 1}

    def test_partial_step_state_only_step_idx(self):
        out = self._call(None, step_idx=4, total=None)
        assert isinstance(out, DiffusionAttentionMetadata)
        assert out.denoising_step == 4
        assert out.total_steps is None

    def test_partial_step_state_only_total(self):
        out = self._call(None, step_idx=None, total=50)
        assert isinstance(out, DiffusionAttentionMetadata)
        assert out.denoising_step is None
        assert out.total_steps == 50
