# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for generic, model-agnostic per-forward state publication.

Covers the transformer forward pre-hook that keeps per-forward state out of model
code entirely (no ``set_forward_context_*`` call in any pipeline / transformer):
- patch-size normalization across the two in-tree conventions
- the pre-hook publishing raw geometry primitives onto the active ForwardContext
- the denoise loop length read off the pipeline's ``num_timesteps`` convention,
  without clobbering a value a pipeline published explicitly
- best-effort no-ops (no patch size / non-5D input / no active context)
- idempotent registration that fires on a real forward
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.per_forward_publish import (
    _make_per_forward_pre_hook,
    _publish_denoise_total,
    _publish_video_geometry,
    _resolve_patch_size,
    _resolve_total_denoise_steps,
    register_per_forward_hooks,
)
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
    set_forward_context_total_denoise_steps,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TestResolvePatchSize:
    """Normalize patch size to (p_t, p_h, p_w) across model conventions."""

    def test_tuple_from_config(self):
        # Wan: config.patch_size is a 3-tuple.
        module = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))
        assert _resolve_patch_size(module) == (1, 2, 2)

    def test_scalar_plus_temporal_from_attrs(self):
        # HunyuanVideo: scalar patch_size + separate patch_size_t, no .config.
        module = SimpleNamespace(patch_size=2, patch_size_t=4)
        assert _resolve_patch_size(module) == (4, 2, 2)

    def test_scalar_without_temporal_defaults_unpatched_time(self):
        # Most video DiTs do not patch time: a scalar patch_size with no
        # patch_size_t must default to p_t = 1, not an isotropic (2, 2, 2)
        # (which would silently halve the published frame count).
        module = SimpleNamespace(patch_size=2)
        assert _resolve_patch_size(module) == (1, 2, 2)

    def test_config_takes_precedence_over_attr(self):
        module = SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)), patch_size=8)
        assert _resolve_patch_size(module) == (1, 2, 2)

    def test_none_when_absent(self):
        assert _resolve_patch_size(SimpleNamespace()) is None
        assert _resolve_patch_size(SimpleNamespace(config=SimpleNamespace())) is None


class TestPreHook:
    """The geometry publisher emits raw primitives; best-effort otherwise."""

    @staticmethod
    def _module():
        return SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2)))

    def test_publishes_from_kwargs(self):
        hs = torch.zeros(2, 16, 21, 60, 104)  # (B, C, T, H, W)
        with override_forward_context(ForwardContext()):
            _publish_video_geometry(self._module(), (), {"hidden_states": hs})
            ctx = get_forward_context()
            assert ctx.latent_shape == (21, 60, 104)
            assert ctx.patch_size == (1, 2, 2)

    def test_publishes_from_positional_args(self):
        hs = torch.zeros(1, 16, 13, 32, 32)
        with override_forward_context(ForwardContext()):
            _publish_video_geometry(self._module(), (hs,), {})
            assert get_forward_context().latent_shape == (13, 32, 32)

    def test_noop_for_non_5d_input(self):
        hs = torch.zeros(1, 256, 16)  # image/audio-like, not a video grid
        with override_forward_context(ForwardContext()):
            _publish_video_geometry(self._module(), (), {"hidden_states": hs})
            assert get_forward_context().latent_shape is None

    def test_noop_when_patch_size_absent(self):
        hs = torch.zeros(1, 16, 13, 32, 32)
        with override_forward_context(ForwardContext()):
            _publish_video_geometry(SimpleNamespace(), (), {"hidden_states": hs})
            assert get_forward_context().patch_size is None

    def test_noop_without_active_context(self):
        # Must not raise when no ForwardContext is active.
        hs = torch.zeros(1, 16, 13, 32, 32)
        with override_forward_context(None):
            _publish_video_geometry(self._module(), (), {"hidden_states": hs})


class TestDenoiseTotal:
    """The step total comes from the pipeline, with no model-side call."""

    def test_resolve_prefers_public_property(self):
        assert _resolve_total_denoise_steps(SimpleNamespace(num_timesteps=40)) == 40

    def test_resolve_falls_back_to_private_attr(self):
        assert _resolve_total_denoise_steps(SimpleNamespace(_num_timesteps=8)) == 8

    def test_resolve_none_for_unconventional_pipeline(self):
        # No convention, not yet set, or a non-positive / non-int value.
        assert _resolve_total_denoise_steps(SimpleNamespace()) is None
        assert _resolve_total_denoise_steps(SimpleNamespace(num_timesteps=None)) is None
        assert _resolve_total_denoise_steps(SimpleNamespace(num_timesteps=0)) is None
        assert _resolve_total_denoise_steps(SimpleNamespace(num_timesteps=True)) is None

    def test_publishes_onto_context(self):
        with override_forward_context(ForwardContext()):
            _publish_denoise_total(SimpleNamespace(num_timesteps=40))
            assert get_forward_context().total_denoise_steps == 40

    def test_explicit_value_is_not_overwritten(self):
        # A pipeline with a non-standard loop may publish its own value; the
        # framework default must not clobber it.
        with override_forward_context(ForwardContext()):
            set_forward_context_total_denoise_steps(4)
            _publish_denoise_total(SimpleNamespace(num_timesteps=40))
            assert get_forward_context().total_denoise_steps == 4

    def test_noop_for_unconventional_pipeline(self):
        with override_forward_context(ForwardContext()):
            _publish_denoise_total(SimpleNamespace())
            assert get_forward_context().total_denoise_steps is None

    def test_noop_without_active_context(self):
        with override_forward_context(None):
            _publish_denoise_total(SimpleNamespace(num_timesteps=40))


class _TinyTransformer(torch.nn.Module):
    config = SimpleNamespace(patch_size=(1, 2, 2))

    def forward(self, hidden_states):
        return hidden_states


class TestRegistration:
    """register_per_forward_hooks is idempotent and fires on forward."""

    def test_hook_fires_on_forward(self):
        pipeline = SimpleNamespace(transformer=_TinyTransformer(), transformer_2=None, num_timesteps=40)
        register_per_forward_hooks(pipeline)
        hs = torch.zeros(1, 16, 21, 60, 104)
        with override_forward_context(ForwardContext()):
            pipeline.transformer(hidden_states=hs)
            ctx = get_forward_context()
            assert ctx.latent_shape == (21, 60, 104)
            assert ctx.patch_size == (1, 2, 2)
            # Same hook publishes the step total: no model-side call anywhere.
            assert ctx.total_denoise_steps == 40

    def test_hook_survives_a_pipeline_that_publishes_nothing(self):
        # A transformer whose latent is not a video grid, on a pipeline with no
        # num_timesteps: the forward must still run, state simply stays unset.
        pipeline = SimpleNamespace(transformer=_TinyTransformer(), transformer_2=None)
        register_per_forward_hooks(pipeline)
        with override_forward_context(ForwardContext()):
            pipeline.transformer(hidden_states=torch.zeros(1, 256, 16))
            ctx = get_forward_context()
            assert ctx.latent_shape is None
            assert ctx.total_denoise_steps is None

    def test_hook_does_not_keep_the_pipeline_alive(self):
        # The hook lives on a module the pipeline owns; a strong ref back would
        # leak the whole pipeline (weights included) for the process lifetime.
        class _Pipeline:
            def __init__(self):
                self.transformer = _TinyTransformer()
                self.transformer_2 = None
                self.num_timesteps = 40

        pipeline = _Pipeline()
        register_per_forward_hooks(pipeline)
        ref = weakref.ref(pipeline)
        del pipeline
        gc.collect()
        assert ref() is None

    def test_pre_hook_never_raises(self):
        # Best-effort contract: a pipeline whose attribute access explodes must
        # not take down the forward.
        class _Exploding:
            @property
            def num_timesteps(self):
                raise RuntimeError("boom")

        hook = _make_per_forward_pre_hook(lambda: _Exploding())
        with override_forward_context(ForwardContext()):
            hook(_TinyTransformer(), (), {"hidden_states": torch.zeros(1, 16, 21, 60, 104)})
            assert get_forward_context().total_denoise_steps is None

    def test_idempotent_registration(self):
        module = _TinyTransformer()
        pipeline = SimpleNamespace(transformer=module, transformer_2=None)
        register_per_forward_hooks(pipeline)
        register_per_forward_hooks(pipeline)  # second call must not double-register
        assert len(module._forward_pre_hooks) == 1

    def test_no_transformer_is_noop(self):
        register_per_forward_hooks(SimpleNamespace())  # must not raise
        register_per_forward_hooks(None)
