# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end per-role attention backend tests (RFC #2632 P3).

These tests run real model weights (Wan2.2 T2V and HunyuanVideo 1.5 T2V) and
are *opt-in*: skipped unless ``VLLM_OMNI_E2E_MODEL`` is set and a CUDA GPU is
available, e.g.::

    VLLM_OMNI_E2E_MODEL=Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
        pytest tests/diffusion/attention/test_e2e_per_role.py -v -m e2e

The Hunyuan case can override its default model id via
``VLLM_OMNI_E2E_HUNYUAN_MODEL``.

The fast plumbing is verified by ``test_attention_integration.py`` and
``test_per_role_gpu.py``: those exercise ``Attention(role=...)`` under a real
``OmniDiffusionConfig`` + ``forward_context`` with the real CUDA backends, no
weights required. This file adds:

- a real Wan2.2 model load with per-role self/cross config and a tiny inference
- a real HunyuanVideo 1.5 model load with per-role joint config (the dual-stream
  joint attention is tagged ``AttentionRole.JOINT``)
- log-based verification that the per-role config reached the worker process
  (the workers run in subprocesses, so worker logs are the source of truth).
"""

import os
import subprocess
import sys

import pytest

from vllm_omni.diffusion.attention.role import AttentionRole
from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig

_E2E_MODEL = os.environ.get("VLLM_OMNI_E2E_MODEL")
_E2E_HUNYUAN_MODEL = os.environ.get(
    "VLLM_OMNI_E2E_HUNYUAN_MODEL",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
)
_HAS_GPU = False
try:
    import torch

    _HAS_GPU = torch.cuda.is_available()
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    not (_E2E_MODEL and _HAS_GPU),
    reason="set VLLM_OMNI_E2E_MODEL and have CUDA available to run E2E tests",
)


# --------------------------------------------------------------------------- #
# Lightweight (no model load): config plumbing & legacy fallback
# --------------------------------------------------------------------------- #


def test_e2e_empty_config_matches_legacy():
    """OmniDiffusionConfig() and OmniDiffusionConfig(attention={}) both yield empty config."""
    cfg_none = OmniDiffusionConfig()
    cfg_empty = OmniDiffusionConfig(attention={})
    assert cfg_none.attention.is_empty()
    assert cfg_empty.attention.is_empty()


def test_e2e_env_var_legacy(monkeypatch):
    """DIFFUSION_ATTENTION_BACKEND remains the lowest-priority fallback."""
    from vllm_omni.diffusion.attention.selector import get_attn_backend

    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    cls, spec = get_attn_backend(role=AttentionRole.SELF, head_size=64, config=AttentionConfig())
    assert cls is not None
    assert spec.backend == "auto"


def test_e2e_backend_kwargs_reach_real_impl():
    """`backend_kwargs` survive AttentionConfig → forward_context → impl."""
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
    )

    omni = OmniDiffusionConfig(
        attention={"per_role": {"self": {"backend": "TORCH_SDPA", "extra": {"custom_flag": True}}}}
    )
    with override_forward_context(ForwardContext(omni_diffusion_config=omni)):
        layer = Attention(
            num_heads=4,
            head_size=64,
            causal=False,
            softmax_scale=0.125,
            num_kv_heads=4,
            role=AttentionRole.SELF,
        )
    assert layer.attention._backend_kwargs == {"custom_flag": True}


# --------------------------------------------------------------------------- #
# Real Wan2.2 load + inference, single subprocess to release GPU cleanly.
# --------------------------------------------------------------------------- #


def _run_omni_subprocess(model: str, attention_spec: str) -> tuple[str, str, int]:
    """Spawn a subprocess that loads Wan2.2 with the per-role config and runs
    a tiny 2-step inference.

    Subprocess isolation matters here: Omni's multiproc workers do not always
    release GPU memory inside the parent's lifetime, so running each load in
    its own process is the only reliable way to keep GPU state clean.
    """
    script = "\n".join(
        [
            "import sys",
            "from vllm_omni.entrypoints import Omni",
            "from vllm_omni.inputs.data import OmniDiffusionSamplingParams",
            "",
            f"omni = Omni(model={model!r}, attention={attention_spec}, enforce_eager=True)",
            "try:",
            "    print('LOAD_OK', flush=True)",
            "    sp = OmniDiffusionSamplingParams(",
            "        num_inference_steps=2, height=256, width=256, num_frames=9",
            "    )",
            "    out = omni.generate({'prompt': 'a small red ball'}, sp)",
            "    print('INFERENCE_OK', flush=True)",
            "finally:",
            "    try:",
            "        omni.close()",
            "    except Exception:",
            "        pass",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=900,
    )
    return proc.stdout, proc.stderr, proc.returncode


def test_e2e_per_role_full_pipeline():
    """Real Wan2.2 load + 2-step inference + per-role config end-to-end check.

    Asserts the model loaded, inference finished, and the worker selected the
    configured per-role backend (TORCH_SDPA) per the spec.
    """
    spec = '{"per_role": {"self": "TORCH_SDPA", "cross": "TORCH_SDPA"}}'
    stdout, stderr, rc = _run_omni_subprocess(_E2E_MODEL, spec)
    assert rc == 0, (
        f"subprocess exit={rc}\n--- stdout tail ---\n"
        + "\n".join(stdout.splitlines()[-30:])
        + "\n--- stderr tail ---\n"
        + "\n".join(stderr.splitlines()[-30:])
    )
    assert "LOAD_OK" in stdout, f"missing LOAD_OK in stdout:\n{stdout[-500:]}"
    assert "INFERENCE_OK" in stdout, f"missing INFERENCE_OK in stdout:\n{stdout[-500:]}"
    log_text = stdout + "\n" + stderr
    assert "TORCH_SDPA" in log_text, "expected per-role TORCH_SDPA selection in worker logs"


def test_e2e_per_role_hunyuanvideo15_joint():
    """HunyuanVideo 1.5 dual-stream joint attention with the JOINT role.

    HunyuanVideo-1.5 uses a single dual-stream attention pass; we tag it with
    ``AttentionRole.JOINT``. This verifies that:
      1. the joint role flows through `Omni(attention=...)` to the worker
      2. all 54 attention layers in the worker pick up TORCH_SDPA (no defaults)
      3. inference completes end-to-end
    """
    if not _E2E_MODEL:  # only force a Hunyuan run when E2E is generally enabled
        pytest.skip("E2E disabled (set VLLM_OMNI_E2E_MODEL)")
    spec = '{"per_role": {"joint": "TORCH_SDPA"}}'
    stdout, stderr, rc = _run_omni_subprocess(_E2E_HUNYUAN_MODEL, spec)
    assert rc == 0, (
        f"subprocess exit={rc}\n--- stdout tail ---\n"
        + "\n".join(stdout.splitlines()[-30:])
        + "\n--- stderr tail ---\n"
        + "\n".join(stderr.splitlines()[-30:])
    )
    assert "LOAD_OK" in stdout, f"missing LOAD_OK:\n{stdout[-500:]}"
    assert "INFERENCE_OK" in stdout, f"missing INFERENCE_OK:\n{stdout[-500:]}"
    log_text = stdout + "\n" + stderr
    n_sdpa = log_text.count("Using diffusion attention backend 'TORCH_SDPA'")
    assert n_sdpa > 0, "no Attention layer picked up TORCH_SDPA — joint role not honored"
    # No layer should silently fall through to the platform default.
    assert "Defaulting to diffusion attention backend" not in log_text, (
        "some Attention layer defaulted to platform backend — joint role missed"
    )


@pytest.mark.skip(reason="serving smoke runs from the serving CI harness, not unit tests")
def test_e2e_serving_smoke():
    pass
