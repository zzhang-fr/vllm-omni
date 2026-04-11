# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latency benchmark for `get_attn_backend` without @cache (RFC #2632 P2).

The selector runs at `Attention.__init__` construction time only — never on
the forward path — so absolute latency is ~free as long as it stays well
under the model load budget. This test asserts an upper bound and also
prints the measured numbers so ``pytest -s`` surfaces the reviewer-facing
data.
"""

import statistics
import time

import pytest

import vllm_omni.platforms as platforms_pkg
from vllm_omni.diffusion.attention.role import AttentionRole
from vllm_omni.diffusion.attention.selector import get_attn_backend
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec


class _DummyBackend:
    @staticmethod
    def get_name() -> str:
        return "DUMMY"


@pytest.fixture
def patched(monkeypatch):
    class _StubPlatform:
        @staticmethod
        def get_diffusion_attn_backend_cls(*, selected_backend, head_size):
            return "x.y.Stub"

    monkeypatch.setattr(platforms_pkg, "current_omni_platform", _StubPlatform)
    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.selector._load_backend_cls",
        lambda path: _DummyBackend,
    )


def test_selector_latency_budget(patched):
    """60 selector calls (Wan2.2-scale init) must stay well under 5 ms."""
    cfg = AttentionConfig(
        per_role={
            "self": AttentionSpec(backend="FLASH_ATTN"),
            "cross": AttentionSpec(backend="TORCH_SDPA"),
        }
    )
    roles = [AttentionRole.SELF, AttentionRole.CROSS] * 30  # 60 calls per round

    # Warmup a few rounds so interpreter caches settle.
    for _ in range(3):
        for r in roles:
            get_attn_backend(r, head_size=64, config=cfg)

    rounds = []
    for _ in range(10):
        t = time.perf_counter()
        for r in roles:
            get_attn_backend(r, head_size=64, config=cfg)
        rounds.append((time.perf_counter() - t) * 1e6)

    median_us = statistics.median(rounds)
    min_us = min(rounds)
    max_us = max(rounds)
    per_call_us = median_us / 60

    # Surface the numbers in pytest -s / captured stdout for reviewer context.
    print()
    print("--- RFC #2632 @cache removal benchmark -----------------")
    print("60 get_attn_backend calls (Wan2.2 scale), 10 rounds:")
    print(f"  median  : {median_us:>6.1f} us  ({per_call_us:.2f} us / call)")
    print(f"  min     : {min_us:>6.1f} us")
    print(f"  max     : {max_us:>6.1f} us")
    print("--------------------------------------------------------")

    # Upper bound: 5 ms per 60 calls is ~2 orders of magnitude above the
    # observed ~120 us on a modern CPU. This catches gross regressions
    # without being flaky on shared CI runners.
    assert median_us < 5000, f"60 selector calls took {median_us:.0f} us (>5 ms)"
