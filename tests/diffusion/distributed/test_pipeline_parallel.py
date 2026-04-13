# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket

import pytest
import torch
from vllm.model_executor.models.utils import PPMissingLayer, make_empty_intermediate_tensors_factory, make_layers
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_classifier_free_guidance_rank,
    get_pp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.pp_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.platforms import current_omni_platform


def _find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def update_environment_variables(envs_dict: dict[str, str]) -> None:
    for k, v in envs_dict.items():
        os.environ[k] = v


# ---------------------------------------------------------------------------
# Shared stubs used by both unit and distributed tests
# ---------------------------------------------------------------------------


class FakeWork:
    """Drop-in for torch.distributed.Work that records whether wait() was called."""

    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class SimpleScheduler:
    """Minimal diffusion-step scheduler: latents -= 0.1 * noise_pred."""

    def step(self, noise_pred: torch.Tensor, t, latents: torch.Tensor, return_dict: bool = False):
        return (latents - 0.1 * noise_pred,)


class MockPipelineParallel(PipelineParallelMixin, CFGParallelMixin):
    """Minimal pipeline used to exercise PipelineParallelMixin.

    Uses vLLM's ``make_layers`` for layer partitioning — the same utility used
    by real DiT models — so the PP layer-split logic is exercised faithfully.

    Each layer's weights are seeded by ``seed + layer_index`` so that layer ``i``
    is initialized identically on every rank regardless of which ranks are active,
    allowing the distributed output to be compared against the single-GPU baseline.

    Args:
        num_layers: Total number of Linear layers.
        dim:        Input / hidden dimension.
        seed:       Base RNG seed; layer ``i`` uses ``seed + i``.
        device:     Target device for layer weights (default: CPU).
        dtype:      Target dtype for layer weights (default: float32).
    """

    def __init__(
        self,
        num_layers: int = 4,
        dim: int = 64,
        seed: int = 42,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers,
            lambda prefix: torch.nn.Linear(dim, dim, bias=False),
            prefix="layers",
        )

        for i, layer in enumerate(self.layers):
            if not isinstance(layer, PPMissingLayer):
                torch.manual_seed(seed + i)
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)

        self.layers.to(device=device, dtype=dtype)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(["hidden_states"], dim)
        self.scheduler = SimpleScheduler()

    def predict_noise(self, x=None, intermediate_tensors=None, **_kwargs) -> torch.Tensor | IntermediateTensors:
        """Layer-split forward pass.

        * First PP rank: uses ``x`` from caller kwargs.
        * Later PP ranks: overrides ``x`` with ``intermediate_tensors["hidden_states"]``
          (which transparently waits for the async receive).
        * Non-last PP ranks return ``IntermediateTensors``; the last rank
          returns the plain noise-prediction tensor.
        """
        if intermediate_tensors is not None:
            x = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            x = self.layers[i](x)

        pp_group = get_pp_group()
        if not pp_group.is_last_rank:
            return IntermediateTensors({"hidden_states": x})
        return x


# ---------------------------------------------------------------------------
# 1.  AsyncLatents – unit tests (no distributed env required)
# ---------------------------------------------------------------------------


class TestAsyncLatents:
    """Verifies the lazy-resolution behaviour of AsyncLatents without a real process group."""

    def _make(self, tensor: torch.Tensor, handles: list | None = None, postproc: list | None = None) -> AsyncLatents:
        return AsyncLatents({"latents": tensor}, handles or [], postproc or [])

    def test_resolve_returns_wrapped_tensor(self):
        t = torch.randn(2, 4)
        al = self._make(t)
        assert al._resolve() is t

    def test_attribute_access_resolves(self):
        t = torch.randn(2, 4)
        al = self._make(t)
        assert al.shape == t.shape
        assert al.dtype == t.dtype

    def test_torch_function_protocol(self):
        """torch ops that receive an AsyncLatents should see the underlying tensor."""
        t = torch.randn(2, 4)
        al = self._make(t)
        mask = torch.ones_like(t)
        result = mask * al  # triggers __torch_function__
        torch.testing.assert_close(result, mask * t)

    def test_torch_function_with_list_arg(self):
        """__torch_function__ must unwrap AsyncLatents inside list/tuple args."""
        t = torch.randn(2, 4)
        al = self._make(t)
        result = torch.cat([al, al], dim=0)
        torch.testing.assert_close(result, torch.cat([t, t], dim=0))

    def test_handles_are_waited_before_resolve(self):
        t = torch.randn(2, 4)
        h1, h2 = FakeWork(), FakeWork()
        al = self._make(t, handles=[h1, h2])
        _ = al.shape  # trigger resolution
        assert h1.waited and h2.waited, "Not all handles were waited on"

    def test_postproc_callbacks_invoked_on_resolve(self):
        t = torch.randn(2, 4)
        log: list[int] = []
        al = self._make(t, postproc=[lambda: log.append(1), lambda: log.append(2)])
        _ = al.shape
        assert log == [1, 2], f"postproc not called in order: {log}"

    def test_idempotent_resolve(self):
        """handle.wait() must not be called twice if _resolve() is called twice."""
        t = torch.randn(2, 4)
        h = FakeWork()
        al = self._make(t, handles=[h])
        _ = al.shape  # first resolve
        h.waited = False  # reset sentinel
        _ = al.dtype  # second resolve
        assert not h.waited, "handle.wait() was called a second time"


# ---------------------------------------------------------------------------
# 2.  sync_pp_send – unit tests (no distributed env required)
# ---------------------------------------------------------------------------


class TestSyncPPSend:
    """Verifies PipelineParallelMixin.sync_pp_send without a real process group."""

    @staticmethod
    def _make_pipeline() -> PipelineParallelMixin:
        # Instantiate a bare mixin — no layers, no distributed env needed.
        # sync_pp_send only touches _pp_send_work, so this is sufficient.
        class _BarePP(PipelineParallelMixin, CFGParallelMixin):
            pass

        return _BarePP()

    def test_noop_when_work_list_empty(self):
        pipeline = self._make_pipeline()
        pipeline.sync_pp_send()
        assert pipeline._pp_send_work == []

    def test_waits_all_pending_handles(self):
        pipeline = self._make_pipeline()
        works = [FakeWork(), FakeWork(), FakeWork()]
        pipeline._pp_send_work = works
        pipeline.sync_pp_send()
        assert all(w.waited for w in works), "Some handles were not waited on"

    def test_clears_work_list_after_sync(self):
        pipeline = self._make_pipeline()
        pipeline._pp_send_work = [FakeWork()]
        pipeline.sync_pp_send()
        assert pipeline._pp_send_work == []


# ---------------------------------------------------------------------------
# Distributed test helpers
# ---------------------------------------------------------------------------


def init_dist(local_rank: int, world_size: int, master_port: str) -> torch.device:
    """Initialise the distributed environment for a spawned worker."""
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
        }
    )
    init_distributed_environment()
    return device


def make_pipeline_and_inputs(
    test_config: dict, dtype: torch.dtype, device: torch.device, do_true_cfg: bool = False
) -> tuple["MockPipelineParallel", dict, dict | None]:
    """Create a MockPipelineParallel and seeded inputs from a test_config dict.

    Must be called after ``initialize_model_parallel`` so that ``make_layers``
    can read the PP group to determine this rank's layer slice.

    Returns ``(pipeline, positive_kwargs, negative_kwargs)``.
    ``negative_kwargs`` is ``None`` when ``do_true_cfg=False``.
    """
    pipeline = MockPipelineParallel(
        num_layers=test_config["num_layers"],
        dim=test_config["dim"],
        seed=test_config["model_seed"],
        device=device,
        dtype=dtype,
    )

    torch.manual_seed(test_config["input_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_config["input_seed"])
    pos_x = {"x": torch.randn(test_config["batch_size"], test_config["dim"], dtype=dtype, device=device)}

    negative_kwargs = None
    if do_true_cfg:
        torch.manual_seed(test_config["input_seed"] + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(test_config["input_seed"] + 1)
        neg_x = torch.randn(test_config["batch_size"], test_config["dim"], dtype=dtype, device=device)
        negative_kwargs = {"x": neg_x}

    return pipeline, pos_x, negative_kwargs


# ---------------------------------------------------------------------------
# 3.  isend_tensor_dict / irecv_tensor_dict  (2 GPUs)
# ---------------------------------------------------------------------------


def isend_irecv_worker(local_rank: int, world_size: int, master_port: str, result_queue):
    device = init_dist(local_rank, world_size, master_port)
    initialize_model_parallel(pipeline_parallel_size=world_size)
    pp_group = get_pp_group()

    if pp_group.is_first_rank:
        torch.manual_seed(77)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(77)
        tensor = torch.randn(3, 5, dtype=torch.float32, device=device)
        handles = pp_group.isend_tensor_dict({"t": tensor})
        for h in handles:
            h.wait()
        result_queue.put(("sent", tensor.cpu()))
    else:
        received = AsyncIntermediateTensors(*pp_group.irecv_tensor_dict())
        result_queue.put(("received", received["t"].cpu()))

    destroy_distributed_env()


@pytest.mark.skipif(current_omni_platform.get_device_count() < 2, reason="Need at least 2 GPUs")
@pytest.mark.parametrize("pp_size", [2])
def test_isend_irecv_tensor_dict(pp_size: int):
    """isend_tensor_dict / irecv_tensor_dict transfer a tensor dict without loss."""
    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    q = manager.Queue()

    port = _find_free_port()
    torch.multiprocessing.spawn(isend_irecv_worker, args=(pp_size, port, q), nprocs=pp_size)

    results = {label: tensor for label, tensor in [q.get(), q.get()]}
    torch.testing.assert_close(
        results["received"], results["sent"], rtol=0, atol=0, msg="isend/irecv transferred tensor incorrectly"
    )


# ---------------------------------------------------------------------------
# 4.  predict_noise_maybe_with_pp_and_cfg
# ---------------------------------------------------------------------------

_baseline_cache: dict[tuple, torch.Tensor] = {}


def compute_single_gpu_baseline(test_config: dict, dtype: torch.dtype, do_true_cfg: bool) -> torch.Tensor:
    """Compute expected single-GPU output using the same MockPipelineParallel.

    Initializes a trivial distributed env (world_size=1) so that ``make_layers`` and the PP/CFG mixins work normally.
    Results are cached so identical configs are only computed once.
    """
    key = (
        test_config["num_layers"],
        test_config["dim"],
        test_config["batch_size"],
        test_config["model_seed"],
        test_config["input_seed"],
        test_config["cfg_scale"],
        dtype,
        do_true_cfg,
    )
    if key in _baseline_cache:
        return _baseline_cache[key]

    device = init_dist(0, 1, _find_free_port())
    initialize_model_parallel(pipeline_parallel_size=1)

    pipeline, positive_kwargs, negative_kwargs = make_pipeline_and_inputs(
        test_config, dtype, device, do_true_cfg=do_true_cfg
    )

    with torch.inference_mode():
        noise_pred = pipeline.predict_noise_maybe_with_pp_and_cfg(
            do_true_cfg=do_true_cfg,
            true_cfg_scale=test_config["cfg_scale"],
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )

    destroy_distributed_env()

    _baseline_cache[key] = noise_pred.cpu()
    return _baseline_cache[key]


def predict_noise_worker(
    local_rank: int,
    world_size: int,
    master_port: str,
    pp_size: int,
    cfg_size: int,
    do_true_cfg: bool,
    dtype: torch.dtype,
    test_config: dict,
    result_queue,
):
    """Generic predict-noise worker parameterized by PP and CFG topology."""
    device = init_dist(local_rank, world_size, master_port)
    initialize_model_parallel(pipeline_parallel_size=pp_size, cfg_parallel_size=cfg_size)

    pp_group = get_pp_group()
    cfg_rank = get_classifier_free_guidance_rank()

    pipeline, positive_kwargs, negative_kwargs = make_pipeline_and_inputs(
        test_config, dtype, device, do_true_cfg=do_true_cfg
    )

    with torch.inference_mode():
        noise_pred = pipeline.predict_noise_maybe_with_pp_and_cfg(
            do_true_cfg=do_true_cfg,
            true_cfg_scale=test_config["cfg_scale"],
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )
    pipeline.sync_pp_send()

    if pp_group.is_last_rank and cfg_rank == 0:
        assert noise_pred is not None
        result_queue.put(noise_pred.cpu())
    else:
        assert noise_pred is None

    destroy_distributed_env()


@pytest.mark.parametrize(
    "pp_size, cfg_size, do_true_cfg, dtype, num_layers, input_seed, rtol, atol",
    [
        pytest.param(
            2,
            1,
            False,
            torch.float32,
            4,
            100,
            1e-5,
            1e-5,
            marks=pytest.mark.skipif(current_omni_platform.get_device_count() < 2, reason="Need at least 2 GPUs"),
            id="pp2-no_cfg-float32",
        ),
        pytest.param(
            2,
            1,
            False,
            torch.bfloat16,
            4,
            100,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(current_omni_platform.get_device_count() < 2, reason="Need at least 2 GPUs"),
            id="pp2-no_cfg-bfloat16",
        ),
        pytest.param(
            2,
            1,
            True,
            torch.bfloat16,
            4,
            200,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(current_omni_platform.get_device_count() < 2, reason="Need at least 2 GPUs"),
            id="pp2-seq_cfg-bfloat16",
        ),
        pytest.param(
            2,
            2,
            True,
            torch.bfloat16,
            4,
            400,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(current_omni_platform.get_device_count() < 4, reason="Need at least 4 GPUs"),
            id="pp2-cfg2-bfloat16",
        ),
        pytest.param(
            3,
            1,
            False,
            torch.bfloat16,
            6,
            500,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(current_omni_platform.get_device_count() < 3, reason="Need at least 3 GPUs"),
            id="pp3-no_cfg-bfloat16",
        ),
    ],
)
def test_predict_noise(pp_size, cfg_size, do_true_cfg, dtype, num_layers, input_seed, rtol, atol):
    """predict_noise_maybe_with_pp_and_cfg output matches the single-GPU baseline across PP / CFG topologies."""
    test_config = {
        "num_layers": num_layers,
        "dim": 64,
        "batch_size": 2,
        "cfg_scale": 7.5,
        "model_seed": 42,
        "input_seed": input_seed,
    }

    baseline_out = compute_single_gpu_baseline(test_config, dtype, do_true_cfg)

    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    pp_q = manager.Queue()

    world_size = pp_size * cfg_size
    port = _find_free_port()
    torch.multiprocessing.spawn(
        predict_noise_worker,
        args=(world_size, port, pp_size, cfg_size, do_true_cfg, dtype, test_config, pp_q),
        nprocs=world_size,
    )

    pp_out = pp_q.get()

    assert baseline_out.shape == pp_out.shape
    torch.testing.assert_close(
        pp_out,
        baseline_out,
        rtol=rtol,
        atol=atol,
        msg=f"PP={pp_size} cfg={cfg_size} {'with' if do_true_cfg else 'no'} CFG output differs from baseline ({dtype=})",
    )


# ---------------------------------------------------------------------------
# 5.  scheduler_step_maybe_with_pp_and_cfg  (PP=2)
# ---------------------------------------------------------------------------


def scheduler_step_pp2_worker(local_rank: int, world_size: int, master_port: str, test_config: dict, result_queue):
    device = init_dist(local_rank, world_size, master_port)
    initialize_model_parallel(pipeline_parallel_size=world_size)

    pp_group = get_pp_group()

    pipeline, positive_kwargs, _ = make_pipeline_and_inputs(test_config, torch.float32, device)
    latents = positive_kwargs["x"]

    # Only the last rank has a noise prediction; middle / first ranks pass None.
    noise_pred = None
    if pp_group.is_last_rank:
        torch.manual_seed(test_config["input_seed"] + 10)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(test_config["input_seed"] + 10)
        noise_pred = torch.randn_like(latents)

    t = torch.tensor(500, device=device)

    with torch.inference_mode():
        result = pipeline.scheduler_step_maybe_with_pp_and_cfg(
            noise_pred=noise_pred, t=t, latents=latents, do_true_cfg=False
        )
    # Last rank must flush pending isend before the process exits.
    pipeline.sync_pp_send()

    if pp_group.is_last_rank:
        # result is the scheduler-updated tensor
        result_queue.put(("last", result.cpu()))
    elif pp_group.is_first_rank:
        # result is AsyncLatents; resolve it by forcing a torch operation
        resolved = result.contiguous()
        result_queue.put(("first", resolved.cpu()))

    destroy_distributed_env()


@pytest.mark.skipif(current_omni_platform.get_device_count() < 2, reason="Need at least 2 GPUs")
def test_scheduler_step_pp2():
    """Rank 0 receives the exact latent tensor produced by the last PP rank via AsyncLatents."""

    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    q = manager.Queue()

    port = _find_free_port()
    torch.multiprocessing.spawn(
        scheduler_step_pp2_worker,
        args=(2, port, {"num_layers": 4, "dim": 64, "batch_size": 2, "model_seed": 42, "input_seed": 300}, q),
        nprocs=2,
    )

    items = {label: tensor for label, tensor in [q.get(), q.get()]}
    assert "first" in items and "last" in items, "Expected results from both PP ranks; got keys: " + str(set(items))
    torch.testing.assert_close(
        items["first"],
        items["last"],
        rtol=0,
        atol=0,
        msg="AsyncLatents on rank 0 does not match the tensor computed on the last PP rank",
    )
