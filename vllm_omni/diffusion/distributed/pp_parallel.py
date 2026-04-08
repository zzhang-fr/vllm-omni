# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_world_size, get_pp_group


class AsyncLatents:
    """Transparent async wrapper returned by scheduler_step on rank 0.

    Wraps a pending ``irecv_tensor_dict`` and defers ``handle.wait()`` until the
    underlying tensor is actually consumed — either via attribute access
    (e.g. ``latents.to(dtype)``, ``latents.shape``) or via a torch operation
    (e.g. ``mask * latents``).  This keeps the first PP rank non-blocking after
    posting the receive, matching the async philosophy used everywhere else in
    the PP communication layer.
    """

    __slots__ = ("_tensor_dict", "_handles", "_postproc", "_tensor")

    def __init__(
        self,
        tensor_dict: dict[str, torch.Tensor],
        handles: list[torch.distributed.Work],
        postproc: list,
    ):
        self._tensor_dict = tensor_dict
        self._handles = handles
        self._postproc = postproc
        self._tensor: torch.Tensor | None = None

    def _resolve(self) -> torch.Tensor:
        if self._tensor is not None:
            return self._tensor
        for h in self._handles:
            h.wait()
        for fn in self._postproc:
            fn()
        self._tensor = self._tensor_dict["latents"]
        return self._tensor

    # Attribute access (e.g. .shape, .to(), .dtype) delegates to the resolved tensor.
    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    # Torch function protocol: any torch op involving an AsyncLatents resolves it first.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def _unwrap(x):
            if isinstance(x, AsyncLatents):
                return x._resolve()
            if isinstance(x, (list, tuple)):
                return type(x)(_unwrap(item) for item in x)  # type(x) return the class of x to preserve its type
            return x

        args = tuple(_unwrap(a) for a in args)
        kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)


class PipelineParallelMixin:
    """
    Mixin providing Pipeline Parallelism for diffusion pipelines.

    All PP ranks run the full denoising loop in `forward()`.
    `predict_noise_maybe_with_pp_and_cfg` and `scheduler_step_maybe_with_pp_and_cfg` encapsulate
    all inter-rank communication.

    Communication pattern per denoising step:
      Forward chain : rank 0 → 1 → … → N-1  via async isend/irecv (AsyncIntermediateTensors)
      Next timestep : last rank → rank 0     via async isend/irecv (AsyncLatents)

    All communication is asynchronous using isend_tensor_dict/irecv_tensor_dict.
    Only rank 0 needs updated latents for the next forward pass start.

    For sequential CFG (cfg_parallel_size=1) with PP, two full forward chains are
    executed — one for the positive pass and one for the negative pass — so that each
    PP stage operates on the correct encoder_hidden_states.
    """

    @property
    def _pp_send_work(self) -> list[torch.distributed.Work]:
        if not hasattr(self, "__pp_send_work"):
            self.__pp_send_work = []
        return self.__pp_send_work

    @_pp_send_work.setter
    def _pp_send_work(self, work: list[torch.distributed.Work]) -> None:
        self.__pp_send_work = work

    def sync_pp_send(self) -> None:
        """
        Wait on all pending non-blocking PP sends.

        Must be called after the denoising loop so that the isend handles
        from the last iteration are completed before any subsequent
        collective (e.g. VAE decode broadcast) or tensor reuse.
        """
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

    def predict_noise_maybe_with_pp_and_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        cfg_normalize: bool = False,
        output_slice: int | None = None,
    ) -> torch.Tensor | None:
        """
        Drop-in replacement for predict_noise_maybe_with_cfg that also handles PP.

        Returns:
            noise_pred on the first PP rank; None on all other ranks.
        """
        if get_pipeline_parallel_world_size() == 1:
            return self.predict_noise_maybe_with_cfg(
                do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize, output_slice
            )

        self.sync_pp_send()

        pp_group = get_pp_group()
        all_kwargs = [positive_kwargs] + ([negative_kwargs] if do_true_cfg else [])

        # Non-first ranks receive all n ITs asynchronously.
        # AsyncIntermediateTensors will wait on handles when .tensors is accessed.
        n = len(all_kwargs)
        its: list[AsyncIntermediateTensors | None] = [None] * n
        if not pp_group.is_first_rank:
            for i in range(n):
                its[i] = AsyncIntermediateTensors(*pp_group.irecv_tensor_dict())

        if not pp_group.is_last_rank:
            # First / middle rank: run partial forwards and propagate ITs downstream.
            for kwargs, it in zip(all_kwargs, its):
                result = self.predict_noise(**kwargs, intermediate_tensors=it)
                self._pp_send_work.extend(pp_group.isend_tensor_dict(result.tensors))
            return None

        # Last rank: run full forwards (second half of transformer layers).
        noise_preds = [self.predict_noise(**kwargs, intermediate_tensors=it) for kwargs, it in zip(all_kwargs, its)]

        # Last rank computes final noise_pred and will run scheduler
        if do_true_cfg:
            return self.combine_cfg_noise(noise_preds[0], noise_preds[1], true_cfg_scale, cfg_normalize)
        return noise_preds[0]

    def scheduler_step_maybe_with_pp_and_cfg(
        self,
        noise_pred: torch.Tensor | None,
        t: torch.Tensor,
        latents: torch.Tensor,
        do_true_cfg: bool,
        per_request_scheduler: Any | None = None,
    ) -> torch.Tensor:
        """
        Drop-in replacement for scheduler_step_maybe_with_cfg that also handles PP.

        Only the last rank runs the scheduler (it already has noise_pred); the result
        is sent to rank 0 which needs it for the next forward pass.

        Returns a ``AsyncLatents`` on rank 0 that transparently defers
        ``handle.wait()`` until the tensor is actually consumed (via attribute
        access or a torch operation), keeping the rank non-blocking after the
        ``irecv`` is posted.
        """
        if get_pipeline_parallel_world_size() == 1:
            return self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg, per_request_scheduler)

        pp_group = get_pp_group()
        if pp_group.is_last_rank:
            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg, per_request_scheduler)
            self._pp_send_work = pp_group.isend_tensor_dict({"latents": latents}, dst=pp_group.first_rank)
        elif pp_group.is_first_rank:
            latents = AsyncLatents(*pp_group.irecv_tensor_dict(src=pp_group.last_rank))
        return latents
