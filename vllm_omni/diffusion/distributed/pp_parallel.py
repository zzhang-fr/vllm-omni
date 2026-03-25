# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_world_size, get_pp_group


class PipelineParallelMixin:
    """
    Mixin providing Pipeline Parallelism for diffusion pipelines.

    All PP ranks run the full denoising loop in `forward()`.
    `predict_noise_maybe_with_pp_and_cfg` and `scheduler_step_maybe_with_pp_and_cfg` encapsulate
    all inter-rank communication.

    Communication pattern per denoising step:
      Forward chain : rank 0 → 1 → … → N-1  via async isend/irecv (AsyncIntermediateTensors)
      Next timestep : last rank → rank 0     via async isend/irecv

    All communication is asynchronous using isend_tensor_dict/irecv_tensor_dict.
    Only rank 0 needs updated latents for the next forward pass start.

    For sequential CFG (cfg_parallel_size=1) with PP, two full forward chains are
    executed — one for the positive pass and one for the negative pass — so that each
    PP stage operates on the correct encoder_hidden_states. This avoids the
    quality degradation of reusing positive-conditioned IntermediateTensors for the
    negative prediction.
    """

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
        pp_size = get_pipeline_parallel_world_size()
        if pp_size == 1:
            return self.predict_noise_maybe_with_cfg(
                do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize, output_slice
            )

        pp_group = get_pp_group()
        all_kwargs = [positive_kwargs] + ([negative_kwargs] if do_true_cfg else [])
        n = len(all_kwargs)

        # Non-first ranks receive all n ITs asynchronously.
        # AsyncIntermediateTensors will wait on handles when .tensors is accessed.
        its: list[AsyncIntermediateTensors | None] = [None] * n
        if not pp_group.is_first_rank:
            for i in range(n):
                its[i] = AsyncIntermediateTensors(*pp_group.irecv_tensor_dict())

        if not pp_group.is_last_rank:
            # First / middle rank: run partial forwards and propagate ITs downstream.
            for kwargs, it in zip(all_kwargs, its):
                kw = dict(kwargs) if it is None else {**kwargs, "intermediate_tensors": it}
                result = self.predict_noise(**kw)
                pp_group.isend_tensor_dict(result.tensors)
            return None

        # Last rank: run full forwards (second half of transformer layers).
        noise_preds = []
        for kwargs, it in zip(all_kwargs, its):
            kw = dict(kwargs) if it is None else {**kwargs, "intermediate_tensors": it}
            noise_preds.append(self.predict_noise(**kw))

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
        """
        pp_size = get_pipeline_parallel_world_size()
        if pp_size == 1:
            return self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg, per_request_scheduler)

        pp_group = get_pp_group()
        if pp_group.is_last_rank:
            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg, per_request_scheduler)
            latents = latents.contiguous()
            pp_group.isend_tensor_dict({"latents": latents}, dst=0)
        elif pp_group.is_first_rank:
            tensor_dict, handles, postproc = pp_group.irecv_tensor_dict(src=pp_group.world_size - 1)
            for handle in handles:
                handle.wait()
            latents = tensor_dict["latents"]
        return latents
