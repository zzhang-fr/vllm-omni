# Related Works

Literature and named methods encountered while researching the vllm-omni
community's direction on world models. Everything here is a pointer for
later reading, **not** a digested summary — use this as a to-read list.

## Source: RFC #1987 "World Model Support" (vllm-project/vllm-omni)

URL: https://github.com/vllm-project/vllm-omni/issues/1987

RFC #1987 partitions world-model workloads into three scenarios: **robotics**,
**interactive video/game**, and **simulation**. The short-term focus is
robotics (DreamZero + an OpenPI-style realtime server). Interactive video
generation and KV-cache / PageAttention for autoregressive diffusion are
explicitly deferred. The literature below is organized by scenario.

### Robotics world models / VLA (near-term RFC focus)

- **DreamZero** — robotics world model; P0 landing target for RFC #1987.
  See also PR #2162 ("DreamZero world model integration + OpenPI WebSocket
  serving"). No public URL in the RFC. *Cited.*
- **Motus** — stateless robotics world model; works with existing vllm-omni
  infra (no engine refactor needed). *Cited.*
- **LingBot-VA / LingBot-World** — video-DiT-based robotics world model.
  Tracked in issues #1045 and #874 (draft I2V pipeline PR). Closest
  in-tree code to a streaming-control pipeline today. *Cited.*
- **π0 (Pi0)** — VLA model; RFC notes it does not require engine changes.
  *Cited.*
- **OpenVLA** — https://openvla.github.io/ — VLA baseline. *Cited.*
- **InternVLA-A1** — MoT (mixture-of-transformers) VLA; issue #1948.
  *Cited.*
- **Groot (Nvidia)** — VLA; P2 target. *Named only.*
- **OpenPI** — realtime-robotics API style; referenced as the **protocol
  shape** for the new `/v1/world/stream` endpoint, not as a model. *Cited.*
- **LeRobot (HuggingFace)** — realtime-API / dataset target. *Cited.*
- **RLinf** — RL framework intended to integrate with vllm-omni's realtime
  server. *Cited.*

### Interactive video / game (deferred in RFC; closest to our project)

- **Genie 3** — https://deepmind.google/discover/blog/genie-3/ — DeepMind
  interactive world model generating explorable 3D environments from a text
  prompt. *Cited.*
- **Matrix Game** — controllable interactive video generation. *Cited.*
- **HunYuan World** — Tencent interactive 3D / video world model. *Cited.*

### Simulation / world foundation models

- **Cosmos WFM (Nvidia Cosmos)** — https://www.nvidia.com/en-us/ai/cosmos/ —
  world foundation model family for simulation-style rollouts. *Cited.*

### Architecture patterns referenced (not models)

- **VGGT** (CVPR 2025) — feed-forward 3D reconstruction from images; basis
  of sibling RFC #2727 "Support 3D World Model (VGGT)". The reason VGGT
  matters for us: #2727 proposes a `stage_type: "reconstruct"` engine with
  **dLLM-style chunked computation + sliding-window KV cache**, which is the
  architectural pattern most relevant to long-horizon video-rollout drift.
  *Cited in #2727.*
  - **VGGT-Long** — overlapping-chunk + Sim(3) alignment extension. *Cited.*
  - **FastVGGT** — token-partitioning, ~4× speedup. *Cited.*
  - **Sparse-VGGT** — block-sparse attention kernel. *Cited.*
- **dLLM (diffusion LLM)** — an architecture family mentioned by maintainer
  Gaohan123 as a possible convergence point with world-model chunked
  generation. *Named*, no specific paper cited in the RFC.

## Conspicuously absent from RFC #1987

None of the following are mentioned anywhere in RFC #1987 or its comments,
even though they'd be natural references for our rollout-drift problem:

- **DIAMOND** (Diffusion for World Modeling; NeurIPS 2024) — learned
  latent-space dynamics for Atari, trained end-to-end as a world model.
- **V-JEPA / V-JEPA 2** (Meta) — representation-only video prediction; no
  pixel reconstruction between steps.
- **CogVideoX** — open-source video DiT with I2V and continuation variants.
- **Wan2.1 / Wan2.2** — Alibaba video DiT family; Wan2.2 has start+end frame
  conditioning.
- **FramePack** — clip-continuation video generation with bounded memory.
- **Pyramidal Flow Matching** — efficient long-video DiT.
- **Open-Sora**, **VideoCrafter**, **MovieGen**, **Sora**, **Lumiere**,
  **CausVid** — general video-generation literature.

If we end up in direction (1) "richer conditioning on the same kind of
model" from the Stage 2 journal entry, Wan2.2 (start+end), CogVideoX
continuation, and FramePack are the three concrete candidates to evaluate
next. DIAMOND is the canonical reference if we go toward direction (3)
"learned dynamics".

## Related issues / PRs in vllm-omni

Noted here so we can recheck status later — none of these are digested,
just indexed.

- **#1987** — [RFC] World Model Support (this one). Open.
- **#2162** — DreamZero + OpenPI WebSocket serving. Open PR; the P0 landing.
- **#2727** — [RFC] Support 3D World Model (VGGT). Open.
  Architecturally most relevant for us: introduces chunked compute +
  sliding-window KV cache.
- **#2073** — [RFC] Refactor engine/runner/pipeline for step-wise and
  continuous batching. Open. Prerequisite for any AR-chunked diffusion.
- **#2590** — [WIP][Perf] AR KV prefix reuse in DiT (Hunyuan-Image-3). Open.
- **#2632** — [RFC] Per-Role Attention Backend Configuration for Diffusion.
  Open. Potential host for blockwise-causal attention.
- **#874** — LingBot-World I2V pipeline PR. Draft; closest in-tree code to
  streaming control.
- **#1045** — [New Model] LingBot-World. Open.
- **#1948** — [New Model] InternVLA-A1. Open.
- **#2063 / #2160 / #2078 / #2423** — CFG parallel refactors. Merged;
  prereqs for DreamZero.
