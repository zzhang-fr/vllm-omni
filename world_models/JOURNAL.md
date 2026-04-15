# World Models — Work Journal

## 2026-04-13 — Project kickoff and plan

Goal: build a **toy video-generation-based world model** on vllm-omni as a
learning vehicle. Deliberately tangible first step before exploring deeper
world-model categories (latent-dynamics like DIAMOND / Genie, representation-
only like V-JEPA, Cosmos-style foundation models).

### Framing

- **State** = image (a single video frame)
- **Action** = text prompt describing what should happen next
- **Transition** = run an image-to-video (I2V) model conditioned on
  `(state, action)`, take the **last frame** of the generated clip as the
  next state
- **Rollout** = iterate the above for N steps, optionally concatenating all
  generated clips into one long video for inspection

This is not yet a "real" world model in the learned-dynamics sense — the
transition function is a frozen pretrained I2V model, not something we train.
The point is to get an end-to-end loop running, see how it behaves (temporal
drift, last-frame artifacts, action compliance), and use that to decide what
to build next.

### Infrastructure choices

- **Repo:** `/data/wijnand/src/vllm-omni` (cloned from
  `github.com/vllm-project/vllm-omni`, main)
- **Docker image:** `vllm-omni:wm`, built from `docker/Dockerfile.cuda`
  (base `vllm/vllm-openai:v0.19.0`)
- **HF cache (host):** `/data/wijnand/hf-cache` → mounted to
  `/root/.cache/huggingface` in the container
- **Experiment dir:** `/data/wijnand/src/vllm-omni/world_models/`
- **Hardware:** 2× A100 80GB PCIe, single GPU per run (default rule)

### Model selection

Considered and rejected:
- **Wan2.1-I2V-1.3B** — no `wan2_1` folder in vllm-omni source, only `wan2_2/`;
  untested path.
- **Wan2.2-I2V-A14B** — ~27B MoE, ~54 GB BF16, too tight on one 80 GB A100.

Chosen: **`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v`**
(~13B DiT, ~35 GB BF16, officially supported in
`examples/online_serving/image_to_video/`). Fits one A100 and has a known-good
serve script.

### Stages

0. **Environment** — build docker image, pre-pull HF weights, smoke-test CLI.
1. **First video generation** — synthetic start frame (red ball on beige table,
   832×480), single POST to `/v1/videos`, save mp4 under `runs/smoke/`.
2. **Rollout loop** — `rollout.py`: 5 steps, fixed action vocab
   `["the ball rolls to the right", "the ball rolls forward", "the ball stops",
   "the camera zooms in on the ball"]`, take last frame of each clip as next
   conditioning image, concatenate clips into `runs/<timestamp>/rollout.mp4`.
3. **Observe & iterate** — look for temporal drift, last-frame artifacts,
   action compliance; decide what to build next (learned dynamics? different
   backbone? representation-only model?).

### Server launch (canonical command, with hard-won flags)

```
docker run -d --name vllm-omni-wm-server \
  --gpus '"device=0"' \
  --shm-size=16g --ipc=host \
  -p 8099:8099 \
  -v /data/wijnand/hf-cache:/root/.cache/huggingface \
  vllm-omni:wm \
  vllm serve hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v \
    --omni --port 8099 --flow-shift 5.0 --host 0.0.0.0 \
    --stage-init-timeout 600
```

Two flags were learned the hard way:
- `--shm-size=16g --ipc=host` — without these, Docker's default 64 MB
  `/dev/shm` fills up on the first inference request and vLLM's multiproc
  executor worker hangs forever in `shm_broadcast`. Diagnosed via
  `docker exec ... df -h /dev/shm` → 64M/64M.
- `--stage-init-timeout 600` — the default 300s isn't enough for a cold-cache
  first load (~6 min observed in practice). 600s is a sane ceiling.

### API shape

- `POST /v1/videos` (multipart): `prompt`, `input_reference=@<image>`,
  `size=832x480`, `num_frames=33`, `fps=24`, `num_inference_steps=30`,
  `guidance_scale=6.0`, `flow_shift=5.0`, `seed`
- `GET /v1/videos/{id}` — poll until `status=completed`
- `GET /v1/videos/{id}/content` — download mp4

Reference script:
`examples/online_serving/image_to_video/run_curl_hunyuan_video_15.sh`.

---

## 2026-04-14 — HunyuanVideo-1.5 I2V memory profile (TP=1 vs TP=2)

Tested `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v` on vllm-omni
at 832x480, 33 frames, 30 inference steps, seed=42, flow-shift=5.0.

### Measured

| Run      | Config          | Steady (per GPU) | Peak (per GPU) | Inference |
|----------|-----------------|------------------|----------------|-----------|
| smoke_01 | 1× A100, TP=1   | 37.8 GB          | 79.95 GB       | 73.8 s    |
| smoke_02 | 2× A100, TP=2   | 32.3 GB          | 76.9 GB        | 145.9 s   |

TP=2 used **more total memory** (64.6 GB aggregate vs 37.8 GB), barely reduced
peak (76.9 vs 79.95 GB per GPU), and was **~2× slower** end-to-end.

### Why TP=2 doesn't help

HunyuanVideo-1.5 I2V is a stack, and only the ~13B DiT shards under
`--tensor-parallel-size`:

- **DiT transformer** (~13B, ~26 GB BF16) — shards with TP ✅
- **Text encoder** (LLaMA-class, ~7B, ~14 GB BF16) — replicated on every GPU
- **VAE encoder/decoder** (~400M) — replicated on every GPU
- Scheduler state, CFG unconditional buffers, activation reservations

Peak memory is dominated by **VAE decode**, not the DiT forward pass: the VAE
decodes DiT latents into full 33×480×832×3 pixel tensors non-sharded, and that
activation pyramid hits every GPU in the pipeline. The `peak_memory_mb` the API
returns is the max on any *single* GPU, not aggregate — which is why TP=2 still
reports 76.9 GB even though the DiT forward under TP=2 peaks much lower.

TP=2 also adds per-step all-reduce overhead on every DiT layer, and diffusion
inference has no KV cache / batched decode to amortize that cost, so wall time
roughly doubles.

### When TP *is* useful on this stack

Concurrent-request throughput, not single-request peak memory. Not relevant for
the autoregressive rollout use case — we issue one request at a time.

### Levers that actually reduce peak memory (priority order)

1. **VAE tiling** — biggest single win. Decodes in spatial/temporal tiles
   instead of one shot. Referenced for 720p in
   `examples/online_serving/image_to_video/run_server_hunyuan_video_15.sh`.
   Reach for this first if 480p/33-frame ever hits OOM under changed conditions.
2. **FP8 DiT quantization** (`--quantization fp8`) — halves the DiT shard;
   does nothing for VAE or text encoder.
3. **Offload text encoder to CPU after prompt encoding** — frees ~14 GB during
   the DiT+VAE steps.

### Conclusion

Stay on **single GPU (TP=1)** for the world-models project. The default
single-GPU rule applies after all; the "use both GPUs" detour was a
misdiagnosis — peak memory isn't a TP-addressable problem on this model.

---

## 2026-04-14 — Stage 2: first autoregressive rollout

Ran `rollout.py` for 5 steps on single GPU (TP=1), 832×480, 33 frames per
clip, 30 inference steps, fixed seed base=42 (+ step index). All outputs
under `runs/20260414_091348/`.

**Action sequence:**
0. the red ball rolls slowly to the right across the beige table
1. the red ball continues rolling to the right across the beige table
2. the red ball slows down and comes to rest on the beige table
3. the camera pushes in slowly toward the red ball on the beige table
4. the red ball rolls forward toward the camera across the beige table

**Timing & memory:** rock-solid across all 5 steps — 73.4–75.0 s inference,
75.2–75.3 s wall, **79952 MB peak every step** (identical to smoke_01, so
memory is deterministic and ~50 MB from OOM).

### What we saw in the video

Observations (watched by hand, clip-by-clip):

- **Step 0** — ball rolls to the right as instructed. Clean.
- **Step 1** — ball suddenly rolls **to the left** (wrong direction), slightly
  bouncing, and a strange drawing appears in the scene. The ball also no
  longer looks like the same ball — the image has moved toward a more
  photorealistic style.
- **Step 2** — ball rolls right again. A specular highlight ("shine") appears
  as if from a light behind the camera. It does **not** slow down or stop
  before the end of the clip — action ignored.
- **Step 3** — camera push-in does not happen. The ball **rotates in place**
  instead of translating.
- **Step 4** — prompted "rolls forward toward camera"; ball moves **away**
  from the camera instead.

Frame file sizes also grow: 9 KB (synthetic start) → 44 → 70 → 73 → 83 → 74
KB. The synthetic flat-color start compresses well; subsequent frames get
larger as the model adds texture and detail — a compression-side signature of
the same drift we see with our eyes.

### Architectural analysis — why all of these fail for the same reason

Every failure maps to a single root cause:
**`state = one PNG` is not enough state.** The transition function
`next_state = I2V(state, action)` takes a single image, and from a single
image the model cannot recover:

1. **Velocity / momentum.** The step 0→1 direction flip is the cleanest
   example. The ball was rolling right, but the last frame is just a still
   image of a ball sitting somewhere on the table. Step 1's model has no way
   to know the ball was moving right, so it picks a direction effectively at
   random (or one the prompt happens to bias). Momentum information is
   destroyed at every clip boundary, because the boundary is a single frame.

2. **Object and scene identity.** "Not the same ball" and "a strange drawing
   appears" are drift on the model's own output manifold. Every time we
   re-condition on a generated frame, the next generation gets pulled
   slightly toward "what HunyuanVideo likes to draw" — more photorealistic,
   more textured, more stylized than our synthetic flat-color start. After 5
   hops the ball has specular highlights, the table has texture, and new
   objects can hallucinate in. Nothing in the loop enforces persistent
   identity.

3. **Spatial grounding.** "Camera pushes in → ball rotates in place" and
   "rolls forward toward camera → moves away" both fail on the same thing:
   there is no persistent 3D understanding across steps. "Forward" relative
   to what? The model interprets each prompt against whatever scene it
   imagines from the current frame, and that scene's geometry is reinvented
   every step.

4. **Temporally extended actions.** "Slows down and comes to rest" didn't
   happen because there is no dynamics model under the hood. The I2V model
   does a text-conditioned generation over 33 frames; "comes to rest" is a
   temporally-extended constraint that a single clip can technically satisfy
   but is not required to.

### What this confirms

This is exactly the lesson the tangible-first-step was meant to teach:

> **Pretrained I2V + feedback loop ≠ world model.** The state bottleneck
> (1 image) destroys information at every step, and the pixel-space
> round-trip pulls each successive state toward the model's output manifold.

A real world model needs either (a) richer state that preserves momentum,
identity, and geometry, or (b) a transition function that sees more than one
frame of history — or both.

### Open directions (to pick from next session)

In rough order of increasing ambition:

1. **Richer conditioning on the same kind of model.** Use an I2V variant that
   accepts more than one conditioning frame, or a video-continuation model
   that takes a clip rather than a frame. Candidates worth checking in
   vllm-omni: CogVideoX variants, Wan2.2 (start+end), FramePack-style clip
   continuation. Addresses momentum loss directly, no training required.
   Cheapest win.
2. **Latent-space rollout.** Stop round-tripping through pixels between
   steps. Encode the generated clip to a VAE latent plus some motion summary
   and pass that as the next "state". Doesn't fix the absence of physics but
   should reduce the cartooning / manifold-drift failure mode.
3. **Learned dynamics (DIAMOND / Genie / GameNGen direction).** Train a small
   transition network in latent space, reuse the pretrained VAE only for
   decoding. This is actual world modeling. Much bigger project.
4. **Representation-only (V-JEPA-style).** Give up on pixel reconstruction
   between steps; predict future representations instead. Good for planning,
   not for watching rollouts. Even bigger commitment.

### Artifacts

- `runs/20260414_091348/rollout.mp4` — concatenated 5-clip rollout
- `runs/20260414_091348/clips/step_{00..04}.mp4` — individual generations
- `runs/20260414_091348/frames/frame_{000..005}.png` — start frame + one last
  frame per step (each served as conditioning for the next step)
- `runs/20260414_091348/metadata.json` — prompts, video IDs, per-step timing
  and peak memory

### Follow-ups for next session

- Pick a direction from the list above.
- Before committing, sanity-check whether vllm-omni actually serves any I2V
  model that takes multi-frame / clip conditioning — option 1 collapses if
  no model on this serving stack supports it.
- Consider whether we even want vllm-omni as the serving layer for options
  2–4, or whether at that point it's easier to drop to raw `diffusers` and a
  Python script (no HTTP hop, no multiproc/stage machinery, full control
  over the latent path).

---

## 2026-04-14 — Pivot to inference-speed contribution; RFC #1987 and SOTA scan

Reframed the project goal. Instead of trying to make the toy rollout into
a "better" world model, the new goal is to **contribute to inference-speed
improvements in vllm-omni**, using the toy rollout as a profiling harness
whose compute shape matches what the community is building.

### RFC #1987 "World Model Support" — what vllm-omni is building

URL: https://github.com/vllm-project/vllm-omni/issues/1987.
Summary of the scan (full bibliographic list in `related_works.md`):

- RFC by TKONIY (Yangshen Deng), state: open. Partitions world-model
  workloads into **robotics**, **interactive video/game**, and
  **simulation**. The near-term (P0) target is robotics: land **DreamZero**
  via PR #2162 and build an **OpenPI-style multi-turn realtime WebSocket
  server** on `/v1/world/stream`. Interactive video and PageAttention for
  autoregressive diffusion are explicitly deferred (P2).
- Maintainers (hsliuustc0106, Gaohan123) insist the session/state machinery
  must be generic — a `SessionStore` reusable by TTS, world models, and
  future realtime apps — and that the KV-buffer MVP should be
  **preallocated** (fixed `max_sessions`) with a documented path to
  PageAttention as a later follow-up.
- Infrastructure deltas called out by the RFC: **action modality** I/O, a
  **multi-turn stateful realtime API** (neither the TTS WebSocket nor
  vLLM's Realtime API supports context accumulation today), and an
  **autoregressive chunked diffusion engine with blockwise-causal
  attention**.
- Sibling RFC **#2727** ("Support 3D World Model (VGGT)") is architecturally
  the closest existing design to what interactive-video rollouts need: a
  new `stage_type: "reconstruct"` engine with **dLLM-style chunked
  computation + a sliding-window KV cache**. VGGT is feed-forward 3D recon,
  not video-DiT, but the architectural pattern transfers.
- Prereqs already merged: CFG-parallel refactors #2063/#2160/#2078/#2423.
  Open and load-bearing: #2073 (step-wise / continuous batching engine
  refactor — needed before any cross-chunk KV-cache work), #2590 (WIP AR
  KV-prefix reuse experiment in DiT), #2632 (per-role attention backend
  config, a possible host for blockwise-causal attention).
- Implementation difficulties quoted in the RFC and siblings:
  *"Neither vllm-omni's TTS WebSocket nor vLLM's Realtime API support
  multi-turn context accumulation"* (core blocker). *"The current
  architecture executes diffusion inference end-to-end per request,
  blocking the engine until completion"* (#2073). PageAttention for AR
  diffusion has no design yet.

### DreamZero (arXiv 2602.15922, NVIDIA, Feb 2026)

**Correction notice:** the DreamZero summary below is from an earlier
reading and contains two silent errors that were fixed later in the
Phase D entry. (1) The 38× speed-up is GB200-only per Table 1; on H100
the cumulative speed-up caps at 9.6×, and CFG parallelism is
multi-GPU from row 2 onwards. (2) The step progression is 16 → 4 → 1,
not "stays at 16" as I initially thought — DiT Caching does intra-
chunk velocity reuse (16 → 4 effective steps), and DreamZero-Flash is
a *training-time* noise-schedule change that further reduces to 1
step. Also: the pre-optimization latency is 5.7 s per chunk and the
bimanual action horizon is 1.6 s per chunk; pypdf had swallowed the
leading digits in my first reading. See the Phase D journal entry
below for the faithfully-reproduced Table 1 and the full list of
corrections.

Fetched the paper. Critical corrections to earlier misreading:

- **DreamZero is literally a video DiT.** *"DreamZero, a 14B robot
  foundation model built upon a pretrained image-to-video diffusion
  backbone (Team Wan, 2025)"* and *"14B autoregressive diffusion
  transformer trained with a teacher-forcing chunk-wise video denoising
  objective"*. Same model family as Wan2.2 in `vllm_omni/diffusion/models/`.
- **Joint video + action generation.** Predicts future frames and actions
  together; authors introduce the term **World Action Model (WAM)**.
  Training objective is teacher-forcing chunk-wise denoising.
- **38× inference speedup, decomposed into three categories:** (1)
  algorithmic — decoupled video/action denoising schedules
  ("DreamZero-Flash"); (2) system-level — parallelism and caching
  strategies; (3) low-level — quantization and CUDA kernel tuning. This
  delivers real-time closed-loop control at ~7 Hz.
- **Target domain:** robotics manipulation (AgiBot G1, Franka, YAM) — but
  the compute shape is pure video-DiT, with the KV-cache/caching work in
  category (2) directly benefiting from RFC #1987's infrastructure plan.

The earlier claim in the Stage 2 entry that robotics world models are
"ViT+LLM+action-head and do not benefit from diffusion KV cache" is wrong
and should be read as superseded by the paragraphs above. DreamZero IS the
P0 target of #1987 AND IS a chunked-AR video DiT.

### StreamDiffusionV2 (arXiv 2511.07399, UT Austin/Berkeley/Stanford/MIT, Nov 2025)

Training-free serving system that adapts existing video DiTs (Wan family —
CausVid, Self-Forcing) to live-streaming SLOs. On 4× H100, without
TensorRT or quantization: **58.28 FPS on a 14B model** and **64.52 FPS on a
1.3B model**, with 0.5 s time-to-first-frame, supporting 1–4 denoising steps.

Core techniques (each is a concrete reference for what a vllm-omni
contribution could implement):

1. **SLO-aware batching scheduler** — reformulates input from
   `1×T×H×W` (offline, T=81+) to `B×T'×H×W` with small `T'` (e.g. 4
   frames), adapts `B` to hardware load.
2. **Sink-token–guided rolling KV cache** — bounded cache across chunks,
   anchored by a small set of attention-sink tokens (StreamingLLM-style)
   that pin the distribution; the rest rolls. This is what prevents drift
   over hour-long streams.
3. **Pipeline orchestration across denoising steps AND network layers** —
   naive pipeline parallelism doesn't scale under per-frame latency
   constraints; they split the DiT across GPUs along both axes.
4. **Motion-aware noise controller** — per-chunk noise schedule adaptation
   to avoid blur/ghosting during high-speed motion.
5. **Adaptive sink-token refresh + RoPE reset at chunk boundaries** —
   drift control for multi-hour streams.

### How DreamZero, StreamDiffusionV2, and RFC #1987 relate

- **DreamZero and StreamDiffusionV2 are the two ends of a tradeoff.**
  DreamZero retrains the model natively as a chunk-AR WAM and bundles
  training + system + low-level optimizations for a 38× speedup that
  bundles everything together. StreamDiffusionV2 takes an existing causal
  video DiT off the shelf (no training) and gets real-time latency on 4×
  H100 from pure serving-system design.
- **StreamDiffusionV2's category (2) maps almost directly onto RFC #1987's
  "system-level parallelism and caching" plan.** Rolling KV cache, chunked
  scheduling, pipeline parallelism across denoising steps and layers. If
  their code is open, it is the single best reference for the vllm-omni
  implementation.
- **Wan2.2 is the common backbone.** DreamZero is built on Wan I2V;
  StreamDiffusionV2's reference efficient DiTs (CausVid, Self-Forcing) are
  distillations of Wan-2.1-T2V. For a profiling prototype, Wan2.2 is the
  most directly representative model family. HunyuanVideo-1.5 is a close
  cousin (same class of 13–14B dense video DiT) and stays as a cross-check.

### Operating point we should actually profile

The streaming regime both papers target is very different from what we ran
in Stage 2:

| Parameter | Stage 2 baseline | Streaming regime (SDv2 / DreamZero) |
|---|---|---|
| Frames per forward pass | 33 | **4** |
| Denoising steps per forward | 30 | **1–4** |
| Context across chunks | 1 last frame (lossy) | Rolling KV cache (bounded) |
| Latency target | 74 s / clip | 17–150 ms / chunk |
| Hardware | 1× A100 80GB | 4× H100 |

Our hardware cannot replicate 17 ms per chunk and that is not a goal. The
goal is to **measure our baseline at both operating points**, attribute
the gap to specific components (hardware, chunk size, steps, caching
absence, multi-GPU), and identify which components are addressable inside
vllm-omni.

### Plan — "computationally representative prototype"

Single GPU, profiling first, no training, no vllm-omni internals yet. All
artifacts under `world_models/profiles/` and `world_models/scripts/`.

**Phase 0 — Pre-flight reconnaissance (10–15 min).** Answer three
questions before committing to any sweep:
1. Does `--enable-diffusion-pipeline-profiler` actually populate
   `stage_durations` cleanly on HunyuanVideo-1.5 I2V? Run one job and
   inspect.
2. Is there a Wan2.2-I2V or Wan2.2-VACE checkpoint that fits one A100 at
   steady state (not the A14B MoE we rejected on day one)?
3. Does `pipeline_wan2_2_vace.py` have a serve path (stage config / CLI
   entry), or is it orphaned code in `main`?

**Phase A — Baseline compute breakdown (25–60 min).** Run one
HunyuanVideo-1.5 I2V job at our current config with the profiler on and
extract `stage_durations`. If Wan2.2-I2V fits, do the same for it.
Deliverable: stacked-bar of % time per stage
(`text_encoder.forward` / `vae.encode` / `diffuse` / `vae.decode`).
Answers "is KV cache on the critical path at all, or is VAE decode the
dominant cost?"

**Phase B — Operating-point sweep (25–40 min).** Fix model, resolution,
seed. Vary `num_inference_steps` ∈ {30, 16, 8, 4, 2, 1}. Independently,
vary `num_frames` ∈ {33, 16, 8, 4}. Record per-call wall time +
`stage_durations`. Deliverable: two plots (latency vs. steps, latency vs.
frames). Tells us where on the cost curve we are and what the "streaming
operating point" costs on this hardware.

**Phase C — Context-length proxy (20–60 min).** At the streaming
operating point from B, measure the cost of re-processing growing
context — the thing a rolling KV cache would eliminate. Preferred path:
Wan2.2-VACE with `reference_images` of length ∈ {1, 2, 4, 8}. Fallback if
VACE is not servable: HunyuanVideo I2V with a tile-strip hack (tile K=1,
2, 4 frames horizontally into one reference image). Deliverable: slope of
per-call latency vs. `K`, quoted as "ms of re-computation per extra
context frame — this is what a rolling KV cache would save."

**Phase D — Literature cross-check (10–15 min).** Tabulate our
measurements against DreamZero's 150 ms / 7 Hz on H100 and
StreamDiffusionV2's 58.28 FPS on 4× H100 / 14B. Attribute our gap to
hardware / chunk size / denoising steps / caching absence / single GPU.
Each gets a rough multiplier. Deliverable: one table.

**Phase E — Journal writeup + recommendation (15–20 min).** Final journal
entry summarizing phases A–D and naming the one highest-leverage
contribution target for vllm-omni given what we measured.

### Explicit non-goals

- Not chasing real-time throughput.
- Not training anything.
- Not implementing a KV cache. We measure the delta a KV cache would
  eliminate.
- Not touching vllm-omni internals. All sweeps run against the HTTP API or
  small Python wrappers.
- Not multi-GPU. One A100, chosen via `nvidia-smi` before each run.

### Risks and fallbacks

- **Wan2.2 doesn't fit or isn't servable.** Phase A baseline becomes
  HunyuanVideo-only; Phase C uses the tile-strip hack on HunyuanVideo. We
  lose directness-to-DreamZero but the slope measurements still
  generalize.
- **`--enable-diffusion-pipeline-profiler` has surprises.** Fallback: wrap
  inference in `OmniTorchProfilerWrapper` from a side script that imports
  the pipeline directly, still inside the existing Docker image.
- **Phase overruns.** Phases are ordered so each earlier phase is useful
  on its own — we don't need all five to produce a useful journal entry.

### Expected wall time

- Optimistic (HunyuanVideo only, no new downloads): ~1.5–2 hours.
- Realistic (one fallback, one new Wan checkpoint): ~2.5–3.5 hours.
- Pessimistic (profiler surprises + Wan doesn't fit + VACE orphaned): ~4
  hours.

---

## 2026-04-14 — Execution: Phases 0–D, measured results

Executed the plan above. Actual wall time ~1h15m (optimistic branch:
HunyuanVideo profiler worked on the first try, Wan2.1-VACE-1.3B fit at
18 GB weights / 26 GB steady state). Phase C was redefined mid-flight —
see below.

Full numbers in `profiles/`. Headline findings below.

### Phase 0 — Pre-flight reconnaissance (~15 min)

1. `--enable-diffusion-pipeline-profiler` is a real CLI flag
   (`vllm_omni/entrypoints/cli/serve.py:405`). It populates the
   `DiffusionPipelineProfiler` wrapper on pipeline methods, and the
   per-stage durations appear in container stderr as
   `[DiffusionPipelineProfiler] <PipelineClass>.<stage> took Ns`.
2. Wan variants that fit on 1× A100: **`Wan-AI/Wan2.1-VACE-1.3B-diffusers`**
   (1.3B dense, 18 GB weights, multi-frame conditioning support via
   `reference_images: list`) and several others. The 14B dense VACE
   would also fit. The A14B MoE variants we rejected on day one remain
   rejected. Downloaded `Wan2.1-VACE-1.3B` (18 GB) into the HF cache.
3. VACE has a real serve path: `Wan22VACEPipeline` in
   `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_vace.py` is
   registered in `diffusion/registry.py` and listed in
   `docs/models/supported_models.md`. Not orphaned.

**Vllm-omni bug discovered while running Phase A.** The profiler flag
populates the pipeline's `_stage_durations` dict (visible in logs) but
`VideoGenerationArtifacts.stage_durations` in the API response is always
`{}`. There is a broken link between the pipeline and the engine's
response assembly. Workaround: scrape container stderr for profiler
lines — the parser is in `scripts/_log_profile.py`. Upstream fix would
be a small two-file PR — identified as **contribution target #1** in
Phase D.

### Phase A — HunyuanVideo-1.5 I2V baseline breakdown

One job, 832×480, 33 frames, 30 denoising steps, guidance 6.0,
flow_shift 5.0, seed 42. Two consecutive runs confirmed results agree
within 0.5 s. First container-internal warmup run is **not**
representative and was excluded (its `forward` was 36.84 s vs the
real 72.4 s — likely a startup validation with a different config).

| Stage | Seconds | % of 73.7 s |
|---|---:|---:|
| text_encoder.forward (×2 for CFG) | 0.197 | 0.27 |
| vae.encode | 0.109 | 0.15 |
| **DiT denoising loop (derived)** | **65.621** | **89.04** |
| vae.decode | 6.470 | 8.78 |
| framework overhead (derived) | 1.303 | 1.77 |

**DiT denoising is 89% of total inference time** at the offline
operating point. VAE decode is 9%. Text encoder and VAE encode are
noise. This directly confirms that KV-cache-for-DiT / blockwise-causal
attention work (RFC #1987 category and #2590) is on the critical path
for this workload — not a secondary optimization.

Peak memory remains 79.95 GB (50 MB from OOM on one 80 GB A100), same
as pre-profiler runs — the profiler itself adds negligible memory.

Artifacts: `profiles/phase_a_hunyuanvideo_1_5_i2v.json`,
`profiles/phase_a_breakdown.png`.

### Phase B — Operating-point sweep

HunyuanVideo only (for the full curves; Wan validated the endpoints in
Phase A'). One axis at a time.

**Sweep 1 — `num_inference_steps` ∈ {30, 16, 8, 4, 2, 1}, frames=33:**

| steps | inference_s | DiT (s) | VAE decode (s) |
|---:|---:|---:|---:|
| 30 | 74.69 | 66.22 | 6.69 |
| 16 | 44.16 | 35.95 | 6.39 |
| 8  | 26.18 | 18.09 | 6.26 |
| 4  | 17.38 |  9.04 | 6.54 |
| 2  | 12.85 |  4.51 | 6.54 |
| 1  | 10.45 |  2.27 | 6.36 |

DiT cost is **linear in steps at ~2.25 s/step**, rock-steady across the
whole range. VAE decode is essentially constant at 6.4 s — it doesn't
care how many denoising steps ran. The residual ~1.8 s (inference − DiT
− VAE decode) is text encoder + VAE encode + framework, also constant.

Single-step inference (1 denoising step) already gets us **7× speedup**
over the 30-step baseline, even at 33 frames. Most of this is not
surprising — it's just "fewer steps" — but it confirms there's no
superlinear scheduler overhead hiding in the loop.

**Sweep 2 — `num_frames` ∈ {33, 16, 8, 4}, steps=30:**

| frames | inference_s | DiT (s) | VAE decode (s) | peak (MB) |
|---:|---:|---:|---:|---:|
| 33 | 75.47 | 67.14 | 6.57 | 79952 |
| 16 | 29.34 | 26.42 | 2.01 | 69754 |
|  8 | 16.55 | 15.29 | 0.73 | 69758 |
|  4 | 11.43 | 10.86 | 0.20 | 47052 |

DiT scales **superlinearly** with frame count. From 4 → 33 frames
(8.25×), DiT climbs from 10.86 → 67.14 s (6.18×). Per-frame cost rises
from 1.11 s/frame (4→8 range) to 2.40 s/frame (16→33 range). That's
the attention O(S²) signature showing up cleanly in the measurement.
VAE decode is also strongly frame-dependent (0.20 → 6.57 s, 33×), as
expected for 3D decoders. Peak memory drops meaningfully at shorter
clips — the 80 GB OOM threat goes away at ≤16 frames.

Artifacts: `profiles/phase_b_sweep.json`, `profiles/phase_b_steps.png`,
`profiles/phase_b_frames.png`.

### Phase C (redefined) + Phase A' — Wan2.1-VACE-1.3B cross-check

Phase C as originally planned collapsed: the `/v1/videos` HTTP endpoint's
`input_reference` field takes a **single** `UploadFile`, not a list.
Multi-frame VACE conditioning is not exposed on this API surface today
(it is exposed on `/v1/chat/completions` via a different code path).
Direct pipeline invocation inside the container would work but was not
worth the session time. Crucially, **Phase B's `num_frames` sweep
already measures the thing a rolling KV cache would save** — attention
cost as a function of total sequence length — so the context-length
sweep was redundant.

Re-purposed the time into **Phase A'**: a second-model cross-check on
Wan2.1-VACE-1.3B so we could argue that the 89% DiT-dominance finding
is not HunyuanVideo-specific.

**Offline baseline (832×480, 33 frames, 30 steps) cross-check:**

| Stage | HunyuanVideo-13B | Wan2.1-VACE-1.3B |
|---|---:|---:|
| Total inference (s) | 73.70 | 53.30 |
| Peak memory (GB) | 80.0 | 26.5 |
| DiT denoising (s / %) | 65.62 / 89.04 | 47.45 / 89.02 |
| VAE decode (s / %) | 6.47 / 8.78 | 1.87 / 3.51 |
| VAE encode (s / %) | 0.11 / 0.15 | 2.20 / 4.13 |
| Text encoder (s / %) | 0.20 / 0.27 | 0.11 / 0.21 |
| Framework (s / %) | 1.30 / 1.77 | 1.67 / 3.13 |

**89% DiT dominance holds on both models** despite a 10× parameter
difference. The reason is subtle and worth internalising: the attention
sequence length is identical (33 frames × 480p latent tokens on both
models), and attention cost dominates over FFN cost at this sequence
length. The 10× weight ratio only moves the FFN portion of DiT, which
is the minority contribution. DiT wall time is only 1.4× faster on
Wan-1.3B than HunyuanVideo-13B (47 s vs 66 s), not 10×.

**Streaming operating point on Wan (4 frames, 4 steps) — measured:**

- Total inference: **2.585 s**
- DiT denoising: 1.536 s (59.5%)
- VAE encode: 0.380 s (14.7%)
- VAE decode: 0.295 s (11.4%)
- text_encoder: 0.114 s (4.4%)
- Framework: 0.261 s (10.1%)
- Peak memory: 26.5 GB

Ratio to the Wan offline baseline on the same hardware and the same
model: 53.30 / 2.585 ≈ 20.6×. This is a fact about our own two
configurations; it is not a claim about what fraction of any paper's
reported speed-up is attributable to hyperparameter choice.

HunyuanVideo at the same streaming point was **not measured** and we
are not attaching a number to it. A rough Phase B extrapolation (DiT
linear in steps: `10.86 × 4/30 ≈ 1.45 s` + VAE + framework) suggests
~2 s per call on 1× A100, but it is an extrapolation, not a measurement.

**Most important finding across Phase A/B/A':** at the streaming-ish
operating point on Wan, the DiT-dominance from the offline baseline
(89%) drops to ~60%, because VAE encode/decode and framework overhead
have fixed components that become non-negligible fractions of a much
smaller chunk budget. See Phase D for the contribution-target
implications.

Artifacts: `profiles/phase_a_Wan2.1-VACE-1.3B-diffusers.json`,
`profiles/phase_a_prime_wan_streaming.json`.

### Phase D — Literature cross-check

Full writeup and the faithfully-reproduced Table 1 from DreamZero are
in `profiles/phase_d_cross_check.md`. An earlier draft of both this
journal section and the cross-check file contained invalid cross-
baseline comparisons; it has been rewritten to report measurements and
reference-paper numbers side-by-side **without** attempting to
attribute a cross-baseline gap to specific optimization categories.

Key corrections relative to the earlier draft:

1. **DreamZero's 5.7 s baseline, not 7 s.** The earlier reading missed
   the leading digit because pypdf split the number across a line
   break. Corresponding "1.6 seconds per chunk" is also 1.6 s, not 6 s.
2. **DreamZero's step-reduction progression is 16 → 4 → 1**, not
   "stays at 16". **DiT Caching** (velocity reuse between successive
   denoising steps within one chunk, based on cosine similarity of
   flow-matching velocities) reduces the effective step count from
   16 to 4. **DreamZero-Flash**, a *training-time* change that
   decouples the noise schedules for video and action modalities,
   further reduces to 1 step. Table 3 in the paper shows the empirical
   cost: naive 1-step inference loses 31 task-progress points on
   table-bussing (83% → 52%) while 1-step DreamZero-Flash loses only
   9 (83% → 74%).
3. **The 38× headline is GB200-only.** It requires NVFP4 weights +
   activations (a Blackwell-generation feature) and DreamZero-Flash.
   The cumulative speed-up **on H100 caps at 9.6×** per Table 1 in the
   paper. The NVFP4 and DreamZero-Flash rows are dashed for H100.
4. **CFG Parallelism is multi-GPU.** DreamZero distributes the two
   classifier-free-guidance forward passes across two GPUs as the
   first optimization row in Table 1, so the post-baseline
   configuration in the paper is never single-GPU.
5. **DreamZero's "DiT Caching" and RFC #1987 / StreamDiffusionV2's
   "rolling KV cache across chunks" are different optimizations** and
   should not be conflated. DiT Caching is intra-chunk velocity reuse
   within a denoising loop. Rolling KV cache is cross-chunk prefix
   reuse across autoregressive chunk boundaries. DreamZero does mention
   KV caching for efficiency in §3.1 but does not break it out with its
   own row in Table 1.

Given those corrections, the cross-check is limited to reporting what
each source measured on its own stack: our numbers on 1× A100, Table 1
from DreamZero on H100 / GB200, and StreamDiffusionV2's 58 FPS /
64 FPS numbers on 4× H100. No attempt is made in this writeup to
attribute the gap between our measurements and theirs to specific
optimization categories — an earlier version tried, and the
attribution did not survive cross-reading.

### Phase E — Contribution targets for vllm-omni (revised)

In priority order by a qualitative assessment of leverage and
feasibility on our 1× A100 setup, with attribution deliberately
conservative:

1. **Fix the `stage_durations` plumbing bug.** Small two-file patch,
   zero risk, directly useful: the `DiffusionPipelineProfiler` already
   populates `_stage_durations` on the pipeline, but the engine never
   forwards it to `VideoGenerationArtifacts.stage_durations` in the
   response. Every future profiling contribution needs this. **Do this
   first** — both as a warmup and as a concrete "I am now a contributor"
   moment.
2. **Framework / per-call overhead reduction.** Our measurement shows
   ~260 ms per call of framework overhead (derived) at the
   streaming-ish operating point on Wan-1.3B, 4 frames × 4 steps.
   That is ≥100% of the 150 ms chunk budget the DreamZero paper
   targets. Profile this more carefully from both client and server
   sides before committing, then argue the concrete numbers on
   RFC #2073 (the step-wise engine refactor). If #2073 is actively in
   progress we may be able to contribute a small focused piece of it.
3. **Rolling KV cache + blockwise-causal attention for Wan /
   HunyuanVideo DiTs.** The core of RFC #1987 and the most direct
   match for StreamDiffusionV2's cross-chunk caching. **We do not have
   a measured speed-up number for this on our hardware**, and the
   numbers in DreamZero's Table 1 and StreamDiffusionV2's headline do
   not transfer across hardware and training regimes without careful
   caveats. If we want a defensible "this much speed-up is on the
   table" argument for the RFC, we need to measure it ourselves on a
   causal Wan variant first. Until we do, target 3 is a research
   commitment, not a quantified target.

Optional follow-up work that would sharpen target 3: reproduce
StreamDiffusionV2's rolling-KV-cache recipe (sink tokens + sliding
window) on a causal Wan checkpoint via `diffusers` directly, measure
the speed-up vs. a naive chunked baseline on our A100, and contribute
the measurement either back as a comment on RFC #2590 or as a
stand-alone benchmark PR.

### Things we deliberately did not do this session

- Run Phase B on Wan. The cross-check on one streaming point was enough
  to argue the pattern holds; the two offline baselines already agreed
  on the 89% DiT number. Full sweeps on Wan would be useful eventually
  but were not decision-relevant today.
- Fit a polynomial to the Phase B frame-sweep to separate attention
  (quadratic) from FFN (linear) components. Tried the math in passing;
  a naive `a·F² + b·F + c` fit undershoots the 33-frame measurement by
  ~7 s, meaning there's either a super-quadratic memory-bandwidth
  effect at high F or VAE encode also grows non-linearly. Worth doing
  properly before contribution #3 above, but not critical today.
- Measure the profile on a real short-horizon rollout with back-to-back
  chunks. The `rollout.py` from Stage 2 can become the harness for
  this; worth doing once the `stage_durations` plumbing is fixed so the
  per-chunk breakdowns come out of the API directly.

### Artifacts written this session

```
world_models/
  JOURNAL.md                        (this file, updated)
  scripts/
    _common.py                      HTTP client helpers
    _log_profile.py                 Container-log scraper for stage_durations
    run_phase_a.py                  Phase A baseline runner
    run_phase_b.py                  Phase B sweep runner
    plot_phase_ab.py                matplotlib plots
  profiles/
    phase_a_hunyuanvideo_1_5_i2v.json
    phase_a_Wan2.1-VACE-1.3B-diffusers.json
    phase_a_prime_wan_streaming.json
    phase_b_sweep.json
    phase_a_breakdown.png
    phase_b_steps.png
    phase_b_frames.png
    phase_d_cross_check.md          Phase D analysis and gap attribution
```
