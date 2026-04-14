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
