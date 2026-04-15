# Phase D — Literature cross-check

All measurements in this document are on **one A100 80GB PCIe**, TP=1,
BF16, no quantization, served via vllm-omni's `HunyuanVideo15I2VPipeline`
or `Wan22VACEPipeline`. Per-stage timings are scraped from the
`DiffusionPipelineProfiler` lines in the container logs (the API
`stage_durations` field is empty — a vllm-omni bug we noted but did not
fix in this session).

This document reports **what we measured** and **what the reference
papers report**, and deliberately does *not* attempt to attribute the
gap between them to specific optimization categories. Neither paper
provides a decomposition that matches our hardware and serving
configuration closely enough for a component-wise attribution to be
defensible, and an earlier draft of this file made invalid
cross-baseline comparisons that we have removed.

## 1. What we measured

### 1a. Offline operating point (our current default)

Config: 832×480, 33 frames, 30 denoising steps, guidance 6.0,
flow_shift 5.0, seed 42, 1× A100 80GB.

| Model | Total inference (s) | DiT loop (s) | DiT % | VAE decode (s) | VAE encode (s) | Text enc (s) | Framework (s) | Peak (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HunyuanVideo-1.5-I2V (13B) | 73.70 | 65.62 | 89.0 | 6.47 | 0.11 | 0.20 | 1.30 | 80.0 |
| Wan2.1-VACE-1.3B | 53.30 | 47.45 | 89.0 | 1.87 | 2.20 | 0.11 | 1.67 | 26.5 |

DiT denoising is ~89% of inference on both models. VAE decode is 9% on
HunyuanVideo, 3.5% on Wan. Despite the 10× weight ratio, Wan's DiT is
only 1.4× faster than HunyuanVideo's — consistent with attention
(O(S²)) dominating over FFN (O(params)) at this sequence length, where
S is the same on both models.

### 1b. Phase B sweeps (HunyuanVideo only)

Steps sweep, num_frames fixed at 33:

| steps | inference (s) | DiT (s) | VAE decode (s) |
|---:|---:|---:|---:|
| 30 | 74.69 | 66.22 | 6.69 |
| 16 | 44.16 | 35.95 | 6.39 |
| 8  | 26.18 | 18.09 | 6.26 |
| 4  | 17.38 |  9.04 | 6.54 |
| 2  | 12.85 |  4.51 | 6.54 |
| 1  | 10.45 |  2.27 | 6.36 |

DiT cost is linear in steps at ~2.25 s/step. VAE decode is essentially
constant — it runs once at the end regardless of denoising step count.

Frames sweep, num_inference_steps fixed at 30:

| frames | inference (s) | DiT (s) | VAE decode (s) | peak (MB) |
|---:|---:|---:|---:|---:|
| 33 | 75.47 | 67.14 | 6.57 | 79 952 |
| 16 | 29.34 | 26.42 | 2.01 | 69 754 |
|  8 | 16.55 | 15.29 | 0.73 | 69 758 |
|  4 | 11.43 | 10.86 | 0.20 | 47 052 |

DiT scales superlinearly with frame count (4 → 33 frames is 8.25× more
frames, 6.18× more DiT time). This is the attention cost curve showing
up cleanly.

### 1c. One streaming-ish operating point on Wan (measured)

Config: 832×480, **4 frames**, **4 denoising steps**, seed 42.
Wan2.1-VACE-1.3B on 1× A100.

| Stage | Seconds | % of 2.585 s |
|---|---:|---:|
| DiT denoising | 1.536 | 59.5 |
| VAE encode | 0.380 | 14.7 |
| VAE decode | 0.295 | 11.4 |
| Framework (derived) | 0.261 | 10.1 |
| text_encoder.forward | 0.114 |  4.4 |

Total: **2.585 s**. Peak memory 26.5 GB.

For HunyuanVideo at the same operating point we did not run a
measurement; linear extrapolation from the Phase B step sweep (frames=4
at 30 steps was 10.86 s DiT; scaling to 4 steps gives ~1.45 s DiT) plus
the fixed-cost stages gives a rough estimate around 2.0–2.5 s per call,
but we are not attaching a number to it without measuring.

### 1d. What happens to the stage-time mix at small chunks

At the offline baseline, DiT denoising is 89% and everything else is
small. At the streaming operating point on Wan, DiT drops to 60%; VAE
encode+decode climbs to 26%; framework overhead climbs to 10%. The
reason is that VAE and framework have large fixed-cost components
(kernel launches, memory staging, Python/HTTP plumbing) that become
non-negligible fractions of a much smaller total budget. This changes
what the highest-leverage optimization targets look like at the
streaming operating point vs. the offline operating point.

### 1e. Our own configuration comparison

For completeness, on Wan2.1-VACE-1.3B:

| Config | Inference (s) | Ratio to offline baseline |
|---|---:|---:|
| 33 frames, 30 steps (1a) | 53.30 | 1.00× |
|  4 frames,  4 steps (1c) |  2.585 | 20.6× faster |

This is a fact about our own two configurations on our own hardware
with the same model. It is **not** a claim about what fraction of any
paper's reported speedup is attributable to hyperparameter choice, and
we are not making any such claim here.

## 2. What DreamZero reports (arXiv 2602.15922, NVIDIA)

Quotations and numbers from §3.2 and Table 1 of the paper.

**Baseline:** *"A naive implementation of DreamZero on a single GPU
requires approximately 5.7 seconds per action chunk due to three
bottlenecks: (1) iterative denoising across 16 diffusion steps required
for smooth actions, (2) the computational cost of a 14B parameter DiT
backbone, and (3) sequential execution that blocks robot motion during
inference."* The hardware for the 5.7 s baseline is not specified as
H100 or GB200 in the text; Table 1 rows are labelled H100 and GB200
columns separately.

**Chunk definition (from Appendix):** K = 2 **latent** video frames per
chunk; M = 4 past chunks kept in the rolling context; max context is
8 latent frames = 33 raw frames ≈ 6 seconds at the native frame rate;
action horizon H = 48 steps at 30 Hz (1.6 seconds of real robot motion
per chunk) for bimanual manipulation.

**Target latency:** *"we target inference latency below approximately
200 ms to ensure sufficient overlap for smooth, reactive control"* —
enabled by asynchronous execution that decouples inference from action
execution.

**Step-count progression:**
- **Naive:** 16 denoising steps per chunk.
- **DiT Caching:** exploits directional consistency of flow-matching
  velocities — *"when cosine similarity between successive velocities
  exceeds a threshold, we reuse cached velocities, reducing effective
  DiT steps from 16 to 4 with minimal quality loss on action
  prediction."* Note that this is **velocity reuse within a chunk's
  denoising loop**, not a rolling KV cache across chunks.
- **DreamZero-Flash:** a **training-time** change. Decoupled noise
  schedules: during training, `t_video ~ 1 − Beta(7, 1)` (biased toward
  high noise) while `t_action` stays uniform. This teaches the model to
  predict clean actions from still-noisy video, closing a train-test
  mismatch that made naive step reduction degrade quality.
  *"As a result, we reduce the diffusion steps from four to one,
  cutting inference from ~350 ms to ~150 ms."*

**Table 3 (§5) — empirical cost of the step reduction on the
table-bussing task:**

| Method | Denoising steps | Task progress | Inference | Speed-up |
|---|---:|---:|---:|---:|
| DreamZero | 4 | 83% ± 6.1% | 350 ms | 1× |
| DreamZero | 1 | 52% ± 10.2% | 150 ms | 2.33× |
| DreamZero-Flash | 1 | 74% ± 10.1% | 150 ms | 2.33× |

So naive 1-step inference loses 31 points of task progress (83 → 52);
Flash 1-step loses only 9 points (83 → 74) at the same latency.

**Table 1 — cumulative inference speed-ups, reproduced faithfully:**

| Optimization | H100 | GB200 |
|---|---:|---:|
| Baseline | 1× | 1.1× |
| + CFG Parallelism | 1.9× | 1.8× |
| + DiT Caching | 5.5× | 5.4× |
| + Torch Compile + CUDA Graphs | 8.9× | 10.9× |
| + Kernel & Scheduler Opts. | 9.6× | 14.8× |
| + Quantization (NVFP4) | — | 16.6× |
| + DreamZero-Flash | — | 38× |

Critical caveats that didn't make it into the earlier draft of this
file:

1. **The 38× headline is GB200-only.** It requires NVFP4 weights +
   activations, which is a Blackwell-generation feature. The
   cumulative speed-up on H100 caps at **9.6×** because the last two
   rows of Table 1 are dashed for H100.
2. **CFG Parallelism is multi-GPU from row 2 onwards.** *"We distribute
   these [conditional and unconditional CFG forward passes] across two
   GPUs, reducing per-step latency by 47%."* So DreamZero's post-baseline
   serving configuration is never single-GPU.
3. **DreamZero's "DiT Caching" is not a rolling KV cache.** It is
   **velocity reuse between successive denoising steps within a single
   chunk**, on the basis of cosine similarity between velocities. It
   acts like an aggressive step-reduction trick (16 → 4 effective
   steps), not a prefix-caching trick across chunks. The rolling KV
   cache across chunks is a separate thing that the paper mentions in
   §3.1 *"leveraging KV caching for efficiency"* but does not break
   out with its own row in Table 1.
4. **DreamZero-Flash requires a training-time change.** It is not a
   serving-layer optimization that can be applied to an existing
   checkpoint.

## 3. What StreamDiffusionV2 reports (arXiv 2511.07399)

- **14B model, 4× H100, no TensorRT, no quantization: 58.28 FPS**, 0.5 s
  TTFF, supports 1–4 denoising steps per chunk.
- **1.3B model, 4× H100, same config: 64.52 FPS.**
- Core techniques: SLO-aware batching scheduler (`B × T' × H × W` with
  small `T'`), **sink-token–guided rolling KV cache across chunks**,
  pipeline orchestration across denoising steps AND network stages,
  motion-aware noise controller, adaptive sink-token refresh, stream-VAE.
- Reference backbones are Wan-family causal DiTs (CausVid,
  Self-Forcing — both distilled from Wan-2.1-T2V).
- **Training-free.** Applies to existing off-the-shelf causal video DiTs.

StreamDiffusionV2's rolling KV cache is the architectural pattern that
most closely matches what RFC #1987's infrastructure is targeting.
Their code, if released, is the most direct reference for a vllm-omni
implementation.

## 4. Stage-time mix implications for contribution targeting

Combining the findings from 1d with the definitions from §2 and §3, the
following observations are specific to what we measured and do not
depend on cross-baseline extrapolation:

1. **At the offline operating point on 1× A100**, DiT denoising is
   ~89% of inference time on both models we tested. Anything that
   reduces DiT denoising cost (velocity reuse like DreamZero's DiT
   Caching, attention kernel improvements, rolling KV cache across
   chunks if the workload is chunked, quantization of DiT weights, step
   reduction via distillation or noise-schedule tricks like
   DreamZero-Flash) is directly on the critical path.
2. **At a streaming-ish operating point on 1× A100** (4 frames, 4
   steps on Wan-1.3B), DiT fraction drops to ~60%. VAE encode+decode
   accounts for ~26% (`0.38 + 0.30 = 0.68 s` of 2.585 s total).
   Framework overhead accounts for ~10% (~260 ms).
3. **Framework overhead is large in absolute terms.** ~260 ms per call
   is ≥100% of a 150 ms chunk budget. Any serving configuration that
   targets DreamZero/SDv2-style latency numbers will need to amortize
   or eliminate this, most plausibly via step-wise scheduling (RFC
   #2073) so overhead is paid once per pipeline fill rather than once
   per call.
4. **"DiT caching" in DreamZero's Table 1 and "rolling KV cache across
   chunks" in RFC #1987 / StreamDiffusionV2 are distinct
   optimizations** and should not be conflated in our internal
   recommendation or in any external argument we make.

## 5. Contribution targets (revised)

In priority order by a rough qualitative assessment of leverage and
feasibility on a 1× A100 + H100-class non-Blackwell deployment, with
attribution kept neutral:

1. **Fix the `stage_durations` plumbing bug.** Small, local, zero
   risk. The `DiffusionPipelineProfiler` already populates
   `_stage_durations` on the pipeline; the engine does not forward it
   to `VideoGenerationArtifacts.stage_durations` in the response. A
   two-file patch. This is the tool every future profiling
   contribution needs, and it is the cheapest way to become a
   contributor to the project.
2. **Framework / per-call overhead reduction.** Our ~260 ms
   measurement is the most concrete motivating number we have for
   arguing on RFC #2073 (step-wise engine refactor). Worth profiling
   from both client and server sides to confirm the breakdown before
   committing to an implementation, then contributing a focused piece
   of the refactor if #2073 is not already blocked on design.
3. **Rolling KV cache + blockwise-causal attention for Wan /
   HunyuanVideo DiTs** — the core of RFC #1987. This is distinct from
   DreamZero's DiT Caching (which is intra-chunk velocity reuse) and
   more directly matches StreamDiffusionV2's cross-chunk rolling KV
   cache. We do **not** have a measured benefit number for this on our
   hardware, and the numbers from the reference papers cannot be
   ported across hardware and training regimes without careful
   caveats. If we want a defensible "this much speed-up is on the
   table" argument for the RFC, we need to measure it ourselves on a
   causal Wan variant first. Until then, this is a research
   commitment, not a quantified target.

Optional follow-up work that would sharpen target #3: reproduce
StreamDiffusionV2's rolling-KV-cache recipe (sink tokens + sliding
window) on a causal Wan checkpoint via `diffusers` directly, and
measure the speed-up vs. a naive chunked baseline on our A100. That
gives us a single-number argument on our own hardware, independent of
Table 1's hardware assumptions.
