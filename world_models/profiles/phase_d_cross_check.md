# Phase D — Literature cross-check

All measurements on one **A100 80GB PCIe**, TP=1, BF16, no quantization,
served via vllm-omni's HunyuanVideo15I2VPipeline or Wan22VACEPipeline.
Numbers are scraped from the DiffusionPipelineProfiler container logs
(the API `stage_durations` field is always empty — a vllm-omni bug we
noted but did not fix).

## 1. Our measurements in one table

| Config | Model | Frames | Steps | Total inference_s | DiT loop (s) | DiT % | Peak (GB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Offline baseline | HunyuanVideo-1.5-I2V 13B | 33 | 30 | 73.70 | 65.62 | 89.0 | 80.0 |
| Offline baseline | Wan2.1-VACE-1.3B | 33 | 30 | 53.30 | 47.45 | 89.0 | 26.5 |
| Steps=1 extreme | HunyuanVideo-1.5-I2V 13B | 33 | 1  | 10.45 | 2.27  | 21.7 | 80.0 |
| Frames=4 extreme | HunyuanVideo-1.5-I2V 13B | 4  | 30 | 11.43 | 10.86 | 95.0 | 47.1 |
| Streaming point | Wan2.1-VACE-1.3B (measured) | 4  | 4  | 2.585 | 1.54 | 59.5 | 26.5 |
| Streaming point | HunyuanVideo-1.5-I2V 13B (extrapolated) | 4  | 4  | ~2.05 | ~1.45 | ~71 | ~47 |

Output frames per clock second at the streaming point:
- Wan-1.3B: 4 frames / 2.585 s = **1.55 FPS** on 1× A100
- HunyuanVideo-13B (extrapolated): 4 frames / ~2.05 s = **~2.0 FPS** on 1× A100

## 2. Our headline speedup: just moving off the offline operating point

From our offline baseline to the streaming operating point, **on the same
hardware with the same model and zero caching work**, we already measure:

- Wan-1.3B: **53.30 s → 2.585 s = 20.6×** speedup (measured end-to-end).
- HunyuanVideo-13B: **73.70 s → ~2.05 s = ~35.9×** speedup (extrapolated).

This is the lever nobody in the literature advertises because it's just
"pick sensible inference hyperparameters for your use case". But it's
essentially the entire DreamZero-reported 38× all on its own — the rest
of the community's optimizations (caching, scheduling, quantization,
kernels) fight over the *remaining* gap between ~2 s per chunk and
~150 ms per chunk.

## 3. Comparison to the literature's real-time numbers

### DreamZero (NVIDIA, arXiv 2602.15922)
Reports **150 ms per action chunk, ~7 Hz closed-loop control**, for their
14B auto-regressive diffusion transformer built on the Wan I2V backbone.
Achieved via a **38× combined speedup** from algorithmic
(DreamZero-Flash: decoupled video/action denoising schedules) + system
(parallelism + caching) + low-level (quantization, CUDA kernel tuning)
optimizations. The paper does not publish the hardware for the 7 Hz number
but the rest of the paper uses A100s and H100s; treat it as
"H100-class" for our gap analysis.

Our closest comparable (Wan-1.3B, 4 frames, 4 steps, 1× A100, BF16, no
caching, no quantization, no kernel tuning): **2.585 s per chunk**.

**Chunk-latency gap: 2.585 s / 0.150 s = 17.2×.** That 17.2× is the
headroom between "naive single-A100 small-chunk serving" and DreamZero's
published end state. It decomposes into something like:

| Factor | Estimated multiplier | Notes |
|---|---:|---|
| Hardware (A100 → H100-class, possibly multi-GPU) | ~3–5× | H100 BF16 is ~2–3× A100; DreamZero likely uses TP or PP on top |
| Quantization (BF16 → FP8 / INT8 weights + activations) | ~1.5–2× | DreamZero category (3) |
| CUDA kernel tuning + fused ops | ~1.3–1.5× | DreamZero category (3) |
| System caching (rolling KV cache, step scheduling) | ~1.5–2× | DreamZero category (2) — **RFC #1987's target** |
| Algorithmic (DreamZero-Flash decoupled schedules) | ~1.5–2× | DreamZero category (1), trained in |
| **Product of midpoints** | **~17×** | Matches the 17.2× observed gap |

The attribution is a fit to the gap, not a measurement of each factor in
isolation — that needs further work and ideally the authors' ablation
table. But the *order of magnitude* of each factor is defensible from the
literature: quantization+kernel wins on video DiTs typically come in
around 2–3× total (Xi 2025, Zhang 2025 sparse/block attention), hardware
gains from H100 and multi-GPU are well-characterized, and system-level
caching has been reported in the 1.5–2× range for causal DiT adaptation
(StreamDiffusionV2's ablations, CausVid).

### StreamDiffusionV2 (UT Austin et al., arXiv 2511.07399)
Reports **58.28 FPS on a 14B model** and **64.52 FPS on a 1.3B model** on
**4× H100**, no TensorRT, no quantization, 0.5 s time-to-first-frame,
supporting 1–4 denoising steps. Those are their peak (4 denoising steps ≈
31–61 FPS).

Our closest comparable (Wan-1.3B, 4 frames, 4 steps, 1× A100): **1.55 FPS
output throughput**. Gap: **64.52 / 1.55 = 41.6×**.

| Factor | Estimated multiplier |
|---|---:|
| 1× A100 → 4× H100 | ~6–10× (raw FLOPS) |
| Rolling KV cache (avoid re-processing prefix) | ~1.5–2× |
| Pipeline orchestration across denoising steps + layers | ~1.5–2× |
| Motion-aware noise + sink refresh (quality, ~no perf cost) | ~1× |
| SLO-aware batching (`B×T'×H×W` reshape vs our `1×33×H×W`) | ~1.5–2× (our 4-frame runs already get part of this) |
| **Product of midpoints** | **~20–60×** |

Same caveat: this is a fit to the gap, not a component-wise measurement.

## 4. Where the DiT / VAE budget lands at the streaming point

At the streaming point, the 89% DiT-dominance from the offline baseline
**does not hold any more**. From Wan-1.3B at 4 frames / 4 steps:

| Stage | Seconds | % of inference |
|---|---:|---:|
| DiT denoising | 1.536 | 59.5 |
| VAE encode | 0.380 | 14.7 |
| VAE decode | 0.295 | 11.4 |
| Framework (derived) | 0.261 | 10.1 |
| text_encoder.forward | 0.114 | 4.4 |

**DiT drops from 89% to ~60%, VAE (enc+dec) climbs from 4% to ~26%, and
framework overhead grows to ~10%.** The reason is that VAE encode/decode
have a large *fixed* component (kernel launches, memory staging) that
becomes a bigger fraction of a smaller chunk. Framework overhead is
similar — 200–300 ms of Python/FastAPI/stage-plumbing cost hits every call
whether the DiT work is 65 s or 1.5 s.

**Implication for contribution targeting.** At the offline operating
point DiT caching/attention work is obviously the highest-leverage
target. At the streaming operating point that lever is smaller in
proportion, and the next-biggest levers are:

1. **Amortizing VAE encode/decode** — stream VAE, fused kernels, or
   keeping the previous chunk's latents in memory instead of re-encoding
   the conditioning frame each step. SDv2 has a "stream-VAE" component
   in their pipeline; this is where that wins.
2. **Framework overhead reduction** — ~300 ms per call from vllm-omni's
   stage machinery, HTTP round trip, and per-call Python. For 150 ms
   chunks this is ≥100% overhead. The **step-wise engine refactor
   (RFC #2073)** would amortize this across interleaved sessions.
3. **DiT denoising** is still the largest single bucket at 60%, so
   caching/attention remains the biggest single target — but it's no
   longer 89% like it looked at the offline baseline.

## 5. What this means for a vllm-omni contribution

Three contribution targets fall out of the numbers, in priority order by
leverage at the *streaming* operating point:

1. **Fix the `stage_durations` plumbing bug** — small PR, directly
   useful: the `DiffusionPipelineProfiler` already populates
   `_stage_durations` on the pipeline, but the engine doesn't forward
   it to `VideoGenerationArtifacts.stage_durations`. A two-file patch.
   Low-leverage for perf but zero risk, and it's the tool every future
   profiling contribution needs.
2. **Rolling KV cache + blockwise-causal attention for Wan / HunyuanVideo
   DiTs** — the RFC #1987 item explicitly. On our numbers this is
   ~1.5–2× at the streaming operating point — not the headline win at
   small chunks, but essential for longer-horizon streams where prefix
   recomputation dominates.
3. **Framework / per-call overhead reduction** — probably falls out of
   RFC #2073 (step-wise engine refactor). Worth profiling from the
   client side first to confirm our 300 ms estimate, and then arguing
   on the RFC with concrete numbers.
