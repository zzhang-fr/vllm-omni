# Related Works

A reference-and-read list for the world-models project. This is a
pointer file, not a digested summary — use it as a to-read queue and a
"have we checked this?" checklist. Everything here is sorted by the
structure it takes in the **ACM CSUR survey** we're using as ground
truth; works *also* found via RFC #1987 and follow-up web searches are
merged in and labelled.

## Primary source: ACM CSUR world-models survey

**Ding et al., *"Understanding World or Predicting Future? A
Comprehensive Survey of World Models"*, ACM Computing Surveys, 2025.**

- Paper: https://arxiv.org/abs/2411.14499
- Companion GitHub list: https://github.com/tsinghua-fib-lab/World-Model

The survey's top-level taxonomy is two-axis. Axis 1 is the **essential
function** of a world model:

1. **Implicit representation of the external world.** Learn internal
   latent variables that *understand* the world's mechanisms, typically
   to support decision-making. Covers model-based RL, LLM-as-world-model,
   JEPA-style representation-only prediction, and "world knowledge
   learned by models".
2. **Future prediction of the physical world.** Generate or simulate
   future states pixel-by-pixel (video world models) or scene-by-scene
   (embodied environments), typically as a forward simulator for
   agents, data generation, or policy evaluation.

Axis 2 is the **application domain**: Game, Embodied, Urban (driving +
logistics + urban analytics), Societal. For this project we care about
Embodied and Game most of all, and we are profiling the video-prediction
side of Axis 1. We are not touching Societal at all.

Sections of the survey and how much of each matters to us:
- §2 Background and categorization — read in full. Core to our framing.
- §3 Implicit representation — read §3.1 (model-based RL, LLM) and §3.2
  (world knowledge). Relevant for *alternative* project directions
  (learned dynamics vs. our current pretrained-video-DiT approach).
- §4.1 World model as video generation — **most relevant section for
  this project**. Covers long-term / multimodal / interactive /
  consistency / diverse-environment video world models.
- §4.2 World model as embodied environment — classifies indoor /
  outdoor / dynamic simulators. Useful context.
- §5.1 Game Intelligence — **directly relevant**. Interactivity,
  consistency, generalization in game-like domains.
- §5.2 Embodied Intelligence — relevant; maps onto the RFC #1987 P0
  robotics target.
- §5.3 Urban Intelligence — driving world models; adjacent but
  technically uses the same video-DiT infrastructure.
- §5.4 Societal Intelligence — skip for now.
- §6.5 Simulation Efficiency — **directly relevant** to our profiling
  and contribution angle; the section is short but it calls out that
  "the autoregressive nature [of transformers] can only generate one
  token at a time" as a key challenge, and discusses small+big model
  ensembles, distillation, and scheduler platforms as acceleration
  strategies.

## Second source: RFC #1987 "World Model Support" (vllm-project/vllm-omni)

URL: https://github.com/vllm-project/vllm-omni/issues/1987

RFC #1987 partitions the world-model workload into three scenarios:
**robotics**, **interactive video/game**, and **simulation**. Short-
term focus is robotics (DreamZero + OpenPI-style realtime server).
Interactive video generation and KV-cache / PageAttention for
autoregressive diffusion are explicitly deferred (P2). The RFC doesn't
cite the ACM survey, but the scenarios map roughly onto the survey's
§5.2 (Embodied) and §5.1 (Game) application domains.

## Annotation conventions

Each entry is tagged with one or more of:

- *Survey*: cited by the ACM CSUR survey (section noted if useful).
- *RFC #1987*: cited by the vllm-omni RFC or its comments.
- *Web*: found via web search while updating this file (post-survey,
  or not in the survey's coverage window).
- **SOTA**: a work I consider field-defining for its sub-area at the
  time of writing. This is a judgement call, not a claim.
- *Adj*: adjacent / secondary interest.
- *Hist*: historically important, not frontier.

---

# Part A — Implicit representation (understanding)

## A.1 Model-based RL / learned latent dynamics

- **Recurrent World Model (Ha & Schmidhuber, 2018)** — the 2018 paper
  that popularized the term "world model" in DL; RNN-based latent
  simulator. *Survey §2.1, Hist.*
- **PlaNet** (2019) — latent-space planning on pixel inputs. *Survey, Hist.*
- **Dreamer V1 / V2 / V3** (Hafner et al., 2020–2025) — the canonical
  latent-dynamics family. **DreamerV3** is published in Nature, 2025
  (https://www.nature.com/articles/s41586-025-08744-2) and masters
  150+ control tasks with a single configuration. *Survey, Web,
  **SOTA for RL / latent dynamics**.*
- **TD-MPC1 / TD-MPC2** (Hansen et al.) — model-predictive control on
  learned latent dynamics; frequently cited alongside Dreamer. *Survey.*
- **DayDreamer** (Wu et al., 2023) — Dreamer applied directly to real
  robots. *Survey §5.2, Adj.*
- **MuDreamer** (2024) — predictive world model without pixel
  reconstruction. *Survey, Adj.*
- **PWM** — Policy World Model; referenced in the survey roadmap. *Survey.*
- **DIAMOND** (Alonso et al., NeurIPS 2024) —
  https://arxiv.org/abs/2405.12399 — first successful application of
  diffusion as a world model for RL; SOTA on Atari 100k at time of
  publication; 3-denoising-step pixel-space predictions via EDM
  framework. *Survey, Web, **SOTA for diffusion-as-world-model in RL**.*

## A.2 LLM-based world models

- **LLM-DM**, **WorldGPT**, **Pandora**, **3D-VLA**, **iVideoGPT**,
  **COKE**, **SimToM** — the survey's §3.1.2 roster. *Survey; not
  central to our project.*
- **Pandora** (Xiang et al., 2024) — world-state simulation with
  free-text actions; listed in the survey's video-world-model table as
  well. *Survey §4.1, §4.2.3.*

## A.3 JEPA / representation-only predictive world models

- **JEPA** (LeCun, 2022 position paper) — joint-embedding predictive
  architecture; predicts abstract representations rather than pixels.
  *Survey §2, Hist.*
- **V-JEPA** (Meta, 2024) — first video version of JEPA. *Survey.*
- **V-JEPA 2** (Meta, June 2025) —
  https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/ and
  https://arxiv.org/abs/2506.09985 — 1.2B parameter open-source
  self-supervised video world model trained on 1M+ hours of video +
  1M images; does zero-shot robot planning via Model Predictive
  Control on its learned latent dynamics. *Survey §3, §5.2; Web;
  **SOTA for representation-only world models**.*
- **DINO-WM** — world model built on DINO visual features; referenced
  in survey roadmap. *Survey, Adj.*

## A.4 World knowledge evaluation / probing

Survey §3.2 covers whether LLMs and multimodal models *have* world
knowledge (global physical, local physical, human society). Not
directly relevant to our project but worth knowing exists if we ever
need to argue "has model X learned physics?"

---

# Part B — Future prediction (generation and simulation)

This is the half of the survey where our project actually lives.

## B.1 Video world models (survey §4.1)

The survey categorizes video world models along five capability axes:
**long-term**, **multimodal**, **interactive**, **consistency**,
**diverse environments**. Table 2 of the survey enumerates the entries
below (among others). Entries marked **SOTA** are the ones I'd point
someone at as the current frontier in their sub-category.

### Long-term video generation
- **NUWA-XL** (Yin et al.) — "Diffusion over Diffusion" for long video. *Survey.*
- **LWM** (Large World Model, Liu et al.) — transformer on long video+text sequences. *Survey.*
- **StreamingT2V** (Henschel et al.) — autoregressive T2V with long/short-term memory. *Survey.*
- **GAIA-1** (Wayve) — driving-specific; see B.1 Urban section. *Survey.*

### Multimodal / action-conditioned
- **3D-VLA** (Zhen et al.) — integrates 3D perception, reasoning, and action. *Survey.*
- **Pandora** (Xiang et al.) — world-state simulation with free-text actions. *Survey.*
- **Genie 1** (Bruce et al., 2024) — unsupervised generative action-controllable environments. *Survey, Hist.*

### Interactive / controllable
- **UniSim** (Yang et al.) — simulates real-world interactions for
  vision-language-and-RL training. *Survey §4.2.3, Adj.*
- **iVideoGPT** — interactive world model via tokens + rewards. *Survey.*
- **PEEKABOO** — spatiotemporal control without extra training. *Survey.*
- **Aether** (2025) — camera trajectories as geometry-aware actions. *Survey.*
- **Yume** — continuous-keyboard-input streaming interactive world. *Survey.*
- **NWM** (Navigation World Model) — controllable video generation on past observations + nav actions. *Survey, Adj.*
- **VideoDecision** — extends video models to planning and RL. *Survey.*

### Consistency
- **WorldGPT**, **DiffDreamer**, **ConsistI2V**, **WorldMem** — the
  survey's §4.1 table. WorldMem integrates a memory mechanism for
  long-term consistency. *Survey.*

### Diverse environments
- **WorldDreamer** — broad-scope video world model across scenarios. *Survey.*
- **Genie** (v1) — unsupervised generative action-controllable virtual environments. *Survey.*

### Frontier / general-purpose video world models
- **Sora** (OpenAI, 2024) — minute-long coherent clips. Closed;
  frequently cited as the spark for the modern "world model" discourse.
  *Survey, Hist-but-also-current-closed-frontier.*
- **Wan 2.1 / 2.2** (Alibaba Tongyi) — open DiT video family; the
  backbone for DreamZero, CausVid, Self-Forcing, and most of the
  streaming-DiT literature. *Survey (Wan entry), Web, **SOTA open
  video DiT backbone**.*
- **HunyuanVideo 1.5** (Tencent) — 13B dense DiT video model; the one
  we profiled in Phase A/B. *Web.*
- **CogVideoX** (Zhipu / THU) — open video DiT with I2V variants. *Web.*
- **MovieGen** (Meta) — frontier closed video model. *Web, Hist-closed.*
- **Cosmos** — see B.3 Simulation / foundation section below.

## B.2 Streaming / fast / efficient video diffusion (survey §6.5 +
## external web coverage)

This is the sub-area most directly relevant to our profiling and
contribution angle. The ACM survey touches it only briefly in §6.5
"Simulation Efficiency", which is why most of the entries here come
from web search and the DreamZero / StreamDiffusionV2 papers.

- **CausVid** (Yin et al., CVPR 2025) —
  https://causvid.github.io/ and https://arxiv.org/abs/2412.07772 —
  distills a bidirectional Wan-2.1-T2V into a causal DiT; ~9.4 FPS
  streaming after ~1.3 s initial latency. **SOTA reference for
  bidirectional → causal distillation.** *Web.*
- **Self-Forcing** (Huang et al., NeurIPS 2025 Spotlight, Adobe Research) —
  https://self-forcing.github.io/ and https://arxiv.org/abs/2506.08009 —
  closes the train-test exposure-bias gap for AR video diffusion;
  17 FPS on 1× H100 with sub-second latency on a 1.3B Wan-family
  model. **SOTA for training-time AR distillation as of late 2025.**
  *Web.*
- **Causal Forcing** (THU-ML, 2025–26) —
  https://github.com/thu-ml/Causal-Forcing — refinement of CausVid-style
  distillation for real-time interactive video. *Web, Adj.*
- **MAGI-1** (Sand AI, May 2025) —
  https://arxiv.org/abs/2505.13211 — autoregressive generation in
  24-frame chunks with monotonically-increasing per-chunk noise,
  concurrent 4-chunk pipelining; first chunk ~2.3 s, subsequent <1 s
  each. **SOTA for chunk-wise AR video generation.** Compute pattern
  is very close to what we're profiling. *Web.*
- **StreamDiffusionV2** (Feng et al., arXiv 2511.07399, UT Austin /
  Berkeley / Stanford / MIT, Nov 2025) —
  https://streamdiffusionv2.github.io/ — training-free serving system;
  sink-token–guided rolling KV cache, SLO-aware batching, pipeline
  orchestration across denoising steps + network stages. 58.28 FPS on
  14B / 64.52 FPS on 1.3B, on 4× H100, no TensorRT, no quantization.
  **SOTA for the serving-system angle; most direct reference for RFC
  #1987's infrastructure plan.** *Web.*
- **MirageLSD** (Decart, 2025) —
  https://decart.ai/publications/mirage — advertised as "the first
  live-stream diffusion AI video model"; closer to video restyling
  than true world modelling but indexed here. *Web, Adj.*

Efficiency techniques that show up in this sub-area but are not
themselves world models:
- **DiT Caching** (velocity reuse within a denoising loop, as used in
  DreamZero §3.2.3 — reduces effective steps from 16 to 4 on AR video
  DiTs). *DreamZero paper.*
- **DreamZero-Flash** (decoupled video/action noise schedules; trains
  the model so single-step inference from noisy visual context still
  yields clean actions; reduces effective steps from 4 to 1). *DreamZero
  paper.*
- **Rolling KV cache + sink tokens** (StreamingLLM-style attention sink
  applied to video DiT KV caches across chunks). *StreamDiffusionV2.*
- **Sparse / block-sparse attention** (Xi et al., Zhang et al. —
  multiple 2025 works on spatio-temporal sparse attention for video
  DiTs). *Referenced indirectly via SDv2 related work.*

## B.3 World foundation models for physical AI / simulation

The survey groups these under §5.2 (Embodied) and §5.3 (Urban); the
Cosmos family is a cross-cutting platform that serves both.

- **NVIDIA Cosmos** platform — https://www.nvidia.com/en-us/ai/cosmos/ —
  the reference family of "world foundation models" for Physical AI.
  Evolved across multiple versions; worth tracking as a platform not
  a single model. *Survey (as "Cosmos" in the roadmap), Web, **SOTA for
  world foundation models**.*
  - **Cosmos-Predict 1** (early 2025) — initial release of predictive
    video WFMs.
  - **Cosmos-Predict 2** (mid 2025) —
    https://github.com/nvidia-cosmos/cosmos-predict2 — next generation.
  - **Cosmos-Predict 2.5** (Oct 2025) —
    https://github.com/nvidia-cosmos/cosmos-predict2.5 and
    https://huggingface.co/blog/nvidia/cosmos-predict-2 — unifies
    Text2World, Image2World, and Video2World in one flow-based model;
    uses **Cosmos-Reason1** as its text encoder (a Physical AI
    reasoning VLM, not a generic text encoder); 2B and 14B scales;
    trained on 200M curated video clips; RL post-training.
    **Current frontier for open world foundation models.**
  - **Cosmos-Transfer 2.5** (Oct 2025) — paired release with Predict
    2.5 for transferring learned world knowledge.
  - **Cosmos-Policy** (early 2026) —
    https://www.therobotreport.com/nvidia-adds-cosmos-policy-world-foundation-models/
    — policy models for RoboCasa and Libero environments.
  - **Cosmos-Predict 2.5 /Robot/Action-Cond Distillation** (Feb 2026)
    — action-conditioned distillation path for robotics.
  - **Cosmos-Reason1** — the Physical AI reasoning VLM used as text
    encoder in Predict 2.5.
- **World Simulation with Video Foundation Models for Physical AI**
  (arXiv 2511.00062) — the Nov 2025 paper describing the Cosmos
  Predict 2.5 architecture. *Web.*

## B.4 Robotics world-action models (RFC #1987 P0)

- **DreamZero** (Seonghyeon Ye et al., NVIDIA, arXiv 2602.15922, Feb
  2026) — https://dreamzero0.github.io/ and
  https://arxiv.org/abs/2602.15922 — 14B autoregressive diffusion
  transformer built on a pretrained Wan I2V backbone, trained with a
  teacher-forcing chunk-wise video denoising objective, jointly
  predicting video and action ("World Action Model"). Target: real-
  time closed-loop control on bimanual manipulators at ~7 Hz.
  **Current SOTA open-architecture world action model for robotics.**
  Landing in vllm-omni via PR #2162. *Web; primary reference for RFC
  #1987's P0 target.*
- **Motus** — stateless robotics world model referenced in RFC #1987;
  no public paper found. *RFC.*
- **LingBot-VA / LingBot-World** — video-DiT-based robotics world
  model in vllm-omni issues #1045 and #874 (draft I2V pipeline PR).
  Closest in-tree code to streaming-control pipelines today. *RFC.*
- **DreamGen** (Jang et al., 2025) — generates synthetic robot data
  ("neural trajectories") to boost VLA training. *Survey §5.2.*
- **Roboscape** (2025) — integrates physical laws during video
  generation; improves VLA training via more physically plausible
  synthetic data. *Survey §5.2.*
- **EVAC** (2025) — expands training data with diverse failure
  trajectories; action-conditioned multi-view generation. *Survey §5.2.*
- **Genie Envisioner (GE-Sim)** (2025) — closed-loop policy ↔ world
  model interaction for scalable simulation. *Survey §5.2.*
- **Vidar** (2025) — two-stage diffusion-based video pre-training +
  masked inverse dynamics for robotic action prediction. *Survey §5.2.*
- **VPP** (Video Pre-training for Policy) — visual-latent-to-motor-
  command translation. *Survey §5.2.*
- **IRASim** (2024) — trajectory-to-video generation for robot policy
  evaluation. *Survey §5.2.*
- **UniPi**, **VIPER**, **GR-1 / GR-2** — earlier (2024) video-as-policy
  works cited in survey Table 4. *Survey §5.2.*

## B.5 VLA baselines (not world models but the comparisons DreamZero makes)

- **π0 / π0.5** (Physical Intelligence) — frontier VLA baseline. *Survey.*
- **OpenVLA** — https://openvla.github.io/ — 7B open academic VLA. *RFC.*
- **GR00T N1.6** (NVIDIA) — NVIDIA's VLA baseline; DreamZero compares
  against this. *Web.*
- **Gemini Robotics** (DeepMind) — frontier closed VLA. *Web.*
- **InternVLA-A1** — mixture-of-transformers VLA; vllm-omni issue #1948.
  *RFC.*

## B.6 Interactive video / game world models (survey §5.1 + web)

- **GameNGen** (Valerio et al., Google, Aug 2024) — modified Stable
  Diffusion runs playable Doom at ~20 FPS on a TPU; the "first neural
  game engine" proof of concept. *Survey §5.1, Web, **SOTA for neural
  game engines at time of publication**.*
- **GameGen-X** — specialized control module for game multimodal
  inputs; unifies character and scene control. *Survey §5.1.*
- **Oasis** (Decart / Etched, 2024) — AI-generated Minecraft via
  next-frame prediction. *Web.*
- **Matrix-Game** (Skywork AI, 2024) — 17B model with fine-grained
  keyboard/mouse control for character and camera. *Survey §5.1.*
- **Matrix-Game 2.0** (Aug 2025) — interactive world foundation model
  for real-time long video generation. *Web.*
- **Matrix-Game 3.0** (March 2026) —
  https://github.com/SkyworkAI/Matrix-Game and arXiv 2604.08995 —
  real-time streaming interactive world model with long-horizon memory.
  **Current frontier for open interactive video world models.** *Web.*
- **Hunyuan-GameCraft** (Tencent, arXiv 2506.17201, June 2025) —
  high-dynamic interactive game video generation with hybrid history
  conditioning. *Web.*
- **Hunyuan-GameCraft 2** (Tencent, arXiv 2511.23429, Nov 2025) —
  https://hunyuan-gamecraft-2.github.io/ — instruction-following
  interactive game world model trained on 1M+ AAA gameplay
  recordings. *Web.*
- **HunyuanWorld 1.0** (Tencent) — 360° immersive 3D world generation
  via semantically layered 3D meshes. *Survey §2.2, §5.1.*
- **Genie 1** (DeepMind, 2024) — unsupervised generative
  action-controllable environments. *Survey.*
- **Genie 2** (DeepMind, Dec 2024) —
  https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/
  — generates interactive 3D worlds from a single image + text;
  explicit scene memory so off-screen content renders consistently on
  return. AR diffusion architecture for interactive video. *Survey,
  Web.*
- **Genie 3** (DeepMind, Aug 2025 announce, Jan 2026 public) —
  real-time interactive worlds from text, image, or sketch; 30 000+
  hours of gameplay training. *Survey, Web, **current SOTA for
  real-time interactive world models**.*
- **WHAM** (World and Human Action Model, Microsoft Research) —
  interactive game-world model from Bleeding Edge footage. *Survey §5.1.*
- **MineWorld** — Minecraft world model in the survey roadmap. *Survey.*
- **Matrix-3D** — panoramic-3D wide-coverage explorable world. *Survey.*
- **WonderWorld** — interactive 3D scene generation from a single 2D
  image. *Survey §2.2.*

## B.7 Driving world models (survey §5.3.1 + web)

Completely missing from the earlier version of this file. Worth
having because driving is the sub-area with the clearest SOTA
benchmarks and several works that share infrastructure with our
project.

- **GAIA-1** (Wayve) — foundational driving world model. *Survey, Hist.*
- **GAIA-2** (Wayve, March 2025) —
  https://wayve.ai/thinking/gaia-2/ and
  https://arxiv.org/abs/2503.20523 — multi-view controllable generative
  driving world model; trained on UK/US/Germany fleets; fine-grained
  ego / agent / environment control. *Web.*
- **GAIA-3** (Wayve, Dec 2025) —
  https://wayve.ai/press/wayve-launches-gaia3/ — 15B parameters
  (double GAIA-2), 2× larger video tokenizer. **Current frontier for
  driving world models.** *Web.*
- **Waymo World Model** (Feb 2026) — Waymo's specialization of Genie 3
  for autonomous-driving simulation. *Web.*
- **DriveDreamer** — driving world model. *Survey §5.3.1.*
- **Drive-WM** — driving world model. *Survey §5.3.1.*
- **Think2Drive** — driving world model with planning. *Survey §5.3.1.*
- **OccWorld** — occupancy-grid world model. *Survey §5.3.1.*
- **Gaussianworld** — Gaussian-splat driving world model. *Survey.*
- **Cosmos-Drive** — NVIDIA's driving specialization of the Cosmos
  platform, in ongoing collaboration with Wayve. *Web.*

## B.8 Embodied environments (survey §4.2)

The survey's §4.2 enumerates static and dynamic embodied environments
that serve as world-model training or evaluation substrates. These are
not world models themselves but context:

- **Static indoor:** AI2-THOR, Matterport 3D, VirtualHome, Habitat,
  SAPIEN, iGibson, AVLEN, ProcTHOR, Holodeck, AnyHome, LEGENT, TDW.
- **Static outdoor:** MetaUrban, UrbanWorld, EmbodiedCity, MineDOJO,
  GRUtopia (in+outdoor).
- **Dynamic (generative):** UniSim, Streetscapes, AVID, EVA, Pandora,
  Roboscape, TesserAct, Aether, Deepverse.

All in survey §4.2 Table 3; not repeating URLs here.

## B.9 3D reconstruction as world model (sibling to video generation)

- **VGGT** (Wang et al., CVPR 2025) — feed-forward 3D reconstruction
  from images. Basis of vllm-omni sibling RFC #2727 "Support 3D World
  Model (VGGT)". *RFC #2727, Web, **SOTA for feed-forward 3D from
  images**.*
- **VGGT-Long** — overlapping-chunk + Sim(3) alignment. *RFC #2727.*
- **FastVGGT** — token-partitioning, ~4× speedup. *RFC #2727.*
- **Sparse-VGGT** — block-sparse attention kernel. *RFC #2727.*

## B.10 Societal / social simulacra (survey §5.4)

Out of scope for this project. Survey §5.4 covers LLM-based social
simulation, agent societies, and generative agents. Indexed here so
we know it exists.

- **Generative Agents** (Park et al.) — Sims-style LLM social
  simulation. *Survey §5.4.*
- **AgentSociety** — larger-scale LLM agent simulator. *Survey §5.4.*

---

# Part C — Supporting infrastructure and backbones

Not world models themselves, but what the world models are built from
or served on.

## C.1 Open video DiT backbones

- **Wan 2.1 / 2.2** (Alibaba) — `Wan-AI/Wan2.1-T2V-{1.3B,14B}-Diffusers`,
  `Wan-AI/Wan2.2-I2V-A14B-Diffusers`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers`,
  `Wan-AI/Wan2.1-VACE-{1.3B,14B}-diffusers`. Dense and MoE variants;
  VACE supports multi-frame reference conditioning. **Backbone of
  choice** for DreamZero, CausVid, Self-Forcing, and SDv2's reference
  pipelines. Already in vllm-omni.
- **HunyuanVideo 1.5** (Tencent) — 13B dense DiT. 480p and 720p I2V
  variants. Already in vllm-omni; what we profiled in Phase A/B.
- **CogVideoX** (Zhipu / THU) — 2B and 5B open T2V/I2V. Not currently
  in vllm-omni.
- **LTX-Video / LTX-2** (Lightricks) — fast DiT with multiple variants;
  already in vllm-omni and used by several of the existing benchmarks.
- **FLUX** (Black Forest Labs) — image DiT family; already in vllm-omni.
- **Stable Diffusion 3 / 3.5** — image DiT; already in vllm-omni.

## C.2 Realtime / robotics serving protocols

- **OpenPI** (Physical Intelligence) — realtime robotics API shape;
  referenced as the protocol model for RFC #1987's `/v1/world/stream`
  endpoint. *RFC.*
- **LeRobot** (HuggingFace) — realtime-robotics API and dataset target. *RFC.*
- **RLinf** — RL framework intended to integrate with vllm-omni's
  realtime server. *RFC.*

## C.3 Surveys and curated indexes

- **Ding et al. 2025, "Understanding World or Predicting Future? A
  Comprehensive Survey of World Models"** — https://arxiv.org/abs/2411.14499
  — the primary source for this file's structure. *Primary source.*
- **tsinghua-fib-lab/World-Model** — companion GitHub list for the
  survey above. https://github.com/tsinghua-fib-lab/World-Model
- **LMD0311/Awesome-World-Model** —
  https://github.com/LMD0311/Awesome-World-Model — curated list focused
  on autonomous driving and robotics.
- **leofan90/Awesome-World-Models** —
  https://github.com/leofan90/Awesome-World-Models — general curated
  list across embodied AI, video generation, and driving.
- **HaoranZhuExplorer/World-Models-Autonomous-Driving-Latest-Survey**
  — driving-specific curated list.
- **Voxel51 "Visual AI in Video: 2026 Landscape"** —
  https://voxel51.com/blog/visual-ai-in-video-2026-landscape
- **Xun Huang "Towards Video World Models" blog** —
  https://www.xunhuang.me/blogs/world_model.html

## C.4 Related issues / PRs in vllm-omni (indexed, not digested)

- **#1987** — [RFC] World Model Support. Open. The umbrella RFC.
- **#2162** — DreamZero world model integration + OpenPI WebSocket
  serving. Open PR; the P0 landing.
- **#2727** — [RFC] Support 3D World Model (VGGT). Open.
  Architecturally closest to our interest: proposes a
  `stage_type: "reconstruct"` engine with dLLM-style chunked
  computation + sliding-window KV cache.
- **#2073** — [RFC] Refactor engine/runner/pipeline for step-wise and
  continuous batching. Open. Prerequisite for any AR-chunked diffusion
  contribution.
- **#2590** — [WIP][Perf] AR KV prefix reuse in DiT (Hunyuan-Image-3).
  Open. Early experiment in the direction of rolling KV cache.
- **#2632** — [RFC] Per-Role Attention Backend Configuration for
  Diffusion. Open. Potential host for blockwise-causal attention.
- **#874** — LingBot-World I2V pipeline PR. Draft; closest in-tree code
  to streaming control.
- **#1045** — [New Model] LingBot-World. Open.
- **#1948** — [New Model] InternVLA-A1. Open.
- **#2063 / #2160 / #2078 / #2423** — CFG parallel refactors. Merged;
  prereqs for DreamZero.

## C.5 Serving patterns for world model inference (added 2026-04-16)

Three serving patterns recur across world model papers. They form a
**stack**, not a set of parallel concerns — each layer presupposes
the one below it:

```
Layer 3 — KV-cache reuse          (performance optimization)
    makes Layer 1+2 efficient: carry forward cached computation
    instead of re-encoding overlapping context each step.
    Presupposes: a rollout loop (Layer 1) generating chunks whose
    context overlaps, managed via a window policy (Layer 2).

Layer 2 — Sliding window context  (what the model sees)
    decides which prior frames condition the next generation,
    and evicts/compresses older context.
    Presupposes: a rollout loop (Layer 1) that produces the
    frames to window over.

Layer 1 — Multi-step rollout      (control flow)
    server orchestrates the output→input feedback loop.
    Works without Layers 2 or 3: each step can be stateless
    (re-encode everything from scratch, pass only the last
    frame). Slow but correct — this is what our rollout.py does.
```

Each layer can be implemented independently of the ones above it:
- Layer 1 alone = our current `rollout.py` (stateless, re-encode
  everything, pass one frame). Correct but slow.
- Layer 1+2 = pass the last N frames as raw input each step,
  let the model re-encode them. Richer context, still redundant
  compute.
- Layer 1+2+3 = carry forward KV-cache entries from chunk N to
  chunk N+1, only compute the new tokens. Efficient.

No single paper unifies all three layers into a reusable serving
framework. **This framing is our synthesis, not established
consensus.** For each layer we document what is sourced vs. inferred.

### Layer 1 — Multi-step rollout (server-managed output→input loop)

*What it is:* The server orchestrates a loop where the output of
generation N (e.g., the last frame of a video clip) feeds as input to
generation N+1. The client submits a session with an action sequence;
the server manages the frame feedback internally.

*Where this comes from:* Generalized from our own `rollout.py` (which
does this client-side) and concrete examples in the literature:

- **DreamZero** (arXiv:2602.15922) — implements a WebSocket server
  where the rollout loop is explicitly server-managed. The
  `ARDroidRoboarenaPolicy` wrapper "handles frame accumulation"
  server-side: "first request uses 1 frame, subsequent requests use
  4 frames for temporal context." *Sourced from paper / code.*
- **StreamDiffusionV2** (arXiv:2511.07399, MLSys 2026) — continuous
  streaming pipeline where the server manages frame-by-frame
  generation with SLO-aware scheduling. The rollout loop is implicit
  in the streaming architecture. *Sourced from paper.*
- **MAGI-1** (arXiv:2505.13211) — chunk pipelining where "the next
  chunk begins generation, conditioned on all preceding chunks." But
  framed as within-request parallelism, not multi-user serving.
  *Sourced from paper.*
- **Genie 3** — generates frames autoregressively "by considering the
  complete history of previously generated frames and the user's
  latest actions." Implies server-managed rollout; no paper published,
  only blog. *Inferred from blog description.*

*What is NOT in the literature:* No paper generalizes this as a
reusable serving primitive with a defined API. Every implementation
is either client-side (our `rollout.py`), application-specific
(DreamZero's WebSocket), or proprietary (Genie 3, Oasis).

### Layer 2 — Sliding window context (server-managed frame history)

*What it is:* The server maintains a configurable window of recent
frames (or their KV-cache entries) as conditioning context for the
next generation. Older frames are evicted or compressed. Presupposes
Layer 1 — there must be a rollout loop producing frames to window
over.

*Where this comes from:* Inferred from model requirements — multiple
models need multi-frame context, and a server that supports them needs
to manage it:

- **MAGI-1** — explicitly supports configurable context windows: "by
  setting the KV range to 8 for all chunks, each newly generated chunk
  depends only on the preceding 8 seconds of video content." Closest
  to a serving-configurable parameter. *Sourced from paper.*
- **StreamDiffusionV2** — manages temporal context through sink-token-
  guided rolling KV cache with periodic RoPE refresh. This is
  server-managed context retention. *Sourced from paper.*
- **Deep Forcing** (arXiv:2512.05081) — introduces "Deep Sink" with
  persistent sink tokens and RoPE realignment. Inference optimization,
  not a serving feature. *Sourced from paper.*
- **Wan 2.1-VACE** — conditions on multiple reference frames (first,
  last, arbitrary). The model needs multi-frame input; somebody has to
  manage which frames to pass. *Inferred from model API.*
- **Genie 2** — "explicit scene memory so off-screen content renders
  consistently on return." *Sourced from blog.*
- **Matrix-Game 3.0** (arXiv:2604.08995) — camera-aware memory
  selection that "retrieves only view-relevant historical content based
  on camera pose." Retrieval-based, not fixed window. *Sourced.*

*What is NOT in the literature:* No serving system exposes a
"configure your context window" API for video world models.

### Layer 3 — Cross-chunk KV-cache reuse

*What it is:* When generating chunk N+1, reuse KV-cache entries
computed during chunk N instead of re-encoding the overlapping
context from scratch. This is a performance optimization on top of
Layers 1+2 — it makes the rollout loop and context window efficient
by avoiding redundant computation. Presupposes both lower layers.

*Where this comes from:* This layer has the clearest prior art,
both in video diffusion and by analogy to LLM serving (vLLM's prefix
caching, multi-turn KV-cache reuse):

- **Ca2-VDM** (ICML 2025, arXiv:2411.16375) — most explicit:
  "unidirectional feature computation ensures that the cache of
  conditional frames can be precomputed in previous autoregression
  steps and reused in every subsequent step, eliminating redundant
  computations." *Sourced from paper.*
- **MAGI-1** — "once a chunk has been sufficiently denoised, its
  features can be cached and reused by subsequent denoising chunks
  without the need for recomputation." *Sourced from paper.*
- **StreamDiffusionV2** — uses adaptive context retention through sink
  tokens rather than full KV-cache forwarding. *Sourced from paper.*
- **Chorus** (arXiv:2604.04451, Apr 2026) — pure serving systems
  paper on inter-*request* KV cache reuse for video diffusion: "up to
  45% speedup on industrial 4-step distilled models." Addresses
  cross-request similarity, not cross-chunk reuse within a rollout.
  *Sourced from paper.*
- **FlowCache** (arXiv:2603.27469) — chunkwise caching policies for
  video rollout. Explicitly calls out the gap between algorithmic
  and serving-level optimization. *Sourced from paper.*
- **TempCache** (arXiv:2602.01801) — "compresses the KV cache via
  temporal correspondence to bound cache growth," 5–10× speedup.
  Inference optimization, not serving infrastructure. *Sourced.*
- **LLM analogy** — vLLM's prefix caching and multi-turn KV-cache
  reuse are the direct precedent in the text domain. The video
  analogue is recognized but not yet implemented in any open serving
  framework. *Inferred by analogy.*

### The gap

No paper or system unifies these three layers into a general-purpose
world model serving framework. StreamDiffusionV2 comes closest (it is
the only MLSys-track paper treating streaming video generation as a
systems problem) but is a standalone system, not integrated into a
general model-serving framework like vLLM. This gap is where
vllm-omni could contribute: building these layers incrementally on
top of its existing diffusion engine.

The stack structure maps naturally onto the tier progression:
- **Tier 1 (simulation)** exercises Layer 1 — the rollout loop.
  Latency is not a concern, so Layers 2–3 are optional.
- **Tier 2 (interactive)** requires Layers 1+2 and benefits greatly
  from Layer 3 — real-time latency demands both context management
  and cache reuse.
- **Tier 3 (robotics)** requires all three layers under the tightest
  constraints, plus action-modality prediction on top.

**Provenance note:** The identification of this gap and the stack
framing are our synthesis from surveying the literature — not a claim
made by any cited paper. FlowCache (arXiv:2603.27469) comes closest
to stating the gap explicitly, noting that "nominal low-bit coding
alone does not solve the deployment problem" due to systems-level
inefficiencies.

---

# Part D — Three-tier taxonomy: simulation → interactive → robotics

The literature supports a three-tier ordering of world model workloads
by increasing difficulty. This section collects the evidence and maps
it to concrete models. It was added on 2026-04-16 after a team
discussion about RFC #1987's prioritization.

## D.1 The tiers

| Tier | Synonym | Action regime | Latency constraint | Key challenge |
|------|---------|---------------|--------------------|---------------|
| **Simulation / offline** | Batch, open-loop | All actions known upfront | None (seconds–minutes OK) | Temporal coherence over long horizons |
| **Interactive / online** | Streaming, game | Actions arrive one at a time from a user | Real-time (~20–60 FPS) | Causal architecture + error accumulation under feedback |
| **Robotics / closed-loop** | Embodied, physical AI | Actions arrive from sensors + policy | Hard real-time (7–30 Hz, <150 ms) | Physical precision + safety + joint action prediction |

Each tier is a strict superset of the previous in terms of
architectural requirements:

- **Simulation → Interactive** adds: causal/autoregressive generation
  (cannot use bidirectional attention across future frames), real-time
  latency, and online error accumulation management.
- **Interactive → Robotics** adds: joint action-modality prediction,
  multi-sensor fusion, contact-rich physics accuracy, safety/reliability
  constraints, and cross-embodiment generalization.

## D.2 Literature supporting this ordering

### Explicit taxonomies

- **"Simulating the Visual World with AI: A Roadmap"** (Yue, Huang,
  Chen, Wang, Wan, Liu; arXiv:2511.08585, CVPR 2025 tutorial) —
  proposes four generations: Gen 1 (Faithfulness ≈ simulation),
  Gen 2 (Interactiveness ≈ interactive), Gen 3 (Planning ≈ robotics),
  Gen 4 (Stochasticity ≈ general-purpose physical simulator). The
  authors describe these as "advancing step by step."
  Tutorial page: https://world-model-tutorial.github.io/
- **ACM CSUR survey** (Ding et al., 2025, arXiv:2411.14499) — splits
  by function (representation vs. prediction) and domain (game,
  embodied, urban, societal). The game → embodied progression aligns
  with interactive → robotics.
- **"Video Generation Models as World Models"** (arXiv:2603.28489,
  2026) — distinguishes batch diffusion (offline) from "autoregressive,
  hybrid AR-diffusion, and causal streaming diffusion paradigms [that]
  explicitly target world-model requirements." States that batch
  approaches "do not by themselves resolve the challenges of persistent
  rollout and long-term error accumulation."

### Simulation < Interactive

- **Xun Huang, "Towards Video World Models"**
  (https://www.xunhuang.me/blogs/world_model.html) — argues non-causal
  video generators are fundamentally unable to support interactivity;
  causality is a prerequisite, making interactive strictly harder.
- **GameNGen** (arXiv:2408.14837) — even for a well-defined game
  (DOOM), interactive generation required novel noise augmentation to
  prevent drift — a challenge absent in offline generation.

### Interactive < Robotics

- **"World-in-World"** (arXiv:2510.18135) — closed-loop evaluation
  finds navigation (simpler) improves with world models, but "robotic
  manipulation demonstrates notably weaker gains" because "accurately
  modeling contact-rich interactions and robot kinematics is
  significantly more challenging than predicting viewpoint changes."
- **DreamZero** — requires 7 Hz closed-loop at 150 ms/chunk, joint
  video+action prediction, cross-embodiment transfer, and real-robot
  safety. None required for gaming.

### Simulation/interactive models as evaluation proxies for robotics

- **"Interactive World Simulator for Robot Policy Training and
  Evaluation"** (arXiv:2603.08546, 2026) — policies trained on 100%
  simulator data achieve 87.9% vs. 90.3% on real data (near parity).
- **V-JEPA 2** (Meta, arXiv:2506.09985) — deployed zero-shot on real
  Franka arms after training on unlabeled robot video only.
- **UniSim** (ICLR 2024, arXiv:2310.06114) — trained RL policies
  purely in simulation with demonstrated real-world transfer.
- **1X World Model Challenge** (CVPR 2025) — 100 hours of humanoid
  data for world model evaluation without physical robots.

## D.3 Mapping existing models to tiers

### Tier 1 — Simulation / offline

Models that generate video from a complete specification (prompt,
trajectory, action sequence) without real-time interaction. **Every
non-causal video generation model is a tier-1 model by definition** —
all inputs are known upfront, no real-time constraint.

Already in vllm-omni (3 families, 10+ variants):

| Model | Pipeline | Scale | Mode | Notes |
|-------|----------|-------|------|-------|
| **LTX-2** | `LTX2Pipeline` | ~2B | T2V | Fastest iteration; good for pipeline debugging |
| **LTX-2 I2V** | `LTX2ImageToVideoPipeline` | ~2B | I2V | |
| **LTX-2 two-stage** | `LTX2TwoStagesPipeline` | ~2B | T2V | Two-stage variant |
| **LTX-2 I2V two-stage** | `LTX2ImageToVideoTwoStagesPipeline` | ~2B | I2V | Two-stage variant |
| **Wan 2.1 T2V** | `WanPipeline` | 1.3B / 14B | T2V | |
| **Wan 2.2 T2V** | `WanPipeline` | — | T2V | |
| **Wan 2.2 TI2V** | `WanPipeline` | 5B | T2V | |
| **Wan 2.2 I2V** | `WanImageToVideoPipeline` | ~14B MoE | I2V | |
| **Wan 2.1-VACE** | `WanVACEPipeline` | 1.3B / 14B | T2V, I2V, V2V, R2V, inpainting | Multi-frame conditioning; previews context management tier 2 needs |
| **HunyuanVideo 1.5** | `HunyuanVideo15Pipeline` | 13B | T2V | |
| **HunyuanVideo 1.5 I2V** | `HunyuanVideo15ImageToVideoPipeline` | 13B | I2V | Our Phase A/B baseline |

Not yet in vllm-omni but open-weight:

| Model | Scale | Notes |
|-------|-------|-------|
| **Cosmos-Predict 2.5** | 2B–14B | NVIDIA WFM; open-weight |
| **MAGI-1** | 24B | Chunk-wise AR; streaming-capable but designed for offline quality |

Our 5-step rollout (`rollout.py`) uses HunyuanVideo 1.5 I2V as a
frozen transition function — a tier-1 demonstration.

### Tier 2 — Interactive / online

Models that generate frames causally, conditioned on live user input:

| Model | Scale | FPS | Notes |
|-------|-------|-----|-------|
| **DIAMOND** | ~100M | N/A (RL) | Diffusion world model for Atari; 3-step denoising |
| **GameNGen** | ~1B | 20 | Modified Stable Diffusion; DOOM |
| **Oasis** | ~500M | 20 | DiT; Minecraft |
| **Lucid** | — | 20+ | VAE with 128× compression; Minecraft |
| **MineWorld** | — | 4–7 | Visual-action AR transformer; Minecraft |
| **Matrix-Game 3.0** | 17B | RT | Streaming with long-horizon memory |
| **Genie 2/3** | — | RT | DeepMind; closed |
| **Hunyuan-GameCraft 2** | — | — | Instruction-following; Tencent |
| **GameGen-X** | — | — | Open-world game generation; Tencent |
| **CausVid** | 1.3B–14B | ~9.4 | Causal distillation of Wan; streaming |
| **Self-Forcing** | 1.3B | 17 | AR distillation; NeurIPS 2025 |

### Tier 3 — Robotics / closed-loop

Models that jointly predict observations and actions for real-time
robot control:

| Model | Scale | Hz | Notes |
|-------|-------|----|-------|
| **DreamZero** | 14B | 7 | World Action Model on Wan backbone |
| **V-JEPA 2** | 1.2B | — | Latent dynamics; zero-shot robot transfer |
| **Cosmos-Policy** | — | — | NVIDIA; RoboCasa/Libero |
| **π0 / π0.5** | — | — | VLA baseline (not a world model per se) |

## D.4 Implications for RFC #1987

RFC #1987 puts robotics (tier 3) as P0 and defers simulation (tier 1)
and interactive (tier 2) to P2. This is inverted relative to the
difficulty ordering above. The concern:

1. **Infrastructure built for tier 3 may be over-engineered for tiers
   1–2**, or conversely may skip foundational serving primitives
   (chunked generation, KV-cache rollover, streaming APIs) that tiers
   1–2 would exercise and validate first.
2. **Evaluation requires simulation-tier infrastructure** — without a
   physical robot, DreamZero can only be tested if we already have a
   working simulation/interactive serving path.
3. **The literature's progression is bottom-up** — every interactive
   model was first validated as a good offline generator; every
   robotics model reuses interactive-tier primitives.

A bottom-up roadmap (simulation → interactive → robotics) would let
each tier validate the serving infrastructure the next tier needs.

---

# Part E — What this means for *our* project

Reading the survey and the web results together, here is the honest
positioning of the project relative to the field (written during the
Phase A/B profiling session; see Part D for the tier-based reframing
added 2026-04-16):

1. **Our "toy rollout" experiments (Stage 1/2) are in the B.6 bucket**
   — interactive video / game world models — and the current frontier
   there is Genie 3 / Matrix-Game 3.0 / Hunyuan-GameCraft 2. Our single-
   frame feedback loop failure in Stage 2 is exactly the long-horizon
   drift problem those works exist to solve, typically by some
   combination of explicit scene memory (Genie 2/3), long-horizon KV
   memory (Matrix-Game 3.0), or multi-frame conditioning.
2. **Our profiling work (Phase A/B) targets B.2** — streaming / fast
   video diffusion — and the serving-system frontier there is
   StreamDiffusionV2, with training-time levers in CausVid /
   Self-Forcing / MAGI-1 / DreamZero-Flash as the comparison points.
3. **RFC #1987's P0 target (DreamZero) is in B.4** — robotics world
   action models. The compute pattern is very close to B.2, but with
   additional action-modality injection, closed-loop KV-cache
   overwriting from ground-truth observations, and robotics-specific
   evaluation. The ACM survey groups DreamGen / Roboscape / Genie
   Envisioner / Vidar / V-JEPA 2 into the same §5.2 bucket as
   concurrent robotics world models worth knowing about.
4. **Sub-areas we are deliberately not touching right now**:
   A.1 learned-latent-dynamics RL (DreamerV3, DIAMOND — the "right
   answer" if we ever decide the toy project needs real dynamics
   learning and not just a frozen I2V model), A.3 JEPA (V-JEPA 2 —
   the "right answer" if we ever decide to give up on pixel rollouts
   entirely), B.7 driving (GAIA-2/3, Waymo World Model — adjacent
   but with its own evaluation methodology), B.10 societal (out of
   scope).

The survey's §6.5 "Simulation Efficiency" is explicitly on our side:
it names AR-per-token generation as the headline challenge for big
generative world models, cites distillation and small+big ensembles
as partial solutions, and notes that holistic platform-level
scheduling improvements are needed. That is roughly the intersection
of RFC #1987's infrastructure plan and the StreamDiffusionV2-style
serving-system angle we've been converging on.
