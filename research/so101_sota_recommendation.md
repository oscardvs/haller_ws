# State of the Art for SO-101 Bimanual Manipulation — Architecture Review & Recommendation

**Project:** Haller / Manipulation — generalist multi-task policy for two SO-101 arms
**Hardware:** 2× SO-101 follower arms on a shared frame · center tower RealSense D455 · 1 wrist cam per gripper (3 cameras total)
**Goal:** learn SOTA by collecting data + implementing/training from scratch, aiming for a generalist (multi-task) policy.

---

## 1. Where the field actually is (and why "behaviour cloning" is only half the story)

The Oier Mees–era talks you heard describe the *foundation*: **behaviour cloning (BC)** — supervised learning of action = f(observation) from teleoperated demonstrations. That paradigm is still the substrate of everything below. What changed in 2024–2025 is *how* the policy is parameterized and *what it's conditioned on*:

- **Action chunking** — predict a short *sequence* of future actions per inference step (not one action), which fixes compounding error and jerky control. Introduced with **ACT** (ALOHA, bimanual).
- **Generative action heads** — instead of regressing a single action, model the *distribution* of expert actions with **diffusion** (Diffusion Policy) or **flow matching** (pi0, SmolVLA, X-VLA). Handles multimodal demonstrations.
- **Vision-Language-Action (VLA) models** — bolt the policy onto a pretrained vision-language model so it's conditioned on a **language instruction** and inherits semantic/visual priors. This is what makes a policy *generalist* — one network, many tasks, told what to do in natural language. This is the current SOTA frontier.

So the modern recipe = BC + action chunking + a flow/diffusion action head + a VLM backbone conditioned on language. That's the lineage from Mees-era BC to today.

---

## 2. The ecosystem you'll build in: LeRobot

For SO-101, the de-facto framework is **HuggingFace LeRobot**. It handles teleop, dataset recording (the `LeRobotDataset` format), training, and deployment, and it ships every architecture below as a first-class policy. Your whole pipeline — record → train → eval — lives here. Bimanual is a supported configuration (`bi_so101_follower`), and multi-camera (your D455 + 2 wrist cams) is standard.

---

## 3. The candidates

See `vla_architecture_comparison.csv` for the full matrix. Summary:

### Foundational (single-task, no language) — for *learning the mechanics*
- **ACT** — Transformer CVAE, action chunking + temporal ensembling. This is *the* bimanual method (born on ALOHA, 2 arms). ~52M params, trains on ~12GB. Implementing/training this first teaches you chunking, the dataset format, teleop, and eval — the bedrock. Not generalist (one task per policy, no language).
- **Diffusion Policy** — conditional diffusion over action trajectories. Excellent on contact-rich/precise tasks. Still single-task, no language.
- **VQ-BeT** — quantizes actions into a codebook, GPT predicts codes. Good for multimodal behaviour.

### Generalist VLAs (language-conditioned, multi-task) — the SOTA frontier
- **SmolVLA (0.45B)** — *purpose-built for SO-100/101 on consumer hardware.* Pretrained on community LeRobot datasets recorded on this exact arm family. Freeze the SigLIP + SmolLM2 backbone, fine-tune only a ~50M flow-matching action expert. The SmolVLA paper/blog (HuggingFace, 2025) reports it **matching or beating ACT and pi0 on SO-100/101 benchmarks** while running on a single modest GPU — worth verifying against the paper's own tables before treating as settled. → **best entry point to generalist VLAs for your setup.**
- **GR00T N1.5 / N1.7 (NVIDIA, ~3B)** — the **best-documented bimanual SO-101 path**: official tutorials fine-tune on 2× SO-100 with dual cameras, LoRA fits ~16GB. Great if you want a big-model generalist with a paved road.
- **pi0 / pi0-FAST / pi0.5 (Physical Intelligence, ~3B)** — flagship flow-matching (pi0) / autoregressive-token (FAST) generalists; pi0.5 targets open-world generalization to unseen homes. pi0 & pi0-FAST weights are open; pi0.5 weights are closed. Heavier (~24GB+).
- **X-VLA (0.9B)** — a recent strong performer: the project page (ICLR 2026 submission) reports SOTA across several simulation + real-robot benchmarks and an AgiBot World Challenge win. Soft-prompt Transformer + flow matching designed for cross-embodiment. The newest "frontier SOTA" candidate — treat the specific rankings as author-reported until independently checked.
- **VLA-0** — a VLM that emits action numbers *as text* with zero architectural change; the VLA-0 paper (2025) reports an improvement over SmolVLA on SO-100 (author-reported, verify against the paper and its authors/affiliation). Conceptually the cleanest to understand, useful as a learning lens.

---

## 4. Recommended path (staged — this is how you actually learn SOTA)

**Stage 0 — Plumbing.** Stand up LeRobot, calibrate both arms as a `bi_so101_follower`, wire all 3 cameras, teleoperate, and record a first small dataset. Get one clean pick-and-place logged end to end.

**Stage 1 — Implement/train ACT (foundations).** Train ACT on a single bimanual task. You learn action chunking, the dataset format, temporal ensembling, and the eval loop. This is your BC baseline and the direct descendant of the ALOHA work. Fast, fits a small GPU.

**Stage 2 — Go generalist with SmolVLA (the SOTA sweet spot for you).** Record 3–5 tasks with language annotations, fine-tune SmolVLA from its community-pretrained checkpoint. Now you have a *single language-conditioned multi-task policy* on your own two arms — the modern generalist recipe, on hardware it was designed for. This is the recommended primary target.

**Stage 3 — Push the frontier (optional, compute-permitting).** Fine-tune **X-VLA** (current champion) or **GR00T N1.5** (best-paved bimanual road) or **pi0.5** for open-world generalization, and benchmark against your SmolVLA. This is where you compare true SOTA generalists head-to-head on your rig.

**Paper-to-implement pairing:**
- Read + reimplement: **ACT** ("Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", Zhao et al. 2023) → hands-on BC + chunking.
- Primary build: **SmolVLA** (Shukor et al., 2025) → generalist VLA on your arms.
- Frontier study: **X-VLA** (ICLR 2026) and/or **pi0/pi0.5** (Physical Intelligence) → SOTA generalization.

---

## 5. Compute

Compute is not a constraint here. You stated you have access to an **Ada-generation GPU**, a **P6000**, a **Blackwell** card, and **RunPod credits** — I'm deliberately not assuming exact model numbers or VRAM, because that determines whether full fine-tunes vs. LoRA are the right call (see caveat below). In general terms, this stack is more than enough to cover the *entire* table, and — depending on the exact VRAM of your Ada/Blackwell cards — likely including **full (non-LoRA) fine-tunes of the 3B VLAs** (pi0, GR00T, X-VLA). Rough guidance by VRAM tier:

- **≥40GB (e.g. RTX 6000 Ada / A100-class)** — full fine-tunes of SmolVLA/X-VLA/pi0/GR00T, big batches, fast eval.
- **24GB (e.g. P6000, or Ada 4000/5000-class)** — ACT, Diffusion Policy, SmolVLA full fine-tune, and LoRA fine-tunes of the 3B models.
- **Note on the P6000**: it's Pascal, which predates bf16 — use fp16/fp32 there. The flow-matching VLAs prefer bf16, so keep those on the Ada/Blackwell cards.
- **Blackwell** — newest arch; needs a recent CUDA/PyTorch build (CUDA 12.4+/recent nightly wheels) before relying on it.
- **RunPod** — good for parallel multi-seed runs or renting an A100/H100 for the largest sweeps.

**To finalize the compute plan I need the exact GPU models and VRAM** — "Ada P6000 Blackwell" is ambiguous (P6000 is a 24GB Pascal Quadro; "Ada" could be an RTX 2000/4000/5000/6000 Ada with very different VRAM). Tell me the precise cards and I'll pin down full-fine-tune vs. LoRA per model. (Note: the T550 shown as this sandbox's local GPU is the hosted environment's, not your training hardware, and isn't usable for training anyway — real training runs on your boxes or RunPod.)


---

## 6. Addendum — "Is there nothing bigger?" and the mobile base

*(Added after clarification: you have ~100GB VRAM available, and the robot will eventually also drive a differential-drive 3-wheel base — two actuated front wheels + rear caster — with an optional 2D lidar.)*

### 6a. Yes there are bigger models — but bigger is *not* better here, and that's the important lesson

The models in the main table top out around 3B not because nothing bigger exists, but because **manipulation VLAs hit diminishing returns on parameter count fast** — the bottleneck is robot data, not model capacity. Concrete, load-bearing evidence:

- **RT-2-X is 55B parameters. OpenVLA is 7B. OpenVLA *beats* RT-2-X by 16.5% absolute task-success across 29 tasks — with 7× fewer parameters** (OpenVLA, Kim et al., CoRL 2024 / arXiv:2406.09246; ~970k OXE training episodes — author-reported). That's the single clearest data point that scale alone plateaus for robot control.
- The field's own framing (π0.5, WholeBodyVLA, the mobile-manipulation survey) is that **data scarcity**, not model size, is the ceiling — mobile-manipulation datasets are 10–100× smaller than fixed-base ones (author-reported, verify against the survey). Your ~100GB VRAM removes the *training* constraint, but a 30B model trained on a few hundred of your own episodes would overfit, not generalize.

So the right use of your big GPU is **not** "find a 30B model." It's: (a) run the strong 3–7B models at **full fine-tune** (no LoRA compromise) with large batches and fast iteration; (b) hold multiple camera streams + longer action-chunk context in memory; (c) later, do multi-model ensembling / distillation experiments. Bigger *than 7B* buys you little for a from-scratch two-arm-plus-base project; more and better-curated *data* is where the wins are.

The **largest genuinely relevant** models to know about:
- **OpenVLA (7B, Apache-2.0)** — the biggest fully-open, commercially-usable manipulation VLA worth running. Full fine-tune fits your ~100GB comfortably. Good "large model" study target.
- **RDT-1B (1.2B)** — largest *diffusion*-based bimanual foundation model; its unified action space **explicitly includes wheeled locomotion**, which is directly relevant to your base.
- **RT-2-X (55B)** — worth knowing as the ceiling reference, but closed and beaten by 7B OpenVLA. Not a build target.

### 6b. The mobile base changes the problem class: this is now *mobile manipulation / whole-body control*

Adding a differential-drive base means the policy's action space grows from "two arms + two grippers" to **"two arms + two grippers + base velocity (v, ω)"**, and — critically — the base and arms must be **coordinated**, not run as separate scripts. The literature calls the hard version *manipulation-aware locomotion*: moving the base to actively create the preconditions for a manipulation (approach, orient, stabilize), rather than "drive, stop, then manipulate." Two design choices you'll face:

1. **Decoupled (easier, standard baseline):** a navigation stack drives the base to a good pose (your 2D lidar → SLAM/costmap + a base-pose selector), *then* a fixed-base manipulation policy (ACT / SmolVLA / RDT) acts. Robust, modular, and the pragmatic first system. Your lidar is genuinely useful here for mapping/obstacle avoidance.
2. **Unified whole-body VLA (harder, SOTA):** one policy outputs base + arm actions jointly from cameras + proprioception + language. This is exactly what **π0.5** does — it drives a mobile manipulator through 10–15 minute multi-stage tasks in unseen homes — and what **WholeBodyVLA** targets via unified latent learning. This is the true frontier for your end goal.

**Key architectural implication for data collection (do this now, before you record anything):** whatever you build toward, **log base odometry (wheel encoders / velocity commands) and lidar in the same synchronized dataset as the arm/camera streams from day one.** Retrofitting base state into an arm-only dataset later is painful; the mobile-manipulation survey's explicit lesson is that hardware must "capture base odometry alongside arm state." LeRobot's dataset format handles extra state/action dimensions, so add the base DoF to the schema from your first recording.

### 6c. Updated recommendation given the mobile base

The staged path still holds, but the **frontier target sharpens from "generic VLA" to π0.5-class whole-body**:

- **Stages 0–2 unchanged** (plumbing → ACT → SmolVLA), done on the arms first. This is still how you learn the fundamentals, and a fixed-base skill is a component of the mobile system.
- **Stage 3 becomes mobile/whole-body**, and the primary paper-to-implement pairing shifts to:
  - **π0.5** ("a VLA with Open-World Generalization", Physical Intelligence 2025) via the **openpi** framework — the reference architecture for base+arm whole-body control. Weights are closed but the code/recipe is usable.
  - **RDT-1B** — if you want an **open-weights** large model whose action space already accommodates wheeled locomotion; strong few-shot behavior suits limited self-collected data.
  - **WholeBodyVLA** — read for the unified-latent / learn-from-video idea that attacks the mobile-data-scarcity problem.
- **Decoupled nav+manip** (lidar SLAM + fixed-base policy) is the recommended *first* mobile system before attempting a unified whole-body VLA.

All claims about specific benchmark numbers above are author-reported from the cited papers — treat them as directional and verify against the source tables before relying on them for a design decision.


---

## 7. Grounding in the actual Haller spec (from haller_ws/README.md, May 2026)

Now working from the real repo rather than chat description. Key hardware facts and what each implies for the architecture choice:

| Spec (from README) | Architectural implication |
| --- | --- |
| **Onboard compute: NVIDIA Jetson Orin Nano** | This is the binding **deployment** constraint, and it's the single most important new fact. An Orin Nano (~8GB shared RAM, ~40–67 TOPS — *verify your exact module*) will **not** run a 3B π0.5/GR00T or 7B OpenVLA in real time onboard. It comfortably runs ACT and is the target class SmolVLA was explicitly designed for (edge/consumer). |
| **2× SO-101 (Feetech STS3215), bimanual** | Matches the main table; ACT/SmolVLA/RDT all fit. |
| **Base: 3-wheel differential drive (2 driven front + rear caster), LK-TECH MF5010 BLDC over CAN** | ✅ **Resolved (2026-07): 3-wheel** — 2 actuated front wheels + rear caster. Differential drive → base action is `(v, ω)` (2 dims); schema-safe. (The old README "4-wheel" wording was stale and has been corrected.) |
| **2D LiDAR: Slamtec RPLIDAR A1M8** | Confirmed — feeds the decoupled nav route (SLAM/costmap → base pose) from Section 6b. |
| **ROS 2 Jazzy + LeRobot conda env; HMI; dataset-collection pipeline wired; RunPod inference docs already scoped for π0.5 / GR00T LoRA** | Stage 0 plumbing is **largely done**. You already have teleop (60 Hz leader↔follower), calibration, camera streams, a CLI dataset recorder, and a RunPod inference/finetune path. So you can jump toward Stage 1–2 sooner than a from-zero build. |

### The deployment split this forces (important)

Because training hardware (your ~100GB GPU / RunPod) and **inference** hardware (Orin Nano) are very different, decide the deployment mode explicitly — it constrains model choice more than training does:

1. **Onboard real-time (recommended for a responsive robot):** model must fit the Orin Nano. → **ACT** and **SmolVLA** are the realistic candidates. SmolVLA is the generalist sweet spot that actually deploys on your hardware. This is the strongest reason SmolVLA — not a 3B/7B model — is your primary generalist target.
2. **Cloud inference (your RunPod path):** run π0.5 / GR00T / OpenVLA on a rented GPU, stream actions to the robot over the network. Lets you use the big generalists, at the cost of network latency and a connectivity dependency. Your repo already has `runpod-inference.md` for exactly this — good for *studying* what a big VLA does on your data, less good for an untethered robot.
3. **Train big, distill small:** fine-tune a 3B generalist in the cloud, distill/quantize to an ACT/SmolVLA-scale student that runs on the Orin Nano. This is the advanced path once you have data and a baseline.

### Revised concrete recommendation for Haller

- **Primary on-robot generalist: SmolVLA.** It's the one model that is both a language-conditioned multi-task generalist *and* deployable on the Orin Nano. Your repo is already set up to record the data it needs.
- **Cloud frontier study: π0.5 (whole-body, mobile) and/or GR00T**, via your existing RunPod docs — for learning the SOTA and for the eventual base+arm unified policy, run as cloud inference or distillation source.
- **Foundations first: ACT**, which also runs onboard and teaches the fundamentals.
- **Base integration:** start **decoupled** (RPLIDAR SLAM → base pose, then arm policy), and **log base odometry + lidar into the LeRobot dataset from the first recording** so a unified whole-body policy (π0.5-class) is trainable later without re-collecting.

**Open items I need from you:** (1) ✅ resolved: **3-wheel** (2 driven front + rear caster, differential drive); (2) confirm the exact Orin Nano module (RAM/TOPS) so I can be precise about what runs onboard; (3) whether the goal is an **untethered** robot (forces onboard-deployable models) or **cloud-inference is acceptable** (opens the big generalists).


---

## 8. Going broad: the paradigm landscape beyond behaviour cloning

You asked to think super broad and aim for humanoid-level results. Good instinct — behaviour cloning (what Mees-era talks describe, and what the whole main table is) is **one of five** live paradigms, and the frontier systems combine several. Full matrix in `paradigm_landscape.csv`. Here's the map and where each fits a from-scratch 2-arm + mobile-base learner on a Jetson Orin Nano.

### Paradigm 1 — Behaviour Cloning / VLA (the mainstream you already have)
Supervised action prediction, action chunking, flow/diffusion head on a VLM. Mature, best-tooled (LeRobot), directly deployable. **Ceiling:** it can only be as good as your demonstrations, cannot learn from its own failures, and gets brittle out-of-distribution — small deviations from expert demos compound and drive the policy into unfamiliar states. This is *why* the other paradigms exist. **Role for you: the baseline and the deployable artifact.**

### Paradigm 2 — World Models (learn to *predict*, then plan/train in imagination)
Instead of only mapping observation→action, learn a model of environment **dynamics** and use it to imagine the consequences of actions. Lineage: Ha & Schmidhuber → PlaNet/Dreamer → DreamerV3 (one algorithm, many control tasks, fixed hyperparameters, Nature 2025) and DayDreamer (Dreamer-style world models learned directly on physical robots for locomotion/manipulation/navigation). Modern manipulation variants:
- **DreMa** (ICLR 2025) — a compositional world model using Gaussian Splatting + a physics engine to *imagine novel object configurations*; reports a real Franka learning new tasks from **one example per variation** (author-reported one-shot). This is a data-amplification engine — highly relevant given your data will be scarce.
- **Diffusion world models for RL** (World4RL, Ctrl-World, WMPO, VLA-RFT) — train/refine the policy inside a learned simulator instead of on the real robot, because real-robot RL is slow, hard to reset, and unsafe.

**Strength:** sample efficiency and imagination-driven generalization. **Cost:** physical accuracy is hard, models hallucinate. **Role for you: advanced — a way to multiply scarce data and do RL without wrecking the robot.**

### Paradigm 3 — Learning from Human / Internet Video (break the data bottleneck)
The core problem in robotics is data scarcity; the internet has effectively unlimited video of people manipulating things. This paradigm learns **latent actions** from *action-less* video and uses them to pretrain a policy, then maps latent→real actions with a small robot-data finetune:
- **LAPA / Latent Action Pretraining** (ICLR 2025) — the first unsupervised method to pretrain a VLA **without ground-truth action labels**: train a VQ-VAE to learn discrete latent actions between frames, pretrain to predict them, then finetune on a little real robot data.
- **UniVLA, Moto, GR-2, Being-H0, ConLA** — variations on task-centric or contrastive latent actions from human video.
- **EMMA** ("Scaling Mobile Manipulation via Egocentric Human Data", arXiv:2509.04443, 2025 — found as a reference in a related paper, not yet read directly; verify title/scope against the arXiv page) — reportedly targets **mobile** manipulation from egocentric human video, i.e. your problem class.

**Strength:** you can cheaply record phone/egocentric video of a task (or use public datasets like Ego-Exo4D, Something-Something) to pretrain, then finetune on far fewer teleop demos. **Role for you: high-leverage — the cheapest route to generalization given limited teleop time.**

### Paradigm 4 — RL Fine-tuning / Autonomous Improvement (get *better than your demos*)
Start from a BC/VLA policy and improve it with reinforcement learning — online, offline, or in a world model — so it learns from its own successes and failures rather than being capped by demonstration quality:
- **SimpleVLA-RL, VLA-RL, ConRFT, pi_RL** — scale VLA training with RL; report improved generalization and long-horizon performance over pure imitation (author-reported).
- **ReWiND** (CoRL 2025) — learns a reward function from data and reports improving a real-robot policy **~5× in ~1 hour** of online interaction (author-reported).

**Strength:** the only paradigm that surpasses your demonstrations and self-improves during deployment. **Cost:** reward design, resets, and safety on real hardware. **Role for you: the step after a working BC baseline — pairs naturally with a world model to avoid real-robot rollouts.**

### Paradigm 5 — Unified Heterogeneous "data-pyramid" models (where humanoid-level actually comes from)
The frontier generalists don't pick one paradigm — they **fuse all of the above**. GR00T N1 is the clearest example: a dual-system (System-1 fast action / System-2 reasoning) model co-trained on a *pyramid* of real-robot trajectories + human videos + synthetic/neural-generated video, using a **latent-action codebook + inverse dynamics model** to put action-less video into a shared latent action space, with a single set of weights spanning single-arm, bimanual, and humanoid embodiments. π0.5 and WholeBodyVLA follow the same "everything, unified" recipe.

The reality check on "humanoid-level": GR00T N1 pretraining reportedly took **~50,000 H100 GPU-hours across 1,024 GPUs**, plus **780k simulation trajectories** from DexMimicGen (author-reported). **You will not pretrain one of these** — nobody outside a handful of labs does. What you *do* is **fine-tune the open checkpoint** (GR00T N1.5 is released) on your own Haller data. That's the realistic path to a "humanoid-level" generalist on your two arms + base.

### How this restructures the recommendation

Think of it as a **capability ladder**, not a single model choice — each rung teaches a paradigm and feeds the next:

1. **BC baseline (ACT → SmolVLA)** — learn the fundamentals, get a deployable policy on the Orin Nano. *Paradigm 1.*
2. **Video pretraining (LAPA-style / fine-tune GR00T N1.5)** — inject internet + your own human-video priors so you need fewer teleop demos and generalize better. *Paradigms 3 + 5.*
3. **RL fine-tuning (SimpleVLA-RL / ReWiND-style)** — push past demonstration quality and let the robot self-improve. *Paradigm 4.*
4. **World-model imagination (DreMa / diffusion WM)** — amplify scarce data and do safe RL in imagination. *Paradigm 2.*

Every rung is a concrete paper you can implement on Haller, and together they *are* the modern state of the art — not just "newer BC." The single most important non-obvious lever for you specifically is **Paradigm 3 (human video)**: it's the cheapest way to fight the data scarcity that will otherwise cap everything, and EMMA shows it works for the mobile-manipulation setting you're building toward.

*All specific quantitative results in this section are author-reported from the cited papers; treat as directional and verify against the source tables before relying on them for a design decision.*


---

## 9. What's genuinely new in 2026 (mid-2026 frontier scan)

*Provenance note: the items below come from 2026 surveys, curated paper lists (Awesome-WAM, awesome-physical-ai, awesome-world-models), and reference sections surfaced by search — not papers I have read end-to-end. arXiv IDs are given where available so you can verify. Treat as leads, not established results.*

### The headline: World-Action Models (WAMs) — world models and VLAs are merging
The single biggest 2026 shift is that the two paradigms I described in Sections 1 and 8.2 are **fusing**. Rather than a reactive VLA (obs→action) *or* a separate world model (predict dynamics), 2026 work builds policies with an **explicit predictive/world-model structure inside** them. There is now a dedicated "World-Action Model" survey and reading list (OpenMOSS/Awesome-WAM) taxonomizing this into cascaded vs. joint WAMs. Representative 2026 entries:
- **VLA-JEPA** (arXiv ID *not independently confirmed — search for the exact title before citing*) — augments a VLA with a JEPA-style *latent* world model (predict in representation space, not pixels).
- **RISE: Self-Improving Robot Policy with Compositional World Model** (arXiv:2602.11075, 2026 — this ID appeared directly in search results) — policy improves itself using a compositional world model.
- **WoVR / Sword / WoVR-style post-training** (2026.02–05) — use world models as *reliable simulators* to post-train VLA policies with RL, attacking the "real-robot RL is slow/unsafe" problem from Section 8.4.
- **"DreamZero" / "Say, Dream, and Act"** (titles as I recall them from paper-list browsing — *not independently confirmed, verify the exact titles/IDs*) — the idea being video/world models used directly as zero-shot or instruction-driven policies.

**Why it matters for you:** this is the concrete realization of the "combine paradigms" thesis from Section 8. If you want to be on the actual 2026 frontier (not 2024's), a WAM-style approach — a VLA with a latent world-model head, used both for prediction and for safe RL post-training — is the direction. It's also *research-grade and unstable*; not a first build.

### The finding that should change your architecture choice: VLM backbone quality ≠ VLA quality
A 2026 study (title/venue as I recall them — *"VLM4VLA: Revisiting Vision-Language-Models in Vision-Language-Action Models", reportedly an ICLR 2026 submission; I have not confirmed the exact title/ID, verify before citing*) reports that **downstream VLA performance has essentially no correlation with the backbone VLM's score on standard vision-language benchmarks** (caveat: benchmark-setting, limited real-robot eval). A companion line of work (reportedly "Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting" — *also unconfirmed, verify*) argues for relabeling robot data as text-like subtasks to avoid catastrophic forgetting when turning a VLM into a VLA.
**Why it matters for you:** don't chase the biggest/best VLM backbone assuming it yields the best robot policy — the evidence says it doesn't. This reinforces the Section 6a conclusion (bigger ≠ better) with a 2026 data point, and it's *good news for a small-GPU-deployable model like SmolVLA*: the backbone doesn't have to be enormous.

### New foundation models announced in 2026 (verify specs before relying)
- **LingBot-VLA** (Ant Group, ~Jan 2026) — a new VLA foundation model for real-world manipulation. *(Announcement-level; I could not confirm details from a primary source — verify.)*
- **Qwen-RobotSuite** — the Qwen team's set of embodied models, referenced in early-2026 coverage. *(Mention-level only; verify.)*
- **π0.5** remains the reference generalist mobile-manipulation system cited through mid-2026 (kitchen/bathroom/bedroom tidying in unseen homes).

### Other live 2026 threads worth a look
- **Video generators as policies** — "VideoVLA: Video generators can be generalizable robot manipulators" (2025/26): use a large pretrained video model as the policy substrate. Extends the Section 8.3 human-video idea.
- **Tactile / contact-aware VLAs** — e.g. TacVLA (2026): fuse tactile sensing into a VLA for contact-rich tasks. Relevant if you ever add fingertip sensors to the SO-101 grippers.
- **Deformable-object world models** — "Learning to unfold cloth: scaling world models to deformable manipulation" (2026.02) — cloth/deformables, a classic hard case.
- **Efficiency/deployment** — a large 2026 sub-literature on distilling/accelerating VLAs (ActDistill, LatBot, parallel/early-exit decoding) — directly relevant to running a generalist on your Orin Nano.

### Net update to the recommendation
Nothing in 2026 dethrones the **staged capability ladder** (Section 8) — it sharpens the top rung:
- Your **Stage-2/3 frontier target** should be read as a **World-Action Model**, not a plain VLA: fine-tune a generalist (GR00T N1.5 / π0.5-class) *and* add a latent world-model / RL-post-training component (VLA-JEPA / RISE / WoVR-style) once you have a BC baseline and data.
- The **VLM4VLA finding** de-risks choosing SmolVLA for onboard deployment: a modest backbone is not the bottleneck.
- The **efficiency/distillation** literature is your practical bridge from "big model in the cloud" to "runs on the Jetson."


---

## 10. JEPA — the paradigm I under-listed (and why it's genuinely distinct)

Good catch. I mentioned VLA-JEPA only in passing; the **JEPA line deserves first-class status** because it is architecturally *unlike* everything else in this document. It's Yann LeCun's long-standing bet, and in 2025 **V-JEPA 2** made it concrete for robotics.

### What JEPA actually is (and how it differs)
- **Joint-Embedding Predictive Architecture**: instead of predicting future *pixels* (generative world models like Cosmos) or mapping observation→action (VLAs), a JEPA predicts the *latent representation* of a future state from the current one. It learns "what happens next" in an abstract feature space, discarding pixel-level detail that doesn't matter for control. This is the "predict in representation space" idea that VLA-JEPA (Section 9) borrows.
- **It plans; it doesn't react.** V-JEPA 2-AC is not a policy. It's a world model wrapped in **Model-Predictive Control**: given a **goal image**, it uses the Cross-Entropy Method to search for an action sequence whose *imagined* latent future is closest (L1 distance) to the goal's latent, executes the first action, then re-plans. This is a fundamentally different control philosophy from BC/VLA — closer to classical MPC, but with a learned neural dynamics model.

### Why it's a big deal (V-JEPA 2, Meta, June 2025 — open weights)
- Pretrained on **>1 million hours of internet video**; the action-conditioned head trained on only **~62 hours of unlabeled Droid interaction data** — no action labels, no reward, no task labels.
- Deployed **zero-shot on Franka arms in two labs not in the training data**, from an **uncalibrated monocular RGB camera**, same weights everywhere. Reported ~100% on reach, ~60–80% on grasp/pick-place (author-reported).
- Reported **~30× faster planning than a diffusion world model (Cosmos)** baseline (~16s/action vs ~4min) — though still slow compared to a reactive VLA running at 30–100 Hz.
- Related: **DINO-WM** (2025 — authors/venue as I recall them, *not confirmed in-session; verify*) builds the same predict-then-plan world model on frozen DINO features.

### Honest assessment of JEPA *for Haller*
**Strengths that matter to you:**
- **Data efficiency is extreme** — it directly attacks your central problem (scarce robot data) by leaning on passive video + a tiny bit of robot interaction. This is the same lever as Section 8.3, taken furthest.
- **Open weights**, works from a **single uncalibrated camera** (you have several), zero-shot generalization is exactly the "generalist" property you want.
- It's the cleanest way to *learn the world-model-planning paradigm* hands-on, which 2026's WAM trend (Section 9) is converging on.

**Real limitations for your use case:**
- **Goal-image, not language.** You said you want language-conditioned generalist behavior ("tell it what to do"). JEPA-AC is conditioned on a *goal image*, which is a different (and for many tasks, more awkward) interface. You'd specify tasks by showing a target photo, not a sentence.
- **Slow control loop** (~16s/action) — fine for slow tabletop tasks and for study, poor for anything dynamic or for a moving base.
- **Not a LeRobot drop-in** — no turnkey SO-101 recipe; you'd wrap the released model in your own MPC/controller. More of a research build than SmolVLA/ACT.

### Where JEPA fits in your ladder
It's a **parallel high-value track**, not a replacement for the BC→VLA spine:
- If your priority is **language-conditioned, real-time, deployable-on-Orin-Nano** generalist behavior → SmolVLA remains the primary spine.
- If you want to **learn the predict-then-plan / world-model paradigm** that 2026 is converging toward, with extreme data efficiency and zero-shot generalization → **V-JEPA 2-AC is the reference to study and reproduce**, and it slots in as an alternative to (or companion for) the Section 8.2 world-model rung. The 2026 endgame (Section 9) — VLA-JEPA, RISE, WoVR — is literally *fusing this JEPA prediction idea back into language-conditioned policies*, so understanding V-JEPA 2 is understanding where the field is heading.

Bottom line: JEPA is not "better or worse" than VLAs — it's a **different axis** (plan-in-latent-space vs. react-from-language). For a learner aiming at the frontier, it's worth a dedicated track; for a shippable language-driven Haller policy, it's complementary rather than primary.
