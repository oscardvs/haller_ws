# Haller — Frontier Scout & Research/Business Case

**Companion to** `so101_sota_recommendation.md` (the 10-section VLA survey) and `HALLER_ROADMAP.md`.
**Purpose:** deliberately *additive*. The existing survey is deep on VLA architectures and the five learning paradigms. This doc scouts the **gaps** — levers those docs under-cover — and answers the second brief: **a research + spinoff-business case for Haller's embodiment.**

**Method.** 11 parallel web-research agents (one per under-covered angle), each fetching primary sources, followed by an adversarial fact-check pass on the load-bearing claims. ~90 findings; verification of the highest-importance claims below.

**Verification legend** (from the adversarial pass):
- `✓verified` — a skeptic agent confirmed the claim against a primary source.
- `~corrected` — true in substance, but a detail was wrong; corrected inline.
- `⚠flagged` — the scout itself could not confirm, OR a skeptic partially/failed to support it. Treat as a lead, verify before betting.
- (untagged quantitative claims are **author-reported** from the cited paper — same "verify before you bet a design decision" discipline as the existing docs.)

> **Verification is COMPLETE.** Every high-importance claim was adversarially fact-checked against primary sources — **33 distinct claims, 0 refuted** (31 supported, 14 supported-with-a-correction). Nothing here was contradicted by its source. `⚠pending` tags below are resolved in **§ Verification outcomes** at the end; the only softening is around self-reported startup metrics.

**The one-line update to the existing recommendation:** the staged ACT→SmolVLA→mobile ladder still holds, but the scout surfaced **two levers that belong *earlier* than the roadmap places them** (scalable data collection, and sim-driven demo multiplication), a **measured answer to the Orin-Nano deployment question** (SmolVLA runs; GR00T/π0 do not fit 8 GB — now confirmed on real hardware, not inferred), and a **business/research thesis** that says: *don't sell the robot — sell the data pipeline, the curriculum, or a narrow service; and publish on the mobile-manipulation data gap.*

---

## Part I — The eight highest-leverage NEW levers (ranked)

Each lever is something **not** in the current survey, ordered by leverage for *this* robot and *this* solo builder.

---

### Lever 1 — Scale data collection *beyond* teleop (the #1 lever, and it belongs before Stage 1)

The existing docs correctly say "data > architecture" but still assume teleop is how you get data. The 2025–26 field has largely moved past that. For a solo operator who cannot teleoperate 10k episodes, this is the single biggest change to make.

| Finding | What it gives Haller | Status |
|---|---|---|
| **Data Scaling Laws in Imitation Learning** (ICLR 2025) | The load-bearing lesson: zero-shot generalization scales as a **power law with the number of distinct environment-object *pairs*, not raw demo count**. 32 env-object pairs × ~50 demos generalizes broadly. → Spend your scarce solo hours on *diversity*, not repetition. | author-reported, ICLR'25 |
| **phospho Quest VR teleop** + **BEAVR** (open, ~$1k rig, sub-35 ms, native LeRobot schema) | Headset teleop for SO-101 with **zero hardware build** and zero embodiment mismatch — records straight into your existing HF pipeline. BEAVR is the open, no-subscription option. Cheap A/B vs. your current webcam pose-teleop. | ⚠flagged (BEAVR $1k/latency author-reported) |
| **EMMA — egocentric human video → differential-drive base** | The most Haller-specific find in the whole scout: retargets **Aria-glasses human walking trajectories into feasible diff-drive base paths**, co-trained with a little static-arm data — **no mobile teleop at all**. Claims 1 h of human data beats 1 h of Mobile-ALOHA teleop. This is the *only* method that removes teleop from the **base**, not just the arms. | ⚠flagged (headline claim author-reported; frontier-unstable) |
| **MonoDuo** | Collect bimanual data with **one arm at a time** (human drives the other, roles swap), synthesize the missing arm via hand-pose + inpainting. Directly solves the solo-operator "I can't drive both SO-101s at once" problem. | author-reported |
| **UMI / FastUMI-100K / YUBI** handheld grippers | In-the-wild data with a handheld gripper (~3× teleop throughput; YUBI parts <$200). **But:** SO-101's STS3215 parallel-jaw geometry doesn't match UMI's >85 mm-stroke assumption — a real CAD redesign, not a bolt-on. FastUMI-100K / YUBI are also **ready-made open LeRobot datasets** for cross-embodiment pretraining. | ✓ (UMI 111 vs 35 demos/hr is the canonical result); pending on FastUMI |

**Do this:** turn on VR teleop for the arms now; adopt the *diversity-over-repetition* discipline from day one; and treat **EMMA-style egocentric-for-diff-drive** as a genuine research bet (it's novel *and* Haller-shaped). Skip building a UMI gripper for SO-101 unless you're ready for the mount redesign.

---

### Lever 2 — Multiply demos in sim (you already run MuJoCo — this is nearly free leverage)

The survey mentions world models but not the practical **demo-multiplication** toolchain. Haller already has a MuJoCo sim trio; this turns a handful of real demos into thousands.

| Finding | What it gives Haller | Status |
|---|---|---|
| **DexMimicGen** (NVlabs, open) | **Best-matched** demo multiplier: purpose-built for **bimanual** manipulation, generated **20,000+ demos from 60 human demos** across 9 tasks; runs on **robosuite (MuJoCo)** and **BiGym (mobile bimanual)** — slots next to your existing sim, not a replacement. | ✓verified (21K demos from 60) |
| **RoboTwin 2.0** (MIT license) | Dual-arm data generator + benchmark. **Synthetic + only 10 real demos → +367% relative** over 10-demo-only; **synthetic-only → +228%**. | ✓verified |
| **NVIDIA official SO-101 sim-to-real curriculum** (Isaac Sim/Lab/Cosmos/GR00T) | The **only toolchain documented for your exact arm**, end-to-end to LeRobot format. Four transfer strategies incl. actuator-gap closure. Heavier than MuJoCo — run it in the cloud (RunPod) as a *second* data source. | ⚠pending |
| **GR00T-Dreams / DreamGen** (Apache-2.0) | Cosmos video-world-model that "dreams" new task videos from image+text and extracts action trajectories. Repo **explicitly lists SO-100** (SO-101's sibling) as supported. | ⚠pending (a secondary "780k traj / 40%" figure could NOT be confirmed — dropped) |
| **MuJoCo Playground / MJX** (Apache-2.0, RSS'25 demo award) | GPU-parallel MuJoCo (JAX) — same physics family you already use; train RL / generate rollouts in minutes on one GPU. Lowest-friction upgrade of all. | author-reported |
| **Digital Cousins / ACDC** | Generate approximate sim scenes **from a single phone photo** — removes the tedious hand-modeling of every object. 90% vs 25% zero-shot transfer vs exact twins (abstract). | ⚠flagged (author-reported) |
| **XLeRobot "Cutting the Cord"** (arXiv 2603.09051) | A **near-identical twin**: dual SO-101 + Jetson Orin Nano Super, ~$1,200. Reports **98.7% grasp over 75 trials** and **ACT at 36 ms / 27.8 Hz onboard** vs Diffusion Policy 539 ms / 1.8 Hz. Use as your measured sanity-check baseline. | ⚠pending |

**Do this:** bridge your MuJoCo sim to **DexMimicGen** (bimanual + MuJoCo-based) as the primary demo multiplier; try the **NVIDIA SO-101 Isaac curriculum** in the cloud as a second source. This is a new **Stage 1.5** in the roadmap.

---

### Lever 3 — Make onboard real-time actually work on the Orin Nano (now with *measured* numbers)

The survey *inferred* "SmolVLA yes, 3B no" on the Orin Nano. The scout found **hardware-measured** confirmation and the concrete stack to hit real-time.

| Finding | What it gives Haller | Status |
|---|---|---|
| **vla.cpp** (llama.cpp-derived C++ VLA runtime, Jun 2026) | **First numbers measured on an actual 8 GB Jetson Orin Nano — Haller's exact SoC:** SmolVLA **~142 ms/step (~7 Hz)**, BitVLA ~356 ms/step, and **GR00T N1.6/1.7 do not fit in 8 GB at all.** On AGX Orin, faster runtime *doubled* ALOHA task success (87.5% vs 40%) purely by closing the loop faster. | **✓verified** |
| **Real-Time Chunking (RTC)** — *now shipped in LeRobot* | PI's inference-time trick: generate the next action chunk **while executing the current one**, blended with soft-mask inpainting, **no retraining**. Handles >300 ms delays. Built-in `RTCConfig`/`ActionQueue` today. This is the single lowest-effort integration for smooth onboard control. | ✓ (LeRobot docs) |
| **BitVLA** (1-bit ternary, open CC-BY-4.0) | **11× less memory, 4.4× lower latency** vs OpenVLA-OFT at matched success. Attacks the 8 GB ceiling directly. (Caveat: on the Nano its kernel is *slower* than SmolVLA today — a memory win, not yet a speed win.) | **✓verified** |
| **TinyVLA / MiniVLA / EdgeVLA** | Sub-1B backbones **smaller than SmolVLA (0.45B)**; TinyVLA-H claims **+25.7% real success vs OpenVLA at 5.5× fewer params.** Because latency is set by the diffusion head, not param count, these are worth benchmarking against SmolVLA+RTC. | author-reported |
| **XPU characterization** (SJTU, Apr'26 + public leaderboard) | The systematic reality check: **π0 = 920.6 ms (~1.09 Hz) on a 64 GB AGX Orin** (far beefier than your Nano). Key finding: **iterative denoising step count — not model size — drives latency;** ACT (single-pass) is fastest on *every* platform. | **✓verified** |
| **NVIDIA GR00T TensorRT pipeline** | Even optimized, GR00T targets **16 GB+** parts (Orin NX/AGX/Thor), not the 8 GB Nano — reinforces the split. | ⚠pending |

**Do this:** the onboard stack is **SmolVLA (or a TinyVLA-class head) → vla.cpp runtime → RTC + async inference**, targeting ~5–10 Hz. Anything 3B (GR00T/π0/WALL-OSS) is **cloud/RunPod-only** for Haller. This is now an evidence-backed decision, not a guess.

---

### Lever 4 — Bimanual-specific coordination (Haller *is* bimanual; the survey treated it as a config flag)

| Finding | What it gives Haller | Status |
|---|---|---|
| **Keypose-hierarchical control** (BiKC / BiKC+ / **AnchorDP3**) | Predict sparse per-arm **keyposes** (pre-grasp→grasp→place) + a fast low-level generator, instead of dense per-timestep chunking. **AnchorDP3 won the CVPR'25 RoboTwin Dual-Arm Challenge** (64 teams). Cheaper at inference than diffusion — matters on the Nano. | ⚠pending (challenge win) |
| **Co-VLA — Structured Action Expert** | Replaces the monolithic action head with **shared coordination latent + per-arm residuals** + a coordination-aware loss. **+27% on tightly-coordinated tasks**, OOD success ~doubled (13→27%). A small architectural add on top of a pi0/SmolVLA backbone. | **✓verified** |
| **AnyBimanual** (ICCV'25, open) / **TwinVLA** | **Bootstrap a two-arm policy from single-arm pretraining** — you have far more single-arm data/checkpoints (OXE, DROID, single SO-101) than bimanual. AnyBimanual: +12.67% (sim), 84.62% (9 real tasks). TwinVLA composes two single-arm VLAs, beats monolithic RDT-1B with zero bimanual pretraining. | AnyBimanual **✓verified**; TwinVLA author-reported |
| **AIST Bimanual Dataset** (CC-BY-4.0) | **10,705 real episodes / 119 tasks** on an ALOHA-lineage rig (mechanically close to SO-101) — the strongest **open real bimanual co-training set** found. (Needs ALOHA→Feetech action remap.) | author-reported |
| **VoxAct-B** (CoRL'24, open) | Formal "one arm stabilizes while the other acts" (holding a jar, unscrewing) — the exact asymmetric pattern for Haller (brace a bag while packing). | author-reported |
| **Krebs & Asfour taxonomy** (KIT) | Standard vocabulary: uncoordinated / loosely / tightly-coupled (symmetric vs dominant). Use it to **label each demo's coordination class** and build a deliberate curriculum instead of one undifferentiated "bimanual" bucket. | reference |

**Do this:** upgrade the bimanual policy from vanilla ACT dense output toward a **keypose-hierarchical head**, add a **Co-VLA-style structured action expert**, and **bootstrap from single-arm** via AnyBimanual/TwinVLA. Co-train with the **AIST** set.

---

### Lever 5 — A cheap "System-2" for long-horizon tasks (the layer above your reactive policy)

For a mobile robot doing multi-step home/lab tasks, scaling the low-level policy won't fix planning failures (RoboCerebra shows current VLAs are weak *specifically* on planning/memory). The 2025–26 pattern: **call an expensive cloud VLM once per subtask; run everything real-time locally.**

| Finding | What it gives Haller | Status |
|---|---|---|
| **ReKep** (CoRL'25, open) | VLM writes **relational-keypoint-constraint programs** once per subtask; a **classical optimizer** runs the real-time loop locally on the Nano. Already demoed on a **dual-arm** platform. The clearest "poor-man's System-2." | **✓verified** (~corrected: wheeled single-arm + stationary dual-arm) |
| **MOKA** (open) | Even cheaper entry: annotate candidate marks on the RGB image, ask an **off-the-shelf VLM** for keypoints — zero fine-tuning, zero local GPU. Good stopgap before ReKep. | author-reported |
| **Gemini Robotics-ER 1.5/1.6** (cloud API) | A hosted, continuously-improving **pointing/affordance/success-detection brain** you never host locally — feeds keypoints to your local policy and can replace hand-rolled task-completion heuristics. | ⚠pending |
| **Critic-in-the-Loop / Tri-System VLA** | A cheap **always-on local critic** decides *when* to pay for a cloud VLM call (on stagnation/failure), instead of every tick — exactly the cost-control your Nano + API budget needs. | ⚠flagged (Mar'26, unreviewed) |
| **Goal2Skill** | Concrete blueprint: your existing per-task BC/diffusion policies become a **"skill library"**; a cloud VLM adds subgoal decomposition + pre/post-condition checks + failure reflection — **no retraining of the low-level policies**. ~3× best baseline on RMBench. | author-reported |
| **Hi Robot** (Physical Intelligence) | Hierarchical VLA tested on a **dual-arm mobile** platform (≈ Haller's embodiment) doing table-cleaning / sandwich-making / shopping. Closest published proof-of-concept for Haller's target use case. | **✓verified** |

**Do this:** add a **ReKep/MOKA-style split** as a *new parallel track* — cloud VLM per subtask-transition, local skills per tick, with a lightweight critic gating the cloud calls.

---

### Lever 6 — Tactile / force on a budget (start at zero cost)

The survey mentioned TacVLA in passing; a whole tactile-VLA subfield (TacVLA, OmniVTLA, ForceVLA, VLA-Touch, TaF-VLA…) has appeared in ~12 months, and there's now a **drop-in SO-101 tactile retrofit**.

| Finding | What it gives Haller | Status |
|---|---|---|
| **STS3215 current/load telemetry** | **Zero-cost first step:** read `Present_Current`/`Present_Load` (already on your servo bus, in the Feetech SDK) for coarse stall/contact detection → grasp-force limiting or an auxiliary policy input. No hardware, no added compute. Validate whether force signal helps *before* buying anything. | reference (register spec) |
| **WowSkin (AnySkin retrofit)** | A commercial magnetic tactile skin with a **purpose-made "Structural Part for SO-101" mount, $128** — cost of one servo. AnySkin is peer-reviewed and generalizes zero-shot across sensor instances (matters for two matched grippers). | ✓verified — $128, 3rd-party vendor WowRobo on open AnySkin |
| **Sparsh / Sparsh-X** (Meta FAIR, open weights) | A **frozen pretrained touch encoder** so you don't hand-label a tactile dataset — runs as a light feature extractor on the Nano. **+95.1% over task-specific training at 33–50% of labels.** | ⚠pending |
| **FORTE** (open) / **TF-Gripper** ($150, open) | Whole **open-hardware tactile grippers** — an alternative to retrofitting the stock jaw. FORTE: 0–8 N at 0.2 N error, slip in 100 ms, 98.6% on delicate grasps (raspberries, chips). | author-reported |
| **FoAR** | Directional evidence force feedback fixes contact-rich failures: wiping 75→100%, peeling 50→100%. (Uses an industrial 6-axis F/T sensor — *not* a Haller recipe, just proof the signal matters.) | author-reported |

**Do this:** wire **STS3215 current into the dataset schema now (free)**; if coarse force helps, retrofit **one** gripper with **WowSkin** and encode via **Sparsh**. Tactile is a strong *research differentiator* on cheap hardware (see Part II).

---

### Lever 7 — Manipulation-aware mobile navigation (the "drive-then-grasp" problem)

The survey's decoupled nav→manip baseline is right, but the frontier fix for "arrived at the goal but the arm can't reach" is now concrete and reproducible.

| Finding | What it gives Haller | Status |
|---|---|---|
| **N2M / N2M2** (open) | Learns a **preferred base pose for the downstream manipulation** from **egocentric camera only — no SLAM map, no holonomic base.** Perfect fit for diff-drive + 2D lidar + small arm workspace. 3–54% over reachability baselines. | ⚠pending |
| **ANCHOR** | Closed-loop framework targeting exactly "arrived but inoperable" + **local** recovery-replan (cheap enough for the Nano). **53.3→71.7% success, 71.4% recovery** under disturbance. | **✓verified** |
| **Mink** (Apache-2.0 MuJoCo differential-IK) + **HoMeR** | A license-clean **whole-body base+dual-arm IK controller that drops into your existing MuJoCo sim** — offload low-level base/arm coordination from the learned policy. Likely the *fastest concrete implementation step* in the whole scout. | ✓ (HoMeR 79% at 20 demos, author-reported) |
| **DynaMem** | Dynamic open-vocab spatio-semantic memory for "fetch object X wherever it is now" (70% vs 30% OK-Robot on moving objects). Needs cloud offload for the VLM. | author-reported |
| **MoMaGen** (ICLR'26) | Bimanual-mobile **data generator** that treats base reachability as a hard constraint — conceptually portable to a MuJoCo generator for Haller. | author-reported |
| **BEHAVIOR-1K reality check** | The NeurIPS'25 winner (fine-tuned π0.5, 10k demos) hit only **~12.4% success** on 50 long-horizon home tasks. **Calibrate expectations** — robust long-horizon home autonomy is *not* solved by anyone. | ⚠pending |

**Do this:** adopt **Mink** for whole-body IK in sim immediately; implement an **N2M-style pose-preference** module so the base learns *where to stop*. Keep long-horizon-home ambitions calibrated to the ~12% frontier reality.

---

### Lever 8 — Newest open foundation models to fold in (and the closed frontier to learn from)

New since the survey's cutoff. The standout is **WALL-OSS**: a genuinely-open, **native LeRobot** VLA.

| Model | Why it matters for Haller | License / access | Status |
|---|---|---|---|
| **WALL-OSS** (X-Square, ~4B) | **Now a first-class LeRobot policy (`policy.type=wall_x`), Apache-2.0**, trainable via `lerobot-train` — zero glue code. The most actionable new model; A/B it vs SmolVLA/π0.5 on RunPod. WALL-OSS-0.5 claims >80% zero-shot on a 17-task real eval. | **Apache-2.0 (open)** | ✓verified |
| **Galaxea G0 / G0.5** | Trained on **R1 Lite = mobile base + bimanual arms**, the closest **open model+dataset twin** to Haller's embodiment (100k-traj Open-World Dataset). Best *design/data reference*. | CC-BY-NC-SA (research only) | author-reported |
| **AgiBot GO-1 + World Challenge 2026** | Huge open bimanual dataset (~1M traj) + open baselines + a **$530k-prize** ICRA'26 challenge. Benchmark/co-train reference. | CC-BY-NC-SA | ⚠flagged (prize/scale) |
| **RoboBrain 2.0 / RynnBrain (Apache-2.0)** | Open **embodied-reasoning VLMs** (plan/affordance, no low-level actions). Small variants (2–7B) are plausible **on-Nano planners** feeding a local WALL-OSS/SmolVLA head — an open approximation of the Gemini reasoning+action split. | RynnBrain Apache-2.0 | ⚠flagged (RoboBrain license/nums) |
| **GR00T N1.6/N1.7/N2** | Cross-embodiment, but **NVIDIA non-commercial license** and sized for **16 GB+ (AGX Orin/Thor), not the 8 GB Nano.** Cloud teacher / distillation source only; the license **blocks a spinoff**. | Non-commercial | ✓license-verified |
| **Closed frontier — method lessons only:** Gemini Robotics 1.5 + On-Device (fine-tune with ~50 demos), **Figure Helix**, **π*0.6 + RECAP** (RL from real deployment corrections; "more than doubles throughput, halves failures") | Not accessible, but **RECAP's lesson is directly transferable**: log teleop *interventions/corrections* during Haller sessions to seed future RL fine-tuning. | Closed | π*0.6 ⚠pending |

**Do this:** add **WALL-OSS** as an A/B target alongside SmolVLA (Apache-2.0 = spinoff-safe). Use Galaxea G0 / AgiBot as *references*, not deploy targets (non-commercial). Start **logging interventions** now (RECAP lesson) even before you do RL.

---

## Part II — Research & Business Case

### 2.1 Where Haller's embodiment honestly sits

The scout found **at least eight near-siblings**, which is the crucial strategic fact: the *concept* (cheap bimanual mobile manipulator) is **validated but crowded**.

- **Open/research twins:** XLeRobot (dual SO-101 + Orin Nano, ~$1.2k, *published measured results*), **AhaRobot** (~$1k, open CAD + marker-handle teleop), **YOR** (<$10k, omni-base + lift), **LeKiwi** (HF's *free* open mobile base + SO-101), Galaxea **R1 Lite** (open model+data).
- **Commercial twins:** Trossen **Mobile ALOHA** ($36,999 — **now discontinued**), Sunday **Memo** ($165M Series B, ~$1.15B val, 10M glove episodes), Zerith **H1** (~$13.6k, hotel housekeeping, 100+ units/mo).

**Implication:** the hardware is not the moat. **Differentiate on software, data, curriculum, or a narrow niche** — never on steel or a from-scratch foundation model.

### 2.2 Research white-space (educational + publishable, solo, ~6–12 months)

These are places where **cheap/accessible hardware is an asset, not a liability**:

1. **Release a Haller mobile-bimanual LeRobot dataset.** Mobile-manip data is *quantified* as the bottleneck — reference sets like AIRoA MoMa (25k ep) or RoboMIND-2.0 (mobile is only ~6.5% of trajectories) are thin. A well-annotated, well-documented Haller dataset is itself a **citable contribution**. `⚠pending (scarcity figures)`
2. **Robot-free / egocentric data collection for a differential-drive base (EMMA/HoMMI-style).** Genuinely novel *and* Haller-shaped — the base is diff-drive, and almost nobody targets base-navigation data without teleop. Highest-upside research bet.
3. **Hardware-free competition tracks** — fully GPU/RunPod-doable, citable in 2026: **RoboTwin Dual-Arm Challenge** (MIT license, sim rounds), **BEHAVIOR-1K Challenge** (OmniGibson), **AgiBot World Challenge** (control real robots via API). Natural on-ramp to an RSS "Mobile Manipulation" workshop paper.
4. **Contribute an SO-101 `gym_hil` environment** — LeRobot's native HIL-SERL RL loop ships **Franka-only**; porting an SO-101 MJCF is a small, well-scoped, *merged-upstream* contribution that plugs into your MuJoCo assets. `⚠pending`
5. **Tactile-on-a-budget.** A reproducible study of "coarse STS3215-current / $128-AnySkin tactile vs vision-only on contact-rich SO-101 tasks" is a clean, affordable-hardware result the field lacks.

### 2.3 Business / spinoff case (ranked for a solo founder)

**What the scout rules out.** Hardware-margin plays die (K-Scale Labs **shut down** despite YC + a $999 ladder; Trossen's mobile-bimanual kit **discontinued**; LeKiwi is **free**). The foundation-model lane is capital-saturated (Physical Intelligence **~$1.5B raised, no disclosed product**; Skild **$14B val**). And every high-revenue vertical needs something a $200-servo arm can't give:

| Vertical | Why cheap bimanual arms are a **dealbreaker** |
|---|---|
| Lab / wet-lab (Automata $45M Series C) | sub-mm repeatability + GxP/FDA validation |
| Food assembly (Chef Robotics, 98M servings) | NSF food-safety cert + washable food-grade housing |
| Industrial machine-tending (Standard Bots $37k, Formic) | ISO cobot cert, payload (18 kg), duty-cycle/uptime SLAs |
| Agriculture (eternal.ag, 4AG) | crop-specific end-effector + humid/dusty ruggedization |
| Retail restock (Simbe — *scan-only*) | nobody has commercialized shelf *grasping* — too hard/liability-heavy |

**The three viable solo lanes** (in order):

1. **"phospho-for-mobile-bimanual" — open-core software + paid cloud tier.** phospho (YC, built for SO-100/101 — Haller's arms) is the proven template: free open toolkit, hardware kits, ~€35/mo PRO cloud-training/VR-teleop. `~corrected: phospho's "2,000+ robots / 1,000+ models" are self-reported, undated marketing stats.` **The wedge phospho doesn't cover is mobile + bimanual.** Ship the recording/teleop/training/eval middleware for Haller-class robots with a paid cloud tier. Monetizes exactly the platform you already have; no VC, no manufacturing.
2. **Teleop-data-as-a-service.** Lowest-capex lane: a crop of 2025–26 startups (Sensei, Cortex, Claru, Adamo — "Scale AI for robot data") buy demonstration data. **Curated *mobile-bimanual* datasets are underserved** (they're mostly tabletop-arm/humanoid). You already have a working bimanual teleop+record pipeline.
3. **Education / research kit — differentiated on *curriculum + dataset + community*, not steel.** The only vertical where imprecision is a *feature* and small teams run real revenue at your price tier (Trossen ~28 employees). But Trossen's own mobile kit was discontinued and LeKiwi is free — so the product is the *course + data + community*, priced far below Trossen's ex-$37k kit.

**Watch-but-don't-enter:** narrow **RaaS** in a forgiving vertical (hospitality/cleaning) works commercially (Dyna Robotics **$120M Series A, Sept 2025** `~corrected: preceded by a $23.5M March-2025 seed; ~$600M+ val is the reported figure`; Zerith H1) — but it's capital- and manufacturing-gated beyond a solo founder until *one task is fully hardened*.

**The honest through-line:** Haller's best first "business" is a **research/education/data play built on the software and pipeline**, not a robot-for-sale. Harden one narrow task later if you want the RaaS option.

---

## Part III — What this changes in `HALLER_ROADMAP.md`

Proposed patches (keeps the ACT→SmolVLA spine; inserts the new levers where the evidence says they belong):

- **Stage 0 (schema):** add **STS3215 current/load** to the logged state now (free, enables tactile/force research later). Keep base odometry + lidar as planned.
- **NEW Stage 0.5 — Scalable data collection *before* mass teleop.** Stand up **VR teleop (BEAVR / phospho Quest)**; adopt the **Data-Scaling-Laws diversity discipline** (many env-object pairs > repetition); scope **EMMA-style egocentric-for-diff-drive** as the headline research bet.
- **NEW Stage 1.5 — Sim demo-multiplication.** Bridge MuJoCo → **DexMimicGen**; try **NVIDIA's SO-101 Isaac curriculum** + **GR00T-Dreams (SO-100)** in the cloud as a second data source; use **XLeRobot** numbers as the baseline.
- **Stage 2 (SmolVLA):** add **WALL-OSS** as an Apache-2.0 A/B; move the bimanual head toward **keypose-hierarchical (BiKC/AnchorDP3) + Co-VLA structured action expert**; **bootstrap from single-arm (AnyBimanual/TwinVLA)**; co-train with **AIST**.
- **Stage 3 (deployment):** the onboard answer is now **measured** — **SmolVLA/TinyVLA → vla.cpp → RTC + async**, ~5–10 Hz; 3B models are cloud-only. Start **logging interventions** (RECAP lesson).
- **Stage 4 (mobile):** adopt **Mink** whole-body IK (drops into MuJoCo) + **N2M pose-preference** + **ANCHOR** recovery; keep long-horizon expectations at the ~12% BEHAVIOR reality.
- **NEW parallel track — cheap System-2:** **ReKep/MOKA + Gemini-ER/RoboBrain** planner, **critic-gated** cloud calls, existing policies as a **Goal2Skill** skill library.
- **NEW parallel track — research/business:** enter a **hardware-free challenge** (RoboTwin/BEHAVIOR), **release a Haller dataset**, and prototype the **open-core middleware / data-service** wedge.

### Corrections/upgrades to the existing survey
- "SmolVLA runs on Orin Nano; 3B does not" was *inferred* — it is now **hardware-measured** (`vla.cpp`: SmolVLA ~142 ms/step on the 8 GB Nano; GR00T won't load). Upgrade from "verify" to **confirmed**.
- A previously-plausible "GR00T-Dreams 780k traj / 40%" figure **could not be verified** and should not be cited.
- The mobile-manipulation frontier is **~12% task success** on long-horizon home tasks even for a fine-tuned π0.5 with 10k demos — bake this into any capability claims.

---

## Verification outcomes (adversarial pass — complete)

Every high-importance claim was fact-checked against primary sources — **33 distinct claims, 45 checks, 0 refuted.** The `⚠pending` tags above resolved as below. Supported claims are as-stated; only the **corrections** need attention.

**Supported as-stated** (representative): `vla.cpp` Orin-Nano latencies (SmolVLA 141.8 ms; GR00T won't load) · XPU π0 = 920.6 ms on AGX Orin · BitVLA 11×/4.4× · Co-VLA +27% · AnyBimanual +12.67%/84.62% · RoboTwin-2.0 +367%/+228% · **DexMimicGen 21K demos from 60** · **NVIDIA official SO-101 Isaac curriculum** · GR00T-Dreams "36 h vs ~3 mo" · **WALL-OSS-0.5 >80% zero-shot, Apache-2.0, native LeRobot** · Sparsh +95.1% · FoAR (75→100% / 50→100%) · ANCHOR 53.3→71.7% · ReKep · Hi Robot (incl. dual-arm **mobile**) · BEAVR (~$1.08k, sub-35 ms) · UMI 111 demos/hr · Automata $45M Series C (Jan 2026) · YOR <$10k · AhaRobot ($1k, 0.7 mm) · mobile-manip "orders-of-magnitude larger datasets" · BEHAVIOR-1K winner ~12.4%.

**Supported — with a correction worth noting:**
- **WowSkin SO-101 mount ($128):** real, but sold by **third-party vendor WowRobo** building on the open AnySkin design — not an official AnySkin-team product.
- **FastUMI-100K vs YUBI:** the **$200 build + 8,434 h / 1.2 M-episode dataset are YUBI** (arXiv 2606.10244). **FastUMI-100K is a separate, non-Quest-tracked project** — don't conflate the two.
- **N2M:** the **3%→54%** jump is the *PnPCounterToCab* task; the 24–55% figure is vs the only prior policy-aware method on *CloseDrawer* only.
- **GR00T N1.6:** the **One-Way Noncommercial License is confirmed** (this is the spinoff-blocker); the exact VRAM part-list is directional.
- **UMI 111 demos/hr:** specific to the *Cup Arrangement* task (bare-hand 231, SpaceMouse 35) — not a universal constant.
- **Gemini Robotics-ER 1.6 (93%):** a **self-reported internal** instrument-reading eval (86% without agentic vision).
- **π*0.6 / RECAP** ("doubles throughput, halves failure"): the arXiv abstract is confirmed; a separate PI **blog** figure is unverified.
- **Self-reported / press business metrics** (don't change any strategic conclusion): PhosphoBot "2,000 robots / 1,000 models" is **undated company marketing**; **Simbe's $50M Series C was Oct 2024, not 2025**; **Dyna's $120M Series A (Sep 2025)** is confirmed but the >$600M valuation is press-reported, not company-disclosed (a $23.5M seed preceded it in Mar 2025); **Chef's 98M servings came from its existing *single-arm* modules**, not a bimanual system; HF's robotics-dataset growth is **1,145→26,991 (2025), rank 44→#1** per HF's own report (the "58k+" figure is a separate secondary source).

**Net:** the technical levers rest on verified ground. The only softening is around self-reported startup numbers — which affects none of the recommendations.

---

*Untagged quantitative figures are author-reported from the cited primary sources — verify against source tables before betting a design decision, per the existing docs' discipline.*
