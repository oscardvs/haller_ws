# Haller — Generalist Manipulation Roadmap (v2)

**Author of plan:** research synthesis for Oscar Devos / Haller
**Scope:** choose a recommended path and lay out a concrete, staged, detailed route from today's `haller_ws` (teleop + calibration + dataset pipeline + MuJoCo sim, all wired) to a **generalist, language-conditioned, multi-task policy** on two SO-101 arms + a differential-drive mobile base — and a **research + spinoff-business** angle alongside it.
**Companion docs:**
- `research/so101_sota_recommendation.md` — the 10-section VLA/paradigm survey.
- `research/haller_frontier_scout.md` — **verified** frontier scout (8 new levers) + research/business case. *(33 claims adversarially fact-checked, 0 refuted; `✓` below points there.)*
- `research/policy_architecture_comparison.csv` (15 architectures) · `research/paradigm_landscape.csv` (6 paradigms).

---

## 0. The recommendation, in one paragraph (revised)

Build a **staged capability ladder**, not a single model — but v2 makes **two structural changes** the frontier scout backs with verified evidence:

1. **Data collection + sim demo-multiplication move *ahead* of policy training.** The bottleneck is demonstrations, and for a solo builder the win is not "teleop more" — it's (a) collecting *diverse* data cheaply (VR teleop, the diversity-over-repetition law) and (b) multiplying a handful of real demos into thousands *in the MuJoCo sim you already run* (DexMimicGen: **21K bimanual demos from 60** ✓). This is the single change most likely to determine whether Haller succeeds.
2. **The onboard deployment question is now *measured*, not guessed.** On Haller's actual 8 GB Jetson Orin Nano, `vla.cpp` measured **SmolVLA at 141.8 ms/step (~7 Hz)** while **GR00T N1.6/1.7 won't even load** ✓. So the deployable generalist is a **SmolVLA-class model + `vla.cpp` runtime + Real-Time Chunking** (already shipped in LeRobot); every 3B model (GR00T / π0.5 / WALL-OSS) is **cloud-only** for this robot.

The spine is unchanged: **BC → VLA**, because it teaches the fundamentals hands-on, deploys onboard, and is fully supported by LeRobot. Concretely: **data flywheel (Stage 0–1.5) → ACT baseline → SmolVLA generalist (+ WALL-OSS A/B, + bimanual coordination head) → measured onboard deployment → manipulation-aware mobile base → advanced paradigms.** Three **parallel tracks** run alongside from early on: a *cheap System-2* planner, *tactile-on-a-budget*, and a *research/business* track.

**Why SmolVLA over bigger models (load-bearing, now verified):** bigger ≠ better in manipulation (7B OpenVLA beats 55B RT-2-X; VLA quality is uncorrelated with backbone-VLM score), **and** the 3B models physically do not fit the Orin Nano's 8 GB (`vla.cpp` ✓). SmolVLA is the model that is simultaneously a real generalist and actually deployable.

---

## Capability-ladder overview

| Stage | Lever | Headline tool / model | Verified anchor | Exit signal |
| - | - | - | - | - |
| **0** | Data schema | `LeRobotDataset` (arms + base + lidar + **STS3215 current** + language + coordination-class) | — | 1 clean multi-stream episode on the Hub |
| **0.5** | *Scale data cheaply* | VR teleop (BEAVR / phospho) · diversity law · EMMA (egocentric→base) | UMI 111 demos/hr ✓; BEAVR ~$1.08k ✓ | ≥20 env-object pairs recorded, not repeats |
| **1** | BC fundamentals | **ACT** (chunking, eval) | XLeRobot: ACT **36 ms / 27.8 Hz** onboard, 98.7% grasp | ACT works onboard on 1 task |
| **1.5** | *Multiply demos in sim* | **DexMimicGen** on MuJoCo · NVIDIA SO-101 Isaac curriculum | DexMimicGen 21K-from-60 ✓; RoboTwin-2.0 +367% ✓ | 1 task's dataset ≥10× via sim |
| **2** | Generalist VLA | **SmolVLA** (+ **WALL-OSS** A/B) + keypose/Co-VLA bimanual head | WALL-OSS-0.5 >80% zero-shot, Apache-2.0 ✓; Co-VLA +27% ✓ | ≥3 tasks by language |
| **3** | Onboard real-time | **`vla.cpp` + RTC + async** | SmolVLA 141.8 ms on 8 GB Nano ✓; RTC in LeRobot | ~5–10 Hz language policy onboard |
| **4** | Mobile (manip-aware) | **Mink** whole-body IK + **N2M** pose-preference + **ANCHOR** | N2M 3%→54% ✓; ANCHOR 53→72% ✓ | robot navigates→manipulates; whole-body data logged |
| **5** | Advanced paradigms | RL-from-corrections (RECAP-style) · world models · JEPA | RECAP "≈2× throughput" ✓ (abstract) | policy beats its own demos |
| **R/B** | *Research + business* | Haller dataset release · RoboTwin/BEHAVIOR challenges · open-core wedge | mobile-manip scarcity ✓; BEHAVIOR winner ~12% ✓ | a citable output + a validated business wedge |

---

## Decision gates (resolve these — they change specific steps, not the overall shape)

| # | Open question | Why it matters | Default assumed here |
| - | -------------- | -------------- | ------------------------------- |
| G1 | **RESOLVED — 3-wheel base:** 2 driven front wheels + rear caster (differential drive). | Base action stays `(v, ω)` (2 dims) — schema-safe. | **3-wheel diff-drive** confirmed; stale README "4-wheel" wording corrected. |
| G2 | **Untethered, or is cloud inference acceptable?** | Decides whether the *deployed* generalist must fit the Orin Nano (→ SmolVLA) or can run in the cloud (→ π0.5/GR00T/WALL-OSS). | Assume **untethered** → onboard SmolVLA is the deploy target; cloud is for training/study. |
| G3 | **First 2–3 concrete tasks?** e.g. pick-place cube, bimanual handover, put-in-drawer. | Drives whether tasks are truly bimanual and how much data per task. | **1 single-arm pick-place + 1 bimanual handover** to start. |
| G4 | **Commercial intent for Haller?** | Some strong models are non-commercial (GR00T ✓license-verified as non-commercial; Galaxea/AgiBot NC). | Prefer **permissive (Apache-2.0)** → SmolVLA + **WALL-OSS** are spinoff-safe. |
| **G5** | **Primary data-collection mode:** current webcam-pose teleop, VR teleop (BEAVR/phospho), a UMI-style handheld, or EMMA egocentric? | Sets your throughput and whether the base is trained from teleop or human video. | **VR teleop for arms now** (zero build); **EMMA egocentric-for-base as the research bet**; skip a UMI SO-101 gripper (needs mount redesign). |
| **G6** | **Sim multiplier:** stay in MuJoCo (DexMimicGen) or add Isaac (NVIDIA SO-101 curriculum) in the cloud? | Determines infra effort vs. turnkey-ness. | **DexMimicGen on existing MuJoCo** primary; **NVIDIA Isaac curriculum** as a cloud second source. |
| **G7** | **Onboard runtime:** `vla.cpp`, NVIDIA TensorRT, or plain PyTorch? | Sets the achievable onboard Hz and which models fit 8 GB. | **`vla.cpp` + RTC** — the only stack with measured 8 GB-Nano numbers. |

---

## Stage 0 — Foundations & data schema (Week 0–1) · *do not skip the schema step*

**Goal:** a clean, synchronized dataset schema and one recorded episode, end-to-end. The schema is the single most consequential design decision in the whole project — everything downstream re-collects data if you get it wrong.

1. **Confirm the environment.** You already have the `lerobot` conda env, ROS 2 Jazzy base, HMI teleop (60 Hz), calibration wizard, camera streams, `record_dataset.sh`, and the MuJoCo sim trio. Verify each still runs: teleop both arms, view all 3 cameras (D455 + 2 wrist), run the calibration wizard once.
2. **Freeze the `LeRobotDataset` schema.** Log, per timestep, synchronized:
   - `observation.images.top` (D455 RGB; optionally depth), `observation.images.left_wrist`, `observation.images.right_wrist`.
   - `observation.state` = `[left arm 6 joints + gripper, right arm 6 joints + gripper]` (14 dims for 2× SO-101).
   - `action` = the same 14 dims (teleop target).
   - **`observation.base`** = wheel odometry / commanded velocity `(v, ω)`; **`observation.lidar`** = RPLIDAR scan. *Add these now even if the first policies ignore them* — retrofitting base state later means re-collecting (resolve G1 for exact dims).
   - **`observation.effort` (NEW):** the **STS3215 `Present_Current` / `Present_Load`** registers per joint (already on your servo bus, ~6.5 mA/count, exposed by the Feetech SDK). Free coarse force/contact signal — logging it now unlocks the tactile track (Stage 5b) and grasp-force limiting later at **zero hardware cost**.
   - `task` = natural-language instruction string (enables language conditioning).
   - **`task.coordination_class` (NEW, metadata):** label each task per the Krebs & Asfour taxonomy — *uncoordinated / loosely-coupled / tightly-coupled (symmetric vs dominant-asymmetric)*. Lets you build a deliberate bimanual curriculum in Stage 2 instead of one undifferentiated "bimanual" bucket.
3. **Record ONE pilot episode** of a trivial task (pick a cube, drop it) with the full schema. Inspect with LeRobot's dataset visualizer; confirm all streams' timestamps align.
4. **Push to the Hugging Face Hub** (per `docs/setup/dataset-collection.md`). This is your data backbone.

**Exit criteria:** one valid episode — all cameras + base + lidar + **effort** + language — visualized and on the Hub.

---

## Stage 0.5 — Scale data collection *before* mass teleop (Week 1–3) · **NEW, highest-leverage**

**Goal:** set up a data *flywheel* that does not depend on you joysticking thousands of episodes. This stage is new in v2 because the verified evidence says data *strategy* — not model choice — is what caps a solo project.

1. **Adopt the diversity-over-repetition law (do this regardless of tool).** Zero-shot generalization scales as a **power law in the number of distinct environment-object *pairs*, not raw demo count** (Data Scaling Laws, ICLR'25 ✓). Practically: **~32 env-object pairs × ~50 demos** generalizes far better than 1,600 demos of one setup. Plan collection as a *diversity budget*, and track pairs, not episode count.
2. **Turn on VR teleop for the arms (zero hardware build).** Two options, both dropping straight into your LeRobot/HF pipeline:
   - **BEAVR** (open, self-hostable, no subscription; Quest 3S hand-tracking, native LeRobot schema, whole rig ≈ **$1.08k** ✓, sub-35 ms, ACT trained on its data hit 100% on a pickup task) — preferred for an open stack.
   - **phospho Quest app** (subscription-gated VR control; also SO-100/101-native) — the fast A/B.
   Compare either against your current webcam-pose teleop for throughput and data quality.
3. **Prototype EMMA-style egocentric→base data collection (the research bet, G5).** EMMA retargets **egocentric human walking trajectories (Aria glasses) into feasible differential-drive base paths**, co-trained with a little static-arm data — *no mobile teleop at all*, and it targets exactly Haller's diff-drive base. Treat as frontier-unstable research to reimplement, but it is the only method that removes teleop from the **base** and is genuinely publishable (see track R/B).
4. **For solo bimanual data, use MonoDuo's pattern.** Drive one arm at a time (a human handles the other, swap roles), synthesize the missing arm via hand-pose + inpainting — solves the "I can't teleoperate both SO-101s at once" constraint.
5. **(Deferred) UMI-style handheld gripper.** Powerful (UMI: **111 demos/hr vs 35 via SpaceMouse** ✓, in-the-wild, no robot needed) — but SO-101's STS3215 parallel-jaw geometry doesn't match UMI's >85 mm-stroke assumption, so it needs a real mount redesign. Skip until Stage 2+ unless you want that CAD work. FastUMI-100K / **YUBI** ($200 build) are also **ready-made open LeRobot datasets** for cross-embodiment pretraining.

**Exit criteria:** VR teleop recording into the schema; a *diversity plan* (≥20 env-object pairs targeted); an EMMA feasibility note (can we retarget human→diff-drive base at all?).

---

## Stage 1 — ACT baseline (Week 2–4) · *learn the fundamentals, get a deployable baseline*

**Goal:** train and deploy the ALOHA-lineage bimanual BC method; understand action chunking end-to-end. Unchanged in intent, now with a **measured baseline to hit**.

1. **Collect a single-task dataset** (~50 demos of one task, e.g. single-arm pick-and-place) via VR teleop from Stage 0.5, kept varied (object position, lighting).
2. **Train ACT** with LeRobot (`lerobot-train ... --policy.type=act`). Runs on any of your GPUs (even 12 GB). Log to W&B / TensorBoard.
3. **Evaluate on the robot** via LeRobot's control loop on the Orin Nano. Measure success over ~20 trials.
   - **Baseline to beat (verified twin):** the near-identical **XLeRobot** (dual SO-101 + Orin Nano) reports **ACT at 36 ms / 27.8 Hz onboard** and **98.7% grasp over 75 trials** (`~` author-reported), with no thermal throttling after 30 min. Use these as your sanity-check targets; if you're far off, suspect the pipeline, not the method.
4. **Study what you built:** action chunking, temporal ensembling, obs/action normalization, the eval harness. Write notes. This is your reference baseline for everything later.

**Exit criteria:** ACT achieves non-trivial success on one task, running onboard; you can explain chunking + the eval loop.

---

## Stage 1.5 — Multiply demos in sim (Week 3–6, overlaps Stage 1) · **NEW**

**Goal:** turn a handful of real demos into thousands of training trajectories using the MuJoCo sim you already have — the cheapest large lever in the whole plan.

1. **Bridge MuJoCo → DexMimicGen (primary, G6).** DexMimicGen (NVlabs, open) is **purpose-built for bimanual** manipulation and runs on **robosuite (MuJoCo)** + **BiGym (mobile bimanual)** — it generated **21K demos from 60 human demos** across 9 tasks ✓. Port an SO-101 MJCF into its pipeline, seed it with your Stage-1 teleop demos, and generate a large synthetic set. (Predecessor MimicGen: 50K from <200 demos.)
2. **Run the NVIDIA official SO-101 Isaac curriculum in the cloud (second source).** "Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac" ✓ is the **only toolchain documented for your exact arm** — Isaac Lab Mimic demo-multiplication, Cosmos/GR00T-Dreams synthetic trajectories ("36 h vs ~3 months" of manual collection ✓), actuator-gap closure, and an **official script that converts to LeRobot format**. Heavier than MuJoCo → run on RunPod as a parallel data source, not a sim replacement.
3. **Use RoboTwin-2.0 for domain-randomized bimanual data (MIT license).** A VLA trained on **synthetic + only 10 real demos got +367%** over 10-real-only; **synthetic-only +228%** ✓. Strong evidence that synthetic co-training slashes your real-demo requirement.
4. **GPU-accelerate rollouts with MuJoCo Playground / MJX** (Apache-2.0) — same physics family, trains policies / generates rollouts in minutes on one GPU on RunPod.
5. **(Optional) Cut scene-building with Digital Cousins / ACDC** — generate approximate sim scenes from single phone photos of your real workspace instead of hand-modeling every object.
6. **Contribute the missing SO-101 `gym_hil` env (feeds track R/B).** LeRobot's native HIL-SERL RL loop ships **Franka-only**; porting an SO-101 MJCF is a small, well-scoped, *upstream-mergeable* contribution that plugs into your MuJoCo assets and sets up Stage 5 RL.

**Exit criteria:** at least one task's dataset multiplied ≥10× via sim; a measured comparison of a policy trained on (real-only) vs (real + synthetic).

---

## Stage 2 — SmolVLA generalist + bimanual coordination (Week 6–12) · *the recommended v1 endpoint*

**Goal:** a single **language-conditioned, multi-task** policy on your two arms, with **bimanual-specific** coordination, deployable on the Orin Nano.

1. **Collect / assemble a multi-task, language-annotated dataset.** 3–5 tasks (pick-place, stack, handover, put-in-container), **each with a `task` string** and a `coordination_class` label, ~50–100 demos/task — now heavily padded by Stage-1.5 synthetic data and Stage-0.5 diversity.
2. **Fine-tune SmolVLA** from the community checkpoint (`--policy.type=smolvla`): freeze the SigLIP + SmolLM2 backbone, train the flow-matching action expert. Fits your GPUs at full fine-tune.
3. **A/B against WALL-OSS (new, spinoff-safe).** **WALL-OSS** (X-Square, ~4B) is now a **first-class LeRobot policy (`--policy.type=wall_x`), Apache-2.0**, trainable via the standard CLI with zero glue code; WALL-OSS-0.5 reports **>80% zero-shot** on a 17-task real eval ✓. It's 4B → **cloud/RunPod training + inference** (not onboard), so use it as a stronger-generalist study/teacher, with SmolVLA remaining the *deployed* student.
4. **Add a bimanual-coordination head (raise the ceiling on two-hand tasks).** In order of leverage:
   - **Keypose-hierarchical control** (BiKC / **AnchorDP3**): predict sparse per-arm keyposes (pre-grasp→grasp→place) + a fast low-level generator instead of dense per-timestep chunking — *cheaper at inference* (matters on the Nano) and won the CVPR'25 RoboTwin Dual-Arm Challenge.
   - **Co-VLA structured action expert**: replace the monolithic head with a **shared coordination latent + per-arm residuals** + a coordination-aware loss — a small add on a SmolVLA/π0 backbone, **+27% on tightly-coordinated tasks** ✓.
   - **Bootstrap from single-arm data** (**AnyBimanual** ✓ / TwinVLA): you have far more single-arm data/checkpoints than bimanual — transfer a unimanual policy into a bimanual one with few bimanual demos.
5. **Co-train with the AIST bimanual dataset (CC-BY-4.0):** 10,705 real episodes / 119 tasks on an ALOHA-lineage rig mechanically close to SO-101 (needs ALOHA→Feetech action remap).
6. **Evaluate multi-task, language-conditioned.** Test each task by *changing only the instruction string*; measure per-task success + generalization to unseen object positions, sliced by coordination-class.
7. **Iterate on data, not architecture.** When a task fails, add targeted demos (or synthetic variants). The data-flywheel discipline is the highest-leverage habit.

**Exit criteria:** one policy performs ≥3 tasks selected by language, with a measurable win from the bimanual head on tightly-coordinated tasks. **This is a legitimate v1 generalist for Haller.**

---

## Stage 3 — Onboard real-time deployment (Week 10–14, overlaps Stage 2) · **now first-class, and measured**

**Goal:** run the language-conditioned policy **onboard the Orin Nano in real time**. v1 promoted this from a footnote to a stage because the numbers are now known.

1. **Adopt the `vla.cpp` runtime (G7).** The llama.cpp-derived C++ VLA server has the **only measured 8 GB-Orin-Nano numbers**: **SmolVLA 141.8 ms/step (~7 Hz)**, BitVLA 356 ms, and **GR00T N1.6/1.7 do not fit** ✓. On AGX Orin, faster runtime alone *doubled* ALOHA success (87.5% vs 40%) — runtime engineering changes task outcomes, not just latency.
2. **Wrap it in Real-Time Chunking + async inference (shipped in LeRobot, no retraining).** RTC generates the next action chunk *while executing the current one* with soft-mask inpainting (`RTCConfig(execution_horizon=10, max_guidance_weight=10.0, prefix_attention_schedule=EXP)`); async inference hides idle wait. Together they absorb the ~140–350 ms/step the Nano actually produces without visible jerkiness. Handles >300 ms delays.
3. **If memory-bound, try BitVLA (1-bit, open CC-BY-4.0):** **11× less memory, 4.4× lower latency** vs OpenVLA-OFT at matched success ✓ — attacks the 8 GB ceiling. (Caveat: on the Nano *today* its kernel is slower than SmolVLA — a memory win, not yet a speed win.) Also benchmark **TinyVLA / MiniVLA / EdgeVLA** (sub-1B) — latency is set by the diffusion head, not param count (SJTU XPU study ✓: ACT single-pass is fastest on every platform).
4. **Keep 3B models in the cloud.** Run WALL-OSS / π0.5 / GR00T on RunPod, stream actions over the network *only* when tethered operation is acceptable (G2), or use them as **distillation teachers** for the onboard SmolVLA student.
5. **Deployment split, decided:** **onboard** = SmolVLA(/TinyVLA) → `vla.cpp` → RTC, target **~5–10 Hz**; **cloud** = anything ≥3B, for study / teacher / tethered use.

**Exit criteria:** SmolVLA-class policy running onboard at ~5–10 Hz with RTC, controlling ≥1 real task without a network dependency.

---

## Stage 4 — Manipulation-aware mobile base (Week 12–18, overlaps Stage 3) · *the Haller-specific step*

**Goal:** coordinate base + arms — and specifically fix the "arrived at the goal but the arm can't reach" failure. Start decoupled, move toward unified.

1. **Whole-body IK in sim with Mink (do this first — fastest concrete step).** **Mink** (Apache-2.0, MuJoCo differential-IK by Kevin Zakka) drops straight into your existing MuJoCo sim to build a **base + dual-arm whole-body IK controller**, offloading low-level base/arm coordination from the learned policy (the HoMeR pattern: 79% at 20 demos).
2. **Decoupled nav→manip baseline (robust first system).** Use RPLIDAR + Nav2 (already in `haller_navigation`) for SLAM/costmap, drive to a task pose, then run the Stage-2/3 arm policy. Your lidar earns its place here.
3. **Add manipulation-aware base placement (the key upgrade).** **N2M / N2M2** (open) learns a **preferred base pose for the downstream manipulation from egocentric camera only — no SLAM map, no holonomic base** — a perfect fit for diff-drive + 2D lidar + a small arm workspace (**3%→54%** on PnPCounterToCab ✓). This is the fix for "drive-then-grasp."
4. **Add closed-loop recovery with ANCHOR's pattern.** ANCHOR targets exactly "arrived but inoperable" with *local* recovery-replanning (cheap enough for the Nano): **53.3→71.7% success, 71.4% recovery** under disturbance ✓.
5. **Log whole-body data.** Now that the base moves during tasks, your Stage-0 schema (base odometry + lidar) pays off — record mobile-manipulation demos with base + arms moving together. (Multiply them with **MoMaGen**-style reachability-constrained augmentation in MuJoCo.)
6. **(Dynamic scenes) DynaMem** for "fetch object X wherever it is now" — an online spatio-semantic memory (70% vs 30% OK-Robot on moving objects); its VLM queries need cloud offload.
7. **Toward unified whole-body (frontier).** Fine-tune a π0.5-class model that outputs base + arm actions jointly on the whole-body dataset — the true generalist endpoint (cloud-side; Galaxea G0's R1-Lite is the closest open architecture reference).
8. **Calibrate expectations.** Even a fine-tuned π0.5 with 10k demos hit only **~12.4% task success** on 50 long-horizon BEHAVIOR-1K home tasks ✓ — robust long-horizon home autonomy is *not* solved by anyone. Aim for narrow, hardened tasks, not open-ended home tidying.

**Exit criteria:** robot navigates to an object and manipulates it (with learned base-pose preference, not naive drive-then-grasp); whole-body dataset recorded for unified-policy training.

---

## Stage 5 — Advanced paradigms (parallel research rungs, Month 4+)

Pick based on what's limiting you; each maps to a paper to implement.

**5a — Get better than your demos (RL / self-improvement).**
- **Start logging teleop *interventions/corrections now*** (the RECAP lesson — π*0.6 "more than doubles throughput, roughly halves failures" ✓, learning from real deployment corrections). Cheap to add; seeds all later RL.
- Then **SimpleVLA-RL / ReWiND / ConRFT**-style RL fine-tuning on top of the BC/VLA policy (ReWiND: ~5× in ~1 h online, author-reported). Pairs with the SO-101 `gym_hil` env from Stage 1.5.

**5b — Tactile on a budget (parallel, cheap first step).**
- **Zero-cost:** you're already logging STS3215 `Present_Current` (Stage 0) — use it for stall/contact detection and grasp-force limiting; test whether coarse force helps.
- **If it helps:** retrofit **one** gripper with **WowSkin** (AnySkin, **$128**, purpose-made SO-101 mount from 3rd-party vendor WowRobo ✓) and encode via **Sparsh** (Meta's open frozen touch encoder, +95.1% over task-specific training at fewer labels ✓) to avoid hand-labeling. Then fuse into the VLA (the new TacVLA/ForceVLA wave). Force feedback flips contact-rich near-failures to near-100% (FoAR: wiping 75→100% ✓).
- This is also a strong **research differentiator** (tactile-on-cheap-hardware is under-studied) → track R/B.

**5c — Learn from human/internet video** (LAPA-style latent-action pretraining; EMMA for mobile — see Stage 0.5). Cheapest route to generalization given scarce teleop.

**5d — World models & JEPA** (DreMa / diffusion WM for safe RL-in-imagination; V-JEPA 2-AC / DINO-WM for predict-then-plan; the 2026 World-Action-Model fusion — VLA-JEPA, RISE, WoVR). The frontier is converging here; study V-JEPA 2 (open weights, monocular, extreme data efficiency) as a parallel track. Full detail in `so101_sota_recommendation.md` §8–10.

---

## Track R/B — Research & business (parallel from day one)

Haller's embodiment is **validated but crowded** — at least 8 near-siblings exist (XLeRobot, AhaRobot, YOR, LeKiwi, Galaxea R1 Lite, Sunday Memo, Zerith H1, the *discontinued* Trossen Mobile ALOHA). **The hardware is not the moat.** Differentiate on software, data, curriculum, or a narrow niche.

**Research (educational + publishable, GPU-only, ~6–12 months):**
1. **Release a Haller mobile-bimanual LeRobot dataset.** Mobile-manip data scarcity is *quantified* ("datasets orders of magnitude larger than stationary" ✓); a well-annotated release is itself a **citable contribution** (à la AIRoA MoMa).
2. **EMMA-style egocentric-for-diff-drive** (Stage 0.5) — genuinely novel *and* Haller-shaped; the highest-upside research bet.
3. **Enter hardware-free competition tracks** (RunPod-only, citable in 2026): **RoboTwin Dual-Arm Challenge** (MIT), **BEHAVIOR-1K Challenge** (OmniGibson), **AgiBot World Challenge** (control real robots via API). Natural on-ramp to an RSS "Mobile Manipulation" workshop paper.
4. **Contribute the SO-101 `gym_hil` env upstream** (Stage 1.5) — small, merged, visible.
5. **Tactile-on-a-budget study** (Stage 5b) — a clean affordable-hardware result the field lacks.

**Business (spinoff, ranked for a solo founder — hardware-margin and foundation-model lanes are dead ends: K-Scale *shut down*, Trossen Mobile ALOHA *discontinued*, LeKiwi is *free*, PI raised ~$1.5B with no product):**
1. **"phospho-for-mobile-bimanual" — open-core middleware + paid cloud tier.** phospho proved the template on SO-100/101 (Haller's arms); the uncovered wedge is **mobile + bimanual**. Ship the recording/teleop/training/eval middleware for Haller-class robots with a paid cloud-training/eval tier. Monetizes exactly the platform you already have; no VC, no manufacturing.
2. **Teleop-data-as-a-service.** Lowest capex: sell curated **mobile-bimanual datasets** (underserved vs tabletop/humanoid) into the Sensei/Cortex-style marketplaces.
3. **Education / research kit** differentiated on **curriculum + dataset + community**, not steel — the one vertical where imprecision is a feature and small teams run real revenue.
- **Watch-but-don't-enter:** narrow **RaaS** in a forgiving vertical (hospitality/cleaning — Dyna, Zerith work commercially) but capital/manufacturing-gated until one task is fully hardened.
- **Dealbreakers for cheap arms (don't chase):** lab automation (sub-mm + GxP), food (NSF cert), industrial (ISO cert + SLA + payload), agriculture (ruggedization).

---

## The through-line (what to internalize)

1. **Data > architecture — and now with concrete levers.** The bottleneck is demonstrations; the wins are *diversity* (power-law law), *cheap collection* (VR/egocentric), and *sim multiplication* (DexMimicGen on your MuJoCo). Build the flywheel *before* chasing models. Log base + lidar + **effort** from episode 1.
2. **Deployment is measured, not guessed.** SmolVLA-class + `vla.cpp` + RTC ≈ 5–10 Hz onboard; 3B models don't fit 8 GB and are cloud-only. Decide by hardware reality, not model prestige.
3. **Bigger ≠ better, and bigger doesn't fit your robot.**
4. **Stage, don't leap.** Each rung teaches and de-risks the next: data flywheel → ACT → SmolVLA(+bimanual) → onboard → mobile → advanced.
5. **The frontier is convergence** (BC + world models + video + RL → World-Action Models). Understanding each paradigm in isolation is how you get there deliberately.
6. **The moat is software/data/niche, not steel.** Research and a spinoff both run through the *pipeline* you already own, not a robot-for-sale.

---

*Provenance: quantitative claims marked `✓` were adversarially fact-checked against primary sources in `research/haller_frontier_scout.md` (33 claims, 0 refuted; 14 supported-with-a-correction — see that doc's "Verification outcomes"). Untagged figures (success rates, GPU-hours, speedups, parameter counts, funding) are author-reported from the cited papers/announcements — treat as directional and verify against source tables before betting a design decision.*
