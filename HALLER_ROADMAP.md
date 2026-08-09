# Haller — Generalist Manipulation Roadmap (v3)

**Author of plan:** research synthesis for Oscar Devos / Haller
**Scope:** choose a recommended path and lay out a concrete, staged, detailed route from today's `haller_ws` (teleop + calibration + dataset pipeline + MuJoCo sim, all wired) to a **generalist, language-conditioned, multi-task policy** on **two SO-101 arms** — and a **research + spinoff-business** angle alongside it. **The mobile base is not built, so the two-arm setup is the whole near-term scope; Stage 4 is explicitly deferred (its section stays, with the trigger that reactivates it).**
**Companion docs:**
- `research/so101_sota_recommendation.md` — the 10-section VLA/paradigm survey.
- `research/haller_frontier_scout.md` — **verified** frontier scout (8 new levers) + research/business case. *(33 claims adversarially fact-checked, 0 refuted; `✓` below points there.)*
- `research/policy_architecture_comparison.csv` (15 architectures) · `research/paradigm_landscape.csv` (6 paradigms).

### What changed from v2

- **π0.5 is promoted from "cloud, study only" to a first-class track.** RunPod compute exists, so `--policy.type=pi05` becomes the **cloud generalist** — training, evaluation, and distillation *teacher*. It does **not** replace SmolVLA as the onboard deploy target. The plan is now explicitly **two models, one pipeline** (see G8).
- **Arms-only refocus.** Stage 4 (mobile base) is deferred. **`observation.base` stays in the schema** (2 floats, already recorded — deleting it forks the dataset format); **`observation.lidar` moves to "not built."**
- **G4's licensing call was wrong.** "GR00T = non-commercial" was true of **N1.5**; **GR00T N1.7 is commercially usable** under the NVIDIA Open Model License, and GR00T is no longer a separate runtime — it ships *inside* lerobot. Corrected in G4 and Stage 3.
- **Training-config decision (2026-08-09), now settled:** π0.5 is fine-tuned **FULL, not LoRA**. lerobot 0.5.1's default π0.5 PEFT targets freeze the SigLIP vision tower *and* the Gemma LLM, which is the configuration measured as catastrophic for adapting to a new embodiment (ATP 0.14 vs 0.76 full, arXiv:2607.10172), and Physical Intelligence say the same of their own model. The cost gap is ~$20–35 on RunPod, so this was never a budget question. See `scripts/runpod/finetune_pi05.sh`.
- **New unresolved licensing risk:** π0.5's weights are **not cleanly Apache-2.0** — the `lerobot/pi05_base` card declares `license: gemma`. Flagged under G4, deliberately *not* resolved.
- **The camera set is now decided: record 3 channels, not 5** — `top` / `left_wrist` / `right_wrist`, decoupled from the 5 the sim renders for the operator. Matches π0.5's pretrained slot count *and* armnet's dataset keys (Stage 0 §3).
- **A units mismatch v2 never mentioned is now the highest-risk item in Stage 0.** Haller logs **degrees**; every public SO-101 dataset logs **normalized [-100, 100]**. Decision made (keep degrees, record the calibrated ranges) — see Stage 0 §4 and G9.
- **Public 12-dim bimanual SO-101 datasets now exist and are named** (Stage 0.5 §2). They let 12-dim bimanual training be validated *before* Haller's own data exists. Stage 0.5 §3 also rules out the big names that *don't* work: **AgiBot World** (CC-BY-NC-SA-4.0, fails G4) and **DROID / BridgeData V2 / Open-X** (end-effector-pose action spaces — structurally uncombinable with a 12-dim joint vector).

---

## 0. The recommendation, in one paragraph (revised)

Build a **staged capability ladder**, not a single model. v2 made two structural changes; v3 makes a third:

1. **Data collection + sim demo-multiplication move *ahead* of policy training.** The bottleneck is demonstrations, and for a solo builder the win is not "teleop more" — it's (a) collecting *diverse* data cheaply (VR teleop, the diversity-over-repetition law) and (b) multiplying a handful of real demos into thousands *in the MuJoCo sim you already run* (DexMimicGen: **21K bimanual demos from 60** ✓). This is the single change most likely to determine whether Haller succeeds.
2. **The onboard deployment question is *measured*, not guessed.** On Haller's actual 8 GB Jetson Orin Nano, `vla.cpp` measured **SmolVLA at 141.8 ms/step (~7 Hz)** while **GR00T N1.6/1.7 won't even load** ✓. So the *deployed* generalist is a **SmolVLA-class model + `vla.cpp` runtime + Real-Time Chunking** (already shipped in LeRobot). Every 3B model stays off the robot.
3. **(v3) "Off the robot" no longer means "out of the plan."** RunPod compute exists, so **π0.5 (`--policy.type=pi05`) becomes a first-class cloud track**: it is where multi-task generalization gets trained and *measured*, and it is the **distillation teacher** for the onboard SmolVLA student. The right mental model is **two models sharing one dataset and one eval harness**, not "the real model" and "the toy." Note also that π0.5 **pads state/action to 32 dims internally and unpads the loss/output back to the dataset's real dimensionality ✓** — so Haller's **12-dim bimanual action space needs no architectural change**. Do not plan around a dimensionality problem that does not exist.

The spine is unchanged: **BC → VLA**, because it teaches the fundamentals hands-on, deploys onboard, and is fully supported by LeRobot. Concretely: **data flywheel (Stage 0–1.5) → ACT baseline → SmolVLA generalist + π0.5 cloud generalist (+ WALL-OSS A/B, + bimanual coordination head) → measured onboard deployment → *(deferred)* mobile base → advanced paradigms.** Three **parallel tracks** run alongside from early on: a *cheap System-2* planner, *tactile-on-a-budget*, and a *research/business* track.

**Why SmolVLA is still the *deploy* target (load-bearing, verified):** bigger ≠ better in manipulation (7B OpenVLA beats 55B RT-2-X; VLA quality is uncorrelated with backbone-VLM score), **and** the 3B models physically do not fit the Orin Nano's 8 GB (`vla.cpp` ✓). π0.5 being promoted to a first-class track changes *where the ceiling is explored*, not *what runs untethered on this robot*.

**And the near-term scope is two arms.** The base is not built. Everything below that assumes a moving base is marked deferred, and Stage 0's schema is written so that deferral costs nothing later.

---

## Capability-ladder overview

| Stage | Lever | Headline tool / model | Verified anchor | Exit signal |
| - | - | - | - | - |
| **0** | Data schema + **units contract** + **camera set** | `LeRobotDataset` (**3 cameras: `top`/`left_wrist`/`right_wrist`** + arms in **degrees** + `base` (2 floats, kept) + **effort (STS3215 load)** + language + coordination-class + **calibrated joint ranges**) · *no lidar — not built* | Haller logs degrees (`arm.py`, `use_degrees=True`) vs public SO-101 normalized [-100,100] ✓ | 1 clean multi-stream episode on the Hub, **ranges in metadata** |
| **0.5** | *Scale data cheaply* · **borrow data that already exists** | VR teleop (BEAVR / phospho) · diversity law · **4 public bimanual SO-101 datasets** | UMI 111 demos/hr ✓; BEAVR ~$1.08k ✓; armnet **1,219 ep / 743,964 frames** ✓ | ≥20 env-object pairs recorded, not repeats; **12-dim training validated on public data first** |
| **1** | BC fundamentals | **ACT** (chunking, eval) | XLeRobot: ACT **36 ms / 27.8 Hz** onboard, 98.7% grasp | ACT works onboard on 1 task |
| **1.5** | *Multiply demos in sim* + **auto-scored sim eval** | **DexMimicGen** on MuJoCo · scene reset + seeded DR + contact-based success predicate (landing now) · NVIDIA SO-101 Isaac curriculum | DexMimicGen 21K-from-60 ✓; RoboTwin-2.0 +367% ✓ | 1 task's dataset ≥10× via sim; **success auto-scorable in sim** |
| **2** | Generalist VLA — **two of them** | **SmolVLA** (onboard student) + **π0.5 on RunPod** (cloud generalist / teacher) + **WALL-OSS** A/B + keypose/Co-VLA bimanual head | π0.5 pads state/action to 32 dims, unpads to real dim ✓ (12-dim is fine); WALL-OSS-0.5 >80% zero-shot ✓; Co-VLA +27% ✓ | ≥3 tasks by language, **cloud vs onboard gap measured** |
| **3** | Onboard real-time + **distillation** | **`vla.cpp` + RTC + async**; π0.5 → SmolVLA distillation | SmolVLA 141.8 ms on 8 GB Nano ✓; **GR00T N1.7 commercially usable** (NVIDIA Open Model License) ✓ | ~5–10 Hz language policy onboard |
| **~~4~~** | ~~Mobile (manip-aware)~~ — **DEFERRED, base not built** | **Mink** whole-body IK + **N2M** pose-preference + **ANCHOR** | N2M 3%→54% ✓; ANCHOR 53→72% ✓ | *(reactivates when the base exists — see Stage 4)* |
| **5** | Advanced paradigms | RL-from-corrections (RECAP-style) · world models · JEPA | RECAP "≈2× throughput" ✓ (abstract) | policy beats its own demos |
| **R/B** | *Research + business* | Haller dataset release · RoboTwin/BEHAVIOR challenges · open-core wedge | mobile-manip scarcity ✓; BEHAVIOR winner ~12% ✓ | a citable output + a validated business wedge |

---

## Decision gates (resolve these — they change specific steps, not the overall shape)

| # | Open question | Why it matters | Default assumed here |
| - | -------------- | -------------- | ------------------------------- |
| G1 | **RESOLVED (on paper) — 3-wheel base:** 2 driven front wheels + rear caster (differential drive). | Base action stays `(v, ω)` (2 dims) — schema-safe, which is *why* the slot survives the Stage-4 deferral. | **3-wheel diff-drive** confirmed; stale README "4-wheel" wording corrected. **Not built** — decision is banked, not exercised. |
| G2 | **RESOLVED — both, deliberately.** Untethered *deployment*; cloud as a first-class *development* environment. | Decides what must fit the Orin Nano (→ SmolVLA) versus what may be big (→ π0.5 on RunPod). v2 answered this as an either/or; it isn't one. | **Untethered SmolVLA is the deploy target.** **π0.5 on RunPod is a first-class track** for training, eval, and distillation-teaching — not "study only." See G8. |
| G3 | **First 2–3 concrete tasks?** e.g. pick-place cube, bimanual handover, put-in-drawer. | Drives whether tasks are truly bimanual and how much data per task. **v3: tasks must be tabletop/arms-only** — nothing that presumes a base. | **1 single-arm pick-place + 1 bimanual handover** to start. |
| G4 | **Commercial intent for Haller?** | Decides which weights are usable in a spinoff. **v2 got this wrong** — see the correction below. | **Corrected:** "GR00T = non-commercial" applied to **N1.5**. **GR00T N1.7 is commercially usable** under the **NVIDIA Open Model License** ✓, and GR00T is no longer a separate runtime — `lerobot/policies/groot/` ships in installed **lerobot 0.5.1** as `--policy.type=groot` and consumes **LeRobotDataset v3.0** directly ✓. **⚠ BUT THE DEFAULT IS THE NON-COMMERCIAL ONE:** `GrootConfig.base_model_path` defaults to **`nvidia/GR00T-N1.5-3B`** ✓ (verified in the installed package), so a plain `--policy.type=groot` run hands you exactly the weights this gate disqualifies. **N1.7 requires an explicit `--policy.base_model_path=nvidia/GR00T-N1.7-3B` override.** WALL-OSS is Apache-2.0 and spinoff-safe. **⚠ CORRECTION (2026-08-09): SmolVLA is NOT verifiably Apache-2.0.** `lerobot/smolvla_base` declares **no `license` field at all** and ships **no LICENSE file** — the HF API returns `cardData.license: None`. Every upstream component is Apache-2.0 (SmolVLM2-500M, SigLIP-so400m, lerobot code), and nothing restrictive is attached, but the artefact itself states nothing, and this repo's own `public-datasets.md` rule is that *an unstated licence is not a permissive one*. This matters MORE than the π0.5 question below, not less: SmolVLA is the **onboard deploy target** — the thing that would actually ship on a robot. Cheapest fix, and free: open an issue on `lerobot/smolvla_base` asking HF to add the field. **Disqualified: AgiBot World is CC-BY-NC-SA-4.0** ✓ (non-commercial *and* share-alike). **⚠ UNRESOLVED — π0.5 weights:** openpi repo code is Apache-2.0 and lerobot's `pi05` source is Apache-2.0, **but the `lerobot/pi05_base` model-card frontmatter declares `license: gemma`** (PaliGemma backbone) ✓. Gemma Terms *permit* commercial use but attach a **Prohibited Use Policy** and **downstream redistribution obligations**. Flagged, not resolved: read the Gemma Terms before any commercial release or weight redistribution built on π0.5. |
| **G5** | **Primary data-collection mode:** current webcam-pose teleop, VR teleop (BEAVR/phospho), a UMI-style handheld, or EMMA egocentric? | Sets your throughput and whether the base is trained from teleop or human video. | **VR teleop for arms now** (zero build); skip a UMI SO-101 gripper (needs mount redesign). **v3: EMMA egocentric-for-base is parked with Stage 4** — it collects data for a base that doesn't exist. **MonoDuo's one-arm-at-a-time pattern is promoted** in its place, since bimanual throughput is now the binding constraint. |
| **G6** | **Sim multiplier:** stay in MuJoCo (DexMimicGen) or add Isaac (NVIDIA SO-101 curriculum) in the cloud? | Determines infra effort vs. turnkey-ness. | **DexMimicGen on existing MuJoCo** primary; **NVIDIA Isaac curriculum** as a cloud second source. |
| **G7** | **Onboard runtime:** `vla.cpp`, NVIDIA TensorRT, or plain PyTorch? | Sets the achievable onboard Hz and which models fit 8 GB. | **`vla.cpp` + RTC** — the only stack with measured 8 GB-Nano numbers. |
| **G8** | **NEW — which cloud generalist?** π0.5 (`--policy.type=pi05`), GR00T N1.7 (`--policy.type=groot`), or WALL-OSS (`--policy.type=wall_x`). | This is the model that sets your ceiling, produces your best evals, and teaches the onboard student. All three are now single-flag LeRobot policies, so the cost of the choice is *training time*, not integration. | **π0.5 primary** — it is the owner's chosen track, it has the strongest generalist story, and 12-dim is a non-issue (32-dim internal padding ✓). **GR00T N1.7 is the fallback and is now license-clean** — but you must pass `--policy.base_model_path=nvidia/GR00T-N1.7-3B`, because the flag alone defaults to the non-commercial N1.5 (G4). **WALL-OSS stays the Apache-2.0 A/B** — and is the one to reach for first if G4's Gemma question resolves badly. |
| **G9** | **RESOLVED — joint units: degrees or normalized?** | Silent data-compat failure: it never hurts training on Haller's own data, but it corrupts co-training with public datasets and blocks foreign checkpoints. | **Keep degrees** (the whole stack speaks degrees) **and record each joint's calibrated range into dataset metadata** so the map to normalized stays exactly recoverable. Full reasoning in Stage 0 §4. |

---

## Stage 0 — Foundations & data schema (Week 0–1) · *do not skip the schema step*

**Goal:** a clean, synchronized dataset schema and one recorded episode, end-to-end. The schema is the single most consequential design decision in the whole project — everything downstream re-collects data if you get it wrong.

1. **Confirm the environment.** You already have the `lerobot` conda env, ROS 2 Jazzy base, HMI teleop (60 Hz), calibration wizard, camera streams, `record_dataset.sh`, and the MuJoCo sim trio. Verify each still runs: teleop both arms, view all 3 cameras (D455 + 2 wrist), run the calibration wizard once.
2. **Freeze the `LeRobotDataset` schema.** Log, per timestep, synchronized:
   - **Exactly three camera channels** — `observation.images.top`, `observation.images.left_wrist`, `observation.images.right_wrist`. **Three, not five** — this is a decision, and it gets its own item below (§3).
   - `observation.state` = `[left arm 6 joints, right arm 6 joints]` = **12 dims** for 2× SO-101. **(v3 correction: v2 said 14.** SO-101 is 6 DOF *including* the gripper — `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`, per `LEROBOT_TO_MJCF` in `hmi/backend/haller_hmi/sim/arm.py`. v2 counted the jaw twice. 12 is also what every public bimanual SO-101 dataset uses, so this correction is what makes Stage 0.5's borrowed data loadable at all.)
   - `action` = the same **12 dims** (teleop target).
   - **`observation.base`** = wheel odometry / commanded velocity `(v, ω)` — **KEEP THIS SLOT.** The base is not built and Stage 4 is deferred, but the field is **2 floats**, it is **already being recorded**, and dropping it would **fork the dataset format**: every episode collected without it becomes incompatible with every episode collected after the base arrives. Two floats per timestep is the cheapest option Haller will ever buy. Write zeros while the base doesn't move.
   - ~~**`observation.lidar`**~~ = RPLIDAR scan — **NOT BUILT. Do not record it, and do not plan to.** Unlike `base`, a lidar scan is not a 2-float placeholder you can zero-fill for free, and it is only consumed by the deferred Stage 4. If the base is ever built, lidar is added *then* — and by construction that is the same moment the whole-body dataset starts anyway, so nothing is lost. **This is written down so nobody re-collects Stage 0–3 data later to "fix" a missing lidar stream.**
   - **`observation.effort` (NEW):** per-joint servo load — the contact signal that answers *"am I actually gripping something?"* Free, already on the bus, and it unlocks the tactile track (Stage 5b) plus grasp-force limiting at **zero hardware cost**. Three implementation facts, all verified against installed lerobot 0.5.1 and worth stating because each one is easy to get wrong:
     - **Read `Present_Load`, not `Present_Current`.** Load is already *signed* and already a *torque fraction* (±1023 per-mille, sign-magnitude bit 10, decoded by the Feetech SDK). `Present_Current`'s often-quoted ~6.5 mA/count scale is **not documented anywhere in the installed package or `scservo_sdk`** — it is a datasheet constant you would be hard-coding on faith.
     - **Sim and real cannot share a unit.** Real gives PWM duty; MuJoCo gives N·m. Both sides therefore normalize to *their own* saturation limit, and the recorded column is a **dimensionless signed fraction of that joint's torque limit** (real: `Present_Load`/1000; sim: `actuator_force`/`forcerange`). Sign = drive direction, |v| → 1 at stall. Without this, one dataset column silently means two different physical quantities depending on which rig recorded it.
     - **The bus cost is round-trips, not bandwidth.** Position reads are ~2.8% of wire time at 1 Mbaud, but telemetry runs synchronously on the event loop while teleop writes from another thread with no lock anywhere in lerobot. A second naive read doubles round-trips; one *combined block read* of the contiguous 56–70 register range gets Load for free.
   - **Grasp detection is partly free already.** `action[gripper] − state[gripper]` — both already recorded — is a standing tracking error whenever the jaw stalls on an object. Effort is still worth adding, because that error *saturates* once stalled while effort keeps carrying force, slip, and contact-anywhere-on-the-arm information.
   - `task` = natural-language instruction string (enables language conditioning).
   - **`task.coordination_class` (NEW, metadata):** label each task per the Krebs & Asfour taxonomy — *uncoordinated / loosely-coupled / tightly-coupled (symmetric vs dominant-asymmetric)*. Lets you build a deliberate bimanual curriculum in Stage 2 instead of one undifferentiated "bimanual" bucket.
3. **Freeze the camera *set* — record 3 channels, not 5 (decided).** The recorded set is now decoupled from the *displayed* set via a per-camera `record:` flag. On the real rig the three are the **D455** (RGB; optionally depth) plus the two wrist cams — the physical set already matches. The **bimanual sim renders five** views, of which three are recorded:

   | Rendered view | Recorded as | |
   | - | - | - |
   | `threequarter_sim` | **`top`** — the scene/base view | ✅ recorded |
   | `wrist_left_sim` | **`left_wrist`** | ✅ recorded |
   | `wrist_right_sim` | **`right_wrist`** | ✅ recorded |
   | `overshoulder_sim` | — | rendered for the operator only |
   | `overhead_sim` | — | rendered for the operator only |

   **Three independent reasons converge here, which is why this is a decision and not a preference:**
   - **It keeps Haller in-distribution for π0.5.** π0.5's pretrained camera slots are `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` — **one base + two wrists**. Matching that *count* means the cloud generalist (G8) receives the input shape it was pretrained on. Five channels is territory with no published run behind it; this **removes that risk instead of betting on it**.
   - **It makes co-training a rename, not a re-record.** `armnet/armnetbench_v01_lerobot_bimanual_so101` — the 1,219-episode bimanual SO-101 set from Stage 0.5 — uses exactly **`top` / `left_wrist` / `right_wrist`**. Adopting those keys now is free; adopting them after bulk collection is not.
   - **It drops fewer frames.** The recorder skips a tick whenever **any** required camera lacks a fresh frame, so the required-channel count multiplies into the drop rate. Three required streams is a materially more robust recorder than five.

   **Why `threequarter` for the base slot specifically — a judgment call, and presented as one.** `overhead` reads `shoulder_pan` well but **flattens lift and elbow**, destroying the height information grasping depends on. `overshoulder` is the egocentric *operator* view: it puts the arms between camera and work, so a policy trained on it would inherit the operator's **self-occlusion**. `threequarter` is angled and face-to-face — it preserves **lateral *and* vertical extent** with the bench and place zone unoccluded, making it the nearest analogue to ALOHA's `cam_high`. Both alternatives stay rendered, so this is **cheap to A/B before bulk collection and expensive to change after** — which is precisely the argument for settling it now.

   **The general lesson, since it is what this whole stage is about:** the camera **set** is as consequential as the feature list. It is a schema decision wearing a UI decision's clothes, and like everything else here, deferring it means re-collecting.
4. **Settle the units contract — the highest-risk item in this stage (G9).** This is the one Stage-0 mistake that does not announce itself.
   - **The mismatch:** Haller records joint angles in **degrees** — `hmi/backend/haller_hmi/arm.py` builds its `SO101FollowerConfig(..., use_degrees=True)` ✓ — while **every public LeRobot SO-101 dataset stores normalized `[-100, 100]`** (gripper `[0, 100]`), which is LeRobot's post-calibration default ✓.
   - **Why it is dangerous rather than merely annoying:** it has **no effect on training on Haller's own data**, because LeRobot normalizes from per-dataset statistics. So every local experiment passes and nothing looks wrong. What it silently breaks is exactly the two things you do *later*: **co-training with public datasets** (Stage 0.5 / Stage 2) and **loading anyone else's checkpoint**. A degree-valued elbow angle and a normalized one are both "small numbers" — nothing crashes, the policy just learns garbage.
   - **Decision: KEEP DEGREES.** The entire stack speaks degrees — the safety clamps, `step_budget_deg`, the collision guard's FK, and the sim. Converting the recording unit to match the world means editing every safety path, which is a far worse trade than owning a conversion.
   - **The obligation that comes with that decision:** record **each joint's calibrated range into dataset metadata at record time** — the `homing_offset` / `range_min` / `range_max` triple the calibration wizard already writes. Those numbers *are* the degrees↔normalized affine map. With them stored per-dataset, conversion in either direction is exact and reversible forever, and co-training becomes a preprocessing step instead of a re-collection. Without them, the map is unrecoverable and the episodes are stranded.
   - **Do this before the first real collection run.** It is metadata, so it cannot be backfilled onto episodes recorded under an unknown calibration.
5. **Record ONE pilot episode** of a trivial task (pick a cube, drop it) with the full schema. Inspect with LeRobot's dataset visualizer; confirm all streams' timestamps align.
6. **Push to the Hugging Face Hub** (per `docs/setup/dataset-collection.md`). This is your data backbone.

**Exit criteria:** one valid episode — **exactly 3 recorded camera channels (`top` / `left_wrist` / `right_wrist`)** + `base` (2 floats) + **effort** + language, **12-dim state/action in degrees, with the calibrated joint ranges written into dataset metadata** — visualized and on the Hub. **No lidar.**

---

## Stage 0.5 — Scale data collection *before* mass teleop (Week 1–3) · **NEW, highest-leverage**

**Goal:** set up a data *flywheel* that does not depend on you joysticking thousands of episodes. This stage is new in v2 because the verified evidence says data *strategy* — not model choice — is what caps a solo project.

1. **Adopt the diversity-over-repetition law (do this regardless of tool).** Zero-shot generalization scales as a **power law in the number of distinct environment-object *pairs*, not raw demo count** (Data Scaling Laws, ICLR'25 ✓). Practically: **~32 env-object pairs × ~50 demos** generalizes far better than 1,600 demos of one setup. Plan collection as a *diversity budget*, and track pairs, not episode count.
2. **Borrow the data that already exists — this is a real lever, not a footnote (NEW in v3).** Public **12-dim bimanual SO-101** datasets now exist in **LeRobotDataset v3.0** with **joint names matching Haller's layout**, all **Apache-2.0** ✓. That means the entire 12-dim bimanual training path — data loading, normalization, policy config, the eval harness, the π0.5 and SmolVLA runs — **can be built and validated before a single Haller episode exists.** Do this while you are still assembling teleop; it converts "does my pipeline work?" from a question you answer with your own scarce data into one you answer for free.

   | Dataset | Size | Use |
   | - | - | - |
   | `armnet/armnetbench_v01_lerobot_bimanual_so101` | **1,219 real episodes / 743,964 frames / ~10.3 h / 43 GB** ✓ | The anchor: real-robot bimanual pretraining + co-training corpus |
   | `Elvinky/bi-so101-insert-screw-562ep` | **562 episodes / 2,070,776 frames / 21 GB** ✓ | Contact-rich, tightly-coupled — the hard end of the coordination taxonomy |
   | `ChihHanShen/bimanual-so101-pickvials-sim` | **263 episodes / 3.97 GB**, simulated ✓ | Sim↔real co-training A/B against your own Stage-1.5 synthetic data |
   | `lerobot/svla_so101_pickplace` | **50 ep / 86 MB**, **unimanual 6-dim, v2.1** ✓ | **The pipeline smoke test.** Small and fast; wrong shape and older format on purpose — if this loads and trains, your plumbing works |

   **Mind the units when you touch any of these (Stage 0 §4):** they are normalized `[-100, 100]`, Haller is degrees. Co-training without the conversion is the silent-corruption case.
3. **Know what is disqualified, so you stop reading about it (NEW in v3).**
   - **🚨 AgiBot World — out.** **CC-BY-NC-SA-4.0** ✓: non-commercial *and* share-alike, so it fails G4 twice over. (The AgiBot World *Challenge* remains fine as a hardware-free competition — see track R/B. Entering a challenge is not redistributing its dataset.)
   - **DROID / BridgeData V2 / Open-X — structurally incompatible, not merely awkward.** They use **end-effector pose action spaces** ✓. A 12-dim joint vector and a 6-DoF pose are different quantities, so these **cannot co-train with Haller's action space at all** — no remap, no adapter, no "just retarget it." Rule them out at the schema level and move on.
4. **Turn on VR teleop for the arms (zero hardware build).** Two options, both dropping straight into your LeRobot/HF pipeline:
   - **BEAVR** (open, self-hostable, no subscription; Quest 3S hand-tracking, native LeRobot schema, whole rig ≈ **$1.08k** ✓, sub-35 ms, ACT trained on its data hit 100% on a pickup task) — preferred for an open stack.
   - **phospho Quest app** (subscription-gated VR control; also SO-100/101-native) — the fast A/B.
   Compare either against your current webcam-pose teleop for throughput and data quality.
5. **For solo bimanual data, use MonoDuo's pattern.** Drive one arm at a time (a human handles the other, swap roles), synthesize the missing arm via hand-pose + inpainting — solves the "I can't teleoperate both SO-101s at once" constraint. **With the base deferred, this is now the most important collection trick in the stage**, because bimanual throughput is the binding constraint on an arms-only plan.
6. **(PARKED with Stage 4) EMMA-style egocentric→base data collection.** EMMA retargets **egocentric human walking trajectories (Aria glasses) into feasible differential-drive base paths**, co-trained with a little static-arm data — *no mobile teleop at all*. It remains the highest-upside research bet (track R/B), but it collects data for a **base that does not exist**, so it is parked alongside Stage 4 rather than pursued now. Reactivate on the same trigger.
7. **(Deferred) UMI-style handheld gripper.** Powerful (UMI: **111 demos/hr vs 35 via SpaceMouse** ✓, in-the-wild, no robot needed) — but SO-101's STS3215 parallel-jaw geometry doesn't match UMI's >85 mm-stroke assumption, so it needs a real mount redesign. Skip until Stage 2+ unless you want that CAD work. FastUMI-100K / **YUBI** ($200 build) are also **ready-made open LeRobot datasets** for cross-embodiment pretraining.

**Exit criteria:** VR teleop recording into the schema; a *diversity plan* (≥20 env-object pairs targeted); and — the new, cheap one — **a 12-dim bimanual training run completed end-to-end on a public dataset before Haller's own data exists**, proving the pipeline rather than assuming it.

---

## Stage 1 — ACT baseline (Week 2–4) · *learn the fundamentals, get a deployable baseline*

**Goal:** train and deploy the ALOHA-lineage bimanual BC method; understand action chunking end-to-end. Unchanged in intent, now with a **measured baseline to hit**.

1. **Collect a single-task dataset** (~50 demos of one task, e.g. single-arm pick-and-place) via VR teleop from Stage 0.5, kept varied (object position, lighting).
2. **Train ACT** with LeRobot (`lerobot-train ... --policy.type=act`). Runs on any of your GPUs (even 12 GB). Log to W&B / TensorBoard.
3. **Evaluate — in sim first, then on the robot.** Measure success over ~20 trials.
   - **Evaluate in sim first, and understand why.** The Stage-1.5 sim work now landing adds an **automatic contact-based task-success predicate**, so closed-loop success in sim is **auto-scorable** — no human watching rollouts and adjudicating. That is the real argument for sim-first, and it is not the one usually given: sim-first is worth doing because it makes the objective **measurable** (and therefore sweepable, regression-testable, and comparable across checkpoints), not merely because it is safe. Safety is a bonus; a scalar you can compute unattended is the point.
   - **Then evaluate on the robot** via LeRobot's control loop on the Orin Nano. Sim success is a proxy — it ranks checkpoints cheaply so that scarce real-robot trials are spent on candidates worth spending them on.
   - **Baseline to beat (verified twin):** the near-identical **XLeRobot** (dual SO-101 + Orin Nano) reports **ACT at 36 ms / 27.8 Hz onboard** and **98.7% grasp over 75 trials** (`~` author-reported), with no thermal throttling after 30 min. Use these as your sanity-check targets; if you're far off, suspect the pipeline, not the method.
4. **Study what you built:** action chunking, temporal ensembling, obs/action normalization, the eval harness. Write notes. This is your reference baseline for everything later.

**Exit criteria:** ACT achieves non-trivial success on one task, running onboard; you can explain chunking + the eval loop.

---

## Stage 1.5 — Multiply demos in sim (Week 3–6, overlaps Stage 1) · **NEW**

**Goal:** turn a handful of real demos into thousands of training trajectories using the MuJoCo sim you already have — the cheapest large lever in the whole plan.

0. **This stage is no longer speculative — its foundation is being built right now (v3).** In-flight in the existing MuJoCo sim: **per-episode scene reset**, **seeded domain randomization**, and an **automatic contact-based task-success predicate**. Those three are exactly the primitives every item below assumes and v2 quietly hand-waved:
   - *scene reset* is what makes episodes independent, so you can generate N of them unattended instead of babysitting one;
   - *seeded DR* is what makes the generated variation **reproducible** — a seed is a citable, re-runnable experiment, and it is the difference between "synthetic data" and "a dataset";
   - *the success predicate* is what turns a rollout into a **label**, which is what makes both demo-multiplication filtering (Stage 1.5) and closed-loop eval (Stage 1) possible without a human in the loop.
   Sequence the rest of this stage behind that landing rather than in parallel with it.
1. **Bridge MuJoCo → DexMimicGen (primary, G6).** DexMimicGen (NVlabs, open) is **purpose-built for bimanual** manipulation and runs on **robosuite (MuJoCo)** + **BiGym (mobile bimanual)** — it generated **21K demos from 60 human demos** across 9 tasks ✓. Port an SO-101 MJCF into its pipeline, seed it with your Stage-1 teleop demos, and generate a large synthetic set. (Predecessor MimicGen: 50K from <200 demos.)
2. **Run the NVIDIA official SO-101 Isaac curriculum in the cloud (second source).** "Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac" ✓ is the **only toolchain documented for your exact arm** — Isaac Lab Mimic demo-multiplication, Cosmos/GR00T-Dreams synthetic trajectories ("36 h vs ~3 months" of manual collection ✓), actuator-gap closure, and an **official script that converts to LeRobot format**. Heavier than MuJoCo → run on RunPod as a parallel data source, not a sim replacement.
3. **Use RoboTwin-2.0 for domain-randomized bimanual data (MIT license).** A VLA trained on **synthetic + only 10 real demos got +367%** over 10-real-only; **synthetic-only +228%** ✓. Strong evidence that synthetic co-training slashes your real-demo requirement.
4. **GPU-accelerate rollouts with MuJoCo Playground / MJX** (Apache-2.0) — same physics family, trains policies / generates rollouts in minutes on one GPU on RunPod.
5. **(Optional) Cut scene-building with Digital Cousins / ACDC** — generate approximate sim scenes from single phone photos of your real workspace instead of hand-modeling every object.
6. **Contribute the missing SO-101 `gym_hil` env (feeds track R/B).** LeRobot's native HIL-SERL RL loop ships **Franka-only**; porting an SO-101 MJCF is a small, well-scoped, *upstream-mergeable* contribution that plugs into your MuJoCo assets and sets up Stage 5 RL.

**Exit criteria:** at least one task's dataset multiplied ≥10× via sim; a measured comparison of a policy trained on (real-only) vs (real + synthetic).

---

## Stage 2 — Two generalists: SmolVLA onboard + π0.5 in the cloud (Week 6–12) · *the recommended v1 endpoint*

**Goal:** a single **language-conditioned, multi-task** policy on your two arms, with **bimanual-specific** coordination — trained at full strength in the cloud and deployed at Orin-Nano strength onboard.

**The v3 shape of this stage:** train **π0.5 on RunPod** and **SmolVLA locally** *from the same dataset, on the same tasks, through the same eval harness*. π0.5 tells you what the data can support; SmolVLA tells you what the robot can run. The gap between the two is the single most informative number this stage produces — it separates "the policy is too small" from "the data is too thin," which are the two failure modes people spend months confusing.

1. **Collect / assemble a multi-task, language-annotated dataset.** 3–5 tasks (pick-place, stack, handover, put-in-container), **each with a `task` string** and a `coordination_class` label, ~50–100 demos/task — now heavily padded by Stage-1.5 synthetic data, Stage-0.5 diversity, and the **public bimanual SO-101 corpora** (Stage 0.5 §2; mind the units, Stage 0 §4).
2. **Fine-tune SmolVLA** from the community checkpoint (`--policy.type=smolvla`): freeze the SigLIP + SmolLM2 backbone, train the flow-matching action expert. Fits your GPUs at full fine-tune. **This remains the model that ships** (Stage 3).
3. **Train π0.5 on RunPod (`--policy.type=pi05`) — first-class, not a side quest (NEW in v3).** This is the cloud generalist: the ceiling-finder, the best-eval producer, and the **distillation teacher** for the onboard student.
   - **12 dims is a non-issue — do not design around it.** π0.5 **pads state and action to 32 dims internally and unpads the loss and output back to the dataset's real dimensionality** ✓. Haller's 12-dim bimanual action space therefore needs **no architectural change, no dimension-matching wrapper, and no padding of your own**. Point this out to yourself now, because "our action space is too small for the big model" is a plausible-sounding blocker that would cost a week to disprove from scratch.
   - **Camera slots already line up.** π0.5's pretrained views are `base_0_rgb` + `left_wrist_0_rgb` + `right_wrist_0_rgb`, which is exactly the 3-channel set frozen in Stage 0 §3. That alignment was the point of that decision.
   - **⚠ Licensing is not settled — see G4.** The code is Apache-2.0 on both sides, but the `lerobot/pi05_base` **model card declares `license: gemma`** ✓. Gemma Terms permit commercial use while attaching a Prohibited Use Policy and redistribution obligations. **Fine for research and for training today; read the terms before shipping or redistributing anything derived from these weights.** This is a flag, not a stop.
4. **A/B against WALL-OSS (spinoff-safe).** **WALL-OSS** (X-Square, ~4B) is a **first-class LeRobot policy (`--policy.type=wall_x`), Apache-2.0**, trainable via the standard CLI with zero glue code; WALL-OSS-0.5 reports **>80% zero-shot** on a 17-task real eval ✓. It's 4B → **cloud/RunPod**, not onboard. **Its role in v3 is sharper than in v2:** it is the *cleanly-licensed* cloud generalist, so it is the fallback that matters if G4's Gemma question resolves badly. **GR00T N1.7 (`--policy.type=groot`) is the other fallback and is license-clean** (G4).
5. **Add a bimanual-coordination head (raise the ceiling on two-hand tasks).** In order of leverage:
   - **Keypose-hierarchical control** (BiKC / **AnchorDP3**): predict sparse per-arm keyposes (pre-grasp→grasp→place) + a fast low-level generator instead of dense per-timestep chunking — *cheaper at inference* (matters on the Nano) and won the CVPR'25 RoboTwin Dual-Arm Challenge.
   - **Co-VLA structured action expert**: replace the monolithic head with a **shared coordination latent + per-arm residuals** + a coordination-aware loss — a small add on a SmolVLA/π0 backbone, **+27% on tightly-coordinated tasks** ✓.
   - **Bootstrap from single-arm data** (**AnyBimanual** ✓ / TwinVLA): you have far more single-arm data/checkpoints than bimanual — transfer a unimanual policy into a bimanual one with few bimanual demos.
6. **Co-train with the AIST bimanual dataset — but price the conversion first (corrected in v3).** **10,705 real episodes / 112 tasks ✓** (v2 said 119) on an ALOHA-lineage rig mechanically close to SO-101, **CC-BY-4.0 ✓** so it clears G4. Two costs v2 didn't state: it needs the **ALOHA→Feetech action remap**, *and* it **ships as HDF5 on Dropbox, not in LeRobot format ✓** — so using it means **writing and maintaining a converter**. That is real work, and it is why the Stage-0.5 datasets come first: they are **already LeRobotDataset v3.0 with matching joint names**, i.e. the same benefit at none of this cost. Reach for AIST when you've exhausted the free-to-load corpora, not before.
7. **Evaluate multi-task, language-conditioned — one harness, both models.** Test each task by *changing only the instruction string*; measure per-task success + generalization to unseen object positions, sliced by coordination-class. Score in sim automatically (Stage 1.5's success predicate) and on the robot for the finalists. **Report π0.5 and SmolVLA side by side on every task** — that comparison is the deliverable, not a nice-to-have.
8. **Iterate on data, not architecture.** When a task fails, add targeted demos (or synthetic variants). The data-flywheel discipline is the highest-leverage habit. **The two-model setup makes this diagnosable:** if π0.5 also fails the task, it's a data problem; if only SmolVLA fails it, it's a capacity problem and distillation (Stage 3) is the lever.

**Exit criteria:** one policy performs ≥3 tasks selected by language, with a measurable win from the bimanual head on tightly-coordinated tasks — **plus a measured π0.5-vs-SmolVLA gap on the same tasks.** **This is a legitimate v1 generalist for Haller.**

---

## Stage 3 — Onboard real-time deployment (Week 10–14, overlaps Stage 2) · **now first-class, and measured**

**Goal:** run the language-conditioned policy **onboard the Orin Nano in real time**. v2 promoted this from a footnote to a stage because the numbers are now known.

1. **Adopt the `vla.cpp` runtime (G7).** The llama.cpp-derived C++ VLA server has the **only measured 8 GB-Orin-Nano numbers**: **SmolVLA 141.8 ms/step (~7 Hz)**, BitVLA 356 ms, and **GR00T N1.6/1.7 do not fit** ✓. On AGX Orin, faster runtime alone *doubled* ALOHA success (87.5% vs 40%) — runtime engineering changes task outcomes, not just latency.
2. **Wrap it in Real-Time Chunking + async inference (shipped in LeRobot, no retraining).** RTC generates the next action chunk *while executing the current one* with soft-mask inpainting (`RTCConfig(execution_horizon=10, max_guidance_weight=10.0, prefix_attention_schedule=EXP)`); async inference hides idle wait. Together they absorb the ~140–350 ms/step the Nano actually produces without visible jerkiness. Handles >300 ms delays.
3. **If memory-bound, try BitVLA (1-bit, open CC-BY-4.0):** **11× less memory, 4.4× lower latency** vs OpenVLA-OFT at matched success ✓ — attacks the 8 GB ceiling. (Caveat: on the Nano *today* its kernel is slower than SmolVLA — a memory win, not yet a speed win.) Also benchmark **TinyVLA / MiniVLA / EdgeVLA** (sub-1B) — latency is set by the diffusion head, not param count (SJTU XPU study ✓: ACT single-pass is fastest on every platform).
4. **Keep 3B models in the cloud — but treat the cloud as a workbench, not a waiting room (revised in v3).** Run **π0.5 / GR00T N1.7 / WALL-OSS** on RunPod. Stream actions over the network *only* when tethered operation is acceptable (G2). Their primary job is **distillation teacher** for the onboard SmolVLA student — that is the mechanism by which cloud-scale capability actually reaches the robot, and it is now a planned step rather than an aside.
   - **Two v2 statements here were wrong and are corrected:** GR00T is **not** non-commercial — **N1.7 is commercially usable under the NVIDIA Open Model License** ✓ (the non-commercial terms applied to **N1.5**). And GR00T is **no longer a separate runtime to integrate**: `lerobot/policies/groot/` ships in installed **lerobot 0.5.1** as `--policy.type=groot` and **consumes LeRobotDataset v3.0 directly** ✓. The practical consequence is that the cost of *trying* GR00T has collapsed from "adopt a second framework" to "change one flag," which is why it is now a real fallback under G8 rather than a footnote.
   - **What has *not* changed:** GR00T N1.6/1.7 still **do not fit** the 8 GB Orin Nano ✓. License-clean and dataset-native does not mean deployable. Cloud-only stands.
5. **Deployment split, decided:** **onboard** = SmolVLA(/TinyVLA) → `vla.cpp` → RTC, target **~5–10 Hz**; **cloud** = anything ≥3B, for training / eval / **teaching the student** / tethered use.

**Exit criteria:** SmolVLA-class policy running onboard at ~5–10 Hz with RTC, controlling ≥1 real task without a network dependency.

---

## Stage 4 — Manipulation-aware mobile base · 🛑 **DEFERRED — OUT OF SCOPE FOR NOW**

> **Status: deferred, not cancelled. The mobile base does not exist yet**, so the two-arm setup is the entire near-term scope. Everything below is preserved because it is researched and correct — it is simply not actionable, and no week should be budgeted against it.
>
> **What this deferral changes today (all of it in Stage 0):**
> - **`observation.base` stays in the schema.** It is 2 floats, already recorded, and deleting it would fork the dataset format — every pre-base episode would become incompatible with every post-base one. Keeping it is the cheapest insurance in the plan.
> - **`observation.lidar` is dropped** — "not built," not "optional." A scan is not a 2-float placeholder, and it feeds nothing outside this stage.
> - **EMMA (Stage 0.5) is parked here too**, since it collects data for a base that does not exist.
>
> **The trigger that reactivates this stage:** *the base is physically built and drivable under `(v, ω)` commands* — i.e. G1's on-paper decision becomes hardware. Nothing else reactivates it: not a good result in Stage 2, not spare time, not a promising paper. When that trigger fires, resume at item 1 below and start logging `observation.lidar` and non-zero `observation.base` from that day forward; the arms-only episodes stay valid because the schema never forked.
>
> **A note on sequencing, since it is easy to read this as a loss:** arms-only is the *right* order regardless of hardware. Stages 0–3 produce the dataset, the eval harness, and the onboard runtime that Stage 4 would otherwise have to build while *also* solving navigation. The base arriving later means it arrives into a working pipeline.

**Goal (when reactivated):** coordinate base + arms — and specifically fix the "arrived at the goal but the arm can't reach" failure. Start decoupled, move toward unified.

1. **Whole-body IK in sim with Mink (do this first — fastest concrete step).** **Mink** (Apache-2.0, MuJoCo differential-IK by Kevin Zakka) drops straight into your existing MuJoCo sim to build a **base + dual-arm whole-body IK controller**, offloading low-level base/arm coordination from the learned policy (the HoMeR pattern: 79% at 20 demos).
2. **Decoupled nav→manip baseline (robust first system).** Use RPLIDAR + Nav2 (already in `haller_navigation`) for SLAM/costmap, drive to a task pose, then run the Stage-2/3 arm policy. Your lidar earns its place here.
3. **Add manipulation-aware base placement (the key upgrade).** **N2M / N2M2** (open) learns a **preferred base pose for the downstream manipulation from egocentric camera only — no SLAM map, no holonomic base** — a perfect fit for diff-drive + 2D lidar + a small arm workspace (**3%→54%** on PnPCounterToCab ✓). This is the fix for "drive-then-grasp."
4. **Add closed-loop recovery with ANCHOR's pattern.** ANCHOR targets exactly "arrived but inoperable" with *local* recovery-replanning (cheap enough for the Nano): **53.3→71.7% success, 71.4% recovery** under disturbance ✓.
5. **Log whole-body data.** Now that the base moves during tasks, the `observation.base` slot held open through Stage 0 pays off — start writing non-zero `(v, ω)`, **add `observation.lidar` at this point**, and record mobile-manipulation demos with base + arms moving together. (Multiply them with **MoMaGen**-style reachability-constrained augmentation in MuJoCo.)
6. **(Dynamic scenes) DynaMem** for "fetch object X wherever it is now" — an online spatio-semantic memory (70% vs 30% OK-Robot on moving objects); its VLM queries need cloud offload.
7. **Toward unified whole-body (frontier).** Fine-tune a π0.5-class model that outputs base + arm actions jointly on the whole-body dataset — the true generalist endpoint (cloud-side; Galaxea G0's R1-Lite is the closest open architecture reference).
8. **Calibrate expectations.** Even a fine-tuned π0.5 with 10k demos hit only **~12.4% task success** on 50 long-horizon BEHAVIOR-1K home tasks ✓ — robust long-horizon home autonomy is *not* solved by anyone. Aim for narrow, hardened tasks, not open-ended home tidying.

**Exit criteria (when reactivated):** robot navigates to an object and manipulates it (with learned base-pose preference, not naive drive-then-grasp); whole-body dataset recorded for unified-policy training.

---

## Stage 5 — Advanced paradigms (parallel research rungs, Month 4+)

Pick based on what's limiting you; each maps to a paper to implement.

**5a — Get better than your demos (RL / self-improvement).**
- **Start logging teleop *interventions/corrections now*** (the RECAP lesson — π*0.6 "more than doubles throughput, roughly halves failures" ✓, learning from real deployment corrections). Cheap to add; seeds all later RL.
- Then **SimpleVLA-RL / ReWiND / ConRFT**-style RL fine-tuning on top of the BC/VLA policy (ReWiND: ~5× in ~1 h online, author-reported). Pairs with the SO-101 `gym_hil` env from Stage 1.5.

**5b — Tactile on a budget (parallel, cheap first step).**
- **Zero-cost:** you're already logging `observation.effort` from STS3215 `Present_Load` (Stage 0) — use it for stall/contact detection and grasp-force limiting; test whether coarse force helps.
- **If it helps:** retrofit **one** gripper with **WowSkin** (AnySkin, **$128**, purpose-made SO-101 mount from 3rd-party vendor WowRobo ✓) and encode via **Sparsh** (Meta's open frozen touch encoder, +95.1% over task-specific training at fewer labels ✓) to avoid hand-labeling. Then fuse into the VLA (the new TacVLA/ForceVLA wave). Force feedback flips contact-rich near-failures to near-100% (FoAR: wiping 75→100% ✓).
- This is also a strong **research differentiator** (tactile-on-cheap-hardware is under-studied) → track R/B.

**5c — Learn from human/internet video** (LAPA-style latent-action pretraining; EMMA for mobile — *parked with Stage 4*). Cheapest route to generalization given scarce teleop, and the LAPA half is arms-only, so it stays live.

**5d — World models & JEPA** (DreMa / diffusion WM for safe RL-in-imagination; V-JEPA 2-AC / DINO-WM for predict-then-plan; the 2026 World-Action-Model fusion — VLA-JEPA, RISE, WoVR). The frontier is converging here; study V-JEPA 2 (open weights, monocular, extreme data efficiency) as a parallel track. Full detail in `so101_sota_recommendation.md` §8–10.

---

## Track R/B — Research & business (parallel from day one)

Haller's embodiment is **validated but crowded** — at least 8 near-siblings exist (XLeRobot, AhaRobot, YOR, LeKiwi, Galaxea R1 Lite, Sunday Memo, Zerith H1, the *discontinued* Trossen Mobile ALOHA). **The hardware is not the moat.** Differentiate on software, data, curriculum, or a narrow niche.

**Research (educational + publishable, GPU-only, ~6–12 months):**
1. **Release a Haller LeRobot dataset — *bimanual* now, *mobile-bimanual* if the base ever lands.** Mobile-manip data scarcity is *quantified* ("datasets orders of magnitude larger than stationary" ✓), so the mobile version is the bigger prize — but it is gated on hardware that does not exist. **The arms-only release is available today and is still a real contribution**, especially with the coordination-class labels, the effort channel, and the degrees↔normalized metadata (Stage 0 §4) that public SO-101 sets do *not* carry. Ship what you can build.
2. **(Parked with Stage 4) EMMA-style egocentric-for-diff-drive** (Stage 0.5) — genuinely novel *and* Haller-shaped, and still the highest-upside research bet, but it needs a base. Hold it.
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

1. **Data > architecture — and now with concrete levers.** The bottleneck is demonstrations; the wins are *diversity* (power-law law), *cheap collection* (VR), *sim multiplication* (DexMimicGen on your MuJoCo), and — new in v3 — **borrowing the public bimanual SO-101 corpora that already exist**. Build the flywheel *before* chasing models. Log `base` + **effort** from episode 1.
2. **Two models, one pipeline (v3).** π0.5 in the cloud finds the ceiling; SmolVLA onboard is what ships. Sharing one dataset and one eval harness between them is what makes the gap between them *diagnostic* — it tells you whether you have a data problem or a capacity problem, and those have opposite fixes.
3. **Deployment is measured, not guessed.** SmolVLA-class + `vla.cpp` + RTC ≈ 5–10 Hz onboard; 3B models don't fit 8 GB and are cloud-only. Decide by hardware reality, not model prestige. **Bigger ≠ better, and bigger doesn't fit your robot** — but bigger is still *useful*, as a teacher.
4. **Interfaces fail silently; performance fails loudly (v3's most expensive lesson).** Units (degrees vs normalized), action-space semantics (joints vs end-effector pose), dimensionality (12 vs 14), camera keys and camera *count* — none of these throw an error. They train fine and produce a worse policy, or they quietly block every future use of someone else's data. **Freeze the schema, then write down the metadata that makes it convertible.** A bad number in a config is cheaper to find than a bad assumption in a dataset.
5. **Check the licence, and check it against the right version.** v2 recorded "GR00T is non-commercial" and was correct — about **N1.5**. **N1.7 is commercially usable**, so carrying that belief forward cost real optionality. Meanwhile π0.5's *weights* carry a Gemma licence the surrounding Apache-2.0 code hides. **Licences are versioned artefacts and model cards disagree with repos**; re-verify at the point of shipping, not at the point of reading.
6. **Scope to the hardware you actually have (v3).** No base means arms-only, and arms-only is not a consolation prize — it is the right build order, since Stages 0–3 produce the dataset, harness, and runtime that a base would otherwise need *while* solving navigation. Defer the stage, keep the 2-float schema slot, drop the stream that costs more than a placeholder.
7. **Stage, don't leap.** Each rung teaches and de-risks the next: data flywheel → ACT → SmolVLA + π0.5 (+bimanual) → onboard → *(deferred: mobile)* → advanced.
8. **The frontier is convergence** (BC + world models + video + RL → World-Action Models). Understanding each paradigm in isolation is how you get there deliberately.
9. **The moat is software/data/niche, not steel.** Research and a spinoff both run through the *pipeline* you already own, not a robot-for-sale.

---

*Provenance: quantitative claims marked `✓` were verified against primary sources in one of two passes. **(1) The v2 pass:** adversarial fact-checking recorded in `research/haller_frontier_scout.md` (33 claims, 0 refuted; 14 supported-with-a-correction — see that doc's "Verification outcomes"). **(2) The v3 pass:** the licensing, dataset, units, and dimensionality claims introduced in this revision, verified against — GR00T N1.7 / NVIDIA Open Model License: `https://huggingface.co/nvidia/GR00T-N1.7-3B`; π0.5 weight licensing: `https://huggingface.co/lerobot/pi05_base/raw/main/README.md` (frontmatter `license: gemma`) and `https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE` (Apache-2.0 code); AIST bimanual 112 tasks / CC-BY-4.0 / HDF5-on-Dropbox: `https://aistairc.github.io/aist_bimanip_site/`; the four public SO-101 datasets and their episode/frame/size figures, from their Hugging Face dataset cards; the degrees-vs-normalized units mismatch, from `hmi/backend/haller_hmi/arm.py` (`use_degrees=True`) against LeRobot's post-calibration default; the 12-dim correction, from `LEROBOT_TO_MJCF` in `hmi/backend/haller_hmi/sim/arm.py`.*

*Untagged figures (success rates, GPU-hours, speedups, parameter counts, funding) are author-reported from the cited papers/announcements — treat as directional and verify against source tables before betting a design decision. **Two v3 additions are deliberately untagged and should be read as reasoning, not as verified fact:** the camera-set rationale in Stage 0 §3 (a schema decision by the project owner, with `threequarter`-for-`top` explicitly a judgment call to A/B before bulk collection), and the Stage 1.5 sim-infrastructure description (work in flight at the time of writing, not yet landed). **One item is flagged UNRESOLVED and must not be treated as settled either way: the π0.5 Gemma licensing question under G4.***
