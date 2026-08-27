# The kit port — plan of record (2026-08-27)

Branch `feat/kit-port`, cut off `refactor/hmi-unify`.
Rollback tag `baseline-2026-08-27-kit-port` at `51844b9`. Anything removed below is
recoverable from it — remove without ceremony.

`~/vr-teleop-kit` is a **read-only source**. Nothing in this port writes to it, ever.
Its venv (`/home/odesha/vr-teleop-kit/.venv/bin/python`) may be imported from and run;
its tree may not be touched. It is the only working end-to-end evidence we have and
staying able to diff against it is worth more than the convenience of editing it.

## The situation, stated plainly

Oscar's VR data collection never worked *in Haller*. He built the kit standalone and it
worked — recording, review, ACT training, a rollout. The kit is now being ported IN.

Evidence on disk, `~/robot-data/lerobot/local/`, read 2026-08-27:

| | kit (`so101_pick_cube`) | Haller (`haller_pick_the_red_cube_…`) |
|---|---|---|
| episodes / frames | **46 / 29,500** | 2 / 997 |
| provenance | real SO-101 hardware | sim only — never recorded real hardware |
| review marks | 35 keep / 11 reject (`review.json`) | none |
| downstream | trained ACT + a rollout | none |
| camera keys | `top` | `top`, `left_wrist`, `right_wrist` |
| fps / version | 30 / v3.0 | 30 / v3.0 |

Haller has the better *schema* and the worse *dataset*. That asymmetry is the whole
port: Haller's safety core, arm manager, sim path and in-headset HUD stay; the kit's
acquisition discipline and data layer come in.

## Why it failed here — four mechanisms

Not "it was flaky". Four specific, separately fixable defects.

1. **Three uncoordinated samplers.** The session commits at 60 Hz; telemetry runs its
   own 20 Hz loop; the recorder scrapes `human_teleop.status()["goal_deg"]` at a third
   instant. Every recorded row pairs a state from one moment with an action from
   another. A policy trained on that is being taught a lie about causality, and no
   amount of episodes fixes it.
2. **The block read never checks `isAvailable()`.** `arm.py::_read_block` checks the
   comm result, then does `getData(id, addr, 2)` per motor. A lost race on the shared
   `bus.sync_reader` returns tick **0** for that motor — and 0 is a legitimate raw
   register value that decodes to a large **angle**, not to zero degrees. The
   docstring's claim that a lost race "degrades to a 0 reading for one tick, not to a
   garbage position" is the bug: for a recorded dataset a 0 reading *is* a garbage
   position.
3. **`fps` is declared, not measured.** `recorder.py:380` —
   `fps = int(round(1.0 / self.telemetry._period))`. That is the rate telemetry was
   *asked* for. It has never been measured against real hardware with real Feetech
   round trips in the loop. Every `timestamp` in every episode is synthesised from it.
4. ~~**Every recorded D455 pixel is wrong.**~~ **RETRACTED 2026-08-27 — see below.**
   The measurement was real; the attribution was wrong, and it was mine.

Mechanisms 1–3 are one fix, not three: **one sampler that owns the tick**. Mechanism 4
does not exist.

## The D455 — a retracted diagnosis (2026-08-27)

An earlier version of this document said the mast cam was read through OpenCV/v4l2 at
66.5/255 with a +31.9 magenta bias against librealsense's 126.1/+9.7, concluded there was
"no OpenCV-side fix", and OVERTURNED PLAN-2026-08-22 decision 6 on that basis. **All of
that rested on measuring the wrong sensor.**

`/dev/video2` is not the colour camera. It is the stereo module's **infrared imager**. All
six D455 nodes report an identical `name`, so nothing in `/dev` distinguishes them, and
`/dev/video2` gets picked because it is the first non-depth node that opens. The RGB
camera is on USB interface `1-4:1.3`. Verified three ways — sysfs interface numbers,
librealsense `physical_port`, and `VIDIOC_ENUM_FMT` (video2 offers only GREY+UYVY;
video4 offers YUYV).

Re-measured on the REAL colour node, same scene:

| node | path | mean | magenta bias | **rb_spread** |
|---|---|---|---|---|
| `/dev/video2` (IR imager) | OpenCV | 60.2 | +32.0 | **5.4** |
| `/dev/haller_cam_mast` → video4 (**RGB**) | OpenCV | 99.3 | +23.2 | **18.6** |
| `/dev/video4` (**RGB**) | librealsense | 101.1 | +18.0 | 27.2 |

**OpenCV decodes the real RGB camera correctly**, within ~0.6 of librealsense on the
bench run in `docs/port/phase0-runtime.md`. There was never an OpenCV-side bug.

**`rb_spread` is the discriminator, not the bias.** Magenta requires R and B to move
TOGETHER against G: the broken decode has them 5.4 apart, a merely warm room has them
18–27 apart. The bias alone separates the two populations by 1.6×; the spread separates
them by ~5×. A correct path in a warm room reads +23, two thirds of the way to the old
"+25 means broken" threshold — which is how a good camera got condemned.

**Haller already solved this and its solution measures as well as librealsense.**
`config.yaml` names no `/dev/videoN`; it uses `/dev/haller_cam_mast`, pinned by
`scripts/99-haller-devices.rules:56` to `ID_USB_INTERFACE_NUM=="03"`, `ATTR{index}=="0"`.
That IS the colour node by construction and cannot resolve onto the IR imager however the
kernel renumbers — and it does renumber: enabling the CSI overlay once moved colour from
video4 to video6.

### Consequences

- **PLAN-2026-08-22 decision 6 stands. It is NOT overturned.** `pyrealsense2` is not
  required for correct colour. It remains a *soft*, optional dependency and the serving
  venv does not need it.
- **LEAVE THE MAST CAMERA PATH ALONE.** It is correct and deployed. Switching it to
  librealsense would add a dependency for no measured gain.
- `haller_hmi/realsense.py` keeps a **smaller** justification: the IR emitter (no V4L2
  equivalent), a box with no udev rule installed, and the colour-health check. Not
  "camera truth".
- Phase 2 shrinks accordingly, and U1/U2 stop gating it.

### Still true, and new

The D455 is negotiating **USB 2.1** here — `/sys/bus/usb/devices/1-4/speed` reads 480,
not 5000. That caps colour resolution and frame rate and will bite during recording,
especially once `fps` is a measured number under invariant 10. Cable or port, not
software.

The emitter figure did not reproduce either: the kit measured Laplacian variance 413→79
with the projector off; here it is 502.8 vs 504.1, i.e. noise. The projector *was* firing
(the IR imager saw it). The kit's number was very likely taken on the IR node — the same
mix-up, independently. Also, the emitter only fires while a DEPTH stream runs, so a
colour-only pipeline is unaffected regardless. `disable_emitter()` stays because it is
free and persists, but 413→79 is not a number this rig reproduces.

## De-risking facts (all verified 2026-08-27)

- **lerobot 0.5.1 and 0.6.1 have byte-identical signatures** for `delete_episodes` and
  `LeRobotDataset.create` / `add_frame` / `save_episode`. The only 0.6.1-only surface we
  need is `lerobot.scripts.lerobot_rollout`. That single module is the entire reason for
  an interpreter split.
- **Haller's hand-rolled pop solved a problem `delete_episodes` already solves** — in
  *both* versions. `recorder.py:539 delete_last_episode` plus its stats-recompute and
  five 409 refusals are ~120 lines answering a question upstream answers. It goes.
- **`vr_teleop/wire.py::normalize_frame` already accepts both spellings** — the kit's
  `xr_frame` (WebXR controllers + indexed gamepad) and Haller's `vr_keypoints`. The
  Haller HUD can drive the kit's teleoperators with **no client change**. This was built
  in the 08-22 unification and is the single largest piece of luck in this port.
- **The kit's data layer passes 38/38 hardware-free checks today**
  (`~/vr-teleop-kit/tools/smoke_test_dataui.py`). It is portable *now*, without servos.
- **Both repos resolve `HF_LEROBOT_HOME` identically.** Datasets already co-locate at
  `~/robot-data/lerobot`. No migration, no copy, no path translation.
- **Haller's recorder schema already follows the arms the rig has** (`9af6a02`) —
  `_state_names`, the calibration block and `_build_frame` enumerate sides from the arm
  manager. Half of "the schema follows the selection" is already true.
- **Duplicate-camera-key collision is already a load error** (`config.py:_cameras_from`)
  and is checked against the runtime record set at `start_episode`.

## Scope correction — bimanual SO-101 is a BUILD, not a port

The kit's `lerobot/bi_quest_teleop.py` targets the **DK1**: 6-DoF per arm,
`{left,right}_joint_{1..6}.pos`. Oscar does not have that hardware and will not.
The kit's SO-101 support (`so101_quest_teleop.py`, `single_arm_quest_teleop.py`,
`ik/so101_ik.py`) is **single-arm only**.

So there is nothing to copy. Stock `BiSOFollower` exists in both lerobot versions; the
work is marrying `BiQuestTeleoperator`'s per-hand structure to `SO101IKSolver` — i.e.
Haller's own `vr_teleop/teleop.py` + `ik/decoupled_ik.py`, which already run two hands
into two 5-DoF arms. Budget it as new work with new tests, not as a port with a diff.

## Decisions of record (Oscar, 2026-08-27)

1. **The desktop is the rig.** Record *and* train on the desktop (RTX 4080 SUPER).
   The Jetson is a later target and is not a constraint on any decision below.
2. **The arm set is a RUNTIME selection** — left only / right only / both, and the same
   three in sim. Not a config file, not a restart. The dataset schema follows the
   selection: a solo dataset simply has no columns for the absent side.
3. **The incoming third camera is WRIST/GRIPPER-mounted** (robot-egocentric). It lands
   in the frozen `left_wrist` key. It is **not** an operator head view — the roadmap
   refuses to record one, because a policy trained on the operator's viewpoint inherits
   the operator's self-occlusion and cannot see past its own hands at inference.
4. **Build everything as far as possible without hardware.** Batch the hardware
   verification into ONE session when the new servos arrive. Sim carries verification
   meanwhile — `SimArmHandle` publishes through the identical path, so anything that is
   true of the commit chain in sim is true of it on hardware.
5. **Recording stays IN-PROCESS.** `/estop` walks every motor in-process; that shape was
   forced on 2026-08-21, when an overloaded shoulder aborted lerobot's bulk
   `disable_torque()` mid-sweep and left four joints energised. A child process cannot
   be allowed to own the bus. This kills the kit's subprocess `record_runner` shape
   outright — its *logic* ports, its *process model* does not.
6. **Interpreter split.** The serving venv (`~/venvs/haller-hmi`, lerobot 0.5.1) stays
   where it is. Detached runners — train, rollout, export — get a new
   `~/venvs/haller-lab` with lerobot 0.6.1. Verified today: `haller-hmi` = 0.5.1,
   `vr-teleop-kit/.venv` = 0.6.1; **`~/venvs/haller-lab` does not exist yet** and is a
   Phase-0 deliverable.

## Unverified — every one of these is a Phase-0 go/no-go

Nothing below is assumed in the plan. If one comes back NO-GO, the phase that depends on
it stops and this document gets amended.

| # | question | how it is answered | blocks |
|---|---|---|---|
| U1 | Does `pyrealsense2` install into `~/venvs/haller-hmi` (`--system-site-packages`) without shadowing or being shadowed? | install + import + one frame | ~~P2~~ **SUPERSEDED** |
| U2 | Can `haller_hmi` hold the D455 colour node through librealsense while anything else on the box wants it? The kit says the colour node **cannot** be shared. | open under librealsense with the ROS node up, then down | ~~P2~~ **SUPERSEDED** |
| U3 | What is the **measured** end-to-end sample rate on real hardware with Feetech round trips in the tick? `fps` must be measured, never declared. | instrument the new sampler, 60 s run | P1, P7-H |
| U4 | Does `delete_episodes` (0.5.1) handle **our** v3.0 layout — one video file per episode, resumed metadata across several `meta/episodes/chunk-*/file-*.parquet`? Identical signatures are not identical behaviour on our data. | record → delete → record → load + resume, on a scratch repo | P4 |
| U5 | Does a dataset **written** by lerobot 0.6.1 (haller-lab) load in 0.5.1 (haller-hmi), and vice versa? Both claim v3.0; that is a claim, not a test. | round-trip both directions on a scratch repo | P6 |
| U6 | Identity-based Quest pairing — flagged hardware-unconfirmed in the 08-22 unification and still unconfirmed. | the batched hardware session | P7-H |
| U7 | ACT training throughput on the 4080 SUPER from within `haller-lab` (the kit trained; not from this interpreter). | one short run, steps/s | P6 |
| U8 | The wrist camera and the new servos are **not on hand**. Mount, cabling, and whether the wrist view is usable at 640×480 are all open. | the batched hardware session | P7-H |

## Invariants — break any of these and the port failed

Carried forward from PLAN-2026-08-22 (1–9 there survive verbatim in meaning) plus four
this port adds. These are not goals. They are the pass/fail.

1. **Zero-error handover.** The mapper re-anchors every frame until a side is DRIVING,
   and once more on the first driving frame. `test_gate_error_stays_zero_through_the_countdown`
   and `test_handover_starts_from_the_hand_where_it_is_now` survive untouched.
2. **The acquisition ramp is load-bearing** — `MATCH_DWELL_MS`, `ACQUIRE_RATE_DEG_S`,
   `ACQUIRE_RAMP_MS`, `_ramp_cap`. A recorder that needs a faster tick does not get it
   by shortening the ramp.
3. **Collision toggle semantics unchanged**: off still measures (`slack_m` keeps
   updating); `available:false` is one-way and enabling on it is 409. `min_tip_z` /
   `min_wrist_z` workspace floors stay ON when the guard is off.
4. **E-STOP path untouched**: `POST /estop` + B/Y in-headset, walking every motor
   in-process. `/arm/{id}/home` stays REFUSED while a session owns the arms.
5. **Controller mapping unchanged**: per-side grip = dead-man, trigger = gripper
   (1−trigger), B/Y = E-STOP, left-stick hold ≈0.8 s = in-session home, left-stick short
   click on RELEASE = view cycle, A/X hold 500 ms = record toggle.

   **The modal end-of-take exception** (ruled 2026-08-27; shipped behaviour since 08-22,
   written down here because it had been living as a consequence nobody recorded). While
   the end-of-take prompt is open it OWNS the sticks — left click = keep, right click =
   redo — and the in-session home hold is refused. That is invariant 5 being honoured,
   not bent, and it holds on two conditions:

   - **the refusal is felt and seen** — the weak 0.2/60 tick `resetArms` already uses,
     plus a HUD line saying home is refused mid-take. Homing through the tail of an
     episode would corrupt a take the operator may be about to keep, so it stays refused;
     what it must never do is fail *silently*.
   - **the prompt is bounded and modal**, so the exception cannot leak into normal driving.

   If either stops being true the exception lapses. The reasoning is asymmetric on
   purpose: banking a take you did not mean to costs one reject mark in a file that
   already carries 11 of 46, whereas asking for home and not getting it is the direction
   that hurts. For the same reason there are no HOLD variants inside the prompt — two
   gestures with the same dwell separated only by modal state is the class of thing that
   passes testing and fails on someone's face in a headset. "Go again" is a MODE instead:
   every decision returns to ARMED.
5b. **Anything that drops torque goes through `_release_torque_per_motor`.** lerobot's
   bulk `disable_torque()` raises on the first refusal and strands the rest energised;
   that is the 2026-08-21 incident, and it has now arrived twice by two different roads.
   No exceptions, including inside a safety check whose whole purpose is to make the arm
   safe.

5c. **Every heavy import happens on the MAIN thread, at build time — never lazily
   inside a request handler.** FastAPI runs plain `def` handlers on anyio worker threads,
   and a first `import pandas` on one worker while another reads a parquet **SEGFAULTS
   the process**. Reproduced 3/3 on this box with no Haller code involved; a main-thread
   pre-import survives 5/5.

   This is an invariant rather than a performance note because of *which* process it is.
   The HMI owns the Feetech bus with torque enabled, and SIGSEGV is not an exception:
   no `finally`, no `atexit`, no `_release_torque_per_motor`, and `/estop` unreachable
   because the server is gone. The arms stay energised holding their last goal, and the
   thing that ends that is a human at the bench PSU. **A page load must never be able to
   take the arms down.**

   Applies to any plain-`def` route doing a deferred heavy import — pandas, pyarrow,
   anything with C extensions. The cost of closing it is a fraction of a second at
   startup.

6. **Single-arm sessions**: the absent side never acquires, is never written, reports
   `reason:"no_arm"`, cannot be homed. Now also: produces a dataset with no columns for
   that side, distinguishable by names alone.
7. **WS-disconnect grace**: a disconnect STARTS the window and never re-stamps; a frame
   clears it.
8. **State and action share one moment.** One sampler owns the tick. Every recorded row
   is state and action read inside the same tick, or the row is not written. *(new)*
9. **A degraded read is a dropped frame, not a recorded one.** No `isAvailable()`
   failure, no comm failure, no stale side may reach a parquet row. *(new)*
10. **`fps` in `info.json` is measured, or the episode does not open.** *(new)*
11. **Recorded pixels are never cosmetically altered.** `facing:"operator"` mirroring
    stays display-only, as it is today. The librealsense switch is a *correctness* fix
    to the source, in the same class as `flip_method`, not a look-nicer filter. *(new)*
12. **Recording stays in-process** — see decision 5. No child process owns the Feetech
    bus, ever.
13. **Review is a sidecar.** Marking an episode bad and destroying it are two acts.
    `review.json` at the dataset ROOT (not under `meta/`, which belongs to lerobot's
    loaders); training passes the kept set as `--dataset.episodes`; pruning is a
    separate explicit export. Ported from the kit unchanged in principle. *(new)*

## Who is building this

Four sessions, disjoint file ownership, each fanning out to its own Workflow subagents.
The integrator holds `server.py` (both backend tracks need routes mounted there), git
integration, verification, and cross-track contract arbitration.

| track | session | territory | phases below |
|---|---|---|---|
| **A** realtime core | `haller-ws-d7` | `arm`/`human_teleop`/`telemetry`/`recorder`/`cameras`/`collision`/`config`/`realsense`.py, `vr_teleop/**`, `sim/**`, `tick.py` (new), config yamls | 1, 2, 3, 7 |
| **B** Lab backend | `haller-ws-ea` | NEW `lab/**`, `api/**`, `runners/**`, `lease.py`, `tests/lab/**` | 4, 6 (backend) |
| **C** Lab frontend | `haller-ws-fd` | all of `hmi/frontend/**` except Track D's files | 5, 6 (frontend) |
| **D** headset client | `haller-ws-1a` | `VRTeleopPanel.tsx`, `lib/vrTeleop.ts`, `lib/humanTeleopClient.ts`, `app/teleop/vr/**` + their tests | 5 (HUD half) |
| — integrator | `haller-ws-13` | `server.py`, this document, git | — |

The phase numbers in this table are **this document's**, and they are authoritative.
Track briefs sent by the integrator used their own local numbering; where the two
disagree, the phase NAMES below win.

## Delivery — eight phases

| P | phase | delivers | risk | gated on |
|---|---|---|---|---|
| 0 | **Probes** | U1–U5, U7 answered with numbers; `~/venvs/haller-lab` built at lerobot 0.6.1; scratch repo harness | **LOW** | — |
| 1 | **One sampler** | the tick that owns state+action+pixels; `isAvailable()` honoured, degraded read → dropped frame; measured fps; mechanisms 1–3 dead | **HIGH** | U3 |
| 2 | **Camera truth** | the IR emitter disable, the no-udev-rule fallback, and the colour-health check — `realsense.py`'s *smaller* justification per §Consequences. The mast-cam OpenCV path is CORRECT and is left alone; `pyrealsense2` stays a soft optional dependency | **LOW** | — |
| 3 | **Runtime arm set** | left / right / both selectable at session start in real and sim; schema follows the selection end to end | **LOW** | — |
| 4 | **Episode lifecycle** | hand-rolled pop deleted for `delete_episodes`; `review.json` keep/reject sidecar; the kit's per-episode grading (grasp, motion, …) with ONE implementation behind both CLI and UI | **MED** | U4 |
| 5 | **Collection workspace** | cockpit dataset tab grows the review/grade surface; HUD gets episode + frame counters and the save/discard choice, in the existing menu style | **LOW** | P4 |
| 6 | **Train + rollout** | detached runners on `haller-lab`; export of a pruned dataset; `lerobot_rollout` reachable; cross-version round-trip proven | **MED** | U5, U7 |
| 7 | **Bimanual build** | `BiQuestTeleoperator`'s per-hand structure married to `SO101IKSolver` over stock `BiSOFollower`; sim-verified through `SimArmHandle` | **HIGH** | P1, P3 |

**Sequencing amendment (2026-08-27, Track A's argument, adopted).** The rollout
ingest runs **after P1–P2 and BEFORE P3 and P7**, scoped SOLO:

    P1 -> P2 -> rollout ingest (solo) -> P3 -> P7

It must be after P2 because it needs the settled commit chain and the TickBus's
measured rate; building it against the three-sampler world means building it twice.
It must NOT wait for P7 because **the rollout ingest is what closes the loop, and
nothing else does.** Until a policy trained on the new data drives the arm, this
port's whole thesis — that the kit's acquisition discipline comes in and the data
gets good — is unfalsified, and P3 doubles the schema and P7 doubles the arms on top
of an unverified premise. Closing it solo and early is also cheap: the kit's proven
end-to-end path *is* single-arm (its bimanual file is DK1), so the first useful
rollout is solo by construction, and gating the one proven path on the one brand-new
path that is waiting on hardware inverts the risk order.

Not blocked on training: the ingest is verifiable against a fake action source
streaming a recorded episode back — no policy, no checkpoint, no GPU.

**A rollout below rate is REFUSED, not warned** (corrects an earlier integrator call).
A policy trained at 30 Hz and run at 4.8 Hz applies action deltas sized for 33 ms over
208 ms — a different dynamical system, and the same class of error as a declared-not-
measured fps. Refusing one while tolerating the other is incoherent. Same 90% constant
as the record gate, refuse below it, alert mid-run, explicit override for someone who
means it, measured rate stamped into the run record either way. The kit's real failure
was not that a 4.8 Hz run happened; it was that "success" was reported with that number
attached to nothing.

**Policy actions are DEGREES on every joint**, declared once at lease time, never
per-message. Normalising in the child would put a calibrated range on both sides of an
interpreter boundary — two things that must match exactly and eventually will not — and
a per-message unit that can flip mid-rollout is a footgun with no legitimate caller.
Convert at the door, as `wire.py::normalize_frame` already does. Related defect being
fixed with it: `human_teleop.py::_to_degrees` clamps the gripper to [0,1] and scales
onto the calibrated range, so a policy emitting degrees collapses to two values —
failing toward DROP THE OBJECT. That [0,1] scaling moves out of the session into the VR
converter that produces it, so nothing inside the session knows one input dialect
normalises one joint.

**Constants tuned against the Quest's cadence must be re-read once a second source
exists.** `frame_age_ms_loss = 700` is ~42 missed frames at 60 Hz but three frames at a
legitimate 4.8 Hz, so a healthy slow policy would trip tracking-loss and be demoted
mid-rollout with a symptom pointing nowhere near the cause. Staleness budgets are
relative to the source's DECLARED rate.

**P7-H — the hardware batch.** Not a phase; a gate. Per decision 4 everything above is
built and verified in sim first, then ONE hardware session with the new servos clears
U3 (re-measured on the real bus), U6, U8, and the bimanual real-arm pass. Nothing waits
on it that does not have to.

Risk notes worth stating rather than discovering:

- **P1 is the phase that can break teleop.** It touches the commit loop that invariants
  1–7 live in. It ships with the existing tests green *before* any recorder change, so a
  regression is attributable.
- ~~**P2's real risk is U2, not the import.**~~ **WITHDRAWN with the D455 retraction.**
  Colour-node sharing only ever mattered because the recorder was going to be moved onto
  librealsense; it is not, so there is no arbitration to design. P2's residual risk is the
  **USB 2.1 negotiation** (480 Mb/s, not 5000), which caps colour rate and is cable-or-port,
  not software.
- **P7 is high risk for schedule, not safety** — it is new code against hardware that has
  not arrived. It is last for that reason and nothing downstream depends on it.

## House rule — cadence-coupled constants

**A constant expressed in ticks or frames is calibrated for exactly one cadence and
lies at every other.** Found by Track A's sweep, 2026-08-27, after `frame_age_ms_loss`
turned out to mean 42 missed frames at 60 Hz and three at 4.8 Hz.

The model to copy is already in the tree: `lpf_tau_s` is a time constant in seconds and
the loop derives `alpha = 1 - exp(-period / tau)` per tick, commented
"frequency-independent". Every new constant that governs motion, staleness or a safety
stop takes that shape.

Found by the sweep, all reachable through `POST /teleop/human/start {hz}` today:

| constant | unit | at 60 Hz | at 10 Hz |
|---|---|---|---|
| `_rate_cap_deg_per_tick = 4.0` | per tick | 240 deg/s (headroom, as designed) | **40 deg/s — now the binding limit** |
| `MAX_CONSECUTIVE_TICK_ERRORS = 50` | ticks | 0.83 s to stop | **2.5 s at 20 Hz, 10.4 s at 4.8 Hz** |
| `pose_filter_alpha = 0.8` | per frame | ~83 ms | ~1 s from a 4.8 Hz source |

The first **degrades invariant 2**, and needs no exotic setup to reach. `_ramp_cap` uses
two unit conventions in one function: its FLOOR converts properly
(`_acquire_rate_deg_s * period`) while its CEILING is the raw per-tick 4.0. So the
acquisition ramp spans 12:1 at 60 Hz, 4:1 at 20 Hz and 2:1 at 10 Hz — at low hz it
quietly stops being a ramp. Fixing it makes the two halves of that function agree, which
they currently do not.

Both corrections are **no-ops at hz=60** — `240 * period` is bit-identical to 4.0 there —
so they land standalone, ahead of the tick restructure, with the no-op pinned. Folding
them into Phase 2 would have destroyed regression attributability for both.

Cleared as already correct, so the sweep is auditable rather than a list of hits:
`ACQUIRE_MS`, `MATCH_DWELL_MS`, `ACQUIRE_RAMP_MS`, `ACQUIRE_RATE_DEG_S`,
`_ws_disconnect_grace_s` — all in physical time or rate units, measured against `now`.

## Baseline to protect

- **645 backend pytest pass, 1 xfailed** (593 pre-port + 52 equivalence).
- **186 frontend vitest pass.**

Backend incantation (the venv fights you — plain `activate` lacks rclpy):

```
source ~/venvs/haller-hmi/bin/activate-haller-hmi
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl python -m pytest -p asyncio <scope> -q
```

Frontend: `npm test` (vitest) plus the repo's typecheck, scoped where possible.
`scripts/vr_smoke.py` against a cold sim backend remains the integration pass, not a
per-session obligation.

## Rules for every session on this branch

- Work in `/home/odesha/haller_ws` on `feat/kit-port` — verify with
  `git branch --show-current` before the first edit; anything else, STOP and report.
- **Commit your own territory, scoped.** `git add <explicit paths>` only — never `-A`,
  never `rebase`/`merge`/`branch`/`tag`, which stay with the integrator. Four tracks
  funnelling every commit through one session would recreate the bottleneck the extra
  sessions exist to remove. (This supersedes the original "no git write operations";
  that rule was written for the single-session plan.)
- **`~/vr-teleop-kit` is read-only.** Read it, import from its venv, never write it.
  Checked by both `git status` and mtime.
- **Touch only files in your ownership list.** A needed change outside it is reported,
  not made. `hmi/backend/tests/equivalence/**` is read-only to *every* track — it is the
  oracle that judges the port, so a track that believes a test there is wrong escalates
  rather than edits. The integrator may grant a narrow exception naming the file, the
  test and the intended post-state.
- **`server.py` belongs to the integrator.** Both the realtime and Lab tracks need routes
  mounted there; report the mount you need rather than making it.
- Match the house style: terse comments that state **constraints**, not history, with
  measured numbers where they exist. Read the neighbours before writing.
- Never add `Co-Authored-By` or any similar trailer, anywhere.
