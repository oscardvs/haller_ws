# Dataset collection (SO-101)

How to record a LeRobot dataset on Haller, what exactly ends up in it, and the
handful of things about that schema you will otherwise get wrong.

There are **two** recording paths, and they are for two different rigs. Pick by
hardware, not by preference:

| path | rig it is for | who drives the arm | HMI |
|---|---|---|---|
| **HMI-integrated recorder** — `POST /record/start` | **bimanual**: two SO-101 *followers* (real or MuJoCo sim), plus base and cameras | human-pose or [Quest VR](../../hmi/QUICKSTART-QUEST.md) teleop, in-process | **must be running** |
| `scripts/record_dataset.sh` — wraps `lerobot-record` | **single arm**: one SO-101 leader driving one SO-101 follower | the physical leader arm | **must be stopped** |

The bimanual path is the one Haller actually collects on. The reason is
structural, not a preference: **both** of Haller's arms are followers. There is
no leader pair to ALOHA-teleoperate them with, so `lerobot-record
--teleop.type=so101_leader` cannot capture a two-arm demonstration on this
hardware at all. The record loop therefore lives where the teleop already
lives — inside the HMI backend, in
[`hmi/backend/haller_hmi/recorder.py`](../../hmi/backend/haller_hmi/recorder.py),
whose module docstring is the schema's specification.

It also means no new serial traffic. During teleop the half-duplex Feetech bus
is already written at ~60 Hz and read at telemetry rate; the recorder opens no
third bus reader and instead samples the streams that exist — telemetry frames
for `observation.state`/`observation.effort`/`observation.base`, the teleop
session's committed goals for `action`, the camera grabber threads for images.

---

## The bimanual path

### Prerequisites

- A config with two arms. Sim:
  [`hmi/backend/config.bimanual-sim.yaml`](../../hmi/backend/config.bimanual-sim.yaml).
  Real: the equivalent two-arm real config (see [`so101-arm.md`](./so101-arm.md)).
- **A teleop session running.** The recorder reads `action` from the teleop
  session's committed joint targets. With no session running, `action` falls
  back to the measured position joint-by-joint, which produces a technically
  valid but useless dataset where action ≡ state.
- At least one camera that can hand over RGB. Cameras marked
  `source: placeholder` are skipped; so are cameras with `record: false`.
- HF CLI authenticated if you intend to push (`hf auth login --token "$HUGGINGFACE_TOKEN"`).

### Driving a take

Three surfaces, one recorder:

- **Cockpit** (`http://localhost:3000` → Dataset tab, or the Record button in
  the command bar). Draft the task string and HF username once; both persist in
  the browser so the headset can start takes from them.
- **In the headset** — hold **A or X** for ~0.5 s to start, hold again to
  stop-and-save. Hold-gated so a thumb brush cannot toggle a take. The HUD shows
  `● REC <frames>`.
- **curl**, which is what the other two do:

```bash
curl -X POST http://localhost:8000/record/start \
    -H 'Content-Type: application/json' \
    -d '{"repo_id": "myuser/haller_pick_red_cube",
         "task": "Pick the red cube and place it on the blue pad"}'

curl http://localhost:8000/record/status

# save the take...
curl -X POST http://localhost:8000/record/stop \
    -H 'Content-Type: application/json' -d '{"save": true}'
# ...or throw it away
curl -X POST http://localhost:8000/record/stop \
    -H 'Content-Type: application/json' -d '{"save": false}'
```

One start/stop pair is **one episode**. Starting again with the same `repo_id`
appends the next episode to the same dataset; a different `repo_id` closes the
open dataset out and opens the new one.

**A take of fewer than 2 frames is refused, not saved.** `last_error` says so.
This is not tidiness: lerobot 0.5.1 cannot compute video statistics over a
one-frame episode, so it omits that episode's `stats/observation.images.*` keys
while every other episode has them, and the ragged result kills the metadata
flush that writes `meta/episodes/` — taking the *whole dataset* with it, not
just the stray take. A mis-click that opens and closes a take in the same
instant is therefore dropped on the floor, loudly.

`GET /record/status` reports:

| field | meaning |
|---|---|
| `recording` | a take is open |
| `repo_id` / `task` | what it is being written into, under what instruction |
| `episode_frames` | frames written so far this take |
| `skipped_frames` | ticks seen but **not** turned into a frame — a required camera had no fresh image, or an arm's telemetry was missing. Nonzero means the take has gaps, and `observation.wall_clock` says where |
| `auto_scored` | whether a task monitor is attached at all (sim only) |
| `success` | tri-state: `null` = nobody scored this, `false` = scored and did not succeed, `true` = succeeded |
| `success_frames` | how many frames the success predicate held for — distinguishes a clean place from a cube that qualified for three frames and rolled off |
| `last_error` | last per-frame failure, if any |

### Where it lands, and at what rate

`$HF_LEROBOT_HOME/<repo_id>`, defaulting to
`~/.cache/huggingface/lerobot/<repo_id>`. Video is **h264** with streaming
encoding (frames compress as they arrive, so memory stays flat over a long take
and `save_episode` at stop time is near-instant). The libsvtav1 default was
rejected: a software AV1 encoder cannot keep up with realtime multi-camera
capture on the machines this runs on.

**Every episode gets its own video file per camera** — `video_files_size_in_mb`
is pinned to 0 in `meta/info.json` so lerobot rotates rather than packs. Stock
lerobot 0.5.1 packs several episodes into one file, and the packer
(`video_utils.concatenate_video_files`) remuxes the appended episode's packets
without re-basing their timestamps, so the mp4 muxer rejects them:

```
av.error.ValueError: [Errno 22] Invalid argument
[mp4] Application provided invalid, non monotonically increasing dts ...
```

That raise lands *after* the frames are written but *before* the episode
metadata is, which loses the take, freezes `total_episodes`, and leaves the
next take reusing the same episode index. One file per episode is a valid v3
layout — each episode records its own chunk/file index — and it removes the
broken path rather than hoping to miss it.

**fps is `telemetry.hz`.** 30 in the sim config, matched to the Quest's ~30 Hz
publish rate and LeRobot's SO-101 convention. The real rig stays at **20 Hz** —
the half-duplex Feetech bus already interleaves 60 Hz writes with those reads.

An **E-STOP or a dead-man release mid-take is safe.** If the teleop session was
running and stops, the recorder notices and saves the episode up to that frame
rather than appending a tail of frames where `action` == measured with the arms
torque-off.

---

## The schema

Frozen. A **bimanual sim** episode at fps 30 produces exactly this and nothing
else:

| feature | dtype | shape |
|---|---|---|
| `observation.state` | float32 | (12,) |
| `action` | float32 | (12,) |
| `observation.effort` | float32 | (12,) |
| `observation.base` | float32 | (2,) |
| `observation.wall_clock` | float32 | (1,) |
| `next.reward` | float32 | (1,) |
| `next.done` | bool | (1,) |
| `observation.images.top` | video | (720, 960, 3) |
| `observation.images.left_wrist` | video | (480, 640, 3) |
| `observation.images.right_wrist` | video | (480, 640, 3) |

plus `task` (str, the natural-language instruction), and LeRobot's own
bookkeeping columns (`timestamp`, `frame_index`, `episode_index`, `index`,
`task_index`) which the recorder never writes itself.

**On a real rig: identical, minus `next.reward`/`next.done`** (nothing on the
real rig can decide whether the task was solved — see
[`haller_scoring`](#haller_scoring-in-infojson) below), **plus whatever cameras
that config records.**

Column layout for the three 12-vectors is the same in all of them: canonical
SO-101 motor order, **left arm then right arm** —

```
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

— so `observation.state[i]`, `action[i]` and `observation.effort[i]` are the
same joint, and the `names` metadata spells them out as
`left_shoulder_pan … right_gripper`.

`observation.base` is `(v, omega)` for the 3-wheel differential drive.
`observation.wall_clock` is the **real** capture time of the frame, measured in
**seconds since that episode started**: LeRobot's own `timestamp` is synthetic
(`frame_index / fps`), so a skipped tick leaves no gap there, and this channel is
the only way to see real sampling holes after the fact.

It is deliberately **not** a Unix timestamp. A float32 carries 24 bits of
mantissa, so one representable step at a 2026 epoch (`1.79e9`) is **128
seconds** — stored absolutely, a three-minute take collapses to two distinct
values and every consecutive difference is zero, which destroys the only thing
the column is for. Measured from episode start it keeps ~10 µs of resolution for
any take under three hours. The absolute start time of the most recent take is
in the `haller_wall_clock` block of `meta/info.json`, so real time is still
recoverable:

```python
abs_t = info["haller_wall_clock"]["episode_started_unix_s"] + wall_clock
```

`observation.lidar` (fixed-length `/scan`) is the one remaining v0.1 slot — an
additive key, not a change to any of the above.

---

## Seven things that bite

### 12 dims, not 14

An SO-101 is **6 DoF including the gripper**, so a bimanual pair is 12. It is
easy to reach for 14 by analogy with ALOHA (7 per arm) or by counting the
gripper separately from the arm — both are wrong here, and every public bimanual
SO-101 dataset is 12-dim. If a config check or a policy head wants 14, the
mismatch is in the assumption, not in the data.

### `action` is not the next `state`

`action` is the **committed** teleop target — the number actually written to the
servo, after the low-pass filter, the velocity ramp and the collision guard have
all had their say. It comes from the teleop session's `goal_deg`, which is what
`send_goal` reports it *actually sent*, not what the retargeter asked for (the
per-call rate cap can legitimately command less than requested).

`observation.state` is what the servo **achieved**.

So the gap between them is real physical information — load and tracking error —
and not noise to be smoothed away. In particular it does not vanish at steady
state when the arm is holding something.

### `observation.effort`, and what 0.0 means

**Unit: a dimensionless *signed fraction* of that joint's own torque limit,
clipped to ±1.** Sign is the drive direction; `|v| → 1` means the joint is
saturated, i.e. stalled or gripping.

| rig | source |
|---|---|
| real | `Present_Load` (sign-magnitude, sign in bit 10) `/ 1000` — the STS3215 reports load as signed per-mille of maximum torque, really the PWM duty it is applying |
| sim | `actuator_force / actuator_forcerange[:, 1]` — N·m over the MJCF's declared saturation bound |

These are **not the same physical quantity and cannot be unit-matched.** A
per-mille PWM duty and a newton-metre do not convert into one another without a
per-joint motor model nobody has measured. That is precisely why both are
normalised to their own saturation limit: so one dataset column means one thing
regardless of which rig recorded it. It is comparable *within* a joint across
takes and across rigs; it is **not** a torque you can integrate into work.

Two sharp edges:

- **0.0 means both "no contact" and "register unreadable."** A flat-zero column
  means there was no effort channel on that take, not that nothing was ever
  touched. (`Present_Current` was rejected as the source: its mA-per-count scale
  appears nowhere in the installed lerobot or scservo_sdk, so using it would
  mean hard-coding a datasheet constant.)
- **Sim effort is only meaningful with torque ON.** Disabling torque zeroes the
  actuator's `gainprm` but leaves its bias term intact, so a limp joint away from
  zero reads as saturated at −1.0. The sim handle reports 0.0 in that case
  rather than the raw number — the same thing a real arm with torque off does.

### Grasp detection

You can partly detect a grasp from columns you already have. When the jaw stalls
on an object, `action[gripper] − state[gripper]` becomes a **standing tracking
error**: the teleop keeps commanding closed, the servo cannot get there.

Effort matters because that error **saturates** once stalled — the commanded
value hits the joint limit and the difference stops growing — while effort keeps
carrying force and slip information after that point.

The usable form is a hysteresis gate on both:

```
grasped = (|action[gripper] − state[gripper]| > θ_e) ∧ (effort[gripper] > θ_i)
          sustained for ≥ k frames
```

θ_e and θ_i are **calibrated, not guessed**: record one free close (jaw shuts on
nothing) and one close on a cube, and take thresholds between the two
distributions. They differ per rig, and the sim's differ from the real arm's for
the unit reason above.

### Units are DEGREES

`observation.state` and `action` are joint **degrees** (lerobot's
`MotorNormMode.DEGREES`, `use_degrees=True` at connect). Every public LeRobot
SO-101 dataset is in normalised **[-100, 100]** (with the gripper in [0, 100]).

This **does not affect training on your own data** — LeRobot normalises from
per-dataset statistics, so a policy trained only on Haller episodes never sees
the raw scale. It **silently corrupts co-training** with public data, where two
differently-scaled action spaces get averaged into one head.

Mitigation, already shipped: every take writes a `haller_joint_calibration`
block into the dataset's `meta/info.json`, so the affine map stays recoverable
after the fact. It is keyed exactly like the state columns
(`left_shoulder_pan`, …) and carries per joint: `range_min_ticks`,
`range_max_ticks`, `homing_offset`, `drive_mode`, `resolution`, `deg_per_tick`,
`norm_mode`, `min_deg`, `max_deg`, and a `source` field (`feetech_calibration`
on a real arm, `declared_joint_range` on a sim arm, which has no Feetech
calibration and leaves the tick-domain fields null).

The map, verbatim from that block's own note:

```
raw   = deg * (resolution - 1) / 360 + (range_min_ticks + range_max_ticks) / 2
norm  = (raw - range_min_ticks) / (range_max_ticks - range_min_ticks),
        scaled per norm_mode (×200−100 for RANGE_M100_100, ×100 for RANGE_0_100),
        negated (or 100 − x for the 0..100 form) when drive_mode is 1
```

> **The subtlety worth naming.** `deg_per_tick` in the block is
> `360 / (resolution - 1)` — 360/4095 — because that is the factor lerobot's
> DEGREES mode actually applies to recorded positions. The `min_deg`/`max_deg`
> clamp limits are the HMI's own, computed with `DEG_PER_TICK = 360/4096`. Both
> constants are in the block on purpose. The 0.02 % difference is irrelevant for
> a safety clamp and very relevant if you invert the wrong one.

Note also that `norm_mode` is per joint: on SO-101 the gripper is `RANGE_0_100`
and the other five are `RANGE_M100_100`, so a single dataset mixes both forms.

The block describes the rig as of the **most recent take** appended to the
dataset — LeRobot v3.0 has no per-episode slot for free-form metadata, so a
mid-dataset recalibration is not representable. If you recalibrate, record into a
new `repo_id`.

### The camera set is 3 of 5, deliberately

The bimanual sim **renders five views and records three**:

| camera `id` | MJCF camera | resolution | `record` | `dataset_key` |
|---|---|---|---|---|
| `overshoulder_sim` | `overshoulder` | 960×720 | `false` | — |
| `threequarter_sim` | `threequarter` | 960×720 | `true` | `top` |
| `overhead_sim` | `overhead` | 640×480 | `false` | — |
| `wrist_left_sim` | `left_wristcam` | 640×480 | `true` | `left_wrist` |
| `wrist_right_sim` | `right_wristcam` | 640×480 | `true` | `right_wrist` |

Three reasons, all of them training decisions rather than plumbing:

- **π0.5's pretrained camera slots are `base_0_rgb` + `left_wrist_0_rgb` +
  `right_wrist_0_rgb`** — one base plus two wrists. Three keeps us in
  distribution; five is territory with no published run behind it.
- **`armnet/armnetbench_v01_lerobot_bimanual_so101` uses exactly
  `top` / `left_wrist` / `right_wrist`.** Matching those keys turns co-training
  into a rename instead of a re-record. See [public datasets](./public-datasets.md).
- **Every recorded camera is a *required* camera.** The frame builder drops the
  whole tick when **any** recorded camera has no fresh image, so each extra
  channel costs sample rate. Fewer required channels, fewer dropped ticks.

Two config fields control this, both on `CameraConfig`:

- `record: bool = true` — does this camera go into the dataset, or is it only
  there for the human to drive from? Default true, so a config written before
  the field existed records exactly what it always did.
- `dataset_key: str | null` — the feature name, i.e.
  `observation.images.<dataset_key>`, falling back to `id`. The split exists
  because the two names answer to different masters: `id` is the HMI's handle and
  has to be unique per rig (`wrist_left_sim`), while the dataset key has to match
  what the datasets you co-train with already call that view (`left_wrist`).
  Renaming the id instead would make the sim and real rigs' camera ids collide in
  the cockpit.

**Two recorded cameras resolving to the same key is a config-load error.** It is
not something LeRobot would catch: both would build the same
`observation.images.<key>` feature, the second spec would win, and every frame
would carry whichever camera was written last — a dataset whose `top` column is
silently half one view and half another. Cameras with `record: false` are
exempt, since a view that never reaches the dataset cannot collide inside it.

### `haller_scoring` in `info.json`

Alongside the calibration block, every take writes a `haller_scoring` block
recording **whether these episodes were machine-labelled, by what predicate, at
what thresholds**:

| field | on the sim rig | on a real rig |
|---|---|---|
| `auto_scored` | `true` | `false` |
| `reward_feature` / `done_feature` | `next.reward` / `next.done` | `null` |
| `monitor` / `predicate` | `TaskMonitor` / `haller_hmi.sim.task.cube_placed` | `null` |
| `predicate_note` | the predicate in words | — |
| `target_cube` | the watched cube, or `null` for "any" | — |
| `spec` | the full `SuccessSpec` — `zone_inset_m`, `lin_vel_eps`, `ang_vel_eps`, `settle_s`, `require_release` | `null` |
| `reward_shape` | `sparse` | — |
| `note` | — | says in words that the episodes are unlabelled |

The thresholds are in there because **the thresholds are the label definition.**
Re-scoring later, or comparing your success rate against someone else's, is
meaningless without them.

> **State it plainly:** on an unscored dataset the **absence of a reward column
> must not be read as "every episode failed."** Those are two very different
> facts that look identical in the schema, which is exactly why the block is
> written even when there is nothing to score. `success` in `/record/status` is
> tri-state for the same reason: `null` is "nobody looked", `false` is "looked,
> and it did not succeed."

`next.reward` is sparse — 1.0 on frames where the predicate holds, 0.0
elsewhere — and `next.done` is true on the final frame of each episode only. A
behaviour-cloning run can ignore both.

### Resume refuses on a feature mismatch, in both directions

Opening an existing dataset resumes it rather than clobbering it. Before the
first frame, the recorder compares the schema this rig produces against the
dataset's frozen features and refuses if **either** side has a key the other
lacks — features missing from the dataset, *and* features the dataset expects
that this rig does not record.

Both directions are checked because `add_frame` validates the key set both ways
too. Without the up-front check the take would run to completion and the operator
would discover at stop time, from an empty episode, that every frame had been
rejected.

Common causes: the recorded camera set or a `dataset_key` changed; or the dataset
was recorded on a rig with a different scorer (`next.reward`/`next.done` exist
only on a rig that can auto-score, i.e. the sim — so a sim dataset and a real
dataset can never be one dataset).

**The fix is to record into a new `repo_id`.** That is the safe move and costs
nothing but a name. If the two really must become one dataset, migrate the older
one to the current schema offline and merge afterwards.

An existing dataset that cannot be resumed at all is also a hard failure, not a
silent recreate: creating over it would destroy episodes someone already drove
for.

---

## Sanity-check the take

```bash
# lerobot 0.5.x calls this `lerobot-dataset-viz` (NOT `lerobot-visualize-dataset`,
# which no longer exists). It opens a rerun viewer.
lerobot-dataset-viz --repo-id myuser/haller_pick_red_cube --episode-index 0
```

Then read `meta/info.json` — it is a plain dict, and both Haller blocks survive
create → `save_episode` → resume → save → finalize:

```bash
python -c "
import json, pathlib
info = json.loads(pathlib.Path('~/.cache/huggingface/lerobot/myuser/haller_pick_red_cube/meta/info.json').expanduser().read_text())
print(info['fps'], info['robot_type'])
print(list(info['features']))
print(info['haller_scoring']['auto_scored'])
print(list(info['haller_joint_calibration']['joints']))
"
```

What to actually check:

- **`skipped_frames` was low or zero** during the take. If it was not, diff
  consecutive `observation.wall_clock` values to find where the gaps are.
- **`action` and `observation.state` differ.** If they are identical, no teleop
  session was running and you recorded the fallback.
- **Feature list matches the table above** — in particular that the three camera
  keys are the ones you meant.
- **`robot_type` is `haller_bimanual`** and `fps` is your `telemetry.hz`.
- **The `task` string is what you intended.**
- On sim, **`success` / `success_frames`** for the take, and `/sim/task/status`
  after it — see [sim.md](./sim.md).

## Push it to the Hub

The recorder writes locally only; publishing is a separate, deliberate step.

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("myuser/haller_pick_red_cube")
ds.push_to_hub(private=True)   # tags/license/branch are kwargs
PY
```

`hf upload myuser/haller_pick_red_cube <local_dir> --repo-type=dataset` works too
and is what [`runpod-inference.md`](./runpod-inference.md) uses for eval
artifacts, but `push_to_hub` also writes the dataset card.

---

## The single-arm path: `scripts/record_dataset.sh`

Still supported, and still the right tool **if and only if** you are recording a
one-arm task with a physical SO-101 **leader** driving a physical SO-101
**follower**. It cannot capture the bimanual rig. Its output is a plain
`lerobot-record` dataset: no `observation.effort`, no `observation.base`, no
`observation.wall_clock`, no Haller metadata blocks.

It needs **exclusive** control of `/dev/haller_arm_leader`,
`/dev/haller_arm_follower` and every camera device — all of which the HMI holds
while running — so **stop the HMI first**:

```bash
# Dev laptop: Ctrl-C the scripts/run_hmi.sh process
# Jetson:     sudo systemctl stop haller-hmi.service
```

The script refuses to start if anything still has those nodes open, and prints
which PID is holding them.

```bash
# Required: task description (1 sentence — the language instruction, also
# slugified into the dataset name). Optional: episode count (default 50).
scripts/record_dataset.sh "Grab the red cube and place it in the box" 20
```

It activates the HMI venv, resolves your HF user from `hf auth whoami`, prints a
confirmation banner and waits for `y`, then runs `lerobot-record` with
`--display_data=true` (a live `rerun` viewer). Per episode: leader → follower
teleop for `EPISODE_TIME_SEC`, then a `RESET_TIME_SEC` pause to reset the scene.
`lerobot-record`'s `push_to_hub` defaults to true, so the finished dataset is
pushed to `<HF_USER>/so101_<slug>` at the end.

| Variable | Default | Notes |
|---|---|---|
| `HF_USER` | from `hf auth whoami` | Override if your HF org differs from your username. |
| `DATASET_REPO` | `${HF_USER}/so101_<slug>` | Set explicitly to pick a custom repo. |
| `FPS` | 30 | Capture + control rate. |
| `EPISODE_TIME_SEC` | 30 | Max time per episode. |
| `RESET_TIME_SEC` | 5 | Pause between episodes for scene reset. |
| `CAMERAS_JSON` | base camera on `/dev/video0` | Full `lerobot --robot.cameras` dict. **This is the script's own camera config — it does not read `hmi/backend/config.yaml`.** |

During the run `lerobot-record` watches the keyboard: `→` ends the episode early
and saves, `←` ends it and discards for a re-record, `ESC` stops the run.

---

## What to do with the dataset

| Goal | Path |
|---|---|
| Train ACT from scratch (single task, ≥50 episodes) | `lerobot-train --policy.type=act --dataset.repo_id=...` — fits on a laptop GPU. |
| LoRA-finetune SmolVLA-base | `--policy.path=lerobot/smolvla_base --policy.peft_config.use_peft=true` on a 16 GB+ cloud GPU. |
| LoRA-finetune π0.5 (recommended VLA path) | See [`runpod-inference.md`](./runpod-inference.md) — `scripts/runpod/finetune_pi05_lora.sh <your-dataset>`. |
| Replay-eval an existing policy on your data | See [`runpod-inference.md`](./runpod-inference.md) — `scripts/runpod/replay_eval.py` runs π0.5 / pi0 against your dataset and dumps per-joint error + plots. |
| Finetune NVIDIA GR00T N1.7 | [Post-Training Isaac GR00T N1.5 for LeRobot SO-101 Arm](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) — official guide, ports cleanly to N1.7. |
| Bootstrap before you have your own data | [Public datasets](./public-datasets.md) — which SO-101 sets are usable, which are licence- or action-space-disqualified, and in what order. |
| Replay a take onto the sim arms to eyeball what you drove | `POST /teleop/sim/start` with `leader.source=replay` — see [`sim.md`](./sim.md). |

## Not yet shipped

- **`observation.lidar`** — the one unfilled v0 schema slot. Additive; wires
  through the same telemetry frame once the broadcaster surfaces `/scan`.
- **Closed-loop policy evaluation in the HMI** — a policy path, a deploy button,
  the existing E-STOP wired into the policy loop. The sim's auto-scorer is the
  half of this that exists: it can already label episodes, it just isn't driving
  a policy yet. See [`sim.md`](./sim.md) for why closed-loop rollout success —
  not training loss — is the number that matters.
- **Per-episode metadata.** LeRobot v3.0 has no free-form per-episode slot, so
  both Haller blocks describe the rig as of the *most recent* take appended.
  Recalibrating or re-scoring mid-dataset means a new `repo_id`.
