# Track A handoff — the realtime core

Written 2026-08-27 by the session that did Phases 0.5–1 and the two standalone
fixes. You are the critical path: Track D is blocked on your `/record/arm`,
everything else is complete and holding.

Read `docs/port/INTEGRATOR-HANDOFF.md` and `PLAN-2026-08-27-kit-port.md` first.
This file is only what is NOT written down elsewhere.

---

## Where things stand

| commit | what |
|---|---|
| `108b104` | Phase 1 — preflight, `isAvailable()`, median-of-3, antipode gate, wire fixes |
| `f995fc8` | gripper clamp window is a percentage, not degrees |
| `fec47cb` | `MIN_RATE_FRACTION = 0.9` published in `safety.py` |
| `99a736d` | rate cap expressed as deg/s; `hz` bounded [10, 120] |

**Nothing is half-done.** The cadence fix is complete and committed. There is no
work-in-progress to pick up — start Phase 2 clean.

`MIN_RATE_FRACTION` is settled: published in `safety.py` (NOT `tick.py` — the
rollout child must read it without importing lerobot, and a test pins that
`safety` stays dependency-light). Track B's probe resolves against it, verified
live by the integrator by publishing `0.5` and watching the resolver follow.
**Do not relitigate this.**

---

## Phase 2 — the tick, as I hold it

The design doc is `docs/port/phase2-tick-contract.md` and it is accurate. What
follows is the reasoning behind it that the doc does not carry.

### Why one sampler

Three uncoordinated samplers today: the session commits at 60 Hz, telemetry runs
its own 20 Hz loop, the recorder scrapes `human_teleop.status()["goal_deg"]` at a
third instant. Every recorded row pairs a state from one moment with an action
from another. That is mechanisms 1–3 of the port's original four, and they are
**one fix, not three**.

### The shape

`haller_hmi/tick.py`:

- **`TickSample`** — frozen. `t_mono`, `seq`, `arms{id: {joints_deg, effort_norm,
  torque}}`, `goal_deg`, `reasons`, `base`, `clutch`, `collision`. Frozen because
  a consumer that can mutate a sample corrupts every other consumer's view of the
  same moment, and one-moment-for-everyone is the entire point.
- **`TickBus`** — synchronous fanout, bounded per-subscriber queues, drop-oldest
  **counted**. Counted is load-bearing: an uncounted drop is indistinguishable
  from a tick that never happened, and invariant 9 turns on telling those apart.
- **Exactly one producer at a time.** `IdleSampler` at `telemetry.hz` when no
  session runs; the session tick when one does. **Test the handover** — two
  producers overlapping for one tick puts two samples on one `seq`.

Consumers: `human_teleop._loop` does the SINGLE bus read (decimated by a config
`read_divisor`), commits, publishes. `telemetry.py` becomes a decimating consumer
and stops touching the bus. `recorder.py` consumes the bus. `snapshot()` returns
state and `goal_deg` under ONE lock at the same `seq`.

### The server.py delta you will need to request

`server.py` is the integrator's. The delta is small — I scoped it:

`human_teleop` is constructed at MODULE level (`server.py:77`) while
`telemetry`/`recorder` are built inside `lifespan`. So the bus cannot simply be a
lifespan local. Cheapest shape: `HumanTeleopSession.__init__` gains an optional
`tick_bus` and constructs its own by default, so line 77 needs no change. Then
lifespan only has to build the `IdleSampler`, start/stop it, and pass
`tick_bus=human_teleop.tick_bus` to `TelemetryBroadcaster` and `DatasetRecorder`.
About five lines. Report it; do not edit it.

### Two constraints the record gate imposes

Both come from Track D's final design and neither is in the original brief.

**C1. `{save: true, rearm: true}` is the HOT PATH.** Track D's end prompt returns
to ARMED on every outcome — L click = KEEP = save and re-arm at the NEXT index. In
a session banking 46 takes it is the most-pressed control on the rig. lerobot's
`save_episode` folds stats and may encode video, so: do not report the next
`episode_index` before the previous save has committed (Track D deleted their own
`episodesTotal()` workaround **because I promised the index is the truth** — getting
it wrong is worse than never promising); a second save must not interleave with
the first still flushing; and the tick must keep running at full rate through the
flush, because the operator is posing the arm during it.

**C2. A long-lived ARMED subscriber must not manufacture drop counts.** ARMED is
now where a session SITS. An attached-but-not-committing subscriber overflows its
bounded queue continuously and those drops mean nothing. Either drain-and-discard
while armed, or do not subscribe until ROLL. What is NOT acceptable is
`skipped_frames` climbing while parked — Track D puts that number on the HUD, and
an operator who learns to ignore it while parked will ignore it mid-take.

### `episode_uid` — ruled, measured, build it

Bare **`episode_uid`**, per-frame **`int64`**, **microseconds since the Unix epoch
UTC**, captured at **ARM time**, monotonic with **+1 on collision**.

**NEVER namespace it under `observation.` or `action.`.** This is the trap.
`dataset_to_policy_features` (`lerobot/utils/feature_utils.py:139-181`) classifies
by key PREFIX: `observation.*` → STATE, `action*` → ACTION, everything else hits a
bare `continue`. That `continue` is what makes an extra column inert to training.
`observation.episode_uid` would be handed to the policy as an input feature — a
dataset that trains on its own episode ids. The instinct to namespace it is
strong and it is exactly wrong.

Why it exists: **pruning renumbers.** `delete_episodes` leaves survivors as
`episode_index` 0..n, measured twice on two datasets (46→35 real, 4→2 synthetic).
So `episode_index` is exact AT RECORD TIME and **not durable across a prune**.
Anything storing it as a lasting key — `review.json` marks, `--dataset.episodes`
keep sets — silently re-points after a prune, and you train on the wrong episodes.

Home measured by the integrator: a per-frame column **survives** a prune;
a key in `info.json` **does not** (the loader warns "Unknown fields in
DatasetInfo … will be ignored"). Arm time, not save time, because at save time a
redo and its keeper are indistinguishable in the ordering; microseconds because it
is sortable, so recording ORDER survives the prune too.

**My addition, agreed: the first real prune must ASSERT the uids come out in the
expected set.** The integrator's probe was a synthetic 4-episode single-camera
dataset, not our three-camera layout with resumed metadata across several
`meta/episodes/chunk-*/file-*.parquet`. The mechanism is generic and the real
layout is known to prune correctly, but nobody has run the two together. Close it
with a measurement, not an inference.

### `GET /record/status` — agreed with Tracks C and D, do not drift

```
state: "idle" | "armed" | "recording"    # `recording` stays == state=="recording"
episode_index: number | null             # known at ARM time
invalidated_reason: string | null        # armed -> idle fallback only
fps_declared: number
fps_measured: number | null              # null only until 30 samples
record_rate_gate: number                 # emit it; C reads it, does not mirror it
skipped_frames: number
drops: {cameras: {<key>: n}, arms: {<id>: n}}
alerts: [{code: "record_rate", ...}]
```

`"prompt"` is NOT a server state — frames keep being written throughout Track D's
end prompt. `"recording"` not `"rolling"`, so the shipped `recording` boolean
stays true.

`invalidated_reason` fires ONLY on armed→idle, never mid-take: a mid-take teleop
stop already saves up to the stop and closes the episode. I told Track C this and
it deleted a whole red-banner state they were about to build. Keep it true.

### Routes (you implement, integrator mounts)

- `POST /record/arm {repo_id, task}` → full RecordStatus. Opens/appends, freezes
  the camera set + feature schema + ARM SET, resolves `episode_index`, freezes
  measured `fps`. 409 here (not at roll) on: colliding camera keys, unknown repo,
  already recording, measured rate below `MIN_RATE_FRACTION`.
- `POST /record/roll {}` → full RecordStatus. 409 if not armed.
- `POST /record/stop {save, rearm?}` → full RecordStatus. **`rearm` OPTIONAL,
  defaults false** — `{save}` alone must keep meaning what it means today, two
  shipped desktop surfaces call it.

| save | rearm | outcome | lands | index |
|---|---|---|---|---|
| true | false | SAVE | idle | advances |
| false | true | RE-RECORD | armed | same |
| false | false | DISCARD | idle | unchanged |
| true | true | SAVE + GO AGAIN | armed | next |

RE-RECORD never touches disk: an episode buffer never `save_episode`'d is dropped
and the index does not advance.

**Report the exact committed bodies to the integrator when they land** — Track D
unblocks on the mount, and the operator docs get written off the committed
protocol, not the planned one.

### What must not move

Every lerobot-0.5.1 workaround in `recorder.py` — `MIN_SAVEABLE_FRAMES = 2`,
`video_files_size_in_mb = 0`, the two-direction resume schema check. Each encodes
a dataset-destroying incident.

---

## Open, carried forward

- **The scanner half of the fixture rule.** The premise test exists
  (`test_every_degrees_window_load_joint_limits_can_emit_is_symmetric`). The
  scanner does not. It MUST carry the scope qualifier: the rule holds for
  fixtures standing in for loader OUTPUT, not for arbitrary limits handed to a
  component that accepts arbitrary ranges — `SO101DecoupledIK` takes whatever it
  is given and `test_so101_ik.py` passes it deliberately asymmetric ranges. A
  naive scanner cries wolf on its first run, which is how a check gets disabled.
- **`pose_filter_alpha` is source-cadence coupled** — a per-FRAME EMA, so ~83 ms
  from a 60 Hz Quest and ~1 s from a 4.8 Hz policy. Goes with the rollout ingest.
- **`frame_age_ms_loss = 700`** is tuned for a 60 Hz stream (~42 missed frames).
  At a legitimate 4.8 Hz rollout it is three frames, so a healthy slow policy
  trips tracking-loss and gets demoted with a symptom pointing nowhere near the
  cause. Make the staleness budget relative to the source's DECLARED rate.
- **The `[0, 1]` gripper scaling should move out of the session** into the VR
  converter, so nothing inside `HumanTeleopSession` knows one input dialect
  normalises one joint. Now a layering cleanup, not a bug — goes with the rollout
  ingest. Track B was told policy actions arrive in DEGREES with the unit declared
  ONCE at lease time, never per message.

---

## How to work here

- **`git commit -F <msgfile> -- <explicit paths>`.** The index is SHARED across
  every session. A bare `git add` + `git commit` will sweep up other tracks' work
  — it happened to me. `git add` only for a brand-new file and only in the same
  breath as its commit. **Never `git stash`** — it is indistinguishable from data
  loss to every other session.
- **A full-suite result on a shifting tree measures the tree, not the code.** If
  a test fails in a full run and passes in isolation, check whether someone is
  mid-edit before reporting a regression. The integrator now runs against a
  detached worktree for this reason.
- **Wall-clock tests flake under four concurrent sessions.** Real-thread teleop
  tests at `hz_override=200.0` can starve. Fix the PRECONDITION, never the
  behavioural assertion.
- Tests: `source ~/venvs/haller-hmi/bin/activate-haller-hmi` then
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl python -m pytest -p asyncio -q`.
  Lint with `~/venvs/haller-hmi/bin/ruff` (0.16.0), NOT the 0.15.1 on PATH.
  Always compare lint against `git show HEAD:<file>` — most findings are
  pre-existing and you only own the delta.

## The disposition that earned its keep

Four corrections to the plan came out of checking a claim instead of building on
it: the kit's `pos_reach_limit` was measured against the wrong arm, the IK
post-processing order already matched so there was nothing to port, the rollout
sequencing was backwards, and the D455 magenta finding was the wrong sensor.

Three lessons the port adopted as rules, all the same shape — **the test was
written from the CODE, so it inherited the code's blind spot**:

- a sequence claim tested per-step walks the same control flow the code does
- an impossible fixture builds its world from the code's assumptions
- a per-point assertion checks the points the code already handles

Counter-discipline: **write the assertion from the CLAIM, in the claim's own
terms, before reading the implementation.** A sweep for a mapping, a
real-loader-derived fixture for a world, an end-to-end for a sequence.

And the two that cost the most to learn:

- **a constant counted in ticks lies at every cadence but one** (`lpf_tau_s` is
  the model to copy)
- **a check that cannot fire in either direction is dead code shaped like a
  safety check** — worse than a bug, because it reassures

I was wrong out loud several times here — a units error in a constant I specified,
a stale read reported as live, a sweep finding that contradicted a correct comment.
Every one was cheaper than being quietly right would have been. Keep doing that.
