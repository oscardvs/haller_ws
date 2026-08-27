# Phase 2 — the tick contract

Design record for the one-sampler fix. Written before the code so the shapes the
other tracks build against are stable; amended in place if a measurement
contradicts one, never silently.

Track A owns everything here. Track D (in-headset client) and Track C (desktop
cockpit) consume the status shape at the bottom; Track B does not touch it.

---

## What this replaces

Three uncoordinated samplers (plan §"Why it failed here", mechanisms 1–3):

- the session commits at 60 Hz,
- `TelemetryBroadcaster` runs its own 20 Hz loop,
- the recorder scrapes `human_teleop.status()["goal_deg"]` at a third instant.

Every recorded row therefore pairs a state from one moment with an action from
another. Mechanisms 1–3 are **one fix**: one sampler that owns the tick.

## The shape

`haller_hmi/tick.py`:

- **`TickSample`** — frozen. `t_mono`, `seq`, `arms{id: {joints_deg,
  effort_norm, torque}}`, `goal_deg`, `reasons`, `base`, `clutch`, `collision`.
  Frozen because a consumer that can mutate a sample can corrupt every other
  consumer's copy of the same tick, and the whole point is that they all see one
  moment.
- **`TickBus`** — synchronous fanout, bounded per-subscriber queues,
  drop-oldest **counted**. Counted is the load-bearing word: an uncounted drop
  is indistinguishable from a tick that never happened, and invariant 9 turns on
  being able to tell those apart.
- **Exactly one producer at a time.** An `IdleSampler` at `telemetry.hz` when no
  session runs; the session tick when one does. The handover between them is
  TESTED, not assumed — two producers briefly overlapping would put two samples
  on one `seq`.

Consumers: `human_teleop._loop` performs the SINGLE bus read (decimated by a
config `read_divisor`), commits, publishes. `telemetry.py` becomes a decimating
consumer and stops touching the bus. `recorder.py` consumes the bus.

`snapshot()` returns state and `goal_deg` under ONE lock at the same `seq`.

## Two constraints the record gate imposes

Settled with Track D (haller-ws-1a) on 2026-08-27. Their END PROMPT resolves
every outcome back to ARMED — ARMED is the resting state of a recording session,
not a step toward one — which makes both of these hot paths rather than edges.

### C1. `{save: true, rearm: true}` is the PRIMARY path, not an edge case

L-stick click = KEEP = save this take and re-arm at the NEXT index. In a session
banking 46 takes this is the most-pressed control on the rig.

lerobot's `save_episode` folds stats incrementally and may encode video. So:

- re-arming must not report the next `episode_index` before the previous
  `save_episode` has actually committed, or the index is a guess — and Track D
  is deleting their own `episodesTotal()` floor **because** we promised the
  index is the truth. Getting it wrong is worse than not having promised it.
- a second save must not interleave with the first still flushing.
- the operator is posing the arm during the flush. That is fine — ARMED writes
  nothing — but the tick must keep running at full rate through it. A save that
  blocks the producer would stall teleop at exactly the moment the operator has
  been told they may drive.

### C2. A long-lived ARMED subscriber must not manufacture drop counts

ARMED is now where a session SITS. A subscriber that is attached but not
committing will overflow its bounded queue continuously, and those drops are
meaningless — nothing was lost, nothing was going to be written.

So an armed-not-rolling recorder either drains and discards, or does not hold a
subscription until ROLL. Either is fine; what is not fine is `skipped_frames`
climbing while armed, because Track D puts that number on the HUD and an
operator reading "dropping" while parked would learn to ignore it — which is
precisely the number that has to be trusted mid-take.

## `GET /record/status` — agreed with Track D, Amendment 4

```
state: "idle" | "armed" | "recording"    # `recording` stays == state=="recording"
episode_index: number | null             # known at ARM time; treated as exact
invalidated_reason: string | null        # armed -> idle fallback, with the why
fps_declared: number
fps_measured: number | null              # always present once a producer has run
skipped_frames: number
drops: {cameras: {<key>: n}, arms: {<id>: n}}
alerts: [{code: "record_rate", ...}]     # measured < 90% of declared for > 2 s
```

`fps` is frozen into `info.json` at ARM time from the bus's rolling measured
rate — which is why the rate refusal lands at arm time rather than mid-take.
Invariant 10: measured, or the episode does not open.

`record_rate_gate` (0.9) is emitted in the payload rather than mirrored in the
UI, so the dashboard cannot come to disagree with the system it is monitoring.
Track C reads it via `recordRateGate(status)` with a constant named
`RECORD_RATE_GATE_FALLBACK` — named so nobody re-promotes it to the authority.

**`fps_declared` MUST be exactly the `fps` written into `info.json`.** Not "the
rate we asked for", not "the rate we hoped for" — the number lerobot synthesises
every `timestamp` from as `frame_index / fps`. The gate is a ratio of measured to
declared, so if those two ever come apart the ratio stays 0.9 and silently starts
meaning something else: a take could pass a rate check while its timestamps are
being generated from a number nothing measured. That is mechanism 3 of the
original four, reintroduced through the back door of the thing built to prevent
it. Anything that sets one must set the other. *(haller-ws-fd, 2026-08-27)*

**Arming freezes the camera set, the feature schema AND the arm set.** If the
session's arm set changes or teleop stops under an armed recorder, the freeze is
stale: state falls back to `idle` with `invalidated_reason` set. It cannot
happen mid-take — a mid-take teleop stop already saves up to the stop and closes
the episode, and that behaviour predates this port and stays.

Exiting VR stops teleop, which invalidates an armed gate by this rule. That is
why the headset has no stand-down gesture: leaving disarms without a command.

## Routes (Track A implements; haller-ws-13 mounts in server.py)

- `POST /record/arm {repo_id, task}` — open/append, freeze schema + camera set +
  arm set, resolve `episode_index`, freeze measured `fps`. Writes nothing.
  409s here rather than at roll: colliding camera keys, unknown repo, already
  recording, measured rate below 90% of declared.
- `POST /record/roll {}` — begin writing. 409 if not armed.
- `POST /record/stop {save, rearm}` — all four combinations legal:

  | save | rearm | outcome | lands in | index |
  |---|---|---|---|---|
  | true | false | SAVE | idle | advances |
  | false | true | RE-RECORD | armed | same |
  | false | false | DISCARD | idle | unchanged |
  | true | true | SAVE + GO AGAIN | armed | next |

RE-RECORD never touches the dataset on disk: an episode buffer that is never
`save_episode`'d is dropped and the index does not advance. No delete, no stats
recompute, none of the hand-rolled pop's five 409 refusals.

## What does not move

Every lerobot-0.5.1 workaround in `recorder.py` stays untouched —
`MIN_SAVEABLE_FRAMES = 2`, `video_files_size_in_mb = 0`, the two-direction
resume schema check. Each encodes a dataset-destroying incident.
