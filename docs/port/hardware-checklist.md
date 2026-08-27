# Hardware verification checklist — the batched session

Per PLAN-2026-08-27 decision 4, everything in this port is built and verified in
sim first, then cleared against real servos in ONE session when the new hardware
arrives. This file is the running list of what sim **cannot** answer, written as
each item is built rather than reconstructed afterwards.

Rules for using it:

- Every item names the claim, how to test it, and what a PASS looks like as a
  number or an observation — not "check it works".
- An item that fails does not get argued with. Record the measurement, mark it
  RED, and it becomes a finding against the phase that wrote it.
- Sim carries everything else. `SimArmHandle` publishes through the identical
  path, so anything true of the commit chain in sim is true of it on hardware —
  that is exactly why the list below is short and specific.

Status legend: ☐ not yet run · ✅ passed · ❌ failed (finding recorded) · ⊘ dropped

Rig at time of writing: desktop (RTX 4080 SUPER), **one** SO-101 arm attached,
new servos on order, wrist camera not on hand.

---

## Track A / Phase 1 — kit reconciliation in the teleop path

### H1. The first-observation range gate actually fires ☐

`vr_teleop/preflight.py::check_first_observation` drops torque and names the
joint when a first reading lands outside its limits ±15°. Unit-tested against a
fake handle; never seen against a real Feetech that genuinely mis-reads.

- **Test:** deliberately mis-calibrate one joint (sweep it, then hold a WRONG
  middle pose at the ENTER prompt), reconnect, and watch `connect_all`.
- **PASS:** torque drops, and the log names *that* joint with its measured angle
  and its limit. A correct calibration on the other joints is not disturbed.
- **Why it can't be sim'd:** sim arms have no Feetech calibration file and no
  encoder wrap; `run_preflight` returns `skipped=True` for them by design.

### H2. The encoder-wrap WARNING does not become an error on a good arm ☐

A fully-folding joint (shoulder_lift, elbow_flex) crosses the 12-bit wrap during
a *correct* sweep, so a ~360° recorded span is normal and must warn, not abort.

- **Test:** connect the existing, known-good `haller_leader` calibration.
- **PASS:** connect completes; any ~360° span logs at WARNING; zero hard problems.
- **RED if:** a known-good arm is refused. That would make the gate unusable and
  it comes out.

### H3. `isAvailable()` — a real lost race is now a dropped read ☐

`arm.py::_read_block` shares `bus.sync_reader` with the 60 Hz teleop thread with
no lock (lerobot's own `sync_read` does the same). Before this fix a lost race
returned tick 0, which decodes to a large ANGLE. Unit tests pin the decode; only
hardware produces the race.

- **Test:** run teleop at 60 Hz with telemetry at 20 Hz for ≥10 min on the real
  bus, with the block effort path active (`_effort_mode == "block"`).
- **PASS:** zero position samples that decode from a raw 0 tick. Any lost race
  shows up as a demotion-counter increment / a dropped read, never as a value.
- **Measure while there:** how often it actually happens. If the answer is
  "never in 10 min", say so — the fix stays either way, but the frequency
  decides how loud the logging should be.

### H4. Median-of-3 on the park-goal-on-present anchor ☐

A single read near the wrap can teleport ±360°; that anchor writes
`Goal_Position := Present_Position`, so a teleported read parks the goal at a
garbage angle and **the arm snaps there the instant torque enables**. This is
the failure mode with the worst consequence in Phase 1.

- **Test:** power-cycle / reconnect ~20 times with a joint parked deliberately
  near its wrap, watching for any lunge at torque-enable.
- **PASS:** 20/20 torque-enables are a no-op — the arm locks where it is.
- **Keep a hand near the E-STOP.** This is the one item on the list that can
  move the arm violently if the fix is wrong.

### H5. The antipode park gate on a real wrist ☐

`rot_err_hold = 2.2 rad` restored to `decoupled_ik.py`. Guards the 180°
come-around flip when the orientation demand nears the antipode — the only
backstop when `rot_reach_limit = 0` (which `config.solo-raw.yaml` sets).

- **Test:** with `HALLER_HMI_CONFIG=config.solo-raw.yaml` (reach limits off),
  deliberately twist the controller past the antipode while driving.
- **PASS:** the wrist PARKS — stops, does not come around — and the operator is
  told, on the haptic channel, at full strength.
- **RED if:** the wrist flips. That is the exact failure the gate exists for.
- **RED also if** the wrist parks *silently*. An earlier draft of this file made
  "pressure ≥ 0.35" the pass criterion, which was wrong and unmeetable: the
  kit's `last_limit_pressure` is in RADIANS and Haller's is in DEGREES, so 0.35
  ported literally lands *below* `_update_haptic`'s 0.5° dead zone
  (`vr_teleop/teleop.py:391`, `_gate(p, 0.5, 4.0)`). The operator would have got
  *less* buzz at a parked wrist than with no gate at all. Assert the buzz, not
  the number behind it.

### H5b. The park gate does not latch on an ordinary over-reach ☐

Distinct from H5 and arguably likelier to bite. The gate parks the wrist when
the orientation error exceeds `rot_err_hold` — but parking removes the only
mechanism that can *reduce* that error, so an unreachable POSITION demand with a
perfectly ordinary orientation can hold the wrist parked indefinitely. Measured
in sim: parked from iteration ~50 through 5000 (83 s at 60 Hz), wrist stuck on
its stop, residual pinned at 1.00.

`rot_reach_limit = 0.6` on a shipped config clamps the demand near the current
orientation and prevents this. `config.solo-raw.yaml` sets `rot_reach_limit: 0`,
so it is reachable there — and that is the config Oscar runs *when the arm is
not going where the hand goes*, i.e. the one where a latched wrist would be
misread as the fault being diagnosed.

- **Test:** on `solo-raw`, drive the tool hard past the edge of the workspace
  with the hand held level, and keep it there.
- **PASS:** the wrist recovers on its own once the reach demand relaxes, and the
  haptic buzz is not a permanent full-strength alarm during ordinary reaching.
  An alarm that cries wolf is worse than no alarm.

### H6. `pos_reach_limit` 0.15 m is the right wall ☐

Moved from 0.12 (port-time value, never measured on Haller) to the kit's own
SO-101 value of 0.15 m — "smaller arm, smaller wall". `config.solo-raw.yaml`'s
header names the 12 cm clutch as a defect mechanism ("the arm stops somewhere
the hand is not"), which is the evidence *against* 0.12 but not yet evidence
*for* 0.15.

- **Test:** fast reaches to the edge of the workspace, solo, both values.
- **PASS:** a fast reach is not silently absorbed; the arm ends where the hand
  is. If 0.15 still absorbs, this is a tuning finding, not a code one — record
  the number that works.

### H6b. The gripper uses its whole travel, and the whole trigger ☐

Found 2026-08-27 by sweeping fixtures for windows the real loader cannot emit.
`_load_joint_limits` centres every window on its tick mid-point, so it can only
ever emit ranges symmetric about zero — and the gripper's is `(-63.59, +63.59)`
on this rig's `haller_follower.json` (ticks 2045..3492). But `_to_degrees` maps
the converter's `[0, 1]` onto *that* window, while lerobot pins the gripper to
`RANGE_0_100` and clamps with `bounded_val = min(100, max(0, val))`. Measured:

| gripper cmd | → degrees | → lerobot |
|---|---|---|
| 0.00 | −63.59 | **0.00 % open** |
| 0.25 | −31.79 | **0.00 % open** |
| 0.50 | 0.00 | 0.00 % open |
| 0.75 | +31.79 | 31.79 % open |
| 1.00 | +63.59 | **63.59 % open** |

With `gripper = 1 − trigger`, the jaw runs its entire range in the first half of
the trigger and the second half is dead — and it never opens past 64 %. Nothing
drives into a stop (lerobot clamps), so this is a control-quality defect, not a
safety one. It would be described as "twitchy, won't open all the way" and
blamed on the servo.

- **Test:** pull the trigger slowly through its full travel and watch the jaw.
- **PASS:** the jaw tracks the trigger across the WHOLE travel and reaches its
  full open stop at trigger released.
- **Also check the recorded column:** every episode recorded before the fix has
  its gripper channel compressed into 0..63.6 with a dead zone, which matters
  for anything trained on it. Do not mix pre- and post-fix episodes in one
  training set without knowing which is which.

### H7. `t_client` — only a kit-shaped client exercises it ⊘ (deferred)

`wire.py` now accepts the kit page's `t_client` as well as Haller's `ts_ms`.
Haller's own HUD sends `ts_ms`, so nothing on the rig exercises the kit path.

- **Deferred, not dropped:** if the kit's page is ever pointed at this backend,
  check the frame clock is non-zero. Nothing downstream consumes it today —
  staleness is measured on arrival — so this is low stakes by construction.

---

## Carried from the plan's unverified list

### U3. The MEASURED end-to-end sample rate ☐ — blocks P1, P7-H

`fps` must be measured, never declared. Mechanism 3 of four: every `timestamp`
in every episode today is synthesised from the rate telemetry was *asked* for.

- **Test:** instrument the new sampler, 60 s run, real Feetech round trips in
  the tick, both arms if they are attached.
- **PASS:** record the number. There is no threshold to hit — the point is that
  `session_hz_measured` is a measurement. The declared/measured ratio then feeds
  the ≥90%-for-2 s record-rate alert.
- **Also measure:** the spread, not just the mean. A 30 Hz mean built from
  alternating 60 Hz and 15 Hz ticks is a different dataset from a steady 30.

### U6. Identity-based Quest pairing ☐

Flagged hardware-unconfirmed in the 08-22 unification and still unconfirmed.
Carried in memory as a "hardware-unconfirmed flip".

### U3b. The D455 is negotiating USB 2.1 — cable or port, not software ☐

`/sys/bus/usb/devices/1-4/speed` reads **480**, not 5000. That caps the colour
node's resolution and frame rate, and it will bite during recording rather than
during teleop, because a recorded camera is a REQUIRED camera — a frame that
does not arrive drops the whole tick (invariant 9).

This makes it a prime suspect for U3 coming back under the declared rate. Rule it
out BEFORE concluding anything about the sampler, or a cable gets diagnosed as a
software defect.

- **Test:** re-seat on a known 5 Gbps port with a known-good cable; re-read
  `speed`; then re-run U3 with the mast cam in the record set.
- **PASS:** `speed` reads 5000, and U3's measured rate does not move when the
  camera is added to the record set.
- **If it stays at 480:** record the achievable resolution/fps and treat that as
  the ceiling. Do not raise the declared fps above what the bus can carry — that
  is mechanism 3 with extra steps.

### U8. The wrist camera ☐

Mount, cabling, and whether the view is usable at 640×480 are all open — the
camera is not on hand. Blocks the Phase 3 third-camera work from being anything
more than sim-verified.

- **Note the ordering constraint from Phase 3:** solo-with-three-cameras must be
  proven BEFORE the second arm connects. Every recorded camera is a REQUIRED
  camera, so going one → three triples the failure surface of a take.

---

## Phase 2 / Phase 3 items

Added as those phases land. Nothing here yet.
