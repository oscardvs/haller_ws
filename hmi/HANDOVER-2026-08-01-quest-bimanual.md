# Handover — Quest bimanual teleop finished, 2026-08-01 (evening)

Branch: **`feat/quest-bimanual-teleop`**, cut from `feat/motion-safety-envelope`
@ `0672b45` and developed in the worktree `~/haller_ws.quest` so the
motion-safety agent's checkout was never touched. Operator path:
**`hmi/QUICKSTART-QUEST.md`** — start there; this file is the engineering
handover.

## What landed (4 commits)

1. **Bimanual collision + workspace guard** (`haller_hmi/collision.py`).
   Four capsules per arm swept along an analytic FK transcribed from the
   vendored SO-101 MJCF, filtering every commanded step inside the 60 Hz
   commit loop: pass if clear of `margin_m`, always pass if it *improves* a
   bad pose (escape is never blocked), otherwise bisect the step back and
   stop at the margin. Height floors keep tip/wrist/elbow off the bench; the
   gripper is never scaled. Per-side grip split: each Quest squeeze is a
   dead-man for its own arm only. Config: `collision:` block in
   `config.yaml`; wiring refuses to fail open (missing mounts disable the
   guard loudly at startup).

2. **The VR cockpit** (`VRTeleopPanel.tsx`, `vrTeleop.ts`). Passthrough AR
   session (VR is a warned fallback) with a cleared XRWebGLLayer — before
   this, the session had no render layer at all, so the operator drove real
   arms while looking at a void. DOM-overlay HUD (authority, countdown,
   blocking joints, clearance, E-STOP button), B/Y hardware E-STOP scanned at
   display rate, per-transition haptics, forced disengage while the Quest
   menu blurs the session, a parting disengage frame on teardown, and a
   re-arm flow after E-STOP.

3. **A real direction bug, found and fixed by the new smoke test**
   (`scripts/vr_smoke.py`). The session's hardwired opposite mirror parities
   (left=`swap`, right=`!swap`) are correct for pre-mirrored webcam frames
   but wrong for egocentric headset frames on two identical same-yaw arms:
   the right arm swung *parallel* to the left instead of toward it. Frames
   now carry `mirror_mode` ("none" for VR, absent = legacy swap convention
   for the camera path, "both" for a genuinely mirrored rig). The panel has
   a persisted mounting selector.

4. **Ops** (`scripts/quest-teleop/`): `up.sh`/`down.sh`, committed Caddyfile
   (env-overridable), Jetson start script + systemd unit, and the
   QUICKSTART.

## Validation state

- Backend **356 passed** (321 baseline + 35 new), frontend **112 passed**,
  `tsc --noEmit` clean.
- `tests/sim/test_collision_sim.py` pins the FK against MuJoCo (<1e-4 m) and
  pins *soundness*: across seeded random configs, every mesh-level contact
  MuJoCo finds (inter-arm AND within one arm) is already gap ≤ 0 in the
  capsule model. If it ever fails: grow the radius, never shrink a margin.
- `scripts/vr_smoke.py`: **15/15**, twice — once against bare uvicorn, once
  through the real Caddy HTTPS/WSS origin. Covers engage/countdown, per-side
  grips, drive, guard clamping a hand-crossing at 0.0 mm slack, release
  freeze, E-STOP + re-arm, WS-drop auto-stop.
- **Not yet done: a human driving the real arms.** The Jetson was off/
  unreachable all evening, so the branch is pushed but not deployed. Follow
  the QUICKSTART's first-hardware-run checklist (measure the real mounts —
  the config still carries the sim's ±0.20 m — verify the clearance readout
  by hand, direction-check one arm at a time).

## Merging with `feat/motion-safety-envelope`

Cut at `0672b45` (their Task 6 done; Tasks 7–8 were still in flight).
Expected conflict surface, all small:

- `server.py` — they rewrite the arm routes + `/estop` (Task 7); this branch
  only adds the guard construction next to `HumanTeleopSession(...)`. Keep
  both. Their `/estop` version adds `executor.cancel()`; ours changed nothing
  there.
- `human_teleop.py` — their Task 8 changes `_commit` to store `send_goal`'s
  return; ours filters the pair *before* `_commit` and touches `start()` not
  at all. Semantically independent; take both sides. Note for their Task 8:
  with the guard active, the commanded pose already equals the filtered
  value, so their fix composes cleanly (recorder gets guard-filtered AND
  cap-truncated actions).
- `config.py` / `config.yaml` — additive blocks on both sides.

## Watch-outs inherited by whoever drives first

- The discrete-move path (`motion.move_to` / home / presets) does **not**
  consult the collision guard — it has its own large-move refusal, but a
  30°-legal preset could still cross arms. Streaming teleop is covered.
  Worth a follow-up once their branch lands.
- Real-arm calibration zero vs MJCF zero: the guard assumes the LeRobot
  degree convention matches the MJCF joint zero (same assumption the sim
  handle makes). Margins absorb small offsets; the clearance-by-hand check
  in the QUICKSTART is the verification.
- The acquisition gate requires ~90 % trigger squeeze to match a mid-range
  jaw (`match: gripper` on the HUD). Deliberate, but surprising the first
  time.
