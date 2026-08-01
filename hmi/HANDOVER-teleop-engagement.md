# Handover — make human-teleop engagement safe and operable

Human-pose teleop works end-to-end against the MuJoCo sim: pose/hand/face
tracking runs in the browser, the backend retargets to SO-101 joint angles, and
the sim arms follow. The pipeline is not the problem. **Authority transfer is.**

Two problems, one root cause: the operator gets no say in *when* the robot
starts following, and no way to see what it will do when it does.

---

## Problem 1 — authority transfers instantly, from wherever your arms happen to be

Today the robot begins following the instant the clutch closes, from whatever
pose your arms are in at that moment. The same thing happens every time a hand
re-enters frame after tracking loss. Neither is safe: the operator's arms are
almost never where the robot's arms are, so engagement means a jump.

On real hardware this is the difference between a demo and a broken arm. It also
makes the sim unusable for judging anything, because every engagement starts
with a lurch.

**Goal.** Engagement becomes a deliberate, visible, timed acquisition rather
than an instant. The operator should be able to see the robot's current pose,
move their own body to match it, and only then have authority transfer — with
enough warning to abort. Re-acquisition after tracking loss must go through the
same path as a cold start; there is no reason for them to differ.

The operator asked specifically for: a countdown that begins once both sides are
trackable, and a render of the robot's current pose to pre-position against. Treat
that as the shape of the requirement, not the required implementation — if a
pose-match gate, a ramp, or something else serves the goal better, argue for it.

## Problem 2 — the mouth clutch's usable band is too narrow to hold

Real numbers from the first operator to ever drive it (2026-07-29):

```
talk_max 0.38   open_min 0.74   →   t_engage 0.60   t_release 0.49
observed peak jawOpen ≈ 0.72
```

That leaves ~0.12 of headroom above the engage threshold, at the very top of his
jaw range, to be sustained continuously while both arms are moving. In practice
the clutch does not stay closed: a 22 s trace never once reached `driving`.

**Goal.** A dead-man the operator can actually hold for a working session without
their jaw giving out, that still cannot be closed by speech. Those two constraints
are in tension and resolving that tension IS the task.

The safety property is non-negotiable and untested: **normal speech must never
engage the robot.** It has never been verified against a real person — do that
first, before changing any constant, so you know what you are trading against.
`MOUTH_MIN_SEPARATION`/`MOUTH_ENGAGE_FRAC`/`MOUTH_RELEASE_FRAC` are tuned on
zero human data. If the honest conclusion is that jawOpen is the wrong signal,
say so rather than tuning the constants until it passes.

---

## Where things are

- `hmi/frontend/components/HumanTeleopPanel.tsx` — webcam, three MediaPipe models,
  ~30 Hz publish loop, calibration state. The rAF loop's dependency comments are
  load-bearing; read them before touching the effect.
- `hmi/frontend/lib/mediapipe.ts` — model loading, landmark fusion, `KeypointFrame`.
- `hmi/backend/haller_hmi/human_teleop.py` — session, retargeting, per-side
  tracking-loss and staleness handling.
- `hmi/backend/haller_hmi/safety.py` — clutch thresholds, clamping, rate caps.
- `hmi/frontend/components/MouthClutchCalibration.tsx` — the two capture windows.

## Invariants not to break

- One dead-man authority per session, fixed for the session's life.
- The backend fails closed: stale keypoints, low confidence, or a lost side
  freeze the affected arm. Confidence floor is 0.4; staleness budget 300 ms.
- Releasing the dead-man freezes both arms immediately. Whatever acquisition
  ramp you add must not put latency in the *release* path.
- E-STOP stops a human-teleop session like any other.
- Existing tests must pass and be updated where a component is restructured,
  not deleted.

## Developing against the sim

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
MUJOCO_GL=egl HALLER_HMI_CONFIG=~/haller_ws/hmi/backend/config.bimanual-sim.yaml \
  uvicorn haller_hmi.server:app --host 0.0.0.0 --port 8000 --app-dir ~/haller_ws/hmi/backend

cd ~/haller_ws/hmi/frontend && NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 pnpm dev
```

`MUJOCO_GL=egl` is mandatory — without it MuJoCo picks GLFW and dies on an X11
assertion. `.env.local` points at the desktop; the inline override is what keeps
you local. Watch the arms at `http://localhost:8000/cameras/threequarter_sim/stream`
in a second window — the dead-man is scoped to the visible tab, so you cannot
watch from the Cameras tab while driving.

No CI typecheck in this repo: run `npx tsc --noEmit` and `pnpm test` yourself.
Stop the backend before running sim pytest (EGL contention makes camera tests fail).

## Verifying you actually fixed it

The bar is behavioural, not unit tests. A session where the operator engages,
drives both arms through a real reach, loses a hand off-frame, recovers, and
re-engages — without a single lurch and without fighting to hold the clutch.
Sim is enough to demonstrate it; hardware has never run this.
