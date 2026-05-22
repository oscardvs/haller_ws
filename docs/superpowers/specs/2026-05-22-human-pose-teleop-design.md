# Human-Pose Teleop for SO-101 — Design

- **Status:** approved (brainstorming output) — implementation plan pending
- **Date:** 2026-05-22
- **Scope:** A new "Human Teleop" mode in the Haller HMI that drives both SO-101 arms simultaneously from a single laptop webcam, using in-browser MediaPipe pose + hand tracking, joint-angle retargeting, and a spacebar dead-man clutch.
- **Predecessor designs:** [`2026-05-22-haller-unified-hmi-design.md`](./2026-05-22-haller-unified-hmi-design.md) (HMI shell + leader/follower teleop), [`2026-05-22-calibration-wizard-design.md`](./2026-05-22-calibration-wizard-design.md) (per-arm calibration).

---

## 1. Goal

Add a new operator workflow to the HMI: **stand in front of your laptop, move your arms, and the two SO-101 arms move with you.** Bimanual from day one, monocular RGB only, in-browser pose estimation, with a live camera-feed-plus-skeleton overlay visible in the HMI. Heavy compute runs on the operator's machine (or in v2, a separate desktop/cloud pose worker). The Jetson Orin Nano only does what it already does: run the HMI backend + lerobot at 60 Hz.

This sits alongside the existing leader/follower teleop, not on top of it. Only one teleop kind runs at a time, enforced by a shared session lock.

## 2. Non-goals

- **Remote operator over WAN.** v1 assumes operator + browser + backend are co-located (or LAN).
- **Depth-anchored absolute cartesian mapping.** Reason we picked joint-angle retargeting: monocular RGB Z is too noisy for direct EE-position teleop. RealSense / stereo / IK comes in v2.
- **Finger articulation.** SO-101 has no fingers; thumb-index pinch is the entire gripper signal.
- **Multi-operator sessions.** One operator at a time.
- **Dataset recording / VLA integration / haptics / force-torque safety beyond joint limits.** All deferred.

## 3. Approaches considered

| Axis | Picked | Rejected |
|---|---|---|
| Capture | Laptop webcam (RGB), browser `getUserMedia` | RealSense (v2), VR headset (v2) |
| Pose model location | In-browser MediaPipe Tasks (HandLandmarker + PoseLandmarker) | Python worker on desktop *(v2 upgrade slot)*, raw-video upload to a remote service |
| EE controller | Joint-angle retargeting, 1:1 topology (SEW-Mimic-style closed-form) | Cartesian IK / Pinocchio dex-retargeting |
| Mapping | Bimanual, mirror-default with swap | Single-arm only; robot-eye fixed orientation |
| Clutch | Spacebar hold-to-drive (dead-man) | Toggle start/stop; gesture (conflicts with pinch=gripper) |
| Tracking loss | Freeze at last commanded pose | Decay-to-home; drop-torque |

### Related prior art

- **[chichonnade/Vision-Based-Hand-Shadowing](https://github.com/chichonnade/Vision-Based-Hand-Shadowing)** — closest existing parallel: SO-ARM101 + MediaPipe Hands + IK. Differences: uses RGB-D egocentric camera, cartesian DLS IK in PyBullet, offline/replay. We diverge on monocular RGB, angular retargeting, and realtime closed-loop.
- **[phospho-app/phosphobot](https://github.com/phospho-app/phosphobot)** — dominant community UI for SO-100/SO-101; covers keyboard, gamepad, drag, and Meta Quest VR teleop. We differentiate by being in-HMI, bimanual, and requiring no headset.
- **[SEW-Mimic (arXiv 2602.01632)](https://arxiv.org/abs/2602.01632)** — closed-form geometric retargeting solver from human shoulder/elbow/wrist to robot joints. Algorithmic spine of `retarget.py`.
- **[dex-retargeting](https://github.com/dexsuite/dex-retargeting)** — Pinocchio-based, would be the v2 path if we ever go to higher-DoF arms.
- **[OpenTeleVision](https://arxiv.org/abs/2407.01512)** / **[Bunny-VisionPro](https://dingry.github.io/projects/bunny_visionpro.html)** — gold standard for VR-headset teleop ergonomics; not direct comparators but inform the v3 roadmap.

## 4. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Browser (operator's laptop, has the camera)            │
│  • new page /teleop/human                                │
│  • <video> + <canvas> overlay (mirror preview)           │
│  • MediaPipe Tasks for Web: HandLandmarker + PoseLandm.  │
│  • spacebar dead-man state                               │
│  • publishes raw keypoint frames over WS                 │
└────────────────────────┬─────────────────────────────────┘
                         │  WS: /ws/teleop/human/in
                         │  ~30 Hz; raw keypoints only
                         ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI backend (existing haller_hmi process)           │
│  • new haller_hmi/human_teleop.py                        │
│      - keypoints → joint angles (retarget module)        │
│      - low-pass smoothing → 60 Hz commit loop            │
│      - dead-man + tracking-loss state machine            │
│      - clamps to each follower's calibrated joint limits │
│  • new REST: /teleop/human/{start,stop,swap,calibrate}   │
│  • status surfaced over existing /ws/telemetry           │
└────────────────────────┬─────────────────────────────────┘
                         │  lerobot.send_action (existing)
                         ▼
                two SO-101 arms (60 Hz each)
```

### Key seams

- **Retargeting is server-side.** Single source of truth; unit-testable in Python; browser only owns capture + overlay + dead-man.
- **`HumanTeleopSession` mirrors `TeleopSession`.** Shares safety primitives (Mode, clamp_joint_goal, E-STOP path).
- **One teleop at a time.** A shared session lock in `server.py` enforces mutual exclusion with leader/follower teleop. Starting one while the other runs returns HTTP 409.
- **Hot-swappable pose source.** In-browser MediaPipe in v1; remote pose worker (WiLoR/HaMeR on desktop) in v2 reuses the same WS schema. Zero backend rework expected.

## 5. Components

### Backend (`hmi/backend/haller_hmi/`)

| File | Purpose | Depends on |
|---|---|---|
| `retarget.py` *(new)* | Pure functions: MediaPipe keypoints → SO-101 joint angles. No side effects. SEW-Mimic-style closed-form math lives here in isolation. | numpy only |
| `human_teleop.py` *(new)* | `HumanTeleopSession`. Owns the 60 Hz commit loop, dead-man state machine, smoothing filter, tracking-loss timeout, per-arm clamping. | `arm.py`, `safety.py`, `retarget.py` |
| `server.py` *(edit)* | New REST: `/teleop/human/{start,stop,swap,calibrate}`. New WS: `/ws/teleop/human/in`. Shared session lock with `teleop.py`. | `human_teleop.py`, `teleop.py` |
| `telemetry.py` *(edit)* | Include `human_teleop` block in `/ws/telemetry` frames. | — |
| `teleop.py` *(edit, minimal)* | Cooperate with the shared session lock — refuse start if `HumanTeleopSession.running`. | — |

### Frontend (`hmi/frontend/`)

| File | Purpose |
|---|---|
| `app/teleop/human/page.tsx` *(new)* | Route, composes the panel. |
| `components/HumanTeleopPanel.tsx` *(new)* | Orchestrator: camera permission, MediaPipe lifecycle, WS connection, start/stop, swap, pinch calibration, status. |
| `components/CameraOverlay.tsx` *(new)* | `<video>` + `<canvas>` overlay. Renders body and hand skeletons + pinch line. Pure visualization. |
| `components/DeadManIndicator.tsx` *(new)* | Visual chip for spacebar state. |
| `components/ScopeBar.tsx` *(new)* | Read-only per-joint bar with center tick, limit ticks, commanded fill, and "intended" ghost tick. |
| `components/PinchCalibrationStep.tsx` *(new)* | Per-hand open/pinch capture. |
| `lib/mediapipe.ts` *(new)* | Loads HandLandmarker + PoseLandmarker; fuses results into a `KeypointFrame`. |
| `lib/humanTeleopClient.ts` *(new)* | WS client; bundles latest keypoints + dead-man state; reconnect-aware. |

### Module boundaries

```
┌────────────────────────────── browser ──────────────────────────────┐
│  HumanTeleopPanel ──► CameraOverlay   (renders only)                │
│        ├──► lib/mediapipe.ts          (in: video, out: KeypointFrame)│
│        ├──► lib/humanTeleopClient.ts  (sends KeypointFrame over WS) │
│        └──► REST /teleop/human/{start,stop,swap,calibrate}          │
└─────────────────────────── WS keypoints ────────────────────────────┘
                                  │
┌──────────────────────────── backend ────────────────────────────────┐
│  server.py (FastAPI) ──► HumanTeleopSession.ingest_frame(...)       │
│                                                                      │
│  HumanTeleopSession                                                  │
│   ├─ uses retarget.compute_joint_angles(keypoints) → JointGoal      │
│   ├─ smoothing filter (one-pole LPF, τ ≈ 100 ms)                    │
│   ├─ dead-man state machine: idle/armed/tracking/driving            │
│   ├─ tracking-loss timer (no keypoints > 300 ms → freeze)           │
│   ├─ 60 Hz commit thread → ArmHandle.robot.send_action (existing)   │
│   └─ E-STOP hook: same path as TeleopSession                        │
│                                                                      │
│  retarget.py — pure functions                                       │
│   ├─ compute_arm_angles(S, E, W) → (pan, lift, elbow_flex)          │
│   ├─ compute_wrist_angles(F, hand_landmarks) → (wrist_flex, roll)   │
│   ├─ compute_pinch(thumb_tip, index_tip, calib) → gripper ∈ [0,1]   │
│   └─ apply_mirror(angles, swap_flag) → angles                        │
│                                                                      │
│  safety.py / arm.py — unchanged surface; clamp_joint_goal reused    │
└──────────────────────────────────────────────────────────────────────┘
```

### What stays unchanged

`arm.py`, `safety.py`, calibration loading, the existing leader/follower `TeleopSession`, the E-STOP route, the base panel, telemetry transport.

## 6. Data flow

### Per tick (camera side, ~33 ms apart)

```
[1] getUserMedia frame
[2] MediaPipe HandLandmarker + PoseLandmarker (in-browser)
[3] CameraOverlay draws skeleton on <canvas>
[4] lib/humanTeleopClient packages KeypointFrame
[5] WS send → /ws/teleop/human/in

[6] HumanTeleopSession.ingest(frame)
     ├─ stash latest KeypointFrame + arrival ts
     ├─ if dead_man=true and frame fresh → update tracking flag
     └─ retarget.compute(...) → JointGoal (left + right)

(parallel) 60 Hz commit thread:
[7] read latest_goal + smoothing state
[8] one-pole LPF: smoothed += alpha * (target - smoothed)
[9] clamp_joint_goal(smoothed, follower.joint_limits_deg)
[10] state==driving? → arm.robot.send_action(...)
     state!=driving? → write nothing (servos hold)
```

### Wire schemas

**Browser → backend** (`/ws/teleop/human/in`, ~30 Hz):

```jsonc
{
  "type": "keypoints",
  "ts_ms": 1716392842123,          // browser monotonic, for staleness
  "dead_man": true,                // spacebar held
  "pinch_calib": {                 // optional; sent once per session
    "left":  { "min_m": 0.020, "max_m": 0.180 },
    "right": { "min_m": 0.022, "max_m": 0.175 }
  },
  "left": {                        // null if not detected this frame
    "pose":  { "shoulder": [x,y,z], "elbow": [x,y,z], "wrist": [x,y,z] },
    "hand":  {
      "wrist":      [x,y,z],
      "thumb_tip":  [x,y,z],
      "index_tip":  [x,y,z],
      "index_mcp":  [x,y,z],
      "middle_mcp": [x,y,z],
      "pinky_mcp":  [x,y,z]
    },
    "confidence": 0.92             // min(pose.visibility, hand.score)
  },
  "right": { ... }                  // same shape, or null
}
```

Coordinates are MediaPipe world-meters (origin at hip center for `pose`, hand-center for `hand`). The browser does no frame translation; backend handles all re-rooting.

**Backend → frontend** (folded into existing `/ws/telemetry`, 20 Hz):

```jsonc
{
  /* …existing base + arms + leader-follower teleop blocks unchanged… */
  "human_teleop": {
    "running": true,
    "state": "driving",                // idle | armed | tracking | driving
    "swap": false,
    "frame_age_ms": 38,
    "tracking": {
      "left":  { "conf": 0.92, "lost": false },
      "right": { "conf": 0.88, "lost": false }
    },
    "goal_deg": {
      "left":  { "shoulder_pan": -12.3, "shoulder_lift": 22.0, "elbow_flex": 47.5,
                 "wrist_flex": -3.2,  "wrist_roll": 18.1, "gripper": 0.42 },
      "right": { ... }
    },
    "last_error": null
  }
}
```

### Retargeting math (high level — full derivations in `retarget.py`)

Given shoulder **S**, elbow **E**, wrist **W**, and the hand landmark set:

- Upper arm vector U = E − S, forearm vector F = W − E.
- `shoulder_pan` = `atan2(U.x, −U.z)`
- `shoulder_lift` = `asin(U.y / |U|)`
- `elbow_flex` = π − `angle(U, F)`
- Build a forearm frame: F̂ along forearm; palm normal n̂ = `(index_mcp − wrist) × (pinky_mcp − wrist)`; up vector û = n̂ × F̂.
- `wrist_flex` = signed angle between F̂ and `(middle_mcp − hand_wrist)`, projected onto F̂×û plane.
- `wrist_roll` = signed rotation of n̂ around F̂ relative to the upper-arm-defined "neutral" palm normal.
- `gripper` = `clip((|thumb_tip − index_tip| − min_m) / (max_m − min_m), 0, 1)`.
- `apply_mirror` flips `shoulder_pan` + `wrist_roll` signs based on the swap flag and human-side → robot-side assignment.

Z is MediaPipe's noisiest axis, but inputs participate through *angles* (atan2, asin, dot/cross), which are far more robust than absolute positions.

### Smoothing & rate

- MediaPipe inference: ~25–40 ms/frame on a modern laptop GPU → ~25–30 Hz effective.
- Commit loop: 60 Hz (same as leader/follower teleop).
- Per-joint one-pole LPF, `alpha = 1 − exp(−dt / τ)`, `τ ≈ 100 ms`. ~70 ms settling, kills per-frame jitter without feeling draggy.
- Filter integrates only when state == `driving`. In `tracking`, the last committed servo target is held.

### Dead-man state machine

```
              session start
                    │
                    ▼
                [armed]    ◄── no keypoints yet
                    │
       first valid frame (any arm)
                    ▼
              [tracking]   ◄── streaming, spacebar not held
                    │  ▲
   spacebar down  │  │  spacebar up │ frame_age > 300 ms
                    ▼  │
              [driving]   ◄── writing to robots
```

Edge rules:
- `session stop` or E-STOP → `idle` from any state; arms restored to MANUAL with torque on (mirrors `TeleopSession.stop()`).
- `frame_age > 300 ms` while `driving` → demote that arm to `tracking`, hold last goal, flag `lost: true`. Per-arm, not session-wide.
- WS disconnect → demote to `armed`; commit loop freezes goals; auto-stop after 5 s reconnect window.

### Failure modes covered

| Failure | Behavior |
|---|---|
| Operator releases spacebar | Robots freeze in place (servos hold last target). Telemetry chip → "ready". |
| Hand goes out of frame | That arm: `tracking.lost=true`, goal held. Other arm continues. |
| WS disconnect | Session demotes to `armed`; commit loop freezes; UI shows reconnect; auto-stop at 5 s. |
| MediaPipe model fails to load | UI shows error, session never starts. |
| E-STOP pressed | Session stops, torque drops on both arms — same path as leader/follower. |
| Joint angle outside robot limits | `clamp_joint_goal` clamps silently; HMI overlay shows the gap between intended and actual via the ghost tick. |
| Camera permission denied | Session never starts; clear UI error. |

## 7. UI design (`/teleop/human`)

### Aesthetic direction

Safety-critical instrument console. Inherits the existing shadcn dark theme so it doesn't feel foreign in the HMI, but the camera viewport gets oscilloscope treatment — acid-lime skeleton, thin lines, small ticks for landmarks — so the human pose reads instantly. Three rules:

1. **The viewport is the hero.** ~65% of the screen.
2. **State is always legible.** Dead-man state has a screen-level border treatment.
3. **No gradients, no decorative chrome.** Hairline borders, mono numerals for live telemetry.

### Layout

```
┌─ TopBar (existing) ─────────────────────────── live ● ── E-STOP ──┐
│                                                                    │
│ Human Teleop                            session: driving · 38 ms   │
│ bimanual · monocular RGB · mirror                       [ stop ▸ ] │
│                                                                    │
│ ┌─────────────── viewport (16:9, ~65% width) ──────┐ ┌────────────┐│
│ │  ╔════════════════════════════════════════════╗  │ │ arm: left  ││
│ │  ║ mirrored <video> feed                       ║  │ │  pan   ▮▮▯ -12.3°│
│ │  ║ skeleton overlay drawn on <canvas>          ║  │ │  lift  ▮▮▮ +22.0°│
│ │  ║   ⊿ shoulder—elbow—wrist (per side)         ║  │ │  elbow ▮▮▮ +47.5°│
│ │  ║   21-pt hand landmarks (lime ticks)         ║  │ │  wflex ▯▮▯ -3.2° │
│ │  ║   pinch line between thumb & index          ║  │ │  wroll ▮▯▯ +18.1°│
│ │  ║                                             ║  │ │  grip  ▮▮▯  0.42 │
│ │  ╚═══════════════════════════════ overlay HUD ╝  │ │ tracking 0.92 ● ││
│ │   left ●0.92  right ●0.88   30.2 fps   age 38ms │ ├────────────┤│
│ │                                                    │ │ arm: right ││
│ │   [ ── DRIVE — hold SPACE ── ] pulse when held    │ │ …          ││
│ └────────────────────────────────────────────────────┘ └────────────┘│
│                                                                    │
│ ┌─ assignment ──────────────┐  ┌─ pinch calibration ─────────────┐│
│ │ default: mirror     [ ⇄ ] │  │ left   open … pinch  → save     ││
│ │ left hand  → arm: left    │  │ right  open … pinch  → save     ││
│ │ right hand → arm: right   │  │ last saved: 18s ago             ││
│ └───────────────────────────┘  └─────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
```

### Viewport details

- `<video>` mirrored with `transform: scaleX(-1)`, object-fit cover, fixed 16:9.
- `<canvas>` stacked above with the same transform, redrawn each MediaPipe result.
  - Body skeleton: shoulder→elbow→wrist segments, 2 px stroke, in `--instrument-line` (acid lime, roughly `oklch(80% 0.18 142)`).
  - Hand: 21 landmarks as 4 px ticks; finger bones connecting them; slightly brighter.
  - **Pinch line** between thumb and index tips: 1.5 px stroke, dashed when pinch < 30%, solid when pinch > 30%. Sole "gripper closing" cue.
  - **Confidence ghosting**: side with confidence < 0.6 fades to 40% opacity. Lost tracking → frozen, amber.
- **HUD chips**, bottom-left: `left ●0.92  right ●0.88` (tracking conf), `30.2 fps · age 38 ms` (mono). Low-contrast 60% alpha on a 50% scrim so they never compete with the skeleton.
- **DRIVE banner**, bottom-center:
  - Idle: muted gray "DRIVE — hold SPACE".
  - Held: text → acid lime; a thin lime border pulses around the *entire viewport* at ~2 Hz. The pulse is the dominant change when driving.
  - Tracking lost while held: amber "HOLD — tracking lost (left)".

### Side panel (per arm, × 2)

Read-only **scope strips** (`ScopeBar`), not draggable sliders. Each joint shows:
- Center tick (0°), ticks at each clamp limit.
- Filled segment = commanded (smoothed, clamped).
- Ghost tick = intended (pre-clamp, pre-smoothing). Visible only when intended ≠ commanded — makes saturation and smoothing lag explicit.
- Right-aligned monospace degree readout.

### Bottom strip

- **Assignment**: title, `mirror ⇄` toggle (default on), two read-only mapping labels. Same `⇄` icon as the existing leader/follower swap.
- **Pinch calibration**: two-step capture per hand ("open" → max distance; "pinch" → min). Persisted to `localStorage` per session.

### State coverage

| State | Viewport border | DRIVE banner | Side panels | Top-right chip |
|---|---|---|---|---|
| `idle` | none | hidden | empty | "stopped" |
| `armed` | dim white | "waiting for camera…" | dashed placeholders | "armed" |
| `tracking` | thin neutral | "DRIVE — hold SPACE" muted | live values, not driving | "ready" |
| `driving` | **pulsing lime** | lime "DRIVING" | live values + ghost ticks | "driving" |
| tracking lost (any side) | amber on lost side's panel | "HOLD — tracking lost (side)" | lost side amber, other normal | "partial" |
| E-STOP | red, animation halts | hidden | grayed, frozen | (global red banner takes over) |

### Keyboard surface

- `Space` (hold) — dead-man drive. `keydown`/`keyup` on the page; ignored when an input is focused.
- `S` — swap arm assignment (same as `⇄`).
- `Esc` — stop session (not E-STOP).
- E-STOP is deliberately *not* bound to a keyboard shortcut — must remain a deliberate click.

### Reuse

- Reuses shadcn `Card`, `Button`, `Badge`, the existing `EStopButton`, the existing `TelemetryBar` chip styling, the existing color tokens.
- New components: `CameraOverlay`, `ScopeBar`, `DeadManIndicator`, `PinchCalibrationStep`, plus the page itself.

## 8. Safety surface (defense in depth)

| Layer | Mechanism | Where it lives |
|---|---|---|
| **L0 — global E-STOP** | Existing `/estop` route. Human teleop registers with the same hook as `TeleopSession`. | `server.py` (existing), `human_teleop.py` (new hook) |
| **L1 — session exclusivity** | Single session lock guards both `TeleopSession` and `HumanTeleopSession`. Starting one while the other runs → HTTP 409. | `server.py` (new shared lock) |
| **L2 — dead-man gate** | Commit loop writes `send_action` only when `state == driving`. Releasing spacebar drops to `tracking` within one tick (~16 ms). | `human_teleop.py` |
| **L3 — tracking-loss freeze** | Per-arm: `frame_age > 300 ms` while driving → demote, hold last goal. Other arm continues. | `human_teleop.py` |
| **L4 — joint-limit clamp** | Every commanded angle through existing `clamp_joint_goal`. Same path as leader/follower. | `safety.py` (existing) |
| **L5 — rate-of-change cap** | One-pole LPF (τ ≈ 100 ms) + hard cap on max delta per joint per tick (≤4°/tick at 60 Hz ≈ 240°/s). | `human_teleop.py` |
| **L6 — WS health** | Disconnect → demote to `armed`; auto-stop after 5 s. | `human_teleop.py` + `humanTeleopClient.ts` |

## 9. Testing strategy

### Backend, pytest (extending existing 25-test suite)

| File | Coverage |
|---|---|
| `test_retarget.py` *(new)* | Synthetic keypoints → expected joint angles (within 0.5°). Cases: arm straight forward, elbow 90°, wrist roll ±90°, pinch open/closed, mirror flip correctness, NaN/Inf-safe handling. |
| `test_human_teleop.py` *(new)* | Session lifecycle (`start` → `armed` → `tracking` → `driving` → `stop`). Dead-man transitions. Tracking-loss timer at 300 ms. Per-arm independence. Smoothing convergence. Rate cap. Session lock prevents starting both teleops. |
| `test_routes.py` *(edit)* | New REST shapes, 409 conflict cases, 400 validation. WS accepts well-formed frames, rejects malformed. |

All hardware paths mocked at the `ArmHandle.robot` boundary — same pattern as existing `test_arm.py`.

### Frontend, vitest (extending existing 6 tests)

| File | Coverage |
|---|---|
| `humanTeleopClient.test.ts` *(new)* | WS framing; spacebar state in outbound frames; pinch-calib once per session; reconnect on close. |
| `mediapipe.test.ts` *(new, light)* | Keypoint-fusion shim: combines HandLandmarker + PoseLandmarker results into a `KeypointFrame`; handles no-hand / no-pose gracefully. MediaPipe mocked. |
| `CameraOverlay.test.tsx` *(new, light)* | Renders without crashing on representative inputs; switches to amber on tracking-loss prop; freezes on `state="tracking"`. Pixel-perfect canvas testing out of scope. |

### Manual smoke tests (added to README)

1. Cold start with no camera → permission prompt → error state → no robot motion.
2. Calibrate pinch → engage with spacebar → wave one arm → second arm stays in place.
3. Mid-drive: hand exits frame → that arm freezes, other continues, chip turns amber.
4. Mid-drive: release spacebar → both arms freeze within ~16 ms.
5. Global E-STOP while driving → session stops, torque drops, E-STOP banner.
6. Try to start leader/follower while human teleop is running → 409, no state change.

### Performance budgets (acceptance criteria)

| Metric | Budget | Rationale |
|---|---|---|
| Browser pipeline frame time (capture → WS send) | < 35 ms | Allows 25–30 Hz on a modern integrated-GPU laptop. |
| Backend ingest → commit | < 5 ms | Trivial; mostly numpy + dict ops. |
| End-to-end glass-to-servo latency | < 120 ms median | Headroom for cloud-side compute in v2. |
| Joint-angle accuracy on a static pose | ±2° of ground truth | Validated against a posed checkerboard + protractor reference. |

## 10. Out of scope & follow-ups

### Out of scope for v1

- Remote operator over WAN; WebRTC/NAT plumbing.
- Depth-anchored absolute cartesian mapping (RealSense / stereo / IK).
- Finger articulation beyond pinch.
- Multi-operator sessions.
- Dataset recording / LeRobot dataset emission.
- VLA / autonomy interaction (blending policy + human).
- Haptic feedback.
- Cartesian force / torque safety beyond joint limits.

### v2 follow-ups (rough priority)

1. **Remote pose worker (Python, on desktop)** — swaps in-browser MediaPipe for a server-side worker. Browser ships frames over WebRTC; worker runs WiLoR/HaMeR-class models; publishes keypoints on the *same* WS schema. No backend change.
2. **RealSense / stereo depth anchor** — adds true 3D wrist position. Enables a second mapping mode: cartesian "reach extension" via clutch.
3. **Per-operator profile** — pinch calibration, swap default, smoothing τ, rate cap, persisted in `~/.haller/`.
4. **Recording-aware teleop** — tag a session as "demonstration"; emit a LeRobot-compatible dataset shard at stop. The imitation-learning on-ramp.
5. **Smoothness/quality dashboard** — post-session report: max delta/s per joint, average tracking confidence, % time driving, jerk metrics.

### v3+ (speculative)

- WebXR / Quest mode via the `XRHand` API — in-HMI architecture preserved, capture mode swapped.
- Active-camera robot head (à la Open-TeleVision).
- Foot-pedal dead-man via WebHID.

### Risks already accepted

| Risk | Why we're accepting it |
|---|---|
| Monocular RGB Z-axis noise leaks into wrist angles | Using *angles* (not positions) bounds the error to a few degrees; smoothing absorbs most of it. |
| MediaPipe occlusion during fast motion | Per-arm independence + tracking-loss freeze; the operator sees the failure on the overlay in real time. |
| Operator outside SO-101's reachable joint range | Hard clamp; UI ghost tick makes the gap visible. |
| Browser-to-backend WS jitter | LPF + commit loop hold last value through brief pauses; tracking-loss timer catches longer ones. |
| Two operators racing on start | Session lock + idempotent stop. Loser gets a 409. |

## 11. Open questions

None blocking v1. Items to resolve during implementation:

- Exact `--instrument-line` color token — pick during the first UI pass, validate against ambient lighting in the lab.
- Whether to expose `τ` and the rate cap as URL params for tuning, or hardcode for v1. Lean: hardcode + log effective values.
- Per-arm pinch calibration persistence key — `localStorage`-by-arm-id is fine, but if we later add per-operator profiles, migrate to a server-side store.
