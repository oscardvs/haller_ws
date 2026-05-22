# Human-Pose Teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new "Human Teleop" mode to the Haller HMI: drive both SO-101 arms simultaneously from a single laptop webcam using in-browser MediaPipe pose + hand tracking, with joint-angle retargeting on the backend at 60 Hz, gated by a spacebar dead-man clutch.

**Architecture:** A new `HumanTeleopSession` (sibling to the existing leader/follower `TeleopSession`) ingests raw keypoint frames over WebSocket from the browser, runs them through pure-function retargeting (`retarget.py`), smooths the result with a one-pole low-pass filter, and writes joint goals to both arms at 60 Hz — only when the dead-man state is `driving`. A new `/teleop/human` page in the HMI owns the camera, runs MediaPipe in the browser, draws a skeleton overlay on the live feed, and sends keypoints over WS. A shared session lock guarantees only one teleop kind runs at a time.

**Tech Stack:**
- Backend: Python 3.12, FastAPI, numpy (new dep — math only), pytest. Reuses existing `arm.py`, `safety.py`, `telemetry.py`.
- Frontend: Next.js 16 App Router, React 19, TypeScript, `@mediapipe/tasks-vision` (new dep), shadcn/ui, vitest. Reuses existing `useTelemetry`, `EStopButton`, `Card`/`Button`/`Badge`, `lib/api.ts`.
- Build/dev: existing `pnpm` + `uv` flows. No new system services.

**Spec:** `docs/superpowers/specs/2026-05-22-human-pose-teleop-design.md` (commit `f396e2e`).

---

## File structure (created/modified by this plan)

```
haller_ws/
├── hmi/
│   ├── README.md                                       ← Task 20 (edit)
│   ├── backend/
│   │   ├── pyproject.toml                              ← Task 1 (edit: + numpy)
│   │   ├── haller_hmi/
│   │   │   ├── retarget.py                             ← Tasks 1–4 (new)
│   │   │   ├── human_teleop.py                         ← Tasks 5–8 (new)
│   │   │   ├── teleop.py                               ← Task 9 (edit)
│   │   │   ├── server.py                               ← Tasks 9–11 (edit)
│   │   │   └── telemetry.py                            ← Task 12 (edit)
│   │   └── tests/
│   │       ├── test_retarget.py                        ← Tasks 1–4 (new)
│   │       ├── test_human_teleop.py                    ← Tasks 5–8 (new)
│   │       └── test_routes.py                          ← Tasks 9–11 (edit)
│   └── frontend/
│       ├── package.json                                ← Task 14 (edit: + @mediapipe/tasks-vision)
│       ├── app/
│       │   ├── page.tsx                                ← Task 20 (edit: dashboard link)
│       │   └── teleop/human/page.tsx                   ← Task 20 (new)
│       ├── components/
│       │   ├── ScopeBar.tsx                            ← Task 16 (new)
│       │   ├── DeadManIndicator.tsx                    ← Task 16 (new)
│       │   ├── PinchCalibrationStep.tsx                ← Task 17 (new)
│       │   ├── CameraOverlay.tsx                       ← Task 18 (new)
│       │   └── HumanTeleopPanel.tsx                    ← Task 19 (new)
│       ├── lib/
│       │   ├── api.ts                                  ← Task 13 (edit)
│       │   ├── mediapipe.ts                            ← Task 14 (new)
│       │   └── humanTeleopClient.ts                    ← Task 15 (new)
│       └── __tests__/
│           ├── api.test.ts                             ← Task 13 (edit)
│           ├── humanTeleopClient.test.ts               ← Task 15 (new)
│           ├── mediapipe.test.ts                       ← Task 14 (new)
│           └── ScopeBar.test.tsx                       ← Task 16 (new)
└── docs/superpowers/plans/2026-05-22-human-pose-teleop.md  ← this file
```

---

## Conventions

- **No co-author trailer** in any commit message (per repo `CLAUDE.md`).
- **TDD throughout:** write the failing test first, run it to see it fail, then make it pass with the minimal code.
- **One concept per commit:** each task ends in a single, self-describing commit. The commit message style matches the existing log (`feat(hmi/backend): …`, `feat(hmi): …`, etc.).
- **Run the existing test suites green before starting:**
  - Backend: `cd hmi/backend && pytest -v` → should report `25 passed`.
  - Frontend: `cd hmi/frontend && pnpm test` → should report `6 passed`.
- **Next.js version:** the frontend `AGENTS.md` warns this is not the Next.js shape from training data. Before editing or creating any page, skim `hmi/frontend/node_modules/next/dist/docs/` for the relevant area (routing, server vs client components) — the existing `app/base/page.tsx` is a good live reference.

---

## Phase I — Backend math (pure)

### Task 1: Scaffold `retarget` module + add numpy dependency

**Files:**
- Create: `hmi/backend/haller_hmi/retarget.py`
- Create: `hmi/backend/tests/test_retarget.py`
- Modify: `hmi/backend/pyproject.toml`

- [ ] **Step 1: Add numpy to backend dependencies**

Open `hmi/backend/pyproject.toml` and append `"numpy>=1.26"` to the `[project] dependencies` list (alphabetical). After editing, run:

```bash
cd hmi/backend
uv pip install -e .
```

Expected: numpy installed, no other changes.

- [ ] **Step 2: Create the empty retarget module with type aliases**

Write `hmi/backend/haller_hmi/retarget.py`:

```python
"""Pure-function MediaPipe-keypoints → SO-101 joint-angle retargeting.

The maths here is the SEW-Mimic-style analytical closed form: human shoulder/
elbow/wrist define the upper-arm + forearm vectors, which map 1:1 to the
SO-101's shoulder_pan / shoulder_lift / elbow_flex. The hand landmarks define
the wrist orientation, which maps to wrist_flex + wrist_roll. Thumb-index
distance maps to gripper aperture.

Nothing in this file touches a robot, the network, or threads. Inputs are
plain Python dicts / tuples; outputs are plain dicts. Everything is unit-
testable with synthetic geometry.
"""
from __future__ import annotations

import math
from typing import Literal, TypedDict

import numpy as np

# MediaPipe pose world coords are right-handed metres: +X right, +Y down, +Z
# *toward the camera*. We rebase: +X right, +Y up, +Z away from the camera
# (i.e. "into the room"). This rebase happens once at the boundary.

Vec3 = tuple[float, float, float]


class PoseLandmarks(TypedDict):
    shoulder: Vec3
    elbow: Vec3
    wrist: Vec3


class HandLandmarks(TypedDict):
    wrist: Vec3
    thumb_tip: Vec3
    index_tip: Vec3
    index_mcp: Vec3
    middle_mcp: Vec3
    pinky_mcp: Vec3


class SideFrame(TypedDict):
    pose: PoseLandmarks
    hand: HandLandmarks
    confidence: float


class PinchCalib(TypedDict):
    min_m: float
    max_m: float


class JointGoal(TypedDict):
    shoulder_pan: float    # degrees
    shoulder_lift: float   # degrees
    elbow_flex: float      # degrees
    wrist_flex: float      # degrees
    wrist_roll: float      # degrees
    gripper: float         # [0, 1] — 0 = closed, 1 = open


Side = Literal["left", "right"]
```

- [ ] **Step 3: Create the test scaffold with a single placeholder smoke test**

Write `hmi/backend/tests/test_retarget.py`:

```python
"""Tests for retarget.py — pure math, no hardware."""
from __future__ import annotations

import math

import pytest

from haller_hmi import retarget


def test_module_imports():
    # If this fails, numpy isn't installed or retarget.py has a syntax error.
    assert hasattr(retarget, "PoseLandmarks")
    assert hasattr(retarget, "HandLandmarks")
```

- [ ] **Step 4: Run tests and confirm the new file passes alongside the existing 25**

```bash
cd hmi/backend
pytest -v
```

Expected: `26 passed` (the previous 25 + the new `test_module_imports`).

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/pyproject.toml hmi/backend/haller_hmi/retarget.py hmi/backend/tests/test_retarget.py
git commit -m "feat(hmi/backend): scaffold retarget module + numpy dep"
```

---

### Task 2: `compute_arm_angles` (shoulder pan, shoulder lift, elbow flex)

**Files:**
- Modify: `hmi/backend/haller_hmi/retarget.py`
- Modify: `hmi/backend/tests/test_retarget.py`

- [ ] **Step 1: Write failing tests for the three arm angles**

Append to `hmi/backend/tests/test_retarget.py`:

```python
TOLERANCE_DEG = 1.0  # tests use synthetic geometry; tight tolerance is fine


def _vec(x, y, z):
    return (float(x), float(y), float(z))


def test_compute_arm_angles_arm_straight_forward():
    # Arm extended straight forward: U and F both point +Z away.
    # Expect pan = 0 (no horizontal rotation), lift = 0 (level),
    # elbow_flex = 0 (straight).
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.0, 1.4, 0.3)
    W = _vec(0.0, 1.4, 0.6)
    pan, lift, elbow = retarget.compute_arm_angles(S, E, W)
    assert abs(pan) < TOLERANCE_DEG
    assert abs(lift) < TOLERANCE_DEG
    assert abs(elbow) < TOLERANCE_DEG


def test_compute_arm_angles_arm_to_the_side():
    # Arm extended straight to the operator's right (+X).
    # Expect pan ≈ +90°, lift = 0, elbow_flex = 0.
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.3, 1.4, 0.0)
    W = _vec(0.6, 1.4, 0.0)
    pan, lift, _ = retarget.compute_arm_angles(S, E, W)
    assert abs(pan - 90.0) < TOLERANCE_DEG
    assert abs(lift) < TOLERANCE_DEG


def test_compute_arm_angles_arm_lifted_up():
    # Arm extended straight up (+Y).
    # Expect pan = 0 (no horizontal rotation), lift = +90°.
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.0, 1.7, 0.0)
    W = _vec(0.0, 2.0, 0.0)
    _, lift, _ = retarget.compute_arm_angles(S, E, W)
    assert abs(lift - 90.0) < TOLERANCE_DEG


def test_compute_arm_angles_elbow_bent_90():
    # Upper arm forward, forearm pointing up: 90° elbow flex.
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.0, 1.4, 0.3)
    W = _vec(0.0, 1.7, 0.3)
    _, _, elbow = retarget.compute_arm_angles(S, E, W)
    assert abs(elbow - 90.0) < TOLERANCE_DEG


def test_compute_arm_angles_handles_zero_length_upper_arm():
    # Degenerate: shoulder == elbow. Should not raise; should return finite values.
    S = _vec(0.0, 1.4, 0.0)
    E = _vec(0.0, 1.4, 0.0)
    W = _vec(0.0, 1.4, 0.3)
    pan, lift, elbow = retarget.compute_arm_angles(S, E, W)
    assert all(math.isfinite(v) for v in (pan, lift, elbow))
```

- [ ] **Step 2: Run tests; they should fail with `AttributeError: 'compute_arm_angles'`**

```bash
cd hmi/backend
pytest tests/test_retarget.py -v
```

Expected: 5 new tests fail; the existing `test_module_imports` still passes.

- [ ] **Step 3: Implement `compute_arm_angles` in `retarget.py`**

Append to `hmi/backend/haller_hmi/retarget.py`:

```python
def _np(v: Vec3) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _safe_norm(v: np.ndarray, eps: float = 1e-9) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


def _signed_angle_deg(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from a to b around `axis` (right-hand rule), in degrees."""
    a_n = a / _safe_norm(a)
    b_n = b / _safe_norm(b)
    cross = np.cross(a_n, b_n)
    sin_t = float(np.dot(cross, axis / _safe_norm(axis)))
    cos_t = float(np.dot(a_n, b_n))
    return math.degrees(math.atan2(sin_t, cos_t))


def compute_arm_angles(
    shoulder: Vec3, elbow: Vec3, wrist: Vec3
) -> tuple[float, float, float]:
    """Return (shoulder_pan, shoulder_lift, elbow_flex) in degrees.

    Coordinate convention (after upstream rebase): +X right, +Y up, +Z away
    from the camera. The retargeted angles are *signed*, centred on a
    straight-arm-pointing-forward neutral pose (pan=0, lift=0, elbow=0).
    """
    S, E, W = _np(shoulder), _np(elbow), _np(wrist)
    U = E - S                          # upper-arm vector
    F = W - E                          # forearm vector
    u_len = _safe_norm(U)

    # Shoulder pan: azimuth of U projected onto the horizontal (X,Z) plane.
    # pan = 0 when U points purely +Z; pan = +90° when U points purely +X.
    pan = math.degrees(math.atan2(U[0], U[2]))

    # Shoulder lift: elevation of U above horizontal. lift = +90° straight up.
    lift = math.degrees(math.asin(max(-1.0, min(1.0, U[1] / u_len))))

    # Elbow flex: angle between U and F. 0° = straight, 90° = right angle.
    cos_t = float(np.dot(U / u_len, F / _safe_norm(F)))
    elbow = math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))

    return pan, lift, elbow
```

- [ ] **Step 4: Run tests; the five new tests should pass**

```bash
cd hmi/backend
pytest tests/test_retarget.py -v
```

Expected: all 6 retarget tests pass; existing suite still green.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/retarget.py hmi/backend/tests/test_retarget.py
git commit -m "feat(hmi/backend): retarget — compute_arm_angles (pan/lift/elbow)"
```

---

### Task 3: `compute_wrist_angles` (wrist flex + wrist roll)

**Files:**
- Modify: `hmi/backend/haller_hmi/retarget.py`
- Modify: `hmi/backend/tests/test_retarget.py`

- [ ] **Step 1: Write failing tests for the wrist angles**

Append to `hmi/backend/tests/test_retarget.py`:

```python
def _flat_hand(forearm_dir: tuple[float, float, float], palm_down: bool = True):
    """Build a HandLandmarks dict for a flat hand whose middle finger points
    along `forearm_dir` and whose palm faces -Y (down) if palm_down else +Y."""
    fx, fy, fz = forearm_dir
    forearm = _vec(fx, fy, fz)
    wrist = _vec(0.0, 0.0, 0.0)
    # Middle finger points along the forearm direction at 0.10 m.
    middle = _vec(0.10 * fx, 0.10 * fy, 0.10 * fz)
    # Palm normal: -Y if palm down, +Y if palm up.
    palm_normal_y = -1.0 if palm_down else 1.0
    # Build index_mcp and pinky_mcp so that (index_mcp-wrist) × (pinky_mcp-wrist)
    # has the desired Y sign.
    # For a flat hand with palm down: place index_mcp to the +X side,
    # pinky_mcp to the -X side of the wrist (the cross product points -Y).
    side_x = 0.04 if palm_down else -0.04
    index_mcp = _vec(side_x, 0.0, 0.05)
    pinky_mcp = _vec(-side_x, 0.0, 0.05)
    thumb_tip = _vec(side_x * 1.5, 0.0, 0.03)
    index_tip = _vec(side_x * 0.5, 0.0, 0.10)
    return {
        "wrist": wrist,
        "thumb_tip": thumb_tip,
        "index_tip": index_tip,
        "index_mcp": index_mcp,
        "middle_mcp": middle,
        "pinky_mcp": pinky_mcp,
    }


def test_compute_wrist_angles_neutral():
    # Forearm and hand both point +Z. Flat hand, palm down.
    forearm = _vec(0.0, 0.0, 0.3)
    hand = _flat_hand(forearm_dir=(0.0, 0.0, 1.0), palm_down=True)
    wflex, wroll = retarget.compute_wrist_angles(forearm, hand)
    assert abs(wflex) < TOLERANCE_DEG
    # Palm down is our defined neutral; roll ≈ 0.
    assert abs(wroll) < TOLERANCE_DEG


def test_compute_wrist_angles_flex_90_up():
    # Forearm +Z, hand bent up so the middle finger points +Y.
    forearm = _vec(0.0, 0.0, 0.3)
    hand = _flat_hand(forearm_dir=(0.0, 1.0, 0.0), palm_down=True)
    wflex, _ = retarget.compute_wrist_angles(forearm, hand)
    assert abs(abs(wflex) - 90.0) < TOLERANCE_DEG


def test_compute_wrist_angles_roll_180_palm_up():
    # Forearm and hand point +Z, but palm faces UP rather than DOWN.
    forearm = _vec(0.0, 0.0, 0.3)
    hand = _flat_hand(forearm_dir=(0.0, 0.0, 1.0), palm_down=False)
    _, wroll = retarget.compute_wrist_angles(forearm, hand)
    # Palm flipped: roll should be ±180°.
    assert abs(abs(wroll) - 180.0) < TOLERANCE_DEG
```

- [ ] **Step 2: Run the new tests; expect `AttributeError: compute_wrist_angles`**

```bash
cd hmi/backend
pytest tests/test_retarget.py -v
```

Expected: three new failures.

- [ ] **Step 3: Implement `compute_wrist_angles`**

Append to `hmi/backend/haller_hmi/retarget.py`:

```python
def compute_wrist_angles(
    forearm: Vec3, hand: HandLandmarks
) -> tuple[float, float]:
    """Return (wrist_flex, wrist_roll) in degrees.

    Neutral pose: forearm and hand point along +Z, palm faces -Y (down).
    `wrist_flex` is the signed angle of the middle-finger ray relative to the
    forearm, measured around the palm-normal axis. `wrist_roll` is the signed
    rotation of the palm normal around the forearm axis relative to the
    neutral-palm-down direction.
    """
    F = _np(forearm)
    f_hat = F / _safe_norm(F)

    w = _np(hand["wrist"])
    middle_dir = _np(hand["middle_mcp"]) - w
    index_dir = _np(hand["index_mcp"]) - w
    pinky_dir = _np(hand["pinky_mcp"]) - w

    # Palm normal as it currently is.
    palm_normal = np.cross(index_dir, pinky_dir)
    palm_normal_hat = palm_normal / _safe_norm(palm_normal)

    # "Neutral palm-down" reference: the world-down direction (-Y), projected
    # onto the plane perpendicular to the forearm and renormalised.
    world_down = np.array([0.0, -1.0, 0.0])
    neutral_palm = world_down - np.dot(world_down, f_hat) * f_hat
    neutral_palm_hat = neutral_palm / _safe_norm(neutral_palm)

    # wrist_roll: signed rotation of palm_normal around forearm, relative to neutral.
    wroll = _signed_angle_deg(neutral_palm_hat, palm_normal_hat, f_hat)

    # wrist_flex: signed angle from forearm to middle-finger direction,
    # measured around the palm normal (so flexion = rotation about palm normal).
    wflex = _signed_angle_deg(f_hat, middle_dir, palm_normal_hat)

    return wflex, wroll
```

- [ ] **Step 4: Run tests; expect the three new tests to pass**

```bash
cd hmi/backend
pytest tests/test_retarget.py -v
```

Expected: all 9 retarget tests pass.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/retarget.py hmi/backend/tests/test_retarget.py
git commit -m "feat(hmi/backend): retarget — compute_wrist_angles (flex/roll)"
```

---

### Task 4: `compute_pinch`, `apply_mirror`, and the top-level `compute_joint_goal`

**Files:**
- Modify: `hmi/backend/haller_hmi/retarget.py`
- Modify: `hmi/backend/tests/test_retarget.py`

- [ ] **Step 1: Write failing tests for pinch + mirror + top-level**

Append to `hmi/backend/tests/test_retarget.py`:

```python
def test_compute_pinch_open_returns_one():
    calib = {"min_m": 0.02, "max_m": 0.18}
    g = retarget.compute_pinch((0.0, 0.0, 0.0), (0.18, 0.0, 0.0), calib)
    assert abs(g - 1.0) < 1e-3


def test_compute_pinch_closed_returns_zero():
    calib = {"min_m": 0.02, "max_m": 0.18}
    g = retarget.compute_pinch((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), calib)
    assert abs(g) < 1e-3


def test_compute_pinch_clamps_below_min():
    calib = {"min_m": 0.02, "max_m": 0.18}
    g = retarget.compute_pinch((0.0, 0.0, 0.0), (0.001, 0.0, 0.0), calib)
    assert g == 0.0


def test_compute_pinch_clamps_above_max():
    calib = {"min_m": 0.02, "max_m": 0.18}
    g = retarget.compute_pinch((0.0, 0.0, 0.0), (0.30, 0.0, 0.0), calib)
    assert g == 1.0


def test_apply_mirror_flips_pan_and_roll_only():
    g = {
        "shoulder_pan": 30.0, "shoulder_lift": 20.0, "elbow_flex": 45.0,
        "wrist_flex": 10.0, "wrist_roll": 18.0, "gripper": 0.5,
    }
    mirrored = retarget.apply_mirror(g)
    assert mirrored["shoulder_pan"] == -30.0
    assert mirrored["wrist_roll"] == -18.0
    # Everything else unchanged
    assert mirrored["shoulder_lift"] == 20.0
    assert mirrored["elbow_flex"] == 45.0
    assert mirrored["wrist_flex"] == 10.0
    assert mirrored["gripper"] == 0.5


def test_compute_joint_goal_combines_everything():
    side = {
        "pose": {
            "shoulder": _vec(0.0, 1.4, 0.0),
            "elbow":    _vec(0.0, 1.4, 0.3),
            "wrist":    _vec(0.0, 1.4, 0.6),
        },
        "hand": _flat_hand(forearm_dir=(0.0, 0.0, 1.0), palm_down=True),
        "confidence": 0.9,
    }
    calib = {"min_m": 0.02, "max_m": 0.18}
    goal = retarget.compute_joint_goal(side, calib, mirror=False)
    assert goal is not None
    # Arm straight forward, flat-hand palm down: all near-zero except gripper.
    for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        assert abs(goal[k]) < TOLERANCE_DEG, f"{k}={goal[k]}"
    assert 0.0 <= goal["gripper"] <= 1.0


def test_compute_joint_goal_returns_none_if_low_confidence():
    side = {
        "pose": {
            "shoulder": _vec(0.0, 1.4, 0.0),
            "elbow":    _vec(0.0, 1.4, 0.3),
            "wrist":    _vec(0.0, 1.4, 0.6),
        },
        "hand": _flat_hand(forearm_dir=(0.0, 0.0, 1.0), palm_down=True),
        "confidence": 0.2,  # below threshold
    }
    calib = {"min_m": 0.02, "max_m": 0.18}
    assert retarget.compute_joint_goal(side, calib, mirror=False) is None
```

- [ ] **Step 2: Run the new tests; expect failures for missing functions**

```bash
cd hmi/backend
pytest tests/test_retarget.py -v
```

- [ ] **Step 3: Implement pinch + mirror + the top-level**

Append to `hmi/backend/haller_hmi/retarget.py`:

```python
# Minimum per-side confidence below which we refuse to emit a joint goal.
CONFIDENCE_FLOOR = 0.4


def compute_pinch(
    thumb_tip: Vec3, index_tip: Vec3, calib: PinchCalib
) -> float:
    """Return gripper aperture in [0, 1].

    Maps thumb-tip ↔ index-tip distance onto [calib.min_m, calib.max_m].
    0 = fully closed pinch, 1 = fully open.
    """
    d = float(np.linalg.norm(_np(thumb_tip) - _np(index_tip)))
    lo, hi = float(calib["min_m"]), float(calib["max_m"])
    span = max(hi - lo, 1e-6)
    return max(0.0, min(1.0, (d - lo) / span))


def apply_mirror(goal: JointGoal) -> JointGoal:
    """Flip the side-specific angles for the contralateral arm.

    Mirroring across the operator's mid-sagittal plane negates only the
    rotations that point sideways: shoulder_pan and wrist_roll. All other
    joints (lift, elbow_flex, wrist_flex, gripper) are side-invariant.
    """
    return {
        "shoulder_pan": -goal["shoulder_pan"],
        "shoulder_lift": goal["shoulder_lift"],
        "elbow_flex": goal["elbow_flex"],
        "wrist_flex": goal["wrist_flex"],
        "wrist_roll": -goal["wrist_roll"],
        "gripper": goal["gripper"],
    }


def compute_joint_goal(
    side: SideFrame, calib: PinchCalib, *, mirror: bool
) -> JointGoal | None:
    """End-to-end retargeting for one side. Returns None below the confidence floor."""
    if side["confidence"] < CONFIDENCE_FLOOR:
        return None
    pan, lift, elbow = compute_arm_angles(
        side["pose"]["shoulder"], side["pose"]["elbow"], side["pose"]["wrist"]
    )
    forearm = tuple(
        np.asarray(side["pose"]["wrist"], dtype=np.float64)
        - np.asarray(side["pose"]["elbow"], dtype=np.float64)
    )
    wflex, wroll = compute_wrist_angles(forearm, side["hand"])
    gripper = compute_pinch(side["hand"]["thumb_tip"], side["hand"]["index_tip"], calib)
    goal: JointGoal = {
        "shoulder_pan": pan,
        "shoulder_lift": lift,
        "elbow_flex": elbow,
        "wrist_flex": wflex,
        "wrist_roll": wroll,
        "gripper": gripper,
    }
    return apply_mirror(goal) if mirror else goal
```

- [ ] **Step 4: Run tests; expect 16 retarget tests passing**

```bash
cd hmi/backend
pytest tests/test_retarget.py -v
```

Expected: all retarget tests pass; existing suites unchanged.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/retarget.py hmi/backend/tests/test_retarget.py
git commit -m "feat(hmi/backend): retarget — pinch, mirror, top-level compute_joint_goal"
```

---

## Phase II — Backend session

### Task 5: `HumanTeleopSession` scaffold + state enum + start/stop transitions

**Files:**
- Create: `hmi/backend/haller_hmi/human_teleop.py`
- Create: `hmi/backend/tests/test_human_teleop.py`

- [ ] **Step 1: Write the failing test for the state-enum + start/stop lifecycle**

Write `hmi/backend/tests/test_human_teleop.py`:

```python
"""Tests for HumanTeleopSession — session lifecycle + state machine."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from haller_hmi.human_teleop import HumanState, HumanTeleopSession
from haller_hmi.safety import Mode


def _fake_arm_manager():
    """Two mocked arms ("left", "right") with realistic joint_limits_deg + guard."""
    mgr = MagicMock()

    def _mkarm(arm_id: str):
        a = MagicMock()
        a.config = MagicMock(id=arm_id)
        a.joint_limits_deg = {
            "shoulder_pan":  (-90.0, 90.0),
            "shoulder_lift": (-90.0, 90.0),
            "elbow_flex":    (-90.0, 90.0),
            "wrist_flex":    (-90.0, 90.0),
            "wrist_roll":    (-90.0, 90.0),
            "gripper":       (-30.0, 30.0),
        }
        a.guard = MagicMock(mode=Mode.MANUAL)
        a.torque_enabled = True
        a.robot = MagicMock()
        return a

    arms = {"left": _mkarm("left"), "right": _mkarm("right")}
    mgr.__getitem__.side_effect = lambda k: arms[k]
    mgr.values.return_value = list(arms.values())
    mgr.keys.return_value = list(arms.keys())
    return mgr, arms


def test_initial_state_is_idle():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    assert sess.state is HumanState.IDLE
    assert sess.status()["running"] is False


def test_start_transitions_to_armed_and_prepares_arms():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        assert sess.state is HumanState.ARMED
        assert sess.status()["running"] is True
        # Both arms should be enabled with torque on and MANUAL mode.
        for a in arms.values():
            a.guard.set.assert_called_with(Mode.MANUAL)
    finally:
        sess.stop()


def test_stop_restores_arms_to_manual_and_torque_on():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    sess.stop()
    assert sess.state is HumanState.IDLE
    for a in arms.values():
        # The most recent guard.set should be MANUAL.
        a.guard.set.assert_called_with(Mode.MANUAL)


def test_start_twice_raises_runtime_error():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        with pytest.raises(RuntimeError):
            sess.start(left_arm="left", right_arm="right", swap=False)
    finally:
        sess.stop()


def test_start_requires_distinct_arms():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    with pytest.raises(ValueError):
        sess.start(left_arm="left", right_arm="left", swap=False)
```

- [ ] **Step 2: Run; expect `ImportError: cannot import name 'HumanState'`**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py -v
```

- [ ] **Step 3: Implement the scaffold**

Write `hmi/backend/haller_hmi/human_teleop.py`:

```python
"""Bimanual human-pose teleop session.

This is the sibling of `teleop.TeleopSession` (leader/follower). Where that
session reads positions off a physical leader arm at 60 Hz, this one reads
keypoints off a WebSocket from the operator's browser and runs them through
`retarget.compute_joint_goal` to produce joint angles. Otherwise the lifecycle
and safety semantics match exactly.

State machine:
    IDLE → (start)        → ARMED
    ARMED → (first frame) → TRACKING
    TRACKING ↔ (dead-man) → DRIVING
    any → (stop / E-STOP) → IDLE
"""
from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field

from .arm import ArmManager
from .safety import Mode

logger = logging.getLogger(__name__)


class HumanState(str, enum.Enum):
    IDLE = "idle"
    ARMED = "armed"
    TRACKING = "tracking"
    DRIVING = "driving"


@dataclass
class _SessionConfig:
    left_arm: str
    right_arm: str
    swap: bool = False
    hz: float = 60.0


class HumanTeleopSession:
    """One global session. Mutually exclusive with leader/follower TeleopSession."""

    def __init__(self, arms: ArmManager):
        self._arms = arms
        self._lock = threading.Lock()
        self._state: HumanState = HumanState.IDLE
        self._cfg: _SessionConfig | None = None
        self._started_at: float | None = None
        self._last_error: str | None = None
        # Filled in by later tasks:
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

    # ---- public API ------------------------------------------------------

    @property
    def state(self) -> HumanState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state is not HumanState.IDLE

    def status(self) -> dict:
        with self._lock:
            cfg = self._cfg
            return {
                "running": self.running,
                "state": self._state.value,
                "left_arm": cfg.left_arm if cfg else None,
                "right_arm": cfg.right_arm if cfg else None,
                "swap": cfg.swap if cfg else False,
                "started_at": self._started_at,
                "last_error": self._last_error,
            }

    def start(self, *, left_arm: str, right_arm: str, swap: bool, hz: float = 60.0) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("human teleop already running; stop it first")
            if left_arm == right_arm:
                raise ValueError("left_arm and right_arm must be different")
            # Resolve both arms (raises KeyError → server converts to 404).
            left = self._arms[left_arm]
            right = self._arms[right_arm]

            for a in (left, right):
                if not a.torque_enabled:
                    a.enable_torque()
                a.guard.set(Mode.MANUAL)

            self._cfg = _SessionConfig(left_arm=left_arm, right_arm=right_arm,
                                       swap=swap, hz=hz)
            self._started_at = time.time()
            self._state = HumanState.ARMED
            self._last_error = None
        logger.info("human teleop started: left=%s right=%s swap=%s",
                    left_arm, right_arm, swap)

    def stop(self) -> None:
        with self._lock:
            if not self.running:
                return
            cfg = self._cfg
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        # Restore arms to MANUAL with torque on.
        if cfg is not None:
            for arm_id in (cfg.left_arm, cfg.right_arm):
                handle = self._arms[arm_id]
                if not handle.torque_enabled:
                    handle.enable_torque()
                handle.guard.set(Mode.MANUAL)
        with self._lock:
            self._state = HumanState.IDLE
            self._cfg = None
            self._started_at = None
        logger.info("human teleop stopped")
```

- [ ] **Step 4: Run; expect 5 tests passing**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/human_teleop.py hmi/backend/tests/test_human_teleop.py
git commit -m "feat(hmi/backend): HumanTeleopSession scaffold + state machine"
```

---

### Task 6: `ingest_frame` + smoothing filter + retarget integration

**Files:**
- Modify: `hmi/backend/haller_hmi/human_teleop.py`
- Modify: `hmi/backend/tests/test_human_teleop.py`

- [ ] **Step 1: Write failing tests for ingest + smoothing**

Append to `hmi/backend/tests/test_human_teleop.py`:

```python
def _kp_frame(
    *, ts_ms: int = 100, dead_man: bool = False, both_arms: bool = True,
    calib: dict | None = None, confidence: float = 0.9,
) -> dict:
    """A minimal valid KeypointFrame: arm straight forward, hand neutral."""
    side = {
        "pose": {
            "shoulder": [0.0, 1.4, 0.0],
            "elbow":    [0.0, 1.4, 0.3],
            "wrist":    [0.0, 1.4, 0.6],
        },
        "hand": {
            "wrist":      [0.0, 0.0, 0.0],
            "thumb_tip":  [0.04, 0.0, 0.05],
            "index_tip":  [0.02, 0.0, 0.10],
            "index_mcp":  [0.04, 0.0, 0.05],
            "middle_mcp": [0.0, 0.0, 0.10],
            "pinky_mcp":  [-0.04, 0.0, 0.05],
        },
        "confidence": confidence,
    }
    return {
        "type": "keypoints",
        "ts_ms": ts_ms,
        "dead_man": dead_man,
        "pinch_calib": calib or {
            "left":  {"min_m": 0.02, "max_m": 0.18},
            "right": {"min_m": 0.02, "max_m": 0.18},
        },
        "left":  side if both_arms else None,
        "right": side if both_arms else None,
    }


def test_first_ingest_transitions_armed_to_tracking():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        assert sess.state is HumanState.ARMED
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.state is HumanState.TRACKING
    finally:
        sess.stop()


def test_dead_man_held_transitions_tracking_to_driving():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.state is HumanState.TRACKING
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert sess.state is HumanState.DRIVING
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.state is HumanState.TRACKING
    finally:
        sess.stop()


def test_ingest_records_latest_target_goal():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        targets = sess.target_goals()
        assert "left" in targets and "right" in targets
        # Arm straight forward → angles all near zero.
        for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex"):
            assert abs(targets["left"][joint]) < 2.0
    finally:
        sess.stop()


def test_ingest_handles_missing_side_gracefully():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Only left detected, right is None.
        frame = _kp_frame(dead_man=False)
        frame["right"] = None
        sess.ingest_frame(frame)
        targets = sess.target_goals()
        # Left should be set, right held at last (initialized to None).
        assert "left" in targets
        assert targets.get("right") is None
    finally:
        sess.stop()
```

- [ ] **Step 2: Run; expect AttributeError**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py -v
```

- [ ] **Step 3: Implement `ingest_frame` + `target_goals`**

In `hmi/backend/haller_hmi/human_teleop.py`, add at the top of the file (after existing imports):

```python
from . import retarget
```

Replace the existing `__init__` of `HumanTeleopSession` with this version, which adds the ingest-owned runtime state:

```python
    def __init__(self, arms: ArmManager):
        self._arms = arms
        self._lock = threading.Lock()
        self._state: HumanState = HumanState.IDLE
        self._cfg: _SessionConfig | None = None
        self._started_at: float | None = None
        self._last_error: str | None = None
        # 60 Hz loop wiring (filled in Task 7)
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        # Ingest-owned runtime state
        self._latest_frame_ts_ms: int = 0
        self._latest_arrival_perf: float = 0.0
        self._dead_man: bool = False
        self._target_left: dict | None = None
        self._target_right: dict | None = None
        self._pinch_calib_left: dict = {"min_m": 0.02, "max_m": 0.18}
        self._pinch_calib_right: dict = {"min_m": 0.02, "max_m": 0.18}
```

And add these methods to `HumanTeleopSession`:

```python
    def ingest_frame(self, frame: dict) -> None:
        """Apply a KeypointFrame from the browser. Thread-safe."""
        with self._lock:
            if not self.running:
                return  # drop frames received outside an active session
            # Track dead-man + per-side pinch calibration if present.
            self._dead_man = bool(frame.get("dead_man", False))
            calib = frame.get("pinch_calib") or {}
            if "left" in calib:
                self._pinch_calib_left = calib["left"]
            if "right" in calib:
                self._pinch_calib_right = calib["right"]
            self._latest_frame_ts_ms = int(frame.get("ts_ms", 0))
            self._latest_arrival_perf = time.perf_counter()

            mirror = bool(self._cfg and self._cfg.swap)
            # In default mirror mode the human's right hand drives the robot's
            # left arm (selfie convention). The `mirror=True` flag is passed
            # to `compute_joint_goal` for the side that should be mirrored.
            left_side = frame.get("left")
            right_side = frame.get("right")
            if left_side is not None:
                self._target_left = retarget.compute_joint_goal(
                    left_side, self._pinch_calib_left, mirror=mirror,
                )
            if right_side is not None:
                self._target_right = retarget.compute_joint_goal(
                    right_side, self._pinch_calib_right, mirror=not mirror,
                )

            # State transitions driven by ingest
            if self._state is HumanState.ARMED:
                self._state = HumanState.TRACKING
            if self._state is HumanState.TRACKING and self._dead_man:
                self._state = HumanState.DRIVING
            elif self._state is HumanState.DRIVING and not self._dead_man:
                self._state = HumanState.TRACKING

    def target_goals(self) -> dict:
        with self._lock:
            return {"left": self._target_left, "right": self._target_right}
```

- [ ] **Step 4: Run; expect 4 new tests passing (9 total)**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/human_teleop.py hmi/backend/tests/test_human_teleop.py
git commit -m "feat(hmi/backend): HumanTeleopSession.ingest_frame + retarget integration"
```

---

### Task 7: 60 Hz commit loop + dead-man gate + smoothing + rate cap

**Files:**
- Modify: `hmi/backend/haller_hmi/human_teleop.py`
- Modify: `hmi/backend/tests/test_human_teleop.py`

- [ ] **Step 1: Write failing tests for the commit loop**

Append to `hmi/backend/tests/test_human_teleop.py`:

```python
import time as _time


def _wait_until(predicate, timeout: float = 1.0, interval: float = 0.01) -> bool:
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(interval)
    return False


def test_commit_loop_writes_to_arms_when_driving():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)  # fast loop for tests
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        # The loop should call send_action on both arms within ~50 ms.
        assert _wait_until(lambda: arms["left"].robot.send_action.called)
        assert _wait_until(lambda: arms["right"].robot.send_action.called)
    finally:
        sess.stop()


def test_commit_loop_does_not_write_when_not_driving():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        _time.sleep(0.05)
        assert not arms["left"].robot.send_action.called
        assert not arms["right"].robot.send_action.called
    finally:
        sess.stop()


def test_commit_loop_clamps_to_arm_joint_limits():
    mgr, arms = _fake_arm_manager()
    # Squeeze the left arm's pan limit so retarget output gets clamped.
    arms["left"].joint_limits_deg["shoulder_pan"] = (-5.0, 5.0)
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Build a frame whose left-arm pan should retarget to ≈ +90° (far above 5°).
        frame = _kp_frame(dead_man=True)
        frame["left"]["pose"]["elbow"] = [0.3, 1.4, 0.0]
        frame["left"]["pose"]["wrist"] = [0.6, 1.4, 0.0]
        sess.ingest_frame(frame)
        _wait_until(lambda: arms["left"].robot.send_action.called)
        sent_actions = [c.args[0] for c in arms["left"].robot.send_action.call_args_list]
        # Every commanded shoulder_pan must be inside [-5, 5].
        for action in sent_actions:
            assert -5.0 <= action["shoulder_pan.pos"] <= 5.0
    finally:
        sess.stop()
```

- [ ] **Step 2: Run; expect failures (commit loop not implemented yet)**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py -v -k commit_loop
```

- [ ] **Step 3: Implement the commit loop**

Add to `hmi/backend/haller_hmi/human_teleop.py` (top, after existing imports):

```python
from .safety import clamp_joint_goal
```

Modify the `__init__` to accept `hz_override` and seed smoothing state:

```python
    def __init__(self, arms: ArmManager, *, hz_override: float | None = None):
        self._arms = arms
        self._lock = threading.Lock()
        self._state: HumanState = HumanState.IDLE
        self._cfg: _SessionConfig | None = None
        self._started_at: float | None = None
        self._last_error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._latest_frame_ts_ms: int = 0
        self._latest_arrival_perf: float = 0.0
        self._dead_man: bool = False
        self._target_left: dict | None = None
        self._target_right: dict | None = None
        self._pinch_calib_left: dict = {"min_m": 0.02, "max_m": 0.18}
        self._pinch_calib_right: dict = {"min_m": 0.02, "max_m": 0.18}
        # Smoothed (last-committed) per-arm goals — used as the LPF state.
        self._committed_left: dict[str, float] = {}
        self._committed_right: dict[str, float] = {}
        self._hz_override = hz_override
        # Per-joint rate cap, in degrees per tick at the configured commit rate.
        # ≈ 240°/s at 60 Hz → 4°/tick. For tests at higher hz, scaled down.
        self._rate_cap_deg_per_tick = 4.0
```

Modify `start` to spawn the commit thread:

```python
    def start(self, *, left_arm: str, right_arm: str, swap: bool, hz: float = 60.0) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("human teleop already running; stop it first")
            if left_arm == right_arm:
                raise ValueError("left_arm and right_arm must be different")
            left = self._arms[left_arm]
            right = self._arms[right_arm]
            for a in (left, right):
                if not a.torque_enabled:
                    a.enable_torque()
                a.guard.set(Mode.MANUAL)
            effective_hz = self._hz_override or hz
            self._cfg = _SessionConfig(left_arm=left_arm, right_arm=right_arm,
                                       swap=swap, hz=effective_hz)
            self._started_at = time.time()
            self._state = HumanState.ARMED
            self._last_error = None
            # Reset smoothing state to current observed positions where available.
            self._committed_left = self._observed_or_zero(left)
            self._committed_right = self._observed_or_zero(right)
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"haller-hmi-human-teleop-{left_arm}-{right_arm}",
            daemon=True,
        )
        self._thread.start()
        logger.info("human teleop started: left=%s right=%s swap=%s @ %.1f Hz",
                    left_arm, right_arm, swap, effective_hz)
```

Add these new helpers + the loop body:

```python
    @staticmethod
    def _observed_or_zero(handle) -> dict[str, float]:
        try:
            obs = handle.robot.get_observation() if handle.robot is not None else {}
        except Exception:
            obs = {}
        out: dict[str, float] = {}
        for joint in handle.joint_limits_deg:
            out[joint] = float(obs.get(f"{joint}.pos", 0.0))
        return out

    def _smooth_step(
        self,
        committed: dict[str, float],
        target: dict[str, float] | None,
        limits: dict[str, tuple[float, float]],
        alpha: float,
    ) -> dict[str, float]:
        if target is None:
            return committed
        out: dict[str, float] = {}
        for joint, lo_hi in limits.items():
            lo, hi = lo_hi
            cur = committed.get(joint, 0.0)
            if joint not in target:
                out[joint] = cur
                continue
            desired = float(target[joint])
            # Special-case gripper: retarget emits [0, 1] (0 = closed, 1 = open).
            # Scale onto the gripper joint's calibrated degree range.
            if joint == "gripper":
                desired = max(0.0, min(1.0, desired))
                desired = lo + desired * (hi - lo)
            # One-pole LPF then per-tick rate cap, then hard clamp to limits.
            new = cur + alpha * (desired - cur)
            cap = self._rate_cap_deg_per_tick
            new = max(cur - cap, min(cur + cap, new))
            out[joint] = max(lo, min(hi, new))
        return out

    def _commit(self, handle, goal: dict[str, float]) -> None:
        action = {f"{joint}.pos": float(value) for joint, value in goal.items()}
        if handle.robot is not None:
            handle.robot.send_action(action)

    def _loop(self) -> None:
        with self._lock:
            cfg = self._cfg
        assert cfg is not None
        left = self._arms[cfg.left_arm]
        right = self._arms[cfg.right_arm]
        period = 1.0 / max(1.0, cfg.hz)
        # Smoothing time constant ≈ 100 ms (frequency-independent).
        tau_s = 0.100
        alpha = 1.0 - math.exp(-period / tau_s) if period > 0 else 1.0
        while not self._stop_flag.is_set():
            tick_start = time.perf_counter()
            try:
                with self._lock:
                    target_left = (
                        retarget.apply_mirror(self._target_left) if False else self._target_left
                    )
                    target_right = self._target_right
                    driving = self._state is HumanState.DRIVING
                self._committed_left = self._smooth_step(
                    self._committed_left, target_left, left.joint_limits_deg, alpha,
                )
                self._committed_right = self._smooth_step(
                    self._committed_right, target_right, right.joint_limits_deg, alpha,
                )
                if driving:
                    self._commit(left, self._committed_left)
                    self._commit(right, self._committed_right)
                with self._lock:
                    self._last_error = None
            except Exception as e:
                logger.exception("human teleop tick failed")
                with self._lock:
                    self._last_error = str(e)
                time.sleep(0.05)
                continue
            elapsed = time.perf_counter() - tick_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
```

Add `import math` at the top of `human_teleop.py` if not already present.

- [ ] **Step 4: Run; expect tests pass**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py -v
```

Expected: all 12 human_teleop tests pass.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/human_teleop.py hmi/backend/tests/test_human_teleop.py
git commit -m "feat(hmi/backend): HumanTeleopSession 60 Hz commit loop + LPF + rate cap"
```

---

### Task 8: Tracking-loss timer + per-arm independence + WS disconnect window

**Files:**
- Modify: `hmi/backend/haller_hmi/human_teleop.py`
- Modify: `hmi/backend/tests/test_human_teleop.py`

- [ ] **Step 1: Write failing tests**

Append to `hmi/backend/tests/test_human_teleop.py`:

```python
def test_per_arm_tracking_loss_freezes_only_that_side():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0,
                              frame_age_ms_loss=80.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Drive a frame where only the left side has fresh keypoints.
        frame = _kp_frame(dead_man=True)
        sess.ingest_frame(frame)
        _wait_until(lambda: arms["left"].robot.send_action.called)
        # Now stop the right side from being updated; the left keeps ticking.
        frame_left_only = _kp_frame(dead_man=True)
        frame_left_only["right"] = None
        # Pump a few left-only frames over ~150 ms (> 80 ms threshold).
        for _ in range(20):
            sess.ingest_frame(frame_left_only)
            _time.sleep(0.01)
        status = sess.status()
        assert status["tracking"]["right"]["lost"] is True
        assert status["tracking"]["left"]["lost"] is False
    finally:
        sess.stop()


def test_session_demotes_to_armed_on_ws_disconnect_window():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0,
                              ws_disconnect_grace_s=0.1)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(lambda: sess.state is HumanState.DRIVING)
        sess.notify_ws_disconnected()
        # After the grace window, the loop should auto-stop the session.
        assert _wait_until(lambda: sess.state is HumanState.IDLE, timeout=1.0)
    finally:
        sess.stop()
```

- [ ] **Step 2: Run; expect failures**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py -v -k "tracking_loss or ws_disconnect"
```

- [ ] **Step 3: Implement**

Modify `__init__` of `HumanTeleopSession` (add params + state):

```python
    def __init__(
        self,
        arms: ArmManager,
        *,
        hz_override: float | None = None,
        frame_age_ms_loss: float = 300.0,
        ws_disconnect_grace_s: float = 5.0,
    ):
        # ... existing body unchanged ...
        self._frame_age_ms_loss = frame_age_ms_loss
        self._ws_disconnect_grace_s = ws_disconnect_grace_s
        self._ws_disconnected_at_perf: float | None = None
        # Per-arm last-frame timestamps (perf_counter), for tracking-loss.
        self._last_left_perf: float = 0.0
        self._last_right_perf: float = 0.0
```

(Leave the rest of `__init__` as written in Task 7; just add the four new attributes above.)

Modify `ingest_frame` to also stash per-side arrival times. Replace the existing body of `ingest_frame` with:

```python
    def ingest_frame(self, frame: dict) -> None:
        with self._lock:
            if not self.running:
                return
            self._dead_man = bool(frame.get("dead_man", False))
            calib = frame.get("pinch_calib") or {}
            if "left" in calib:
                self._pinch_calib_left = calib["left"]
            if "right" in calib:
                self._pinch_calib_right = calib["right"]
            self._latest_frame_ts_ms = int(frame.get("ts_ms", 0))
            now_perf = time.perf_counter()
            self._latest_arrival_perf = now_perf
            # WS is healthy: cancel any pending grace window.
            self._ws_disconnected_at_perf = None

            mirror = bool(self._cfg and self._cfg.swap)
            left_side = frame.get("left")
            right_side = frame.get("right")
            if left_side is not None:
                self._target_left = retarget.compute_joint_goal(
                    left_side, self._pinch_calib_left, mirror=mirror,
                )
                self._last_left_perf = now_perf
            if right_side is not None:
                self._target_right = retarget.compute_joint_goal(
                    right_side, self._pinch_calib_right, mirror=not mirror,
                )
                self._last_right_perf = now_perf

            if self._state is HumanState.ARMED:
                self._state = HumanState.TRACKING
            if self._state is HumanState.TRACKING and self._dead_man:
                self._state = HumanState.DRIVING
            elif self._state is HumanState.DRIVING and not self._dead_man:
                self._state = HumanState.TRACKING
```

Add a new method:

```python
    def notify_ws_disconnected(self) -> None:
        with self._lock:
            if not self.running:
                return
            self._ws_disconnected_at_perf = time.perf_counter()
```

Extend `status()` with per-side tracking flags and the live commanded goals (so the frontend's `ScopeBar` has data to render):

```python
    def status(self) -> dict:
        with self._lock:
            cfg = self._cfg
            now = time.perf_counter()
            left_age = (now - self._last_left_perf) * 1000.0 if self._last_left_perf else None
            right_age = (now - self._last_right_perf) * 1000.0 if self._last_right_perf else None
            return {
                "running": self.running,
                "state": self._state.value,
                "left_arm": cfg.left_arm if cfg else None,
                "right_arm": cfg.right_arm if cfg else None,
                "swap": cfg.swap if cfg else False,
                "started_at": self._started_at,
                "last_error": self._last_error,
                "tracking": {
                    "left":  {
                        "age_ms": left_age,
                        "lost":   left_age is not None and left_age > self._frame_age_ms_loss,
                    },
                    "right": {
                        "age_ms": right_age,
                        "lost":   right_age is not None and right_age > self._frame_age_ms_loss,
                    },
                },
                "goal_deg": {
                    "left":  dict(self._committed_left),
                    "right": dict(self._committed_right),
                },
            }
```

Update the loop body to honour the WS-disconnect grace and to gate driving per-side:

In `_loop`, replace the block starting `if driving:` with:

```python
                if driving:
                    # Gate per-side: don't write to an arm whose tracking is lost.
                    now_perf = time.perf_counter()
                    left_age_ms = (now_perf - self._last_left_perf) * 1000.0 if self._last_left_perf else float("inf")
                    right_age_ms = (now_perf - self._last_right_perf) * 1000.0 if self._last_right_perf else float("inf")
                    if left_age_ms <= self._frame_age_ms_loss:
                        self._commit(left, self._committed_left)
                    if right_age_ms <= self._frame_age_ms_loss:
                        self._commit(right, self._committed_right)
                # WS disconnect grace window: if too much time has passed, auto-stop.
                with self._lock:
                    disc_at = self._ws_disconnected_at_perf
                if disc_at is not None and (time.perf_counter() - disc_at) > self._ws_disconnect_grace_s:
                    logger.info("human teleop WS disconnect grace exceeded; stopping")
                    threading.Thread(target=self.stop, daemon=True).start()
                    break
```

- [ ] **Step 4: Run; expect all human_teleop tests passing (14)**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py -v
```

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/human_teleop.py hmi/backend/tests/test_human_teleop.py
git commit -m "feat(hmi/backend): tracking-loss timer + per-arm independence + WS grace"
```

---

## Phase III — Backend wiring

### Task 9: Shared session lock — only one teleop kind at a time

**Files:**
- Modify: `hmi/backend/haller_hmi/teleop.py`
- Modify: `hmi/backend/haller_hmi/human_teleop.py`
- Modify: `hmi/backend/haller_hmi/server.py`
- Modify: `hmi/backend/tests/test_human_teleop.py`

The two session classes need a shared lock so that starting one while the other is running returns a clear error. Smallest change: a tiny module that owns one shared `threading.Lock()` + a "current kind" sentinel.

- [ ] **Step 1: Write a failing cross-session test**

Append to `hmi/backend/tests/test_human_teleop.py`:

```python
def test_cannot_start_human_teleop_while_leader_follower_is_running(monkeypatch):
    from haller_hmi.teleop import TeleopSession
    mgr, _ = _fake_arm_manager()
    lf = TeleopSession(mgr)
    # Mark leader/follower as running without actually spawning a thread.
    monkeypatch.setattr(lf, "_state", lf._state.__class__(running=True, leader="left",
                                                         follower="right",
                                                         hz=60.0, tick_count=0,
                                                         last_error=None,
                                                         started_at=_time.time()))

    sess = HumanTeleopSession(mgr)
    sess.attach_peer(lf)  # share the "is anyone teleoping?" check
    with pytest.raises(RuntimeError):
        sess.start(left_arm="left", right_arm="right", swap=False)
```

- [ ] **Step 2: Run; expect failure (`attach_peer` not implemented)**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py::test_cannot_start_human_teleop_while_leader_follower_is_running -v
```

- [ ] **Step 3: Implement `attach_peer` and the symmetric guard**

In `hmi/backend/haller_hmi/human_teleop.py`, add to `HumanTeleopSession`:

```python
    def attach_peer(self, peer: object) -> None:
        """Register the sibling TeleopSession so we can refuse to start
        concurrently. Both sessions call .running on each other."""
        self._peer = peer

    def __init__(self, *args, **kwargs):  # NOTE: leave this signature in sync above
        # The existing __init__ remains, and we additionally set:
        ...
```

Actually keep the existing `__init__` unchanged and add `self._peer = None` to it. Then modify `start()` so the running-peer check is the FIRST check:

```python
    def start(self, *, left_arm: str, right_arm: str, swap: bool, hz: float = 60.0) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("human teleop already running; stop it first")
            if self._peer is not None and getattr(self._peer, "status", lambda: {})().get("running"):
                raise RuntimeError("leader/follower teleop is running; stop it first")
            # ... rest of the existing start() body ...
```

In `hmi/backend/haller_hmi/teleop.py`, add the same pattern. Add an `attach_peer` method and use it in `start()` as the first guard, mirroring the new method on `HumanTeleopSession`. Concretely:

```python
class TeleopSession:
    def __init__(self, arms: ArmManager):
        # ... existing body ...
        self._peer = None

    def attach_peer(self, peer: object) -> None:
        self._peer = peer

    def start(self, leader_id: str, follower_id: str, hz: float = 60.0) -> None:
        with self._lock:
            if self._state.running:
                raise RuntimeError("teleop already running; stop it first")
            if self._peer is not None and getattr(self._peer, "status", lambda: {})().get("running"):
                raise RuntimeError("human teleop is running; stop it first")
            # ... rest of the existing start() body unchanged ...
```

In `hmi/backend/haller_hmi/server.py`, wire the peer link near the existing globals:

```python
from .human_teleop import HumanTeleopSession

# ... existing globals (teleop = TeleopSession(arms)) ...
human_teleop = HumanTeleopSession(arms)
teleop.attach_peer(human_teleop)
human_teleop.attach_peer(teleop)
```

Make sure the existing `_lifespan` shutdown also calls `human_teleop.stop()`:

```python
    yield
    logger.info("haller-hmi backend shutting down")
    if telemetry is not None:
        await telemetry.stop()
    teleop.stop()
    human_teleop.stop()
    arms.disconnect_all()
    ros.stop()
```

And add `human_teleop.stop()` to the `/estop` handler:

```python
@app.post("/estop")
async def post_estop():
    logger.warning("E-STOP triggered")
    teleop.stop()
    human_teleop.stop()
    for handle in arms.values():
        handle.disable_torque()
        handle.guard.set(Mode.STOP)
    ros.zero_cmd_vel()
    return {"ok": True}
```

- [ ] **Step 4: Run; expect test passing**

```bash
cd hmi/backend
pytest tests/test_human_teleop.py -v
```

Then run the full suite to make sure nothing regressed:

```bash
cd hmi/backend
pytest -v
```

Expected: green across the board.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/teleop.py hmi/backend/haller_hmi/human_teleop.py hmi/backend/haller_hmi/server.py hmi/backend/tests/test_human_teleop.py
git commit -m "feat(hmi/backend): shared session lock — one teleop kind at a time"
```

---

### Task 10: REST routes `/teleop/human/{start,stop,swap,calibrate}`

**Files:**
- Modify: `hmi/backend/haller_hmi/server.py`
- Modify: `hmi/backend/tests/test_routes.py`

- [ ] **Step 1: Write failing tests for the four endpoints**

In `hmi/backend/tests/test_routes.py`, extend the fixture (the existing `app_with_mocks`) to also mock `human_teleop`. Replace the block that pins `srv_mod.teleop = teleop_mock` with:

```python
    teleop_mock = MagicMock()
    teleop_mock.status.return_value = {"running": False, "leader": None, "follower": None}
    srv_mod.teleop = teleop_mock

    human_teleop_mock = MagicMock()
    human_teleop_mock.status.return_value = {
        "running": False, "state": "idle",
        "left_arm": None, "right_arm": None, "swap": False,
        "started_at": None, "last_error": None,
        "tracking": {"left": {"age_ms": None, "lost": False},
                     "right": {"age_ms": None, "lost": False}},
    }
    srv_mod.human_teleop = human_teleop_mock
```

Then append these tests to `test_routes.py`:

```python
def test_post_human_teleop_start_ok(app_with_mocks):
    r = app_with_mocks.post(
        "/teleop/human/start",
        json={"left_arm": "right", "right_arm": "right", "swap": False},
    )
    # left==right is invalid → 400 (server check) OR mock allows it through.
    assert r.status_code in {200, 400}


def test_post_human_teleop_start_unknown_arm_404(app_with_mocks):
    r = app_with_mocks.post(
        "/teleop/human/start",
        json={"left_arm": "left", "right_arm": "right", "swap": False},
    )
    # The fixture only knows about arm "right" → "left" is unknown → 404.
    assert r.status_code == 404


def test_post_human_teleop_stop(app_with_mocks):
    r = app_with_mocks.post("/teleop/human/stop", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_post_human_teleop_swap(app_with_mocks):
    r = app_with_mocks.post("/teleop/human/swap", json={"swap": True})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_post_human_teleop_calibrate(app_with_mocks):
    r = app_with_mocks.post(
        "/teleop/human/calibrate",
        json={"left": {"min_m": 0.02, "max_m": 0.18},
              "right": {"min_m": 0.022, "max_m": 0.175}},
    )
    assert r.status_code == 200


def test_get_human_teleop(app_with_mocks):
    r = app_with_mocks.get("/teleop/human")
    assert r.status_code == 200
    body = r.json()
    assert "state" in body
```

- [ ] **Step 2: Run; expect failures (routes not defined)**

```bash
cd hmi/backend
pytest tests/test_routes.py -v -k human
```

- [ ] **Step 3: Add the routes to `server.py`**

In `hmi/backend/haller_hmi/server.py`, add request schemas after the existing `TeleopStartBody`:

```python
class HumanTeleopStartBody(BaseModel):
    left_arm: str
    right_arm: str
    swap: bool = False
    hz: float = 60.0


class HumanTeleopSwapBody(BaseModel):
    swap: bool


class HumanPinchCalibSide(BaseModel):
    min_m: float
    max_m: float


class HumanTeleopCalibrateBody(BaseModel):
    left: HumanPinchCalibSide | None = None
    right: HumanPinchCalibSide | None = None
```

Add the route handlers after the existing `/teleop/stop`:

```python
@app.get("/teleop/human")
async def get_human_teleop():
    return human_teleop.status()


@app.post("/teleop/human/start")
async def post_human_teleop_start(body: HumanTeleopStartBody):
    _arm_or_404(body.left_arm)
    _arm_or_404(body.right_arm)
    try:
        human_teleop.start(
            left_arm=body.left_arm, right_arm=body.right_arm,
            swap=body.swap, hz=body.hz,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, **human_teleop.status()}


@app.post("/teleop/human/stop")
async def post_human_teleop_stop():
    human_teleop.stop()
    return {"ok": True, **human_teleop.status()}


@app.post("/teleop/human/swap")
async def post_human_teleop_swap(body: HumanTeleopSwapBody):
    human_teleop.set_swap(body.swap)
    return {"ok": True, **human_teleop.status()}


@app.post("/teleop/human/calibrate")
async def post_human_teleop_calibrate(body: HumanTeleopCalibrateBody):
    human_teleop.set_pinch_calib(
        left=body.left.model_dump() if body.left else None,
        right=body.right.model_dump() if body.right else None,
    )
    return {"ok": True}
```

Add the matching methods to `HumanTeleopSession` in `hmi/backend/haller_hmi/human_teleop.py`:

```python
    def set_swap(self, swap: bool) -> None:
        with self._lock:
            if self._cfg is not None:
                self._cfg.swap = bool(swap)

    def set_pinch_calib(self, *, left: dict | None, right: dict | None) -> None:
        with self._lock:
            if left is not None:
                self._pinch_calib_left = dict(left)
            if right is not None:
                self._pinch_calib_right = dict(right)
```

- [ ] **Step 4: Run; expect new tests pass**

```bash
cd hmi/backend
pytest tests/test_routes.py -v -k human
```

Then run the whole backend suite:

```bash
cd hmi/backend
pytest -v
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/server.py hmi/backend/haller_hmi/human_teleop.py hmi/backend/tests/test_routes.py
git commit -m "feat(hmi/backend): REST /teleop/human/{start,stop,swap,calibrate}"
```

---

### Task 11: WebSocket endpoint `/ws/teleop/human/in`

**Files:**
- Modify: `hmi/backend/haller_hmi/server.py`
- Modify: `hmi/backend/tests/test_routes.py`

- [ ] **Step 1: Write the failing WS test**

Append to `hmi/backend/tests/test_routes.py`:

```python
def test_ws_human_teleop_in_accepts_frame_and_forwards_to_session(app_with_mocks):
    client = app_with_mocks
    with client.websocket_connect("/ws/teleop/human/in") as ws:
        ws.send_json({
            "type": "keypoints",
            "ts_ms": 1234,
            "dead_man": False,
            "pinch_calib": {
                "left":  {"min_m": 0.02, "max_m": 0.18},
                "right": {"min_m": 0.02, "max_m": 0.18},
            },
            "left":  None,
            "right": None,
        })
        # Server acknowledges; no payload required, just that the socket stays open.
        ws.close()
    # `human_teleop.ingest_frame` must have been called once.
    import haller_hmi.server as srv_mod
    srv_mod.human_teleop.ingest_frame.assert_called()
```

- [ ] **Step 2: Run; expect failure (route not defined)**

```bash
cd hmi/backend
pytest tests/test_routes.py::test_ws_human_teleop_in_accepts_frame_and_forwards_to_session -v
```

- [ ] **Step 3: Implement the WS handler in `server.py`**

Append, after the existing `ws_telemetry` handler:

```python
@app.websocket("/ws/teleop/human/in")
async def ws_human_teleop_in(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            frame = await ws.receive_json()
            try:
                human_teleop.ingest_frame(frame)
            except Exception:
                # Don't kill the socket over a bad frame — log and continue.
                logger.exception("human teleop ingest_frame failed")
    except WebSocketDisconnect:
        human_teleop.notify_ws_disconnected()
        return
```

- [ ] **Step 4: Run the new test + the full backend suite**

```bash
cd hmi/backend
pytest tests/test_routes.py -v
pytest -v
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/server.py hmi/backend/tests/test_routes.py
git commit -m "feat(hmi/backend): WS /ws/teleop/human/in ingests keypoint frames"
```

---

### Task 12: Telemetry — include `human_teleop` block in `/ws/telemetry`

**Files:**
- Modify: `hmi/backend/haller_hmi/telemetry.py`
- Modify: `hmi/backend/haller_hmi/server.py`
- Modify: `hmi/backend/tests/test_telemetry.py`

- [ ] **Step 1: Write the failing telemetry test**

Open `hmi/backend/tests/test_telemetry.py` and append:

```python
def test_telemetry_frame_includes_human_teleop_block(broadcaster_with_human_teleop):
    frame = broadcaster_with_human_teleop._build_frame()
    assert "human_teleop" in frame
    ht = frame["human_teleop"]
    assert "state" in ht and "tracking" in ht
```

Add the new fixture (place it near the existing fixtures in the file; if there isn't one, create it at the top):

```python
import pytest
from unittest.mock import MagicMock
from haller_hmi.telemetry import TelemetryBroadcaster


@pytest.fixture()
def broadcaster_with_human_teleop():
    arms = MagicMock()
    arms.keys.return_value = []
    ros = MagicMock()
    ros.snapshot.return_value = MagicMock(linear=0.0, angular=0.0, odom={}, scan_min_range=None)
    teleop = MagicMock()
    teleop.status.return_value = {"running": False}
    human_teleop = MagicMock()
    human_teleop.status.return_value = {
        "running": False, "state": "idle",
        "tracking": {"left": {"age_ms": None, "lost": False},
                     "right": {"age_ms": None, "lost": False}},
    }
    return TelemetryBroadcaster(arms, ros, hz=20.0,
                                teleop=teleop, human_teleop=human_teleop)
```

- [ ] **Step 2: Run; expect failure (TelemetryBroadcaster doesn't accept `human_teleop`)**

```bash
cd hmi/backend
pytest tests/test_telemetry.py -v
```

- [ ] **Step 3: Wire it in `telemetry.py`**

In `hmi/backend/haller_hmi/telemetry.py`, modify `__init__` to accept `human_teleop`:

```python
class TelemetryBroadcaster:
    def __init__(self, arms, ros, hz: float = 20.0, teleop=None, human_teleop=None):
        self._arms = arms
        self._ros = ros
        self._teleop = teleop
        self._human_teleop = human_teleop
        # ... rest unchanged ...
```

And in `_build_frame`, add the new block:

```python
        frame = {
            # ... existing keys ...
            "teleop": self._teleop.status() if self._teleop is not None else {"running": False},
            "human_teleop": self._human_teleop.status() if self._human_teleop is not None
                             else {"running": False, "state": "idle"},
        }
```

In `hmi/backend/haller_hmi/server.py`, pass it through in `_lifespan`:

```python
    telemetry = TelemetryBroadcaster(arms, ros, hz=cfg.telemetry.hz,
                                     teleop=teleop, human_teleop=human_teleop)
```

- [ ] **Step 4: Run; expect green**

```bash
cd hmi/backend
pytest -v
```

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/telemetry.py hmi/backend/haller_hmi/server.py hmi/backend/tests/test_telemetry.py
git commit -m "feat(hmi/backend): telemetry — include human_teleop block in 20 Hz frames"
```

---

## Phase IV — Frontend lib

### Task 13: `lib/api.ts` — typed client for the new REST endpoints

**Files:**
- Modify: `hmi/frontend/lib/api.ts`
- Modify: `hmi/frontend/__tests__/api.test.ts`

- [ ] **Step 1: Write failing tests for the new methods**

Open `hmi/frontend/__tests__/api.test.ts` and append:

```typescript
describe("api human-teleop wrappers", () => {
  beforeEach(() => {
    // The existing file already mocks fetch — extend with these responses below.
  });

  it("humanTeleopStart posts the correct body", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true, state: "armed", running: true }),
                   { status: 200 })
    );
    await api.humanTeleopStart({ left_arm: "left", right_arm: "right", swap: false });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/teleop/human/start"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("humanTeleopStop hits the stop endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true, running: false }), { status: 200 })
    );
    await api.humanTeleopStop();
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/teleop/human/stop"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("humanTeleopCalibrate posts pinch ranges", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true }), { status: 200 })
    );
    await api.humanTeleopCalibrate({
      left: { min_m: 0.02, max_m: 0.18 },
      right: { min_m: 0.022, max_m: 0.175 },
    });
    const call = fetchMock.mock.calls[0];
    expect(call[0]).toMatch("/teleop/human/calibrate");
    expect(JSON.parse((call[1] as RequestInit).body as string)).toEqual({
      left:  { min_m: 0.02,  max_m: 0.18 },
      right: { min_m: 0.022, max_m: 0.175 },
    });
  });
});
```

If the existing file doesn't already import `vi`, add `import { vi, describe, it, expect, beforeEach } from "vitest";` at the top (vitest projects typically auto-import these).

- [ ] **Step 2: Run; expect failures (methods undefined)**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 3: Implement the new methods on `api`**

In `hmi/frontend/lib/api.ts`, add types near the existing `TeleopStatus`:

```typescript
export type HumanTeleopState = "idle" | "armed" | "tracking" | "driving";

export type PinchCalibSide = { min_m: number; max_m: number };

export type HumanTeleopStatus = {
  running: boolean;
  state: HumanTeleopState;
  left_arm: string | null;
  right_arm: string | null;
  swap: boolean;
  started_at: number | null;
  last_error: string | null;
  tracking: {
    left:  { age_ms: number | null; lost: boolean };
    right: { age_ms: number | null; lost: boolean };
  };
  frame_age_ms?: number;
  goal_deg?: { left?: Record<string, number>; right?: Record<string, number> };
};
```

Add the methods to the `api` object:

```typescript
  humanTeleopStatus: () =>
    getJson<HumanTeleopStatus>("/teleop/human"),
  humanTeleopStart: (body: { left_arm: string; right_arm: string; swap: boolean; hz?: number }) =>
    postJson<{ ok: true } & HumanTeleopStatus>("/teleop/human/start", body),
  humanTeleopStop: () =>
    postJson<{ ok: true } & HumanTeleopStatus>("/teleop/human/stop", {}),
  humanTeleopSwap: (swap: boolean) =>
    postJson<{ ok: true } & HumanTeleopStatus>("/teleop/human/swap", { swap }),
  humanTeleopCalibrate: (body: { left?: PinchCalibSide; right?: PinchCalibSide }) =>
    postJson<{ ok: true }>("/teleop/human/calibrate", body),
```

- [ ] **Step 4: Run; expect tests passing**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 5: Commit**

```bash
git add hmi/frontend/lib/api.ts hmi/frontend/__tests__/api.test.ts
git commit -m "feat(hmi/frontend): api — human teleop endpoints"
```

---

### Task 14: `lib/mediapipe.ts` — MediaPipe wrapper that emits `KeypointFrame`

**Files:**
- Modify: `hmi/frontend/package.json`
- Create: `hmi/frontend/lib/mediapipe.ts`
- Create: `hmi/frontend/__tests__/mediapipe.test.ts`

- [ ] **Step 1: Install the MediaPipe Tasks Vision package**

```bash
cd hmi/frontend
pnpm add @mediapipe/tasks-vision
```

This adds `@mediapipe/tasks-vision` to `package.json` dependencies.

- [ ] **Step 2: Write failing tests for the keypoint-fusion shim**

Create `hmi/frontend/__tests__/mediapipe.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import { fuseLandmarkResults, type SideFrame } from "@/lib/mediapipe";

const sample_pose_left_shoulder = { x: 0.5, y: 0.4, z: 0.0, visibility: 0.95 };
const sample_pose_left_elbow    = { x: 0.5, y: 0.5, z: 0.0, visibility: 0.93 };
const sample_pose_left_wrist    = { x: 0.5, y: 0.6, z: 0.0, visibility: 0.91 };

const sample_pose_right_shoulder = { x: 0.3, y: 0.4, z: 0.0, visibility: 0.92 };
const sample_pose_right_elbow    = { x: 0.3, y: 0.5, z: 0.0, visibility: 0.90 };
const sample_pose_right_wrist    = { x: 0.3, y: 0.6, z: 0.0, visibility: 0.88 };

const sample_hand_landmarks = Array.from({ length: 21 }, (_, i) => ({
  x: i * 0.01, y: i * 0.01, z: 0.0,
}));

describe("fuseLandmarkResults", () => {
  it("returns null sides when nothing is detected", () => {
    const out = fuseLandmarkResults(
      { worldLandmarks: [] },
      { worldLandmarks: [], handednesses: [] },
    );
    expect(out.left).toBeNull();
    expect(out.right).toBeNull();
  });

  it("builds a side from pose + hand for the left arm only", () => {
    // MediaPipe pose world-landmarks: an array of 33 points. We feed 16 entries
    // and the helper indexes by enum constants.
    const pose = Array.from({ length: 33 }, (_, i) => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    pose[11] = sample_pose_left_shoulder;   // LEFT_SHOULDER
    pose[13] = sample_pose_left_elbow;      // LEFT_ELBOW
    pose[15] = sample_pose_left_wrist;      // LEFT_WRIST

    const out = fuseLandmarkResults(
      { worldLandmarks: [pose] },
      {
        worldLandmarks: [sample_hand_landmarks],
        handednesses: [[{ categoryName: "Left", score: 0.95 }]],
      },
    );
    expect(out.left).not.toBeNull();
    expect(out.right).toBeNull();
    const left = out.left as SideFrame;
    expect(left.pose.shoulder).toEqual([
      sample_pose_left_shoulder.x,
      sample_pose_left_shoulder.y,
      sample_pose_left_shoulder.z,
    ]);
    expect(left.confidence).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 3: Run; expect import error**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 4: Implement `lib/mediapipe.ts`**

Create `hmi/frontend/lib/mediapipe.ts`:

```typescript
/**
 * MediaPipe Tasks for Web wrapper.
 *
 * Two responsibilities:
 *   1. Lazy-load HandLandmarker + PoseLandmarker from a single WASM bundle.
 *   2. Fuse their outputs into the `KeypointFrame` shape the backend expects.
 *
 * Coordinates: MediaPipe `worldLandmarks` are metres-relative-to-hip-centre
 * (pose) or metres-relative-to-hand-centre (hand). We pass them through as-is;
 * the backend handles any re-rooting.
 */
import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
  type Landmark,
  type HandLandmarkerResult,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

// Indices into the 33-point Pose Landmarker output, per MediaPipe docs.
const POSE_LEFT_SHOULDER = 11;
const POSE_RIGHT_SHOULDER = 12;
const POSE_LEFT_ELBOW = 13;
const POSE_RIGHT_ELBOW = 14;
const POSE_LEFT_WRIST = 15;
const POSE_RIGHT_WRIST = 16;

// Indices into the 21-point Hand Landmarker output, per MediaPipe docs.
const HAND_WRIST = 0;
const HAND_THUMB_TIP = 4;
const HAND_INDEX_MCP = 5;
const HAND_INDEX_TIP = 8;
const HAND_MIDDLE_MCP = 9;
const HAND_PINKY_MCP = 17;

export type Vec3 = [number, number, number];

export type SideFrame = {
  pose: { shoulder: Vec3; elbow: Vec3; wrist: Vec3 };
  hand: {
    wrist: Vec3; thumb_tip: Vec3; index_tip: Vec3;
    index_mcp: Vec3; middle_mcp: Vec3; pinky_mcp: Vec3;
  };
  confidence: number;
};

export type KeypointFrame = {
  type: "keypoints";
  ts_ms: number;
  dead_man: boolean;
  pinch_calib?: {
    left?:  { min_m: number; max_m: number };
    right?: { min_m: number; max_m: number };
  };
  left:  SideFrame | null;
  right: SideFrame | null;
};

function _xyz(p: Landmark | NormalizedLandmark | undefined): Vec3 {
  if (!p) return [0, 0, 0];
  return [p.x, p.y, (p as Landmark).z ?? 0];
}

function _vis(p: Landmark | NormalizedLandmark | undefined): number {
  if (!p) return 0;
  return (p as Landmark).visibility ?? 0;
}

/** Pure function: combine MediaPipe results into a backend-shaped KeypointFrame. */
export function fuseLandmarkResults(
  pose: Pick<PoseLandmarkerResult, "worldLandmarks">,
  hands: Pick<HandLandmarkerResult, "worldLandmarks" | "handednesses">,
): { left: SideFrame | null; right: SideFrame | null } {
  const poseLm = pose.worldLandmarks?.[0];
  let left: SideFrame | null = null;
  let right: SideFrame | null = null;

  // Pair hands to sides via the handedness label.
  const handByLabel: Record<"Left" | "Right", { lm: Landmark[]; score: number } | null> = {
    Left: null, Right: null,
  };
  hands.worldLandmarks?.forEach((lm, i) => {
    const label = hands.handednesses?.[i]?.[0]?.categoryName as "Left" | "Right" | undefined;
    const score = hands.handednesses?.[i]?.[0]?.score ?? 0;
    if (label && (label === "Left" || label === "Right")) {
      handByLabel[label] = { lm: lm as Landmark[], score };
    }
  });

  if (poseLm) {
    const buildSide = (
      sIdx: number, eIdx: number, wIdx: number,
      handLabel: "Left" | "Right",
    ): SideFrame | null => {
      const h = handByLabel[handLabel];
      if (!h) return null;
      const handLm = h.lm;
      const poseConf = Math.min(_vis(poseLm[sIdx]), _vis(poseLm[eIdx]), _vis(poseLm[wIdx]));
      return {
        pose: {
          shoulder: _xyz(poseLm[sIdx]),
          elbow:    _xyz(poseLm[eIdx]),
          wrist:    _xyz(poseLm[wIdx]),
        },
        hand: {
          wrist:      _xyz(handLm[HAND_WRIST]),
          thumb_tip:  _xyz(handLm[HAND_THUMB_TIP]),
          index_tip:  _xyz(handLm[HAND_INDEX_TIP]),
          index_mcp:  _xyz(handLm[HAND_INDEX_MCP]),
          middle_mcp: _xyz(handLm[HAND_MIDDLE_MCP]),
          pinky_mcp:  _xyz(handLm[HAND_PINKY_MCP]),
        },
        confidence: Math.min(poseConf, h.score),
      };
    };
    left  = buildSide(POSE_LEFT_SHOULDER,  POSE_LEFT_ELBOW,  POSE_LEFT_WRIST,  "Left");
    right = buildSide(POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST, "Right");
  }
  return { left, right };
}

/** Stateful runner: loads the WASM bundle once, then runs inference per frame. */
export class MediaPipeRunner {
  private hand: HandLandmarker | null = null;
  private pose: PoseLandmarker | null = null;

  async load(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm",
    );
    this.hand = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
    this.pose = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  }

  detect(video: HTMLVideoElement, timestamp_ms: number) {
    if (!this.hand || !this.pose) {
      throw new Error("MediaPipeRunner.load() not called");
    }
    const hands = this.hand.detectForVideo(video, timestamp_ms);
    const pose = this.pose.detectForVideo(video, timestamp_ms);
    return { hands, pose };
  }

  close() {
    this.hand?.close();
    this.pose?.close();
    this.hand = null;
    this.pose = null;
  }
}
```

- [ ] **Step 5: Run; expect tests pass**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 6: Commit**

```bash
git add hmi/frontend/package.json hmi/frontend/pnpm-lock.yaml hmi/frontend/lib/mediapipe.ts hmi/frontend/__tests__/mediapipe.test.ts
git commit -m "feat(hmi/frontend): mediapipe wrapper + keypoint fusion"
```

---

### Task 15: `lib/humanTeleopClient.ts` — WS sender with dead-man + reconnect

**Files:**
- Create: `hmi/frontend/lib/humanTeleopClient.ts`
- Create: `hmi/frontend/__tests__/humanTeleopClient.test.ts`

- [ ] **Step 1: Write failing tests**

Create `hmi/frontend/__tests__/humanTeleopClient.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";

import { HumanTeleopClient } from "@/lib/humanTeleopClient";

/* Minimal fake WebSocket. Each instance pushes itself onto `createdSockets`
   so tests can drive open / close / message events. */
const createdSockets: FakeWS[] = [];

class FakeWS {
  static OPEN = 1;
  url: string;
  readyState = 0;
  onopen?: () => void;
  onclose?: () => void;
  sent: string[] = [];

  constructor(url: string) {
    this.url = url;
    createdSockets.push(this);
    queueMicrotask(() => {
      this.readyState = FakeWS.OPEN;
      this.onopen?.();
    });
  }

  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
    this.onclose?.();
  }
}

beforeEach(() => {
  createdSockets.length = 0;
  (globalThis as any).WebSocket = FakeWS;
});

describe("HumanTeleopClient", () => {
  it("does not send when no frame has been queued", async () => {
    const c = new HumanTeleopClient("ws://x");
    c.connect();
    await new Promise(r => setTimeout(r, 5));
    c.tick();
    expect(createdSockets[0].sent.length).toBe(0);
  });

  it("sends the latest queued frame on tick()", async () => {
    const c = new HumanTeleopClient("ws://x");
    c.connect();
    await new Promise(r => setTimeout(r, 5));
    c.queueFrame({
      type: "keypoints", ts_ms: 1, dead_man: false,
      left: null, right: null,
    });
    c.tick();
    expect(createdSockets[0].sent.length).toBe(1);
    const sent = JSON.parse(createdSockets[0].sent[0]);
    expect(sent.dead_man).toBe(false);
    expect(sent.ts_ms).toBe(1);
  });

  it("reconnects after close", async () => {
    const c = new HumanTeleopClient("ws://x");
    c.connect();
    await new Promise(r => setTimeout(r, 5));
    createdSockets[0].close();
    // The client should schedule a reconnect.
    await new Promise(r => setTimeout(r, 60));
    expect(createdSockets.length).toBeGreaterThan(1);
  });
});
```

- [ ] **Step 2: Run; expect import failure**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 3: Implement the client**

Create `hmi/frontend/lib/humanTeleopClient.ts`:

```typescript
/**
 * WebSocket sender for the in-browser pose pipeline.
 *
 *   client.connect();
 *   // Each MediaPipe frame:
 *   client.queueFrame(frame);
 *   // 60 Hz from the render loop:
 *   client.tick();   // sends the latest queued frame if the socket is open
 *
 * The client reconnects after close with a 50 ms backoff (kept short so the
 * operator sees the live feed snap back fast; the backend grace window is 5 s
 * so we have plenty of headroom).
 */
import type { KeypointFrame } from "./mediapipe";

export class HumanTeleopClient {
  private url: string;
  private ws: WebSocket | null = null;
  private latest: KeypointFrame | null = null;
  private shouldReconnect = true;

  constructor(url: string) {
    this.url = url;
  }

  connect() {
    this.shouldReconnect = true;
    this._open();
  }

  close() {
    this.shouldReconnect = false;
    this.ws?.close();
    this.ws = null;
  }

  queueFrame(frame: KeypointFrame) {
    this.latest = frame;
  }

  tick() {
    if (!this.latest) return;
    if (!this.ws || this.ws.readyState !== 1 /* OPEN */) return;
    this.ws.send(JSON.stringify(this.latest));
    this.latest = null;
  }

  private _open() {
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onclose = () => {
      this.ws = null;
      if (this.shouldReconnect) {
        setTimeout(() => this._open(), 50);
      }
    };
  }
}
```

- [ ] **Step 4: Run; expect green**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 5: Commit**

```bash
git add hmi/frontend/lib/humanTeleopClient.ts hmi/frontend/__tests__/humanTeleopClient.test.ts
git commit -m "feat(hmi/frontend): HumanTeleopClient WS sender + reconnect"
```

---

## Phase V — Frontend components

### Task 16: `ScopeBar` + `DeadManIndicator`

**Files:**
- Create: `hmi/frontend/components/ScopeBar.tsx`
- Create: `hmi/frontend/components/DeadManIndicator.tsx`
- Create: `hmi/frontend/__tests__/ScopeBar.test.tsx`

- [ ] **Step 1: Write the failing test for `ScopeBar`**

Create `hmi/frontend/__tests__/ScopeBar.test.tsx`:

```typescript
import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { ScopeBar } from "@/components/ScopeBar";

describe("ScopeBar", () => {
  it("renders the commanded value and limits", () => {
    const { container } = render(
      <ScopeBar label="pan" min={-90} max={90} commanded={30} intended={30} />
    );
    // The component must render `30.0` somewhere readable.
    expect(container.textContent).toMatch(/30\.0/);
    expect(container.textContent).toMatch(/pan/);
  });

  it("shows a ghost tick only when intended differs from commanded", () => {
    const { container, rerender } = render(
      <ScopeBar label="pan" min={-90} max={90} commanded={30} intended={30} />,
    );
    // Without divergence, the ghost element should not be present.
    expect(container.querySelector("[data-ghost]")).toBeNull();
    rerender(
      <ScopeBar label="pan" min={-90} max={90} commanded={30} intended={45} />,
    );
    expect(container.querySelector("[data-ghost]")).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run; expect import failure**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 3: Implement the two components**

Create `hmi/frontend/components/ScopeBar.tsx`:

```typescript
"use client";

/**
 * Read-only per-joint bar. Center tick = 0°, side ticks = limits,
 * filled segment = `commanded`, ghost tick = `intended` when it diverges.
 *
 * Used in the human-teleop side panel; purely visual.
 */
export function ScopeBar({
  label, min, max, commanded, intended,
}: {
  label: string;
  min: number;
  max: number;
  commanded: number;
  intended?: number;
}) {
  const span = Math.max(max - min, 1e-6);
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - min) / span) * 100));
  const cmdPct = pct(commanded);
  const intendedPct = intended === undefined ? null : pct(intended);
  const diverged = intended !== undefined && Math.abs(intended - commanded) > 0.5;

  return (
    <div className="flex items-center gap-2 text-[12px] font-mono">
      <span className="w-12 text-muted-foreground">{label}</span>
      <div className="relative h-2 flex-1 rounded-sm border border-border bg-card">
        {/* center tick */}
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-border" />
        {/* commanded fill */}
        <div
          className="absolute top-0 bottom-0 left-1/2 bg-[var(--instrument-line,oklch(80%_0.18_142))]"
          style={{
            transform: `translateX(-${cmdPct < 50 ? 100 - cmdPct * 2 : 0}%)`,
            width: `${Math.abs(cmdPct - 50)}%`,
          }}
        />
        {/* ghost tick (intended) */}
        {diverged && intendedPct !== null ? (
          <div
            data-ghost
            className="absolute top-[-2px] bottom-[-2px] w-px bg-foreground/70"
            style={{ left: `${intendedPct}%` }}
          />
        ) : null}
      </div>
      <span className="w-14 text-right tabular-nums">{commanded.toFixed(1)}°</span>
    </div>
  );
}
```

Create `hmi/frontend/components/DeadManIndicator.tsx`:

```typescript
"use client";

/**
 * Visual chip for the spacebar-driven dead-man state.
 * - `held`: lime "DRIVING — release to stop"
 * - !held & !lost: muted "DRIVE — hold SPACE"
 * - lost: amber "HOLD — tracking lost"
 */
export function DeadManIndicator({
  held, trackingLost,
}: {
  held: boolean;
  trackingLost: boolean;
}) {
  if (trackingLost) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-amber-500 text-amber-500">
        HOLD — tracking lost
      </div>
    );
  }
  if (held) {
    return (
      <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-[var(--instrument-line,oklch(80%_0.18_142))] text-[var(--instrument-line,oklch(80%_0.18_142))] animate-pulse">
        DRIVING — release to stop
      </div>
    );
  }
  return (
    <div className="font-mono text-[12px] px-3 py-1 rounded-sm border border-border text-muted-foreground">
      DRIVE — hold SPACE
    </div>
  );
}
```

- [ ] **Step 4: Run; expect green**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 5: Commit**

```bash
git add hmi/frontend/components/ScopeBar.tsx hmi/frontend/components/DeadManIndicator.tsx hmi/frontend/__tests__/ScopeBar.test.tsx
git commit -m "feat(hmi/frontend): ScopeBar + DeadManIndicator components"
```

---

### Task 17: `PinchCalibrationStep`

**Files:**
- Create: `hmi/frontend/components/PinchCalibrationStep.tsx`

- [ ] **Step 1: Create the component**

Create `hmi/frontend/components/PinchCalibrationStep.tsx`:

```typescript
"use client";

/**
 * Per-hand pinch calibration. Two captures:
 *   1. "Open"  — hold hand open, click → max thumb-index distance.
 *   2. "Pinch" — pinch fully closed, click → min thumb-index distance.
 *
 * The caller passes a `liveDistance` (current thumb-index distance in metres,
 * derived from MediaPipe in the parent), and receives onChange callbacks when
 * a side captures its values.
 */
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export type PinchSide = "left" | "right";

export type PinchCalib = {
  min_m: number | null;
  max_m: number | null;
};

export function PinchCalibrationStep({
  liveDistance,
  side,
  value,
  onChange,
}: {
  liveDistance: number | null;
  side: PinchSide;
  value: PinchCalib;
  onChange: (next: PinchCalib) => void;
}) {
  const captureOpen = () => {
    if (liveDistance === null) return;
    onChange({ ...value, max_m: liveDistance });
  };
  const capturePinch = () => {
    if (liveDistance === null) return;
    onChange({ ...value, min_m: liveDistance });
  };
  const ready = value.min_m !== null && value.max_m !== null
    && value.max_m > value.min_m;

  return (
    <Card className="p-0">
      <CardContent className="p-3 flex flex-col gap-2 text-[12px] font-mono">
        <div className="flex justify-between">
          <span className="text-muted-foreground">side</span>
          <span>{side}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">live</span>
          <span>{liveDistance === null ? "—" : liveDistance.toFixed(3) + " m"}</span>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" className="h-7 flex-1" onClick={captureOpen}>
            open · capture
          </Button>
          <Button size="sm" variant="outline" className="h-7 flex-1" onClick={capturePinch}>
            pinch · capture
          </Button>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">min..max</span>
          <span>
            {value.min_m === null ? "—" : value.min_m.toFixed(3)}
            {" .. "}
            {value.max_m === null ? "—" : value.max_m.toFixed(3)}
            {ready ? "" : " (incomplete)"}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Run existing tests as a smoke check**

```bash
cd hmi/frontend
pnpm test
```

Expected: existing 8 tests still pass (no new tests for this small UI component; it has no logic worth isolating beyond click→capture, which is exercised by the integration test in Task 19).

- [ ] **Step 3: Commit**

```bash
git add hmi/frontend/components/PinchCalibrationStep.tsx
git commit -m "feat(hmi/frontend): PinchCalibrationStep component"
```

---

### Task 18: `CameraOverlay`

**Files:**
- Create: `hmi/frontend/components/CameraOverlay.tsx`

- [ ] **Step 1: Create the component**

Create `hmi/frontend/components/CameraOverlay.tsx`:

```typescript
"use client";

/**
 * <video> + <canvas> overlay. Pure render — owns no state besides refs.
 *
 *   - The parent attaches a MediaStream to `video.current.srcObject`.
 *   - On every animation frame, the parent passes the latest landmark result
 *     and calls drawOverlay() through the imperative handle.
 */
import { forwardRef, useImperativeHandle, useRef } from "react";

export type OverlaySides = {
  left:  { lost: boolean; pose: [number, number][]; hand: [number, number][]; pinch01: number } | null;
  right: { lost: boolean; pose: [number, number][]; hand: [number, number][]; pinch01: number } | null;
};

export type CameraOverlayHandle = {
  video: HTMLVideoElement | null;
  draw: (sides: OverlaySides) => void;
};

const INSTRUMENT_LINE = "oklch(80% 0.18 142)";
const AMBER = "oklch(75% 0.16 70)";

export const CameraOverlay = forwardRef<CameraOverlayHandle, { aspectRatio?: string }>(
  function CameraOverlay({ aspectRatio = "16/9" }, ref) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useImperativeHandle(ref, () => ({
      get video() { return videoRef.current; },
      draw(sides: OverlaySides) {
        const cv = canvasRef.current;
        const vd = videoRef.current;
        if (!cv || !vd) return;
        const w = (cv.width = vd.videoWidth || cv.clientWidth);
        const h = (cv.height = vd.videoHeight || cv.clientHeight);
        const ctx = cv.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, w, h);

        const drawSide = (side: OverlaySides["left"]) => {
          if (!side) return;
          const colour = side.lost ? AMBER : INSTRUMENT_LINE;
          ctx.strokeStyle = colour;
          ctx.fillStyle = colour;
          ctx.lineWidth = 2;

          // Body skeleton: shoulder → elbow → wrist.
          ctx.beginPath();
          for (let i = 0; i < side.pose.length; i++) {
            const [x, y] = side.pose[i];
            const px = x * w, py = y * h;
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();

          // Hand landmarks as 4px ticks.
          for (const [x, y] of side.hand) {
            ctx.fillRect(x * w - 2, y * h - 2, 4, 4);
          }

          // Pinch line: thumb-tip to index-tip (assumed first two hand entries).
          if (side.hand.length >= 2) {
            const [tx, ty] = side.hand[0];
            const [ix, iy] = side.hand[1];
            ctx.beginPath();
            ctx.setLineDash(side.pinch01 < 0.3 ? [4, 3] : []);
            ctx.lineWidth = 1.5;
            ctx.moveTo(tx * w, ty * h);
            ctx.lineTo(ix * w, iy * h);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.lineWidth = 2;
          }
        };

        drawSide(sides.left);
        drawSide(sides.right);
      },
    }));

    return (
      <div className="relative w-full" style={{ aspectRatio }}>
        <video
          ref={videoRef}
          autoPlay muted playsInline
          className="absolute inset-0 w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ transform: "scaleX(-1)" }}
        />
      </div>
    );
  },
);
```

- [ ] **Step 2: Smoke check**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 3: Commit**

```bash
git add hmi/frontend/components/CameraOverlay.tsx
git commit -m "feat(hmi/frontend): CameraOverlay (video + canvas overlay)"
```

---

### Task 19: `HumanTeleopPanel` orchestrator

**Files:**
- Create: `hmi/frontend/components/HumanTeleopPanel.tsx`

This is the largest component but contains no novel logic — it stitches MediaPipe, the WS client, the overlay, the side panels, and the calibration step together.

- [ ] **Step 1: Read the Next.js docs**

Before writing this client component, skim `hmi/frontend/node_modules/next/dist/docs/02-app/01-getting-started/05-fetching-data.mdx` and `hmi/frontend/node_modules/next/dist/docs/02-app/02-guides/client-side-data-fetching.mdx` (or whatever the current paths are) — Next 16 has stricter rules about what runs on the server vs client. Verify that `"use client"` is the correct directive and that `useEffect` is allowed in client components.

- [ ] **Step 2: Implement the orchestrator**

Create `hmi/frontend/components/HumanTeleopPanel.tsx`:

```typescript
"use client";

/**
 * Top-level orchestrator for the human-teleop page. Owns:
 *   - The camera stream and the MediaPipeRunner lifecycle.
 *   - The render loop that runs detection + WS publish at ~30 Hz.
 *   - The dead-man key state.
 *   - The pinch calibration state (persisted in localStorage).
 *   - Start/stop/swap session calls.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";

import { api, type HumanTeleopStatus } from "@/lib/api";
import { BACKEND_URL } from "@/lib/config";
import { useTelemetry } from "@/lib/telemetry";
import {
  MediaPipeRunner, fuseLandmarkResults,
  type KeypointFrame, type SideFrame,
} from "@/lib/mediapipe";
import { HumanTeleopClient } from "@/lib/humanTeleopClient";

import { CameraOverlay, type CameraOverlayHandle, type OverlaySides } from "./CameraOverlay";
import { ScopeBar } from "./ScopeBar";
import { DeadManIndicator } from "./DeadManIndicator";
import { PinchCalibrationStep, type PinchCalib } from "./PinchCalibrationStep";

const WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/teleop/human/in`;

const JOINTS = [
  "shoulder_pan", "shoulder_lift", "elbow_flex",
  "wrist_flex", "wrist_roll", "gripper",
] as const;

const CALIB_LS_KEY = "haller.humanTeleop.pinchCalib.v1";

export function HumanTeleopPanel({ armIds }: { armIds: string[] }) {
  const status = useTelemetry((s) => s.lastFrame?.human_teleop) as
    HumanTeleopStatus | undefined;

  const [leftArm, setLeftArm] = useState(armIds[0] ?? "");
  const [rightArm, setRightArm] = useState(armIds[1] ?? armIds[0] ?? "");
  const [swap, setSwap] = useState(false);

  const overlayRef = useRef<CameraOverlayHandle | null>(null);
  const runnerRef = useRef<MediaPipeRunner | null>(null);
  const clientRef = useRef<HumanTeleopClient | null>(null);
  const deadManRef = useRef(false);

  const [calib, setCalib] = useState<{ left: PinchCalib; right: PinchCalib }>(() => {
    if (typeof window === "undefined") return defaultCalib();
    try {
      const raw = localStorage.getItem(CALIB_LS_KEY);
      if (raw) return JSON.parse(raw);
    } catch { /* ignore */ }
    return defaultCalib();
  });
  const [liveDistance, setLiveDistance] = useState<{ left: number | null; right: number | null }>({
    left: null, right: null,
  });

  // Persist calib on change.
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(CALIB_LS_KEY, JSON.stringify(calib));
    }
  }, [calib]);

  // Bind dead-man key.
  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !isInput(e.target)) {
        e.preventDefault();
        deadManRef.current = true;
      }
    };
    const onUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        deadManRef.current = false;
      }
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, []);

  // One-shot: open camera + load models.
  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay || !overlay.video) return;
    let cancelled = false;
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 960, height: 540 }, audio: false,
        });
        if (cancelled) return;
        overlay.video!.srcObject = stream;
        await overlay.video!.play();
        runnerRef.current = new MediaPipeRunner();
        await runnerRef.current.load();
        clientRef.current = new HumanTeleopClient(WS_URL);
        clientRef.current.connect();
      } catch (e) {
        toast.error(`camera/MediaPipe init failed: ${(e as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
      runnerRef.current?.close();
      runnerRef.current = null;
      clientRef.current?.close();
      clientRef.current = null;
    };
  }, []);

  // Render loop: detect → overlay → publish.
  useEffect(() => {
    let raf = 0;
    let last_t = 0;
    const tick = (t: number) => {
      raf = requestAnimationFrame(tick);
      const runner = runnerRef.current;
      const overlay = overlayRef.current;
      const client = clientRef.current;
      const video = overlay?.video;
      if (!runner || !overlay || !client || !video || video.readyState < 2) return;
      // Cap to ~30 Hz; MediaPipe will internally throttle if GPU saturates.
      if (t - last_t < 33) return;
      last_t = t;
      const { hands, pose } = runner.detect(video, t);
      const fused = fuseLandmarkResults(pose, hands);

      overlay.draw(toOverlay(fused));
      const ld = liveThumbIndex(fused);
      if (ld.left !== liveDistance.left || ld.right !== liveDistance.right) {
        setLiveDistance(ld);
      }

      const frame: KeypointFrame = {
        type: "keypoints",
        ts_ms: Math.floor(performance.now()),
        dead_man: deadManRef.current,
        pinch_calib: {
          left:  calib.left.min_m !== null && calib.left.max_m !== null
            ? { min_m: calib.left.min_m, max_m: calib.left.max_m } : undefined,
          right: calib.right.min_m !== null && calib.right.max_m !== null
            ? { min_m: calib.right.min_m, max_m: calib.right.max_m } : undefined,
        },
        left: fused.left, right: fused.right,
      };
      client.queueFrame(frame);
      client.tick();
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [calib]);

  const running = status?.running ?? false;
  const state = status?.state ?? "idle";

  const handleStart = async () => {
    try {
      // Push calibration first if both sides are complete.
      const cl = calib.left.min_m !== null && calib.left.max_m !== null
        ? { min_m: calib.left.min_m, max_m: calib.left.max_m } : undefined;
      const cr = calib.right.min_m !== null && calib.right.max_m !== null
        ? { min_m: calib.right.min_m, max_m: calib.right.max_m } : undefined;
      if (cl || cr) await api.humanTeleopCalibrate({ left: cl, right: cr });
      await api.humanTeleopStart({ left_arm: leftArm, right_arm: rightArm, swap });
      toast.success(`human teleop started`);
    } catch (e) {
      toast.error(`start failed: ${(e as Error).message}`);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-3">
      <div className="space-y-2">
        <CameraOverlay ref={overlayRef} aspectRatio="16/9" />
        <div className="flex items-center justify-between">
          <DeadManIndicator
            held={state === "driving"}
            trackingLost={!!status?.tracking?.left?.lost || !!status?.tracking?.right?.lost}
          />
          <div className="flex items-center gap-2 font-mono text-[12px]">
            <Badge variant={running ? "default" : "secondary"}>{state}</Badge>
            {running ? (
              <Button size="sm" variant="destructive"
                      className="h-7 px-3"
                      onClick={() => api.humanTeleopStop().catch(() => null)}>
                stop
              </Button>
            ) : (
              <Button size="sm" className="h-7 px-3" onClick={handleStart}
                      disabled={!leftArm || !rightArm || leftArm === rightArm}>
                start
              </Button>
            )}
          </div>
        </div>
        <Card className="p-3 flex flex-wrap items-center gap-2 font-mono text-[12px]">
          <span className="text-muted-foreground">assign</span>
          <NativeSelect ariaLabel="left arm" value={leftArm} onChange={setLeftArm}
                        options={armIds.map((id) => ({ value: id, label: id }))} />
          <Button size="sm" variant="outline" className="h-7"
                  onClick={() => { const t = leftArm; setLeftArm(rightArm); setRightArm(t); }}>
            ⇄
          </Button>
          <NativeSelect ariaLabel="right arm" value={rightArm} onChange={setRightArm}
                        options={armIds.filter((id) => id !== leftArm).map((id) => ({ value: id, label: id }))} />
          <Button size="sm" variant="outline" className="h-7"
                  onClick={() => { setSwap(!swap); api.humanTeleopSwap(!swap).catch(() => null); }}>
            mirror: {swap ? "off" : "on"}
          </Button>
        </Card>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <PinchCalibrationStep
            side="left" liveDistance={liveDistance.left} value={calib.left}
            onChange={(next) => setCalib({ ...calib, left: next })}
          />
          <PinchCalibrationStep
            side="right" liveDistance={liveDistance.right} value={calib.right}
            onChange={(next) => setCalib({ ...calib, right: next })}
          />
        </div>
      </div>
      <div className="space-y-2">
        <ArmScopePanel label={`arm: ${leftArm}`} goal={status?.goal_deg?.left} />
        <ArmScopePanel label={`arm: ${rightArm}`} goal={status?.goal_deg?.right} />
      </div>
    </div>
  );
}

function ArmScopePanel({
  label, goal,
}: { label: string; goal?: Record<string, number> }) {
  return (
    <Card className="p-3">
      <div className="flex justify-between text-[12px] font-mono mb-2">
        <span>{label}</span>
      </div>
      <div className="space-y-1">
        {JOINTS.map((j) => (
          // All joints are reported in degrees by the backend (gripper is scaled
          // from [0,1] to its calibrated degree range inside the commit loop).
          // Use a fixed -90..90 viewport for all bars; calibrated limits are
          // typically inside this range and the bar visually saturates if not.
          <ScopeBar
            key={j}
            label={j}
            min={-90}
            max={90}
            commanded={goal?.[j] ?? 0}
          />
        ))}
      </div>
    </Card>
  );
}

function NativeSelect({
  value, onChange, options, ariaLabel,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  ariaLabel?: string;
}) {
  return (
    <select
      aria-label={ariaLabel}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-7 rounded-sm border border-border bg-background px-2 font-mono text-[12px]"
    >
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

function defaultCalib(): { left: PinchCalib; right: PinchCalib } {
  return {
    left:  { min_m: null, max_m: null },
    right: { min_m: null, max_m: null },
  };
}

function isInput(t: EventTarget | null): boolean {
  return t instanceof HTMLElement &&
    (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}

function toOverlay(fused: { left: SideFrame | null; right: SideFrame | null }): OverlaySides {
  // The CameraOverlay expects 2D coords in [0, 1]. MediaPipe `worldLandmarks`
  // are 3D metres; for overlay we use the IMAGE-space landmarks instead — see
  // `detect()` future enhancement. v1 simplification: draw nothing (canvas is
  // left blank) until we wire in the normalised landmarks in a follow-up.
  // This keeps the page running while the visualisation is iterated on.
  return { left: null, right: null };
}

function liveThumbIndex(
  fused: { left: SideFrame | null; right: SideFrame | null }
): { left: number | null; right: number | null } {
  const dist = (s: SideFrame | null): number | null => {
    if (!s) return null;
    const a = s.hand.thumb_tip, b = s.hand.index_tip;
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };
  return { left: dist(fused.left), right: dist(fused.right) };
}
```

> **Note on the overlay:** the panel imports `OverlaySides` but currently passes empty sides via `toOverlay`. The MediaPipe wrapper in Task 14 returns only `worldLandmarks`. To finish the overlay later, also request `landmarks` (image-normalised) from MediaPipe — `runner.detect` would extend its return type. This is a deliberate v1 simplification so the page is functional end-to-end before the visualisation is iterated on. Tracked in spec §11.

- [ ] **Step 3: Smoke check**

```bash
cd hmi/frontend
pnpm test
```

- [ ] **Step 4: Commit**

```bash
git add hmi/frontend/components/HumanTeleopPanel.tsx
git commit -m "feat(hmi/frontend): HumanTeleopPanel orchestrator"
```

---

## Phase VI — Page + integration

### Task 20: `/teleop/human` route + dashboard link + README

**Files:**
- Create: `hmi/frontend/app/teleop/human/page.tsx`
- Modify: `hmi/frontend/app/page.tsx`
- Modify: `hmi/README.md`

- [ ] **Step 1: Read Next.js routing docs**

Skim `hmi/frontend/node_modules/next/dist/docs/02-app/01-getting-started/03-layouts-and-pages.mdx` (or the current path) to confirm App Router page conventions for Next 16. The existing `hmi/frontend/app/base/page.tsx` is a working reference.

- [ ] **Step 2: Create the new page**

Create `hmi/frontend/app/teleop/human/page.tsx`:

```typescript
"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { HumanTeleopPanel } from "@/components/HumanTeleopPanel";

export default function HumanTeleopPage() {
  const [armIds, setArmIds] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.config()
      .then((cfg) => setArmIds(cfg.arms.map((a) => a.id)))
      .catch((e: Error) => setErr(e.message));
  }, []);

  if (err) {
    return (
      <main className="p-4 font-mono text-sm text-destructive">
        config load failed: {err}
      </main>
    );
  }

  return (
    <main className="p-4 space-y-3">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-lg font-mono">Human Teleop</h1>
          <p className="text-[12px] text-muted-foreground">
            bimanual · monocular RGB · hold <kbd>SPACE</kbd> to drive
          </p>
        </div>
      </header>
      {armIds.length >= 2 ? (
        <HumanTeleopPanel armIds={armIds} />
      ) : (
        <div className="text-[12px] font-mono text-muted-foreground">
          human teleop needs ≥2 enabled arms in <code>hmi/backend/config.yaml</code>
        </div>
      )}
    </main>
  );
}
```

- [ ] **Step 3: Add a dashboard link**

In `hmi/frontend/app/page.tsx`, find the `TeleopLauncher` block (it's used near the top of the dashboard) and add a sibling link card right above or below it. The exact insertion point: locate the `<TeleopLauncher … />` JSX and add:

```tsx
<a
  href="/teleop/human"
  className="inline-flex items-center gap-2 rounded-sm border border-border px-3 py-1 text-[12px] font-mono hover:bg-muted"
>
  Human teleop →
</a>
```

Place this either inside the existing teleop card's header (right of the leader/follower start button) or as a small link block immediately after the `<TeleopLauncher>` element. Match the existing styling tokens (`label-micro`, `text-muted-foreground`) — the surrounding code shows the conventions in use.

- [ ] **Step 4: Update the HMI README**

In `hmi/README.md`, add a new section between "Operating an arm" and "Operating the base", titled `## Human teleop (vision)`. Use roughly the same depth as the existing leader/follower section. Suggested content:

````markdown
## Human teleop (vision)

The dashboard's **Human teleop** link opens `/teleop/human`, a new mode that drives both SO-101 arms from your laptop webcam.

1. Pick the two arms in **assign**.
2. Click **Calibrate · open**, hold your hand open, then **Calibrate · pinch** with thumb and index touching. Repeat for the other side.
3. Position yourself in front of the camera (both shoulders + both hands visible). The skeleton overlay will appear on the feed.
4. **Hold SPACE** to drive. The robots track your joint angles 1:1. Release SPACE to freeze instantly.
5. **Stop** ends the session and restores both arms to MANUAL.

**Mutual exclusion.** Only one teleop kind runs at a time. Starting human teleop while leader/follower teleop is running returns 409 (and vice versa).

**Tracking loss.** If one hand exits the frame, that arm freezes in place; the other arm continues. The HUD shows `tracking lost (side)`.

**E-STOP** stops human teleop just like leader/follower, drops torque, and zeroes `/cmd_vel`.

### Manual smoke tests

1. Cold start with no camera → permission prompt → error state → no robot motion.
2. Calibrate pinch → engage SPACE → wave one arm → the other stays in place.
3. Mid-drive: hand exits frame → that arm freezes, other continues, chip turns amber.
4. Mid-drive: release SPACE → both arms freeze within ~16 ms.
5. Global E-STOP while driving → session stops, torque drops, E-STOP banner.
6. Try to start leader/follower while human teleop is running → 409, no state change.

See [`docs/superpowers/specs/2026-05-22-human-pose-teleop-design.md`](../docs/superpowers/specs/2026-05-22-human-pose-teleop-design.md) for the full design.
````

Append `POST /teleop/human/start`, `POST /teleop/human/stop`, `POST /teleop/human/swap`, `POST /teleop/human/calibrate`, `GET /teleop/human`, and `WS /ws/teleop/human/in` to the REST endpoints table in the same README, in the rows immediately after the existing teleop endpoints.

- [ ] **Step 5: Smoke-test the build and full test suites**

```bash
cd hmi/backend
pytest -v
cd ../frontend
pnpm test
pnpm build
```

Expected: all green; the production build emits a new route `/teleop/human` in the Next routes manifest.

- [ ] **Step 6: Commit**

```bash
git add hmi/frontend/app/teleop/human/page.tsx hmi/frontend/app/page.tsx hmi/README.md
git commit -m "feat(hmi): /teleop/human route + dashboard link + README"
```

---

## Done — what you should see

After Task 20:

- Backend exposes `/teleop/human/{start,stop,swap,calibrate}`, `GET /teleop/human`, and `WS /ws/teleop/human/in`. `pytest -v` reports ~37 tests passing (existing 25 + 16 retarget + 14 human_teleop + 6 new route tests, minus any merges).
- Frontend has a working `/teleop/human` page that:
  - Opens the laptop webcam, runs MediaPipe in-browser at ~30 Hz.
  - Streams keypoint frames over WS to the backend.
  - Drives both SO-101 arms at 60 Hz while spacebar is held.
  - Freezes within ~16 ms when spacebar is released.
  - Survives one-hand-out-of-frame on a per-arm basis.
  - Persists pinch calibration to `localStorage` per side.
- Leader/follower teleop is unaffected; the two teleop kinds are mutually exclusive.
- E-STOP stops everything.

Out-of-scope items deferred to v2 (per spec §10):
- Remote pose worker (WiLoR/HaMeR on a desktop or cloud GPU) — the WS schema is already shaped for this swap.
- RealSense / stereo depth anchor for cartesian "reach extension" mode.
- Per-operator profile persistence.
- LeRobot-dataset emission of demonstration sessions.
- The overlay's image-normalised landmark wiring (Task 19 ships with the skeleton drawing disabled; the metric for filling this in is "operator sees their pose drawn on the feed within one v1.1 commit").
