# Human-Teleop Diagnostic Chain + Skeleton Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the human-teleop page legible — draw the MediaPipe skeleton on the video, and expose the landmarks → target → committed → actual chain so a surprising arm motion can be attributed to a stage instead of guessed at.

**Architecture:** `HumanTeleopSession._smooth_step` already computes the rate-cap and joint-limit conditions and discards them; it is changed to return a per-joint record (`target`, `committed`, `reason`) which `status()` publishes under a new `joints` key. That rides the existing 20 Hz telemetry WebSocket, which already embeds `human_teleop.status()` wholesale — no new route. On the frontend, a new pure function `buildOverlaySides` reads MediaPipe's normalized `.landmarks` (where `fuseLandmarkResults` reads `.worldLandmarks`) and feeds the already-written `CameraOverlay`.

**Tech Stack:** Python 3.12 / FastAPI / MuJoCo (backend, `hmi/backend`), pytest. Next.js 16 / React / TypeScript / Tailwind (frontend, `hmi/frontend`), vitest.

**Spec:** [`docs/superpowers/specs/2026-07-28-teleop-diagnostic-chain-design.md`](../specs/2026-07-28-teleop-diagnostic-chain-design.md)

## Global Constraints

- **`goal_deg` must not change shape.** `DatasetRecorder` reads `HumanTeleopSession.status()["goal_deg"]` as the `action` column of every recorded LeRobot episode (`hmi/backend/haller_hmi/recorder.py:224`). All new data is additive under a new `joints` key. A change here silently corrupts recorded datasets and would not surface until training.
- **Backend tests must run in the project venv**, not system Python. System Python loads a ROS `launch_testing` pytest plugin that fails on a missing `lark` module. Always: `MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest ...` run from `hmi/backend`.
- **Frontend commands run from `hmi/frontend`** with `pnpm` (not npm).
- **No `git add -A` or `git add .`** in this repo. Stage explicit paths only, and verify with `git diff --cached --name-only` before committing. Concurrent sessions leave untracked WIP that must not ride along.
- **No `Co-Authored-By:` trailers** in commit messages (repo rule, `CLAUDE.md`).
- **`reason` vocabulary is exactly** `"ok" | "rate_capped" | "clamped" | "held"`. Same four strings in Python and TypeScript.

---

### Task 1: `_smooth_step` reports per-joint reasons

**Files:**
- Modify: `hmi/backend/haller_hmi/human_teleop.py:280-307` (`_smooth_step`), `:319-` (`_loop` call sites)
- Test: `hmi/backend/tests/test_human_teleop.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `JointStep` dataclass with fields `target: float | None`, `committed: float`, `reason: str`. `HumanTeleopSession._smooth_step(committed, target, limits, alpha) -> dict[str, JointStep]` (was `-> dict[str, float]`). Instance attributes `self._steps_left` / `self._steps_right`, both `dict[str, JointStep]`. Static helper `HumanTeleopSession._held_steps(committed: dict[str, float]) -> dict[str, JointStep]`.

**Note on float comparison:** reason detection compares floats with `!=`, deliberately. `max(lo, min(hi, x))` returns `x` bitwise unchanged when no clamping occurs, so exact equality is the correct test. Do not add an epsilon tolerance — it would make a joint resting exactly at its limit report `ok`.

- [ ] **Step 1: Write the failing tests**

Add to `hmi/backend/tests/test_human_teleop.py`:

```python
def test_smooth_step_reports_ok_when_nothing_intervenes():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0)}
    # alpha=1.0 -> the filter passes `desired` straight through; small step so
    # the 4 deg/tick cap does not bite and the value is far from the limits.
    steps = sess._smooth_step({"shoulder_pan": 0.0}, {"shoulder_pan": 2.0}, limits, 1.0)
    assert steps["shoulder_pan"].reason == "ok"
    assert steps["shoulder_pan"].committed == pytest.approx(2.0)
    assert steps["shoulder_pan"].target == pytest.approx(2.0)


def test_smooth_step_reports_rate_capped():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0)}
    # Ask for a 50 deg jump with alpha=1.0; the 4 deg/tick cap must bite.
    steps = sess._smooth_step({"shoulder_pan": 0.0}, {"shoulder_pan": 50.0}, limits, 1.0)
    assert steps["shoulder_pan"].reason == "rate_capped"
    assert steps["shoulder_pan"].committed == pytest.approx(4.0)
    # target is what was ASKED for, not what was delivered.
    assert steps["shoulder_pan"].target == pytest.approx(50.0)


def test_smooth_step_reports_clamped_and_clamped_beats_rate_capped():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 5.0)}
    # Sitting at 4 deg, asked for 50: the cap would allow 8, the limit allows 5.
    # Both conditions fire; `clamped` must win.
    steps = sess._smooth_step({"shoulder_pan": 4.0}, {"shoulder_pan": 50.0}, limits, 1.0)
    assert steps["shoulder_pan"].reason == "clamped"
    assert steps["shoulder_pan"].committed == pytest.approx(5.0)


def test_smooth_step_reports_held_when_side_has_no_target():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0), "elbow_flex": (-90.0, 90.0)}
    steps = sess._smooth_step({"shoulder_pan": 12.0, "elbow_flex": 3.0}, None, limits, 1.0)
    for joint in limits:
        assert steps[joint].reason == "held"
        assert steps[joint].target is None
    assert steps["shoulder_pan"].committed == pytest.approx(12.0)


def test_smooth_step_reports_held_for_a_joint_missing_from_the_target():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0), "elbow_flex": (-90.0, 90.0)}
    steps = sess._smooth_step(
        {"shoulder_pan": 0.0, "elbow_flex": 7.0}, {"shoulder_pan": 2.0}, limits, 1.0,
    )
    assert steps["shoulder_pan"].reason == "ok"
    assert steps["elbow_flex"].reason == "held"
    assert steps["elbow_flex"].target is None
    assert steps["elbow_flex"].committed == pytest.approx(7.0)


def test_smooth_step_reports_gripper_target_in_degrees_not_unit_interval():
    """retarget emits gripper in [0,1]; _smooth_step scales it onto the joint's
    degree range. `target` must be reported post-scaling so it is comparable
    with `committed`, which is always degrees."""
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"gripper": (-30.0, 30.0)}
    # 1.0 == fully open == the joint's max, 30 deg.
    steps = sess._smooth_step({"gripper": 0.0}, {"gripper": 1.0}, limits, 1.0)
    assert steps["gripper"].target == pytest.approx(30.0)
    assert steps["gripper"].target > 1.0, "gripper target leaked as [0,1]"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_human_teleop.py -k smooth_step -v
```

Expected: FAIL — `AttributeError: 'float' object has no attribute 'reason'` (`_smooth_step` currently returns `dict[str, float]`).

- [ ] **Step 3: Add the `JointStep` dataclass**

In `hmi/backend/haller_hmi/human_teleop.py`, below the existing `_SessionConfig` dataclass:

```python
@dataclass
class JointStep:
    """One joint's outcome for one tick of the commit loop.

    `target` is what the retargeter asked for, in degrees (the gripper's
    [0,1] output is already scaled onto its calibrated range). `committed`
    is what was actually written. `reason` explains any difference.
    """
    target: float | None
    committed: float
    reason: str   # "ok" | "rate_capped" | "clamped" | "held"
```

- [ ] **Step 4: Rewrite `_smooth_step`**

Replace `_smooth_step` (currently `human_teleop.py:280-307`) with:

```python
    def _smooth_step(
        self,
        committed: dict[str, float],
        target: dict[str, float] | None,
        limits: dict[str, tuple[float, float]],
        alpha: float,
    ) -> dict[str, JointStep]:
        out: dict[str, JointStep] = {}
        for joint, lo_hi in limits.items():
            lo, hi = lo_hi
            cur = committed.get(joint, 0.0)
            if target is None or joint not in target:
                out[joint] = JointStep(target=None, committed=cur, reason="held")
                continue
            desired = float(target[joint])
            # Special-case gripper: retarget emits [0, 1] (0 = closed, 1 = open).
            # Scale onto the gripper joint's calibrated degree range so that
            # `target` and `committed` are always the same unit.
            if joint == "gripper":
                desired = max(0.0, min(1.0, desired))
                desired = lo + desired * (hi - lo)
            # One-pole LPF, then per-tick rate cap, then hard clamp to limits.
            # Each stage records whether it altered the value. Exact float
            # equality is correct here: these clamps return their input
            # bitwise unchanged when they don't bite.
            lpf = cur + alpha * (desired - cur)
            cap = self._rate_cap_deg_per_tick
            capped = max(cur - cap, min(cur + cap, lpf))
            final = max(lo, min(hi, capped))
            if final != capped:
                reason = "clamped"        # a hard limit outranks a transient cap
            elif capped != lpf:
                reason = "rate_capped"
            else:
                reason = "ok"
            out[joint] = JointStep(target=desired, committed=final, reason=reason)
        return out
```

Behaviour note: the old code did `if target is None: return committed` — returning the dict verbatim. The new version always iterates `limits`, so a joint present in `limits` but missing from `committed` now appears with `0.0` instead of being absent. In practice `_committed_*` is seeded from `joint_limits_deg` so the key sets already match; this only makes the shape unconditional.

- [ ] **Step 5: Add the `_held_steps` helper**

Add as a `@staticmethod` on `HumanTeleopSession`, next to `_observed_or_zero`:

```python
    @staticmethod
    def _held_steps(committed: dict[str, float]) -> dict[str, JointStep]:
        """Seed the diagnostic block before any frame has arrived: everything
        is being held at its seeded position, nothing has been asked for."""
        return {joint: JointStep(target=None, committed=value, reason="held")
                for joint, value in committed.items()}
```

- [ ] **Step 6: Initialise the step dicts in `__init__` and `start`**

In `__init__`, next to `self._committed_left` / `self._committed_right` (currently `human_teleop.py:72-73`), add:

```python
        self._steps_left: dict[str, JointStep] = {}
        self._steps_right: dict[str, JointStep] = {}
```

In `start()`, immediately after the two `_observed_or_zero` seeding lines, add:

```python
            self._steps_left = self._held_steps(self._committed_left)
            self._steps_right = self._held_steps(self._committed_right)
```

- [ ] **Step 7: Update the `_loop` call sites**

In `_loop`, replace the two `self._committed_* = self._smooth_step(...)` assignments with:

```python
                steps_left = self._smooth_step(
                    self._committed_left, target_left, left.joint_limits_deg, alpha,
                )
                steps_right = self._smooth_step(
                    self._committed_right, target_right, right.joint_limits_deg, alpha,
                )
                self._committed_left = {j: s.committed for j, s in steps_left.items()}
                self._committed_right = {j: s.committed for j, s in steps_right.items()}
                # Rebinding a dict is atomic in CPython, so status() always sees
                # a whole tick's worth of steps — never a half-updated dict.
                self._steps_left = steps_left
                self._steps_right = steps_right
```

- [ ] **Step 8: Run the new tests plus the full suite**

```bash
cd hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_human_teleop.py -k smooth_step -v
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/ -q
```

Expected: the six new tests PASS; the full suite reports **155 passed** (149 existing + 6 new).

- [ ] **Step 9: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/backend/haller_hmi/human_teleop.py hmi/backend/tests/test_human_teleop.py
git diff --cached --name-only
git commit -m "feat(hmi): _smooth_step reports per-joint target and reason

The commit loop already computed the rate-cap and joint-limit conditions
and threw them away. It now returns a JointStep per joint carrying the
requested target (gripper scaled to degrees), the committed value, and a
reason: ok | rate_capped | clamped | held. clamped outranks rate_capped
when both fire, because a hard limit is the more fundamental reason the
arm stopped where it did."
```

---

### Task 2: Publish the `joints` block in `status()`

**Files:**
- Modify: `hmi/backend/haller_hmi/human_teleop.py:113-141` (`status`)
- Test: `hmi/backend/tests/test_human_teleop.py`, `hmi/backend/tests/sim/test_human_teleop_sim.py`

**Interfaces:**
- Consumes: `JointStep`, `self._steps_left`, `self._steps_right` from Task 1.
- Produces: `status()["joints"]` — `{"left": {joint: {"target": float | None, "committed": float, "reason": str}}, "right": {...}}`. `status()["goal_deg"]` is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `hmi/backend/tests/test_human_teleop.py`:

```python
def test_status_joints_block_mirrors_goal_deg_keys():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        st = sess.status()
        assert set(st["joints"]["left"]) == set(st["goal_deg"]["left"])
        for entry in st["joints"]["left"].values():
            assert set(entry) == {"target", "committed", "reason"}
    finally:
        sess.stop()


def test_status_joints_are_held_before_any_frame_arrives():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        st = sess.status()
        for side in ("left", "right"):
            for entry in st["joints"][side].values():
                assert entry["reason"] == "held"
                assert entry["target"] is None
    finally:
        sess.stop()


def test_status_joints_revert_to_held_after_stop():
    """After stop() nothing is being asked for, so no joint may still advertise
    a live reason. A retained CLAMPED badge from an ended session would tell the
    operator the arm is at a limit it is no longer being driven into."""
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    sess.ingest_frame(_kp_frame(dead_man=True))
    _time.sleep(0.05)
    sess.stop()

    st = sess.status()
    for side in ("left", "right"):
        for joint, entry in st["joints"][side].items():
            assert entry["reason"] == "held", f"{side}.{joint} kept a live reason after stop"
            assert entry["target"] is None
    # The committed values themselves are still retained, matching goal_deg.
    assert st["joints"]["left"].keys() == st["goal_deg"]["left"].keys()


def test_status_goal_deg_shape_is_unchanged_by_the_joints_block():
    """goal_deg is DatasetRecorder's `action` column. It must stay a plain
    joint -> float mapping."""
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        _time.sleep(0.05)
        goal = sess.status()["goal_deg"]["left"]
        assert goal, "goal_deg must not be empty while driving"
        for value in goal.values():
            assert isinstance(value, float)
    finally:
        sess.stop()
```

Add to `hmi/backend/tests/sim/test_human_teleop_sim.py`:

```python
def test_reason_reports_clamped_against_a_real_sim_joint_limit(sim_arms):
    """Drive a real MuJoCo arm past a real calibrated limit and check the
    reason. Mock-arm tests cannot catch a wrong limit source; this can."""
    mgr, handles, _world = sim_arms
    lo, hi = handles["left"].joint_limits_deg["shoulder_pan"]

    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # The arm must be swung PAST the real limit, which for the vendored
        # SO-101 MJCF is shoulder_pan = +/-110.008 deg (`range="-1.92 1.92"`
        # radians, so_arm100.xml:35). A pose in the shoulder's XY plane
        # retargets to exactly 90 deg and would never clamp — the arm has to
        # swing behind the shoulder. This pose retargets to ~123.7 deg.
        frame = _kp_frame(dead_man=True, elbow=[0.3, 1.4, -0.2], wrist=[0.6, 1.4, -0.4])

        def _clamped() -> bool:
            sess.ingest_frame(frame)
            entry = sess.status()["joints"]["left"].get("shoulder_pan", {})
            return entry.get("reason") == "clamped"

        assert _wait_until(_clamped), (
            f"shoulder_pan never reported clamped; limits were ({lo}, {hi}), "
            f"status={sess.status()['joints']['left'].get('shoulder_pan')}"
        )
        entry = sess.status()["joints"]["left"]["shoulder_pan"]
        assert lo <= entry["committed"] <= hi
        assert entry["target"] is not None
    finally:
        sess.stop()


def test_reason_is_ok_for_a_joint_tracking_freely(sim_arms):
    mgr, _handles, _world = sim_arms
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Neutral pose: nothing should clamp.
        frame = _kp_frame(dead_man=True, elbow=[0.0, 1.4, 0.3], wrist=[0.0, 1.4, 0.6])

        def _settled() -> bool:
            sess.ingest_frame(frame)
            reasons = {j: e["reason"] for j, e in sess.status()["joints"]["left"].items()}
            return reasons.get("shoulder_pan") == "ok"

        assert _wait_until(_settled), (
            f"shoulder_pan never settled to ok; "
            f"status={sess.status()['joints']['left']}"
        )
    finally:
        sess.stop()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_human_teleop.py -k "joints_block or held_before or goal_deg_shape" tests/sim/test_human_teleop_sim.py -k "reason" -v
```

Expected: FAIL with `KeyError: 'joints'`.

- [ ] **Step 3: Add the `joints` key to `status()`**

In `status()` (`human_teleop.py:119-141`), add a `joints` key immediately after the existing `goal_deg` key. Leave `goal_deg` exactly as it is:

```python
                "goal_deg": {
                    "left":  dict(self._committed_left),
                    "right": dict(self._committed_right),
                },
                # Additive diagnostic block. `goal_deg` above is the recorder's
                # `action` column and must keep its plain joint -> float shape.
                "joints": {
                    "left":  self._steps_as_dict(self._steps_left),
                    "right": self._steps_as_dict(self._steps_right),
                },
```

- [ ] **Step 4: Add the serialisation helper**

Add as a `@staticmethod` on `HumanTeleopSession`, next to `_held_steps`:

```python
    @staticmethod
    def _steps_as_dict(steps: dict[str, JointStep]) -> dict[str, dict]:
        return {
            joint: {
                "target": step.target,
                "committed": step.committed,
                "reason": step.reason,
            }
            for joint, step in steps.items()
        }
```

- [ ] **Step 5: Revert the steps to `held` on stop**

Without this, `_steps_*` keep the last tick's live reasons after the session ends, so the UI would show a `CLAMPED` badge for an arm nobody is driving. In `stop()`, inside the final `with self._lock:` block (alongside `self._state = HumanState.IDLE`), add:

```python
            # Nothing is being asked for any more. Keep the committed values —
            # goal_deg retains them too — but no joint may still advertise a
            # live reason.
            self._steps_left = self._held_steps(self._committed_left)
            self._steps_right = self._held_steps(self._committed_right)
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_human_teleop.py tests/sim/test_human_teleop_sim.py -v
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/ -q
```

Expected: all PASS; full suite **161 passed** (155 after Task 1 + 4 unit + 2 sim).

- [ ] **Step 7: Verify the recorder still works**

The recorder reads `goal_deg`; this task must not have disturbed it.

```bash
cd hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_recorder.py -v
```

Expected: PASS, unchanged.

- [ ] **Step 8: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/backend/haller_hmi/human_teleop.py hmi/backend/tests/test_human_teleop.py hmi/backend/tests/sim/test_human_teleop_sim.py
git diff --cached --name-only
git commit -m "feat(hmi): publish per-joint target/committed/reason in status

Additive `joints` block alongside goal_deg, which keeps its exact shape
because DatasetRecorder reads it as the action column. Rides the existing
20 Hz telemetry WS, which already embeds human_teleop.status() wholesale,
so no new route is needed.

The sim tests drive a real MuJoCo arm into a real calibrated limit — the
mock-arm suite cannot catch a wrong limit source."
```

---

### Task 3: `buildOverlaySides` pure function

**Files:**
- Modify: `hmi/frontend/lib/mediapipe.ts`, `hmi/frontend/components/CameraOverlay.tsx:10-15` (import the moved type)
- Test: `hmi/frontend/__tests__/mediapipe.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `OverlaySides` type exported from `lib/mediapipe.ts` (moved out of `components/CameraOverlay.tsx`, shape unchanged). `buildOverlaySides(pose, hands, opts) -> OverlaySides` where `opts` is `{ leftLost: boolean; rightLost: boolean; leftPinch01: number; rightPinch01: number }`.

**Critical ordering contract:** `CameraOverlay` draws the pinch line between `hand[0]` and `hand[1]` (`CameraOverlay.tsx:63-65`) and draws `pose` as a polyline in array order. So `buildOverlaySides` **must** emit `hand` as `[thumb_tip, index_tip, index_mcp, middle_mcp, pinky_mcp, wrist]` and `pose` as `[shoulder, elbow, wrist]`. Getting this order wrong draws a pinch line between the wrong two fingers.

- [ ] **Step 1: Move the `OverlaySides` type into `lib/mediapipe.ts`**

Delete this block from `hmi/frontend/components/CameraOverlay.tsx` (lines 12-15):

```ts
export type OverlaySides = {
  left:  { lost: boolean; pose: [number, number][]; hand: [number, number][]; pinch01: number } | null;
  right: { lost: boolean; pose: [number, number][]; hand: [number, number][]; pinch01: number } | null;
};
```

Add it to `hmi/frontend/lib/mediapipe.ts` after the `KeypointFrame` type, and re-export it from `CameraOverlay` so existing importers keep working:

```ts
// lib/mediapipe.ts
export type OverlaySide = {
  lost: boolean;
  /** Image-normalized [x, y] in [0,1]. Drawn as a polyline: shoulder, elbow, wrist. */
  pose: [number, number][];
  /** Image-normalized [x, y]. ORDER MATTERS: [thumb_tip, index_tip, ...rest].
   *  CameraOverlay draws the pinch line between entries 0 and 1. */
  hand: [number, number][];
  /** Pinch aperture in [0,1]; 0 = closed. Below 0.3 the pinch line is dashed. */
  pinch01: number;
};

export type OverlaySides = { left: OverlaySide | null; right: OverlaySide | null };
```

```ts
// components/CameraOverlay.tsx — replace the deleted block with:
import type { OverlaySides } from "@/lib/mediapipe";
export type { OverlaySides } from "@/lib/mediapipe";
```

- [ ] **Step 2: Write the failing tests**

Add to `hmi/frontend/__tests__/mediapipe.test.ts`:

```ts
import { buildOverlaySides } from "../lib/mediapipe";

const norm_hand = Array.from({ length: 21 }, (_, i) => ({
  x: 0.10 + i * 0.01, y: 0.20 + i * 0.01, z: 0.0,
}));

function norm_pose() {
  const p = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
  p[11] = { x: 0.50, y: 0.40, z: 0, visibility: 0.95 };  // LEFT_SHOULDER
  p[13] = { x: 0.55, y: 0.50, z: 0, visibility: 0.93 };  // LEFT_ELBOW
  p[15] = { x: 0.60, y: 0.60, z: 0, visibility: 0.91 };  // LEFT_WRIST
  return p;
}

const NO_OPTS = { leftLost: false, rightLost: false, leftPinch01: 0.5, rightPinch01: 0.5 };

describe("buildOverlaySides", () => {
  it("returns null sides when nothing is detected", () => {
    const out = buildOverlaySides({ landmarks: [] }, { landmarks: [], handednesses: [] }, NO_OPTS);
    expect(out.left).toBeNull();
    expect(out.right).toBeNull();
  });

  it("returns null for a side with a pose but no matching hand", () => {
    const out = buildOverlaySides(
      { landmarks: [norm_pose()] },
      { landmarks: [], handednesses: [] },
      NO_OPTS,
    );
    expect(out.left).toBeNull();
    expect(out.right).toBeNull();
  });

  it("builds the left side with pose as shoulder,elbow,wrist in order", () => {
    const out = buildOverlaySides(
      { landmarks: [norm_pose()] },
      { landmarks: [norm_hand], handednesses: [[{ categoryName: "Left", score: 0.95 }]] },
      NO_OPTS,
    );
    expect(out.right).toBeNull();
    expect(out.left!.pose).toEqual([[0.50, 0.40], [0.55, 0.50], [0.60, 0.60]]);
  });

  it("emits thumb_tip then index_tip as the first two hand entries", () => {
    const out = buildOverlaySides(
      { landmarks: [norm_pose()] },
      { landmarks: [norm_hand], handednesses: [[{ categoryName: "Left", score: 0.95 }]] },
      NO_OPTS,
    );
    // CameraOverlay draws the pinch line between hand[0] and hand[1].
    expect(out.left!.hand[0]).toEqual([norm_hand[4].x, norm_hand[4].y]);   // THUMB_TIP
    expect(out.left!.hand[1]).toEqual([norm_hand[8].x, norm_hand[8].y]);   // INDEX_TIP
  });

  it("passes the lost flag and pinch01 through per side", () => {
    const out = buildOverlaySides(
      { landmarks: [norm_pose()] },
      { landmarks: [norm_hand], handednesses: [[{ categoryName: "Left", score: 0.95 }]] },
      { leftLost: true, rightLost: false, leftPinch01: 0.12, rightPinch01: 0.9 },
    );
    expect(out.left!.lost).toBe(true);
    expect(out.left!.pinch01).toBeCloseTo(0.12);
  });
});
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd hmi/frontend
pnpm vitest run __tests__/mediapipe.test.ts
```

Expected: FAIL — `buildOverlaySides is not a function`.

- [ ] **Step 4: Implement `buildOverlaySides`**

Add to `hmi/frontend/lib/mediapipe.ts`, after `fuseLandmarkResults`:

```ts
function _xy(p: NormalizedLandmark | undefined): [number, number] {
  if (!p) return [0, 0];
  return [p.x, p.y];
}

/**
 * Build image-space overlay geometry from the SAME MediaPipe result that
 * `fuseLandmarkResults` consumes — but from `.landmarks` (normalized to the
 * image) rather than `.worldLandmarks` (metric).
 *
 * Kept separate from `fuseLandmarkResults` on purpose: that function builds
 * the backend's KeypointFrame, this one builds a view. Same handedness
 * pairing logic, so the overlay and the commanded motion can never disagree
 * about which hand is which.
 */
export function buildOverlaySides(
  pose:  Pick<PoseLandmarkerResult, "landmarks">,
  hands: Pick<HandLandmarkerResult, "landmarks" | "handednesses">,
  opts: {
    leftLost: boolean; rightLost: boolean;
    leftPinch01: number; rightPinch01: number;
  },
): OverlaySides {
  const poseLm = pose.landmarks?.[0];

  const handByLabel: Record<"Left" | "Right", NormalizedLandmark[] | null> = {
    Left: null, Right: null,
  };
  hands.landmarks?.forEach((lm, i) => {
    const label = hands.handednesses?.[i]?.[0]?.categoryName as "Left" | "Right" | undefined;
    if (label === "Left" || label === "Right") handByLabel[label] = lm;
  });

  const buildSide = (
    sIdx: number, eIdx: number, wIdx: number,
    handLabel: "Left" | "Right", lost: boolean, pinch01: number,
  ): OverlaySide | null => {
    const handLm = handByLabel[handLabel];
    if (!poseLm || !handLm) return null;
    return {
      lost,
      pose: [_xy(poseLm[sIdx]), _xy(poseLm[eIdx]), _xy(poseLm[wIdx])],
      // ORDER MATTERS: CameraOverlay draws the pinch line between [0] and [1].
      hand: [
        _xy(handLm[HAND_THUMB_TIP]),
        _xy(handLm[HAND_INDEX_TIP]),
        _xy(handLm[HAND_INDEX_MCP]),
        _xy(handLm[HAND_MIDDLE_MCP]),
        _xy(handLm[HAND_PINKY_MCP]),
        _xy(handLm[HAND_WRIST]),
      ],
      pinch01,
    };
  };

  return {
    left: buildSide(POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW, POSE_LEFT_WRIST,
                    "Left", opts.leftLost, opts.leftPinch01),
    right: buildSide(POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST,
                     "Right", opts.rightLost, opts.rightPinch01),
  };
}
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd hmi/frontend
pnpm vitest run
```

Expected: all PASS (32 existing + 5 new = 37).

- [ ] **Step 6: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/frontend/lib/mediapipe.ts hmi/frontend/components/CameraOverlay.tsx hmi/frontend/__tests__/mediapipe.test.ts
git diff --cached --name-only
git commit -m "feat(hmi): buildOverlaySides — image-space geometry for the overlay

Reads MediaPipe's normalized .landmarks where fuseLandmarkResults reads
.worldLandmarks, reusing the same handedness pairing so the overlay and
the commanded motion cannot disagree about which hand is which.

OverlaySides moves from components/CameraOverlay.tsx to lib/mediapipe.ts;
it is a data shape, and leaving it in the component would make lib import
from components."
```

---

### Task 4: Wire the overlay into the render loop

**Files:**
- Modify: `hmi/frontend/components/HumanTeleopPanel.tsx:132-172` (render loop), `:308-314` (delete `toOverlay`), `:41-48` (drop the stale cast)

**Interfaces:**
- Consumes: `buildOverlaySides`, `OverlaySides` from Task 3.
- Produces: a visibly drawn skeleton. No new exports.

**Note:** `HumanTeleopPanel.tsx:41-48` casts telemetry through `as unknown as` with a comment saying `TelemetryFrame` doesn't type `human_teleop`. **That comment is stale** — `lib/telemetry.ts:52` already declares `human_teleop?: HumanTeleopStatus`. Delete the cast and the comment; do not add the type.

- [ ] **Step 1: Replace the stale telemetry cast**

Replace `HumanTeleopPanel.tsx:41-48`:

```ts
  const status = useTelemetry((s) => s.lastFrame?.human_teleop);
```

- [ ] **Step 2: Add the `statusRef` mirror**

The render loop runs on `requestAnimationFrame` and must not re-subscribe on every telemetry frame, so it reads `status` through a ref rather than taking it as an effect dependency. Add next to the other refs (near `HumanTeleopPanel.tsx:55-58`):

```ts
  const statusRef = useRef<HumanTeleopStatus | undefined>(undefined);
```

And immediately after the `status` selector from Step 1, keep it current:

```ts
  useEffect(() => { statusRef.current = status; }, [status]);
```

- [ ] **Step 3: Compute `pinch01` per side in the render loop**

`pinch01` is the calibrated pinch aperture. Add this helper next to `liveThumbIndex` at the bottom of `HumanTeleopPanel.tsx`:

```ts
/** Map a raw thumb-index distance onto [0,1] using the captured calibration.
 *  Returns 0.5 (neutral) when that side isn't calibrated yet, so the overlay
 *  still draws rather than showing a permanently-dashed pinch line. */
function pinch01For(distance: number | null, calib: PinchCalib): number {
  if (distance === null || calib.min_m === null || calib.max_m === null) return 0.5;
  const span = calib.max_m - calib.min_m;
  if (span <= 0) return 0.5;
  return Math.max(0, Math.min(1, (distance - calib.min_m) / span));
}
```

- [ ] **Step 4: Call `buildOverlaySides` in the render loop**

The current render loop reads (`HumanTeleopPanel.tsx:146-153`):

```ts
      const { hands, pose } = runner.detect(video, t);
      const fused = fuseLandmarkResults(pose, hands);

      overlay.draw(toOverlay(fused));
      const ld = liveThumbIndex(fused);
      if (ld.left !== liveDistance.left || ld.right !== liveDistance.right) {
        setLiveDistance(ld);
      }
```

Replace that whole block with — note `ld` moves *above* the `overlay.draw` call because the pinch values now feed it, so there is exactly one `const ld` declaration:

```ts
      const { hands, pose } = runner.detect(video, t);
      const fused = fuseLandmarkResults(pose, hands);
      const ld = liveThumbIndex(fused);

      overlay.draw(buildOverlaySides(pose, hands, {
        leftLost:  statusRef.current?.tracking?.left?.lost ?? false,
        rightLost: statusRef.current?.tracking?.right?.lost ?? false,
        leftPinch01:  pinch01For(ld.left, calib.left),
        rightPinch01: pinch01For(ld.right, calib.right),
      }));

      if (ld.left !== liveDistance.left || ld.right !== liveDistance.right) {
        setLiveDistance(ld);
      }
```

- [ ] **Step 5: Delete `toOverlay` and fix imports**

Delete the whole `toOverlay` function (`HumanTeleopPanel.tsx:308-314`). Update the imports at the top:

```ts
import {
  MediaPipeRunner, fuseLandmarkResults, buildOverlaySides,
  type KeypointFrame, type SideFrame,
} from "@/lib/mediapipe";

import { CameraOverlay, type CameraOverlayHandle } from "./CameraOverlay";
```

(`OverlaySides` is no longer referenced in this file.)

- [ ] **Step 6: Typecheck and test**

```bash
cd hmi/frontend
pnpm tsc --noEmit
pnpm vitest run
```

Expected: no type errors; all tests PASS.

- [ ] **Step 7: Verify visually against the sim backend**

Start the backend on the bimanual sim preset and the frontend against it:

```bash
cd /home/oscar-devos/haller_ws
nohup bash -c 'source "$HOME/venvs/haller-hmi/bin/activate-haller-hmi" >/dev/null 2>&1
export HALLER_HMI_CONFIG=/home/oscar-devos/haller_ws/hmi/backend/config.bimanual-sim.yaml
export MUJOCO_GL=egl
cd /home/oscar-devos/haller_ws/hmi/backend
exec python -m uvicorn haller_hmi.server:app --host 127.0.0.1 --port 8077' > /tmp/backend.log 2>&1 &
disown
cd hmi/frontend && NEXT_PUBLIC_BACKEND_URL=http://localhost:8077 pnpm dev -p 3001
```

Open `http://localhost:3001/teleop/human` and confirm: a green skeleton tracks your shoulder → elbow → wrist, hand landmark ticks appear, and the thumb-to-index pinch line dashes when you close your fingers. This is a human check — there is no automated substitute for "does the skeleton land on the body".

- [ ] **Step 8: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/frontend/components/HumanTeleopPanel.tsx
git diff --cached --name-only
git commit -m "feat(hmi): draw the pose skeleton on the teleop video feed

Replaces the toOverlay() stub that returned null sides and left the video
panel blank since the feature shipped. Lost-state colour comes from the
backend's tracking status rather than a second frontend opinion.

Also drops the as-unknown-as telemetry cast: its comment claimed
TelemetryFrame doesn't type human_teleop, but lib/telemetry.ts:52 already
declares it."
```

---

### Task 5: Scope bars use real per-joint limits

**Files:**
- Modify: `hmi/frontend/components/HumanTeleopPanel.tsx:243-274` (`ArmScopePanel` and its call sites)

**Interfaces:**
- Consumes: `useTelemetry` arm state (`lib/telemetry.ts` `ArmState.joints[j].{min,max}`), already typed.
- Produces: `ArmScopePanel` gains a `limits?: Record<string, { min: number; max: number }>` prop.

**Why this is in scope:** `ArmScopePanel` hardcodes `min={-90} max={90}` for every joint. Real limits come from calibration and differ per joint — the gripper especially. A CLAMPED badge (Task 6) beside a bar showing the joint nowhere near its limit would actively mislead, so the scale must be truthful first.

- [ ] **Step 1: Read real limits from telemetry**

At the top of `HumanTeleopPanel`, next to the `status` selector, add:

```ts
  const armsState = useTelemetry((s) => s.lastFrame?.arms);
```

Add this helper at the bottom of the file:

```ts
/** Per-joint {min,max} in degrees for one arm, straight from calibration via
 *  telemetry. Returns undefined when that arm isn't reporting yet, in which
 *  case ScopeBar falls back to its own default range. */
function limitsFor(
  armsState: Record<string, ArmState> | undefined, armId: string,
): Record<string, { min: number; max: number }> | undefined {
  const joints = armsState?.[armId]?.joints;
  if (!joints) return undefined;
  return Object.fromEntries(
    Object.entries(joints).map(([j, s]) => [j, { min: s.min, max: s.max }]),
  );
}
```

Import the type: `import { useTelemetry, type ArmState } from "@/lib/telemetry";`

- [ ] **Step 2: Thread limits into `ArmScopePanel`**

Replace the two call sites (`HumanTeleopPanel.tsx:244-245`):

```tsx
        <ArmScopePanel label={`arm: ${leftArm}`} goal={status?.goal_deg?.left}
                       limits={limitsFor(armsState, leftArm)} />
        <ArmScopePanel label={`arm: ${rightArm}`} goal={status?.goal_deg?.right}
                       limits={limitsFor(armsState, rightArm)} />
```

And update the component:

```tsx
function ArmScopePanel({
  label, goal, limits,
}: {
  label: string;
  goal?: Record<string, number>;
  limits?: Record<string, { min: number; max: number }>;
}) {
  return (
    <Card className="p-3">
      <div className="flex justify-between text-[12px] font-mono mb-2">
        <span>{label}</span>
      </div>
      <div className="space-y-1">
        {JOINTS.map((j) => (
          // Real calibrated limits when the arm is reporting; the +/-90 default
          // is only a placeholder for an arm that hasn't sent telemetry yet.
          <ScopeBar
            key={j}
            label={j}
            min={limits?.[j]?.min ?? -90}
            max={limits?.[j]?.max ?? 90}
            commanded={goal?.[j] ?? 0}
          />
        ))}
      </div>
    </Card>
  );
}
```

- [ ] **Step 3: Typecheck and test**

```bash
cd hmi/frontend
pnpm tsc --noEmit
pnpm vitest run
```

Expected: no type errors; all tests PASS.

- [ ] **Step 4: Verify against the sim backend**

With the sim backend running (Task 4 Step 7), open `http://localhost:3001/teleop/human`. The gripper bar should now scale differently from the other joints — its calibrated range is not ±90°. Compare against `curl -s http://127.0.0.1:8077/config` and the telemetry `arms.left.joints.gripper.{min,max}`:

```bash
curl -s --max-time 3 http://127.0.0.1:8077/health
```

- [ ] **Step 5: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/frontend/components/HumanTeleopPanel.tsx
git diff --cached --name-only
git commit -m "fix(hmi): teleop scope bars use real per-joint limits

ArmScopePanel hardcoded min=-90 max=90 for every joint, so every bar was
mis-scaled and the gripper badly so. Real calibrated limits already ride
telemetry at arms[id].joints[j].{min,max}."
```

---

### Task 6: Ghost target tick and reason badges

**Files:**
- Modify: `hmi/frontend/lib/api.ts:129-144` (`HumanTeleopStatus`), `hmi/frontend/components/HumanTeleopPanel.tsx` (`ArmScopePanel`)
- Test: `hmi/frontend/__tests__/ScopeBar.test.tsx`

**Interfaces:**
- Consumes: `status().joints` from Task 2; `limits` prop from Task 5.
- Produces: `JointDiag` type exported from `lib/api.ts`; `HumanTeleopStatus.joints`.

**Note:** `ScopeBar` already accepts `intended` and renders a ghost tick when it diverges from `commanded` by more than 0.5° (`ScopeBar.tsx:37-45`). That code is currently unreachable — no caller passes `intended`. This task supplies the data; `ScopeBar` itself needs no change.

- [ ] **Step 1: Add the types**

In `hmi/frontend/lib/api.ts`, above `HumanTeleopStatus`:

```ts
export type JointReason = "ok" | "rate_capped" | "clamped" | "held";

export type JointDiag = {
  /** What the retargeter asked for, in degrees. null when the joint is held. */
  target: number | null;
  committed: number;
  reason: JointReason;
};
```

And add to `HumanTeleopStatus`, after `goal_deg`:

```ts
  joints?: {
    left?:  Record<string, JointDiag>;
    right?: Record<string, JointDiag>;
  };
```

- [ ] **Step 2: Write the failing test**

Add to `hmi/frontend/__tests__/ScopeBar.test.tsx`:

```tsx
it("renders the ghost tick when intended diverges from commanded", () => {
  const { container } = render(
    <ScopeBar label="elbow_flex" min={-90} max={90} commanded={58.4} intended={62.1} />,
  );
  expect(container.querySelector("[data-ghost]")).not.toBeNull();
});

it("omits the ghost tick when intended matches commanded", () => {
  const { container } = render(
    <ScopeBar label="elbow_flex" min={-90} max={90} commanded={58.4} intended={58.4} />,
  );
  expect(container.querySelector("[data-ghost]")).toBeNull();
});
```

- [ ] **Step 3: Run the test to verify it passes already**

```bash
cd hmi/frontend
pnpm vitest run __tests__/ScopeBar.test.tsx
```

Expected: **PASS** — this is a characterisation test locking in behaviour that already exists but had no caller. If it fails, `ScopeBar` is not what this plan assumes; stop and re-read `ScopeBar.tsx` before continuing.

- [ ] **Step 4: Render the badge and pass `intended`**

Update `ArmScopePanel` in `HumanTeleopPanel.tsx` to take a `diag` prop and render both:

```tsx
const REASON_LABEL: Record<string, string> = {
  clamped: "CLAMPED",
  rate_capped: "RATE-CAP",
};

function ArmScopePanel({
  label, goal, limits, diag,
}: {
  label: string;
  goal?: Record<string, number>;
  limits?: Record<string, { min: number; max: number }>;
  diag?: Record<string, JointDiag>;
}) {
  return (
    <Card className="p-3">
      <div className="flex justify-between text-[12px] font-mono mb-2">
        <span>{label}</span>
      </div>
      <div className="space-y-1">
        {JOINTS.map((j) => {
          const d = diag?.[j];
          const badge = d ? REASON_LABEL[d.reason] : undefined;
          return (
            <div key={j} className="flex items-center gap-2">
              <div className="flex-1">
                <ScopeBar
                  label={j}
                  min={limits?.[j]?.min ?? -90}
                  max={limits?.[j]?.max ?? 90}
                  commanded={goal?.[j] ?? 0}
                  intended={d?.target ?? undefined}
                />
              </div>
              <span
                className="w-16 text-right font-mono text-[10px] text-[var(--instrument-warn,oklch(75%_0.16_70))]"
              >
                {badge ?? ""}
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
```

Import the type in `HumanTeleopPanel.tsx`:

```ts
import { api, type HumanTeleopStatus, type JointDiag } from "@/lib/api";
```

- [ ] **Step 5: Pass `diag` at the call sites**

```tsx
        <ArmScopePanel label={`arm: ${leftArm}`} goal={status?.goal_deg?.left}
                       limits={limitsFor(armsState, leftArm)}
                       diag={status?.joints?.left} />
        <ArmScopePanel label={`arm: ${rightArm}`} goal={status?.goal_deg?.right}
                       limits={limitsFor(armsState, rightArm)}
                       diag={status?.joints?.right} />
```

- [ ] **Step 6: Typecheck and test**

```bash
cd hmi/frontend
pnpm tsc --noEmit
pnpm vitest run
```

Expected: no type errors; all tests PASS (39 total).

- [ ] **Step 7: Verify end to end against the sim**

With the sim backend running, drive an arm into a joint limit — swing one arm hard to the side while holding SPACE. Confirm: the ghost tick separates from the filled bar during fast motion, `RATE-CAP` appears during the fast part, and `CLAMPED` appears and stays once the joint reaches its limit.

- [ ] **Step 8: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/frontend/lib/api.ts hmi/frontend/components/HumanTeleopPanel.tsx hmi/frontend/__tests__/ScopeBar.test.tsx
git diff --cached --name-only
git commit -m "feat(hmi): show retarget target and clamp/rate-cap reason per joint

ScopeBar's ghost-tick support existed but no caller passed intended. It
now receives the raw retarget target next to the committed value, with a
CLAMPED / RATE-CAP badge sourced from the backend stage that actually
made the decision."
```

---

### Task 7: Per-side confidence readout

**Files:**
- Modify: `hmi/frontend/components/HumanTeleopPanel.tsx` (render loop + the per-side calibration card)

**Interfaces:**
- Consumes: `SideFrame.confidence` from `fuseLandmarkResults` (already computed at `mediapipe.ts:115`).
- Produces: no new exports.

- [ ] **Step 1: Track confidence in state**

Next to the existing `liveDistance` state (`HumanTeleopPanel.tsx:68-70`), add:

```ts
  const [liveConf, setLiveConf] = useState<{ left: number | null; right: number | null }>({
    left: null, right: null,
  });
```

- [ ] **Step 2: Update it in the render loop**

Immediately after the existing `setLiveDistance` block in the render loop:

```ts
      // Functional update with an identity bail-out: returning `prev` unchanged
      // makes React skip the re-render, so this needs no effect dependency.
      // Do NOT add `liveConf` to the effect's dep array — that would tear down
      // and recreate the requestAnimationFrame loop on every confidence change.
      const lc = { left: fused.left?.confidence ?? null, right: fused.right?.confidence ?? null };
      setLiveConf((prev) =>
        prev.left === lc.left && prev.right === lc.right ? prev : lc,
      );
```

Leave the effect's dependency array as `[calib]`.

- [ ] **Step 3: Add a `confidence` prop to `PinchCalibrationStep`**

In `hmi/frontend/components/PinchCalibrationStep.tsx`, extend the signature (currently lines 22-32) to:

```tsx
export function PinchCalibrationStep({
  liveDistance,
  confidence,
  side,
  value,
  onChange,
}: {
  liveDistance: number | null;
  /** MediaPipe tracking confidence for this side, [0,1]. Display only —
   *  a low value does not reduce authority. */
  confidence?: number | null;
  side: PinchSide;
  value: PinchCalib;
  onChange: (next: PinchCalib) => void;
}) {
```

Insert a `conf` row directly after the existing `live` row (currently lines 51-54), so the two live readouts sit together:

```tsx
        <div className="flex justify-between">
          <span className="text-muted-foreground">conf</span>
          <span className={
            confidence !== null && confidence !== undefined && confidence < 0.5
              ? "tabular-nums text-[var(--instrument-warn,oklch(75%_0.16_70))]"
              : "tabular-nums"
          }>
            {confidence === null || confidence === undefined ? "—" : confidence.toFixed(2)}
          </span>
        </div>
```

- [ ] **Step 4: Pass it at both call sites**

At the two `PinchCalibrationStep` usages in `HumanTeleopPanel.tsx` (lines 233 and 237), add the prop — `liveConf.left` on the left-side card, `liveConf.right` on the right-side card:

```tsx
            confidence={liveConf.left}
```

- [ ] **Step 5: Typecheck and test**

```bash
cd hmi/frontend
pnpm tsc --noEmit
pnpm vitest run
```

Expected: no type errors; all tests PASS.

- [ ] **Step 6: Verify against the sim**

Open the page. Confidence should read ~0.9 with both arms clearly visible, and drop (turning amber below 0.5) when you occlude an arm or step out of frame. Note: confidence is **displayed only** — a low value does not reduce authority. That gate is explicitly out of scope (spec §2, §8).

- [ ] **Step 7: Commit**

```bash
cd /home/oscar-devos/haller_ws
git add hmi/frontend/components/HumanTeleopPanel.tsx hmi/frontend/components/PinchCalibrationStep.tsx
git diff --cached --name-only
git commit -m "feat(hmi): per-side tracking confidence readout

MediaPipe confidence was computed and shipped to the backend but never
surfaced. Display only — it does not gate authority; that remains out of
scope and is recorded as a pre-hardware risk in the spec."
```

---

## Final verification

- [ ] **Full backend suite**

```bash
cd hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/ -q
```

Expected: **161 passed**.

- [ ] **Full frontend suite and typecheck**

```bash
cd hmi/frontend
pnpm tsc --noEmit
pnpm vitest run
```

Expected: no type errors; **39 passed**.

- [ ] **Recorder regression**

`goal_deg` is the dataset `action` column and must be untouched by all of the above.

```bash
cd hmi/backend
MUJOCO_GL=egl ~/venvs/haller-hmi/bin/python -m pytest tests/test_recorder.py -v
```

Expected: PASS, unchanged from before this plan.

- [ ] **End-to-end in sim**

Backend on `config.bimanual-sim.yaml`, frontend against it, `/teleop/human` open. Hold SPACE and confirm all four deliverables at once: skeleton tracks the body, confidence reads plausibly, ghost ticks separate from the bars during fast motion, and CLAMPED appears when a joint reaches its limit.

## Out of scope — do not implement

These are recorded in spec §8 and must **not** be added while executing this plan:

- **Confidence gating.** Display only. Reducing authority on low confidence is a separate decision.
- **Vendoring the MediaPipe WASM bundle and models.** The CDN dependency at `mediapipe.ts:130-146` stays. It is the biggest remaining robustness gap and deserves its own piece of work.
- **Scrubbing, freeze-frame, or a rolling buffer.** Live display only.
- **Clearing `goal_deg` on stop** to fix the stale-pose-while-idle display. Cosmetic, and it touches the recorder's action column.
