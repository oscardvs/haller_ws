# Motion Safety Envelope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make it impossible for a discrete arm command to sweep an unplanned path across the workspace, and make the real and sim paths run the same motion-safety code so the defect cannot exist in one and not the other.

**Architecture:** Three pure functions in `safety.py` (step cap, move-size check, ramp planner), a `MoveExecutor` that plays ramp waypoints on a background thread while re-checking the mode guard, and a shared `move_to`/`home` in a new `motion.py` that both `ArmHandle` and `SimArmHandle` delegate to. Handles keep their own `send_goal` because the transport genuinely differs; only the distance/velocity policy is shared.

**Tech Stack:** Python 3.12, FastAPI, pytest, MuJoCo (sim tests), lerobot (real handle only).

**Spec:** `docs/superpowers/specs/2026-08-01-motion-safety-envelope-design.md`

## Global Constraints

- Backend tests run from `hmi/backend` with ROS's pytest plugins suppressed: `source /home/odesha/venvs/haller-hmi/bin/activate && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio`. Baseline before this plan: **269 passed**.
- **Never add `Co-Authored-By:` or any co-author trailer to a commit message.** Project rule in `CLAUDE.md`.
- Do not add `set -u` to any script that sources ROS — `setup.bash` dereferences `AMENT_TRACE_SETUP_FILES` and dies.
- Work on branch `feat/motion-safety-envelope` (already created, spec already committed there).
- All work in this plan is verifiable without hardware. The rig is down pending a 10 A buck.
- Joint angles are degrees everywhere in this layer. Both handles expose `read_joints_deg() -> dict[str, float]` keyed by LeRobot snake_case names.

---

## File Structure

| File | Responsibility |
|---|---|
| `hmi/backend/haller_hmi/safety.py` | **Modify.** Add `limit_step`, `check_move_size`, `plan_ramp` beside `clamp_joint_goal`. Pure functions, no I/O. |
| `hmi/backend/haller_hmi/config.py` | **Modify.** Add `MotionConfig` + per-arm overrides + `resolve_motion()`. |
| `hmi/backend/haller_hmi/motion.py` | **Create.** `MoveRefused`, `MoveExecutor`, shared `move_to()` and `home()`. |
| `hmi/backend/haller_hmi/arm.py` | **Modify.** Drop silent torque re-enable, add step cap, delete `home()`. |
| `hmi/backend/haller_hmi/sim/arm.py` | **Modify.** Same three changes. |
| `hmi/backend/haller_hmi/server.py` | **Modify.** Routes call shared `move_to`/`home`, map `MoveRefused` → 409. |
| `hmi/backend/config.yaml` | **Modify.** Add the `motion:` block. |
| `hmi/backend/tests/test_safety.py` | **Modify.** Primitives. |
| `hmi/backend/tests/test_config.py` | **Modify.** Config resolution. |
| `hmi/backend/tests/test_motion.py` | **Create.** Executor + shared policy. |
| `hmi/backend/tests/sim/test_motion_sim.py` | **Create.** Incident reproduction + parity. |

---

### Task 1: Motion primitives in `safety.py`

**Files:**
- Modify: `hmi/backend/haller_hmi/safety.py` (after `clamp_joint_goal`, which ends at line 49)
- Test: `hmi/backend/tests/test_safety.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `limit_step(current: dict[str,float], goal: dict[str,float], max_step_deg: float) -> dict[str,float]`; `check_move_size(current: dict[str,float], goal: dict[str,float], threshold_deg: float) -> dict[str,float]`; `plan_ramp(current: dict[str,float], goal: dict[str,float], max_speed_deg_s: float, hz: float) -> list[dict[str,float]]`

- [ ] **Step 1: Write the failing tests**

Append to `hmi/backend/tests/test_safety.py`:

```python
from haller_hmi.safety import check_move_size, limit_step, plan_ramp


def test_limit_step_caps_large_delta_both_directions():
    current = {"shoulder_pan": 0.0, "elbow_flex": 10.0}
    goal = {"shoulder_pan": 100.0, "elbow_flex": -90.0}
    out = limit_step(current, goal, max_step_deg=1.2)
    assert out == {"shoulder_pan": 1.2, "elbow_flex": 8.8}


def test_limit_step_passes_small_delta_through_untouched():
    current = {"shoulder_pan": 5.0}
    out = limit_step(current, {"shoulder_pan": 5.5}, max_step_deg=1.2)
    assert out == {"shoulder_pan": 5.5}


def test_limit_step_passes_through_joints_with_no_reference_position():
    # clamp_joint_goal already dropped unknown joints; a joint missing from
    # `current` means we have no measurement, not that the joint is bogus.
    out = limit_step({}, {"wrist_roll": 42.0}, max_step_deg=1.2)
    assert out == {"wrist_roll": 42.0}


def test_check_move_size_reports_only_offending_joints_with_signed_delta():
    current = {"shoulder_pan": 0.0, "elbow_flex": 0.0, "gripper": 0.0}
    goal = {"shoulder_pan": 45.0, "elbow_flex": -31.0, "gripper": 5.0}
    assert check_move_size(current, goal, threshold_deg=30.0) == {
        "shoulder_pan": 45.0,
        "elbow_flex": -31.0,
    }


def test_check_move_size_empty_when_all_within_threshold():
    assert check_move_size({"a": 0.0}, {"a": 29.9}, threshold_deg=30.0) == {}


def test_plan_ramp_bounds_every_consecutive_step():
    current = {"shoulder_pan": 0.0, "elbow_flex": 0.0}
    goal = {"shoulder_pan": 20.0, "elbow_flex": -10.0}
    wps = plan_ramp(current, goal, max_speed_deg_s=60.0, hz=50.0)
    step = 60.0 / 50.0
    prev = current
    for wp in wps:
        for j, v in wp.items():
            assert abs(v - prev[j]) <= step + 1e-9
        prev = wp
    assert wps[-1] == goal


def test_plan_ramp_returns_empty_when_already_at_goal():
    assert plan_ramp({"a": 5.0}, {"a": 5.0}, max_speed_deg_s=60.0, hz=50.0) == []


def test_plan_ramp_rejects_nonpositive_rates():
    with pytest.raises(ValueError):
        plan_ramp({"a": 0.0}, {"a": 1.0}, max_speed_deg_s=0.0, hz=50.0)
    with pytest.raises(ValueError):
        plan_ramp({"a": 0.0}, {"a": 1.0}, max_speed_deg_s=60.0, hz=0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/odesha/haller_ws/hmi/backend && source /home/odesha/venvs/haller-hmi/bin/activate && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio tests/test_safety.py -k "limit_step or check_move_size or plan_ramp"
```

Expected: collection error or FAIL — `ImportError: cannot import name 'check_move_size' from 'haller_hmi.safety'`.

- [ ] **Step 3: Implement the primitives**

Insert into `hmi/backend/haller_hmi/safety.py` immediately after `clamp_joint_goal` (line 49), and add `import math` to the imports at the top:

```python
def limit_step(
    current: dict[str, float],
    goal: dict[str, float],
    max_step_deg: float,
) -> dict[str, float]:
    """Cap each joint's per-call delta at `max_step_deg`.

    The streaming half of the motion envelope. A corrupted input frame
    commanding a 100° jump becomes one bounded step, and the next good frame
    corrects it. Joints absent from `current` pass through: callers run
    clamp_joint_goal first, so a missing key means "no measurement", not
    "unknown joint".
    """
    out: dict[str, float] = {}
    for joint, value in goal.items():
        if joint not in current:
            out[joint] = value
            continue
        ref = current[joint]
        delta = value - ref
        if delta > max_step_deg:
            out[joint] = ref + max_step_deg
        elif delta < -max_step_deg:
            out[joint] = ref - max_step_deg
        else:
            out[joint] = value
    return out


def check_move_size(
    current: dict[str, float],
    goal: dict[str, float],
    threshold_deg: float,
) -> dict[str, float]:
    """Return {joint: signed delta} for joints moving further than the threshold.

    Empty means the move is small enough to ramp. Non-empty means refuse — see
    the 2026-08-01 incident, where Home right after a recalibration commanded a
    slew across the whole workspace because 0° had just been redefined.
    """
    out: dict[str, float] = {}
    for joint, value in goal.items():
        if joint not in current:
            continue
        delta = value - current[joint]
        if abs(delta) > threshold_deg:
            out[joint] = delta
    return out


def plan_ramp(
    current: dict[str, float],
    goal: dict[str, float],
    max_speed_deg_s: float,
    hz: float,
) -> list[dict[str, float]]:
    """Interpolated waypoints from `current` to `goal`, bounded by max_speed_deg_s.

    All joints share one step count so they arrive together; the joint with the
    largest excursion sets the pace. The final waypoint is exactly `goal`.
    """
    if max_speed_deg_s <= 0:
        raise ValueError("max_speed_deg_s must be positive")
    if hz <= 0:
        raise ValueError("hz must be positive")
    joints = [j for j in goal if j in current]
    if not joints:
        return []
    largest = max(abs(goal[j] - current[j]) for j in joints)
    if largest == 0.0:
        return []
    steps = math.ceil(largest / (max_speed_deg_s / hz))
    return [
        {j: current[j] + (goal[j] - current[j]) * (i / steps) for j in joints}
        for i in range(1, steps + 1)
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Same command as Step 2. Expected: all PASS.

- [ ] **Step 5: Run the full suite**

```bash
cd /home/odesha/haller_ws/hmi/backend && source /home/odesha/venvs/haller-hmi/bin/activate && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio
```

Expected: 269 + 8 new = **277 passed**.

- [ ] **Step 6: Commit**

```bash
git add hmi/backend/haller_hmi/safety.py hmi/backend/tests/test_safety.py
git commit -m "feat(hmi): add step-cap, move-size and ramp primitives to safety.py"
```

---

### Task 2: `MotionConfig`

**Files:**
- Modify: `hmi/backend/haller_hmi/config.py`
- Modify: `hmi/backend/config.yaml`
- Test: `hmi/backend/tests/test_config.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `MotionConfig(max_speed_deg_s: float = 60.0, large_move_deg: float = 30.0, ramp_hz: float = 50.0)`; `Config.motion: MotionConfig`; `ArmConfig.max_speed_deg_s: float | None`, `ArmConfig.large_move_deg: float | None`; `resolve_motion(arm: ArmConfig, default: MotionConfig) -> MotionConfig`

- [ ] **Step 1: Write the failing tests**

Append to `hmi/backend/tests/test_config.py`:

```python
from haller_hmi.config import ArmConfig, MotionConfig, resolve_motion


def test_motion_defaults_are_conservative():
    m = MotionConfig()
    # STS3215 does ~375 deg/s at 7.4 V; 60 is ~16% of capability.
    assert m.max_speed_deg_s == 60.0
    assert m.large_move_deg == 30.0
    assert m.ramp_hz == 50.0


def test_resolve_motion_uses_global_when_arm_sets_no_override():
    arm = ArmConfig(id="right", model="so101_follower", port="/dev/null",
                    calibration_id="haller_follower")
    assert resolve_motion(arm, MotionConfig()) == MotionConfig()


def test_resolve_motion_applies_per_arm_overrides():
    arm = ArmConfig(id="right", model="so101_follower", port="/dev/null",
                    calibration_id="haller_follower",
                    max_speed_deg_s=25.0, large_move_deg=15.0)
    got = resolve_motion(arm, MotionConfig())
    assert got.max_speed_deg_s == 25.0
    assert got.large_move_deg == 15.0
    assert got.ramp_hz == 50.0  # not overridden, inherits the global


def test_load_config_reads_motion_block(tmp_path):
    from haller_hmi.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text(
        "arms: []\ncameras: []\nmotion:\n  max_speed_deg_s: 30.0\n  large_move_deg: 20.0\n"
    )
    cfg = load_config(p)
    assert cfg.motion.max_speed_deg_s == 30.0
    assert cfg.motion.large_move_deg == 20.0
    assert cfg.motion.ramp_hz == 50.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/odesha/haller_ws/hmi/backend && source /home/odesha/venvs/haller-hmi/bin/activate && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio tests/test_config.py -k motion
```

Expected: FAIL — `ImportError: cannot import name 'MotionConfig'`.

- [ ] **Step 3: Implement the config**

In `hmi/backend/haller_hmi/config.py`, add these two optional fields to `ArmConfig` (after `sim_arm_name`, line 25):

```python
    # Per-arm motion overrides. None means "inherit the global motion block".
    max_speed_deg_s: float | None = None
    large_move_deg: float | None = None
```

Add the dataclass after `TelemetryConfig` (line 39):

```python
@dataclass
class MotionConfig:
    """Bounds on commanded arm motion. See
    docs/superpowers/specs/2026-08-01-motion-safety-envelope-design.md.
    """
    # The STS3215 reaches ~375 deg/s at 7.4 V. This is deliberately ~16% of
    # capability: two arms share one bench.
    max_speed_deg_s: float = 60.0
    # A discrete move needing more than this on any joint is refused outright
    # rather than ramped, because ramping still sweeps an unplanned path.
    large_move_deg: float = 30.0
    # Waypoint rate. Also sets the streaming per-step cap, at
    # max_speed_deg_s / ramp_hz.
    ramp_hz: float = 50.0
```

Add to `Config` (after `cameras`, line 78):

```python
    motion: MotionConfig = field(default_factory=MotionConfig)
```

Add to the `Config(...)` construction inside `load_config`:

```python
        motion=MotionConfig(**raw.get("motion", {})),
```

And add this module-level function at the end of the file:

```python
def resolve_motion(arm: ArmConfig, default: MotionConfig) -> MotionConfig:
    """Merge an arm's overrides over the global motion config."""
    return MotionConfig(
        max_speed_deg_s=(arm.max_speed_deg_s
                         if arm.max_speed_deg_s is not None
                         else default.max_speed_deg_s),
        large_move_deg=(arm.large_move_deg
                        if arm.large_move_deg is not None
                        else default.large_move_deg),
        ramp_hz=default.ramp_hz,
    )
```

- [ ] **Step 4: Add the block to `config.yaml`**

Add at top level of `hmi/backend/config.yaml`:

```yaml
# Bounds on commanded arm motion. Added after the 2026-08-01 incident, where
# Home straight after a recalibration slewed the right arm into the bench and
# burnt the 7.4 V DC-DC. See
# docs/superpowers/specs/2026-08-01-motion-safety-envelope-design.md.
motion:
  max_speed_deg_s: 60.0
  large_move_deg: 30.0
  ramp_hz: 50.0
```

- [ ] **Step 5: Run the tests to verify they pass**

Same command as Step 2, then the full suite. Expected: **281 passed**.

- [ ] **Step 6: Commit**

```bash
git add hmi/backend/haller_hmi/config.py hmi/backend/config.yaml hmi/backend/tests/test_config.py
git commit -m "feat(hmi): add motion bounds config with per-arm overrides"
```

---

### Task 3: Stop `send_goal` silently energizing a limp arm

**Files:**
- Modify: `hmi/backend/haller_hmi/arm.py:100-111` (`send_goal`)
- Modify: `hmi/backend/haller_hmi/sim/arm.py:69-81` (`send_goal`)
- Test: `hmi/backend/tests/test_arm.py`, `hmi/backend/tests/sim/test_sim_arm_handle.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `send_goal` on both handles no longer calls `enable_torque()`.

**Context for the implementer:** every legitimate caller already enables torque explicitly first — `server.py:275`, `human_teleop.py:498` and `:558`, `calibration.py:191`. The only path that relied on the side effect was `home()`, which is the one that caused the incident: a limp arm was silently energized and then commanded to slew.

- [ ] **Step 1: Write the failing tests**

Append to `hmi/backend/tests/test_arm.py`:

```python
def test_send_goal_does_not_silently_enable_torque(monkeypatch):
    """A limp arm must stay limp. Silently energizing it is half of what made
    the 2026-08-01 Home command dangerous."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.torque_enabled = False

    handle.send_goal({"shoulder_pan": 10.0})

    handle.robot.bus.enable_torque.assert_not_called()
    assert handle.torque_enabled is False
```

Append the equivalent to `hmi/backend/tests/sim/test_sim_arm_handle.py`. That file has **no fixtures** — there is no `conftest.py` anywhere under `tests/`. Use its existing module-level helper `_make_world_and_handle()` (line 46), which returns `(world, handle)` built on `TINY_XML`:

```python
def test_sim_send_goal_does_not_silently_enable_torque():
    _world, handle = _make_world_and_handle()
    handle.connect()
    handle.guard.set(Mode.MANUAL)
    handle.torque_enabled = False

    handle.send_goal({"shoulder_pan": 10.0})

    assert handle.torque_enabled is False
```

Note: `TINY_XML` models only two joints — `right_Rotation` and `right_Jaw` — so the handle exposes exactly two LeRobot names, `shoulder_pan` and `gripper` (`LEROBOT_TO_MJCF`, `sim/arm.py:26`). Do not reference any other joint in a TINY_XML-backed test.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/odesha/haller_ws/hmi/backend && source /home/odesha/venvs/haller-hmi/bin/activate && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio -k "silently_enable_torque"
```

Expected: FAIL — `torque_enabled` is True because `send_goal` re-enabled it.

- [ ] **Step 3: Remove the side effect from both handles**

In `hmi/backend/haller_hmi/arm.py`, delete these four lines from `send_goal`:

```python
        # If the user previously disabled torque (free-drive), re-engage it before
        # commanding new positions — otherwise the goal is silently stored but the
        # arm doesn't move.
        if not self.torque_enabled:
            self.enable_torque()
```

In `hmi/backend/haller_hmi/sim/arm.py`, delete the equivalent two lines from `send_goal`:

```python
        if not self.torque_enabled:
            self.enable_torque()
```

- [ ] **Step 4: Run the tests to verify they pass, then the full suite**

Expected: the two new tests PASS, **283 passed** overall. If any existing test fails because it relied on the side effect, that test was asserting the buggy behaviour — update it to enable torque explicitly first, and note which one in the commit body.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/arm.py hmi/backend/haller_hmi/sim/arm.py \
        hmi/backend/tests/test_arm.py hmi/backend/tests/sim/test_sim_arm_handle.py
git commit -m "fix(hmi): stop send_goal silently energizing a torque-disabled arm"
```

---

### Task 4: Per-step cap on the streaming path

**Files:**
- Modify: `hmi/backend/haller_hmi/arm.py` (`ArmHandle`)
- Modify: `hmi/backend/haller_hmi/sim/arm.py` (`SimArmHandle`)
- Test: `hmi/backend/tests/test_arm.py`

**Interfaces:**
- Consumes: `limit_step` from Task 1; `MotionConfig` from Task 2.
- Produces: both handles gain `motion: MotionConfig` (field, default `MotionConfig()`) and `_last_commanded: dict[str, float] | None`; `send_goal` applies the cap.

**Why against the last command, not a fresh read:** `read_joints_deg()` on the real arm is a synchronous serial round-trip. The teleop loop runs at 60 Hz and already reads the leader each tick; adding a follower read inside `send_goal` would double serial traffic on the arm that is already showing intermittent UART failures. Limiting against the last *commanded* position bounds the command slew directly and costs no I/O. `_last_commanded` is seeded from a real read whenever it is `None`, which covers first use and any torque toggle.

- [ ] **Step 1: Write the failing tests**

Append to `hmi/backend/tests/test_arm.py`:

```python
def test_send_goal_caps_a_garbage_jump_to_one_step(monkeypatch):
    """A corrupted frame commanding +100 deg must yield one bounded step. Also
    covers the suspected UART corruption that poisoned the right arm's sweep."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = {"shoulder_pan": 0.0}

    sent = handle.send_goal({"shoulder_pan": 100.0})

    assert sent["shoulder_pan"] == pytest.approx(1.2)


def test_send_goal_tracks_last_commanded_across_calls(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = {"shoulder_pan": 0.0}

    handle.send_goal({"shoulder_pan": 100.0})
    second = handle.send_goal({"shoulder_pan": 100.0})

    assert second["shoulder_pan"] == pytest.approx(2.4)


def test_send_goal_seeds_last_commanded_from_a_real_read(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = None
    handle.robot.get_observation.return_value = {"shoulder_pan.pos": 50.0}

    sent = handle.send_goal({"shoulder_pan": 100.0})

    assert sent["shoulder_pan"] == pytest.approx(51.2)
```

Add `from haller_hmi.config import MotionConfig` to the imports at the top of `tests/test_arm.py`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/odesha/haller_ws/hmi/backend && source /home/odesha/venvs/haller-hmi/bin/activate && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio tests/test_arm.py -k "garbage_jump or last_commanded or seeds_last"
```

Expected: FAIL — `AttributeError: 'ArmHandle' object has no attribute 'motion'`.

- [ ] **Step 3: Implement the cap in `ArmHandle`**

In `hmi/backend/haller_hmi/arm.py`, add to the imports:

```python
from .config import ArmConfig, MotionConfig
from .safety import Mode, ModeGuard, clamp_joint_goal, limit_step
```

Add two fields to the `ArmHandle` dataclass (after `torque_enabled`):

```python
    motion: MotionConfig = field(default_factory=MotionConfig)
    _last_commanded: dict[str, float] | None = None
```

Replace the body of `send_goal` with:

```python
    def send_goal(self, goal_deg: dict[str, float]) -> dict[str, float]:
        self.guard.assert_manual()
        clamped = clamp_joint_goal(goal_deg, self.joint_limits_deg)
        if self._last_commanded is None:
            # First command since connect or a torque toggle: seed from a real
            # read. Every later call limits against the last command, so the
            # 60 Hz teleop path costs no extra serial traffic.
            self._last_commanded = self.read_joints_deg()
        capped = limit_step(
            self._last_commanded,
            clamped,
            self.motion.max_speed_deg_s / self.motion.ramp_hz,
        )
        action = {f"{j}.pos": v for j, v in capped.items()}
        assert self.robot is not None
        self.robot.send_action(action)
        self._last_commanded = {**self._last_commanded, **capped}
        return capped
```

Invalidate the seed in both torque methods, so a hand-moved arm is re-read:

```python
    def disable_torque(self) -> None:
        if self.robot is not None:
            self.robot.bus.disable_torque()
            self.torque_enabled = False
            self._last_commanded = None

    def enable_torque(self) -> None:
        if self.robot is not None:
            self.robot.bus.enable_torque()
            self.torque_enabled = True
            self._last_commanded = None
```

- [ ] **Step 4: Implement the same cap in `SimArmHandle`**

In `hmi/backend/haller_hmi/sim/arm.py`, add `MotionConfig` and `limit_step` imports, add the same two dataclass fields, and apply the identical cap between `clamp_joint_goal` and the MJCF translation:

```python
    def send_goal(self, goal_deg: dict[str, float]) -> dict[str, float]:
        self.guard.assert_manual()
        clamped = clamp_joint_goal(goal_deg, self.joint_limits_deg)
        if self._last_commanded is None:
            self._last_commanded = self.read_joints_deg()
        capped = limit_step(
            self._last_commanded,
            clamped,
            self.motion.max_speed_deg_s / self.motion.ramp_hz,
        )
        # Translate snake_case → CamelCase + add arm prefix for the world.
        mjcf_goal = {
            f"{self._prefix}{LEROBOT_TO_MJCF[j]}": v
            for j, v in capped.items()
            if j in LEROBOT_TO_MJCF
        }
        self.world.write_ctrl_deg(self.config.sim_arm_name, mjcf_goal)
        self._last_commanded = {**self._last_commanded, **capped}
        return capped
```

Add `self._last_commanded = None` to `SimArmHandle.disable_torque` and `enable_torque` too.

- [ ] **Step 5: Wire the config through `ArmManager`**

In `ArmManager.__init__`, accept the global motion config and hand each handle its resolved copy:

```python
    def __init__(self, arm_configs: list[ArmConfig],
                 motion: MotionConfig | None = None):
        self._configs = [c for c in arm_configs if c.enabled]
        self._motion = motion or MotionConfig()
        self._handles: dict[str, "ArmHandle | SimArmHandle"] = {}
        self._world = None
```

In `connect_all`, set `handle.motion = resolve_motion(cfg, self._motion)` immediately after each handle is constructed (both the sim and real branches). Import `resolve_motion` from `.config`. In `server.py`, pass it: `ArmManager(cfg.arms, motion=cfg.motion)`.

- [ ] **Step 6: Run the tests to verify they pass, then the full suite**

Expected: the three new tests PASS, **286 passed** overall.

Note for the implementer: existing teleop tests assert exact forwarded values (e.g. `test_arm.py`'s teleop loop test asserts `{"shoulder_pan": 42.0, ...}`). With a 1.2° cap and `_last_commanded` seeded from a mock reading 0.0, those will now fail legitimately. Fix them by setting a `MotionConfig` with a large `max_speed_deg_s` on the test handle, so the test keeps testing forwarding rather than capping.

- [ ] **Step 7: Commit**

```bash
git add hmi/backend/haller_hmi/arm.py hmi/backend/haller_hmi/sim/arm.py \
        hmi/backend/haller_hmi/server.py hmi/backend/tests/
git commit -m "feat(hmi): cap per-call joint delta on the streaming goal path"
```

---

### Task 5: `MoveExecutor`

**Files:**
- Create: `hmi/backend/haller_hmi/motion.py`
- Test: `hmi/backend/tests/test_motion.py` (create)

**Interfaces:**
- Consumes: `ModeError` from `safety.py`.
- Produces: `class MoveRefused(Exception)`; `class MoveExecutor` with `__init__(handle)`, `run(waypoints: list[dict[str,float]], hz: float) -> None`, `cancel() -> None`, `is_running: bool`

- [ ] **Step 1: Write the failing tests**

Create `hmi/backend/tests/test_motion.py`:

```python
import threading
import time
from unittest.mock import MagicMock

from haller_hmi.motion import MoveExecutor
from haller_hmi.safety import Mode, ModeGuard


def _fake_handle():
    h = MagicMock()
    h.guard = ModeGuard(Mode.MANUAL)
    h.config.id = "right"
    h.sent = []
    h.send_goal.side_effect = lambda wp: h.sent.append(wp)
    return h


def test_executor_plays_every_waypoint_in_order():
    h = _fake_handle()
    ex = MoveExecutor(h)
    wps = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]

    ex.run(wps, hz=200.0)
    ex.wait(timeout=5.0)

    assert h.sent == wps
    assert ex.is_running is False


def test_estop_mid_ramp_halts_the_move():
    """/estop sets Mode.STOP on every guard. The executor re-checks the guard
    before each waypoint, so that is the whole cancellation mechanism."""
    h = _fake_handle()
    ex = MoveExecutor(h)
    wps = [{"a": float(i)} for i in range(200)]

    ex.run(wps, hz=200.0)
    time.sleep(0.05)
    h.guard.set(Mode.STOP)
    ex.wait(timeout=5.0)

    assert len(h.sent) < len(wps), "ramp should have stopped early"
    assert ex.is_running is False


def test_cancel_stops_the_ramp():
    h = _fake_handle()
    ex = MoveExecutor(h)
    ex.run([{"a": float(i)} for i in range(200)], hz=200.0)
    time.sleep(0.05)
    ex.cancel()

    assert ex.is_running is False


def test_a_new_run_cancels_the_one_in_flight():
    h = _fake_handle()
    ex = MoveExecutor(h)
    ex.run([{"a": float(i)} for i in range(200)], hz=200.0)
    time.sleep(0.02)
    ex.run([{"b": 1.0}], hz=200.0)
    ex.wait(timeout=5.0)

    assert h.sent[-1] == {"b": 1.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/odesha/haller_ws/hmi/backend && source /home/odesha/venvs/haller-hmi/bin/activate && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio tests/test_motion.py
```

Expected: FAIL — `ModuleNotFoundError: No module named 'haller_hmi.motion'`.

- [ ] **Step 3: Implement `motion.py`**

Create `hmi/backend/haller_hmi/motion.py`:

```python
# hmi/backend/haller_hmi/motion.py
"""Discrete arm moves: bounded, cancellable, and shared between real and sim.

The HTTP routes that trigger a move are `async def` calling synchronous motion
code, so a ramp must never run inline — it would stall the event loop for every
other client. Hence the background executor.

Cancellation is not a separate channel: `/estop` already sets Mode.STOP on every
arm's guard, so re-checking the guard before each waypoint means E-STOP, a mode
change and a teleop takeover all stop a ramp for free.
"""
from __future__ import annotations

import logging
import threading
import time

from .safety import ModeError

logger = logging.getLogger(__name__)


class MoveRefused(Exception):
    """A discrete move was rejected before any motion was commanded.

    Deliberately not `calibration.ConflictError`: motion must not depend on the
    calibration module. Routes map this to HTTP 409.
    """


class MoveExecutor:
    """Plays ramp waypoints on a background thread, one arm per instance."""

    def __init__(self, handle) -> None:
        self._handle = handle
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def run(self, waypoints: list[dict[str, float]], hz: float) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        with self._lock:
            self._cancel_locked()
            self._cancel = threading.Event()
            self._thread = threading.Thread(
                target=self._play,
                args=(waypoints, hz, self._cancel),
                name=f"move-{getattr(self._handle.config, 'id', '?')}",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            self._cancel_locked()

    def _cancel_locked(self) -> None:
        self._cancel.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current ramp finishes. Test helper."""
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)

    def _play(self, waypoints, hz: float, cancel: threading.Event) -> None:
        period = 1.0 / hz
        arm_id = getattr(self._handle.config, "id", "?")
        for waypoint in waypoints:
            if cancel.is_set():
                logger.warning("move on arm %s cancelled", arm_id)
                return
            try:
                self._handle.guard.assert_manual()
            except ModeError:
                logger.warning("move on arm %s stopped: mode left manual", arm_id)
                return
            self._handle.send_goal(waypoint)
            time.sleep(period)
```

- [ ] **Step 4: Run the tests to verify they pass, then the full suite**

Expected: 4 new tests PASS, **290 passed** overall.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/motion.py hmi/backend/tests/test_motion.py
git commit -m "feat(hmi): add cancellable background MoveExecutor for ramped moves"
```

---

### Task 6: Shared `move_to` / `home`, deleting both duplicates

**Files:**
- Modify: `hmi/backend/haller_hmi/motion.py`
- Modify: `hmi/backend/haller_hmi/arm.py` (delete `home()`, currently at line 114)
- Modify: `hmi/backend/haller_hmi/sim/arm.py` (delete `home()`, currently at line 83)
- Test: `hmi/backend/tests/test_motion.py`

**Interfaces:**
- Consumes: `check_move_size`, `plan_ramp`, `clamp_joint_goal` (Task 1); `MotionConfig` (Task 2); `MoveExecutor`, `MoveRefused` (Task 5).
- Produces: `move_to(handle, goal_deg: dict[str,float]) -> dict[str,float]`; `home(handle) -> dict[str,float]`. Both read `handle.motion` and use `handle.executor`.

- [ ] **Step 1: Write the failing tests**

Append to `hmi/backend/tests/test_motion.py`:

```python
import pytest

from haller_hmi.config import MotionConfig
from haller_hmi.motion import MoveExecutor, MoveRefused, home, move_to


def _movable_handle(current, limits=None):
    h = _fake_handle()
    h.torque_enabled = True
    h.motion = MotionConfig(max_speed_deg_s=60.0, large_move_deg=30.0, ramp_hz=50.0)
    h.joint_limits_deg = limits or {j: (-180.0, 180.0) for j in current}
    h.read_joints_deg.return_value = current
    h.executor = MoveExecutor(h)
    return h


def test_move_to_refuses_when_any_joint_exceeds_the_threshold():
    h = _movable_handle({"shoulder_pan": 0.0, "gripper": 0.0})
    with pytest.raises(MoveRefused) as e:
        move_to(h, {"shoulder_pan": 90.0, "gripper": 1.0})
    assert "shoulder_pan" in str(e.value)
    assert "gripper" not in str(e.value)
    assert h.sent == [], "nothing may be commanded when a move is refused"


def test_home_refuses_right_after_a_recalibration():
    """The 2026-08-01 incident. Calibration redefines 0 deg, leaving the arm far
    from it; Home then slewed the arm across the bench. It must refuse."""
    h = _movable_handle({"shoulder_pan": -126.5, "wrist_flex": 148.9})
    with pytest.raises(MoveRefused):
        home(h)
    assert h.sent == []


def test_move_to_ramps_a_small_move():
    h = _movable_handle({"shoulder_pan": 0.0})
    move_to(h, {"shoulder_pan": 10.0})
    h.executor.wait(timeout=5.0)
    assert h.sent, "a small move should have been commanded"
    assert h.sent[-1]["shoulder_pan"] == pytest.approx(10.0)


def test_move_to_refuses_when_torque_is_disabled():
    h = _movable_handle({"shoulder_pan": 0.0})
    h.torque_enabled = False
    with pytest.raises(MoveRefused) as e:
        move_to(h, {"shoulder_pan": 1.0})
    assert "torque" in str(e.value).lower()
    assert h.sent == []


def test_move_to_refuses_when_a_commanded_joint_has_no_current_reading():
    """read_joints_deg drops a joint whose .pos was missing from the
    observation, which this rig's UART does intermittently. Ramping the rest
    would command a partial move and report it as complete."""
    h = _movable_handle({"shoulder_pan": 0.0})
    h.joint_limits_deg = {"shoulder_pan": (-180.0, 180.0),
                          "wrist_flex": (-180.0, 180.0)}
    h.read_joints_deg.return_value = {"shoulder_pan": 0.0}  # wrist_flex dropped

    with pytest.raises(MoveRefused) as e:
        move_to(h, {"shoulder_pan": 1.0, "wrist_flex": 1.0})

    assert "wrist_flex" in str(e.value)
    assert h.sent == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/odesha/haller_ws/hmi/backend && source /home/odesha/venvs/haller-hmi/bin/activate && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio tests/test_motion.py -k "move_to or home_refuses"
```

Expected: FAIL — `ImportError: cannot import name 'move_to' from 'haller_hmi.motion'`.

- [ ] **Step 3: Implement the shared policy**

Append to `hmi/backend/haller_hmi/motion.py`, and add `from .safety import ModeError, check_move_size, clamp_joint_goal, plan_ramp` to its imports:

```python
def move_to(handle, goal_deg: dict[str, float]) -> dict[str, float]:
    """Ramp `handle` to `goal_deg`, or refuse if any joint moves too far.

    Shared by ArmHandle and SimArmHandle so the two cannot diverge — that
    duplication is what let the 2026-08-01 defect exist in both.
    """
    handle.guard.assert_manual()
    motion = handle.motion
    arm_id = getattr(handle.config, "id", "?")

    if not handle.torque_enabled:
        raise MoveRefused(
            f"arm {arm_id!r} has torque disabled; enable it before a discrete move"
        )

    clamped = clamp_joint_goal(goal_deg, handle.joint_limits_deg)
    current = handle.read_joints_deg()

    # read_joints_deg omits any joint whose .pos was missing from the
    # observation — it tolerates the intermittent UART failures this rig has.
    # Without this guard, plan_ramp would silently drop those joints from every
    # waypoint while move_to still returned them in `clamped`, so the route
    # would report a partial move as complete. Refuse instead.
    unmeasured = sorted(set(clamped) - set(current))
    if unmeasured:
        raise MoveRefused(
            f"move refused on arm {arm_id!r}: no current position for "
            f"{', '.join(unmeasured)}. Commanding the rest would be a partial "
            "move reported as complete. Retry the move."
        )

    oversize = check_move_size(current, clamped, motion.large_move_deg)
    if oversize:
        detail = ", ".join(
            f"{joint} {delta:+.1f}°" for joint, delta in sorted(oversize.items())
        )
        raise MoveRefused(
            f"move refused on arm {arm_id!r}: {detail} exceeds the "
            f"{motion.large_move_deg:.0f}° limit. Jog the arm closer by hand first."
        )

    waypoints = plan_ramp(current, clamped, motion.max_speed_deg_s, motion.ramp_hz)
    if waypoints:
        handle.executor.run(waypoints, motion.ramp_hz)
    return clamped


def home(handle) -> dict[str, float]:
    """Go to the calibrated home pose (0° on every joint), bounded."""
    return move_to(handle, {joint: 0.0 for joint in handle.joint_limits_deg})
```

- [ ] **Step 4: Delete both duplicate `home()` methods**

Delete from `hmi/backend/haller_hmi/arm.py`:

```python
    def home(self) -> dict[str, float]:
        """Go to the calibrated home pose (0° on every joint)."""
        goal = {j: 0.0 for j in self.joint_limits_deg}
        return self.send_goal(goal)
```

Delete from `hmi/backend/haller_hmi/sim/arm.py`:

```python
    def home(self) -> dict[str, float]:
        goal = {j: 0.0 for j in self.joint_limits_deg}
        return self.send_goal(goal)
```

Give each handle an executor. Add to both dataclasses:

```python
    executor: "MoveExecutor" = None  # set in __post_init__
```

and to both classes:

```python
    def __post_init__(self) -> None:
        from .motion import MoveExecutor   # sim/arm.py uses: from ..motion import MoveExecutor
        self.executor = MoveExecutor(self)
```

The import is deferred to avoid a cycle: `motion.py` imports nothing from `arm.py`, but `arm.py` needs `MoveExecutor` at construction time.

- [ ] **Step 5: Run the tests to verify they pass, then the full suite**

Expected: 4 new tests PASS. Existing tests calling `handle.home()` will now fail with `AttributeError` — that is the point. Task 7 updates the routes; for this task, update any direct `handle.home()` call in a test to `motion.home(handle)`.

- [ ] **Step 6: Commit**

```bash
git add hmi/backend/haller_hmi/motion.py hmi/backend/haller_hmi/arm.py \
        hmi/backend/haller_hmi/sim/arm.py hmi/backend/tests/test_motion.py
git commit -m "feat(hmi): fold home() into shared bounded move policy

Both handles carried their own copy of home(), so the defect that drove
the right arm into the bench existed identically in the sim path. There
is now one implementation, and it refuses moves that are too large."
```

---

### Task 7: Wire the routes, reproduce the incident in sim, lock parity

**Files:**
- Modify: `hmi/backend/haller_hmi/server.py:249-256` (goal), `:279-286` (home), `:316-327` (preset)
- Create: `hmi/backend/tests/sim/test_motion_sim.py`
- Test: `hmi/backend/tests/test_routes.py`

**Interfaces:**
- Consumes: `move_to`, `home`, `MoveRefused` from Task 6.
- Produces: no new interfaces. `POST /arm/{id}/home`, `/goal` and `/preset` return 409 with the refusal detail.

- [ ] **Step 1: Write the failing tests**

Create `hmi/backend/tests/sim/test_motion_sim.py`. Copy `TINY_XML`, `ARM_JOINT_MAP` and `_make_world_and_handle` from `tests/sim/test_sim_arm_handle.py` (there is no `conftest.py` to share them), or import them from that module.

The test asserts the **policy** — that a far-from-zero pose is refused and nothing reaches the actuators. It deliberately does not drive MuJoCo dynamics to get the arm into position, because stepping the world to a settled pose makes the test slow and timing-dependent without testing anything extra. `read_joints_deg` is stubbed to report the post-calibration pose; everything else is the real handle and the real world.

```python
"""The 2026-08-01 incident, reproduced.

Home was pressed straight after recalibrating the right arm. Calibration had
just redefined 0°, the arm was parked far from it, and home() issued a single
unbounded position write. The arm slewed into the bench and stalled six servos,
which burnt the 7.4 V DC-DC.
"""
from unittest.mock import MagicMock

import pytest

from haller_hmi import motion
from haller_hmi.config import MotionConfig
from haller_hmi.motion import MoveRefused

from .test_sim_arm_handle import _make_world_and_handle


def test_home_refuses_from_a_post_calibration_pose():
    world, handle = _make_world_and_handle()
    handle.connect()
    handle.motion = MotionConfig(max_speed_deg_s=60.0, large_move_deg=30.0,
                                 ramp_hz=50.0)
    handle.enable_torque()

    # Where the right arm actually sat after its sweep, in the frame the new
    # calibration had just established. TINY_XML models shoulder_pan + gripper.
    handle.read_joints_deg = MagicMock(
        return_value={"shoulder_pan": -126.5, "gripper": 0.0}
    )
    world.write_ctrl_deg = MagicMock()

    with pytest.raises(MoveRefused) as e:
        motion.home(handle)

    assert "shoulder_pan" in str(e.value)
    world.write_ctrl_deg.assert_not_called()


def test_both_handles_share_one_home_implementation():
    """Parity guard. Fails if anyone reintroduces a per-handle home(), which is
    the structural regression that made the incident possible."""
    from haller_hmi.arm import ArmHandle
    from haller_hmi.sim.arm import SimArmHandle

    assert not hasattr(ArmHandle, "home")
    assert not hasattr(SimArmHandle, "home")
```

Append to `hmi/backend/tests/test_routes.py`. The fixture is named **`app_with_mocks`** (line 8) and it returns a `TestClient` — there is no fixture called `client`:

```python
def test_home_route_returns_409_when_the_move_is_too_large(app_with_mocks, monkeypatch):
    from haller_hmi import motion
    from haller_hmi.motion import MoveRefused

    def _refuse(handle):
        raise MoveRefused("move refused on arm 'right': shoulder_pan +126.5° "
                          "exceeds the 30° limit. Jog the arm closer by hand first.")

    monkeypatch.setattr(motion, "home", _refuse)
    r = app_with_mocks.post("/arm/right/home")
    assert r.status_code == 409
    assert "shoulder_pan" in r.json()["detail"]
```

Note: `server.py` imports the module as `motion_policy`, so patching the attribute on the `haller_hmi.motion` module object (as above) is what takes effect — do not patch `haller_hmi.server.motion_policy.home` by string path, since both names refer to the same module object and the module-level patch is clearer.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/odesha/haller_ws/hmi/backend && source /home/odesha/venvs/haller-hmi/bin/activate && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio \
  tests/sim/test_motion_sim.py tests/test_routes.py -k "home or share_one_home"
```

Expected: FAIL — the route still returns 200 because it calls the deleted `handle.home()`.

- [ ] **Step 3: Rewire the three routes**

In `hmi/backend/haller_hmi/server.py`, add to the imports:

```python
from . import motion as motion_policy
from .motion import MoveRefused
```

Replace the three route bodies:

```python
@app.post("/arm/{arm_id}/goal")
async def post_arm_goal(arm_id: str, body: dict[str, float]):
    handle = _arm_or_404(arm_id)
    try:
        clamped = motion_policy.move_to(handle, body)
    except ModeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except MoveRefused as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "sent": clamped}


@app.post("/arm/{arm_id}/home")
async def post_arm_home(arm_id: str):
    handle = _arm_or_404(arm_id)
    try:
        sent = motion_policy.home(handle)
    except ModeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except MoveRefused as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "sent": sent}


@app.post("/arm/{arm_id}/preset")
async def post_arm_preset(arm_id: str, body: PresetBody):
    handle = _arm_or_404(arm_id)
    try:
        goal = presets.get(body.name, arm_id)
    except PresetNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        clamped = motion_policy.move_to(handle, goal)
    except ModeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except MoveRefused as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "sent": clamped}
```

Add executor cancellation to the E-STOP route (`server.py:371`), inside the existing loop over handles:

```python
    for handle in arms.values():
        handle.executor.cancel()
        handle.disable_torque()
        handle.guard.set(Mode.STOP)
```

- [ ] **Step 4: Run the tests to verify they pass**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Run the full suite and typecheck the frontend**

```bash
cd /home/odesha/haller_ws/hmi/backend && source /home/odesha/venvs/haller-hmi/bin/activate && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p pytest_asyncio.plugin -p anyio
cd /home/odesha/haller_ws/hmi/frontend && npx tsc --noEmit && pnpm test
```

Expected: backend **~296 passed**, frontend 101 passed and typecheck clean. The frontend is untouched by this plan; run it to confirm the 409 shape did not break a client expectation.

- [ ] **Step 6: Commit**

```bash
git add hmi/backend/haller_hmi/server.py hmi/backend/tests/
git commit -m "feat(hmi): refuse oversized discrete moves at the arm routes

Home, goal and preset now go through the shared bounded move policy and
return 409 naming the offending joints. E-STOP cancels any ramp in
flight. Adds the sim reproduction of the 2026-08-01 collision and a
parity test that fails if home() is ever duplicated back onto a handle."
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §4.1 shared primitives | 1 |
| §4.2 `MoveExecutor` + guard-based cancellation | 5, and E-STOP wiring in 7 |
| §4.3 shared `home()`, duplicates deleted | 6 |
| §4.4 non-blocking ramp | 5 (background thread), 7 (routes) |
| §5.1 streaming step cap | 4 |
| §5.1 torque side effect removed | 3 |
| §5.2 discrete refuse/ramp | 6 |
| §6 config | 2 |
| §7 error handling → 409 | 6 (raises), 7 (maps) |
| §8 test 1 incident reproduced | 7 |
| §8 test 2 ramp bounds | 1 |
| §8 test 3 E-STOP mid-ramp | 5 |
| §8 test 4 streaming outlier | 4 |
| §8 test 5 no silent energize | 3 |
| §8 test 6 parity | 7 |

No spec requirement is unimplemented.

**Deviations from the spec, and why:**
- The spec said the discrete path reads current position fresh, and implied the streaming path would too. Task 4 limits the streaming path against the *last commanded* position instead. A fresh read per tick is a serial round-trip at 60 Hz on an arm already showing intermittent UART failures. The discrete path still reads fresh, as specified.
- The spec named `ConflictError` for refusals. That class lives in `calibration.py`, so using it would make motion depend on calibration. Task 5 defines `MoveRefused` in `motion.py` instead; routes map it to the same 409.

**Type consistency:** `MotionConfig` field names are identical in Tasks 2, 4, 6, 7. `read_joints_deg() -> dict[str, float]` matches both handles' existing signatures. `plan_ramp` returns `list[dict[str, float]]`, which is exactly what `MoveExecutor.run` consumes.

**Test infrastructure, verified against the tree rather than assumed:**
- There is **no `conftest.py`** anywhere under `hmi/backend/tests/`. Sim tests build their own world via the module-level helper `_make_world_and_handle()` in `tests/sim/test_sim_arm_handle.py:46`.
- `TINY_XML` models only `right_Rotation` and `right_Jaw`, so a TINY_XML-backed handle exposes exactly `shoulder_pan` and `gripper`. Any other joint name in such a test silently does nothing, because `send_goal` filters on `LEROBOT_TO_MJCF` membership.
- The route fixture in `tests/test_routes.py` is `app_with_mocks`, not `client`.

**Known gap left deliberately:** the frontend still shows a plain Home button and will surface the 409 as a generic error. Improving that copy belongs with spec B's UI work, not here.
