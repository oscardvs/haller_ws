# SO-101 MuJoCo sim trio — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three HMI-driven MuJoCo simulations of the SO-101 arms — solo follower, bimanual follower pair, and leader+follower — that drop into the existing HMI via a `SimArmHandle` (interface-compatible with the real `ArmHandle`) and a `SimCamera` (compatible with the existing MJPEG plumbing).

**Architecture:** A single `MuJoCoWorld` per HMI process owns one `mjModel`+`mjData` and runs a 500 Hz physics stepper thread. `SimArmHandle` writes actuator `ctrl` and reads `qpos` for one named arm. `SimCamera` runs an offscreen `mujoco.Renderer` and feeds JPEGs through the existing camera HTTP routes. Three new `config.*-sim.yaml` presets compose the scenarios; `run_hmi.sh --config <path>` selects one. A small `SimLeaderTeleop` handles the mouse-drag and dataset-replay leader modes; the real-leader → sim-follower mode is a config-only override that reuses the existing `TeleopSession`.

**Tech Stack:** `mujoco>=3.2` (pure-Python wheel, EGL/OSMesa for headless render), existing LeRobot stack, FastAPI, PyYAML.

**Spec:** `docs/superpowers/specs/2026-05-23-so101-mujoco-sim-trio-design.md`

---

## File Structure

| Path | Created/Modified | Responsibility |
| --- | --- | --- |
| `hmi/backend/pyproject.toml` | Modify | Add `mujoco>=3.2` dep |
| `sim/assets/so101/` | Create dir | Vendored SO-101 (trs_so_arm100) MJCF + meshes + LICENSE + CHANGELOG |
| `sim/assets/scenes/workbench.xml` | Create | MJCF snippet: flat workbench plane |
| `sim/assets/scenes/cube.xml` | Create | MJCF snippet: a 4 cm cube body |
| `hmi/backend/haller_hmi/arm.py` | Modify | Add `ArmHandle.read_joints_deg()` |
| `hmi/backend/haller_hmi/teleop.py` | Modify | Tick loop calls `read_joints_deg()` + `send_goal()` instead of reaching into `.robot` |
| `hmi/backend/haller_hmi/config.py` | Modify | `ArmConfig.source` + `sim_arm_name`; `CameraConfig` `sim_camera`/`mjcf_camera`; `SimLeaderConfig`; `Config.sim_leader` |
| `hmi/backend/haller_hmi/cameras.py` | Modify | `CameraManager` constructs `SimCamera` for `source: sim_camera` |
| `hmi/backend/haller_hmi/server.py` | Modify | Wire `SimLeaderTeleop` into lifespan; expose `/teleop/sim/{start,stop}` |
| `scripts/run_hmi.sh` | Modify | Accept `--config <path>` → exports `HALLER_HMI_CONFIG` |
| `hmi/backend/haller_hmi/sim/__init__.py` | Create | Package marker |
| `hmi/backend/haller_hmi/sim/world.py` | Create | `MuJoCoWorld`: model+data, stepper thread, viewer launch, per-arm ctrl/qpos lookup |
| `hmi/backend/haller_hmi/sim/builder.py` | Create | Compose solo / bimanual / leader-follower MJCFs by namespacing the SO-101 model |
| `hmi/backend/haller_hmi/sim/arm.py` | Create | `SimArmHandle`: drop-in for `ArmHandle` |
| `hmi/backend/haller_hmi/sim/camera.py` | Create | `SimCamera`: offscreen `mujoco.Renderer` → JPEG, MJPEG-plumbing-compatible |
| `hmi/backend/haller_hmi/sim/sources.py` | Create | `LeaderSource` protocol + `MouseDragSource` + `DatasetReplaySource` |
| `hmi/backend/haller_hmi/sim/teleop.py` | Create | `SimLeaderTeleop` session (mouse / replay leader modes) |
| `hmi/backend/config.solo-sim.yaml` | Create | One sim follower, workbench + cube |
| `hmi/backend/config.bimanual-sim.yaml` | Create | Two sim followers, workbench + 2 cubes |
| `hmi/backend/config.leader-follower-sim.yaml` | Create | Two sim arms (one leader, one follower), workbench |
| `hmi/backend/tests/sim/__init__.py` | Create | Test package marker |
| `hmi/backend/tests/sim/test_builder.py` | Create | MJCF composition tests |
| `hmi/backend/tests/sim/test_world.py` | Create | World init / stepper / ctrl writes |
| `hmi/backend/tests/sim/test_sim_arm_handle.py` | Create | SimArmHandle interface contract |
| `hmi/backend/tests/sim/test_sim_camera.py` | Create | Headless render → JPEG |
| `hmi/backend/tests/sim/test_sources.py` | Create | LeaderSource impls |
| `hmi/backend/tests/sim/test_sim_teleop.py` | Create | SimLeaderTeleop loop |
| `hmi/backend/tests/test_arm.py` | Modify | Add `read_joints_deg()` test |
| `docs/setup/sim.md` | Create | Install + run instructions |
| `hmi/README.md` | Modify | Link the three sim configs + docs |
| `README.md` | Modify | One-line status bullet |

---

## Task 1: Add `mujoco` dependency and verify install

**Files:**
- Modify: `hmi/backend/pyproject.toml:14-30`

- [ ] **Step 1: Add `mujoco>=3.2` to the deps list**

In `hmi/backend/pyproject.toml`, find the `dependencies = [` block and add one line:

```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pyyaml>=6.0",
    "pandas>=2.2.2",
    "numexpr>=2.10",
    "bottleneck>=1.4",
    "numpy>=1.26",
    "lerobot[feetech]>=0.5,<0.6",
    "mujoco>=3.2",
]
```

- [ ] **Step 2: Install into the HMI venv**

Run:
```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
pip install -e hmi/backend
```

Expected: `Successfully installed mujoco-3.x.x ...` (no errors).

- [ ] **Step 3: Verify the import works headless**

Run:
```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
MUJOCO_GL=egl python -c "import mujoco; print(mujoco.__version__)"
```

Expected: prints `3.x.x` and exits 0. If EGL is missing, fall back to `MUJOCO_GL=osmesa` and confirm that works instead.

- [ ] **Step 4: Commit**

```bash
git add hmi/backend/pyproject.toml
git diff --cached --name-only   # verify only that file is staged
git commit -m "feat(hmi/backend): add mujoco>=3.2 dependency"
```

---

## Task 2: `ArmHandle.read_joints_deg()` — interface tightening (real side)

**Files:**
- Modify: `hmi/backend/haller_hmi/arm.py:108-122` (insert after `disable_torque`/`enable_torque`)
- Modify: `hmi/backend/tests/test_arm.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `hmi/backend/tests/test_arm.py`:

```python
def test_read_joints_deg_strips_pos_suffix(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.robot.get_observation.return_value = {
        "shoulder_pan.pos": 12.5,
        "elbow_flex.pos": -30.0,
        "gripper.pos": 0.0,
        # lerobot also emits non-joint keys (e.g. ".vel"); read_joints_deg must ignore them
        "shoulder_pan.vel": 999.0,
    }
    joints = handle.read_joints_deg()
    assert joints == {"shoulder_pan": 12.5, "elbow_flex": -30.0, "gripper": 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
pytest hmi/backend/tests/test_arm.py::test_read_joints_deg_strips_pos_suffix -v
```

Expected: FAIL with `AttributeError: 'ArmHandle' object has no attribute 'read_joints_deg'`.

- [ ] **Step 3: Add the method**

In `hmi/backend/haller_hmi/arm.py`, add to `ArmHandle` (after `enable_torque`, before `state_snapshot`):

```python
    def read_joints_deg(self) -> dict[str, float]:
        """Latest joint positions in degrees, keyed by joint name (no `.pos` suffix).

        Filters lerobot's observation dict to only `<joint>.pos` entries that
        belong to a known joint, and strips the suffix so callers don't need to.
        """
        assert self.robot is not None
        obs = self.robot.get_observation()
        out: dict[str, float] = {}
        for joint in self.joint_limits_deg:
            key = f"{joint}.pos"
            if key in obs:
                out[joint] = float(obs[key])
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest hmi/backend/tests/test_arm.py::test_read_joints_deg_strips_pos_suffix -v
```

Expected: PASS.

- [ ] **Step 5: Run the full arm test file to confirm no regression**

Run:
```bash
pytest hmi/backend/tests/test_arm.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add hmi/backend/haller_hmi/arm.py hmi/backend/tests/test_arm.py
git diff --cached --name-only
git commit -m "refactor(hmi/backend): ArmHandle.read_joints_deg() — push raw lerobot read behind the interface"
```

---

## Task 3: Refactor `teleop.py` tick loop to use `read_joints_deg()` + `send_goal()`

Goal: the leader↔follower tick loop currently calls `leader.robot.get_observation()` and `follower.robot.send_action(...)` directly. Replace both with handle-level calls so the loop works against any `ArmHandle`-shaped object (including the upcoming `SimArmHandle`). This preserves observed behavior — same clamp, same shape, same cadence.

**Files:**
- Modify: `hmi/backend/haller_hmi/teleop.py:147-160` (the tick body)

- [ ] **Step 1: Write a failing test asserting the loop uses the new method**

Append to `hmi/backend/tests/test_arm.py` (or create `hmi/backend/tests/test_teleop_loop.py` — same file is fine for now):

```python
def test_teleop_loop_uses_read_joints_deg_and_send_goal(monkeypatch):
    """The teleop tick must go through handle.read_joints_deg() + handle.send_goal(),
    not reach into handle.robot directly. This is what makes SimArmHandle a drop-in."""
    from unittest.mock import MagicMock
    from haller_hmi.teleop import TeleopSession
    from haller_hmi.config import ArmConfig
    from haller_hmi.arm import ArmManager
    import time

    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "haller_hmi.arm.ArmHandle._load_joint_limits",
        lambda self: {"shoulder_pan": (-120.0, 120.0), "gripper": (0.0, 100.0)},
    )
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )

    cfg_l = ArmConfig(id="left",  model="so101_follower", port="/dev/null", calibration_id="x")
    cfg_r = ArmConfig(id="right", model="so101_follower", port="/dev/null", calibration_id="y")
    mgr = ArmManager([cfg_l, cfg_r])
    mgr.connect_all()

    leader = mgr["left"]
    follower = mgr["right"]

    # Stub the two interface methods we expect the loop to call.
    leader.read_joints_deg = MagicMock(return_value={"shoulder_pan": 42.0, "gripper": 50.0})
    follower.send_goal = MagicMock(return_value={"shoulder_pan": 42.0, "gripper": 50.0})
    # Also disable any auto-enable side-effects.
    leader.disable_torque = MagicMock()
    follower.enable_torque = MagicMock()

    session = TeleopSession(mgr)
    session.start(leader_id="left", follower_id="right", hz=120.0)
    time.sleep(0.1)  # let a few ticks happen
    session.stop()

    assert leader.read_joints_deg.called, "loop must call leader.read_joints_deg()"
    assert follower.send_goal.called, "loop must call follower.send_goal()"
    last_call = follower.send_goal.call_args
    assert last_call.args[0] == {"shoulder_pan": 42.0, "gripper": 50.0}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pytest hmi/backend/tests/test_arm.py::test_teleop_loop_uses_read_joints_deg_and_send_goal -v
```

Expected: FAIL (the loop currently calls `leader.robot.get_observation()` and `follower.robot.send_action(...)`, not the handle methods).

- [ ] **Step 3: Refactor the loop body**

In `hmi/backend/haller_hmi/teleop.py`, replace the body of `_loop` (lines around 146–160) so the per-tick read+write goes through the handle interface:

Replace this block:

```python
            try:
                obs = leader.robot.get_observation() if leader.robot else {}
                # Clamp every joint to the follower's calibrated range and pack
                # into lerobot's "<joint>.pos" action shape.
                action = {}
                for joint, (lo, hi) in follower.joint_limits_deg.items():
                    raw = float(obs.get(f"{joint}.pos", 0.0))
                    action[f"{joint}.pos"] = max(lo, min(hi, raw))
                if follower.robot is not None:
                    follower.robot.send_action(action)
                with self._lock:
                    self._state.tick_count += 1
                    self._state.last_error = None
```

with:

```python
            try:
                # Go through the ArmHandle interface (works for real + sim arms).
                leader_joints = leader.read_joints_deg()
                # Restrict to joints the follower knows about — send_goal handles
                # clamping against the follower's own limits.
                goal = {j: leader_joints[j]
                        for j in follower.joint_limits_deg
                        if j in leader_joints}
                # send_goal asserts MANUAL mode; the leader is in STOP, follower in MANUAL,
                # so this hits the follower's path correctly. Override the leader's STOP
                # guard for the duration of the read by going through .read_joints_deg(),
                # which doesn't go through the mode guard.
                follower.send_goal(goal)
                with self._lock:
                    self._state.tick_count += 1
                    self._state.last_error = None
```

Also delete the now-stale "lerobot's mode-guard goes via ArmHandle.send_goal, but the teleop loop bypasses ModeError…" paragraph from the module docstring at the top of `hmi/backend/haller_hmi/teleop.py` — we now go through `send_goal` properly.

- [ ] **Step 4: Run the new test plus the full teleop-adjacent suite**

Run:
```bash
pytest hmi/backend/tests/test_arm.py hmi/backend/tests/test_routes.py hmi/backend/tests/test_telemetry.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/teleop.py hmi/backend/tests/test_arm.py
git diff --cached --name-only
git commit -m "refactor(hmi/backend): teleop tick uses ArmHandle interface, not raw .robot"
```

---

## Task 4: Vendor SO-101 MJCF assets + smoke-load test

**Files:**
- Create: `sim/assets/so101/` (dir, plus contents from upstream)
- Create: `sim/assets/so101/LICENSE`
- Create: `sim/assets/so101/CHANGELOG.md`
- Create: `sim/assets/so101/README.md`
- Create: `hmi/backend/tests/sim/__init__.py`
- Create: `hmi/backend/tests/sim/test_assets.py`

- [ ] **Step 1: Verify the target dir does not exist yet**

Run:
```bash
ls sim 2>/dev/null || echo "no sim dir yet (good)"
```

Expected: `no sim dir yet (good)`.

- [ ] **Step 2: Vendor `trs_so_arm100` from `google-deepmind/mujoco_menagerie`**

The upstream is the canonical maintained MuJoCo model for the SO-ARM100 family (SO-101 shares kinematics; gripper geometry may differ — we'll note this in the CHANGELOG and refine later if it matters).

Run:
```bash
mkdir -p sim/assets/so101
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$TMP/menagerie"
cp -r "$TMP/menagerie/trs_so_arm100/." sim/assets/so101/
UPSTREAM_SHA=$(git -C "$TMP/menagerie" rev-parse HEAD)
echo "vendored from mujoco_menagerie@$UPSTREAM_SHA"
ls sim/assets/so101/
```

Expected: lists `so_arm100.xml` (or similarly named scene file), an `assets/` subdir with meshes, and a `LICENSE`-style file. Note the `UPSTREAM_SHA` value — needed in Step 3.

- [ ] **Step 3: Write `sim/assets/so101/CHANGELOG.md`**

Create `sim/assets/so101/CHANGELOG.md` with:

```markdown
# SO-101 MJCF — vendor log

## Source

Vendored from [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100).

- Upstream commit: `<UPSTREAM_SHA from vendor step>`
- Date pulled: 2026-05-23

## SO-100 vs SO-101

The upstream model is `trs_so_arm100` (SO-100). SO-101 differs from SO-100
primarily in the gripper assembly; the 6-DOF arm chain is identical. We
ship the SO-100 MJCF unchanged for the arm; if gripper visuals/contact
matter for a future task, replace `assets/SO_ARM100_Gripper_*.stl` with
SO-101 meshes (TheRobotStudio/SO-ARM100 repo).

## Local edits

(none yet — record any future hand-edits here with rationale and a `git diff`-style summary)

## Refresh procedure

```bash
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$TMP/m"
rsync -a --delete --exclude='CHANGELOG.md' --exclude='README.md' \
      "$TMP/m/trs_so_arm100/" sim/assets/so101/
git -C "$TMP/m" rev-parse HEAD   # paste into the entry above
```
```

- [ ] **Step 4: Write `sim/assets/so101/README.md`**

Create `sim/assets/so101/README.md`:

```markdown
# SO-101 MuJoCo assets

Vendored MJCF + meshes for the SO-101 arm. See `CHANGELOG.md` for the upstream
source and refresh procedure.

Consumed by `hmi/backend/haller_hmi/sim/`.

License: see `LICENSE` (Apache-2.0 from upstream mujoco_menagerie).
```

If the vendor step (Step 2) copied a license file under a different name (e.g. `LICENSE`, `LICENSE.txt`), keep it as-is — do not rename.

- [ ] **Step 5: Write the smoke-load test**

Create `hmi/backend/tests/sim/__init__.py` (empty file).

Create `hmi/backend/tests/sim/test_assets.py`:

```python
"""The vendored SO-101 MJCF loads cleanly under MuJoCo."""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import pytest

# Headless render backend — set before any MuJoCo OpenGL is touched.
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[4]
SO101_DIR = REPO_ROOT / "sim" / "assets" / "so101"


def _find_scene_xml() -> Path:
    # mujoco_menagerie folders ship a "scene.xml" and a robot-named xml; prefer scene.
    for name in ("scene.xml", "so_arm100.xml", "so_arm100_scene.xml"):
        cand = SO101_DIR / name
        if cand.exists():
            return cand
    xmls = list(SO101_DIR.glob("*.xml"))
    if not xmls:
        pytest.skip("no MJCF in sim/assets/so101/")
    return xmls[0]


def test_so101_mjcf_loads():
    xml = _find_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.nq >= 6, f"expected at least 6 joints, got {model.nq}"
    assert model.nu >= 6, f"expected at least 6 actuators, got {model.nu}"
```

- [ ] **Step 6: Run the smoke test**

Run:
```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_assets.py -v
```

Expected: PASS. If `MUJOCO_GL=egl` errors, retry with `MUJOCO_GL=osmesa`.

- [ ] **Step 7: Commit (explicit paths only)**

Stage the vendored tree, the changelog/readme, and the test — **explicit paths, never `git add -A`**:

```bash
git add sim/assets/so101 \
        hmi/backend/tests/sim/__init__.py \
        hmi/backend/tests/sim/test_assets.py
git diff --cached --name-only | head -30   # spot-check
git commit -m "feat(hmi/sim): vendor SO-101 MJCF (trs_so_arm100) + smoke-load test"
```

---

## Task 5: `MuJoCoWorld` — model+data, stepper thread, per-arm ctrl/qpos lookup

**Files:**
- Create: `hmi/backend/haller_hmi/sim/__init__.py`
- Create: `hmi/backend/haller_hmi/sim/world.py`
- Create: `hmi/backend/tests/sim/test_world.py`

- [ ] **Step 1: Write the failing test**

Create `hmi/backend/haller_hmi/sim/__init__.py` (empty).

Create `hmi/backend/tests/sim/test_world.py`:

```python
"""MuJoCoWorld owns model+data, runs a stepper, and exposes per-arm ctrl/qpos."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.sim.world import MuJoCoWorld

# Minimal 2-DOF MJCF with named joints + actuators we can address by "arm".
TINY_XML = """
<mujoco>
  <compiler angle="degree"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="right_base" pos="0 0 0">
      <joint name="right_shoulder_pan" type="hinge" axis="0 0 1" range="-180 180" limited="true"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0  0.2 0 0"/>
      <body name="right_link2" pos="0.2 0 0">
        <joint name="right_gripper" type="hinge" axis="0 1 0" range="-90 90" limited="true"/>
        <geom type="capsule" size="0.02 0.1" fromto="0 0 0  0.2 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="right_shoulder_pan_act" joint="right_shoulder_pan" kp="50" ctrlrange="-180 180"/>
    <position name="right_gripper_act"      joint="right_gripper"      kp="50" ctrlrange="-90 90"/>
  </actuator>
</mujoco>
"""

ARM_JOINT_MAP = {"right": ["right_shoulder_pan", "right_gripper"]}


def test_world_loads_and_starts_stepper():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    world.start()
    try:
        time.sleep(0.1)
        assert world.is_running()
        assert world.tick_count() > 0
    finally:
        world.stop()


def test_write_ctrl_moves_qpos_toward_goal():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    world.start()
    try:
        # 30 deg goal on shoulder_pan
        world.write_ctrl_deg("right", {"right_shoulder_pan": 30.0})
        time.sleep(0.5)  # let the actuator drive it
        q = world.read_qpos_deg("right")
        assert abs(q["right_shoulder_pan"] - 30.0) < 5.0, f"got {q!r}"
    finally:
        world.stop()


def test_unknown_arm_raises():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    with pytest.raises(KeyError):
        world.write_ctrl_deg("left", {"shoulder_pan": 0.0})


def test_disable_actuator_kp_zeros_gain():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    world.start()
    try:
        world.set_arm_torque("right", enabled=False)
        # All actuators for this arm should have kp = 0
        for joint in ARM_JOINT_MAP["right"]:
            assert world.actuator_kp_for_joint(joint) == 0.0
        world.set_arm_torque("right", enabled=True)
        for joint in ARM_JOINT_MAP["right"]:
            assert world.actuator_kp_for_joint(joint) > 0.0
    finally:
        world.stop()
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_world.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'haller_hmi.sim.world'`.

- [ ] **Step 3: Implement `MuJoCoWorld`**

Create `hmi/backend/haller_hmi/sim/world.py`:

```python
"""MuJoCoWorld: owns one mjModel+mjData, runs a physics stepper, exposes per-arm ctrl/qpos.

A single world is shared by every SimArmHandle and SimCamera in the HMI process.
Stepper runs in a daemon thread at the model's configured timestep.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import mujoco

logger = logging.getLogger(__name__)


@dataclass
class _ArmIndex:
    joint_names: list[str]
    qpos_addr: dict[str, int]          # joint name -> qpos index
    actuator_id: dict[str, int]        # joint name -> actuator id
    default_kp: dict[str, float]       # joint name -> kp at world-construct time


class MuJoCoWorld:
    def __init__(self, mjcf_xml: str, arm_joint_map: dict[str, list[str]]):
        """`arm_joint_map` maps an arm id (e.g. "left", "right") to the list of
        joint names that arm owns in the MJCF. Joint names should be the
        already-namespaced names (e.g. "right_shoulder_pan")."""
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.data = mujoco.MjData(self.model)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0

        self._arms: dict[str, _ArmIndex] = {}
        for arm_id, joint_names in arm_joint_map.items():
            qpos_addr: dict[str, int] = {}
            actuator_id: dict[str, int] = {}
            default_kp: dict[str, float] = {}
            for jname in joint_names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    raise ValueError(f"joint {jname!r} not found in MJCF")
                qpos_addr[jname] = int(self.model.jnt_qposadr[jid])
                # find the actuator that drives this joint
                act_id = None
                for a in range(self.model.nu):
                    if int(self.model.actuator_trnid[a, 0]) == jid:
                        act_id = a
                        break
                if act_id is None:
                    raise ValueError(f"no actuator drives joint {jname!r}")
                actuator_id[jname] = act_id
                # gainprm[0] for `position` actuators is kp
                default_kp[jname] = float(self.model.actuator_gainprm[act_id, 0])
            self._arms[arm_id] = _ArmIndex(
                joint_names=list(joint_names),
                qpos_addr=qpos_addr,
                actuator_id=actuator_id,
                default_kp=default_kp,
            )

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._paused.clear()
        self._thread = threading.Thread(
            target=self._step_loop, name="MuJoCoWorld-stepper", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def tick_count(self) -> int:
        return self._tick_count

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    # ---- per-arm I/O ----

    def write_ctrl_deg(self, arm_id: str, goal_deg: dict[str, float]) -> None:
        arm = self._arms[arm_id]
        with self._lock:
            for joint, deg in goal_deg.items():
                if joint not in arm.actuator_id:
                    continue
                self.data.ctrl[arm.actuator_id[joint]] = math.radians(deg)

    def read_qpos_deg(self, arm_id: str) -> dict[str, float]:
        arm = self._arms[arm_id]
        with self._lock:
            return {
                jname: math.degrees(float(self.data.qpos[addr]))
                for jname, addr in arm.qpos_addr.items()
            }

    def joint_range_deg(self, arm_id: str, joint: str) -> tuple[float, float]:
        arm = self._arms[arm_id]
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0 or joint not in arm.joint_names:
            raise KeyError(joint)
        lo, hi = self.model.jnt_range[jid]
        # MuJoCo jnt_range is in the joint's native units; for hinge joints that's radians.
        return math.degrees(float(lo)), math.degrees(float(hi))

    def set_arm_torque(self, arm_id: str, enabled: bool) -> None:
        arm = self._arms[arm_id]
        with self._lock:
            for joint, act_id in arm.actuator_id.items():
                self.model.actuator_gainprm[act_id, 0] = (
                    arm.default_kp[joint] if enabled else 0.0
                )

    def actuator_kp_for_joint(self, joint: str) -> float:
        for arm in self._arms.values():
            if joint in arm.actuator_id:
                return float(self.model.actuator_gainprm[arm.actuator_id[joint], 0])
        raise KeyError(joint)

    # ---- stepper ----

    def _step_loop(self) -> None:
        timestep = float(self.model.opt.timestep)
        next_t = time.perf_counter()
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.01)
                next_t = time.perf_counter()
                continue
            with self._lock:
                mujoco.mj_step(self.model, self.data)
                self._tick_count += 1
            next_t += timestep
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.perf_counter()  # we're behind; resync
```

- [ ] **Step 4: Run the world tests**

Run:
```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_world.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/sim/__init__.py \
        hmi/backend/haller_hmi/sim/world.py \
        hmi/backend/tests/sim/test_world.py
git diff --cached --name-only
git commit -m "feat(hmi/sim): MuJoCoWorld — model+data, stepper thread, per-arm ctrl/qpos"
```

---

## Task 6: `builder.py` — compose the three scenes with namespaced arms

The vendored SO-101 MJCF uses unprefixed joint/body names. For bimanual and leader+follower scenes we need two instances side-by-side. We do this by string-prefixing every name in the SO-101 MJCF with an arm prefix (`left_` / `right_`) and composing into a parent scene that also has a workbench, optional cubes, and an overhead camera.

**Files:**
- Create: `sim/assets/scenes/workbench.xml`
- Create: `sim/assets/scenes/cube.xml`
- Create: `hmi/backend/haller_hmi/sim/builder.py`
- Create: `hmi/backend/tests/sim/test_builder.py`

- [ ] **Step 1: Write workbench + cube MJCF snippets**

Create `sim/assets/scenes/workbench.xml`:

```xml
<mujocoinclude>
  <!-- Flat workbench: a thin box at z=0 with a neutral grey surface. -->
  <worldbody>
    <geom name="workbench"
          type="box"
          size="0.4 0.4 0.01"
          pos="0 0 -0.01"
          rgba="0.35 0.35 0.35 1"
          friction="1 0.005 0.0001"/>
    <light name="overhead_light" pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
  </worldbody>
</mujocoinclude>
```

Create `sim/assets/scenes/cube.xml`:

```xml
<mujocoinclude>
  <!-- One 4 cm cube. The caller must wrap this in a <body name="..."> to namespace it. -->
  <worldbody>
    <body name="cube" pos="0.25 0 0.05">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.05"
            rgba="0.85 0.2 0.2 1"
            friction="1 0.005 0.0001"/>
    </body>
  </worldbody>
</mujocoinclude>
```

- [ ] **Step 2: Write the failing builder tests**

Create `hmi/backend/tests/sim/test_builder.py`:

```python
"""builder.py composes solo/bimanual/leader-follower MJCFs by namespacing arms."""
from __future__ import annotations

import os
import re

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

from haller_hmi.sim.builder import build_scene, SO101_JOINTS

PRESETS = {
    "solo":              {"arms": ["right"], "cubes": 1},
    "bimanual":          {"arms": ["left", "right"], "cubes": 2},
    "leader_follower":   {"arms": ["left", "right"], "cubes": 0},
}


@pytest.mark.parametrize("preset", list(PRESETS.keys()))
def test_build_scene_loads_and_has_expected_arms(preset):
    cfg = PRESETS[preset]
    mjcf_xml, arm_joint_map = build_scene(arms=cfg["arms"], cubes=cfg["cubes"])
    model = mujoco.MjModel.from_xml_string(mjcf_xml)

    # Each arm contributes len(SO101_JOINTS) named joints, prefixed.
    for arm_id in cfg["arms"]:
        for j in SO101_JOINTS:
            qualified = f"{arm_id}_{j}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, qualified)
            assert jid >= 0, f"missing joint {qualified} in {preset}"
        assert arm_joint_map[arm_id] == [f"{arm_id}_{j}" for j in SO101_JOINTS]


def test_build_scene_has_overhead_camera():
    mjcf_xml, _ = build_scene(arms=["right"], cubes=1)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    assert cam_id >= 0


def test_build_scene_cubes_have_unique_names():
    mjcf_xml, _ = build_scene(arms=["left", "right"], cubes=2)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    names = []
    for i in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n and re.match(r"cube_\d+", n):
            names.append(n)
    assert sorted(names) == ["cube_0", "cube_1"]
```

- [ ] **Step 3: Run to confirm failure**

Run:
```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_builder.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'haller_hmi.sim.builder'`.

- [ ] **Step 4: Implement `builder.py`**

Create `hmi/backend/haller_hmi/sim/builder.py`:

```python
"""Compose SO-101 sim scenes by namespacing the vendored arm MJCF.

Strategy: parse the SO-101 MJCF as XML, prefix every `name="..."` attribute and
every reference to those names (`joint="..."`, `body="..."`, etc.) with an arm
prefix, then assemble the prefixed arm(s) plus workbench/cubes/overhead camera
into one parent MJCF. Keeps us off `dm_control` (heavy dep) while staying
deterministic and easy to inspect.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

# Canonical SO-101 joint names (LeRobot convention).
SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

REPO_ROOT = Path(__file__).resolve().parents[4]
SO101_DIR = REPO_ROOT / "sim" / "assets" / "so101"
SCENES_DIR = REPO_ROOT / "sim" / "assets" / "scenes"

# Attributes that reference a named element by name (must be prefixed if the
# referenced element was prefixed). This list covers the elements used in the
# trs_so_arm100 MJCF; if upstream adds new reference attrs, extend it.
_NAME_REF_ATTRS = {
    "joint", "body1", "body2", "site", "geom", "mesh", "material",
    "tendon", "actuator", "class", "childclass", "target",
}


def _find_arm_xml() -> Path:
    for name in ("so_arm100.xml", "so_arm100_scene.xml", "scene.xml"):
        p = SO101_DIR / name
        if p.exists():
            return p
    # Fall back to the first *.xml that has an <actuator> section.
    for p in sorted(SO101_DIR.glob("*.xml")):
        if "<actuator" in p.read_text():
            return p
    raise FileNotFoundError(f"no SO-101 MJCF found in {SO101_DIR}")


def _collect_named_elements(root: ET.Element) -> set[str]:
    """Names that get prefixed: every element with a `name` attribute, EXCEPT
    top-level asset filenames (those are paths, not symbol refs)."""
    names: set[str] = set()
    for el in root.iter():
        n = el.get("name")
        if n is not None:
            names.add(n)
    return names


def _prefix_element_tree(root: ET.Element, prefix: str, names_to_prefix: set[str]) -> None:
    """In place: prefix every `name="..."` and every reference attribute whose
    value is in `names_to_prefix`."""
    for el in root.iter():
        n = el.get("name")
        if n is not None and n in names_to_prefix:
            el.set("name", f"{prefix}{n}")
        for attr in _NAME_REF_ATTRS:
            v = el.get(attr)
            if v is not None and v in names_to_prefix:
                el.set(attr, f"{prefix}{v}")


@dataclass
class _ArmSubtree:
    worldbody_inner: str
    asset_inner: str
    actuator_inner: str
    default_inner: str
    contact_inner: str
    sensor_inner: str
    tendon_inner: str
    equality_inner: str
    compiler_attrs: dict[str, str]  # mesh dir etc.; only the first arm's are used
    joint_names: list[str]


# Sections we extract from the upstream MJCF and recompose under the parent.
# We INTENTIONALLY drop <option> and <size> — the parent owns those, and MuJoCo
# errors on duplicates.
_EXTRACTED_SECTIONS = {
    "worldbody", "asset", "actuator", "default",
    "contact", "sensor", "tendon", "equality",
}


def _load_arm_subtree(prefix: str, x_offset: float) -> _ArmSubtree:
    """Parse the SO-101 MJCF, prefix every name + name-ref, return the per-section
    inner XML so the caller can recompose under a single parent <mujoco> root.
    Wraps the worldbody contents in a positioning body so multi-arm scenes don't
    overlap.
    """
    arm_path = _find_arm_xml()
    tree = ET.parse(arm_path)
    root = tree.getroot()
    names = _collect_named_elements(root)
    _prefix_element_tree(root, prefix, names)

    sections: dict[str, list[str]] = {s: [] for s in _EXTRACTED_SECTIONS}
    compiler_attrs: dict[str, str] = {}
    for child in list(root):
        if child.tag == "compiler":
            compiler_attrs = dict(child.attrib)
        elif child.tag in _EXTRACTED_SECTIONS:
            for sub in list(child):
                sections[child.tag].append(ET.tostring(sub, encoding="unicode"))

    wrapped_worldbody = (
        f'<body name="{prefix}root" pos="{x_offset} 0 0">\n'
        + "\n".join(sections["worldbody"])
        + "\n</body>"
    )

    return _ArmSubtree(
        worldbody_inner=wrapped_worldbody,
        asset_inner="\n".join(sections["asset"]),
        actuator_inner="\n".join(sections["actuator"]),
        default_inner="\n".join(sections["default"]),
        contact_inner="\n".join(sections["contact"]),
        sensor_inner="\n".join(sections["sensor"]),
        tendon_inner="\n".join(sections["tendon"]),
        equality_inner="\n".join(sections["equality"]),
        compiler_attrs=compiler_attrs,
        joint_names=[f"{prefix}{j}" for j in SO101_JOINTS],
    )


def build_scene(arms: list[str], cubes: int) -> tuple[str, dict[str, list[str]]]:
    """Compose a scene MJCF.

    `arms`: list of arm ids (e.g. ["right"] or ["left", "right"]).
    `cubes`: number of 4cm cubes on the workbench.

    Returns (mjcf_xml_string, arm_joint_map). arm_joint_map maps each arm id to
    its list of prefixed joint names — what `MuJoCoWorld` consumes.
    """
    if not arms:
        raise ValueError("scene needs at least one arm")

    # Horizontal offsets so arms don't overlap. Single arm: centered. Two arms:
    # +/- 0.20 m on x.
    if len(arms) == 1:
        offsets = [0.0]
    elif len(arms) == 2:
        offsets = [-0.20, 0.20]
    else:
        raise ValueError(f"only 1 or 2 arms supported, got {len(arms)}")

    arm_joint_map: dict[str, list[str]] = {}
    subtrees: list[_ArmSubtree] = []
    for arm_id, x in zip(arms, offsets):
        sub = _load_arm_subtree(prefix=f"{arm_id}_", x_offset=x)
        subtrees.append(sub)
        arm_joint_map[arm_id] = sub.joint_names

    # Workbench (always) + cubes.
    workbench_inner = _extract_worldbody_inner(SCENES_DIR / "workbench.xml")
    cube_chunks: list[str] = []
    cube_template = (SCENES_DIR / "cube.xml").read_text()
    cube_body_re = re.compile(r"<body\s+name=\"cube\"", re.M)
    for i in range(cubes):
        x = -0.10 + 0.10 * i
        per_cube = cube_body_re.sub(
            f'<body name="cube_{i}" pos="{x} 0 0.05"', cube_template
        )
        cube_chunks.append(_extract_worldbody_inner_from_string(per_cube))

    # <compiler> — meshdir is relative to the MJCF on disk. We're producing an
    # in-memory string, so resolve meshdir to an absolute path so meshes load.
    compiler_attrs = dict(subtrees[0].compiler_attrs) if subtrees else {}
    meshdir = compiler_attrs.get("meshdir", "assets")
    if not Path(meshdir).is_absolute():
        compiler_attrs["meshdir"] = str((SO101_DIR / meshdir).resolve())
    texturedir = compiler_attrs.get("texturedir")
    if texturedir and not Path(texturedir).is_absolute():
        compiler_attrs["texturedir"] = str((SO101_DIR / texturedir).resolve())
    compiler_attr_str = " ".join(f'{k}="{v}"' for k, v in compiler_attrs.items())

    parts: list[str] = ['<mujoco model="haller-sim">']
    if compiler_attr_str:
        parts.append(f"<compiler {compiler_attr_str}/>")
    parts.append('<option timestep="0.002" gravity="0 0 -9.81"/>')

    def _wrap(tag: str, inners: list[str]) -> None:
        joined = "\n".join(s for s in inners if s.strip())
        if joined:
            parts.append(f"<{tag}>\n{joined}\n</{tag}>")

    _wrap("default",  [s.default_inner  for s in subtrees])
    _wrap("asset",    [s.asset_inner    for s in subtrees])

    parts.append("<worldbody>")
    parts.append(workbench_inner)
    for s in subtrees:
        parts.append(s.worldbody_inner)
    parts.extend(cube_chunks)
    # Overhead camera looking straight down at the workbench.
    parts.append(
        '<camera name="overhead" pos="0 0 1.0" '
        'xyaxes="1 0 0  0 1 0" fovy="60"/>'
    )
    parts.append("</worldbody>")

    _wrap("contact",   [s.contact_inner  for s in subtrees])
    _wrap("equality",  [s.equality_inner for s in subtrees])
    _wrap("tendon",    [s.tendon_inner   for s in subtrees])
    _wrap("actuator",  [s.actuator_inner for s in subtrees])
    _wrap("sensor",    [s.sensor_inner   for s in subtrees])

    parts.append("</mujoco>")
    return "\n".join(parts), arm_joint_map


def _extract_worldbody_inner(path: Path) -> str:
    root = ET.parse(path).getroot()
    wb = root.find("worldbody")
    if wb is None:
        return ""
    return "\n".join(ET.tostring(c, encoding="unicode") for c in wb)


def _extract_worldbody_inner_from_string(xml: str) -> str:
    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    if wb is None:
        return ""
    return "\n".join(ET.tostring(c, encoding="unicode") for c in wb)
```

- [ ] **Step 5: Run the builder tests**

Run:
```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_builder.py -v
```

Expected: all PASS. If a test fails because the upstream SO-100 joint names differ from `SO101_JOINTS`, **stop and inspect**: print `[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]` against the vendored file, then update `SO101_JOINTS` in `builder.py` to match the actual joint names (and update the spec's joint-name list to match).

- [ ] **Step 6: Commit**

```bash
git add sim/assets/scenes/workbench.xml \
        sim/assets/scenes/cube.xml \
        hmi/backend/haller_hmi/sim/builder.py \
        hmi/backend/tests/sim/test_builder.py
git diff --cached --name-only
git commit -m "feat(hmi/sim): scene builder — namespaced arms + workbench + cubes + overhead camera"
```

---

## Task 7: `SimArmHandle` — drop-in for `ArmHandle`

> **Ordering note:** Task 7's tests construct `ArmConfig(source="sim", sim_arm_name="right")`, which requires the schema extension in Task 8 Step 3. If you're doing tasks strictly in order, jump to Task 8 Step 3 first (it's ~15 lines), land it on its own commit, then come back here. Otherwise the tests below will fail with `TypeError: ArmConfig.__init__() got an unexpected keyword argument 'source'`.

**Files:**
- Create: `hmi/backend/haller_hmi/sim/arm.py`
- Create: `hmi/backend/tests/sim/test_sim_arm_handle.py`

- [ ] **Step 1: Write the failing tests**

Create `hmi/backend/tests/sim/test_sim_arm_handle.py`:

```python
"""SimArmHandle implements the same public surface as ArmHandle."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.config import ArmConfig
from haller_hmi.safety import Mode, ModeError
from haller_hmi.sim.arm import SimArmHandle
from haller_hmi.sim.world import MuJoCoWorld


TINY_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="right_base">
      <joint name="right_shoulder_pan" type="hinge" axis="0 0 1" range="-180 180"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0  0.2 0 0"/>
      <body name="right_link2" pos="0.2 0 0">
        <joint name="right_gripper" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" size="0.02 0.1" fromto="0 0 0  0.2 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="right_shoulder_pan" kp="50" ctrlrange="-3.14 3.14"/>
    <position name="a2" joint="right_gripper"      kp="50" ctrlrange="-1.57 1.57"/>
  </actuator>
</mujoco>
"""

ARM_JOINT_MAP = {"right": ["right_shoulder_pan", "right_gripper"]}


def _make_world_and_handle():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    cfg = ArmConfig(
        id="right", model="so101_follower", port="(sim)", calibration_id="(sim)",
        source="sim", sim_arm_name="right",
    )
    handle = SimArmHandle(cfg, world=world)
    return world, handle


def test_connect_populates_joint_limits_deg():
    world, handle = _make_world_and_handle()
    handle.connect()
    # We use plain LeRobot joint shortnames (no prefix) on the HMI side.
    assert "shoulder_pan" in handle.joint_limits_deg
    assert "gripper" in handle.joint_limits_deg
    lo, hi = handle.joint_limits_deg["shoulder_pan"]
    assert lo == pytest.approx(-180.0, abs=0.1)
    assert hi == pytest.approx(180.0, abs=0.1)


def test_send_goal_writes_ctrl_and_returns_clamped():
    world, handle = _make_world_and_handle()
    handle.connect()
    world.start()
    try:
        sent = handle.send_goal({"shoulder_pan": 30.0, "gripper": 999.0})
        assert sent["shoulder_pan"] == pytest.approx(30.0, abs=1e-3)
        assert sent["gripper"] == pytest.approx(90.0, abs=1e-3)  # clamped to range
        time.sleep(0.3)
        q = handle.read_joints_deg()
        assert abs(q["shoulder_pan"] - 30.0) < 5.0
    finally:
        world.stop()


def test_send_goal_in_auto_mode_raises():
    world, handle = _make_world_and_handle()
    handle.connect()
    handle.guard.set(Mode.AUTO)
    with pytest.raises(ModeError):
        handle.send_goal({"shoulder_pan": 0.0})


def test_state_snapshot_shape_matches_real_arm():
    world, handle = _make_world_and_handle()
    handle.connect()
    snap = handle.state_snapshot()
    assert snap["mode"] in {"auto", "manual", "stop"}
    assert "torque" in snap and "joints" in snap
    assert set(snap["joints"]) == {"shoulder_pan", "gripper"}
    for j, info in snap["joints"].items():
        assert set(info) == {"pos", "min", "max", "torque"}


def test_disable_torque_zeros_kp():
    world, handle = _make_world_and_handle()
    handle.connect()
    handle.disable_torque()
    assert not handle.torque_enabled
    assert world.actuator_kp_for_joint("right_shoulder_pan") == 0.0
    handle.enable_torque()
    assert world.actuator_kp_for_joint("right_shoulder_pan") > 0.0
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sim_arm_handle.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'haller_hmi.sim.arm'` (and `ArmConfig` will reject `source=`/`sim_arm_name=` until Task 8).

- [ ] **Step 3: Implement `SimArmHandle`**

Create `hmi/backend/haller_hmi/sim/arm.py`:

```python
"""SimArmHandle: drop-in for ArmHandle backed by a MuJoCoWorld.

Public surface matches ArmHandle exactly: connect, disconnect, send_goal,
home, disable_torque, enable_torque, state_snapshot, read_joints_deg.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import ArmConfig
from ..safety import Mode, ModeGuard, clamp_joint_goal
from .builder import SO101_JOINTS
from .world import MuJoCoWorld

logger = logging.getLogger(__name__)


@dataclass
class SimArmHandle:
    config: ArmConfig
    world: MuJoCoWorld
    joint_limits_deg: dict[str, tuple[float, float]] = field(default_factory=dict)
    guard: ModeGuard = field(default_factory=lambda: ModeGuard(Mode.MANUAL))
    torque_enabled: bool = True

    @property
    def _prefix(self) -> str:
        return f"{self.config.sim_arm_name}_"

    def connect(self) -> None:
        # Map plain LeRobot joint names ("shoulder_pan") to prefixed MJCF names
        # ("right_shoulder_pan") so we can clamp/read in HMI-native units while
        # the world speaks MJCF names.
        self.joint_limits_deg = {}
        for short in SO101_JOINTS:
            mjcf_name = f"{self._prefix}{short}"
            try:
                self.joint_limits_deg[short] = self.world.joint_range_deg(
                    self.config.sim_arm_name, mjcf_name
                )
            except KeyError:
                # Joint missing in MJCF (e.g. SO-100 vs SO-101 gripper variant) —
                # skip it rather than failing the whole connect.
                logger.warning("sim arm %s: joint %s missing in MJCF; skipping",
                               self.config.id, short)
        logger.info("sim arm %s connected; joints: %s",
                    self.config.id, list(self.joint_limits_deg))

    def disconnect(self) -> None:
        # World lifecycle is owned by ArmManager; nothing per-arm to release.
        pass

    def send_goal(self, goal_deg: dict[str, float]) -> dict[str, float]:
        self.guard.assert_manual()
        if not self.torque_enabled:
            self.enable_torque()
        clamped = clamp_joint_goal(goal_deg, self.joint_limits_deg)
        # Translate short joint names to MJCF names for the world.
        mjcf_goal = {f"{self._prefix}{j}": v for j, v in clamped.items()}
        self.world.write_ctrl_deg(self.config.sim_arm_name, mjcf_goal)
        return clamped

    def home(self) -> dict[str, float]:
        goal = {j: 0.0 for j in self.joint_limits_deg}
        return self.send_goal(goal)

    def disable_torque(self) -> None:
        self.world.set_arm_torque(self.config.sim_arm_name, enabled=False)
        self.torque_enabled = False

    def enable_torque(self) -> None:
        self.world.set_arm_torque(self.config.sim_arm_name, enabled=True)
        self.torque_enabled = True

    def read_joints_deg(self) -> dict[str, float]:
        raw = self.world.read_qpos_deg(self.config.sim_arm_name)
        # Strip the arm-name prefix to match the real ArmHandle's contract.
        prefix = self._prefix
        return {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}

    def state_snapshot(self) -> dict:
        joints_now = self.read_joints_deg()
        joints = {}
        for joint, (lo, hi) in self.joint_limits_deg.items():
            joints[joint] = {
                "pos": float(joints_now.get(joint, 0.0)),
                "min": float(lo),
                "max": float(hi),
                "torque": self.torque_enabled,
            }
        return {
            "mode": self.guard.mode.value,
            "torque": self.torque_enabled,
            "joints": joints,
        }
```

- [ ] **Step 4: Run the sim arm tests**

Run (note: depends on Task 8 for the new `ArmConfig` fields; if you're working out of order, peek at Task 8 first or temporarily pass the new fields via `**kwargs`).

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sim_arm_handle.py -v
```

Expected: PASS once Task 8 lands. If you're running this task first, add the two new `ArmConfig` fields now (Task 8 Step 3) and commit them together.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/sim/arm.py \
        hmi/backend/tests/sim/test_sim_arm_handle.py
git diff --cached --name-only
git commit -m "feat(hmi/sim): SimArmHandle — drop-in for ArmHandle, backed by MuJoCoWorld"
```

---

## Task 8: Config schema — `ArmConfig.source` / `sim_arm_name` and `Config.sim_leader`

**Files:**
- Modify: `hmi/backend/haller_hmi/config.py`
- Create or extend: `hmi/backend/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `hmi/backend/tests/test_config.py` (or append if it exists):

```python
"""Config schema accepts `source: sim` on arms, `source: sim_camera` on cameras,
and an optional top-level `sim_leader` block."""
from __future__ import annotations

from pathlib import Path

from haller_hmi.config import load_config


def test_sim_arm_and_sim_camera_and_sim_leader(tmp_path: Path):
    cfg_file = tmp_path / "sim.yaml"
    cfg_file.write_text(
        """
arms:
  - id: left
    model: so101_follower
    port: "(sim)"
    calibration_id: "(sim)"
    source: sim
    sim_arm_name: left
  - id: right
    model: so101_follower
    port: "(sim)"
    calibration_id: "(sim)"
    source: sim
    sim_arm_name: right
cameras:
  - id: overhead
    role: base
    source: sim_camera
    mjcf_camera: overhead
    width: 640
    height: 480
    fps: 15
sim_leader:
  source: mouse
"""
    )
    cfg = load_config(cfg_file)
    assert len(cfg.arms) == 2
    assert cfg.arms[0].source == "sim"
    assert cfg.arms[0].sim_arm_name == "left"
    assert cfg.cameras[0].source == "sim_camera"
    assert cfg.cameras[0].mjcf_camera == "overhead"
    assert cfg.sim_leader is not None
    assert cfg.sim_leader.source == "mouse"
    assert cfg.sim_leader.dataset_path is None


def test_arm_source_defaults_to_real(tmp_path: Path):
    cfg_file = tmp_path / "real.yaml"
    cfg_file.write_text(
        """
arms:
  - id: right
    model: so101_follower
    port: /dev/null
    calibration_id: haller_follower
"""
    )
    cfg = load_config(cfg_file)
    assert cfg.arms[0].source == "real"
    assert cfg.arms[0].sim_arm_name is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest hmi/backend/tests/test_config.py -v
```

Expected: FAIL (`TypeError: ArmConfig.__init__() got an unexpected keyword argument 'source'`, etc.).

- [ ] **Step 3: Extend the schema**

Edit `hmi/backend/haller_hmi/config.py`. Replace `ArmConfig`, `CameraConfig`, `Config`, and `load_config` as follows (everything else in the file stays):

```python
@dataclass
class ArmConfig:
    id: str
    model: str
    port: str
    calibration_id: str
    enabled: bool = True
    # "real" (default — drives an actual SO-101 over serial) or "sim" (MuJoCo).
    source: str = "real"
    # Required when source == "sim": which arm body in the composed MJCF this
    # handle owns. Typically "left" or "right".
    sim_arm_name: str | None = None


@dataclass
class CameraConfig:
    id: str
    role: str  # "wrist" or "base"
    source: str  # "placeholder" | "opencv" | "mjpeg" | "webrtc" | "sim_camera"
    arm_id: str | None = None
    # OpenCV-specific.
    index_or_path: str | int | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    # sim_camera-specific: which <camera name="..."> in the composed MJCF.
    mjcf_camera: str | None = None


@dataclass
class SimLeaderConfig:
    source: str  # "mouse" | "replay"
    dataset_path: str | None = None  # required when source == "replay"


@dataclass
class Config:
    arms: list[ArmConfig] = field(default_factory=list)
    ros: RosConfig = field(default_factory=RosConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    sim_leader: SimLeaderConfig | None = None


def load_config(path: Path | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("HALLER_HMI_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(cfg_path.read_text())
    sim_leader_raw = raw.get("sim_leader")
    return Config(
        arms=[ArmConfig(**a) for a in raw.get("arms", [])],
        ros=RosConfig(**raw.get("ros", {})),
        telemetry=TelemetryConfig(**raw.get("telemetry", {})),
        cameras=[CameraConfig(**c) for c in raw.get("cameras", [])],
        sim_leader=SimLeaderConfig(**sim_leader_raw) if sim_leader_raw else None,
    )
```

- [ ] **Step 4: Run the config tests + the full suite**

```bash
pytest hmi/backend/tests/test_config.py -v
pytest hmi/backend/tests -v -x
```

Expected: new tests PASS; nothing else regresses.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/config.py hmi/backend/tests/test_config.py
git diff --cached --name-only
git commit -m "feat(hmi/backend): config — source: real|sim on arms, sim_camera on cameras, sim_leader block"
```

---

## Task 9: `ArmManager` constructs `SimArmHandle` with a lazy shared `MuJoCoWorld`

**Files:**
- Modify: `hmi/backend/haller_hmi/arm.py` (extend `ArmManager`)
- Create: `hmi/backend/tests/sim/test_arm_manager_sim.py`

- [ ] **Step 1: Write the failing test**

Create `hmi/backend/tests/sim/test_arm_manager_sim.py`:

```python
"""ArmManager constructs SimArmHandle for sim arms and shares one MuJoCoWorld."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from unittest.mock import patch

from haller_hmi.arm import ArmManager
from haller_hmi.config import ArmConfig
from haller_hmi.sim.arm import SimArmHandle


def test_mixed_real_and_sim_arms_share_one_world(monkeypatch):
    # Stub the real-arm bring-up path so we don't open /dev/null.
    from unittest.mock import MagicMock
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    monkeypatch.setattr(
        "haller_hmi.arm.ArmHandle._load_joint_limits",
        lambda self: {"shoulder_pan": (-120.0, 120.0)},
    )

    cfg_real = ArmConfig(id="real_arm", model="so101_follower",
                         port="/dev/null", calibration_id="x", source="real")
    cfg_sim_l = ArmConfig(id="sim_left", model="so101_follower",
                          port="(sim)", calibration_id="(sim)",
                          source="sim", sim_arm_name="left")
    cfg_sim_r = ArmConfig(id="sim_right", model="so101_follower",
                          port="(sim)", calibration_id="(sim)",
                          source="sim", sim_arm_name="right")

    mgr = ArmManager([cfg_real, cfg_sim_l, cfg_sim_r])
    mgr.connect_all()
    try:
        assert isinstance(mgr["sim_left"], SimArmHandle)
        assert isinstance(mgr["sim_right"], SimArmHandle)
        assert not isinstance(mgr["real_arm"], SimArmHandle)
        # Same world instance shared between the two sim arms.
        assert mgr["sim_left"].world is mgr["sim_right"].world
    finally:
        mgr.disconnect_all()


def test_all_real_arms_dont_construct_a_world(monkeypatch):
    from unittest.mock import MagicMock
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    monkeypatch.setattr(
        "haller_hmi.arm.ArmHandle._load_joint_limits",
        lambda self: {"shoulder_pan": (-120.0, 120.0)},
    )

    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="x", source="real")
    with patch("haller_hmi.sim.world.MuJoCoWorld") as MockWorld:
        mgr = ArmManager([cfg])
        mgr.connect_all()
        mgr.disconnect_all()
        MockWorld.assert_not_called()
```

- [ ] **Step 2: Run to confirm failure**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_arm_manager_sim.py -v
```

Expected: FAIL (`ArmManager` currently always constructs `ArmHandle`).

- [ ] **Step 3: Update `ArmManager`**

Edit `hmi/backend/haller_hmi/arm.py`. Replace the `ArmManager` class with:

```python
class ArmManager:
    """Lookup-by-id collection of arm handles (real or sim)."""

    def __init__(self, arm_configs: list[ArmConfig]):
        self._configs = [c for c in arm_configs if c.enabled]
        self._handles: dict[str, "ArmHandle | SimArmHandle"] = {}
        self._world = None  # lazily constructed if any sim arm/camera needs it

    def _ensure_world(self) -> "MuJoCoWorld":
        if self._world is not None:
            return self._world
        from .sim.builder import build_scene
        from .sim.world import MuJoCoWorld

        sim_arm_names = [c.sim_arm_name for c in self._configs
                         if c.source == "sim" and c.sim_arm_name is not None]
        mjcf_xml, arm_joint_map = build_scene(arms=sim_arm_names, cubes=0)
        self._world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
        self._world.start()
        return self._world

    def connect_all(self) -> None:
        from .calibration_bootstrap import ensure_follower_calibrations
        from .sim.arm import SimArmHandle

        real_configs = [c for c in self._configs if c.source == "real"]
        if real_configs:
            ensure_follower_calibrations(real_configs)

        for cfg in self._configs:
            if cfg.source == "sim":
                if not cfg.sim_arm_name:
                    raise ValueError(
                        f"arm {cfg.id!r} has source=sim but no sim_arm_name"
                    )
                world = self._ensure_world()
                handle = SimArmHandle(cfg, world=world)
                handle.connect()
            else:
                handle = ArmHandle(cfg)
                handle.connect()
            self._handles[cfg.id] = handle

    def disconnect_all(self) -> None:
        for handle in self._handles.values():
            handle.disconnect()
        if self._world is not None:
            self._world.stop()
            self._world = None

    def world(self) -> "MuJoCoWorld | None":
        """Exposed so SimCamera and SimLeaderTeleop can share the same world."""
        return self._world

    def __getitem__(self, arm_id: str):
        if arm_id not in self._handles:
            raise KeyError(f"unknown arm id {arm_id!r}; known: {list(self._handles)}")
        return self._handles[arm_id]

    def values(self):
        return self._handles.values()

    def keys(self):
        return self._handles.keys()
```

Note: this changes `ArmManager.__init__` from "construct handles eagerly" to "construct handles in `connect_all`" — read the next two methods that previously assumed eager construction (`ros_bridge`, `server.py` globals) and make sure they only iterate `mgr.values()` after `connect_all()` ran. They already do (the lifespan hook calls `arms.connect_all()` before anything else uses the manager).

- [ ] **Step 4: Tweak the eager-construction test still in `test_arm.py`**

The existing `test_arm_manager_lookup_by_id` constructs `ArmManager` and accesses `mgr["right"]` without calling `connect_all`. Update it:

In `hmi/backend/tests/test_arm.py`, find `test_arm_manager_lookup_by_id` and replace its body:

```python
def test_arm_manager_lookup_by_id(monkeypatch):
    cfg_right = ArmConfig(id="right", model="so101_follower",
                          port="/dev/null", calibration_id="haller_follower")
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr("haller_hmi.arm.ArmHandle._load_joint_limits",
                        lambda self: {"gripper": (0, 100)})
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    mgr = ArmManager([cfg_right])
    mgr.connect_all()
    assert mgr["right"].config.id == "right"
```

- [ ] **Step 5: Run the new tests and the full suite**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_arm_manager_sim.py -v
pytest hmi/backend/tests -v -x
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add hmi/backend/haller_hmi/arm.py \
        hmi/backend/tests/test_arm.py \
        hmi/backend/tests/sim/test_arm_manager_sim.py
git diff --cached --name-only
git commit -m "feat(hmi/backend): ArmManager constructs SimArmHandle, shares lazy MuJoCoWorld"
```

---

## Task 10: `SimCamera` — offscreen `mujoco.Renderer` → JPEG

**Files:**
- Create: `hmi/backend/haller_hmi/sim/camera.py`
- Create: `hmi/backend/tests/sim/test_sim_camera.py`

- [ ] **Step 1: Write the failing test**

Create `hmi/backend/tests/sim/test_sim_camera.py`:

```python
"""SimCamera renders a JPEG from a named MJCF camera, headlessly."""
from __future__ import annotations

import io
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.config import CameraConfig
from haller_hmi.sim.builder import build_scene
from haller_hmi.sim.camera import SimCamera
from haller_hmi.sim.world import MuJoCoWorld


def test_sim_camera_renders_a_nonblank_jpeg():
    mjcf_xml, arm_joint_map = build_scene(arms=["right"], cubes=1)
    world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
    world.start()
    try:
        cfg = CameraConfig(
            id="overhead_sim", role="base", source="sim_camera",
            mjcf_camera="overhead", width=320, height=240, fps=10,
        )
        cam = SimCamera(cfg, world=world)
        cam.connect()
        try:
            time.sleep(0.2)  # let the render thread produce at least one frame
            jpeg = cam.latest_jpeg()
            assert jpeg is not None
            assert jpeg[:2] == b"\xff\xd8", "not a JPEG"
            assert len(jpeg) > 500, f"suspiciously small JPEG ({len(jpeg)} bytes)"
        finally:
            cam.disconnect()
    finally:
        world.stop()
```

- [ ] **Step 2: Run to confirm failure**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sim_camera.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'haller_hmi.sim.camera'`.

- [ ] **Step 3: Implement `SimCamera`**

Create `hmi/backend/haller_hmi/sim/camera.py`:

```python
"""SimCamera: an HMI Camera-shaped object backed by mujoco.Renderer.

Implements the same surface CameraManager/HTTP routes already consume from
`CameraHandle`: `.cfg`, `.active`, `connect`, `disconnect`, `latest_jpeg`.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import mujoco
import numpy as np

from ..config import CameraConfig
from .world import MuJoCoWorld

logger = logging.getLogger(__name__)

JPEG_QUALITY = 80


class SimCamera:
    def __init__(self, cfg: CameraConfig, world: MuJoCoWorld):
        if cfg.source != "sim_camera":
            raise ValueError(f"SimCamera requires source=sim_camera, got {cfg.source!r}")
        if not cfg.mjcf_camera:
            raise ValueError(f"camera {cfg.id!r}: source=sim_camera requires mjcf_camera")
        self.cfg = cfg
        self.world = world
        self._renderer: mujoco.Renderer | None = None
        self._latest: bytes | None = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def connect(self) -> None:
        self._renderer = mujoco.Renderer(self.world.model,
                                         height=self.cfg.height,
                                         width=self.cfg.width)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._render_loop,
            name=f"SimCamera-{self.cfg.id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("sim camera %s connected: %dx%d @ %d fps (mjcf cam %s)",
                    self.cfg.id, self.cfg.width, self.cfg.height, self.cfg.fps,
                    self.cfg.mjcf_camera)

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def latest_jpeg(self, max_age_ms: int = 500) -> bytes | None:
        with self._latest_lock:
            return self._latest

    def _render_loop(self) -> None:
        assert self._renderer is not None
        period = 1.0 / max(1, self.cfg.fps)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                # Snapshot data under the world lock so we don't race the stepper.
                with self.world._lock:  # noqa: SLF001 — intentional internal sync
                    self._renderer.update_scene(self.world.data,
                                                camera=self.cfg.mjcf_camera)
                rgb = self._renderer.render()  # H,W,3 uint8
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with self._latest_lock:
                        self._latest = buf.tobytes()
            except Exception:
                logger.exception("sim camera %s: render failed", self.cfg.id)
                time.sleep(0.1)
            sleep_for = period - (time.perf_counter() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
```

- [ ] **Step 4: Run the sim camera test**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sim_camera.py -v
```

Expected: PASS. If the JPEG comes out tiny (<500 bytes), the scene may be all-black; check that the `<light>` in workbench.xml is actually being included and the camera is pointed at the scene (`pos="0 0 1.0"` looking down requires a non-zero `xyaxes` or `mode="fixed"` with proper orientation — adjust the camera in `builder.py` if needed).

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/sim/camera.py \
        hmi/backend/tests/sim/test_sim_camera.py
git diff --cached --name-only
git commit -m "feat(hmi/sim): SimCamera — offscreen mujoco.Renderer thread → JPEG"
```

---

## Task 11: `CameraManager` constructs `SimCamera` for `source: sim_camera`

**Files:**
- Modify: `hmi/backend/haller_hmi/cameras.py` (extend `CameraManager.__init__` and signature)
- Modify: `hmi/backend/haller_hmi/server.py` (pass world reference to CameraManager)

- [ ] **Step 1: Write the failing test**

Append to `hmi/backend/tests/sim/test_sim_camera.py`:

```python
def test_camera_manager_constructs_sim_camera_for_sim_camera_source():
    from haller_hmi.cameras import CameraManager
    from haller_hmi.sim.camera import SimCamera
    from haller_hmi.sim.world import MuJoCoWorld

    mjcf_xml, arm_joint_map = build_scene(arms=["right"], cubes=0)
    world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
    cfg = CameraConfig(id="overhead_sim", role="base", source="sim_camera",
                       mjcf_camera="overhead", width=160, height=120, fps=5)
    mgr = CameraManager([cfg], world=world)
    assert isinstance(mgr["overhead_sim"], SimCamera)


def test_camera_manager_without_world_skips_sim_cameras(caplog):
    from haller_hmi.cameras import CameraManager

    cfg = CameraConfig(id="overhead_sim", role="base", source="sim_camera",
                       mjcf_camera="overhead", width=160, height=120, fps=5)
    mgr = CameraManager([cfg], world=None)
    # Camera is skipped (not constructed), not crashed.
    assert "overhead_sim" not in mgr.keys() if hasattr(mgr, "keys") else True
```

(The second test will be slightly relaxed if `CameraManager` doesn't expose `keys()`; the important behavior is "no crash when no world is available.")

- [ ] **Step 2: Run to confirm failure**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sim_camera.py -v
```

Expected: FAIL with `TypeError: CameraManager.__init__() got an unexpected keyword argument 'world'`.

- [ ] **Step 3: Extend `CameraManager`**

In `hmi/backend/haller_hmi/cameras.py`:

```python
class CameraManager:
    """Lookup-by-id collection of camera handles (real OpenCV or sim)."""

    def __init__(self, camera_configs: list[CameraConfig], world=None):
        from .sim.camera import SimCamera  # late import so non-sim setups don't need mujoco

        self._handles: dict[str, "CameraHandle | SimCamera"] = {}
        for c in camera_configs:
            if c.source == "sim_camera":
                if world is None:
                    logger.warning(
                        "camera %s: source=sim_camera but no MuJoCoWorld available; skipping",
                        c.id,
                    )
                    continue
                self._handles[c.id] = SimCamera(c, world=world)
            else:
                self._handles[c.id] = CameraHandle(c)
```

(Leave the rest of `CameraManager` — `connect_all`, `disconnect_all`, `list`, `__getitem__`, `mjpeg_stream` — unchanged. They call methods that both `CameraHandle` and `SimCamera` implement.)

Also add a `keys(self)` method to `CameraManager` if not present:

```python
    def keys(self):
        return self._handles.keys()
```

- [ ] **Step 4: Wire the world through in `server.py`**

In `hmi/backend/haller_hmi/server.py`, change the initial wiring so `CameraManager` is constructed AFTER `arms.connect_all()` runs (because the world only exists after that). Replace the module-level globals + lifespan body:

Find this block near the top:

```python
cfg = load_config()
arms = ArmManager(cfg.arms)
cameras = CameraManager(cfg.cameras)
```

Replace with:

```python
cfg = load_config()
arms = ArmManager(cfg.arms)
cameras: CameraManager | None = None   # constructed in lifespan, after arms.connect_all()
```

In the lifespan function, change:

```python
    arms.connect_all()
    cameras.connect_all()
```

to:

```python
    global cameras
    arms.connect_all()
    cameras = CameraManager(cfg.cameras, world=arms.world())
    cameras.connect_all()
```

And any other reference to `cameras.` in the file needs to be guarded for the period before lifespan (there shouldn't be any — HTTP routes always run inside the lifespan-active window). Grep for `cameras.` in `server.py` and confirm.

- [ ] **Step 5: Run the new + full suites**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sim_camera.py -v
pytest hmi/backend/tests -v -x
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add hmi/backend/haller_hmi/cameras.py \
        hmi/backend/haller_hmi/server.py \
        hmi/backend/tests/sim/test_sim_camera.py
git diff --cached --name-only
git commit -m "feat(hmi/backend): CameraManager wires SimCamera for source=sim_camera"
```

---

## Task 12: `LeaderSource` protocol + `MouseDragSource` + `DatasetReplaySource`

**Files:**
- Create: `hmi/backend/haller_hmi/sim/sources.py`
- Create: `hmi/backend/tests/sim/test_sources.py`

- [ ] **Step 1: Write the failing tests**

Create `hmi/backend/tests/sim/test_sources.py`:

```python
"""LeaderSource implementations: mouse-drag (reads sim leader qpos) and dataset replay."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.sim.builder import build_scene, SO101_JOINTS
from haller_hmi.sim.sources import DatasetReplaySource, MouseDragSource
from haller_hmi.sim.world import MuJoCoWorld


def test_mouse_drag_source_reads_sim_leader_qpos():
    mjcf_xml, arm_joint_map = build_scene(arms=["left", "right"], cubes=0)
    world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
    src = MouseDragSource(world=world, arm_name="left")
    out = src.read()
    # Joint shortnames (no prefix) — what SimLeaderTeleop will feed into send_goal.
    assert set(out) == set(SO101_JOINTS) or set(out).issubset(set(SO101_JOINTS))
    for v in out.values():
        assert isinstance(v, float)


def test_dataset_replay_source_walks_observation_state(tmp_path, monkeypatch):
    """Fake out lerobot.datasets.LeRobotDataset so we don't pull data off the network."""
    from unittest.mock import MagicMock

    fake_rows = [
        {"observation.state": [0.0]  * 6},
        {"observation.state": [10.0] * 6},
        {"observation.state": [20.0] * 6},
    ]

    class FakeDataset:
        meta = MagicMock(fps=30, features={
            "observation.state": {"names": SO101_JOINTS}
        })
        def __len__(self): return len(fake_rows)
        def __getitem__(self, i): return fake_rows[i]

    monkeypatch.setattr(
        "haller_hmi.sim.sources._load_lerobot_dataset",
        lambda path: FakeDataset(),
    )
    src = DatasetReplaySource(dataset_path=str(tmp_path))
    src.start()
    try:
        first = src.read()
        assert first == {j: 0.0 for j in SO101_JOINTS}
        second = src.read()
        assert second == {j: 10.0 for j in SO101_JOINTS}
        third = src.read()
        assert third == {j: 20.0 for j in SO101_JOINTS}
        # Looping back to start
        fourth = src.read()
        assert fourth == {j: 0.0 for j in SO101_JOINTS}
    finally:
        src.stop()
```

- [ ] **Step 2: Run to confirm failure**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sources.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'haller_hmi.sim.sources'`.

- [ ] **Step 3: Implement `sources.py`**

Create `hmi/backend/haller_hmi/sim/sources.py`:

```python
"""LeaderSource implementations for SimLeaderTeleop.

A LeaderSource just promises `read() -> dict[shortname -> degrees]` returning
the next leader pose. SimLeaderTeleop ticks it at a fixed rate and forwards
the result to a sim follower via the existing send_goal path.
"""
from __future__ import annotations

import logging
import threading
from typing import Protocol

from .builder import SO101_JOINTS
from .world import MuJoCoWorld

logger = logging.getLogger(__name__)


class LeaderSource(Protocol):
    def read(self) -> dict[str, float]: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class MouseDragSource:
    """Reads the sim leader's own qpos. The user drags joints in the MuJoCo
    viewer (`MUJOCO_VIEWER=1`); we simply forward the resulting pose."""

    def __init__(self, world: MuJoCoWorld, arm_name: str):
        self.world = world
        self.arm_name = arm_name
        self.prefix = f"{arm_name}_"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> dict[str, float]:
        raw = self.world.read_qpos_deg(self.arm_name)
        return {k[len(self.prefix):]: v for k, v in raw.items()
                if k.startswith(self.prefix)}


def _load_lerobot_dataset(path: str):
    """Indirection so tests can monkeypatch without importing lerobot.datasets."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(path)


class DatasetReplaySource:
    """Walks observation.state from a recorded LeRobot dataset at real time, looping."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._ds = None
        self._idx = 0
        self._joint_names: list[str] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        self._ds = _load_lerobot_dataset(self.dataset_path)
        # Prefer the dataset's own joint-name list; fall back to canonical SO-101.
        try:
            names = list(self._ds.meta.features["observation.state"]["names"])
        except Exception:
            names = list(SO101_JOINTS)
        # Truncate or pad to SO101_JOINTS length to keep downstream code stable.
        self._joint_names = (names + SO101_JOINTS)[: len(SO101_JOINTS)]

    def stop(self) -> None:
        self._ds = None

    def read(self) -> dict[str, float]:
        if self._ds is None:
            raise RuntimeError("DatasetReplaySource: start() not called")
        with self._lock:
            row = self._ds[self._idx]
            self._idx = (self._idx + 1) % len(self._ds)
        state = row["observation.state"]
        # state may be a list or a tensor; convert defensively.
        try:
            values = [float(x) for x in state]
        except TypeError:
            values = [float(x) for x in state.tolist()]
        return {name: values[i] for i, name in enumerate(self._joint_names) if i < len(values)}
```

- [ ] **Step 4: Run the source tests**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sources.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/haller_hmi/sim/sources.py \
        hmi/backend/tests/sim/test_sources.py
git diff --cached --name-only
git commit -m "feat(hmi/sim): MouseDragSource + DatasetReplaySource leader inputs"
```

---

## Task 13: `SimLeaderTeleop` session + REST endpoints + session-lock peer wiring

**Files:**
- Create: `hmi/backend/haller_hmi/sim/teleop.py`
- Modify: `hmi/backend/haller_hmi/server.py` (instantiate, attach peers, mount routes)
- Create: `hmi/backend/tests/sim/test_sim_teleop.py`

- [ ] **Step 1: Write the failing test**

Create `hmi/backend/tests/sim/test_sim_teleop.py`:

```python
"""SimLeaderTeleop ticks a LeaderSource and forwards the pose to a sim follower."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

from unittest.mock import MagicMock

from haller_hmi.config import ArmConfig
from haller_hmi.arm import ArmManager
from haller_hmi.sim.teleop import SimLeaderTeleop


def test_sim_teleop_calls_send_goal_at_configured_rate(monkeypatch):
    # Sim-only setup so we don't need real hardware mocks.
    cfg_l = ArmConfig(id="sim_left",  model="so101_follower", port="(sim)",
                      calibration_id="(sim)", source="sim", sim_arm_name="left")
    cfg_r = ArmConfig(id="sim_right", model="so101_follower", port="(sim)",
                      calibration_id="(sim)", source="sim", sim_arm_name="right")
    mgr = ArmManager([cfg_l, cfg_r])
    mgr.connect_all()
    try:
        # Replace the source with a stub returning a fixed pose.
        fake_source = MagicMock()
        fake_source.read.return_value = {"shoulder_pan": 5.0}
        fake_source.start = MagicMock()
        fake_source.stop = MagicMock()

        # Spy on follower send_goal
        follower = mgr["sim_right"]
        follower.send_goal = MagicMock(return_value={"shoulder_pan": 5.0})

        session = SimLeaderTeleop(arms=mgr)
        session.start(follower_id="sim_right", source=fake_source, hz=120.0)
        time.sleep(0.1)
        session.stop()

        assert fake_source.start.called
        assert fake_source.read.call_count >= 2
        assert follower.send_goal.called
        assert fake_source.stop.called
    finally:
        mgr.disconnect_all()


def test_sim_teleop_status_shape():
    from haller_hmi.sim.teleop import SimLeaderTeleop
    s = SimLeaderTeleop(arms=MagicMock())
    out = s.status()
    assert set(out) >= {"running", "follower", "hz", "tick_count", "last_error"}
```

- [ ] **Step 2: Run to confirm failure**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sim_teleop.py -v
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement `SimLeaderTeleop`**

Create `hmi/backend/haller_hmi/sim/teleop.py`:

```python
"""SimLeaderTeleop: drive a sim follower from a LeaderSource (mouse / replay).

Mirrors the structure of `TeleopSession` so the existing session-lock peer
wiring (`attach_peer`) accepts it as a coequal teleop kind.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from ..arm import ArmManager
from ..safety import Mode
from .sources import LeaderSource

logger = logging.getLogger(__name__)


@dataclass
class _State:
    running: bool = False
    follower: str | None = None
    hz: float = 0.0
    tick_count: int = 0
    last_error: str | None = None
    started_at: float | None = None


class SimLeaderTeleop:
    def __init__(self, arms: ArmManager):
        self._arms = arms
        self._state = _State()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peer = None
        self._source: LeaderSource | None = None

    def attach_peer(self, peer) -> None:
        self._peer = peer

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._state.running,
                "follower": self._state.follower,
                "hz": self._state.hz,
                "tick_count": self._state.tick_count,
                "last_error": self._state.last_error,
                "started_at": self._state.started_at,
            }

    def start(self, follower_id: str, source: LeaderSource, hz: float = 60.0) -> None:
        with self._lock:
            if self._state.running:
                raise RuntimeError("sim teleop already running")
            if self._peer is not None and getattr(self._peer, "status",
                                                  lambda: {})().get("running"):
                raise RuntimeError("another teleop is already running")
            follower = self._arms[follower_id]
            follower.guard.set(Mode.MANUAL)
            if not follower.torque_enabled:
                follower.enable_torque()
            self._source = source
            self._source.start()
            self._stop.clear()
            self._state = _State(
                running=True, follower=follower_id, hz=hz,
                tick_count=0, last_error=None, started_at=time.time(),
            )
        self._thread = threading.Thread(
            target=self._loop, name="haller-hmi-sim-teleop", daemon=True
        )
        self._thread.start()
        logger.info("sim teleop started -> %s @ %.1f Hz", follower_id, hz)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            if self._source is not None:
                try:
                    self._source.stop()
                except Exception:
                    logger.exception("sim teleop: source.stop() failed")
                self._source = None
            self._state.running = False

    def _loop(self) -> None:
        with self._lock:
            follower_id = self._state.follower
            hz = self._state.hz
            source = self._source
        assert follower_id is not None and source is not None
        follower = self._arms[follower_id]
        period = 1.0 / max(1.0, hz)

        while not self._stop.is_set():
            tick_start = time.perf_counter()
            try:
                goal = source.read()
                # Only joints the follower knows about; send_goal clamps for us.
                trimmed = {j: v for j, v in goal.items() if j in follower.joint_limits_deg}
                follower.send_goal(trimmed)
                with self._lock:
                    self._state.tick_count += 1
                    self._state.last_error = None
            except Exception as e:
                logger.exception("sim teleop tick failed")
                with self._lock:
                    self._state.last_error = str(e)
                time.sleep(0.05)
                continue
            slack = period - (time.perf_counter() - tick_start)
            if slack > 0:
                time.sleep(slack)
```

- [ ] **Step 4: Mount in `server.py`**

In `hmi/backend/haller_hmi/server.py`, after the existing `teleop = TeleopSession(arms)` / `human_teleop = HumanTeleopSession(arms)` / `attach_peer` block, add:

```python
from .sim.teleop import SimLeaderTeleop
sim_teleop = SimLeaderTeleop(arms)
teleop.attach_peer(sim_teleop)
human_teleop.attach_peer(sim_teleop)
sim_teleop.attach_peer(teleop)   # any of the peers' .status().running blocks the others
```

Add REST routes (placed alongside the existing `/teleop/start` etc.):

```python
class SimTeleopStartBody(BaseModel):
    follower: str
    hz: float = 60.0
    # Body of the leader source — one of:
    #   {"source": "mouse", "arm_name": "left"}
    #   {"source": "replay", "dataset_path": "/path/to/lerobot/dataset"}
    leader: dict


@app.post("/teleop/sim/start")
def teleop_sim_start(body: SimTeleopStartBody):
    leader_cfg = body.leader
    src_kind = leader_cfg.get("source")
    if src_kind == "mouse":
        from .sim.sources import MouseDragSource
        world = arms.world()
        if world is None:
            raise HTTPException(status_code=409, detail="sim world not active")
        src = MouseDragSource(world=world, arm_name=leader_cfg["arm_name"])
    elif src_kind == "replay":
        from .sim.sources import DatasetReplaySource
        src = DatasetReplaySource(dataset_path=leader_cfg["dataset_path"])
    else:
        raise HTTPException(status_code=400, detail=f"unknown leader source {src_kind!r}")
    try:
        sim_teleop.start(follower_id=body.follower, source=src, hz=body.hz)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return sim_teleop.status()


@app.post("/teleop/sim/stop")
def teleop_sim_stop():
    sim_teleop.stop()
    return sim_teleop.status()


@app.get("/teleop/sim/status")
def teleop_sim_status():
    return sim_teleop.status()
```

Add `sim_teleop.stop()` to the shutdown sequence in the lifespan teardown (alongside `teleop.stop()` and `human_teleop.stop()`).

- [ ] **Step 5: Run the sim-teleop tests and the routes suite**

```bash
MUJOCO_GL=egl pytest hmi/backend/tests/sim/test_sim_teleop.py hmi/backend/tests/test_routes.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add hmi/backend/haller_hmi/sim/teleop.py \
        hmi/backend/haller_hmi/server.py \
        hmi/backend/tests/sim/test_sim_teleop.py
git diff --cached --name-only
git commit -m "feat(hmi/sim): SimLeaderTeleop + /teleop/sim REST routes + session-lock peer"
```

---

## Task 14: Three preset YAML configs

**Files:**
- Create: `hmi/backend/config.solo-sim.yaml`
- Create: `hmi/backend/config.bimanual-sim.yaml`
- Create: `hmi/backend/config.leader-follower-sim.yaml`

- [ ] **Step 1: Write `config.solo-sim.yaml`**

Create `hmi/backend/config.solo-sim.yaml`:

```yaml
# Solo SO-101 sim: one follower, workbench + one cube + overhead camera.
arms:
  - id: right
    model: so101_follower
    port: "(sim)"
    calibration_id: "(sim)"
    source: sim
    sim_arm_name: right
    enabled: true
ros:
  cmd_vel_topic: /cmd_vel
  odom_topic: /odom
  scan_topic: /scan
  max_linear: 1.0
  max_angular: 2.0
telemetry:
  hz: 20
cameras:
  - id: overhead_sim
    role: base
    source: sim_camera
    mjcf_camera: overhead
    width: 640
    height: 480
    fps: 15
```

- [ ] **Step 2: Write `config.bimanual-sim.yaml`**

Create `hmi/backend/config.bimanual-sim.yaml`:

```yaml
# Bimanual sim: two SO-101 followers facing the same workbench, two cubes.
arms:
  - id: left
    model: so101_follower
    port: "(sim)"
    calibration_id: "(sim)"
    source: sim
    sim_arm_name: left
    enabled: true
  - id: right
    model: so101_follower
    port: "(sim)"
    calibration_id: "(sim)"
    source: sim
    sim_arm_name: right
    enabled: true
ros:
  cmd_vel_topic: /cmd_vel
  odom_topic: /odom
  scan_topic: /scan
  max_linear: 1.0
  max_angular: 2.0
telemetry:
  hz: 20
cameras:
  - id: overhead_sim
    role: base
    source: sim_camera
    mjcf_camera: overhead
    width: 640
    height: 480
    fps: 15
```

- [ ] **Step 3: Write `config.leader-follower-sim.yaml`**

Create `hmi/backend/config.leader-follower-sim.yaml`:

```yaml
# Leader+follower sim, default = mouse-drag the sim leader in the MuJoCo viewer.
#
# Variants:
#   - Mouse (default): MUJOCO_VIEWER=1 ./scripts/run_hmi.sh --config <this file>,
#     then drag the LEFT arm's joints in the viewer; the RIGHT arm mirrors via
#     POST /teleop/sim/start {follower: "right", leader: {source: "mouse", arm_name: "left"}}.
#
#   - Replay: POST /teleop/sim/start {follower: "right", leader: {source: "replay",
#     dataset_path: "/path/to/lerobot/dataset"}}.
#
#   - Real leader -> sim follower: REPLACE the `left` arm below with `source: real`
#     and `port: /dev/haller_arm_leader` / `calibration_id: haller_leader`, then use
#     the regular leader<->follower POST /teleop/start endpoint (not /teleop/sim/start).
arms:
  - id: left
    model: so101_follower
    port: "(sim)"
    calibration_id: "(sim)"
    source: sim
    sim_arm_name: left
    enabled: true
  - id: right
    model: so101_follower
    port: "(sim)"
    calibration_id: "(sim)"
    source: sim
    sim_arm_name: right
    enabled: true
ros:
  cmd_vel_topic: /cmd_vel
  odom_topic: /odom
  scan_topic: /scan
  max_linear: 1.0
  max_angular: 2.0
telemetry:
  hz: 20
cameras:
  - id: overhead_sim
    role: base
    source: sim_camera
    mjcf_camera: overhead
    width: 640
    height: 480
    fps: 15
sim_leader:
  source: mouse
```

- [ ] **Step 4: Smoke-load every preset under load_config**

Run:
```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
for f in hmi/backend/config.solo-sim.yaml \
         hmi/backend/config.bimanual-sim.yaml \
         hmi/backend/config.leader-follower-sim.yaml; do
  echo "--- $f ---"
  python -c "from haller_hmi.config import load_config; import pathlib; \
             c = load_config(pathlib.Path('$f')); \
             print('arms:', [(a.id, a.source) for a in c.arms]); \
             print('cams:', [(c2.id, c2.source) for c2 in c.cameras]); \
             print('sim_leader:', c.sim_leader)"
done
```

Expected: prints three preset summaries, no exceptions.

- [ ] **Step 5: Commit**

```bash
git add hmi/backend/config.solo-sim.yaml \
        hmi/backend/config.bimanual-sim.yaml \
        hmi/backend/config.leader-follower-sim.yaml
git diff --cached --name-only
git commit -m "feat(hmi): config presets — solo-sim, bimanual-sim, leader-follower-sim"
```

---

## Task 15: `run_hmi.sh --config <path>` flag

**Files:**
- Modify: `scripts/run_hmi.sh`

- [ ] **Step 1: Inspect current arg-parsing**

Run:
```bash
cat scripts/run_hmi.sh
```

Confirm there's no existing arg-parsing — the current script takes no positional args.

- [ ] **Step 2: Add `--config` flag**

Edit `scripts/run_hmi.sh`. Right after the `source ... activate-haller-hmi` line, insert:

```bash
# --config <path>: select a non-default HMI config (e.g. one of the sim presets).
# Exported as HALLER_HMI_CONFIG, which haller_hmi.config.load_config respects.
while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            shift
            if [ -z "${1:-}" ]; then
                echo "run_hmi.sh: --config requires a path" >&2
                exit 2
            fi
            export HALLER_HMI_CONFIG="$1"
            shift
            ;;
        *)
            echo "run_hmi.sh: unknown arg $1" >&2
            exit 2
            ;;
    esac
done

if [ -n "${HALLER_HMI_CONFIG:-}" ]; then
    echo "run_hmi.sh: using config $HALLER_HMI_CONFIG"
fi
```

(Do NOT add `set -u` — per [[project-local-env-quirks]], `activate-haller-hmi` sources ROS which breaks under `set -u`. The script already uses `set -euo pipefail`; **change it to `set -eo pipefail`** at the top of the file if it currently has `-u`.)

- [ ] **Step 3: Verify shebang and flags didn't break the no-arg path**

Run:
```bash
bash -n scripts/run_hmi.sh
echo "shellcheck-style parse OK"
```

Expected: no syntax errors. (We don't actually launch the HMI here — that would start a real backend.)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_hmi.sh
git diff --cached --name-only
git commit -m "feat(hmi/scripts): run_hmi.sh --config <path> selects HALLER_HMI_CONFIG"
```

---

## Task 16: `docs/setup/sim.md` + `hmi/README.md` + `README.md` updates

**Files:**
- Create: `docs/setup/sim.md`
- Modify: `hmi/README.md` (add a "Simulation" section)
- Modify: `README.md` (one-line status bullet)

- [ ] **Step 1: Write `docs/setup/sim.md`**

Create `docs/setup/sim.md`:

```markdown
# SO-101 MuJoCo simulation

Three HMI-driven MuJoCo simulations of the SO-101 arms — solo follower, bimanual,
and leader+follower — that reuse the existing HMI control surfaces (per-arm
panels, leader↔follower teleop, human-pose teleop, dataset recorder, MJPEG
camera streams). One feature, four use cases: dev without hardware, dataset
generation, VLA closed-loop eval, and demos.

## Install

The `mujoco` Python package is already pinned in `hmi/backend/pyproject.toml`.
If you've installed the backend in editable mode, you have it. Otherwise:

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
pip install -e hmi/backend
```

The HMI runs headless by default and uses EGL for offscreen rendering. If your
host lacks EGL, fall back to OSMesa:

```bash
export MUJOCO_GL=osmesa   # or 'egl' (default) or 'glfw' for a viewer
```

## The three presets

| Preset | Config | Arms | Scene |
| --- | --- | --- | --- |
| Solo follower    | `hmi/backend/config.solo-sim.yaml` | 1 sim follower | workbench + 1 cube |
| Bimanual         | `hmi/backend/config.bimanual-sim.yaml` | 2 sim followers | workbench + 2 cubes |
| Leader+follower  | `hmi/backend/config.leader-follower-sim.yaml` | 2 sim arms | workbench |

Bring any one up with:

```bash
./scripts/run_hmi.sh --config hmi/backend/config.solo-sim.yaml
```

Then open the HMI in a browser (default `http://localhost:3000`). Joint sliders
and the overhead camera both work against the sim.

## Watching the physics

By default the HMI runs headless and you watch the simulated scene through the
overhead camera in the browser (the MJPEG stream is at
`http://localhost:8000/cameras/overhead_sim/stream`).

For a desktop MuJoCo viewer with interactive mouse-drag perturbation:

```bash
MUJOCO_VIEWER=1 ./scripts/run_hmi.sh --config hmi/backend/config.solo-sim.yaml
```

(See "Leader+follower modes" below for what mouse-drag enables.)

## Leader+follower modes

The leader+follower preset has three operating modes. Default is mouse-drag.

### Mouse-drag (default)

```bash
MUJOCO_VIEWER=1 ./scripts/run_hmi.sh --config hmi/backend/config.leader-follower-sim.yaml
```

Then start a sim teleop session:

```bash
curl -X POST http://localhost:8000/teleop/sim/start \
     -H 'Content-Type: application/json' \
     -d '{"follower":"right","leader":{"source":"mouse","arm_name":"left"},"hz":60}'
```

Drag the LEFT arm's joints in the MuJoCo viewer — the RIGHT arm mirrors.

### Dataset replay

```bash
curl -X POST http://localhost:8000/teleop/sim/start \
     -H 'Content-Type: application/json' \
     -d '{"follower":"right","leader":{"source":"replay",
          "dataset_path":"/path/to/lerobot/dataset"},"hz":30}'
```

### Real leader → sim follower

Edit `hmi/backend/config.leader-follower-sim.yaml` and change the LEFT arm to
`source: real` with the right `port` and `calibration_id`. Then use the regular
leader↔follower endpoint (the HMI's existing TeleopSession does the rest):

```bash
curl -X POST http://localhost:8000/teleop/start \
     -H 'Content-Type: application/json' \
     -d '{"leader":"left","follower":"right","hz":60}'
```

## Troubleshooting

### "GLFWError: X11: Failed to open display"

Set `MUJOCO_GL=egl` (default) or `MUJOCO_GL=osmesa` for headless. Only set
`MUJOCO_GL=glfw` if you have an X11 / Wayland display AND want the desktop
viewer (`MUJOCO_VIEWER=1`).

### Sim camera frame is all black

Check that the `<light>` element in `sim/assets/scenes/workbench.xml` is in the
composed MJCF (it always is — the builder includes the workbench unconditionally)
and that the overhead camera's `pos` / `euler` actually point at the scene.
Adjust the `<camera name="overhead" ...>` line in
`hmi/backend/haller_hmi/sim/builder.py` if your scene is tall or off-center.

### `lerobot.policies` import error

Unrelated to the sim. See the project's notes on the local scipy/numpy ABI
issue — VLA policy code lives on RunPod, not on the dev laptop.

## Out of scope (for now)

- Wrist cameras in sim.
- Gripper / cube friction tuning for reliable picks (default MJCF likely needs
  work for real pick-and-place).
- Domain randomization (textures, lighting, object pose).
- A closed-loop policy-eval CLI (belongs on RunPod alongside the existing
  `scripts/runpod/` recipes).
```

- [ ] **Step 2: Add a "Simulation" section to `hmi/README.md`**

Open `hmi/README.md`. Find a sensible insertion point near the top (after the intro / before deeper subsections) and append:

```markdown
## Simulation

Three MuJoCo presets let you bring the HMI up against simulated arms — solo,
bimanual, leader+follower. See [docs/setup/sim.md](../docs/setup/sim.md).

```bash
./scripts/run_hmi.sh --config hmi/backend/config.solo-sim.yaml
./scripts/run_hmi.sh --config hmi/backend/config.bimanual-sim.yaml
./scripts/run_hmi.sh --config hmi/backend/config.leader-follower-sim.yaml
```
```

- [ ] **Step 3: Add a one-line status bullet to `README.md`**

In `README.md`, find the "Status (May 2026):" paragraph and append (in the same paragraph):

> Three MuJoCo sim presets (solo, bimanual, leader+follower) drop into the same HMI surface; see [`docs/setup/sim.md`](./docs/setup/sim.md).

- [ ] **Step 4: Commit**

```bash
git add docs/setup/sim.md hmi/README.md README.md
git diff --cached --name-only
git commit -m "docs: sim setup guide + HMI README + top-level status pointer"
```

---

## Self-Review Pass (run before declaring the plan complete)

Check the plan against the spec:

- [x] §1 goal — solo / bimanual / leader+follower presets → Tasks 14 + smoke runs
- [x] §1 in-scope: MuJoCo world wrapping community MJCF → Tasks 4–6
- [x] §1 in-scope: `SimArmHandle` drop-in → Task 7
- [x] §1 in-scope: `SimCamera` adapter → Tasks 10–11
- [x] §1 in-scope: three config presets → Task 14
- [x] §1 in-scope: `SimLeaderTeleop` with mouse / real / replay → Task 13 (mouse + replay) + leader-follower preset note (real)
- [x] §1 in-scope: tests for builder / sim arm / sim camera / sources → Tasks 6/7/10/12
- [x] §1 in-scope: `docs/setup/sim.md` → Task 16
- [x] §3 drop-in interface methods → Task 7
- [x] §3a interface tightening (`read_joints_deg` + teleop refactor) → Tasks 2 + 3
- [x] §4 stepper at 500 Hz / E-STOP path → Task 5 (timestep=0.002, pause/resume)
- [x] §5 SimCamera plugged into existing MJPEG routes → Task 11
- [x] §6 three preset YAMLs → Task 14
- [x] §7 leader sources, real-leader via mixed-mode config → Tasks 12, 13, 14
- [x] §8 session-lock peer → Task 13 step 4
- [x] §9 vendor a community MJCF → Task 4
- [x] §10 testing layout → tests under `hmi/backend/tests/sim/`
- [x] §11 docs → Task 16
- [x] §13 small `feat(scope): ...` commits, no co-author trailer → every commit step

No placeholders detected. No "implement later". Every code step shows actual code. Method signatures used in later tasks (`world.read_qpos_deg`, `world.write_ctrl_deg`, `world.set_arm_torque`, `world.actuator_kp_for_joint`, `world.joint_range_deg`, `world.world()` on `ArmManager`, `SimArmHandle.world`, `SimCamera(cfg, world=world)`, `SimLeaderTeleop.start(follower_id=, source=, hz=)`) match across files.

One known soft spot worth surfacing during execution: the `MUJOCO_GL=egl` default may not work on every Linux host. The plan documents the OSMesa fallback in Tasks 1, 4, and 16. If EGL fails in Task 1's verify step, set the test runner's default to OSMesa for the rest of the plan rather than per-task.
