# Haller Unified HMI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing single-file `web_teleop.py` with a unified, two-arm-ready HMI consisting of a FastAPI Python backend (wrapping `lerobot` for the SO-101 arm and `rclpy` for the base) and a Next.js 16 + shadcn/ui frontend. Deployable to both the operator laptop and the Jetson Orin Nano.

**Architecture:** One backend process (Python venv with system-site-packages so it sees ROS 2 Jazzy's rclpy) exposes REST for commands and WebSocket for 20 Hz telemetry. One Next.js app talks to it. Communication is JSON. Frontend is shadcn-themed dark dashboard with a fixed-top-right E-STOP. The legacy `web_teleop.py` is disabled (not deleted) until the new system has run stably for two weeks.

**Tech Stack:**
- Backend: Python 3.12, FastAPI, uvicorn, rclpy (ROS 2 Jazzy), lerobot 0.5.x, pyyaml, pytest
- Frontend: Next.js 16 (App Router), React 19, TypeScript, Tailwind v4, shadcn/ui, zustand, lucide-react, vitest + React Testing Library
- Tooling: pnpm, ruff (Python lint), pyright (Python types optional)
- Deployment: systemd, Node.js 20+ for Next.js standalone build

**Spec:** `docs/superpowers/specs/2026-05-22-haller-unified-hmi-design.md` (commit `bb7ff00`).

---

## File structure (created by this plan)

```
haller_ws/
├── hmi/
│   ├── README.md                                  ← Phase III, Task 17
│   ├── backend/
│   │   ├── pyproject.toml                         ← Task 1
│   │   ├── config.yaml                            ← Task 2
│   │   ├── haller_hmi/
│   │   │   ├── __init__.py                        ← Task 1
│   │   │   ├── config.py                          ← Task 2
│   │   │   ├── safety.py                          ← Task 3
│   │   │   ├── arm.py                             ← Task 4
│   │   │   ├── ros_bridge.py                      ← Task 5
│   │   │   ├── presets.py                         ← Task 6
│   │   │   ├── telemetry.py                       ← Task 7
│   │   │   └── server.py                          ← Task 8
│   │   └── tests/
│   │       ├── __init__.py                        ← Task 3
│   │       ├── test_safety.py                     ← Task 3
│   │       ├── test_arm.py                        ← Task 4
│   │       ├── test_presets.py                    ← Task 6
│   │       ├── test_telemetry.py                  ← Task 7
│   │       └── test_routes.py                     ← Task 8
│   └── frontend/
│       ├── package.json                           ← Task 9
│       ├── tsconfig.json                          ← Task 9
│       ├── next.config.ts                         ← Task 9
│       ├── tailwind.config.ts                     ← Task 9
│       ├── postcss.config.js                      ← Task 9
│       ├── components.json                        ← Task 9 (shadcn init)
│       ├── app/
│       │   ├── globals.css                        ← Task 9
│       │   ├── layout.tsx                         ← Task 10
│       │   ├── page.tsx                           ← Task 11
│       │   ├── arm/[id]/page.tsx                  ← Task 13
│       │   ├── base/page.tsx                      ← Task 14
│       │   └── settings/page.tsx                  ← Task 15
│       ├── components/
│       │   ├── ui/                                ← shadcn primitives via CLI
│       │   ├── ArmPanel.tsx                       ← Task 13
│       │   ├── BasePanel.tsx                      ← Task 14
│       │   ├── EStopButton.tsx                    ← Task 12
│       │   ├── ModeToggle.tsx                     ← Task 12
│       │   ├── JointSlider.tsx                    ← Task 13
│       │   ├── CameraTile.tsx                     ← Task 13
│       │   └── TelemetryBar.tsx                   ← Task 11
│       ├── lib/
│       │   ├── api.ts                             ← Task 11
│       │   ├── telemetry.ts                       ← Task 11
│       │   └── config.ts                          ← Task 9
│       └── __tests__/
│           ├── EStopButton.test.tsx               ← Task 12
│           ├── JointSlider.test.tsx               ← Task 13
│           └── api.test.ts                        ← Task 11
└── scripts/
    ├── run_hmi.sh                                 ← Task 16
    └── haller-hmi.service                         ← Task 16
```

---

## Riskiest steps (heads-up)

1. **Task 5 (ROS bridge inside the same process)** — running `rclpy.spin` inside a FastAPI asyncio loop is non-trivial. The plan uses `rclpy.executors.SingleThreadedExecutor.spin_once` in a background thread; if that misbehaves, the fallback is `rclpy.executors.MultiThreadedExecutor` on its own thread.
2. **Task 1 (Python env)** — `--system-site-packages` venvs sometimes shadow installed wheels with system versions. Always set `PYTHONNOUSERSITE=1` and verify `pip show` reports the venv path. If lerobot's PyTorch conflicts with system Python's PyTorch (unlikely, system Python has none), fall back to RoboStack conda env (notes in Task 1).
3. **Task 16 (systemd unit on the Orin)** — the Orin currently runs the legacy `haller-robot.service`. The new unit MUST not race or conflict. We add `Conflicts=` to the systemd unit and we disable the launch arg in `haller_bringup.launch.py` before enabling the new unit.
4. **Task 18 (parity check)** — base teleop UX must match the old joystick feel; the keyboard handler order and the 10 Hz polling pattern from the original `web_teleop.py` need preserving.

---

# Phase I — Backend (FastAPI in a ROS-aware venv)

## Task 1: Bootstrap the backend Python environment

**Files:**
- Create: `hmi/backend/pyproject.toml`
- Create: `hmi/backend/haller_hmi/__init__.py`
- New venv at: `~/venvs/haller-hmi/`

- [ ] **Step 1: Confirm system-Python and ROS Jazzy availability**

```bash
python3 --version                                                   # expect 3.12.x
ls /opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py    # expect to exist
```

Expected: Python 3.12.3 (or newer 3.12), file path exists.

If `/opt/ros/jazzy` is missing, ROS isn't installed — stop and install `sudo apt install ros-jazzy-ros-base ros-jazzy-rclpy` before continuing.

- [ ] **Step 2: Create the backend venv with system-site-packages**

```bash
# IMPORTANT: source ROS so its libs are on the path the venv will inherit
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages ~/venvs/haller-hmi
echo "Created venv at ~/venvs/haller-hmi"
```

- [ ] **Step 3: Add isolation hook to the venv**

The user's `~/.local/lib/python3.12/site-packages` has a stale `nvidia-nccl-cu12` that breaks PyTorch's CUDA load (see `feedback_lerobot_env_isolation` memory). The same hook pattern used for the conda env applies here.

```bash
mkdir -p ~/venvs/haller-hmi/etc
cat > ~/venvs/haller-hmi/bin/activate-haller-hmi <<'EOF'
#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source ~/venvs/haller-hmi/bin/activate
export PYTHONNOUSERSITE=1
EOF
chmod +x ~/venvs/haller-hmi/bin/activate-haller-hmi
```

Use this from now on: `source ~/venvs/haller-hmi/bin/activate-haller-hmi`.

- [ ] **Step 4: Smoke-test rclpy + isolation**

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
python -c "
import sys
print('sys.path:')
for p in sys.path:
    print(' ', p)
import rclpy
print('rclpy:', rclpy.__file__)
"
```

Expected: `/opt/ros/jazzy/lib/python3.12/site-packages` appears in sys.path, no `~/.local/...` paths, `rclpy:` line points at `/opt/ros/jazzy/...`. If the rclpy line points at `~/.local/...`, `PYTHONNOUSERSITE=1` didn't apply — re-check the activate script.

- [ ] **Step 5: Create the package skeleton**

```bash
mkdir -p ~/haller_ws/hmi/backend/haller_hmi ~/haller_ws/hmi/backend/tests
touch ~/haller_ws/hmi/backend/haller_hmi/__init__.py
touch ~/haller_ws/hmi/backend/tests/__init__.py
```

- [ ] **Step 6: Write `pyproject.toml`**

```toml
# hmi/backend/pyproject.toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "haller-hmi"
version = "0.1.0"
description = "Haller robot unified HMI backend (FastAPI + lerobot + rclpy)"
readme = "../README.md"
requires-python = ">=3.12"
license = { text = "Apache-2.0" }
authors = [{ name = "Oscar Devos" }]
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pyyaml>=6.0",
    "lerobot[feetech]>=0.5,<0.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
    "ruff>=0.7",
]

[project.scripts]
haller-hmi = "haller_hmi.server:run"

[tool.setuptools.packages.find]
where = ["."]
include = ["haller_hmi*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 7: Install the backend in editable mode**

```bash
cd ~/haller_ws/hmi/backend
pip install -e ".[dev]"
```

Expected: lerobot, fastapi, uvicorn, pytest installed inside `~/venvs/haller-hmi`. Wheel cache from the earlier conda install speeds this up significantly.

- [ ] **Step 8: Verify both libs import in one process**

```bash
python -c "import rclpy, lerobot, fastapi; print('rclpy', rclpy.__file__); print('lerobot', lerobot.__version__); print('fastapi', fastapi.__version__)"
```

Expected: rclpy from /opt/ros/jazzy, lerobot 0.5.x, fastapi 0.115+.

- [ ] **Step 9: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/pyproject.toml hmi/backend/haller_hmi/__init__.py hmi/backend/tests/__init__.py
git commit -m "feat(hmi/backend): bootstrap FastAPI backend package skeleton"
```

---

## Task 2: Config loader

**Files:**
- Create: `hmi/backend/config.yaml`
- Create: `hmi/backend/haller_hmi/config.py`
- Modify: `hmi/backend/haller_hmi/__init__.py` (re-export Config)

- [ ] **Step 1: Write `config.yaml` (the default, lives in the repo)**

```yaml
# hmi/backend/config.yaml
arms:
  - id: right
    model: so101_follower
    port: /dev/ttyACM0
    calibration_id: haller_follower  # matches the haller_follower.json on disk
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
  - id: wrist_right
    role: wrist
    arm_id: right
    source: placeholder  # one of: placeholder, mjpeg, webrtc
  - id: base_front
    role: base
    source: placeholder
```

- [ ] **Step 2: Write `haller_hmi/config.py`**

```python
# hmi/backend/haller_hmi/config.py
"""Loads the HMI runtime config from YAML, with $HALLER_HMI_CONFIG override."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class ArmConfig:
    id: str
    model: str
    port: str
    calibration_id: str
    enabled: bool = True


@dataclass
class RosConfig:
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    max_linear: float = 1.0
    max_angular: float = 2.0


@dataclass
class TelemetryConfig:
    hz: float = 20.0


@dataclass
class CameraConfig:
    id: str
    role: str  # "wrist" or "base"
    source: str  # "placeholder" | "mjpeg" | "webrtc"
    arm_id: str | None = None


@dataclass
class Config:
    arms: list[ArmConfig] = field(default_factory=list)
    ros: RosConfig = field(default_factory=RosConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    cameras: list[CameraConfig] = field(default_factory=list)


def load_config(path: Path | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("HALLER_HMI_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(cfg_path.read_text())
    return Config(
        arms=[ArmConfig(**a) for a in raw.get("arms", [])],
        ros=RosConfig(**raw.get("ros", {})),
        telemetry=TelemetryConfig(**raw.get("telemetry", {})),
        cameras=[CameraConfig(**c) for c in raw.get("cameras", [])],
    )
```

- [ ] **Step 3: Add a quick verification test in REPL**

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
cd ~/haller_ws/hmi/backend
python -c "from haller_hmi.config import load_config; c = load_config(); print(c)"
```

Expected: prints a `Config(arms=[ArmConfig(id='right', ...)], ros=RosConfig(...), ...)`.

- [ ] **Step 4: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/config.yaml hmi/backend/haller_hmi/config.py
git commit -m "feat(hmi/backend): add YAML config loader (arms, ROS topics, cameras)"
```

---

## Task 3: Safety primitives (joint-limit clamp + mode guard) — TDD

**Files:**
- Create: `hmi/backend/haller_hmi/safety.py`
- Create: `hmi/backend/tests/test_safety.py`

- [ ] **Step 1: Write the failing test**

```python
# hmi/backend/tests/test_safety.py
import pytest

from haller_hmi.safety import (
    clamp_joint_goal,
    ModeGuard,
    ModeError,
    Mode,
)


def test_clamp_joint_goal_clamps_above_max():
    limits = {"shoulder_pan": (-120.0, 120.0), "gripper": (0.0, 100.0)}
    out = clamp_joint_goal({"shoulder_pan": 200.0, "gripper": 50.0}, limits)
    assert out == {"shoulder_pan": 120.0, "gripper": 50.0}


def test_clamp_joint_goal_clamps_below_min():
    limits = {"shoulder_pan": (-120.0, 120.0)}
    out = clamp_joint_goal({"shoulder_pan": -200.0}, limits)
    assert out == {"shoulder_pan": -120.0}


def test_clamp_joint_goal_ignores_unknown_joint():
    limits = {"shoulder_pan": (-120.0, 120.0)}
    out = clamp_joint_goal({"unknown_joint": 50.0, "shoulder_pan": 0.0}, limits)
    assert out == {"shoulder_pan": 0.0}


def test_mode_guard_blocks_writes_in_auto():
    guard = ModeGuard(initial=Mode.AUTO)
    with pytest.raises(ModeError):
        guard.assert_manual()


def test_mode_guard_allows_writes_in_manual():
    guard = ModeGuard(initial=Mode.MANUAL)
    guard.assert_manual()  # must not raise


def test_mode_guard_transitions():
    guard = ModeGuard(initial=Mode.AUTO)
    guard.set(Mode.MANUAL)
    assert guard.mode is Mode.MANUAL
    guard.set(Mode.STOP)
    with pytest.raises(ModeError):
        guard.assert_manual()
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
cd ~/haller_ws/hmi/backend
pytest tests/test_safety.py -v
```

Expected: ImportError or ModuleNotFoundError because `haller_hmi.safety` doesn't exist yet.

- [ ] **Step 3: Implement `haller_hmi/safety.py`**

```python
# hmi/backend/haller_hmi/safety.py
"""Safety primitives: joint-limit clamps, mode guards, E-STOP orchestrator."""
from __future__ import annotations

import enum
from dataclasses import dataclass


class Mode(str, enum.Enum):
    AUTO = "auto"
    MANUAL = "manual"
    STOP = "stop"


class ModeError(Exception):
    """Raised when an operation is attempted in the wrong mode."""


@dataclass
class ModeGuard:
    """Per-resource mode tracker. Backend uses one per arm and one for the base."""

    mode: Mode = Mode.MANUAL

    def __init__(self, initial: Mode = Mode.MANUAL):
        self.mode = initial

    def set(self, mode: Mode) -> None:
        self.mode = mode

    def assert_manual(self) -> None:
        if self.mode is not Mode.MANUAL:
            raise ModeError(f"resource is in mode {self.mode.value!r}, manual required")


def clamp_joint_goal(
    goal: dict[str, float],
    limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Clamp each joint in `goal` to its (min, max) range; drop unknown joints."""
    out: dict[str, float] = {}
    for joint, value in goal.items():
        if joint not in limits:
            continue
        lo, hi = limits[joint]
        out[joint] = max(lo, min(hi, value))
    return out
```

- [ ] **Step 4: Run the tests, verify they pass**

```bash
pytest tests/test_safety.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/safety.py hmi/backend/tests/test_safety.py
git commit -m "feat(hmi/backend): joint clamp + mode guard primitives (TDD)"
```

---

## Task 4: Arm wrapper (`ArmManager`) — TDD with mocked SO101Follower

**Files:**
- Create: `hmi/backend/haller_hmi/arm.py`
- Create: `hmi/backend/tests/test_arm.py`

- [ ] **Step 1: Write the failing test (mocks lerobot at the boundary)**

```python
# hmi/backend/tests/test_arm.py
from unittest.mock import MagicMock

import pytest

from haller_hmi.arm import ArmManager, ArmHandle
from haller_hmi.config import ArmConfig
from haller_hmi.safety import Mode, ModeError


def _make_handle(monkeypatch) -> ArmHandle:
    # Patch SO101Follower so we never touch real hardware.
    fake_robot = MagicMock()
    fake_robot.bus.motors = {
        "shoulder_pan": MagicMock(id=1),
        "shoulder_lift": MagicMock(id=2),
        "elbow_flex": MagicMock(id=3),
        "wrist_flex": MagicMock(id=4),
        "wrist_roll": MagicMock(id=5),
        "gripper": MagicMock(id=6),
    }
    fake_robot.calibration = {
        "shoulder_pan":  MagicMock(range_min=0,    range_max=4095),
        "shoulder_lift": MagicMock(range_min=0,    range_max=4095),
        "elbow_flex":    MagicMock(range_min=0,    range_max=4095),
        "wrist_flex":    MagicMock(range_min=0,    range_max=4095),
        "wrist_roll":    MagicMock(range_min=0,    range_max=4095),
        "gripper":       MagicMock(range_min=0,    range_max=4095),
    }
    fake_robot.get_observation.return_value = {
        f"{j}.pos": 0.0 for j in fake_robot.bus.motors
    }
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: fake_robot)
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    handle = ArmHandle(cfg, joint_limits_deg={
        "shoulder_pan": (-120, 120), "shoulder_lift": (-100, 100),
        "elbow_flex": (-110, 110), "wrist_flex": (-90, 90),
        "wrist_roll": (-180, 180), "gripper": (0, 100),
    })
    handle.robot = fake_robot
    return handle


def test_send_goal_in_auto_mode_raises(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.AUTO)
    with pytest.raises(ModeError):
        handle.send_goal({"shoulder_pan": 30.0})


def test_send_goal_clamps_and_calls_lerobot(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    sent = handle.send_goal({"shoulder_pan": 999.0, "gripper": -50.0, "unknown": 10.0})
    assert sent == {"shoulder_pan": 120.0, "gripper": 0.0}
    handle.robot.send_action.assert_called_once_with({"shoulder_pan.pos": 120.0,
                                                      "gripper.pos": 0.0})


def test_state_snapshot_returns_joints_with_limits(monkeypatch):
    handle = _make_handle(monkeypatch)
    snap = handle.state_snapshot()
    assert set(snap["joints"].keys()) == {
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    }
    assert snap["joints"]["shoulder_pan"]["min"] == -120
    assert snap["joints"]["shoulder_pan"]["max"] == 120
    assert snap["mode"] in {"auto", "manual", "stop"}


def test_arm_manager_lookup_by_id(monkeypatch):
    cfg_right = ArmConfig(id="right", model="so101_follower",
                          port="/dev/null", calibration_id="haller_follower")
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr("haller_hmi.arm.ArmHandle._load_joint_limits",
                        lambda self: {"gripper": (0, 100)})
    mgr = ArmManager([cfg_right])
    assert mgr["right"].config.id == "right"
    with pytest.raises(KeyError):
        _ = mgr["left"]
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_arm.py -v
```

Expected: ImportError, `haller_hmi.arm` not found.

- [ ] **Step 3: Implement `haller_hmi/arm.py`**

```python
# hmi/backend/haller_hmi/arm.py
"""Per-arm wrapper around `lerobot.robots.so_follower.SO101Follower`.

The HMI's safety surface lives on top of lerobot's raw API:
  - mode gating (only Mode.MANUAL accepts goals from the HMI)
  - joint-limit clamping in DEGREES against the calibration
  - keys translated between HMI ("shoulder_pan": deg) and lerobot ("shoulder_pan.pos": deg)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .config import ArmConfig
from .safety import Mode, ModeGuard, clamp_joint_goal

logger = logging.getLogger(__name__)


# Conservative defaults if calibration metadata doesn't expose explicit deg ranges;
# we derive these per-joint from each motor's calibrated range converted to degrees.
TICKS_PER_REV = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_REV


@dataclass
class ArmHandle:
    config: ArmConfig
    joint_limits_deg: dict[str, tuple[float, float]] = field(default_factory=dict)
    guard: ModeGuard = field(default_factory=lambda: ModeGuard(Mode.MANUAL))
    robot: SO101Follower | None = None

    def connect(self) -> None:
        cfg = SO101FollowerConfig(
            port=self.config.port,
            id=self.config.calibration_id,
            use_degrees=True,
        )
        self.robot = SO101Follower(cfg)
        self.robot.connect(calibrate=True)
        # Load joint limits from the now-loaded calibration.
        self.joint_limits_deg = self._load_joint_limits()
        logger.info(
            "arm %s connected; joint limits (deg): %s",
            self.config.id,
            self.joint_limits_deg,
        )

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def _load_joint_limits(self) -> dict[str, tuple[float, float]]:
        # SO101Follower stores calibration as dict[motor_name, MotorCalibration]
        # with range_min/range_max in raw ticks. Convert to centered degrees by
        # subtracting the motor's home (encoded via homing_offset).
        out: dict[str, tuple[float, float]] = {}
        if self.robot is None or not self.robot.calibration:
            return out
        for motor, mc in self.robot.calibration.items():
            center = (mc.range_min + mc.range_max) / 2.0
            min_deg = (mc.range_min - center) * DEG_PER_TICK
            max_deg = (mc.range_max - center) * DEG_PER_TICK
            out[motor] = (min_deg, max_deg)
        return out

    def send_goal(self, goal_deg: dict[str, float]) -> dict[str, float]:
        self.guard.assert_manual()
        clamped = clamp_joint_goal(goal_deg, self.joint_limits_deg)
        # lerobot expects keys suffixed with ".pos"
        action = {f"{j}.pos": v for j, v in clamped.items()}
        assert self.robot is not None
        self.robot.send_action(action)
        return clamped

    def disable_torque(self) -> None:
        if self.robot is not None:
            self.robot.bus.disable_torque()

    def state_snapshot(self) -> dict:
        assert self.robot is not None
        obs = self.robot.get_observation()
        joints = {}
        for joint, (lo, hi) in self.joint_limits_deg.items():
            joints[joint] = {
                "pos": float(obs.get(f"{joint}.pos", 0.0)),
                "min": float(lo),
                "max": float(hi),
                "torque": True,  # lerobot doesn't expose per-joint torque cheaply; placeholder
            }
        return {
            "mode": self.guard.mode.value,
            "joints": joints,
        }


class ArmManager:
    """Lookup-by-id collection of ArmHandle instances."""

    def __init__(self, arm_configs: list[ArmConfig]):
        self._handles: dict[str, ArmHandle] = {}
        for cfg in arm_configs:
            if not cfg.enabled:
                continue
            self._handles[cfg.id] = ArmHandle(cfg)

    def connect_all(self) -> None:
        for handle in self._handles.values():
            handle.connect()

    def disconnect_all(self) -> None:
        for handle in self._handles.values():
            handle.disconnect()

    def __getitem__(self, arm_id: str) -> ArmHandle:
        if arm_id not in self._handles:
            raise KeyError(f"unknown arm id {arm_id!r}; known: {list(self._handles)}")
        return self._handles[arm_id]

    def values(self):
        return self._handles.values()

    def keys(self):
        return self._handles.keys()
```

- [ ] **Step 4: Run, verify passes**

```bash
pytest tests/test_arm.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/arm.py hmi/backend/tests/test_arm.py
git commit -m "feat(hmi/backend): ArmManager + per-arm safety on top of SO101Follower (TDD)"
```

---

## Task 5: ROS 2 bridge (`RosBridge`)

**Files:**
- Create: `hmi/backend/haller_hmi/ros_bridge.py`

This task does NOT have a unit test — rclpy is hard to mock cleanly and the behaviour is verified end-to-end in Task 8's `test_routes.py` (which mocks rclpy at module load) and Task 18's parity check against real ROS.

- [ ] **Step 1: Implement `haller_hmi/ros_bridge.py`**

```python
# hmi/backend/haller_hmi/ros_bridge.py
"""ROS 2 bridge that runs in a background thread.

Responsibilities:
  - publish geometry_msgs/Twist on /cmd_vel from POST /base/cmd_vel
  - subscribe to /odom and /scan, expose latest snapshot for telemetry frames
  - keep the rclpy executor spinning without blocking the FastAPI event loop

The executor lives on a dedicated thread. `latest()` is a thread-safe read of the
last odom and scan messages (held as plain dicts so the JSON serializer is happy).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from .config import RosConfig

logger = logging.getLogger(__name__)


@dataclass
class BaseSnapshot:
    linear: float = 0.0
    angular: float = 0.0
    odom: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0})
    scan_min_range: float | None = None


class _HmiNode(Node):
    def __init__(self, cfg: RosConfig, snap: BaseSnapshot, lock: threading.Lock):
        super().__init__("haller_hmi")
        self._cfg = cfg
        self._snap = snap
        self._lock = lock
        self._pub = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self.create_subscription(Odometry, cfg.odom_topic, self._on_odom, 10)
        self.create_subscription(LaserScan, cfg.scan_topic, self._on_scan, 10)

    def publish_cmd_vel(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._pub.publish(msg)
        with self._lock:
            self._snap.linear = float(linear)
            self._snap.angular = float(angular)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # quaternion → yaw (small util to avoid pulling tf_transformations)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        import math
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self._lock:
            self._snap.odom = {"x": float(p.x), "y": float(p.y), "yaw": float(yaw)}

    def _on_scan(self, msg: LaserScan) -> None:
        # min finite range; +inf if all infinite
        rng = [r for r in msg.ranges if (r > 0.0 and r < float("inf"))]
        with self._lock:
            self._snap.scan_min_range = float(min(rng)) if rng else None


class RosBridge:
    def __init__(self, cfg: RosConfig):
        self._cfg = cfg
        self._snap = BaseSnapshot()
        self._lock = threading.Lock()
        self._node: _HmiNode | None = None
        self._exec: SingleThreadedExecutor | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        rclpy.init(args=None)
        self._node = _HmiNode(self._cfg, self._snap, self._lock)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, name="haller-hmi-ros", daemon=True)
        self._thread.start()
        logger.info("ROS bridge started; node=%s", self._node.get_name())

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._exec is not None:
            self._exec.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        rclpy.shutdown()

    def _spin(self) -> None:
        assert self._exec is not None
        while not self._stop.is_set():
            self._exec.spin_once(timeout_sec=0.1)

    # public API used by FastAPI routes / telemetry

    def publish_cmd_vel(self, linear: float, angular: float) -> tuple[float, float]:
        # clamp to configured maxes
        linear = max(-self._cfg.max_linear, min(self._cfg.max_linear, float(linear)))
        angular = max(-self._cfg.max_angular, min(self._cfg.max_angular, float(angular)))
        assert self._node is not None
        self._node.publish_cmd_vel(linear, angular)
        return (linear, angular)

    def zero_cmd_vel(self) -> None:
        self.publish_cmd_vel(0.0, 0.0)

    def snapshot(self) -> BaseSnapshot:
        with self._lock:
            # return a copy so the reader can't see torn state
            return BaseSnapshot(
                linear=self._snap.linear,
                angular=self._snap.angular,
                odom=dict(self._snap.odom),
                scan_min_range=self._snap.scan_min_range,
            )
```

- [ ] **Step 2: Manual smoke test against running ROS**

In one terminal:

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
ros2 run demo_nodes_cpp talker &           # any node so executor has something to do
ros2 topic echo /cmd_vel &
```

In the venv shell:

```bash
python -c "
from haller_hmi.config import load_config
from haller_hmi.ros_bridge import RosBridge
import time
cfg = load_config()
bridge = RosBridge(cfg.ros)
bridge.start()
print('publishing 0.2, 0.0 ...')
bridge.publish_cmd_vel(0.2, 0.0)
time.sleep(0.5)
print('snap:', bridge.snapshot())
bridge.stop()
print('stopped')
"
```

Expected: `ros2 topic echo /cmd_vel` prints a Twist with linear.x=0.2. Process exits cleanly.

If `ros2` isn't on PATH, the `source /opt/ros/jazzy/setup.bash` in the activate-haller-hmi hook didn't run — re-source it manually.

- [ ] **Step 3: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/ros_bridge.py
git commit -m "feat(hmi/backend): ROS 2 bridge thread (Twist pub + odom/scan subs)"
```

---

## Task 6: Presets store — TDD

**Files:**
- Create: `hmi/backend/haller_hmi/presets.py`
- Create: `hmi/backend/tests/test_presets.py`

- [ ] **Step 1: Write the failing test**

```python
# hmi/backend/tests/test_presets.py
import json

import pytest

from haller_hmi.presets import PresetStore, PresetNotFound


def test_save_and_load_roundtrip(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    store.save("home", "right", {"shoulder_pan": 0.0, "gripper": 0.0})
    store.save("ready", "right", {"shoulder_pan": 30.0})
    out = store.get("home", "right")
    assert out == {"shoulder_pan": 0.0, "gripper": 0.0}
    out2 = store.get("ready", "right")
    assert out2 == {"shoulder_pan": 30.0}


def test_missing_preset_raises(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    with pytest.raises(PresetNotFound):
        store.get("home", "right")


def test_list_returns_all_for_arm(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    store.save("home", "right", {"shoulder_pan": 0.0})
    store.save("home", "left", {"shoulder_pan": 0.0})
    assert sorted(store.list("right")) == ["home"]
    assert sorted(store.list("left")) == ["home"]


def test_file_persists_on_disk(tmp_path):
    path = tmp_path / "presets.json"
    store1 = PresetStore(path)
    store1.save("home", "right", {"shoulder_pan": 12.5})
    # re-open and read
    store2 = PresetStore(path)
    assert store2.get("home", "right") == {"shoulder_pan": 12.5}
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_presets.py -v
```

- [ ] **Step 3: Implement `haller_hmi/presets.py`**

```python
# hmi/backend/haller_hmi/presets.py
"""On-disk JSON preset store, keyed by (arm_id, name)."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PRESETS_PATH = Path.home() / ".haller" / "presets.json"


class PresetNotFound(Exception):
    pass


class PresetStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or DEFAULT_PRESETS_PATH)
        self._data: dict[str, dict[str, dict[str, float]]] = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def save(self, name: str, arm_id: str, joints_deg: dict[str, float]) -> None:
        self._data.setdefault(arm_id, {})[name] = dict(joints_deg)
        self._write()

    def get(self, name: str, arm_id: str) -> dict[str, float]:
        try:
            return dict(self._data[arm_id][name])
        except KeyError as e:
            raise PresetNotFound(f"no preset {name!r} for arm {arm_id!r}") from e

    def list(self, arm_id: str) -> list[str]:
        return list(self._data.get(arm_id, {}).keys())
```

- [ ] **Step 4: Run, verify passes**

```bash
pytest tests/test_presets.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/presets.py hmi/backend/tests/test_presets.py
git commit -m "feat(hmi/backend): on-disk preset store keyed by (arm, name) (TDD)"
```

---

## Task 7: Telemetry broadcaster — TDD

**Files:**
- Create: `hmi/backend/haller_hmi/telemetry.py`
- Create: `hmi/backend/tests/test_telemetry.py`

The broadcaster runs an asyncio task that polls `ArmManager` + `RosBridge` at `hz` and emits a frame dict to all subscribed asyncio queues.

- [ ] **Step 1: Write the failing test**

```python
# hmi/backend/tests/test_telemetry.py
import asyncio
from unittest.mock import MagicMock

import pytest

from haller_hmi.telemetry import TelemetryBroadcaster


@pytest.mark.asyncio
async def test_broadcaster_emits_frames():
    arm = MagicMock()
    arm.state_snapshot.return_value = {
        "mode": "manual",
        "joints": {"gripper": {"pos": 0.0, "min": 0.0, "max": 100.0, "torque": True}},
    }
    arms = MagicMock()
    arms.keys.return_value = ["right"]
    arms.__getitem__.return_value = arm

    ros = MagicMock()
    snap = MagicMock(linear=0.0, angular=0.0, odom={"x": 1.0, "y": 0.0, "yaw": 0.0}, scan_min_range=2.0)
    ros.snapshot.return_value = snap

    bcast = TelemetryBroadcaster(arms, ros, hz=200.0)  # high hz for fast test
    bcast.start()
    try:
        sub = bcast.subscribe()
        frame = await asyncio.wait_for(sub.__anext__(), timeout=0.2)
    finally:
        await bcast.stop()
    assert "t" in frame
    assert frame["base"]["odom"]["x"] == 1.0
    assert "right" in frame["arms"]
    assert frame["arms"]["right"]["joints"]["gripper"]["pos"] == 0.0


@pytest.mark.asyncio
async def test_multiple_subscribers_get_same_frame():
    arms = MagicMock()
    arms.keys.return_value = []
    ros = MagicMock()
    ros.snapshot.return_value = MagicMock(linear=0.0, angular=0.0, odom={}, scan_min_range=None)

    bcast = TelemetryBroadcaster(arms, ros, hz=200.0)
    bcast.start()
    try:
        s1 = bcast.subscribe()
        s2 = bcast.subscribe()
        f1 = await asyncio.wait_for(s1.__anext__(), timeout=0.2)
        f2 = await asyncio.wait_for(s2.__anext__(), timeout=0.2)
    finally:
        await bcast.stop()
    # both subscribers see a frame (timing may differ by one tick, that's fine)
    assert "t" in f1 and "t" in f2
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_telemetry.py -v
```

- [ ] **Step 3: Implement `haller_hmi/telemetry.py`**

```python
# hmi/backend/haller_hmi/telemetry.py
"""Broadcasts a state frame to N subscribers at a fixed rate."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class TelemetryBroadcaster:
    def __init__(self, arms, ros, hz: float = 20.0):
        self._arms = arms
        self._ros = ros
        self._period = 1.0 / hz
        self._subscribers: list[asyncio.Queue] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.get_event_loop().create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        self._task = None

    def subscribe(self) -> AsyncIterator[dict]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._subscribers.append(queue)
        async def gen():
            try:
                while True:
                    frame = await queue.get()
                    yield frame
            finally:
                self._subscribers.remove(queue)
        return gen()

    def _build_frame(self) -> dict:
        base_snap = self._ros.snapshot()
        frame = {
            "t": time.time(),
            "base": {
                "linear": base_snap.linear,
                "angular": base_snap.angular,
                "odom": dict(base_snap.odom),
                "scan_min_range": base_snap.scan_min_range,
            },
            "arms": {},
            "alerts": [],
        }
        for arm_id in self._arms.keys():
            try:
                frame["arms"][arm_id] = self._arms[arm_id].state_snapshot()
            except Exception as e:
                logger.warning("arm %s telemetry failed: %s", arm_id, e)
                frame["alerts"].append({
                    "level": "warn",
                    "code": "arm_telemetry_failed",
                    "message": str(e),
                    "source": f"arm:{arm_id}",
                })
        return frame

    async def _run(self) -> None:
        while not self._stop.is_set():
            tick_start = time.perf_counter()
            try:
                frame = self._build_frame()
                for q in list(self._subscribers):
                    if q.full():
                        # drop oldest to keep latency bounded
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    q.put_nowait(frame)
            except Exception as e:
                logger.exception("telemetry tick failed: %s", e)
            elapsed = time.perf_counter() - tick_start
            await asyncio.sleep(max(0.0, self._period - elapsed))
```

- [ ] **Step 4: Run, verify passes**

```bash
pytest tests/test_telemetry.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/telemetry.py hmi/backend/tests/test_telemetry.py
git commit -m "feat(hmi/backend): 20 Hz telemetry broadcaster (TDD)"
```

---

## Task 8: FastAPI app — TDD against TestClient with mocked rclpy/lerobot

**Files:**
- Create: `hmi/backend/haller_hmi/server.py`
- Create: `hmi/backend/tests/test_routes.py`

- [ ] **Step 1: Write the failing test (mocks ArmManager, RosBridge, presets at the module boundary)**

```python
# hmi/backend/tests/test_routes.py
import json
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_with_mocks(monkeypatch, tmp_path):
    # Mock ArmManager + RosBridge + PresetStore before importing server
    arm = MagicMock()
    arm.send_goal.return_value = {"shoulder_pan": 30.0}
    arm.state_snapshot.return_value = {
        "mode": "manual",
        "joints": {"shoulder_pan": {"pos": 0.0, "min": -120.0, "max": 120.0, "torque": True}},
    }
    arm.config = MagicMock(id="right")
    arm.guard = MagicMock(mode=MagicMock(value="manual"))

    arm_mgr = MagicMock()
    arm_mgr.keys.return_value = ["right"]
    def _lookup(key: str):
        if key == "right":
            return arm
        raise KeyError(f"unknown arm id {key!r}")
    arm_mgr.__getitem__.side_effect = _lookup
    arm_mgr.values.return_value = [arm]

    ros = MagicMock()
    ros.publish_cmd_vel.return_value = (0.1, 0.2)

    monkeypatch.setattr("haller_hmi.server.ArmManager", lambda *a, **kw: arm_mgr)
    monkeypatch.setattr("haller_hmi.server.RosBridge", lambda *a, **kw: ros)
    monkeypatch.setattr(
        "haller_hmi.server.PresetStore",
        lambda *a, **kw: MagicMock(get=lambda name, arm: {"shoulder_pan": 0.0},
                                   save=MagicMock(),
                                   list=lambda arm: ["home"]),
    )

    import importlib
    import haller_hmi.server as srv_mod
    importlib.reload(srv_mod)
    return TestClient(srv_mod.app)


def test_get_config(app_with_mocks):
    r = app_with_mocks.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "arms" in body
    assert "version" in body


def test_post_base_cmd_vel(app_with_mocks):
    r = app_with_mocks.post("/base/cmd_vel", json={"linear": 0.1, "angular": 0.2})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_post_arm_goal(app_with_mocks):
    r = app_with_mocks.post("/arm/right/goal", json={"shoulder_pan": 30.0})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "sent" in body


def test_post_arm_mode(app_with_mocks):
    r = app_with_mocks.post("/arm/right/mode", json={"mode": "auto"})
    assert r.status_code == 200


def test_post_arm_mode_invalid(app_with_mocks):
    r = app_with_mocks.post("/arm/right/mode", json={"mode": "blender"})
    assert r.status_code == 400


def test_post_arm_preset(app_with_mocks):
    r = app_with_mocks.post("/arm/right/preset", json={"name": "home"})
    assert r.status_code == 200


def test_post_estop(app_with_mocks):
    r = app_with_mocks.post("/estop", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_unknown_arm_returns_404(app_with_mocks):
    # The mock raises KeyError for any id != "right"; _arm_or_404 converts it to 404.
    r = app_with_mocks.post("/arm/left/goal", json={"shoulder_pan": 0.0})
    assert r.status_code == 404


def test_get_health(app_with_mocks):
    r = app_with_mocks.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_routes.py -v
```

- [ ] **Step 3: Implement `haller_hmi/server.py`**

```python
# hmi/backend/haller_hmi/server.py
"""FastAPI app. The only place that ties lerobot, ROS, presets, and HTTP together."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from .arm import ArmManager
from .config import load_config
from .presets import PresetNotFound, PresetStore
from .ros_bridge import RosBridge
from .safety import Mode, ModeError
from .telemetry import TelemetryBroadcaster

logger = logging.getLogger(__name__)

VERSION = "0.1.0"

# Globals — wired in lifespan
cfg = load_config()
arms = ArmManager(cfg.arms)
ros = RosBridge(cfg.ros)
presets = PresetStore()
telemetry: TelemetryBroadcaster | None = None


# ---- request schemas -----------------------------------------------------

class CmdVel(BaseModel):
    linear: float
    angular: float


class ArmGoal(BaseModel):
    model_config = ConfigDict(extra="allow")  # any subset of joint names

    # No declared fields — the joint dict comes through as `model_extra`
    # so we read it via `.__dict__`.


class ArmModeBody(BaseModel):
    mode: str = Field(pattern="^(auto|manual|stop)$")


class PresetBody(BaseModel):
    name: str


# ---- lifespan ------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global telemetry
    logger.info("haller-hmi backend starting (version %s)", VERSION)
    arms.connect_all()
    ros.start()
    telemetry = TelemetryBroadcaster(arms, ros, hz=cfg.telemetry.hz)
    telemetry.start()
    yield
    logger.info("haller-hmi backend shutting down")
    if telemetry is not None:
        await telemetry.stop()
    arms.disconnect_all()
    ros.stop()


app = FastAPI(title="haller-hmi", version=VERSION, lifespan=_lifespan)


# ---- helpers -------------------------------------------------------------

def _arm_or_404(arm_id: str):
    try:
        return arms[arm_id]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---- routes --------------------------------------------------------------

@app.get("/health")
def get_health():
    return {"status": "ok", "arms_online": len(list(arms.keys())), "base_online": True}


@app.get("/config")
def get_config():
    return {
        "version": VERSION,
        "arms": [
            {
                "id": h.config.id,
                "model": h.config.model,
                "port": h.config.port,
                "mode": h.guard.mode.value,
            }
            for h in arms.values()
        ],
        "cameras": [c.__dict__ for c in cfg.cameras],
    }


@app.post("/base/cmd_vel")
def post_cmd_vel(body: CmdVel):
    sent = ros.publish_cmd_vel(body.linear, body.angular)
    return {"ok": True, "linear": sent[0], "angular": sent[1]}


@app.post("/arm/{arm_id}/goal")
async def post_arm_goal(arm_id: str, body: dict[str, float]):
    handle = _arm_or_404(arm_id)
    try:
        clamped = handle.send_goal(body)
    except ModeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "sent": clamped}


@app.post("/arm/{arm_id}/mode")
async def post_arm_mode(arm_id: str, body: ArmModeBody):
    handle = _arm_or_404(arm_id)
    new_mode = Mode(body.mode)
    handle.guard.set(new_mode)
    if new_mode is Mode.STOP:
        handle.disable_torque()
    return {"ok": True, "mode": new_mode.value}


@app.post("/arm/{arm_id}/preset")
async def post_arm_preset(arm_id: str, body: PresetBody):
    handle = _arm_or_404(arm_id)
    try:
        goal = presets.get(body.name, arm_id)
    except PresetNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        clamped = handle.send_goal(goal)
    except ModeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "sent": clamped}


@app.post("/arm/{arm_id}/preset/record")
async def post_arm_preset_record(arm_id: str, body: PresetBody):
    handle = _arm_or_404(arm_id)
    snap = handle.state_snapshot()
    current = {j: v["pos"] for j, v in snap["joints"].items()}
    presets.save(body.name, arm_id, current)
    return {"ok": True, "saved": current}


@app.post("/estop")
async def post_estop():
    logger.warning("E-STOP triggered")
    for handle in arms.values():
        handle.disable_torque()
        handle.guard.set(Mode.STOP)
    ros.zero_cmd_vel()
    return {"ok": True}


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    assert telemetry is not None
    sub = telemetry.subscribe()
    try:
        async for frame in sub:
            await ws.send_json(frame)
    except WebSocketDisconnect:
        return


def run() -> None:
    """Entry point for the `haller-hmi` console script."""
    import uvicorn
    uvicorn.run("haller_hmi.server:app", host="0.0.0.0", port=8000, log_level="info")
```

- [ ] **Step 4: Run, verify passes**

```bash
pytest tests/test_routes.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Run the full backend test suite**

```bash
pytest -v
```

Expected: 25 passed (6 safety + 4 arm + 4 presets + 2 telemetry + 9 routes).

- [ ] **Step 6: Manual smoke against real arm (optional but recommended before frontend work)**

Make sure no other process holds `/dev/ttyACM0`, then in one terminal:

```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
sudo chmod 666 /dev/ttyACM0
haller-hmi   # uses the console script defined in pyproject.toml
```

In another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/config | jq .
curl http://localhost:8000/arm/right/mode -d '{"mode":"manual"}' -H "Content-Type: application/json"
curl http://localhost:8000/arm/right/goal -d '{"gripper": 30.0}' -H "Content-Type: application/json"
# arm should move the gripper a bit
curl http://localhost:8000/estop -X POST
# torque should release; arm becomes back-drivable
```

If everything works, kill the server (Ctrl-C).

- [ ] **Step 7: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/server.py hmi/backend/tests/test_routes.py
git commit -m "feat(hmi/backend): FastAPI app with arm + base + telemetry + estop routes (TDD)"
```

---

# Phase II — Frontend (Next.js 16 + shadcn/ui)

## Task 9: Scaffold Next.js + shadcn

**Files:**
- New directory: `hmi/frontend/`
- Creates many files via the create-next-app + shadcn CLIs.

- [ ] **Step 1: Confirm Node toolchain**

```bash
node --version    # expect 20+
corepack enable
pnpm --version    # expect 9+; if missing: corepack prepare pnpm@latest --activate
```

- [ ] **Step 2: Scaffold the Next.js app**

```bash
cd ~/haller_ws/hmi
pnpm create next-app@latest frontend --typescript --tailwind --eslint --app --import-alias "@/*" --src-dir=false --use-pnpm --turbopack
```

Defaults to accept: yes to Tailwind, yes to App Router, yes to `@/*` alias.

- [ ] **Step 3: Initialize shadcn/ui**

```bash
cd ~/haller_ws/hmi/frontend
pnpm dlx shadcn@latest init -d
# accept defaults: TypeScript, Default theme, Slate base color, css variables: yes
```

This writes `components.json`, `app/globals.css` with CSS vars, and creates `components/ui/`.

- [ ] **Step 4: Add the shadcn primitives we need**

```bash
pnpm dlx shadcn@latest add button card slider switch toggle badge separator input toast sonner
```

- [ ] **Step 5: Install runtime + dev deps we'll need**

```bash
pnpm add zustand lucide-react
pnpm add -D vitest @testing-library/react @testing-library/dom jsdom @vitejs/plugin-react
```

- [ ] **Step 6: Add `lib/config.ts`**

```ts
// hmi/frontend/lib/config.ts
export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export const WS_URL =
  BACKEND_URL.replace(/^http/, "ws") + "/ws/telemetry";
```

- [ ] **Step 7: Smoke-test the dev server**

```bash
pnpm dev
# open http://localhost:3000 — should see the default Next.js page
```

Ctrl-C to stop.

- [ ] **Step 8: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend
git commit -m "feat(hmi/frontend): scaffold Next.js 16 + Tailwind + shadcn/ui"
```

---

## Task 10: Root layout with fixed E-STOP slot

**Files:**
- Modify: `hmi/frontend/app/layout.tsx`
- Create placeholder: `hmi/frontend/components/EStopButton.tsx` (real impl in Task 12)

- [ ] **Step 1: Create EStopButton placeholder**

```tsx
// hmi/frontend/components/EStopButton.tsx
"use client";
import { Button } from "@/components/ui/button";

export function EStopButton({ className }: { className?: string }) {
  return (
    <Button
      variant="destructive"
      size="lg"
      className={`rounded-full h-16 w-16 text-xs font-bold ${className ?? ""}`}
      aria-label="Emergency stop"
    >
      E-STOP
    </Button>
  );
}
```

- [ ] **Step 2: Update `app/layout.tsx`**

```tsx
// hmi/frontend/app/layout.tsx
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { EStopButton } from "@/components/EStopButton";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Haller HMI",
  description: "Unified control surface for the Haller robot",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground min-h-screen`}>
        <EStopButton className="fixed top-3 right-3 z-50" />
        {children}
        <Toaster richColors closeButton />
      </body>
    </html>
  );
}
```

- [ ] **Step 3: Verify visually**

```bash
cd ~/haller_ws/hmi/frontend && pnpm dev
# open http://localhost:3000 — should see dark background, big red E-STOP top-right
```

- [ ] **Step 4: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/app/layout.tsx hmi/frontend/components/EStopButton.tsx
git commit -m "feat(hmi/frontend): dark root layout + fixed E-STOP slot"
```

---

## Task 11: Backend client + telemetry store + TelemetryBar + dashboard skeleton

**Files:**
- Create: `hmi/frontend/lib/api.ts`
- Create: `hmi/frontend/lib/telemetry.ts`
- Create: `hmi/frontend/__tests__/api.test.ts`
- Create: `hmi/frontend/components/TelemetryBar.tsx`
- Modify: `hmi/frontend/app/page.tsx`

- [ ] **Step 1: Write the failing test (api.ts)**

```ts
// hmi/frontend/__tests__/api.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { postJson, getJson } from "../lib/api";

describe("postJson", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("posts JSON and parses response", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 })
    );
    const out = await postJson<{ ok: boolean }>("/foo", { a: 1 });
    expect(out).toEqual({ ok: true });
    const call = fetchSpy.mock.calls[0];
    expect((call[1] as RequestInit).method).toBe("POST");
    expect((call[1] as RequestInit).body).toBe(JSON.stringify({ a: 1 }));
  });

  it("throws on non-2xx", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: "bad" }), { status: 409 })
    );
    await expect(postJson("/foo", {})).rejects.toThrow(/bad|409/);
  });
});

describe("getJson", () => {
  it("returns parsed body", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ x: 1 }), { status: 200 })
    );
    expect(await getJson<{ x: number }>("/x")).toEqual({ x: 1 });
  });
});
```

- [ ] **Step 2: Configure vitest**

Create `hmi/frontend/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: [],
    globals: true,
  },
  resolve: {
    alias: { "@": "/" },
  },
});
```

Add to `package.json` scripts: `"test": "vitest run"`.

- [ ] **Step 3: Implement `lib/api.ts`**

```ts
// hmi/frontend/lib/api.ts
import { BACKEND_URL } from "./config";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = body.error ?? body.detail ?? detail;
    } catch {
      /* ignore parse error */
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handle<T>(res);
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`);
  return handle<T>(res);
}

// Convenience wrappers
export type ArmGoal = Record<string, number>;

export const api = {
  health: () => getJson<{ status: string }>("/health"),
  config: () => getJson<{
    version: string;
    arms: { id: string; model: string; port: string; mode: string }[];
    cameras: { id: string; role: string; source: string; arm_id?: string }[];
  }>("/config"),
  cmdVel: (linear: number, angular: number) =>
    postJson<{ ok: true; linear: number; angular: number }>("/base/cmd_vel", { linear, angular }),
  armGoal: (armId: string, goal: ArmGoal) =>
    postJson<{ ok: true; sent: ArmGoal }>(`/arm/${armId}/goal`, goal),
  armMode: (armId: string, mode: "auto" | "manual" | "stop") =>
    postJson<{ ok: true; mode: string }>(`/arm/${armId}/mode`, { mode }),
  armPreset: (armId: string, name: string) =>
    postJson<{ ok: true; sent: ArmGoal }>(`/arm/${armId}/preset`, { name }),
  armPresetRecord: (armId: string, name: string) =>
    postJson<{ ok: true; saved: ArmGoal }>(`/arm/${armId}/preset/record`, { name }),
  estop: () => postJson<{ ok: true }>("/estop", {}),
};
```

- [ ] **Step 4: Run the API tests, verify they pass**

```bash
cd ~/haller_ws/hmi/frontend
pnpm test
```

Expected: 3 passed.

- [ ] **Step 5: Implement `lib/telemetry.ts` (WebSocket → zustand)**

```ts
// hmi/frontend/lib/telemetry.ts
"use client";
import { create } from "zustand";
import { WS_URL } from "./config";

export type JointState = {
  pos: number;
  min: number;
  max: number;
  torque: boolean;
};

export type ArmState = {
  mode: "auto" | "manual" | "stop";
  joints: Record<string, JointState>;
};

export type BaseState = {
  linear: number;
  angular: number;
  odom: { x: number; y: number; yaw: number };
  scan_min_range: number | null;
};

export type TelemetryFrame = {
  t: number;
  base: BaseState;
  arms: Record<string, ArmState>;
  alerts: { level: string; code: string; message: string; source: string }[];
};

type Store = {
  connected: boolean;
  lastFrame: TelemetryFrame | null;
  start: () => void;
  stop: () => void;
};

let socket: WebSocket | null = null;

export const useTelemetry = create<Store>((set, get) => ({
  connected: false,
  lastFrame: null,
  start: () => {
    if (socket) return;
    const ws = new WebSocket(WS_URL);
    socket = ws;
    ws.addEventListener("open", () => set({ connected: true }));
    ws.addEventListener("close", () => {
      set({ connected: false });
      socket = null;
      // simple reconnect after 1 s
      setTimeout(() => get().start(), 1000);
    });
    ws.addEventListener("message", (e) => {
      try {
        const frame = JSON.parse(e.data) as TelemetryFrame;
        set({ lastFrame: frame });
      } catch {
        /* drop malformed frame */
      }
    });
  },
  stop: () => {
    socket?.close();
    socket = null;
  },
}));
```

- [ ] **Step 6: Implement `components/TelemetryBar.tsx`**

```tsx
// hmi/frontend/components/TelemetryBar.tsx
"use client";
import { useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { useTelemetry } from "@/lib/telemetry";

export function TelemetryBar() {
  const { connected, lastFrame, start } = useTelemetry();
  useEffect(() => { start(); }, [start]);
  const t = lastFrame?.t;
  return (
    <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground">
      <Badge variant={connected ? "default" : "destructive"}>
        {connected ? "live" : "disconnected"}
      </Badge>
      {t ? <span>t={new Date(t * 1000).toLocaleTimeString()}</span> : null}
      {lastFrame ? (
        <>
          <span>v={lastFrame.base.linear.toFixed(2)} m/s</span>
          <span>ω={lastFrame.base.angular.toFixed(2)} rad/s</span>
          <span>x={lastFrame.base.odom.x.toFixed(2)}</span>
          <span>y={lastFrame.base.odom.y.toFixed(2)}</span>
        </>
      ) : null}
    </div>
  );
}
```

- [ ] **Step 7: Implement dashboard skeleton `app/page.tsx`**

```tsx
// hmi/frontend/app/page.tsx
"use client";
import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { TelemetryBar } from "@/components/TelemetryBar";
import { api } from "@/lib/api";

export default function Dashboard() {
  const [cfg, setCfg] = useState<Awaited<ReturnType<typeof api.config>> | null>(null);
  useEffect(() => { api.config().then(setCfg).catch(console.error); }, []);
  if (!cfg) return <div className="p-3 text-sm">Loading config…</div>;
  return (
    <main className="p-3 space-y-3">
      <div className="flex items-baseline justify-between">
        <h1 className="text-lg font-semibold">Haller HMI</h1>
        <TelemetryBar />
      </div>
      <div className="grid grid-cols-12 gap-3">
        <Card className="col-span-7">
          <CardHeader><CardTitle>Base</CardTitle></CardHeader>
          <CardContent>Base panel coming in Task 14.</CardContent>
        </Card>
        {cfg.arms.map((arm) => (
          <Card key={arm.id} className="col-span-5">
            <CardHeader><CardTitle>Arm: {arm.id}</CardTitle></CardHeader>
            <CardContent>Arm panel coming in Task 13.</CardContent>
          </Card>
        ))}
      </div>
    </main>
  );
}
```

- [ ] **Step 8: Smoke against the running backend**

In one terminal: `source ~/venvs/haller-hmi/bin/activate-haller-hmi && haller-hmi`.
In another: `cd ~/haller_ws/hmi/frontend && pnpm dev`.

Open http://localhost:3000 — expect: dark page, "Haller HMI" title, "live" badge in green, t/v/ω/x/y values updating, one card per arm.

- [ ] **Step 9: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/lib hmi/frontend/components/TelemetryBar.tsx hmi/frontend/app/page.tsx hmi/frontend/__tests__/api.test.ts hmi/frontend/vitest.config.ts hmi/frontend/package.json
git commit -m "feat(hmi/frontend): API client, telemetry WS store, dashboard skeleton"
```

---

## Task 12: Wire EStopButton + ModeToggle

**Files:**
- Modify: `hmi/frontend/components/EStopButton.tsx`
- Create: `hmi/frontend/components/ModeToggle.tsx`
- Create: `hmi/frontend/__tests__/EStopButton.test.tsx`

- [ ] **Step 1: Failing test**

```tsx
// hmi/frontend/__tests__/EStopButton.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { EStopButton } from "../components/EStopButton";

describe("EStopButton", () => {
  it("calls api.estop on click", async () => {
    const mockEstop = vi.fn().mockResolvedValue({ ok: true });
    vi.doMock("@/lib/api", () => ({ api: { estop: mockEstop } }));
    // re-import after mock
    const { EStopButton: Reloaded } = await import("../components/EStopButton");
    render(<Reloaded />);
    fireEvent.click(screen.getByRole("button", { name: /emergency stop/i }));
    await Promise.resolve();
    expect(mockEstop).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Wire `EStopButton`**

```tsx
// hmi/frontend/components/EStopButton.tsx
"use client";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { toast } from "sonner";

export function EStopButton({ className }: { className?: string }) {
  const handleClick = async () => {
    try {
      await api.estop();
      toast.error("E-STOP triggered — torque off, base zeroed");
    } catch (e) {
      toast.error(`E-STOP failed: ${(e as Error).message}`);
    }
  };
  return (
    <Button
      variant="destructive"
      size="lg"
      onClick={handleClick}
      className={`rounded-full h-16 w-16 text-xs font-bold ${className ?? ""}`}
      aria-label="Emergency stop"
    >
      E-STOP
    </Button>
  );
}
```

- [ ] **Step 3: Implement `ModeToggle`**

```tsx
// hmi/frontend/components/ModeToggle.tsx
"use client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { toast } from "sonner";

const ORDER: Array<"auto" | "manual" | "stop"> = ["auto", "manual", "stop"];

export function ModeToggle({ armId, mode }: { armId: string; mode: "auto" | "manual" | "stop" }) {
  return (
    <div className="flex items-center gap-2">
      <Badge variant={mode === "stop" ? "destructive" : mode === "auto" ? "default" : "secondary"}>
        {mode}
      </Badge>
      <div className="flex gap-1">
        {ORDER.map((m) => (
          <Button
            key={m}
            size="sm"
            variant={m === mode ? "default" : "outline"}
            onClick={async () => {
              try {
                await api.armMode(armId, m);
              } catch (e) {
                toast.error(`mode change failed: ${(e as Error).message}`);
              }
            }}
          >
            {m}
          </Button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run the test**

```bash
cd ~/haller_ws/hmi/frontend && pnpm test EStopButton
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/components/EStopButton.tsx hmi/frontend/components/ModeToggle.tsx hmi/frontend/__tests__/EStopButton.test.tsx
git commit -m "feat(hmi/frontend): wire E-STOP + per-arm ModeToggle to backend"
```

---

## Task 13: Arm panel with JointSlider + presets + camera tile

**Files:**
- Create: `hmi/frontend/components/JointSlider.tsx`
- Create: `hmi/frontend/components/CameraTile.tsx`
- Create: `hmi/frontend/components/ArmPanel.tsx`
- Modify: `hmi/frontend/app/page.tsx` (use ArmPanel)
- Create: `hmi/frontend/app/arm/[id]/page.tsx` (per-arm detail)
- Create: `hmi/frontend/__tests__/JointSlider.test.tsx`

- [ ] **Step 1: Failing test for JointSlider**

```tsx
// hmi/frontend/__tests__/JointSlider.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { JointSlider } from "../components/JointSlider";

describe("JointSlider", () => {
  it("calls onChange (debounced) when moved", async () => {
    const onChange = vi.fn();
    render(
      <JointSlider name="shoulder_pan" pos={0} min={-100} max={100}
                   onChange={onChange} disabled={false} />
    );
    expect(screen.getByText("shoulder_pan")).toBeTruthy();
    // Radix slider uses keyboard for tests:
    const slider = screen.getByRole("slider");
    fireEvent.keyDown(slider, { key: "ArrowRight" });
    await new Promise((r) => setTimeout(r, 80));
    expect(onChange).toHaveBeenCalled();
  });

  it("disables interaction when disabled prop is set", () => {
    render(
      <JointSlider name="gripper" pos={0} min={0} max={100}
                   onChange={() => {}} disabled />
    );
    const slider = screen.getByRole("slider");
    expect(slider.getAttribute("aria-disabled")).toBe("true");
  });
});
```

- [ ] **Step 2: Implement JointSlider with debouncing**

```tsx
// hmi/frontend/components/JointSlider.tsx
"use client";
import { useEffect, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";

export type JointSliderProps = {
  name: string;
  pos: number;       // current observed position (deg)
  min: number;       // calibrated min (deg)
  max: number;       // calibrated max (deg)
  onChange: (value: number) => void;
  disabled?: boolean;
};

export function JointSlider({ name, pos, min, max, onChange, disabled }: JointSliderProps) {
  // Locally controlled while user drags; snaps back to `pos` when released.
  const [local, setLocal] = useState<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // If the upstream telemetry changes more than 1° from our local value, accept it
  useEffect(() => {
    if (local === null) return;
    if (Math.abs(local - pos) > 5) setLocal(null);
  }, [pos, local]);

  const value = local ?? pos;

  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-mono">{name}</span>
        <span className="text-xs font-mono text-muted-foreground">
          {value.toFixed(1)}° <span className="opacity-50">/ [{min.toFixed(0)}, {max.toFixed(0)}]</span>
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={0.5}
        value={[value]}
        onValueChange={([v]) => {
          setLocal(v);
          if (timer.current) clearTimeout(timer.current);
          timer.current = setTimeout(() => onChange(v), 50);
        }}
        onValueCommit={() => {
          setLocal(null);
        }}
        disabled={disabled}
      />
    </div>
  );
}
```

- [ ] **Step 3: Implement CameraTile placeholder**

```tsx
// hmi/frontend/components/CameraTile.tsx
"use client";
import { Badge } from "@/components/ui/badge";

export function CameraTile({ id, role }: { id: string; role: "wrist" | "base" }) {
  return (
    <div className="aspect-video bg-muted/30 rounded-md flex items-center justify-center border border-dashed">
      <div className="flex flex-col items-center gap-1 text-muted-foreground">
        <span className="text-xs uppercase">{role}</span>
        <span className="font-mono text-xs">{id}</span>
        <Badge variant="outline" className="text-xs">no feed</Badge>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Implement ArmPanel**

```tsx
// hmi/frontend/components/ArmPanel.tsx
"use client";
import { useMemo, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

import { useTelemetry } from "@/lib/telemetry";
import { api } from "@/lib/api";
import { JointSlider } from "./JointSlider";
import { ModeToggle } from "./ModeToggle";
import { CameraTile } from "./CameraTile";

export function ArmPanel({ armId }: { armId: string }) {
  const arm = useTelemetry((s) => s.lastFrame?.arms[armId]);
  const [presetName, setPresetName] = useState("");
  const joints = useMemo(() => Object.entries(arm?.joints ?? {}), [arm]);

  if (!arm) return <div className="text-xs text-muted-foreground">no telemetry yet</div>;
  const disabled = arm.mode !== "manual";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2">
        <CardTitle>Arm: {armId}</CardTitle>
        <ModeToggle armId={armId} mode={arm.mode} />
      </CardHeader>
      <CardContent className="space-y-4">
        <CameraTile id={`${armId}_wrist`} role="wrist" />
        <div className="space-y-2">
          {joints.map(([name, j]) => (
            <JointSlider
              key={name}
              name={name}
              pos={j.pos}
              min={j.min}
              max={j.max}
              disabled={disabled}
              onChange={async (v) => {
                try {
                  await api.armGoal(armId, { [name]: v });
                } catch (e) {
                  toast.error(`${name} goal failed: ${(e as Error).message}`);
                }
              }}
            />
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Input
            placeholder="preset name"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
            className="max-w-[160px] h-8"
          />
          <Button
            size="sm"
            disabled={!presetName}
            onClick={async () => {
              try { await api.armPresetRecord(armId, presetName); toast.success(`saved ${presetName}`); }
              catch (e) { toast.error((e as Error).message); }
            }}
          >Save pose</Button>
          <Button
            size="sm"
            variant="outline"
            disabled={!presetName || disabled}
            onClick={async () => {
              try { await api.armPreset(armId, presetName); }
              catch (e) { toast.error((e as Error).message); }
            }}
          >Go to pose</Button>
        </div>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 5: Use ArmPanel in the dashboard**

```tsx
// hmi/frontend/app/page.tsx — replace the arm placeholder Card with:
{cfg.arms.map((arm) => (
  <div key={arm.id} className="col-span-5">
    <ArmPanel armId={arm.id} />
  </div>
))}
```

Plus add: `import { ArmPanel } from "@/components/ArmPanel";` at top.

- [ ] **Step 6: Per-arm detail page**

```tsx
// hmi/frontend/app/arm/[id]/page.tsx
import { ArmPanel } from "@/components/ArmPanel";

export default async function ArmDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <main className="p-3">
      <ArmPanel armId={id} />
    </main>
  );
}
```

- [ ] **Step 7: Run JointSlider test**

```bash
cd ~/haller_ws/hmi/frontend && pnpm test JointSlider
```

Expected: 2 passed.

- [ ] **Step 8: Manual smoke against arm**

With backend + frontend running: open http://localhost:3000.
- Set right arm to manual via ModeToggle.
- Drag the gripper slider — the gripper should actually move.
- In auto mode, sliders are disabled.

- [ ] **Step 9: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/components/JointSlider.tsx hmi/frontend/components/CameraTile.tsx hmi/frontend/components/ArmPanel.tsx hmi/frontend/app/page.tsx hmi/frontend/app/arm/[id]/page.tsx hmi/frontend/__tests__/JointSlider.test.tsx
git commit -m "feat(hmi/frontend): ArmPanel with joint sliders, presets, camera placeholder"
```

---

## Task 14: BasePanel — joystick + keyboard + speed slider

**Files:**
- Create: `hmi/frontend/components/BasePanel.tsx`
- Create: `hmi/frontend/app/base/page.tsx`
- Modify: `hmi/frontend/app/page.tsx`

The control vocabulary mirrors `web_teleop.py` (joystick + WASD + speed slider) so operators have muscle memory.

- [ ] **Step 1: Implement BasePanel**

```tsx
// hmi/frontend/components/BasePanel.tsx
"use client";
import { useEffect, useRef, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useTelemetry } from "@/lib/telemetry";
import { CameraTile } from "./CameraTile";

const SEND_HZ = 10;

export function BasePanel() {
  const base = useTelemetry((s) => s.lastFrame?.base);
  const [speed, setSpeed] = useState(0.4);
  const cmd = useRef({ linear: 0, angular: 0 });
  const pad = useRef<HTMLDivElement>(null);
  const [knob, setKnob] = useState({ x: 0, y: 0 });

  // Send at fixed rate while non-zero, then one final zero
  useEffect(() => {
    const t = setInterval(() => {
      api.cmdVel(cmd.current.linear * speed, cmd.current.angular * speed).catch(() => {});
    }, 1000 / SEND_HZ);
    return () => clearInterval(t);
  }, [speed]);

  // keyboard
  useEffect(() => {
    const pressed = new Set<string>();
    const update = () => {
      let l = 0, a = 0;
      if (pressed.has("w") || pressed.has("ArrowUp")) l += 1;
      if (pressed.has("s") || pressed.has("ArrowDown")) l -= 1;
      if (pressed.has("a") || pressed.has("ArrowLeft")) a += 1;
      if (pressed.has("d") || pressed.has("ArrowRight")) a -= 1;
      cmd.current = { linear: l, angular: a };
    };
    const down = (e: KeyboardEvent) => { pressed.add(e.key.toLowerCase()); update(); };
    const up = (e: KeyboardEvent) => { pressed.delete(e.key.toLowerCase()); update(); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, []);

  // joystick (mouse + touch)
  useEffect(() => {
    const el = pad.current;
    if (!el) return;
    let dragging = false;
    const r = () => el.getBoundingClientRect();
    const move = (clientX: number, clientY: number) => {
      const rect = r();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (clientX - cx) / (rect.width / 2);
      const dy = (clientY - cy) / (rect.height / 2);
      const clipped = (v: number) => Math.max(-1, Math.min(1, v));
      const x = clipped(dx);
      const y = clipped(dy);
      setKnob({ x, y });
      cmd.current = { linear: -y, angular: -x };
    };
    const onDown = (e: MouseEvent) => { dragging = true; move(e.clientX, e.clientY); };
    const onMove = (e: MouseEvent) => { if (dragging) move(e.clientX, e.clientY); };
    const onUp = () => { dragging = false; cmd.current = { linear: 0, angular: 0 }; setKnob({ x: 0, y: 0 }); };
    el.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { el.removeEventListener("mousedown", onDown); window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);

  return (
    <Card>
      <CardHeader><CardTitle>Base</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <CameraTile id="base_front" role="base" />
        <div className="grid grid-cols-2 gap-3 items-center">
          <div
            ref={pad}
            className="relative w-full aspect-square bg-muted/30 rounded-full border touch-none"
            tabIndex={0}
          >
            <div
              className="absolute w-8 h-8 -mt-4 -ml-4 rounded-full bg-emerald-500"
              style={{
                left: `${50 + knob.x * 40}%`,
                top: `${50 + knob.y * 40}%`,
              }}
            />
          </div>
          <div className="space-y-2">
            <div className="text-xs font-mono">speed {speed.toFixed(2)}×</div>
            <Slider min={0.1} max={1.0} step={0.05} value={[speed]} onValueChange={([v]) => setSpeed(v)} />
            <Button variant="destructive" onClick={() => { cmd.current = { linear: 0, angular: 0 }; setKnob({ x: 0, y: 0 }); }}>
              STOP
            </Button>
            <div className="text-xs font-mono text-muted-foreground">
              v={base?.linear.toFixed(2) ?? "—"} m/s  ω={base?.angular.toFixed(2) ?? "—"} rad/s
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Use BasePanel in the dashboard and base page**

In `app/page.tsx`:

```tsx
// replace the Base placeholder card with:
<div className="col-span-7">
  <BasePanel />
</div>
```

…and add import.

```tsx
// hmi/frontend/app/base/page.tsx
import { BasePanel } from "@/components/BasePanel";
export default function BasePage() {
  return <main className="p-3"><BasePanel /></main>;
}
```

- [ ] **Step 3: Manual smoke**

With backend + frontend running, the base joystick should drive the robot (or the `ros2 topic echo /cmd_vel` you ran in Task 5 step 2) at 10 Hz, matching the existing teleop behaviour.

- [ ] **Step 4: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/components/BasePanel.tsx hmi/frontend/app/base/page.tsx hmi/frontend/app/page.tsx
git commit -m "feat(hmi/frontend): BasePanel with joystick + WASD + speed slider (10 Hz)"
```

---

## Task 15: Settings page

**Files:**
- Create: `hmi/frontend/app/settings/page.tsx`

- [ ] **Step 1: Implement settings page**

```tsx
// hmi/frontend/app/settings/page.tsx
"use client";
import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

type ConfigBody = Awaited<ReturnType<typeof api.config>>;

export default function SettingsPage() {
  const [cfg, setCfg] = useState<ConfigBody | null>(null);
  const [health, setHealth] = useState<{ status: string } | null>(null);
  useEffect(() => {
    api.config().then(setCfg).catch(console.error);
    api.health().then(setHealth).catch(console.error);
  }, []);
  return (
    <main className="p-3 space-y-3">
      <h1 className="text-lg font-semibold">Settings</h1>
      <Card>
        <CardHeader><CardTitle>Health</CardTitle></CardHeader>
        <CardContent className="text-sm font-mono">
          status: <Badge>{health?.status ?? "…"}</Badge>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Arms</CardTitle></CardHeader>
        <CardContent>
          {cfg?.arms.map((a) => (
            <div key={a.id} className="text-sm font-mono py-1 flex gap-3">
              <span className="w-16">{a.id}</span>
              <span>{a.model}</span>
              <span className="text-muted-foreground">{a.port}</span>
              <Badge variant="outline">{a.mode}</Badge>
            </div>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Cameras</CardTitle></CardHeader>
        <CardContent>
          {cfg?.cameras.map((c) => (
            <div key={c.id} className="text-sm font-mono py-1 flex gap-3">
              <span className="w-32">{c.id}</span>
              <span>{c.role}</span>
              <Badge variant="outline">{c.source}</Badge>
            </div>
          ))}
        </CardContent>
      </Card>
    </main>
  );
}
```

- [ ] **Step 2: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/app/settings/page.tsx
git commit -m "feat(hmi/frontend): settings page (health, arms, cameras)"
```

---

## Task 16: Frontend styling pass via `frontend-design` skill

**Files:**
- Modify: various components (component-by-component design polish)

This is the explicit hook-in the user asked for: invoke the `frontend-design` skill so component styling is distinctive and high-quality rather than vanilla shadcn.

- [ ] **Step 1: Invoke the frontend-design skill**

```text
[invoke Skill: frontend-design with this argument:]
Polish the Haller HMI frontend (Next.js 16 + shadcn/ui, dark dashboard) for a supervisory-control robotics product. Files:
  - hmi/frontend/app/layout.tsx
  - hmi/frontend/app/page.tsx
  - hmi/frontend/components/EStopButton.tsx
  - hmi/frontend/components/ArmPanel.tsx
  - hmi/frontend/components/BasePanel.tsx
  - hmi/frontend/components/JointSlider.tsx
  - hmi/frontend/components/TelemetryBar.tsx
  - hmi/frontend/components/CameraTile.tsx
  - hmi/frontend/components/ModeToggle.tsx
Goals: high information density (operator monitoring, not marketing), distinctive identity (this is "Haller", not generic shadcn). Mono for numeric readouts, sans for labels. E-STOP must be unmistakable. Mode badges should communicate state at a glance with motion (e.g. pulsing dot for "auto" — emerald). Joint sliders should show range visually (filled track between min/max, tick at home).
Do NOT change component prop shapes — only styling, internal markup, and class composition. After applying changes, run `pnpm test` and `pnpm build` in hmi/frontend to make sure nothing broke.
```

Wait for the skill to apply changes, review them, accept or push back.

- [ ] **Step 2: Verify nothing broke**

```bash
cd ~/haller_ws/hmi/frontend
pnpm test
pnpm build
```

Both must pass.

- [ ] **Step 3: Manual smoke + screenshot**

Open the dashboard, take a screenshot for the README. Save to `hmi/frontend/public/screenshot-dashboard.png`.

- [ ] **Step 4: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend
git commit -m "style(hmi/frontend): design pass via frontend-design skill"
```

---

# Phase III — Deployment & migration

## Task 17: Production build + run script

**Files:**
- Create: `scripts/run_hmi.sh`
- Modify: `hmi/frontend/next.config.ts` (enable standalone output)

- [ ] **Step 1: Configure Next.js standalone output**

Edit `hmi/frontend/next.config.ts`:

```ts
import type { NextConfig } from "next";
const nextConfig: NextConfig = {
  output: "standalone",
};
export default nextConfig;
```

- [ ] **Step 2: Build the frontend**

```bash
cd ~/haller_ws/hmi/frontend
pnpm build
ls .next/standalone/server.js   # expect the file to exist
```

- [ ] **Step 3: Write `scripts/run_hmi.sh`**

```bash
#!/usr/bin/env bash
# scripts/run_hmi.sh — launches the unified HMI (backend + frontend) on this host.
set -euo pipefail

# Activate the backend env (sources ROS + venv + isolation hooks)
source "$HOME/venvs/haller-hmi/bin/activate-haller-hmi"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

cd "$HOME/haller_ws"

# Start backend
uvicorn haller_hmi.server:app --host 0.0.0.0 --port "$BACKEND_PORT" --app-dir hmi/backend &
BACKEND_PID=$!

# Start prebuilt frontend (standalone Node server)
HOSTNAME="0.0.0.0" PORT="$FRONTEND_PORT" \
NEXT_PUBLIC_BACKEND_URL="http://localhost:$BACKEND_PORT" \
node hmi/frontend/.next/standalone/server.js &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT
wait
```

```bash
chmod +x ~/haller_ws/scripts/run_hmi.sh
```

- [ ] **Step 4: Local smoke (laptop)**

```bash
~/haller_ws/scripts/run_hmi.sh &
# open http://localhost:3000 — should look like the dev build
kill %1
```

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/next.config.ts scripts/run_hmi.sh
git commit -m "build(hmi): standalone Next.js output + run_hmi.sh launcher"
```

---

## Task 18: systemd unit + Orin migration parity check

**Files:**
- Create: `scripts/haller-hmi.service`
- Modify: `src/haller_ros/haller_robot/haller_hardware/launch/haller_bringup.launch.py` (default `enable_web_teleop=False`)

- [ ] **Step 1: Write the unit file**

```ini
# scripts/haller-hmi.service
[Unit]
Description=Haller Unified HMI (FastAPI + Next.js)
After=network-online.target haller-ap.service
Wants=network-online.target
Conflicts=haller-robot.service

[Service]
Type=simple
User=orin
WorkingDirectory=/home/orin/haller_ws
Environment=ROS_DOMAIN_ID=0
ExecStart=/home/orin/haller_ws/scripts/run_hmi.sh
Restart=on-failure
RestartSec=3s

[Install]
WantedBy=multi-user.target
```

Note: `Conflicts=haller-robot.service` is intentional — for the migration window we want to choose one or the other, not run both servers on the same ROS network publishing `/cmd_vel`.

- [ ] **Step 2: Disable the legacy web_teleop in the launch file**

Find the line in `haller_bringup.launch.py` that declares `enable_web_teleop` and flip the default:

```python
# before
DeclareLaunchArgument('enable_web_teleop', default_value='true', description='Enable web teleop UI'),
# after
DeclareLaunchArgument('enable_web_teleop', default_value='false',
                      description='DEPRECATED: enable legacy web_teleop. Use haller-hmi.service instead.'),
```

- [ ] **Step 3: Parity check (one-time, before swapping units on the Orin)**

On the Orin, install both unit files but enable only the legacy one. Bring it up; verify the operator workflow still works on port 8080. Stop it. Now bring up the new unit:

```bash
# on the Orin
scp -r ~/haller_ws orin:~/   # or git pull if already cloned there
sudo cp ~/haller_ws/scripts/haller-hmi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl stop haller-robot.service     # ensure conflict is clean
sudo systemctl start haller-hmi.service
sudo journalctl -u haller-hmi.service -f     # watch startup
```

From the operator laptop, open `http://orin.local:3000` (or whatever the Orin's hostname/IP is). Drive the joystick — `/cmd_vel` should publish. Toggle the arm to manual, drag a slider — arm should move.

- [ ] **Step 4: Enable on boot, disable legacy**

```bash
# on the Orin, only after the new unit is verified working
sudo systemctl enable haller-hmi.service
sudo systemctl disable haller-robot.service
sudo systemctl daemon-reload
```

Note: we do NOT delete `web_teleop.py` here. It stays in-tree for one release for rollback per the spec's migration plan. The launch arg default flip is enough to keep it from running.

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add scripts/haller-hmi.service src/haller_ros/haller_robot/haller_hardware/launch/haller_bringup.launch.py
git commit -m "deploy(hmi): systemd unit, disable legacy web_teleop launch default"
```

---

## Task 19: hmi/README.md

**Files:**
- Create: `hmi/README.md`
- Modify: top-level `README.md` (link to hmi/README from the "Getting started → Arms" section)

- [ ] **Step 1: Write `hmi/README.md`**

````markdown
# Haller HMI

Unified web HMI for the Haller robot. Replaces the legacy `web_teleop.py`.

Frontend: Next.js 16 + shadcn/ui (Node).
Backend: FastAPI wrapping `lerobot` (arms) and `rclpy` (base) in one process.

## Repository layout

```
hmi/
├── backend/      Python service (uvicorn + FastAPI)
└── frontend/     Next.js app (standalone-built)
```

## Dev (operator laptop)

Backend:
```bash
source ~/venvs/haller-hmi/bin/activate-haller-hmi
cd hmi/backend
pip install -e ".[dev]"
haller-hmi          # serves http://localhost:8000
```

Frontend:
```bash
cd hmi/frontend
pnpm install
pnpm dev            # serves http://localhost:3000
```

Pointing the frontend at a non-default backend:
```bash
NEXT_PUBLIC_BACKEND_URL=http://orin.local:8000 pnpm dev
```

## Production (Jetson Orin Nano)

```bash
sudo cp scripts/haller-hmi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now haller-hmi.service
```

Logs: `journalctl -u haller-hmi.service -f`

## REST endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| GET | `/health` | — | liveness |
| GET | `/config` | — | arms, cameras, version |
| POST | `/base/cmd_vel` | `{linear, angular}` | publishes `/cmd_vel` |
| POST | `/arm/{id}/goal` | `{joint: deg, …}` | manual mode only |
| POST | `/arm/{id}/mode` | `{mode: "auto"\|"manual"\|"stop"}` | |
| POST | `/arm/{id}/preset` | `{name}` | replay preset |
| POST | `/arm/{id}/preset/record` | `{name}` | save current pose |
| POST | `/estop` | `{}` | torque off all arms + zero `/cmd_vel` |
| WS | `/ws/telemetry` | — | 20 Hz frames |

See `docs/superpowers/specs/2026-05-22-haller-unified-hmi-design.md` for the design rationale.

## Tests

Backend:
```bash
cd hmi/backend && pytest
```

Frontend:
```bash
cd hmi/frontend && pnpm test
```

## Configuration

`hmi/backend/config.yaml` — arm ids, ports, ROS topics, telemetry rate, camera roles.
Override path with `HALLER_HMI_CONFIG=/path/to/your.yaml haller-hmi`.
````

- [ ] **Step 2: Patch top-level `README.md`**

Find the "Arms (LeRobot + SO-101)" section and add:

```markdown
3. **[`hmi/README.md`](./hmi/README.md)** — bring up the unified HMI (FastAPI backend + Next.js + shadcn frontend) that replaces the legacy `web_teleop.py`.
```

- [ ] **Step 3: Commit**

```bash
cd ~/haller_ws
git add hmi/README.md README.md
git commit -m "docs(hmi): repo README + link from top-level README"
```

---

## Task 20: Push everything

- [ ] **Step 1: Verify branch state**

```bash
cd ~/haller_ws
git log --oneline -30
```

Expect about 15-20 new commits on top of `bb7ff00` covering Tasks 1–19.

- [ ] **Step 2: Push**

```bash
git push origin main
```

If auth prompts, see top-level README's git config notes.

- [ ] **Step 3: Final smoke against the deployed unit**

Open `http://orin.local:3000` (or laptop equivalent) and run the operator workflow once end-to-end:
- dashboard loads, "live" badge green
- drive base briefly with joystick
- switch arm to manual, drag gripper slider
- record a pose called "home"
- E-STOP — torque releases, arm becomes back-drivable

---

# Done criteria

- [ ] All 19 implementation tasks closed.
- [ ] `cd hmi/backend && pytest` → 25 passed.
- [ ] `cd hmi/frontend && pnpm test` → all passed.
- [ ] `cd hmi/frontend && pnpm build` → standalone output produced.
- [ ] `haller-hmi.service` running on the Orin without restarts.
- [ ] Legacy `web_teleop.py` disabled but in-tree.
- [ ] All changes pushed to `origin/main`.
- [ ] `hmi/README.md` describes the dev + prod workflows and lives next to the top-level README link.
