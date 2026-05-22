# HMI Calibration Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an in-browser, single-arm calibration wizard (homing → range-of-motion sweep → review → save) that replaces the `lerobot-calibrate` CLI step, with backup-then-overwrite semantics and an auto-prompt when calibration is missing.

**Architecture:** New `CalibrationSession` / `CalibrationManager` in the FastAPI backend orchestrates state; the existing 20 Hz telemetry stream carries live tick / min / max data while a session is active; six new REST routes drive the state machine; a Next.js `CalibrationWizard` sheet plus a settings-page status card and a dashboard banner cover the UI. Backend hits the bus via lerobot's low-level primitives only (`sync_read("Present_Position", …, normalize=False)`, `bus.write("Homing_Offset", …)`, `bus.disable_torque/enable_torque`).

**Tech Stack:** FastAPI + pydantic + pytest (backend), Next.js 16 + React 19 + shadcn/ui + zustand + vitest + React Testing Library (frontend), draccus (lerobot's JSON serializer for `MotorCalibration`).

**Spec:** `docs/superpowers/specs/2026-05-22-calibration-wizard-design.md` (commit `b0f5d75`).

---

## File structure (created or modified by this plan)

```
haller_ws/
├── hmi/
│   ├── backend/
│   │   ├── haller_hmi/
│   │   │   ├── calibration.py                       ← Tasks 1, 2 (new)
│   │   │   ├── server.py                            ← Tasks 4, 5 (modify)
│   │   │   ├── telemetry.py                         ← Task 3 (modify)
│   │   │   └── arm.py                               ← Task 1 (small additive change: expose `bus` accessor if needed; nothing if already accessible via `handle.robot.bus`)
│   │   └── tests/
│   │       ├── test_calibration.py                  ← Tasks 1, 2 (new)
│   │       ├── test_telemetry.py                    ← Task 3 (modify, add cases)
│   │       └── test_routes.py                       ← Tasks 4, 5 (modify, add cases)
│   ├── frontend/
│   │   ├── lib/
│   │   │   └── calibration.ts                       ← Task 6 (new)
│   │   ├── components/
│   │   │   ├── CalibrationStatusCard.tsx            ← Task 7 (new)
│   │   │   ├── CalibrationWizard.tsx                ← Tasks 10, 11, 12 (new)
│   │   │   ├── ArmPanel.tsx                         ← Task 13 (modify)
│   │   │   └── ui/                                  ← shadcn primitives (sheet, alert-dialog if missing) added via CLI
│   │   ├── app/
│   │   │   ├── settings/page.tsx                    ← Task 8 (modify)
│   │   │   └── page.tsx                             ← Task 9 (modify)
│   │   └── __tests__/
│   │       ├── calibration.test.ts                  ← Task 6 (new)
│   │       └── CalibrationWizard.test.tsx           ← Tasks 10–12 (new)
│   └── README.md                                    ← Task 14 (modify)
└── README.md                                        ← Task 14 (modify; update "Next" line)
```

---

## Riskiest steps (heads-up)

1. **Task 2 (Save mechanics).** Format must be byte-compatible with `lerobot-calibrate`'s output so files round-trip cleanly. Plan uses `draccus.dump(self.calibration, f, indent=4)` exactly as lerobot's `Robot._save_calibration` does — anything else risks subtle drift.
2. **Task 3 (Telemetry tick reads the bus).** A `sync_read("Present_Position", …, normalize=False)` on every telemetry frame for the calibrating arm doubles the bus traffic for that arm. At 20 Hz on a 1 Mbaud half-duplex chain this is well within budget, but `state_snapshot` already calls `get_observation` (which does its own sync_read). Plan is to fold the calibration read into the same tick path — one extra read per tick, not two.
3. **Task 5 (Mode-change gating).** Adding a 409 to `/arm/<id>/mode` could break the existing teleop launcher if it relies on bouncing modes. Verify by running the teleop test suite after the change.
4. **Task 13 (ArmPanel gating).** Mirrors the existing teleop gating pattern; the risk is forgetting one of the controls (joint sliders, Home button, free-drive toggle, preset chips).

---

# Phase I — Backend (CalibrationSession + state machine)

## Task 1: `CalibrationSession` + `CalibrationManager` skeleton — TDD

**Files:**
- Create: `hmi/backend/haller_hmi/calibration.py`
- Create: `hmi/backend/tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests (state transitions, conflict checks, capture math)**

```python
# hmi/backend/tests/test_calibration.py
from unittest.mock import MagicMock

import pytest

from haller_hmi.calibration import (
    CalibrationManager,
    CalibrationSession,
    CalibrationState,
    ConflictError,
    UnmovedJointsError,
    WrongStateError,
)
from haller_hmi.config import ArmConfig
from haller_hmi.safety import Mode


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]


def _make_handle(arm_id: str = "right", mode: Mode = Mode.MANUAL):
    """Build a MagicMock that quacks like ArmHandle for the calibration session."""
    handle = MagicMock()
    handle.config = ArmConfig(id=arm_id, model="so101_follower",
                              port="/dev/null", calibration_id=f"haller_{arm_id}")
    handle.guard = MagicMock()
    handle.guard.mode = mode
    handle.torque_enabled = True
    # The bus: sync_read returns ticks; write captures Homing_Offset writes
    bus = MagicMock()
    bus.sync_read.return_value = {j: 2048 for j in JOINTS}
    handle.robot = MagicMock()
    handle.robot.bus = bus
    # Motor models — used by the homing math
    handle.robot.bus.motors = {j: MagicMock(model="sts3215") for j in JOINTS}
    handle.robot.bus.model_resolution_table = {"sts3215": 4096}
    return handle


def _make_arms(*handles):
    arms = MagicMock()
    arms.keys.return_value = [h.config.id for h in handles]
    arms.values.return_value = list(handles)
    by_id = {h.config.id: h for h in handles}
    arms.__getitem__.side_effect = lambda k: by_id[k]
    return arms


def test_start_from_idle_transitions_to_homing_and_disables_torque():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    assert session.state is CalibrationState.HOMING
    handle.disable_torque.assert_called_once()


def test_start_rejected_when_another_arm_not_manual():
    left = _make_handle("left", mode=Mode.AUTO)
    right = _make_handle("right")
    arms = _make_arms(left, right)
    mgr = CalibrationManager()
    with pytest.raises(ConflictError, match="left"):
        mgr.start(arms, "right")


def test_start_rejected_when_session_already_active():
    h1 = _make_handle("right")
    arms = _make_arms(h1)
    mgr = CalibrationManager()
    mgr.start(arms, "right")
    with pytest.raises(ConflictError, match="session active"):
        mgr.start(arms, "right")


def test_capture_neutral_writes_homing_offsets_and_transitions():
    handle = _make_handle()
    handle.robot.bus.sync_read.return_value = {j: 2200 for j in JOINTS}
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    # Feetech: homing_offset = pos - (resolution - 1) // 2 = 2200 - 2047 = 153
    expected = 2200 - (4096 - 1) // 2
    for j in JOINTS:
        handle.robot.bus.write.assert_any_call("Homing_Offset", j, expected)
    assert session.state is CalibrationState.SWEEPING
    assert session.homing_offsets == {j: expected for j in JOINTS}


def test_capture_neutral_wrong_state_raises():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    with pytest.raises(WrongStateError):
        session.capture_neutral(handle)


def test_tick_sweep_accumulates_min_max():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    handle.robot.bus.sync_read.return_value = {j: 1000 for j in JOINTS}
    session.tick_sweep(handle)
    handle.robot.bus.sync_read.return_value = {j: 3500 for j in JOINTS}
    session.tick_sweep(handle)
    handle.robot.bus.sync_read.return_value = {j: 2500 for j in JOINTS}
    ticks = session.tick_sweep(handle)
    assert session.mins == {j: 1000 for j in JOINTS}
    assert session.maxes == {j: 3500 for j in JOINTS}
    assert ticks == {j: 2500 for j in JOINTS}


def test_finish_sweep_unmoved_joints_raises():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    # mins == maxes for every joint because we never moved
    handle.robot.bus.sync_read.return_value = {j: 2048 for j in JOINTS}
    session.tick_sweep(handle)
    with pytest.raises(UnmovedJointsError) as ei:
        session.finish_sweep(handle)
    for j in JOINTS:
        assert j in str(ei.value)
    assert session.state is CalibrationState.SWEEPING  # stays


def test_finish_sweep_builds_lerobot_shaped_proposed():
    handle = _make_handle()
    handle.robot.bus.motors = {
        "shoulder_pan":  MagicMock(model="sts3215", id=1),
        "shoulder_lift": MagicMock(model="sts3215", id=2),
        "elbow_flex":    MagicMock(model="sts3215", id=3),
        "wrist_flex":    MagicMock(model="sts3215", id=4),
        "wrist_roll":    MagicMock(model="sts3215", id=5),
        "gripper":       MagicMock(model="sts3215", id=6),
    }
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    handle.robot.bus.sync_read.return_value = {j: 500 for j in JOINTS}
    session.tick_sweep(handle)
    handle.robot.bus.sync_read.return_value = {j: 3600 for j in JOINTS}
    session.tick_sweep(handle)
    proposed = session.finish_sweep(handle)
    assert session.state is CalibrationState.REVIEW
    for joint in JOINTS:
        entry = proposed[joint]
        assert set(entry.keys()) == {"id", "drive_mode", "homing_offset",
                                     "range_min", "range_max"}
        assert entry["range_min"] == 500
        assert entry["range_max"] == 3600
        assert entry["drive_mode"] == 0


def test_abort_from_homing_re_enables_torque_and_clears():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    mgr.start(arms, "right")
    mgr.abort()
    handle.enable_torque.assert_called_once()
    assert mgr.current is None


def test_abort_is_idempotent():
    mgr = CalibrationManager()
    mgr.abort()  # no current session — must not raise
    assert mgr.current is None
```

- [ ] **Step 2: Run, verify failures**

```bash
cd ~/haller_ws/hmi/backend
source ~/venvs/haller-hmi/bin/activate-haller-hmi
pytest tests/test_calibration.py -v
```

Expected: ImportError (module doesn't exist yet).

- [ ] **Step 3: Implement `haller_hmi/calibration.py` (state + capture + sweep + finish; save lives in Task 2)**

```python
# hmi/backend/haller_hmi/calibration.py
"""Per-arm calibration session — homing offsets + range-of-motion sweep.

Only one session exists at a time across the whole HMI. The session reuses the
ArmHandle's existing MotorsBus (no second serial connection).

Save mechanics live in this module too (see Task 2 in the plan).
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

from .safety import Mode

logger = logging.getLogger(__name__)


class CalibrationState(str, enum.Enum):
    HOMING = "homing"
    SWEEPING = "sweeping"
    REVIEW = "review"
    DONE = "done"
    ABORTED = "aborted"


class CalibrationError(Exception):
    """Base for calibration-specific errors mapped to HTTP responses."""


class ConflictError(CalibrationError):
    """Pre-flight failure: another session active, or an arm not in manual."""


class WrongStateError(CalibrationError):
    """A method was called in the wrong session state."""


class UnmovedJointsError(CalibrationError):
    """finish_sweep called while one or more joints have min == max."""

    def __init__(self, joints: list[str]):
        super().__init__(f"joints with no motion: {joints}")
        self.joints = joints


@dataclass
class CalibrationSession:
    arm_id: str
    state: CalibrationState = CalibrationState.HOMING
    homing_offsets: dict[str, int] = field(default_factory=dict)
    mins: dict[str, int] = field(default_factory=dict)
    maxes: dict[str, int] = field(default_factory=dict)
    proposed: dict[str, dict] | None = None
    current_on_disk: dict[str, dict] | None = None  # snapshot for the review diff

    def capture_neutral(self, handle) -> None:
        if self.state is not CalibrationState.HOMING:
            raise WrongStateError(f"capture_neutral requires HOMING, got {self.state.value}")
        bus = handle.robot.bus
        motors = list(bus.motors.keys())
        positions = bus.sync_read("Present_Position", motors, normalize=False)
        homings: dict[str, int] = {}
        for motor, pos in positions.items():
            model = bus.motors[motor].model
            max_res = bus.model_resolution_table[model] - 1
            offset = int(pos) - (max_res // 2)
            bus.write("Homing_Offset", motor, offset)
            homings[motor] = offset
        self.homing_offsets = homings
        # Seed the sweep accumulators from the just-captured position.
        post = bus.sync_read("Present_Position", motors, normalize=False)
        self.mins = {m: int(v) for m, v in post.items()}
        self.maxes = dict(self.mins)
        self.state = CalibrationState.SWEEPING

    def tick_sweep(self, handle) -> dict[str, int]:
        if self.state is not CalibrationState.SWEEPING:
            raise WrongStateError(f"tick_sweep requires SWEEPING, got {self.state.value}")
        bus = handle.robot.bus
        motors = list(bus.motors.keys())
        positions = bus.sync_read("Present_Position", motors, normalize=False)
        ticks: dict[str, int] = {}
        for motor, val in positions.items():
            ival = int(val)
            ticks[motor] = ival
            if ival < self.mins[motor]:
                self.mins[motor] = ival
            if ival > self.maxes[motor]:
                self.maxes[motor] = ival
        return ticks

    def finish_sweep(self, handle) -> dict[str, dict]:
        if self.state is not CalibrationState.SWEEPING:
            raise WrongStateError(f"finish_sweep requires SWEEPING, got {self.state.value}")
        unmoved = sorted(j for j in self.mins if self.mins[j] == self.maxes[j])
        if unmoved:
            raise UnmovedJointsError(unmoved)
        bus = handle.robot.bus
        proposed: dict[str, dict] = {}
        prior = self.current_on_disk or {}
        for motor in bus.motors.keys():
            prior_entry = prior.get(motor, {})
            proposed[motor] = {
                "id": int(bus.motors[motor].id),
                "drive_mode": int(prior_entry.get("drive_mode", 0)),
                "homing_offset": int(self.homing_offsets[motor]),
                "range_min": int(self.mins[motor]),
                "range_max": int(self.maxes[motor]),
            }
        self.proposed = proposed
        self.state = CalibrationState.REVIEW
        return proposed


class CalibrationManager:
    """Singleton; at most one session across the HMI."""

    def __init__(self) -> None:
        self.current: CalibrationSession | None = None
        self._handle = None  # type: ignore[var-annotated]  # set on start()

    def start(self, arms, arm_id: str) -> CalibrationSession:
        if self.current is not None:
            raise ConflictError(f"session active for arm {self.current.arm_id!r}")
        # Every configured arm must be in MANUAL.
        for handle in arms.values():
            if handle.guard.mode is not Mode.MANUAL:
                raise ConflictError(
                    f"arm {handle.config.id!r} is in mode {handle.guard.mode.value!r}, "
                    "all arms must be manual"
                )
        # Look up the target (raises KeyError if unknown — the route converts to 404).
        handle = arms[arm_id]
        handle.disable_torque()
        session = CalibrationSession(arm_id=arm_id)
        session.current_on_disk = _read_current_calibration(handle)
        self.current = session
        self._handle = handle
        logger.info("calibration: session started for arm %s", arm_id)
        return session

    def abort(self) -> None:
        if self.current is None:
            return
        if self._handle is not None and not self._handle.torque_enabled:
            self._handle.enable_torque()
        prev = self.current.arm_id
        self.current = None
        self._handle = None
        logger.info("calibration: session aborted (was arm %s)", prev)


def _read_current_calibration(handle) -> dict[str, dict] | None:
    """Return the arm's current calibration as a plain dict, or None if absent."""
    robot = handle.robot
    if robot is None or not getattr(robot, "calibration", None):
        return None
    out: dict[str, dict] = {}
    for motor, cal in robot.calibration.items():
        out[motor] = {
            "id": int(cal.id),
            "drive_mode": int(cal.drive_mode),
            "homing_offset": int(cal.homing_offset),
            "range_min": int(cal.range_min),
            "range_max": int(cal.range_max),
        }
    return out
```

- [ ] **Step 4: Run, verify all 9 tests pass**

```bash
pytest tests/test_calibration.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/calibration.py hmi/backend/tests/test_calibration.py
git commit -m "feat(hmi/backend): CalibrationSession + manager (TDD, no save yet)"
```

---

## Task 2: Save mechanics (backup, dual-path write, reload) — TDD

**Files:**
- Modify: `hmi/backend/haller_hmi/calibration.py`
- Modify: `hmi/backend/tests/test_calibration.py`

- [ ] **Step 1: Add failing tests for save**

Append to `tests/test_calibration.py`:

```python
import json
from pathlib import Path

from haller_hmi.calibration import CalibrationManager, _calibration_paths


def _populate_session_through_review(handle):
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, handle.config.id)
    session.capture_neutral(handle)
    handle.robot.bus.sync_read.return_value = {j: 500 for j in JOINTS}
    session.tick_sweep(handle)
    handle.robot.bus.sync_read.return_value = {j: 3600 for j in JOINTS}
    session.tick_sweep(handle)
    session.finish_sweep(handle)
    return mgr, arms


def test_save_writes_follower_file_and_creates_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Pre-existing calibration to be backed up
    follower_dir = tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower"
    follower_dir.mkdir(parents=True)
    existing = follower_dir / "haller_right.json"
    existing.write_text(json.dumps({"shoulder_pan": {"id": 1, "drive_mode": 0,
        "homing_offset": 0, "range_min": 100, "range_max": 200}}))

    handle = _make_handle("right")
    handle.robot.bus.motors = {
        j: MagicMock(model="sts3215", id=i + 1) for i, j in enumerate(JOINTS)
    }
    mgr, arms = _populate_session_through_review(handle)

    target, backup = mgr.save(arms)
    assert target == existing
    assert backup is not None and backup.exists()
    assert backup.name.startswith("haller_right.json.bak-")
    written = json.loads(target.read_text())
    assert set(written.keys()) == set(JOINTS)
    assert written["shoulder_pan"]["range_min"] == 500
    assert written["shoulder_pan"]["range_max"] == 3600


def test_save_also_updates_teleop_sibling(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cal_root = tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration"
    follower_dir = cal_root / "robots" / "so_follower"
    teleop_dir = cal_root / "teleoperators" / "so_leader"
    follower_dir.mkdir(parents=True)
    teleop_dir.mkdir(parents=True)
    teleop_path = teleop_dir / "haller_right.json"
    teleop_path.write_text(json.dumps({"shoulder_pan": {"id": 1, "drive_mode": 0,
        "homing_offset": 0, "range_min": 100, "range_max": 200}}))

    handle = _make_handle("right")
    handle.robot.bus.motors = {
        j: MagicMock(model="sts3215", id=i + 1) for i, j in enumerate(JOINTS)
    }
    mgr, arms = _populate_session_through_review(handle)
    mgr.save(arms)

    written = json.loads(teleop_path.read_text())
    assert set(written.keys()) == set(JOINTS)
    bak = list(teleop_dir.glob("haller_right.json.bak-*"))
    assert len(bak) == 1


def test_save_reconnects_arm(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower").mkdir(parents=True)
    handle = _make_handle("right")
    handle.robot.bus.motors = {
        j: MagicMock(model="sts3215", id=i + 1) for i, j in enumerate(JOINTS)
    }
    mgr, arms = _populate_session_through_review(handle)
    mgr.save(arms)
    handle.disconnect.assert_called_once()
    handle.connect.assert_called_once()
    assert mgr.current is None  # session cleared after save


def test_calibration_paths_returns_follower_and_existing_teleop(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cal_root = tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration"
    (cal_root / "robots" / "so_follower").mkdir(parents=True)
    (cal_root / "teleoperators" / "so_leader").mkdir(parents=True)
    teleop_path = cal_root / "teleoperators" / "so_leader" / "haller_right.json"
    teleop_path.write_text("{}")
    paths = _calibration_paths("haller_right")
    assert paths[0].name == "haller_right.json"
    assert "so_follower" in str(paths[0])
    assert teleop_path in paths
```

- [ ] **Step 2: Run, verify failures (no `save`, no `_calibration_paths`)**

```bash
pytest tests/test_calibration.py -v
```

- [ ] **Step 3: Add `save` to `CalibrationManager` + path helper**

Append to `haller_hmi/calibration.py`:

```python
import datetime as _dt
import shutil
from pathlib import Path


CALIB_ROOT_REL = ".cache/huggingface/lerobot/calibration"


def _cal_root() -> Path:
    return Path.home() / CALIB_ROOT_REL


def _calibration_paths(calibration_id: str) -> list[Path]:
    """Return the follower path plus every existing teleop sibling for this id.

    The follower path is always first and is always returned (even when the file
    doesn't exist yet — save() needs the target).
    """
    root = _cal_root()
    follower = root / "robots" / "so_follower" / f"{calibration_id}.json"
    paths: list[Path] = [follower]
    teleop_root = root / "teleoperators"
    if teleop_root.exists():
        for candidate in teleop_root.glob(f"*/{calibration_id}.json"):
            if candidate.is_file():
                paths.append(candidate)
    return paths


def _backup_path(path: Path) -> Path:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return path.with_name(path.name + f".bak-{ts}")


def _save_calibration_to(path: Path, proposed: dict[str, dict]) -> None:
    """Write the proposed JSON in the exact shape lerobot's _save_calibration emits.

    We re-use lerobot's draccus serializer via MotorCalibration so the file is
    byte-compatible with `lerobot-calibrate`.
    """
    import draccus
    from lerobot.motors.motors_bus import MotorCalibration

    payload = {
        motor: MotorCalibration(
            id=entry["id"],
            drive_mode=entry["drive_mode"],
            homing_offset=entry["homing_offset"],
            range_min=entry["range_min"],
            range_max=entry["range_max"],
        )
        for motor, entry in proposed.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f, draccus.config_type("json"):
        draccus.dump(payload, f, indent=4)


class CalibrationManager:  # extend the existing class
    # ... (keep existing __init__/start/abort) ...

    def save(self, arms) -> tuple[Path, Path | None]:
        if self.current is None or self.current.state is not CalibrationState.REVIEW:
            raise WrongStateError("save requires an active session in REVIEW")
        assert self.current.proposed is not None
        proposed = self.current.proposed
        arm_id = self.current.arm_id
        handle = arms[arm_id]
        calibration_id = handle.config.calibration_id

        paths = _calibration_paths(calibration_id)
        first_backup: Path | None = None
        for p in paths:
            bak: Path | None = None
            if p.exists():
                bak = _backup_path(p)
                shutil.move(str(p), str(bak))
                if first_backup is None:
                    first_backup = bak
            _save_calibration_to(p, proposed)

        # Reload the arm with the new calibration so subsequent commands use it.
        handle.disconnect()
        handle.connect()

        target = paths[0]
        # Clear session state — it's done.
        self.current = None
        self._handle = None
        logger.info("calibration: saved arm %s to %s (backup=%s)", arm_id, target, first_backup)
        return target, first_backup
```

Replace the `class CalibrationManager:` block from Task 1 in place — do not leave two classes. The full final class:

```python
class CalibrationManager:
    """Singleton; at most one session across the HMI."""

    def __init__(self) -> None:
        self.current: CalibrationSession | None = None
        self._handle = None  # set on start()

    def start(self, arms, arm_id: str) -> CalibrationSession:
        if self.current is not None:
            raise ConflictError(f"session active for arm {self.current.arm_id!r}")
        for handle in arms.values():
            if handle.guard.mode is not Mode.MANUAL:
                raise ConflictError(
                    f"arm {handle.config.id!r} is in mode {handle.guard.mode.value!r}, "
                    "all arms must be manual"
                )
        handle = arms[arm_id]
        handle.disable_torque()
        session = CalibrationSession(arm_id=arm_id)
        session.current_on_disk = _read_current_calibration(handle)
        self.current = session
        self._handle = handle
        logger.info("calibration: session started for arm %s", arm_id)
        return session

    def abort(self) -> None:
        if self.current is None:
            return
        if self._handle is not None and not self._handle.torque_enabled:
            self._handle.enable_torque()
        prev = self.current.arm_id
        self.current = None
        self._handle = None
        logger.info("calibration: session aborted (was arm %s)", prev)

    def save(self, arms) -> tuple[Path, Path | None]:
        if self.current is None or self.current.state is not CalibrationState.REVIEW:
            raise WrongStateError("save requires an active session in REVIEW")
        assert self.current.proposed is not None
        proposed = self.current.proposed
        arm_id = self.current.arm_id
        handle = arms[arm_id]
        calibration_id = handle.config.calibration_id

        paths = _calibration_paths(calibration_id)
        first_backup: Path | None = None
        for p in paths:
            if p.exists():
                bak = _backup_path(p)
                shutil.move(str(p), str(bak))
                if first_backup is None:
                    first_backup = bak
            _save_calibration_to(p, proposed)

        handle.disconnect()
        handle.connect()

        target = paths[0]
        self.current = None
        self._handle = None
        logger.info("calibration: saved arm %s to %s (backup=%s)", arm_id, target, first_backup)
        return target, first_backup
```

- [ ] **Step 4: Run, verify all 13 tests pass**

```bash
pytest tests/test_calibration.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/calibration.py hmi/backend/tests/test_calibration.py
git commit -m "feat(hmi/backend): calibration save (backup, dual-path, reload)"
```

---

## Task 3: Wire calibration tick into the telemetry broadcaster — TDD

**Files:**
- Modify: `hmi/backend/haller_hmi/telemetry.py`
- Modify: `hmi/backend/tests/test_telemetry.py`

- [ ] **Step 1: Add failing tests for the calibration block in the telemetry frame**

Append to `tests/test_telemetry.py`:

```python
import asyncio
from unittest.mock import MagicMock

import pytest

from haller_hmi.telemetry import TelemetryBroadcaster
from haller_hmi.calibration import CalibrationManager, CalibrationState


@pytest.mark.asyncio
async def test_frame_contains_calibration_block_when_session_active():
    handle = MagicMock()
    handle.state_snapshot.return_value = {
        "mode": "manual", "torque": False,
        "joints": {"gripper": {"pos": 0.0, "min": 0.0, "max": 100.0, "torque": False}},
    }
    handle.config = MagicMock(id="right")
    arms = MagicMock()
    arms.keys.return_value = ["right"]
    arms.__getitem__.return_value = handle

    ros = MagicMock()
    ros.snapshot.return_value = MagicMock(linear=0.0, angular=0.0, odom={}, scan_min_range=None)

    cal_mgr = CalibrationManager()
    # Move the session to SWEEPING manually (capture_neutral is fully tested elsewhere).
    handle.guard = MagicMock(); handle.guard.mode.__ne__ = lambda *_: False
    handle.robot = MagicMock()
    handle.robot.bus.motors = {"gripper": MagicMock(model="sts3215", id=6)}
    handle.robot.bus.model_resolution_table = {"sts3215": 4096}
    handle.robot.bus.sync_read.return_value = {"gripper": 2048}
    handle.robot.calibration = None
    handle.torque_enabled = True
    arms.values.return_value = [handle]
    session = cal_mgr.start(arms, "right")
    session.capture_neutral(handle)

    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, calibration=cal_mgr)
    bcast.start()
    try:
        sub = bcast.subscribe()
        frame = await asyncio.wait_for(sub.__anext__(), timeout=0.2)
    finally:
        await bcast.stop()

    block = frame["arms"]["right"]["calibration"]
    assert block["state"] == "sweeping"
    assert "ticks" in block and "gripper" in block["ticks"]
    assert "min" in block and "max" in block


@pytest.mark.asyncio
async def test_frame_has_no_calibration_block_when_idle():
    handle = MagicMock()
    handle.state_snapshot.return_value = {
        "mode": "manual", "torque": True, "joints": {},
    }
    arms = MagicMock()
    arms.keys.return_value = ["right"]
    arms.__getitem__.return_value = handle

    ros = MagicMock()
    ros.snapshot.return_value = MagicMock(linear=0.0, angular=0.0, odom={}, scan_min_range=None)

    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, calibration=CalibrationManager())
    bcast.start()
    try:
        sub = bcast.subscribe()
        frame = await asyncio.wait_for(sub.__anext__(), timeout=0.2)
    finally:
        await bcast.stop()
    assert "calibration" not in frame["arms"]["right"]
```

- [ ] **Step 2: Run, verify failures (broadcaster doesn't accept `calibration=`)**

```bash
pytest tests/test_telemetry.py -v
```

- [ ] **Step 3: Modify `TelemetryBroadcaster` to accept the manager and inject the block**

In `haller_hmi/telemetry.py`, change the constructor and `_build_frame`. Only the changed parts are shown — leave the rest as-is:

```python
class TelemetryBroadcaster:
    def __init__(self, arms, ros, hz: float = 20.0, teleop=None, calibration=None):
        self._arms = arms
        self._ros = ros
        self._teleop = teleop
        self._calibration = calibration   # CalibrationManager | None
        self._period = 1.0 / hz
        self._subscribers: list[asyncio.Queue] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    # ... start / stop / subscribe unchanged ...

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
        active = (
            self._calibration.current
            if self._calibration is not None
            else None
        )
        for arm_id in self._arms.keys():
            try:
                snap = self._arms[arm_id].state_snapshot()
            except Exception as e:
                logger.warning("arm %s telemetry failed: %s", arm_id, e)
                frame["alerts"].append({
                    "level": "warn",
                    "code": "arm_telemetry_failed",
                    "message": str(e),
                    "source": f"arm:{arm_id}",
                })
                continue
            if active is not None and active.arm_id == arm_id:
                snap["calibration"] = self._calibration_block(active, arm_id)
            frame["arms"][arm_id] = snap
        return frame

    def _calibration_block(self, session, arm_id: str) -> dict:
        block: dict = {"state": session.state.value}
        try:
            handle = self._arms[arm_id]
            if session.state.value == "homing":
                bus = handle.robot.bus
                motors = list(bus.motors.keys())
                block["ticks"] = {m: int(v) for m, v in
                                  bus.sync_read("Present_Position", motors, normalize=False).items()}
            elif session.state.value == "sweeping":
                block["ticks"] = session.tick_sweep(handle)
                block["min"] = dict(session.mins)
                block["max"] = dict(session.maxes)
        except Exception as e:
            logger.warning("calibration tick failed: %s", e)
            block["error"] = str(e)
        return block
```

- [ ] **Step 4: Run all telemetry tests, verify pass**

```bash
pytest tests/test_telemetry.py -v
```

Expected: 4 passed (2 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/telemetry.py hmi/backend/tests/test_telemetry.py
git commit -m "feat(hmi/backend): emit per-arm calibration block in telemetry"
```

---

## Task 4: FastAPI routes for the wizard — TDD

**Files:**
- Modify: `hmi/backend/haller_hmi/server.py`
- Modify: `hmi/backend/tests/test_routes.py`

- [ ] **Step 1: Extend the routes test fixture and add failing route cases**

In `tests/test_routes.py`, locate the `app_with_mocks` fixture and add a mocked `CalibrationManager` to it:

```python
# tests/test_routes.py — patch the existing fixture
from unittest.mock import MagicMock
from pathlib import Path

@pytest.fixture()
def app_with_mocks(monkeypatch, tmp_path):
    # ... existing arm/ros/preset mocks unchanged ...
    cal_mgr = MagicMock()
    cal_mgr.current = None

    def _start(arms_arg, arm_id):
        if arm_id != "right":
            raise KeyError(arm_id)
        from haller_hmi.calibration import CalibrationSession, CalibrationState
        cal_mgr.current = CalibrationSession(arm_id="right", state=CalibrationState.HOMING)
        return cal_mgr.current
    cal_mgr.start.side_effect = _start
    cal_mgr.save.return_value = (tmp_path / "haller_right.json",
                                 tmp_path / "haller_right.json.bak-2026-05-22T00-00-00Z")
    monkeypatch.setattr("haller_hmi.server.CalibrationManager", lambda *a, **kw: cal_mgr)
    monkeypatch.setattr("haller_hmi.server.calibration", cal_mgr, raising=False)
    # ... existing reload(srv_mod) at end of fixture remains ...
```

Add these new tests at the end:

```python
def test_get_calibration_status(app_with_mocks):
    r = app_with_mocks.get("/calibration/status")
    assert r.status_code == 200
    body = r.json()
    assert "arms" in body and isinstance(body["arms"], list)
    assert "current_session" in body


def test_post_calibration_start(app_with_mocks):
    r = app_with_mocks.post("/calibration/right/start")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "state": "homing"}


def test_post_calibration_start_unknown_arm_404(app_with_mocks):
    r = app_with_mocks.post("/calibration/left/start")
    assert r.status_code == 404


def test_post_calibration_capture_neutral(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    r = app_with_mocks.post("/calibration/right/capture_neutral")
    assert r.status_code == 200
    assert r.json()["state"] == "sweeping"


def test_post_calibration_finish_sweep(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    app_with_mocks.post("/calibration/right/capture_neutral")
    r = app_with_mocks.post("/calibration/right/finish_sweep")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "review"
    assert "proposed" in body


def test_post_calibration_save_returns_paths(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    app_with_mocks.post("/calibration/right/capture_neutral")
    app_with_mocks.post("/calibration/right/finish_sweep")
    r = app_with_mocks.post("/calibration/right/save")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "done"
    assert "path" in body and "backup_path" in body


def test_post_calibration_abort_is_idempotent(app_with_mocks):
    r1 = app_with_mocks.post("/calibration/right/abort")
    assert r1.status_code == 200
    r2 = app_with_mocks.post("/calibration/right/abort")
    assert r2.status_code == 200
```

Important: the `app_with_mocks` fixture relies on the real `CalibrationSession` driving the state machine on the mock manager. The simplest path is to back the mock with a real session and shim only `save`. Update the fixture's `cal_mgr` setup as follows (replace the lines starting `cal_mgr = MagicMock()` through the `monkeypatch.setattr` lines for it):

```python
from haller_hmi.calibration import (
    CalibrationManager, CalibrationSession, CalibrationState, _calibration_paths,
)
# Build a real manager but stub out save/disconnect/connect so we don't touch disk.
cal_mgr = CalibrationManager()

def _fake_save(arms_arg):
    cal_mgr.current = None
    return (tmp_path / "haller_right.json",
            tmp_path / "haller_right.json.bak-2026-05-22T00-00-00Z")
cal_mgr.save = _fake_save  # type: ignore[method-assign]
monkeypatch.setattr("haller_hmi.server.CalibrationManager", lambda *a, **kw: cal_mgr)
```

Add this to the arm mock so `capture_neutral` / sweep works against `bus`:

```python
arm.robot = MagicMock()
arm.robot.bus.motors = {
    "shoulder_pan": MagicMock(model="sts3215", id=1),
    "gripper":      MagicMock(model="sts3215", id=6),
}
arm.robot.bus.model_resolution_table = {"sts3215": 4096}
arm.robot.bus.sync_read.return_value = {"shoulder_pan": 1000, "gripper": 3500}
arm.robot.calibration = None
arm.guard = MagicMock()
arm.guard.mode = MagicMock()
arm.guard.mode.__ne__ = lambda *_args, **_kw: False  # always equal to MANUAL
arm.torque_enabled = True
```

- [ ] **Step 2: Run, verify failures (routes don't exist yet)**

```bash
pytest tests/test_routes.py -v
```

- [ ] **Step 3: Add the routes + lifespan wiring to `server.py`**

Add at module scope alongside the other globals (after `presets = PresetStore()`):

```python
from .calibration import (
    CalibrationManager,
    CalibrationError,
    ConflictError,
    UnmovedJointsError,
    WrongStateError,
    _calibration_paths,
)

calibration = CalibrationManager()
```

In the lifespan, pass the manager to the broadcaster:

```python
telemetry = TelemetryBroadcaster(arms, ros, hz=cfg.telemetry.hz,
                                 teleop=teleop, calibration=calibration)
```

Add at the bottom of the routes section:

```python
@app.get("/calibration/status")
def get_calibration_status():
    out_arms = []
    for h in arms.values():
        paths = _calibration_paths(h.config.calibration_id)
        target = paths[0]
        in_session = calibration.current is not None and calibration.current.arm_id == h.config.id
        out_arms.append({
            "id": h.config.id,
            "has_file": target.exists(),
            "path": str(target),
            "mtime": target.stat().st_mtime if target.exists() else None,
            "in_session": in_session,
        })
    session = calibration.current
    current = None
    if session is not None:
        current = {"arm_id": session.arm_id, "state": session.state.value}
        if session.state.value == "review":
            current["proposed"] = session.proposed
            current["current"] = session.current_on_disk
    return {"arms": out_arms, "current_session": current}


def _session_or_404_and_check(arm_id: str):
    handle = _arm_or_404(arm_id)
    return handle


@app.post("/calibration/{arm_id}/start")
async def post_calibration_start(arm_id: str):
    _arm_or_404(arm_id)
    try:
        session = calibration.start(arms, arm_id)
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "state": session.state.value}


@app.post("/calibration/{arm_id}/capture_neutral")
async def post_calibration_capture_neutral(arm_id: str):
    handle = _arm_or_404(arm_id)
    if calibration.current is None or calibration.current.arm_id != arm_id:
        raise HTTPException(status_code=409, detail="no active session for this arm")
    try:
        calibration.current.capture_neutral(handle)
    except WrongStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "state": calibration.current.state.value,
            "homing_offsets": calibration.current.homing_offsets}


@app.post("/calibration/{arm_id}/finish_sweep")
async def post_calibration_finish_sweep(arm_id: str):
    handle = _arm_or_404(arm_id)
    if calibration.current is None or calibration.current.arm_id != arm_id:
        raise HTTPException(status_code=409, detail="no active session for this arm")
    try:
        proposed = calibration.current.finish_sweep(handle)
    except UnmovedJointsError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except WrongStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {
        "ok": True,
        "state": calibration.current.state.value,
        "proposed": proposed,
        "current": calibration.current.current_on_disk,
    }


@app.post("/calibration/{arm_id}/save")
async def post_calibration_save(arm_id: str):
    _arm_or_404(arm_id)
    if calibration.current is None or calibration.current.arm_id != arm_id:
        raise HTTPException(status_code=409, detail="no active session for this arm")
    try:
        target, backup = calibration.save(arms)
    except WrongStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"save failed: {e}")
    return {"ok": True, "state": "done", "path": str(target),
            "backup_path": str(backup) if backup else None}


@app.post("/calibration/{arm_id}/abort")
async def post_calibration_abort(arm_id: str):
    _arm_or_404(arm_id)
    calibration.abort()
    return {"ok": True, "state": "aborted"}
```

- [ ] **Step 4: Run routes tests, verify pass**

```bash
pytest tests/test_routes.py -v
```

Expected: all routes tests pass (existing + 7 new).

- [ ] **Step 5: Run the full backend suite**

```bash
pytest -v
```

Expected: every test from prior tasks still green.

- [ ] **Step 6: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/server.py hmi/backend/tests/test_routes.py
git commit -m "feat(hmi/backend): /calibration/* routes (start/capture/sweep/save/abort)"
```

---

## Task 5: Extend `/estop` and `/arm/<id>/mode` — TDD

**Files:**
- Modify: `hmi/backend/haller_hmi/server.py`
- Modify: `hmi/backend/tests/test_routes.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_routes.py`:

```python
def test_estop_aborts_calibration_session(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    r = app_with_mocks.post("/estop")
    assert r.status_code == 200
    status = app_with_mocks.get("/calibration/status").json()
    assert status["current_session"] is None


def test_arm_mode_blocked_during_calibration(app_with_mocks):
    app_with_mocks.post("/calibration/right/start")
    r = app_with_mocks.post("/arm/right/mode", json={"mode": "auto"})
    assert r.status_code == 409
    assert "calibrat" in r.json()["detail"].lower()


def test_arm_mode_other_arm_unaffected_by_calibration(app_with_mocks):
    # The fixture only has 'right', so this test just confirms the gate is keyed by arm_id.
    app_with_mocks.post("/calibration/right/start")
    # /arm/right/mode is blocked above; a different arm_id 404s, not 409.
    r = app_with_mocks.post("/arm/left/mode", json={"mode": "manual"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run, verify failures**

```bash
pytest tests/test_routes.py::test_estop_aborts_calibration_session tests/test_routes.py::test_arm_mode_blocked_during_calibration -v
```

- [ ] **Step 3: Modify `/estop` and `/arm/<id>/mode` in `server.py`**

`/estop` — call `calibration.abort()` first:

```python
@app.post("/estop")
async def post_estop():
    logger.warning("E-STOP triggered")
    calibration.abort()
    for handle in arms.values():
        handle.disable_torque()
        handle.guard.set(Mode.STOP)
    teleop.stop()
    ros.zero_cmd_vel()
    return {"ok": True}
```

`/arm/<id>/mode` — refuse to change mode for the calibrating arm:

```python
@app.post("/arm/{arm_id}/mode")
async def post_arm_mode(arm_id: str, body: ArmModeBody):
    handle = _arm_or_404(arm_id)
    if calibration.current is not None and calibration.current.arm_id == arm_id:
        raise HTTPException(status_code=409,
                            detail=f"arm {arm_id!r} is being calibrated")
    try:
        new_mode = Mode(body.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid mode {body.mode!r}")
    handle.guard.set(new_mode)
    if new_mode is Mode.STOP:
        handle.disable_torque()
    else:
        if not handle.torque_enabled:
            handle.enable_torque()
    return {"ok": True, "mode": new_mode.value}
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_routes.py -v
```

- [ ] **Step 5: Run the full backend suite + the teleop tests specifically to check Task 5 didn't break anything**

```bash
pytest -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
cd ~/haller_ws
git add hmi/backend/haller_hmi/server.py hmi/backend/tests/test_routes.py
git commit -m "feat(hmi/backend): E-STOP aborts calibration; mode change blocked during session"
```

---

# Phase II — Frontend (Next.js + shadcn)

## Task 6: Typed REST client `lib/calibration.ts`

**Files:**
- Create: `hmi/frontend/lib/calibration.ts`
- Create: `hmi/frontend/__tests__/calibration.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// hmi/frontend/__tests__/calibration.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  fetchCalibrationStatus,
  startCalibration,
  captureNeutral,
  finishSweep,
  saveCalibration,
  abortCalibration,
} from "../lib/calibration";

const okJson = (body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

describe("calibration client", () => {
  it("GETs /calibration/status", async () => {
    (fetch as any).mockReturnValue(okJson({
      arms: [{ id: "right", has_file: true, path: "/x", mtime: 1, in_session: false }],
      current_session: null,
    }));
    const r = await fetchCalibrationStatus("http://b");
    expect(fetch).toHaveBeenCalledWith("http://b/calibration/status",
      expect.objectContaining({ method: "GET" }));
    expect(r.arms[0].id).toBe("right");
  });

  it("POSTs start/capture/finish/save/abort", async () => {
    (fetch as any).mockReturnValue(okJson({ ok: true, state: "homing" }));
    await startCalibration("http://b", "right");
    expect(fetch).toHaveBeenCalledWith("http://b/calibration/right/start",
      expect.objectContaining({ method: "POST" }));

    (fetch as any).mockReturnValue(okJson({ ok: true, state: "sweeping", homing_offsets: {} }));
    await captureNeutral("http://b", "right");
    (fetch as any).mockReturnValue(okJson({ ok: true, state: "review", proposed: {}, current: null }));
    await finishSweep("http://b", "right");
    (fetch as any).mockReturnValue(okJson({ ok: true, state: "done", path: "/x", backup_path: null }));
    await saveCalibration("http://b", "right");
    (fetch as any).mockReturnValue(okJson({ ok: true, state: "aborted" }));
    await abortCalibration("http://b", "right");
  });

  it("throws on non-OK response with the detail message", async () => {
    (fetch as any).mockReturnValue(Promise.resolve(
      new Response(JSON.stringify({ detail: "session active for arm 'right'" }), { status: 409 })));
    await expect(startCalibration("http://b", "right"))
      .rejects.toThrow(/session active/);
  });
});
```

- [ ] **Step 2: Run, verify failures**

```bash
cd ~/haller_ws/hmi/frontend
pnpm exec vitest run __tests__/calibration.test.ts
```

- [ ] **Step 3: Implement `lib/calibration.ts`**

```typescript
// hmi/frontend/lib/calibration.ts
export type CalibrationState = "homing" | "sweeping" | "review" | "done" | "aborted";

export interface CalibrationArmStatus {
  id: string;
  has_file: boolean;
  path: string;
  mtime: number | null;
  in_session: boolean;
}

export interface JointCalibration {
  id: number;
  drive_mode: number;
  homing_offset: number;
  range_min: number;
  range_max: number;
}

export interface CalibrationCurrentSession {
  arm_id: string;
  state: CalibrationState;
  proposed?: Record<string, JointCalibration>;
  current?: Record<string, JointCalibration> | null;
}

export interface CalibrationStatusResponse {
  arms: CalibrationArmStatus[];
  current_session: CalibrationCurrentSession | null;
}

async function postNoBody<T>(url: string): Promise<T> {
  const r = await fetch(url, { method: "POST" });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { detail = (await r.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export async function fetchCalibrationStatus(base: string): Promise<CalibrationStatusResponse> {
  const r = await fetch(`${base}/calibration/status`, { method: "GET" });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as CalibrationStatusResponse;
}

export const startCalibration = (base: string, id: string) =>
  postNoBody<{ ok: true; state: CalibrationState }>(`${base}/calibration/${id}/start`);

export const captureNeutral = (base: string, id: string) =>
  postNoBody<{ ok: true; state: CalibrationState; homing_offsets: Record<string, number> }>(
    `${base}/calibration/${id}/capture_neutral`);

export const finishSweep = (base: string, id: string) =>
  postNoBody<{ ok: true; state: CalibrationState;
               proposed: Record<string, JointCalibration>;
               current: Record<string, JointCalibration> | null }>(
    `${base}/calibration/${id}/finish_sweep`);

export const saveCalibration = (base: string, id: string) =>
  postNoBody<{ ok: true; state: "done"; path: string; backup_path: string | null }>(
    `${base}/calibration/${id}/save`);

export const abortCalibration = (base: string, id: string) =>
  postNoBody<{ ok: true; state: "aborted" }>(`${base}/calibration/${id}/abort`);
```

- [ ] **Step 4: Run, verify pass**

```bash
pnpm exec vitest run __tests__/calibration.test.ts
```

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/lib/calibration.ts hmi/frontend/__tests__/calibration.test.ts
git commit -m "feat(hmi/frontend): typed REST client for /calibration/* routes"
```

---

## Task 7: `CalibrationStatusCard` component

**Files:**
- Create: `hmi/frontend/components/CalibrationStatusCard.tsx`

- [ ] **Step 1: Install the shadcn `sheet` and `alert-dialog` primitives if not already present**

```bash
cd ~/haller_ws/hmi/frontend
pnpm dlx shadcn@latest add sheet alert-dialog
```

If they're already installed the CLI is a no-op.

- [ ] **Step 2: Implement the card**

```tsx
// hmi/frontend/components/CalibrationStatusCard.tsx
"use client";

import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { CalibrationArmStatus } from "@/lib/calibration";

interface Props {
  status: CalibrationArmStatus;
  canStart: boolean;            // false when any arm not in manual / another session running
  blockedReason?: string;
  onCalibrate: () => void;
}

export function CalibrationStatusCard({ status, canStart, blockedReason, onCalibrate }: Props) {
  const fileLabel = status.has_file
    ? `Calibrated · ${new Date((status.mtime ?? 0) * 1000).toLocaleString()}`
    : "No calibration file";
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-base font-medium">arm: {status.id}</CardTitle>
        <Badge variant={status.has_file ? "secondary" : "destructive"}>
          {status.has_file ? "OK" : "MISSING"}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-2">
        <p className="text-sm text-muted-foreground">{fileLabel}</p>
        <p className="text-xs text-muted-foreground break-all">{status.path}</p>
        <Button
          onClick={onCalibrate}
          disabled={!canStart || status.in_session}
          title={!canStart ? blockedReason : undefined}
          className="w-full"
        >
          {status.in_session ? "In session…" : "Calibrate"}
        </Button>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 3: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/components/CalibrationStatusCard.tsx hmi/frontend/components.json hmi/frontend/components/ui/sheet.tsx hmi/frontend/components/ui/alert-dialog.tsx 2>/dev/null || true
git add hmi/frontend
git commit -m "feat(hmi/frontend): CalibrationStatusCard + shadcn sheet/alert-dialog primitives"
```

---

## Task 8: Settings page wiring

**Files:**
- Modify: `hmi/frontend/app/settings/page.tsx`

- [ ] **Step 1: Read the current settings page to learn the layout**

```bash
cat ~/haller_ws/hmi/frontend/app/settings/page.tsx
```

- [ ] **Step 2: Add a "Calibration" section that lists `<CalibrationStatusCard>` per arm**

In `app/settings/page.tsx`:

```tsx
"use client";

import * as React from "react";
import { CalibrationStatusCard } from "@/components/CalibrationStatusCard";
import { CalibrationWizard } from "@/components/CalibrationWizard";
import { useTelemetry } from "@/lib/telemetry";
import { fetchCalibrationStatus, type CalibrationStatusResponse } from "@/lib/calibration";
import { BACKEND_URL } from "@/lib/config";

export default function SettingsPage() {
  const armsTelemetry = useTelemetry(s => s.lastFrame?.arms ?? {});
  const [status, setStatus] = React.useState<CalibrationStatusResponse | null>(null);
  const [wizardArm, setWizardArm] = React.useState<string | null>(null);

  const refresh = React.useCallback(async () => {
    try { setStatus(await fetchCalibrationStatus(BACKEND_URL)); } catch { /* offline */ }
  }, []);
  React.useEffect(() => { void refresh(); }, [refresh]);

  const allManual = Object.values(armsTelemetry).every(a => a.mode === "manual");
  const blockedReason = !allManual ? "Put every arm in manual first" : undefined;
  const canStart = allManual && status?.current_session == null;

  return (
    <main className="container mx-auto py-6 space-y-6">
      <h1 className="text-2xl font-semibold">Settings</h1>
      <section className="space-y-3">
        <h2 className="text-lg font-medium">Calibration</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {(status?.arms ?? []).map(s => (
            <CalibrationStatusCard
              key={s.id}
              status={s}
              canStart={!!canStart}
              blockedReason={blockedReason}
              onCalibrate={() => setWizardArm(s.id)}
            />
          ))}
        </div>
      </section>
      {wizardArm && (
        <CalibrationWizard
          armId={wizardArm}
          onClose={async () => { setWizardArm(null); await refresh(); }}
        />
      )}
    </main>
  );
}
```

- [ ] **Step 3: Manually verify the page renders**

```bash
cd ~/haller_ws/hmi/frontend
pnpm dev
```

Open `http://localhost:3000/settings`. Confirm the section appears with one card per configured arm; Calibrate disables when any arm is non-manual.

Stop the dev server.

- [ ] **Step 4: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/app/settings/page.tsx
git commit -m "feat(hmi/frontend): settings page Calibration section"
```

---

## Task 9: Dashboard banner when an arm has no calibration

**Files:**
- Modify: `hmi/frontend/app/page.tsx`

- [ ] **Step 1: Read the current dashboard to find the banner area**

```bash
cat ~/haller_ws/hmi/frontend/app/page.tsx
```

- [ ] **Step 2: Add a banner and wire it to the wizard**

Inside the existing dashboard component, after the top bar, add:

```tsx
import { fetchCalibrationStatus, type CalibrationStatusResponse } from "@/lib/calibration";
import { CalibrationWizard } from "@/components/CalibrationWizard";
import { BACKEND_URL } from "@/lib/config";

// inside the component:
const [calStatus, setCalStatus] = React.useState<CalibrationStatusResponse | null>(null);
const [wizardArm, setWizardArm] = React.useState<string | null>(null);
React.useEffect(() => {
  let cancelled = false;
  void (async () => {
    try {
      const s = await fetchCalibrationStatus(BACKEND_URL);
      if (!cancelled) setCalStatus(s);
    } catch { /* offline */ }
  })();
  return () => { cancelled = true; };
}, []);

const uncalibrated = (calStatus?.arms ?? []).filter(a => !a.has_file);
```

Render under the top bar:

```tsx
{uncalibrated.map(a => (
  <div key={a.id}
       className="bg-amber-900/30 border border-amber-700 rounded-md px-4 py-2 flex items-center justify-between">
    <span>Arm <strong>{a.id}</strong> has no calibration file.</span>
    <button
      className="underline text-amber-300"
      onClick={() => setWizardArm(a.id)}
    >Calibrate {a.id}</button>
  </div>
))}
{wizardArm && (
  <CalibrationWizard
    armId={wizardArm}
    onClose={async () => {
      setWizardArm(null);
      try { setCalStatus(await fetchCalibrationStatus(BACKEND_URL)); } catch {}
    }}
  />
)}
```

- [ ] **Step 3: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/app/page.tsx
git commit -m "feat(hmi/frontend): dashboard banner prompts wizard when arm uncalibrated"
```

---

## Task 10: `CalibrationWizard` — step 1 (homing) + skeleton

**Files:**
- Create: `hmi/frontend/components/CalibrationWizard.tsx`
- Create: `hmi/frontend/__tests__/CalibrationWizard.test.tsx`

- [ ] **Step 1: Write the failing tests (step 1 only — sweep/review follow in 11/12)**

```tsx
// hmi/frontend/__tests__/CalibrationWizard.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { CalibrationWizard } from "../components/CalibrationWizard";
import * as cal from "../lib/calibration";
import * as tele from "../lib/telemetry";

vi.mock("../lib/calibration");
vi.mock("../lib/telemetry");

beforeEach(() => {
  vi.resetAllMocks();
  (tele.useTelemetry as any).mockImplementation((sel: any) =>
    sel({ lastFrame: { arms: { right: { mode: "manual", torque: false,
      joints: {}, calibration: { state: "homing", ticks: { shoulder_pan: 2048 } } } } } }));
  (cal.fetchCalibrationStatus as any).mockResolvedValue({
    arms: [], current_session: { arm_id: "right", state: "homing" },
  });
});

describe("CalibrationWizard step 1 (homing)", () => {
  it("renders the live ticks table", async () => {
    render(<CalibrationWizard armId="right" onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText(/shoulder_pan/)).toBeInTheDocument());
    expect(screen.getByText("2048")).toBeInTheDocument();
  });

  it("clicking Capture neutral calls the backend and advances to step 2", async () => {
    (cal.captureNeutral as any).mockResolvedValue({ ok: true, state: "sweeping", homing_offsets: {} });
    // re-mock telemetry to advance to sweeping on next render
    (tele.useTelemetry as any).mockImplementation((sel: any) =>
      sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {},
        calibration: { state: "sweeping",
                       ticks: { shoulder_pan: 2048 },
                       min: { shoulder_pan: 2048 },
                       max: { shoulder_pan: 2048 } } } } } }));
    render(<CalibrationWizard armId="right" onClose={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: /capture neutral/i }));
    await waitFor(() => expect(cal.captureNeutral).toHaveBeenCalledWith(expect.any(String), "right"));
    expect(await screen.findByText(/done sweeping/i)).toBeInTheDocument();
  });

  it("calls /abort exactly once on unmount", async () => {
    const { unmount } = render(<CalibrationWizard armId="right" onClose={() => {}} />);
    unmount();
    await waitFor(() => expect(cal.abortCalibration).toHaveBeenCalledTimes(1));
  });
});
```

- [ ] **Step 2: Run, verify failures**

```bash
cd ~/haller_ws/hmi/frontend
pnpm exec vitest run __tests__/CalibrationWizard.test.tsx
```

- [ ] **Step 3: Extend the telemetry `ArmState` type to know about the calibration block**

In `hmi/frontend/lib/telemetry.ts`, change the `ArmState` type:

```ts
export type CalibrationTelemetryBlock = {
  state: "homing" | "sweeping" | "review" | "done" | "aborted";
  ticks?: Record<string, number>;
  min?: Record<string, number>;
  max?: Record<string, number>;
  error?: string;
};

export type ArmState = {
  mode: "auto" | "manual" | "stop";
  torque?: boolean;
  joints: Record<string, JointState>;
  calibration?: CalibrationTelemetryBlock;
};
```

- [ ] **Step 4: Implement the wizard with step 1 + skeleton for steps 2/3**

```tsx
// hmi/frontend/components/CalibrationWizard.tsx
"use client";

import * as React from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useTelemetry } from "@/lib/telemetry";
import {
  startCalibration, captureNeutral, finishSweep, saveCalibration, abortCalibration,
  fetchCalibrationStatus,
  type CalibrationState, type JointCalibration,
} from "@/lib/calibration";
import { BACKEND_URL } from "@/lib/config";

interface Props { armId: string; onClose: () => void; }

type ProposedMap = Record<string, JointCalibration>;

export function CalibrationWizard({ armId, onClose }: Props) {
  const armFrame = useTelemetry(s => s.lastFrame?.arms?.[armId]);
  const calBlock = armFrame?.calibration;
  const stateFromTele = (calBlock?.state ?? null) as CalibrationState | null;
  const [phase, setPhase] = React.useState<CalibrationState>("homing");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [proposed, setProposed] = React.useState<ProposedMap | null>(null);
  const [current, setCurrent] = React.useState<ProposedMap | null>(null);

  // Bootstrap: start the session if no session exists for this arm; re-attach if one already does.
  React.useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const status = await fetchCalibrationStatus(BACKEND_URL);
        if (cancelled) return;
        const active = status.current_session;
        if (active && active.arm_id === armId) {
          setPhase(active.state);
          if (active.state === "review") {
            setProposed(active.proposed ?? null);
            setCurrent(active.current ?? null);
          }
        } else if (!active) {
          await startCalibration(BACKEND_URL, armId);
          setPhase("homing");
        } else {
          setError(`Another session is active for arm '${active.arm_id}'.`);
        }
      } catch (e: any) { setError(e.message); }
    })();
    return () => { cancelled = true; };
  }, [armId]);

  // Mirror telemetry-driven state for non-review phases.
  React.useEffect(() => {
    if (stateFromTele && stateFromTele !== "review") setPhase(stateFromTele);
  }, [stateFromTele]);

  // Always abort on unmount unless the wizard reached "done" (Save handles its own cleanup).
  const isDoneRef = React.useRef(false);
  React.useEffect(() => () => {
    if (!isDoneRef.current) void abortCalibration(BACKEND_URL, armId).catch(() => {});
  }, [armId]);

  const guarded = async (fn: () => Promise<void>) => {
    setBusy(true); setError(null);
    try { await fn(); } catch (e: any) { setError(e.message); }
    finally { setBusy(false); }
  };

  const ticks = (calBlock?.ticks ?? {}) as Record<string, number>;
  const mins  = (calBlock?.min   ?? {}) as Record<string, number>;
  const maxes = (calBlock?.max   ?? {}) as Record<string, number>;
  const joints = Object.keys(ticks).length ? Object.keys(ticks)
                 : Object.keys(proposed ?? current ?? {});

  return (
    <Sheet open onOpenChange={(o) => { if (!o) onClose(); }}>
      <SheetContent side="right" className="w-[480px] sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Calibrate: {armId} arm</SheetTitle>
        </SheetHeader>

        {error && <p className="text-sm text-destructive py-2">{error}</p>}

        {phase === "homing" && (
          <section className="space-y-4 py-4">
            <h3 className="font-medium">Step 1 of 3 — Set neutral pose</h3>
            <p className="text-sm text-muted-foreground">
              Move the arm by hand into the pose you want to be "0°". Torque is off; the arm is back-drivable.
            </p>
            <Table>
              <TableHeader><TableRow><TableHead>Joint</TableHead><TableHead>Ticks</TableHead></TableRow></TableHeader>
              <TableBody>
                {joints.map(j => <TableRow key={j}><TableCell>{j}</TableCell><TableCell>{ticks[j] ?? "–"}</TableCell></TableRow>)}
              </TableBody>
            </Table>
            <div className="flex gap-2">
              <Button
                disabled={busy}
                onClick={() => guarded(async () => {
                  await captureNeutral(BACKEND_URL, armId);
                  setPhase("sweeping");
                })}
              >Capture neutral</Button>
              <Button variant="ghost" onClick={onClose}>Cancel</Button>
            </div>
          </section>
        )}

        {phase === "sweeping" && (
          <SweepingStep
            armId={armId} busy={busy} joints={joints}
            ticks={ticks} mins={mins} maxes={maxes}
            onDone={() => guarded(async () => {
              const res = await finishSweep(BACKEND_URL, armId);
              setProposed(res.proposed); setCurrent(res.current);
              setPhase("review");
            })}
            onCancel={onClose}
          />
        )}

        {phase === "review" && (
          <ReviewStep
            armId={armId} busy={busy} proposed={proposed} current={current}
            onSave={() => guarded(async () => {
              await saveCalibration(BACKEND_URL, armId);
              isDoneRef.current = true;
              onClose();
            })}
            onCancel={onClose}
          />
        )}
      </SheetContent>
    </Sheet>
  );
}

// Stubs for Task 11/12; implemented incrementally.
function SweepingStep(props: any) {
  return <section className="py-4"><Button onClick={props.onDone}>Done sweeping</Button></section>;
}
function ReviewStep(props: any) {
  return <section className="py-4"><Button onClick={props.onSave}>Save</Button></section>;
}
```

- [ ] **Step 5: Run wizard tests, verify pass**

```bash
pnpm exec vitest run __tests__/CalibrationWizard.test.tsx
```

- [ ] **Step 6: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/components/CalibrationWizard.tsx hmi/frontend/__tests__/CalibrationWizard.test.tsx hmi/frontend/lib/telemetry.ts
git commit -m "feat(hmi/frontend): CalibrationWizard step 1 (homing) + skeleton"
```

---

## Task 11: Wizard — step 2 (sweeping)

**Files:**
- Modify: `hmi/frontend/components/CalibrationWizard.tsx`

- [ ] **Step 1: Add a failing assertion for the sweep table**

Append to `__tests__/CalibrationWizard.test.tsx`:

```tsx
it("renders min | POS | max columns in the sweep step", async () => {
  (tele.useTelemetry as any).mockImplementation((sel: any) =>
    sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {},
      calibration: { state: "sweeping",
                     ticks: { shoulder_pan: 2200 },
                     min:   { shoulder_pan: 1000 },
                     max:   { shoulder_pan: 3500 } } } } } }));
  (cal.fetchCalibrationStatus as any).mockResolvedValue({
    arms: [], current_session: { arm_id: "right", state: "sweeping" },
  });
  render(<CalibrationWizard armId="right" onClose={() => {}} />);
  expect(await screen.findByText("1000")).toBeInTheDocument();
  expect(await screen.findByText("2200")).toBeInTheDocument();
  expect(await screen.findByText("3500")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run, verify failure**

```bash
pnpm exec vitest run __tests__/CalibrationWizard.test.tsx
```

- [ ] **Step 3: Replace the `SweepingStep` stub with the real implementation**

```tsx
function SweepingStep({
  busy, joints, ticks, mins, maxes, onDone, onCancel,
}: {
  busy: boolean;
  joints: string[];
  ticks: Record<string, number>;
  mins: Record<string, number>;
  maxes: Record<string, number>;
  onDone: () => void;
  onCancel: () => void;
}) {
  return (
    <section className="space-y-4 py-4">
      <h3 className="font-medium">Step 2 of 3 — Range of motion</h3>
      <p className="text-sm text-muted-foreground">
        Wiggle every joint to its physical limits. The table records the
        extremes; click <strong>Done sweeping</strong> when every joint has both a min and a max.
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Joint</TableHead>
            <TableHead className="text-right">min</TableHead>
            <TableHead className="text-right">POS</TableHead>
            <TableHead className="text-right">max</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {joints.map(j => (
            <TableRow key={j}>
              <TableCell>{j}</TableCell>
              <TableCell className="text-right tabular-nums">{mins[j] ?? "–"}</TableCell>
              <TableCell className="text-right tabular-nums font-medium">{ticks[j] ?? "–"}</TableCell>
              <TableCell className="text-right tabular-nums">{maxes[j] ?? "–"}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <div className="flex gap-2">
        <Button disabled={busy} onClick={onDone}>Done sweeping</Button>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Run, verify pass**

```bash
pnpm exec vitest run __tests__/CalibrationWizard.test.tsx
```

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/components/CalibrationWizard.tsx hmi/frontend/__tests__/CalibrationWizard.test.tsx
git commit -m "feat(hmi/frontend): CalibrationWizard step 2 (sweep table)"
```

---

## Task 12: Wizard — step 3 (review) + Save

**Files:**
- Modify: `hmi/frontend/components/CalibrationWizard.tsx`

- [ ] **Step 1: Add failing tests for review + Save**

Append to `__tests__/CalibrationWizard.test.tsx`:

```tsx
it("renders the diff table in review and Save calls saveCalibration", async () => {
  (tele.useTelemetry as any).mockImplementation((sel: any) =>
    sel({ lastFrame: { arms: { right: { mode: "manual", torque: false, joints: {} } } } }));
  (cal.fetchCalibrationStatus as any).mockResolvedValue({
    arms: [],
    current_session: {
      arm_id: "right", state: "review",
      proposed: { shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 0, range_min: 500, range_max: 3500 } },
      current:  { shoulder_pan: { id: 1, drive_mode: 0, homing_offset: 0, range_min: 100, range_max: 200  } },
    },
  });
  (cal.saveCalibration as any).mockResolvedValue({ ok: true, state: "done", path: "/x", backup_path: "/x.bak-1" });

  render(<CalibrationWizard armId="right" onClose={() => {}} />);
  expect(await screen.findByText("500")).toBeInTheDocument();   // proposed min
  expect(await screen.findByText("3500")).toBeInTheDocument();  // proposed max
  expect(await screen.findByText("100")).toBeInTheDocument();   // current min
  fireEvent.click(screen.getByRole("button", { name: /save/i }));
  await waitFor(() => expect(cal.saveCalibration).toHaveBeenCalledWith(expect.any(String), "right"));
});
```

- [ ] **Step 2: Run, verify failure**

```bash
pnpm exec vitest run __tests__/CalibrationWizard.test.tsx
```

- [ ] **Step 3: Replace the `ReviewStep` stub with the real implementation**

```tsx
function ReviewStep({
  busy, proposed, current, onSave, onCancel,
}: {
  busy: boolean;
  proposed: ProposedMap | null;
  current: ProposedMap | null;
  onSave: () => void;
  onCancel: () => void;
}) {
  const joints = Object.keys(proposed ?? {});
  return (
    <section className="space-y-4 py-4">
      <h3 className="font-medium">Step 3 of 3 — Review</h3>
      <p className="text-sm text-muted-foreground">
        Review the new calibration. The previous file is preserved as a <code>.bak-&lt;ts&gt;</code> sibling.
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Joint</TableHead>
            <TableHead className="text-right">old min → new min</TableHead>
            <TableHead className="text-right">old max → new max</TableHead>
            <TableHead className="text-right">old offset → new offset</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {joints.map(j => {
            const p = proposed?.[j]; const c = current?.[j];
            return (
              <TableRow key={j}>
                <TableCell>{j}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {c?.range_min ?? "—"} → <strong>{p?.range_min}</strong>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {c?.range_max ?? "—"} → <strong>{p?.range_max}</strong>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {c?.homing_offset ?? "—"} → <strong>{p?.homing_offset}</strong>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      <div className="flex gap-2">
        <Button disabled={busy} onClick={onSave}>Save</Button>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Run all wizard tests + the calibration client tests**

```bash
pnpm exec vitest run
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/components/CalibrationWizard.tsx hmi/frontend/__tests__/CalibrationWizard.test.tsx
git commit -m "feat(hmi/frontend): CalibrationWizard step 3 (review + save)"
```

---

## Task 13: ArmPanel gating during a session

**Files:**
- Modify: `hmi/frontend/components/ArmPanel.tsx`

- [ ] **Step 1: Add the calibration flag to the existing `disabled` predicate**

In `hmi/frontend/components/ArmPanel.tsx`, the current pattern (around line 26) is:

```tsx
const arm = useTelemetry((s) => s.lastFrame?.arms[armId]);
const teleop = useTelemetry((s) => s.lastFrame?.teleop);
const teleopRole: "leader" | "follower" | null =
    teleop?.running && teleop.leader === armId
      ? "leader"
      : teleop?.running && teleop.follower === armId
      ? "follower"
      : null;
// ...
const disabled = arm.mode !== "manual" || teleopRole !== null;
```

Add an `isCalibrating` derivation just after `teleopRole` and fold it into `disabled`:

```tsx
const isCalibrating = Boolean(arm?.calibration);
// ...
const disabled = arm.mode !== "manual" || teleopRole !== null || isCalibrating;
```

- [ ] **Step 2: Render a `CALIBRATING` chip in the same slot as the teleop chip**

The existing chip render (around line 81) looks like:

```tsx
{teleopRole ? (
  <span className="bg-emerald-700 text-emerald-50 ...">
    teleop · {teleopRole}
  </span>
) : (
  /* default chip */
)}
```

Extend the ternary so calibration takes precedence over teleop (a session can't run at the same time anyway, but be explicit):

```tsx
{isCalibrating ? (
  <span className="bg-blue-700 text-blue-50 ...">  {/* same other classes as teleop chip */}
    calibrating
  </span>
) : teleopRole ? (
  <span className="bg-emerald-700 text-emerald-50 ...">
    teleop · {teleopRole}
  </span>
) : (
  /* default chip */
)}
```

Copy the exact non-color Tailwind classes from the existing teleop chip — don't rewrite them.

- [ ] **Step 3: Manually verify**

```bash
cd ~/haller_ws/hmi/frontend && pnpm dev
```

Start the backend in another terminal, open the dashboard, click Calibrate from the banner, and confirm the corresponding ArmPanel renders the `calibrating` chip with all controls disabled. Cancel the wizard and confirm the panel restores.

- [ ] **Step 4: Commit**

```bash
cd ~/haller_ws
git add hmi/frontend/components/ArmPanel.tsx
git commit -m "feat(hmi/frontend): ArmPanel locks controls while calibration session active"
```

---

# Phase III — Docs + manual smoke

## Task 14: README updates + manual smoke procedure

**Files:**
- Modify: `README.md`
- Modify: `hmi/README.md`

- [ ] **Step 1: Top-level `README.md`**

Replace the "Next: HMI-driven calibration wizard." sentence with a statement of what shipped, and link to the new HMI section:

```diff
-> **Status (May 2026):** mobile base operational under ROS 2. Both SO-101 arms running through the unified HMI (FastAPI + Next.js + shadcn) on main. Per-arm controls (joint sliders, home, free-drive, pose presets) and a leader↔follower teleop launcher at 60 Hz are live. Next: HMI-driven calibration wizard.
+> **Status (May 2026):** mobile base operational under ROS 2. Both SO-101 arms running through the unified HMI (FastAPI + Next.js + shadcn) on main. Per-arm controls (joint sliders, home, free-drive, pose presets), a leader↔follower teleop launcher at 60 Hz, and an in-browser calibration wizard (homing + range-of-motion sweep + save, with automatic backup) are live.
```

- [ ] **Step 2: `hmi/README.md` — replace the "calibration caveat" paragraph**

Replace the paragraph that begins "If the two arms were calibrated independently…" with:

```markdown
**Calibrating an arm.** Open the Settings page and click **Calibrate** on the arm's card, or use the dashboard banner that appears when an arm has no calibration file. The wizard walks you through three steps:

1. **Set neutral pose** — torque off; pose the arm by hand; click *Capture neutral*.
2. **Range of motion** — wiggle every joint to its limits; the live table shows `min | POS | max`; click *Done sweeping*.
3. **Review** — verify the old → new diff; click *Save*. The previous calibration file (and any sibling teleop file) is preserved as `<id>.json.bak-<timestamp>`.

To fix a leader↔follower midpoint mismatch (`shoulder_lift` looks the most off), re-run the wizard on one arm while it holds the same physical neutral pose as the other.
```

- [ ] **Step 3: Manual smoke (no automated test) — document in `hmi/README.md` under "Troubleshooting"**

Append:

```markdown
### Verifying the calibration wizard end-to-end

1. Stop the backend; move `~/.cache/huggingface/lerobot/calibration/robots/so_follower/haller_follower.json` aside.
2. Restart the backend, open the dashboard — the banner should read "Arm right has no calibration file."
3. Click *Calibrate right*. Hand-pose the arm; click *Capture neutral*.
4. Wiggle every joint; verify the `min | POS | max` table moves; click *Done sweeping*.
5. Click *Save*. Confirm `haller_follower.json` is back on disk and a `haller_follower.json.bak-<ts>` sibling exists.
6. Repeat for the leader-as-follower (`haller_leader`) to verify the teleop sibling file at `teleoperators/*/haller_leader.json` is also written and backed up.
7. Run a short leader↔follower teleop session to confirm the new calibration loads correctly.
```

- [ ] **Step 4: Commit**

```bash
cd ~/haller_ws
git add README.md hmi/README.md
git commit -m "docs: HMI calibration wizard shipped (top-level README + hmi/README)"
```

---

## Wrap-up

After Task 14, run one more full sanity pass:

```bash
cd ~/haller_ws/hmi/backend && pytest -v
cd ~/haller_ws/hmi/frontend && pnpm exec vitest run
```

If any test fails, fix it before declaring the feature done. Then run the Task 14 Step 3 manual smoke against real hardware before merging.
