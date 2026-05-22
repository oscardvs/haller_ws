# hmi/backend/haller_hmi/calibration.py
"""Per-arm calibration session — homing offsets + range-of-motion sweep.

Only one session exists at a time across the whole HMI. The session reuses the
ArmHandle's existing MotorsBus (no second serial connection).

Save mechanics live in this module too (see Task 2 in the plan).
"""
from __future__ import annotations

import datetime as _dt
import enum
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

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
    current_on_disk: dict[str, dict] | None = None

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
        self._handle = None

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

    def save(self, arms) -> tuple[Path, Path | None]:
        if self.current is None or self.current.state is not CalibrationState.REVIEW:
            raise WrongStateError("save requires an active session in REVIEW")
        if self.current.proposed is None:
            raise WrongStateError("save invariant violated: proposed is None in REVIEW state")
        proposed = self.current.proposed
        arm_id = self.current.arm_id
        handle = self._handle
        if handle is None:
            raise WrongStateError("save invariant: _handle is None despite active session")
        calibration_id = handle.config.calibration_id

        paths = _calibration_paths(calibration_id)
        first_backup: Path | None = None
        for p in paths:
            tmp = p.with_suffix(p.suffix + ".tmp")
            _save_calibration_to(tmp, proposed)
            if p.exists():
                bak = _backup_path(p)
                shutil.move(str(p), str(bak))
                if first_backup is None:
                    first_backup = bak
            os.replace(str(tmp), str(p))

        handle.disconnect()
        handle.connect()

        target = paths[0]
        self.current = None
        self._handle = None
        logger.info("calibration: saved arm %s to %s (backup=%s)", arm_id, target, first_backup)
        return target, first_backup

    def abort(self) -> None:
        if self.current is None:
            return
        if self._handle is not None:
            self._handle.enable_torque()
        prev = self.current.arm_id
        self.current = None
        self._handle = None
        logger.info("calibration: session aborted (was arm %s)", prev)


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
    """Write the proposed JSON in the exact shape lerobot's _save_calibration emits."""
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
