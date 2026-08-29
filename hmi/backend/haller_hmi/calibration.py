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
import time
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


class RecenterError(CalibrationError):
    """capture_neutral's homing write did not land the joints at centre.

    A sweep started off-centre wraps through 0/4095 and dies two steps later
    at finish_sweep, where it reads as operator error. Refuse at the step
    that actually failed.
    """

    def __init__(self, positions: dict[str, int]):
        pretty = ", ".join(f"{j}@{p}" for j, p in sorted(positions.items()))
        super().__init__(
            "re-centre failed — joints still off-centre after the homing "
            f"write: {pretty}. The servo did not follow the write; check for "
            "alarm states or a locked EEPROM, then retry the capture."
        )
        self.positions = positions


#: How far (ticks) a joint may sit from mid-range after the re-centre write
#: before capture refuses. The write lands exactly at mid; the slack covers
#: the operator's hand moving the held arm between the write and the check.
RECENTER_TOL_TICKS = 150


class SweepWrapError(CalibrationError):
    """finish_sweep found a range no physical joint has.

    Min/max accumulation amplifies a single bad sample into a ruined range,
    and a ruined range becomes a ruined ZERO (zero = range centre), so this
    refuses at the wizard instead of at the bench. Measured 2026-08-29: one
    sweep recorded ~356° on every joint but wrist_roll — near-full-tick-range
    garbage from wrap/corrupt reads — and the save went through silently.
    """

    def __init__(self, widths_deg: dict[str, float]):
        pretty = ", ".join(f"{j} {w:.0f}°" for j, w in sorted(widths_deg.items()))
        super().__init__(
            "sweep recorded a physically impossible range (encoder wrap or a "
            f"corrupt read got past the filter): {pretty}. Redo the capture "
            "pose and sweep again."
        )
        self.widths_deg = widths_deg


#: A raw sweep sample farther than this (ticks) from the last ACCEPTED sample
#: is held back until a second consecutive sample agrees with it (within the
#: same window). At the ~20 Hz sweep tick, 400 ticks is ≈ 700 °/s of hand
#: motion — no honest sweep moves that fast — while a 0/4095 wrap crossing
#: (≈ ±4000) or a corrupt read lands far beyond it. Real fast motion clears
#: the filter one tick late; a one-tick glitch never gets in.
SWEEP_MAX_JUMP_TICKS = 400

#: A non-full-turn joint whose recorded width exceeds this is wrap/garbage —
#: the widest real travel on the SO-101 is shoulder_pan at ~238°.
WRAP_SUSPECT_DEG = 300.0
FULL_TURN_JOINTS = frozenset({"wrist_roll"})


@dataclass
class CalibrationSession:
    arm_id: str
    state: CalibrationState = CalibrationState.HOMING
    homing_offsets: dict[str, int] = field(default_factory=dict)
    mins: dict[str, int] = field(default_factory=dict)
    maxes: dict[str, int] = field(default_factory=dict)
    proposed: dict[str, dict] | None = None
    current_on_disk: dict[str, dict] | None = None
    # Sweep glitch-filter state — see tick_sweep. Per motor: the last sample
    # folded into min/max, an unconfirmed outlier awaiting its second vote,
    # and how many samples have been accepted (1 = only the capture seed).
    _last_accepted: dict[str, int] = field(default_factory=dict)
    _pending: dict[str, int] = field(default_factory=dict)
    _accepted_count: dict[str, int] = field(default_factory=dict)
    #: The bus's in-memory calibration as it stood before capture reset it.
    #: `set_half_turn_homings` clears it (via `reset_calibration`), which is
    #: honest DURING a sweep — the old zeros are meaningless mid-wizard, and
    #: every normalized read failing loudly beats reading fiction — but it
    #: must come back on ABORT or a failed capture, or the whole backend is
    #: left unable to read the arm until a restart (lived 2026-08-29).
    prior_bus_calibration: dict | None = None

    def capture_neutral(self, handle) -> None:
        if self.state is not CalibrationState.HOMING:
            raise WrongStateError(f"capture_neutral requires HOMING, got {self.state.value}")
        bus = handle.robot.bus
        motors = list(bus.motors.keys())
        # Stock lerobot, not a reimplementation: `set_half_turn_homings`
        # resets the homing offset to ZERO (and the position limits to full
        # range) before measuring, then centres — the same call the kit's
        # one-command calibrate runs. The code this replaces wrote
        # `pos - mid` as an ABSOLUTE offset, which is correct only when the
        # prior offset is zero — and it never is: lerobot applies the loaded
        # calibration file's offsets to the servos at connect. With a
        # poisoned file on disk the "re-centred" joints landed at
        # mid + prior_offset, several against the 0/4095 wrap, and every
        # sweep wrapped: the wizard needed a good file to produce a good
        # file (bench-measured 2026-08-29).
        cal = getattr(bus, "calibration", None)
        self.prior_bus_calibration = dict(cal) if isinstance(cal, dict) and cal else None
        try:
            self.homing_offsets = {
                m: int(v) for m, v in bus.set_half_turn_homings(motors).items()
            }
            # Correct any joint the stock write could not centre. The
            # Homing_Offset register is sign-magnitude with an 11-BIT
            # magnitude (±2047 ticks); a joint whose raw encoder sits past
            # the encoder's zero point needs a nominal offset outside that
            # range, and the register silently truncates the write (measured
            # on the bench: writing 2315 stored 267, parking the gripper at
            # 268 instead of 2047 — both wizard runs of 2026-08-29). The
            # same physical shift expressed mod 4096 fits: offset −1780
            # centred the joint the stock call left at 268, bench-verified.
            # Firmware model (also lerobot feetech.py:285):
            # reported = actual − offset, mod 4096 — so correct on the
            # MEASURED reported error, the short way round, and iterate in
            # case a first pass lands partway.
            for _ in range(3):
                post = bus.sync_read("Present_Position", motors,
                                     normalize=False)
                fixes: dict[str, int] = {}
                for motor, v in post.items():
                    mid = (bus.model_resolution_table[
                        bus.motors[motor].model] - 1) // 2
                    err = ((int(v) - mid + 2048) % 4096) - 2048
                    if abs(err) > RECENTER_TOL_TICKS:
                        fixes[motor] = err
                if not fixes:
                    break
                for motor, err in fixes.items():
                    cur = int(bus.read("Homing_Offset", motor,
                                       normalize=False))
                    if 0x800 <= cur <= 0xFFF:   # undecoded sign-magnitude
                        cur = -(cur & 0x7FF)
                    new = ((cur + err + 2047) % 4096) - 2047
                    bus.write("Homing_Offset", motor, new)
                    self.homing_offsets[motor] = new
            # Trust nothing: verify the centre actually landed. This write
            # path has failed silently before (alarm states, locked EEPROM,
            # the truncation above), and an off-centre start is invisible
            # until the sweep wraps.
            post = bus.sync_read("Present_Position", motors, normalize=False)
            off_centre: dict[str, int] = {}
            for motor, v in post.items():
                mid = (bus.model_resolution_table[bus.motors[motor].model] - 1) // 2
                if abs(int(v) - mid) > RECENTER_TOL_TICKS:
                    off_centre[motor] = int(v)
            if off_centre:
                logger.warning("calibration: re-centre failed, post positions %s",
                               off_centre)
                raise RecenterError(off_centre)
        except Exception:
            # A capture that fails must hand back the calibration it reset,
            # or normalized reads stay broken far beyond the wizard.
            if self.prior_bus_calibration:
                try:
                    bus.write_calibration(self.prior_bus_calibration)
                except Exception:
                    logger.warning("calibration: could not restore prior "
                                   "calibration after failed capture",
                                   exc_info=True)
            raise
        self.mins = {m: int(v) for m, v in post.items()}
        self.maxes = dict(self.mins)
        self._last_accepted = dict(self.mins)
        self._pending = {}
        self._accepted_count = {m: 1 for m in self.mins}
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
            # Glitch filter — see SWEEP_MAX_JUMP_TICKS. Min/max accumulation
            # is maximally sensitive to a single outlier: one wrapped or
            # corrupt read anywhere in a 1000-read sweep ruins the range and
            # with it the zero. A sample implausibly far from the last
            # accepted one must be confirmed by its successor before it
            # counts; the raw value is still returned so the operator's live
            # readout never lies about what the bus said.
            prev = self._last_accepted.get(motor, ival)
            if abs(ival - prev) <= SWEEP_MAX_JUMP_TICKS:
                self._pending.pop(motor, None)
            else:
                pend = self._pending.get(motor)
                if pend is None or abs(ival - pend) > SWEEP_MAX_JUMP_TICKS:
                    self._pending[motor] = ival
                    continue        # unconfirmed outlier: not folded in
                # Confirmed jump. If the only accepted sample so far was the
                # capture seed, the SEED was the glitch — evict it from the
                # range instead of keeping a poisoned endpoint.
                self._pending.pop(motor, None)
                if self._accepted_count.get(motor, 0) <= 1:
                    self.mins[motor] = ival
                    self.maxes[motor] = ival
            self._last_accepted[motor] = ival
            self._accepted_count[motor] = self._accepted_count.get(motor, 0) + 1
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
        # Physical-plausibility guard — see SweepWrapError. The filter above
        # stops single glitches; this refuses whatever still got through,
        # because a range wider than the arm's widest real travel cannot be a
        # measurement and the zero derived from it lands nowhere physical.
        wrapped: dict[str, float] = {}
        for motor in self.mins:
            if motor in FULL_TURN_JOINTS:
                continue
            res = bus.model_resolution_table[bus.motors[motor].model]
            width = (self.maxes[motor] - self.mins[motor]) * 360.0 / res
            if width > WRAP_SUSPECT_DEG:
                wrapped[motor] = width
        if wrapped:
            logger.warning("calibration: sweep refused, widths %s | mins %s | "
                           "maxes %s", wrapped, self.mins, self.maxes)
            raise SweepWrapError(wrapped)
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


#: Seconds `CalibrationManager.start` waits between claiming the singleton
#: and its first bus op — long enough for an idle-sampler read already in
#: flight on its own thread to drain (two 20 Hz sampler periods).
START_DRAIN_S = 0.1


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
        session = CalibrationSession(arm_id=arm_id)
        # Claim the singleton BEFORE the first bus op. The idle sampler
        # stands down while `current` is set (the server wires its sample
        # source through this attribute), but a read already in flight on
        # its thread cannot be recalled — and the wizard's single-error
        # abort turns one collision into a dead session. The 0.1 s drain is
        # two 20 Hz sampler periods, comfortably longer than one read.
        self.current = session
        self._handle = handle
        try:
            time.sleep(START_DRAIN_S)
            handle.disable_torque()
            session.current_on_disk = _read_current_calibration(handle)
        except Exception:
            self.current = None
            self._handle = None
            raise
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

        # Files are committed to disk. Reload the arm so subsequent commands use the
        # new calibration — but clear the session regardless of reload outcome.
        try:
            handle.disconnect()
            handle.connect()
        except Exception as e:
            logger.warning("calibration: arm reload after save failed: %s", e)
            raise
        finally:
            target = paths[0]
            self.current = None
            self._handle = None
            logger.info("calibration: saved arm %s to %s (backup=%s)", arm_id, target, first_backup)

        return target, first_backup

    def abort(self) -> None:
        if self.current is None:
            return
        prev = self.current.arm_id
        # An aborted wizard must hand back the calibration capture reset, or
        # the backend cannot do a normalized read until a restart — the arm
        # panel sits on "awaiting telemetry" with the hardware fine.
        prior = self.current.prior_bus_calibration
        if prior and self._handle is not None:
            try:
                self._handle.robot.bus.write_calibration(prior)
            except Exception:
                logger.warning("calibration: could not restore prior "
                               "calibration during abort", exc_info=True)
        try:
            if self._handle is not None:
                self._handle.enable_torque()
        except Exception as e:
            logger.warning("calibration: enable_torque during abort failed: %s", e)
        finally:
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
