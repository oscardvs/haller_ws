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


# ---- mouth-open dead-man clutch ---------------------------------------
#
# Pure policy: score in, boolean out. No clock, no lock, no I/O — the caller
# supplies `held_ms` and `stale`. Kept here beside clamp_joint_goal because
# every gate that can stop the arms lives backend-side, and this is one.
#
# NOTE: engage and release are deliberately asymmetric. Engaging demands a
# sustained hold above a high threshold; releasing takes one sample and no
# hold at all. Every ambiguous or faulted state resolves to disengaged.

MOUTH_MIN_SEPARATION = 0.25   # min (open_min - talk_max) for any safe threshold
MOUTH_ENGAGE_FRAC = 0.60      # t_engage position within the calibrated gap
MOUTH_RELEASE_FRAC = 0.30     # t_release position; the difference is hysteresis
MOUTH_HOLD_MS = 200.0         # sustained time above t_engage before engaging


@dataclass(frozen=True)
class MouthClutchCalib:
    """Per-operator jaw-open calibration, both raw MediaPipe blendshape scores.

    talk_max: highest jawOpen observed while speaking normally — the noise
              floor that must never engage.
    open_min: lowest jawOpen held during a deliberate wide open.
    """

    talk_max: float
    open_min: float


def mouth_clutch_thresholds(c: MouthClutchCalib) -> tuple[float, float] | None:
    """(t_engage, t_release), or None when no safe threshold exists.

    Returns None when the operator's speech range overlaps their deliberate
    open. There is no correct threshold in that case, so mouth mode refuses
    to arm rather than picking a dangerous constant.
    """
    gap = c.open_min - c.talk_max
    if gap < MOUTH_MIN_SEPARATION:
        return None
    return (c.talk_max + MOUTH_ENGAGE_FRAC * gap,
            c.talk_max + MOUTH_RELEASE_FRAC * gap)


def mouth_clutch_decision(
    score: float | None,
    thresholds: tuple[float, float],
    held_ms: float,
    stale: bool,
    engaged: bool,
) -> bool:
    """Next engaged state.

    score:   most recent jawOpen sample, or None if none has ever arrived.
             A decimated frame is NOT None — the caller passes the last
             known score and reports ageing through `stale`.
    held_ms: how long `score` has been continuously at or above t_engage.
    stale:   the last face sample is older than the staleness budget.
    engaged: current state, for hysteresis.
    """
    t_engage, t_release = thresholds
    if stale or score is None:
        return False
    if engaged:
        return score >= t_release
    return score >= t_engage and held_ms >= MOUTH_HOLD_MS
