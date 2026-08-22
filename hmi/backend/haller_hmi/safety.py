# hmi/backend/haller_hmi/safety.py
"""Safety primitives: joint-limit clamps, mode guards, E-STOP orchestrator."""
from __future__ import annotations

import enum
import math
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


# Largest gap that still earns proportional budget. Beyond this a caller has
# stalled, and a stalled caller must not bank a large jump.
MAX_STEP_DT_S = 0.1


def step_budget_deg(
    dt_s: float,
    max_speed_deg_s: float,
    max_dt_s: float = MAX_STEP_DT_S,
) -> float:
    """Degrees one joint may move on a single call, given time since the last.

    Steady-state speed is exactly `max_speed_deg_s` at any loop rate, because
    the budget is strictly proportional to real elapsed time. `max_dt_s` caps
    it so a stalled or long-idle caller cannot bank an unbounded step.
    """
    return max_speed_deg_s * min(max(dt_s, 0.0), max_dt_s)


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

    Every joint in `goal` must also appear in `current`; the caller guarantees
    this. Joints missing from `current` are dropped, so an unmatched key would
    silently omit that joint from every waypoint.
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
