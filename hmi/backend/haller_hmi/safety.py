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
