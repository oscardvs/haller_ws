"""SimArmHandle: drop-in for ArmHandle backed by a MuJoCoWorld.

Public surface matches ArmHandle exactly: connect, disconnect, send_goal,
disable_torque, enable_torque, state_snapshot, read_joints_deg, executor.
`home()` used to be part of that surface; it is now `motion.home(handle)`,
shared with ArmHandle instead of duplicated per class — see motion.py.

The HMI speaks LeRobot snake_case joint names (e.g. "shoulder_pan", "gripper")
because the real ArmHandle gets those names from LeRobot's SO101Follower
calibration. The vendored SO-101 MJCF uses CamelCase joint names ("Rotation",
"Jaw", etc.). This handle translates between them at the boundary so a single
HMI goal dict works against either a real or a sim arm.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import ArmConfig, MotionConfig
from ..safety import Mode, ModeGuard, clamp_joint_goal, limit_step, step_budget_deg
from .world import MuJoCoWorld

if TYPE_CHECKING:
    from ..motion import MoveExecutor

logger = logging.getLogger(__name__)


# LeRobot (snake_case, HMI-facing) → MJCF (CamelCase, world-facing).
# Order is the canonical LeRobot SO-101 joint order.
LEROBOT_TO_MJCF: dict[str, str] = {
    "shoulder_pan":  "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex":    "Elbow",
    "wrist_flex":    "Wrist_Pitch",
    "wrist_roll":    "Wrist_Roll",
    "gripper":       "Jaw",
}
MJCF_TO_LEROBOT: dict[str, str] = {v: k for k, v in LEROBOT_TO_MJCF.items()}


@dataclass
class SimArmHandle:
    config: ArmConfig
    world: MuJoCoWorld
    joint_limits_deg: dict[str, tuple[float, float]] = field(default_factory=dict)
    guard: ModeGuard = field(default_factory=lambda: ModeGuard(Mode.MANUAL))
    torque_enabled: bool = True
    motion: MotionConfig = field(default_factory=MotionConfig)
    # init=False: an `executor=` constructor argument would otherwise
    # type-check and then be silently discarded, since __post_init__
    # overwrites it unconditionally below. compare=False/repr=False: two
    # otherwise-identical handles must still compare equal and print sanely —
    # MoveExecutor has no meaningful equality of its own and holds a live
    # thread/lock, neither of which belongs in a dataclass repr.
    executor: MoveExecutor | None = field(
        init=False, repr=False, compare=False, default=None,
    )  # set in __post_init__
    _last_commanded: dict[str, float] | None = None
    _last_command_at: float | None = None

    def __post_init__(self) -> None:
        # Deferred, not because a cycle exists today — motion.py imports only
        # stdlib and .safety, and .safety imports only stdlib, so a top-level
        # `from ..motion import MoveExecutor` here would work right now — but
        # to keep it that way. This is the natural place a future edit to
        # either module would grow one, so the import waits until it's
        # actually needed, here at construction time. The TYPE_CHECKING
        # import above is separate: it's only so the annotation above
        # resolves for static analysis, and never runs.
        from ..motion import MoveExecutor
        self.executor = MoveExecutor(self)

    @property
    def _prefix(self) -> str:
        return f"{self.config.sim_arm_name}_"

    def connect(self) -> None:
        """Populate joint_limits_deg using LeRobot names (HMI-facing). Joints
        absent from the MJCF (e.g. if a future MJCF lacks one) are skipped."""
        self.joint_limits_deg = {}
        for lerobot_name, mjcf_short in LEROBOT_TO_MJCF.items():
            mjcf_name = f"{self._prefix}{mjcf_short}"
            try:
                self.joint_limits_deg[lerobot_name] = self.world.joint_range_deg(
                    self.config.sim_arm_name, mjcf_name
                )
            except KeyError:
                logger.warning("sim arm %s: joint %s (%s) missing in MJCF; skipping",
                               self.config.id, lerobot_name, mjcf_name)
        logger.info("sim arm %s connected; joints: %s",
                    self.config.id, list(self.joint_limits_deg))

    def disconnect(self) -> None:
        # World lifecycle is owned by ArmManager; nothing per-arm to release.
        pass

    def send_goal(self, goal_deg: dict[str, float]) -> dict[str, float]:
        self.guard.assert_manual()
        clamped = clamp_joint_goal(goal_deg, self.joint_limits_deg)
        now = time.monotonic()
        if self._last_commanded is None:
            self._last_commanded = self.read_joints_deg()
        if any(j not in self._last_commanded for j in clamped):
            # A flaky read can drop a joint at seed time, or leave it dropped
            # from an earlier call. Retry so it rejoins as soon as one read
            # succeeds, rather than staying unmeasured indefinitely.
            self._last_commanded = {**self._last_commanded, **self.read_joints_deg()}
        # Don't move what you can't measure: a joint missing from
        # `_last_commanded` has no reference for limit_step to cap against,
        # and limit_step's own contract is to pass such a joint through
        # UNCAPPED — exactly the fail-open a flaky read must not produce. Drop
        # it here instead; it rejoins on whichever later call next reads it.
        measurable = {j: v for j, v in clamped.items() if j in self._last_commanded}
        # No previous call to measure real elapsed time from: a seeded first
        # call still earns one ramp period's worth of motion. Every call
        # after that is governed by real elapsed time — step_budget_deg
        # itself has no floor, or a loop faster than ramp_hz would land back
        # on a fixed per-call cap and reintroduce the over-speed this fixed.
        dt = (1.0 / self.motion.ramp_hz) if self._last_command_at is None \
            else (now - self._last_command_at)
        max_step_deg = step_budget_deg(dt, self.motion.max_speed_deg_s)
        capped = limit_step(self._last_commanded, measurable, max_step_deg)
        # Translate snake_case → CamelCase + add arm prefix for the world.
        mjcf_goal = {
            f"{self._prefix}{LEROBOT_TO_MJCF[j]}": v
            for j, v in capped.items()
            if j in LEROBOT_TO_MJCF
        }
        self.world.write_ctrl_deg(self.config.sim_arm_name, mjcf_goal)
        self._last_commanded = {**self._last_commanded, **capped}
        self._last_command_at = now
        return capped

    def disable_torque(self) -> None:
        self.world.set_arm_torque(self.config.sim_arm_name, enabled=False)
        self.torque_enabled = False
        self._last_commanded = None
        self._last_command_at = None

    def enable_torque(self) -> None:
        self.world.set_arm_torque(self.config.sim_arm_name, enabled=True)
        self.torque_enabled = True
        self._last_commanded = None
        self._last_command_at = None

    def read_joints_deg(self) -> dict[str, float]:
        """Latest joint positions in degrees, keyed by LeRobot snake_case names —
        matches ArmHandle.read_joints_deg() exactly so callers (teleop loop,
        telemetry, etc.) don't care whether the arm is real or sim."""
        raw = self.world.read_qpos_deg(self.config.sim_arm_name)
        prefix = self._prefix
        out: dict[str, float] = {}
        for mjcf_key, value in raw.items():
            if not mjcf_key.startswith(prefix):
                continue
            mjcf_short = mjcf_key[len(prefix):]
            lerobot = MJCF_TO_LEROBOT.get(mjcf_short)
            if lerobot is not None:
                out[lerobot] = value
        return out

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
