"""SimArmHandle: drop-in for ArmHandle backed by a MuJoCoWorld.

Public surface matches ArmHandle exactly: connect, disconnect, send_goal,
home, disable_torque, enable_torque, state_snapshot, read_joints_deg.

The HMI speaks LeRobot snake_case joint names (e.g. "shoulder_pan", "gripper")
because the real ArmHandle gets those names from LeRobot's SO101Follower
calibration. The vendored SO-101 MJCF uses CamelCase joint names ("Rotation",
"Jaw", etc.). This handle translates between them at the boundary so a single
HMI goal dict works against either a real or a sim arm.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import ArmConfig
from ..safety import Mode, ModeGuard, clamp_joint_goal
from .world import MuJoCoWorld

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
        # Translate snake_case → CamelCase + add arm prefix for the world.
        mjcf_goal = {
            f"{self._prefix}{LEROBOT_TO_MJCF[j]}": v
            for j, v in clamped.items()
            if j in LEROBOT_TO_MJCF
        }
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
