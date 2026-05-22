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
        # with range_min/range_max in raw ticks. We center on the mid-point of
        # that range and convert to degrees — symmetric clamping, independent
        # of the motor's homing_offset.
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
