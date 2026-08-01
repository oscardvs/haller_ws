# hmi/backend/haller_hmi/arm.py
"""Per-arm wrapper around `lerobot.robots.so_follower.SO101Follower`.

The HMI's safety surface lives on top of lerobot's raw API:
  - mode gating (only Mode.MANUAL accepts goals from the HMI)
  - joint-limit clamping in DEGREES against the calibration
  - keys translated between HMI ("shoulder_pan": deg) and lerobot ("shoulder_pan.pos": deg)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .config import ArmConfig, MotionConfig
from .safety import Mode, ModeGuard, clamp_joint_goal, limit_step, step_budget_deg

if TYPE_CHECKING:
    from .motion import MoveExecutor

logger = logging.getLogger(__name__)


# Conservative defaults if calibration metadata doesn't expose explicit deg ranges;
# we derive these per-joint from each motor's calibrated range converted to degrees.
TICKS_PER_REV = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_REV


def _write_calibration_to_motors(robot: SO101Follower) -> None:
    """Non-interactive stand-in for `SOFollower.calibrate()`.

    lerobot resolves a motors-vs-file calibration mismatch by asking on stdin
    whether to adopt the file or re-run calibration. Answering ENTER — the
    default — writes the file into the motors' registers, and that is the only
    answer the HMI ever wants: the file is the artefact the calibration wizard
    just committed. Since uvicorn has no stdin, leaving lerobot to ask raises
    EOFError instead, so we answer it ourselves.
    """
    if not robot.calibration:
        raise RuntimeError(
            f"arm {robot.id!r} has no calibration file at {robot.calibration_fpath}; "
            "run the calibration wizard before connecting"
        )
    logger.info("arm %s: writing calibration from %s into motors",
                robot.id, robot.calibration_fpath)
    robot.bus.write_calibration(robot.calibration)


@dataclass
class ArmHandle:
    config: ArmConfig
    joint_limits_deg: dict[str, tuple[float, float]] = field(default_factory=dict)
    guard: ModeGuard = field(default_factory=lambda: ModeGuard(Mode.MANUAL))
    robot: SO101Follower | None = None
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
        # `from .motion import MoveExecutor` here would work right now — but
        # to keep it that way. This is the natural place a future edit to
        # either module would grow one, so the import waits until it's
        # actually needed, here at construction time. The TYPE_CHECKING
        # import above is separate: it's only so the annotation above
        # resolves for static analysis, and never runs.
        from .motion import MoveExecutor
        self.executor = MoveExecutor(self)

    def connect(self) -> None:
        cfg = SO101FollowerConfig(
            port=self.config.port,
            id=self.config.calibration_id,
            use_degrees=True,
        )
        self.robot = SO101Follower(cfg)
        # connect() delegates to robot.calibrate() whenever the motors disagree
        # with the calibration file — precisely the state the wizard leaves
        # behind the instant it writes a new one. Substitute the prompt-free
        # equivalent before connecting, so lerobot keeps its own ordering and
        # still applies the calibration before configure() touches the motors.
        self.robot.calibrate = partial(_write_calibration_to_motors, self.robot)
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
        assert self.robot is not None
        clamped = clamp_joint_goal(goal_deg, self.joint_limits_deg)
        now = time.monotonic()
        if self._last_commanded is None:
            # First command since connect or a torque toggle: seed from a real
            # read. Every later call limits against the last command, so the
            # 60 Hz teleop path costs no extra serial traffic.
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
        # lerobot expects keys suffixed with ".pos"
        action = {f"{j}.pos": v for j, v in capped.items()}
        self.robot.send_action(action)
        self._last_commanded = {**self._last_commanded, **capped}
        self._last_command_at = now
        return capped

    def disable_torque(self) -> None:
        if self.robot is not None:
            self.robot.bus.disable_torque()
            self.torque_enabled = False
            self._last_commanded = None
            self._last_command_at = None

    def enable_torque(self) -> None:
        if self.robot is not None:
            self.robot.bus.enable_torque()
            self.torque_enabled = True
            self._last_commanded = None
            self._last_command_at = None

    def read_joints_deg(self) -> dict[str, float]:
        """Latest joint positions in degrees, keyed by joint name (no `.pos` suffix).

        Filters lerobot's observation dict to only `<joint>.pos` entries that
        belong to a known joint, and strips the suffix so callers don't need to.
        """
        assert self.robot is not None
        obs = self.robot.get_observation()
        out: dict[str, float] = {}
        for joint in self.joint_limits_deg:
            key = f"{joint}.pos"
            if key in obs:
                out[joint] = float(obs[key])
        return out

    def state_snapshot(self) -> dict:
        assert self.robot is not None
        obs = self.robot.get_observation()
        joints = {}
        for joint, (lo, hi) in self.joint_limits_deg.items():
            joints[joint] = {
                "pos": float(obs.get(f"{joint}.pos", 0.0)),
                "min": float(lo),
                "max": float(hi),
                "torque": self.torque_enabled,
            }
        return {
            "mode": self.guard.mode.value,
            "torque": self.torque_enabled,
            "joints": joints,
        }


class ArmManager:
    """Lookup-by-id collection of arm handles (real or sim)."""

    def __init__(self, arm_configs: list[ArmConfig],
                 motion: MotionConfig | None = None):
        self._configs = [c for c in arm_configs if c.enabled]
        self._motion = motion or MotionConfig()
        self._handles: dict[str, "ArmHandle | SimArmHandle"] = {}
        self._world = None  # lazily constructed if any sim arm/camera needs it

    def _ensure_world(self) -> "MuJoCoWorld":
        if self._world is not None:
            return self._world
        from .sim.builder import build_scene
        from .sim.world import MuJoCoWorld

        sim_arm_names = [c.sim_arm_name for c in self._configs
                         if c.source == "sim" and c.sim_arm_name is not None]
        mjcf_xml, arm_joint_map = build_scene(arms=sim_arm_names, cubes=0)
        self._world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
        self._world.start()
        return self._world

    def connect_all(self) -> None:
        from .calibration_bootstrap import ensure_follower_calibrations
        from .config import resolve_motion
        from .sim.arm import SimArmHandle

        real_configs = [c for c in self._configs if c.source == "real"]
        if real_configs:
            ensure_follower_calibrations(real_configs)

        for cfg in self._configs:
            if cfg.source == "sim":
                if not cfg.sim_arm_name:
                    raise ValueError(
                        f"arm {cfg.id!r} has source=sim but no sim_arm_name"
                    )
                world = self._ensure_world()
                handle = SimArmHandle(cfg, world=world)
                handle.connect()
            else:
                handle = ArmHandle(cfg)
                handle.connect()
            handle.motion = resolve_motion(cfg, self._motion)
            self._handles[cfg.id] = handle

    def disconnect_all(self) -> None:
        for handle in self._handles.values():
            handle.disconnect()
        if self._world is not None:
            self._world.stop()
            self._world = None

    def world(self) -> "MuJoCoWorld | None":
        """Exposed so SimCamera and SimLeaderTeleop can share the same world."""
        return self._world

    def __getitem__(self, arm_id: str):
        if arm_id not in self._handles:
            raise KeyError(f"unknown arm id {arm_id!r}; known: {list(self._handles)}")
        return self._handles[arm_id]

    def values(self):
        return self._handles.values()

    def keys(self):
        return self._handles.keys()
