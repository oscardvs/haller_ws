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

from lerobot.motors.encoding_utils import decode_sign_magnitude
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
    # Which of the three effort-read paths this arm settled on — decided once
    # in connect() by _probe_effort_path(), never per tick. See read_effort_norm.
    _effort_mode: str = "unprobed"
    _effort_fail_streak: int = 0

    # ---- effort (servo load) constants -----------------------------------
    #
    # UNIT of every effort number this class returns: a DIMENSIONLESS SIGNED
    # FRACTION of the joint's own torque limit, clipped to [-1, 1]. Sign is the
    # drive direction; |v| -> 1 at stall (the servo is holding against
    # something). It is NOT N·m and NOT amps.
    #
    # Why a fraction and not a physical unit: the STS3215 can only report
    # `Present_Load`, a signed per-mille of maximum torque (really the PWM duty
    # the servo is applying), while the MuJoCo sim reports actuator force in
    # N·m. Those two cannot be unit-matched, so both sides normalise against
    # their OWN saturation limit — real: Present_Load/1000, sim:
    # actuator_force/forcerange — and one dataset column keeps one meaning
    # whichever rig recorded the episode. See sim/world.read_effort_norm.
    EFFORT_UNIT = "fraction_of_torque_limit"

    # Present_Load is sign-magnitude with the direction in bit 10 (lerobot's
    # STS_SMS_SERIES_ENCODINGS_TABLE), magnitude in per-mille of max torque.
    _LOAD_SIGN_BIT = 10
    _LOAD_FULL_SCALE = 1000.0

    # ONE block read, not two sync_reads. In the STS control table
    # Present_Position(56,2) .. Present_Current(69,2) are contiguous, so a
    # single 15-byte read starting at 56 returns position AND load in the same
    # round trip: the recorder gets an effort channel for +13 bytes of rx per
    # motor instead of doubling the number of read round trips on a bus that
    # is already being written at 60 Hz by the teleop thread.
    _BLOCK_ADDR = 56
    _BLOCK_LEN = 15
    _POS_ADDR = 56
    _LOAD_ADDR = 60
    # Consecutive block-read failures before giving up on the fast path. One
    # transient comm error must not permanently cost an extra round trip per
    # tick, but a bus/firmware that genuinely can't serve the block read must
    # not be retried (and time out) forever either.
    _EFFORT_DEMOTE_AFTER = 3

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
        # Decide the effort-read path ONCE, here, so the 20 Hz telemetry tick
        # never pays for probing and the log says which path is live exactly
        # once per connect instead of forty times a second.
        self._effort_mode = self._probe_effort_path()
        self._effort_fail_streak = 0
        logger.info("arm %s: effort read path = %s", self.config.id, self._effort_mode)

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
        # A later connect() re-probes: the same arm can come back on a
        # different adapter, or with different firmware, than it left on.
        self._effort_mode = "unprobed"
        self._effort_fail_streak = 0

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

    # ---- effort ----------------------------------------------------------

    @classmethod
    def _load_fraction(cls, signed_load: int) -> float:
        """Signed Present_Load counts -> fraction of torque limit, clipped.

        Clipping is not paranoia: the register is a 10-bit magnitude, so 1023
        counts is 1.023 of "full torque". Everything at or past 1.0 means the
        same thing — the servo is saturated — and clipping keeps the recorded
        column inside the [-1, 1] contract the sim side also honours.
        """
        return max(-1.0, min(1.0, float(signed_load) / cls._LOAD_FULL_SCALE))

    def _read_block(self) -> tuple[dict[str, float], dict[str, float]]:
        """One bus round trip -> (positions in deg, effort fractions).

        Uses lerobot's PRIVATE bus API on purpose — there is no public call
        that reads two registers in one packet — so every caller wraps this in
        a try/except and falls back. What the private bits are:
          - `_setup_sync_reader(ids, addr, len)` points the shared GroupSyncRead
            at a byte range instead of a named register,
          - `getData(id, addr, 2)` slices a register out of that block; it
            returns 0 for a range the block does not cover, which is why the
            comm result is checked first rather than trusting the values,
          - `_decode_sign` / `_normalize` are lerobot's own, so position comes
            out of here byte-identical to `get_observation()` — including the
            gripper's 0..100 normalisation, which differs from the other five
            joints' degrees.

        Not locked, deliberately: `bus.sync_reader` is already shared with the
        60 Hz teleop thread by lerobot's own `sync_read` with no lock anywhere
        (see motion.py). This adds no new class of race — and a lost race here
        degrades to `isAvailable() == False`, i.e. a 0 reading for one tick,
        not to a garbage position.
        """
        assert self.robot is not None
        bus = self.robot.bus
        ids = [m.id for m in bus.motors.values()]
        bus._setup_sync_reader(ids, self._BLOCK_ADDR, self._BLOCK_LEN)
        comm = bus.sync_reader.txRxPacket()
        if not bus._is_comm_success(comm):
            raise ConnectionError(
                f"block sync read @{self._BLOCK_ADDR}+{self._BLOCK_LEN} failed: "
                f"{bus.packet_handler.getTxRxResult(comm)}"
            )
        raw_pos = {i: bus.sync_reader.getData(i, self._POS_ADDR, 2) for i in ids}
        raw_load = {i: bus.sync_reader.getData(i, self._LOAD_ADDR, 2) for i in ids}
        pos = bus._normalize(bus._decode_sign("Present_Position", raw_pos))
        return (
            {bus._id_to_name(i): float(v) for i, v in pos.items()},
            # RAW registers here: unlike `sync_read`, `getData` does no sign
            # decoding of its own, so bit 10 is still the direction bit.
            {bus._id_to_name(i): self._load_fraction(
                decode_sign_magnitude(int(v), self._LOAD_SIGN_BIT))
             for i, v in raw_load.items()},
        )

    def _read_load_registers(self) -> dict[str, float]:
        """Effort via a plain `sync_read` — the safe path, one extra round trip.

        `sync_read` runs `_decode_sign` itself and Present_Load is in the
        sign-magnitude table, so these values arrive ALREADY SIGNED; decoding
        them again here would turn every negative load into a positive one.
        `normalize=False` because Present_Load is not in lerobot's
        NORMALIZED_DATA — asking for normalisation would try to map load counts
        through the POSITION calibration.
        """
        assert self.robot is not None
        raw = self.robot.bus.sync_read("Present_Load", normalize=False)
        return {name: self._load_fraction(int(v)) for name, v in raw.items()}

    def _probe_effort_path(self) -> str:
        """Try the fast path, then the safe one, then give up — once, at connect.

        Returns "block" (position+load in one round trip), "sync_read" (load in
        its own round trip) or "none" (no effort channel; the recorder writes
        0.0). Never raises: an arm that cannot report load must still teleop.
        """
        try:
            self._read_block()
            return "block"
        except Exception as e:
            logger.warning(
                "arm %s: combined position+load block read unavailable (%s); "
                "falling back to a second sync_read for effort", self.config.id, e)
        try:
            self._read_load_registers()
            return "sync_read"
        except Exception as e:
            logger.warning(
                "arm %s: Present_Load unreadable (%s); effort will be reported "
                "as 0.0", self.config.id, e)
        return "none"

    def _demote_effort_path(self, exc: Exception) -> None:
        """Step down one path after `_EFFORT_DEMOTE_AFTER` consecutive failures."""
        self._effort_fail_streak += 1
        if self._effort_fail_streak < self._EFFORT_DEMOTE_AFTER:
            return
        nxt = {"block": "sync_read", "sync_read": "none"}.get(self._effort_mode, "none")
        logger.warning("arm %s: effort read path %s -> %s after %d failures (%s)",
                       self.config.id, self._effort_mode, nxt,
                       self._effort_fail_streak, exc)
        self._effort_mode = nxt
        self._effort_fail_streak = 0

    def read_effort_norm(self) -> dict[str, float]:
        """Per-joint effort as a signed fraction of the joint's torque limit,
        keyed by joint name (no `.pos` suffix), same keys `read_joints_deg` uses.

        Dimensionless, NOT N·m and NOT amps — see `EFFORT_UNIT` above for why
        the real and sim arms both normalise to their own saturation limit.

        Returns `{}` rather than raising when the load register cannot be read:
        an effort channel is a nice-to-have, and it must never be the reason
        telemetry drops an arm or teleop stops. Callers substitute 0.0.

        Unlike the sim handle this is NOT gated on `torque_enabled`: a real
        STS3215 with torque off applies no PWM, so Present_Load already reads
        ~0. The sim's position actuator keeps its bias term and has to be
        gated; see sim/arm.py.
        """
        if self.robot is None:
            return {}
        try:
            if self._effort_mode == "block":
                effort = self._read_block()[1]
            elif self._effort_mode == "sync_read":
                effort = self._read_load_registers()
            else:
                return {}
        except Exception as e:
            self._demote_effort_path(e)
            return {}
        self._effort_fail_streak = 0
        return effort

    def _read_state_and_effort(self) -> tuple[dict[str, float], dict[str, float]]:
        """Positions (deg) + effort for one telemetry tick, in as few round
        trips as the arm allows: one on the block path, two otherwise."""
        if self._effort_mode == "block":
            try:
                pos, effort = self._read_block()
            except Exception as e:
                self._demote_effort_path(e)
                # Deliberately no effort retry on this tick: a second attempt
                # would pay a second serial timeout on a bus that just failed,
                # and position — the channel telemetry actually needs — has
                # still to be read. The demotion counter handles the rest.
                return self.read_joints_deg(), {}
            self._effort_fail_streak = 0
            return pos, effort
        # Fallback path: lerobot's own position read, plus effort on its own
        # round trip if this arm can serve one at all. read_joints_deg raising
        # here is the pre-existing "arm telemetry failed" behaviour and stays.
        return self.read_joints_deg(), self.read_effort_norm()

    def state_snapshot(self) -> dict:
        assert self.robot is not None
        pos, effort = self._read_state_and_effort()
        joints = {}
        for joint, (lo, hi) in self.joint_limits_deg.items():
            joints[joint] = {
                "pos": float(pos.get(joint, 0.0)),
                "min": float(lo),
                "max": float(hi),
                "torque": self.torque_enabled,
                # Dimensionless signed fraction of this joint's torque limit —
                # the contact/grasp signal. Same key and same meaning as the
                # sim handle's, so the recorder gets one column whichever rig
                # is driving. 0.0 when unreadable, so the key always exists.
                "effort": float(effort.get(joint, 0.0)),
            }
        return {
            "mode": self.guard.mode.value,
            "torque": self.torque_enabled,
            "joints": joints,
        }

    def calibration_metadata(self) -> dict[str, dict]:
        """Per-joint calibration, in the shape the recorder persists into the
        dataset. Keyed by joint name.

        WHY this has to be recorded: the HMI records joint angles in DEGREES
        (`use_degrees=True` in connect()), while every public LeRobot SO-101
        dataset is in normalised [-100, 100]. Nothing about a column of degrees
        says which affine map produced it, so without this block a Haller
        episode cannot be replayed against, or merged with, a normalised
        dataset. With it the map is exact and reversible, in both directions:

            raw_ticks = deg * (resolution - 1) / 360 + (range_min + range_max)/2
            norm_m100_100 = ((raw - range_min) / (range_max - range_min)) * 200 - 100
            norm_0_100    = ((raw - range_min) / (range_max - range_min)) * 100

        which is lerobot's `_normalize`/`_unnormalize` verbatim, with the sign
        flipped when `drive_mode` is set. `norm_mode` says which of the two
        normalised forms a joint uses — on SO-101 the gripper is 0..100 and the
        other five are -100..100, so a single dataset mixes both.

        NOTE `deg_per_tick` is 360/(resolution-1), the factor lerobot's DEGREES
        mode actually applies to the recorded positions. The clamp limits in
        `min_deg`/`max_deg` are the HMI's own, computed with 360/resolution
        (`DEG_PER_TICK`); the 0.02% difference does not matter for a safety
        clamp but would matter if you inverted the wrong one.
        """
        out: dict[str, dict] = {}
        if self.robot is None or not self.robot.calibration:
            return out
        bus = self.robot.bus
        for joint, (lo, hi) in self.joint_limits_deg.items():
            mc = self.robot.calibration.get(joint)
            if mc is None:
                continue
            motor = bus.motors.get(joint)
            resolution = None
            norm_mode = None
            if motor is not None:
                resolution = bus.model_resolution_table.get(motor.model)
                # MotorNormMode is a str-Enum; .value is the wire form lerobot
                # writes elsewhere, and it is what belongs in JSON.
                norm_mode = getattr(motor.norm_mode, "value", motor.norm_mode)
            out[joint] = {
                "source": "feetech_calibration",
                "range_min_ticks": int(mc.range_min),
                "range_max_ticks": int(mc.range_max),
                "homing_offset": int(mc.homing_offset),
                "drive_mode": int(mc.drive_mode),
                "resolution": int(resolution) if resolution else None,
                "deg_per_tick": (360.0 / (resolution - 1)) if resolution else None,
                "norm_mode": str(norm_mode) if norm_mode is not None else None,
                "min_deg": float(lo),
                "max_deg": float(hi),
            }
        return out


class ArmManager:
    """Lookup-by-id collection of arm handles (real or sim)."""

    def __init__(self, arm_configs: list[ArmConfig],
                 motion: MotionConfig | None = None,
                 sim_cubes: int = 0):
        self._configs = [c for c in arm_configs if c.enabled]
        self._motion = motion or MotionConfig()
        self._sim_cubes = sim_cubes
        self._handles: dict[str, "ArmHandle | SimArmHandle"] = {}
        self._world = None  # lazily constructed if any sim arm/camera needs it

    def _ensure_world(self) -> "MuJoCoWorld":
        if self._world is not None:
            return self._world
        from .sim.builder import build_scene
        from .sim.world import MuJoCoWorld

        sim_arm_names = [c.sim_arm_name for c in self._configs
                         if c.source == "sim" and c.sim_arm_name is not None]
        mjcf_xml, arm_joint_map = build_scene(arms=sim_arm_names,
                                              cubes=self._sim_cubes)
        self._world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
        self._world.start()
        return self._world

    def connect_all(self, *, teleop_peers: list) -> None:
        """Connect every configured arm.

        `teleop_peers` — normally [TeleopSession, HumanTeleopSession,
        SimLeaderTeleop] — gets attached to every handle's `executor` right
        here, in the one place that constructs every handle, instead of the
        caller looping over `arms.values()` after the fact (3 sessions x N
        arms, repeated at every call site that ever constructs a handle).
        `MoveExecutor.teleop_owner` returns None on an empty peer list, so a
        handle nobody wires silently allows every discrete move with no
        error and no log — see motion.py and amendment A6 in the plan.

        The three sessions take this ArmManager as a constructor argument, so
        they cannot be constructed before it and cannot be passed to
        `__init__`. Passing them here instead — the one place in `_lifespan`
        where `arms`, `teleop`, `human_teleop` and `sim_teleop` all already
        exist — is the earliest that's possible while still keeping the
        actual attachment loop inside this method rather than in the caller.

        No default: a caller must pass a list, `[]` if it genuinely has no
        peers to wire. A `None`/`()` default let `teleop_peers=[...]` be
        deleted from server.py's one call site with all 332 tests staying
        green while the guard failed open in production — obligation 1's own
        failure mode, one level up. Making this required turns that deletion
        into an immediate `TypeError` instead.
        """
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
            for peer in teleop_peers:
                handle.executor.attach_peer(peer)
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
