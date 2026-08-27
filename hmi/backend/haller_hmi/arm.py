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
    from .vr_teleop.preflight import PreflightReport

logger = logging.getLogger(__name__)


# Conservative defaults if calibration metadata doesn't expose explicit deg ranges;
# we derive these per-joint from each motor's calibrated range converted to degrees.
TICKS_PER_REV = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_REV

# `MotorNormMode` values, as lerobot spells them on the wire. Compared as
# strings so a handle built against a stub motor (the sim path, the test
# fakes) works without importing the enum, and pinned by
# `test_norm_mode_spellings_still_match_lerobots` so a rename upstream fails
# loudly here instead of silently falling through to the degrees branch.
_NORM_0_100 = "range_0_100"
_NORM_M100_100 = "range_m100_100"

# Reads that ANCHOR a relative motion take a median of this many. One corrupted
# Feetech status packet is enough to place an anchor most of a revolution away,
# and everything downstream is measured FROM the anchor — so the bad value is
# not a glitch, it is where the arm goes. Reads that only feed telemetry stay
# single: at 20-60 Hz three round trips are the whole tick budget, and one bad
# frame there costs one frame.
ANCHOR_READS = 3


class SyncReaderRace(ConnectionError):
    """The shared `bus.sync_reader` was re-pointed under a block read.

    A `ConnectionError` subclass so every caller that already treats a failed
    block read as "no effort this tick" keeps working unchanged, and a distinct
    type so `_demote_effort_path` can tell a lost race — which costs one tick's
    effort column on a reader that is contended by design — from a bus that
    genuinely cannot serve the read, which must retire the fast path.
    """


def _median_present_position(robot: SO101Follower,
                             n: int = ANCHOR_READS) -> dict[str, int]:
    """`Present_Position` in RAW TICKS, per-joint median of `n` reads.

    `sorted(v)[len(v) // 2]`, not `statistics.median`: on an even count the
    latter averages the two middle reads, and averaging a corrupted tick into
    a good one parks the goal at a position no servo ever reported.

    Same rule as `preflight.get_observation_median`, which cannot be reused
    here: that one reads degrees through the calibration via `ArmHandle`, and
    this runs on a bare `SO101Follower` inside `configure()`, before
    `_load_joint_limits`, on the raw registers the goal write needs.

    No availability gate, unlike `ArmHandle._read_block`, and the median is
    why. `sync_read` shares `bus.sync_reader`, so a re-point under it makes
    `getData` answer 0 for a raced joint — and 0 is the MINIMUM raw tick, so
    it sorts to the outside of a 3-sample median and one race is absorbed
    whole. Being fooled needs the SAME joint raced in TWO of the three reads,
    i.e. two hits on the ~10 us `getData` loop of two separate reads. Callers
    are `connect()` (no other thread is on this arm's bus yet) and
    `enable_torque()` (telemetry at 20 Hz), which puts the residual around
    1e-6 per park against ~1 in 30 s for the 60 Hz block reader. A gate that
    REFUSED to park would also be the wrong trade here: not parking is the
    lunge `_park_goal_on_present` exists to prevent.
    """
    reads = [robot.bus.sync_read("Present_Position", normalize=False)
             for _ in range(n)]
    out: dict[str, int] = {}
    for joint in {k for r in reads for k in r}:
        ticks = sorted(int(r[joint]) for r in reads if joint in r)
        out[joint] = ticks[len(ticks) // 2]
    return out


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


def _park_goal_on_present(robot: SO101Follower) -> None:
    """Point every servo's Goal_Position at where the arm actually IS.

    lerobot's `configure()` does its register writes inside
    `bus.torque_disabled()`, and that context manager re-enables torque
    unconditionally on the way out. Nothing between here and there ever tells
    the servos where the arm is standing, so torque comes back on against
    whatever Goal_Position the registers already held. On a cold power-up
    that value is 0 for every joint, and enabling torque commands all six to
    tick 0 at once, at whatever speed they can manage — from a rest pose
    that is most of a revolution on the elbow and the wrist roll.

    Measured on the bench 2026-08-21, one arm, cold power-up, before this
    function existed: torque off on all six, every Goal_Position reading 0,
    against a present pose of 642 / 1095 / 3715 / 1 / 3913 / 3312 raw. So
    connect() alone was one command away from ~327 deg on the elbow and ~344
    deg on the wrist roll.

    Whether this is also what happened on 2026-08-01 — the arm that slewed
    into the bench after a recalibration and took the 7.4 V DC-DC with it —
    is not something the logs can still settle, and the note in
    config.solo-real.yaml blames the Home that followed. Worth knowing that
    the Home is not needed for it: on a cold bus the slew is already
    committed when connect() returns, before any goal is ever sent.

    So: raw ticks, read then written back, before lerobot's configure() runs.
    Raw on purpose — Goal_Position and Present_Position share one register
    space, so present-into-goal is exact whatever the calibration says, and
    stays exact across the offsets `_write_calibration_to_motors` just wrote.
    Torque is still off here, so this moves nothing; it only makes the
    torque-enable that follows a hold instead of a lunge.

    Median of three reads, never one: the tick parked here is the tick the
    servo drives to the instant torque comes back, so a single corrupted read
    re-enters this function's own failure through the read instead of through
    the register — same slew, same distance, and nothing downstream to catch
    it because the goal IS the reference.

    lerobot's own note on connect() — "we assume that at connection time, arm
    is in a rest position, and torque can be safely disabled to run
    calibration" — is about *disabling* torque. Nothing there covers turning
    it back on, which is the half that moves the arm.
    """
    present = _median_present_position(robot)
    robot.bus.sync_write("Goal_Position", present, normalize=False)
    logger.info(
        "arm %s: parked Goal_Position on Present_Position before torque enable (raw ticks: %s)",
        robot.id,
        {j: int(v) for j, v in present.items()},
    )


def _configure_holding_position(robot: SO101Follower) -> None:
    """`robot.configure()`, with the goal registers parked first."""
    _park_goal_on_present(robot)
    SO101Follower.configure(robot)


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
    # Lost races to `bus.sync_reader` since connect. Counted, not folded into
    # `_effort_fail_streak` — see _demote_effort_path.
    _effort_race_count: int = 0

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
        # Same substitution trick, for the step that actually moves the arm:
        # configure() re-enables torque on its way out, and does it against
        # goal registers nobody has set. See _park_goal_on_present.
        self.robot.configure = partial(_configure_holding_position, self.robot)
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
        self._effort_race_count = 0
        logger.info("arm %s: effort read path = %s", self.config.id, self._effort_mode)

    def _release_torque_per_motor(self) -> list[str]:
        """Torque off, one servo at a time. Returns the ids that refused.

        lerobot's `bus.disconnect(disable_torque=True)` releases the arm with
        a single `disable_torque()` that walks the motors and raises on the
        first one to answer with an error. A servo in a latched alarm answers
        with an error to everything, including "turn your torque off" — so the
        one motor that most needs releasing is also the one that aborts the
        walk, and every motor after it in the chain stays energised.

        Bench, 2026-08-21: shoulder_lift (id 2) overloaded holding the arm
        cantilevered, and the shutdown died on
        `Failed to write 'Torque_Enable' on id_=2 with '0' after 6 tries.
        [RxPacketError] Overload error!`. Torque afterwards read pan=0,
        lift=0, elbow=1, wrist_flex=1, wrist_roll=1, gripper=1 — the backend
        had exited and four joints were still holding. `Application shutdown
        failed` was the only sign, in a log nobody reads while an arm is
        stiff on the bench.

        So: never let one servo's refusal decide the fate of the other five.
        Per-motor, each in its own try, worst case six failures instead of
        one abort.
        """
        assert self.robot is not None
        refused: list[str] = []
        for joint in list(self.robot.bus.motors):
            try:
                self.robot.bus.write("Torque_Enable", joint, 0, normalize=False)
            except Exception:
                refused.append(joint)
                logger.exception(
                    "arm %s: %s refused Torque_Enable=0; continuing to the rest",
                    self.config.id, joint)
        return refused

    def disconnect(self) -> None:
        if self.robot is not None:
            refused = self._release_torque_per_motor()
            if refused:
                logger.error(
                    "arm %s: %d motor(s) would not release torque: %s. They are "
                    "in an alarm state (overload/overheat is what does this) and "
                    "will stay stiff until the arm is power-cycled.",
                    self.config.id, len(refused), ", ".join(refused))
            # Every motor has now been asked individually, so lerobot must not
            # repeat the bulk walk that raises on the first refusal — that
            # exception is what escaped `disconnect_all` and stranded the rest
            # of the shutdown.
            self.robot.config.disable_torque_on_disconnect = False
            try:
                self.robot.disconnect()
            except Exception:
                logger.exception("arm %s: port close failed", self.config.id)
            self.robot = None
            self.torque_enabled = False
        # A later connect() re-probes: the same arm can come back on a
        # different adapter, or with different firmware, than it left on.
        self._effort_mode = "unprobed"
        self._effort_fail_streak = 0
        self._effort_race_count = 0

    def _load_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Per-joint clamp window, IN THE UNIT THAT JOINT IS READ AND WRITTEN IN.

        The unit is per motor, not per robot. lerobot sets the five body joints
        from `use_degrees` but pins the gripper to RANGE_0_100 unconditionally
        (`so_follower.py:50,59`), so one arm reports degrees on five joints and
        a 0..100 percentage on the sixth. `_normalize`/`_unnormalize` are the
        authority and this mirrors them exactly:

          DEGREES        tick range centred on its own mid-point, converted —
                         symmetric about zero by construction, and independent
                         of the motor's homing_offset.
          RANGE_0_100    (0, 100).
          RANGE_M100_100 (-100, 100).

        Reading the unit off the motor rather than assuming degrees is
        load-bearing, not tidiness. Before 2026-08-27 every joint got the
        degrees treatment, so the gripper's window came out (-63.59, +63.59) on
        this rig — and `_to_degrees` maps the converter's [0, 1] onto whatever
        window it is handed. Measured consequence: commands 0.00 through 0.50
        all landed at or below 0 and lerobot clamped them to a shut jaw, so the
        whole lower half of the trigger did nothing, and 1.00 reached 63.59 %
        rather than the open stop. The jaw ran its entire travel in the top
        half of the trigger and never fully opened. Every gripper column
        recorded before that date is compressed into 0..63.6 with a dead band.
        """
        out: dict[str, tuple[float, float]] = {}
        if self.robot is None or not self.robot.calibration:
            return out
        motors = self.robot.bus.motors
        for motor, mc in self.robot.calibration.items():
            norm = getattr(motors.get(motor), "norm_mode", None)
            # Exact match, never a substring test: "0_100" is also a substring
            # of "range_m100_100", so `in` would silently give the ±100 joints
            # a 0..100 window — the same class of unit error this method is
            # being fixed for.
            name = str(getattr(norm, "value", norm) or "")
            if name == _NORM_0_100:
                out[motor] = (0.0, 100.0)
            elif name == _NORM_M100_100:
                out[motor] = (-100.0, 100.0)
            else:
                center = (mc.range_min + mc.range_max) / 2.0
                out[motor] = ((mc.range_min - center) * DEG_PER_TICK,
                              (mc.range_max - center) * DEG_PER_TICK)
        return out

    def _anchor_read(self) -> dict[str, float]:
        """Positions in degrees, median of `ANCHOR_READS` reads.

        For the reads that become a MOTION REFERENCE rather than a datapoint.
        `limit_step` caps the commanded step against `_last_commanded`, so a
        single corrupted seed does not produce a bounded error — it produces a
        bounded step away from a garbage reference, i.e. a goal a whole
        revolution from where the arm is, sent at full speed. The cap is only
        as good as what it caps from.

        Neither caller is on the steady-state path: the seed runs once per
        connect or torque toggle, the retry only after a read has already
        dropped a joint. The healthy 60 Hz tick reads nothing here at all —
        that is what `_last_commanded` is for.

        Deferred import for `__post_init__`'s reason, one module further out:
        `vr_teleop` is a leaf of this package today and importing it at module
        scope is the edit that would stop it being one.
        """
        from .vr_teleop.preflight import get_observation_median
        return get_observation_median(self, ANCHOR_READS)

    def send_goal(self, goal_deg: dict[str, float]) -> dict[str, float]:
        self.guard.assert_manual()
        assert self.robot is not None
        clamped = clamp_joint_goal(goal_deg, self.joint_limits_deg)
        now = time.monotonic()
        if self._last_commanded is None:
            # First command since connect or a torque toggle: seed from a real
            # read. Every later call limits against the last command, so the
            # 60 Hz teleop path costs no extra serial traffic.
            self._last_commanded = self._anchor_read()
        if any(j not in self._last_commanded for j in clamped):
            # A flaky read can drop a joint at seed time, or leave it dropped
            # from an earlier call. Retry so it rejoins as soon as one read
            # succeeds, rather than staying unmeasured indefinitely.
            self._last_commanded = {**self._last_commanded, **self._anchor_read()}
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

    def disable_torque(self) -> list[str]:
        """Release every motor, and report the ones that refused.

        Walks per motor rather than calling `bus.disable_torque()`, which
        writes in a loop and RAISES ON THE FIRST REFUSAL — leaving every motor
        after it energised. That is the 2026-08-21 incident (an overloaded
        shoulder aborted the sweep mid-way and stranded four joints stiff),
        and every caller of this method carried it: `/arm/{id}/mode` into STOP,
        `/arm/{id}/torque`, the shutdown walk, `calibration.py`, and the
        preflight drop. Fixing it here fixes all of them at once.

        Returns the joints that would not release. Non-empty means the arm is
        PART LIMP AND PART STIFF — the one state an operator must not be told
        is "holding" — so a caller that reports torque state must report this
        list, not just the fact that a release was attempted.
        """
        refused: list[str] = []
        if self.robot is not None:
            refused = self._release_torque_per_motor()
            # Set unconditionally, even on a partial refusal: `post_arm_mode`
            # re-energises on leaving STOP only `if not handle.torque_enabled`,
            # so a stale True strands a limp arm displayed as holding with no
            # way back short of a restart.
            self.torque_enabled = False
            self._last_commanded = None
            self._last_command_at = None
        return refused

    def enable_torque(self) -> None:
        if self.robot is not None:
            # Park before enabling, every time — not just at connect. The
            # servo keeps whatever Goal_Position it last held, and anything
            # that moved the arm or shifted its frame while torque was off
            # (hand-repositioning, an E-STOP mid-motion, a homing-offset
            # rewrite) turns that stale goal into a full-speed lunge on
            # enable. 2026-08-24: a recalibration's offset rewrite did
            # exactly that — the arm slewed to full extension the moment
            # torque came back.
            _park_goal_on_present(self.robot)
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

    def _assert_block_held(self, bus, slices: list[tuple[int, int]]) -> None:
        """Raise unless the shared reader still holds every slice `_read_block`
        is about to take (or has just taken)."""
        for i, addr in slices:
            if not bus.sync_reader.isAvailable(i, addr, 2):
                raise SyncReaderRace(
                    f"block sync read @{self._BLOCK_ADDR}+{self._BLOCK_LEN}: "
                    f"motor id {i} has no data at register {addr}"
                )

    def _read_block(self) -> tuple[dict[str, float], dict[str, float]]:
        """One bus round trip -> (positions in deg, effort fractions).

        Uses lerobot's PRIVATE bus API on purpose — there is no public call
        that reads two registers in one packet — so every caller wraps this in
        a try/except and falls back. What the private bits are:
          - `_setup_sync_reader(ids, addr, len)` points the shared GroupSyncRead
            at a byte range instead of a named register,
          - `getData(id, addr, 2)` slices a register out of that block and
            answers 0 for anything the reader does not hold, so every slice is
            gated on `isAvailable(id, addr, 2)` — see the race note below,
          - `_decode_sign` / `_normalize` are lerobot's own, so position comes
            out of here byte-identical to `get_observation()` — including the
            gripper's 0..100 normalisation, which differs from the other five
            joints' degrees.

        Not locked, deliberately: `bus.sync_reader` is already shared with the
        60 Hz teleop thread by lerobot's own `sync_read` with no lock anywhere
        (see motion.py). This adds no new class of race — but a lost race is
        not harmless. The winner re-points the shared reader (`clearParam()`,
        then a new start_address), and `getData` then answers 0. 0 is a
        legitimate raw tick, not a sentinel: through `_decode_sign` /
        `_normalize` it decodes to -180.0 deg on a full-range calibration, not
        to zero degrees. In telemetry that is a visible teleport; in a
        recorded episode it is a row that teaches a policy the arm was
        somewhere it never was.

        So every slice is availability-checked BEFORE any slice is read, and
        checked again after the last one, and either miss raises
        `SyncReaderRace`. What that does and does not buy, precisely:
          - it is NOT a lock and does NOT make a 0 impossible. `getData` runs
            `isAvailable` itself and answers 0 on a miss with NO raise
            (group_sync_read.py), so a re-point landing between a check and
            the matching `getData` still yields 0 for that slice.
          - the two passes bracket every read, so a wrong value only escapes
            if the reader spends the WHOLE bracket pointing at a window that
            still covers both slices of every motor. lerobot's own reads are
            single named registers and the only code that points the reader at
            a window this wide is this method, so in practice that means a
            second copy of this same read — a valid second reading of the same
            arm, not garbage.
          - the common case is closed outright: the contender is lerobot's own
            `sync_read("Present_Position")`, which re-points to (56, 2) — that
            fails the check at `_LOAD_ADDR` = 60 and raises.
        Callers treat the raise as "no effort this tick" (see
        `_read_state_and_effort`), which is the honest outcome for data that
        did not arrive; `_demote_effort_path` deliberately does not count a
        race as a comm failure.
        """
        assert self.robot is not None
        bus = self.robot.bus
        ids = [m.id for m in bus.motors.values()]
        slices = [(i, addr) for i in ids
                  for addr in (self._POS_ADDR, self._LOAD_ADDR)]
        bus._setup_sync_reader(ids, self._BLOCK_ADDR, self._BLOCK_LEN)
        comm = bus.sync_reader.txRxPacket()
        if not bus._is_comm_success(comm):
            raise ConnectionError(
                f"block sync read @{self._BLOCK_ADDR}+{self._BLOCK_LEN} failed: "
                f"{bus.packet_handler.getTxRxResult(comm)}"
            )
        # Every slice checked before the first read: checking motor 6 only
        # after motor 1 has been read widens the window for nothing.
        self._assert_block_held(bus, slices)
        raw_pos = {i: bus.sync_reader.getData(i, self._POS_ADDR, 2) for i in ids}
        raw_load = {i: bus.sync_reader.getData(i, self._LOAD_ADDR, 2) for i in ids}
        # And again: the reader has to have stayed ours across all the reads,
        # not merely have been ours before the first.
        self._assert_block_held(bus, slices)
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
        """Step down one path after `_EFFORT_DEMOTE_AFTER` consecutive COMM
        failures. A lost race is not one of those.

        `_EFFORT_DEMOTE_AFTER` exists so one transient error cannot permanently
        cost an extra round trip per tick, and `bus.sync_reader` is contended
        by design at 60 Hz — so counting a race would hand three raced ticks
        exactly the permanent demotion that constant is there to prevent. A
        race means the fast path did not get its data this tick; it says
        nothing about whether the bus can serve a block read, which is the only
        question demotion answers. It does not clear the streak either: genuine
        failures interleaved with races still add up.
        """
        if isinstance(exc, SyncReaderRace):
            self._effort_race_count += 1
            # First one per connect is worth a line; at 60 Hz the rest are not.
            logger.log(
                logging.WARNING if self._effort_race_count == 1 else logging.DEBUG,
                "arm %s: effort read lost the shared sync_reader (%s); race #%d "
                "since connect, path stays %s",
                self.config.id, exc, self._effort_race_count, self._effort_mode)
            return
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
                # still to be read. The demotion counter handles a bus that
                # cannot serve the block read; a lost race is not that and is
                # counted separately (see _demote_effort_path).
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
                 sim_cubes: int = 0, sim_task: str = "cubes"):
        self._configs = [c for c in arm_configs if c.enabled]
        self._motion = motion or MotionConfig()
        self._sim_cubes = sim_cubes
        self._sim_task = sim_task
        self._handles: dict[str, "ArmHandle | SimArmHandle"] = {}
        self._world = None  # lazily constructed if any sim arm/camera needs it
        # Kept, not logged-and-dropped: the log line scrolls away and the named
        # offending joints are the only thing that tells an operator which
        # servo to re-sweep. See preflight_reports().
        self._preflight_reports: dict[str, PreflightReport] = {}

    def _ensure_world(self) -> "MuJoCoWorld":
        if self._world is not None:
            return self._world
        from .sim.builder import build_scene
        from .sim.world import MuJoCoWorld

        sim_arm_names = [c.sim_arm_name for c in self._configs
                         if c.source == "sim" and c.sim_arm_name is not None]
        mjcf_xml, arm_joint_map = build_scene(arms=sim_arm_names,
                                              cubes=self._sim_cubes,
                                              task=self._sim_task)
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
            # On the books BEFORE the preflight, so an arm that faults during
            # the check is still an arm disconnect_all releases. An energised
            # arm missing from `_handles` is stranded stiff at shutdown.
            self._handles[cfg.id] = handle
            self._preflight_arm(handle)

    def _preflight_arm(self, handle) -> None:
        """Check one just-connected arm and act on the report.

        Ordering: `handle.connect()` has already run lerobot's `configure()`,
        which is where `_park_goal_on_present` parks the goals and torque comes
        back on. So the arm is energised and holding by the time this runs —
        which is exactly why the check happens here and not later.

        A failed preflight does NOT drop torque. The arm is holding its own
        weight; cutting torque on a report about the calibration FILE drops it
        onto the bench, which is the collapse the report exists to prevent.
        `Mode.STOP` is the refusal instead — `send_goal` raises on it — and it
        costs nothing to undo when the operator has looked. The one case where
        torque must go is a first reading outside the limits, and preflight
        drops that itself before returning, because it cannot wait for a round
        trip through here.

        Both effects are already in every `state_snapshot`, so the operator
        sees an arm sitting in STOP in the HMI rather than an ERROR line in a
        log nobody reads while an arm is on the bench.

        Each arm in its own try, for `disconnect_all`'s reason one step
        earlier: `run_preflight` promises never to raise, but a promise from
        another module is not a guarantee, and a fault here must not leave the
        arms behind this one connected, energised and unchecked.
        """
        from .vr_teleop.preflight import run_preflight

        arm_id = getattr(handle.config, "id", "?")
        try:
            report = run_preflight(handle, logger)
            self._preflight_reports[arm_id] = report
            if report.ok():
                if report.calibration_warnings:
                    # run_preflight logs an ok report at INFO, and the per-joint
                    # warnings it emits do not name the arm — on a bimanual rig
                    # that is the half the operator needs.
                    logger.warning("%s", report.message())
                return
            handle.guard.set(Mode.STOP)
            logger.error(
                "arm %s: preflight failed; arm set to mode %s and will refuse "
                "goals until an operator clears it. Torque: %s",
                arm_id, Mode.STOP.value,
                # `torque_phrase()`, not the bare `torque_dropped` bool: the
                # bool is True on a PARTIAL release, so branching on it prints
                # "the arm is limp" about an arm that is limp in some joints
                # and stiff in the rest — the state most likely to hurt
                # someone who believes the log and reaches in.
                report.torque_phrase())
        except Exception:
            logger.exception("arm %s: preflight raised; arm set to mode %s",
                             arm_id, Mode.STOP.value)
            handle.guard.set(Mode.STOP)

    def preflight_reports(self) -> dict[str, PreflightReport]:
        """Last preflight per arm id — the named offending joints, for a caller
        that can put them in front of the operator."""
        return dict(self._preflight_reports)

    def disconnect_all(self) -> None:
        # Each arm in its own try: on a bimanual rig one arm's bad servo must
        # not leave the OTHER arm energised, which is the same failure as
        # ArmHandle._release_torque_per_motor one level up.
        for handle in self._handles.values():
            try:
                handle.disconnect()
            except Exception:
                logger.exception("arm %s: disconnect failed; continuing",
                                 getattr(handle.config, "id", "?"))
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
