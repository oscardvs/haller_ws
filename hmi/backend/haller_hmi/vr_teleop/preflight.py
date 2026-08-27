# hmi/backend/haller_hmi/vr_teleop/preflight.py
"""The checks that run between `connect()` and the first goal of a session.

Ported from the kit's `lerobot/so101_utils.py`, retargeted from lerobot's
`SOFollower` onto `ArmHandle`, so the sim handle goes through the same call.

Two failures live here, and only one of them is visible in the calibration
file. A calibration sweep that crossed the 12-bit encoder wrap records a
~360 deg span on a joint that physically travels ~200; that alone is
harmless, because the zeros still come from the middle pose held at the
first ENTER. The same wrap with a WRONG middle pose records the identical
span and yields garbage zeros. Nothing in the file separates them — so the
file check only warns, and the decisive test is the first reading after
connect landing where the mechanism can actually be.

Nothing here decides anything: `run_preflight` returns a report and the
caller acts on it. The one action it does take is dropping torque, because
that cannot wait for a round trip through the caller.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

# How far outside its limits a joint's first reading may land before the
# calibration is treated as wrong. Slack for a joint parked against a hard
# stop the sweep did not quite reach; well under the tens of degrees a bad
# middle pose puts in.
FIRST_OBS_TOLERANCE_DEG = 15.0

# The rest ramp's speed ceiling. ArmHandle.send_goal caps each step against
# real elapsed time at MotionConfig.max_speed_deg_s (60), so this sits at 40%
# of what the streaming path already allows: the ramp is never the thing
# re-capped, and a collapsed arm still unfolds slowly enough to be caught.
MAX_RAMP_DEG_S = 25.0

# Waypoint rate of the ramp. At MAX_RAMP_DEG_S this is 0.83 deg per step
# against send_goal's ~2 deg budget at 30 Hz, so every increment passes
# through uncapped and the schedule that runs is the schedule that was
# planned.
RAMP_STEP_HZ = 30.0

# Physical travel per joint, from the URDF — and the set of joints every
# check here runs on, because these five are the ones lerobot reports in
# DEGREES. The gripper is absent deliberately and is checked nowhere: its
# range is a jaw opening, not a URDF joint limit, and its reading is not in
# these units at all. See `_effective_limits`.
URDF_SPAN_DEG: dict[str, float] = {
    "shoulder_pan": 220.0,
    "shoulder_lift": 200.0,
    "elbow_flex": 193.6,
    "wrist_flex": 190.0,
    "wrist_roll": 320.0,
}

# A recorded span this far past the physical travel means the sweep crossed
# the encoder wrap. shoulder_lift and elbow_flex reach past +/-180 deg from
# center at their hard stops, so a perfect sweep of a correctly-held arm does
# it — hence a warning, never an error.
SPAN_SLACK_DEG = 30.0

# Below this the sweep never reached the stops, so the recorded range is not
# the joint's range and every limit derived from it is fiction.
MIN_SWEPT_DEG = 60.0

# 12-bit Feetech encoder. Duplicated from arm.py rather than imported: that
# module pulls in lerobot at import time, and preflight has to stay
# importable on the sim path.
DEG_PER_TICK = 360.0 / 4096


@dataclass(frozen=True)
class PreflightReport:
    arm_id: str
    calibration_problems: list[str] = field(default_factory=list)
    calibration_warnings: list[str] = field(default_factory=list)
    out_of_range: list[str] = field(default_factory=list)
    # True once the release has ASKED EVERY MOTOR — not once every motor
    # obeyed. False means the walk never finished, so torque is wherever it
    # was. `torque_refused` names the servos that would not release; a
    # non-empty list is an arm that is part limp and part stiff, which is
    # neither "dropped" nor "still enabled" to an operator standing over it.
    torque_dropped: bool = False
    torque_refused: list[str] = field(default_factory=list)
    skipped: bool = False

    def ok(self) -> bool:
        """Warnings do not fail a preflight — an encoder wrap with a correct
        middle pose is a normal calibration, and failing it would train
        operators to re-run a wizard that cannot fix anything."""
        return not self.calibration_problems and not self.out_of_range

    def message(self) -> str:
        if self.skipped:
            return f"arm {self.arm_id}: preflight skipped (no calibration surface)"
        parts = ["preflight ok" if self.ok() else "PREFLIGHT FAILED"]
        if self.calibration_problems:
            parts.append("calibration: " + "; ".join(self.calibration_problems))
        if self.out_of_range:
            parts.append("first reading outside limits: "
                         + "; ".join(self.out_of_range))
            parts.append(self.torque_phrase())
        if self.calibration_warnings:
            parts.append("warnings: " + "; ".join(self.calibration_warnings))
        return f"arm {self.arm_id}: " + " | ".join(parts)

    def torque_phrase(self) -> str:
        """What the operator is walking up to. Never a bare "torque dropped"
        while a servo refused: a half-released arm is neither limp nor
        holding, and either word alone gets someone to let go of it."""
        if not self.torque_dropped:
            return "TORQUE STILL ENABLED"
        if not self.torque_refused:
            return "torque dropped"
        return ("TORQUE ONLY PARTLY DROPPED — "
                + ", ".join(self.torque_refused)
                + " refused; the arm is part limp and part stiff, support it. "
                  "Those servos are in an alarm state and stay stiff until the "
                  "arm is power-cycled")


def check_calibration_plausible(
    handle, logger: logging.Logger,
) -> tuple[list[str], list[str]]:
    """Cross-check the calibration file against the URDF spans, before
    anything moves. Returns (hard problems, warnings).

    A joint missing from the file, or one whose recorded range is too small
    to be the real range, is a hard problem: every limit derived from it is
    wrong and the arm must not be driven against it. An oversized range is a
    wrap and only warns — see the module docstring for why the file cannot
    tell a good wrap from a bad one.
    """
    problems: list[str] = []
    warnings: list[str] = []
    calib = getattr(getattr(handle, "robot", None), "calibration", None) or {}
    for name, span_urdf in URDF_SPAN_DEG.items():
        entry = calib.get(name)
        if entry is None:
            problems.append(f"{name}: missing from calibration")
            continue
        lo = getattr(entry, "range_min", None)
        hi = getattr(entry, "range_max", None)
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            problems.append(f"{name}: calibration carries no numeric tick range")
            continue
        span_deg = (hi - lo) * DEG_PER_TICK
        if span_deg > span_urdf + SPAN_SLACK_DEG:
            warnings.append(
                f"{name}: recorded range {span_deg:.0f} deg exceeds the physical "
                f"~{span_urdf:.0f} deg (encoder wrap during the sweep); zeros are "
                f"still fine if the middle pose was held"
            )
        elif span_deg < MIN_SWEPT_DEG:
            problems.append(
                f"{name}: recorded range only {span_deg:.0f} deg — joint barely swept"
            )
    for w in warnings:
        logger.warning("calibration check: %s", w)
    for p in problems:
        logger.error("calibration check: %s", p)
    return problems, warnings


def get_observation_median(handle, n: int = 3) -> dict[str, float]:
    """Median of n reads per joint.

    A Feetech bus returns the occasional corrupted status packet, and a joint
    sitting near the 12-bit wrap can teleport +/-360 deg between two reads.
    Never anchor a relative move on a single read.

    `sorted(vals)[len(vals) // 2]`, not `statistics.median`: on an even count
    the latter averages the two middle reads, and averaging a wrapped read
    with a good one invents a position no servo ever reported. A joint no
    read returned numerically is absent from the result — callers must not
    move what they cannot measure.
    """
    reads = [handle.read_joints_deg() or {} for _ in range(max(1, int(n)))]
    out: dict[str, float] = {}
    for joint in {k for r in reads for k in r}:
        vals = sorted(float(r[joint]) for r in reads
                      if isinstance(r.get(joint), (int, float))
                      and not isinstance(r.get(joint), bool))
        if vals:
            out[joint] = vals[len(vals) // 2]
    return out


def _effective_limits(handle) -> dict[str, tuple[float, float]]:
    """Per-joint bounds for the first-observation gate: the calibration's own
    limits, intersected with the joint's physical travel.

    The intersection is what gives the gate teeth on the case it exists for.
    A wrapped sweep records ~360 deg, so ArmHandle derives limits of +/-180
    from it and a reading 90 deg off a wrong middle pose sails through. The
    URDF span is the mechanism and cannot be wrong.

    The five BODY joints and nothing else — URDF_SPAN_DEG is a unit contract
    here, not a lookup. lerobot pins the gripper to MotorNormMode.RANGE_0_100
    whatever `use_degrees` says (so_follower.py:59), so
    `read_joints_deg()["gripper"]` is a PERCENT of jaw opening, while
    ArmHandle._load_joint_limits converts its tick range into a symmetric
    DEGREES window: this rig's jaw ticks 2045..3492 give [-63.6, 63.6],
    tripping at 78.6. An open jaw reads 100, so the gate cut torque on a
    healthy arm — worst right after a calibration, whose sweep ends at the
    open stop.

    Gating it in its own unit is not the fix either: lerobot's `_normalize`
    clamps RANGE_0_100 into [0, 100] against the same range_min/range_max any
    window would be derived from, so the reading is inside its window by
    construction and the check could never fire. And a wrong jaw calibration
    cannot collapse an arm — the pose this gate exists to catch is the body's.
    """
    out: dict[str, tuple[float, float]] = {}
    limits = getattr(handle, "joint_limits_deg", None) or {}
    for joint, span in URDF_SPAN_DEG.items():
        bounds = limits.get(joint)
        if bounds is None:
            continue
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError, IndexError):
            continue
        out[joint] = (max(lo, -span / 2.0), min(hi, span / 2.0))
    return out


def check_first_observation(
    handle, logger: logging.Logger, tolerance_deg: float = FIRST_OBS_TOLERANCE_DEG,
) -> list[str]:
    """The decisive gate: every BODY joint's first reading must land inside
    its limits widened by `tolerance_deg`. Returns the offenders, named.

    Naming them is the whole point. "calibration looks wrong" sends an
    operator nowhere; "wrist_flex reads 212 deg, limit is [-95, 95]" sends
    them to one servo.

    The gripper is not judged here at all — its reading is a percentage, not
    degrees, and nothing in it can be out of range. See `_effective_limits`.

    A joint with no numeric reading is NOT reported here: an unreadable joint
    is a bus fault, not evidence of bad calibration, and send_goal already
    refuses to move a joint it cannot measure.
    """
    obs = get_observation_median(handle)
    offenders: list[str] = []
    for joint, (lo, hi) in _effective_limits(handle).items():
        value = obs.get(joint)
        if value is None:
            logger.warning("first-observation check: %s returned no reading", joint)
            continue
        if value < lo - tolerance_deg or value > hi + tolerance_deg:
            offenders.append(
                f"{joint} reads {value:.0f} deg, limit is [{lo:.0f}, {hi:.0f}]"
            )
    for o in offenders:
        logger.error("first-observation check: %s", o)
    return offenders


def ramp_to_rest(
    handle,
    target_deg: dict[str, float],
    duration_s: float,
    steps: int,
    logger: logging.Logger,
) -> None:
    """Drive the arm from where it is to `target_deg` in linearly
    interpolated increments.

    `duration_s` is a MINIMUM: it stretches so no joint is scheduled faster
    than MAX_RAMP_DEG_S. A fixed-time schedule from a collapsed pose would
    otherwise sweep 90 deg+ at whatever speed 3 s implies.

    Requires the handle in MANUAL — send_goal gates on the mode guard.
    """
    if not target_deg:
        return
    obs = get_observation_median(handle)
    # A joint that returned nothing starts at its target, so it contributes
    # no delta and is commanded straight to rest rather than swept from a
    # guessed pose.
    start = {j: float(obs.get(j, target_deg[j])) for j in target_deg}

    worst_delta = max(abs(target_deg[j] - start[j]) for j in target_deg)
    duration_s = max(float(duration_s), worst_delta / MAX_RAMP_DEG_S)
    steps = max(int(steps), int(duration_s * RAMP_STEP_HZ), 1)
    logger.info("ramp start : %s", {j: round(v, 1) for j, v in start.items()})
    logger.info("ramp target: %s", {j: round(v, 1) for j, v in target_deg.items()})
    logger.info("ramping to rest over %.1fs in %d steps (worst joint delta %.1f deg)",
                duration_s, steps, worst_delta)
    dt = duration_s / steps
    for i in range(1, steps + 1):
        alpha = i / steps
        handle.send_goal(
            {j: start[j] + alpha * (target_deg[j] - start[j]) for j in target_deg}
        )
        time.sleep(dt)
    logger.info("rest pose reached")


def _drop_torque(handle, logger: logging.Logger) -> list[str]:
    """Release the arm one servo at a time. Returns the joints that refused.

    NOT a bare `handle.disable_torque()`: on `ArmHandle` that reaches
    lerobot's bulk `bus.disable_torque()`, which writes Torque_Enable per
    motor and RAISES on the first refusal. A servo in a latched alarm refuses
    every write, including "turn your torque off", and the servo most likely
    to be in alarm is shoulder_lift (id 2) holding the cantilever — so the one
    motor that must be released is the one that aborts the walk, and
    elbow_flex through gripper stay energised behind an already-released
    shoulder_pan. Bench, 2026-08-21; `ArmHandle._release_torque_per_motor`
    was written for exactly this and this routes through it.
    """
    # `ArmHandle.disable_torque()` walks per motor and returns the refusals
    # itself, so this neither reaches for a private name nor writes
    # `torque_enabled` on someone else's object — both of which it used to do,
    # and both of which belonged in arm.py. `or []` covers a handle whose
    # release reports nothing: SimArmHandle has one call and no serial bus for
    # a motor to refuse it on.
    refused = [str(j) for j in (handle.disable_torque() or [])]
    if refused:
        logger.error(
            "torque drop: %d motor(s) refused Torque_Enable=0: %s. The arm is "
            "part limp and part stiff — support it. Those servos stay stiff "
            "until the arm is power-cycled.",
            len(refused), ", ".join(refused))
    return refused


def run_preflight(handle, logger: logging.Logger | None = None) -> PreflightReport:
    """Calibration plausibility, then the first-observation gate. Torque is
    dropped if the gate fails; everything else is reported, not acted on.

    NEVER raises. This runs inside connect_all, where one arm's exception
    strands every arm already energised behind it.
    """
    log = logger if logger is not None else _log
    arm_id = str(getattr(getattr(handle, "config", None), "id", "unknown"))
    # getattr, not attribute access: SimArmHandle does not define `robot` at
    # all. A sim arm has no Feetech calibration to cross-check and no encoder
    # wrap to fear, so preflight is a no-op on it.
    if getattr(handle, "robot", None) is None:
        report = PreflightReport(arm_id=arm_id, skipped=True)
        log.info("%s", report.message())
        return report

    problems: list[str] = []
    warnings: list[str] = []
    out_of_range: list[str] = []
    torque_dropped = False
    torque_refused: list[str] = []
    try:
        problems, warnings = check_calibration_plausible(handle, log)
        # Run the gate even when the file check already failed: one pass
        # should hand the operator every offending joint, and a bad file plus
        # bad readings still needs the torque dropped.
        out_of_range = check_first_observation(handle, log)
        if out_of_range:
            torque_refused = _drop_torque(handle, log)
            # After the walk, not before: an exception here leaves this False
            # and the report says TORQUE STILL ENABLED, which is the truth
            # about a release that never finished asking.
            torque_dropped = True
    except Exception as exc:
        # A preflight that could not finish is not a preflight that passed.
        log.exception("arm %s: preflight aborted", arm_id)
        problems = [*problems, f"preflight aborted: {exc!r}"]

    report = PreflightReport(
        arm_id=arm_id,
        calibration_problems=problems,
        calibration_warnings=warnings,
        out_of_range=out_of_range,
        torque_dropped=torque_dropped,
        torque_refused=torque_refused,
    )
    log.log(logging.INFO if report.ok() else logging.ERROR, "%s", report.message())
    return report
