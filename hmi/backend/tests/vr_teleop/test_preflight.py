# hmi/backend/tests/vr_teleop/test_preflight.py
"""Pre-flight against fakes. No servos exist on this bench, and the cases
that matter — a wrapped encoder, a corrupted read, a calibration that never
reached the stops — are ones you cannot ask real hardware to produce on
demand anyway.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from haller_hmi.vr_teleop.preflight import (
    FIRST_OBS_TOLERANCE_DEG,
    MAX_RAMP_DEG_S,
    URDF_SPAN_DEG,
    PreflightReport,
    check_calibration_plausible,
    check_first_observation,
    get_observation_median,
    ramp_to_rest,
    run_preflight,
)

LOG = logging.getLogger("test_preflight")

JOINTS = (*URDF_SPAN_DEG, "gripper")

# The jaw's recorded travel on this rig: ticks 2045..3492 in
# haller_follower.json, (3492 - 2045) * 360 / 4096.
GRIPPER_SPAN_DEG = 127.2

# Limits as ArmHandle derives them from a clean, non-wrapped sweep: symmetric
# about the recorded range's centre, in DEGREES — the gripper's included, and
# that is the trap. `_load_joint_limits` has no percent branch, so the jaw
# gets a degrees window it is never read in ([-63.6, 63.6] here). A fixture
# that hands the gate (0, 100) there invents a window no calibration file can
# produce, and hides the unit mismatch it is supposed to expose.
CLEAN_LIMITS = {j: (-URDF_SPAN_DEG[j] / 2, URDF_SPAN_DEG[j] / 2) for j in URDF_SPAN_DEG}
CLEAN_LIMITS["gripper"] = (-GRIPPER_SPAN_DEG / 2, GRIPPER_SPAN_DEG / 2)

REST = {j: 0.0 for j in URDF_SPAN_DEG}


@dataclass
class _Calib:
    """The slice of lerobot's MotorCalibration preflight reads: raw ticks."""
    range_min: int
    range_max: int


def _ticks(span_deg: float) -> _Calib:
    half = round(span_deg / 2 * 4096 / 360)
    return _Calib(range_min=2048 - half, range_max=2048 + half)


def _calibration(**overrides) -> dict[str, _Calib]:
    """A plausible sweep of every joint, with per-joint spans overridden in
    degrees; `None` removes the joint from the file entirely."""
    calib = {j: _ticks(URDF_SPAN_DEG[j]) for j in URDF_SPAN_DEG}
    calib["gripper"] = _ticks(GRIPPER_SPAN_DEG)
    for joint, span in overrides.items():
        if span is None:
            calib.pop(joint, None)
        else:
            calib[joint] = _ticks(span)
    return calib


class _FakeRobot:
    def __init__(self, calibration):
        self.calibration = calibration


class FakeHandle:
    """The ArmHandle surface preflight uses, and nothing else.

    `reads` is a script: one dict per call to read_joints_deg(), the last one
    repeating once exhausted. That is how a single corrupted read gets placed
    in the middle of a median.
    """

    def __init__(self, *, arm_id="right", calibration=None, reads=None,
                 limits=None, robot=True):
        self.config = type("Cfg", (), {"id": arm_id})()
        self.joint_limits_deg = dict(CLEAN_LIMITS if limits is None else limits)
        if robot:
            self.robot = _FakeRobot(
                _calibration() if calibration is None else calibration
            )
        self._reads = list(reads) if reads else [{j: 0.0 for j in JOINTS}]
        self.read_calls = 0
        self.sent: list[dict[str, float]] = []
        self.torque_enabled = True
        self.bulk_drops = 0

    def read_joints_deg(self) -> dict[str, float]:
        i = min(self.read_calls, len(self._reads) - 1)
        self.read_calls += 1
        return dict(self._reads[i])

    def send_goal(self, goal_deg: dict[str, float]) -> dict[str, float]:
        self.sent.append(dict(goal_deg))
        return dict(goal_deg)

    def disable_torque(self) -> None:
        self.bulk_drops += 1
        self.torque_enabled = False


class PerMotorHandle(FakeHandle):
    """ArmHandle's shape including `_release_torque_per_motor`, and the two
    torque paths behaving as the real ones do.

    `refuse` names servos in a latched alarm. They answer EVERY write with an
    error, so lerobot's bulk `disable_torque()` raises on the first of them and
    leaves every motor after it energised — the 2026-08-21 bench incident —
    while the per-motor walk asks all six and releases the ones that will.
    """

    MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper")

    def __init__(self, *, refuse=(), **kwargs):
        super().__init__(**kwargs)
        self._refuse = set(refuse)
        self.motor_torque = dict.fromkeys(self.MOTORS, True)

    def _release_torque_per_motor(self) -> list[str]:
        refused = []
        for motor in self.MOTORS:
            if motor in self._refuse:
                refused.append(motor)
                continue
            self.motor_torque[motor] = False
        return refused

    def disable_torque(self) -> list[str]:
        """The real `ArmHandle.disable_torque` since 2026-08-27: it IS the
        per-motor walk and returns the refusals.

        It used to be lerobot's bulk write, which raises on the first refusal
        and strands every motor after it — so this fake used to raise here and
        `preflight._drop_torque` reached past it for the private walk. Folding
        the walk into `disable_torque` fixed that at all five call sites
        (`/arm/{id}/mode`, `/arm/{id}/torque`, the shutdown walk,
        `calibration.py`, preflight), so a fake that still raised would be
        modelling a contract that no longer exists.

        `bulk_drops` counts calls to the lerobot-level bulk write, which this
        path must never reach; `_bus_disable_torque` below is that call.
        """
        refused = self._release_torque_per_motor()
        self.torque_enabled = False
        return refused

    def _bus_disable_torque(self) -> None:
        """lerobot's `bus.disable_torque()`: writes per motor, raises on the
        first refusal. Nothing in the preflight path may reach this."""
        self.bulk_drops += 1
        for motor in self.MOTORS:
            if motor in self._refuse:
                raise RuntimeError(
                    f"Failed to write 'Torque_Enable' on id_={motor} with '0' "
                    "after 6 tries. [RxPacketError] Overload error!")
            self.motor_torque[motor] = False
        self.torque_enabled = False


class SimShapedHandle:
    """A SimArmHandle stand-in: same call surface, no `robot` ATTRIBUTE AT
    ALL — not None, absent. Preflight must probe with getattr."""

    def __init__(self):
        self.config = type("Cfg", (), {"id": "sim_right"})()
        self.joint_limits_deg = dict(CLEAN_LIMITS)
        self.torque_dropped = False

    def read_joints_deg(self) -> dict[str, float]:
        return {j: 0.0 for j in JOINTS}

    def send_goal(self, goal_deg):
        return dict(goal_deg)

    def disable_torque(self) -> None:
        self.torque_dropped = True


@pytest.fixture()
def no_sleep(monkeypatch):
    """Record the ramp's schedule instead of living through it."""
    slept: list[float] = []
    monkeypatch.setattr("haller_hmi.vr_teleop.preflight.time.sleep", slept.append)
    return slept


# ---- calibration plausibility ------------------------------------------


def test_wrapped_span_is_a_warning_not_a_problem():
    """A ~360 deg span means the sweep crossed the 12-bit wrap, which
    shoulder_lift and elbow_flex do even from a perfect middle pose — they
    reach past +/-180 deg from centre at their hard stops. Failing it would
    reject correctly-calibrated arms."""
    handle = FakeHandle(calibration=_calibration(shoulder_lift=359.0,
                                                 elbow_flex=359.0))
    problems, warnings = check_calibration_plausible(handle, LOG)
    assert problems == []
    assert len(warnings) == 2
    assert {w.split(":")[0] for w in warnings} == {"shoulder_lift", "elbow_flex"}


def test_span_just_inside_the_slack_is_neither_warned_nor_failed():
    handle = FakeHandle(calibration=_calibration(
        wrist_flex=URDF_SPAN_DEG["wrist_flex"] + 25.0))
    assert check_calibration_plausible(handle, LOG) == ([], [])


def test_barely_swept_joint_is_a_hard_problem():
    """Under 60 deg the sweep never reached the stops, so the recorded range
    is not the joint's range and every limit derived from it is fiction."""
    handle = FakeHandle(calibration=_calibration(wrist_roll=30.0))
    problems, warnings = check_calibration_plausible(handle, LOG)
    assert warnings == []
    assert len(problems) == 1
    assert problems[0].startswith("wrist_roll:")
    assert "barely swept" in problems[0]


def test_missing_joint_is_a_hard_problem():
    handle = FakeHandle(calibration=_calibration(elbow_flex=None))
    problems, _ = check_calibration_plausible(handle, LOG)
    assert problems == ["elbow_flex: missing from calibration"]


def test_gripper_is_not_span_checked():
    """The gripper's range is a jaw opening, not a URDF joint limit. A 90 deg
    gripper sweep is normal and must not read as 'barely swept'."""
    handle = FakeHandle(calibration=_calibration(gripper=40.0))
    assert check_calibration_plausible(handle, LOG) == ([], [])


def test_non_numeric_tick_range_is_a_problem_not_a_crash():
    calib = _calibration()
    calib["shoulder_pan"] = _Calib(range_min=None, range_max=4095)
    problems, _ = check_calibration_plausible(FakeHandle(calibration=calib), LOG)
    assert problems == ["shoulder_pan: calibration carries no numeric tick range"]


# ---- median reads -------------------------------------------------------


def test_median_of_three_rejects_a_single_wrap_teleport():
    """A joint near the 12-bit wrap teleports +/-360 deg on one read. Anchor
    a relative move on that read and the ramp plans a full revolution."""
    handle = FakeHandle(reads=[
        {"shoulder_pan": 10.0},
        {"shoulder_pan": 10.0 - 360.0},
        {"shoulder_pan": 10.0},
    ])
    assert get_observation_median(handle) == {"shoulder_pan": 10.0}
    assert handle.read_calls == 3


def test_median_never_averages_the_two_middle_reads():
    """With an even count, averaging would mix a wrapped read into the
    answer and invent a position no servo reported."""
    handle = FakeHandle(reads=[{"wrist_roll": 5.0}, {"wrist_roll": -355.0}])
    assert get_observation_median(handle, n=2)["wrist_roll"] in (5.0, -355.0)


def test_median_survives_a_joint_dropping_out_of_one_read():
    handle = FakeHandle(reads=[
        {"shoulder_pan": 1.0, "elbow_flex": 7.0},
        {"shoulder_pan": 1.0},
        {"shoulder_pan": 1.0, "elbow_flex": 7.0},
    ])
    assert get_observation_median(handle) == {"shoulder_pan": 1.0, "elbow_flex": 7.0}


def test_median_drops_a_joint_no_read_returned_numerically():
    """Absent beats guessed: send_goal refuses to move a joint it cannot
    measure, and a non-numeric value must not reach the arithmetic."""
    handle = FakeHandle(reads=[{"shoulder_pan": 1.0, "wrist_flex": None}])
    assert get_observation_median(handle) == {"shoulder_pan": 1.0}


# ---- the first-observation gate ----------------------------------------


def test_reading_inside_the_tolerance_passes():
    """Slack for a joint parked against a hard stop the sweep did not quite
    reach — 10 deg past a limit is not a broken calibration."""
    hi = CLEAN_LIMITS["wrist_flex"][1]
    handle = FakeHandle(reads=[{**REST, "wrist_flex": hi + 10.0}])
    assert check_first_observation(handle, LOG) == []


def test_reading_past_the_tolerance_is_named_with_its_number():
    """'calibration looks wrong' is unactionable; a joint name, its reading
    and its limit send the operator to one servo."""
    handle = FakeHandle(reads=[{**REST, "wrist_flex": 212.0}])
    offenders = check_first_observation(handle, LOG)
    assert len(offenders) == 1
    assert offenders[0].startswith("wrist_flex reads 212 deg")
    assert "[-95, 95]" in offenders[0]


def test_gate_is_bounded_by_physical_travel_not_only_the_wrapped_file():
    """The case the gate exists for. A wrapped sweep records ~360 deg, so
    ArmHandle derives limits of +/-180 from it, and a reading 130 deg off a
    wrong middle pose sails through those. The URDF span is the mechanism
    and cannot be wrong."""
    wrapped = {**CLEAN_LIMITS, "shoulder_lift": (-180.0, 180.0)}
    handle = FakeHandle(limits=wrapped, reads=[{**REST, "shoulder_lift": 130.0}])
    offenders = check_first_observation(handle, LOG)
    assert len(offenders) == 1
    assert offenders[0].startswith("shoulder_lift reads 130 deg")


def test_unreadable_joint_is_not_reported_as_out_of_range():
    """A joint no read returned is a bus fault, not evidence of bad
    calibration, and send_goal already refuses to move it."""
    handle = FakeHandle(reads=[{j: 0.0 for j in JOINTS if j != "wrist_roll"}])
    assert check_first_observation(handle, LOG) == []


def test_an_open_jaw_on_a_centred_arm_is_not_an_offender():
    """The gripper is a PERCENT, not degrees: lerobot pins it to
    MotorNormMode.RANGE_0_100 whatever `use_degrees` says. Compared against
    the degrees window ArmHandle derives from the jaw's ticks ([-63.6, 63.6]
    on this rig, tripping at 78.6), a simply-OPEN jaw reads 100 and the gate
    named a healthy arm — worst right after a calibration, whose sweep ends at
    the open stop."""
    handle = FakeHandle(reads=[{**REST, "gripper": 100.0}])
    assert check_first_observation(handle, LOG) == []


def test_the_gate_has_no_opinion_on_the_jaw_at_any_opening():
    """Not "the percentage happens to fit the window" — the gripper is outside
    the gate. Its reading is clamped into [0, 100] against the same
    range_min/range_max any window would come from, so a check there could
    never fire in either direction, and a jaw calibration cannot collapse an
    arm the way a body one can."""
    for jaw in (0.0, 50.0, 100.0):
        handle = FakeHandle(reads=[{**REST, "gripper": jaw}])
        assert check_first_observation(handle, LOG) == []


def test_the_body_joints_are_still_gated_alongside_an_open_jaw():
    """Excluding the jaw must not excuse the five joints that matter: one
    pass still hands the operator every offending body joint."""
    handle = FakeHandle(reads=[{**REST, "gripper": 100.0, "wrist_flex": 212.0,
                                "shoulder_pan": -140.0}])
    offenders = check_first_observation(handle, LOG)
    assert {o.split()[0] for o in offenders} == {"wrist_flex", "shoulder_pan"}


def test_tolerance_is_applied_at_the_documented_width():
    lo = CLEAN_LIMITS["shoulder_pan"][0]
    inside = FakeHandle(reads=[{**REST, "shoulder_pan": lo - FIRST_OBS_TOLERANCE_DEG}])
    outside = FakeHandle(
        reads=[{**REST, "shoulder_pan": lo - FIRST_OBS_TOLERANCE_DEG - 1.0}])
    assert check_first_observation(inside, LOG) == []
    assert len(check_first_observation(outside, LOG)) == 1


# ---- run_preflight ------------------------------------------------------


def test_out_of_range_first_observation_drops_torque_and_names_the_joint():
    handle = FakeHandle(reads=[{**REST, "wrist_flex": 212.0}])
    report = run_preflight(handle, LOG)
    assert not report.ok()
    assert report.torque_dropped is True
    assert handle.torque_enabled is False
    assert "wrist_flex" in report.message()
    assert "212" in report.message()


def test_clean_arm_passes_and_keeps_its_torque():
    handle = FakeHandle()
    report = run_preflight(handle, LOG)
    assert report.ok()
    assert (report.calibration_problems, report.out_of_range) == ([], [])
    assert report.torque_dropped is False
    assert handle.torque_enabled is True


def test_a_centred_arm_with_an_open_jaw_keeps_its_torque():
    """The collapse the gate exists to prevent, caused by the gate: every body
    joint at its calibrated centre, the jaw simply open, and preflight cut
    torque on an arm holding its own weight."""
    handle = FakeHandle(reads=[{**REST, "gripper": 100.0}])
    report = run_preflight(handle, LOG)
    assert report.ok()
    assert report.out_of_range == []
    assert report.torque_dropped is False
    assert handle.torque_enabled is True


def test_warnings_alone_do_not_fail_the_preflight():
    """An encoder wrap with a correct middle pose is a normal calibration.
    Failing it trains operators to re-run a wizard that cannot fix it."""
    handle = FakeHandle(calibration=_calibration(elbow_flex=359.0))
    report = run_preflight(handle, LOG)
    assert report.ok()
    assert len(report.calibration_warnings) == 1
    assert handle.torque_enabled is True


def test_both_lists_are_reported_in_one_pass():
    """A bad file plus bad readings must not cost the operator two runs, and
    the torque still has to drop."""
    handle = FakeHandle(calibration=_calibration(wrist_roll=30.0),
                        reads=[{**REST, "wrist_flex": 212.0}])
    report = run_preflight(handle, LOG)
    assert len(report.calibration_problems) == 1
    assert len(report.out_of_range) == 1
    assert report.torque_dropped is True


def test_sim_shaped_handle_is_skipped_and_never_touched():
    """SimArmHandle has no `robot` attribute at all. A sim arm has no Feetech
    calibration to check and no encoder wrap to fear."""
    handle = SimShapedHandle()
    report = run_preflight(handle, LOG)
    assert report.skipped is True
    assert report.ok()
    assert report.torque_dropped is False
    assert handle.torque_dropped is False
    assert "skipped" in report.message()


def test_preflight_never_raises_and_a_crash_is_not_a_pass():
    """run_preflight sits inside connect_all: one arm's exception strands
    every arm already energised behind it."""
    handle = FakeHandle()

    def boom():
        raise RuntimeError("serial port went away")

    handle.read_joints_deg = boom
    report = run_preflight(handle, LOG)
    assert not report.ok()
    assert any("preflight aborted" in p for p in report.calibration_problems)


def test_a_failed_torque_drop_is_reported_not_swallowed():
    handle = FakeHandle(reads=[{**REST, "wrist_flex": 212.0}])

    def boom():
        raise RuntimeError("bus write failed")

    handle.disable_torque = boom
    report = run_preflight(handle, LOG)
    assert not report.ok()
    assert report.torque_dropped is False
    assert "TORQUE STILL ENABLED" in report.message()


def test_report_message_names_the_arm():
    assert "arm left" in PreflightReport(arm_id="left").message()


# ---- the torque drop ----------------------------------------------------


def _alarmed(**kwargs):
    """An arm that fails the gate on wrist_flex with shoulder_lift — the servo
    holding the cantilever, and so the one most likely to be in alarm —
    refusing to release."""
    return PerMotorHandle(refuse={"shoulder_lift"},
                          reads=[{**REST, "wrist_flex": 212.0}], **kwargs)


def test_one_servo_in_alarm_does_not_strand_the_rest_energised():
    """2026-08-21: lerobot's bulk disable_torque writes Torque_Enable per motor
    and raises on the FIRST refusal, so shoulder_pan is released, id 2 aborts
    the walk, and elbow_flex through gripper stay energised. Half a limp arm,
    on the arm the check just decided it does not trust."""
    handle = _alarmed()
    report = run_preflight(handle, LOG)
    assert handle.bulk_drops == 0
    assert handle.motor_torque == {
        "shoulder_pan": False, "shoulder_lift": True, "elbow_flex": False,
        "wrist_flex": False, "wrist_roll": False, "gripper": False,
    }
    assert report.torque_refused == ["shoulder_lift"]


def test_a_partial_drop_is_reported_as_neither_dropped_nor_still_enabled():
    """An operator reading "TORQUE STILL ENABLED" walks up to an arm they
    believe is holding; one reading "torque dropped" lets go of it. Both are
    wrong about an arm with one servo still stiff and five limp."""
    message = run_preflight(_alarmed(), LOG).message()
    assert "TORQUE STILL ENABLED" not in message
    assert "torque dropped" not in message
    assert "shoulder_lift" in message
    assert "part limp and part stiff" in message


def test_a_partial_drop_does_not_tell_the_caller_the_arm_is_holding():
    """ArmHandleManager._preflight_arm logs "still enabled, the arm is
    holding" on `not report.torque_dropped`. A release that asked every motor
    must not land there, whatever any one of them answered."""
    assert run_preflight(_alarmed(), LOG).torque_dropped is True


def test_a_release_that_never_finished_asking_is_not_a_drop():
    """The other direction: torque_dropped is about the walk completing, so a
    drop that blew up mid-way still reports the torque as it left it."""
    handle = _alarmed()

    def boom():
        raise RuntimeError("bus fell over mid-release")

    handle._release_torque_per_motor = boom
    report = run_preflight(handle, LOG)
    assert report.torque_dropped is False
    assert "TORQUE STILL ENABLED" in report.message()


def test_a_clean_release_names_no_refusals():
    handle = PerMotorHandle(reads=[{**REST, "wrist_flex": 212.0}])
    report = run_preflight(handle, LOG)
    assert (report.torque_dropped, report.torque_refused) == (True, [])
    assert "torque dropped" in report.message()
    assert not any(handle.motor_torque.values())


def test_the_walk_leaves_the_handle_flag_matching_the_servos():
    """`_release_torque_per_motor` writes Torque_Enable and nothing else. A
    stale torque_enabled=True renders a limp arm as holding, and
    server.post_arm_mode re-energises only `if not handle.torque_enabled` — so
    leaving STOP would leave the arm limp with no way back but a restart."""
    handle = _alarmed()
    run_preflight(handle, LOG)
    assert handle.torque_enabled is False


def test_a_handle_with_no_per_motor_walk_still_gets_its_torque_dropped():
    """SimArmHandle has one call and no serial bus for a motor to refuse it
    on; so would a handle whose disable_torque already walks per motor."""
    handle = FakeHandle(reads=[{**REST, "wrist_flex": 212.0}])
    report = run_preflight(handle, LOG)
    assert handle.bulk_drops == 1
    assert (report.torque_dropped, report.torque_refused) == (True, [])
    assert handle.torque_enabled is False


# ---- ramp ---------------------------------------------------------------


def test_ramp_stretches_duration_so_no_joint_exceeds_the_speed_ceiling(no_sleep):
    """A fixed-time schedule from a collapsed pose sweeps 90 deg+ at whatever
    speed 3 s implies."""
    start = {**REST, "elbow_flex": -90.0}
    handle = FakeHandle(reads=[start])
    ramp_to_rest(handle, dict(REST), duration_s=1.0, steps=10, logger=LOG)

    total_s = sum(no_sleep)
    assert total_s >= 90.0 / MAX_RAMP_DEG_S
    assert len(no_sleep) == len(handle.sent)
    per_step_dt = total_s / len(handle.sent)
    previous = start
    for command in handle.sent:
        worst = max(abs(command[j] - previous[j]) for j in command)
        assert worst / per_step_dt <= MAX_RAMP_DEG_S + 1e-9
        previous = command


def test_ramp_lands_exactly_on_target_and_never_overshoots(no_sleep):
    start = {**REST, "shoulder_pan": 40.0, "wrist_flex": -20.0}
    handle = FakeHandle(reads=[start])
    target = {**REST, "shoulder_pan": -10.0}
    ramp_to_rest(handle, target, duration_s=0.1, steps=4, logger=LOG)

    assert handle.sent[-1] == pytest.approx(target)
    for command in handle.sent:
        for joint, value in command.items():
            lo, hi = sorted((start[joint], target[joint]))
            assert lo - 1e-9 <= value <= hi + 1e-9


def test_ramp_honours_a_duration_longer_than_the_speed_ceiling_requires(no_sleep):
    """duration_s is a MINIMUM, not a target — a slow ramp stays slow."""
    handle = FakeHandle(reads=[{**REST, "shoulder_pan": 5.0}])
    ramp_to_rest(handle, dict(REST), duration_s=4.0, steps=1, logger=LOG)
    assert sum(no_sleep) == pytest.approx(4.0)
    # steps floors at duration * RAMP_STEP_HZ, so a coarse `steps` cannot turn
    # the schedule into a handful of step-changes the servo chases at speed.
    assert len(handle.sent) == 120


def test_ramp_anchors_on_the_median_not_a_single_read(no_sleep):
    """One wrapped read as the start pose plans a 360 deg sweep."""
    handle = FakeHandle(reads=[
        {**REST, "shoulder_pan": 20.0},
        {**REST, "shoulder_pan": 20.0 - 360.0},
        {**REST, "shoulder_pan": 20.0},
    ])
    ramp_to_rest(handle, dict(REST), duration_s=0.1, steps=2, logger=LOG)
    assert max(abs(c["shoulder_pan"]) for c in handle.sent) <= 20.0


def test_ramp_on_an_empty_target_does_nothing(no_sleep):
    handle = FakeHandle()
    ramp_to_rest(handle, {}, duration_s=1.0, steps=10, logger=LOG)
    assert handle.sent == []
    assert no_sleep == []
