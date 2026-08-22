# hmi/backend/tests/test_safety.py
import pytest

from haller_hmi.safety import (
    check_move_size,
    clamp_joint_goal,
    limit_step,
    ModeGuard,
    ModeError,
    Mode,
    plan_ramp,
    step_budget_deg,
    MAX_STEP_DT_S,
)


def test_clamp_joint_goal_clamps_above_max():
    limits = {"shoulder_pan": (-120.0, 120.0), "gripper": (0.0, 100.0)}
    out = clamp_joint_goal({"shoulder_pan": 200.0, "gripper": 50.0}, limits)
    assert out == {"shoulder_pan": 120.0, "gripper": 50.0}


def test_clamp_joint_goal_clamps_below_min():
    limits = {"shoulder_pan": (-120.0, 120.0)}
    out = clamp_joint_goal({"shoulder_pan": -200.0}, limits)
    assert out == {"shoulder_pan": -120.0}


def test_clamp_joint_goal_ignores_unknown_joint():
    limits = {"shoulder_pan": (-120.0, 120.0)}
    out = clamp_joint_goal({"unknown_joint": 50.0, "shoulder_pan": 0.0}, limits)
    assert out == {"shoulder_pan": 0.0}


def test_mode_guard_blocks_writes_in_auto():
    guard = ModeGuard(initial=Mode.AUTO)
    with pytest.raises(ModeError):
        guard.assert_manual()


def test_mode_guard_allows_writes_in_manual():
    guard = ModeGuard(initial=Mode.MANUAL)
    guard.assert_manual()  # must not raise


def test_mode_guard_transitions():
    guard = ModeGuard(initial=Mode.AUTO)
    guard.set(Mode.MANUAL)
    assert guard.mode is Mode.MANUAL
    guard.set(Mode.STOP)
    with pytest.raises(ModeError):
        guard.assert_manual()


def test_limit_step_caps_large_delta_both_directions():
    current = {"shoulder_pan": 0.0, "elbow_flex": 10.0}
    goal = {"shoulder_pan": 100.0, "elbow_flex": -90.0}
    out = limit_step(current, goal, max_step_deg=1.2)
    assert out == {"shoulder_pan": 1.2, "elbow_flex": 8.8}


def test_limit_step_passes_small_delta_through_untouched():
    current = {"shoulder_pan": 5.0}
    out = limit_step(current, {"shoulder_pan": 5.5}, max_step_deg=1.2)
    assert out == {"shoulder_pan": 5.5}


def test_limit_step_passes_through_joints_with_no_reference_position():
    # clamp_joint_goal already dropped unknown joints; a joint missing from
    # `current` means we have no measurement, not that the joint is bogus.
    out = limit_step({}, {"wrist_roll": 42.0}, max_step_deg=1.2)
    assert out == {"wrist_roll": 42.0}


def test_step_budget_deg_is_zero_for_a_zero_gap():
    assert step_budget_deg(0.0, max_speed_deg_s=60.0) == pytest.approx(0.0)


def test_step_budget_deg_negative_gap_clamps_to_zero():
    # Clock skew or a mis-ordered call must read as "no time earned", never
    # as a negative (i.e. backwards) budget.
    assert step_budget_deg(-1.0, max_speed_deg_s=60.0) == pytest.approx(0.0)


def test_step_budget_deg_is_exactly_max_speed_at_60hz():
    """The regression case for the floored version this replaced: with
    ramp_hz=50, `max(dt_s, 1/ramp_hz)` clipped 60 Hz's 1/60 s period up to
    1/50 s and returned 1.2 regardless of the real gap — 60 calls/s x 1.2 deg
    = 72 deg/s, the exact over-speed the fix exists to close. No floor means
    this is now strictly proportional."""
    got = step_budget_deg(1.0 / 60.0, max_speed_deg_s=60.0)
    assert got == pytest.approx(1.0)
    assert got * 60.0 == pytest.approx(60.0)  # 60 calls/s x 1.0 deg = 60 deg/s


def test_step_budget_deg_is_exactly_max_speed_at_200hz():
    """Same regression, sharper: the floored version also returned 1.2 here
    (1/200 s is further below the 1/50 s floor), giving 240 deg/s instead
    of 60."""
    got = step_budget_deg(1.0 / 200.0, max_speed_deg_s=60.0)
    assert got == pytest.approx(0.3)
    assert got * 200.0 == pytest.approx(60.0)  # 200 calls/s x 0.3 deg = 60 deg/s


def test_step_budget_deg_is_proportional_to_elapsed_time():
    got = step_budget_deg(0.05, max_speed_deg_s=60.0)
    assert got == pytest.approx(60.0 * 0.05)


def test_step_budget_deg_ceilings_at_max_dt_s_for_a_stalled_caller():
    # A caller that stalls for a long time must not bank an unbounded step:
    # 5 s at 60 deg/s would be 300 deg; the ceiling caps it at 6.
    got = step_budget_deg(5.0, max_speed_deg_s=60.0)
    assert got == pytest.approx(60.0 * MAX_STEP_DT_S)
    assert got == pytest.approx(6.0)


def test_step_budget_deg_ceiling_is_flat_above_max_dt_s():
    at_ceiling = step_budget_deg(MAX_STEP_DT_S, max_speed_deg_s=60.0)
    way_past = step_budget_deg(50.0, max_speed_deg_s=60.0)
    assert at_ceiling == pytest.approx(way_past)


def test_step_budget_deg_custom_max_dt_s_overrides_the_default_ceiling():
    got = step_budget_deg(1.0, max_speed_deg_s=60.0, max_dt_s=0.5)
    assert got == pytest.approx(60.0 * 0.5)


def test_check_move_size_reports_only_offending_joints_with_signed_delta():
    current = {"shoulder_pan": 0.0, "elbow_flex": 0.0, "gripper": 0.0}
    goal = {"shoulder_pan": 45.0, "elbow_flex": -31.0, "gripper": 5.0}
    assert check_move_size(current, goal, threshold_deg=30.0) == {
        "shoulder_pan": 45.0,
        "elbow_flex": -31.0,
    }


def test_check_move_size_empty_when_all_within_threshold():
    assert check_move_size({"a": 0.0}, {"a": 29.9}, threshold_deg=30.0) == {}


def test_check_move_size_boundary_at_threshold():
    # delta exactly at threshold should NOT be reported
    assert check_move_size({"a": 0.0}, {"a": 30.0}, threshold_deg=30.0) == {}


def test_check_move_size_skips_unmeasured_joints():
    # check_move_size skips joints absent from current (opposite of limit_step)
    current = {"shoulder_pan": 0.0}
    goal = {"shoulder_pan": 90.0, "wrist_flex": 200.0}
    assert check_move_size(current, goal, threshold_deg=30.0) == {"shoulder_pan": 90.0}


def test_plan_ramp_bounds_every_consecutive_step():
    current = {"shoulder_pan": 0.0, "elbow_flex": 0.0}
    goal = {"shoulder_pan": 20.0, "elbow_flex": -10.0}
    wps = plan_ramp(current, goal, max_speed_deg_s=60.0, hz=50.0)
    step = 60.0 / 50.0
    prev = current
    for wp in wps:
        for j, v in wp.items():
            assert abs(v - prev[j]) <= step + 1e-9
        prev = wp
    assert wps[-1] == goal


def test_plan_ramp_returns_empty_when_already_at_goal():
    assert plan_ramp({"a": 5.0}, {"a": 5.0}, max_speed_deg_s=60.0, hz=50.0) == []


def test_plan_ramp_rejects_nonpositive_rates():
    with pytest.raises(ValueError):
        plan_ramp({"a": 0.0}, {"a": 1.0}, max_speed_deg_s=0.0, hz=50.0)
    with pytest.raises(ValueError):
        plan_ramp({"a": 0.0}, {"a": 1.0}, max_speed_deg_s=-1.0, hz=50.0)
    with pytest.raises(ValueError):
        plan_ramp({"a": 0.0}, {"a": 1.0}, max_speed_deg_s=60.0, hz=0.0)
    with pytest.raises(ValueError):
        plan_ramp({"a": 0.0}, {"a": 1.0}, max_speed_deg_s=60.0, hz=-1.0)
