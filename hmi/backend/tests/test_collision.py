"""CollisionGuard geometry + step-filter policy, no MuJoCo required.

The poses here were chosen against the analytic FK (see fk_points) and are
pinned by tests/sim/test_collision_sim.py against the actual MJCF, so a
vendored-model change that invalidates them fails loudly over there.
"""
from __future__ import annotations

import pytest

from haller_hmi.collision import CollisionGuard, POSE_JOINTS, fk_points
from haller_hmi.config import CollisionConfig


def _pose(**kw) -> dict[str, float]:
    p = {j: 0.0 for j in POSE_JOINTS}
    p.update(kw)
    return p


def _guard(**cfg_kw) -> CollisionGuard:
    return CollisionGuard(CollisionConfig(**cfg_kw))


# Hands raised well clear of the bench, panned outward: safely apart.
APART = {
    "left": _pose(shoulder_pan=-30.0, shoulder_lift=-60.0, elbow_flex=30.0),
    "right": _pose(shoulder_pan=30.0, shoulder_lift=-60.0, elbow_flex=30.0),
}
# Same lift, both panned inward: the hands overrun each other's position
# across the centreline — a real collision, not a near-miss.
CROSSED = {
    "left": _pose(shoulder_pan=45.0, shoulder_lift=-60.0, elbow_flex=30.0),
    "right": _pose(shoulder_pan=-45.0, shoulder_lift=-60.0, elbow_flex=30.0),
}
# Fingertip driven ~0.23 m below the mount plane: through the bench.
DIG = {
    "left": _pose(shoulder_lift=10.0, elbow_flex=80.0, wrist_flex=-5.0),
    "right": _pose(shoulder_pan=30.0, shoulder_lift=-60.0, elbow_flex=30.0),
}


def test_apart_pose_reads_clear():
    assert _guard().clearance(APART).slack > 0.0


def test_crossed_hands_read_deep_negative_and_name_the_pair():
    cl = _guard().clearance(CROSSED)
    assert cl.slack < -0.05
    assert "hand" in cl.worst


def test_filter_passes_a_clear_step_untouched():
    res = _guard().filter_step(prev=APART, want=APART)
    assert res.limited is False
    assert res.alpha == 1.0
    assert res.poses is APART


def test_filter_stops_an_approach_at_the_margin():
    g = _guard()
    res = g.filter_step(prev=APART, want=CROSSED)
    assert res.limited is True
    assert 0.0 <= res.alpha < 1.0
    # Wherever it stopped must actually clear the margin (within the
    # bisection's resolution), and be strictly short of the collision.
    assert res.clearance.slack >= -1e-3
    for arm in ("left", "right"):
        want_pan = CROSSED[arm]["shoulder_pan"]
        prev_pan = APART[arm]["shoulder_pan"]
        got = res.poses[arm]["shoulder_pan"]
        assert min(prev_pan, want_pan) <= got <= max(prev_pan, want_pan)
        assert got != want_pan


def test_filter_never_blocks_escape():
    """From inside the margin, a step that improves clearance must pass whole.

    This is the property that makes a bad starting pose (or a mount
    reconfiguration under a parked arm) recoverable instead of a lockup.
    """
    res = _guard().filter_step(prev=CROSSED, want=APART)
    assert res.limited is False
    assert res.alpha == 1.0


def test_filter_tolerates_holding_still_inside_the_margin():
    res = _guard().filter_step(prev=CROSSED, want=CROSSED)
    assert res.limited is False


def test_gripper_is_never_scaled_by_the_guard():
    want = {arm: {**pose, "gripper": 33.3} for arm, pose in CROSSED.items()}
    prev = {arm: {**pose, "gripper": 0.0} for arm, pose in APART.items()}
    res = _guard().filter_step(prev=prev, want=want)
    assert res.limited is True
    for arm in ("left", "right"):
        assert res.poses[arm]["gripper"] == 33.3


def test_table_floor_stops_a_dig():
    g = _guard()
    start = {"left": _pose(shoulder_lift=-30.0, elbow_flex=30.0),
             "right": DIG["right"]}
    assert g.clearance(start).slack > 0.0
    res = g.filter_step(prev=start, want=DIG)
    assert res.limited is True
    tip_z = float(fk_points((-0.20, 0.0, 0.0), 0.0, res.poses["left"])["tip"][2])
    assert tip_z >= -1e-3


def test_table_floor_can_be_disabled():
    g = _guard(table_z_m=None)
    assert g.clearance(DIG).slack > 0.0


def test_an_arm_without_a_mount_raises_instead_of_passing():
    with pytest.raises(KeyError):
        _guard().clearance({"left": _pose(), "centre": _pose()})


def test_fk_treats_missing_joints_as_zero():
    full = fk_points((0.0, 0.0, 0.0), 0.0, _pose())
    sparse = fk_points((0.0, 0.0, 0.0), 0.0, {})
    for key in full:
        assert full[key] == pytest.approx(sparse[key])


def test_parked_rest_pose_is_escape_only_not_a_lockup():
    """The SO-101 'rest' keyframe tucks the gripper against its own base —
    inside any honest capsule model. The guard must report negative slack
    there yet still let the arm move OUT."""
    g = _guard()
    rest = _pose(shoulder_lift=-190.0, elbow_flex=178.0, wrist_flex=68.0)
    parked = {"left": rest, "right": dict(rest)}
    assert g.clearance(parked).slack < 0.0
    res = g.filter_step(prev=parked, want=APART)
    assert res.limited is False
    assert res.alpha == 1.0
