"""The decoupled SO-101 solver.

Four of these are load-bearing rather than routine:

  * `test_reanchor_takes_no_step` — the session's acquisition gate hands the
    arm over when the commanded pose matches the measured one, and a VR
    handover is near-instant only because squeezing the grip anchors the
    target ON the arm. A solver that drifts when asked for where it already
    is breaks that, and it breaks it silently.
  * `test_pure_rotation_leaves_the_arm_joints_alone` — the decoupling claim.
  * `test_unreachable_yaw_does_not_drag_the_tool` — the 5-DoF adaptation.
    The reference places its position target with the DEMANDED orientation;
    on a 5-DoF arm an unreachable yaw then sits in that term and pulls the
    tool off position for as long as the operator holds their hand over.
  * `test_no_elbow_flip_at_the_straight_singularity` — what the damping is
    for.
"""
from __future__ import annotations

import numpy as np
import pytest

from haller_hmi.so101_kinematics import fk_frames
from haller_hmi.vr_teleop.core import quat
from haller_hmi.vr_teleop.ik.decoupled_ik import SO101DecoupledIK
from haller_hmi.vr_teleop.ik.model import DEFAULT_LIMITS_DEG, POSE_JOINTS

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")


def _pose(**kw) -> dict[str, float]:
    return {j: float(kw.get(j, 0.0)) for j in POSE_JOINTS}


def _ik(**kw) -> SO101DecoupledIK:
    return SO101DecoupledIK(DEFAULT_LIMITS_DEG, **kw)


def _settle(ik, target_pos, target_quat, seed, n=300):
    cur = dict(seed)
    for _ in range(n):
        cur = ik.solve(target_pos, target_quat, cur)
    return cur


@pytest.mark.parametrize("q", [
    _pose(),                                                    # home, near singular
    _pose(shoulder_lift=-60, elbow_flex=90),                    # the rest posture
    _pose(shoulder_pan=25, shoulder_lift=-80, elbow_flex=70,
          wrist_flex=30, wrist_roll=-40),
])
def test_reanchor_takes_no_step(q):
    ik = _ik()
    out = ik.solve(*ik.fk(q), q)
    assert max(abs(out[j] - q[j]) for j in POSE_JOINTS) < 1e-3


def test_converges_to_a_reachable_pose():
    ik = _ik()
    for target in (_pose(shoulder_pan=20, shoulder_lift=-70, elbow_flex=80,
                         wrist_flex=20, wrist_roll=30),
                   _pose(shoulder_pan=-45, shoulder_lift=-110, elbow_flex=120,
                         wrist_flex=-40, wrist_roll=90)):
        tp, tq = ik.fk(target)
        reached = _settle(ik, tp, tq, _pose(shoulder_lift=-60, elbow_flex=90))
        ap, aq = ik.fk(reached)
        assert np.linalg.norm(ap - tp) < 1e-3
        assert np.degrees(quat.angle_between(aq, tq)) < 0.5


@pytest.mark.parametrize("joint", ["wrist_roll", "wrist_flex"])
@pytest.mark.parametrize("deg", [10.0, 30.0, 60.0])
def test_pure_rotation_leaves_the_arm_joints_alone(joint, deg):
    """A twist about a real wrist axis, pivoted at the wrist anchor, must be
    taken entirely by the wrist."""
    ik = _ik()
    q0 = _pose(shoulder_lift=-70, elbow_flex=85, wrist_flex=10)
    p0, o0 = ik.fk(q0)
    anchor = ik.wrist_anchor(q0)
    axis = fk_frames(q0).joint_axis[joint]
    dq = quat.from_rotvec(axis * np.radians(deg))
    target_pos = anchor + quat.rotate(dq, p0 - anchor)
    reached = _settle(ik, target_pos, quat.mul(dq, o0), q0)
    assert max(abs(reached[j] - q0[j]) for j in ARM_JOINTS) < 0.05
    assert abs(reached[joint] - q0[joint] - deg) < 0.5


@pytest.mark.parametrize("deg", [5.0, 15.0, 45.0])
def test_unreachable_yaw_does_not_drag_the_tool(deg):
    """Yaw about world +z with position held: two wrist axes cannot deliver
    it, and the tool must stay put rather than being pulled off station."""
    ik = _ik()
    q0 = _pose(shoulder_lift=-70, elbow_flex=85)
    p0, o0 = ik.fk(q0)
    target_quat = quat.mul(quat.from_rotvec(np.array([0, 0, np.radians(deg)])), o0)
    reached = _settle(ik, p0, target_quat, q0, n=200)
    assert np.linalg.norm(ik.fk(reached)[0] - p0) < 2e-3
    # ...and the operator is told why, rather than the arm just sitting there.
    assert ik.last_orient_residual > 0.8


def test_no_elbow_flip_at_the_straight_singularity():
    ik = _ik()
    q = _pose()                      # all-zero: near the straight-elbow collapse
    p, o = ik.fk(q)
    cur = dict(q)
    worst_step = 0.0
    for _ in range(300):
        prev = dict(cur)
        cur = ik.solve(p + np.array([0.0, -0.30, 0.0]), o, cur)   # push past reach
        worst_step = max(worst_step, max(abs(cur[j] - prev[j]) for j in POSE_JOINTS))
    assert all(np.isfinite(v) for v in cur.values())
    assert cur["elbow_flex"] >= DEFAULT_LIMITS_DEG["elbow_flex"][0]
    assert worst_step <= 8.0 + 1e-6          # the configured step cap, nothing worse
    assert ik.last_singularity_proximity > 0.5


def test_out_of_reach_stays_finite_and_bounded():
    ik = _ik()
    cur = _pose(shoulder_lift=-60, elbow_flex=90)
    cur = _settle(ik, np.array([2.0, -2.0, 1.5]), quat.IDENTITY, cur, n=200)
    assert all(np.isfinite(v) for v in cur.values())
    assert ik.last_pos_err_norm > 1.0        # honest about how far off it is


def test_respects_a_narrower_calibrated_range():
    """A real arm's range comes from its LeRobot calibration, not the MJCF —
    the solver must never command past the limits it was handed."""
    narrow = {"shoulder_pan": (-90.0, 90.0), "shoulder_lift": (-100.0, 0.0),
              "elbow_flex": (0.0, 100.0), "wrist_flex": (-60.0, 60.0),
              "wrist_roll": (-90.0, 90.0)}
    ik = SO101DecoupledIK(narrow)
    cur = _settle(ik, np.array([0.5, -0.5, 0.6]), quat.IDENTITY,
                  _pose(shoulder_lift=-60, elbow_flex=90))
    for joint, (lo, hi) in narrow.items():
        assert lo - 1e-9 <= cur[joint] <= hi + 1e-9
    assert ik.last_limit_pressure > 0.0


def test_rest_pose_is_clamped_into_the_arms_range():
    """A calibration that cannot reach the default posture must not leave the
    bias pulling permanently into a stop."""
    narrow = {**DEFAULT_LIMITS_DEG, "elbow_flex": (0.0, 40.0)}
    ik = SO101DecoupledIK(narrow)
    assert ik.q_rest[2] == pytest.approx(40.0)


def test_step_cap_is_per_group():
    ik = _ik(max_dq_deg={"shoulder_pan": 1.0, "shoulder_lift": 1.0,
                         "elbow_flex": 1.0, "wrist_flex": 5.0, "wrist_roll": 5.0})
    q = _pose(shoulder_lift=-70, elbow_flex=85)
    far = _pose(shoulder_pan=60, shoulder_lift=-20, elbow_flex=160,
                wrist_flex=60, wrist_roll=120)
    out = ik.solve(*ik.fk(far), q)
    for joint in ARM_JOINTS:
        assert abs(out[joint] - q[joint]) <= 1.0 + 1e-9
    for joint in ("wrist_flex", "wrist_roll"):
        assert abs(out[joint] - q[joint]) <= 5.0 + 1e-9
