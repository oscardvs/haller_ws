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
  * `test_the_pre_solve_gate_holds_the_position_joints_too` — the half of
    the antipode gate that a wrist_roll twist provably cannot see. Read its
    docstring before touching `_twisted`.
"""
from __future__ import annotations

import numpy as np
import pytest

from haller_hmi.so101_kinematics import fk_frames
from haller_hmi.vr_teleop.core import quat
from haller_hmi.vr_teleop.ik.decoupled_ik import (
    PARK_LIMIT_PRESSURE_DEG,
    PARK_MAX_SOLVES,
    SO101DecoupledIK,
)
from haller_hmi.vr_teleop.ik.model import DEFAULT_LIMITS_DEG, POSE_JOINTS
from haller_hmi.vr_teleop.teleop import _LIMIT_PRESSURE_GATE, _gate

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
WRIST_JOINTS = ("wrist_flex", "wrist_roll")


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


# ---- the near-antipodal park gate ---------------------------------------

_GATE_SEED = _pose(shoulder_lift=-70, elbow_flex=85, wrist_flex=15)


def _twisted(ik, q0, deg, joint="wrist_roll"):
    """(target_pos, target_quat): the tool exactly where it already is,
    rotated `deg` about one of its own wrist axes.

    A direction the wrist CAN take, deliberately — so the only thing the
    gate can be reacting to is the error angle, not unreachability.

    The axis is NOT interchangeable, and `wrist_roll` is the blind one. At
    `_GATE_SEED` the tool→anchor offset is [0, 0.0601, 0] and the world
    wrist_roll axis is [0, 0.866, 0.5], so cross(axis, R @ offset) is
    EXACTLY 0: rolling cannot move the position anchor, `R_reachable` drops
    out of the solve, and the pre-solve half of the gate is a provable no-op.
    About `wrist_flex` the cross is 0.0601 m and it is not.
    """
    p0, o0 = ik.fk(q0)
    axis = fk_frames(q0).joint_axis[joint]
    return p0, quat.mul(quat.from_rotvec(axis * np.radians(deg)), o0)


def test_wrist_parks_past_the_antipode_gate():
    """Past `rot_err_hold` the shortest-way direction of the quaternion error
    flips under a hair of jitter, so the wrist must take NO step — a damped
    one still comes around through 180°. 150° is outside the 2.2 rad (126°)
    default; `test_wrist_tracks_inside_the_antipode_gate` is the other side.
    """
    ik = _ik()
    q0 = dict(_GATE_SEED)
    out = ik.solve(*_twisted(ik, q0, 150.0), q0)
    assert ik.last_wrist_parked
    for joint in WRIST_JOINTS:
        assert out[joint] == q0[joint]        # no step at all, not a small one
    assert ik.last_orient_residual == pytest.approx(1.0)


def test_wrist_tracks_inside_the_antipode_gate():
    ik = _ik()
    q0 = dict(_GATE_SEED)
    out = ik.solve(*_twisted(ik, q0, 100.0), q0)
    assert not ik.last_wrist_parked
    assert max(abs(out[j] - q0[j]) for j in WRIST_JOINTS) > 1e-3


def test_parked_wrist_reports_saturating_pressure():
    """The operator-feedback half of the gate, asserted against the gate it
    has to clear rather than against a number.

    A parked wrist takes no step, so the joint-limit clamp measures nothing
    — without the sentinel the HUD and the haptic mix would read zero
    pressure at the exact moment the wrist stopped obeying the operator. The
    UNITS are the whole point: the reference stack's 0.35 is RADIANS and
    this field is degrees, and 0.35 taken as degrees is below
    `teleop._LIMIT_PRESSURE_GATE`'s dead zone, so the gate added to raise
    the alarm would have silenced it instead.
    """
    ik = _ik()
    q0 = dict(_GATE_SEED)
    ik.solve(*_twisted(ik, q0, 150.0), q0)
    assert ik.last_wrist_parked
    assert ik.last_limit_pressure >= PARK_LIMIT_PRESSURE_DEG
    assert _gate(ik.last_limit_pressure, *_LIMIT_PRESSURE_GATE) == 1.0
    assert _gate(0.35, *_LIMIT_PRESSURE_GATE) == 0.0     # the units trap


def test_the_pre_solve_gate_holds_the_position_joints_too():
    """The gate is tested in BOTH `_wrist_step` calls, and the pre-solve half
    is what keeps joints 1-3 still while the wrist is parked.

    Twisted about `wrist_flex`, because a `wrist_roll` twist cannot see this
    at all — see `_twisted`. Ungating the pre-solve costs, in a SINGLE
    solve, +3.0° of shoulder_lift and -3.0° of elbow_flex, both saturating
    `DEFAULT_MAX_DQ_DEG`, and 65° of drift over 60 solves: the anchor is
    placed against an orientation the real solve then refuses to take, so
    the position joints chase a tool pose that never arrives.
    """
    ik = _ik()
    q0 = dict(_GATE_SEED)
    tp, tq = _twisted(ik, q0, 150.0, joint="wrist_flex")
    out = ik.solve(tp, tq, q0)
    assert ik.last_wrist_parked
    # Tolerances sized against the defect, not against zero. The FK's own
    # floor at this seed is 9.6e-8 m — the same value for either twist axis,
    # with the pre-solve step exactly [0, 0] — and the ungated pre-solve is
    # worth 0.116 m and 3.0°, so 1e-3 separates them by three decades. It is
    # also `test_reanchor_takes_no_step`'s tolerance for "took no step".
    assert ik.last_pos_err_norm == pytest.approx(0.0, abs=1e-6)
    for joint in ARM_JOINTS:
        assert out[joint] == pytest.approx(q0[joint], abs=1e-3)
    cur = dict(out)
    for _ in range(PARK_MAX_SOLVES - 1):        # the rest of the park budget
        cur = ik.solve(tp, tq, cur)
        assert ik.last_wrist_parked
        assert max(abs(cur[j] - q0[j]) for j in ARM_JOINTS) < 1e-3


def test_the_park_expires_so_the_wrist_cannot_latch():
    """A park is a fixed point unless it is bounded: a wrist that takes no
    step cannot reduce its own orientation error, so on any demand that
    stays outside the hold the gate would hold it forever.

    Measured on this exact case before the budget went in: parked from
    iteration ~50 straight through 5000 (83 s at 60 Hz), `orient_residual`
    pinned at 1.00, the wrist stuck on a stop — and with the pressure
    sentinel of `test_parked_wrist_reports_saturating_pressure` working,
    that is a permanent full-strength haptic alarm during an ordinary
    over-reach. An alarm that cries wolf is worse than no alarm.

    The budget must also not be refundable by a dip. Two rejected rules,
    measured over these same 1200 solves: re-arm on any re-entry parked
    1164 of 1200 in bursts of a full budget; re-arm on a one-solve dip
    parked 618 in 559 single-solve flickers — half-rate wrist tracking under
    a permanent half-strength buzz. Only a sustained return refunds it, so
    the total below is one budget, not many.
    """
    tp = np.array([0.2, -0.2, 0.0])
    gated, ungated = _ik(), _ik(rot_err_hold=3.2)
    a = b = dict(_GATE_SEED)
    parked = 0
    for _ in range(1200):
        a, b = gated.solve(tp, quat.IDENTITY, a), ungated.solve(tp, quat.IDENTITY, b)
        parked += bool(gated.last_wrist_parked)
    assert parked <= PARK_MAX_SOLVES
    assert not gated.last_wrist_parked
    # ...and having let go, the gate costs the settled pose nothing.
    assert max(abs(a[j] - b[j]) for j in POSE_JOINTS) < 1e-6


def test_a_held_antipodal_demand_is_obeyed_once_the_debounce_is_spent():
    """The trade-off the expiry buys, pinned so it is not discovered by
    surprise. The wrist refuses the come-around for `PARK_MAX_SOLVES` — 1 s
    at 60 Hz, long enough to outlast any demand glitch — and then takes it,
    because a demand still outside the hold a second later is what the
    operator is asking for rather than quaternion jitter.
    """
    ik = _ik()
    q0 = dict(_GATE_SEED)
    tp, tq = _twisted(ik, q0, 150.0)
    cur = dict(q0)
    for _ in range(PARK_MAX_SOLVES):
        cur = ik.solve(tp, tq, cur)
        assert ik.last_wrist_parked
    assert max(abs(cur[j] - q0[j]) for j in POSE_JOINTS) < 1e-3
    cur = ik.solve(tp, tq, cur)
    assert not ik.last_wrist_parked
    assert max(abs(cur[j] - q0[j]) for j in WRIST_JOINTS) > 1e-3


def test_the_wrist_is_solved_against_a_pose_the_arm_can_hold():
    """Both intermediate poses of the decoupled chain are clamped to the
    arm's limits: the step-2 wrist prediction that places the position
    anchor, and the step-4 position pose the wrist is solved at.

    Unclamped they are fantasies exactly when it matters — during an
    over-reach, where the raw damped step runs past a stop the arm is
    already sitting on. Measured on this pose with a fully pinned arm: the
    pre-solve read an error angle of 3.11 rad while the real solve read
    1.74 on the same tick, so the two halves of the antipode gate disagreed
    about whether to park, and the gate cycled park/release on a 120-solve
    period forever at a pose where no joint was moving.
    """
    tp = np.array([0.2, 0.0, 0.2])
    gated, ungated = _ik(), _ik(rot_err_hold=3.2)
    a = b = _pose()
    for _ in range(400):
        a, b = gated.solve(tp, quat.IDENTITY, a), ungated.solve(tp, quat.IDENTITY, b)
    settled = dict(a)
    for _ in range(200):
        a, b = gated.solve(tp, quat.IDENTITY, a), ungated.solve(tp, quat.IDENTITY, b)
        assert not gated.last_wrist_parked
    assert max(abs(a[j] - settled[j]) for j in POSE_JOINTS) < 1e-9
    assert max(abs(a[j] - b[j]) for j in POSE_JOINTS) < 1e-9


def test_rot_err_hold_past_pi_disables_the_gate():
    """The error angle cannot exceed π, so a hold past 3.15 can never fire.
    The comparison demo that shows the raw come-around depends on that."""
    ik = _ik(rot_err_hold=3.2)
    q0 = dict(_GATE_SEED)
    out = ik.solve(*_twisted(ik, q0, 179.0), q0)
    assert not ik.last_wrist_parked
    assert max(abs(out[j] - q0[j]) for j in WRIST_JOINTS) > 1e-3


def test_the_gate_is_inert_in_ordinary_working_poses():
    """The gate must not be a tax on normal teleop: with the demand inside
    it, a gated solver and a disabled one agree step for step."""
    gated, ungated = _ik(), _ik(rot_err_hold=3.2)
    target = _pose(shoulder_pan=20, shoulder_lift=-75, elbow_flex=80,
                   wrist_flex=25, wrist_roll=-35)
    tp, tq = gated.fk(target)
    a = b = _pose(shoulder_lift=-60, elbow_flex=90)
    for _ in range(200):
        a, b = gated.solve(tp, tq, a), ungated.solve(tp, tq, b)
        assert not gated.last_wrist_parked
        assert a == b


def test_a_pure_position_demand_never_touches_the_gate():
    """A translation-only demand — the orientation asked for each tick is the
    one the tool already has — keeps the error angle nowhere near π however
    hard the position target is pushed, so the gated solver is step for step
    the ungated one right through the reach."""
    gated, ungated = _ik(), _ik(rot_err_hold=3.2)
    a = b = _pose(shoulder_lift=-70, elbow_flex=85)
    reach = gated.fk(_pose(shoulder_pan=15, shoulder_lift=-100, elbow_flex=110))[0]
    for _ in range(200):
        a = gated.solve(reach, gated.fk(a)[1], a)
        b = ungated.solve(reach, ungated.fk(b)[1], b)
        assert not gated.last_wrist_parked
        assert a == b


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
