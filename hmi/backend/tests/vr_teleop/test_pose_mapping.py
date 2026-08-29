"""The robot-agnostic clutch mapping: gains, frames, and the reach limits.

The reach-limit tests are the ones worth keeping honest. They encode the two
hardware misbehaviours the absorbing limit exists to kill — a demand winding
up past a stop, and an orientation demand coming around the far side — and
either would come back the moment the limit turned into a plain clamp.
"""
from __future__ import annotations

import numpy as np
import pytest

from haller_hmi.vr_teleop.core import frames, quat
from haller_hmi.vr_teleop.core.pose_mapping import ClutchPoseMapper

EE_P = np.array([0.0, -0.30, 0.10])
EE_Q = quat.IDENTITY.copy()
CTRL_P = np.array([0.0, 1.20, -0.30])
CTRL_Q = quat.IDENTITY.copy()


def _mapper(**kw) -> ClutchPoseMapper:
    m = ClutchPoseMapper(**kw)
    m.engage(CTRL_P, CTRL_Q, EE_P, EE_Q)
    return m


# ---- quaternion helpers --------------------------------------------------

@pytest.mark.parametrize("v", [
    [0.3, 0.0, 0.0], [0.0, -1.2, 0.4], [2.9, 0.1, 0.2], [0.0, 0.0, 0.0],
])
def test_rotvec_roundtrip(v):
    got = quat.to_rotvec(quat.from_rotvec(np.array(v)))
    assert got == pytest.approx(v, abs=1e-9)


def test_from_mat_survives_a_180_degree_rotation():
    """Shepperd branch selection, not the trace form — a hand turned all the
    way over is reachable, and the trace form divides by zero there."""
    for axis in np.eye(3):
        R = quat.to_mat(quat.from_rotvec(axis * np.pi))
        assert quat.angle_between(quat.from_mat(R), quat.from_rotvec(axis * np.pi)) \
            == pytest.approx(0.0, abs=1e-6)


def test_power_scales_the_angle_and_keeps_the_axis():
    q = quat.from_rotvec(np.array([0.0, 0.6, 0.0]))
    assert quat.to_rotvec(quat.power(q, 0.5)) == pytest.approx([0.0, 0.3, 0.0], abs=1e-9)
    assert quat.to_rotvec(quat.power(q, 2.0)) == pytest.approx([0.0, 1.2, 0.0], abs=1e-9)


# ---- mapping -------------------------------------------------------------

def test_disengaged_returns_none():
    m = _mapper()
    m.disengage()
    assert m.target(CTRL_P, CTRL_Q) is None


def test_no_motion_means_no_target_change():
    m = _mapper()
    p, q = m.target(CTRL_P, CTRL_Q, EE_P, EE_Q)
    assert p == pytest.approx(EE_P, abs=1e-12)
    assert quat.angle_between(q, EE_Q) == pytest.approx(0.0, abs=1e-9)


def test_translation_gain():
    m = _mapper(scale=0.5, pos_reach_limit=0.0)
    p, _ = m.target(CTRL_P + np.array([0.06, 0, 0]), CTRL_Q)
    assert p == pytest.approx(EE_P + np.array([0.03, 0, 0]), abs=1e-9)


def test_reengage_moves_the_origin():
    m = _mapper(pos_reach_limit=0.0)
    new_ee = np.array([0.10, -0.25, 0.20])
    m.engage(CTRL_P, CTRL_Q, new_ee, EE_Q)
    p, _ = m.target(CTRL_P + np.array([0.05, 0, 0]), CTRL_Q)
    assert p == pytest.approx(new_ee + np.array([0.05, 0, 0]), abs=1e-9)


@pytest.mark.parametrize("stance,expected", [
    # Hand +x is the operator's right; hand −z is forward. See
    # `core.frames.STANCES` for why each stance maps where it does.
    ("behind", {"right": [-0.05, 0, 0], "forward": [0, -0.05, 0]}),
    ("mirror", {"right": [0.05, 0, 0], "forward": [0, -0.05, 0]}),
    ("front", {"right": [0.05, 0, 0], "forward": [0, 0.05, 0]}),
])
def test_stance_direction_mapping(stance, expected):
    for name, delta_quest in (("right", [0.05, 0, 0]), ("forward", [0, 0, -0.05])):
        m = ClutchPoseMapper(R=frames.stance_rotation(stance), pos_reach_limit=0.0)
        m.engage(CTRL_P, CTRL_Q, EE_P, EE_Q)
        p, _ = m.target(CTRL_P + np.array(delta_quest), CTRL_Q)
        assert p - EE_P == pytest.approx(expected[name], abs=1e-9)


def test_up_is_up_in_every_stance():
    for stance in frames.STANCES:
        m = ClutchPoseMapper(R=frames.stance_rotation(stance), pos_reach_limit=0.0)
        m.engage(CTRL_P, CTRL_Q, EE_P, EE_Q)
        p, _ = m.target(CTRL_P + np.array([0, 0.05, 0]), CTRL_Q)
        assert p - EE_P == pytest.approx([0, 0, 0.05], abs=1e-9)


def test_yaw_on_engage_cancels_the_operators_heading():
    """Turn the operator 90° in the room and 'push forward' must still mean
    'the arm extends'."""
    for yaw_deg in (-90.0, -30.0, 0.0, 45.0, 120.0):
        yaw = np.radians(yaw_deg)
        R = frames.stance_rotation("behind", yaw)
        m = ClutchPoseMapper(R=R, pos_reach_limit=0.0)
        m.engage(CTRL_P, CTRL_Q, EE_P, EE_Q)
        # "Forward" for an operator facing `yaw` is Ry(yaw) applied to −z.
        fwd = quat.rot_y(yaw) @ np.array([0.0, 0.0, -1.0])
        p, _ = m.target(CTRL_P + 0.05 * fwd, CTRL_Q)
        assert p - EE_P == pytest.approx([0, -0.05, 0], abs=1e-9)


def test_mirror_stance_is_a_reflection_and_rotations_still_work():
    """det = −1 is a CHOICE, not a bug. It also means rotations cannot go
    across by quaternion conjugation — this pins that they still go across."""
    R = frames.stance_rotation("mirror")
    assert np.linalg.det(R) == pytest.approx(-1.0, abs=1e-9)
    m = ClutchPoseMapper(R=R, rot_reach_limit=0.0, pos_reach_limit=0.0)
    m.engage(CTRL_P, CTRL_Q, EE_P, EE_Q)
    # A 40° twist of the hand must still produce a 40° twist of the tool.
    twist = quat.from_rotvec(np.array([0.0, np.radians(40.0), 0.0]))
    _, q = m.target(CTRL_P, quat.mul(twist, CTRL_Q))
    assert np.degrees(quat.angle_between(q, EE_Q)) == pytest.approx(40.0, abs=1e-6)


# ---- reach limits --------------------------------------------------------

def test_position_reach_limit_absorbs_and_reversal_bites():
    m = _mapper(pos_reach_limit=0.10)
    # Push the hand a metre while the arm stays exactly where it is.
    for _ in range(20):
        p, _ = m.target(CTRL_P + np.array([1.0, 0, 0]), CTRL_Q, EE_P, EE_Q)
    assert np.linalg.norm(p - EE_P) == pytest.approx(0.10, abs=1e-9)
    # Reverse 4 cm: the target must come back 4 cm NOW, not after the
    # 0.9 m of absorbed overshoot has been retraced.
    p2, _ = m.target(CTRL_P + np.array([0.96, 0, 0]), CTRL_Q, EE_P, EE_Q)
    assert np.linalg.norm(p2 - p) == pytest.approx(0.04, abs=1e-9)


def test_rotation_reach_limit_never_comes_around():
    """Twist 300° in 2° steps against a frozen arm. Without the incremental
    limit the target would eventually agree that 300° clockwise is 60°
    anticlockwise and snap the wrist in from the wrong side."""
    m = _mapper(rot_reach_limit=0.5)
    worst = 0.0
    for deg in range(2, 302, 2):
        q_ctrl = quat.from_rotvec(np.array([0.0, np.radians(deg), 0.0]))
        _, q = m.target(CTRL_P, q_ctrl, EE_P, EE_Q)
        worst = max(worst, quat.angle_between(q, EE_Q))
    assert worst <= 0.5 + 1e-6


def test_incremental_equals_absolute_inside_the_limit():
    """At unit gain and with the arm keeping up, the limited path must be
    the plain absolute mapping — the limit is meant to be invisible until
    it bites."""
    limited = ClutchPoseMapper(pos_reach_limit=10.0, rot_reach_limit=3.0)
    plain = ClutchPoseMapper(pos_reach_limit=0.0, rot_reach_limit=0.0)
    for m in (limited, plain):
        m.engage(CTRL_P, CTRL_Q, EE_P, EE_Q)
    ee_q = EE_Q
    ee_p = EE_P
    worst_rot = worst_pos = 0.0
    for i in range(1, 41):
        q_ctrl = quat.from_rotvec(np.radians(i) * np.array([0.4, 0.8, 0.2]))
        p_ctrl = CTRL_P + np.array([0.002 * i, -0.001 * i, 0.003 * i])
        p_l, q_l = limited.target(p_ctrl, q_ctrl, ee_p, ee_q)
        p_a, q_a = plain.target(p_ctrl, q_ctrl)
        worst_rot = max(worst_rot, quat.angle_between(q_l, q_a))
        worst_pos = max(worst_pos, float(np.linalg.norm(p_l - p_a)))
        ee_p, ee_q = p_l, q_l        # the arm tracks perfectly
    assert worst_rot < 1e-6
    assert worst_pos < 1e-9


def test_rotation_pivot_swings_the_tool_around_the_anchor():
    """With a pivot set, a pure hand twist moves the tool along an arc about
    the pivot instead of spinning it on the spot."""
    pivot = EE_P + np.array([0.0, 0.06, 0.0])
    m = ClutchPoseMapper(pos_reach_limit=0.0, rot_reach_limit=0.0)
    m.engage(CTRL_P, CTRL_Q, EE_P, EE_Q, pivot_armbase=pivot)
    twist = quat.from_rotvec(np.array([0.0, 0.0, np.radians(30.0)]))
    p, _ = m.target(CTRL_P, quat.mul(twist, CTRL_Q))
    # Still the same distance from the pivot, but no longer the same point.
    assert np.linalg.norm(p - pivot) == pytest.approx(
        np.linalg.norm(EE_P - pivot), abs=1e-9)
    assert np.linalg.norm(p - EE_P) > 1e-3


def test_absorbed_diagnostics_saturate_at_the_limit():
    m = _mapper(pos_reach_limit=0.10, rot_reach_limit=0.5)
    m.target(CTRL_P + np.array([0.01, 0, 0]), CTRL_Q, EE_P, EE_Q)
    assert m.last_pos_absorbed < 0.5
    for _ in range(10):
        m.target(CTRL_P + np.array([1.0, 0, 0]), CTRL_Q, EE_P, EE_Q)
    assert m.last_pos_absorbed == pytest.approx(1.0)
