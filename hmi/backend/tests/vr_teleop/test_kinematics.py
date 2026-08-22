"""The SO-101 kinematics the guard and the IK now share.

Two of these restate claims the module docstring makes. That is deliberate:
the decoupled solver is built on "the wrist anchor is invariant to both wrist
joints" and "the two wrist axes are never parallel", and if a future MJCF
update broke either, the solver would keep running and quietly stop being
decoupled. `tests/sim/test_collision_sim.py` pins the chain against MuJoCo;
these pin what the chain has to MEAN.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from haller_hmi import collision
from haller_hmi.so101_kinematics import (
    ORIENTATION_JOINTS,
    POSE_JOINTS,
    POSITION_JOINTS,
    fk_frames,
    fk_points,
    jacobian_position,
    jacobian_rotation,
)


def _pose(**kw) -> dict[str, float]:
    return {j: float(kw.get(j, 0.0)) for j in POSE_JOINTS}


def test_collision_still_re_exports_the_chain():
    """The guard's public names survived the move to a shared module."""
    assert collision.fk_points is fk_points
    assert collision.POSE_JOINTS == POSE_JOINTS
    assert len(collision._CHAIN) == 5


def test_fk_frames_agrees_with_fk_points():
    q = _pose(shoulder_pan=17.0, shoulder_lift=-63.0, elbow_flex=81.0,
              wrist_flex=-22.0, wrist_roll=45.0)
    pts = fk_points((0.2, 0.0, 0.0), 15.0, q)
    frames = fk_frames(q, (0.2, 0.0, 0.0), 15.0)
    for name, value in pts.items():
        assert frames.points[name] == pytest.approx(value, abs=1e-12)


def test_wrist_anchor_is_invariant_to_both_wrist_joints():
    """The premise of the whole position/orientation split."""
    base = fk_frames({}).wrist_pos
    for joint in ORIENTATION_JOINTS:
        for angle in (-80.0, -10.0, 40.0, 90.0):
            moved = fk_frames({joint: angle}).wrist_pos
            assert moved == pytest.approx(base, abs=1e-12)


def test_wrist_axes_never_align():
    """No gimbal lock inside this wrist — which is why the ported solver
    carries no gimbal-proximity damping."""
    for angle in np.linspace(-95.0, 95.0, 41):
        f = fk_frames({"wrist_flex": float(angle)})
        cos = abs(float(np.dot(f.joint_axis["wrist_flex"], f.joint_axis["wrist_roll"])))
        assert cos < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_position_jacobian_matches_central_differences(seed):
    rng = np.random.default_rng(seed)
    q = {j: float(rng.uniform(-70, 70)) for j in POSE_JOINTS}
    f = fk_frames(q)
    analytic = jacobian_position(f, f.wrist_pos, POSITION_JOINTS)
    h = 1e-5
    for k, joint in enumerate(POSITION_JOINTS):
        plus = fk_frames({**q, joint: q[joint] + math.degrees(h)}).wrist_pos
        minus = fk_frames({**q, joint: q[joint] - math.degrees(h)}).wrist_pos
        assert (plus - minus) / (2 * h) == pytest.approx(analytic[:, k], abs=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotation_jacobian_matches_central_differences(seed):
    rng = np.random.default_rng(seed)
    q = {j: float(rng.uniform(-70, 70)) for j in POSE_JOINTS}
    f = fk_frames(q)
    analytic = jacobian_rotation(f, ORIENTATION_JOINTS)
    h = 1e-5
    for k, joint in enumerate(ORIENTATION_JOINTS):
        dR = (fk_frames({**q, joint: q[joint] + math.degrees(h)}).tool_R
              @ fk_frames({**q, joint: q[joint] - math.degrees(h)}).tool_R.T)
        w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                      dR[1, 0] - dR[0, 1]]) / 2.0
        assert w / (2 * h) == pytest.approx(analytic[:, k], abs=1e-4)


def test_base_frame_orientation():
    """The convention every stance matrix is written against: z up, the arm
    reaching along −y at the all-zero pose."""
    f = fk_frames({})
    assert f.tip_pos[2] > 0.0
    assert f.tip_pos[1] < -0.4
    # Loose, not exact: the MJCF's base quaternion is written to six figures
    # (0.707105, 0.707108) and is not quite a 90° rotation, so the chain
    # carries a sub-micron lateral offset and a few microradians of tilt at
    # the zero pose. Tightening this would pin a rounding in a vendored model
    # rather than a property of the arm.
    assert f.tip_pos[0] == pytest.approx(0.0, abs=1e-6)
    assert f.joint_axis["shoulder_pan"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-5)
