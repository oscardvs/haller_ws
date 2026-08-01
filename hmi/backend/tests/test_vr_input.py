"""Tests for vr_input.py — pure math, no headset.

The point of most of these is not that the numbers are pretty but that the
*conventions* agree with retarget.py. A sign error here does not crash; it
produces an arm that drives smoothly in the wrong direction, or a wrist_roll
inverted on one side only. So the assertions are deliberately about direction
and handedness rather than just "returns a float".
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from haller_hmi import retarget, vr_input
from haller_hmi.vr_input import (
    BodyModel,
    ControllerSample,
    HeadSample,
    VR_PINCH_CALIB,
    build_side_frame,
    head_basis_xr,
    rotate,
    shoulder_xr,
    solve_elbow_rt,
    xr_to_rt,
)

TOL_DEG = 0.5
IDENTITY: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
HEAD_ORIGIN = HeadSample(position=(0.0, 0.0, 0.0), orientation=IDENTITY)
BODY = BodyModel()


def _quat_about(axis: tuple[float, float, float], deg: float):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2.0
    s = math.sin(h)
    return (float(a[0] * s), float(a[1] * s), float(a[2] * s), math.cos(h))


def test_module_imports():
    assert hasattr(vr_input, "BodyModel")
    assert hasattr(vr_input, "build_side_frame")


# ---- frame conversion -----------------------------------------------------


def test_xr_to_rt_flips_y_and_z():
    # XR "up" must become retarget "up", which is -Y there.
    assert list(xr_to_rt((0.0, 1.0, 0.0))) == [0.0, -1.0, 0.0]
    # XR forward (-Z) must become retarget "away from camera" (+Z).
    assert list(xr_to_rt((0.0, 0.0, -1.0))) == [0.0, 0.0, 1.0]
    # X is untouched, so the conversion is a rotation and not a mirror.
    assert list(xr_to_rt((1.0, 0.0, 0.0))) == [1.0, 0.0, 0.0]


def test_xr_to_rt_preserves_handedness():
    """A mirror here would invert every cross product downstream."""
    x, y, z = (xr_to_rt(v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    assert np.allclose(np.cross(x, y), z)


def test_rotate_identity_is_noop():
    assert np.allclose(rotate(IDENTITY, (0.3, -0.2, 0.5)), [0.3, -0.2, 0.5])


def test_rotate_90_about_y():
    # +Y is up in XR; rotating forward (-Z) by +90° about +Y gives -X.
    out = rotate(_quat_about((0, 1, 0), 90.0), (0.0, 0.0, -1.0))
    assert np.allclose(out, [-1.0, 0.0, 0.0], atol=1e-9)


# ---- head basis: yaw only -------------------------------------------------


def test_head_basis_identity():
    fwd, right = head_basis_xr(HEAD_ORIGIN)
    assert np.allclose(fwd, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.allclose(right, [1.0, 0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("pitch_deg", [-40.0, -15.0, 15.0, 40.0])
def test_pitching_the_head_does_not_move_the_shoulders(pitch_deg):
    """Nodding must not drag the body offsets around. This is the whole reason
    head_basis_xr throws away everything but yaw."""
    nodded = HeadSample(
        position=(0.0, 0.0, 0.0), orientation=_quat_about((1, 0, 0), pitch_deg)
    )
    assert np.allclose(
        shoulder_xr(nodded, "right", BODY),
        shoulder_xr(HEAD_ORIGIN, "right", BODY),
        atol=1e-9,
    )


def test_rolling_the_head_does_not_move_the_shoulders():
    tilted = HeadSample(
        position=(0.0, 0.0, 0.0), orientation=_quat_about((0, 0, 1), 25.0)
    )
    assert np.allclose(
        shoulder_xr(tilted, "left", BODY),
        shoulder_xr(HEAD_ORIGIN, "left", BODY),
        atol=1e-9,
    )


def test_yawing_the_head_does_move_the_shoulders():
    """The converse: turning really must swing the shoulder line, or the
    operator could never drive the robot anywhere but straight ahead."""
    turned = HeadSample(
        position=(0.0, 0.0, 0.0), orientation=_quat_about((0, 1, 0), 90.0)
    )
    assert not np.allclose(
        shoulder_xr(turned, "right", BODY),
        shoulder_xr(HEAD_ORIGIN, "right", BODY),
        atol=1e-3,
    )


def test_shoulders_sit_below_behind_and_either_side_of_the_head():
    r = shoulder_xr(HEAD_ORIGIN, "right", BODY)
    ll = shoulder_xr(HEAD_ORIGIN, "left", BODY)
    assert r[1] == pytest.approx(-BODY.shoulder_drop)      # below (XR: -Y)
    assert r[2] == pytest.approx(BODY.shoulder_back)        # behind (XR: +Z)
    assert r[0] > 0 and ll[0] < 0                           # right is +X
    assert r[0] == pytest.approx(-ll[0])                    # symmetric


# ---- elbow IK -------------------------------------------------------------


def test_elbow_hangs_below_the_shoulder_wrist_line():
    """In the retarget frame down is +Y, and a relaxed elbow drops. Same
    convention retarget.arm_direction_vectors picks for the inverse map."""
    s = np.array([0.0, 0.0, 0.0])
    w = np.array([0.0, 0.0, 0.5])  # straight ahead of the shoulder
    e = solve_elbow_rt(s, w, BODY.upper_arm, BODY.fore_arm)
    assert e[1] > 0.0, "elbow should be displaced toward +Y (operator's down)"


def test_elbow_respects_both_link_lengths():
    s = np.array([0.1, -0.2, 0.05])
    w = np.array([0.1, 0.05, 0.45])
    e = solve_elbow_rt(s, w, BODY.upper_arm, BODY.fore_arm)
    assert float(np.linalg.norm(e - s)) == pytest.approx(BODY.upper_arm, abs=1e-9)
    assert float(np.linalg.norm(w - e)) == pytest.approx(BODY.fore_arm, abs=1e-6)


def test_beyond_reach_the_arm_straightens_rather_than_failing():
    s = np.array([0.0, 0.0, 0.0])
    w = np.array([0.0, 0.0, 5.0])  # far out of reach
    e = solve_elbow_rt(s, w, BODY.upper_arm, BODY.fore_arm)
    assert np.all(np.isfinite(e))
    # Straight arm: elbow on the shoulder→wrist ray at exactly l1.
    assert np.allclose(e, [0.0, 0.0, BODY.upper_arm], atol=1e-9)
    pan, lift, elbow_deg = retarget.compute_arm_angles(
        tuple(s), tuple(e), tuple(w)
    )
    assert elbow_deg == pytest.approx(0.0, abs=TOL_DEG)


def test_degenerate_zero_length_does_not_nan():
    s = np.array([0.2, 0.2, 0.2])
    e = solve_elbow_rt(s, s.copy(), BODY.upper_arm, BODY.fore_arm)
    assert np.all(np.isfinite(e))


# ---- end to end, through the real retargeter ------------------------------


def _reach_forward(distance: float, side="right") -> ControllerSample:
    """Controller placed `distance` straight ahead of that shoulder."""
    s = shoulder_xr(HEAD_ORIGIN, side, BODY)
    return ControllerSample(
        position=tuple(s + np.array([0.0, 0.0, -distance])),
        orientation=IDENTITY,
        trigger=0.0,
        tracked=True,
    )


def test_reaching_straight_ahead_gives_zero_pan():
    frame = build_side_frame(HEAD_ORIGIN, _reach_forward(0.50), "right", BODY)
    assert frame is not None
    pan, lift, elbow_deg = retarget.compute_arm_angles(
        frame["pose"]["shoulder"], frame["pose"]["elbow"], frame["pose"]["wrist"]
    )
    assert pan == pytest.approx(0.0, abs=TOL_DEG)
    # Elbow drops, so the upper arm tilts downward: lift is positive in a frame
    # whose +Y is down. See retarget.py's module docstring.
    assert lift > 0.0
    # Triangle 0.29 / 0.27 / 0.50 — interior angle at the elbow is 126.4°, and
    # elbow_flex is measured from straight, so 180 - 126.4.
    assert elbow_deg == pytest.approx(53.6, abs=1.0)


def test_extending_further_straightens_the_elbow():
    near = build_side_frame(HEAD_ORIGIN, _reach_forward(0.35), "right", BODY)
    far = build_side_frame(HEAD_ORIGIN, _reach_forward(0.55), "right", BODY)
    _, _, e_near = retarget.compute_arm_angles(
        near["pose"]["shoulder"], near["pose"]["elbow"], near["pose"]["wrist"]
    )
    _, _, e_far = retarget.compute_arm_angles(
        far["pose"]["shoulder"], far["pose"]["elbow"], far["pose"]["wrist"]
    )
    assert e_far < e_near, "reaching out must open the elbow toward straight"


def test_moving_the_controller_right_pans_positive():
    s = shoulder_xr(HEAD_ORIGIN, "right", BODY)
    out = ControllerSample(
        position=tuple(s + np.array([0.30, 0.0, -0.30])),
        orientation=IDENTITY, trigger=0.0, tracked=True,
    )
    frame = build_side_frame(HEAD_ORIGIN, out, "right", BODY)
    pan, _, _ = retarget.compute_arm_angles(
        frame["pose"]["shoulder"], frame["pose"]["elbow"], frame["pose"]["wrist"]
    )
    assert pan > 0.0


def test_lowering_the_controller_increases_lift():
    s = shoulder_xr(HEAD_ORIGIN, "right", BODY)
    low = ControllerSample(
        position=tuple(s + np.array([0.0, -0.35, -0.25])),
        orientation=IDENTITY, trigger=0.0, tracked=True,
    )
    high = ControllerSample(
        position=tuple(s + np.array([0.0, 0.10, -0.40])),
        orientation=IDENTITY, trigger=0.0, tracked=True,
    )
    f_low = build_side_frame(HEAD_ORIGIN, low, "right", BODY)
    f_high = build_side_frame(HEAD_ORIGIN, high, "right", BODY)
    _, lift_low, _ = retarget.compute_arm_angles(
        f_low["pose"]["shoulder"], f_low["pose"]["elbow"], f_low["pose"]["wrist"]
    )
    _, lift_high, _ = retarget.compute_arm_angles(
        f_high["pose"]["shoulder"], f_high["pose"]["elbow"], f_high["pose"]["wrist"]
    )
    assert lift_low > lift_high


# ---- gripper --------------------------------------------------------------


@pytest.mark.parametrize("trigger", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_trigger_maps_exactly_onto_gripper_aperture(trigger):
    """Squeezing shuts the gripper. The separation is constructed, not
    measured, so this should be exact rather than approximate."""
    c = ControllerSample(
        position=(0.2, -0.2, -0.4), orientation=IDENTITY,
        trigger=trigger, tracked=True,
    )
    frame = build_side_frame(HEAD_ORIGIN, c, "right", BODY)
    aperture = retarget.compute_pinch(
        frame["hand"]["thumb_tip"], frame["hand"]["index_tip"], VR_PINCH_CALIB
    )
    assert aperture == pytest.approx(1.0 - trigger, abs=1e-6)


def test_trigger_is_clamped_outside_unit_range():
    for t, expected in ((-0.5, 1.0), (1.5, 0.0)):
        c = ControllerSample(
            position=(0.2, -0.2, -0.4), orientation=IDENTITY,
            trigger=t, tracked=True,
        )
        frame = build_side_frame(HEAD_ORIGIN, c, "right", BODY)
        assert retarget.compute_pinch(
            frame["hand"]["thumb_tip"], frame["hand"]["index_tip"], VR_PINCH_CALIB
        ) == pytest.approx(expected, abs=1e-6)


# ---- handedness -----------------------------------------------------------


def _palm_normal(frame) -> np.ndarray:
    h = frame["hand"]
    w = np.asarray(h["wrist"])
    return np.cross(np.asarray(h["index_mcp"]) - w, np.asarray(h["pinky_mcp"]) - w)


def test_palm_normal_flips_between_hands():
    """Same controller pose, opposite hands, opposite palm normal — as on a
    real pair of hands. If this ever passes with the normals *equal*,
    wrist_roll is silently inverted on exactly one arm."""
    c = ControllerSample(
        position=(0.0, -0.2, -0.4), orientation=IDENTITY, trigger=0.0, tracked=True
    )
    n_r = _palm_normal(build_side_frame(HEAD_ORIGIN, c, "right", BODY))
    n_l = _palm_normal(build_side_frame(HEAD_ORIGIN, c, "left", BODY))
    assert np.allclose(n_r, -n_l, atol=1e-12)


def test_neutral_grip_is_near_zero_wrist_angles():
    """Controller held level, pointing where the forearm points: the wrist
    should read as roughly straight and unrolled, so an operator's comfortable
    neutral is not already halfway into a joint limit."""
    frame = build_side_frame(HEAD_ORIGIN, _reach_forward(0.50), "right", BODY)
    forearm = tuple(
        np.asarray(frame["pose"]["wrist"]) - np.asarray(frame["pose"]["elbow"])
    )
    wflex, wroll = retarget.compute_wrist_angles(forearm, frame["hand"])
    assert abs(wflex) < 35.0, f"neutral grip already flexed {wflex:.1f}°"
    assert abs(wroll) < 35.0, f"neutral grip already rolled {wroll:.1f}°"


def test_rolling_the_controller_rolls_the_wrist():
    s = shoulder_xr(HEAD_ORIGIN, "right", BODY)
    pos = tuple(s + np.array([0.0, 0.0, -0.5]))
    angles = []
    for deg in (-50.0, 0.0, 50.0):
        c = ControllerSample(
            position=pos, orientation=_quat_about((0, 0, 1), deg),
            trigger=0.0, tracked=True,
        )
        f = build_side_frame(HEAD_ORIGIN, c, "right", BODY)
        forearm = tuple(
            np.asarray(f["pose"]["wrist"]) - np.asarray(f["pose"]["elbow"])
        )
        angles.append(retarget.compute_wrist_angles(forearm, f["hand"])[1])
    assert angles[0] != pytest.approx(angles[2], abs=5.0), (
        "rolling the controller must change wrist_roll"
    )


# ---- tracking loss --------------------------------------------------------


def test_untracked_controller_yields_none():
    c = ControllerSample(
        position=(0.0, 0.0, 0.0), orientation=IDENTITY, trigger=0.0, tracked=False
    )
    assert build_side_frame(HEAD_ORIGIN, c, "right", BODY) is None


def test_tracked_frame_is_full_confidence():
    frame = build_side_frame(HEAD_ORIGIN, _reach_forward(0.45), "right", BODY)
    assert frame["confidence"] == 1.0
    assert frame["confidence"] >= retarget.CONFIDENCE_FLOOR


def test_every_landmark_is_finite():
    """NaN reaching the retargeter would clamp to a joint limit and lurch."""
    frame = build_side_frame(HEAD_ORIGIN, _reach_forward(0.45), "right", BODY)
    for group in ("pose", "hand"):
        for name, v in frame[group].items():
            assert all(math.isfinite(c) for c in v), f"{group}.{name} = {v}"
