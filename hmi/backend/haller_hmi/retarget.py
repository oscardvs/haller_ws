"""Pure-function MediaPipe-keypoints → SO-101 joint-angle retargeting.

The maths here is the SEW-Mimic-style analytical closed form: human shoulder/
elbow/wrist define the upper-arm + forearm vectors, which map 1:1 to the
SO-101's shoulder_pan / shoulder_lift / elbow_flex. The hand landmarks define
the wrist orientation, which maps to wrist_flex + wrist_roll. Thumb-index
distance maps to gripper aperture.

Nothing in this file touches a robot, the network, or threads. Inputs are
plain Python dicts / tuples; outputs are plain dicts. Everything is unit-
testable with synthetic geometry.
"""
from __future__ import annotations

import math
from typing import Literal, TypedDict, cast

import numpy as np

# Coordinates are MediaPipe pose-world metres, passed through unchanged from
# the browser: +X image-right, +Y image-DOWN, +Z away from the camera (per
# MediaPipe, a smaller z is nearer the lens). Nothing here rebases them.
#
# Read the joint names as labels for these axes rather than as anatomy:
# `shoulder_lift` is asin(U_y / |U|) in a frame whose +Y points down, so it
# grows as the arm points DOWNWARD. That has been the deployed convention since
# the retargeter shipped and the operator's calibration is built on it — the
# note this replaced claimed a rebase to +Y up that no code has ever performed,
# which is a trap for anyone deriving new geometry from it (see
# `arm_direction_vectors`, which must invert what the code does, not what the
# comment said).

Vec3 = tuple[float, float, float]


class PoseLandmarks(TypedDict):
    shoulder: Vec3
    elbow: Vec3
    wrist: Vec3


class HandLandmarks(TypedDict):
    wrist: Vec3
    thumb_tip: Vec3
    index_tip: Vec3
    index_mcp: Vec3
    middle_mcp: Vec3
    pinky_mcp: Vec3


class SideFrame(TypedDict):
    pose: PoseLandmarks
    hand: HandLandmarks
    confidence: float


class PinchCalib(TypedDict):
    min_m: float
    max_m: float


class JointGoal(TypedDict):
    shoulder_pan: float    # degrees
    shoulder_lift: float   # degrees
    elbow_flex: float      # degrees
    wrist_flex: float      # degrees
    wrist_roll: float      # degrees
    gripper: float         # [0, 1] — 0 = closed, 1 = open


Side = Literal["left", "right"]


def _np(v: Vec3) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _safe_norm(v: np.ndarray, eps: float = 1e-9) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


def _signed_angle_deg(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from a to b around `axis` (right-hand rule), in degrees.

    Returns 0.0 if either input vector is degenerate (length < 1e-9), so
    callers don't have to guard zero-length cases themselves.
    """
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 0.0
    a_n = a / a_norm
    b_n = b / b_norm
    cross = np.cross(a_n, b_n)
    sin_t = float(np.dot(cross, axis / _safe_norm(axis)))
    cos_t = float(np.dot(a_n, b_n))
    return math.degrees(math.atan2(sin_t, cos_t))


def compute_arm_angles(
    shoulder: Vec3, elbow: Vec3, wrist: Vec3
) -> tuple[float, float, float]:
    """Return (shoulder_pan, shoulder_lift, elbow_flex) in degrees.

    Coordinate convention (after upstream rebase): +X right, +Y up, +Z away
    from the camera. The retargeted angles are *signed*, centred on a
    straight-arm-pointing-forward neutral pose (pan=0, lift=0, elbow=0).
    """
    S, E, W = _np(shoulder), _np(elbow), _np(wrist)
    U = E - S                          # upper-arm vector
    F = W - E                          # forearm vector
    u_len = _safe_norm(U)

    # Shoulder pan: azimuth of U projected onto the horizontal (X,Z) plane.
    # pan = 0 when U points purely +Z; pan = +90° when U points purely +X.
    pan = math.degrees(math.atan2(U[0], U[2]))

    # Shoulder lift: elevation of U above horizontal. lift = +90° straight up.
    lift = math.degrees(math.asin(max(-1.0, min(1.0, U[1] / u_len))))

    # Elbow flex: angle between U and F. 0° = straight, 90° = right angle.
    cos_t = float(np.dot(U / u_len, F / _safe_norm(F)))
    elbow = math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))

    return pan, lift, elbow


def arm_direction_vectors(
    pan_deg: float, lift_deg: float, elbow_deg: float
) -> tuple[Vec3, Vec3]:
    """Inverse of `compute_arm_angles`: angles → unit upper-arm + forearm.

    Used to draw the robot's current pose back onto the operator's camera as a
    ghost to match before authority transfers. Kept here, beside the forward
    map and tested against it by round-trip, so the two can never drift into
    disagreeing about a convention; the browser only draws what this returns.

    `compute_arm_angles` throws away one degree of freedom — the elbow's swivel
    about the upper-arm axis — because the SO-101 has no joint for it. Any
    inverse must therefore pick a swivel. This one bends the forearm toward +Y,
    which in the frame above is image-down: the resting human choice, and the
    one that makes the ghost reachable without contortion. When the upper arm
    is itself along ±Y (a hanging arm — common, not exotic) that direction is
    undefined, and the forearm falls back to -Z, i.e. forward, out of the
    screen toward the camera.
    """
    pan = math.radians(pan_deg)
    lift = math.radians(lift_deg)
    elbow = math.radians(elbow_deg)

    # Exactly inverts pan = atan2(U_x, U_z) and lift = asin(U_y / |U|).
    u = np.array([
        math.cos(lift) * math.sin(pan),
        math.sin(lift),
        math.cos(lift) * math.cos(pan),
    ])

    # Bend direction: the component of +Y perpendicular to the upper arm. `d`
    # and `u` are orthonormal, so rotating u toward d by the elbow angle is a
    # plain 2-D combination — no rotation matrix needed, and the round trip
    # through acos(u . f) returns `elbow` by construction.
    y_axis = np.array([0.0, 1.0, 0.0])
    d = y_axis - float(np.dot(y_axis, u)) * u
    if float(np.linalg.norm(d)) < 1e-6:
        d = np.array([0.0, 0.0, -1.0])
        d = d - float(np.dot(d, u)) * u
    d = d / _safe_norm(d)

    f = u * math.cos(elbow) + d * math.sin(elbow)
    return (
        cast(Vec3, tuple(float(v) for v in u)),
        cast(Vec3, tuple(float(v) for v in f)),
    )


def compute_wrist_angles(
    forearm: Vec3, hand: HandLandmarks
) -> tuple[float, float]:
    """Return (wrist_flex, wrist_roll) in degrees.

    Neutral pose: forearm and hand point along +Z, palm faces -Y (down).
    `wrist_flex` is the signed angle of the middle-finger ray relative to the
    forearm, measured around the wrist's lateral axis — `cross(palm_normal,
    forearm)`, i.e. the axis pointing sideways out of the hand. Positive
    flex = hand bends up (middle finger rotates toward palm-normal direction).
    `wrist_roll` is the signed rotation of the palm normal around the forearm
    axis relative to the neutral palm-down direction.
    """
    F = _np(forearm)
    f_hat = F / _safe_norm(F)

    w = _np(hand["wrist"])
    middle_dir = _np(hand["middle_mcp"]) - w
    index_dir = _np(hand["index_mcp"]) - w
    pinky_dir = _np(hand["pinky_mcp"]) - w

    # Palm normal as it currently is.
    palm_normal = np.cross(index_dir, pinky_dir)
    palm_normal_hat = palm_normal / _safe_norm(palm_normal)

    # "Neutral palm-down" reference: the world-down direction (-Y), projected
    # onto the plane perpendicular to the forearm and renormalised.
    world_down = np.array([0.0, -1.0, 0.0])
    neutral_palm = world_down - np.dot(world_down, f_hat) * f_hat
    neutral_palm_hat = neutral_palm / _safe_norm(neutral_palm)

    # wrist_roll: signed rotation of palm_normal around forearm, relative to neutral.
    wroll = _signed_angle_deg(neutral_palm_hat, palm_normal_hat, f_hat)

    # wrist_flex: signed angle from forearm to middle-finger direction,
    # measured around the wrist-side axis (perpendicular to both forearm and
    # palm normal). For a flat palm-down hand with forearm +Z, this axis is
    # along ±X, so bending the hand up (middle finger +Y) gives ±90°.
    # flex_axis collapses when palm_normal is parallel to forearm (wrist
    # supinated along the arm axis). _signed_angle_deg returns 0.0 in that
    # case via its zero-length-input guard.
    flex_axis = np.cross(palm_normal_hat, f_hat)
    wflex = _signed_angle_deg(f_hat, middle_dir, flex_axis)

    return wflex, wroll


# Minimum per-side confidence below which we refuse to emit a joint goal.
CONFIDENCE_FLOOR = 0.4


def compute_pinch(
    thumb_tip: Vec3, index_tip: Vec3, calib: PinchCalib
) -> float:
    """Return gripper aperture in [0, 1].

    Maps thumb-tip ↔ index-tip distance onto [calib.min_m, calib.max_m].
    0 = fully closed pinch, 1 = fully open.
    """
    d = float(np.linalg.norm(_np(thumb_tip) - _np(index_tip)))
    lo, hi = float(calib["min_m"]), float(calib["max_m"])
    span = max(hi - lo, 1e-6)
    return max(0.0, min(1.0, (d - lo) / span))


def apply_mirror(goal: JointGoal) -> JointGoal:
    """Flip the side-specific angles for the contralateral arm.

    Mirroring across the operator's mid-sagittal plane negates only the
    rotations that point sideways: shoulder_pan and wrist_roll. All other
    joints (lift, elbow_flex, wrist_flex, gripper) are side-invariant.
    """
    return {
        "shoulder_pan": -goal["shoulder_pan"],
        "shoulder_lift": goal["shoulder_lift"],
        "elbow_flex": goal["elbow_flex"],
        "wrist_flex": goal["wrist_flex"],
        "wrist_roll": -goal["wrist_roll"],
        "gripper": goal["gripper"],
    }


def compute_joint_goal(
    side: SideFrame, calib: PinchCalib, *, mirror: bool
) -> JointGoal | None:
    """End-to-end retargeting for one side. Returns None below the confidence floor."""
    if side["confidence"] < CONFIDENCE_FLOOR:
        return None
    pan, lift, elbow = compute_arm_angles(
        side["pose"]["shoulder"], side["pose"]["elbow"], side["pose"]["wrist"]
    )
    forearm = tuple(
        np.asarray(side["pose"]["wrist"], dtype=np.float64)
        - np.asarray(side["pose"]["elbow"], dtype=np.float64)
    )
    wflex, wroll = compute_wrist_angles(forearm, side["hand"])
    gripper = compute_pinch(side["hand"]["thumb_tip"], side["hand"]["index_tip"], calib)
    goal: JointGoal = {
        "shoulder_pan": pan,
        "shoulder_lift": lift,
        "elbow_flex": elbow,
        "wrist_flex": wflex,
        "wrist_roll": wroll,
        "gripper": gripper,
    }
    return apply_mirror(goal) if mirror else goal
