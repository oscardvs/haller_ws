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
from typing import Literal, TypedDict

import numpy as np

# MediaPipe pose world coords are right-handed metres: +X right, +Y down, +Z
# *toward the camera*. We rebase: +X right, +Y up, +Z away from the camera
# (i.e. "into the room"). This rebase happens once at the boundary.

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
    """Signed angle from a to b around `axis` (right-hand rule), in degrees."""
    a_n = a / _safe_norm(a)
    b_n = b / _safe_norm(b)
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
