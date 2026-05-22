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
