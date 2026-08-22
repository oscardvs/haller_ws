"""The SO-101 as the IK sees it: named points, rest posture, joint limits.

Plays the role the reference stack's `ik/model.py` plays — decorate the arm
model with the handful of named things the solver needs — with one structural
difference. That stack loads a URDF into MuJoCo and adds sites to it, because
its arm is a 6-DoF machine whose geometry is only written down in a URDF.
Ours is five revolute joints already transcribed, pinned against MuJoCo by
`tests/sim/test_collision_sim.py`, and shared with the collision guard in
`haller_hmi.so101_kinematics`. So the model here is that chain plus:

    TOOL          the frame the operator's hand commands (`Fixed_Jaw`).
    WRIST_ANCHOR  the position task's target point (`Wrist_Pitch_Roll`
                  origin) — the most distal point neither wrist joint can
                  move, and therefore the one that lets joints 1-3 own
                  position outright.

The reference stack has to *construct* its anchor by hand (a site 10 cm past
joint 4, offset to keep it off the joint-1 axis at folded poses). On this arm
the geometry hands it over: `wrist_flex` pivots exactly at that origin.

`DEFAULT_REST_DEG` deliberately is NOT the home pose. Home is all-zeros, and
all-zeros is close to the straight-elbow singularity (smallest singular value
of the position Jacobian ≈ 0.015 m/rad there, against ≈ 0.08 at the best-
conditioned pose). Biasing the solver toward a singular posture is the
opposite of what a posture bias is for, so the default sits near the
manipulability peak instead, on the elbow-positive branch the arm actually
works in.
"""
from __future__ import annotations

import numpy as np

from ...so101_kinematics import (
    ORIENTATION_JOINTS,
    POSE_JOINTS,
    POSITION_JOINTS,
    TOOL_BODY,
    WRIST_POINT,
    ChainFrames,
    fk_frames,
    jacobian_position,
    jacobian_rotation,
)

__all__ = [
    "ORIENTATION_JOINTS",
    "POSE_JOINTS",
    "POSITION_JOINTS",
    "TOOL_BODY",
    "WRIST_POINT",
    "ChainFrames",
    "DEFAULT_REST_DEG",
    "DEFAULT_LIMITS_DEG",
    "fk_frames",
    "jacobian_position",
    "jacobian_rotation",
    "clamp_to_limits",
]

#: Posture the solver's Tikhonov term biases toward near singularities.
#: (0°, −60°, +90°) sits close to the position Jacobian's best conditioning
#: and unambiguously on the elbow-positive branch, which is the one this arm
#: folds into (its elbow folds downward, unlike a human's).
DEFAULT_REST_DEG: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -60.0,
    "elbow_flex": 90.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}

#: Fallback joint ranges, degrees, from the vendored MJCF. Only used when the
#: caller has no live arm to ask. A REAL arm's calibrated range differs — it
#: comes from the LeRobot calibration file, not from this table — so every
#: solver takes its limits as a constructor argument and this is the last
#: resort, not the default path. Getting that backwards would let the IK
#: command a real arm past its own calibrated stops.
DEFAULT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-190.0, 10.0),
    "elbow_flex": (-10.0, 180.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
}


def clamp_to_limits(joints_deg: dict[str, float],
                    limits: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Clamp a pose into the arm's own range. Joints with no limit pass."""
    out: dict[str, float] = {}
    for joint, value in joints_deg.items():
        lo_hi = limits.get(joint)
        if lo_hi is None:
            out[joint] = float(value)
        else:
            lo, hi = lo_hi
            out[joint] = float(min(hi, max(lo, value)))
    return out


def rest_pose_deg(limits: dict[str, tuple[float, float]] | None = None,
                  override: dict[str, float] | None = None) -> np.ndarray:
    """Rest posture as a 5-vector in `POSE_JOINTS` order, clamped to limits.

    Clamped rather than trusted: an arm whose calibration puts `elbow_flex`
    below +90° would otherwise get a bias pulling permanently into its own
    stop, which reads to the operator as the arm fighting them.
    """
    pose = {**DEFAULT_REST_DEG, **(override or {})}
    if limits:
        pose = clamp_to_limits(pose, limits)
    return np.array([pose[j] for j in POSE_JOINTS], dtype=float)
