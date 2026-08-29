"""Quaternion helpers in the `wxyz` convention, on plain numpy.

The reference stack this layer is ported from leans on MuJoCo's `mju_*`
quaternion routines. Here they are hand-rolled instead, for one reason worth
stating: `core/` is the robot-agnostic layer. Its whole job is to be the part
that carries over to any arm, and a hard dependency on a physics engine to
multiply two quaternions would make it the part that carries over only where
that engine is installed. Everything below is numpy and `math`.

`wxyz` throughout, matching MJCF (and so the rest of this codebase). WebXR
speaks `xyzw`; `from_xyzw` is the only place that conversion is allowed to
happen, so a reversed quaternion has exactly one function to be wrong in.
"""
from __future__ import annotations

import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def from_xyzw(q_xyzw) -> np.ndarray:
    """WebXR's (x, y, z, w) → our (w, x, y, z)."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array([w, x, y, z])


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    return IDENTITY.copy() if n < 1e-12 else q / n


def conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def mul(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Hamilton product `qa · qb` — apply `qb` first, then `qa`."""
    w1, x1, y1, z1 = (float(v) for v in qa)
    w2, x2, y2, z2 = (float(v) for v in qb)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a quaternion."""
    return to_mat(q) @ np.asarray(v, dtype=float)


def to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in normalize(q))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def from_mat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → wxyz quaternion (Shepperd's branch selection).

    Branching on the largest diagonal term rather than always going through
    the trace: the trace form divides by `sqrt(1 + tr)`, which collapses at
    180° rotations — reachable here whenever the operator turns their hand
    all the way over.
    """
    R = np.asarray(R, dtype=float)
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s,
                         (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis · angle, rad), taking the SHORTEST way.

    `q` and `−q` are the same rotation; picking the hemisphere with `w ≥ 0`
    is what makes the returned angle ≤ π. Every consumer here treats the
    magnitude as "how far off are we", so the long way round would report a
    350° error for a 10° mistake.
    """
    q = normalize(q)
    if q[0] < 0.0:
        q = -q
    v = q[1:]
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(n, float(q[0])))
    return (v / n) * angle


def from_rotvec(v: np.ndarray) -> np.ndarray:
    """wxyz quaternion of a rotation vector (axis · angle, rad)."""
    v = np.asarray(v, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return IDENTITY.copy()
    axis = v / angle
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), s * axis[0], s * axis[1], s * axis[2]])


def power(q: np.ndarray, k: float) -> np.ndarray:
    """Keep the axis, scale the angle by `k`.

    `power(q, 1) == q`, `power(q, 0) == identity`. This is how a rotation
    gain is applied: at k > 1 a small wrist twist becomes a large tool twist,
    which is what an arm whose roll range exceeds a human wrist's needs.
    """
    q = normalize(q)
    w = float(q[0])
    v = np.asarray(q[1:], dtype=float)
    half = float(np.arctan2(float(np.linalg.norm(v)), w))
    if half < 1e-9:
        return IDENTITY.copy()
    axis = v / np.sin(half)
    new_half = k * half
    s = float(np.sin(new_half))
    return np.array([float(np.cos(new_half)), s * axis[0], s * axis[1], s * axis[2]])


def angle_between(qa: np.ndarray, qb: np.ndarray) -> float:
    """Shortest rotation angle between two orientations, radians."""
    return float(np.linalg.norm(to_rotvec(mul(qa, conj(qb)))))


def hemisphere_align(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return whichever of `±q` is on the same hemisphere as `reference`.

    Needed anywhere successive samples are differenced or blended: the WebXR
    runtime is free to flip the sign of a controller quaternion between
    frames, and an unaligned difference reads that free flip as a 360°
    rotation in one tick.
    """
    q = np.asarray(q, dtype=float)
    return q if float(np.dot(q, np.asarray(reference, dtype=float))) >= 0.0 else -q


def yaw_about_up(q_xyzw, up_axis: int = 1) -> float:
    """Heading of a WebXR pose about the world up axis, radians.

    WebXR `local-floor` is y-up, so the default reads a headset quaternion's
    yaw while ignoring pitch and roll — nodding must not steer the mapping.
    """
    q = from_xyzw(q_xyzw)
    w, x, y, z = (float(v) for v in q)
    if up_axis != 1:
        raise ValueError("only the WebXR y-up convention is implemented")
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def rot_y(angle: float) -> np.ndarray:
    """Rotation about +Y (WebXR's up), as a matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
