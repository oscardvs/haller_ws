"""The kit's T1-T11 pose-mapping self-tests, as a table two venvs can run.

`vr_teleop_kit.core.pose_mapping.__main__` prints eleven checks and asserts
almost none of them. Transcribed here as data plus a tiny interpreter so the
SAME byte-identical inputs go into the kit's mapper (under the kit's venv,
via `gen/gen_pose_mapping.py`) and into Haller's (under pytest), and the two
output streams can be compared number for number.

Constraints this file must keep to, because it is imported by both:

  * numpy only. No mujoco, no `haller_hmi`, no `vr_teleop_kit` — whichever
    of those is missing is exactly the one the other side has.
  * Orientations are stored as ROTATION VECTORS and converted here, not
    passed as quaternions. The two stacks have separate quaternion modules;
    converting locally means a divergence in the mapper cannot be masked (or
    manufactured) by a divergence in `from_rotvec`.
  * Every case reconstructs its own mapper. The kit's `main()` reuses `m`
    across T1/T2 and T5/T6, and across T8/T9 — that carried state is
    reproduced by keeping those steps inside one case, never by ordering
    between cases.
"""
from __future__ import annotations

import math

import numpy as np

# The kit's own fixture values, transcribed from `main()`.
EE_P = [0.50, 0.00, 0.42]
EE_R = [0.0, 0.0, 0.0]              # identity orientation, as a rotvec
CTRL_P = [0.0, 1.4, -0.3]
CTRL_R = [0.0, 0.0, 0.0]
NEW_EE_P = [0.55, 0.10, 0.42]       # T7's re-engage pose

#: T4's frame change: Quest +X must land on arm-base +Y.
R_Z90 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

#: T8/T9's reach limits, and the angle they clamp to. Named because three
#: cases and four assertions quote them.
ROT_REACH = 0.5
POS_REACH = 0.25


def quat_from_rotvec(v) -> np.ndarray:
    """wxyz quaternion of a rotation vector. Deliberately local — see module docstring."""
    v = np.asarray(v, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = math.sin(angle / 2.0)
    axis = v / angle
    return np.array([math.cos(angle / 2.0), s * axis[0], s * axis[1], s * axis[2]])


def _shift(p, d):
    return [p[0] + d[0], p[1] + d[1], p[2] + d[2]]


# ---- the case table ------------------------------------------------------
#
# Each case is (name, mapper_kwargs, ops). An op is a tuple:
#
#   ("engage",    ctrl_p, ctrl_rotvec, ee_p, ee_rotvec)
#   ("disengage",)
#   ("target",    ctrl_p, ctrl_rotvec)                  — no EE → absolute path
#   ("target_ee", ctrl_p, ctrl_rotvec, ee_p, ee_rotvec) — EE → reach limits live
#   ("target_follow", ctrl_p, ctrl_rotvec, ee_p)        — EE orientation is the
#                                                         PREVIOUS target's, i.e.
#                                                         an arm tracking perfectly
#
# Only `target*` ops append to the recorded output; `None` is recorded when
# the mapper is disengaged, which is itself T6's whole assertion.

_T8_SWEEP = [("target_ee", CTRL_P, [0.0, math.radians(i), 0.0], EE_P, EE_R)
             for i in range(1, 121)]
_T9_BACKOFF = [("target_ee", CTRL_P,
                [0.0, math.radians(120.0 - math.degrees(ROT_REACH)), 0.0],
                EE_P, EE_R)]
_T11_PATH = [math.radians(i) * np.array([0.5, 0.7, 0.2]) for i in range(1, 41)]

CASES: tuple[tuple[str, dict, tuple], ...] = (
    # T1 + T2 share one mapper in the kit: T2 asserts the delta is measured
    # from the ENGAGE pose, which only means anything after T1's no-op call.
    ("T1_T2_absolute_translation", {}, (
        ("engage", CTRL_P, CTRL_R, EE_P, EE_R),
        ("target", CTRL_P, CTRL_R),
        ("target", _shift(CTRL_P, [0.05, 0, 0]), CTRL_R),
    )),
    ("T3_linear_gain", {"scale": 0.5}, (
        ("engage", CTRL_P, CTRL_R, EE_P, EE_R),
        ("target", _shift(CTRL_P, [0.05, 0, 0]), CTRL_R),
    )),
    ("T4_frame_change", {"R": R_Z90}, (
        ("engage", CTRL_P, CTRL_R, EE_P, EE_R),
        ("target", _shift(CTRL_P, [0.05, 0, 0]), CTRL_R),
    )),
    ("T5_absolute_rotation", {}, (
        ("engage", CTRL_P, CTRL_R, EE_P, EE_R),
        ("target", CTRL_P, [0.0, math.pi / 6, 0.0]),
    )),
    ("T6_disengaged_returns_none", {}, (
        ("engage", CTRL_P, CTRL_R, EE_P, EE_R),
        ("target", CTRL_P, [0.0, math.pi / 6, 0.0]),
        ("disengage",),
        ("target", CTRL_P, CTRL_R),
    )),
    ("T7_reengage_moves_origin", {}, (
        ("engage", CTRL_P, CTRL_R, NEW_EE_P, EE_R),
        ("target", _shift(CTRL_P, [0.05, 0, 0]), CTRL_R),
    )),
    # T8 then T9 on the same mapper: T9's whole point is that the clutch has
    # ALREADY absorbed 120° − 28.6° of twist when the reversal arrives.
    ("T8_T9_rot_reach_limit",
     {"rot_reach_limit": ROT_REACH, "pos_reach_limit": POS_REACH},
     (("engage", CTRL_P, CTRL_R, EE_P, EE_R), *_T8_SWEEP, *_T9_BACKOFF)),
    ("T10_pos_reach_limit",
     {"rot_reach_limit": 1.0, "pos_reach_limit": POS_REACH}, (
         ("engage", CTRL_P, CTRL_R, EE_P, EE_R),
         ("target_ee", _shift(CTRL_P, [1.00, 0, 0]), CTRL_R, EE_P, EE_R),
         ("target_ee", _shift(CTRL_P, [0.95, 0, 0]), CTRL_R, EE_P, EE_R),
     )),
    # T11 compares two mappers. Split into two cases because the interpreter
    # drives one mapper at a time; the comparison is the test's job.
    ("T11_incremental", {"rot_reach_limit": 3.0, "pos_reach_limit": 10.0},
     (("engage", CTRL_P, CTRL_R, EE_P, EE_R),
      *[("target_follow", CTRL_P, v, EE_P) for v in _T11_PATH])),
    ("T11_absolute", {},
     (("engage", CTRL_P, CTRL_R, EE_P, EE_R),
      *[("target", CTRL_P, v) for v in _T11_PATH])),
)

CASE_NAMES = tuple(name for name, _, _ in CASES)


def run_case(mapper_cls, case) -> tuple[list, list]:
    """Drive one case through a `ClutchPoseMapper`-shaped class.

    Returns (positions, quats) with `None` in both wherever `target()`
    returned None. `mapper_cls` is the kit's class under the kit's venv and
    Haller's under pytest; nothing else about them is assumed.
    """
    _, kwargs, ops = case
    kwargs = dict(kwargs)
    if "R" in kwargs:
        kwargs["R"] = np.asarray(kwargs["R"], dtype=float)
    m = mapper_cls(**kwargs)
    positions: list = []
    quats: list = []
    last_quat = None
    for op in ops:
        kind = op[0]
        if kind == "engage":
            _, cp, cr, ep, er = op
            last_quat = quat_from_rotvec(er)
            m.engage(np.array(cp, float), quat_from_rotvec(cr),
                     np.array(ep, float), last_quat)
            continue
        if kind == "disengage":
            m.disengage()
            continue
        if kind == "target":
            _, cp, cr = op
            out = m.target(np.array(cp, float), quat_from_rotvec(cr))
        elif kind == "target_ee":
            _, cp, cr, ep, er = op
            out = m.target(np.array(cp, float), quat_from_rotvec(cr),
                           np.array(ep, float), quat_from_rotvec(er))
        elif kind == "target_follow":
            _, cp, cr, ep = op
            out = m.target(np.array(cp, float), quat_from_rotvec(cr),
                           np.array(ep, float), last_quat)
        else:
            raise ValueError(f"unknown op {kind!r}")
        if out is None:
            positions.append(None)
            quats.append(None)
        else:
            p, q = out
            positions.append(np.asarray(p, float).copy())
            quats.append(np.asarray(q, float).copy())
            last_quat = quats[-1]
    return positions, quats


def pack(positions: list, quats: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten one case's output to arrays an `.npz` can hold.

    A disengaged step has no pose, so `valid` carries the None-ness rather
    than a sentinel that could be mistaken for a number.
    """
    n = len(positions)
    valid = np.array([p is not None for p in positions], dtype=bool)
    P = np.zeros((n, 3))
    Q = np.zeros((n, 4))
    for i, (p, q) in enumerate(zip(positions, quats)):
        if p is not None:
            P[i] = p
            Q[i] = q
    return P, Q, valid


def quat_angle_deg(qa, qb) -> float:
    """Angle between two wxyz quaternions, degrees, sign-insensitive.

    Two constraints, both learned the hard way on this comparison:

      * q and −q are the same rotation, so the sign is taken out. A
        comparison that forgets this reports 180° where there is none.
      * `2·acos(dot)` cannot resolve below ~1.7e-6 deg — acos is flat at 1,
        so float noise in the last bit of the dot product comes back
        magnified to sqrt(eps). `atan2` of the relative quaternion's vector
        part against its scalar part is exact there, which is the only
        reason this file can assert agreement at 1e-6 deg.
    """
    a = np.asarray(qa, float)
    b = np.asarray(qb, float)
    if float(a @ b) < 0.0:
        b = -b
    rel = np.array([
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
        -a[0] * b[1] + a[1] * b[0] - a[2] * b[3] + a[3] * b[2],
        -a[0] * b[2] + a[1] * b[3] + a[2] * b[0] - a[3] * b[1],
        -a[0] * b[3] - a[1] * b[2] + a[2] * b[1] + a[3] * b[0],
    ])
    return float(math.degrees(
        2.0 * math.atan2(float(np.linalg.norm(rel[1:])), abs(float(rel[0])))))
