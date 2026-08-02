"""Bimanual self-collision + workspace guard for the streaming teleop path.

Two SO-101 arms share one bench. Nothing upstream stops the operator from
commanding both hands into the same cubic decimetre of air — the retargeter is
per-side, the joint clamps are per-joint, and the step caps bound speed, not
destination. This module is the piece that knows where both arms are at once.

The model is deliberately coarse: each arm is four capsules (base column,
upper arm, forearm, hand) swept along an analytic FK of the vendored SO-101
MJCF. Coarse is the point — capsule distance is exact, cheap enough to run
inside the 60 Hz commit loop with room for a bisection search, and *sound* as
long as the capsules contain the meshes. `tests/sim/test_collision_sim.py`
checks both properties against MuJoCo: the FK against body kinematics, and
soundness by asserting that whenever MuJoCo finds an inter-arm contact, this
model's gap is already ≤ 0.

The guard filters *steps*, not poses: `filter_step(prev, want)` passes any
step that keeps clearance, passes any step that improves a bad situation
(escape must never be blocked), and otherwise scales the whole bimanual step
back along its own direction until it stops at the margin. Both arms are
scaled together — the commanded step is one 10-DOF motion, and scaling its
components independently could turn a safe diagonal into an unsafe slide.
The gripper never participates: opening or closing the jaw moves nothing that
this model tracks, and freezing the gripper because the *arms* are close would
cost the operator a grasp for no safety gain.

Angles are LeRobot degrees throughout, matching the rest of the HMI. The FK
interprets them as the MJCF joint angles in degrees — exactly the equivalence
`SimArmHandle` already relies on. On the real rig the calibrated zero is the
same physical pose only to within calibration quality; margins are chosen to
absorb small offsets, and the QUICKSTART's first-run check verifies the live
clearance readout reacts correctly before anything moves fast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import CollisionConfig

#: The five pose joints, in kinematic order. `gripper` is deliberately absent.
POSE_JOINTS: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
)


def _quat_to_mat(q: tuple[float, float, float, float]) -> np.ndarray:
    """(w, x, y, z) — MJCF's ordering, not WebXR's — to a rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle(axis: tuple[float, float, float], theta: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    half = theta / 2.0
    w = math.cos(half)
    xyz = ax * math.sin(half)
    return _quat_to_mat((w, float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _rx(a: float) -> np.ndarray:
    return _axis_angle((1.0, 0.0, 0.0), a)


def _ry(a: float) -> np.ndarray:
    return _axis_angle((0.0, 1.0, 0.0), a)


# Kinematic chain transcribed from sim/assets/so101/so_arm100.xml: each entry
# is (body name, body pos in parent frame, body rotation, joint axis, HMI
# joint name). tests/sim/test_collision_sim.py pins this against the MJCF
# itself, so a vendored-model update that moves a link fails a test here
# rather than silently skewing the guard.
_CHAIN: tuple[tuple[str, tuple[float, float, float], np.ndarray,
                    tuple[float, float, float], str], ...] = (
    ("Rotation_Pitch", (0.0, -0.0452, 0.0165),
     _quat_to_mat((0.707105, 0.707108, 0.0, 0.0)), (0.0, 1.0, 0.0),
     "shoulder_pan"),
    ("Upper_Arm", (0.0, 0.1025, 0.0306), _rx(1.57079), (1.0, 0.0, 0.0),
     "shoulder_lift"),
    ("Lower_Arm", (0.0, 0.11257, 0.028), _rx(-1.57079), (1.0, 0.0, 0.0),
     "elbow_flex"),
    ("Wrist_Pitch_Roll", (0.0, 0.0052, 0.1349), _rx(-1.57079),
     (1.0, 0.0, 0.0), "wrist_flex"),
    ("Fixed_Jaw", (0.0, -0.0601, 0.0), _ry(1.57079), (0.0, 1.0, 0.0),
     "wrist_roll"),
)

#: Fingertip in the Fixed_Jaw frame — just past the last jaw pad (y=-0.1014).
_TIP_LOCAL = np.array([0.010, -0.105, 0.0])

#: Capsule radii, metres. Sized to contain the STL meshes with a little slack;
#: the sim soundness test is what holds them honest. Grow a radius rather than
#: shrinking a margin if that test ever reports a violation.
_RADII: dict[str, float] = {
    "column": 0.050,   # Base block + Rotation_Pitch housing
    "upper": 0.035,
    "fore": 0.035,
    "hand": 0.045,     # Wrist_Pitch_Roll + both jaws
}

#: Inter-arm capsule pairs. column×column is static geometry — if the mounts
#: put the bases in contact the config is wrong, which is a startup problem,
#: not a per-tick one.
_INTER_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (a, b) for a in _RADII for b in _RADII if (a, b) != ("column", "column")
)

#: Same-arm pairs. Only the distal links can fold back onto the base column;
#: adjacent links share a joint and are always "in contact" by construction.
_SELF_PAIRS: tuple[tuple[str, str], ...] = (("hand", "column"),
                                            ("fore", "column"))


def fk_points(mount_pos: tuple[float, float, float], mount_yaw_deg: float,
              joints_deg: dict[str, float]) -> dict[str, np.ndarray]:
    """World-frame chain points for one arm. Missing joints read as 0°."""
    R = _axis_angle((0.0, 0.0, 1.0), math.radians(mount_yaw_deg))
    t = np.asarray(mount_pos, dtype=float)
    pts: dict[str, np.ndarray] = {"root": t.copy()}
    for body, pos, rot, axis, joint in _CHAIN:
        t = t + R @ np.asarray(pos, dtype=float)
        R = R @ rot
        pts[body] = t.copy()
        R = R @ _axis_angle(axis, math.radians(float(joints_deg.get(joint, 0.0))))
    pts["tip"] = t + R @ _TIP_LOCAL
    return pts


def _capsule_segments(pts: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "column": (pts["root"] + np.array([0.0, -0.0452, 0.0]), pts["Upper_Arm"]),
        "upper": (pts["Upper_Arm"], pts["Lower_Arm"]),
        "fore": (pts["Lower_Arm"], pts["Wrist_Pitch_Roll"]),
        "hand": (pts["Wrist_Pitch_Roll"], pts["tip"]),
    }


def _seg_seg_dist(p1: np.ndarray, q1: np.ndarray,
                  p2: np.ndarray, q2: np.ndarray) -> float:
    """Closest distance between segments [p1,q1] and [p2,q2] (Ericson §5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(p1 - p2))
    if a <= 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


@dataclass(frozen=True)
class Clearance:
    """The single worst constraint over every check, as slack in metres.

    `slack` ≥ 0 means every capsule pair is at least the margin apart and
    every height floor is respected; 0 is exactly at the limit. `worst` names
    the binding constraint (e.g. "left:hand|right:hand" or "right:tip_floor")
    so the operator sees *what* is about to touch, not just a number.
    """
    slack: float
    worst: str


@dataclass(frozen=True)
class GuardResult:
    poses: dict[str, dict[str, float]]
    alpha: float          # 1.0 = the wanted step went through untouched
    limited: bool
    clearance: Clearance


class CollisionGuard:
    """Stateless filter: every decision is a pure function of (prev, want)."""

    def __init__(self, cfg: CollisionConfig):
        self.cfg = cfg
        self._mounts = {
            arm_id: (tuple(m.pos), float(m.yaw_deg))
            for arm_id, m in cfg.mounts.items()
        }

    # ---- clearance -------------------------------------------------------

    def clearance(self, poses: dict[str, dict[str, float]]) -> Clearance:
        """Worst slack over inter-arm, self, and table checks.

        `poses` maps arm id → joints in degrees. Arm ids without a configured
        mount raise KeyError loudly: a guard silently ignoring an arm it was
        asked about is the exact fail-open this module exists to prevent.
        """
        pts = {
            arm_id: fk_points(*self._mounts[arm_id], joints)
            for arm_id, joints in poses.items()
        }
        caps = {arm_id: _capsule_segments(p) for arm_id, p in pts.items()}
        margin = self.cfg.margin_m
        worst = Clearance(slack=float("inf"), worst="none")

        def consider(slack: float, label: str) -> None:
            nonlocal worst
            if slack < worst.slack:
                worst = Clearance(slack=slack, worst=label)

        arm_ids = sorted(pts)
        for i, a in enumerate(arm_ids):
            for b in arm_ids[i + 1:]:
                for ca, cb in _INTER_PAIRS:
                    d = _seg_seg_dist(*caps[a][ca], *caps[b][cb])
                    consider(d - _RADII[ca] - _RADII[cb] - margin,
                             f"{a}:{ca}|{b}:{cb}")
        for arm_id in arm_ids:
            for ca, cb in _SELF_PAIRS:
                d = _seg_seg_dist(*caps[arm_id][ca], *caps[arm_id][cb])
                consider(d - _RADII[ca] - _RADII[cb] - self.cfg.self_margin_m,
                         f"{arm_id}:{ca}|{arm_id}:{cb}")
            if self.cfg.table_z_m is not None:
                z0 = self.cfg.table_z_m
                consider(float(pts[arm_id]["tip"][2]) - z0 - self.cfg.tip_min_m,
                         f"{arm_id}:tip_floor")
                consider(float(pts[arm_id]["Wrist_Pitch_Roll"][2]) - z0
                         - self.cfg.wrist_min_m, f"{arm_id}:wrist_floor")
                consider(float(pts[arm_id]["Lower_Arm"][2]) - z0
                         - self.cfg.elbow_min_m, f"{arm_id}:elbow_floor")
        return worst

    # ---- step filter -----------------------------------------------------

    def filter_step(self, prev: dict[str, dict[str, float]],
                    want: dict[str, dict[str, float]]) -> GuardResult:
        """Bound one commit-loop step. Never blocks escape.

        Three regimes, in order:
          1. The wanted pose clears everything → pass through.
          2. It doesn't, but it is no worse than where the arms already are →
             pass through. This is what lets an operator back *out* of the
             margin, slide along it, or recover from a pose that started bad
             (say, after the mounts were reconfigured under a live session).
          3. It is strictly worse → bisect the largest step fraction that
             keeps slack ≥ min(0, current slack), i.e. stop at the margin
             when coming from safety, and merely refuse to deteriorate when
             already inside it.
        """
        want_cl = self.clearance(want)
        if want_cl.slack >= 0.0:
            return GuardResult(poses=want, alpha=1.0, limited=False,
                               clearance=want_cl)
        prev_cl = self.clearance(prev)
        if want_cl.slack >= prev_cl.slack - 1e-9:
            return GuardResult(poses=want, alpha=1.0, limited=False,
                               clearance=want_cl)

        target = min(0.0, prev_cl.slack)

        def lerp(alpha: float) -> dict[str, dict[str, float]]:
            out: dict[str, dict[str, float]] = {}
            for arm_id, want_j in want.items():
                prev_j = prev.get(arm_id, {})
                pose: dict[str, float] = {}
                for joint, w in want_j.items():
                    if joint in POSE_JOINTS and joint in prev_j:
                        p = prev_j[joint]
                        pose[joint] = p + (w - p) * alpha
                    else:
                        # Gripper, and any joint with no previous value:
                        # passes through at the wanted value regardless of
                        # alpha — see the module docstring.
                        pose[joint] = w
                out[arm_id] = pose
            return out

        lo, hi = 0.0, 1.0  # slack(0) == prev slack ≥ target by construction
        for _ in range(10):
            mid = (lo + hi) / 2.0
            if self.clearance(lerp(mid)).slack >= target:
                lo = mid
            else:
                hi = mid
        poses = lerp(lo)
        return GuardResult(poses=poses, alpha=lo, limited=True,
                           clearance=self.clearance(poses))
