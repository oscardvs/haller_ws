"""SO-101 forward kinematics and geometric Jacobians — the one definition.

Two consumers need to agree about where this arm's links are: the collision
guard (`collision.py`, which sweeps capsules along the chain) and the VR
teleop IK (`vr_teleop/ik/`, which steps joints toward a Cartesian target).
They used to be able to disagree, because the guard owned the chain and the
IK reached into it through a private name. This module is that chain, lifted
out and given the extra read-outs the IK needs — full link *frames* and
analytic Jacobians, not just points.

The chain is transcribed from `sim/assets/so101/so_arm100.xml`;
`tests/sim/test_collision_sim.py` pins it against MuJoCo's own body
kinematics, so a vendored-model update that moves a link fails a test rather
than silently skewing either consumer.

Angles are LeRobot degrees on the public surface, matching the rest of the
HMI. Jacobians are per-radian (the SI convention every damping constant in
the literature assumes); `vr_teleop.ik` converts at its own boundary.

Frame conventions, measured (see `_self_test`):

    base:       z up, the arm reaching along −y at the all-zero pose
    shoulder_pan   axis +z          — swings the whole arm about the mount
    shoulder_lift  axis +x  ┐
    elbow_flex     axis +x  ├ three parallel pitch axes: the arm is planar
    wrist_flex     axis +x  ┘ within the plane shoulder_pan selects
    wrist_roll     axis +y          — along the forearm, i.e. tool roll

That structure is why this arm splits 3 + 2 rather than the 3 + 3 a
spherical-wrist 6-DoF arm splits into: `WRIST_POINT` (the Wrist_Pitch_Roll
origin) is invariant to *both* wrist joints, so joints 1-3 own position
outright, and joints 4-5 own whatever orientation two axes can reach. The
two wrist axes are mutually perpendicular at every wrist_flex angle
(wrist_roll's axis is Rx(θ₄)·ŷ, always ⟂ x̂), so unlike a 3-axis wrist this
one has no internal gimbal lock — only the standing 1-DoF orientation
deficit every 5-DoF arm has.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The five pose joints, in kinematic order. `gripper` is deliberately absent
#: from every list here: it moves jaws, not links, and neither the guard nor
#: the IK models it.
POSE_JOINTS: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
)

#: Joints the position task owns — everything upstream of `WRIST_POINT`.
POSITION_JOINTS: tuple[str, ...] = POSE_JOINTS[:3]

#: Joints the orientation task owns. Two, not three: see the module docstring.
ORIENTATION_JOINTS: tuple[str, ...] = POSE_JOINTS[3:]

#: The body whose origin the position task targets. Chosen because it is the
#: most distal point that neither wrist joint can move — the direct analogue
#: of the `j4_anchor` site a 6-DoF decoupled solver places by hand. On this
#: arm the geometry hands it to us: `wrist_flex` rotates *about* this origin
#: and `wrist_roll` lives further out still.
WRIST_POINT: str = "Wrist_Pitch_Roll"

#: The body whose frame is the tool frame. Its origin sits at the wrist_roll
#: pivot; `TIP_LOCAL` carries on to the fingertip.
TOOL_BODY: str = "Fixed_Jaw"


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


# Each entry is (body name, body pos in parent frame, body rotation, joint
# axis in the body frame, HMI joint name).
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


def fk_points(mount_pos: tuple[float, float, float], mount_yaw_deg: float,
              joints_deg: dict[str, float]) -> dict[str, np.ndarray]:
    """World-frame chain points for one arm. Missing joints read as 0°.

    Kept byte-for-byte as the collision guard has always had it: this is the
    function `tests/sim/test_collision_sim.py` pins against MuJoCo, and the
    guard's soundness argument rests on it.
    """
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


@dataclass(frozen=True)
class ChainFrames:
    """One arm's kinematics at one pose — everything an IK step needs.

    `points` is exactly what `fk_points` returns, so a caller holding a
    `ChainFrames` never has to run FK twice to also get the capsule
    geometry. The rest is what points alone cannot give: the tool's
    orientation, and the world origin/axis of every joint, which is all a
    geometric Jacobian is made of.
    """

    points: dict[str, np.ndarray]
    #: World rotation of the tool body's frame.
    tool_R: np.ndarray
    #: Per-joint world-frame pivot and unit axis, keyed by HMI joint name.
    joint_origin: dict[str, np.ndarray]
    joint_axis: dict[str, np.ndarray]

    @property
    def tool_pos(self) -> np.ndarray:
        """World position of the tool body's origin (the wrist_roll pivot)."""
        return self.points[TOOL_BODY]

    @property
    def tip_pos(self) -> np.ndarray:
        """World position of the fingertip."""
        return self.points["tip"]

    @property
    def wrist_pos(self) -> np.ndarray:
        """World position of the wrist-invariant position-task anchor."""
        return self.points[WRIST_POINT]


def fk_frames(joints_deg: dict[str, float],
              mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
              mount_yaw_deg: float = 0.0) -> ChainFrames:
    """Full FK: chain points, the tool frame's rotation, and joint axes.

    Same traversal as `fk_points` — deliberately, so the two can never drift
    — with the intermediate rotation and each joint's world axis retained
    instead of discarded.
    """
    R = _axis_angle((0.0, 0.0, 1.0), math.radians(mount_yaw_deg))
    t = np.asarray(mount_pos, dtype=float)
    pts: dict[str, np.ndarray] = {"root": t.copy()}
    origins: dict[str, np.ndarray] = {}
    axes: dict[str, np.ndarray] = {}
    for body, pos, rot, axis, joint in _CHAIN:
        t = t + R @ np.asarray(pos, dtype=float)
        R = R @ rot
        pts[body] = t.copy()
        # The joint pivots at this body's origin, about `axis` expressed in
        # the frame as it stands BEFORE the joint rotation is applied —
        # which is exactly the `R` we hold right here.
        origins[joint] = t.copy()
        axes[joint] = R @ np.asarray(axis, dtype=float)
        R = R @ _axis_angle(axis, math.radians(float(joints_deg.get(joint, 0.0))))
    tool_R = R.copy()
    pts["tip"] = t + R @ _TIP_LOCAL
    return ChainFrames(points=pts, tool_R=tool_R,
                       joint_origin=origins, joint_axis=axes)


def jacobian_position(frames: ChainFrames, point: np.ndarray,
                      joints: tuple[str, ...]) -> np.ndarray:
    """3×N linear-velocity Jacobian of `point` w.r.t. `joints`, per radian.

    Analytic, not finite-differenced: for a revolute joint with unit world
    axis ẑ pivoting at o, a point p downstream moves at ẑ × (p − o). Exact
    to machine precision and one cross product per joint, where the finite
    difference this replaced cost a full FK per joint and carried a step-size
    tuning parameter that had to be re-justified every time the units moved.

    Joints the point does not depend on must simply not be passed; this
    function does not check reachability order.
    """
    p = np.asarray(point, dtype=float)
    return np.column_stack([
        np.cross(frames.joint_axis[j], p - frames.joint_origin[j])
        for j in joints
    ])


def jacobian_rotation(frames: ChainFrames,
                      joints: tuple[str, ...]) -> np.ndarray:
    """3×N angular-velocity Jacobian w.r.t. `joints`, per radian.

    For a revolute joint the angular contribution is its world axis, full
    stop — independent of which downstream frame you are asking about.
    """
    return np.column_stack([frames.joint_axis[j] for j in joints])


# ---- self-test -----------------------------------------------------------

def _self_test() -> None:
    """Analytic Jacobians vs. central differences on `fk_frames`, plus the
    structural claims the module docstring makes."""
    rng = np.random.default_rng(0)
    worst_pos = 0.0
    worst_rot = 0.0
    for _ in range(50):
        q = {j: float(rng.uniform(-70, 70)) for j in POSE_JOINTS}
        f = fk_frames(q)

        # Position Jacobian of the wrist anchor w.r.t. joints 1-3.
        j_an = jacobian_position(f, f.wrist_pos, POSITION_JOINTS)
        for k, name in enumerate(POSITION_JOINTS):
            h = 1e-5
            qp = {**q, name: q[name] + math.degrees(h)}
            qm = {**q, name: q[name] - math.degrees(h)}
            num = (fk_frames(qp).wrist_pos - fk_frames(qm).wrist_pos) / (2 * h)
            worst_pos = max(worst_pos, float(np.max(np.abs(num - j_an[:, k]))))

        # Rotation Jacobian of the tool w.r.t. the wrist joints, read back
        # through the rotation-vector of the frame difference.
        j_rot = jacobian_rotation(f, ORIENTATION_JOINTS)
        for k, name in enumerate(ORIENTATION_JOINTS):
            h = 1e-5
            qp = {**q, name: q[name] + math.degrees(h)}
            qm = {**q, name: q[name] - math.degrees(h)}
            dR = fk_frames(qp).tool_R @ fk_frames(qm).tool_R.T
            w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                          dR[1, 0] - dR[0, 1]]) / 2.0
            worst_rot = max(worst_rot, float(np.max(np.abs(w / (2 * h) - j_rot[:, k]))))
    print(f"position Jacobian  max abs err: {worst_pos:.2e}  "
          f"[{'ok' if worst_pos < 1e-5 else 'FAIL'}]")
    print(f"rotation Jacobian  max abs err: {worst_rot:.2e}  "
          f"[{'ok' if worst_rot < 1e-5 else 'FAIL'}]")

    # WRIST_POINT is invariant to both wrist joints.
    base = fk_frames({}).wrist_pos
    moved = max(
        float(np.linalg.norm(fk_frames({j: a}).wrist_pos - base))
        for j in ORIENTATION_JOINTS for a in (-80.0, 80.0)
    )
    print(f"wrist anchor invariance: max drift {moved:.2e} m  "
          f"[{'ok' if moved < 1e-12 else 'FAIL'}]")

    # The two wrist axes never align — no gimbal lock in this wrist.
    worst_align = 0.0
    for a in np.linspace(-90, 90, 37):
        f = fk_frames({"wrist_flex": float(a)})
        c = abs(float(np.dot(f.joint_axis["wrist_flex"], f.joint_axis["wrist_roll"])))
        worst_align = max(worst_align, c)
    print(f"wrist axis orthogonality: max |cos| {worst_align:.2e}  "
          f"[{'ok' if worst_align < 1e-6 else 'FAIL'}]")


if __name__ == "__main__":
    _self_test()
