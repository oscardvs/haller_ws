"""VR position mode: the operator's hand drives the gripper, not the joints.

The angle-copying path (`vr_input.py`) treats the SO-101 as if it were a human
arm, and it is nearly the opposite of one: the shoulder pitch stops +10° below
horizontal while a human shoulder drops freely, and the elbow folds DOWNWARD
while a human elbow folds up. Copying human joint angles onto that kinematics
means a natural reach-down-and-grab either clamps against the pitch limit or
demands the operator bend their elbow backwards — which is exactly how the
first live session ended.

This mode is clutch-relative position tracking, the way practical VR teleop
rigs work:

  - Squeezing a grip ANCHORS: the arm's current pose and the hand's current
    pose become the shared origin. Nothing jumps — at the moment of the
    squeeze the target IS the arm's own pose, so the acquisition gate matches
    instantly and the countdown runs while the operator simply holds still.
  - While the grip is held, hand deltas (in the operator's own heading frame,
    fixed at anchor time) move the arm's wrist point through damped
    least-squares IK on the REAL kinematic chain (collision.fk_points), which
    is free to use the elbow branch the robot actually has. Controller pitch
    and roll steer the wrist joints relative to their anchored values; the
    trigger is the gripper, absolute.
  - Releasing freezes the arm where it is; the next squeeze re-anchors from
    wherever the hand is then. Big workspaces are covered by ratcheting:
    drag, release, reposition the hand, drag again.

Everything downstream is unchanged: this module only *produces the target*.
The session still owns the countdown, staleness, rate ramps, per-side
authority, the collision guard, and the recorder. Frames it emits carry
per-side ``joint_goal`` dicts (degrees; gripper in [0, 1], matching the
retargeter's contract) which ``ingest_frame`` uses in place of the landmark
retargeter. No BodyModel is involved anywhere — position mode needs no
limb-length calibration at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .collision import fk_points

#: The joints the wrist-point IK owns. Wrist pitch/roll ride the controller's
#: own orientation instead, and the gripper rides the trigger.
_IK_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
_POSE_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll")


def _wpr(joints: dict[str, float]) -> np.ndarray:
    """Wrist-point (Wrist_Pitch_Roll origin) in the arm's own base frame."""
    return fk_points((0.0, 0.0, 0.0), 0.0, joints)["Wrist_Pitch_Roll"]


def solve_wrist_point(
    seed: dict[str, float],
    target: np.ndarray,
    limits: dict[str, tuple[float, float]],
    iters: int = 24,
) -> dict[str, float]:
    """Damped least-squares IK: pan/lift/elbow so the wrist point hits target.

    Numeric on purpose — the chain has non-collinear link offsets, and running
    Gauss-Newton against the same fk_points the collision guard trusts means
    there is exactly one definition of the kinematics in this codebase. Joint
    limits clamp every iterate, so an unreachable target (too far, or through
    the floor of the joint space) converges to the closest reachable pose
    instead of diverging; the per-iteration step cap keeps a distant target
    from swinging the linearisation wildly. ~30 fk evaluations per solve, at
    30 Hz per driven side: microseconds of numpy.
    """
    joints = {j: float(seed.get(j, 0.0)) for j in _POSE_JOINTS}
    eps = 0.25          # finite-difference step, degrees
    max_step = 20.0     # per-iteration joint step cap, degrees
    for _ in range(iters):
        here = _wpr(joints)
        err = target - here
        if float(np.linalg.norm(err)) < 1e-4:
            break
        jac = np.zeros((3, 3))
        for k, name in enumerate(_IK_JOINTS):
            probe = dict(joints)
            probe[name] += eps
            jac[:, k] = (_wpr(probe) - here) / eps
        # Damping keeps the solve stable at singularities (straight elbow).
        # RELATIVE to the Jacobian's own scale: with links in metres and
        # joints in degrees, J^T J entries sit around 1e-5 — an absolute
        # lambda would either dominate them (and freeze the solve, the first
        # version of this function) or vanish. 1% of the mean diagonal keeps
        # the same conditioning at any unit choice.
        jtj = jac.T @ jac
        lam = 1e-2 * float(np.trace(jtj)) / 3.0 + 1e-12
        dq = np.linalg.solve(jtj + lam * np.eye(3), jac.T @ err)
        dq = np.clip(dq, -max_step, max_step)
        for k, name in enumerate(_IK_JOINTS):
            lo, hi = limits.get(name, (-180.0, 180.0))
            joints[name] = min(hi, max(lo, joints[name] + float(dq[k])))
    return joints


def _ray_pitch_roll_deg(orientation: list[float]) -> tuple[float, float]:
    """Controller pitch (up positive) and roll (right-hand-down positive),
    from the target-ray quaternion (x, y, z, w) in XR space (+Y up)."""
    x, y, z, w = (float(v) for v in orientation)
    # Forward = q * (0,0,-1); right = q * (1,0,0). Inlined rotation.
    fwd = (
        -2.0 * (x * z + w * y),
        -2.0 * (y * z - w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    )
    right = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    )
    pitch = math.degrees(math.atan2(fwd[1], math.hypot(fwd[0], fwd[2])))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, right[1]))))
    return pitch, roll


@dataclass
class _SideAnchor:
    hand: np.ndarray
    fwd: np.ndarray          # operator heading at anchor time, XR frame
    right: np.ndarray
    joints: dict[str, float]
    pitch_deg: float
    roll_deg: float
    last_solution: dict[str, float] = field(default_factory=dict)


class VRPoseMode:
    """One per websocket connection — the anchors are conversation state."""

    def __init__(self, session, arms) -> None:
        self._session = session
        self._arms = arms
        self._anchors: dict[str, _SideAnchor | None] = {"left": None, "right": None}

    # ---- helpers -----------------------------------------------------------

    def _limits(self, status: dict, side: str) -> dict[str, tuple[float, float]] | None:
        arm_id = status.get(f"{side}_arm")
        if not arm_id:
            return None
        try:
            return self._arms[arm_id].joint_limits_deg
        except KeyError:
            return None

    @staticmethod
    def _gripper_unit(committed: dict[str, float],
                      limits: dict[str, tuple[float, float]]) -> float:
        lo, hi = limits.get("gripper", (0.0, 1.0))
        if hi <= lo:
            return 1.0
        g = committed.get("gripper", lo)
        return max(0.0, min(1.0, (g - lo) / (hi - lo)))

    @staticmethod
    def _heading(head_orientation: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Yaw-only forward/right in XR space — the same convention
        vr_input.head_basis_xr uses: nodding must not steer the mapping."""
        x, y, z, w = (float(v) for v in head_orientation)
        fx = -2.0 * (x * z + w * y)
        fz = -(1.0 - 2.0 * (x * x + y * y))
        yaw = math.atan2(fx, -fz)
        fwd = np.array([math.sin(yaw), 0.0, -math.cos(yaw)])
        right = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
        return fwd, right

    # ---- conversion ----------------------------------------------------------

    def convert(self, frame: dict) -> dict:
        """Raw WebXR frame → KeypointFrame whose sides carry joint_goal."""
        status = self._session.status()
        frame_goal = status.get("goal_deg") or {}
        head = frame.get("head")
        heading = (self._heading(head["orientation"])
                   if isinstance(head, dict) else None)

        sides: dict[str, dict | None] = {}
        squeezes: dict[str, bool] = {}
        for side in ("left", "right"):
            raw = frame.get(side)
            squeeze = bool(isinstance(raw, dict) and raw.get("squeeze"))
            squeezes[side] = squeeze
            sides[side] = self._convert_side(
                side, raw, squeeze, heading, frame_goal.get(side) or {},
                self._limits(status, side),
            )

        dead_man = squeezes["left"] or squeezes["right"]
        return {
            "type": "keypoints",
            "ts_ms": int(frame.get("ts_ms", 0)),
            "clutch_source": "vr_grip",
            "dead_man": dead_man or bool(frame.get("dead_man", False)),
            "dead_man_sides": dict(squeezes),
            "jaw_open": None,
            # Joint goals are already in robot joint space; mirroring them
            # again would be applying handedness twice.
            "mirror_mode": "none",
            "left": sides["left"],
            "right": sides["right"],
        }

    def _convert_side(
        self,
        side: str,
        raw: dict | None,
        squeeze: bool,
        heading: tuple[np.ndarray, np.ndarray] | None,
        committed: dict[str, float],
        limits: dict[str, tuple[float, float]] | None,
    ) -> dict | None:
        if (not isinstance(raw, dict) or not raw.get("tracked", False)
                or heading is None or limits is None or not committed):
            # Untracked (or nothing to anchor to): drop the anchor and report
            # the side lost, exactly like the camera path does. Recovery goes
            # back through acquisition, which re-anchors on the next squeeze.
            self._anchors[side] = None
            return None

    # The open-grip target is wherever the arm already is: zero gate error,
    # so the moment the operator squeezes, matching is already done and the
    # countdown is the only wait. This is what "easy to lock into" means —
    # the anchor moves the target to the robot instead of moving the
    # operator to the robot.
        trigger = float(raw.get("trigger", 0.0))
        if not squeeze:
            self._anchors[side] = None
            goal = {j: float(committed[j]) for j in _POSE_JOINTS if j in committed}
            goal["gripper"] = self._gripper_unit(committed, limits)
            return {"joint_goal": goal, "confidence": 1.0}

        pos = np.asarray(raw["position"], dtype=float)
        pitch, roll = _ray_pitch_roll_deg(raw["orientation"])
        anchor = self._anchors[side]
        if anchor is None:
            fwd, right = heading
            anchor = _SideAnchor(
                hand=pos.copy(), fwd=fwd, right=right,
                joints={j: float(committed.get(j, 0.0)) for j in _POSE_JOINTS},
                pitch_deg=pitch, roll_deg=roll,
            )
            anchor.last_solution = dict(anchor.joints)
            self._anchors[side] = anchor

        # Hand delta in the operator's anchored heading frame → arm base
        # frame. Operator right → +x, forward → −y (the arm reaches −y at
        # zero pan), up → +z. Same mapping for both arms: the mounts are
        # identical and side correspondence is world-direction based, which
        # is what the mirror_mode=none smoke test pinned.
        d = pos - anchor.hand
        fwd_amt = float(np.dot(d, anchor.fwd))
        right_amt = float(np.dot(d, anchor.right))
        up_amt = float(d[1])
        delta_robot = np.array([right_amt, -fwd_amt, up_amt])

        target = _wpr(anchor.joints) + delta_robot
        solved = solve_wrist_point(anchor.last_solution, target, limits)
        anchor.last_solution = dict(solved)

        # Wrist joints track the controller's attitude RELATIVE to its
        # anchored attitude, so however the controller was held at the
        # squeeze reads as "keep the wrist where it is". Signs: positive
        # wrist_flex pitches the hand DOWN (tests/sim pin this against FK),
        # so controller pitch-up subtracts.
        wf_lo, wf_hi = limits.get("wrist_flex", (-180.0, 180.0))
        wr_lo, wr_hi = limits.get("wrist_roll", (-180.0, 180.0))
        goal = dict(solved)
        goal["wrist_flex"] = min(wf_hi, max(wf_lo,
            anchor.joints.get("wrist_flex", 0.0) - (pitch - anchor.pitch_deg)))
        goal["wrist_roll"] = min(wr_hi, max(wr_lo,
            anchor.joints.get("wrist_roll", 0.0) + (roll - anchor.roll_deg)))
        goal["gripper"] = max(0.0, min(1.0, 1.0 - trigger))
        return {"joint_goal": goal, "confidence": 1.0}
