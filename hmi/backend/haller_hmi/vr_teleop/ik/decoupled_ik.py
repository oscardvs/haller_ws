"""Decoupled differential IK for the SO-101 — 3 joints position, 2 orientation.

Ported from the `vr-teleop-kit` reference solver, which splits a 6-DoF arm
into two 3-DoF sub-problems: joints 1-3 satisfy position, joints 4-6 satisfy
orientation, each as one damped-least-squares step per call, warm-started
from the caller's pose. The reason for the split is not efficiency — it is
that "a pure rotation of your hand keeps the robot's wrist still", which is
what an operator expects and what a coupled solve does not give you.

The SO-101 splits 3 + **2**, not 3 + 3, and that difference is the whole
adaptation:

  * Position is unchanged in spirit. `WRIST_ANCHOR` is invariant to both
    wrist joints, so joints 1-3 own it outright and the sub-problem is a
    square 3×3 — the same shape the reference solves, and we get the anchor
    from the geometry instead of having to place a site for it.

  * Orientation has two axes for a three-dimensional demand. The wrist
    Jacobian is 3×2, so the damped least-squares step tracks the reachable
    part of the demanded rotation and simply ignores the rest. That is not a
    failure mode to guard against, it is the standing 1-DoF deficit of any
    5-DoF arm: with position fixed, the gripper's yaw is decided by
    `shoulder_pan`, and no wrist can argue. What we do about it is *report*
    it — `last_orient_residual` is the fraction of the demand that is
    unreachable — so the operator feels a buzz telling them to move their
    hand instead of twisting harder at something that cannot move.

  * There is no wrist gimbal lock to damp. `wrist_roll`'s axis is Rx(θ₄)·ŷ
    and `wrist_flex`'s is x̂, so the two are perpendicular at every pose
    (`so101_kinematics._self_test` pins this). The reference's
    gimbal-proximity ramp and its near-antipodal park gate both exist for a
    3-axis wrist that can fold onto itself; neither is portable here, and
    shipping them as dead code would suggest the arm has a failure mode it
    does not have.

Two further deliberate deviations from the reference, both measured:

  * **Conditioning is read from σ_min, not |det J|.** On a 25 cm arm the
    determinant is small everywhere (peak 0.0035, median 0.0011), so it
    conflates "near singular" with "short lever arms" and a threshold that
    catches the first also damps most of the workspace. The smallest
    singular value is in m/rad and says the honest thing: how far the tool
    moves, in the worst direction, per radian. Median 0.045, best 0.082,
    collapsing toward 0 in the singular set.

  * **The posture bias is ramped with the singularity, not constant.** The
    position sub-problem is square, so a constant Tikhonov pull has no null
    space to hide in: it shows up directly as steady-state tracking error
    (a couple of mm of tool sag toward the rest pose). Its only real job is
    to pick an elbow branch where JᵀJ has collapsed, so it is scaled by the
    same ramp as the damping and is exactly zero in ordinary poses.

Angles are LeRobot degrees on the public surface — matching the arm handles,
the session, the guard and the recorder. Radians internally, because every
damping constant in the literature assumes them.
"""
from __future__ import annotations

import numpy as np

from ..core import quat
from .model import (
    ORIENTATION_JOINTS,
    POSE_JOINTS,
    POSITION_JOINTS,
    fk_frames,
    jacobian_position,
    jacobian_rotation,
    rest_pose_deg,
)

_DEG = np.pi / 180.0

#: Position error (m) at which the posture bias reaches full strength. Below
#: it the bias fades to zero so a stationary re-anchor takes no step at all —
#: see the gate in `solve`. 5 mm is comfortably under any motion the operator
#: would call deliberate and comfortably over the solver's own noise floor.
_POSTURE_BIAS_ERR_REF_M = 0.005


#: Per-joint step cap, degrees per solve. The wrist gets a looser cap than
#: the position joints for the same reason the reference stack gives it one:
#: at a shared cap the wrist feels held back during fine work while position
#: is fine. These are not the binding speed limit — the session's rate cap
#: and `ArmHandle.send_goal`'s `max_speed_deg_s` both sit downstream — they
#: exist so no single solve can ask for a lurch.
DEFAULT_MAX_DQ_DEG: dict[str, float] = {
    "shoulder_pan": 3.0,
    "shoulder_lift": 3.0,
    "elbow_flex": 3.0,
    "wrist_flex": 8.0,
    "wrist_roll": 8.0,
}


class SO101DecoupledIK:
    """One arm's differential IK. One damped step per `solve()`.

    Always returns a valid pose — never None, never an exception for a bad
    target. An unreachable demand produces the closest step the arm can
    take, with the residual left for the operator's own visual loop to
    close, exactly as the reference solver does. Freezing instead would be
    worse: an operator whose arm stops dead has no way to tell "unreachable"
    from "crashed".

    Tunables (all live-settable; the web panel writes them):
        lam_pos     Base DLS damping on the position sub-solve, m/rad —
                    same units as σ_min, so `lam_pos / σ` reads directly as
                    the attenuation at that pose.
        lam0        Extra damping amplitude near the position singularity.
                    λ² = lam_pos² + lam0²·ramp², ramp = max(0, 1 − σ/w0).
        w0          σ_min (m/rad) where the ramp starts. 0.02 keeps ordinary
                    working poses (0.04–0.08) undamped and catches the
                    straight-elbow and on-axis collapses.
        mu          Posture-bias stiffness toward `q_rest`, scaled by ramp².
        lam_rot     Base DLS damping on the 2-DoF orientation sub-solve.
                    The wrist Jacobian is orthonormal, so σ = 1 and this is
                    a plain relative attenuation.
        q_rest_deg  Posture the bias pulls toward near singularities.
        max_dq_deg  Per-joint step cap, degrees. The last safety layer
                    inside this module; the session's rate cap and the arm
                    handle's speed limit sit downstream of it as well.
    """

    def __init__(
        self,
        limits_deg: dict[str, tuple[float, float]],
        *,
        lam_pos: float = 0.010,
        lam0: float = 0.050,
        w0: float = 0.020,
        mu: float = 0.020,
        lam_rot: float = 0.05,
        q_rest_deg: dict[str, float] | None = None,
        max_dq_deg: dict[str, float] | None = None,
    ) -> None:
        self.limits_deg = dict(limits_deg)
        self.lam_pos = float(lam_pos)
        self.lam0 = float(lam0)
        self.w0 = float(w0)
        self.mu = float(mu)
        self.lam_rot = float(lam_rot)
        self.q_rest = rest_pose_deg(self.limits_deg, q_rest_deg)
        self.max_dq_deg = dict(max_dq_deg or DEFAULT_MAX_DQ_DEG)

        # Set by every solve(); consumed by the haptic mix and the HUD.
        #: Residual position-task error after the step, metres. Reads as
        #: "how far past its reach the operator is pushing".
        self.last_pos_err_norm: float = 0.0
        #: How hard this step pushed joints into their stops, degrees (L2 of
        #: the clamp). Distinct from the rate cap below it, which is a speed
        #: limit rather than an unreachability.
        self.last_limit_pressure: float = 0.0
        #: Position-singularity damping ramp, 0..1.
        self.last_singularity_proximity: float = 0.0
        #: Fraction of the demanded rotation the 2-DoF wrist cannot reach,
        #: 0..1. The honest analogue of the reference's gimbal proximity.
        self.last_orient_residual: float = 0.0
        #: Smallest singular value of the position Jacobian, m/rad.
        self.last_sigma_min: float = 0.0

    # ---- forward kinematics ---------------------------------------------

    def fk(self, joints_deg: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Tool pose for a joint dict: (position, orientation as wxyz)."""
        frames = fk_frames(joints_deg)
        return frames.tool_pos.copy(), quat.from_mat(frames.tool_R)

    def wrist_anchor(self, joints_deg: dict[str, float]) -> np.ndarray:
        """World position of the position task's anchor point.

        The pose mapper takes this as its rotation pivot, so that a pure
        hand twist swings the gripper about the wrist the way the operator's
        own hand does rather than spinning the tool in place and dragging
        joints 1-3 after it.
        """
        return fk_frames(joints_deg).wrist_pos.copy()

    # ---- the step -------------------------------------------------------

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        seed_deg: dict[str, float],
    ) -> dict[str, float]:
        """One decoupled damped step from `seed_deg` toward the target pose.

        Returns a full `POSE_JOINTS` dict in degrees, clamped to the arm's
        limits. Joints absent from the seed read as 0°.
        """
        q = np.array([float(seed_deg.get(j, 0.0)) for j in POSE_JOINTS])
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        R_target = quat.to_mat(target_quat_wxyz)

        # ---- 1. FK at the seed. Read the tool→anchor offset in the TOOL
        # frame, re-derived every tick rather than cached at some reference
        # wrist pose. That is what makes a re-anchor with target == current
        # produce exactly zero error: the same configuration is on both
        # sides of the subtraction.
        seed_pose = {j: float(v) for j, v in zip(POSE_JOINTS, q)}
        frames = fk_frames(seed_pose)
        anchor_pos = frames.wrist_pos
        anchor_in_tool = frames.tool_R.T @ (anchor_pos - frames.tool_pos)

        # ---- 2. Predict the orientation the wrist can actually reach, and
        # place the position target against THAT rather than against the
        # demand.
        #
        # The reference solver places its anchor with the target rotation:
        # `target_anchor = target_pos + R_target · offset`. On a 6-DoF arm
        # that is right — the demanded rotation is (almost always) reachable,
        # so pre-positioning the anchor for it is anticipation, not error.
        # On a 5-DoF arm it is a trap: a yaw demand the wrist can never take
        # would sit in that term forever and drag the TOOL off position by
        # up to |offset| ≈ 6 cm for as long as the operator holds their hand
        # over. Measured before this pre-solve went in: a 45° unreachable
        # yaw pulled the tool 52 mm off target and kept it there.
        #
        # So the wrist is solved first, against the seed pose, purely to ask
        # "how far over CAN we get?" — and the answer places the anchor. The
        # commanded wrist step is still the one taken in step 4, at the new
        # joints 1-3, so this pre-solve costs one extra FK and changes
        # nothing when the demand is reachable.
        dq_pre = self._wrist_step(frames, R_target)
        pre_pose = dict(seed_pose)
        for j, dv in zip(ORIENTATION_JOINTS, dq_pre / _DEG):
            pre_pose[j] = seed_pose[j] + float(dv)
        R_reachable = fk_frames(pre_pose).tool_R

        # ---- 3. Position sub-solve on joints 1-3, against the anchor.
        target_anchor = target_pos + R_reachable @ anchor_in_tool
        pos_err = target_anchor - anchor_pos
        self.last_pos_err_norm = float(np.linalg.norm(pos_err))

        J_pos = jacobian_position(frames, anchor_pos, POSITION_JOINTS)
        sigma_min = float(np.linalg.svd(J_pos, compute_uv=False)[-1])
        self.last_sigma_min = sigma_min
        ramp = max(0.0, 1.0 - sigma_min / max(self.w0, 1e-12))
        self.last_singularity_proximity = float(ramp)

        lam2 = self.lam_pos ** 2 + (self.lam0 ** 2) * (ramp ** 2)
        # Posture bias, gated twice. By the singularity ramp, because
        # picking an elbow branch is the only thing it is for and JᵀJ is
        # only rank-deficient there. And by the position error, because the
        # sub-problem is SQUARE — there is no null space for a posture term
        # to live in, so an ungated one shows up as motion even when the
        # target is exactly where the arm already is.
        #
        # That second gate is load-bearing, not tidiness. The session's
        # acquisition gate hands the arm over only once the commanded pose
        # matches the measured one, and the whole reason a VR handover is
        # near-instant here is that squeezing the grip anchors the target ON
        # the arm — zero error by construction. An ungated bias broke that:
        # measured 3° of drift per solve at the home pose (which sits near
        # the straight-elbow singularity, so the ramp is open there), i.e.
        # the arm creeping away from the operator during the countdown.
        err_gate = min(1.0, self.last_pos_err_norm / _POSTURE_BIAS_ERR_REF_M)
        mu2 = (self.mu ** 2) * (ramp ** 2) * err_gate
        q_rest_rad = self.q_rest[:3] * _DEG
        q_now_rad = q[:3] * _DEG
        A = J_pos.T @ J_pos + (lam2 + mu2) * np.eye(3)
        b = J_pos.T @ pos_err + mu2 * (q_rest_rad - q_now_rad)
        dq_pos_rad = np.linalg.solve(A, b)

        q_next = q.copy()
        q_next[:3] = q[:3] + dq_pos_rad / _DEG

        # ---- 4. Orientation sub-solve on joints 4-5, at the joints 1-3 the
        # arm is about to be at, so the wrist corrects the orientation it
        # will actually see rather than the one it is leaving.
        after_pos = {j: float(v) for j, v in zip(POSE_JOINTS, q_next)}
        frames_after = fk_frames(after_pos)
        dq_rot_rad = self._wrist_step(frames_after, R_target, record=True)
        q_next[3:] = q[3:] + dq_rot_rad / _DEG

        # ---- 5. Joint limits. Clamp, don't reject: the operator sees the
        # arm reach as far as its joints allow and feels the pressure.
        lo = np.array([self.limits_deg.get(j, (-360.0, 360.0))[0] for j in POSE_JOINTS])
        hi = np.array([self.limits_deg.get(j, (-360.0, 360.0))[1] for j in POSE_JOINTS])
        q_reachable = np.clip(q_next, lo, hi)
        self.last_limit_pressure = float(np.linalg.norm(q_next - q_reachable))

        # ---- 6. Per-joint step cap, then re-clamp (the cap can only shrink
        # a step, but a seed already outside its limits — a freshly
        # recalibrated arm, say — must not be allowed to stay there).
        cap = np.array([float(self.max_dq_deg.get(j, 180.0)) for j in POSE_JOINTS])
        dq = np.clip(q_reachable - q, -cap, cap)
        q_out = np.clip(q + dq, lo, hi)

        return {j: float(v) for j, v in zip(POSE_JOINTS, q_out)}

    def _wrist_step(self, frames, R_target: np.ndarray,
                    record: bool = False) -> np.ndarray:
        """Damped least-squares step for the two wrist joints, radians.

        Called twice per solve: once to predict what the wrist can reach (so
        the anchor can be placed against a reachable orientation), once for
        real at the post-position pose. Only the second call records the
        diagnostics, or the pre-solve's numbers would be what the HUD showed.
        """
        # e_rot is the world-frame angular displacement carrying the current
        # tool orientation onto the target — the same frame the rotational
        # Jacobian maps joint rates into.
        e_rot = quat.to_rotvec(quat.from_mat(R_target @ frames.tool_R.T))
        J_rot = jacobian_rotation(frames, ORIENTATION_JOINTS)   # 3x2
        A_rot = J_rot.T @ J_rot + (self.lam_rot ** 2) * np.eye(2)
        dq = np.linalg.solve(A_rot, J_rot.T @ e_rot)
        if record:
            e_norm = float(np.linalg.norm(e_rot))
            # Threshold in radians, well above the ~1e-8 float noise a
            # converged solve leaves behind: dividing that noise by itself
            # produced a saturated "unreachable" reading on a perfectly
            # tracked pose.
            if e_norm > 1e-6:
                residual = float(np.linalg.norm(e_rot - J_rot @ dq))
                self.last_orient_residual = float(min(1.0, residual / e_norm))
            else:
                self.last_orient_residual = 0.0
        return dq
