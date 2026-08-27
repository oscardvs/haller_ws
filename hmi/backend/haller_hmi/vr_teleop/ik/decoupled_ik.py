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
    gimbal-proximity ramp exists for a 3-axis wrist that can fold onto
    itself and is not portable here; shipping it as dead code would suggest
    the arm has a failure mode it does not have.

  * The reference's near-antipodal park gate IS ported, because it guards a
    different failure mode. It is a property of the QUATERNION ERROR, not of
    the wrist's DoF count: `e_rot` is read off a quaternion, so as the error
    angle approaches π its shortest-way direction is unstable — an
    arbitrarily small change in the demanded orientation flips it to the
    opposite side, and chasing that is the 180° come-around where the wrist
    slams through half a revolution. Past `rot_err_hold` the wrist takes no
    step at all.

    Ported with one addition the reference does not need: the park EXPIRES
    after `PARK_MAX_SOLVES`, and only a run of that many solves back inside
    the hold refunds it. A parked wrist cannot reduce its own orientation
    error, so an UNBOUNDED park is a fixed point — it latched for a whole
    5000-solve run (83 s at 60 Hz) on a plain over-reach with an identity
    orientation demand, wrist pinned on its stop, `last_orient_residual` at
    1.00 and the haptic alarm saturated throughout. The expiry makes the
    park a DEBOUNCE: a demand still outside the hold a second later is not
    quaternion jitter, it is what the operator is asking for, so the arm
    goes there at its capped rate instead of refusing forever. See
    `_wrist_step`.

    What normally keeps the demand from ever reaching the antipode is the
    orientation reach limit. `vr_teleop.config.BOUNDS` permits
    `rot_reach_limit = 0.0` and `config.solo-raw.yaml` sets exactly that, so
    on the tracing config this gate is the only backstop left — and that is
    also the only config in which the latch above is reachable at all, i.e.
    the diagnostic config, where a stuck wrist reads as the fault under
    diagnosis.

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


#: Limit pressure, DEGREES, reported while the antipode gate holds the wrist.
#: A SENTINEL, not a measurement of joint-stop pressure: a parked wrist takes
#: no step, so the joint clamp measures exactly zero at the one moment the
#: wrist has stopped obeying the operator. Its job is to saturate the haptic
#: mix — `teleop._update_haptic` runs limit pressure through
#: `_gate(p, *_LIMIT_PRESSURE_GATE)`, currently (0.5, 4.0) deg, so anything
#: from 4.0 up is a full buzz. The value is the reference stack's 0.35
#: correctly converted: its pressure is in RADIANS, ours in degrees, and
#: 0.35 rad = 20.05 deg. 5x the ceiling is deliberate margin — it keeps
#: saturating if someone raises `_LIMIT_PRESSURE_GATE`.
PARK_LIMIT_PRESSURE_DEG = 20.05

#: Budget, in solves, that the antipode gate may spend holding the wrist on
#: one excursion past `rot_err_hold`, and equally the run of solves back
#: inside the hold that refunds it. 60 at the session's 60 Hz is 1.0 s.
#:
#: Sized against what it is refusing. The come-around it guards is a ONE-TIME
#: commitment, not an oscillation: measured ungated against a near-antipodal
#: demand jittered by 2°, the wrist travels 160° with ZERO direction
#: reversals and settles, in ~20 solves at the 8 deg default cap and ~6 at
#: config.solo-raw's 30. So the only thing a longer hold buys is outlasting a
#: transient — a dropped tracking frame, a filter spike, a re-anchor — and 60
#: solves is an order of magnitude more than any of those. Past it the demand
#: is being held rather than glitched, and a wrist that refuses a held demand
#: for longer than an operator's own "is it broken?" reflex reads as a crash,
#: which is the one thing this solver must never look like.
PARK_MAX_SOLVES = 60


class SO101DecoupledIK:
    """One arm's differential IK. One damped step per `solve()`.

    Always returns a valid pose — never None, never an exception for a bad
    target. An unreachable demand produces the closest step the arm can
    take, with the residual left for the operator's own visual loop to
    close, exactly as the reference solver does. Freezing instead would be
    worse: an operator whose arm stops dead has no way to tell "unreachable"
    from "crashed".

    The near-antipodal park gate is the ONE deliberate exception to that,
    and it is bounded on three sides so it cannot become the freeze the
    paragraph above rejects. It holds the two WRIST joints only — joints 1-3
    keep tracking position throughout. It holds for at most
    `PARK_MAX_SOLVES` per excursion. And while it holds it reports
    `PARK_LIMIT_PRESSURE_DEG` of limit pressure, which saturates the haptic
    mix, so the operator is told by a full buzz instead of being left to
    infer a freeze from an arm that went quiet.

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
        rot_err_hold
                    Park the wrist above this orientation error angle, rad —
                    the antipode gate of the module docstring, for at most
                    `PARK_MAX_SOLVES` solves per excursion past it. Above π
                    it can never fire, so anything past 3.15 disables it.
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
        rot_err_hold: float = 2.2,
        q_rest_deg: dict[str, float] | None = None,
        max_dq_deg: dict[str, float] | None = None,
    ) -> None:
        self.limits_deg = dict(limits_deg)
        self.lam_pos = float(lam_pos)
        self.lam0 = float(lam0)
        self.w0 = float(w0)
        self.mu = float(mu)
        self.lam_rot = float(lam_rot)
        self.rot_err_hold = float(rot_err_hold)
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
        #: Reads 1.0 while the wrist is parked, which is the truth: none of
        #: the demand was taken.
        self.last_orient_residual: float = 0.0
        #: Whether the antipode gate held the wrist still on this step. Read
        #: by `solve` itself to saturate the limit pressure below. NOT on the
        #: `ik_state` wire, and `last_orient_residual` is not a stand-in for
        #: it — the client's cue fires on a RISING crossing of 0.4 with
        #: hysteresis, and on a 5-DoF arm the residual is routinely already
        #: past 0.4 while driving, so entering a park often fires nothing.
        #: The saturated limit pressure below is what the operator actually
        #: feels; a dedicated wire field is the honest fix.
        self.last_wrist_parked: bool = False
        #: The antipode gate's budget, 0..`PARK_MAX_SOLVES`, spent one per
        #: solve outside `rot_err_hold`; the gate holds the wrist only while
        #: it is unspent. Refunded in full, and only, after `_park_inside`
        #: consecutive solves back inside the hold.
        self._park_spent: int = 0
        self._park_inside: int = 0
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
        # The predicted step is CLAMPED to the wrist's own limits, because
        # the question is "how far over can we get" and a joint stop is part
        # of the answer. Unclamped it is not: measured over a sweep of
        # over-reach targets the raw step ran past a stop on 64,321 solves,
        # by up to 78°, and `R_reachable` was then an orientation no wrist
        # of this arm can hold — placing the position anchor where the tool
        # will never be. That fantasy moved the position joints far enough
        # that the real solve's error angle read 1.74 rad on a tick where
        # this one read 3.11: the two halves of the antipode gate disagreed
        # about the same tick and the gate oscillated on a 120-solve period.
        dq_pre = self._wrist_step(frames, R_target)
        pre_pose = dict(seed_pose)
        for j, dv in zip(ORIENTATION_JOINTS, dq_pre / _DEG):
            j_lo, j_hi = self.limits_deg.get(j, (-360.0, 360.0))
            pre_pose[j] = min(j_hi, max(j_lo, seed_pose[j] + float(dv)))
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

        lo = np.array([self.limits_deg.get(j, (-360.0, 360.0))[0] for j in POSE_JOINTS])
        hi = np.array([self.limits_deg.get(j, (-360.0, 360.0))[1] for j in POSE_JOINTS])
        q_next = q.copy()
        # Clamped here rather than at step 5 alone, so that "the joints 1-3
        # the arm is about to be at" below is true. Past a stop the raw DLS
        # step is a pose the arm cannot hold, and solving the wrist against
        # it is solving for an orientation that will never exist — the whole
        # error the pre-solve of step 2 was added to remove. Clipping early
        # cannot change the returned position joints: step 5 clips to the
        # same bounds regardless.
        q_next[:3] = np.clip(q[:3] + dq_pos_rad / _DEG, lo[:3], hi[:3])

        # ---- 4. Orientation sub-solve on joints 4-5, at the joints 1-3 the
        # arm is about to be at, so the wrist corrects the orientation it
        # will actually see rather than the one it is leaving.
        after_pos = {j: float(v) for j, v in zip(POSE_JOINTS, q_next)}
        frames_after = fk_frames(after_pos)
        dq_rot_rad = self._wrist_step(frames_after, R_target, record=True)
        q_next[3:] = q[3:] + dq_rot_rad / _DEG

        # ---- 5. Joint limits. Clamp, don't reject: the operator sees the
        # arm reach as far as its joints allow and feels the pressure. Only
        # the wrist can still be outside them here; joints 1-3 were clipped
        # at step 3 so that step 4 saw a pose the arm can hold.
        q_reachable = np.clip(q_next, lo, hi)
        self.last_limit_pressure = float(np.linalg.norm(q_next - q_reachable))
        if self.last_wrist_parked:
            # A parked wrist takes no step, so the clamp above measures
            # nothing and the pressure would read ZERO at the exact moment
            # the wrist stopped obeying the operator — HUD and haptics both
            # saying everything is fine. Saturate it instead; this is the
            # operator-feedback half of the gate, not a metric detail.
            #
            # UNITS: the sentinel is in DEGREES because this field is —
            # see `PARK_LIMIT_PRESSURE_DEG`, which carries the conversion
            # and the dependency on `teleop._LIMIT_PRESSURE_GATE`.
            self.last_limit_pressure = max(self.last_limit_pressure,
                                           PARK_LIMIT_PRESSURE_DEG)

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
        e_norm = float(np.linalg.norm(e_rot))
        J_rot = jacobian_rotation(frames, ORIENTATION_JOINTS)   # 3x2

        # Antipode gate — see the module docstring. Tested BEFORE the damped
        # solve, because past the gate it is the DIRECTION of e_rot that is
        # untrustworthy and no amount of damping fixes a step aimed the wrong
        # way round. Both calls are gated: a pre-solve that stepped where the
        # real solve parks would place the position anchor against an
        # orientation the wrist is never going to take. Worth 3.0 deg on
        # shoulder_lift and -3.0 on elbow_flex in a SINGLE solve if ungated,
        # both saturating their step cap, so it is not a tidiness detail.
        #
        # The park is a DEBOUNCE, not a veto, and `_park_spent` is what makes
        # it one. A parked wrist takes no step, so it cannot reduce its own
        # orientation error: an unbounded park is a FIXED POINT, and on any
        # demand that stays outside the hold it never releases — measured
        # latching for a whole 5000-solve run on a plain over-reach. Past the
        # budget the wrist is let go and takes the come-around at its capped
        # rate.
        #
        # Re-arming needs a SUSTAINED return, not a dip, because the park
        # makes its own limit cycle otherwise: a held wrist lets the error
        # grow back over the hold, one step drops it under, and a gate that
        # re-arms on that alternates every solve. Measured on an
        # identity-orientation over-reach — re-arm on any dip parked 1164 of
        # 1200 solves in 13 full-budget bursts; re-arm on a dip lasting one
        # solve parked 618 in 559 single-solve flickers, half-rate wrist
        # under a permanent half-strength buzz. Neither is an alarm worth
        # having.
        over = e_norm > self.rot_err_hold
        parked = over and self._park_spent < PARK_MAX_SOLVES
        if parked:
            dq = np.zeros(2)
        else:
            A_rot = J_rot.T @ J_rot + (self.lam_rot ** 2) * np.eye(2)
            dq = np.linalg.solve(A_rot, J_rot.T @ e_rot)

        if record:
            # Moved by the recording call only, so both calls within one
            # solve read the same budget and cannot disagree about the gate.
            if over:
                self._park_spent = min(PARK_MAX_SOLVES, self._park_spent + 1)
                self._park_inside = 0
            else:
                self._park_inside += 1
                if self._park_inside >= PARK_MAX_SOLVES:
                    self._park_spent = 0
            self.last_wrist_parked = parked
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
