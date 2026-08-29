# Vendored byte-faithfully from vr-teleop-kit v0.1.0 (Apache-2.0),
# origin: src/vr_teleop_kit/ik/so101_ik.py. Only the script-mode import
# fallback was rewritten for this package layout. Do not edit here.
"""Decoupled IK for the SO-101 arm — position/wrist joint decoupling.

The SO-101 is a 5-DoF arm: shoulder_pan (yaw about the base vertical),
then shoulder_lift / elbow_flex / wrist_flex (three parallel pitch axes),
then wrist_roll (roll about the gripper axis). Joints 1-3 satisfy
position of a wrist-invariant anchor; joints 4-5 satisfy orientation.
Both sub-problems are one damped-least-squares step per call,
warm-started from the caller's qpos.

With only two wrist joints the orientation task is underactuated: the
reachable orientations at a given position form a 2-parameter family
(pitch × roll), and the gripper's yaw is dictated by shoulder_pan. The
3×2 DLS step projects the demanded rotation onto the reachable subspace;
the unreachable yaw component simply stays as residual error. That is
intentional and mirrors the DK1 solver's handling of its 6.2 cm wrist
non-sphericity: the operator's visual feedback loop closes the gap, and
the mapper's incremental rot reach limit keeps the residual bounded.

Unlike a 3-DoF spherical wrist there is no gimbal lock here — the
wrist_flex and wrist_roll axes are perpendicular by construction, so the
wrist manipulability w = sqrt(det(JᵀJ)) stays ≈ 1 everywhere. The
adaptive-damping ramp is kept anyway (same recipe as the arm) as a
guard, but it is essentially inert.

Always returns a valid qpos5 (never None). Boundary cases are handled
in-line so the arm degrades gracefully instead of freezing:

  1. Near workspace boundary — manipulability-adaptive damping on the
     position sub-solve; the step smoothly shrinks at the singularity.
  2. Joint limit violation — elementwise clamp into URDF limits, with
     the residual EE error left for the operator's visual loop.
  3. Near-antipodal orientation demand (error angle past rot_err_hold) —
     the shortest-way error direction is unstable there, so the wrist
     parks and reports saturating limit pressure.

A `max_dq_per_joint` cap is applied at the end as a per-joint velocity
bound — the operator-safety layer against any residual fast motion.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

try:
    from .so101_model import ARM_JOINT_NAMES, DEFAULT_Q_REST, build_so101_model
except ImportError:  # allow running this file directly as a script
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from haller_hmi.vr_teleop.kit.so101_model import (
        ARM_JOINT_NAMES,
        DEFAULT_Q_REST,
        build_so101_model,
    )

ARM_DOFS = 5


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
    return out.reshape(3, 3)


class SO101IKSolver:
    """Decoupled IK for the SO-101.

    Holds a mujoco model with the tool0 / wrist_anchor sites, plus cached
    joint-index lookups (resolved by name, not position). `solve()` is the
    main entry point; the interface — including the haptic signal
    attributes — matches `DecoupledIKSolver` so the teleop layer treats
    the two interchangeably.

    Tunables (same recipe as the DK1 solver):
      lam_pos           DLS base damping on the 3-DoF position sub-solve.
      lam0              Extra damping ramp amplitude near the joints-1-3
                        singularity. Total λ² = lam_pos² + lam0² · ramp²
                        where ramp = max(0, 1 - w/w0), w = |det(J_pos)|.
      w0                Manipulability threshold where the ramp starts.
                        The SO-101's links are ~3× shorter than the
                        DK1's, so det(J_pos) runs ~30× smaller; default
                        0.002 (vs the DK1's 0.05).
      mu                Tikhonov stiffness pulling joints 1-3 toward q_rest.
      lam_rot           DLS base damping on the 2-DoF wrist sub-solve.
      lam0_rot, w0_rot  Adaptive-damping ramp on the wrist. Inert on this
                        geometry (w ≈ 1 everywhere, w0_rot default 0.5) —
                        kept for interface parity.
      rot_err_hold      Park the wrist when the orientation error angle
                        exceeds this (rad). Near the antipode the
                        shortest-way error direction flips under tiny
                        jitter; the wrist holds and reports saturating
                        limit pressure instead. Default 2.2 (~126°).
      q_rest            Rest pose (5 joints); the first three feed the
                        Tikhonov bias.
      max_dq_per_joint  Per-joint Δq cap, length 5.
    """

    def __init__(
        self,
        urdf_path: Path | None = None,
        lam_pos: float = 0.05,
        lam0: float = 0.15,
        w0: float = 0.002,
        mu: float = 0.02,
        lam_rot: float = 0.05,
        lam0_rot: float = 0.4,
        w0_rot: float = 0.5,
        rot_err_hold: float = 2.2,
        q_rest: np.ndarray | None = None,
        max_dq_per_joint: list[float] | None = None,
    ) -> None:
        self.lam_pos = float(lam_pos)
        self.lam0 = float(lam0)
        self.w0 = float(w0)
        self.mu = float(mu)
        self.lam_rot = float(lam_rot)
        self.lam0_rot = float(lam0_rot)
        self.w0_rot = float(w0_rot)
        self.rot_err_hold = float(rot_err_hold)
        q_rest_full = (
            DEFAULT_Q_REST.copy() if q_rest is None
            else np.asarray(q_rest, dtype=float)[:ARM_DOFS].copy()
        )
        self.q_rest_123 = q_rest_full[:3].copy()
        self.max_dq_per_joint = (
            None if max_dq_per_joint is None
            else np.asarray(max_dq_per_joint, dtype=float).reshape(ARM_DOFS).copy()
        )

        self.model, self.data = build_so101_model(urdf_path)

        # Resolve the five arm joints by name; derive qpos addresses, dof
        # (velocity/jacobian) indices, and limits straight from the
        # compiled model — nothing positional, nothing transcribed by hand.
        self._qpos_adr = np.zeros(ARM_DOFS, dtype=int)
        self._dof_adr = np.zeros(ARM_DOFS, dtype=int)
        self.joint_limits = np.zeros((ARM_DOFS, 2))
        for i, name in enumerate(ARM_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"SO101IKSolver: joint {name!r} missing from model")
            self._qpos_adr[i] = self.model.jnt_qposadr[jid]
            self._dof_adr[i] = self.model.jnt_dofadr[jid]
            self.joint_limits[i] = self.model.jnt_range[jid]
        self._dof_pos = self._dof_adr[:3]   # shoulder_pan, shoulder_lift, elbow_flex
        self._dof_wrist = self._dof_adr[3:]  # wrist_flex, wrist_roll

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tool0"
        )
        self.anchor_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "wrist_anchor"
        )
        if -1 in (self.site_id, self.anchor_site_id):
            raise RuntimeError("SO101IKSolver: required site missing from model")

        # Set by each solve(). Signals fed into the controller-haptic mix
        # downstream (same contract as DecoupledIKSolver):
        #   last_limit_pressure (rad)          — joint-limit clip this tick.
        #   last_pos_err_norm (m)              — workspace-boundary reach error.
        #   last_singularity_proximity (0..1)  — joints-1-3 damping ramp.
        #   last_wrist_gimbal_proximity (0..1) — wrist damping ramp (≈0 here).
        self.last_limit_pressure: float = 0.0
        self.last_pos_err_norm: float = 0.0
        self.last_singularity_proximity: float = 0.0
        self.last_wrist_gimbal_proximity: float = 0.0

    def _fk(self, qpos: np.ndarray) -> None:
        """Run kinematics + comPos (required for the site Jacobians in solve()).
        `qpos` is the 5 arm joints (extra trailing values ignored)."""
        self.data.qpos[:] = 0
        self.data.qpos[self._qpos_adr] = np.asarray(qpos, dtype=float)[:ARM_DOFS]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # for jacobians

    def fk(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics. Returns (tool0 world pos, tool0 world quat wxyz)."""
        self._fk(qpos)
        pos = self.data.site_xpos[self.site_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return pos, quat

    def wrist_anchor_xpos(self) -> np.ndarray:
        """World position of the wrist_anchor site at the most recent FK.
        This is the wrist-invariant point the position task targets; the
        pose mapper should use it as the rotation pivot so pure controller
        rotations leave joints 1-3 at rest."""
        return self.data.site_xpos[self.anchor_site_id].copy()

    # Alias so teleop code written against DecoupledIKSolver's anchor
    # accessor keeps working unmodified.
    j4_anchor_xpos = wrist_anchor_xpos

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        qpos_seed: np.ndarray,
    ) -> np.ndarray:
        """One step of decoupled IK.

        Always returns a valid 5-vector. Boundary cases (workspace edge,
        joint limits, near-antipodal demand) are handled in-line — the arm
        degrades gracefully instead of freezing. See module docstring.
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        R_target = _quat_wxyz_to_R(target_quat_wxyz)
        seed = np.asarray(qpos_seed, dtype=float)[:ARM_DOFS]

        # ----- Step 1: FK at the seed; read the *current* (tool0 → anchor)
        # vector in tool0's local frame. Re-deriving this every tick keeps
        # the position-task math self-consistent with the wrist
        # configuration we're actually at, so a re-anchor with
        # target == current_ee produces pos_err = 0 exactly. -----
        self._fk(seed)
        current_tool0 = self.data.site_xpos[self.site_id].copy()
        current_R_tool0 = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        current_anchor = self.data.site_xpos[self.anchor_site_id].copy()
        ee_to_anchor_in_tool0 = current_R_tool0.T @ (current_anchor - current_tool0)

        # ----- Step 2: target anchor in arm-base; pos_err for Newton step -----
        target_anchor = target_pos + R_target @ ee_to_anchor_in_tool0
        pos_err = target_anchor - current_anchor

        self.last_pos_err_norm = float(np.linalg.norm(pos_err))

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.anchor_site_id)
        J_pos_arm = jacp[:, self._dof_pos]       # 3x3, joints 1-3 only

        # Manipulability-adaptive damping, same recipe as the DK1 solver:
        # w = |det(J_pos_arm)| → damping climbs smoothly as w → 0, bounding
        # joint velocities near the shoulder singularity.
        w = abs(float(np.linalg.det(J_pos_arm)))
        ramp = max(0.0, 1.0 - w / max(self.w0, 1e-12))
        self.last_singularity_proximity = float(ramp)
        lam2 = self.lam_pos ** 2 + (self.lam0 ** 2) * (ramp ** 2)
        mu2 = self.mu ** 2
        A = J_pos_arm.T @ J_pos_arm + (lam2 + mu2) * np.eye(3)
        b = J_pos_arm.T @ pos_err + mu2 * (self.q_rest_123 - seed[:3])
        dq_arm = np.linalg.solve(A, b)

        new_q123 = seed[:3] + dq_arm

        # ----- Step 3: FK at (new_q123, seed wrist); read tool0 orientation -----
        qpos_after_arm = seed.copy()
        qpos_after_arm[:3] = new_q123
        self._fk(qpos_after_arm)
        R_cur = self.data.site_xmat[self.site_id].reshape(3, 3)

        # ----- Step 4: orientation error as a world-frame rotation vector -----
        R_err = R_target @ R_cur.T
        q_err = np.zeros(4)
        mujoco.mju_mat2Quat(q_err, np.ascontiguousarray(R_err).ravel())
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, q_err, 1.0)

        # ----- Step 5: damped LS on the 2-DoF wrist -----
        # J_rot is 3x2 (wrist_flex, wrist_roll columns): the normal
        # equations project the demanded rotation onto the reachable
        # pitch×roll subspace; the unreachable yaw component stays as
        # residual (see module docstring). w = sqrt(det(JᵀJ)) is the area
        # spanned by the two unit axis directions — ≈ 1 everywhere on this
        # geometry since the axes are perpendicular by construction.
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        J_rot = jacr[:, self._dof_wrist]         # 3x2, wrist joints 4-5
        JtJ = J_rot.T @ J_rot
        w_rot = float(np.sqrt(max(0.0, np.linalg.det(JtJ))))
        ramp_rot = max(0.0, 1.0 - w_rot / max(self.w0_rot, 1e-12))
        self.last_wrist_gimbal_proximity = float(ramp_rot)
        # Antipode gate: with the error angle past `rot_err_hold`, the
        # shortest-way direction of e_rot is unstable — park the wrist
        # instead of chasing it.
        wrist_parked = float(np.linalg.norm(e_rot)) > self.rot_err_hold
        if wrist_parked:
            dq_wrist = np.zeros(2)
        else:
            lam2_rot = self.lam_rot ** 2 + (self.lam0_rot ** 2) * (ramp_rot ** 2)
            A_rot = JtJ + lam2_rot * np.eye(2)
            dq_wrist = np.linalg.solve(A_rot, J_rot.T @ e_rot)
        new_q45 = seed[3:] + dq_wrist

        # ----- Step 6: joint-limit clamp; limit-pressure metric -----
        qpos5 = np.concatenate([new_q123, new_q45])
        qpos5_reachable = np.clip(qpos5, self.joint_limits[:, 0], self.joint_limits[:, 1])

        # Limit pressure = L2 distance from the unclamped step to the
        # clamped one — how hard this tick pushes joints into their stops.
        # A parked wrist takes no step at all, so it reports a saturating
        # pressure directly.
        self.last_limit_pressure = float(
            np.linalg.norm(qpos5 - qpos5_reachable)
        )
        if wrist_parked:
            self.last_limit_pressure = max(self.last_limit_pressure, 0.35)

        # ----- Step 7: per-joint Δq cap (operator-safety velocity bound) -----
        qpos5 = qpos5_reachable
        dq_total = qpos5 - seed
        if self.max_dq_per_joint is not None:
            dq_total = np.clip(dq_total, -self.max_dq_per_joint, self.max_dq_per_joint)
        qpos5 = seed + dq_total

        # ----- Step 8: clamp into joint limits (don't reject) -----
        qpos5 = np.clip(qpos5, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return qpos5


# ---------- self-test ----------

def _self_test() -> None:
    """Round-trip: pick known qpos → FK → solve from that target → check."""
    solver = SO101IKSolver(max_dq_per_joint=[0.05] * ARM_DOFS)

    rest = DEFAULT_Q_REST.copy()
    test_set = [
        ("rest",            rest.copy()),
        ("slight rotation", rest + np.array([0.2, 0.1, -0.1, 0.1, 0.3])),
        ("forward reach",   np.array([0.0, 0.5, 0.5, 0.3, 0.0])),
        ("yaw + pitch",     np.array([0.6, -0.2, 0.9, 0.9, 0.8])),
    ]

    print(f"{'pose':22s}  Δqpos  |  pos_err  |  rot_resid")
    print("-" * 70)
    for label, q in test_set:
        solver._fk(q)
        target_pos = solver.data.site_xpos[solver.site_id].copy()
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, solver.data.site_xmat[solver.site_id])

        seed = rest.copy()
        result = solver.solve(target_pos, target_quat, seed)

        solver._fk(result)
        actual_pos = solver.data.site_xpos[solver.site_id].copy()
        actual_R = solver.data.site_xmat[solver.site_id].reshape(3, 3)
        target_R = _quat_wxyz_to_R(target_quat)
        rot_err = float(np.linalg.norm(actual_R - target_R, ord='fro'))
        pos_err = float(np.linalg.norm(actual_pos - target_pos))
        diff = float(np.linalg.norm(result - q))
        print(f"{label:22s}  {diff:.4f}  |  {pos_err*1000:5.1f} mm  |  {rot_err:.3f}")

    # Convergence: iterating on a fixed reachable target must drive the
    # position error to ~zero and the orientation error down to the
    # unreachable-yaw residual (which is 0 for a target generated by FK).
    raw = SO101IKSolver()   # no Δq cap — observe raw steps
    q_t = np.array([0.4, 0.3, 0.8, 0.5, 0.6])
    raw._fk(q_t)
    t_pos = raw.data.site_xpos[raw.site_id].copy()
    t_quat = np.zeros(4)
    mujoco.mju_mat2Quat(t_quat, raw.data.site_xmat[raw.site_id])
    cur = DEFAULT_Q_REST.copy()
    for _ in range(200):
        cur = raw.solve(t_pos, t_quat, cur)
    _, q_reached = raw.fk(cur)
    R_a, R_b = _quat_wxyz_to_R(q_reached), _quat_wxyz_to_R(t_quat)
    cos_err = (float(np.trace(R_a @ R_b.T)) - 1.0) / 2.0
    rot_err_deg = float(np.degrees(np.arccos(np.clip(cos_err, -1.0, 1.0))))
    pos_err_mm = float(np.linalg.norm(raw.data.site_xpos[raw.site_id] - t_pos)) * 1000
    ok = rot_err_deg < 2.0 and pos_err_mm < 10.0
    print(f"\n  convergence (200 iters): rot_err={rot_err_deg:.3f}°  "
          f"pos_err={pos_err_mm:.1f} mm  [{'ok' if ok else 'FAIL'}]")
    assert ok, "iterated solve did not converge to the FK-generated target"

    # Underactuation: a pure-yaw orientation demand (position held) must
    # produce a small, bounded wrist step — the yaw component projects
    # out — and must never blow up joints 4-5.
    raw._fk(q_t)
    p0 = raw.data.site_xpos[raw.site_id].copy()
    R0 = raw.data.site_xmat[raw.site_id].reshape(3, 3).copy()
    yaw = np.radians(30.0)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                   [np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    qt = np.zeros(4)
    mujoco.mju_mat2Quat(qt, np.ascontiguousarray(Rz @ R0).ravel())
    out = raw.solve(p0, qt, q_t)
    step = float(np.linalg.norm(out[3:] - q_t[3:]))
    ok = step < np.radians(35.0) and bool(np.all(np.isfinite(out)))
    print(f"  underactuated yaw demand (30°): wrist step |Δq45|={np.degrees(step):.2f}°  "
          f"[{'ok' if ok else 'FAIL'}]")
    assert ok, "pure-yaw demand produced an unbounded wrist step"

    # Boundary: far-out-of-reach target → finite qpos, bounded step.
    seed = DEFAULT_Q_REST.copy()
    out = solver.solve(np.array([2.0, 0.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]), seed)
    print(f"  far-out-of-reach: qpos5 finite={bool(np.all(np.isfinite(out)))}  "
          f"|Δq|={float(np.linalg.norm(out - seed)):.4f}")

    # Antipode gate: a demand ~150° away must PARK the wrist while ~100°
    # away still tracks.
    q0 = np.array([0.0, 0.3, 0.6, 0.3, 0.0])
    raw._fk(q0)
    p0 = raw.data.site_xpos[raw.site_id].copy()
    R0 = raw.data.site_xmat[raw.site_id].reshape(3, 3).copy()

    def _twisted_target(deg: float) -> np.ndarray:
        ang = np.radians(deg)
        K = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        R_t = (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)) @ R0
        qt = np.zeros(4)
        mujoco.mju_mat2Quat(qt, np.ascontiguousarray(R_t).ravel())
        return qt

    out150 = raw.solve(p0, _twisted_target(150.0), q0)
    parked = bool(np.all(out150[3:] == q0[3:]))
    pressure_at_park = raw.last_limit_pressure
    out100 = raw.solve(p0, _twisted_target(100.0), q0)
    tracks = float(np.linalg.norm(out100[3:] - q0[3:])) > 1e-3
    ok = parked and pressure_at_park >= 0.35 and tracks
    print(f"  antipode gate: parked@150°={parked} (pressure={pressure_at_park:.2f})  "
          f"tracks@100°={tracks}  [{'ok' if ok else 'FAIL'}]")
    assert ok, "antipode gate broken (wrist should park at 150°, track at 100°)"


if __name__ == "__main__":
    _self_test()
