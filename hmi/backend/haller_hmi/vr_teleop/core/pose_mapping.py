"""Clutch-relative pose mapping: WebXR controller pose → EE target in arm base.

Robot-agnostic. Nothing here knows what an SO-101 is; it turns "where the
operator's hand is" into "where the tool should be", and the IK layer turns
that into joints. Ported from the `vr-teleop-kit` reference stack
(https://github.com/Dream-Machines-Robotics/vr-teleop-kit), whose reasoning
about reach limits and incremental rotation is reproduced below because it is
the part that fixes real hardware misbehaviour, not a style preference.

`ClutchPoseMapper` holds the "engage" state — the controller pose and the EE
pose captured at the moment the operator squeezed the grip. While the clutch
is held, the EE target is the engaged EE pose composed with an effective
controller delta, rotated from Quest world into the arm's base frame.

Pass the arm's CURRENT EE pose into `target()` and the two reach limits come
alive: the delta then accumulates per-tick increments and the target is held
to within `pos_reach_limit` / `rot_reach_limit` of where the arm actually is,
with the excess ABSORBED. That absorbing behaviour is the one to understand,
because it is what the previous mapping in this codebase did not have:

  * A demand pressed past a joint stop can never wind up. Without a limit,
    an operator whose hand keeps going while the arm cannot builds an
    unbounded error; the arm then runs at its rate cap in whatever direction
    the error points, and reversing does nothing until the whole overshoot
    has been retraced.
  * Orientation can never "come around". 350° clockwise and 10°
    anticlockwise are the same orientation, and an absolute mapping must
    eventually agree with that — which reads as the wrist snapping in from
    the wrong side. The incremental path never has to agree, because it
    never accumulates a large angle in the first place.
  * Reversal bites immediately, like a mouse cursor at the edge of a screen:
    push past the edge and the extra travel is simply gone, so pulling back
    a centimetre moves the target a centimetre.

The costs are real and worth stating: absorbed motion does not come back, so
hand↔tool correspondence drifts within one engagement, and with a rotation
gain ≠ 1 the per-increment scaling makes curved hand paths path-dependent.
Re-clutching realigns either way, and re-clutching is a ratchet the operator
is already doing to cover a workspace bigger than their arm.

On clutch release the mapper disengages and `target()` returns None until the
next engage, which is what lets the operator reposition without moving the
robot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import quat


@dataclass
class ClutchPoseMapper:
    """One hand's clutch-relative controller→EE mapping.

    Parameters:
        R: 3×3 rotation taking Quest world vectors to arm base vectors
            (``v_armbase = R @ v_quest``). Built by `frames.stance_rotation`;
            replaced per engage with a yaw-corrected version so the operator
            can turn their body between engagements and still have
            "controller forward" mean "arm forward".
        scale: linear gain on translation (1.0 = 1:1).
        scale_rotation: gain on rotation (1.0 = 1:1). Above 1 a small wrist
            twist becomes a large tool twist — which this arm needs, since
            its wrist_roll range is far wider than a human wrist holding a
            controller can cover.
        rotation_pivot: optional 3-vector in arm-base frame. When set,
            controller rotation is read as "rotate the tool ABOUT this
            point" rather than "rotate the tool in place". The teleop passes
            the wrist anchor here, so a pure hand twist swings the gripper
            around the wrist the way the operator's own hand does, instead
            of spinning the tool on the spot and dragging the arm after it.
        pos_reach_limit: max distance (m) the position target may run ahead
            of the arm's CURRENT position. 0/None disables (absolute
            mapping). The 0.15 m default is the reference stack's own SO-101
            value — "smaller arm, smaller wall" against the 0.25 m it uses
            on the DK1 — and the only one with recorded episodes behind it
            (46 / 29,500 frames). The 0.12 it replaces came in with the
            port-time snapshot and was never measured on this arm.
        rot_reach_limit: max angle (rad) the orientation target may run ahead
            of the arm's CURRENT orientation. 0/None disables.
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: float = 1.0
    scale_rotation: float = 1.0
    rotation_pivot: np.ndarray | None = None
    pos_reach_limit: float | None = 0.15
    rot_reach_limit: float | None = 0.6

    def __post_init__(self) -> None:
        self._engaged: bool = False
        self._ctrl_engage_pos: np.ndarray | None = None
        self._ctrl_engage_quat: np.ndarray | None = None
        self._ee_engage_pos: np.ndarray | None = None
        self._ee_engage_quat: np.ndarray | None = None
        self.set_R(self.R)
        # Incremental reach-limit state: the previous tick's controller pose
        # (Quest frame) and the accumulated effective deltas (arm-base frame).
        # All reset on every engage.
        self._ctrl_prev_pos: np.ndarray | None = None
        self._ctrl_prev_quat: np.ndarray | None = None
        self._d_pos_eff: np.ndarray = np.zeros(3)
        self._d_quat_eff: np.ndarray = quat.IDENTITY.copy()
        #: Diagnostics for the HUD / haptics: how hard the operator is
        #: pressing past what the arm can follow, 0..1 of the reach limit.
        self.last_pos_absorbed: float = 0.0
        self.last_rot_absorbed: float = 0.0

    # ---- state ----------------------------------------------------------

    @property
    def engaged(self) -> bool:
        return self._engaged

    def set_R(self, R: np.ndarray) -> None:
        """Replace the Quest→arm-base rotation used for delta mapping.

        `R` is allowed to be IMPROPER (det = −1). One of this rig's operator
        stances is a mirror — face-to-face, the arm as your reflection — and
        that is a considered choice, not a bug to engineer away (see
        `frames.STANCES`). It does mean the frame change cannot be done by
        quaternion conjugation, which only exists for proper rotations, so
        `_rotate_delta` below carries rotations across on the axial-vector
        rule instead. The determinant is cached here because that rule needs
        it on every tick.
        """
        self.R = np.asarray(R, dtype=float).copy()
        self._R_det = float(np.sign(np.linalg.det(self.R))) or 1.0

    def _rotate_delta(self, q_delta: np.ndarray) -> np.ndarray:
        """Carry a rotation from Quest world into the arm base frame.

        For orthogonal S the identity ``S·[ω]ₓ·Sᵀ = [det(S)·S·ω]ₓ`` holds,
        so a rotation of angle θ about axis n̂ becomes one of the same angle
        about ``det(S)·S·n̂``. For a proper S that is ordinary quaternion
        conjugation; for the mirror stance the determinant is what keeps the
        handedness of the twist right, which is exactly the property a
        mirror metaphor needs.
        """
        return quat.from_rotvec(self._R_det * (self.R @ quat.to_rotvec(q_delta)))

    def engage(
        self,
        controller_pos_quest: np.ndarray,
        controller_quat_quest: np.ndarray,
        ee_pos_armbase: np.ndarray,
        ee_quat_armbase: np.ndarray,
        pivot_armbase: np.ndarray | None = None,
    ) -> None:
        """Capture the engage frame. Call on the rising clutch edge.

        Note what this makes true: at the instant of the squeeze the target
        IS the arm's own pose, so nothing jumps and the session's
        acquisition gate matches by construction. That is not incidental —
        it is why the countdown for a VR session can be short.
        """
        self._ctrl_engage_pos = np.array(controller_pos_quest, float, copy=True)
        self._ctrl_engage_quat = quat.normalize(controller_quat_quest)
        self._ee_engage_pos = np.array(ee_pos_armbase, float, copy=True)
        self._ee_engage_quat = quat.normalize(ee_quat_armbase)
        self.rotation_pivot = (
            None if pivot_armbase is None
            else np.array(pivot_armbase, float, copy=True)
        )
        self._ctrl_prev_pos = self._ctrl_engage_pos.copy()
        self._ctrl_prev_quat = self._ctrl_engage_quat.copy()
        self._d_pos_eff = np.zeros(3)
        self._d_quat_eff = quat.IDENTITY.copy()
        self.last_pos_absorbed = 0.0
        self.last_rot_absorbed = 0.0
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False

    # ---- the mapping ----------------------------------------------------

    def target(
        self,
        controller_pos_quest: np.ndarray,
        controller_quat_quest: np.ndarray,
        ee_pos_armbase: np.ndarray | None = None,
        ee_quat_armbase: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Current EE target in arm-base frame, or None while disengaged.

        Passing the arm's CURRENT EE pose enables the reach limits; leaving
        them out gives the plain absolute delta-since-engage mapping, which
        is kept only so the limits can be compared against something.
        """
        if not self._engaged:
            return None
        assert self._ctrl_engage_pos is not None
        assert self._ctrl_engage_quat is not None
        assert self._ee_engage_pos is not None
        assert self._ee_engage_quat is not None

        # ---- translation ----
        # Reach-limited path accumulates per-tick increments. Vectors
        # commute, so at any fixed scale the increments telescope to exactly
        # the absolute scaled delta right up until the limit absorbs some.
        p_now = np.asarray(controller_pos_quest, dtype=float)
        pos_limited = ee_pos_armbase is not None and bool(self.pos_reach_limit)
        if pos_limited:
            assert self._ctrl_prev_pos is not None
            self._d_pos_eff = self._d_pos_eff + self.R @ (
                self.scale * (p_now - self._ctrl_prev_pos))
            d_pos_arm = self._d_pos_eff
        else:
            d_pos_arm = self.R @ (self.scale * (p_now - self._ctrl_engage_pos))
        self._ctrl_prev_pos = p_now.copy()

        # ---- rotation ----
        q_now = quat.normalize(controller_quat_quest)
        rot_limited = ee_quat_armbase is not None and bool(self.rot_reach_limit)
        if rot_limited:
            # Per-tick increments are a degree or two — never
            # direction-ambiguous — so accumulating them cannot produce the
            # near-180° error whose shortest-way direction flips under hand
            # tremor. At scale_rotation == 1 they telescope to exactly the
            # absolute delta; at other gains the gain applies per increment
            # (a RATE gain, like mouse sensitivity), which is path-dependent
            # on SO(3) but never wraps.
            assert self._ctrl_prev_quat is not None
            q_now = quat.hemisphere_align(q_now, self._ctrl_prev_quat)
            inc = quat.mul(q_now, quat.conj(self._ctrl_prev_quat))
            self._ctrl_prev_quat = q_now.copy()
            if self.scale_rotation != 1.0:
                inc = quat.power(inc, self.scale_rotation)
            inc_arm = self._rotate_delta(inc)
            d_quat_arm = quat.normalize(quat.mul(inc_arm, self._d_quat_eff))
            self._d_quat_eff = d_quat_arm
        else:
            # Absolute path. Keep the incremental state fresh anyway, so
            # toggling a reach limit live doesn't resume from a stale prev.
            if self._ctrl_prev_quat is not None:
                q_now = quat.hemisphere_align(q_now, self._ctrl_prev_quat)
                self._ctrl_prev_quat = q_now.copy()
            d_quat_quest = quat.mul(q_now, quat.conj(self._ctrl_engage_quat))
            if self.scale_rotation != 1.0:
                d_quat_quest = quat.power(d_quat_quest, self.scale_rotation)
            d_quat_arm = self._rotate_delta(d_quat_quest)

        target_quat = quat.mul(d_quat_arm, self._ee_engage_quat)

        if rot_limited:
            ee_q = quat.normalize(ee_quat_armbase)
            e = quat.to_rotvec(quat.mul(target_quat, quat.conj(ee_q)))
            e_norm = float(np.linalg.norm(e))
            limit = float(self.rot_reach_limit)
            self.last_rot_absorbed = min(1.0, e_norm / limit) if limit > 0 else 0.0
            if e_norm > limit:
                e = e * (limit / e_norm)
                target_quat = quat.mul(quat.from_rotvec(e), ee_q)
                # Absorb: fold the clamp back into the effective delta so the
                # excess is GONE rather than pending.
                self._d_quat_eff = quat.mul(target_quat,
                                            quat.conj(self._ee_engage_quat))
                d_quat_arm = self._d_quat_eff

        # A pivot turns "rotate in place" into "swing about this point": the
        # engaged offset from pivot to tool is carried round by the same
        # delta. With no pivot the offset term vanishes and this is the
        # plain in-place rotation.
        if self.rotation_pivot is not None:
            offset = self._ee_engage_pos - self.rotation_pivot
            target_pos = (self.rotation_pivot
                          + quat.rotate(d_quat_arm, offset) + d_pos_arm)
        else:
            target_pos = self._ee_engage_pos + d_pos_arm

        if pos_limited:
            ee_p = np.asarray(ee_pos_armbase, dtype=float)
            dp = target_pos - ee_p
            dp_norm = float(np.linalg.norm(dp))
            limit = float(self.pos_reach_limit)
            self.last_pos_absorbed = min(1.0, dp_norm / limit) if limit > 0 else 0.0
            if dp_norm > limit:
                clamped = ee_p + dp * (limit / dp_norm)
                self._d_pos_eff = self._d_pos_eff + (clamped - target_pos)
                target_pos = clamped

        return target_pos, target_quat
