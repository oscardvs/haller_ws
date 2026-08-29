"""KitSideTeleop — one side's kit-faithful VR tracking loop, ported line by line.

This is phase 2 of the kit port: the DRIVEN path's per-arm logic, adapted from
the reference single-arm teleop
    /home/odesha/vr-teleop-kit/src/vr_teleop_kit/lerobot/so101_quest_teleop.py
onto the byte-faithful vendored math in `haller_hmi.vr_teleop.kit` (equivalence
pinned at 1e-12 by tests/test_kit_equivalence.py — never edit those files).

The session owns the 60 Hz tick, the write gate, authority, the guard and the
recorder; this class owns exactly what the kit's per-arm dict owned: the
clutch, the EMA pose filter, the mapper, the solver, and the OPEN-LOOP qpos
integrator that is never re-read from the arm mid-teleop.

Kit line map (so101_quest_teleop.py, v0.1.0, 934 lines):

    XR_FRAME_STALE_TIMEOUT_S      <- kit 106
    KIT_MAX_DQ_PER_JOINT          <- kit 168-171  (config default; the solver
                                      constructor itself defaults to None)
    __init__                      <- kit 210-283  (solver 232-244, mapper
                                      258-264, per-arm state 253-283)
    seed_from_observed            <- kit seed_qpos_from_obs, 448-474
    update                        <- kit get_action 476-505 (yaw extraction
                                      493-495, latest-frame semantics) +
                                      _update_arm 545-675:
      absent/untracked skip         <- kit 546-547
      staleness gate                <- kit 550-556
      EMA pose filter               <- kit 558-573
      button decode                 <- kit 575-583 (wire spelling, see DEV-2)
      precision edge + gains        <- kit 598-614
      stale-recovery re-anchor      <- kit 616-619
      clutch edges                  <- kit 621-630
      gripper follows trigger       <- kit 632-637
      rest ramp                     <- kit 584-596, 639-652 (NOT ported, DEV-5)
      FK -> mapper -> solve         <- kit 654-659
      haptic mix                    <- kit 660-675 (exact gate numbers)
    _anchor_mapper                <- kit 523-543
    _build_action                 <- kit 509-521 (radians -> signs * degrees)
    joint_signs semantics         <- kit 186-203 (config), 459-462, 515

Intentional deviations (each also marked `DEV-n` where it occurs):

  DEV-1  Threading, websockets, ik_state publishing, live config_update and
         the force-haptic/handoff machinery are gone. The session calls
         `update()` once per 60 Hz tick with the LATEST frame's controller
         dict and that frame's age — the same latest-frame-at-consumer-rate
         shape the kit's WS-thread + get_action() pair produced.
  DEV-2  Frames arrive as the wire dict from `vr_teleop.wire`
         ({position, orientation(xyzw), trigger, squeeze, tracked,
         precision}), not as an xr_frame button array: squeeze == buttons[1]
         "p" (grip clutch), trigger == buttons[0] "v", precision is the
         wire's own per-side flag (never gamepad index 4 — see wire.py), and
         `tracked: false` takes the kit's absent-controller exit (the kit
         page omits an untracked hand from the frame entirely).
  DEV-3  R_calib comes from `core.frames.stance_rotation`, composed with the
         constant change of basis `URDF_FROM_MOUNT` below, because the
         vendored solver lives in the new-calib URDF base frame while the
         stance matrices speak the classic haller mount frame. VERIFIED
         2026-08-29 (and pinned by tests/test_kit_teleop.py):
             URDF_FROM_MOUNT @ stance_rotation('behind', yaw)
                 == DEFAULT_R_CALIB @ R_y(-yaw)      for all yaw,
         i.e. the default stance IS the kit's shipped mapping, exactly —
         `quat.rot_y` matches the kit's `_R_y` and `frames.head_yaw` matches
         `_yaw_from_quat_xyzw` element for element. Yaw handling follows the
         kit: a MISSING head pose keeps the previous R (kit 531-533); the
         `yaw_on_engage=False` switch (no kit equivalent) anchors with the
         uncorrected stance matrix instead, so stance changes still apply.
  DEV-4  The gripper's internal state and ACTION stay in [0, 1] (1 = open);
         the kit speaks LeRobot's 0..100 on the wire. The session scales the
         action onto the calibrated range downstream (`_to_degrees`),
         unchanged — and the one INPUT that arrives in follower units, the
         observed pose handed to `seed_from_observed`, is normalized against
         that same calibrated range on the way in (the kit's /100, kit 464,
         generalized to the (lo, hi) this arm actually reports).
  DEV-5  The rest ramp (thumbstick-click "go home", kit 584-596/639-652) is
         not ported — the session's request_home covers it — and the kit's
         qpos[5] gripper-jaw viewer mirror is dropped: qpos here is the five
         arm joints only.
  DEV-6  The solver is built on its OWN constructor defaults (the kit's
         shipped tune) plus the kit config's shipped per-joint dq caps
         (KIT_MAX_DQ_PER_JOINT). None of haller's old IK knobs
         (lam_pos/lam0/w0/mu/lam_rot/rot_err_hold/max_dq_deg_*) are wired
         in: they parameterise a different algorithm. Only
         `rest_pose_deg`, if an operator set one, feeds the solver's q_rest
         (Tikhonov posture bias + initial qpos).
  DEV-7  diag() is wire-shaped for the existing frontend: the kit's
         radian-unit last_limit_pressure is converted to DEGREES (the parked
         wrist's 0.35 rad floor lands on the 20.05° the frontend gates were
         sized against); `orient_residual` is measured here with one extra
         FK after the solve (achieved-vs-demand fraction — the kit never
         reported it; diagnostic only, it feeds nothing in the control
         path); `pos_absorbed`/`rot_absorbed` are 0.0 because the vendored
         kit mapper exposes no absorbed diagnostics and the vendored files
         must not be modified; `sigma_min` is omitted (the kit solver does
         not measure it, and the frontend type-guards it); `driving` is
         omitted — authority belongs to the session, which overlays it.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import QuestTeleopConfig
from .core import frames as frames_mod
from .core import quat
from .kit.pose_mapping import ClutchPoseMapper
from .kit.so101_ik import ARM_DOFS, SO101IKSolver
from .kit.so101_model import ARM_JOINT_NAMES, DEFAULT_Q_REST

logger = logging.getLogger(__name__)

#: If the newest frame is older than this, treat the controller stream as
#: stale: freeze, and silently re-anchor on recovery. Kit line 106.
XR_FRAME_STALE_TIMEOUT_S = 0.2

#: The kit's SHIPPED per-joint per-solve dq cap (rad/tick), length 5 —
#: 3 position joints + 2 wrist. This is `SO101QuestTeleoperatorConfig.
#: max_dq_per_joint`'s default (kit 168-171), always passed to the solver by
#: the kit's own loop; the solver constructor alone would default to None
#: (uncapped), which is NOT the shipped behavior. With joint limits and servo
#: physics, this cap is one of the kit's only three governors — no post-IK
#: shaping exists downstream of it.
KIT_MAX_DQ_PER_JOINT: tuple[float, ...] = (0.06, 0.06, 0.06, 0.24, 0.24)

#: DEV-3 — change of basis taking classic haller mount-frame vectors into the
#: new-calib URDF base frame the vendored solver computes in. The two frames
#: describe the same bench with axes relabeled about z: the URDF base reaches
#: along +x at qpos 0 (FK-verified: tool0 = (0.391, 0, 0.227)), the haller
#: mount frame reaches along -y (old fk_frames tool = (0, -0.383, 0.096) — see
#: core/frames.py, "the mounts put each arm's reach along -y"). This constant
#: is a pure axis relabeling of the MAPPING matrices, not a bridge between the
#: two FK models (no such bridge exists — see the drift finding of record in
#: tests/test_kit_equivalence.py). With it, the default stance reproduces the
#: kit's shipped DEFAULT_R_CALIB exactly:
#:     URDF_FROM_MOUNT @ stance_rotation('behind', yaw)
#:         == DEFAULT_R_CALIB @ R_y(-yaw)
#: and 'mirror'/'front' carry over under the same relabeling.
URDF_FROM_MOUNT: np.ndarray = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])


def _mapper_R(stance: str, yaw_rad: float | None) -> np.ndarray:
    """`stance_rotation` expressed in the vendored solver's URDF frame."""
    return URDF_FROM_MOUNT @ frames_mod.stance_rotation(stance, yaw_rad)


class KitSideTeleop:
    """One side's kit-faithful tracking state: vendored mapper + solver +
    open-loop qpos.

    The session constructs one per side, seeds it from the observed pose on
    handover (`seed_from_observed`), then calls `update()` exactly once per
    60 Hz tick with the latest frame's controller dict. `update()` never
    reads the arm: the solve is seeded from this object's own integrated
    qpos, the audit's fix for the throttled-committed-seed feedback loop.
    """

    def __init__(
        self,
        joint_limits_deg: dict[str, tuple[float, float]],
        config: QuestTeleopConfig,
        *,
        urdf_path: str | None = None,
    ) -> None:
        unknown = [n for n in joint_limits_deg
                   if n != "gripper" and n not in ARM_JOINT_NAMES]
        if unknown:
            raise ValueError(
                f"KitSideTeleop: joint_limits_deg names {unknown!r} are not "
                f"SO-101 arm joints {ARM_JOINT_NAMES}")
        self._cfg = config
        self._limits = dict(joint_limits_deg)
        self._emit_gripper = "gripper" in joint_limits_deg

        # Per-joint sign between the model convention (new-calib URDF) and
        # the follower's degrees: follower_deg = sign * model_deg (kit
        # 186-203). QuestTeleopConfig has no such field today, so the default
        # is [1.0] * 5 — the standard follower build, and the convention this
        # very arm recorded the kit's 46 episodes on.
        signs = getattr(config, "joint_signs", None)
        signs = list(signs) if signs is not None else [1.0] * ARM_DOFS
        if len(signs) != ARM_DOFS or any(s not in (1.0, -1.0, 1, -1) for s in signs):
            raise ValueError(
                f"joint_signs must be 5 values of +1/-1, got {signs!r}")
        self._signs = [float(s) for s in signs]

        # DEV-6: rest pose — the operator's configured one if present
        # (follower degrees -> signs -> model radians), else the kit's
        # DEFAULT_Q_REST, which is also the solver constructor's default.
        q_rest = DEFAULT_Q_REST.copy()
        rest_deg = getattr(config, "rest_pose_deg", None)
        if rest_deg:
            for j, name in enumerate(ARM_JOINT_NAMES):
                if name in rest_deg:
                    q_rest[j] = float(
                        np.radians(self._signs[j] * float(rest_deg[name])))

        # Kit 232-244 — the solver on its own defaults (the shipped tune)
        # plus the kit config's shipped dq caps. Haller's old IK knobs are
        # deliberately NOT wired in (DEV-6).
        self._solver = SO101IKSolver(
            urdf_path=urdf_path or None,
            q_rest=q_rest.copy(),
            max_dq_per_joint=list(KIT_MAX_DQ_PER_JOINT),
        )
        # Kit 258-264 — mapper, with R in the solver's frame (DEV-3).
        self._mapper = ClutchPoseMapper(
            R=_mapper_R(getattr(config, "stance", frames_mod.DEFAULT_STANCE), None),
            scale=config.scale_translation,
            scale_rotation=config.scale_rotation,
            rot_reach_limit=config.rot_reach_limit,
            pos_reach_limit=config.pos_reach_limit,
        )

        # Kit 253-283 — the per-arm state dict, as attributes. qpos is the
        # open-loop integrator: five arm joints, radians, model convention
        # (no jaw mirror — DEV-5).
        self._qpos: np.ndarray = q_rest.copy()
        self._trigger: float = 0.0
        self._engaged: bool = False
        self._last_grip: bool = False
        self._last_precision: bool = False
        self._needs_reanchor: bool = False
        self._pos_filt: np.ndarray | None = None
        self._quat_filt: np.ndarray | None = None
        self._haptic: float = 0.0
        self._tracked: bool = False
        self._diag: dict = self._make_diag(solved=False)

    # ------------------------------------------------------------------ API

    @property
    def engaged(self) -> bool:
        return self._engaged

    def seed_from_observed(self, joints_deg: dict[str, float]) -> None:
        """Reset qpos + gripper + engagement from an OBSERVED pose in
        FOLLOWER units — joints in degrees, gripper on its own
        `joint_limits_deg` range (LeRobot's 0..100 on a real SO-101, MJCF
        degrees on a sim arm). Kit seed_qpos_from_obs, 448-474: call before
        handing control to the operator so the IK anchors to the robot's
        actual pose and the next squeeze re-anchors the mapper cleanly.
        Missing keys are skipped.

        The gripper is normalized onto [0, 1] HERE, against the calibrated
        (lo, hi) this side was built with — the kit's own seed divides its
        0..100 by 100 (kit 464), and scaling by the range is that same move
        on whatever range this arm actually reports. This is the exact
        inverse of the session's `_to_degrees`, so a freeze-write of the
        seeded action commands the very degrees that were observed. Seeding
        the raw follower value into [0, 1] instead reads any jaw >1 unit
        open as FULLY open, and the first freeze-write drops whatever the
        jaw was holding. Internal state and the action stay [0, 1] (DEV-4).
        """
        for j, name in enumerate(ARM_JOINT_NAMES):
            if name in joints_deg:
                self._qpos[j] = float(
                    np.radians(self._signs[j] * float(joints_deg[name])))
        if "gripper" in joints_deg and "gripper" in self._limits:
            lo, hi = self._limits["gripper"]
            g_open = (float(joints_deg["gripper"]) - lo) / ((hi - lo) or 1.0)
            # trigger 1 = squeezed = closed
            self._trigger = 1.0 - float(np.clip(g_open, 0.0, 1.0))
        self._mapper.disengage()
        self._engaged = False
        self._last_grip = False
        self._pos_filt = None
        self._quat_filt = None
        self._needs_reanchor = False

    def update(
        self,
        ctrl: dict | None,
        head_orientation_xyzw,
        stance: str,
        frame_age_s: float,
    ) -> tuple[dict[str, float], bool]:
        """One session tick (60 Hz): kit get_action 476-505 + _update_arm.

        `ctrl` is this side's wire dict or None; the returned action maps
        every joint in joint_limits_deg to follower DEGREES, gripper to
        [0, 1]. When not engaged the action is the held qpos — the session's
        write gate decides whether to send it.
        """
        # Kit 493-495: yaw from the head pose of the SAME frame. The
        # yaw_on_engage=False switch reads as "no head pose ever" here and is
        # resolved at anchor time (DEV-3).
        yaw_now = (frames_mod.head_yaw(head_orientation_xyzw)
                   if self._cfg.yaw_on_engage else None)
        if stance not in frames_mod.STANCES:
            stance = self._cfg.stance
        self._update_arm(ctrl, yaw_now, stance, float(frame_age_s))
        return self._build_action(), self._engaged

    def diag(self) -> dict:
        """Wire-shaped diagnostics for the `ik_state` sides payload (DEV-7)."""
        return dict(self._diag)

    # ------------------------------------------------------- kit internals

    def _update_arm(self, ctrl: dict | None, yaw_now: float | None,
                    stance: str, gap_s: float) -> None:
        # Kit 546-547: an absent hand skips the tick outright — integrator
        # holds, anchor and filters survive. DEV-2: `tracked: false` takes
        # the same exit (the kit page omits an untracked controller).
        if ctrl is None or not ctrl.get("tracked", False):
            self._tracked = False
            self._diag = self._make_diag(solved=False)
            return
        self._tracked = True

        # Kit 550-556: staleness gate. Freeze; decay the haptic; remember to
        # re-anchor silently once frames recover (while still gripped).
        if gap_s > XR_FRAME_STALE_TIMEOUT_S:
            if self._engaged and not self._needs_reanchor:
                self._needs_reanchor = True
                logger.warning("kit teleop: frame stale (%.2fs gap) — pausing",
                               gap_s)
            self._haptic *= 0.6
            self._diag = self._make_diag(solved=False)
            return

        # Kit 558-560: wire orientation is WebXR xyzw; everything internal is
        # wxyz.
        pos_raw = np.asarray(ctrl["position"], dtype=float)
        quat_raw = quat.from_xyzw(ctrl["orientation"])

        # Kit 562-573: EMA smoothing on the controller pose (nlerp with
        # hemisphere check).
        alpha = float(self._cfg.pose_filter_alpha)
        if self._pos_filt is None or self._quat_filt is None:
            self._pos_filt = pos_raw.copy()
            self._quat_filt = quat_raw.copy()
        else:
            self._pos_filt = (1.0 - alpha) * self._pos_filt + alpha * pos_raw
            q_in = (quat_raw
                    if float(np.dot(self._quat_filt, quat_raw)) >= 0.0
                    else -quat_raw)
            qf = (1.0 - alpha) * self._quat_filt + alpha * q_in
            self._quat_filt = qf / np.linalg.norm(qf)
        pos = self._pos_filt
        quat_wxyz = self._quat_filt

        # Kit 575-583, wire spelling (DEV-2). Per-SIDE precision: this side's
        # own flag, never the other hand's. Rest button (kit 581-596): not
        # ported (DEV-5).
        grip = bool(ctrl.get("squeeze", False))
        trigger = float(ctrl.get("trigger", 0.0))
        precision = bool(ctrl.get("precision", False))

        # Kit 598-614: precision toggle — re-anchor on the transition (at the
        # OLD gains: the accumulated delta must not be reinterpreted under
        # the new scale), THEN scale the gains. Reach limits re-read every
        # tick so live tuning applies.
        if precision != self._last_precision:
            if self._engaged:
                self._anchor_mapper(pos, quat_wxyz, yaw_now, stance,
                                    "PRECISION" if precision else "FULL-SCALE")
        self._last_precision = precision
        scale_factor = self._cfg.precision_factor if precision else 1.0
        self._mapper.scale = self._cfg.scale_translation * scale_factor
        self._mapper.scale_rotation = self._cfg.scale_rotation * scale_factor
        self._mapper.rot_reach_limit = self._cfg.rot_reach_limit
        self._mapper.pos_reach_limit = self._cfg.pos_reach_limit

        # Kit 616-619: stale-recovery re-anchor, silent, only while gripped.
        if self._needs_reanchor and self._engaged and grip:
            self._anchor_mapper(pos, quat_wxyz, yaw_now, stance, "RE-ANCHOR")
        self._needs_reanchor = False

        # Kit 621-630: edge-detect clutch. Rising -> anchor; falling ->
        # disengage and clear the pose filters.
        if grip and not self._last_grip:
            self._anchor_mapper(pos, quat_wxyz, yaw_now, stance, "ENGAGE")
        elif not grip and self._last_grip:
            self._mapper.disengage()
            self._engaged = False
            self._pos_filt = None
            self._quat_filt = None
            logger.debug("kit teleop: clutch RELEASE")
        self._last_grip = grip

        # Kit 632-637: gripper tracks the trigger ONLY while engaged (holds
        # its last value otherwise — no stray closes between corrections).
        # No jaw-qpos viewer mirror here (DEV-5).
        if self._engaged:
            self._trigger = trigger

        # Kit 639-652: rest ramp — not ported (DEV-5).

        # Kit 654-675: FK at the OWN qpos -> mapper target (reach-limited,
        # absorbing) -> one solver step, integrated in place.
        ee_pos_now, ee_quat_now = self._solver.fk(self._qpos)
        out = self._mapper.target(pos, quat_wxyz, ee_pos_now, ee_quat_now)
        if out is not None:
            tgt_pos, tgt_quat = out
            demand_rad = quat.angle_between(tgt_quat, ee_quat_now)
            self._qpos[:] = self._solver.solve(tgt_pos, tgt_quat, self._qpos)

            # Kit 660-673: haptic mix — the kit's exact gate numbers.
            # pressure is in RADIANS here, as in the kit (degrees only on
            # the diag wire, DEV-7).
            pressure = float(getattr(self._solver, "last_limit_pressure", 0.0))
            pos_err = float(getattr(self._solver, "last_pos_err_norm", 0.0))
            singular = float(
                getattr(self._solver, "last_singularity_proximity", 0.0))
            gimbal = float(
                getattr(self._solver, "last_wrist_gimbal_proximity", 0.0))
            i_limit = min(1.0, max(0.0, (pressure - 0.05) / 0.25))
            i_reach = min(1.0, max(0.0, (pos_err - 0.02) / 0.06))
            i_singular = max(0.0, (singular - 0.95) / 0.2)
            i_gimbal = min(1.0, max(0.0, (gimbal - 0.5) / 0.5))
            raw_intensity = max(i_limit, i_reach, i_singular, i_gimbal)
            self._haptic = 0.6 * self._haptic + 0.4 * raw_intensity

            # DEV-7: orientation deficit for the HUD/one-shot cue, measured
            # achieved-vs-demand with one extra FK. Diagnostic only.
            _, q_after = self._solver.fk(self._qpos)
            resid_rad = quat.angle_between(tgt_quat, q_after)
            orient_residual = (min(1.0, resid_rad / demand_rad)
                               if demand_rad > 1e-6 else 0.0)
            self._diag = self._make_diag(solved=True,
                                         orient_residual=orient_residual)
        else:
            self._haptic *= 0.6
            self._diag = self._make_diag(solved=False)

    def _anchor_mapper(self, pos: np.ndarray, quat_wxyz: np.ndarray,
                       yaw_now: float | None, stance: str, label: str) -> None:
        """Kit _anchor_mapper, 523-543: capture the engage frame at the
        current robot+controller state — used by every re-anchor path."""
        ee_pos, ee_quat = self._solver.fk(self._qpos)
        anchor_pos = self._solver.wrist_anchor_xpos()
        if yaw_now is not None:
            # Kit 531-533: R_engage = R_CALIB @ R_y(-yaw); here that whole
            # product is stance_rotation(stance, yaw) in the solver's frame
            # (DEV-3), identical for the default stance.
            self._mapper.set_R(_mapper_R(stance, yaw_now))
        elif not self._cfg.yaw_on_engage:
            # DEV-3: explicit no-correction switch — the uncorrected stance
            # matrix, so a stance change still applies.
            self._mapper.set_R(_mapper_R(stance, None))
        # else: head pose missing — KEEP the previous R, the kit's rule
        # (a stale yaw beats snapping to an uncorrected frame mid-session).
        self._mapper.engage(pos, quat_wxyz, ee_pos, ee_quat,
                            pivot_armbase=anchor_pos)
        self._engaged = True
        logger.debug("kit teleop: clutch %s  ee_pos=%s  anchor=%s  yaw=%s",
                     label, ee_pos.round(3), anchor_pos.round(3),
                     f"{np.degrees(yaw_now):.1f}deg" if yaw_now is not None
                     else "-")

    def _build_action(self) -> dict[str, float]:
        """Kit _build_action, 509-521: solver qpos (radians, model
        convention) -> follower action (signs * degrees), gripper in [0, 1]
        with 1 = open (DEV-4)."""
        out: dict[str, float] = {
            name: float(self._signs[j] * np.degrees(self._qpos[j]))
            for j, name in enumerate(ARM_JOINT_NAMES)
            if name in self._limits
        }
        if self._emit_gripper:
            out["gripper"] = float(np.clip(1.0 - self._trigger, 0.0, 1.0))
        return out

    def _make_diag(self, *, solved: bool,
                   orient_residual: float = 0.0) -> dict:
        """DEV-7 — the `ik_state` side dict the frontend already reads.
        Solver metrics are reported only for a tick that actually solved;
        limit pressure converts kit radians -> the wire's degrees (parked
        wrist: 0.35 rad -> 20.05°, saturating the frontend's gate as
        before). The vendored mapper has no absorbed diagnostics, so
        pos/rot_absorbed are 0.0 (the vendored files must not be modified
        to add them). `driving` is the session's to overlay."""
        s = self._solver
        return {
            "tracked": bool(self._tracked),
            "engaged": bool(self._engaged),
            "haptic": float(self._haptic),
            "limit_pressure_deg": (
                float(np.degrees(s.last_limit_pressure)) if solved else 0.0),
            "pos_err_m": float(s.last_pos_err_norm) if solved else 0.0,
            "singularity": (
                float(s.last_singularity_proximity) if solved else 0.0),
            "orient_residual": float(orient_residual),
            "pos_absorbed": 0.0,
            "rot_absorbed": 0.0,
        }
