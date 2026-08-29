"""QuestTeleoperator — the RETIRED per-frame converter. OFF the driven path.

As of the kit rewire, `/ws/teleop/vr/in` stores frames RAW on the session and
`HumanTeleopSession` solves them at its own 60 Hz tick through
`vr_teleop.kit_teleop.KitSideTeleop` (the vendored kit mapper + solver).
Nothing in the server constructs this class any more. It remains because its
mapper/IK layers are still exercised by `tests/vr_teleop/` and it documents
the pre-kit architecture; the audit's findings against it (solve-per-frame
seeded from the throttled committed pose, the wrist_roll TCP, the hand-rolled
FK) are why it is no longer driven.

This was the layer the reference stack calls `lerobot/bi_quest_teleop.py`: it
owns per-arm clutch state, runs the pose mapper and the IK, and hands the
result to whatever drives the robot. The difference is what it hands it to.
The reference emits a LeRobot action dict and a follower writes it; here the
consumer is `HumanTeleopSession`, which already owns everything a bench
session needs and which nothing about this port should bypass:

    * per-side authority — acquisition countdown, pose-match gate, the
      handover rate ramp,
    * the one-pole command filter and per-tick rate caps,
    * the bimanual collision guard,
    * the mode guard and E-STOP,
    * the dataset recorder's `action` column.

So the seam is a narrow one: this module only *produces the target*. It emits a `KeypointFrame` carrying a per-side
`joint_goal` in robot joint space — degrees, gripper in [0, 1] — and the
session does the rest. That is also what keeps it working identically
against a real arm and a MuJoCo one: it talks to `ArmHandle` and
`SimArmHandle` only through joint names and limits, which the two share.

One state instance per WebSocket connection, because the clutch anchors are
exactly as long-lived as the operator's connection.

Per-frame pipeline, per side:

  1. Tracking gate — an untracked hand drops its anchor and reports the side
     lost, which the session turns into a re-acquire.
  2. EMA pose filter on the controller pose (`pose_filter_alpha`).
  3. Precision button edge — re-anchor before changing the gains, so the
     accumulated delta is not reinterpreted under a new scale.
  4. Clutch edge — squeeze anchors (controller pose, tool pose, and the
     wrist anchor as the rotation pivot, with the heading correction
     applied); release disengages and freezes the arm.
  5. NOT-DRIVING re-anchor — while the session has not handed this side
     over, the anchor is refreshed every frame so the commanded pose stays
     exactly on the arm. This is what keeps the acquisition gate's error at
     zero through the countdown; without it the operator's hand motion
     during the countdown accumulates against a stationary arm and the gate
     they are waiting on drifts out of tolerance.
  6. Mapper → reach-limited tool target → IK step → joint goal.
  7. Haptic mix, published in `state()` for the client to vibrate with.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..so101_kinematics import _TIP_LOCAL, fk_frames
from .config import QuestTeleopConfig, apply_update
from .core import frames as frames_mod
from .core import quat
from .core.pose_mapping import ClutchPoseMapper
from .ik.decoupled_ik import SO101DecoupledIK
from .ik.model import POSE_JOINTS

logger = logging.getLogger(__name__)

SIDES: tuple[str, ...] = ("left", "right")


@dataclass
class _SideState:
    """One hand's clutch, filter and solver state."""

    mapper: ClutchPoseMapper
    solver: SO101DecoupledIK
    engaged: bool = False
    last_squeeze: bool = False
    last_precision: bool = False
    last_driving: bool = False
    pos_filt: np.ndarray | None = None
    quat_filt: np.ndarray | None = None
    #: EMA-smoothed 0..1 haptic intensity, published for the client.
    haptic: float = 0.0
    #: Last emitted joint goal — the fallback IK seed when the session has
    #: no committed pose yet (the very first frames of a session).
    last_goal: dict[str, float] = field(default_factory=dict)


class QuestTeleoperator:
    """WebXR frames → per-side joint goals. One per connection."""

    def __init__(self, session, arms, config: QuestTeleopConfig | None = None):
        self._session = session
        self._arms = arms
        self.config = config or QuestTeleopConfig()
        self._sides: dict[str, _SideState] = {}
        #: Diagnostics from the last converted frame, per side. Read by
        #: `state()` for the `ik_state` broadcast.
        self._diag: dict[str, dict] = {s: {} for s in SIDES}

    # ---- live configuration ---------------------------------------------

    def apply_config_update(self, update: dict) -> dict:
        """Write an operator's settings change. Returns what was applied.

        Gains and reach limits are re-read from `self.config` on every frame,
        so they take effect on the next one. The IK parameters live on the
        solver objects, so they are pushed across here.
        """
        applied = apply_update(self.config, update)
        if applied:
            for side in self._sides.values():
                self._push_ik_config(side.solver)
            logger.info("vr teleop config_update: %s", applied)
        return applied

    def _push_ik_config(self, solver: SO101DecoupledIK) -> None:
        cfg = self.config
        solver.lam_pos = cfg.lam_pos
        solver.lam0 = cfg.lam0
        solver.w0 = cfg.w0
        solver.mu = cfg.mu
        solver.lam_rot = cfg.lam_rot
        solver.rot_err_hold = cfg.rot_err_hold
        pos, rot = cfg.max_dq_deg_pos, cfg.max_dq_deg_rot
        solver.max_dq_deg = {
            "shoulder_pan": pos, "shoulder_lift": pos, "elbow_flex": pos,
            "wrist_flex": rot, "wrist_roll": rot,
        }

    # ---- per-frame conversion -------------------------------------------

    def convert(self, frame: dict) -> dict:
        """Raw WebXR frame → the frame `HumanTeleopSession.ingest_frame` eats:
        the clutch, per side, and a per-side `joint_goal` already solved in
        robot joint space. Handedness is applied here and nowhere else."""
        status = self._session.status()
        committed = status.get("goal_deg") or {}
        acquire = status.get("acquire") or {}

        head = frame.get("head")
        head_orientation = (head.get("orientation")
                            if isinstance(head, dict) else None)
        yaw = (frames_mod.head_yaw(head_orientation)
               if self.config.yaw_on_engage else None)

        stance = frame.get("stance")
        if stance not in frames_mod.STANCES:
            stance = self.config.stance

        sides: dict[str, dict | None] = {}
        squeezes: dict[str, bool] = {}
        for side in SIDES:
            raw = frame.get(side)
            squeeze = bool(isinstance(raw, dict) and raw.get("squeeze"))
            squeezes[side] = squeeze
            driving = ((acquire.get(side) or {}).get("authority") == "driving")
            sides[side] = self._convert_side(
                side, raw, squeeze, yaw, stance,
                committed.get(side) or {}, driving,
                self._limits(status.get(f"{side}_arm")),
            )

        dead_man = squeezes["left"] or squeezes["right"]
        return {
            "type": "keypoints",
            "ts_ms": int(frame.get("ts_ms", 0)),
            "dead_man": dead_man or bool(frame.get("dead_man", False)),
            "dead_man_sides": dict(squeezes),
            "left": sides["left"],
            "right": sides["right"],
        }

    def _limits(self, status_side_arm: str | None):
        if not status_side_arm:
            return None
        try:
            return self._arms[status_side_arm].joint_limits_deg
        except KeyError:
            return None

    def _side_state(self, side: str, limits: dict) -> _SideState:
        state = self._sides.get(side)
        if state is None:
            solver = SO101DecoupledIK(
                limits, q_rest_deg=self.config.rest_pose_deg)
            self._push_ik_config(solver)
            state = _SideState(
                mapper=ClutchPoseMapper(
                    R=frames_mod.stance_rotation(self.config.stance),
                    scale=self.config.scale_translation,
                    scale_rotation=self.config.scale_rotation,
                    pos_reach_limit=self.config.pos_reach_limit,
                    rot_reach_limit=self.config.rot_reach_limit,
                ),
                solver=solver,
            )
            self._sides[side] = state
        return state

    def _convert_side(
        self,
        side: str,
        raw: dict | None,
        squeeze: bool,
        yaw: float | None,
        stance: str,
        committed: dict[str, float],
        driving: bool,
        limits: dict[str, tuple[float, float]] | None,
    ) -> dict | None:
        if (not isinstance(raw, dict) or not raw.get("tracked", False)
                or limits is None or not committed):
            # Untracked, or nothing to anchor against: drop the anchor and
            # report the side lost, exactly as the camera path does.
            # Recovery goes back through acquisition, which re-anchors on
            # the next squeeze.
            state = self._sides.get(side)
            if state is not None:
                state.mapper.disengage()
                state.engaged = False
                state.last_squeeze = False
                state.last_driving = False
                state.pos_filt = None
                state.quat_filt = None
            self._diag[side] = {"tracked": False}
            return None

        state = self._side_state(side, limits)
        state.solver.limits_deg = dict(limits)
        trigger = float(raw.get("trigger", 0.0))
        gripper = max(0.0, min(1.0, 1.0 - trigger))

        # The seed is the session's COMMITTED pose, not this module's own
        # integrated one. That matters: the session's committed pose is what
        # the arm was actually told to do — after the rate caps, the
        # acquisition ramp and the collision guard have each had their say —
        # so seeding from it means the IK model can never quietly diverge
        # from the robot and hand back a lurch when a clamp releases. It also
        # gives the reach limit the right thing to measure against: "how far
        # ahead of the ARM is the target", not "how far ahead of my own
        # optimistic integration".
        seed = {j: float(committed[j]) for j in POSE_JOINTS if j in committed}
        if not seed:
            seed = dict(state.last_goal)

        if not squeeze:
            state.mapper.disengage()
            state.engaged = False
            state.last_squeeze = False
            state.last_driving = False
            state.pos_filt = None
            state.quat_filt = None
            self._diag[side] = {"tracked": True, "engaged": False,
                                "haptic": state.haptic}
            state.haptic *= 0.6
            # Open grip: ask for exactly where the arm already is. Zero gate
            # error, so the moment the operator squeezes there is nothing to
            # match — the countdown is the only wait.
            goal = dict(seed)
            goal["gripper"] = gripper
            state.last_goal = dict(goal)
            return {"joint_goal": goal, "confidence": 1.0}

        # ---- pose, filtered ----
        pos_raw = np.asarray(raw["position"], dtype=float)
        quat_raw = quat.from_xyzw(raw["orientation"])
        alpha = float(self.config.pose_filter_alpha)
        if state.pos_filt is None or state.quat_filt is None:
            state.pos_filt = pos_raw.copy()
            state.quat_filt = quat_raw.copy()
        else:
            state.pos_filt = (1.0 - alpha) * state.pos_filt + alpha * pos_raw
            q_in = quat.hemisphere_align(quat_raw, state.quat_filt)
            state.quat_filt = quat.normalize(
                (1.0 - alpha) * state.quat_filt + alpha * q_in)
        pos, orient = state.pos_filt, state.quat_filt

        # ---- gains, and the precision modifier ----
        precision = bool(raw.get("precision", False))
        factor = self.config.precision_factor if precision else 1.0
        state.mapper.scale = self.config.scale_translation * factor
        state.mapper.scale_rotation = self.config.scale_rotation * factor
        state.mapper.pos_reach_limit = self.config.pos_reach_limit
        state.mapper.rot_reach_limit = self.config.rot_reach_limit

        # Re-anchor whenever the engagement is not currently steering the
        # arm. Four cases fold into one rule:
        #   * rising clutch edge — the ordinary engage,
        #   * the precision button changing — otherwise the delta already
        #     accumulated gets reinterpreted under the new gain, which is a
        #     target snap,
        #   * the session has not handed this side over yet — see the module
        #     docstring; this is what holds the acquisition gate at zero
        #     error through the countdown,
        #   * the FIRST frame after it has. The session flips to DRIVING
        #     inside its own 60 Hz loop and this converter only learns about
        #     it on the next frame, so without this the very first driven
        #     frame carries however far the hand travelled in between — a
        #     centimetre or so at a normal reach speed, applied as a step.
        #     Anchoring once more on the rising edge makes handover start
        #     from zero delta by construction, which is the same property
        #     the countdown relies on.
        needs_anchor = (
            not state.engaged
            or not state.last_squeeze
            or precision != state.last_precision
            or not driving
            or not state.last_driving
        )
        state.last_squeeze = True
        state.last_precision = precision
        state.last_driving = driving

        if needs_anchor:
            self._anchor(state, seed, pos, orient, yaw, stance)

        tool_pos, tool_quat = state.solver.fk(seed)
        out = state.mapper.target(pos, orient, tool_pos, tool_quat)
        if out is None:                     # cannot happen: we just anchored
            goal = dict(seed)
            goal["gripper"] = gripper
            return {"joint_goal": goal, "confidence": 1.0}
        target_pos, target_quat = out
        target_pos = self._apply_floor(target_pos, target_quat, seed)

        solved = state.solver.solve(target_pos, target_quat, seed)
        goal = dict(solved)
        goal["gripper"] = gripper
        state.last_goal = dict(goal)

        self._diag[side] = self._update_haptic(state, driving)
        return {"joint_goal": goal, "confidence": 1.0}

    def _anchor(self, state: _SideState, seed: dict[str, float],
                pos: np.ndarray, orient: np.ndarray,
                yaw: float | None, stance: str) -> None:
        """Bind hand↔arm at the current pose, in the operator's own frame."""
        state.mapper.set_R(frames_mod.stance_rotation(stance, yaw))
        tool_pos, tool_quat = state.solver.fk(seed)
        # The wrist anchor as the rotation pivot: a pure hand twist then
        # swings the gripper about the wrist the way the operator's own hand
        # does, instead of spinning the tool on the spot and dragging the
        # whole arm round after it.
        pivot = state.solver.wrist_anchor(seed)
        state.mapper.engage(pos, orient, tool_pos, tool_quat, pivot_armbase=pivot)
        state.engaged = True

    def _apply_floor(self, target_pos: np.ndarray, target_quat: np.ndarray,
                     seed: dict[str, float]) -> np.ndarray:
        """Raise a commanded tool position so the demand stays above the bench.

        Closed-form rather than the solve-and-retry loop this replaces: given
        the target tool frame, the fingertip and the wrist anchor it implies
        are both a fixed offset away, so the smallest lift that clears both
        floors can be computed directly. Bounding the DEMAND is the point —
        an operator whose hand drops below the bench asks for a pose under
        it, and if that ask is left intact then every step toward it makes
        the guard's floor slack worse, at which point the guard scales the
        whole step and the arm freezes solid, sideways motion included.
        Clamping turns "push too low" into "slide along the bench".
        """
        cfg = self.config
        if not cfg.floor_enabled:
            return target_pos
        R = quat.to_mat(target_quat)
        # Tool origin → fingertip, and tool origin → wrist anchor, both in
        # the tool frame, so the target's own orientation places them.
        tip_offset = R @ _TIP_LOCAL
        f = fk_frames(seed)
        anchor_in_tool = f.tool_R.T @ (f.wrist_pos - f.tool_pos)
        wrist_offset = R @ anchor_in_tool
        lift = max(
            0.0,
            cfg.min_tip_z - (target_pos[2] + tip_offset[2]),
            cfg.min_wrist_z - (target_pos[2] + wrist_offset[2]),
        )
        if lift <= 0.0:
            return target_pos
        out = np.array(target_pos, dtype=float, copy=True)
        out[2] += lift
        return out

    # ---- haptics + diagnostics ------------------------------------------

    def _update_haptic(self, state: _SideState, driving: bool) -> dict:
        """Mix the IK's trouble signals into one 0..1 buzz, and report them.

        The four are gated, not summed: each has a dead zone below which it
        means nothing, and above it the operator should feel the WORST one
        rather than an average that hides it.
        """
        s = state.solver
        m = state.mapper
        # Joints being clipped into their stops this step, degrees.
        i_limit = _gate(s.last_limit_pressure, *_LIMIT_PRESSURE_GATE)
        # Reach limit absorbing travel: the "wall" at the workspace edge.
        # Gated high, because the limit is always partly engaged during fast
        # motion and only means something once it is saturating.
        i_reach = _gate(m.last_pos_absorbed, 0.85, 1.0)
        i_twist = _gate(m.last_rot_absorbed, 0.85, 1.0)
        # Approaching the position singularity. Gated very high: the damping
        # ramp opens well before the arm is in trouble, and buzzing through
        # the whole ramp region would be a constant hum.
        i_singular = _gate(s.last_singularity_proximity, 0.90, 1.0)
        # The 1-DoF orientation deficit (last_orient_residual) is deliberately
        # NOT in this mix. On a 5-DoF arm some unreachable twist is the
        # STANDING state of ordinary driving — tool yaw belongs to the
        # shoulder — so blending it here turns a structural fact into a
        # permanent tremble at 20 Hz. It is still reported below; the client
        # gives it a dedicated one-shot cue and HUD line (ikHapticCues),
        # which is what the operator doc promises: one firm buzz on crossing
        # into the deficit, then no nagging.
        raw = max(i_limit, i_reach, i_twist, i_singular)
        state.haptic = 0.6 * state.haptic + 0.4 * (raw if driving else 0.0)
        return {
            "tracked": True,
            "engaged": True,
            "driving": bool(driving),
            "haptic": float(state.haptic),
            "limit_pressure_deg": float(s.last_limit_pressure),
            "pos_err_m": float(s.last_pos_err_norm),
            "sigma_min": float(s.last_sigma_min),
            "singularity": float(s.last_singularity_proximity),
            "orient_residual": float(s.last_orient_residual),
            "pos_absorbed": float(m.last_pos_absorbed),
            "rot_absorbed": float(m.last_rot_absorbed),
        }

    def state(self) -> dict:
        """The `ik_state` payload broadcast back to the headset."""
        return {
            "type": "ik_state",
            "config": self.config.to_dict(),
            "sides": {s: dict(self._diag.get(s) or {}) for s in SIDES},
        }


#: Dead zone and saturation for joint-limit pressure, degrees. Named because
#: something outside this file depends on the ceiling: a wrist parked by the
#: antipode gate takes no step, so the joint clamp measures nothing, and
#: `ik.decoupled_ik.PARK_LIMIT_PRESSURE_DEG` (20.05) is reported in its place
#: PURELY to saturate this ramp. Raise the ceiling past 20.05 and the buzz
#: goes silent at exactly the pose where the wrist stopped obeying.
_LIMIT_PRESSURE_GATE = (0.5, 4.0)


def _gate(value: float, lo: float, hi: float) -> float:
    """Dead-zone linear ramp: 0 at or below `lo`, 1 at or above `hi`."""
    if hi <= lo:
        return 0.0
    return float(min(1.0, max(0.0, (float(value) - lo) / (hi - lo))))
