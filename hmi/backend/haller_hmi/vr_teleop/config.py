"""Live-tunable knobs for the VR teleoperator, with hard bounds.

Everything here can be written at runtime from the headset's settings panel
via a `config_update` message, which is the reference stack's model and the
right one: the numbers that decide how an arm feels cannot be tuned from a
desk, and an operator who has to take the headset off, edit a YAML file and
restart the backend will simply stop tuning them.

Every knob carries a hard range. A slider is a physical control on a machine
that can hurt someone, so "the panel sent 400" has to clamp rather than
apply. `BOUNDS` is the single place those ranges live; `apply_update` is the
single place a value from the wire can reach the dataclass.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QuestTeleopConfig:
    """Mapping + IK parameters for one VR teleop connection."""

    # ---- operator frame ----
    #: Which stance the operator is driving from — see `core.frames.STANCES`.
    #: Frozen into each engagement, so changing it mid-squeeze cannot
    #: teleport a held arm.
    stance: str = "behind"
    #: Divide out the headset's heading at engage so "controller forward"
    #: means "arm forward" wherever the operator is standing. Off only for
    #: diagnosing a suspected frame problem.
    yaw_on_engage: bool = True

    # ---- gains ----
    #: Translation gain, hand → tool. 1.0 is 1:1.
    scale_translation: float = 1.0
    #: Rotation gain. Above 1 because this arm's wrist_roll spans ±160°
    #: while a human wrist holding a controller comfortably covers maybe
    #: ±90 — at 1:1 the arm's roll extremes need a re-clutch to reach.
    scale_rotation: float = 1.6
    #: Both gains are multiplied by this while the precision button is held.
    precision_factor: float = 0.4

    # ---- filtering ----
    #: EMA on the incoming controller pose. 1.0 = raw passthrough, lower =
    #: smoother and laggier. Smooths sub-tick WebXR jitter before the IK
    #: sees it; the session's own one-pole filter sits downstream of this
    #: and does a different job (bounding the commanded joint rate).
    pose_filter_alpha: float = 0.8

    # ---- reach limits (the absorbing clutch) ----
    #: Max distance (m) the position target may run ahead of where the arm
    #: actually is; overshoot is absorbed. 0 disables. 0.15 is the reference
    #: stack's own SO-101 number — "smaller arm, smaller wall" against the
    #: 0.25 m it uses on the DK1 — and the only value here with recorded
    #: episodes behind it (46 / 29,500 frames). The 0.12 it replaces arrived
    #: with the port-time snapshot unmeasured, and absorbed enough of a fast
    #: reach that the arm stopped somewhere the hand was not.
    pos_reach_limit: float = 0.15
    #: Max angle (rad) the orientation target may run ahead. 0 disables.
    rot_reach_limit: float = 0.6

    # ---- IK ----
    lam_pos: float = 0.010
    lam0: float = 0.050
    w0: float = 0.020
    mu: float = 0.020
    lam_rot: float = 0.05
    #: Park the wrist when the orientation error angle (rad) exceeds this.
    #: Near the antipode the shortest-way direction of the quaternion error
    #: is unstable and chasing it is a 180° come-around. With rot_reach_limit
    #: on, the demand never gets there and this never fires; with it at 0
    #: (config.solo-raw.yaml) it is the only backstop. Past 3.15 it is off,
    #: since the error angle cannot exceed π. The park it arms is bounded to
    #: ik.decoupled_ik.PARK_MAX_SOLVES per excursion, so a demand held past
    #: the hold is eventually obeyed rather than refused forever.
    rot_err_hold: float = 2.2
    #: Per-solve step cap, degrees, split into the position joints and the
    #: wrist for the reason the reference stack splits them: one shared cap
    #: holds the wrist back during fine work while position is fine.
    max_dq_deg_pos: float = 3.0
    max_dq_deg_rot: float = 8.0
    #: Posture the IK biases toward at singularities; None = module default.
    rest_pose_deg: dict[str, float] | None = None

    # ---- workspace floor ----
    #: Minimum commanded height, metres in the arm's mount frame, for the
    #: fingertip and for the wrist. Deliberately NOT gated on the collision
    #: guard: these bound the DEMAND (an operator whose hand drops below the
    #: bench asks for a pose under it), and they have to keep working when
    #: the guard is switched off, which is exactly when nothing else is
    #: watching the bench. The guard, when on, is the precise backstop.
    min_tip_z: float = 0.005
    min_wrist_z: float = 0.035
    #: Master switch for the two floors above. Off is for a rig with no
    #: bench under the arm, not for "the floor annoys me".
    floor_enabled: bool = True

    #: Extra per-connection state the panel round-trips but the mapper does
    #: not read; kept so a `config_update` echo can carry it back to a
    #: second client without a special case.
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


#: (low, high) for every numeric knob a client may write. A key absent here
#: cannot be set from the wire at all — that is the allow-list, not an
#: oversight.
BOUNDS: dict[str, tuple[float, float]] = {
    "scale_translation": (0.1, 4.0),
    "scale_rotation": (0.1, 4.0),
    "precision_factor": (0.05, 1.0),
    "pose_filter_alpha": (0.05, 1.0),
    # 0 is legal on both reach limits and means "off" — the absolute mapping
    # the limits replaced, kept reachable so the two can be compared on the
    # bench rather than argued about.
    "pos_reach_limit": (0.0, 0.60),
    "rot_reach_limit": (0.0, 2.0),
    "lam_pos": (0.001, 0.20),
    "lam0": (0.0, 0.50),
    "w0": (0.001, 0.10),
    "mu": (0.0, 0.20),
    "lam_rot": (0.001, 1.0),
    # The top of this range is the "off" position, the way 0 is on the reach
    # limits above: the orientation error angle cannot exceed π, so a hold
    # past 3.15 can never fire (the reference CLI documents it the same way,
    # for its no-limit comparison demo). The floor is the DEFAULT
    # rot_reach_limit, which keeps the gate outside the demand's working
    # range at the shipped tuning — and that is all it is. It is NOT a
    # guarantee: rot_reach_limit is a live slider up to 2.0, apply_update
    # validates every key on its own, and rot_err_hold=0.6 with
    # rot_reach_limit=2.0 is a legal pair that parks the wrist during
    # ordinary reaching. Deliberately not cross-clamped — both are operator
    # tuning knobs, one slider silently moving another is worse than the
    # misconfiguration, and the park's own expiry
    # (ik.decoupled_ik.PARK_MAX_SOLVES) bounds that misconfiguration to a
    # stutter rather than a freeze.
    "rot_err_hold": (0.6, 3.20),
    # Upper bounds sized against the downstream limiters rather than against
    # comfort: the session's rate cap and `ArmHandle.send_goal`'s
    # max_speed_deg_s both sit below these, so a maxed slider makes the IK
    # stop being the binding constraint — it cannot make the arm faster than
    # the motion envelope allows.
    "max_dq_deg_pos": (0.25, 15.0),
    "max_dq_deg_rot": (0.25, 30.0),
    "min_tip_z": (-0.20, 0.40),
    "min_wrist_z": (-0.20, 0.40),
}

#: Boolean knobs a client may write.
BOOL_KEYS: tuple[str, ...] = ("yaw_on_engage", "floor_enabled")

#: String knobs, with their permitted values.
ENUM_KEYS: dict[str, tuple[str, ...]] = {
    "stance": ("behind", "mirror", "front"),
}


def apply_update(cfg: QuestTeleopConfig, update: dict) -> dict:
    """Write an operator's `config_update` onto `cfg`, clamped.

    Returns the subset that was actually applied, which the caller echoes
    back so the panel's sliders snap to the clamped value instead of
    silently disagreeing with the robot.
    """
    if not isinstance(update, dict):
        return {}
    applied: dict[str, float | bool | str] = {}
    for key, (lo, hi) in BOUNDS.items():
        if key not in update:
            continue
        try:
            value = float(update[key])
        except (TypeError, ValueError):
            logger.warning("config_update: %s=%r is not a number; ignoring",
                           key, update[key])
            continue
        if math.isnan(value):   # a slider bug, not an operator intent
            logger.warning("config_update: %s=NaN; ignoring", key)
            continue
        clamped = max(lo, min(hi, value))
        if clamped != value:
            logger.warning("config_update: %s=%.4f out of [%.3f, %.3f]; clamping",
                           key, value, lo, hi)
        setattr(cfg, key, clamped)
        applied[key] = clamped
    for key in BOOL_KEYS:
        if key in update:
            setattr(cfg, key, bool(update[key]))
            applied[key] = bool(update[key])
    for key, allowed in ENUM_KEYS.items():
        if key not in update:
            continue
        value = str(update[key])
        if value not in allowed:
            logger.warning("config_update: %s=%r not one of %s; ignoring",
                           key, value, allowed)
            continue
        setattr(cfg, key, value)
        applied[key] = value
    return applied
