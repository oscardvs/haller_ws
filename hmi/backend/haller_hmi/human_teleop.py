"""Bimanual teleop session — the safety core the headset drives through.

This is the sibling of `teleop.TeleopSession` (leader/follower). Where that
session reads positions off a physical leader arm at 60 Hz, this one stores
RAW WebXR frames off a WebSocket — per-side controller pose + buttons,
latest-wins — and solves them ITSELF, once per tick, through one
`vr_teleop.kit_teleop.KitSideTeleop` per driven side (the vendored
vr-teleop-kit mapper + solver). That is the kit's own loop shape: the
consumer solves at its cadence on the latest frame and integrates its own
open-loop qpos, so no downstream limiter's withheld degrees ever feed back
into the mapper's reach accounting as operator over-drive.

Everything a bench session needs still lives here: per-side authority, the
collision guard, the mode guard, E-STOP, and the recorder's `action` column.
What deliberately does NOT exist on the driven path any more: the one-pole
LPF and the per-tick rate cap. The kit ships neither; its governors are the
solver's per-joint per-solve dq caps, the joint limits, and servo physics —
and while a side is DRIVING, this session writes the adapter's action in
full, with an unbounded speed budget. Both filters remain, byte-for-byte, on
the `request_home` slew path, which is not a kit path.

Otherwise the lifecycle and safety semantics match `TeleopSession` exactly.

Session state machine:
    IDLE → (start)        → ARMED
    ARMED → (first frame) → TRACKING
    any → (stop / E-STOP) → IDLE

TRACKING / ACQUIRING / DRIVING are *derived* from the two per-side authorities
below rather than set directly — see `_derive_state`.

Authority transfer (per side):
    HELD → (clutch closed, side trackable) → ACQUIRING
    ACQUIRING → (countdown elapsed) → DRIVING
    ACQUIRING/DRIVING → (clutch released, or side lost) → HELD

The robot only ever starts following through ACQUIRING, whether the session is
seconds old or the operator's hand just came back into frame. That is the whole
point: a hand re-entering frame is a cold start, and used to be a lurch because
the two paths differed.
"""
from __future__ import annotations

import enum
import logging
import math
import threading
import time
from dataclasses import dataclass

from .arm import ArmManager
from .safety import Mode
from .tick import ProducerConflict, TickBus
from .vr_teleop.config import QuestTeleopConfig
from .vr_teleop.core.frames import STANCES

logger = logging.getLogger(__name__)


def _default_side_teleop_factory(joint_limits_deg, config, *, urdf_path=None):
    """One side's kit-faithful tracker. Imported lazily: the vendored kit
    carries a mujoco model, which a session that never starts (and every
    consumer of this module's constants) should not pay for."""
    from .vr_teleop.kit_teleop import KitSideTeleop
    return KitSideTeleop(joint_limits_deg, config, urdf_path=urdf_path)

# ---- acquisition ------------------------------------------------------
#
# Closing the dead-man used to hand the robot over instantly, from wherever the
# operator's hand happened to be — which is essentially never where the robot's
# arm is. Worse, the smoothing state kept slewing toward the operator's pose
# the whole time the clutch was OPEN, so the first commit after engaging was a
# single step command to the operator's current pose. The rate cap had already
# been spent against an arm that was not moving; on hardware that is a step
# input to the servos, i.e. maximum velocity.
#
# Two mechanisms replace that instant, and they do different jobs:
#
#   the COUNTDOWN gives the operator warning and a window to abort,
#   the RAMP      bounds the speed at which whatever error remains is closed.
#
# Only the ramp is load-bearing for safety — it holds unconditionally, needs no
# cooperation from the operator, and cannot be satisfied by accident. The
# countdown makes engagement deliberate and legible.
#
# There was a third: a per-joint MATCH GATE that refused handover until the
# commanded pose was within tolerance of the measured one. It went with the
# pose-reconstruction input path it existed for. That path GUESSED the
# operator's joint angles from webcam landmarks, so the commanded pose could
# sit tens of degrees from the arm at the moment of handover, and the gate was
# what stopped that becoming a lurch. The headset path anchors instead: squeezing
# the grip binds the target to wherever the arm already IS, and the mapper
# re-anchors every frame until the side is DRIVING — the error is zero by
# construction, so a gate on it can only ever be satisfied. What the gate
# actually bounded, the ramp still bounds.

#: Countdown from the clutch closing to handover. ZERO: the clutch engages on
#: the rising edge, the way `SO101QuestTeleoperator._update_arm` does and the
#: way 46 recorded episodes were driven.
#:
#: The paragraph above already contains the argument for this. The headset
#: path anchors — squeezing the grip binds the target to wherever the arm
#: already IS — so the error at handover is zero BY CONSTRUCTION, and what a
#: countdown filters is therefore only "an accidental grip". That is not worth
#: what it costs: it is charged on EVERY engage, including the dozens of
#: deliberate re-clutches it takes to ratchet across a workspace, and every
#: tracking blip past the grace window bills it again. The operator reads that
#: as an arm that does not follow their hand.
#:
#: The kit has no equivalent and never needed one. What actually bounds the
#: rig is unchanged and still unconditional: `motion.max_speed_deg_s`, the
#: per-joint limits, RATE_CAP_DEG_S, the workspace floors and the motion
#: envelope.
ACQUIRE_MS = 0.0
#: Kept at zero for the same reason, and kept as a name because the wire and
#: the tests both read it. Handover fires at max(ACQUIRE_MS, MATCH_DWELL_MS).
MATCH_DWELL_MS = 0.0
#: Per-side tracking-loss grace. Quest controller tracking flickers — hands at
#: the FOV edge, occlusions — and a long grace is SAFE by construction here:
#: during the gap no new targets arrive (the arm holds), and on recovery the
#: next frame re-anchors at the hand's new position, so there is no stale-goal
#: jump for the grace to protect against.
FRAME_AGE_MS_LOSS = 700.0
#: What `status()["clutch"]["reason"]` says when nothing is squeezed. A
#: resting state, not a fault — the grip is armed and simply not held — which
#: is why it is named for the control rather than for the absence. Mirrored in
#: TypeScript as `ClutchReason`.
CLUTCH_RESTING = "vr_grip_mode"
#: Joint speed the first instant of DRIVING is allowed to command. Retired
#: with the ramp below; retained because the constructor and the status block
#: still name it.
ACQUIRE_RATE_DEG_S = 20.0
#: Time over which the cap returned to the session's normal rate limit. ZERO:
#: the ramp existed to absorb "a handover the anchor somehow did not zero",
#: and on the headset path the anchor always zeroes it — the mapper re-anchors
#: every frame until the side drives. A 1.5 s climb from 20 deg/s on every
#: clutch is the single biggest reason this rig tracks a hand differently from
#: the kit, and the kit ships without it.
ACQUIRE_RAMP_MS = 0.0
#: The session's normal joint-rate ceiling, DEGREES PER SECOND.
#:
#: Expressed as a rate and converted per tick, not stored as degrees-per-tick:
#: a per-tick constant is calibrated for exactly one cadence and lies at every
#: other. `hz` is a field of `POST /teleop/human/start`, so this was reachable
#: — at the hard-coded 4 deg/tick it meant 240 deg/s at hz=60 but 80 deg/s at
#: hz=20, silently taking over the binding-speed-limit role that belongs to
#: motion.max_speed_deg_s.
#:
#: It also made `_ramp_cap` disagree with itself: its floor was already a
#: proper `ACQUIRE_RATE_DEG_S * period` conversion while its ceiling was the
#: raw per-tick number, so the acquisition ramp spanned 12:1 at hz=60 and 2:1
#: at hz=10 — and invariant 2 calls that ramp load-bearing. Both ends are
#: rates now and the ramp is the same ratio at every cadence.
#:
#: 240 deg/s IS 4 deg/tick at 60 Hz exactly, so this changes nothing at the
#: only cadence ever driven on hardware. `lpf_tau_s` is the model: a physical
#: constant, converted at the point of use.
RATE_CAP_DEG_S = 240.0

#: Duration of the in-session home slew, seconds — the kit's rest ramp
#: (`rest_ramp_duration_s`, so101_quest_teleop.py:184/640-652): a
#: fixed-DURATION joint-space lerp from wherever the arm stands, every
#: joint arriving together. Duration-shaped, not rate-shaped, because a
#: rate-shaped slew moves at whatever the cap is the moment the filter is
#: off — with lpf_tau_s 0 that was the full 240 deg/s, a park maneuver
#: moving at teleop speed.
HOME_RAMP_S = 2.0

#: Bounds on the session tick rate a caller may request.
#:
#: The floor is set by the acquisition ramp, not by speed: ACQUIRE_RAMP_MS is
#: 1.5 s, so below 10 Hz the ramp gets fewer than 15 ticks and stops being a
#: ramp — and invariant 2 calls it load-bearing. The ceiling is the Feetech
#: bus: past ~120 Hz the round trips cannot be served and the loop only burns
#: CPU discovering that.
#:
#: Note this bound is WEAKER than it needed to be before the rate cap became a
#: rate. While the cap was degrees-per-TICK, `hz` silently reconfigured the
#: speed limit and the safety-stop budget; now it changes tick RESOLUTION and
#: nothing else, so the bound is about the ramp having enough samples rather
#: than about the envelope moving underneath the operator.
MIN_SESSION_HZ = 10.0
MAX_SESSION_HZ = 120.0

#: How many consecutive failed ticks the 60 Hz loop tolerates before it stops
#: the session outright. A failed tick sleeps 50 ms and retries, which rides
#: through transient bus glitches; a PERMANENT fault (a dead arm handle) would
#: otherwise spin forever, "running" the whole time. 50 x 50 ms ≈ 2.5 s.
MAX_CONSECUTIVE_TICK_ERRORS = 50


class SideAuthority(str, enum.Enum):
    """Whether one arm is being written to, and if not, how far off it is."""

    HELD = "held"
    ACQUIRING = "acquiring"
    DRIVING = "driving"


@dataclass
class _SideAcquire:
    """Per-side authority plus everything the operator needs to see about it."""

    authority: SideAuthority = SideAuthority.HELD
    #: Countdown origin — when ACQUIRING began.
    since_perf: float | None = None
    #: Ramp origin — when DRIVING began.
    driving_since_perf: float | None = None
    #: Why this side is where it is. The state an operator is most likely to be
    #: stuck in is also the one they are least able to diagnose, since a
    #: countdown that silently restarts looks identical to one that is frozen.
    reason: str = "clutch_open"

    def release(self, reason: str) -> None:
        self.authority = SideAuthority.HELD
        self.since_perf = None
        self.driving_since_perf = None
        self.reason = reason


class HumanState(str, enum.Enum):
    IDLE = "idle"
    ARMED = "armed"
    TRACKING = "tracking"
    ACQUIRING = "acquiring"
    DRIVING = "driving"


@dataclass
class _SessionConfig:
    #: Arm id driven by each hand, or None for a side this session has no
    #: arm for. Exactly one side may be None — a session with neither would
    #: be a loop that writes to nothing, which is a bug worth refusing at
    #: start() rather than discovering on the bench.
    left_arm: str | None
    right_arm: str | None
    hz: float = 60.0

    def arm_for(self, side: str) -> str | None:
        return self.left_arm if side == "left" else self.right_arm

    @property
    def sides(self) -> tuple[str, ...]:
        """The sides this session actually drives."""
        return tuple(s for s in ("left", "right") if self.arm_for(s))


@dataclass
class JointStep:
    """One joint's outcome for one tick of the commit loop.

    `target` is what the converter asked for, in degrees (the gripper's
    [0,1] input is already scaled onto its calibrated range). `committed`
    is what was actually written. `reason` explains any difference.
    """
    target: float | None
    committed: float
    reason: str   # "ok" | "rate_capped" | "clamped" | "held"


class HumanTeleopSession:
    """One global session. Mutually exclusive with leader/follower TeleopSession."""

    def __init__(
        self,
        arms: ArmManager,
        *,
        hz_override: float | None = None,
        frame_age_ms_loss: float = FRAME_AGE_MS_LOSS,
        ws_disconnect_grace_s: float = 5.0,
        acquire_ms: float = ACQUIRE_MS,
        match_dwell_ms: float = MATCH_DWELL_MS,
        acquire_rate_deg_s: float = ACQUIRE_RATE_DEG_S,
        acquire_ramp_ms: float = ACQUIRE_RAMP_MS,
        lpf_tau_s: float = 0.100,
        collision_guard=None,
        tick_bus: TickBus | None = None,
        sample_hz: float | None = None,
        vr_config: QuestTeleopConfig | None = None,
        side_teleop_factory=None,
    ):
        self._arms = arms
        # The live-tunable VR mapping config, SHARED with the teleop socket:
        # the socket writes `config_update`s onto this instance and the
        # per-side adapters below read it on every tick, so a slider moved in
        # the headset reaches the running session without a restart.
        self._vr_cfg = vr_config if vr_config is not None else QuestTeleopConfig()
        # (joint_limits_deg, config, *, urdf_path=None) -> KitSideTeleop-like.
        # Injectable so session tests can pin the SESSION contract with a
        # deterministic tracker; production uses the vendored kit adapter.
        self._side_factory = (side_teleop_factory if side_teleop_factory
                              is not None else _default_side_teleop_factory)
        # One kit-faithful tracker per driven side, built at start().
        self._kit: dict[str, object] = {}
        # THE tick bus. Constructed here rather than handed in from the
        # lifespan because this object is built at module scope in server.py
        # while telemetry and the recorder are lifespan locals — so the bus
        # cannot be a lifespan local without reordering construction. Owning
        # it here keeps that line unchanged; consumers read
        # `human_teleop.tick_bus`.
        self.tick_bus = tick_bus if tick_bus is not None else TickBus()
        self._tick_token = None
        # Where `base` in a published sample comes from. Injected after
        # construction (`set_base_source`) because ROS is a lifespan object
        # and this session is not.
        self._base_source = None
        # Sampling rate for the published tick, in HERTZ — not a divisor.
        #
        # A count of ticks is calibrated for exactly one cadence and lies at
        # every other: `read_divisor = 2` means 30 Hz at hz=60 and 5 Hz at
        # hz=10, so the recorder's fps would silently follow the control rate.
        # That is the constant class `_rate_cap_deg_per_tick` was just fixed
        # for, and the house rule this codebase adopted from that fix. The
        # divisor is DERIVED from this rate and the loop's real period, so the
        # sampler targets the same real rate at any control cadence.
        #
        # None samples every tick, which is what the three-sampler world
        # effectively did for the commit loop and changes nothing on its own.
        self._sample_hz = sample_hz
        # Bimanual collision/workspace guard (collision.CollisionGuard), or
        # None to run unguarded. Duck-typed so tests can inject a stub; the
        # session only calls filter_step()/clearance() and reads .cfg.margin_m.
        self._collision = collision_guard
        self._collision_last: dict | None = None
        self._lock = threading.Lock()
        self._state: HumanState = HumanState.IDLE
        self._cfg: _SessionConfig | None = None
        self._started_at: float | None = None
        self._last_error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._latest_frame_ts_ms: int = 0
        self._latest_arrival_perf: float = 0.0
        self._dead_man: bool = False
        # Per-side dead-man. A Quest controller's squeeze is per hand, and a
        # grip held on one controller must not hand over the arm the *other*
        # hand happens to be waving around. A frame carrying only the global
        # boolean mirrors it onto both sides.
        self._dead_man_sides: dict[str, bool] = {"left": False, "right": False}
        # The latest RAW controller frame per side (latest-wins, the kit's
        # storage discipline) plus the head pose and stance it arrived with.
        # These are STORED, not solved: solving happens once per tick in the
        # loop, on whatever is newest then.
        self._ctrl: dict[str, dict | None] = {"left": None, "right": None}
        self._head_orientation = None
        self._stance_frame: str | None = None
        # Sides asked to slew HOME while their clutch is open (the headset's
        # hold-the-left-stick reset). Cleared the moment a side starts
        # driving: the operator's hand always outranks a parked reset.
        self._home_req: dict[str, bool] = {"left": False, "right": False}
        # The kit-style fixed-duration lerp behind an accepted home request:
        # {"start": committed-at-request, "t0": perf} per side, None when no
        # ramp is running. See HOME_RAMP_S.
        self._home_ramp: dict[str, dict | None] = {"left": None, "right": None}
        self._committed_left: dict[str, float] = {}
        self._committed_right: dict[str, float] = {}
        self._steps_left: dict[str, JointStep] = {}
        self._steps_right: dict[str, JointStep] = {}
        self._hz_override = hz_override
        self._rate_cap_deg_s = RATE_CAP_DEG_S
        # Set from cfg.hz when the loop starts. The default matters only
        # for `_smooth_step` called outside a running session.
        self._period = 1.0 / 60.0
        # Smoothing time constant for the one-pole filter, config
        # motion.lpf_tau_s. HOME-SLEW ONLY: the driven path is the kit's and
        # carries no output filter at all — see the module docstring.
        self._lpf_tau_s = lpf_tau_s
        # T8: tracking-loss + WS disconnect grace
        self._frame_age_ms_loss = frame_age_ms_loss
        self._ws_disconnect_grace_s = ws_disconnect_grace_s
        self._ws_disconnected_at_perf: float | None = None
        # Per-arm last-frame timestamps (perf_counter), for tracking-loss.
        self._last_left_perf: float = 0.0
        self._last_right_perf: float = 0.0
        self._peers: list = []
        self._clutch_reason: str = CLUTCH_RESTING
        # Acquisition: per-side authority, and the countdown that gates it.
        self._acquire_ms = acquire_ms
        self._match_dwell_ms = match_dwell_ms
        self._acquire_rate_deg_s = acquire_rate_deg_s
        self._acquire_ramp_ms = acquire_ramp_ms
        self._acq: dict[str, _SideAcquire] = {
            "left": _SideAcquire(), "right": _SideAcquire(),
        }
        # Set exactly when the side's adapter and the arm may disagree about
        # where the arm is — today: the side just entered the session, a
        # request_home slew wrote the arm behind the adapter's back, or the
        # adapter itself ran AHEAD of the arm (see _kit_ran_ahead). The
        # pending side's adapter is re-seeded from a real read
        # (`seed_from_observed`) before it may drive again; everywhere else
        # the adapter's qpos is OPEN LOOP and never re-read mid-teleop, which
        # is the kit's discipline and the point of the port.
        self._reseed_pending: dict[str, bool] = {"left": True, "right": True}
        # The one divergence the kit structurally cannot have and this
        # session can: an ENGAGED adapter integrating on ticks where nothing
        # is written — a configured acquisition countdown, or a
        # demote-and-recover with the grip still squeezed. The kit's rising
        # edge IS the write gate, so its integrator is never unwritten; here
        # the flag records that it happened, and the side re-seeds (and so
        # re-anchors, zero delta by construction) before its next driven
        # tick. At the shipped acquire_ms=0 this never fires and no extra
        # read is ever taken.
        self._kit_ran_ahead: dict[str, bool] = {"left": False, "right": False}
        self._seen_frame: bool = False

    # ---- public API ------------------------------------------------------

    @property
    def state(self) -> HumanState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state is not HumanState.IDLE

    def attach_peer(self, peer) -> None:
        """Register a sibling teleop session — at start time, if any registered
        peer reports running=True, this session refuses to start (HTTP 409 in
        the route)."""
        self._peers.append(peer)

    @staticmethod
    def _sample_divisor(hz: float, sample_hz: float | None) -> int:
        """Ticks per published sample, from two RATES.

        The house rule this codebase adopted after `_rate_cap_deg_per_tick`:
        a constant counted in ticks is calibrated for exactly one cadence and
        lies at every other. A configured divisor of 2 would mean 30 Hz of
        samples at hz=60 and 5 Hz at hz=10 — and `fps` is frozen from the
        measured sample rate, so the recorder's declared frame rate would
        quietly follow the control rate. Expressed as a rate, the sampler
        targets the same real cadence at any `hz`.

        Never below 1: a sample rate above the control rate cannot be served
        by decimating it, and rounding to 0 would divide by zero every tick.
        """
        if not sample_hz or sample_hz <= 0:
            return 1
        return max(1, round(hz / sample_hz))

    def set_base_source(self, ros) -> None:
        """Where `base` in a published sample comes from.

        Injected rather than constructed because ROS is a lifespan object and
        this session is built at module scope. Anything with a `snapshot()`
        answering linear/angular/odom/scan_min_range will do, which is what
        telemetry already reads.
        """
        self._base_source = ros

    def _base_block(self) -> dict:
        src = self._base_source
        if src is None:
            return {}
        try:
            snap = src.snapshot()
        except Exception:
            logger.warning("base snapshot failed for this tick", exc_info=True)
            return {}
        return {
            "linear": snap.linear,
            "angular": snap.angular,
            "odom": dict(snap.odom),
            "scan_min_range": snap.scan_min_range,
        }

    def idle_sample(self) -> dict | None:
        """One tick's fields while NO session is driving. The IdleSampler's source.

        Lives here rather than in the lifespan because this class already owns
        what it takes to sample an arm, and `tick.py` has to stay stdlib-only
        so the rollout child can read `safety.MIN_RATE_FRACTION` without
        pulling lerobot in. The mount is then one line.

        Returns None while a session runs: the commit loop owns the tick then,
        and `publish_once` would refuse anyway. Returning None means the bus is
        never even asked, which keeps the handover a fact about this method
        rather than a race the bus has to arbitrate.

        `goal_deg` is the last committed target, which is what `status()`
        reports while idle too. Nothing is being commanded, and an idle sample
        cannot reach a dataset row — arming freezes the arm set and a recorder
        whose teleop has stopped falls back to idle — so this is a readout, not
        an instruction.
        """
        if self.running:
            return None
        arms, errors = self._sample_arms()
        # Stamped after the read, as in the loop: a clock taken before a bus
        # round trip is stale by however long the round trip took.
        t_mono = time.perf_counter()
        t_unix = time.time()
        with self._lock:
            goal = {"left": dict(self._committed_left),
                    "right": dict(self._committed_right)}
        return {
            "t_mono": t_mono,
            "t_unix": t_unix,
            "arms": arms,
            "arm_errors": errors,
            "goal_deg": goal,
            "base": self._base_block(),
            "degraded": bool(errors),
        }

    def _sample_arms(self) -> tuple[dict, dict]:
        """One state read per arm, for the moment this tick owns.

        EVERY arm the manager has, not only the session's. The cockpit shows
        arms this session is not driving, and once telemetry consumes the bus
        instead of reading for itself, an arm nobody sampled is an arm nobody
        can see. One sampler means one sampler for all of them.

        A read that fails puts the arm in `errors` and leaves it OUT of
        `arms`, so a consumer meets a hole rather than a plausible number
        standing in for a measurement that did not happen. That is invariant 9
        at the point where the read is taken: mechanism 2's tick 0 decodes to
        -180.0 deg, not to zero, so a substituted value is not a small error.
        """
        arms: dict[str, dict] = {}
        errors: dict[str, str] = {}
        # `.keys()`, and it must stay `.keys()`. ArmManager is NOT a dict: it
        # has __getitem__/keys/values and no __iter__, so `for x in manager`
        # falls back to the legacy integer protocol and raises
        # `KeyError: unknown arm id 0` on the first tick — every tick, until
        # MAX_CONSECUTIVE_TICK_ERRORS stops the session about 2.5 s after it
        # starts. Ruff's SIM118 asks for the dict form here and is wrong about
        # this receiver; the rule is sound one type over.
        for arm_id in self._arms.keys():  # noqa: SIM118  (ArmManager, not a dict)
            try:
                snap = self._arms[arm_id].state_snapshot()
            except Exception as e:  # noqa: BLE001  (any bus fault is a hole)
                errors[arm_id] = str(e)
                continue
            # Verbatim. Projecting here would make this loop the thing that
            # decides which per-joint keys exist, and telemetry has a test
            # saying nothing between the handle and a subscriber may do that.
            arms[arm_id] = snap
        return arms, errors

    def _reset_clutch_state(self) -> None:
        """Clear every clutch transient. Caller holds the lock.

        Called from both ends of the lifecycle. At stop() this matters as much
        as at start(): nothing is being asked for any more, so `engaged` and
        `reason` have to clear — an ended session that still advertises
        `{"engaged": true, "reason": "engaged"}` beside `"state": "idle"` is
        telling the operator the arms are live when they are not.
        """
        self._dead_man = False
        self._dead_man_sides = {"left": False, "right": False}
        self._clutch_reason = CLUTCH_RESTING

    def status(self) -> dict:
        with self._lock:
            cfg = self._cfg
            now = time.perf_counter()
            left_age = (now - self._last_left_perf) * 1000.0 if self._last_left_perf else None
            right_age = (now - self._last_right_perf) * 1000.0 if self._last_right_perf else None
            return {
                "running": self.running,
                "state": self._state.value,
                "left_arm": cfg.left_arm if cfg else None,
                "right_arm": cfg.right_arm if cfg else None,
                "started_at": self._started_at,
                "last_error": self._last_error,
                "tracking": {
                    "left":  {
                        "age_ms": left_age,
                        "lost":   left_age is not None and left_age > self._frame_age_ms_loss,
                    },
                    "right": {
                        "age_ms": right_age,
                        "lost":   right_age is not None and right_age > self._frame_age_ms_loss,
                    },
                },
                "goal_deg": {
                    "left":  dict(self._committed_left),
                    "right": dict(self._committed_right),
                },
                # Additive diagnostic block. `goal_deg` above is the recorder's
                # `action` column and must keep its plain joint -> float shape.
                "joints": {
                    "left":  self._steps_as_dict(self._steps_left),
                    "right": self._steps_as_dict(self._steps_right),
                },
                # Whatever the last ingested frame decided — these only move
                # when a frame arrives. That is not what catches a total frame
                # stall: if frames stop, the per-side tracking-loss gate in
                # `_update_authority` releases both sides regardless of what
                # this block still says.
                "clutch": {
                    "engaged": self._dead_man,
                    # Which sides that global bool actually covers: the two
                    # squeeze buttons, or a mirror of `engaged` for a frame
                    # that carried no split.
                    "sides": dict(self._dead_man_sides),
                    "reason": self._clutch_reason,
                },
                # Bimanual collision guard. `enabled: false` means no guard is
                # wired; otherwise `slack_m` is metres of clearance left before
                # the guard clamps (< 0 while it is actively holding a step
                # back), and `worst` names the binding constraint.
                "collision": self._collision_status(),
                # Authority transfer, per side. `engaged` above says the
                # operator is ASKING to drive; this says whether either arm has
                # actually been handed over, and if not, what is still in the
                # way. The two were the same event before acquisition existed.
                "acquire": {
                    "acquire_ms": self._handover_ms(),
                    "match_dwell_ms": self._match_dwell_ms,
                    "left":  self._acquire_status("left", now),
                    "right": self._acquire_status("right", now),
                },
            }

    def _collision_status(self) -> dict:
        """Caller holds the lock.

        `enabled` is whether the guard may clamp a step; `available` is
        whether it *could* be enabled at all (it cannot without mount
        geometry for every arm). The two are reported separately because a
        UI that shows one switch for both would offer the operator a toggle
        that silently does nothing on a rig with no mounts configured.

        `slack_m` and `worst` are published either way — a disabled guard
        still measures, so the operator can watch the clearance they chose
        to stop enforcing.
        """
        if self._collision is None:
            return {"enabled": False, "available": False}
        out = {
            "enabled": bool(getattr(self._collision, "enabled", True)),
            "available": bool(getattr(self._collision, "available", True)),
            "margin_m": float(self._collision.cfg.margin_m),
        }
        if self._collision_last is not None:
            out.update(self._collision_last)
        return out

    def _acquire_status(self, side: str, now: float) -> dict:
        """One side's acquisition block. Caller holds the lock.

        `remaining_ms` is recomputed against the live clock rather than cached
        from the last tick, so the operator's countdown runs smoothly off a
        20 Hz telemetry poll instead of stepping with the commit loop. Both it
        and `authority` are read back by `QuestTeleoperator.convert` and the
        in-headset HUD — that contract outranks tidiness here.
        """
        acq = self._acq[side]
        remaining: float | None = None
        if acq.authority is SideAuthority.ACQUIRING and acq.since_perf is not None:
            remaining = max(0.0, self._handover_ms() - (now - acq.since_perf) * 1000.0)
        ramp: float | None = None
        if acq.authority is SideAuthority.DRIVING and acq.driving_since_perf is not None:
            ramp = min(1.0, (now - acq.driving_since_perf) * 1000.0
                       / max(1.0, self._acquire_ramp_ms))
        return {
            "authority": acq.authority.value,
            "reason": acq.reason,
            "remaining_ms": remaining,
            "ramp": ramp,
        }

    def _handover_ms(self) -> float:
        """How long an unbroken engagement must run before authority
        transfers. The dwell is a floor under the countdown — see
        MATCH_DWELL_MS."""
        return max(self._acquire_ms, self._match_dwell_ms)

    def start(self, *, left_arm: str | None, right_arm: str | None,
              hz: float = 60.0) -> None:
        """Begin a session on one or both arms.

        Passing None for a side starts a SINGLE-ARM session: that hand's
        controller is ignored, its authority stays HELD, and nothing is ever
        written to it. This is the shape a first hardware bring-up wants —
        one arm on the bench, one hand on the grip, half as much that can go
        wrong — and it is also what a rig with one working servo board is
        left with.
        """
        with self._lock:
            for _peer in self._peers:
                if getattr(_peer, "status", lambda: {})().get("running"):
                    raise RuntimeError("leader/follower teleop is running; stop it first")
            if self.running:
                raise RuntimeError("human teleop already running; stop it first")
            if not left_arm and not right_arm:
                raise ValueError("at least one of left_arm/right_arm is required")
            if left_arm and right_arm and left_arm == right_arm:
                raise ValueError("left_arm and right_arm must be different")
            if not (MIN_SESSION_HZ <= hz <= MAX_SESSION_HZ):
                raise ValueError(
                    f"hz must be between {MIN_SESSION_HZ:g} and "
                    f"{MAX_SESSION_HZ:g}; got {hz:g}")
            arm_ids = {side: arm_id
                       for side, arm_id in (("left", left_arm),
                                            ("right", right_arm))
                       if arm_id}
            handles = {side: self._arms[arm_id]
                       for side, arm_id in arm_ids.items()}
            # See TeleopSession.start's identical guard (teleop.py): once
            # this session sets both arms to Mode.MANUAL below, an in-flight
            # discrete-move ramp is no longer visible to the mode guard, so
            # it must be refused explicitly here rather than trusted to
            # self-cancel. Mirrors move_to's refusal when a teleop session
            # already owns the arm — see motion.py and A6 in the plan.
            for side, arm_id in arm_ids.items():
                handle = handles[side]
                if handle.executor.is_running:
                    raise RuntimeError(
                        f"arm {arm_id!r} has a move in progress; wait for it "
                        "to finish or cancel it before starting teleop"
                    )
            for a in handles.values():
                if not a.torque_enabled:
                    a.enable_torque()
                a.guard.set(Mode.MANUAL)
            effective_hz = self._hz_override or hz
            self._cfg = _SessionConfig(left_arm=left_arm, right_arm=right_arm,
                                       hz=effective_hz)
            self._started_at = time.time()
            self._state = HumanState.ARMED
            self._last_error = None
            # Clear every per-session transient. Two of these are load-bearing:
            #   _ws_disconnected_at_perf — set when the operator's tab closes,
            #     which happens *before* stop(); leaving it set makes the next
            #     session's first tick see an expired grace window and auto-stop.
            #   _ctrl — the last raw controller frames of the previous
            #     session; leaving them set makes a freshly-ARMED session
            #     track hands that are no longer there, before a single new
            #     frame has arrived.
            self._ws_disconnected_at_perf = None
            self._ctrl = {"left": None, "right": None}
            self._head_orientation = None
            self._stance_frame = None
            self._home_req = {"left": False, "right": False}
            self._home_ramp = {"left": None, "right": None}
            self._last_left_perf = 0.0
            self._last_right_perf = 0.0
            # A new session starts with neither arm handed over, whatever the
            # previous one ended holding.
            self._seen_frame = False
            for acq in self._acq.values():
                acq.release("clutch_open")
            self._reseed_pending = {"left": True, "right": True}
            self._kit_ran_ahead = {"left": False, "right": False}
            self._reset_clutch_state()
            self._latest_frame_ts_ms = 0
            self._latest_arrival_perf = 0.0
            # The smoothing state is seeded AFTER the producer is claimed —
            # see below. Empty here so no stale pose from a previous session
            # survives into the gap.
            self._committed_left = {}
            self._committed_right = {}
            self._steps_left = self._held_steps({})
            self._steps_right = self._held_steps({})
        self._stop_flag.clear()
        # Claim the bus BEFORE the loop exists, so there is no window in which
        # the loop is running and the idle sampler still owns the tick. The
        # rate window is dropped explicitly: two consecutive sessions share a
        # producer NAME, so the automatic reset cannot see a cadence change
        # between them, and fps is frozen from this number.
        self.tick_bus.reset_rate()
        try:
            # The rate this session AIMS to publish at. Recorded beside the
            # measurement and never written as `fps` — see
            # `recorder._freeze_fps`. `_sample_divisor` is the same conversion
            # the loop uses, so the two cannot disagree about what the target
            # is.
            target = effective_hz / self._sample_divisor(effective_hz,
                                                         self._sample_hz)
            self._tick_token = self.tick_bus.attach_producer(
                "human-teleop", target_hz=target)
        except ProducerConflict:
            # The state was already set to RUNNING under the lock above. A
            # session marked running with no thread and no producer is worse
            # than a failed start: `stop()` would clean up a session that
            # never began, and every surface would report teleop live.
            with self._lock:
                self._state = HumanState.IDLE
                self._cfg = None
                self._started_at = None
            raise
        # Build one kit tracker per driven side and seed it — plus the
        # committed pose — from where the arm actually is. AFTER claiming the
        # producer, never before: the idle sampler checks the bus's owner
        # before it reads, so from the attach onward the serial line belongs
        # to this session alone; seeded inside the lock block above, these
        # reads raced the idle sampler's 20 Hz sampling by construction, and
        # a read that lost the race fell back to 0° on every joint — an
        # anchor to a pose the arm is not in.
        #
        # A read that FAILS leaves that side's reseed pending: the loop
        # retries, and until it succeeds the side is frozen — an unseeded
        # tracker anchors the operator's hand to a pose nothing observed,
        # which is the zero-seed fiction this machinery exists to prevent.
        # The committed pose still falls back to zeros for status() only.
        # Thread-safety: the loop thread does not exist yet, so these fields
        # have exactly one writer here.
        try:
            self._kit = {side: self._side_factory(handle.joint_limits_deg,
                                                  self._vr_cfg)
                         for side, handle in handles.items()}
        except Exception:
            # Same reasoning as the ProducerConflict handler above: a session
            # marked running with a claimed bus and no loop is worse than a
            # failed start.
            self._tick_token.detach()
            self._tick_token = None
            with self._lock:
                self._state = HumanState.IDLE
                self._cfg = None
                self._started_at = None
            raise
        committed_seed: dict[str, dict[str, float]] = {"left": {}, "right": {}}
        for side, handle in handles.items():
            observed = self._read_observed(handle)
            if observed is None:
                committed_seed[side] = {j: 0.0 for j in handle.joint_limits_deg}
                continue                # reseed stays pending; the loop retries
            committed_seed[side] = observed
            self._kit[side].seed_from_observed(dict(observed))
            with self._lock:
                self._reseed_pending[side] = False
        self._committed_left = committed_seed["left"]
        self._committed_right = committed_seed["right"]
        self._steps_left = self._held_steps(self._committed_left)
        self._steps_right = self._held_steps(self._committed_right)
        self._thread = threading.Thread(
            target=self._loop,
            name=f"haller-hmi-human-teleop-{left_arm or '-'}-{right_arm or '-'}",
            daemon=True,
        )
        self._thread.start()
        logger.info("human teleop started: left=%s right=%s @ %.1f Hz",
                    left_arm or "-", right_arm or "-", effective_hz)

    def request_home(self) -> list[str]:
        """Slew every non-driving side to home (0° joints, gripper open),
        INSIDE the session.

        This exists because the discrete move path (`POST /arm/{id}/home`)
        is — correctly — refused while a session owns the arms, yet the
        operator standing in a headset with the grips open plainly should be
        able to park the arms between takes. Done here, the home target rides
        the exact machinery every teleop step rides: the one-pole LPF, the
        per-tick rate caps, and the bimanual collision guard — which the
        discrete path never consults.

        Returns the sides that accepted the request. A DRIVING side is
        skipped, and any accepted request self-cancels the moment its side
        starts driving: the operator's hand outranks a parked reset.
        """
        with self._lock:
            if not self.running:
                return []
            sides = [s for s in self._cfg.sides
                     if self._acq[s].authority is not SideAuthority.DRIVING]
            now = time.perf_counter()
            for s in sides:
                self._home_req[s] = True
                # The ramp's fixed frame: where the arm stands NOW, and when
                # the request landed. Re-requesting home restarts the ramp
                # from the current pose, never from a stale one.
                committed = (self._committed_left if s == "left"
                             else self._committed_right)
                self._home_ramp[s] = {"start": dict(committed), "t0": now}
        return sides

    @staticmethod
    def _home_target(handle) -> dict[str, float]:
        """Home pose in target units: 0° on every joint, gripper OPEN (1.0 in
        the converter's [0,1] gripper convention) — a reset that ends with
        closed jaws would fight the very next grab."""
        target = {j: 0.0 for j in handle.joint_limits_deg}
        target["gripper"] = 1.0
        return target

    def stop(self) -> None:
        with self._lock:
            if not self.running:
                return
            cfg = self._cfg
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        # After the join, never before: releasing while the loop can still
        # publish would let the idle sampler back on the bus beside it.
        if self._tick_token is not None:
            self._tick_token.detach()
            self._tick_token = None
        # Restore arms to MANUAL with torque on.
        if cfg is not None:
            for side in cfg.sides:
                handle = self._arms[cfg.arm_for(side)]
                if not handle.torque_enabled:
                    handle.enable_torque()
                handle.guard.set(Mode.MANUAL)
        with self._lock:
            self._state = HumanState.IDLE
            self._cfg = None
            self._started_at = None
            # Nothing is being asked for any more. Keep the committed values —
            # goal_deg retains them too — but no joint may still advertise a
            # live reason, and no clutch may still advertise authority.
            self._steps_left = self._held_steps(self._committed_left)
            self._steps_right = self._held_steps(self._committed_right)
            self._reset_clutch_state()
            # Same reasoning as the clutch block: a stopped session must not
            # still advertise an arm as DRIVING or mid-countdown — nor a
            # tracker as engaged. The adapters are session-lived, like the
            # clutch anchors they hold.
            self._seen_frame = False
            self._kit = {}
            for acq in self._acq.values():
                acq.release("idle")
        logger.info("human teleop stopped")

    def ingest_frame(self, frame: dict) -> None:
        """Store one RAW wire frame — per-side controller state, latest-wins —
        plus the clutch booleans derived from the per-side squeeze.
        Thread-safe.

        NOTHING is solved here. The kit's loop shape solves at the CONSUMER's
        cadence on the latest frame; solving per arriving frame, seeded from
        the throttled committed pose, is exactly the structure the audit
        found bleeding hand-to-tool correspondence away, so this method's
        whole job is storage plus the authority bookkeeping a release must
        never wait a tick for.
        """
        with self._lock:
            if not self.running:
                return
            left_raw = self._usable_side(frame.get("left"))
            right_raw = self._usable_side(frame.get("right"))
            # A plain held boolean per hand: the Quest's squeeze button,
            # carried on the side. Read off the UNFILTERED side dict — a hand
            # whose pose arrived as junk can still be squeezing, and opening
            # the clutch on a pose glitch would restart acquisition on a
            # fault the staleness budget exists to absorb. The backend
            # applies no threshold and no hold timer of its own. `dead_man`
            # remains as two compat rails: an explicit False overrules any
            # squeeze the same frame claims (an inconsistent frame reads
            # disengaged — the safe reading), and a harness side that
            # carries no `squeeze` key at all mirrors the global boolean
            # rather than silently engaging neither.
            dm = frame.get("dead_man")
            sq = {}
            for side in ("left", "right"):
                raw = frame.get(side)
                sq[side] = (bool(raw.get("squeeze", bool(dm)))
                            if isinstance(raw, dict) else False)
            if dm is False:
                sq = {"left": False, "right": False}
            engaged = sq["left"] or sq["right"] or bool(dm)
            self._dead_man = engaged
            self._clutch_reason = "engaged" if engaged else CLUTCH_RESTING
            self._dead_man_sides = dict(sq)

            self._latest_frame_ts_ms = int(frame.get("ts_ms", 0))
            now_perf = time.perf_counter()
            self._latest_arrival_perf = now_perf
            # WS is healthy: cancel any pending grace window.
            self._ws_disconnected_at_perf = None

            # Head + stance ride the frame and are stored per frame, absence
            # included — the kit keeps the previous engage rotation when the
            # latest frame carries no head pose, and that decision belongs to
            # the adapter, not to a cache here inventing one. A head pose
            # that is malformed or non-finite is stored as ABSENT for the
            # same reason `_usable_side` refuses one: a NaN quaternion rides
            # `atan2` into a poisoned engage rotation without ever raising.
            head = frame.get("head")
            head_orient = (head.get("orientation")
                           if isinstance(head, dict) else None)
            self._head_orientation = (head_orient
                                      if self._finite_quat(head_orient)
                                      else None)
            stance = frame.get("stance")
            self._stance_frame = stance if stance in STANCES else None

            # Latest-wins per side, absent-or-junk included: the loop solves
            # whatever is newest, and the adapter freezes on an untracked or
            # missing hand (the kit's own rule). The AGE stamp moves only for
            # a side that is present and tracked — an untracked flicker ages
            # out through the staleness budget instead of resetting it, and a
            # side that really has stopped producing usable poses reads as
            # lost, which is the truth.
            self._ctrl["left"] = left_raw
            self._ctrl["right"] = right_raw
            if isinstance(left_raw, dict) and left_raw.get("tracked", False):
                self._last_left_perf = now_perf
            if isinstance(right_raw, dict) and right_raw.get("tracked", False):
                self._last_right_perf = now_perf

            self._seen_frame = True
            # Evaluated here as well as in the commit loop, and for different
            # reasons: the loop is what advances a countdown when no frame
            # arrives, and this is what makes a RELEASE take effect on the
            # frame that reports it rather than up to a tick later. Release
            # must never wait for the loop.
            self._update_authority(now_perf)

    @staticmethod
    def _usable_side(raw) -> dict | None:
        """One side's raw controller dict, or None for anything the adapter
        could not safely consume. The wire (`vr_teleop.wire`) is the schema;
        this refuses shapes that would throw inside the 60 Hz loop AND any
        non-finite pose/trigger number — a junk frame must cost one side one
        tick, never the session, and NEVER a motion.

        Finiteness is load-bearing, not tidiness: stdlib `json.loads`
        accepts NaN/Infinity literals, the adapter's EMA and the solver's
        `np.clip` dq caps propagate NaN instead of raising, one NaN then
        sticks in the OPEN-LOOP qpos integrator until a reseed, and
        `_kit_step`'s `max(lo, min(hi, nan))` resolves to the UPPER joint
        limit — which the DRIVING write budget (`float("inf")`, by the kit
        contract) would send at servo speed, every tick. Refused here, the
        side takes the adapter's untracked exit: integrator holds, anchor
        and filters survive, and trackability ages out through the normal
        staleness budget. The clutch bookkeeping stays on the UNFILTERED
        dict by design (see `ingest_frame`) — a hand whose pose arrived as
        junk can still be squeezing."""
        if not isinstance(raw, dict):
            return None
        pos, orient = raw.get("position"), raw.get("orientation")
        if not (isinstance(pos, (list, tuple)) and len(pos) >= 3
                and isinstance(orient, (list, tuple)) and len(orient) >= 4):
            return None
        try:
            values = [*pos, *orient, raw.get("trigger", 0.0)]
            if not all(math.isfinite(float(v)) for v in values):
                return None
        except (TypeError, ValueError):
            return None
        return raw

    @staticmethod
    def _finite_quat(orient) -> bool:
        """True only for a 4+-element sequence of finite numbers. The yaw
        correction consumes the head pose through `atan2`, which propagates
        NaN rather than raising, and a poisoned R sticks to the mapper at
        the next anchor. A junk head pose must read as ABSENT — the kit
        keeps the previous engage rotation — never as facing anywhere."""
        if not isinstance(orient, (list, tuple)) or len(orient) < 4:
            return False
        try:
            return all(math.isfinite(float(v)) for v in orient)
        except (TypeError, ValueError):
            return False

    # ---- authority transfer ---------------------------------------------

    def _side_trackable(self, side: str, now: float) -> bool:
        last = self._last_left_perf if side == "left" else self._last_right_perf
        return last != 0.0 and (now - last) * 1000.0 <= self._frame_age_ms_loss

    def _update_authority(self, now: float) -> None:
        """Advance both sides' authority. Caller holds the lock.

        Idempotent and safe to call from either thread; it reads the clock the
        caller passes and nothing else. Sides are independent on purpose — one
        hand leaving frame must not freeze the arm the other hand is mid-reach
        with, and re-acquiring the side that dropped should not restart the
        countdown for the side that never did.
        """
        for side in ("left", "right"):
            acq = self._acq[side]
            # A side this session has no arm for can never acquire. It is
            # not "lost" — nothing was ever expected of it — so it reports a
            # reason of its own rather than borrowing the tracking-loss one
            # and making the operator hunt for a hand that is not missing.
            if self._cfg is not None and self._cfg.arm_for(side) is None:
                acq.release("no_arm")
                continue
            # The age stamp only ever moves for a present, tracked hand
            # (`ingest_frame`), so trackability alone is the acquisition gate:
            # a hand the headset reports untracked ages out exactly like an
            # absent one.
            tracked = self._side_trackable(side, now)
            engaged = self._dead_man_sides.get(side, self._dead_man)
            if not engaged or not tracked:
                # No reseed on a demote: nothing wrote the arm during the
                # loss (the goals froze), so the adapter's open-loop qpos is
                # still the last thing commanded — and the kit's discipline
                # is exactly that it is never re-read mid-teleop. The reseed
                # rule lives with the two events that actually move the arm
                # under the adapter: session entry and the home slew.
                #
                # Tracking outranks the clutch: an operator holding the
                # dead-man over a side the robot cannot see needs to be told
                # about the side, not about the clutch they are already
                # holding.
                acq.release("no_tracking" if not tracked else "clutch_open")
                continue

            if acq.authority is SideAuthority.HELD:
                acq.since_perf = now
                if self._handover_ms() <= 0.0:
                    # The default, and the kit's clutch: engage on the rising
                    # edge. ACQUIRING is not entered at all — passing through
                    # it for a single tick would advertise a state the operator
                    # can never act on.
                    acq.authority = SideAuthority.DRIVING
                    acq.driving_since_perf = now
                    acq.reason = "driving"
                    logger.info("human teleop: %s arm engaged", side)
                    continue
                acq.authority = SideAuthority.ACQUIRING

            if acq.authority is SideAuthority.DRIVING:
                acq.reason = "driving"
                continue

            # Only reachable if a non-zero handover window is configured (the
            # constructor still takes one, and the tests exercise it).
            acq.reason = "counting"
            held_for_ms = ((now - acq.since_perf) * 1000.0
                           if acq.since_perf is not None else 0.0)
            if held_for_ms >= self._handover_ms():
                acq.authority = SideAuthority.DRIVING
                acq.driving_since_perf = now
                acq.reason = "driving"
                logger.info("human teleop: %s arm acquired", side)
        self._derive_state()

    def _derive_state(self) -> None:
        """Session state is a view of the two authorities. Caller holds the lock."""
        if self._state is HumanState.IDLE:
            return
        authorities = {acq.authority for acq in self._acq.values()}
        if SideAuthority.DRIVING in authorities:
            self._state = HumanState.DRIVING
        elif SideAuthority.ACQUIRING in authorities:
            self._state = HumanState.ACQUIRING
        elif self._seen_frame:
            self._state = HumanState.TRACKING
        else:
            self._state = HumanState.ARMED

    def _ramp_cap(self, side: str, period: float, now: float) -> float:
        """Per-tick rate cap for one side, in degrees.

        A time-varying cap rather than a blended offset: it composes with the
        clamp/reason machinery already in `_smooth_step` (the operator sees
        RATE-CAP while the ramp is biting, which is the truth), and it cannot
        make `committed` disagree with what was actually written.
        """
        full = self._rate_cap_deg_s * period
        acq = self._acq[side]
        if acq.authority is not SideAuthority.DRIVING or acq.driving_since_perf is None:
            return full
        # No ramp configured (the default): the session's normal cap from the
        # first tick, which is what the kit commands and what the envelope
        # limits already bound.
        if self._acquire_ramp_ms <= 0.0:
            return full
        frac = (now - acq.driving_since_perf) * 1000.0 / self._acquire_ramp_ms
        if frac >= 1.0:
            return full
        start = self._acquire_rate_deg_s * period
        return min(full, start + max(0.0, frac) * (full - start))

    def latest_ctrl(self) -> dict:
        """The raw controller frame each side would be solved from next tick
        (None for a side the latest frame did not carry). Diagnostic."""
        with self._lock:
            return {"left": self._ctrl["left"], "right": self._ctrl["right"]}

    def ik_sides(self) -> dict:
        """Per-side solver diagnostics for the `ik_state` broadcast.

        Assembled from each side's `KitSideTeleop.diag()` with the one fact
        the adapter cannot know — whether the session is actually WRITING its
        output — overlaid as `driving`. Field names are the frontend's
        `IkSideDiag` contract (VRTeleopPanel reads them off `msg.sides`).
        Both side keys are always present; a side with no tracker (no arm, or
        no session) is an empty dict, which the panel renders as idle.
        """
        with self._lock:
            kits = dict(self._kit) if self.running else {}
            driving = {side: self._acq[side].authority is SideAuthority.DRIVING
                       for side in ("left", "right")}
        out: dict[str, dict] = {"left": {}, "right": {}}
        for side, kit in kits.items():
            try:
                diag = dict(kit.diag())
            except Exception:       # diagnostics must never take a route down
                logger.warning("kit diag() failed for %s", side, exc_info=True)
                diag = {}
            diag["driving"] = bool(driving[side])
            out[side] = diag
        return out

    def notify_ws_disconnected(self) -> None:
        """Start the grace window after which a session with no operator on
        the other end stops itself.

        STARTS it — it does not restart one that is already running. That
        distinction is load-bearing now that more than one socket can call
        this: the VR relay hands the same session frames, and every idle
        client it drops calls in here too. Re-stamping the clock on each of
        those turned the window into something a second page could hold open
        indefinitely, so a session whose driving headset had genuinely died
        never stopped — measured, with one stray browser tab sitting on the
        relay page reconnecting every three seconds.

        What actually clears the window is a FRAME arriving (`ingest_frame`),
        which is the only evidence that an operator is still there. So the
        window means "time since the last frame", not "time since some socket
        closed", and no number of unrelated sockets opening and closing can
        extend it.
        """
        with self._lock:
            if not self.running:
                return
            if self._ws_disconnected_at_perf is None:
                self._ws_disconnected_at_perf = time.perf_counter()

    @staticmethod
    def _read_observed(handle) -> dict[str, float] | None:
        """Where the arm is, through `read_joints_deg()` (the ArmHandle
        interface, so sim arms read correctly too) — or None if the read
        failed. It must never quietly become 0° per joint: a zero-seeded
        tracker anchors the operator's hand to a pose the arm is not in, and
        the first squeeze slews the arm toward the fiction."""
        try:
            observed = handle.read_joints_deg()
        except Exception:
            logger.warning("could not read pose for arm %s; reseed pending",
                           getattr(handle.config, "id", "?"), exc_info=True)
            return None
        return {joint: float(observed.get(joint, 0.0))
                for joint in handle.joint_limits_deg}

    @staticmethod
    def _held_steps(committed: dict[str, float]) -> dict[str, JointStep]:
        """Seed the diagnostic block before any frame has arrived: everything
        is being held at its seeded position, nothing has been asked for."""
        return {joint: JointStep(target=None, committed=value, reason="held")
                for joint, value in committed.items()}

    @staticmethod
    def _steps_as_dict(steps: dict[str, JointStep]) -> dict[str, dict]:
        return {
            joint: {
                "target": step.target,
                "committed": step.committed,
                "reason": step.reason,
            }
            for joint, step in steps.items()
        }

    @staticmethod
    def _to_degrees(joint: str, value: float, lo: float, hi: float) -> float:
        """One target joint value in degrees, unclamped.

        Special-cases the gripper: the converter emits [0, 1] (0 = closed,
        1 = open), which is scaled onto the joint's calibrated degree range so
        that every comparison downstream — smoothing, rate caps, the ramp — is
        between two numbers in the same unit.
        """
        if joint == "gripper":
            return lo + max(0.0, min(1.0, value)) * (hi - lo)
        return value

    def _smooth_step(
        self,
        committed: dict[str, float],
        target: dict[str, float] | None,
        limits: dict[str, tuple[float, float]],
        alpha: float,
        *,
        cap: float | None = None,
    ) -> dict[str, JointStep]:
        out: dict[str, JointStep] = {}
        cap = (self._rate_cap_deg_s * self._period) if cap is None else cap
        for joint, lo_hi in limits.items():
            lo, hi = lo_hi
            cur = committed.get(joint, 0.0)
            if target is None or joint not in target:
                out[joint] = JointStep(target=None, committed=cur, reason="held")
                continue
            desired = self._to_degrees(joint, float(target[joint]), lo, hi)
            # One-pole LPF, then per-tick rate cap, then hard clamp to limits.
            # Each stage records whether it altered the value. Exact float
            # equality is correct here: these clamps return their input
            # bitwise unchanged when they don't bite.
            lpf = cur + alpha * (desired - cur)
            capped = max(cur - cap, min(cur + cap, lpf))
            final = max(lo, min(hi, capped))
            if final != capped:
                reason = "clamped"        # a hard limit outranks a transient cap
            elif capped != lpf:
                reason = "rate_capped"
            else:
                reason = "ok"
            out[joint] = JointStep(target=desired, committed=final, reason=reason)
        return out

    def _kit_step(
        self,
        committed: dict[str, float],
        action: dict[str, float],
        limits: dict[str, tuple[float, float]],
    ) -> dict[str, JointStep]:
        """One DRIVING tick's steps: the adapter's action, IN FULL.

        No LPF, no per-tick rate cap — the kit ships neither, and with the
        solver integrating its own open-loop qpos any attenuation here would
        be measured by the mapper's reach limits as operator over-drive and
        absorbed. The only shaping is the joint-limit clamp (recorded here so
        the diagnostic tells the truth; `send_goal` re-applies it as the
        enforcement point) and the gripper's [0,1] → calibrated-degrees
        scaling, which is the session's contract with the recorder.
        """
        out: dict[str, JointStep] = {}
        for joint, lo_hi in limits.items():
            lo, hi = lo_hi
            if joint not in action:
                # The adapter maps every joint in joint_limits_deg; a hole
                # would be a contract break upstream. Hold rather than invent.
                out[joint] = JointStep(target=None,
                                       committed=committed.get(joint, 0.0),
                                       reason="held")
                continue
            desired = self._to_degrees(joint, float(action[joint]), lo, hi)
            final = max(lo, min(hi, desired))
            reason = "clamped" if final != desired else "ok"
            out[joint] = JointStep(target=desired, committed=final, reason=reason)
        return out

    @staticmethod
    def _commit(handle, goal: dict[str, float],
                *, speed_cap_deg_s: float) -> dict[str, float]:
        """Write one tick's goal through the ArmHandle interface and report
        what was actually sent.

        `send_goal` does the joint-limit clamp and the mode-guard check itself,
        and works against both `ArmHandle` and `SimArmHandle` — so the same loop
        drives real arms and MuJoCo arms. `send_goal` can legitimately command
        LESS than it was asked when a finite cap is passed, because it caps
        each call against real elapsed time rather than this loop's nominal
        tick — so its return, not `goal`, is what the arm actually received.
        The caller folds this back into the committed pose, which is what
        status()["goal_deg"] — and recorder.py's `action` column downstream of
        it — reports.

        The cap is the CALLER's statement about the path the goal rode in on:

          * a DRIVING side passes float("inf") — the kit writes raw, and its
            governors already had their say (the vendored solver's per-joint
            per-solve dq caps and the joint-limit clamp inside send_goal;
            the mode guard and E-STOP stand regardless). Any finite number
            here re-imposes the downstream limiter the audit tore out.
          * a home slew passes RATE_CAP_DEG_S — that path keeps the session's
            LPF + rate cap, and the write-side bound matches it.
        """
        return handle.send_goal(
            {joint: float(value) for joint, value in goal.items()},
            speed_cap_deg_s=speed_cap_deg_s)

    def _side_steps(self, handle, kit, ctrl, prev, *, driving, homing,
                    pending, head, stance, frame_age_s, alpha, cap,
                    home_ramp=None, now=0.0):
        """One side's steps for this tick — the three-way fork the loop takes.
        Returns (steps, engaged): `engaged` mirrors the adapter's clutch when
        `kit.update()` ran this tick, and is None when it did not.

        * HOMING: the in-session park, shaped like the kit's rest ramp — a
          fixed-duration joint-space lerp (HOME_RAMP_S) from where the arm
          stood at the request, every joint arriving together. The per-tick
          rate cap stays as the backstop under it (a 2 s ramp never reaches
          it), and the collision guard downstream still has its say.
        * PENDING a re-seed: frozen. The tracker must neither integrate nor
          drive from state nothing observed; the loop's seed step retries.
        * otherwise: exactly one `kit.update()` on the latest raw frame —
          clutch edges, the EMA pose filter, the 0.2 s staleness gate and the
          open-loop IK integration all live in the adapter. While DRIVING the
          returned action is this side's target IN FULL; while held the
          adapter still ran (so its state machine stays true to the hand) but
          nothing is asked of the arm.
        """
        limits = handle.joint_limits_deg
        if homing:
            home = self._home_target(handle)
            target = home
            if home_ramp is not None:
                frac = min(1.0, max(0.0, (now - home_ramp["t0"]) / HOME_RAMP_S))
                start = home_ramp["start"]
                target = {}
                for j, v in home.items():
                    s0 = start.get(j)
                    lo, hi = limits.get(j, (0.0, 1.0))
                    if s0 is None:
                        # A joint with no committed start has nothing to ramp
                        # from — ask for the target and let the cap bound it.
                        target[j] = v
                        continue
                    if j == "gripper":
                        # committed is degrees-on-range; the home target (and
                        # _smooth_step's input) speak the converter's [0, 1].
                        s0 = (s0 - lo) / ((hi - lo) or 1.0)
                    target[j] = s0 + frac * (v - s0)
            # alpha 1.0: the lerp IS the shaping — filtering it would turn
            # the kit's straight ramp back into an exponential.
            return self._smooth_step(prev, target, limits, 1.0, cap=cap), None
        if pending or kit is None:
            return self._held_steps(prev), None
        action, engaged = kit.update(ctrl, head, stance, frame_age_s)
        if driving:
            return self._kit_step(prev, action, limits), engaged
        return self._held_steps(prev), engaged

    def _loop(self) -> None:
        with self._lock:
            cfg = self._cfg
        assert cfg is not None
        # None for a side this session has no arm for. Every use below is
        # guarded; a single-arm session runs the same loop with one half of
        # it inert rather than a second loop that could drift from this one.
        left = self._arms[cfg.left_arm] if cfg.left_arm else None
        right = self._arms[cfg.right_arm] if cfg.right_arm else None
        period = 1.0 / max(1.0, cfg.hz)
        # `_smooth_step`'s default cap converts against this; the loop
        # always passes an explicit cap, so this is for callers outside it.
        self._period = period
        # Smoothing time constant (frequency-independent), HOME-SLEW ONLY —
        # the driven path writes the adapter's action in full. Zero means the
        # filter is OFF (alpha 1, pure passthrough).
        tau_s = self._lpf_tau_s
        if tau_s <= 0.0 or period <= 0:
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(-period / tau_s)
        consecutive_errors = 0
        # Ticks per PUBLISHED sample. Derived from a rate rather than
        # configured as a count, so the sampler targets the same real cadence
        # whatever `hz` this session runs at — a divisor of 2 would mean 30 Hz
        # at hz=60 and 5 Hz at hz=10, and the recorder's fps would silently
        # follow the control rate.
        sample_every = self._sample_divisor(cfg.hz, self._sample_hz)
        token = self._tick_token
        tick_index = -1
        unix_at_read = 0.0
        while not self._stop_flag.is_set():
            tick_start = time.perf_counter()
            tick_index += 1
            sampling = token is not None and (tick_index % sample_every) == 0
            try:
                # Re-seed any side whose arm was moved by something other
                # than its own adapter — session entry, or the home slew —
                # BEFORE that adapter is asked for a step. Outside the lock:
                # this is bus traffic on real hardware, and status() must not
                # block on it. Edge-triggered, so a steady session pays
                # nothing. A side still mid-home-slew is skipped (the arm is
                # still being moved; a seed now is stale by the next write) —
                # unless it has just gone DRIVING, which is the handover
                # tick: the home request is dead (cancelled below), and the
                # adapter must anchor to the arm's real pose before its
                # first driven step.
                for side, handle in (("left", left), ("right", right)):
                    if handle is None:
                        continue
                    with self._lock:
                        pending = self._reseed_pending[side]
                        homing_now = self._home_req[side]
                        driving_now = (self._acq[side].authority
                                       is SideAuthority.DRIVING)
                    if not pending or (homing_now and not driving_now):
                        continue
                    observed = self._read_observed(handle)
                    if observed is None:
                        # Keep the request pending and the previous committed
                        # pose in place, and try again next tick. A failed
                        # read must never quietly become 0° per joint here:
                        # with the acquisition countdown at zero, a
                        # zero-seeded side anchors the operator's hand to a
                        # pose the arm is not in, and the first squeeze slews
                        # the arm toward the fiction.
                        continue
                    kit = self._kit.get(side)
                    if kit is not None:
                        # Resets the tracker's qpos AND its clutch state, so
                        # a squeeze already held re-anchors on this tick's
                        # rising edge — the engage starts from zero delta by
                        # construction.
                        kit.seed_from_observed(dict(observed))
                    with self._lock:
                        self._reseed_pending[side] = False
                        self._kit_ran_ahead[side] = False
                        if self._acq[side].authority is SideAuthority.DRIVING:
                            continue    # this tick's own write refreshes it
                        if side == "left":
                            self._committed_left = observed
                            self._steps_left = self._held_steps(observed)
                        else:
                            self._committed_right = observed
                            self._steps_right = self._held_steps(observed)

                # Read the clock HERE, not at the top of the tick. Authority is
                # judged against a 700 ms freshness budget and the ramp against
                # a 1.5 s one, while the re-seed above is a bus read on
                # hardware and a locked world step in sim — either can block.
                # A `tick_start` captured before that work is stale by however
                # long it took, which stretches the ramp and softens the
                # freshness test by the same amount.
                #
                # Note this errs PERMISSIVE, not dangerous: an old `now` makes
                # `now - last_frame` smaller, so it cannot invent a tracking
                # loss. Both budgets should still be measured against the
                # moment they are being applied, not against the moment the
                # tick happened to begin.
                #
                # THE read also happens here, in the same breath as the clock:
                # one round trip per arm, once per published tick, so the
                # state in a sample and the action committed below describe
                # one moment rather than two (invariant 8). This is the read
                # telemetry and the recorder each used to take for themselves
                # at their own instants.
                arms_snap: dict = {}
                arm_errors: dict = {}
                if sampling:
                    arms_snap, arm_errors = self._sample_arms()
                    unix_at_read = time.time()
                now = time.perf_counter()
                with self._lock:
                    self._update_authority(now)
                    driving_left = (self._acq["left"].authority
                                    is SideAuthority.DRIVING)
                    driving_right = (self._acq["right"].authority
                                     is SideAuthority.DRIVING)
                    # A driving hand cancels a pending home request; a held
                    # side with one keeps slewing home through the session's
                    # LPF/rate-cap/guard chain — the one path that keeps them.
                    if driving_left:
                        self._home_req["left"] = False
                        self._home_ramp["left"] = None
                    if driving_right:
                        self._home_req["right"] = False
                        self._home_ramp["right"] = None
                    # An adapter that integrated while its side was NOT being
                    # written has run ahead of the arm (configured countdown,
                    # or a demote-recover with the grip held). Before such a
                    # side drives, re-align it from a real read — the reseed
                    # also resets the adapter's clutch edge, so the engage
                    # re-anchors at the hand's current position and the first
                    # driven write is the arm's own pose, zero delta by
                    # construction. Never fires at the shipped acquire_ms=0.
                    for side, driving_now in (("left", driving_left),
                                              ("right", driving_right)):
                        if driving_now and self._kit_ran_ahead[side]:
                            self._reseed_pending[side] = True
                            self._kit_ran_ahead[side] = False
                    homing_left = self._home_req["left"] and left is not None
                    homing_right = self._home_req["right"] and right is not None
                    pending_left = self._reseed_pending["left"]
                    pending_right = self._reseed_pending["right"]
                    # The latest raw frames, and the operator context they
                    # arrived with. Snapshotted so the adapter steps below run
                    # outside the lock against one consistent moment.
                    ctrl_left = self._ctrl["left"]
                    ctrl_right = self._ctrl["right"]
                    head = self._head_orientation
                    stance = self._stance_frame or self._vr_cfg.stance
                    frame_age_s = (now - self._latest_arrival_perf
                                   if self._latest_arrival_perf
                                   else float("inf"))
                    cap_left = self._ramp_cap("left", period, now)
                    cap_right = self._ramp_cap("right", period, now)
                    home_ramp_left = self._home_ramp["left"]
                    home_ramp_right = self._home_ramp["right"]
                    prev_left = dict(self._committed_left)
                    prev_right = dict(self._committed_right)
                # A DRIVING side that still awaits its re-seed is frozen this
                # tick — never driven from state nothing observed.
                drive_left = driving_left and not pending_left
                drive_right = driving_right and not pending_right
                steps_left, engaged_left = (self._side_steps(
                    left, self._kit.get("left"), ctrl_left, prev_left,
                    driving=drive_left, homing=homing_left,
                    pending=pending_left, head=head, stance=stance,
                    frame_age_s=frame_age_s, alpha=alpha, cap=cap_left,
                    home_ramp=home_ramp_left, now=now,
                ) if left is not None else ({}, None))
                steps_right, engaged_right = (self._side_steps(
                    right, self._kit.get("right"), ctrl_right, prev_right,
                    driving=drive_right, homing=homing_right,
                    pending=pending_right, head=head, stance=stance,
                    frame_age_s=frame_age_s, alpha=alpha, cap=cap_right,
                    home_ramp=home_ramp_right, now=now,
                ) if right is not None else ({}, None))
                # Record an engaged-but-unwritten adapter tick (see the
                # handover re-align above). Conservative on purpose: a frozen
                # engaged adapter (untracked hand, stale frames) sets it too —
                # the reseed it buys is one read and a fresh anchor, and the
                # miss it prevents is a handover lurch.
                if engaged_left and not drive_left:
                    with self._lock:
                        self._kit_ran_ahead["left"] = True
                if engaged_right and not drive_right:
                    with self._lock:
                        self._kit_ran_ahead["right"] = True
                committed_left = {j: s.committed for j, s in steps_left.items()}
                committed_right = {j: s.committed for j, s in steps_right.items()}
                # Bimanual guard, applied to the pair that would actually be
                # written — after smoothing and the rate caps, before the
                # handles — because a bound enforced on any earlier quantity
                # can be re-violated by a later stage. While nothing is
                # driving it still publishes live clearance, so the operator
                # can sanity-check the mount geometry against the real arms
                # BEFORE the first engagement.
                collision_last: dict | None = None
                # Only the sides this session actually drives go to the
                # guard. A single-arm session still gets the self-collision
                # pairs and the bench floors — which is most of what bites on
                # one arm anyway — while the inter-arm checks simply have no
                # second arm to find.
                by_side = {"left": (cfg.left_arm, prev_left, committed_left,
                                    steps_left),
                           "right": (cfg.right_arm, prev_right,
                                     committed_right, steps_right)}
                present = {s: v for s, v in by_side.items() if v[0]}
                if self._collision is not None and present:
                    pair_prev = {arm_id: prev
                                 for arm_id, prev, _, _ in present.values()}
                    pair_want = {arm_id: want
                                 for arm_id, _, want, _ in present.values()}
                    if driving_left or driving_right:
                        result = self._collision.filter_step(pair_prev, pair_want)
                        for arm_id, _, committed, steps in present.values():
                            filtered = result.poses[arm_id]
                            for joint, value in filtered.items():
                                if abs(value - committed.get(joint, value)) > 1e-9:
                                    st = steps[joint]
                                    steps[joint] = JointStep(
                                        target=st.target, committed=value,
                                        reason="collision",
                                    )
                        if cfg.left_arm:
                            committed_left = result.poses[cfg.left_arm]
                        if cfg.right_arm:
                            committed_right = result.poses[cfg.right_arm]
                        collision_last = {
                            "limited": result.limited,
                            "alpha": result.alpha,
                            "slack_m": result.clearance.slack,
                            "worst": result.clearance.worst,
                        }
                    else:
                        cl = self._collision.clearance(pair_want)
                        collision_last = {"limited": False, "alpha": 1.0,
                                          "slack_m": cl.slack,
                                          "worst": cl.worst}
                # Rebinding a single dict is atomic in CPython, but that does not
                # make this four-way update atomic — a reader in status() could
                # otherwise interleave and see committed_* from this tick paired
                # with steps_* from the previous one. Hold the lock across all
                # four assignments together so status() always sees one tick's
                # worth, consistently.
                with self._lock:
                    self._committed_left = committed_left
                    self._committed_right = committed_right
                    self._steps_left = steps_left
                    self._steps_right = steps_right
                    self._collision_last = collision_last
                # Authority IS the write gate. The per-side tracking-loss check
                # that used to live here moved into `_update_authority`, which
                # runs off the same `now` a few microseconds earlier —
                # keeping it here as well would be a second opinion about
                # whether an arm may move, and two of those is how they come to
                # disagree. Losing a side now DEMOTES it rather than merely
                # skipping the write, so recovery re-runs acquisition.
                #
                # A pending home request is the one sanctioned exception: the
                # side is HELD (no operator authority), but the target was set
                # by request_home, not by tracking — and it has been through
                # the session's smoothing, caps and guard by here.
                #
                # The speed budget names the path. A DRIVING write is the
                # kit's: unbounded here, because its governors already ran
                # (the vendored solver's per-solve dq caps; the joint-limit
                # clamp inside send_goal; the mode guard and E-STOP) — any
                # finite number at this seam is a downstream limiter whose
                # withheld degrees the mapper would misread as over-drive.
                # The home slew keeps the session's ceiling, matching the
                # rate cap it rode in on.
                write_left = left is not None and (drive_left or homing_left)
                write_right = right is not None and (drive_right or homing_right)
                if write_left:
                    sent_left = self._commit(
                        left, self._committed_left,
                        speed_cap_deg_s=(float("inf") if drive_left
                                         else RATE_CAP_DEG_S))
                if write_right:
                    sent_right = self._commit(
                        right, self._committed_right,
                        speed_cap_deg_s=(float("inf") if drive_right
                                         else RATE_CAP_DEG_S))
                # `_commit` reports what send_goal ACTUALLY sent, which can be
                # less than `_committed_left`/`_right` asked for (see
                # `_commit`'s docstring). Fold it back in — merged, not
                # replaced, so a joint send_goal had to drop (an unmeasured
                # joint; see ArmHandle.send_goal) keeps its last known value
                # rather than resetting to 0 on the next `_smooth_step` — and
                # do it as one lock acquisition covering both sides so
                # status() can never observe this tick's send reflected on one
                # side and last tick's on the other. Both `send_goal` calls
                # above stay outside any lock: real hardware traffic, or a
                # locked sim step, that status() must not block on.
                with self._lock:
                    if write_left:
                        self._committed_left = {**self._committed_left, **sent_left}
                        if homing_left:
                            # The slew moved the arm behind the adapter's
                            # back: its open-loop qpos is stale until the
                            # next re-seed, which the loop's seed step takes
                            # once the slew no longer owns the arm.
                            self._reseed_pending["left"] = True
                    if write_right:
                        self._committed_right = {**self._committed_right, **sent_right}
                        if homing_right:
                            self._reseed_pending["right"] = True
                # Publish LAST, once this tick's commit is final — including
                # the fold-back of what send_goal actually sent. Publishing
                # the intended goal instead of the committed one would put an
                # action in the dataset that the arm was never asked for.
                if sampling:
                    with self._lock:
                        goal_deg = {"left": dict(self._committed_left),
                                    "right": dict(self._committed_right)}
                        reasons = {
                            "left": {j: st.reason
                                     for j, st in self._steps_left.items()},
                            "right": {j: st.reason
                                      for j, st in self._steps_right.items()},
                        }
                        clutch = {"engaged": self._dead_man,
                                  "sides": dict(self._dead_man_sides),
                                  "reason": self._clutch_reason}
                        collision_block = self._collision_last
                    try:
                        token.publish(
                            t_mono=now, t_unix=unix_at_read,
                            arms=arms_snap, arm_errors=arm_errors,
                            goal_deg=goal_deg, reasons=reasons,
                            base=self._base_block(), clutch=clutch,
                            collision=collision_block,
                            degraded=bool(arm_errors),
                        )
                    except ProducerConflict:
                        # stop() detached between the sample and here. The
                        # session is going away; dropping this sample is the
                        # correct outcome, and it must not be an error.
                        pass
                # WS disconnect grace window: if too much time has passed, auto-stop.
                with self._lock:
                    disc_at = self._ws_disconnected_at_perf
                if disc_at is not None and (time.perf_counter() - disc_at) > self._ws_disconnect_grace_s:
                    logger.info("human teleop WS disconnect grace exceeded; stopping")
                    threading.Thread(target=self.stop, daemon=True).start()
                    break
                with self._lock:
                    self._last_error = None
                consecutive_errors = 0
            except Exception as e:
                logger.exception("human teleop tick failed")
                consecutive_errors += 1
                with self._lock:
                    self._last_error = str(e)
                if consecutive_errors >= MAX_CONSECUTIVE_TICK_ERRORS:
                    # A persistent fault (dead arm handle, broken guard) would
                    # otherwise retry at 20 Hz forever, session "running" the
                    # whole time. Stop it the same way the WS grace does —
                    # from another thread, since stop() joins this one.
                    logger.error("human teleop loop failed %d ticks in a row; "
                                 "stopping session", consecutive_errors)
                    threading.Thread(target=self.stop, daemon=True).start()
                    break
                time.sleep(0.05)
                continue
            elapsed = time.perf_counter() - tick_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
