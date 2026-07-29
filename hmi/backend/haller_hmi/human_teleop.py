"""Bimanual human-pose teleop session.

This is the sibling of `teleop.TeleopSession` (leader/follower). Where that
session reads positions off a physical leader arm at 60 Hz, this one reads
keypoints off a WebSocket from the operator's browser and runs them through
`retarget.compute_joint_goal` to produce joint angles. Otherwise the lifecycle
and safety semantics match exactly.

Session state machine:
    IDLE → (start)        → ARMED
    ARMED → (first frame) → TRACKING
    any → (stop / E-STOP) → IDLE

TRACKING / ACQUIRING / DRIVING are *derived* from the two per-side authorities
below rather than set directly — see `_derive_state`.

Authority transfer (per side):
    HELD → (clutch closed, side trackable) → ACQUIRING
    ACQUIRING → (countdown elapsed AND pose matched for the dwell) → DRIVING
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
from dataclasses import dataclass, field
from typing import Literal, cast

from .arm import ArmManager
from .safety import (
    Mode,
    MouthClutchCalib,
    MouthClutchState,
    mouth_clutch_thresholds,
)
from . import retarget

logger = logging.getLogger(__name__)

# ---- acquisition ------------------------------------------------------
#
# Closing the dead-man used to hand the robot over instantly, from wherever the
# operator's arms happened to be — which is essentially never where the robot's
# arms are. Worse, the smoothing state kept slewing toward the operator's pose
# the whole time the clutch was OPEN, so the first commit after engaging was a
# single step command to the operator's current pose. The rate cap had already
# been spent against an arm that was not moving; on hardware that is a step
# input to the servos, i.e. maximum velocity.
#
# Three mechanisms replace that instant, and they do different jobs:
#
#   the GATE   bounds how far apart the two poses may be at handover,
#   the COUNTDOWN gives the operator warning and a window to abort,
#   the RAMP   bounds the speed at which whatever error remains is closed.
#
# Only the ramp is load-bearing for safety — it holds unconditionally, needs no
# cooperation from the operator, and cannot be satisfied by accident. The gate
# and the countdown make engagement deliberate and legible. Do not drop the
# ramp on the grounds that the gate already bounds the error: the gate is
# checked against a pose estimate, and the ramp is what makes being wrong about
# it survivable.

#: Per-joint match tolerance, in degrees. Generous on purpose — the gate exists
#: to make the operator look at the robot and pre-position, not to demand
#: precision the pose estimate cannot deliver, and the ramp covers the residual.
ACQUIRE_TOL_DEFAULT_DEG = 20.0
ACQUIRE_TOL_DEG: dict[str, float] = {
    # The palm-normal cross product is the noisiest term in the retargeter.
    "wrist_roll": 30.0,
    # One low-inertia DOF whose approach speed the ramp bounds anyway; a tight
    # tolerance here would block acquisition over a pinch the operator cannot
    # see the robot holding.
    "gripper": 30.0,
}
#: Countdown from the clutch closing to the earliest possible handover.
ACQUIRE_MS = 3000.0
#: How long the pose match must hold before handover. Separate from the
#: countdown so estimator jitter across the tolerance boundary costs the
#: operator this much, not the whole countdown — a hard reset on every flicker
#: is how a gate becomes unreachable.
MATCH_DWELL_MS = 400.0
#: Joint speed the first instant of DRIVING is allowed to command. At the
#: default tolerance a worst-case matched pose closes in ~1.25 s, inside the
#: ramp, so the two constants stay consistent if either is retuned.
ACQUIRE_RATE_DEG_S = 20.0
#: Time over which the cap returns to the session's normal rate limit.
ACQUIRE_RAMP_MS = 1500.0


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
    #: Start of the current unbroken pose match, or None if not matched now.
    matched_since_perf: float | None = None
    #: Ramp origin — when DRIVING began.
    driving_since_perf: float | None = None
    #: Signed (clamped target − committed) per joint, in degrees.
    error_deg: dict[str, float] = field(default_factory=dict)
    blocking: tuple[str, ...] = ()
    matched: bool = False
    #: Why this side is where it is. The blocking-joint list explains a failed
    #: MATCH, but says nothing about a side that never got as far as matching —
    #: which is the state an operator is most likely to be stuck in and least
    #: able to diagnose, since a countdown that silently restarts looks
    #: identical to one that is frozen.
    reason: str = "clutch_open"

    def release(self, reason: str) -> None:
        self.authority = SideAuthority.HELD
        self.since_perf = None
        self.matched_since_perf = None
        self.driving_since_perf = None
        self.error_deg = {}
        self.blocking = ()
        self.matched = False
        self.reason = reason


#: Which source holds dead-man authority. Exactly these two strings, mirrored
#: in TypeScript as `ClutchSource` — nothing else may reach the session.
ClutchSource = Literal["spacebar", "mouth"]
CLUTCH_SOURCES: tuple[str, ...] = ("spacebar", "mouth")


class HumanState(str, enum.Enum):
    IDLE = "idle"
    ARMED = "armed"
    TRACKING = "tracking"
    ACQUIRING = "acquiring"
    DRIVING = "driving"


@dataclass
class _SessionConfig:
    left_arm: str
    right_arm: str
    swap: bool = False
    hz: float = 60.0


@dataclass
class JointStep:
    """One joint's outcome for one tick of the commit loop.

    `target` is what the retargeter asked for, in degrees (the gripper's
    [0,1] output is already scaled onto its calibrated range). `committed`
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
        frame_age_ms_loss: float = 300.0,
        ws_disconnect_grace_s: float = 5.0,
        face_stale_ms: float = 250.0,
        acquire_ms: float = ACQUIRE_MS,
        match_dwell_ms: float = MATCH_DWELL_MS,
        acquire_tol_deg: dict[str, float] | None = None,
        acquire_tol_default_deg: float = ACQUIRE_TOL_DEFAULT_DEG,
        acquire_rate_deg_s: float = ACQUIRE_RATE_DEG_S,
        acquire_ramp_ms: float = ACQUIRE_RAMP_MS,
    ):
        self._arms = arms
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
        self._target_left: dict | None = None
        self._target_right: dict | None = None
        self._pinch_calib_left: dict = {"min_m": 0.02, "max_m": 0.18}
        self._pinch_calib_right: dict = {"min_m": 0.02, "max_m": 0.18}
        self._committed_left: dict[str, float] = {}
        self._committed_right: dict[str, float] = {}
        self._steps_left: dict[str, JointStep] = {}
        self._steps_right: dict[str, JointStep] = {}
        self._hz_override = hz_override
        self._rate_cap_deg_per_tick = 4.0
        # T8: tracking-loss + WS disconnect grace
        self._frame_age_ms_loss = frame_age_ms_loss
        self._ws_disconnect_grace_s = ws_disconnect_grace_s
        self._ws_disconnected_at_perf: float | None = None
        # Per-arm last-frame timestamps (perf_counter), for tracking-loss.
        self._last_left_perf: float = 0.0
        self._last_right_perf: float = 0.0
        self._peers: list = []
        # Mouth-clutch state. The hold timer and the last-sample memory live in
        # safety.MouthClutchState so that the calibration analyser and the
        # replay-based speech test drive the same object this session does.
        self._mouth = MouthClutchState(stale_after_ms=face_stale_ms)
        # Set once by start() and never by a frame: the operator selects the
        # authority before the session exists, and the browser cannot hand it
        # over afterwards (spec 1, 6.4).
        self._clutch_source: ClutchSource = "spacebar"
        self._mouth_calib: MouthClutchCalib | None = None
        self._clutch_reason: str = "spacebar_mode"
        # Acquisition: per-side authority, and the parameters that gate it.
        self._acquire_ms = acquire_ms
        self._match_dwell_ms = match_dwell_ms
        self._acquire_tol_deg = dict(acquire_tol_deg or ACQUIRE_TOL_DEG)
        self._acquire_tol_default_deg = acquire_tol_default_deg
        self._acquire_rate_deg_s = acquire_rate_deg_s
        self._acquire_ramp_ms = acquire_ramp_ms
        self._acq: dict[str, _SideAcquire] = {
            "left": _SideAcquire(), "right": _SideAcquire(),
        }
        # Set on every authority change: the smoothing state has to be re-read
        # from the arm before it is used to judge a pose match or to command a
        # first step, or both are judged against a number nothing measured.
        self._reseed_pending: dict[str, bool] = {"left": True, "right": True}
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

    def set_swap(self, swap: bool) -> None:
        with self._lock:
            if self._cfg is not None:
                self._cfg.swap = bool(swap)

    def set_pinch_calib(self, *, left: dict | None, right: dict | None) -> None:
        with self._lock:
            if left is not None:
                self._pinch_calib_left = dict(left)
            if right is not None:
                self._pinch_calib_right = dict(right)

    def set_mouth_calib(self, calib: dict | None) -> None:
        """Store per-operator jaw calibration. None clears it.

        Both levels are SUSTAINED ones, measured over a hold window — see
        safety.MouthClutchCalib. This deliberately does not accept the old
        peak-based `talk_max`/`open_min` pair: those numbers are not the ones
        the current policy reasons about, and silently reinterpreting a peak as
        a sustained level would place t_engage far above where the operator can
        hold, which is exactly the failure this replaced.
        """
        with self._lock:
            self._mouth_calib = (
                MouthClutchCalib(
                    talk_hold=float(calib["talk_hold"]),
                    open_hold=float(calib["open_hold"]),
                    talk_peak=(float(calib["talk_peak"])
                               if calib.get("talk_peak") is not None else None),
                )
                if calib else None
            )

    def mouth_calib_is_valid(self) -> bool:
        """True when a calibration is set AND yields safe thresholds."""
        with self._lock:
            return (self._mouth_calib is not None
                    and mouth_clutch_thresholds(self._mouth_calib) is not None)

    def _reset_clutch_state(self, source: ClutchSource | None) -> None:
        """Clear every clutch transient. Caller holds the lock.

        Called from both ends of the lifecycle. At start(), `source` is the
        newly selected authority: a stale jaw sample from the previous session
        must not let a fresh one engage. At stop(), `source` is None — nothing
        is being asked for any more, so `engaged`/`reason`/the hold timer all
        have to clear (an ended session that still advertises
        `{"engaged": true, "reason": "engaged"}` beside `"state": "idle"` is
        telling the operator the arms are live when they are not) — but
        `_clutch_source` itself is deliberately left alone.

        That last part is a considered choice, not an oversight: a session
        that ran in mouth mode and then stopped must keep reporting "mouth" in
        this diagnostic block, not silently revert to the default. Rewriting
        it to "spacebar" here made the block self-contradictory — a stopped
        mouth session claiming the spacebar was armed, which it never was.
        `start()` is the only place authority is actually chosen, and it
        always passes a concrete `source` here.

        The initial reason is honest about why a fresh mouth session cannot
        engage yet: no face has ever been seen, which is exactly what status()
        reports live as `stale`.
        """
        if source is not None:
            self._clutch_source = source
        self._mouth.reset()
        self._dead_man = False
        self._clutch_reason = (
            "spacebar_mode" if self._clutch_source == "spacebar" else "stale"
        )

    def _mouth_thresholds(self) -> tuple[float, float] | None:
        """Caller holds the lock."""
        return (mouth_clutch_thresholds(self._mouth_calib)
                if self._mouth_calib else None)

    def status(self) -> dict:
        with self._lock:
            cfg = self._cfg
            now = time.perf_counter()
            left_age = (now - self._last_left_perf) * 1000.0 if self._last_left_perf else None
            right_age = (now - self._last_right_perf) * 1000.0 if self._last_right_perf else None
            th = self._mouth_thresholds()
            face_stale = (self._clutch_source == "mouth"
                          and self._mouth.is_stale(now))
            return {
                "running": self.running,
                "state": self._state.value,
                "left_arm": cfg.left_arm if cfg else None,
                "right_arm": cfg.right_arm if cfg else None,
                "swap": cfg.swap if cfg else False,
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
                # Two different clocks in one block, on purpose. `engaged`
                # and `reason` are whatever the last ingested frame decided —
                # they only move when a frame arrives. `stale` is recomputed
                # here against the current time, so a face that stopped being
                # seen reads as stale from a poll alone.
                #
                # That asymmetry is safe because it is not what catches a
                # total frame stall: if frames stop entirely, the pre-existing
                # 300 ms per-side tracking-loss gate in _loop stops writing to
                # the arms regardless of what the clutch block still says. The
                # face budget only ever judges the freshness of the jaw signal
                # inside a stream of frames that is otherwise arriving.
                "clutch": {
                    "source": self._clutch_source,
                    "jaw_open": self._mouth.score,
                    "t_engage": th[0] if th else None,
                    "t_release": th[1] if th else None,
                    "engaged": self._dead_man,
                    "stale": face_stale,
                    "reason": self._clutch_reason,
                },
                # Authority transfer, per side. `engaged` above says the
                # operator is ASKING to drive; this says whether either arm has
                # actually been handed over, and if not, what is still in the
                # way. The two were the same event before acquisition existed.
                "acquire": {
                    "acquire_ms": self._acquire_ms,
                    "match_dwell_ms": self._match_dwell_ms,
                    "left":  self._acquire_status("left", now),
                    "right": self._acquire_status("right", now),
                },
            }

    def _acquire_status(self, side: str, now: float) -> dict:
        """One side's acquisition block. Caller holds the lock.

        `remaining_ms` is recomputed against the live clock rather than cached
        from the last tick, so the operator's countdown runs smoothly off a
        20 Hz telemetry poll instead of stepping with the commit loop.
        """
        acq = self._acq[side]
        committed = self._committed_left if side == "left" else self._committed_right
        remaining: float | None = None
        if acq.authority is SideAuthority.ACQUIRING and acq.since_perf is not None:
            remaining = max(0.0, self._acquire_ms - (now - acq.since_perf) * 1000.0)
        ramp: float | None = None
        if acq.authority is SideAuthority.DRIVING and acq.driving_since_perf is not None:
            ramp = min(1.0, (now - acq.driving_since_perf) * 1000.0
                       / max(1.0, self._acquire_ramp_ms))
        return {
            "authority": acq.authority.value,
            "reason": acq.reason,
            "matched": acq.matched,
            "blocking": list(acq.blocking),
            "error_deg": dict(acq.error_deg),
            "tol_deg": {j: self._tol_for(j) for j in acq.error_deg},
            "remaining_ms": remaining,
            "ramp": ramp,
            # Where the arm actually is, as a human arm pointing the same way.
            # Emitted whether or not the clutch is closed: the operator needs
            # it BEFORE they engage, to pre-position against.
            "ghost": self._ghost(side, committed),
        }

    def _tol_for(self, joint: str) -> float:
        return self._acquire_tol_deg.get(joint, self._acquire_tol_default_deg)

    def _side_mirrored(self, side: str) -> bool:
        """Whether this side's goals went through `apply_mirror`.

        Mirrors the call in `ingest_frame` exactly — left follows `swap`, right
        follows its inverse — because the ghost has to be un-mirrored back into
        the operator's own frame or it renders the wrong arm's pose.
        """
        swap = bool(self._cfg and self._cfg.swap)
        return swap if side == "left" else not swap

    def _ghost(self, side: str, committed: dict[str, float]) -> dict | None:
        """The arm's current pose as unit upper-arm/forearm vectors, in the
        operator's frame. Caller holds the lock."""
        if not committed:
            return None
        pan = committed.get("shoulder_pan", 0.0)
        if self._side_mirrored(side):
            # apply_mirror negates shoulder_pan and wrist_roll; only the former
            # reaches the arm's direction, and negating it again inverts it.
            pan = -pan
        upper, fore = retarget.arm_direction_vectors(
            pan,
            committed.get("shoulder_lift", 0.0),
            committed.get("elbow_flex", 0.0),
        )
        return {"upper": list(upper), "fore": list(fore)}

    def start(self, *, left_arm: str, right_arm: str, swap: bool, hz: float = 60.0,
              clutch_source: str = "spacebar") -> None:
        with self._lock:
            for _peer in self._peers:
                if getattr(_peer, "status", lambda: {})().get("running"):
                    raise RuntimeError("leader/follower teleop is running; stop it first")
            if self.running:
                raise RuntimeError("human teleop already running; stop it first")
            if left_arm == right_arm:
                raise ValueError("left_arm and right_arm must be different")
            if clutch_source not in CLUTCH_SOURCES:
                raise ValueError(
                    f"clutch_source must be one of {CLUTCH_SOURCES}, got {clutch_source!r}"
                )
            left = self._arms[left_arm]
            right = self._arms[right_arm]
            for a in (left, right):
                if not a.torque_enabled:
                    a.enable_torque()
                a.guard.set(Mode.MANUAL)
            effective_hz = self._hz_override or hz
            self._cfg = _SessionConfig(left_arm=left_arm, right_arm=right_arm,
                                       swap=swap, hz=effective_hz)
            self._started_at = time.time()
            self._state = HumanState.ARMED
            self._last_error = None
            # Clear every per-session transient. Two of these are load-bearing:
            #   _ws_disconnected_at_perf — set when the operator's tab closes,
            #     which happens *before* stop(); leaving it set makes the next
            #     session's first tick see an expired grace window and auto-stop.
            #   _target_left/_target_right — the last retarget goal of the
            #     previous session; leaving them set makes a freshly-ARMED
            #     session drift toward where the operator's hands used to be,
            #     before a single new frame has arrived.
            self._ws_disconnected_at_perf = None
            self._target_left = None
            self._target_right = None
            self._last_left_perf = 0.0
            self._last_right_perf = 0.0
            # A new session starts with neither arm handed over, whatever the
            # previous one ended holding.
            self._seen_frame = False
            for acq in self._acq.values():
                acq.release("clutch_open")
            self._reseed_pending = {"left": True, "right": True}
            # The membership check above is what narrows the bare str.
            self._reset_clutch_state(cast(ClutchSource, clutch_source))
            self._latest_frame_ts_ms = 0
            self._latest_arrival_perf = 0.0
            # Reset smoothing state to current observed positions where available.
            self._committed_left = self._observed_or_zero(left)
            self._committed_right = self._observed_or_zero(right)
            self._steps_left = self._held_steps(self._committed_left)
            self._steps_right = self._held_steps(self._committed_right)
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"haller-hmi-human-teleop-{left_arm}-{right_arm}",
            daemon=True,
        )
        self._thread.start()
        logger.info("human teleop started: left=%s right=%s swap=%s @ %.1f Hz",
                    left_arm, right_arm, swap, effective_hz)

    def stop(self) -> None:
        with self._lock:
            if not self.running:
                return
            cfg = self._cfg
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        # Restore arms to MANUAL with torque on.
        if cfg is not None:
            for arm_id in (cfg.left_arm, cfg.right_arm):
                handle = self._arms[arm_id]
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
            self._reset_clutch_state(None)
            # Same reasoning as the clutch block: a stopped session must not
            # still advertise an arm as DRIVING or mid-countdown.
            self._seen_frame = False
            for acq in self._acq.values():
                acq.release("idle")
        logger.info("human teleop stopped")

    def ingest_frame(self, frame: dict) -> None:
        """Apply a KeypointFrame from the browser. Thread-safe."""
        with self._lock:
            if not self.running:
                return
            # The session's authority is whatever start() was told, full stop.
            # The frame's own clutch_source is read only to detect a browser
            # asserting through the *other* source; it can never reassign
            # authority, or two frames (one to disengage, one to drive) would
            # be enough to hand the arms to the spacebar mid-session.
            frame_source = frame.get("clutch_source")
            source_mismatch = (
                frame_source is not None and str(frame_source) != self._clutch_source
            )
            now_perf_clutch = time.perf_counter()
            if self._clutch_source == "mouth":
                jaw = frame.get("jaw_open")
                engaged = self._mouth.observe(
                    now_perf_clutch,
                    float(jaw) if jaw is not None else None,
                    self._mouth_thresholds(),
                )
                self._clutch_reason = self._mouth.reason
            else:
                engaged = bool(frame.get("dead_man", False))
                self._clutch_reason = "spacebar_mode"
                self._mouth.above_since_t = None
            if source_mismatch:
                # Authority never hands over mid-motion. A frame speaking for
                # the source that does NOT hold authority disengages and wipes
                # the hold timer, so the operator has to re-assert through the
                # armed source deliberately. The mismatch itself is the fault
                # here — not "closed mouth" (below_threshold) and not "SPACE
                # holds authority" (spacebar_mode), both of which the frontend
                # treats as non-blocking resting states and prints nothing
                # for. Overriding whichever reason the branch above just set
                # is what makes this visible on the operator's chip instead
                # of silent.
                engaged = False
                self._mouth.above_since_t = None
                self._mouth.engaged = False
                self._clutch_reason = "source_mismatch"
            self._dead_man = engaged

            # Frame-carried calibration follows the same rule as the clutch
            # itself: a spacebar session must not be able to arm the mouth
            # clutch from frame data alone.
            mc = frame.get("mouth_calib")
            if mc and self._clutch_source == "mouth":
                self._mouth_calib = MouthClutchCalib(
                    talk_hold=float(mc["talk_hold"]),
                    open_hold=float(mc["open_hold"]),
                    talk_peak=(float(mc["talk_peak"])
                               if mc.get("talk_peak") is not None else None),
                )

            calib = frame.get("pinch_calib") or {}
            if "left" in calib:
                self._pinch_calib_left = calib["left"]
            if "right" in calib:
                self._pinch_calib_right = calib["right"]
            self._latest_frame_ts_ms = int(frame.get("ts_ms", 0))
            now_perf = time.perf_counter()
            self._latest_arrival_perf = now_perf
            # WS is healthy: cancel any pending grace window.
            self._ws_disconnected_at_perf = None

            mirror = bool(self._cfg and self._cfg.swap)
            left_side = frame.get("left")
            right_side = frame.get("right")
            # A side counts as tracked only when it produced a USABLE goal.
            #
            # `compute_joint_goal` returns None below the confidence floor, and
            # this used to store that None while still stamping the side as
            # freshly seen. The effect was invisible and nasty: one sub-floor
            # frame — routine when a hand is near the edge of frame — made the
            # side momentarily not-live, which released its acquisition and
            # restarted the 3 s countdown, while `tracking.age_ms` went on
            # reporting a healthy few milliseconds because a frame HAD
            # arrived. The operator saw a countdown that would not count down
            # and nothing anywhere saying why.
            #
            # Treating an unusable frame exactly like an absent one fixes both
            # halves: the 300 ms staleness budget now absorbs a flicker instead
            # of resetting on it, and a side that really is below the floor
            # ages out and reads as lost, which is the truth.
            if left_side is not None:
                goal = retarget.compute_joint_goal(
                    left_side, self._pinch_calib_left, mirror=mirror,
                )
                if goal is not None:
                    self._target_left = goal
                    self._last_left_perf = now_perf
            if right_side is not None:
                goal = retarget.compute_joint_goal(
                    right_side, self._pinch_calib_right, mirror=not mirror,
                )
                if goal is not None:
                    self._target_right = goal
                    self._last_right_perf = now_perf

            self._seen_frame = True
            # Evaluated here as well as in the commit loop, and for different
            # reasons: the loop is what advances a countdown when no frame
            # arrives, and this is what makes a RELEASE take effect on the
            # frame that reports it rather than up to a tick later. Release
            # must never wait for the loop.
            self._update_authority(now_perf)

    # ---- authority transfer ---------------------------------------------

    def _side_trackable(self, side: str, now: float) -> bool:
        last = self._last_left_perf if side == "left" else self._last_right_perf
        return last != 0.0 and (now - last) * 1000.0 <= self._frame_age_ms_loss

    def _update_authority(self, now: float) -> None:
        """Advance both sides' authority. Caller holds the lock.

        Idempotent and safe to call from either thread; it reads the clock the
        caller passes and nothing else. Sides are independent on purpose — one
        hand leaving frame must not freeze the arm the other hand is mid-reach
        with, and re-acquiring the side that dropped should not make the
        operator re-match a pose for the side that never did.
        """
        for side in ("left", "right"):
            acq = self._acq[side]
            target = self._target_left if side == "left" else self._target_right
            # `target is None` also covers a side below the confidence floor:
            # compute_joint_goal refuses to emit a goal there, so the
            # confidence gate is an acquisition gate for free.
            tracked = self._side_trackable(side, now) and target is not None
            if not self._dead_man or not tracked:
                if acq.authority is not SideAuthority.HELD:
                    # Coming back from a handover: re-read the arm before its
                    # position is trusted again.
                    self._reseed_pending[side] = True
                # Tracking outranks the clutch: an operator holding the
                # dead-man over a side the robot cannot see needs to be told
                # about the side, not about the clutch they are already
                # holding.
                acq.release("no_tracking" if not tracked else "clutch_open")
                continue

            if acq.authority is SideAuthority.HELD:
                acq.authority = SideAuthority.ACQUIRING
                acq.since_perf = now
                acq.matched_since_perf = None
                self._reseed_pending[side] = True

            limits = self._handle(side).joint_limits_deg
            committed = (self._committed_left if side == "left"
                         else self._committed_right)
            desired = self._target_deg(target, limits)
            # Against the CLAMPED target, so a joint the operator is asking to
            # drive past its limit matches at the limit. Comparing against the
            # raw ask would make such a pose permanently unmatchable, and the
            # arm cannot go there in any case.
            acq.error_deg = {j: v - committed.get(j, 0.0) for j, v in desired.items()}
            acq.blocking = tuple(j for j, e in acq.error_deg.items()
                                 if abs(e) > self._tol_for(j))
            acq.matched = not acq.blocking

            if acq.authority is SideAuthority.DRIVING:
                acq.reason = "driving"
                continue    # the error stays published; the ramp handles it

            if acq.matched:
                if acq.matched_since_perf is None:
                    acq.matched_since_perf = now
            else:
                acq.matched_since_perf = None
            acq.reason = "counting" if acq.matched else "matching"

            counted_down = (acq.since_perf is not None
                            and (now - acq.since_perf) * 1000.0 >= self._acquire_ms)
            dwelled = (acq.matched_since_perf is not None
                       and (now - acq.matched_since_perf) * 1000.0
                       >= self._match_dwell_ms)
            if counted_down and dwelled:
                acq.authority = SideAuthority.DRIVING
                acq.driving_since_perf = now
                # Here, not only on the next pass: for one tick the block would
                # otherwise advertise `authority: driving` beside
                # `reason: counting`, which is the kind of self-contradiction
                # that teaches an operator to stop reading the diagnostics.
                acq.reason = "driving"
                logger.info("human teleop: %s arm acquired (max error %.1f deg)",
                            side, max((abs(e) for e in acq.error_deg.values()),
                                      default=0.0))
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

    def _handle(self, side: str):
        """Caller holds the lock. Attribute reads only — no bus traffic."""
        assert self._cfg is not None
        return self._arms[self._cfg.left_arm if side == "left"
                          else self._cfg.right_arm]

    def _ramp_cap(self, side: str, period: float, now: float) -> float:
        """Per-tick rate cap for one side, in degrees.

        A time-varying cap rather than a blended offset: it composes with the
        clamp/reason machinery already in `_smooth_step` (the operator sees
        RATE-CAP while the ramp is biting, which is the truth), and it cannot
        make `committed` disagree with what was actually written.
        """
        full = self._rate_cap_deg_per_tick
        acq = self._acq[side]
        if acq.authority is not SideAuthority.DRIVING or acq.driving_since_perf is None:
            return full
        frac = (now - acq.driving_since_perf) * 1000.0 / max(1.0, self._acquire_ramp_ms)
        if frac >= 1.0:
            return full
        start = self._acquire_rate_deg_s * period
        return min(full, start + max(0.0, frac) * (full - start))

    def target_goals(self) -> dict:
        with self._lock:
            return {"left": self._target_left, "right": self._target_right}

    def notify_ws_disconnected(self) -> None:
        with self._lock:
            if not self.running:
                return
            self._ws_disconnected_at_perf = time.perf_counter()

    @staticmethod
    def _observed_or_zero(handle) -> dict[str, float]:
        """Seed the smoothing state from where the arm currently is.

        Goes through `read_joints_deg()` (the ArmHandle interface) rather than
        the underlying lerobot robot, so sim arms seed correctly too. Falls back
        to zero per joint if the read fails or omits a joint.
        """
        try:
            observed = handle.read_joints_deg()
            return {joint: float(observed.get(joint, 0.0))
                    for joint in handle.joint_limits_deg}
        except Exception:
            logger.warning("could not read start pose for arm %s; seeding at zero",
                           getattr(handle.config, "id", "?"), exc_info=True)
            return {joint: 0.0 for joint in handle.joint_limits_deg}

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
        """One retargeted joint value in degrees, unclamped.

        Special-cases the gripper: retarget emits [0, 1] (0 = closed, 1 = open),
        which is scaled onto the joint's calibrated degree range so that every
        comparison downstream — smoothing, rate caps, the acquisition match —
        is between two numbers in the same unit.
        """
        if joint == "gripper":
            return lo + max(0.0, min(1.0, value)) * (hi - lo)
        return value

    @classmethod
    def _target_deg(
        cls,
        target: dict[str, float] | None,
        limits: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """A retarget goal in degrees, clamped to the arm — i.e. the pose the
        arm would actually reach if it were handed this goal."""
        if target is None:
            return {}
        out: dict[str, float] = {}
        for joint, (lo, hi) in limits.items():
            if joint not in target:
                continue
            out[joint] = max(lo, min(hi, cls._to_degrees(
                joint, float(target[joint]), lo, hi)))
        return out

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
        cap = self._rate_cap_deg_per_tick if cap is None else cap
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

    @staticmethod
    def _commit(handle, goal: dict[str, float]) -> None:
        """Write one tick's goal through the ArmHandle interface.

        `send_goal` does the joint-limit clamp and the mode-guard check itself,
        and works against both `ArmHandle` and `SimArmHandle` — so the same loop
        drives real arms and MuJoCo arms.
        """
        handle.send_goal({joint: float(value) for joint, value in goal.items()})

    def _loop(self) -> None:
        with self._lock:
            cfg = self._cfg
        assert cfg is not None
        left = self._arms[cfg.left_arm]
        right = self._arms[cfg.right_arm]
        period = 1.0 / max(1.0, cfg.hz)
        # Smoothing time constant ≈ 100 ms (frequency-independent).
        tau_s = 0.100
        alpha = 1.0 - math.exp(-period / tau_s) if period > 0 else 1.0
        while not self._stop_flag.is_set():
            tick_start = time.perf_counter()
            try:
                # Re-read any side that is not being driven, BEFORE authority
                # is judged against its position. Outside the lock: this is bus
                # traffic on real hardware, and status() must not block on it.
                # Edge-triggered, so a steady session pays nothing.
                for side, handle in (("left", left), ("right", right)):
                    with self._lock:
                        pending = self._reseed_pending[side]
                    if not pending:
                        continue
                    observed = self._observed_or_zero(handle)
                    with self._lock:
                        self._reseed_pending[side] = False
                        if self._acq[side].authority is SideAuthority.DRIVING:
                            continue    # handed over while we were reading
                        if side == "left":
                            self._committed_left = observed
                            self._steps_left = self._held_steps(observed)
                        else:
                            self._committed_right = observed
                            self._steps_right = self._held_steps(observed)

                # Read the clock HERE, not at the top of the tick. Authority is
                # judged against a 300 ms freshness budget and the ramp against
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
                now = time.perf_counter()
                with self._lock:
                    self._update_authority(now)
                    # A side that is not DRIVING gets NO target, so its
                    # smoothing state stays where the arm is instead of slewing
                    # toward the operator. This is the bug that made engaging a
                    # lurch: `committed` used to track the operator's pose the
                    # whole time the clutch was open, so the first commit after
                    # engaging was a single step to wherever they were standing
                    # — the rate cap had been spent on an arm that never moved.
                    driving_left = (self._acq["left"].authority
                                    is SideAuthority.DRIVING)
                    driving_right = (self._acq["right"].authority
                                     is SideAuthority.DRIVING)
                    target_left = self._target_left if driving_left else None
                    target_right = self._target_right if driving_right else None
                    cap_left = self._ramp_cap("left", period, now)
                    cap_right = self._ramp_cap("right", period, now)
                steps_left = self._smooth_step(
                    self._committed_left, target_left, left.joint_limits_deg, alpha,
                    cap=cap_left,
                )
                steps_right = self._smooth_step(
                    self._committed_right, target_right, right.joint_limits_deg, alpha,
                    cap=cap_right,
                )
                committed_left = {j: s.committed for j, s in steps_left.items()}
                committed_right = {j: s.committed for j, s in steps_right.items()}
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
                # Authority IS the write gate. The per-side tracking-loss check
                # that used to live here moved into `_update_authority`, which
                # runs off the same `tick_start` a few microseconds earlier —
                # keeping it here as well would be a second opinion about
                # whether an arm may move, and two of those is how they come to
                # disagree. Losing a side now DEMOTES it rather than merely
                # skipping the write, so recovery re-runs acquisition.
                if driving_left:
                    self._commit(left, self._committed_left)
                if driving_right:
                    self._commit(right, self._committed_right)
                # WS disconnect grace window: if too much time has passed, auto-stop.
                with self._lock:
                    disc_at = self._ws_disconnected_at_perf
                if disc_at is not None and (time.perf_counter() - disc_at) > self._ws_disconnect_grace_s:
                    logger.info("human teleop WS disconnect grace exceeded; stopping")
                    threading.Thread(target=self.stop, daemon=True).start()
                    break
                with self._lock:
                    self._last_error = None
            except Exception as e:
                logger.exception("human teleop tick failed")
                with self._lock:
                    self._last_error = str(e)
                time.sleep(0.05)
                continue
            elapsed = time.perf_counter() - tick_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
