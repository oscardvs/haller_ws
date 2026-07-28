"""Bimanual human-pose teleop session.

This is the sibling of `teleop.TeleopSession` (leader/follower). Where that
session reads positions off a physical leader arm at 60 Hz, this one reads
keypoints off a WebSocket from the operator's browser and runs them through
`retarget.compute_joint_goal` to produce joint angles. Otherwise the lifecycle
and safety semantics match exactly.

State machine:
    IDLE → (start)        → ARMED
    ARMED → (first frame) → TRACKING
    TRACKING ↔ (dead-man) → DRIVING
    any → (stop / E-STOP) → IDLE
"""
from __future__ import annotations

import enum
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Literal

from .arm import ArmManager
from .safety import (
    Mode,
    MouthClutchCalib,
    mouth_clutch_thresholds,
    mouth_clutch_decision,
)
from . import retarget

logger = logging.getLogger(__name__)


#: Which source holds dead-man authority. Exactly these two strings, mirrored
#: in TypeScript as `ClutchSource` — nothing else may reach the session.
ClutchSource = Literal["spacebar", "mouth"]
CLUTCH_SOURCES: tuple[str, ...] = ("spacebar", "mouth")


class HumanState(str, enum.Enum):
    IDLE = "idle"
    ARMED = "armed"
    TRACKING = "tracking"
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
        # Mouth-clutch state. _last_face_perf mirrors the per-side staleness
        # pattern above: 0.0 means "no sample has ever arrived", which reads
        # as stale and therefore disengaged.
        self._face_stale_ms = face_stale_ms
        # Set once by start() and never by a frame: the operator selects the
        # authority before the session exists, and the browser cannot hand it
        # over afterwards (spec 1, 6.4).
        self._clutch_source: ClutchSource = "spacebar"
        self._mouth_calib: MouthClutchCalib | None = None
        self._jaw_open: float | None = None
        self._last_face_perf: float = 0.0
        self._jaw_above_since_perf: float | None = None
        self._clutch_reason: str = "spacebar_mode"

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
        """Store per-operator jaw-open calibration. None clears it."""
        with self._lock:
            self._mouth_calib = (
                MouthClutchCalib(talk_max=float(calib["talk_max"]),
                                 open_min=float(calib["open_min"]))
                if calib else None
            )

    def mouth_calib_is_valid(self) -> bool:
        """True when a calibration is set AND yields safe thresholds."""
        with self._lock:
            return (self._mouth_calib is not None
                    and mouth_clutch_thresholds(self._mouth_calib) is not None)

    def _reset_clutch_state(self, source: ClutchSource) -> None:
        """Clear every clutch transient and arm `source`. Caller holds the lock.

        Called from both ends of the lifecycle. At start(), for the same
        reason as the per-side timestamps: a stale jaw sample from the previous
        session must not let a fresh one engage. At stop(), because nothing is
        being asked for any more — an ended session that still advertises
        `{"engaged": true, "reason": "engaged"}` beside `"state": "idle"` is
        telling the operator the arms are live when they are not.

        The initial reason is honest about why a fresh mouth session cannot
        engage yet: `_last_face_perf == 0.0` means no face has ever been seen,
        which is exactly what status() reports live as `stale`.
        """
        self._clutch_source = source
        self._jaw_open = None
        self._last_face_perf = 0.0
        self._jaw_above_since_perf = None
        self._dead_man = False
        self._clutch_reason = "spacebar_mode" if source == "spacebar" else "stale"

    def _mouth_engaged(self, now_perf: float) -> bool:
        """Evaluate the mouth clutch. Caller holds the lock.

        Owns only the stateful parts — the clock reads and the hold timer.
        The actual engage/release policy is safety.mouth_clutch_decision.
        """
        th = (mouth_clutch_thresholds(self._mouth_calib)
              if self._mouth_calib else None)
        if th is None:
            self._jaw_above_since_perf = None
            self._clutch_reason = "uncalibrated"
            return False

        t_engage, _t_release = th
        stale = (
            self._last_face_perf == 0.0
            or (now_perf - self._last_face_perf) * 1000.0 > self._face_stale_ms
        )

        if self._jaw_open is not None and not stale and self._jaw_open >= t_engage:
            if self._jaw_above_since_perf is None:
                self._jaw_above_since_perf = now_perf
            # Measure the hold across OBSERVED samples, not the wall clock.
            # `_jaw_open` deliberately survives `jaw_open: null` frames, and
            # FACE_STALE_MS (250) is above MOUTH_HOLD_MS (200) — so a wall-clock
            # timer lets one above-threshold sample followed by a face dropout
            # satisfy the hold with nothing confirming it, making tracking loss
            # ENGAGE the arms. `_jaw_above_since_perf` was set inside the same
            # ingest that recorded a sample, so this is the span between the
            # first and the most recent REAL observation: it reads 0 on the
            # crossing frame, and engaging needs at least two real samples
            # MOUTH_HOLD_MS apart.
            held_ms = (self._last_face_perf - self._jaw_above_since_perf) * 1000.0
        else:
            self._jaw_above_since_perf = None
            held_ms = 0.0

        engaged = mouth_clutch_decision(
            self._jaw_open, th, held_ms, stale, self._dead_man,
        )

        if stale:
            self._clutch_reason = "stale"
        elif engaged:
            self._clutch_reason = "engaged"
        elif self._jaw_above_since_perf is not None:
            self._clutch_reason = "holding"
        else:
            self._clutch_reason = "below_threshold"
        return engaged

    def status(self) -> dict:
        with self._lock:
            cfg = self._cfg
            now = time.perf_counter()
            left_age = (now - self._last_left_perf) * 1000.0 if self._last_left_perf else None
            right_age = (now - self._last_right_perf) * 1000.0 if self._last_right_perf else None
            th = (mouth_clutch_thresholds(self._mouth_calib)
                  if self._mouth_calib else None)
            face_stale = (
                self._clutch_source == "mouth"
                and (self._last_face_perf == 0.0
                     or (now - self._last_face_perf) * 1000.0 > self._face_stale_ms)
            )
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
                "clutch": {
                    "source": self._clutch_source,
                    "jaw_open": self._jaw_open,
                    "t_engage": th[0] if th else None,
                    "t_release": th[1] if th else None,
                    "engaged": self._dead_man,
                    "stale": face_stale,
                    "reason": self._clutch_reason,
                },
            }

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
            self._reset_clutch_state(clutch_source)   # type: ignore[arg-type]
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
            self._reset_clutch_state("spacebar")
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
                if jaw is not None:
                    self._jaw_open = float(jaw)
                    self._last_face_perf = now_perf_clutch
                engaged = self._mouth_engaged(now_perf_clutch)
            else:
                engaged = bool(frame.get("dead_man", False))
                self._clutch_reason = "spacebar_mode"
                self._jaw_above_since_perf = None
            if source_mismatch:
                # Authority never hands over mid-motion. A frame speaking for
                # the source that does NOT hold authority disengages and wipes
                # the hold timer, so the operator has to re-assert through the
                # armed source deliberately.
                engaged = False
                self._jaw_above_since_perf = None
                if self._clutch_source == "mouth":
                    # Do not leave the previous frame's "engaged" standing
                    # beside engaged=False for a tick.
                    self._clutch_reason = "below_threshold"
            self._dead_man = engaged

            # Frame-carried calibration follows the same rule as the clutch
            # itself: a spacebar session must not be able to arm the mouth
            # clutch from frame data alone.
            mc = frame.get("mouth_calib")
            if mc and self._clutch_source == "mouth":
                self._mouth_calib = MouthClutchCalib(
                    talk_max=float(mc["talk_max"]), open_min=float(mc["open_min"]),
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
            if left_side is not None:
                self._target_left = retarget.compute_joint_goal(
                    left_side, self._pinch_calib_left, mirror=mirror,
                )
                self._last_left_perf = now_perf
            if right_side is not None:
                self._target_right = retarget.compute_joint_goal(
                    right_side, self._pinch_calib_right, mirror=not mirror,
                )
                self._last_right_perf = now_perf

            if self._state is HumanState.ARMED:
                self._state = HumanState.TRACKING
            if self._state is HumanState.TRACKING and self._dead_man:
                self._state = HumanState.DRIVING
            elif self._state is HumanState.DRIVING and not self._dead_man:
                self._state = HumanState.TRACKING

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

    def _smooth_step(
        self,
        committed: dict[str, float],
        target: dict[str, float] | None,
        limits: dict[str, tuple[float, float]],
        alpha: float,
    ) -> dict[str, JointStep]:
        out: dict[str, JointStep] = {}
        for joint, lo_hi in limits.items():
            lo, hi = lo_hi
            cur = committed.get(joint, 0.0)
            if target is None or joint not in target:
                out[joint] = JointStep(target=None, committed=cur, reason="held")
                continue
            desired = float(target[joint])
            # Special-case gripper: retarget emits [0, 1] (0 = closed, 1 = open).
            # Scale onto the gripper joint's calibrated degree range so that
            # `target` and `committed` are always the same unit.
            if joint == "gripper":
                desired = max(0.0, min(1.0, desired))
                desired = lo + desired * (hi - lo)
            # One-pole LPF, then per-tick rate cap, then hard clamp to limits.
            # Each stage records whether it altered the value. Exact float
            # equality is correct here: these clamps return their input
            # bitwise unchanged when they don't bite.
            lpf = cur + alpha * (desired - cur)
            cap = self._rate_cap_deg_per_tick
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
                with self._lock:
                    target_left = self._target_left
                    target_right = self._target_right
                    driving = self._state is HumanState.DRIVING
                steps_left = self._smooth_step(
                    self._committed_left, target_left, left.joint_limits_deg, alpha,
                )
                steps_right = self._smooth_step(
                    self._committed_right, target_right, right.joint_limits_deg, alpha,
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
                if driving:
                    # Gate per-side: don't write to an arm whose tracking is lost.
                    now_perf = time.perf_counter()
                    left_age_ms = (now_perf - self._last_left_perf) * 1000.0 if self._last_left_perf else float("inf")
                    right_age_ms = (now_perf - self._last_right_perf) * 1000.0 if self._last_right_perf else float("inf")
                    if left_age_ms <= self._frame_age_ms_loss:
                        self._commit(left, self._committed_left)
                    if right_age_ms <= self._frame_age_ms_loss:
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
