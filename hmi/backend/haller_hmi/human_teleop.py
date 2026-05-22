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

from .arm import ArmManager
from .safety import Mode, clamp_joint_goal
from . import retarget

logger = logging.getLogger(__name__)


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


class HumanTeleopSession:
    """One global session. Mutually exclusive with leader/follower TeleopSession."""

    def __init__(
        self,
        arms: ArmManager,
        *,
        hz_override: float | None = None,
        frame_age_ms_loss: float = 300.0,
        ws_disconnect_grace_s: float = 5.0,
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
        self._hz_override = hz_override
        self._rate_cap_deg_per_tick = 4.0
        # T8: tracking-loss + WS disconnect grace
        self._frame_age_ms_loss = frame_age_ms_loss
        self._ws_disconnect_grace_s = ws_disconnect_grace_s
        self._ws_disconnected_at_perf: float | None = None
        # Per-arm last-frame timestamps (perf_counter), for tracking-loss.
        self._last_left_perf: float = 0.0
        self._last_right_perf: float = 0.0

    # ---- public API ------------------------------------------------------

    @property
    def state(self) -> HumanState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state is not HumanState.IDLE

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
            }

    def start(self, *, left_arm: str, right_arm: str, swap: bool, hz: float = 60.0) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("human teleop already running; stop it first")
            if left_arm == right_arm:
                raise ValueError("left_arm and right_arm must be different")
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
            # Reset smoothing state to current observed positions where available.
            self._committed_left = self._observed_or_zero(left)
            self._committed_right = self._observed_or_zero(right)
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
        logger.info("human teleop stopped")

    def ingest_frame(self, frame: dict) -> None:
        """Apply a KeypointFrame from the browser. Thread-safe."""
        with self._lock:
            if not self.running:
                return
            self._dead_man = bool(frame.get("dead_man", False))
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
        try:
            obs = handle.robot.get_observation() if handle.robot is not None else {}
        except Exception:
            obs = {}
        out: dict[str, float] = {}
        for joint in handle.joint_limits_deg:
            out[joint] = float(obs.get(f"{joint}.pos", 0.0))
        return out

    def _smooth_step(
        self,
        committed: dict[str, float],
        target: dict[str, float] | None,
        limits: dict[str, tuple[float, float]],
        alpha: float,
    ) -> dict[str, float]:
        if target is None:
            return committed
        out: dict[str, float] = {}
        for joint, lo_hi in limits.items():
            lo, hi = lo_hi
            cur = committed.get(joint, 0.0)
            if joint not in target:
                out[joint] = cur
                continue
            desired = float(target[joint])
            # Special-case gripper: retarget emits [0, 1] (0 = closed, 1 = open).
            # Scale onto the gripper joint's calibrated degree range.
            if joint == "gripper":
                desired = max(0.0, min(1.0, desired))
                desired = lo + desired * (hi - lo)
            # One-pole LPF then per-tick rate cap, then hard clamp to limits.
            new = cur + alpha * (desired - cur)
            cap = self._rate_cap_deg_per_tick
            new = max(cur - cap, min(cur + cap, new))
            out[joint] = max(lo, min(hi, new))
        return out

    def _commit(self, handle, goal: dict[str, float]) -> None:
        clamped = clamp_joint_goal(goal, handle.joint_limits_deg)
        action = {f"{joint}.pos": float(value) for joint, value in clamped.items()}
        if handle.robot is not None:
            handle.robot.send_action(action)

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
                self._committed_left = self._smooth_step(
                    self._committed_left, target_left, left.joint_limits_deg, alpha,
                )
                self._committed_right = self._smooth_step(
                    self._committed_right, target_right, right.joint_limits_deg, alpha,
                )
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
