"""SimLeaderTeleop: drive a sim follower from a LeaderSource (mouse / replay).

Mirrors the structure of `TeleopSession` so the existing session-lock peer
wiring (`attach_peer`) accepts it as a coequal teleop kind.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from ..arm import ArmManager
from ..safety import Mode
from .sources import LeaderSource

logger = logging.getLogger(__name__)


@dataclass
class _State:
    running: bool = False
    follower: str | None = None
    hz: float = 0.0
    tick_count: int = 0
    last_error: str | None = None
    started_at: float | None = None


class SimLeaderTeleop:
    def __init__(self, arms: ArmManager):
        self._arms = arms
        self._state = _State()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peer = None
        self._source: LeaderSource | None = None

    def attach_peer(self, peer) -> None:
        self._peer = peer

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._state.running,
                "follower": self._state.follower,
                "hz": self._state.hz,
                "tick_count": self._state.tick_count,
                "last_error": self._state.last_error,
                "started_at": self._state.started_at,
            }

    def start(self, follower_id: str, source: LeaderSource, hz: float = 60.0) -> None:
        with self._lock:
            if self._state.running:
                raise RuntimeError("sim teleop already running")
            if self._peer is not None and getattr(self._peer, "status",
                                                  lambda: {})().get("running"):
                raise RuntimeError("another teleop is already running")
            follower = self._arms[follower_id]
            follower.guard.set(Mode.MANUAL)
            if not follower.torque_enabled:
                follower.enable_torque()
            self._source = source
            self._source.start()
            self._stop.clear()
            self._state = _State(
                running=True, follower=follower_id, hz=hz,
                tick_count=0, last_error=None, started_at=time.time(),
            )
        self._thread = threading.Thread(
            target=self._loop, name="haller-hmi-sim-teleop", daemon=True
        )
        self._thread.start()
        logger.info("sim teleop started -> %s @ %.1f Hz", follower_id, hz)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            if self._source is not None:
                try:
                    self._source.stop()
                except Exception:
                    logger.exception("sim teleop: source.stop() failed")
                self._source = None
            self._state.running = False

    def _loop(self) -> None:
        with self._lock:
            follower_id = self._state.follower
            hz = self._state.hz
            source = self._source
        assert follower_id is not None and source is not None
        follower = self._arms[follower_id]
        period = 1.0 / max(1.0, hz)

        while not self._stop.is_set():
            tick_start = time.perf_counter()
            try:
                goal = source.read()
                trimmed = {j: v for j, v in goal.items() if j in follower.joint_limits_deg}
                follower.send_goal(trimmed)
                with self._lock:
                    self._state.tick_count += 1
                    self._state.last_error = None
            except Exception as e:
                logger.exception("sim teleop tick failed")
                with self._lock:
                    self._state.last_error = str(e)
                time.sleep(0.05)
                continue
            slack = period - (time.perf_counter() - tick_start)
            if slack > 0:
                time.sleep(slack)
