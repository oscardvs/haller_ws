# hmi/backend/haller_hmi/motion.py
"""Discrete arm moves: bounded, cancellable, and shared between real and sim.

The HTTP routes that trigger a move are `async def` calling synchronous motion
code, so a ramp must never run inline — it would stall the event loop for every
other client. Hence the background executor.

Cancellation is not a separate channel: `/estop` already sets Mode.STOP on every
arm's guard, so re-checking the guard before each waypoint means E-STOP, a mode
change and a teleop takeover all stop a ramp for free.
"""
from __future__ import annotations

import logging
import threading
import time

from .safety import ModeError

logger = logging.getLogger(__name__)


class MoveRefused(Exception):
    """A discrete move was rejected before any motion was commanded.

    Deliberately not `calibration.ConflictError`: motion must not depend on the
    calibration module. Routes map this to HTTP 409.
    """


class MoveExecutor:
    """Plays ramp waypoints on a background thread, one arm per instance."""

    def __init__(self, handle) -> None:
        self._handle = handle
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        #: Set by `_play` when a waypoint raises anything other than
        #: ModeError (e.g. a comm failure). None means either "no run yet"
        #: or "the most recent run finished without one" — cleared at the
        #: start of every `run()`, so it never outlives the ramp it describes.
        self.last_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        # Snapshot under the lock: `run()` holds `_lock` across both
        # constructing the Thread and calling `.start()`, so a lock-taking
        # read here can only ever observe a thread that is either fully
        # started already or not yet assigned — never the gap in between.
        with self._lock:
            t = self._thread
        return t is not None and t.is_alive()

    def run(self, waypoints: list[dict[str, float]], hz: float) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        with self._lock:
            self._cancel_locked()
            self.last_error = None
            self._cancel = threading.Event()
            self._thread = threading.Thread(
                target=self._play,
                args=(waypoints, hz, self._cancel),
                name=f"move-{getattr(self._handle.config, 'id', '?')}",
                daemon=True,
            )
            self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            self._cancel_locked()

    def _cancel_locked(self) -> None:
        self._cancel.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
            if t.is_alive():
                arm_id = getattr(self._handle.config, "id", "?")
                logger.error(
                    "move on arm %s: ramp thread did not stop within 2.0s of "
                    "cancellation; it may still be commanding",
                    arm_id,
                )

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current ramp finishes. Test helper."""
        # Same snapshot-under-the-lock reasoning as `is_running`: without it,
        # a caller can read `self._thread` in the gap between `run()`
        # assigning it and calling `.start()`, and `Thread.join()` raises
        # RuntimeError on a thread that has not been started yet.
        with self._lock:
            t = self._thread
        if t is not None:
            t.join(timeout=timeout)

    def _play(self, waypoints, hz: float, cancel: threading.Event) -> None:
        period = 1.0 / hz
        arm_id = getattr(self._handle.config, "id", "?")
        for waypoint in waypoints:
            if cancel.is_set():
                logger.warning("move on arm %s cancelled", arm_id)
                return
            try:
                self._handle.guard.assert_manual()
                self._handle.send_goal(waypoint)
            except ModeError:
                logger.warning("move on arm %s stopped: mode left manual", arm_id)
                return
            except Exception as exc:
                self.last_error = exc
                logger.exception(
                    "move on arm %s: ramp failed sending a waypoint", arm_id
                )
                return
            time.sleep(period)
