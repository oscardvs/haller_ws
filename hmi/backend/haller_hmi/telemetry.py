# hmi/backend/haller_hmi/telemetry.py
"""Broadcasts a state frame to N subscribers at a fixed rate."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class TelemetryBroadcaster:
    def __init__(self, arms, ros, hz: float = 20.0, teleop=None):
        self._arms = arms
        self._ros = ros
        self._teleop = teleop
        self._period = 1.0 / hz
        self._subscribers: list[asyncio.Queue] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.get_event_loop().create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        self._task = None

    def subscribe(self) -> AsyncIterator[dict]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._subscribers.append(queue)
        async def gen():
            try:
                while True:
                    frame = await queue.get()
                    yield frame
            finally:
                self._subscribers.remove(queue)
        return gen()

    def _build_frame(self) -> dict:
        base_snap = self._ros.snapshot()
        frame = {
            "t": time.time(),
            "base": {
                "linear": base_snap.linear,
                "angular": base_snap.angular,
                "odom": dict(base_snap.odom),
                "scan_min_range": base_snap.scan_min_range,
            },
            "arms": {},
            "alerts": [],
            "teleop": self._teleop.status() if self._teleop is not None else {"running": False},
        }
        for arm_id in self._arms.keys():
            try:
                frame["arms"][arm_id] = self._arms[arm_id].state_snapshot()
            except Exception as e:
                logger.warning("arm %s telemetry failed: %s", arm_id, e)
                frame["alerts"].append({
                    "level": "warn",
                    "code": "arm_telemetry_failed",
                    "message": str(e),
                    "source": f"arm:{arm_id}",
                })
        return frame

    async def _run(self) -> None:
        while not self._stop.is_set():
            tick_start = time.perf_counter()
            try:
                frame = self._build_frame()
                for q in list(self._subscribers):
                    if q.full():
                        # drop oldest to keep latency bounded
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    q.put_nowait(frame)
            except Exception as e:
                logger.exception("telemetry tick failed: %s", e)
            elapsed = time.perf_counter() - tick_start
            await asyncio.sleep(max(0.0, self._period - elapsed))
