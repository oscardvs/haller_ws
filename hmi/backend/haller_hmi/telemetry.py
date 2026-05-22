# hmi/backend/haller_hmi/telemetry.py
"""Broadcasts a state frame to N subscribers at a fixed rate."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class TelemetryBroadcaster:
    def __init__(self, arms, ros, hz: float = 20.0, teleop=None, human_teleop=None, calibration=None):
        self._arms = arms
        self._ros = ros
        self._teleop = teleop
        self._human_teleop = human_teleop
        self._calibration = calibration
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
            "human_teleop": self._human_teleop.status() if self._human_teleop is not None else {"running": False, "state": "idle"},
        }
        active = (
            self._calibration.current
            if self._calibration is not None
            else None
        )
        for arm_id in self._arms.keys():
            try:
                snap = self._arms[arm_id].state_snapshot()
            except Exception as e:
                logger.warning("arm %s telemetry failed: %s", arm_id, e)
                frame["alerts"].append({
                    "level": "warn",
                    "code": "arm_telemetry_failed",
                    "message": str(e),
                    "source": f"arm:{arm_id}",
                })
                continue
            if active is not None and active.arm_id == arm_id:
                cal_block = self._calibration_block(active, arm_id)
                snap["calibration"] = cal_block
                if cal_block.get("error"):
                    frame["alerts"].append({
                        "level": "error",
                        "code": "calibration_bus_error",
                        "message": cal_block["error"],
                        "source": f"arm:{arm_id}",
                    })
                    # Single-tick auto-abort per spec §6. The next tick will not emit a
                    # calibration block (current is None), which the frontend uses as the
                    # signal that the session ended.
                    if self._calibration is not None:
                        self._calibration.abort()
            frame["arms"][arm_id] = snap
        return frame

    def _calibration_block(self, session, arm_id: str) -> dict:
        block: dict = {"state": session.state.value}
        try:
            handle = self._arms[arm_id]
            if session.state.value == "homing":
                bus = handle.robot.bus
                motors = list(bus.motors.keys())
                block["ticks"] = {m: int(v) for m, v in
                                  bus.sync_read("Present_Position", motors, normalize=False).items()}
            elif session.state.value == "sweeping":
                block["ticks"] = session.tick_sweep(handle)
                block["min"] = dict(session.mins)
                block["max"] = dict(session.maxes)
        except Exception as e:
            logger.warning("calibration tick failed: %s", e)
            block["error"] = str(e)
        return block

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
