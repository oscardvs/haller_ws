# hmi/backend/haller_hmi/telemetry.py
"""Broadcasts a state frame to N subscribers at a fixed rate.

Since Phase 2 this is a DECIMATING CONSUMER of the tick, not a sampler. It no
longer reads the arms itself: it takes the newest `TickSample` whenever it
wakes, so the arm state a subscriber sees is the same moment the recorder
wrote and the same moment the session committed against (invariant 8).

`latest()` on the bus rather than a subscription, deliberately. A queue would
overflow continuously between telemetry's 20 Hz wakeups and count drops that
mean nothing — this consumer is decimating BY DESIGN and loses nothing by
skipping samples it never wanted. That is contract C2's failure mode arriving
at a different consumer, and the fix is to not hold a queue at all.

Base, teleop/human-teleop status and the calibration block stay telemetry's
own. None of them is Feetech bus traffic, none was part of the duplicated
read, and the recorder takes its own `base` from the sample where pairing
actually reaches a dataset column.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

from .tick import plain

logger = logging.getLogger(__name__)


class TelemetryBroadcaster:
    def __init__(self, arms, ros, hz: float = 20.0, teleop=None, human_teleop=None,
                 calibration=None, tick_bus=None, tick_stale_after_s: float = 0.25):
        self._arms = arms
        # THE tick. None falls back to reading the arms directly, which is
        # what this class did before Phase 2 — see `_fresh_sample`.
        self._tick_bus = tick_bus
        # How old the newest sample may be before this falls back, in SECONDS.
        # A count of ticks would mean something different at every telemetry
        # rate; 0.25 s is comfortably above any sane producer cadence (a 20 Hz
        # idle sampler is 50 ms) and far below the point where a frozen
        # cockpit reads as a dead one.
        self._tick_stale_after_s = tick_stale_after_s
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

    def _fresh_sample(self):
        """The newest tick, if there is one and it is recent enough.

        Returns None when there is no bus, nothing has been published, or the
        newest sample is older than `tick_stale_after_s` — all three of which
        mean the same thing to the caller: nobody is producing the tick right
        now, so read the arms directly and say so.
        """
        if self._tick_bus is None:
            return None
        sample = self._tick_bus.latest()
        if sample is None:
            return None
        if (time.perf_counter() - sample.t_mono) > self._tick_stale_after_s:
            return None
        return sample

    def _base_block(self) -> dict:
        base_snap = self._ros.snapshot()
        return {
            "linear": base_snap.linear,
            "angular": base_snap.angular,
            "odom": dict(base_snap.odom),
            "scan_min_range": base_snap.scan_min_range,
        }

    def _read_arms_directly(self, frame: dict) -> dict:
        """Pre-Phase-2 behaviour, kept for when nothing is producing the tick.

        EXPIRY: this branch stops being reachable in normal operation once the
        `IdleSampler` is mounted in `server.py`'s lifespan, because something
        then produces the tick whenever no session does. It is not deleted at
        that point — a producer can still die — but it stops being the
        ordinary path, which is why it announces itself in `alerts` rather
        than quietly standing in. A fallback nobody can see is how a rig runs
        for a week on the path that was supposed to be the exception.
        """
        snaps: dict = {}
        # `.keys()` and it must stay `.keys()` — ArmManager has
        # __getitem__/keys/values and no __iter__, so the dict form ruff
        # suggests here raises `KeyError: unknown arm id 0` on every
        # call. Same trap as human_teleop._sample_arms; an autofixer is a
        # caller that has not read your class.
        for arm_id in self._arms.keys():  # noqa: SIM118  (ArmManager, not a dict)
            try:
                snaps[arm_id] = self._arms[arm_id].state_snapshot()
            except Exception as e:
                logger.warning("arm %s telemetry failed: %s", arm_id, e)
                frame["alerts"].append({
                    "level": "warn",
                    "code": "arm_telemetry_failed",
                    "message": str(e),
                    "source": f"arm:{arm_id}",
                })
        return snaps

    def _build_frame(self) -> dict:
        sample = self._fresh_sample()
        frame = {
            "t": time.time() if sample is None else sample.t_unix,
            "base": self._base_block(),
            "arms": {},
            "alerts": [],
            "teleop": self._teleop.status() if self._teleop is not None else {"running": False},
            "human_teleop": self._human_teleop.status() if self._human_teleop is not None else {"running": False, "state": "idle"},
        }
        if sample is None:
            snaps = self._read_arms_directly(frame)
            if self._tick_bus is not None:
                frame["alerts"].append({
                    "level": "warn",
                    "code": "tick_bus_idle",
                    "message": ("no tick published recently; arm state read "
                                "directly and is not paired with any action"),
                    "source": "tick",
                })
        else:
            # `plain()` because these go to `ws.send_json` and a sample's maps
            # are read-only views, which json.dumps refuses outright.
            snaps = {arm_id: plain(snap) for arm_id, snap in sample.arms.items()}
            for arm_id, message in sample.arm_errors.items():
                logger.warning("arm %s telemetry failed: %s", arm_id, message)
                frame["alerts"].append({
                    "level": "warn",
                    "code": "arm_telemetry_failed",
                    "message": message,
                    "source": f"arm:{arm_id}",
                })

        active = (
            self._calibration.current
            if self._calibration is not None
            else None
        )
        if active is not None and active.arm_id not in snaps:
            # Mid-session the arm often has no snapshot at all: capture resets
            # the bus calibration, so every normalized read fails until save
            # registers the new one. Block-presence is the frontend's liveness
            # signal (spec §6), and tick_sweep records the sweep only where the
            # block is built — so the block must not ride on the snapshot.
            snaps[active.arm_id] = {}
        for arm_id, snap in snaps.items():
            if active is not None and active.arm_id == arm_id:
                cal_block = self._calibration_block(active, arm_id)
                # A new dict rather than a mutation: a bus-backed snapshot is
                # a frozen copy shared with every other consumer of this tick,
                # and the calibration block belongs to this frame alone.
                snap = {**snap, "calibration": cal_block}
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
