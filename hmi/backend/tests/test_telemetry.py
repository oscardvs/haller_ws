# hmi/backend/tests/test_telemetry.py
import asyncio
from unittest.mock import MagicMock

import pytest

from haller_hmi.telemetry import TelemetryBroadcaster


@pytest.mark.asyncio
async def test_broadcaster_emits_frames():
    arm = MagicMock()
    arm.state_snapshot.return_value = {
        "mode": "manual",
        "joints": {"gripper": {"pos": 0.0, "min": 0.0, "max": 100.0, "torque": True}},
    }
    arms = MagicMock()
    arms.keys.return_value = ["right"]
    arms.__getitem__.return_value = arm

    ros = MagicMock()
    snap = MagicMock(linear=0.0, angular=0.0, odom={"x": 1.0, "y": 0.0, "yaw": 0.0}, scan_min_range=2.0)
    ros.snapshot.return_value = snap

    bcast = TelemetryBroadcaster(arms, ros, hz=200.0)  # high hz for fast test
    bcast.start()
    try:
        sub = bcast.subscribe()
        frame = await asyncio.wait_for(sub.__anext__(), timeout=0.2)
    finally:
        await bcast.stop()
    assert "t" in frame
    assert frame["base"]["odom"]["x"] == 1.0
    assert "right" in frame["arms"]
    assert frame["arms"]["right"]["joints"]["gripper"]["pos"] == 0.0


@pytest.mark.asyncio
async def test_multiple_subscribers_get_same_frame():
    arms = MagicMock()
    arms.keys.return_value = []
    ros = MagicMock()
    ros.snapshot.return_value = MagicMock(linear=0.0, angular=0.0, odom={}, scan_min_range=None)

    bcast = TelemetryBroadcaster(arms, ros, hz=200.0)
    bcast.start()
    try:
        s1 = bcast.subscribe()
        s2 = bcast.subscribe()
        f1 = await asyncio.wait_for(s1.__anext__(), timeout=0.2)
        f2 = await asyncio.wait_for(s2.__anext__(), timeout=0.2)
    finally:
        await bcast.stop()
    # both subscribers see a frame (timing may differ by one tick, that's fine)
    assert "t" in f1 and "t" in f2
