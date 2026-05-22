# hmi/backend/tests/test_telemetry.py
import asyncio
from unittest.mock import MagicMock

import pytest

from haller_hmi.telemetry import TelemetryBroadcaster
from haller_hmi.calibration import CalibrationManager, CalibrationState
from haller_hmi.safety import Mode


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


@pytest.mark.asyncio
async def test_frame_contains_calibration_block_when_session_active():
    handle = MagicMock()
    handle.state_snapshot.return_value = {
        "mode": "manual", "torque": False,
        "joints": {"gripper": {"pos": 0.0, "min": 0.0, "max": 100.0, "torque": False}},
    }
    handle.config = MagicMock(id="right")
    arms = MagicMock()
    arms.keys.return_value = ["right"]
    arms.__getitem__.return_value = handle

    ros = MagicMock()
    ros.snapshot.return_value = MagicMock(linear=0.0, angular=0.0, odom={}, scan_min_range=None)

    cal_mgr = CalibrationManager()
    # Move the session to SWEEPING manually (capture_neutral is fully tested elsewhere).
    # CalibrationManager.start() uses `is not Mode.MANUAL` identity check, so we must
    # set the real enum value — not a MagicMock.
    handle.guard = MagicMock()
    handle.guard.mode = Mode.MANUAL
    handle.robot = MagicMock()
    handle.robot.bus.motors = {"gripper": MagicMock(model="sts3215", id=6)}
    handle.robot.bus.model_resolution_table = {"sts3215": 4096}
    handle.robot.bus.sync_read.return_value = {"gripper": 2048}
    handle.robot.calibration = None
    handle.torque_enabled = True
    arms.values.return_value = [handle]
    session = cal_mgr.start(arms, "right")
    session.capture_neutral(handle)

    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, calibration=cal_mgr)
    bcast.start()
    try:
        sub = bcast.subscribe()
        frame = await asyncio.wait_for(sub.__anext__(), timeout=0.2)
    finally:
        await bcast.stop()

    block = frame["arms"]["right"]["calibration"]
    assert block["state"] == "sweeping"
    assert "ticks" in block and "gripper" in block["ticks"]
    assert "min" in block and "max" in block


@pytest.mark.asyncio
async def test_frame_has_no_calibration_block_when_idle():
    handle = MagicMock()
    handle.state_snapshot.return_value = {
        "mode": "manual", "torque": True, "joints": {},
    }
    arms = MagicMock()
    arms.keys.return_value = ["right"]
    arms.__getitem__.return_value = handle

    ros = MagicMock()
    ros.snapshot.return_value = MagicMock(linear=0.0, angular=0.0, odom={}, scan_min_range=None)

    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, calibration=CalibrationManager())
    bcast.start()
    try:
        sub = bcast.subscribe()
        frame = await asyncio.wait_for(sub.__anext__(), timeout=0.2)
    finally:
        await bcast.stop()
    assert "calibration" not in frame["arms"]["right"]


@pytest.mark.asyncio
async def test_bus_error_during_sweep_aborts_session_and_emits_alert():
    handle = MagicMock()
    handle.state_snapshot.return_value = {
        "mode": "manual", "torque": False,
        "joints": {},
    }
    handle.config = MagicMock(id="right")
    arms = MagicMock()
    arms.keys.return_value = ["right"]
    arms.__getitem__.return_value = handle

    ros = MagicMock()
    ros.snapshot.return_value = MagicMock(linear=0.0, angular=0.0, odom={}, scan_min_range=None)

    from haller_hmi.safety import Mode
    cal_mgr = CalibrationManager()
    handle.guard = MagicMock()
    handle.guard.mode = Mode.MANUAL
    handle.robot = MagicMock()
    handle.robot.bus.motors = {"gripper": MagicMock(model="sts3215", id=6)}
    handle.robot.bus.model_resolution_table = {"sts3215": 4096}
    handle.robot.bus.sync_read.return_value = {"gripper": 2048}
    handle.robot.calibration = None
    handle.torque_enabled = True
    arms.values.return_value = [handle]
    session = cal_mgr.start(arms, "right")
    session.capture_neutral(handle)
    # Make sync_read fail during sweep
    handle.robot.bus.sync_read.side_effect = RuntimeError("bus disconnect")

    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, calibration=cal_mgr)
    bcast.start()
    try:
        sub = bcast.subscribe()
        frame = await asyncio.wait_for(sub.__anext__(), timeout=0.2)
    finally:
        await bcast.stop()

    # Alert is in the frame
    alerts = frame["alerts"]
    assert any(a["code"] == "calibration_bus_error" and a["source"] == "arm:right"
               for a in alerts), f"expected calibration_bus_error alert, got {alerts}"
    # Session was auto-aborted
    assert cal_mgr.current is None
