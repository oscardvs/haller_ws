# hmi/backend/tests/test_telemetry_tick.py
"""Telemetry as a decimating CONSUMER of the tick (Phase 2b).

The pre-Phase-2 tests in `test_telemetry.py` construct a broadcaster with no
bus and still pass — that is the fallback path, and it staying green is the
point. These cover the bus-backed path.
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

from haller_hmi.arm import ArmManager
from haller_hmi.telemetry import TelemetryBroadcaster
from haller_hmi.tick import TickBus


def _snapshot(pos: float = 1.0, **extra):
    return {
        "mode": "manual", "torque": True,
        "joints": {"gripper": {"pos": pos, "min": 0.0, "max": 100.0,
                               "torque": True, "effort": -0.42, **extra}},
    }


def _arms_and_ros():
    arm = MagicMock()
    arm.state_snapshot.return_value = _snapshot(pos=9.0)
    arms = MagicMock(spec=ArmManager)
    arms.keys.return_value = ["right"]
    arms.__getitem__.return_value = arm
    ros = MagicMock()
    ros.snapshot.return_value = MagicMock(
        linear=0.0, angular=0.0, odom={"x": 1.0}, scan_min_range=2.0)
    return arms, arm, ros


def _bus_with(sample_arms, arm_errors=None):
    bus = TickBus()
    bus.publish_once("test", t_mono=time.perf_counter(), t_unix=1234.5,
                     arms=sample_arms, arm_errors=arm_errors or {})
    return bus


def test_a_bus_backed_frame_does_not_read_the_arms_itself():
    """The point of 2b: the duplicated Feetech read is gone."""
    arms, arm, ros = _arms_and_ros()
    bus = _bus_with({"right": _snapshot(pos=1.0)})
    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, tick_bus=bus)

    frame = bcast._build_frame()

    arm.state_snapshot.assert_not_called()
    assert frame["arms"]["right"]["joints"]["gripper"]["pos"] == 1.0


def test_the_frame_is_still_json_serialisable():
    """It goes straight to `ws.send_json`, and a sample's maps are read-only
    views that json.dumps refuses. This is the assertion that stops the first
    bus-backed frame from 500ing the telemetry socket."""
    arms, _, ros = _arms_and_ros()
    bcast = TelemetryBroadcaster(arms, ros, hz=200.0,
                                 tick_bus=_bus_with({"right": _snapshot()}))
    encoded = json.dumps(bcast._build_frame())
    assert json.loads(encoded)["arms"]["right"]["joints"]["gripper"]["pos"] == 1.0


def test_a_per_joint_key_still_reaches_a_subscriber_untouched():
    """`test_telemetry.py::test_effort_passes_through_the_frame_verbatim` says
    the broadcaster owns no part of the per-joint dict. The producer is now
    upstream of it, so the guarantee has to survive the whole path."""
    arms, _, ros = _arms_and_ros()
    bus = _bus_with({"right": _snapshot(a_channel_added_later=7)})
    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, tick_bus=bus)

    gripper = bcast._build_frame()["arms"]["right"]["joints"]["gripper"]
    assert gripper["a_channel_added_later"] == 7
    assert gripper["effort"] == -0.42


def test_the_frames_timestamp_is_when_the_arms_were_read():
    arms, _, ros = _arms_and_ros()
    bcast = TelemetryBroadcaster(arms, ros, hz=200.0,
                                 tick_bus=_bus_with({"right": _snapshot()}))
    assert bcast._build_frame()["t"] == 1234.5


def test_an_arm_that_failed_upstream_is_reported_not_invented():
    arms, _, ros = _arms_and_ros()
    bus = _bus_with({"left": _snapshot()}, arm_errors={"right": "bus went away"})
    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, tick_bus=bus)

    frame = bcast._build_frame()
    assert "right" not in frame["arms"]
    codes = [a["code"] for a in frame["alerts"]]
    assert "arm_telemetry_failed" in codes


def test_a_stale_tick_falls_back_to_a_direct_read_and_says_so():
    """A fallback nobody can see is how a rig runs for a week on the exception."""
    arms, arm, ros = _arms_and_ros()
    bus = TickBus()
    bus.publish_once("test", t_mono=time.perf_counter() - 10.0, t_unix=1.0,
                     arms={"right": _snapshot(pos=1.0)})
    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, tick_bus=bus,
                                 tick_stale_after_s=0.25)

    frame = bcast._build_frame()
    arm.state_snapshot.assert_called()
    assert frame["arms"]["right"]["joints"]["gripper"]["pos"] == 9.0
    assert "tick_bus_idle" in [a["code"] for a in frame["alerts"]]


def test_a_bus_that_has_never_published_falls_back_the_same_way():
    arms, arm, ros = _arms_and_ros()
    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, tick_bus=TickBus())
    frame = bcast._build_frame()
    arm.state_snapshot.assert_called()
    assert "tick_bus_idle" in [a["code"] for a in frame["alerts"]]


def test_no_bus_at_all_is_the_pre_phase_2_path_and_says_nothing():
    """An unwired broadcaster is not a degraded one — it is the old one."""
    arms, arm, ros = _arms_and_ros()
    frame = TelemetryBroadcaster(arms, ros, hz=200.0)._build_frame()
    arm.state_snapshot.assert_called()
    assert frame["alerts"] == []


def test_the_calibration_block_does_not_mutate_the_shared_sample():
    """A sample is one moment shared by every consumer. Telemetry's own
    calibration block belongs to this frame alone, and a frozen sample makes
    the alternative a TypeError rather than a corruption — but only if nothing
    tries to write through it."""
    arms, _, ros = _arms_and_ros()
    bus = _bus_with({"right": _snapshot()})
    cal = MagicMock()
    cal.current = MagicMock(arm_id="right")
    cal.current.state.value = "idle"
    bcast = TelemetryBroadcaster(arms, ros, hz=200.0, tick_bus=bus,
                                 calibration=cal)

    frame = bcast._build_frame()
    assert "calibration" in frame["arms"]["right"]
    assert "calibration" not in bus.latest().arms["right"]


@pytest.mark.asyncio
async def test_a_bus_backed_broadcaster_still_emits_to_subscribers():
    arms, _, ros = _arms_and_ros()
    bcast = TelemetryBroadcaster(arms, ros, hz=200.0,
                                 tick_bus=_bus_with({"right": _snapshot()}))
    bcast.start()
    try:
        sub = bcast.subscribe()
        frame = await asyncio.wait_for(sub.__anext__(), timeout=2.0)
    finally:
        await bcast.stop()
    assert frame["arms"]["right"]["joints"]["gripper"]["pos"] == 1.0
