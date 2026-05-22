from unittest.mock import MagicMock

import pytest

from haller_hmi.calibration import (
    CalibrationManager,
    CalibrationSession,
    CalibrationState,
    ConflictError,
    UnmovedJointsError,
    WrongStateError,
)
from haller_hmi.config import ArmConfig
from haller_hmi.safety import Mode


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]


def _make_handle(arm_id: str = "right", mode: Mode = Mode.MANUAL):
    """Build a MagicMock that quacks like ArmHandle for the calibration session."""
    handle = MagicMock()
    handle.config = ArmConfig(id=arm_id, model="so101_follower",
                              port="/dev/null", calibration_id=f"haller_{arm_id}")
    handle.guard = MagicMock()
    handle.guard.mode = mode
    handle.torque_enabled = True
    bus = MagicMock()
    bus.sync_read.return_value = {j: 2048 for j in JOINTS}
    handle.robot = MagicMock()
    handle.robot.bus = bus
    handle.robot.bus.motors = {j: MagicMock(model="sts3215") for j in JOINTS}
    handle.robot.bus.model_resolution_table = {"sts3215": 4096}
    return handle


def _make_arms(*handles):
    arms = MagicMock()
    arms.keys.return_value = [h.config.id for h in handles]
    arms.values.return_value = list(handles)
    by_id = {h.config.id: h for h in handles}
    arms.__getitem__.side_effect = lambda k: by_id[k]
    return arms


def test_start_from_idle_transitions_to_homing_and_disables_torque():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    assert session.state is CalibrationState.HOMING
    handle.disable_torque.assert_called_once()


def test_start_rejected_when_another_arm_not_manual():
    left = _make_handle("left", mode=Mode.AUTO)
    right = _make_handle("right")
    arms = _make_arms(left, right)
    mgr = CalibrationManager()
    with pytest.raises(ConflictError, match="left"):
        mgr.start(arms, "right")


def test_start_rejected_when_session_already_active():
    h1 = _make_handle("right")
    arms = _make_arms(h1)
    mgr = CalibrationManager()
    mgr.start(arms, "right")
    with pytest.raises(ConflictError, match="session active"):
        mgr.start(arms, "right")


def test_capture_neutral_writes_homing_offsets_and_transitions():
    handle = _make_handle()
    handle.robot.bus.sync_read.return_value = {j: 2200 for j in JOINTS}
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    expected = 2200 - (4096 - 1) // 2
    for j in JOINTS:
        handle.robot.bus.write.assert_any_call("Homing_Offset", j, expected)
    assert session.state is CalibrationState.SWEEPING
    assert session.homing_offsets == {j: expected for j in JOINTS}


def test_capture_neutral_wrong_state_raises():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    with pytest.raises(WrongStateError):
        session.capture_neutral(handle)


def test_tick_sweep_accumulates_min_max():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    handle.robot.bus.sync_read.return_value = {j: 1000 for j in JOINTS}
    session.tick_sweep(handle)
    handle.robot.bus.sync_read.return_value = {j: 3500 for j in JOINTS}
    session.tick_sweep(handle)
    handle.robot.bus.sync_read.return_value = {j: 2500 for j in JOINTS}
    ticks = session.tick_sweep(handle)
    assert session.mins == {j: 1000 for j in JOINTS}
    assert session.maxes == {j: 3500 for j in JOINTS}
    assert ticks == {j: 2500 for j in JOINTS}


def test_finish_sweep_unmoved_joints_raises():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    handle.robot.bus.sync_read.return_value = {j: 2048 for j in JOINTS}
    session.tick_sweep(handle)
    with pytest.raises(UnmovedJointsError) as ei:
        session.finish_sweep(handle)
    for j in JOINTS:
        assert j in str(ei.value)
    assert session.state is CalibrationState.SWEEPING


def test_finish_sweep_builds_lerobot_shaped_proposed():
    handle = _make_handle()
    handle.robot.bus.motors = {
        "shoulder_pan":  MagicMock(model="sts3215", id=1),
        "shoulder_lift": MagicMock(model="sts3215", id=2),
        "elbow_flex":    MagicMock(model="sts3215", id=3),
        "wrist_flex":    MagicMock(model="sts3215", id=4),
        "wrist_roll":    MagicMock(model="sts3215", id=5),
        "gripper":       MagicMock(model="sts3215", id=6),
    }
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    handle.robot.bus.sync_read.return_value = {j: 500 for j in JOINTS}
    session.tick_sweep(handle)
    handle.robot.bus.sync_read.return_value = {j: 3600 for j in JOINTS}
    session.tick_sweep(handle)
    proposed = session.finish_sweep(handle)
    assert session.state is CalibrationState.REVIEW
    for joint in JOINTS:
        entry = proposed[joint]
        assert set(entry.keys()) == {"id", "drive_mode", "homing_offset",
                                     "range_min", "range_max"}
        assert entry["range_min"] == 500
        assert entry["range_max"] == 3600
        assert entry["drive_mode"] == 0


def test_abort_from_homing_re_enables_torque_and_clears():
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    mgr.start(arms, "right")
    mgr.abort()
    handle.enable_torque.assert_called_once()
    assert mgr.current is None


def test_abort_is_idempotent():
    mgr = CalibrationManager()
    mgr.abort()
    assert mgr.current is None
