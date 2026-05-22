import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from haller_hmi.calibration import (
    CalibrationManager,
    CalibrationSession,
    CalibrationState,
    ConflictError,
    UnmovedJointsError,
    WrongStateError,
    _calibration_paths,
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


# ---------------------------------------------------------------------------
# Task 2: Save mechanics
# ---------------------------------------------------------------------------


def _populate_session_through_review(handle):
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, handle.config.id)
    session.capture_neutral(handle)
    handle.robot.bus.sync_read.return_value = {j: 500 for j in JOINTS}
    session.tick_sweep(handle)
    handle.robot.bus.sync_read.return_value = {j: 3600 for j in JOINTS}
    session.tick_sweep(handle)
    session.finish_sweep(handle)
    return mgr, arms


def test_save_writes_follower_file_and_creates_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    follower_dir = tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower"
    follower_dir.mkdir(parents=True)
    existing = follower_dir / "haller_right.json"
    existing.write_text(json.dumps({"shoulder_pan": {"id": 1, "drive_mode": 0,
        "homing_offset": 0, "range_min": 100, "range_max": 200}}))

    handle = _make_handle("right")
    handle.robot.bus.motors = {
        j: MagicMock(model="sts3215", id=i + 1) for i, j in enumerate(JOINTS)
    }
    mgr, arms = _populate_session_through_review(handle)

    target, backup = mgr.save(arms)
    assert target == existing
    assert backup is not None and backup.exists()
    assert backup.name.startswith("haller_right.json.bak-")
    written = json.loads(target.read_text())
    assert set(written.keys()) == set(JOINTS)
    assert written["shoulder_pan"]["range_min"] == 500
    assert written["shoulder_pan"]["range_max"] == 3600


def test_save_also_updates_teleop_sibling(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cal_root = tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration"
    follower_dir = cal_root / "robots" / "so_follower"
    teleop_dir = cal_root / "teleoperators" / "so_leader"
    follower_dir.mkdir(parents=True)
    teleop_dir.mkdir(parents=True)
    teleop_path = teleop_dir / "haller_right.json"
    teleop_path.write_text(json.dumps({"shoulder_pan": {"id": 1, "drive_mode": 0,
        "homing_offset": 0, "range_min": 100, "range_max": 200}}))

    handle = _make_handle("right")
    handle.robot.bus.motors = {
        j: MagicMock(model="sts3215", id=i + 1) for i, j in enumerate(JOINTS)
    }
    mgr, arms = _populate_session_through_review(handle)
    mgr.save(arms)

    written = json.loads(teleop_path.read_text())
    assert set(written.keys()) == set(JOINTS)
    bak = list(teleop_dir.glob("haller_right.json.bak-*"))
    assert len(bak) == 1


def test_save_reconnects_arm(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower").mkdir(parents=True)
    handle = _make_handle("right")
    handle.robot.bus.motors = {
        j: MagicMock(model="sts3215", id=i + 1) for i, j in enumerate(JOINTS)
    }
    mgr, arms = _populate_session_through_review(handle)
    mgr.save(arms)
    handle.disconnect.assert_called_once()
    handle.connect.assert_called_once()
    assert mgr.current is None


def test_calibration_paths_returns_follower_and_existing_teleop(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cal_root = tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration"
    (cal_root / "robots" / "so_follower").mkdir(parents=True)
    (cal_root / "teleoperators" / "so_leader").mkdir(parents=True)
    teleop_path = cal_root / "teleoperators" / "so_leader" / "haller_right.json"
    teleop_path.write_text("{}")
    paths = _calibration_paths("haller_right")
    assert paths[0].name == "haller_right.json"
    assert "so_follower" in str(paths[0])
    assert teleop_path in paths


def test_save_keeps_original_when_write_fails(tmp_path, monkeypatch):
    """If _save_calibration_to raises, the canonical file is preserved (not lost to backup)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    follower_dir = tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower"
    follower_dir.mkdir(parents=True)
    existing = follower_dir / "haller_right.json"
    existing.write_text('{"original": true}')

    handle = _make_handle("right")
    handle.robot.bus.motors = {
        j: MagicMock(model="sts3215", id=i + 1) for i, j in enumerate(JOINTS)
    }
    mgr, arms = _populate_session_through_review(handle)

    # Force _save_calibration_to to fail
    import haller_hmi.calibration as cal_mod
    def boom(*_args, **_kwargs):
        raise OSError("simulated disk full")
    monkeypatch.setattr(cal_mod, "_save_calibration_to", boom)

    with pytest.raises(OSError, match="disk full"):
        mgr.save(arms)

    # Canonical file must still be readable with the original content
    assert existing.exists()
    assert existing.read_text() == '{"original": true}'


# ---------------------------------------------------------------------------
# FU1: Safety hardening — try/finally cleanup
# ---------------------------------------------------------------------------


def test_abort_clears_session_even_if_enable_torque_raises():
    handle = _make_handle()
    handle.enable_torque.side_effect = RuntimeError("bus down")
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    mgr.start(arms, "right")
    mgr.abort()  # must NOT raise
    assert mgr.current is None
    assert mgr._handle is None


def test_save_clears_session_even_if_reload_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower").mkdir(parents=True)
    handle = _make_handle("right")
    handle.robot.bus.motors = {
        j: MagicMock(model="sts3215", id=i + 1) for i, j in enumerate(JOINTS)
    }
    handle.connect.side_effect = RuntimeError("reload failed")
    mgr, arms = _populate_session_through_review(handle)
    with pytest.raises(RuntimeError, match="reload failed"):
        mgr.save(arms)
    # Files are written, session is cleared, even though connect raised
    assert mgr.current is None
    assert mgr._handle is None
