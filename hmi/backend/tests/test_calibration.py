import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from haller_hmi.calibration import (
    CalibrationManager,
    CalibrationSession,
    CalibrationState,
    ConflictError,
    SweepWrapError,
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
    bus.set_half_turn_homings.return_value = {j: 0 for j in JOINTS}
    bus.read.return_value = 0    # Homing_Offset after the stock reset+write
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


def _sweeping_session(handle):
    """A session through capture, reading 2048 everywhere at the seed."""
    mgr = CalibrationManager()
    session = mgr.start(_make_arms(handle), handle.config.id)
    session.capture_neutral(handle)
    return session


def _sweep_with(handle, session, values_by_tick):
    """Feed tick_sweep one read per entry; each entry maps joint→raw ticks
    (unnamed joints hold 2048)."""
    for overrides in values_by_tick:
        reading = {j: 2048 for j in JOINTS}
        reading.update(overrides)
        handle.robot.bus.sync_read.return_value = reading
        session.tick_sweep(handle)


def _walk_all(handle, session, lo, hi, start=2048):
    """Sweep every joint start→lo→hi in ≤350-tick steps. The glitch filter
    (SWEEP_MAX_JUMP_TICKS) holds implausible teleports back for confirmation,
    so a synthetic sweep has to move the way a hand does."""
    def _steps(frm, to):
        step = 350 if to > frm else -350
        return list(range(frm + step, to, step)) + [to]
    for v in _steps(start, lo) + _steps(lo, hi):
        handle.robot.bus.sync_read.return_value = {j: v for j in JOINTS}
        session.tick_sweep(handle)


def test_one_glitched_read_does_not_poison_the_range():
    """Min/max accumulation amplifies a single outlier into a ruined range —
    and a ruined range is a ruined ZERO (zero = range centre). A sample
    implausibly far from the last accepted one must be confirmed by its
    successor before it counts. Measured 2026-08-29: one poisoned sweep
    recorded ~356° on every joint and the save went through silently."""
    handle = _make_handle()
    session = _sweeping_session(handle)
    _sweep_with(handle, session, [
        {"shoulder_pan": 2200},
        {"shoulder_pan": 4090},   # wrap/corrupt spike, one tick only
        {"shoulder_pan": 2250},
        {"shoulder_pan": 2300},
    ])
    assert session.maxes["shoulder_pan"] == 2300, "the spike got in"


def test_sustained_fast_motion_is_accepted_one_tick_late():
    handle = _make_handle()
    session = _sweeping_session(handle)
    _sweep_with(handle, session, [
        {"shoulder_pan": 2600},   # > jump window from 2048: held pending
        {"shoulder_pan": 2650},   # confirms — real motion, folded in
        {"shoulder_pan": 3000},
    ])
    assert session.maxes["shoulder_pan"] == 3000
    # The capture seed (2048) was evicted by the confirmed jump — deliberate:
    # a seed cannot earn confirmation, and the capture pose is mid-travel, so
    # it is never a real endpoint; the sweep re-finds the true minimum.
    assert session.mins["shoulder_pan"] == 2650


def test_a_glitched_capture_seed_is_evicted_by_the_confirmed_sweep():
    """The capture read is one sample and gets no confirmation vote; if the
    first CONFIRMED sweep samples disagree with it wholesale, the seed was
    the glitch and must not survive as a range endpoint."""
    handle = _make_handle()
    handle.robot.bus.sync_read.return_value = {j: 2048 for j in JOINTS} | {
        "shoulder_pan": 2150}     # glitched seed, inside the re-centre tol
    mgr = CalibrationManager()
    session = mgr.start(_make_arms(handle), handle.config.id)
    session.capture_neutral(handle)
    _sweep_with(handle, session, [
        {"shoulder_pan": 2600},   # held pending against the 2150 seed
        {"shoulder_pan": 2610},   # confirms; the held sample itself is not
        {"shoulder_pan": 2900},   # folded — its confirmer is
    ])
    assert session.maxes["shoulder_pan"] == 2900
    assert session.mins["shoulder_pan"] == 2610, "the glitched seed survived"


def test_finish_sweep_refuses_a_physically_impossible_range():
    """A non-full-turn joint reading ~356° of travel is wrap or garbage —
    the widest real SO-101 travel is shoulder_pan at ~238°. Refuse at the
    wizard, not at the bench. wrist_roll IS full-turn and its ~353° must
    pass."""
    handle = _make_handle()
    session = _sweeping_session(handle)
    # Every joint moves a little (so `unmoved` cannot mask the wrap check);
    # pan and roll walk to near-full range in plausible steps.
    steps = [{j: 2200 for j in JOINTS}]
    steps += [{"shoulder_pan": t, "wrist_roll": t} for t in
              range(2200, 4060, 300)]
    steps += [{"shoulder_pan": t, "wrist_roll": t} for t in
              range(4050, 40, -300)]
    _sweep_with(handle, session, steps)
    with pytest.raises(SweepWrapError) as e:
        session.finish_sweep(handle)
    assert "shoulder_pan" in str(e.value)
    assert "wrist_roll" not in str(e.value)


def test_start_claims_the_singleton_before_the_first_bus_op_and_rolls_back():
    """`current` must be set before any bus write, and unset if start fails.

    The idle sampler stands down while `current` is set (server.py wires its
    sample source through it); a `disable_torque` issued before the claim
    races the sampler's 20 Hz reads on a lock-free half-duplex bus, and the
    wizard's single-error abort dies on the first "Port is in use!". A failed
    start must release the claim, or the sampler stays down for a session
    that does not exist.
    """
    handle = _make_handle()
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    seen: list[bool] = []
    handle.disable_torque.side_effect = (
        lambda: seen.append(mgr.current is not None))
    mgr.start(arms, "right")
    assert seen == [True], "disable_torque ran before the singleton was claimed"
    mgr.abort()

    handle2 = _make_handle("left")
    arms2 = _make_arms(handle2)
    mgr2 = CalibrationManager()
    handle2.disable_torque.side_effect = RuntimeError("port is in use")
    with pytest.raises(RuntimeError):
        mgr2.start(arms2, "left")
    assert mgr2.current is None, "a failed start must release the claim"


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


def test_capture_neutral_delegates_to_stock_lerobot_homing():
    """Capture must run `set_half_turn_homings` — the same call the kit's
    one-command calibrate runs, which resets the offset to ZERO before
    measuring — never a reimplementation. The hand-rolled absolute-offset
    math it replaces only centred when the prior offset was zero, and
    lerobot loads the calibration file's offsets into the servos at connect,
    so with a poisoned file the wizard could never produce a good one
    (2026-08-29)."""
    handle = _make_handle()
    handle.robot.bus.set_half_turn_homings.return_value = {
        j: 153 for j in JOINTS}
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    session.capture_neutral(handle)
    handle.robot.bus.set_half_turn_homings.assert_called_once()
    assert session.state is CalibrationState.SWEEPING
    assert session.homing_offsets == {j: 153 for j in JOINTS}


def test_failed_capture_restores_the_calibration_it_reset():
    """`set_half_turn_homings` clears the bus's in-memory calibration; a
    capture that then fails must hand it back, or every normalized read in
    the backend stays broken until a restart — the cockpit sat on 'awaiting
    telemetry' with healthy hardware (2026-08-29)."""
    from haller_hmi.calibration import RecenterError
    handle = _make_handle()
    prior = {"shoulder_pan": "CAL"}
    handle.robot.bus.calibration = dict(prior)
    handle.robot.bus.sync_read.return_value = {j: 2048 for j in JOINTS} | {
        "shoulder_pan": 2400}
    mgr = CalibrationManager()
    session = mgr.start(_make_arms(handle), "right")
    with pytest.raises(RecenterError):
        session.capture_neutral(handle)
    handle.robot.bus.write_calibration.assert_called_once_with(prior)


def test_abort_after_capture_restores_the_calibration_it_reset():
    handle = _make_handle()
    prior = {"shoulder_pan": "CAL"}
    handle.robot.bus.calibration = dict(prior)
    mgr = CalibrationManager()
    session = mgr.start(_make_arms(handle), "right")
    session.capture_neutral(handle)
    mgr.abort()
    handle.robot.bus.write_calibration.assert_called_once_with(prior)
    assert mgr.current is None


def test_capture_corrects_an_offset_the_register_cannot_hold():
    """The Homing_Offset register is sign-magnitude with an 11-bit magnitude
    (±2047); a joint whose raw encoder sits past the encoder zero needs
    more, the stock write silently truncates (2315 stored as 267, gripper
    parked at 268 — both wizard runs, 2026-08-29), and the SAME physical
    shift expressed mod 4096 fits. Capture must finish the job itself:
    measure the reported error, correct the short way round, and only ever
    write values the register can hold."""
    handle = _make_handle()
    bus = handle.robot.bus
    state = {"centred": False}

    def _read_pos(_reg, _motors, **_kw):
        base = {j: 2048 for j in JOINTS}
        if not state["centred"]:
            base["gripper"] = 268
        return base

    def _write(reg, motor, value):
        if reg == "Homing_Offset" and motor == "gripper":
            assert -2047 <= value <= 2047, \
                "wrote a value the register cannot hold"
            state["centred"] = True

    bus.sync_read.side_effect = _read_pos
    bus.write.side_effect = _write
    mgr = CalibrationManager()
    session = mgr.start(_make_arms(handle), "right")
    session.capture_neutral(handle)
    assert session.state is CalibrationState.SWEEPING
    assert session.homing_offsets["gripper"] == -1779


def test_capture_neutral_refuses_when_the_recenter_did_not_land():
    """The homing write has failed silently before (alarm states, locked
    EEPROM); a sweep started off-centre wraps and dies two steps later at
    finish_sweep, reading as operator error. Refuse at the step that
    failed."""
    from haller_hmi.calibration import RecenterError
    handle = _make_handle()
    handle.robot.bus.sync_read.return_value = {j: 2048 for j in JOINTS} | {
        "shoulder_pan": 2400}   # 353 ticks off-centre after the write
    arms = _make_arms(handle)
    mgr = CalibrationManager()
    session = mgr.start(arms, "right")
    with pytest.raises(RecenterError) as e:
        session.capture_neutral(handle)
    assert "shoulder_pan" in str(e.value)


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
    _walk_all(handle, session, 1000, 3500)
    # A 3500→2500 teleport: the raw value still comes back in `ticks` (the
    # live readout must never lie about the bus) while the range holds the
    # unconfirmed sample back.
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
    _walk_all(handle, session, 500, 3600)
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
    _walk_all(handle, session, 500, 3600)
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
