# hmi/backend/tests/test_arm.py
from unittest.mock import MagicMock

import pytest

from haller_hmi.arm import ArmManager, ArmHandle
from haller_hmi.config import ArmConfig, MotionConfig
from haller_hmi.safety import Mode, ModeError


def _make_handle(monkeypatch) -> ArmHandle:
    # Patch SO101Follower so we never touch real hardware.
    fake_robot = MagicMock()
    fake_robot.bus.motors = {
        "shoulder_pan": MagicMock(id=1),
        "shoulder_lift": MagicMock(id=2),
        "elbow_flex": MagicMock(id=3),
        "wrist_flex": MagicMock(id=4),
        "wrist_roll": MagicMock(id=5),
        "gripper": MagicMock(id=6),
    }
    fake_robot.calibration = {
        "shoulder_pan":  MagicMock(range_min=0,    range_max=4095),
        "shoulder_lift": MagicMock(range_min=0,    range_max=4095),
        "elbow_flex":    MagicMock(range_min=0,    range_max=4095),
        "wrist_flex":    MagicMock(range_min=0,    range_max=4095),
        "wrist_roll":    MagicMock(range_min=0,    range_max=4095),
        "gripper":       MagicMock(range_min=0,    range_max=4095),
    }
    fake_robot.get_observation.return_value = {
        f"{j}.pos": 0.0 for j in fake_robot.bus.motors
    }
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: fake_robot)
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    handle = ArmHandle(cfg, joint_limits_deg={
        "shoulder_pan": (-120, 120), "shoulder_lift": (-100, 100),
        "elbow_flex": (-110, 110), "wrist_flex": (-90, 90),
        "wrist_roll": (-180, 180), "gripper": (0, 100),
    })
    handle.robot = fake_robot
    return handle


def test_send_goal_in_auto_mode_raises(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.AUTO)
    with pytest.raises(ModeError):
        handle.send_goal({"shoulder_pan": 30.0})


def test_send_goal_clamps_and_calls_lerobot(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    # A large max_speed_deg_s keeps this test about joint-limit clamping and
    # forwarding to lerobot, not about the per-step cap (covered separately by
    # test_send_goal_caps_a_garbage_jump_to_one_step and friends below).
    handle.motion = MotionConfig(max_speed_deg_s=100000.0, ramp_hz=50.0)
    sent = handle.send_goal({"shoulder_pan": 999.0, "gripper": -50.0, "unknown": 10.0})
    assert sent == {"shoulder_pan": 120.0, "gripper": 0.0}
    handle.robot.send_action.assert_called_once_with({"shoulder_pan.pos": 120.0,
                                                      "gripper.pos": 0.0})


def test_state_snapshot_returns_joints_with_limits(monkeypatch):
    handle = _make_handle(monkeypatch)
    snap = handle.state_snapshot()
    assert set(snap["joints"].keys()) == {
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    }
    assert snap["joints"]["shoulder_pan"]["min"] == -120
    assert snap["joints"]["shoulder_pan"]["max"] == 120
    assert snap["mode"] in {"auto", "manual", "stop"}


def test_arm_manager_lookup_by_id(monkeypatch):
    cfg_right = ArmConfig(id="right", model="so101_follower",
                          port="/dev/null", calibration_id="haller_follower")
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr("haller_hmi.arm.ArmHandle._load_joint_limits",
                        lambda self: {"gripper": (0, 100)})
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    mgr = ArmManager([cfg_right])
    mgr.connect_all()
    assert mgr["right"].config.id == "right"


def test_read_joints_deg_strips_pos_suffix(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.robot.get_observation.return_value = {
        "shoulder_pan.pos": 12.5,
        "elbow_flex.pos": -30.0,
        "gripper.pos": 0.0,
        # lerobot also emits non-joint keys (e.g. ".vel"); read_joints_deg must ignore them
        "shoulder_pan.vel": 999.0,
    }
    joints = handle.read_joints_deg()
    assert joints == {"shoulder_pan": 12.5, "elbow_flex": -30.0, "gripper": 0.0}


def test_teleop_loop_uses_read_joints_deg_and_send_goal(monkeypatch):
    """The teleop tick must go through handle.read_joints_deg() + handle.send_goal(),
    not reach into handle.robot directly. This is what makes SimArmHandle a drop-in."""
    from unittest.mock import MagicMock
    from haller_hmi.teleop import TeleopSession
    from haller_hmi.config import ArmConfig
    from haller_hmi.arm import ArmManager
    import time

    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "haller_hmi.arm.ArmHandle._load_joint_limits",
        lambda self: {"shoulder_pan": (-120.0, 120.0), "gripper": (0.0, 100.0)},
    )
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )

    cfg_l = ArmConfig(id="left",  model="so101_follower", port="/dev/null", calibration_id="x")
    cfg_r = ArmConfig(id="right", model="so101_follower", port="/dev/null", calibration_id="y")
    mgr = ArmManager([cfg_l, cfg_r])
    mgr.connect_all()

    leader = mgr["left"]
    follower = mgr["right"]

    # Stub the two interface methods we expect the loop to call.
    leader.read_joints_deg = MagicMock(return_value={"shoulder_pan": 42.0, "gripper": 50.0})
    follower.send_goal = MagicMock(return_value={"shoulder_pan": 42.0, "gripper": 50.0})
    # Also disable any auto-enable side-effects.
    leader.disable_torque = MagicMock()
    follower.enable_torque = MagicMock()

    session = TeleopSession(mgr)
    session.start(leader_id="left", follower_id="right", hz=120.0)
    time.sleep(0.1)  # let a few ticks happen
    session.stop()

    assert leader.read_joints_deg.called, "loop must call leader.read_joints_deg()"
    assert follower.send_goal.called, "loop must call follower.send_goal()"
    last_call = follower.send_goal.call_args
    assert last_call.args[0] == {"shoulder_pan": 42.0, "gripper": 50.0}


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]


class _FakeSOFollower:
    """The slice of lerobot's SOFollower contract that connect() depends on.

    lerobot's connect() delegates to self.calibrate() whenever the motors
    disagree with the calibration file, and the stock calibrate() asks a
    question on stdin. Under uvicorn there is no stdin, so it raises EOFError
    on the spot. The MagicMock used elsewhere in this file cannot express that,
    which is why the reload path went untested.
    """

    def __init__(self, is_calibrated: bool) -> None:
        self.is_calibrated = is_calibrated
        self.id = "haller_follower"
        self.calibration_fpath = "/nonexistent/haller_follower.json"
        self.calibration = {j: MagicMock(range_min=0, range_max=4095) for j in JOINTS}
        self.bus = MagicMock()
        self.bus.motors = {j: MagicMock(id=i + 1) for i, j in enumerate(JOINTS)}

    def connect(self, calibrate: bool = True) -> None:
        if not self.is_calibrated and calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        raise EOFError("EOF when reading a line")


def _handle_with(monkeypatch, fake) -> ArmHandle:
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: fake)
    monkeypatch.setattr("haller_hmi.arm.SO101FollowerConfig",
                        lambda **kw: MagicMock(**kw))
    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="haller_follower")
    return ArmHandle(cfg)


def test_connect_writes_calibration_instead_of_prompting(monkeypatch):
    """Regression: saving a new calibration is exactly what makes the motors
    disagree with the file, so the wizard's post-save reload always landed in
    lerobot's interactive calibrate() and died with EOFError — turning an
    already-committed save into a 500, and the retry into a 409."""
    fake = _FakeSOFollower(is_calibrated=False)
    handle = _handle_with(monkeypatch, fake)

    handle.connect()  # must not raise

    fake.bus.write_calibration.assert_called_once_with(fake.calibration)
    assert set(handle.joint_limits_deg) == set(JOINTS)


def test_connect_leaves_motors_alone_when_calibration_already_matches(monkeypatch):
    """The happy path must stay untouched: no redundant EEPROM writes."""
    fake = _FakeSOFollower(is_calibrated=True)
    handle = _handle_with(monkeypatch, fake)

    handle.connect()

    fake.bus.write_calibration.assert_not_called()


def test_connect_without_a_calibration_file_raises_a_clear_error(monkeypatch):
    """With no file to push there is genuinely nothing a headless connect can
    do, so it must say so rather than surface lerobot's EOFError."""
    fake = _FakeSOFollower(is_calibrated=False)
    fake.calibration = {}
    handle = _handle_with(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="no calibration"):
        handle.connect()


def test_send_goal_does_not_silently_enable_torque(monkeypatch):
    """A limp arm must stay limp. Silently energizing it is half of what made
    the 2026-08-01 Home command dangerous."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.torque_enabled = False

    handle.send_goal({"shoulder_pan": 10.0})

    handle.robot.bus.enable_torque.assert_not_called()
    assert handle.torque_enabled is False


def test_send_goal_caps_a_garbage_jump_to_one_step(monkeypatch):
    """A corrupted frame commanding +100 deg must yield one bounded step. Also
    covers the suspected UART corruption that poisoned the right arm's sweep."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = {"shoulder_pan": 0.0}

    sent = handle.send_goal({"shoulder_pan": 100.0})

    assert sent["shoulder_pan"] == pytest.approx(1.2)


def test_send_goal_tracks_last_commanded_across_calls(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = {"shoulder_pan": 0.0}
    # The cap is governed by real elapsed time between calls, so pin the
    # clock rather than relying on however long two back-to-back Python
    # calls actually take — one ramp period must have "elapsed" for the
    # second call to earn another full step.
    clock = iter([100.0, 100.0 + 1.0 / handle.motion.ramp_hz])
    monkeypatch.setattr("haller_hmi.arm.time.monotonic", lambda: next(clock))

    handle.send_goal({"shoulder_pan": 100.0})
    second = handle.send_goal({"shoulder_pan": 100.0})

    assert second["shoulder_pan"] == pytest.approx(2.4)


def test_send_goal_seeds_last_commanded_from_a_real_read(monkeypatch):
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = None
    handle.robot.get_observation.return_value = {"shoulder_pan.pos": 50.0}

    sent = handle.send_goal({"shoulder_pan": 100.0})

    assert sent["shoulder_pan"] == pytest.approx(51.2)


def test_send_goal_drops_a_joint_the_seed_read_could_not_measure(monkeypatch):
    """A flaky seed read (first call, or right after a torque toggle) can come
    back missing a joint. limit_step's own contract passes a joint absent from
    `current` through UNCAPPED — the right behaviour when the caller already
    knows every joint is measured, but a fail-open if send_goal handed it an
    unmeasured one. send_goal must refuse to command that joint at all rather
    than inherit the pass-through."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = None
    # The seed read comes back missing shoulder_pan entirely.
    handle.robot.get_observation.return_value = {"gripper.pos": 0.0}

    sent = handle.send_goal({"shoulder_pan": 100.0, "gripper": 50.0})

    assert "shoulder_pan" not in sent
    assert sent["gripper"] == pytest.approx(1.2)
    handle.robot.send_action.assert_called_once_with({"gripper.pos": pytest.approx(1.2)})


def test_send_goal_recovers_a_dropped_joint_once_a_read_succeeds(monkeypatch):
    """The joint that a flaky seed read dropped must rejoin as soon as a later
    call's retry read actually measures it — not stay locked out forever."""
    handle = _make_handle(monkeypatch)
    handle.guard.set(Mode.MANUAL)
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    handle._last_commanded = None
    handle.robot.get_observation.return_value = {"gripper.pos": 0.0}
    # Pin the clock so the second call's budget is deterministic — see
    # test_send_goal_tracks_last_commanded_across_calls.
    clock = iter([100.0, 100.0 + 1.0 / handle.motion.ramp_hz])
    monkeypatch.setattr("haller_hmi.arm.time.monotonic", lambda: next(clock))

    first = handle.send_goal({"shoulder_pan": 100.0, "gripper": 50.0})
    assert "shoulder_pan" not in first

    # The read recovers: shoulder_pan is now measured too.
    handle.robot.get_observation.return_value = {
        "shoulder_pan.pos": 10.0, "gripper.pos": 1.2,
    }
    second = handle.send_goal({"shoulder_pan": 100.0, "gripper": 50.0})

    assert second["shoulder_pan"] == pytest.approx(11.2)
