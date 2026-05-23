# hmi/backend/tests/test_arm.py
from unittest.mock import MagicMock

import pytest

from haller_hmi.arm import ArmManager, ArmHandle
from haller_hmi.config import ArmConfig
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
