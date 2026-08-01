"""SimArmHandle implements the same public surface as ArmHandle."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.config import ArmConfig
from haller_hmi.safety import Mode, ModeError
from haller_hmi.sim.arm import SimArmHandle
from haller_hmi.sim.world import MuJoCoWorld


# Tiny single-arm MJCF for unit tests. Uses CamelCase joint names (matching the
# real SO-101 MJCF convention) namespaced with a "right_" prefix.
TINY_XML = """
<mujoco>
  <compiler angle="degree"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="right_base">
      <joint name="right_Rotation" type="hinge" axis="0 0 1" range="-180 180"
             limited="true" damping="2"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0  0.2 0 0"/>
      <body name="right_link2" pos="0.2 0 0">
        <joint name="right_Jaw" type="hinge" axis="0 1 0" range="-90 90"
               limited="true" damping="2"/>
        <geom type="capsule" size="0.02 0.1" fromto="0 0 0  0.2 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="right_Rotation" kp="50" ctrlrange="-180 180"/>
    <position name="a2" joint="right_Jaw"      kp="50" ctrlrange="-90 90"/>
  </actuator>
</mujoco>
"""

# Map of HMI/LeRobot joint names to the MJCF names in TINY_XML.
ARM_JOINT_MAP = {"right": ["right_Rotation", "right_Jaw"]}


def _make_world_and_handle():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    cfg = ArmConfig(
        id="right", model="so101_follower", port="(sim)", calibration_id="(sim)",
        source="sim", sim_arm_name="right",
    )
    handle = SimArmHandle(cfg, world=world)
    return world, handle


def test_connect_populates_joint_limits_deg_in_lerobot_naming():
    """SimArmHandle must expose joint_limits_deg keyed by LeRobot snake_case names
    (e.g. 'shoulder_pan', 'gripper') so the HMI's per-arm sliders and the existing
    teleop session work without modification."""
    world, handle = _make_world_and_handle()
    handle.connect()
    # TINY_XML only includes Rotation and Jaw → snake_case shoulder_pan + gripper.
    assert "shoulder_pan" in handle.joint_limits_deg
    assert "gripper" in handle.joint_limits_deg
    lo, hi = handle.joint_limits_deg["shoulder_pan"]
    assert lo == pytest.approx(-180.0, abs=0.5)
    assert hi == pytest.approx(180.0, abs=0.5)


def test_send_goal_takes_lerobot_keys_and_writes_to_mjcf_actuators():
    world, handle = _make_world_and_handle()
    handle.connect()
    world.start()
    try:
        sent = handle.send_goal({"shoulder_pan": 30.0, "gripper": 999.0})
        assert sent["shoulder_pan"] == pytest.approx(30.0, abs=1e-3)
        assert sent["gripper"] == pytest.approx(90.0, abs=1e-3)  # clamped
        time.sleep(0.5)
        q = handle.read_joints_deg()
        assert abs(q["shoulder_pan"] - 30.0) < 5.0, f"got {q!r}"
    finally:
        world.stop()


def test_send_goal_in_auto_mode_raises():
    world, handle = _make_world_and_handle()
    handle.connect()
    handle.guard.set(Mode.AUTO)
    with pytest.raises(ModeError):
        handle.send_goal({"shoulder_pan": 0.0})


def test_state_snapshot_shape_matches_real_arm():
    world, handle = _make_world_and_handle()
    handle.connect()
    snap = handle.state_snapshot()
    assert snap["mode"] in {"auto", "manual", "stop"}
    assert "torque" in snap and "joints" in snap
    assert set(snap["joints"]) == {"shoulder_pan", "gripper"}
    for j, info in snap["joints"].items():
        assert set(info) == {"pos", "min", "max", "torque"}


def test_disable_torque_zeros_kp():
    world, handle = _make_world_and_handle()
    handle.connect()
    handle.disable_torque()
    assert not handle.torque_enabled
    assert world.actuator_kp_for_joint("right_Rotation") == 0.0
    handle.enable_torque()
    assert world.actuator_kp_for_joint("right_Rotation") > 0.0


def test_read_joints_deg_returns_lerobot_keys():
    """read_joints_deg() must return snake_case keys (no prefix, no CamelCase),
    matching ArmHandle.read_joints_deg() from the real side."""
    world, handle = _make_world_and_handle()
    handle.connect()
    out = handle.read_joints_deg()
    assert set(out) == {"shoulder_pan", "gripper"}


def test_sim_send_goal_does_not_silently_enable_torque():
    _world, handle = _make_world_and_handle()
    handle.connect()
    handle.guard.set(Mode.MANUAL)
    handle.torque_enabled = False

    handle.send_goal({"shoulder_pan": 10.0})

    assert handle.torque_enabled is False
