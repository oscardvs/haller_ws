"""MuJoCoWorld owns model+data, runs a stepper, and exposes per-arm ctrl/qpos."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.sim.world import MuJoCoWorld

# Minimal 2-DOF MJCF with named joints + actuators we can address by "arm".
TINY_XML = """
<mujoco>
  <compiler angle="degree"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="right_base" pos="0 0 0">
      <joint name="right_shoulder_pan" type="hinge" axis="0 0 1" range="-180 180" limited="true" damping="2"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0  0.2 0 0"/>
      <body name="right_link2" pos="0.2 0 0">
        <joint name="right_gripper" type="hinge" axis="0 1 0" range="-90 90" limited="true" damping="2"/>
        <geom type="capsule" size="0.02 0.1" fromto="0 0 0  0.2 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="right_shoulder_pan_act" joint="right_shoulder_pan" kp="50" ctrlrange="-180 180"/>
    <position name="right_gripper_act"      joint="right_gripper"      kp="50" ctrlrange="-90 90"/>
  </actuator>
</mujoco>
"""

ARM_JOINT_MAP = {"right": ["right_shoulder_pan", "right_gripper"]}


def test_world_loads_and_starts_stepper():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    world.start()
    try:
        time.sleep(0.1)
        assert world.is_running()
        assert world.tick_count() > 0
    finally:
        world.stop()


def test_write_ctrl_moves_qpos_toward_goal():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    world.start()
    try:
        # 30 deg goal on shoulder_pan
        world.write_ctrl_deg("right", {"right_shoulder_pan": 30.0})
        time.sleep(0.5)  # let the actuator drive it
        q = world.read_qpos_deg("right")
        assert abs(q["right_shoulder_pan"] - 30.0) < 5.0, f"got {q!r}"
    finally:
        world.stop()


def test_unknown_arm_raises():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    with pytest.raises(KeyError):
        world.write_ctrl_deg("left", {"shoulder_pan": 0.0})


def test_disable_actuator_kp_zeros_gain():
    world = MuJoCoWorld(TINY_XML, arm_joint_map=ARM_JOINT_MAP)
    world.start()
    try:
        world.set_arm_torque("right", enabled=False)
        # All actuators for this arm should have kp = 0
        for joint in ARM_JOINT_MAP["right"]:
            assert world.actuator_kp_for_joint(joint) == 0.0
        world.set_arm_torque("right", enabled=True)
        for joint in ARM_JOINT_MAP["right"]:
            assert world.actuator_kp_for_joint(joint) > 0.0
    finally:
        world.stop()
