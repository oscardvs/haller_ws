"""SimArmHandle implements the same public surface as ArmHandle."""
from __future__ import annotations

import os
import time
from unittest.mock import MagicMock

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import pytest

from haller_hmi.config import ArmConfig, MotionConfig
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
    <!-- forcerange mirrors the real SO-101 MJCF's <default> (-3.5 3.5 N·m).
         It is the divisor read_effort_norm normalises against, so without it
         the effort column here would exercise nothing. -->
    <position name="a1" joint="right_Rotation" kp="50" ctrlrange="-180 180"
              forcerange="-3.5 3.5"/>
    <position name="a2" joint="right_Jaw"      kp="50" ctrlrange="-90 90"
              forcerange="-3.5 3.5"/>
  </actuator>
</mujoco>
"""

#: The forcerange above, i.e. what a normalised effort of 1.0 means here.
FORCE_LIMIT_NM = 3.5

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


def test_two_freshly_constructed_handles_with_equal_fields_compare_equal():
    """Same regression as ArmHandle's (see test_arm.py): `executor` used to
    make two otherwise-identical handles compare unequal, purely because each
    builds its own MoveExecutor in __post_init__."""
    world, handle1 = _make_world_and_handle()
    handle2 = SimArmHandle(handle1.config, world=world)
    assert handle1 == handle2


def test_executor_constructor_argument_is_rejected():
    world, handle = _make_world_and_handle()
    with pytest.raises(TypeError):
        SimArmHandle(handle.config, world=world, executor=MagicMock())


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
    # A large max_speed_deg_s keeps this test about clamping-and-forwarding to
    # the MJCF actuators, not about the per-step motion-safety cap (covered
    # separately in test_arm.py).
    handle.motion = MotionConfig(max_speed_deg_s=100000.0, ramp_hz=50.0)
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
        # "effort" is a dimensionless signed fraction of the joint's torque
        # limit. The real ArmHandle publishes the same key with the same
        # meaning (normalised against Present_Load's ±1023 PWM duty instead of
        # against N·m) so the recorder gets ONE column whichever rig is
        # driving — the two quantities cannot be unit-matched, so neither side
        # reports raw units.
        assert set(info) == {"pos", "min", "max", "torque", "effort"}
        assert isinstance(info["effort"], float)


def test_effort_norm_saturates_at_the_actuator_force_limit():
    """A position actuator commanded far from where the joint is asks for far
    more than its forcerange allows, so the normalised reading pins at ±1."""
    world, handle = _make_world_and_handle()
    handle.connect()

    world.write_ctrl_deg("right", {"right_Jaw": 90.0})
    for _ in range(10):
        mujoco.mj_step(world.model, world.data)
    assert world.read_effort_norm("right")["right_Jaw"] == pytest.approx(1.0)
    assert handle.read_effort_norm()["gripper"] == pytest.approx(1.0)

    world.write_ctrl_deg("right", {"right_Jaw": -90.0})
    for _ in range(10):
        mujoco.mj_step(world.model, world.data)
    assert world.read_effort_norm("right")["right_Jaw"] == pytest.approx(-1.0)

    # Once it arrives, the actuator is no longer fighting: the reading comes off
    # the rail and stays inside the band.
    world.write_ctrl_deg("right", {"right_Jaw": 45.0})
    for _ in range(4000):
        mujoco.mj_step(world.model, world.data)
    settled = world.read_effort_norm("right")["right_Jaw"]
    assert -1.0 < settled < 1.0
    assert abs(settled) == pytest.approx(
        abs(float(world.data.actuator_force[1])) / FORCE_LIMIT_NM, abs=1e-9)


def test_effort_norm_uses_lerobot_keys_on_the_handle():
    world, handle = _make_world_and_handle()
    handle.connect()
    assert set(handle.read_effort_norm()) == {"shoulder_pan", "gripper"}


def test_effort_with_torque_off_saturates_negative_and_is_reported_as_zero():
    """KNOWN CAVEAT, pinned so it cannot change silently.

    `set_arm_torque(enabled=False)` zeroes gainprm[0] but leaves biasprm at
    [0, -kp, -kv], so a torque-off position actuator still applies
    -kp*qpos - kv*qvel and saturates at the NEGATIVE force limit the moment the
    joint is anywhere but zero. Raw, that reads as "straining hard" on an arm
    that is in fact limp — so state_snapshot reports 0.0 while torque is off,
    which is what a real arm with its torque disabled is doing anyway.
    """
    world, handle = _make_world_and_handle()
    handle.connect()
    world.write_ctrl_deg("right", {"right_Jaw": 45.0})
    for _ in range(4000):
        mujoco.mj_step(world.model, world.data)

    handle.disable_torque()
    for _ in range(20):
        mujoco.mj_step(world.model, world.data)

    assert world.actuator_kp_for_joint("right_Jaw") == 0.0
    assert world.read_effort_norm("right")["right_Jaw"] == pytest.approx(-1.0)
    assert handle.state_snapshot()["joints"]["gripper"]["effort"] == 0.0
    assert handle.state_snapshot()["joints"]["gripper"]["torque"] is False


def test_effort_norm_reports_zero_when_no_forcerange_is_declared(caplog):
    """An actuator with no forcerange has no saturation limit to be a fraction
    OF. Report 0.0 and warn at construction rather than inventing a divisor —
    and never divide by zero."""
    xml = TINY_XML.replace(' forcerange="-3.5 3.5"', "")
    with caplog.at_level("WARNING"):
        world = MuJoCoWorld(xml, arm_joint_map=ARM_JOINT_MAP)
    assert any("forcerange" in r.getMessage() for r in caplog.records)

    world.write_ctrl_deg("right", {"right_Jaw": 90.0})
    for _ in range(10):
        mujoco.mj_step(world.model, world.data)
    assert world.read_effort_norm("right") == {"right_Rotation": 0.0, "right_Jaw": 0.0}


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


def test_send_goal_drops_a_joint_the_seed_read_could_not_measure(monkeypatch):
    """Same fail-open guard as ArmHandle: a joint missing from the seed read
    must be dropped from the command, not passed through uncapped."""
    _world, handle = _make_world_and_handle()
    handle.connect()
    handle.motion = MotionConfig(max_speed_deg_s=60.0, ramp_hz=50.0)
    # Simulate a flaky seed read that only measured gripper.
    monkeypatch.setattr(handle, "read_joints_deg", lambda: {"gripper": 0.0})

    sent = handle.send_goal({"shoulder_pan": 30.0, "gripper": 10.0})

    assert "shoulder_pan" not in sent
    assert sent["gripper"] == pytest.approx(1.2)
