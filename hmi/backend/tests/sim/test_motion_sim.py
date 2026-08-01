"""The 2026-08-01 incident, reproduced.

Home was pressed straight after recalibrating the right arm. Calibration had
just redefined 0°, the arm was parked far from it, and home() issued a single
unbounded position write. The arm slewed into the bench and stalled six servos,
which burnt the 7.4 V DC-DC.
"""
from unittest.mock import MagicMock

import pytest

from haller_hmi import motion
from haller_hmi.config import MotionConfig
from haller_hmi.motion import MoveRefused

from .test_sim_arm_handle import _make_world_and_handle


def test_home_refuses_from_a_post_calibration_pose():
    world, handle = _make_world_and_handle()
    handle.connect()
    handle.motion = MotionConfig(max_speed_deg_s=60.0, large_move_deg=30.0,
                                 ramp_hz=50.0)
    handle.enable_torque()

    # Where the right arm actually sat after its sweep, in the frame the new
    # calibration had just established. TINY_XML models shoulder_pan + gripper.
    handle.read_joints_deg = MagicMock(
        return_value={"shoulder_pan": -126.5, "gripper": 0.0}
    )
    world.write_ctrl_deg = MagicMock()

    with pytest.raises(MoveRefused) as e:
        motion.home(handle)

    assert "shoulder_pan" in str(e.value)
    world.write_ctrl_deg.assert_not_called()


def test_both_handles_share_one_home_implementation():
    """Parity guard. Fails if anyone reintroduces a per-handle home(), which is
    the structural regression that made the incident possible."""
    from haller_hmi.arm import ArmHandle
    from haller_hmi.sim.arm import SimArmHandle

    assert not hasattr(ArmHandle, "home")
    assert not hasattr(SimArmHandle, "home")
