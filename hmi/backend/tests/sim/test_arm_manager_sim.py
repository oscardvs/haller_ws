"""ArmManager constructs SimArmHandle for sim arms and shares one MuJoCoWorld."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from unittest.mock import patch

from haller_hmi.arm import ArmManager
from haller_hmi.config import ArmConfig
from haller_hmi.sim.arm import SimArmHandle


def test_mixed_real_and_sim_arms_share_one_world(monkeypatch):
    # Stub the real-arm bring-up path so we don't open /dev/null.
    from unittest.mock import MagicMock
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    monkeypatch.setattr(
        "haller_hmi.arm.ArmHandle._load_joint_limits",
        lambda self: {"shoulder_pan": (-120.0, 120.0)},
    )

    cfg_real = ArmConfig(id="real_arm", model="so101_follower",
                         port="/dev/null", calibration_id="x", source="real")
    cfg_sim_l = ArmConfig(id="sim_left", model="so101_follower",
                          port="(sim)", calibration_id="(sim)",
                          source="sim", sim_arm_name="left")
    cfg_sim_r = ArmConfig(id="sim_right", model="so101_follower",
                          port="(sim)", calibration_id="(sim)",
                          source="sim", sim_arm_name="right")

    mgr = ArmManager([cfg_real, cfg_sim_l, cfg_sim_r])
    mgr.connect_all()
    try:
        assert isinstance(mgr["sim_left"], SimArmHandle)
        assert isinstance(mgr["sim_right"], SimArmHandle)
        assert not isinstance(mgr["real_arm"], SimArmHandle)
        # Same world instance shared between the two sim arms.
        assert mgr["sim_left"].world is mgr["sim_right"].world
    finally:
        mgr.disconnect_all()


def test_all_real_arms_dont_construct_a_world(monkeypatch):
    from unittest.mock import MagicMock
    monkeypatch.setattr("haller_hmi.arm.SO101Follower", lambda cfg: MagicMock())
    monkeypatch.setattr(
        "haller_hmi.calibration_bootstrap.ensure_follower_calibrations",
        lambda configs: None,
    )
    monkeypatch.setattr(
        "haller_hmi.arm.ArmHandle._load_joint_limits",
        lambda self: {"shoulder_pan": (-120.0, 120.0)},
    )

    cfg = ArmConfig(id="right", model="so101_follower",
                    port="/dev/null", calibration_id="x", source="real")
    with patch("haller_hmi.sim.world.MuJoCoWorld") as MockWorld:
        mgr = ArmManager([cfg])
        mgr.connect_all()
        mgr.disconnect_all()
        MockWorld.assert_not_called()
