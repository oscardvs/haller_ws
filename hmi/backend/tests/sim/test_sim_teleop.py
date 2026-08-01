"""SimLeaderTeleop ticks a LeaderSource and forwards the pose to a sim follower."""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

from unittest.mock import MagicMock

from haller_hmi.config import ArmConfig
from haller_hmi.arm import ArmManager
from haller_hmi.sim.teleop import SimLeaderTeleop


def test_sim_teleop_calls_send_goal_at_configured_rate():
    cfg_l = ArmConfig(id="sim_left",  model="so101_follower", port="(sim)",
                      calibration_id="(sim)", source="sim", sim_arm_name="left")
    cfg_r = ArmConfig(id="sim_right", model="so101_follower", port="(sim)",
                      calibration_id="(sim)", source="sim", sim_arm_name="right")
    mgr = ArmManager([cfg_l, cfg_r])
    mgr.connect_all(teleop_peers=[])
    try:
        fake_source = MagicMock()
        fake_source.read.return_value = {"shoulder_pan": 5.0}
        fake_source.start = MagicMock()
        fake_source.stop = MagicMock()

        follower = mgr["sim_right"]
        follower.send_goal = MagicMock(return_value={"shoulder_pan": 5.0})

        session = SimLeaderTeleop(arms=mgr)
        session.start(follower_id="sim_right", source=fake_source, hz=120.0)
        time.sleep(0.1)
        session.stop()

        assert fake_source.start.called
        assert fake_source.read.call_count >= 2
        assert follower.send_goal.called
        assert fake_source.stop.called
    finally:
        mgr.disconnect_all()


def test_sim_teleop_status_shape():
    s = SimLeaderTeleop(arms=MagicMock())
    out = s.status()
    assert set(out) >= {"running", "follower", "hz", "tick_count", "last_error"}


def test_sim_teleop_blocks_when_peer_running():
    """Session lock: if a peer (e.g. real TeleopSession or HumanTeleopSession) is
    already running, sim teleop must refuse to start."""
    cfg = ArmConfig(id="sim_right", model="so101_follower", port="(sim)",
                    calibration_id="(sim)", source="sim", sim_arm_name="right")
    mgr = ArmManager([cfg])
    mgr.connect_all(teleop_peers=[])
    try:
        session = SimLeaderTeleop(arms=mgr)
        peer = MagicMock()
        peer.status = MagicMock(return_value={"running": True})
        session.attach_peer(peer)
        fake_source = MagicMock()
        import pytest
        with pytest.raises(RuntimeError, match="already running"):
            session.start(follower_id="sim_right", source=fake_source, hz=60.0)
    finally:
        mgr.disconnect_all()
