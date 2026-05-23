"""All teleop sessions block each other via attach_peer."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from unittest.mock import MagicMock

import pytest

from haller_hmi.teleop import TeleopSession
from haller_hmi.human_teleop import HumanTeleopSession
from haller_hmi.sim.teleop import SimLeaderTeleop


def _make_three_sessions():
    arms = MagicMock()
    t = TeleopSession(arms)
    h = HumanTeleopSession(arms)
    s = SimLeaderTeleop(arms)
    t.attach_peer(h); t.attach_peer(s)
    h.attach_peer(t); h.attach_peer(s)
    s.attach_peer(t); s.attach_peer(h)
    return t, h, s


def test_teleop_blocks_human_teleop():
    t, h, _ = _make_three_sessions()
    t.status = MagicMock(return_value={"running": True})
    with pytest.raises(RuntimeError):
        # human_teleop.start signature: start(*, left_arm, right_arm, swap, hz)
        h.start(left_arm="left", right_arm="right", swap=False, hz=60.0)


def test_human_teleop_blocks_teleop():
    t, h, _ = _make_three_sessions()
    h.status = MagicMock(return_value={"running": True})
    with pytest.raises(RuntimeError):
        # teleop.start signature: start(leader_id, follower_id, hz)
        t.start(leader_id="left", follower_id="right", hz=60.0)


def test_sim_teleop_blocks_teleop_and_human():
    t, h, s = _make_three_sessions()
    s.status = MagicMock(return_value={"running": True})
    with pytest.raises(RuntimeError):
        t.start(leader_id="left", follower_id="right", hz=60.0)
    with pytest.raises(RuntimeError):
        h.start(left_arm="left", right_arm="right", swap=False, hz=60.0)


def test_teleop_blocks_sim():
    t, h, s = _make_three_sessions()
    t.status = MagicMock(return_value={"running": True})
    src = MagicMock()
    with pytest.raises(RuntimeError):
        s.start(follower_id="right", source=src, hz=60.0)


def test_human_blocks_sim():
    t, h, s = _make_three_sessions()
    h.status = MagicMock(return_value={"running": True})
    src = MagicMock()
    with pytest.raises(RuntimeError):
        s.start(follower_id="right", source=src, hz=60.0)
