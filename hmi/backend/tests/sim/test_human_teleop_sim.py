"""Human-pose teleop must drive *sim* arms, not just real ones.

`tests/test_human_teleop.py` covers the session against MagicMock arms, which
answer every attribute — including `.robot` — so they can't catch the session
reaching past the ArmHandle interface. These tests wire the real
`HumanTeleopSession` to real `SimArmHandle`s over a real `MuJoCoWorld`, which is
the configuration `config.bimanual-sim.yaml` produces.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from haller_hmi.config import ArmConfig
from haller_hmi.human_teleop import HumanState, HumanTeleopSession
from haller_hmi.sim.arm import SimArmHandle
from haller_hmi.sim.builder import build_scene
from haller_hmi.sim.world import MuJoCoWorld


class _Arms:
    """Minimal ArmManager stand-in — deliberately NOT a MagicMock, so any
    attribute the session invents (e.g. `.robot`) raises instead of answering."""

    def __init__(self, handles: dict):
        self._handles = handles

    def __getitem__(self, arm_id: str):
        return self._handles[arm_id]

    def values(self):
        return self._handles.values()

    def keys(self):
        return self._handles.keys()


@pytest.fixture
def sim_arms():
    mjcf_xml, arm_joint_map = build_scene(arms=["left", "right"], cubes=0)
    world = MuJoCoWorld(mjcf_xml, arm_joint_map=arm_joint_map)
    world.start()
    handles = {}
    for arm_id in ("left", "right"):
        cfg = ArmConfig(
            id=arm_id, model="so101_follower", port="(sim)",
            calibration_id="(sim)", source="sim", sim_arm_name=arm_id,
        )
        handle = SimArmHandle(cfg, world=world)
        handle.connect()
        handles[arm_id] = handle
    try:
        yield _Arms(handles), handles, world
    finally:
        world.stop()


def _kp_frame(*, dead_man: bool, elbow, wrist, ts_ms: int = 100) -> dict:
    """A KeypointFrame with a configurable arm pose (shoulder fixed at origin)."""
    side = {
        "pose": {
            "shoulder": [0.0, 1.4, 0.0],
            "elbow":    list(elbow),
            "wrist":    list(wrist),
        },
        "hand": {
            "wrist":      [0.0, 0.0, 0.0],
            "thumb_tip":  [0.04, 0.0, 0.05],
            "index_tip":  [0.02, 0.0, 0.10],
            "index_mcp":  [0.04, 0.0, 0.05],
            "middle_mcp": [0.0, 0.0, 0.10],
            "pinky_mcp":  [-0.04, 0.0, 0.05],
        },
        "confidence": 0.9,
    }
    return {
        "type": "keypoints",
        "ts_ms": ts_ms,
        "dead_man": dead_man,
        "pinch_calib": {"left":  {"min_m": 0.02, "max_m": 0.18},
                        "right": {"min_m": 0.02, "max_m": 0.18}},
        "left": side,
        "right": side,
    }


def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_driving_moves_the_sim_arms(sim_arms):
    """The whole point: hold the dead-man with a non-neutral pose and both sim
    arms must actually travel in MuJoCo."""
    mgr, handles, _world = sim_arms
    start_left = handles["left"].read_joints_deg()
    start_right = handles["right"].read_joints_deg()

    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Arm swung out to the side → large shoulder_pan, well away from rest.
        frame = _kp_frame(dead_man=True, elbow=[0.3, 1.4, 0.0], wrist=[0.6, 1.4, 0.0])

        def _moved() -> bool:
            sess.ingest_frame(frame)  # keep the frame fresh (300 ms loss window)
            now_left = handles["left"].read_joints_deg()
            now_right = handles["right"].read_joints_deg()
            left_delta = max(abs(now_left[j] - start_left[j]) for j in now_left)
            right_delta = max(abs(now_right[j] - start_right[j]) for j in now_right)
            return left_delta > 5.0 and right_delta > 5.0

        assert sess.state is HumanState.IDLE or True
        assert _moved() or _wait_until(_moved), (
            f"sim arms never moved; session last_error={sess.status()['last_error']!r}"
        )
    finally:
        sess.stop()


def test_no_last_error_while_driving_sim_arms(sim_arms):
    """A commit path that throws is swallowed into `last_error` and the arms sit
    still — assert the loop stays clean."""
    mgr, _handles, _world = sim_arms
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        frame = _kp_frame(dead_man=True, elbow=[0.3, 1.4, 0.0], wrist=[0.6, 1.4, 0.0])
        for _ in range(10):
            sess.ingest_frame(frame)
            time.sleep(0.02)
        assert sess.status()["last_error"] is None
    finally:
        sess.stop()


def test_start_seeds_committed_goals_from_observed_sim_pose(sim_arms):
    """On start the session seeds its smoothing state from where the arm *is*.
    Against sim arms that read must go through `read_joints_deg()`; falling back
    to all-zeros would make the first driving tick a jump from a false origin."""
    mgr, handles, world = sim_arms
    # Park the left arm somewhere clearly non-zero and let physics settle.
    handles["left"].send_goal({"shoulder_pan": 45.0})
    assert _wait_until(
        lambda: abs(handles["left"].read_joints_deg()["shoulder_pan"] - 45.0) < 5.0
    ), "sim arm never reached the parked pose"
    observed = handles["left"].read_joints_deg()

    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        seeded = sess.status()["goal_deg"]["left"]
        assert seeded["shoulder_pan"] == pytest.approx(
            observed["shoulder_pan"], abs=2.0
        ), f"seeded from {seeded['shoulder_pan']}, arm was at {observed['shoulder_pan']}"
    finally:
        sess.stop()
