"""Tests for HumanTeleopSession — session lifecycle + state machine."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from haller_hmi.human_teleop import HumanState, HumanTeleopSession
from haller_hmi.safety import Mode


def _fake_arm_manager():
    """Two mocked arms ("left", "right") with realistic joint_limits_deg + guard."""
    mgr = MagicMock()

    def _mkarm(arm_id: str):
        a = MagicMock()
        a.config = MagicMock(id=arm_id)
        a.joint_limits_deg = {
            "shoulder_pan":  (-90.0, 90.0),
            "shoulder_lift": (-90.0, 90.0),
            "elbow_flex":    (-90.0, 90.0),
            "wrist_flex":    (-90.0, 90.0),
            "wrist_roll":    (-90.0, 90.0),
            "gripper":       (-30.0, 30.0),
        }
        a.guard = MagicMock(mode=Mode.MANUAL)
        a.torque_enabled = True
        a.robot = MagicMock()
        return a

    arms = {"left": _mkarm("left"), "right": _mkarm("right")}
    mgr.__getitem__.side_effect = lambda k: arms[k]
    mgr.values.return_value = list(arms.values())
    mgr.keys.return_value = list(arms.keys())
    return mgr, arms


def test_initial_state_is_idle():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    assert sess.state is HumanState.IDLE
    assert sess.status()["running"] is False


def test_start_transitions_to_armed_and_prepares_arms():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        assert sess.state is HumanState.ARMED
        assert sess.status()["running"] is True
        # Both arms should be enabled with torque on and MANUAL mode.
        for a in arms.values():
            a.guard.set.assert_called_with(Mode.MANUAL)
    finally:
        sess.stop()


def test_stop_restores_arms_to_manual_and_torque_on():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    sess.stop()
    assert sess.state is HumanState.IDLE
    for a in arms.values():
        # The most recent guard.set should be MANUAL.
        a.guard.set.assert_called_with(Mode.MANUAL)


def test_start_twice_raises_runtime_error():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        with pytest.raises(RuntimeError):
            sess.start(left_arm="left", right_arm="right", swap=False)
    finally:
        sess.stop()


def test_start_requires_distinct_arms():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    with pytest.raises(ValueError):
        sess.start(left_arm="left", right_arm="left", swap=False)


def _kp_frame(
    *, ts_ms: int = 100, dead_man: bool = False, both_arms: bool = True,
    calib: dict | None = None, confidence: float = 0.9,
) -> dict:
    """A minimal valid KeypointFrame: arm straight forward, hand neutral."""
    side = {
        "pose": {
            "shoulder": [0.0, 1.4, 0.0],
            "elbow":    [0.0, 1.4, 0.3],
            "wrist":    [0.0, 1.4, 0.6],
        },
        "hand": {
            "wrist":      [0.0, 0.0, 0.0],
            "thumb_tip":  [0.04, 0.0, 0.05],
            "index_tip":  [0.02, 0.0, 0.10],
            "index_mcp":  [0.04, 0.0, 0.05],
            "middle_mcp": [0.0, 0.0, 0.10],
            "pinky_mcp":  [-0.04, 0.0, 0.05],
        },
        "confidence": confidence,
    }
    return {
        "type": "keypoints",
        "ts_ms": ts_ms,
        "dead_man": dead_man,
        "pinch_calib": calib or {
            "left":  {"min_m": 0.02, "max_m": 0.18},
            "right": {"min_m": 0.02, "max_m": 0.18},
        },
        "left":  side if both_arms else None,
        "right": side if both_arms else None,
    }


def test_first_ingest_transitions_armed_to_tracking():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        assert sess.state is HumanState.ARMED
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.state is HumanState.TRACKING
    finally:
        sess.stop()


def test_dead_man_held_transitions_tracking_to_driving():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.state is HumanState.TRACKING
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert sess.state is HumanState.DRIVING
        sess.ingest_frame(_kp_frame(dead_man=False))
        assert sess.state is HumanState.TRACKING
    finally:
        sess.stop()


def test_ingest_records_latest_target_goal():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        targets = sess.target_goals()
        assert "left" in targets and "right" in targets
        # Arm straight forward → angles all near zero.
        for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex"):
            assert abs(targets["left"][joint]) < 2.0
    finally:
        sess.stop()


def test_ingest_handles_missing_side_gracefully():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Only left detected, right is None.
        frame = _kp_frame(dead_man=False)
        frame["right"] = None
        sess.ingest_frame(frame)
        targets = sess.target_goals()
        # Left should be set, right held at last (initialized to None).
        assert "left" in targets
        assert targets.get("right") is None
    finally:
        sess.stop()


import time as _time


def _wait_until(predicate, timeout: float = 1.0, interval: float = 0.01) -> bool:
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(interval)
    return False


def test_commit_loop_writes_to_arms_when_driving():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)  # fast loop for tests
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        # The loop should call send_action on both arms within ~50 ms.
        assert _wait_until(lambda: arms["left"].robot.send_action.called)
        assert _wait_until(lambda: arms["right"].robot.send_action.called)
    finally:
        sess.stop()


def test_commit_loop_does_not_write_when_not_driving():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        _time.sleep(0.05)
        assert not arms["left"].robot.send_action.called
        assert not arms["right"].robot.send_action.called
    finally:
        sess.stop()


def test_commit_loop_clamps_to_arm_joint_limits():
    mgr, arms = _fake_arm_manager()
    # Squeeze the left arm's pan limit so retarget output gets clamped.
    arms["left"].joint_limits_deg["shoulder_pan"] = (-5.0, 5.0)
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Build a frame whose left-arm pan should retarget to ≈ +90° (far above 5°).
        frame = _kp_frame(dead_man=True)
        frame["left"]["pose"]["elbow"] = [0.3, 1.4, 0.0]
        frame["left"]["pose"]["wrist"] = [0.6, 1.4, 0.0]
        sess.ingest_frame(frame)
        _wait_until(lambda: arms["left"].robot.send_action.called)
        sent_actions = [c.args[0] for c in arms["left"].robot.send_action.call_args_list]
        # Every commanded shoulder_pan must be inside [-5, 5].
        for action in sent_actions:
            assert -5.0 <= action["shoulder_pan.pos"] <= 5.0
    finally:
        sess.stop()
