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
        # spec_set mirrors the public surface shared by ArmHandle and
        # SimArmHandle. It deliberately omits `.robot` — the lerobot object only
        # real arms carry — so a session that reaches past the handle interface
        # blows up here instead of passing on mocks and then failing against sim
        # arms. See tests/sim/test_human_teleop_sim.py.
        a = MagicMock(spec_set=[
            "config", "joint_limits_deg", "guard", "torque_enabled",
            "connect", "disconnect", "send_goal", "home",
            "enable_torque", "disable_torque", "read_joints_deg", "state_snapshot",
        ])
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
        a.read_joints_deg.return_value = {j: 0.0 for j in a.joint_limits_deg}
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
        # The loop should call send_goal on both arms within ~50 ms.
        assert _wait_until(lambda: arms["left"].send_goal.called)
        assert _wait_until(lambda: arms["right"].send_goal.called)
    finally:
        sess.stop()


def test_commit_loop_does_not_write_when_not_driving():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=False))
        _time.sleep(0.05)
        assert not arms["left"].send_goal.called
        assert not arms["right"].send_goal.called
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
        assert _wait_until(lambda: arms["left"].send_goal.called)
        sent_goals = [c.args[0] for c in arms["left"].send_goal.call_args_list]
        # Every commanded shoulder_pan must be inside [-5, 5].
        for goal in sent_goals:
            assert -5.0 <= goal["shoulder_pan"] <= 5.0
    finally:
        sess.stop()


def test_restart_after_ws_disconnect_does_not_immediately_auto_stop():
    """The WS-disconnect grace timer is per-session state.

    Closing the browser tab fires notify_ws_disconnected() while the session is
    still running, so the timestamp is set at the moment the operator stops.
    If start() doesn't clear it, the *next* session sees an already-expired
    grace window on its first tick and auto-stops — i.e. human teleop works
    exactly once per backend process.
    """
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0, ws_disconnect_grace_s=0.05)
    sess.start(left_arm="left", right_arm="right", swap=False)
    sess.ingest_frame(_kp_frame(dead_man=True))
    sess.notify_ws_disconnected()   # browser tab closed, session still running
    sess.stop()

    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Well past the (tiny) grace window — the fresh session must survive.
        _time.sleep(0.2)
        assert sess.running, "second session auto-stopped on a stale grace timer"
        assert sess.state is HumanState.ARMED
    finally:
        sess.stop()


def test_restart_does_not_inherit_previous_session_targets():
    """A new session must not carry the last session's retarget goals.

    Otherwise pressing Start makes the arms drift toward wherever the operator's
    hands were when they last stopped, before a single new frame has arrived.
    """
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    # A pose well away from neutral, so a leaked target would be obvious.
    frame = _kp_frame(dead_man=True)
    frame["left"]["pose"]["elbow"] = [0.3, 1.4, 0.0]
    frame["left"]["pose"]["wrist"] = [0.6, 1.4, 0.0]
    sess.ingest_frame(frame)
    assert _wait_until(lambda: arms["left"].send_goal.called)
    sess.stop()

    arms["left"].send_goal.reset_mock()
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        assert sess.target_goals()["left"] is None
        assert sess.target_goals()["right"] is None
        assert sess.status()["tracking"]["left"]["age_ms"] is None
        # ARMED with no frames yet: nothing may be commanded.
        _time.sleep(0.05)
        assert not arms["left"].send_goal.called
    finally:
        sess.stop()


def test_per_arm_tracking_loss_freezes_only_that_side():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0,
                              frame_age_ms_loss=80.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        # Drive a frame where only the left side has fresh keypoints.
        frame = _kp_frame(dead_man=True)
        sess.ingest_frame(frame)
        assert _wait_until(lambda: arms["left"].send_goal.called)
        # Now stop the right side from being updated; the left keeps ticking.
        frame_left_only = _kp_frame(dead_man=True)
        frame_left_only["right"] = None
        # Pump a few left-only frames over ~150 ms (> 80 ms threshold).
        for _ in range(20):
            sess.ingest_frame(frame_left_only)
            _time.sleep(0.01)
        status = sess.status()
        assert status["tracking"]["right"]["lost"] is True
        assert status["tracking"]["left"]["lost"] is False
    finally:
        sess.stop()


def test_session_demotes_to_armed_on_ws_disconnect_window():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0,
                              ws_disconnect_grace_s=0.1)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(lambda: sess.state is HumanState.DRIVING)
        sess.notify_ws_disconnected()
        # After the grace window, the loop should auto-stop the session.
        assert _wait_until(lambda: sess.state is HumanState.IDLE, timeout=1.0)
    finally:
        sess.stop()


def test_cannot_start_human_teleop_while_leader_follower_is_running(monkeypatch):
    from haller_hmi.teleop import TeleopSession
    mgr, _ = _fake_arm_manager()
    lf = TeleopSession(mgr)
    # Mark leader/follower as running without actually spawning a thread.
    monkeypatch.setattr(lf, "_state", lf._state.__class__(running=True, leader="left",
                                                         follower="right",
                                                         hz=60.0, tick_count=0,
                                                         last_error=None,
                                                         started_at=_time.time()))

    sess = HumanTeleopSession(mgr)
    sess.attach_peer(lf)  # share the "is anyone teleoping?" check
    with pytest.raises(RuntimeError):
        sess.start(left_arm="left", right_arm="right", swap=False)
