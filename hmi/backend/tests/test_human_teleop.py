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


def test_smooth_step_reports_ok_when_nothing_intervenes():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0)}
    # alpha=1.0 -> the filter passes `desired` straight through; small step so
    # the 4 deg/tick cap does not bite and the value is far from the limits.
    steps = sess._smooth_step({"shoulder_pan": 0.0}, {"shoulder_pan": 2.0}, limits, 1.0)
    assert steps["shoulder_pan"].reason == "ok"
    assert steps["shoulder_pan"].committed == pytest.approx(2.0)
    assert steps["shoulder_pan"].target == pytest.approx(2.0)


def test_smooth_step_reports_rate_capped():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0)}
    # Ask for a 50 deg jump with alpha=1.0; the 4 deg/tick cap must bite.
    steps = sess._smooth_step({"shoulder_pan": 0.0}, {"shoulder_pan": 50.0}, limits, 1.0)
    assert steps["shoulder_pan"].reason == "rate_capped"
    assert steps["shoulder_pan"].committed == pytest.approx(4.0)
    # target is what was ASKED for, not what was delivered.
    assert steps["shoulder_pan"].target == pytest.approx(50.0)


def test_smooth_step_reports_clamped_and_clamped_beats_rate_capped():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 5.0)}
    # Sitting at 4 deg, asked for 50: the cap would allow 8, the limit allows 5.
    # Both conditions fire; `clamped` must win.
    steps = sess._smooth_step({"shoulder_pan": 4.0}, {"shoulder_pan": 50.0}, limits, 1.0)
    assert steps["shoulder_pan"].reason == "clamped"
    assert steps["shoulder_pan"].committed == pytest.approx(5.0)


def test_smooth_step_reports_held_when_side_has_no_target():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0), "elbow_flex": (-90.0, 90.0)}
    steps = sess._smooth_step({"shoulder_pan": 12.0, "elbow_flex": 3.0}, None, limits, 1.0)
    for joint in limits:
        assert steps[joint].reason == "held"
        assert steps[joint].target is None
    assert steps["shoulder_pan"].committed == pytest.approx(12.0)


def test_smooth_step_reports_held_for_a_joint_missing_from_the_target():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"shoulder_pan": (-90.0, 90.0), "elbow_flex": (-90.0, 90.0)}
    steps = sess._smooth_step(
        {"shoulder_pan": 0.0, "elbow_flex": 7.0}, {"shoulder_pan": 2.0}, limits, 1.0,
    )
    assert steps["shoulder_pan"].reason == "ok"
    assert steps["elbow_flex"].reason == "held"
    assert steps["elbow_flex"].target is None
    assert steps["elbow_flex"].committed == pytest.approx(7.0)


def test_smooth_step_reports_gripper_target_in_degrees_not_unit_interval():
    """retarget emits gripper in [0,1]; _smooth_step scales it onto the joint's
    degree range. `target` must be reported post-scaling so it is comparable
    with `committed`, which is always degrees."""
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    limits = {"gripper": (-30.0, 30.0)}
    # 1.0 == fully open == the joint's max, 30 deg.
    steps = sess._smooth_step({"gripper": 0.0}, {"gripper": 1.0}, limits, 1.0)
    assert steps["gripper"].target == pytest.approx(30.0)
    assert steps["gripper"].target > 1.0, "gripper target leaked as [0,1]"


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


def test_status_joints_block_mirrors_goal_deg_keys():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        st = sess.status()
        assert set(st["joints"]["left"]) == set(st["goal_deg"]["left"])
        for entry in st["joints"]["left"].values():
            assert set(entry) == {"target", "committed", "reason"}
    finally:
        sess.stop()


def test_status_joints_are_held_before_any_frame_arrives():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        st = sess.status()
        for side in ("left", "right"):
            for entry in st["joints"][side].values():
                assert entry["reason"] == "held"
                assert entry["target"] is None
    finally:
        sess.stop()


def test_status_joints_revert_to_held_after_stop():
    """After stop() nothing is being asked for, so no joint may still advertise
    a live reason. A retained CLAMPED badge from an ended session would tell the
    operator the arm is at a limit it is no longer being driven into."""
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    sess.ingest_frame(_kp_frame(dead_man=True))
    _time.sleep(0.05)
    sess.stop()

    st = sess.status()
    for side in ("left", "right"):
        for joint, entry in st["joints"][side].items():
            assert entry["reason"] == "held", f"{side}.{joint} kept a live reason after stop"
            assert entry["target"] is None
    # The committed values themselves are still retained, matching goal_deg.
    assert st["joints"]["left"].keys() == st["goal_deg"]["left"].keys()


def test_status_goal_deg_shape_is_unchanged_by_the_joints_block():
    """goal_deg is DatasetRecorder's `action` column. It must stay a plain
    joint -> float mapping."""
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, hz_override=200.0)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        _time.sleep(0.05)
        goal = sess.status()["goal_deg"]["left"]
        assert goal, "goal_deg must not be empty while driving"
        for value in goal.values():
            assert isinstance(value, float)
    finally:
        sess.stop()


# ---- mouth-open dead-man clutch: session wiring -----------------------
#
# This file has no shared "started session" fixture; every test above builds
# its own mgr/session and stops it in a `finally`. The mouth-clutch tests
# follow the same pattern rather than introducing a new one.

from haller_hmi.safety import MOUTH_HOLD_MS


def _mouth_frame(jaw, *, ts_ms=0):
    """A minimal mouth-mode keypoint frame carrying no side data."""
    return {
        "type": "keypoints", "ts_ms": ts_ms,
        "clutch_source": "mouth", "dead_man": False,
        "jaw_open": jaw, "left": None, "right": None,
    }


def test_mouth_calib_is_valid_on_a_real_session():
    """The HTTP-400 start gate is exactly this predicate — and the route tests
    mock it out, so nothing else asserts what a real session computes. Both
    halves of that seam were "covered"; the seam itself was not."""
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    # Never calibrated at all.
    assert sess.mouth_calib_is_valid() is False
    # Separation 0.05 — speech overlaps the deliberate open, no safe threshold.
    sess.set_mouth_calib({"talk_max": 0.50, "open_min": 0.55})
    assert sess.mouth_calib_is_valid() is False
    # Inverted captures are not valid either.
    sess.set_mouth_calib({"talk_max": 0.90, "open_min": 0.10})
    assert sess.mouth_calib_is_valid() is False
    sess.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
    assert sess.mouth_calib_is_valid() is True
    # Clearing it takes the gate back to refusing.
    sess.set_mouth_calib(None)
    assert sess.mouth_calib_is_valid() is False


def test_mouth_releases_immediately_at_session_level(monkeypatch):
    """Release must never be debounced.

    test_safety.py guards that for the pure function, but the session is what
    adds the hold timer and feeds `self._dead_man` in as the hysteresis input.
    A regression there — a release that had to be sustained the way an engage
    does — would leave the pure-function test green while the arms kept
    driving after the operator closed their mouth.
    """
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="mouth")
    try:
        sess.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
        clock = {"t": 1000.0}
        monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                            lambda: clock["t"])
        sess.ingest_frame(_mouth_frame(0.95))
        clock["t"] += (MOUTH_HOLD_MS + 10) / 1000.0
        sess.ingest_frame(_mouth_frame(0.95))
        assert sess.status()["clutch"]["engaged"] is True
        assert sess.status()["state"] == "driving"

        # ONE sample below t_release (0.34 for this calibration), and NO clock
        # advance at all: the very next frame must have dropped the arms.
        sess.ingest_frame(_mouth_frame(0.20))
        st = sess.status()
        assert st["clutch"]["engaged"] is False
        assert st["clutch"]["reason"] == "below_threshold"
        assert st["state"] != "driving"
    finally:
        sess.stop()


def test_spacebar_mode_ignores_jaw_open():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        sess.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
        sess.ingest_frame({
            "type": "keypoints", "ts_ms": 0,
            "clutch_source": "spacebar", "dead_man": False,
            "jaw_open": 0.99, "left": None, "right": None,
        })
        st = sess.status()
        assert st["state"] != "driving"
        assert st["clutch"]["source"] == "spacebar"
        assert st["clutch"]["reason"] == "spacebar_mode"
    finally:
        sess.stop()


def test_mouth_mode_ignores_dead_man_boolean():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="mouth")
    try:
        sess.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
        sess.ingest_frame({
            "type": "keypoints", "ts_ms": 0,
            "clutch_source": "mouth", "dead_man": True,
            "jaw_open": 0.01, "left": None, "right": None,
        })
        assert sess.status()["state"] != "driving"
    finally:
        sess.stop()


def test_mouth_uncalibrated_never_engages():
    # No calibration set at all.
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="mouth")
    try:
        sess.ingest_frame(_mouth_frame(0.99))
        st = sess.status()
        assert st["state"] != "driving"
        assert st["clutch"]["reason"] == "uncalibrated"
    finally:
        sess.stop()


def test_mouth_invalid_calibration_never_engages():
    # Separation 0.05 — below MOUTH_MIN_SEPARATION.
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="mouth")
    try:
        sess.set_mouth_calib({"talk_max": 0.50, "open_min": 0.55})
        sess.ingest_frame(_mouth_frame(0.99))
        st = sess.status()
        assert st["state"] != "driving"
        assert st["clutch"]["reason"] == "uncalibrated"
    finally:
        sess.stop()


def test_mouth_engages_after_sustained_hold(monkeypatch):
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="mouth")
    try:
        sess.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
        clock = {"t": 1000.0}
        monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                            lambda: clock["t"])
        sess.ingest_frame(_mouth_frame(0.95))
        assert sess.status()["clutch"]["reason"] == "holding"
        assert sess.status()["state"] != "driving"

        clock["t"] += (MOUTH_HOLD_MS + 10) / 1000.0
        sess.ingest_frame(_mouth_frame(0.95))
        assert sess.status()["clutch"]["engaged"] is True
        assert sess.status()["clutch"]["reason"] == "engaged"
    finally:
        sess.stop()


def test_mouth_hold_is_not_satisfied_by_a_face_dropout(monkeypatch):
    """Losing the face must never ENGAGE the arms.

    FACE_STALE_MS (250) sits deliberately above MOUTH_HOLD_MS (200), so there
    is a ~50 ms window in which a wall-clock hold timer is satisfied purely by
    extrapolation from one sample that nothing is confirming any more —
    `_jaw_open` retains its last value across `jaw_open: null` frames by
    design. Measured against the wall clock, a face dropout after a single
    above-threshold sample therefore *starts* the arms, inverting the fail-safe
    the whole design rests on (spec §6.1 "sustained continuously", §6.4 "every
    fault resolves to disengaged").

    The hold is measured between OBSERVATIONS instead, so a stream of nulls
    accrues nothing.
    """
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="mouth")
    try:
        sess.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
        clock = {"t": 1000.0}
        monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                            lambda: clock["t"])
        # ONE real above-threshold sample, then the face is lost: every later
        # frame carries jaw_open=None.
        sess.ingest_frame(_mouth_frame(0.95))
        assert sess.status()["clutch"]["engaged"] is False

        # Walk right through the ~50 ms window between MOUTH_HOLD_MS and the
        # staleness budget, where the wall-clock timer would fire.
        for _ in range(7):          # 7 x 33 ms = 231 ms: past 200, under 250
            clock["t"] += 0.033
            sess.ingest_frame(_mouth_frame(None))
            st = sess.status()
            assert st["clutch"]["engaged"] is False, (
                "a face dropout engaged the clutch from a single sample"
            )
            assert st["state"] != "driving"

        # And the budget expiring still reports the fault as a fault.
        clock["t"] += 0.033          # 264 ms since the only real sample
        sess.ingest_frame(_mouth_frame(None))
        st = sess.status()
        assert st["clutch"]["stale"] is True
        assert st["clutch"]["reason"] == "stale"
        assert st["state"] != "driving"
    finally:
        sess.stop()


def test_mouth_decimated_nulls_within_budget_do_not_disengage(monkeypatch):
    """Normal operation is NOT a fault: face runs every 3rd frame, so two
    frames in three legitimately carry jaw_open=None."""
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="mouth")
    try:
        sess.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
        clock = {"t": 1000.0}
        monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                            lambda: clock["t"])
        sess.ingest_frame(_mouth_frame(0.95))
        clock["t"] += (MOUTH_HOLD_MS + 10) / 1000.0
        sess.ingest_frame(_mouth_frame(0.95))
        assert sess.status()["clutch"]["engaged"] is True

        # Two null frames, 33ms apart — well inside the 250ms budget.
        clock["t"] += 0.033
        sess.ingest_frame(_mouth_frame(None))
        clock["t"] += 0.033
        sess.ingest_frame(_mouth_frame(None))
        assert sess.status()["clutch"]["engaged"] is True
        assert sess.status()["clutch"]["stale"] is False
    finally:
        sess.stop()


def test_mouth_stale_face_disengages(monkeypatch):
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="mouth")
    try:
        sess.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
        clock = {"t": 1000.0}
        monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                            lambda: clock["t"])
        sess.ingest_frame(_mouth_frame(0.95))
        clock["t"] += (MOUTH_HOLD_MS + 10) / 1000.0
        sess.ingest_frame(_mouth_frame(0.95))
        assert sess.status()["clutch"]["engaged"] is True

        # 300ms with no real sample — past the 250ms budget.
        clock["t"] += 0.300
        sess.ingest_frame(_mouth_frame(None))
        st = sess.status()
        assert st["clutch"]["engaged"] is False
        assert st["clutch"]["stale"] is True
        assert st["clutch"]["reason"] == "stale"
        assert st["state"] != "driving"
    finally:
        sess.stop()


def _spacebar_frame(dead_man: bool):
    return {
        "type": "keypoints", "ts_ms": 0,
        "clutch_source": "spacebar", "dead_man": dead_man,
        "jaw_open": None, "left": None, "right": None,
    }


def test_the_browser_cannot_hand_authority_to_the_other_source(monkeypatch):
    """The session's authority is what start() was told, for its whole life.

    A frame does not get to reassign it. When authority lived in the frame,
    a mouth session took exactly two spacebar frames to become a spacebar
    session — the first disengaged, the second drove — which is spec 1's
    "sole authority for that session" weakened to "authority for ~33 ms".
    """
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False,
               clutch_source="mouth")
    try:
        sess.set_mouth_calib({"talk_max": 0.10, "open_min": 0.90})
        clock = {"t": 1000.0}
        monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                            lambda: clock["t"])
        sess.ingest_frame(_mouth_frame(0.95))
        clock["t"] += (MOUTH_HOLD_MS + 10) / 1000.0
        sess.ingest_frame(_mouth_frame(0.95))
        assert sess.status()["state"] == "driving"

        # Authority must never hand over mid-motion, even though every one of
        # these frames arrives with dead_man=True.
        for i in range(5):
            clock["t"] += 0.033
            sess.ingest_frame(_spacebar_frame(True))
            st = sess.status()
            assert st["state"] != "driving", f"spacebar frame {i} took the arms"
            assert st["clutch"]["engaged"] is False
            # And the session still reports the source it was started with.
            assert st["clutch"]["source"] == "mouth"
    finally:
        sess.stop()


def test_a_spacebar_session_ignores_mouth_frames_entirely(monkeypatch):
    """The mirror image: frame-carried mouth data cannot arm a spacebar
    session, calibration included. Otherwise a session started under the
    spacebar could be driven by a face it was never armed for."""
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    sess.start(left_arm="left", right_arm="right", swap=False)
    try:
        clock = {"t": 1000.0}
        monkeypatch.setattr("haller_hmi.human_teleop.time.perf_counter",
                            lambda: clock["t"])
        for _ in range(5):
            clock["t"] += 0.100
            frame = _mouth_frame(0.99)
            # Calibration riding on the frame must not arm the mouth clutch
            # in a session that did not select it.
            frame["mouth_calib"] = {"talk_max": 0.10, "open_min": 0.90}
            sess.ingest_frame(frame)
            st = sess.status()
            assert st["state"] != "driving"
            assert st["clutch"]["source"] == "spacebar"
        assert sess.mouth_calib_is_valid() is False
    finally:
        sess.stop()


def test_start_rejects_an_unknown_clutch_source():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr)
    with pytest.raises(ValueError):
        sess.start(left_arm="left", right_arm="right", swap=False,
                   clutch_source="eyebrow")
    assert sess.state is HumanState.IDLE
