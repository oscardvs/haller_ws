"""Single-arm sessions: one arm on the bench, one hand on the grip.

The session was bimanual by construction — it refused to start unless it was
handed two distinct arm ids, and its commit loop wrote to both unconditionally.
That made a first hardware bring-up impossible to do on one arm, which is the
way you would actually want to do it, and left a rig with one working servo
board with no teleop at all.

What these pin is that the absent side is genuinely inert: never acquires,
never written, never able to be homed, and honest about why in `status()`.
"""
from __future__ import annotations

import time as _time

import pytest

from haller_hmi.human_teleop import HumanTeleopSession, SideAuthority

from .test_human_teleop import (
    _fake_arm_manager,
    _fast_acquire,
    _kp_frame,
    _wait_until,
)


def _session(left, right, **kw):
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, **_fast_acquire(**kw))
    sess.start(left_arm=left, right_arm=right)
    return sess, arms


def _sided_frame(*, left: bool, right: bool):
    frame = _kp_frame(dead_man=left or right)
    frame["dead_man_sides"] = {"left": left, "right": right}
    return frame


def test_starts_with_only_a_right_arm():
    sess, _ = _session(None, "right")
    try:
        st = sess.status()
        assert st["running"] is True
        assert st["left_arm"] is None
        assert st["right_arm"] == "right"
    finally:
        sess.stop()


def test_starts_with_only_a_left_arm():
    sess, _ = _session("left", None)
    try:
        assert sess.status()["left_arm"] == "left"
        assert sess.status()["right_arm"] is None
    finally:
        sess.stop()


def test_refuses_a_session_with_no_arms_at_all():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, **_fast_acquire())
    with pytest.raises(ValueError, match="at least one"):
        sess.start(left_arm=None, right_arm=None)


def test_still_refuses_the_same_arm_twice():
    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, **_fast_acquire())
    with pytest.raises(ValueError, match="must be different"):
        sess.start(left_arm="right", right_arm="right")


def test_the_absent_side_says_why_rather_than_claiming_tracking_loss():
    """`no_tracking` would send the operator hunting for a hand that is not
    missing; the side simply has no arm."""
    sess, _ = _session(None, "right")
    try:
        sess.ingest_frame(_sided_frame(left=True, right=True))
        acq = sess.status()["acquire"]
        assert acq["left"]["authority"] == SideAuthority.HELD.value
        assert acq["left"]["reason"] == "no_arm"
        assert acq["right"]["authority"] == SideAuthority.DRIVING.value
    finally:
        sess.stop()


def test_the_present_arm_is_driven_and_nothing_else_is():
    sess, arms = _session(None, "right", hz_override=200.0)
    try:
        sess.ingest_frame(_sided_frame(left=True, right=True))
        assert _wait_until(lambda: arms["right"].send_goal.called)
        assert not arms["left"].send_goal.called
    finally:
        sess.stop()


def test_the_loop_survives_a_missing_side_for_a_while():
    """The commit loop runs the same code with one half inert. If any of the
    guards were missing it would raise, and the session would stop itself
    after MAX_CONSECUTIVE_TICK_ERRORS — so a session still running with no
    last_error is the assertion."""
    sess, _ = _session(None, "right", hz_override=200.0)
    try:
        for _ in range(4):
            sess.ingest_frame(_sided_frame(left=True, right=True))
            _time.sleep(0.02)
        st = sess.status()
        assert st["running"] is True
        assert st["last_error"] is None
    finally:
        sess.stop()


def test_home_only_touches_the_side_that_has_an_arm():
    sess, _ = _session(None, "right")
    try:
        assert sess.request_home() == ["right"]
    finally:
        sess.stop()


def test_stopping_restores_only_the_arm_it_took():
    sess, arms = _session(None, "right")
    sess.ingest_frame(_sided_frame(left=False, right=True))
    sess.stop()
    assert arms["right"].guard.mode.value == "manual"
    # The untouched arm was never put into MANUAL by this session, so its
    # mode is whatever it already was — the fixture's default.
    assert not arms["left"].enable_torque.called


def test_collision_guard_sees_only_the_driven_arm():
    """A single-arm session still gets the self-collision pairs and the bench
    floors; there is simply no second arm for the inter-arm checks to find."""
    seen: list[set[str]] = []

    class RecordingGuard:
        cfg = type("C", (), {"margin_m": 0.025})()
        enabled = True
        available = True

        def clearance(self, poses):
            seen.append(set(poses))
            return type("Cl", (), {"slack": 1.0, "worst": "none"})()

        def filter_step(self, prev, want):
            seen.append(set(want))
            return type("R", (), {"poses": want, "alpha": 1.0, "limited": False,
                                  "clearance": self.clearance(want)})()

    mgr, _ = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, collision_guard=RecordingGuard(),
                              **_fast_acquire())
    sess.start(left_arm=None, right_arm="right")
    try:
        for _ in range(4):
            sess.ingest_frame(_sided_frame(left=False, right=True))
        assert seen, "the guard was never consulted"
        assert all(s == {"right"} for s in seen)
    finally:
        sess.stop()


def test_a_second_socket_dropping_cannot_extend_the_disconnect_grace():
    """The grace window measures time since the last FRAME, not since the
    last socket close.

    More than one socket can now feed a session — the VR relay hands frames to
    the same one, and every idle client it drops calls
    `notify_ws_disconnected` too. Re-stamping the clock on each of those made
    the window something an unrelated page could hold open forever: measured
    on the bench with one stray browser tab reconnecting every three seconds,
    a session whose driving headset had genuinely gone never stopped itself.
    """
    sess, _ = _session(None, "right", hz_override=200.0)
    try:
        sess.ingest_frame(_sided_frame(left=False, right=True))
        sess.notify_ws_disconnected()
        started = sess._ws_disconnected_at_perf
        assert started is not None
        _time.sleep(0.05)
        # A second socket going away must not push the deadline out.
        sess.notify_ws_disconnected()
        assert sess._ws_disconnected_at_perf == started
        # A frame is what proves an operator is still there, and it clears it.
        sess.ingest_frame(_sided_frame(left=False, right=True))
        assert sess._ws_disconnected_at_perf is None
    finally:
        sess.stop()
