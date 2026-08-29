"""The session ↔ collision-guard contract.

The guard itself is covered by test_collision.py; here it is a stub, and what
is under test is the wiring: the loop must write what the guard returned (not
what smoothing produced), surface the guard's verdict in status(), and stay
alive if the guard blows up.
"""
from __future__ import annotations

import time as _time
from types import SimpleNamespace

from haller_hmi.collision import Clearance, GuardResult
from haller_hmi.human_teleop import HumanTeleopSession

from .test_human_teleop import (
    _StubSideTeleop, _fake_arm_manager, _fast_acquire, _kp_frame,
)


def _wait_until(predicate, timeout: float = 1.0, interval: float = 0.01) -> bool:
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(interval)
    return False


class _StubGuard:
    """Duck-type of collision.CollisionGuard, rigged to visibly rewrite."""

    PAN = 7.7

    def __init__(self, *, explode: bool = False):
        self.cfg = SimpleNamespace(margin_m=0.025)
        self.filter_calls: list = []
        self.explode = explode

    def filter_step(self, prev, want):
        if self.explode:
            raise RuntimeError("guard exploded")
        self.filter_calls.append((prev, want))
        poses = {arm: dict(pose) for arm, pose in want.items()}
        for pose in poses.values():
            if "shoulder_pan" in pose:
                pose["shoulder_pan"] = self.PAN
        return GuardResult(
            poses=poses, alpha=0.5, limited=True,
            clearance=Clearance(slack=-0.011, worst="left:hand|right:hand"),
        )

    def clearance(self, poses):
        return Clearance(slack=0.123, worst="none")


def test_driving_commits_go_through_the_guard():
    mgr, arms = _fake_arm_manager()
    guard = _StubGuard()
    sess = HumanTeleopSession(mgr, collision_guard=guard,
                              side_teleop_factory=_StubSideTeleop,
                              **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(lambda: bool(guard.filter_calls))
        assert _wait_until(lambda: arms["left"].send_goal.called)
        sent = arms["left"].send_goal.call_args[0][0]
        assert sent["shoulder_pan"] == guard.PAN
        status = sess.status()
        assert status["collision"]["enabled"] is True
        assert status["collision"]["limited"] is True
        assert status["collision"]["slack_m"] == -0.011
        assert status["collision"]["worst"] == "left:hand|right:hand"
        assert status["collision"]["margin_m"] == 0.025
        assert (status["joints"]["left"]["shoulder_pan"]["reason"]
                == "collision")
    finally:
        sess.stop()


def test_idle_session_still_publishes_live_clearance():
    """The operator pre-positions against this number BEFORE engaging; it must
    not wait for the first handover to exist."""
    mgr, _arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, collision_guard=_StubGuard(),
                              side_teleop_factory=_StubSideTeleop,
                              **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        assert _wait_until(
            lambda: sess.status()["collision"].get("slack_m") == 0.123)
        assert sess.status()["collision"]["limited"] is False
    finally:
        sess.stop()


def test_no_guard_reports_disabled_and_writes_unfiltered():
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, side_teleop_factory=_StubSideTeleop,
                              **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        # `available` joined `enabled` when the guard gained a runtime
        # switch: a UI needs to tell "off, flip it back on" apart from "this
        # rig has no mount geometry, the switch does nothing".
        assert sess.status()["collision"] == {"enabled": False,
                                              "available": False}
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(lambda: arms["left"].send_goal.called)
        sent = arms["left"].send_goal.call_args[0][0]
        assert sent.get("shoulder_pan") != _StubGuard.PAN
    finally:
        sess.stop()


def test_a_crashing_guard_fails_safe_not_silent():
    """An exception inside the guard must stop that tick's writes (the loop's
    existing catch) and surface in last_error — never crash the thread."""
    mgr, arms = _fake_arm_manager()
    sess = HumanTeleopSession(mgr, collision_guard=_StubGuard(explode=True),
                              side_teleop_factory=_StubSideTeleop,
                              **_fast_acquire(hz_override=200.0))
    sess.start(left_arm="left", right_arm="right")
    try:
        sess.ingest_frame(_kp_frame(dead_man=True))
        assert _wait_until(
            lambda: "guard exploded" in (sess.status()["last_error"] or ""))
        assert not arms["left"].send_goal.called
        assert sess.status()["running"] is True
    finally:
        sess.stop()
