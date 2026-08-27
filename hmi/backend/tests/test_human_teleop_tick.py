# hmi/backend/tests/test_human_teleop_tick.py
"""The session as the tick's single producer (Phase 2, invariant 8).

The fakes here are spec'd from the REAL `ArmManager` on purpose. The manager
fake in `test_human_teleop.py` is a bare MagicMock, which supports `__iter__`
and yields nothing — so a producer that iterated the manager instead of asking
it for `keys()` samples no arms, publishes empty samples, and every one of
those tests still passes. `ArmManager` has no `__iter__` at all; iterating it
raises `KeyError: unknown arm id 0`. A fake more permissive than production is
the "impossible fixture" rule pointing the other way: it does not invent a
world where the bug cannot exist, it invents one where the bug cannot be seen.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from haller_hmi.arm import ArmManager
from haller_hmi.human_teleop import HumanTeleopSession
from haller_hmi.safety import Mode
from haller_hmi.tick import ProducerConflict

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")


def _arm(arm_id: str, *, pos: float = 0.0, fail: bool = False):
    a = MagicMock(spec_set=[
        "config", "joint_limits_deg", "guard", "torque_enabled",
        "connect", "disconnect", "send_goal", "executor",
        "enable_torque", "disable_torque", "read_joints_deg", "state_snapshot",
    ])
    a.config = MagicMock(id=arm_id)
    a.joint_limits_deg = {j: (-90.0, 90.0) for j in JOINTS}
    a.guard = MagicMock(mode=Mode.MANUAL)
    a.torque_enabled = True
    a.executor.is_running = False
    a.read_joints_deg.return_value = {j: pos for j in JOINTS}
    if fail:
        a.state_snapshot.side_effect = ConnectionError("bus went away")
    else:
        a.state_snapshot.return_value = {
            "mode": "manual", "torque": True,
            "joints": {j: {"pos": pos, "min": -90.0, "max": 90.0,
                           "torque": True, "effort": 0.25} for j in JOINTS},
        }
    return a


def _manager(**arms):
    """A manager fake with EXACTLY ArmManager's surface — no __iter__."""
    mgr = MagicMock(spec=ArmManager)
    mgr.__getitem__.side_effect = lambda k: arms[k]
    mgr.keys.return_value = list(arms.keys())
    mgr.values.return_value = list(arms.values())
    return mgr


def _run_briefly(sess, sub, *, want: int = 3, timeout_s: float = 5.0):
    """Wait for `want` samples. Counts SAMPLES, never elapsed time."""
    deadline = time.monotonic() + timeout_s
    seen = []
    while len(seen) < want and time.monotonic() < deadline:
        seen.extend(sub.drain())
        time.sleep(0.005)
    return seen


def test_the_manager_this_session_samples_is_not_iterable():
    """Pins the reason `_sample_arms` must ask for `keys()`.

    Not a style note: `for arm_id in manager` raises on the first tick and
    goes on raising, so MAX_CONSECUTIVE_TICK_ERRORS stops the session about
    2.5 s after every start. A linter asked for that change and was right
    about dicts and wrong about this receiver.
    """
    assert not hasattr(ArmManager, "__iter__")
    with pytest.raises(KeyError):
        for _ in ArmManager([]):
            break


def test_a_running_session_publishes_the_tick():
    arms = {"left": _arm("left", pos=5.0), "right": _arm("right", pos=-5.0)}
    sess = HumanTeleopSession(_manager(**arms), hz_override=120.0)
    sub = sess.tick_bus.subscribe(name="test")
    sess.start(left_arm="left", right_arm="right")
    try:
        samples = _run_briefly(sess, sub)
    finally:
        sess.stop()
    assert len(samples) >= 3


def test_every_published_sample_carries_every_arm_the_manager_reports():
    """The empty-sample failure is silent, so assert the CONTENT, not the count."""
    arms = {"left": _arm("left", pos=5.0), "right": _arm("right", pos=-5.0)}
    sess = HumanTeleopSession(_manager(**arms), hz_override=120.0)
    sub = sess.tick_bus.subscribe(name="test")
    sess.start(left_arm="left", right_arm="right")
    try:
        samples = _run_briefly(sess, sub)
    finally:
        sess.stop()

    assert samples, "no samples published at all"
    for s in samples:
        assert set(s.arms) == {"left", "right"}
        assert s.arms["left"]["joints_deg"]["shoulder_pan"] == 5.0
        assert s.arms["right"]["joints_deg"]["shoulder_pan"] == -5.0
        assert s.arms["left"]["effort_norm"]["gripper"] == 0.25
        assert s.degraded is False
        assert dict(s.arm_errors) == {}


def test_a_sample_pairs_state_and_goal_from_one_tick():
    """Invariant 8, in the shape the recorder consumes.

    Both halves come off ONE sample, so there is no second instant available
    to pair with — which is the structural fix, not a timing improvement.
    """
    arms = {"left": _arm("left", pos=5.0)}
    sess = HumanTeleopSession(_manager(**arms), hz_override=120.0)
    sub = sess.tick_bus.subscribe(name="test")
    sess.start(left_arm="left", right_arm=None)
    try:
        samples = _run_briefly(sess, sub)
    finally:
        sess.stop()

    s = samples[-1]
    assert set(s.arms["left"]["joints_deg"]) == set(JOINTS)
    assert set(s.goal_deg["left"]) == set(JOINTS)
    assert s.t_mono > 0.0 and s.t_unix > 0.0


def test_an_arm_whose_read_fails_is_a_hole_not_a_number():
    """Invariant 9 at the point the read is taken.

    Mechanism 2's lost race decodes tick 0 to -180.0 deg, so substituting any
    plausible value for a read that did not happen is not a small error.
    """
    arms = {"left": _arm("left", pos=5.0), "right": _arm("right", fail=True)}
    sess = HumanTeleopSession(_manager(**arms), hz_override=120.0)
    sub = sess.tick_bus.subscribe(name="test")
    sess.start(left_arm="left", right_arm=None)
    try:
        samples = _run_briefly(sess, sub)
    finally:
        sess.stop()

    s = samples[-1]
    assert "right" not in s.arms
    assert "right" in s.arm_errors
    assert s.degraded is True
    assert s.arms["left"]["joints_deg"]["shoulder_pan"] == 5.0


def test_the_session_owns_the_bus_only_while_it_runs():
    arms = {"left": _arm("left")}
    sess = HumanTeleopSession(_manager(**arms), hz_override=120.0)
    assert sess.tick_bus.producer_name is None

    sess.start(left_arm="left", right_arm=None)
    try:
        assert sess.tick_bus.producer_name == "human-teleop"
        assert sess.tick_bus.publish_once("idle-sampler", t_mono=0.0) is None
    finally:
        sess.stop()

    assert sess.tick_bus.producer_name is None
    assert sess.tick_bus.publish_once("idle-sampler", t_mono=0.0) is not None


def test_a_session_that_cannot_claim_the_bus_does_not_come_up_half_started():
    """A session marked running with no thread is worse than a failed start."""
    arms = {"left": _arm("left")}
    sess = HumanTeleopSession(_manager(**arms), hz_override=120.0)
    sess.tick_bus.attach_producer("someone-else")

    with pytest.raises(ProducerConflict):
        sess.start(left_arm="left", right_arm=None)
    assert sess.running is False
    assert sess.status()["running"] is False


# --- the sample rate is a RATE ------------------------------------------

@pytest.mark.parametrize(("hz", "sample_hz", "expected"), [
    (60.0, None, 1),      # unset samples every tick
    (60.0, 30.0, 2),
    (30.0, 30.0, 1),      # the same requested rate, a different cadence
    (120.0, 30.0, 4),
    (10.0, 30.0, 1),      # cannot sample faster than the loop runs
    (60.0, 0.0, 1),
    (60.0, -5.0, 1),
])
def test_the_divisor_is_derived_so_the_sample_rate_survives_a_cadence_change(
        hz, sample_hz, expected):
    """A configured divisor would mean 30 Hz at hz=60 and 5 Hz at hz=10.

    `fps` is frozen from the measured sample rate, so a ticks-denominated
    setting would make the recorder's declared frame rate follow the control
    rate — the constant class `_rate_cap_deg_per_tick` was just fixed for.
    """
    assert HumanTeleopSession._sample_divisor(hz, sample_hz) == expected


def test_a_decimating_session_publishes_slower_than_it_commits():
    """The divisor arithmetic, observed on a running loop rather than asserted."""
    arms = {"left": _arm("left")}
    sess = HumanTeleopSession(_manager(**arms), hz_override=120.0, sample_hz=30.0)
    sub = sess.tick_bus.subscribe(name="test")
    sess.start(left_arm="left", right_arm=None)
    try:
        samples = _run_briefly(sess, sub, want=5)
    finally:
        sess.stop()

    assert len(samples) >= 5
    # Consecutive seq numbers: the bus allocates one per PUBLISH, so a
    # decimating producer still yields a gapless sequence. What decimation
    # changes is how many commits happened between them, which is why the
    # count of published samples is the thing worth pinning and the sequence
    # is not.
    seqs = [s.seq for s in samples]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))
