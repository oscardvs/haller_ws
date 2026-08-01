import os

# Required before importing anything under haller_hmi.sim (here: sim/teleop.py,
# pulled in transitively by the real-session teleop_owner tests below) —
# mirrors tests/test_session_lock.py and tests/sim/test_sim_arm_handle.py.
os.environ.setdefault("MUJOCO_GL", "egl")

import time
from unittest.mock import MagicMock

import pytest

from haller_hmi.arm import ArmHandle
from haller_hmi.config import MotionConfig
from haller_hmi.human_teleop import HumanTeleopSession
from haller_hmi.motion import MoveExecutor, MoveRefused, home, move_to
from haller_hmi.safety import Mode, ModeGuard
from haller_hmi.sim.arm import SimArmHandle
from haller_hmi.sim.teleop import SimLeaderTeleop
from haller_hmi.teleop import TeleopSession


def _fake_handle():
    h = MagicMock()
    h.guard = ModeGuard(Mode.MANUAL)
    h.config.id = "right"
    h.sent = []
    h.send_goal.side_effect = lambda wp: h.sent.append(wp)
    return h


def test_executor_plays_every_waypoint_in_order():
    h = _fake_handle()
    ex = MoveExecutor(h)
    wps = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]

    ex.run(wps, hz=200.0)
    ex.wait(timeout=5.0)

    assert h.sent == wps
    assert ex.is_running is False


def test_estop_mid_ramp_halts_the_move():
    """/estop sets Mode.STOP on every guard. The executor re-checks the guard
    before each waypoint, so that is the whole cancellation mechanism."""
    h = _fake_handle()
    ex = MoveExecutor(h)
    wps = [{"a": float(i)} for i in range(200)]

    ex.run(wps, hz=200.0)
    time.sleep(0.05)
    h.guard.set(Mode.STOP)
    ex.wait(timeout=5.0)

    # Pins the check granularity, not just "stopped somewhere": at hz=200 a
    # 50 ms sleep lets through ~10-12 waypoints before the guard is re-read.
    # A looser bound (e.g. "< 200") would also pass an implementation that
    # only re-checks the guard every 50 waypoints.
    assert len(h.sent) < 25, "ramp should have stopped within a few waypoints"
    assert ex.is_running is False


def test_cancel_stops_the_ramp():
    h = _fake_handle()
    ex = MoveExecutor(h)
    ex.run([{"a": float(i)} for i in range(200)], hz=200.0)
    time.sleep(0.05)
    ex.cancel()

    assert ex.is_running is False
    # Not just "eventually not running": cancel() must have cut the ramp off,
    # not merely waited for it to finish on its own. Without
    # `self._cancel.set()` in `_cancel_locked`, the surviving `join(timeout=
    # 2.0)` would just wait out the ~1s ramp and this would pass vacuously —
    # confirmed locally by deleting that line (see task-5-report.md).
    assert len(h.sent) <= 20, "cancel() should have cut the ramp short"


def test_a_new_run_cancels_the_one_in_flight():
    h = _fake_handle()
    ex = MoveExecutor(h)
    ex.run([{"a": float(i)} for i in range(200)], hz=200.0)
    time.sleep(0.02)
    ex.run([{"b": 1.0}], hz=200.0)
    ex.wait(timeout=5.0)

    assert h.sent[-1] == {"b": 1.0}
    # `h.sent[-1]` alone only proves runs are serialised (the second run's
    # one waypoint is appended after the first run's last one, whatever that
    # was) — it says nothing about cancellation. Pin that the first ramp was
    # actually cut short rather than left to finish underneath the second.
    assert len(h.sent) <= 20, "the in-flight ramp should have been cancelled"


def test_send_goal_exception_is_recorded_and_stops_the_ramp():
    """A non-ModeError exception (e.g. the rig's documented intermittent UART
    failure surfacing as ConnectionError out of send_goal) must not kill the
    background thread silently: it is recorded on the executor and the ramp
    stops rather than continuing on to the next waypoint."""
    h = _fake_handle()

    def _boom(wp):
        h.sent.append(wp)
        if len(h.sent) == 3:
            raise ConnectionError("comm failure")

    h.send_goal.side_effect = _boom
    ex = MoveExecutor(h)
    wps = [{"a": float(i)} for i in range(200)]

    ex.run(wps, hz=200.0)
    ex.wait(timeout=5.0)

    assert h.sent == wps[:3], "the ramp should have stopped at the failing waypoint"
    assert ex.is_running is False
    assert isinstance(ex.last_error, ConnectionError)


def test_last_error_is_cleared_at_the_start_of_the_next_run():
    h = _fake_handle()
    calls = {"n": 0}

    def _flaky_once(wp):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("comm failure")
        h.sent.append(wp)

    h.send_goal.side_effect = _flaky_once
    ex = MoveExecutor(h)

    ex.run([{"a": 1.0}], hz=200.0)
    ex.wait(timeout=5.0)
    assert isinstance(ex.last_error, ConnectionError)

    ex.run([{"a": 2.0}], hz=200.0)
    ex.wait(timeout=5.0)

    assert ex.last_error is None
    assert h.sent == [{"a": 2.0}]


# ---- move_to / home: the shared bounded-move policy ------------------------


def _movable_handle(current, limits=None):
    h = _fake_handle()
    h.torque_enabled = True
    h.motion = MotionConfig(max_speed_deg_s=60.0, large_move_deg=30.0, ramp_hz=50.0)
    h.joint_limits_deg = limits or {j: (-180.0, 180.0) for j in current}
    h.read_joints_deg.return_value = current
    h.executor = MoveExecutor(h)
    return h


def test_move_to_refuses_when_any_joint_exceeds_the_threshold():
    h = _movable_handle({"shoulder_pan": 0.0, "gripper": 0.0})
    with pytest.raises(MoveRefused) as e:
        move_to(h, {"shoulder_pan": 90.0, "gripper": 1.0})
    assert "shoulder_pan" in str(e.value)
    assert "gripper" not in str(e.value)
    assert h.sent == [], "nothing may be commanded when a move is refused"


def test_home_refuses_right_after_a_recalibration():
    """The 2026-08-01 incident. Calibration redefines 0 deg, leaving the arm far
    from it; Home then slewed the arm across the bench. It must refuse."""
    h = _movable_handle({"shoulder_pan": -126.5, "wrist_flex": 148.9})
    with pytest.raises(MoveRefused):
        home(h)
    assert h.sent == []


def test_move_to_ramps_a_small_move():
    h = _movable_handle({"shoulder_pan": 0.0})
    move_to(h, {"shoulder_pan": 10.0})
    h.executor.wait(timeout=5.0)
    assert h.sent, "a small move should have been commanded"
    assert h.sent[-1]["shoulder_pan"] == pytest.approx(10.0)


def test_move_to_refuses_when_torque_is_disabled():
    h = _movable_handle({"shoulder_pan": 0.0})
    h.torque_enabled = False
    with pytest.raises(MoveRefused) as e:
        move_to(h, {"shoulder_pan": 1.0})
    assert "torque" in str(e.value).lower()
    assert h.sent == []


def test_move_to_refuses_when_a_commanded_joint_has_no_current_reading():
    """read_joints_deg drops a joint whose .pos was missing from the
    observation, which this rig's UART does intermittently. Ramping the rest
    would command a partial move and report it as complete."""
    h = _movable_handle({"shoulder_pan": 0.0})
    h.joint_limits_deg = {"shoulder_pan": (-180.0, 180.0),
                          "wrist_flex": (-180.0, 180.0)}
    h.read_joints_deg.return_value = {"shoulder_pan": 0.0}  # wrist_flex dropped

    with pytest.raises(MoveRefused) as e:
        move_to(h, {"shoulder_pan": 1.0, "wrist_flex": 1.0})

    assert "wrist_flex" in str(e.value)
    assert h.sent == []


# ---- move_to vs. a running teleop session -----------------------------
#
# Task 5's review: POST /arm/{id}/home during an active teleop session starts
# a ramp thread that writes Goal_Position on the same serial port a 60 Hz
# teleop loop is already streaming to — there is no lock anywhere in lerobot,
# and the overlap lasts the whole ramp, not a moment. `_fake_handle` names its
# arm "right" (see the top of this file); TeleopSession/HumanTeleopSession/
# SimLeaderTeleop all report which arm(s) they own through a `status()` dict
# ("follower/leader" or "left_arm/right_arm"), so a stand-in with the same
# shape is enough to exercise `MoveExecutor.teleop_owner` without importing
# any of the three.

def test_move_to_refuses_when_a_teleop_session_owns_the_arm():
    h = _movable_handle({"shoulder_pan": 0.0})
    peer = MagicMock()
    peer.status.return_value = {"running": True, "follower": "right"}
    h.executor.attach_peer(peer)

    with pytest.raises(MoveRefused) as e:
        move_to(h, {"shoulder_pan": 10.0})

    assert "right" in str(e.value)
    assert "teleop" in str(e.value).lower()
    assert h.sent == []


def test_move_to_ignores_a_teleop_session_that_owns_a_different_arm():
    """The check is scoped to the arm being moved, not "is any teleop running
    anywhere": an idle arm must stay movable while a teleop session drives its
    sibling (e.g. SimLeaderTeleop, which only ever occupies its follower)."""
    h = _movable_handle({"shoulder_pan": 0.0})
    peer = MagicMock()
    peer.status.return_value = {"running": True, "follower": "left"}
    h.executor.attach_peer(peer)

    move_to(h, {"shoulder_pan": 10.0})
    h.executor.wait(timeout=5.0)

    assert h.sent, "a move on an arm no running peer owns must not be refused"


def test_move_to_is_not_blocked_by_a_peer_that_is_not_running():
    h = _movable_handle({"shoulder_pan": 0.0})
    peer = MagicMock()
    peer.status.return_value = {"running": False, "follower": "right"}
    h.executor.attach_peer(peer)

    move_to(h, {"shoulder_pan": 10.0})
    h.executor.wait(timeout=5.0)

    assert h.sent, "a stopped peer must not block a move"


def test_home_is_not_reintroduced_as_a_handle_method():
    """Task 6 deleted the duplicated home() on both handles in favour of the
    shared motion.home(handle) — the two per-class copies being wrong in the
    same way is exactly what let the 2026-08-01 incident happen in sim too.
    If either copy comes back, fail loudly here rather than the two handles
    silently diverging again."""
    assert not hasattr(ArmHandle, "home")
    assert not hasattr(SimArmHandle, "home")


# ---- teleop_owner vs. the real session classes --------------------------
#
# Every test above authors its own MagicMock `status()` dict, which pins
# nothing against the three real session classes: if TeleopState.to_dict()
# (or HumanTeleopSession.status()/SimLeaderTeleop.status()) ever renamed
# "follower", `teleop_owner` would return None for every arm forever, and
# every test above would keep passing regardless. These start a REAL session
# of each kind and read its REAL status() — the same construction pattern
# tests/test_session_lock.py uses for the existing three-way attach_peer
# lock, just started for real instead of having `.status` monkey-patched.

def _two_arm_manager():
    """ArmManager-shaped stand-in with two arms ("left", "right"), realistic
    enough that a real TeleopSession/HumanTeleopSession/SimLeaderTeleop can
    start() and tick without raising. Deliberately not spec_set: this helper
    exists to drive the *real* status() of each session below, not to police
    which handle methods a session may call (that's test_human_teleop.py's
    _fake_arm_manager, elsewhere)."""
    mgr = MagicMock()

    def _mkarm(arm_id):
        a = MagicMock()
        a.config.id = arm_id
        a.joint_limits_deg = {"shoulder_pan": (-90.0, 90.0)}
        a.guard = ModeGuard(Mode.MANUAL)
        a.torque_enabled = True
        a.read_joints_deg.return_value = {"shoulder_pan": 0.0}
        # A bare Mock's .executor.is_running is a truthy Mock, which would
        # make every session.start() below refuse immediately ("a move in
        # progress") before ever reaching the real status()/teleop_owner
        # behaviour these tests exist to exercise. See A6 obligation 2.
        a.executor.is_running = False
        return a

    arms = {"left": _mkarm("left"), "right": _mkarm("right")}
    mgr.__getitem__.side_effect = lambda k: arms[k]
    mgr.values.return_value = list(arms.values())
    return mgr


def test_teleop_owner_reads_a_real_running_teleop_session():
    mgr = _two_arm_manager()
    session = TeleopSession(mgr)
    session.start(leader_id="left", follower_id="right", hz=200.0)
    try:
        ex = MoveExecutor(_fake_handle())
        ex.attach_peer(session)
        assert ex.teleop_owner("right") == "TeleopSession"
        # The leader is owned too, not just the follower: its guard is set to
        # Mode.STOP (see teleop.py), but teleop_owner doesn't special-case
        # that — it reports ownership from status() alone.
        assert ex.teleop_owner("left") == "TeleopSession"
        assert ex.teleop_owner("spare") is None
    finally:
        session.stop()


def test_teleop_owner_reads_a_real_running_human_teleop_session():
    mgr = _two_arm_manager()
    session = HumanTeleopSession(mgr)
    session.start(left_arm="left", right_arm="right", swap=False)
    try:
        ex = MoveExecutor(_fake_handle())
        ex.attach_peer(session)
        assert ex.teleop_owner("left") == "HumanTeleopSession"
        assert ex.teleop_owner("right") == "HumanTeleopSession"
        assert ex.teleop_owner("spare") is None
    finally:
        session.stop()


def test_teleop_owner_reads_a_real_running_sim_leader_teleop():
    """SimLeaderTeleop's "leader" is a synthetic mouse/replay source, not an
    arm, so a running session only ever occupies its follower. This confirms
    that against the real status(), not just the hand-authored fake in
    test_move_to_ignores_a_teleop_session_that_owns_a_different_arm above."""
    mgr = _two_arm_manager()
    source = MagicMock()
    source.read.return_value = {}
    session = SimLeaderTeleop(mgr)
    session.start(follower_id="right", source=source, hz=200.0)
    try:
        ex = MoveExecutor(_fake_handle())
        ex.attach_peer(session)
        assert ex.teleop_owner("right") == "SimLeaderTeleop"
        assert ex.teleop_owner("left") is None
        assert ex.teleop_owner("spare") is None
    finally:
        session.stop()


# ---- the other direction: a ramp must block teleop from starting ----------
#
# move_to refuses when a teleop session already owns the arm (above). A6
# obligation 2 closes the reverse gap: Home, then start teleop before the
# ramp finishes. TeleopSession's follower and HumanTeleopSession's two arms
# are commanded into Mode.MANUAL by start() itself — exactly the mode the
# ramp thread's own guard check lets through — so an in-flight ramp would not
# self-cancel; two threads would write Goal_Position to the same serial port
# with no lock anywhere in lerobot, for the whole ramp. See teleop.py,
# human_teleop.py and sim/teleop.py's `executor.is_running` checks.

def _long_running_executor(handle) -> MoveExecutor:
    """A MoveExecutor mid-ramp on `handle`, for testing the refusal itself —
    not a race against a real serial write, which test_estop_mid_ramp_halts_
    the_move and friends already cover."""
    ex = MoveExecutor(handle)
    ex.run([{"shoulder_pan": float(i)} for i in range(200)], hz=200.0)
    return ex


def test_teleop_session_start_refuses_while_a_ramp_is_in_flight():
    mgr = _two_arm_manager()
    follower = mgr["right"]
    follower.executor = _long_running_executor(follower)
    try:
        session = TeleopSession(mgr)
        with pytest.raises(RuntimeError) as e:
            session.start(leader_id="left", follower_id="right", hz=60.0)
        assert "right" in str(e.value)
    finally:
        follower.executor.cancel()


def test_human_teleop_session_start_refuses_while_a_ramp_is_in_flight():
    mgr = _two_arm_manager()
    right = mgr["right"]
    right.executor = _long_running_executor(right)
    try:
        session = HumanTeleopSession(mgr)
        with pytest.raises(RuntimeError) as e:
            session.start(left_arm="left", right_arm="right", swap=False)
        assert "right" in str(e.value)
    finally:
        right.executor.cancel()


def test_sim_leader_teleop_start_refuses_while_a_ramp_is_in_flight():
    mgr = _two_arm_manager()
    follower = mgr["right"]
    follower.executor = _long_running_executor(follower)
    try:
        session = SimLeaderTeleop(mgr)
        source = MagicMock()
        with pytest.raises(RuntimeError) as e:
            session.start(follower_id="right", source=source, hz=60.0)
        assert "right" in str(e.value)
    finally:
        follower.executor.cancel()
