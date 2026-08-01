import time
from unittest.mock import MagicMock

from haller_hmi.motion import MoveExecutor
from haller_hmi.safety import Mode, ModeGuard


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
