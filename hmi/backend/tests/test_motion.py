import threading
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

    assert len(h.sent) < len(wps), "ramp should have stopped early"
    assert ex.is_running is False


def test_cancel_stops_the_ramp():
    h = _fake_handle()
    ex = MoveExecutor(h)
    ex.run([{"a": float(i)} for i in range(200)], hz=200.0)
    time.sleep(0.05)
    ex.cancel()

    assert ex.is_running is False


def test_a_new_run_cancels_the_one_in_flight():
    h = _fake_handle()
    ex = MoveExecutor(h)
    ex.run([{"a": float(i)} for i in range(200)], hz=200.0)
    time.sleep(0.02)
    ex.run([{"b": 1.0}], hz=200.0)
    ex.wait(timeout=5.0)

    assert h.sent[-1] == {"b": 1.0}
