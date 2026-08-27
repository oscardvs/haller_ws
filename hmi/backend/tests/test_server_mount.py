"""The wiring in `_lifespan` that no behavioural test can see.

`IdleSampler` itself is covered (test_tick.py, test_human_teleop_tick.py). What
is NOT covered by any of that is whether `server.py` actually MOUNTS it — and
the failure is silent in the worst direction: delete the four lines and every
sampler test stays green while `measured_hz()` is None on a fresh backend, so
arming refuses forever under invariant 10.

Source-level on purpose, and for the same reason the rollout child greps its own
source: a wiring fact is not a behaviour you can assert on without standing up
the whole app, and it is exactly what a future edit breaks QUIETLY. Driving the
real `_lifespan` needs a sim config and a mujoco context — done by hand on
config.bimanual-sim (measured_hz None for ~0.98 s / 30 samples, then 29.9), but
too slow and too environment-bound to sit in the default suite.
"""
from __future__ import annotations

import inspect

from haller_hmi import server


def _lifespan_source() -> str:
    return inspect.getsource(server._lifespan)


def _lifespan_code() -> str:
    """`_lifespan` with comments stripped.

    Ordering assertions MUST use this. The first version of
    `test_the_idle_sampler_stops_before_the_arms_are_disconnected` indexed the
    raw source and failed against correct code, because the comment explaining
    the constraint says "BEFORE arms.disconnect_all()" and therefore matched
    ~300 chars ahead of the call. A test that greps prose is measuring the
    documentation, and here the documentation was written to describe the very
    ordering under test — so the better the comment, the more wrong the test.
    """
    out = []
    for line in inspect.getsource(server._lifespan).splitlines():
        code = line.split("#", 1)[0]
        if code.strip():
            out.append(code)
    return "\n".join(out)


def test_the_lifespan_mounts_the_idle_sampler():
    """Without this the tick has no producer while idle, and arming refuses.

    `recorder._freeze_fps` RAISES when `rate_detail()` is None, so an unmounted
    sampler fails CLOSED — arming is impossible rather than silently permitted.
    Safe, and completely broken.

    (This cited `rate_ok` until 2026-08-27. The conclusion was right and the
    mechanism named was not: `rate_ok` had no production caller at all, and is
    now deleted outright at `3ae8320`. Caught by grepping AFTER a deletion, twice
    over — the compiler sees neither citation, and this one outlived its subject.)
    """
    src = _lifespan_source()
    assert "IdleSampler(" in src
    assert "idle_sampler.start()" in src
    assert "human_teleop.idle_sample" in src, "the sampler must read the session's sampler"


def test_telemetry_is_wired_to_the_same_bus_the_session_owns():
    """One moment, published once. A second bus is two moments again."""
    src = _lifespan_source()
    assert "tick_bus=human_teleop.tick_bus" in src


def test_the_base_source_is_set_before_any_sample_is_taken():
    """`base` is a field of every sample; unset, every tick publishes it empty."""
    src = _lifespan_code()          # ordering assertion — must ignore comments
    assert "human_teleop.set_base_source(ros)" in src
    assert src.index("set_base_source") < src.index("IdleSampler("), \
        "the source must be attached before the sampler starts reading"


def test_the_idle_sampler_stops_before_the_arms_are_disconnected():
    """ORDERING, and the only one that bites.

    The sampler is a daemon thread doing per-arm reads. Disconnecting underneath
    it throws on every remaining tick — caught and logged rather than fatal, so
    the cost is a shutdown full of noise that reads as a fault. Pinned as an
    ordering fact because both calls are present either way: this cannot be
    caught by asserting that each exists.
    """
    src = _lifespan_code()          # comments stripped — see _lifespan_code
    stop_at = src.index("idle_sampler.stop()")
    disconnect_at = src.index("arms.disconnect_all()")
    assert stop_at < disconnect_at, "idle_sampler.stop() must precede arms.disconnect_all()"


def test_the_sampler_is_stopped_at_all():
    """A started daemon thread with no stop outlives the backend it belongs to."""
    assert "idle_sampler.stop()" in _lifespan_source()
