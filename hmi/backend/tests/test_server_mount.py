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

import ast
import inspect
import textwrap

import pytest
from haller_hmi import server
from haller_hmi.recorder import DatasetRecorder
from haller_hmi.telemetry import TelemetryBroadcaster

#: Everything in `_lifespan` that must be handed the session's bus. A consumer
#: built without one does not fail loudly — it reads an empty world forever.
BUS_CONSUMERS = {"TelemetryBroadcaster", "DatasetRecorder"}


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
    """One moment, published once. A second bus is two moments again.

    KEPT, and deliberately NOT the whole guarantee — see
    `test_every_bus_consumer_is_wired_to_the_session_bus` below, which exists
    because this assertion cannot say WHICH call site got the argument.
    """
    src = _lifespan_source()
    assert "tick_bus=human_teleop.tick_bus" in src


def test_every_bus_consumer_is_wired_to_the_session_bus():
    """Per CALL, because a substring over `_lifespan` cannot say which one.

    THIS DEFECT WAS LIVE, from 2c until 2d. `DatasetRecorder` was constructed
    without `tick_bus=`, so `recorder.tick_bus` was None on every real backend,
    `_freeze_fps` raised "no tick bus: fps cannot be measured", and **every
    `/record/start` 409'd — recording was dead the whole time.** `fps_measured`
    was permanently null with it, so `_check_rate` returned early and the rate
    alert could never fire either.

    The test above stayed green throughout, and could not have done otherwise:
    it looks for `"tick_bus=human_teleop.tick_bus"` anywhere in the function,
    and TELEMETRY satisfies it at the first of three call sites. A presence
    check over a whole function is satisfied by its easiest instance — which is
    the same shape as an absence assertion in a teardown-on-success path, one
    level up: the harness, not the code, decided the outcome.

    Parsed rather than grepped, so a `tick_bus` in a comment or a neighbouring
    call cannot satisfy it, and asserted per constructor rather than in
    aggregate.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(server._lifespan)))
    built: dict[str, list[set[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in BUS_CONSUMERS:
            built.setdefault(name, []).append({k.arg for k in node.keywords})

    # A loop over zero matches passes every assertion inside it, so pin that we
    # FOUND them before judging them: a rename would otherwise silently empty
    # this test rather than fail it.
    assert set(built) == BUS_CONSUMERS, (
        f"expected to find {sorted(BUS_CONSUMERS)} constructed in _lifespan, "
        f"found {sorted(built)} — a rename empties this test rather than "
        f"failing it, so the mismatch is the finding")

    for name, calls in sorted(built.items()):
        for kw in calls:
            assert "tick_bus" in kw, (
                f"{name} is constructed without tick_bus — it will read an "
                f"empty world, and for the recorder that means every take 409s")


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


def _bus_consumer_calls():
    """Every `_lifespan` call that constructs a bus consumer, as an AST node."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(server._lifespan)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in BUS_CONSUMERS:
            yield name, node


def test_every_bus_consumer_call_satisfies_its_constructor_signature():
    """A missing REQUIRED argument is a boot-time `TypeError` the suite cannot see.

    Nothing in this file constructs `_lifespan` — driving it needs a sim config
    and a mujoco context (see the module docstring) — so a call site that omits a
    required parameter raises at startup, the backend never comes up, and **the
    whole suite stays green through it.** Unbootable server, green suite, on a
    branch four sessions pull from.

    That is not hypothetical: `tick_bus` is about to stop being optional on
    `DatasetRecorder`, because 2c deleted the mode `None` stood for and left the
    default behind — an optional parameter is a standing promise that `None` is a
    working mode, and the promise outlived the mode. Making it required is the
    right fix and it is precisely the edit that can strand this file green.

    Binds the real signature against the real call rather than naming arguments,
    so it covers every required parameter either class ever grows, and catches an
    unknown keyword in the same motion.
    """
    classes = {"TelemetryBroadcaster": TelemetryBroadcaster,
               "DatasetRecorder": DatasetRecorder}
    seen = set()
    for name, node in _bus_consumer_calls():
        seen.add(name)
        # Sentinels: only the ARITY and the names matter to `bind`.
        args = [object()] * len(node.args)
        kwargs = {k.arg: object() for k in node.keywords if k.arg is not None}
        try:
            inspect.signature(classes[name]).bind(*args, **kwargs)
        except TypeError as e:
            pytest.fail(f"{name}(...) in _lifespan does not satisfy its own "
                        f"constructor: {e}. The backend raises this at startup "
                        f"and every other test stays green.")
    assert seen == BUS_CONSUMERS, (
        f"expected {sorted(BUS_CONSUMERS)}, found {sorted(seen)} — a rename "
        f"empties this test rather than failing it")
