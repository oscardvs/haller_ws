# hmi/backend/tests/test_tick.py
"""Claims made by `haller_hmi.tick`, asserted in the claims' own terms.

Written from the contract (docs/port/phase2-tick-contract.md) before reading
the implementation back, per the port's counter-discipline: a test written
from the code inherits the code's blind spot.

The clock is INJECTED throughout. Every rate assertion here passes an explicit
`t_mono`, so none of them measures this box — four concurrent sessions share
it, and a wall-clock rate assertion is the shape that goes red for reasons
that have nothing to do with the code. The one thing a clock-injected test
cannot check is that the producer passes a real clock; that is pinned where
the producer is wired, not here.
"""
from __future__ import annotations

import asyncio

import pytest

from haller_hmi.tick import (
    RATE_MIN_SAMPLES,
    IdleSampler,
    ProducerConflict,
    TickBus,
    TickSample,
    plain,
)


def _publish(token, *, seq_hint: int = 0, t: float | None = None, **kw):
    return token.publish(t_mono=t if t is not None else float(seq_hint), **kw)


# --- one moment, and it cannot change afterwards -------------------------

def test_two_consumers_of_one_publish_see_the_same_moment():
    """The whole point: state and action from ONE instant, for everyone."""
    bus = TickBus()
    a = bus.subscribe(name="a")
    b = bus.subscribe(name="b")
    token = bus.attach_producer("test")
    _publish(token, arms={"left": {"joints_deg": {"j": 1.0}}},
             goal_deg={"left": {"j": 2.0}})

    got_a = a.drain()
    got_b = b.drain()
    assert len(got_a) == len(got_b) == 1
    assert got_a[0].seq == got_b[0].seq
    assert got_a[0].arms["left"]["joints_deg"]["j"] == 1.0
    assert got_b[0].goal_deg["left"]["j"] == 2.0


def test_a_consumer_cannot_mutate_a_delivered_sample():
    bus = TickBus()
    sub = bus.subscribe(name="a")
    token = bus.attach_producer("test")
    _publish(token, arms={"left": {"joints_deg": {"j": 1.0}}})
    sample = sub.drain()[0]

    with pytest.raises(TypeError):
        sample.arms["left"]["joints_deg"]["j"] = 99.0
    with pytest.raises(AttributeError):
        sample.seq = 99


def test_the_producer_mutating_its_own_dict_after_publish_cannot_reach_a_consumer():
    """The half that a wrap-without-copy would fail, looking identical.

    `MappingProxyType` over a dict the producer still owns is a read-only VIEW
    of a mutable object: the consumer above still cannot write through it, so
    the previous test passes either way. This one is the reason `_freeze`
    copies. The teleop loop reuses its dicts every tick, so without the copy a
    consumer holding "one moment" would silently be holding the CURRENT one.
    """
    bus = TickBus()
    sub = bus.subscribe(name="a")
    token = bus.attach_producer("test")

    live = {"j": 1.0}
    _publish(token, goal_deg={"left": live})
    live["j"] = 99.0          # the producer's next tick, before the consumer reads

    assert sub.drain()[0].goal_deg["left"]["j"] == 1.0


# --- drops are counted, and the arithmetic closes -------------------------

def test_a_full_subscriber_drops_the_oldest_and_counts_every_drop():
    """Invariant 9 turns on telling a lost tick from one that never happened."""
    bus = TickBus()
    sub = bus.subscribe(name="slow", maxsize=3)
    token = bus.attach_producer("test")
    for i in range(10):
        _publish(token, seq_hint=i, goal_deg={"left": {"j": float(i)}})

    queued = sub.drain()
    assert [s.goal_deg["left"]["j"] for s in queued] == [7.0, 8.0, 9.0]
    assert sub.dropped == 7
    assert sub.delivered == 10
    assert sub.delivered == len(queued) + sub.dropped


def test_a_draining_subscriber_does_not_manufacture_drops():
    """Contract C2 — `skipped_frames` must not climb while parked in ARMED.

    An operator who learns to ignore that number while parked ignores it
    mid-take, which is the one moment it has to be trusted.
    """
    bus = TickBus()
    sub = bus.subscribe(name="armed", maxsize=2)
    sub.set_draining(True)
    token = bus.attach_producer("test")
    for i in range(50):
        _publish(token, seq_hint=i)

    assert sub.dropped == 0
    assert sub.pending == 0
    assert sub.delivered == 50


def test_decimating_with_latest_discards_without_counting_a_drop():
    """A consumer declining samples it never wanted is not a consumer losing them."""
    bus = TickBus()
    sub = bus.subscribe(name="telemetry", maxsize=64)
    token = bus.attach_producer("test")
    for i in range(10):
        _publish(token, seq_hint=i, goal_deg={"left": {"j": float(i)}})

    newest = sub.latest()
    assert newest is not None and newest.goal_deg["left"]["j"] == 9.0
    assert sub.pending == 0
    assert sub.dropped == 0


def test_a_full_subscriber_never_blocks_the_producer():
    bus = TickBus()
    bus.subscribe(name="stalled", maxsize=1)
    token = bus.attach_producer("test")
    for i in range(500):
        _publish(token, seq_hint=i)
    assert bus.published == 500


# --- exactly one producer -------------------------------------------------

def test_a_second_producer_is_refused_while_one_is_attached():
    bus = TickBus()
    bus.attach_producer("idle-sampler")
    with pytest.raises(ProducerConflict):
        bus.attach_producer("session")


def test_the_handover_is_detach_then_attach_and_seq_never_repeats():
    """Two producers across a handover cannot land on one `seq`.

    `seq` is allocated by the BUS, so this holds by construction rather than
    by the producers cooperating — which is the point. See
    `TickBus.attach_producer` for why exclusivity is still enforced: the cost
    of an overlap is a doubled `measured_hz`, not a duplicate seq, and a
    doubled rate is what gets frozen into `info.json` as fps.
    """
    bus = TickBus()
    sub = bus.subscribe(name="a")

    idle = bus.attach_producer("idle-sampler")
    _publish(idle, seq_hint=0)
    _publish(idle, seq_hint=1)
    idle.detach()

    session = bus.attach_producer("session")
    _publish(session, seq_hint=2)
    session.detach()

    seqs = [s.seq for s in sub.drain()]
    assert seqs == [0, 1, 2]
    assert len(seqs) == len(set(seqs))


def test_a_detached_producer_cannot_still_publish():
    bus = TickBus()
    token = bus.attach_producer("session")
    token.detach()
    with pytest.raises(ProducerConflict):
        _publish(token)


def test_the_producer_token_releases_on_leaving_its_block():
    bus = TickBus()
    with bus.attach_producer("session"):
        assert bus.producer_name == "session"
    assert bus.producer_name is None
    bus.attach_producer("idle-sampler")   # would raise if the release leaked


# --- the measured rate ----------------------------------------------------

def test_the_rate_is_unknown_rather_than_guessed_below_the_sample_floor():
    """Invariant 10: fps is measured, or the episode does not open.

    Mechanism 3 was a plausible number standing in for a measured one, so the
    honest report while we do not know is None — never an extrapolation from
    two samples.
    """
    bus = TickBus()
    token = bus.attach_producer("test")
    for i in range(RATE_MIN_SAMPLES - 1):
        _publish(token, t=i * 0.01)
    assert bus.measured_hz() is None

    _publish(token, t=(RATE_MIN_SAMPLES - 1) * 0.01)
    assert bus.measured_hz() is not None


def test_the_measured_rate_is_the_rate_the_samples_actually_arrived_at():
    bus = TickBus()
    token = bus.attach_producer("test")
    for i in range(60):
        _publish(token, t=i * (1.0 / 30.0))       # exactly 30 Hz
    assert bus.measured_hz() == pytest.approx(30.0, rel=1e-9)


def test_a_producer_running_slow_reports_slow_rather_than_its_intention():
    """The defect in one line: the old fps was the rate telemetry was ASKED for."""
    bus = TickBus()
    token = bus.attach_producer("test")
    for i in range(60):
        _publish(token, t=i * (1.0 / 4.8))        # a 4.8 Hz reality
    measured = bus.measured_hz()
    assert measured == pytest.approx(4.8, rel=1e-9)


def test_a_stalled_window_reports_None_rather_than_dividing_by_its_own_span():
    """The bus says "I do not know" instead of producing a number, and that
    is what makes `recorder._freeze_fps` fail CLOSED: it RAISES on a None
    `rate_detail()`, so an unmounted or stalled sampler cannot open an
    episode. Invariant 10 from the reporting side.

    Pointed at the state only the SPAN guard can rescue. `rate_detail` has two
    ways to answer None — below `RATE_MIN_SAMPLES`, and a non-positive window
    — and the test above already pins the sample floor at its boundary. A
    handful of publishes trips BOTH, so it would pass with either guard
    deleted and has therefore tested neither. A FULL window stamped at one
    instant clears the floor and leaves the span guard as the only thing
    between the caller and `(n - 1) / 0`.

    Not a contrived input: `t_mono` comes from the producer, and a producer
    whose clock has stopped advancing is exactly the stall this reports.
    """
    bus = TickBus()
    token = bus.attach_producer("test")
    for _ in range(RATE_MIN_SAMPLES + 10):
        _publish(token, t=0.0)                    # a window with no span at all
    assert bus.rate_detail() is None
    assert bus.measured_hz() is None


def test_tick_does_not_define_its_own_rate_fraction():
    from haller_hmi import tick
    assert not any("RATE_FRACTION" in n for n in vars(tick))


def test_reset_rate_drops_the_window_for_a_producer_that_did_not_change():
    """The case the AUTOMATIC reset cannot see: one producer, a new cadence.

    The auto-reset keys on the producer's NAME changing, so two consecutive
    sessions — same name, different `hz` — would share one window and freeze
    an fps that describes neither. The idle sampler running between them
    happens to cover this today, but that is a coincidence of wiring rather
    than a guarantee, so the session clears the window itself at start.

    This test replaced one that called `reset_rate()` across a producer
    CHANGE. That version went on passing with the method's body deleted,
    because the auto-reset was doing the work — a live assertion that had
    quietly stopped pinning the thing it named. The mutation pass found it;
    reading it would not have.
    """
    bus = TickBus()
    token = bus.attach_producer("session")
    for i in range(60):
        _publish(token, t=i * (1.0 / 20.0))
    assert bus.measured_hz() == pytest.approx(20.0, rel=1e-9)

    bus.reset_rate()
    assert bus.measured_hz() is None, "the previous cadence survived the reset"

    for i in range(RATE_MIN_SAMPLES):
        _publish(token, t=100.0 + i * (1.0 / 60.0))
    assert bus.measured_hz() == pytest.approx(60.0, rel=1e-9)


# --- asyncio delivery -----------------------------------------------------

async def test_an_async_consumer_is_woken_by_a_publish_from_a_plain_thread():
    """The producer is a plain thread; the recorder lives on the event loop.

    The consumer is parked in the await BEFORE the thread publishes, and that
    ordering is asserted rather than hoped for. Written the obvious way round
    — start the thread, then await — this test passes whenever the thread wins
    the race, because the sample is already queued and the wakeup is never
    exercised. It would then hold identically with `call_soon_threadsafe`
    deleted: an assertion that passes for a reason other than the one claimed.
    """
    import threading

    bus = TickBus()
    loop = asyncio.get_running_loop()
    sub = bus.subscribe(name="recorder", loop=loop)
    token = bus.attach_producer("session")

    getter = asyncio.create_task(sub.get())
    await asyncio.sleep(0)          # let it reach the await
    assert not getter.done(), "consumer must be parked, or this proves nothing"

    threading.Thread(
        target=lambda: token.publish(t_mono=0.0, goal_deg={"left": {"j": 7.0}}),
        daemon=True,
    ).start()

    sample = await asyncio.wait_for(getter, timeout=2.0)
    assert sample is not None
    assert sample.goal_deg["left"]["j"] == 7.0


async def test_an_async_consumer_gets_every_sample_in_order():
    bus = TickBus()
    sub = bus.subscribe(name="recorder", loop=asyncio.get_running_loop())
    token = bus.attach_producer("session")
    for i in range(5):
        _publish(token, seq_hint=i, goal_deg={"left": {"j": float(i)}})

    seen = [(await sub.get()).goal_deg["left"]["j"] for _ in range(5)]
    assert seen == [0.0, 1.0, 2.0, 3.0, 4.0]


async def test_a_closed_subscription_stops_the_consumer_rather_than_hanging():
    bus = TickBus()
    sub = bus.subscribe(name="recorder", loop=asyncio.get_running_loop())
    sub.close()
    assert await asyncio.wait_for(sub.get(), timeout=2.0) is None
    assert bus.subscriber_count == 0


def test_drain_and_latest_work_without_an_event_loop():
    """A plain-thread consumer must not need asyncio to read the bus."""
    bus = TickBus()
    sub = bus.subscribe(name="thread-consumer")
    token = bus.attach_producer("test")
    _publish(token, seq_hint=0)
    assert len(sub.drain()) == 1

    with pytest.raises(RuntimeError):
        asyncio.run(_await_get(sub))


async def _await_get(sub):
    return await sub.get()


# --- the sample itself ----------------------------------------------------

def test_a_sample_carries_both_clocks_read_in_one_breath():
    """`observation.wall_clock` needs an epoch; a recorder calling time.time()
    itself would be a third instant — mechanism 1 through the timestamp column.
    """
    sample = TickSample(seq=0, t_mono=1.0, t_unix=2.0)
    assert sample.t_mono == 1.0
    assert sample.t_unix == 2.0


def test_a_sample_defaults_to_not_degraded():
    """`degraded` is declared here and CONSUMED where the arms are read.

    This test pins the default only. It deliberately does not assert that a
    degraded read sets it — nothing in this module can produce one, and an
    assertion about a flag no producer sets yet would pass however the
    producer is later written. The real pin belongs with the producer, and
    invariant 9 is where it is owed.
    """
    assert TickSample(seq=0, t_mono=0.0, t_unix=0.0).degraded is False


# --- the producer handover ------------------------------------------------

def test_publish_once_publishes_while_the_bus_is_free():
    bus = TickBus()
    sub = bus.subscribe(name="a")
    assert bus.publish_once("idle-sampler", t_mono=0.0) is not None
    assert len(sub.drain()) == 1


def test_the_idle_sampler_is_silent_while_a_session_holds_the_bus():
    """The handover cannot overlap, so nothing has to stand the sampler down.

    `publish_once` attaches, publishes and detaches inside ONE lock hold, so a
    session claiming the bus wins it strictly before or strictly after — never
    during. This is the property the contract wanted exclusivity for.
    """
    bus = TickBus()
    sub = bus.subscribe(name="a")
    session = bus.attach_producer("session")
    _publish(session, seq_hint=0, goal_deg={"left": {"j": 1.0}})

    assert bus.publish_once("idle-sampler", t_mono=99.0) is None

    session.detach()
    assert bus.publish_once("idle-sampler", t_mono=1.0) is not None

    seqs = [s.seq for s in sub.drain()]
    assert seqs == [0, 1], "an idle sample slipped in beside the session's"


def test_a_producer_change_drops_the_rate_window_without_being_asked():
    """A window spanning two cadences describes neither, and fps comes from it."""
    bus = TickBus()
    idle = bus.attach_producer("idle-sampler")
    for i in range(60):
        _publish(idle, t=i * (1.0 / 20.0))
    assert bus.measured_hz() == pytest.approx(20.0, rel=1e-9)
    idle.detach()

    session = bus.attach_producer("session")
    assert bus.measured_hz() is None, "the 20 Hz window survived the handover"
    for i in range(60):
        _publish(session, t=100.0 + i * (1.0 / 60.0))
    assert bus.measured_hz() == pytest.approx(60.0, rel=1e-9)


def test_the_idle_sampler_accumulates_a_rate_across_its_own_publishes():
    """The other side of the reset rule, and the one that breaks quietly.

    `publish_once` attaches every tick. If attaching reset the window each
    time, the window would never reach the sample floor and `measured_hz`
    would be None forever — so the recorder could never refuse against a rate
    while idle, and the reset built to protect fps would have removed it.
    Reset is keyed on the producer's NAME changing, not on attaching.
    """
    bus = TickBus()
    for i in range(60):
        bus.publish_once("idle-sampler", t_mono=i * (1.0 / 20.0))
    assert bus.measured_hz() == pytest.approx(20.0, rel=1e-9)


def test_the_idle_sampler_skips_a_tick_its_source_cannot_answer():
    bus = TickBus()
    sub = bus.subscribe(name="a")
    sampler = IdleSampler(bus, sample=lambda: None, hz=1000.0)
    assert sampler.tick_once() is None
    assert sub.pending == 0


def test_the_idle_sampler_publishes_what_its_source_returns():
    bus = TickBus()
    sub = bus.subscribe(name="a")
    sampler = IdleSampler(
        bus, sample=lambda: {"t_mono": 0.0, "arms": {"left": {"joints_deg": {"j": 3.0}}}},
        hz=1000.0)
    sampler.tick_once()
    assert sub.drain()[0].arms["left"]["joints_deg"]["j"] == 3.0


def test_the_idle_sampler_stands_aside_for_a_session_and_returns_after():
    """While a session holds the bus, the sampler must not ASK its source.

    The claim is about the read, not the publish. On hardware the source is
    a serial-bus read of every servo, and a refused publish does not un-send
    it: the previous version of this test pinned `calls["n"] == 3` with the
    comment "the source is still asked, the bus just refuses" — written from
    the code, inheriting its blind spot — and the rig it described spent a
    whole session colliding 20 Hz idle reads with the session's goal writes
    on a lock-free half-duplex line (solo rig, 2026-08-29).
    """
    bus = TickBus()
    calls = {"n": 0}

    def sample():
        calls["n"] += 1
        return {"t_mono": float(calls["n"])}

    sampler = IdleSampler(bus, sample=sample, hz=1000.0)
    assert sampler.tick_once() is not None

    session = bus.attach_producer("session")
    assert sampler.tick_once() is None
    assert calls["n"] == 1, "an owned bus means the source is never touched"
    session.detach()

    assert sampler.tick_once() is not None
    assert calls["n"] == 2


def test_a_sample_carries_the_handles_snapshot_verbatim():
    """Nothing between the handle and a consumer decides which keys exist.

    Telemetry already has a test for this property on its own frame; the
    producer is now upstream of it, so the property has to hold here or the
    guarantee is gone one layer earlier with that test still green.
    """
    bus = TickBus()
    sub = bus.subscribe(name="a")
    token = bus.attach_producer("test")
    _publish(token, arms={"left": {
        "mode": "manual", "torque": True,
        "joints": {"gripper": {"pos": 1.0, "min": 0.0, "max": 100.0,
                               "torque": True, "effort": -0.42,
                               "a_channel_added_later": 7}},
    }})
    sample = sub.drain()[0]
    assert sample.arms["left"]["joints"]["gripper"]["a_channel_added_later"] == 7
    assert sample.arms["left"]["joints"]["gripper"]["effort"] == -0.42


def test_the_derived_views_are_computed_rather_than_stored():
    """One representation, so there is no copy to drift from the original."""
    bus = TickBus()
    sub = bus.subscribe(name="a")
    token = bus.attach_producer("test")
    _publish(token, arms={"left": {"joints": {
        "j1": {"pos": 3.0, "effort": 0.5},
        "j2": {"pos": -3.0},            # no effort channel on this joint
    }}})
    sample = sub.drain()[0]
    assert sample.joints_deg("left") == {"j1": 3.0, "j2": -3.0}
    assert sample.effort_norm("left") == {"j1": 0.5, "j2": 0.0}
    assert sample.joints_deg("nonexistent-arm") == {}


def test_a_sample_can_be_serialised_after_plain():
    """The telemetry frame goes to `ws.send_json`, and a frozen map cannot.

    `MappingProxyType` is not a dict subclass, so `json.dumps` refuses it. A
    bus-backed telemetry frame would 500 the socket on its first send, and a
    test that stopped at reading the values back would never see it.
    """
    import json

    bus = TickBus()
    sub = bus.subscribe(name="a")
    token = bus.attach_producer("test")
    _publish(token, arms={"left": {"joints": {"g": {"pos": 1.0}}}},
             goal_deg={"left": {"g": 2.0}})
    sample = sub.drain()[0]

    with pytest.raises(TypeError):
        json.dumps({"arms": sample.arms})

    round_tripped = json.loads(json.dumps({"arms": plain(sample.arms)}))
    assert round_tripped["arms"]["left"]["joints"]["g"]["pos"] == 1.0


def test_plain_hands_back_containers_the_caller_may_keep():
    """What a consumer passes onward must not reach back into a shared sample."""
    bus = TickBus()
    sub = bus.subscribe(name="a")
    token = bus.attach_producer("test")
    _publish(token, arms={"left": {"joints": {"g": {"pos": 1.0}}}})
    sample = sub.drain()[0]

    copy = plain(sample.arms)
    copy["left"]["joints"]["g"]["pos"] = 99.0
    assert sample.arms["left"]["joints"]["g"]["pos"] == 1.0
