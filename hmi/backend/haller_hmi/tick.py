# hmi/backend/haller_hmi/tick.py
"""One sampler owns the tick: state, action and pixels from a single moment.

Replaces three uncoordinated samplers (plan §"Why it failed here", mechanisms
1-3): the session committing at 60 Hz, telemetry running its own 20 Hz loop,
and the recorder scraping `human_teleop.status()["goal_deg"]` at a third
instant. Every recorded row therefore paired a state from one moment with an
action from another, which teaches a policy a lie about causality that no
number of episodes corrects.

This module is the fix's foundation: a producer publishes one `TickSample`, and
every consumer that reads that sample reads the SAME moment. Invariant 8.

STDLIB ONLY, and it must stay that way: the detached rollout child imports
this without wanting lerobot, which `arm.py` would drag in. This module also
never becomes a second home for a number that belongs elsewhere — it MEASURES
the rate and reports it, and every threshold a rate is judged against lives
with the surface doing the judging. Pinned by
`test_tick_does_not_define_its_own_rate_fraction`.

That discipline extends to VERDICTS, which is why there is no `rate_ok` here
any more (deleted 2026-08-27, no production caller). The record gate is a
symmetric tolerance — `timestamp` is `frame_index / fps`, so fast is as wrong
as slow — and the rollout gate is a one-sided floor. One helper on the object
they both read can only ever express one of those, under a name general enough
for the other to reach for.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Self

logger = logging.getLogger(__name__)

#: Publish intervals kept for the rolling rate measurement. At 60 Hz this is a
#: two-second window: long enough that one late tick does not move the number,
#: short enough that a real slowdown shows up while the operator is still
#: standing there.
RATE_WINDOW = 120

#: Publishes required before `measured_hz` reports anything at all. Below this
#: the window is too short to distinguish a rate from a startup transient, and
#: invariant 10 says fps is measured or the episode does not open — so the
#: honest answer while we do not know is None, never a guess.
RATE_MIN_SAMPLES = 30

#: Default depth of a subscriber's queue. Deep enough to absorb a consumer that
#: blocks for a few ticks (a video encode, a stats fold), shallow enough that a
#: consumer which has genuinely fallen behind is told so rather than served
#: stale samples forever.
DEFAULT_QUEUE_DEPTH = 64


class ProducerConflict(RuntimeError):
    """Raised when a second producer attaches while one is already attached.

    The bus takes exactly one producer at a time — `IdleSampler` when no
    session runs, the session tick when one does. See `TickBus.attach_producer`
    for what an overlap actually costs, which is not what it looks like.
    """


def _freeze(value: Any) -> Any:
    """Deep-freeze into read-only views, COPYING every container on the way.

    The copy is the load-bearing half and it is easy to leave out. A
    `MappingProxyType` over a dict the producer still owns is a read-only
    VIEW of a mutable object: the consumer cannot write through it, but the
    producer's next tick mutates what the consumer is holding, and a sample
    that changes after publication is precisely the "two moments in one row"
    defect this module exists to end. Wrapping without copying would look
    exactly like this function and be worth nothing.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    # str/bytes are iterable and already immutable; leave them whole.
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze(v) for v in value)
    return value


def plain(value: Any) -> Any:
    """The inverse of `_freeze`: read-only views back to plain containers.

    Any consumer that SERIALISES part of a sample must call this first.
    `MappingProxyType` is not a `dict` subclass, so `json.dumps` raises
    `Object of type mappingproxy is not JSON serializable` — and the telemetry
    frame goes straight to `ws.send_json`. Without this the first bus-backed
    frame would 500 the telemetry socket, and no unit test that stops short of
    serialising would see it.

    Returns fresh containers, so what a consumer hands onward is theirs to
    mutate and cannot reach back into a sample other consumers hold.
    """
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, frozenset):
        return {plain(v) for v in value}
    return value


@dataclass(frozen=True)
class TickSample:
    """One moment. Every consumer of a given `seq` sees exactly these values.

    Frozen, and every container inside it is a read-only copy — see `_freeze`.
    A consumer that could mutate a sample would corrupt every other consumer's
    view of the same moment, and one-moment-for-everyone is the entire point.

    `t_mono` is `time.perf_counter()`, for durations and the rate measurement.
    `t_unix` is `time.time()`, captured in the SAME breath, because the
    recorder's `observation.wall_clock` column needs an absolute epoch and a
    recorder calling `time.time()` itself would be a third instant — mechanism
    1 re-entering through the timestamp column.
    """

    seq: int
    t_mono: float
    t_unix: float
    #: {arm_id: the handle's `state_snapshot()`, VERBATIM}.
    #:
    #: Verbatim rather than a {joints_deg, effort_norm, torque} projection, and
    #: the reason is a property telemetry already has a test for: the
    #: broadcaster owns no part of the per-joint dict, so a new key inside
    #: `joints` reaches subscribers untouched (`test_telemetry.py::
    #: test_effort_passes_through_the_frame_verbatim`). A projection here would
    #: silently become the place that decides which per-joint keys exist, and
    #: the next channel added to `state_snapshot()` would be dropped at the
    #: producer with every test still green.
    #:
    #: Derived views are METHODS below, never stored fields — one
    #: representation, so there is no copy to drift from the original.
    arms: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: {side: {joint: degrees}} — the commanded target, the recorder's `action`.
    goal_deg: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    #: {side: {joint: reason}} — why a joint was held back, e.g. "collision".
    reasons: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: {arm_id: message} for arms whose state read FAILED this tick. An arm
    #: here is absent from `arms` above, so a consumer sees the hole rather
    #: than a plausible number standing in for it.
    arm_errors: Mapping[str, str] = field(default_factory=dict)
    base: Mapping[str, Any] = field(default_factory=dict)
    clutch: Mapping[str, Any] = field(default_factory=dict)
    collision: Mapping[str, Any] | None = None
    #: True when any arm in this tick was read degraded (a lost `isAvailable()`
    #: race, a comm failure, a stale side). Invariant 9: such a tick is a
    #: DROPPED frame, never a recorded one, and the flag is what lets a
    #: consumer tell "degraded" from "never happened".
    degraded: bool = False

    def joints_deg(self, arm_id: str) -> dict[str, float]:
        """{joint: degrees} for one arm, derived from its snapshot."""
        joints = self.arms.get(arm_id, {}).get("joints", {})
        return {j: float(v["pos"]) for j, v in joints.items()}

    def effort_norm(self, arm_id: str) -> dict[str, float]:
        """{joint: signed fraction of torque limit} for one arm.

        0.0 for a joint whose snapshot carries no effort. Read
        `recorder.py`'s `observation.effort` docstring before using this: a
        flat-zero column means "no effort channel on that take", not "no
        contact", which is why a transient degradation is a dropped frame
        rather than a zero.
        """
        joints = self.arms.get(arm_id, {}).get("joints", {})
        return {j: float(v.get("effort", 0.0)) for j, v in joints.items()}

    def __post_init__(self) -> None:
        for name in ("arms", "goal_deg", "reasons", "arm_errors", "base",
                     "clutch", "collision"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


class TickSubscription:
    """A bounded, drop-oldest, COUNTED view of the tick stream.

    Counted is the load-bearing word. A silent drop is indistinguishable from
    a tick that never happened, and invariant 9 turns entirely on telling those
    two apart: one is a consumer that could not keep up, the other is a sampler
    that stalled, and they call for opposite responses.

    Drop-oldest rather than drop-newest because every consumer here wants the
    freshest moment it can get; a recorder that has fallen behind writing the
    stale half of its queue would be recording the past at the wrong timestamps.
    """

    def __init__(self, bus: TickBus, *, name: str,
                 maxsize: int = DEFAULT_QUEUE_DEPTH,
                 loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.name = name
        self._bus = bus
        self._maxsize = max(1, int(maxsize))
        self._items: deque[TickSample] = deque()
        self._dropped = 0
        self._delivered = 0
        self._lock = threading.Lock()
        self._loop = loop
        self._event = asyncio.Event()
        self._closed = False
        #: While True, samples are counted as delivered and discarded rather
        #: than queued. This is how an ARMED-but-not-rolling recorder holds a
        #: subscription without manufacturing drop counts (contract C2): the
        #: operator's HUD must never show `skipped_frames` climbing while
        #: parked, or they learn to ignore the one number that has to be
        #: trusted mid-take.
        self._draining = False

    @property
    def dropped(self) -> int:
        """Samples this subscriber lost to a full queue. Monotonic."""
        with self._lock:
            return self._dropped

    @property
    def delivered(self) -> int:
        with self._lock:
            return self._delivered

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._items)

    def set_draining(self, draining: bool) -> None:
        """Discard-on-arrival, without dropping the subscription.

        A subscriber that is attached but not consuming would otherwise
        overflow continuously and count drops that mean nothing — nothing was
        lost, nothing was going to be written.

        NO PRODUCTION CALLER, ON PURPOSE — and this paragraph is the reason it
        is not cruft. Contract C2 says an ARMED-but-not-rolling recorder must
        not manufacture drop counts, and it has two legal answers: subscribe at
        ARM and drain until ROLL, or do not subscribe until ROLL. Phase 2d took
        the second (`recorder.arm` takes no subscription at all), so this is the
        answer that was not used rather than a mechanism nobody needed.

        Left standing because C2 is still in force and subscribing at ARM is the
        more natural shape — the next consumer that wants a long-lived armed
        subscription needs exactly this, and rebuilding it from the symptom is
        how the manufactured drop counts get shipped once first. What is NOT
        legal is subscribing early and leaving the queue to overflow.
        """
        with self._lock:
            self._draining = bool(draining)
            if self._draining:
                self._items.clear()

    def _deliver(self, sample: TickSample) -> None:
        """Called by the bus, holding the bus lock. Never blocks the producer."""
        with self._lock:
            if self._closed:
                return
            if self._draining:
                self._delivered += 1
                return
            if len(self._items) >= self._maxsize:
                self._items.popleft()
                self._dropped += 1
            self._items.append(sample)
            self._delivered += 1
        self._wake()

    def _wake(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._event.set)
        except RuntimeError:
            # Loop closed or shutting down. A consumer that is going away does
            # not get to take the producer down with it.
            pass

    def drain(self) -> list[TickSample]:
        """Every queued sample, oldest first. Non-blocking, safe from any thread."""
        with self._lock:
            items = list(self._items)
            self._items.clear()
            self._event.clear()
        return items

    def latest(self) -> TickSample | None:
        """The newest queued sample, discarding older ones.

        The decimating read: telemetry runs at its own cadence and wants the
        current moment, not a backlog. Discards here are NOT drops — they are
        this consumer declining samples it never asked for — so they are not
        counted against `dropped`.
        """
        with self._lock:
            if not self._items:
                return None
            sample = self._items[-1]
            self._items.clear()
            self._event.clear()
        return sample

    async def get(self, timeout: float | None = None) -> TickSample | None:
        """Await the next sample. Requires a `loop` bound at subscribe time."""
        if self._loop is None:
            raise RuntimeError(
                "TickSubscription.get() needs an event loop bound at "
                "subscribe() time; use drain()/latest() from a plain thread"
            )
        while True:
            with self._lock:
                if self._items:
                    sample = self._items.popleft()
                    if not self._items:
                        self._event.clear()
                    return sample
                if self._closed:
                    return None
                self._event.clear()
            try:
                if timeout is None:
                    await self._event.wait()
                else:
                    await asyncio.wait_for(self._event.wait(), timeout)
            except TimeoutError:
                return None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._items.clear()
        self._wake()
        self._bus._unsubscribe(self)


class ProducerToken:
    """Proof that the holder is the bus's single current producer."""

    def __init__(self, bus: TickBus, name: str) -> None:
        self.name = name
        self._bus = bus
        self._live = True

    @property
    def live(self) -> bool:
        return self._live

    def publish(self, **kwargs: Any) -> TickSample:
        return self._bus._publish(self, **kwargs)

    def detach(self) -> None:
        self._bus._detach(self)
        self._live = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.detach()


class TickBus:
    """Synchronous fanout of one moment to every consumer, with counted drops.

    Thread-safe by construction: the producer is a plain thread (the teleop
    loop) while consumers live on the asyncio loop, so nothing here may assume
    an event loop and every asyncio wakeup goes through `call_soon_threadsafe`.

    `seq` is allocated HERE, not by the producer, so no two samples can ever
    share one — see `attach_producer` for why that changes what a producer
    overlap actually costs.
    """

    def __init__(self, *, rate_window: int = RATE_WINDOW,
                 rate_min_samples: int = RATE_MIN_SAMPLES) -> None:
        self._lock = threading.RLock()
        self._subs: list[TickSubscription] = []
        self._seq = 0
        self._published = 0
        self._latest: TickSample | None = None
        self._stamps: deque[float] = deque(maxlen=max(2, int(rate_window)))
        self._rate_min_samples = max(2, int(rate_min_samples))
        self._producer: ProducerToken | None = None
        self._rate_producer: str | None = None
        # What the current producer is AIMING at. Recorded beside the
        # measurement and NEVER written as `fps` — see `rate_detail`.
        self._target_hz: float | None = None

    # -- producers ---------------------------------------------------------

    def attach_producer(self, name: str, *,
                        target_hz: float | None = None) -> ProducerToken:
        """Claim the bus. Raises `ProducerConflict` if someone already holds it.

        NOTE, and it corrects the reasoning in the tick contract rather than
        the ruling: the contract justifies exclusivity with "two producers
        overlapping for one tick puts two samples on one `seq`". With `seq`
        allocated by the bus under this lock, that specific symptom CANNOT
        occur — every sample gets its own number whoever published it.

        What an overlap actually costs is quieter and worse. Two producers
        interleaving means the bus sees roughly double the publish interval
        density, so `measured_hz` reads high for as long as the overlap lasts.
        `fps` is frozen into `info.json` from that measured rate at ARM time,
        and every `timestamp` in the episode is then synthesised as
        `frame_index / fps`. An overlap across an arm is therefore mechanism 3
        — a declared-not-measured fps — re-entering through the machinery built
        to prevent it, with a number that looks measured because it was.

        So the exclusivity stands and the handover stays tested; the symptom
        to test for is the RATE, not a duplicate seq.
        """
        with self._lock:
            if self._producer is not None and self._producer.live:
                raise ProducerConflict(
                    f"{self._producer.name!r} already produces this bus; "
                    f"{name!r} must wait for the handover"
                )
            token = ProducerToken(self, name)
            self._producer = token
            self._note_producer(name, target_hz)
            return token

    def _note_producer(self, name: str,
                       target_hz: float | None = None) -> None:
        """Caller holds the lock. Drop the rate window when the cadence changes.

        The idle sampler and the session run at different rates, so a window
        spanning a handover describes neither of them — and `fps` is frozen
        into `info.json` from that number. Automatic rather than a call the
        handover has to remember, because the one that forgets is the one that
        writes a wrong fps into a real episode.
        """
        if name != self._rate_producer:
            self._stamps.clear()
            self._rate_producer = name
        self._target_hz = target_hz

    def publish_once(self, name: str, *, target_hz: float | None = None,
                     **fields: Any) -> TickSample | None:
        """Attach, publish one sample, and detach — all under one lock hold.

        The idle sampler's whole API, and the reason the handover needs no
        takeover protocol. Because the token is created and released inside a
        single lock hold, a session claiming the bus for its lifetime either
        wins it strictly before this call or strictly after it, never during.
        Two producers therefore cannot overlap even for one tick, and neither
        has to know the other exists.

        Returns None when a long-lived producer already holds the bus. That is
        the ordinary state while a session runs — not an error, and not
        something to log every tick.
        """
        with self._lock:
            if self._producer is not None and self._producer.live:
                return None
            token = ProducerToken(self, name)
            self._producer = token
            self._note_producer(name, target_hz)
            try:
                return self._publish(token, **fields)
            finally:
                self._producer = None
                token._live = False

    @property
    def target_hz(self) -> float | None:
        with self._lock:
            return self._target_hz

    @property
    def producer_name(self) -> str | None:
        with self._lock:
            return self._producer.name if self._producer is not None else None

    def _detach(self, token: ProducerToken) -> None:
        with self._lock:
            if self._producer is token:
                self._producer = None

    def _publish(self, token: ProducerToken, **kwargs: Any) -> TickSample:
        with self._lock:
            if self._producer is not token or not token.live:
                raise ProducerConflict(
                    f"{token.name!r} is not the current producer"
                )
            now_mono = kwargs.pop("t_mono", None)
            now_unix = kwargs.pop("t_unix", None)
            sample = TickSample(
                seq=self._seq,
                t_mono=time.perf_counter() if now_mono is None else now_mono,
                t_unix=time.time() if now_unix is None else now_unix,
                **kwargs,
            )
            self._seq += 1
            self._published += 1
            self._latest = sample
            self._stamps.append(sample.t_mono)
            for sub in self._subs:
                sub._deliver(sample)
            return sample

    # -- consumers ---------------------------------------------------------

    def subscribe(self, *, name: str, maxsize: int = DEFAULT_QUEUE_DEPTH,
                  loop: asyncio.AbstractEventLoop | None = None
                  ) -> TickSubscription:
        sub = TickSubscription(self, name=name, maxsize=maxsize, loop=loop)
        with self._lock:
            self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub: TickSubscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def latest(self) -> TickSample | None:
        with self._lock:
            return self._latest

    @property
    def published(self) -> int:
        with self._lock:
            return self._published

    # -- rate --------------------------------------------------------------

    def rate_detail(self) -> dict | None:
        """The measurement behind `measured_hz`, with what it was taken from.

        `hz` is unrounded. The recorder writes an INTEGER `fps` and records
        this beside it, because 29.4 -> 29 is a fact worth being able to see
        later and lerobot's `fps` field cannot hold it: `DatasetInfo` types it
        `int`, and every `timestamp` in the episode is synthesised as
        `frame_index / fps`.

        `samples` and `window_s` are here so a reader can tell a rate measured
        over two seconds of steady running from one measured across a stall.
        """
        with self._lock:
            if len(self._stamps) < self._rate_min_samples:
                return None
            span = self._stamps[-1] - self._stamps[0]
            if span <= 0:
                return None
            return {
                "hz": (len(self._stamps) - 1) / span,
                "samples": len(self._stamps),
                "window_s": span,
                "target_hz": self._target_hz,
            }

    def measured_hz(self) -> float | None:
        """Real publish rate over the rolling window, or None while unknown.

        None until `RATE_MIN_SAMPLES` publishes. Invariant 10 — `fps` in
        `info.json` is measured or the episode does not open — so an
        unmeasured rate reports as absent rather than as a plausible number.
        Mechanism 3 was exactly a plausible number.

        Delegates to `rate_detail` rather than recomputing: two
        implementations of one measurement is how they come to disagree.
        """
        detail = self.rate_detail()
        return None if detail is None else detail["hz"]

    def reset_rate(self) -> None:
        """Forget the rate window — used at a producer handover.

        The two producers run at different cadences, so a window spanning the
        handover measures neither of them.
        """
        with self._lock:
            self._stamps.clear()


class IdleSampler:
    """Produces the tick while no teleop session owns the bus.

    A cockpit with no session still needs live arms, and the recorder still
    needs a measured rate to refuse against before a session ever starts. This
    runs at telemetry's cadence and steps aside the instant a session attaches
    — `tick_once` checks the bus's producer before it so much as reads, so
    there is no stand-down handshake to get wrong and no idle read left on
    the wire beside a session's writes (see `tick_once` for why the check
    must precede the read).

    Takes a `sample` callable rather than the arm manager so this module stays
    stdlib-only: the wiring that knows about arms and ROS lives in the lifespan
    that builds it, which keeps `server.py`'s delta to a few lines.

    `sample()` returns the `TickSample` field dict for one moment, or None to
    skip this tick (nothing to report, or a read that failed outright).
    """

    def __init__(self, bus: TickBus, *, sample: Callable[[], dict | None],
                 hz: float = 20.0, name: str = "idle-sampler") -> None:
        self._bus = bus
        self._sample = sample
        self._period = 1.0 / max(1e-6, float(hz))
        self._name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def tick_once(self) -> TickSample | None:
        """One sample, published if the bus is free. Synchronous, for tests.

        The ownership check comes BEFORE `self._sample()`, and that ordering
        is the point, not an optimisation. On hardware the sample callable is
        a serial-bus read of every servo, and `publish_once` refusing the
        RESULT does not un-send the read: sampled-then-refused, this thread
        kept transmitting on the half-duplex Feetech line at its full cadence
        for as long as a session ran beside it — colliding with the session
        thread's goal writes and its own reads (`arm.py` serialises bus access
        by architecture, not by locks). Measured on the solo rig 2026-08-29:
        a continuous "no status packet" storm for a whole session, corrupted
        reseeds, dropped goals.

        A session attaching between this check and the read can still overlap
        it for at most one in-flight sample at handover — the same one-shot
        race the effort path already tolerates — which is why the session
        seeds its own state only after claiming the bus. Holding the bus lock
        across a hardware read would close that window and is worse: it would
        block the session's attach for the duration of every idle read.
        """
        if self._bus.producer_name is not None:
            return None
        fields = self._sample()
        if fields is None:
            return None
        return self._bus.publish_once(self._name,
                                      target_hz=1.0 / self._period,
                                      **fields)

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                self.tick_once()
            except Exception:
                # A sampler that dies takes the cockpit's only view of the
                # arms with it. Log and keep the cadence.
                logger.exception("idle sampler tick failed")
            self._stop.wait(
                max(0.0, self._period - (time.perf_counter() - started)))
