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

Deliberately light on dependencies (stdlib only). `safety.py` is the one Haller
import, for `MIN_RATE_FRACTION`, and it is stdlib-only for the same reason.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Self

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
    #: {arm_id: {"joints_deg": {j: deg}, "effort_norm": {j: frac},
    #:           "torque": bool}}
    arms: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: {side: {joint: degrees}} — the commanded target, the recorder's `action`.
    goal_deg: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    #: {side: {joint: reason}} — why a joint was held back, e.g. "collision".
    reasons: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    base: Mapping[str, Any] = field(default_factory=dict)
    clutch: Mapping[str, Any] = field(default_factory=dict)
    collision: Mapping[str, Any] | None = None
    #: True when any arm in this tick was read degraded (a lost `isAvailable()`
    #: race, a comm failure, a stale side). Invariant 9: such a tick is a
    #: DROPPED frame, never a recorded one, and the flag is what lets a
    #: consumer tell "degraded" from "never happened".
    degraded: bool = False

    def __post_init__(self) -> None:
        for name in ("arms", "goal_deg", "reasons", "base", "clutch",
                     "collision"):
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

    # -- producers ---------------------------------------------------------

    def attach_producer(self, name: str) -> ProducerToken:
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
            return token

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

    def measured_hz(self) -> float | None:
        """Real publish rate over the rolling window, or None while unknown.

        None until `RATE_MIN_SAMPLES` publishes. Invariant 10 — `fps` in
        `info.json` is measured or the episode does not open — so an
        unmeasured rate reports as absent rather than as a plausible number.
        Mechanism 3 was exactly a plausible number.
        """
        with self._lock:
            if len(self._stamps) < self._rate_min_samples:
                return None
            span = self._stamps[-1] - self._stamps[0]
            if span <= 0:
                return None
            return (len(self._stamps) - 1) / span

    def rate_ok(self, declared_hz: float, fraction: float) -> bool | None:
        """Is the measured rate at least `fraction` of `declared_hz`?

        None while the rate is not yet known — which is NOT a pass. Callers
        refuse on None; a gate that treats "do not know" as "fine" is a check
        that cannot fire.

        `fraction` is passed in rather than read here so this module never
        becomes a second home for `safety.MIN_RATE_FRACTION`. One threshold,
        one home, read by both measuring surfaces.
        """
        measured = self.measured_hz()
        if measured is None or declared_hz <= 0:
            return None
        return measured >= declared_hz * fraction

    def reset_rate(self) -> None:
        """Forget the rate window — used at a producer handover.

        The two producers run at different cadences, so a window spanning the
        handover measures neither of them.
        """
        with self._lock:
            self._stamps.clear()
