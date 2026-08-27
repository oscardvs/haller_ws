# hmi/backend/tests/test_record_gate.py
"""Phase 2d — the ARM / ROLL / STOP gate, asserted from the contract.

Written from `docs/port/phase2-tick-contract.md` and Track D's four-row
save/rearm matrix, in the contract's own terms, before reading the
implementation back. The port's counter-discipline: a test written from the
code inherits the code's blind spot.

The fakes are IMPORTED from `test_recorder` rather than re-declared. One fake
with one source of truth beats two that can drift — and drift is the specific
hazard here, because a fake MORE PERMISSIVE than production is the
impossible-fixture rule pointing the other way. `_FakeArms` deliberately has
`__getitem__` and no `__iter__`, exactly like the real `ArmManager`, which is
what makes `for x in manager` fail in a test the way it fails on the rig.

Rates are measured off an INJECTED clock throughout (`_measured_bus`), so no
assertion here measures this box. Four sessions share it.
"""
from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from haller_hmi.recorder import (
    ARMED,
    EPISODE_UID_FEATURE,
    IDLE,
    RECORDING,
    DatasetRecorder,
    RateNotMeasuredYet,
    RateUnfaithful,
)
from haller_hmi.tick import TickBus

from .test_recorder import (
    SIX,
    _FakeArm,
    _FakeArms,
    _FakeCameras,
    _FakeDataset,
    _FakeHumanTeleop,
    _FakeTelemetry,
    _joints_block,
    _measured_bus,
)

DRIVING = {"running": True, "left_arm": "left", "right_arm": "right", "goal_deg": {}}
STOPPED = {"running": False, "left_arm": None, "right_arm": None, "goal_deg": {}}


class _MutableTeleop:
    """A session whose status the test can change mid-flight.

    `_FakeHumanTeleop` holds one dict forever, which cannot express the thing
    every invalidation test needs: teleop being one way at ARM time and another
    way afterwards.
    """

    def __init__(self, status):
        self.status_dict = dict(status)

    def status(self):
        return dict(self.status_dict)


class _CountingDataset(_FakeDataset):
    """A fake dataset that records WHEN its episode index was read.

    The C1 index promise is an ORDERING claim — the next index is never
    reported before the previous save has committed — and an ordering claim
    cannot be pinned by reading the final value, which is identical either way.
    So `meta.total_episodes` becomes a property that appends to a log, and the
    test asserts on the log.
    """

    def __init__(self, on_save=None, repo_id: str = "smoke/gate"):
        super().__init__(repo_id=repo_id)
        self.log: list[str] = []
        self.meta = _LoggingMeta(self.log)
        self._on_save = on_save
        self.concurrent_saves = 0
        self.max_concurrent_saves = 0
        self._lock = threading.Lock()

    def save_episode(self):
        with self._lock:
            self.concurrent_saves += 1
            self.max_concurrent_saves = max(self.max_concurrent_saves,
                                            self.concurrent_saves)
        self.log.append("save:enter")
        if self._on_save is not None:
            self._on_save()
        self.saved += 1
        self.meta.total_episodes += 1
        self.log.append("save:exit")
        with self._lock:
            self.concurrent_saves -= 1


class _LoggingMeta:
    def __init__(self, log, fps: int = 30):
        self._log = log
        self._total = 0
        self.info = {"fps": fps}

    @property
    def total_episodes(self):
        self._log.append("read_index")
        return self._total

    @total_episodes.setter
    def total_episodes(self, v):
        self._total = v


def _gate_recorder(teleop=None, cams=None, bus=None, dataset=None):
    """A recorder wired for the gate routes, with a filled rate window.

    No `_dataset` is pre-set unless the caller asks: `arm` opens one, and half
    of what these tests check is what `arm` itself does.
    """
    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    r = DatasetRecorder(
        tick_bus=bus if bus is not None else _measured_bus(),
        telemetry=_FakeTelemetry(arms),
        human_teleop=teleop if teleop is not None else _FakeHumanTeleop(dict(STOPPED)),
        cameras=cams if cams is not None else _FakeCameras([]),
    )
    if dataset is not None:
        r._dataset = dataset
    return r


async def _armed(**kw):
    """A recorder sitting in ARMED over a fake dataset, as after `/record/arm`."""
    r = _gate_recorder(dataset=kw.pop("dataset", None) or _FakeDataset(), **kw)
    await r.arm("smoke/gate", "lift the cube")
    return r


# --- ARM writes nothing; ROLL is what starts the frames -------------------

async def test_arming_opens_the_gate_and_writes_no_frames():
    """`POST /record/arm` — open, freeze, resolve, and write NOTHING.

    The claim is an ABSENCE, so it is asserted against the two things that
    would have to exist for a frame to have been written: a record loop, and a
    frame on the dataset.
    """
    r = await _armed()
    st = r.status()
    assert st["state"] == ARMED
    assert st["recording"] is False
    assert st["episode_index"] == 0
    assert st["invalidated_reason"] is None
    assert r._dataset.frames == []
    assert r._task_handle is None          # no record loop exists yet
    assert r._state.started_at is None     # a take that has not begun has no start


async def test_rolling_is_what_starts_the_frames():
    r = await _armed()
    await r.roll()
    st = r.status()
    assert st["state"] == RECORDING
    assert st["recording"] is True
    assert st["started_at"] is not None
    assert r._task_handle is not None
    await r.stop_episode(save=False)


async def test_roll_without_arm_is_refused():
    """Every refusal lives at arm time; roll's only job is to refuse the
    sequence itself."""
    r = _gate_recorder()
    with pytest.raises(RuntimeError, match="not armed"):
        await r.roll()
    assert r.status()["state"] == IDLE


async def test_arming_while_a_take_is_rolling_is_refused():
    r = await _armed()
    await r.roll()
    with pytest.raises(RuntimeError, match="already recording"):
        await r.arm("smoke/gate", "another")
    await r.stop_episode(save=False)


async def test_start_episode_is_exactly_arm_then_roll():
    """The shipped `/record/start` keeps working, and by COMPOSITION rather
    than by a second implementation of the open sequence."""
    r = _gate_recorder(dataset=_FakeDataset())
    await r.start_episode("smoke/gate", "lift the cube")
    st = r.status()
    assert st["state"] == RECORDING
    assert st["episode_index"] == 0        # arm's freeze survived the roll
    assert st["started_at"] is not None
    await r.stop_episode(save=False)


# --- the four-row save/rearm matrix --------------------------------------
#
# One test per row. Each asserts all three columns the contract names — where
# it LANDS, what happens to the INDEX, and whether disk was touched — because
# a row is only distinguishable from its neighbours by the combination.

async def _rolled_take(frames: int = 3, dataset=None):
    r = await _armed(dataset=dataset)
    await r.roll()
    for _ in range(frames):
        r._dataset.add_frame({"task": "t"})
        r._state.episode_frames += 1
    return r


async def test_save_without_rearm_lands_idle_and_advances_the_index():
    r = await _rolled_take()
    st = await r.stop_episode(save=True)
    assert st["state"] == IDLE
    assert st["episode_index"] is None      # nothing is armed, so no index
    assert r._dataset.saved == 1
    assert r._dataset.meta.total_episodes == 1   # it advanced on disk


async def test_re_record_stays_armed_at_the_SAME_index_and_never_touches_disk():
    """`{save: false, rearm: true}` — the headset's REDO.

    An episode buffer that was never `save_episode`'d is simply dropped. No
    delete, no stats recompute, and the index does not move: the next take
    lands exactly where this one would have.
    """
    r = await _rolled_take()
    before = r.status()["episode_index"]
    st = await r.stop_episode(save=False, rearm=True)
    assert st["state"] == ARMED
    assert st["episode_index"] == before
    assert r._dataset.saved == 0
    assert r._dataset.cleared == 1
    assert r._dataset.meta.total_episodes == 0


async def test_discard_lands_idle_and_leaves_the_index_where_it_was():
    r = await _rolled_take()
    st = await r.stop_episode(save=False, rearm=False)
    assert st["state"] == IDLE
    assert st["episode_index"] is None
    assert r._dataset.saved == 0
    assert r._dataset.cleared == 1
    assert r._dataset.meta.total_episodes == 0


async def test_save_and_go_again_lands_armed_at_the_NEXT_index():
    """The hot path. L-stick click = KEEP = save this take and re-arm."""
    r = await _rolled_take()
    assert r.status()["episode_index"] == 0
    st = await r.stop_episode(save=True, rearm=True)
    assert st["state"] == ARMED
    assert st["episode_index"] == 1
    assert r._dataset.saved == 1


async def test_ten_takes_without_ever_leaving_armed():
    """A claim about a SEQUENCE cannot be pinned by tests about single
    transitions — the one-cycle test walks the same control flow the code does
    and adopts its ordering assumptions.

    So: ten consecutive keeps, the way an operator actually banks a session,
    asserting the index at every rung rather than only at the end. An
    off-by-one that cancels itself over a round trip is invisible to a test
    that only reads the total.
    """
    r = await _armed()
    seen = []
    for _ in range(10):
        await r.roll()
        for _ in range(3):
            r._dataset.add_frame({"task": "t"})
            r._state.episode_frames += 1
        seen.append(r.status()["episode_index"])
        st = await r.stop_episode(save=True, rearm=True)
        assert st["state"] == ARMED
    assert seen == list(range(10))
    assert r._dataset.saved == 10
    assert r.status()["episode_index"] == 10


async def test_stop_with_save_alone_still_means_exactly_what_it_meant():
    """`rearm` is OPTIONAL and defaults false. Two shipped desktop surfaces
    call `{save}` alone and predate the headset's state machine entirely."""
    r = await _rolled_take()
    st = await r.stop_episode(save=True)      # no rearm argument at all
    assert st["state"] == IDLE
    assert st["recording"] is False
    assert r._dataset.saved == 1


async def test_standing_down_from_armed_leaves_no_reason_to_explain():
    """Stopping from ARMED without ever rolling is a deliberate act, so it
    must not leave an `invalidated_reason` behind — that field means the gate
    went stale under the operator, and a HUD explaining a stand-down as a
    fault is worse than one saying nothing."""
    r = await _armed()
    st = await r.stop_episode(save=False)
    assert st["state"] == IDLE
    assert st["invalidated_reason"] is None
    assert r._dataset.saved == 0


# --- C1: the index is a fact, not a prediction ---------------------------

async def test_the_next_index_is_read_AFTER_the_save_has_committed():
    """C1's index promise, asserted as the ORDERING it actually is.

    Track D deleted its own `episodesTotal()` floor because we promised this
    index is the truth, so reporting it before `save_episode` commits would be
    worse than the guess it replaced. Reading the final VALUE cannot tell the
    two apart — both end at 1. The call ORDER can.
    """
    ds = _CountingDataset()
    r = await _armed(dataset=ds)
    await r.roll()
    for _ in range(3):
        ds.add_frame({"task": "t"})
        r._state.episode_frames += 1
    ds.log.clear()
    st = await r.stop_episode(save=True, rearm=True)

    assert st["episode_index"] == 1
    # The save's own pre-read, then the save, then the re-arm's read.
    assert ds.log.index("save:exit") < ds.log.index("read_index",
                                                    ds.log.index("save:exit"))
    assert ds.log[-1] == "read_index"


async def test_a_second_stop_cannot_interleave_with_the_first_still_flushing():
    """Two stops racing must serialise, because `save_episode` is not
    re-entrant: a second one arriving mid-flush is how a buffer gets saved
    twice or half."""
    gate = threading.Event()
    ds = _CountingDataset(on_save=lambda: gate.wait(timeout=5.0))
    r = await _armed(dataset=ds)
    await r.roll()
    for _ in range(3):
        ds.add_frame({"task": "t"})
        r._state.episode_frames += 1

    first = asyncio.create_task(r.stop_episode(save=True, rearm=True))
    await asyncio.sleep(0.05)                 # let the first reach the flush
    second = asyncio.create_task(r.stop_episode(save=True, rearm=False))
    await asyncio.sleep(0.05)                 # the second must be waiting
    assert not second.done()
    gate.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)
    assert ds.max_concurrent_saves == 1


async def test_nothing_on_the_event_loop_stalls_while_the_save_flushes():
    """C1's third requirement, pointed at the half that can actually fail.

    Two claims live here and only one of them is at risk, so they are asserted
    separately rather than as one comfortable pass:

    THE PRODUCER is the teleop session's own `threading.Thread`, so teleop
    cannot be stalled by anything the event loop does. That is structural, it
    was true before this commit, and a test asserting it would hold with the
    save running synchronously on the loop — it is a guard against a future
    save that takes a lock the producer needs, and nothing more. It is the
    weaker assertion and it is labelled as such.

    THE CONSUMERS are the half at risk. `save_episode` folds stats and may
    encode video, so run on the event loop it freezes telemetry's websocket for
    the whole flush — at exactly the moment the HUD is telling the operator to
    go again. That is what the `asyncio.to_thread` hop buys, and a heartbeat
    coroutine is what can tell the two apart: it advances across the flush now
    and would flatline if the save came back onto the loop.
    """
    gate = threading.Event()
    ds = _CountingDataset(on_save=lambda: gate.wait(timeout=5.0))
    bus = _measured_bus()
    r = await _armed(dataset=ds, bus=bus)
    await r.roll()
    for _ in range(3):
        ds.add_frame({"task": "t"})
        r._state.episode_frames += 1

    published: list[int] = []
    beats = 0
    stop = threading.Event()

    def produce():
        token = bus.attach_producer("test-thread")
        try:
            i = 0
            while not stop.is_set():
                token.publish(t_mono=1000.0 + i * 0.01,
                              arms={"left": _joints_block(0.0)}, goal_deg={})
                published.append(i)
                i += 1
                stop.wait(0.002)
        finally:
            token.detach()

    async def heartbeat():
        nonlocal beats
        while True:
            await asyncio.sleep(0.005)
            beats += 1

    r._end_tick_stream()                       # release the loop's own hold
    await asyncio.sleep(0.02)
    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    beat_task = asyncio.create_task(heartbeat())
    try:
        stopping = asyncio.create_task(r.stop_episode(save=True))
        await asyncio.sleep(0.05)              # the save is now blocked
        # THE MUTATION LANDS HERE, not on the heartbeat below, and the message
        # says so because the symptom is confusing otherwise. With the save
        # back on the event loop this line is not reached until the flush has
        # ALREADY finished — observing a flush in progress itself requires the
        # loop to come back to us, which is the very thing being stalled. So a
        # zero here is the stall, not a slow fake.
        assert ds.concurrent_saves == 1, (
            "the flush was never observable in progress — the event loop did "
            "not come back while save_episode was running, which IS the stall "
            "this test exists to catch")
        beats_at_flush, published_at_flush = beats, len(published)
        await asyncio.sleep(0.15)

        # THE ONE THAT CAN FIRE: the loop kept turning through the flush.
        assert beats > beats_at_flush, (
            "the event loop stalled for the whole save — telemetry and the HUD "
            "freeze with it")
        # The weaker, structural one.
        assert len(published) > published_at_flush
        assert ds.concurrent_saves == 1        # still flushing, nothing raced

        gate.set()
        await asyncio.wait_for(stopping, timeout=5.0)
        assert ds.saved == 1
    finally:
        beat_task.cancel()
        stop.set()
        thread.join(timeout=2.0)


# --- C2: an armed recorder is not a consumer -----------------------------

async def test_an_armed_recorder_cannot_manufacture_drop_counts():
    """C2. ARMED is where a session SITS, so an attached-but-not-committing
    subscriber would overflow its bounded queue continuously and count drops
    that mean nothing.

    Pointed at the state that would actually produce them: far more publishes
    than any queue is deep, with the recorder parked in ARMED the whole time.
    `skipped_frames` is the number Track D puts on the HUD, and an operator who
    learns to ignore it while parked will ignore it mid-take.
    """
    bus = _measured_bus()
    r = await _armed(bus=bus)
    token = bus.attach_producer("test-flood")
    for i in range(500):                       # >> DEFAULT_QUEUE_DEPTH
        token.publish(t_mono=1000.0 + i * 0.01,
                      arms={"left": _joints_block(0.0)}, goal_deg={})
    token.detach()

    st = r.status()
    assert st["state"] == ARMED
    assert st["skipped_frames"] == 0
    assert st["drops"] == {"cameras": {}, "arms": {}}
    assert r._tick_sub is None                 # it never subscribed at all


# --- the armed freeze going stale ----------------------------------------

async def test_teleop_stopping_under_an_armed_gate_drops_it_to_idle():
    tele = _MutableTeleop(DRIVING)
    r = await _armed(teleop=tele)
    assert r.status()["state"] == ARMED

    tele.status_dict = dict(STOPPED)
    st = r.status()
    assert st["state"] == IDLE
    assert st["episode_index"] is None
    assert "teleop stopped" in st["invalidated_reason"]


async def test_a_stale_armed_gate_refuses_to_ROLL():
    """The reconcile has to fail CLOSED, not merely report.

    Pointed at the path that writes data: if `roll` trusted the stored state
    instead of re-checking, a take would open against a frozen schema that no
    longer describes the rig — and nothing downstream would say so.
    """
    tele = _MutableTeleop(DRIVING)
    r = await _armed(teleop=tele)
    tele.status_dict = dict(STOPPED)
    with pytest.raises(RuntimeError, match="not armed"):
        await r.roll()
    assert r._task_handle is None
    assert r._dataset.frames == []


async def test_teleop_switching_arms_under_an_armed_gate_invalidates_it():
    """The frozen schema NAMES the pair it was built for, so a different pair
    is a different dataset shape — invisible at the column level because both
    are `left_*`/`right_*`."""
    tele = _MutableTeleop(DRIVING)
    r = await _armed(teleop=tele)
    tele.status_dict = {**DRIVING, "right_arm": None}
    st = r.status()
    assert st["state"] == IDLE
    assert "switched arms" in st["invalidated_reason"]


async def test_a_gate_armed_with_teleop_IDLE_is_never_invalidated_by_it():
    """The bring-up path: a schema take driven with the arms idle.

    Same asymmetry `_run` already makes with `teleop_was_running` — there is
    no transition to detect, so there is nothing to go stale. Without this
    exemption every bring-up take would disarm itself on the first status
    poll.
    """
    tele = _MutableTeleop(STOPPED)
    r = await _armed(teleop=tele)
    for _ in range(5):
        assert r.status()["state"] == ARMED
    await r.roll()
    assert r.status()["state"] == RECORDING
    await r.stop_episode(save=False)


async def test_invalidated_reason_never_fires_MID_TAKE():
    """Track C deleted a whole red-banner state on this promise.

    A mid-take teleop stop already saves up to the stop and closes the episode
    — behaviour that predates this port — so the mid-take signal is a SAVED
    take, never an invalidation.
    """
    tele = _MutableTeleop(DRIVING)
    r = await _armed(teleop=tele)
    await r.roll()
    tele.status_dict = dict(STOPPED)
    st = r.status()
    assert st["state"] == RECORDING            # the reconcile did not touch it
    assert st["invalidated_reason"] is None
    await r.stop_episode(save=False)


# --- the re-arm that is refused ------------------------------------------

async def test_a_save_whose_re_arm_is_refused_still_reports_the_SAVE():
    """The take was banked; the next gate could not open. Reporting that as a
    failure would be a lie about the take the operator just drove, so it is a
    200 that lands in idle and says why.

    Refused through the rate gate, which is the way it will actually happen: a
    rig that degraded during the take cannot open the next one.
    """
    ds = _FakeDataset()
    r = await _armed(dataset=ds)
    await r.roll()
    for _ in range(3):
        ds.add_frame({"task": "t"})
        r._state.episode_frames += 1

    r.tick_bus = TickBus()                     # no measurement -> arm refuses
    st = await r.stop_episode(save=True, rearm=True)
    assert ds.saved == 1                       # the take IS on disk
    assert st["state"] == IDLE
    assert "re-arm refused" in st["invalidated_reason"]
    assert st["episode_index"] is None


async def test_the_two_rate_refusals_are_different_types():
    """Both are a 409, but only one is worth telling the operator to retry.

    Separating them by matching message text is the thing that rots, so they
    are types. Both subclass `RuntimeError`, so every handler that predates
    them behaves exactly as it did.
    """
    assert issubclass(RateNotMeasuredYet, RuntimeError)
    assert issubclass(RateUnfaithful, RuntimeError)

    unmeasured = _gate_recorder(bus=TickBus())
    with pytest.raises(RateNotMeasuredYet):
        await unmeasured.arm("smoke/gate", "t")

    # 4.8 Hz measured, appended to a dataset written at 30 fps.
    slow = _gate_recorder(bus=_measured_bus(hz=4.8), dataset=_FakeDataset())
    slow._existing_fps = lambda repo_id: 30
    with pytest.raises(RateUnfaithful):
        await slow.arm("smoke/gate", "t")


# --- episode_uid ---------------------------------------------------------

async def test_the_uid_is_stamped_at_arm_time_and_is_one_value_per_episode():
    r = await _armed()
    uid = r.status()["episode_index"], r._state.episode_uid
    assert isinstance(uid[1], int)
    await r.roll()
    frames = [r._build_frame(_gate_sample()) for _ in range(4)]
    stamped = {int(f[EPISODE_UID_FEATURE][0]) for f in frames}
    assert stamped == {r._state.episode_uid}
    await r.stop_episode(save=False)


async def test_a_REDO_gets_a_new_uid_though_it_keeps_the_same_index():
    """Why the uid is stamped at ARM time and not at save time: at save time a
    redo and its keeper are indistinguishable in the ordering.

    The index is deliberately the same on both — that is what REDO means — so
    the uid is the only thing that can tell the discarded attempt from the one
    that was kept.
    """
    r = await _rolled_take()
    first_uid = r._state.episode_uid
    first_index = r.status()["episode_index"]

    st = await r.stop_episode(save=False, rearm=True)
    assert st["episode_index"] == first_index
    assert r._state.episode_uid != first_uid
    assert r._state.episode_uid > first_uid


async def test_uids_keep_increasing_when_the_clock_does_not():
    """Monotonic by +1 on collision. Two arms inside one microsecond collide,
    and `time.time()` is not monotonic — an NTP step backwards would otherwise
    hand a later episode a smaller id.

    Order wins over absolute accuracy: the column exists so recording ORDER
    survives a prune that renumbers `episode_index`, and an id that sorts
    wrongly has lost the only property it was added for.
    """
    r = _gate_recorder()
    r._last_episode_uid = None
    uids = [r._next_episode_uid() for _ in range(50)]
    assert uids == sorted(uids)
    assert len(set(uids)) == 50

    # A clock that jumped backwards by an hour.
    r._last_episode_uid = uids[-1] + 3_600_000_000
    assert r._next_episode_uid() > uids[-1] + 3_600_000_000


def test_the_uid_is_inert_to_training():
    """THE trap this column exists inside, asserted against lerobot's own
    classifier rather than against a string prefix.

    `dataset_to_policy_features` classifies by key PREFIX and `continue`s on
    anything it does not recognise. That `continue` is the entire reason the
    column is free. Under `observation.` it would be handed to the policy as an
    input feature and we would be training on our own episode ids — so the
    claim is "the policy never sees it", and the only thing that can answer it
    is the function that decides.
    """
    from lerobot.datasets.feature_utils import dataset_to_policy_features

    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    r = DatasetRecorder(tick_bus=_measured_bus(), telemetry=_FakeTelemetry(arms),
                        human_teleop=_FakeHumanTeleop(dict(STOPPED)),
                        cameras=_FakeCameras([]))
    features = r._build_features([])
    assert EPISODE_UID_FEATURE in features

    policy_features = dataset_to_policy_features(features)
    assert EPISODE_UID_FEATURE not in policy_features
    # And the columns that SHOULD reach the policy still do, so the assertion
    # above is about this key rather than about a classifier that returned
    # nothing.
    assert "observation.state" in policy_features
    assert "action" in policy_features


def test_namespacing_the_uid_would_feed_it_to_the_policy():
    """The counter-case, which is what makes the test above mean something.

    Without this, `episode_uid not in policy_features` would pass just as
    happily against a classifier that dropped everything.
    """
    from lerobot.datasets.feature_utils import dataset_to_policy_features

    trapped = dataset_to_policy_features({
        f"observation.{EPISODE_UID_FEATURE}": {
            "dtype": "int64", "shape": (1,), "names": None,
        },
    })
    assert f"observation.{EPISODE_UID_FEATURE}" in trapped


def _gate_sample():
    from haller_hmi.tick import TickSample
    return TickSample(
        seq=0, t_mono=0.0, t_unix=0.0,
        arms={"left": _joints_block(0.0), "right": _joints_block(0.0)},
        base={}, goal_deg={}, arm_errors={},
    )


# --- the status payload the other two tracks type against ----------------

async def test_status_carries_every_key_the_contract_names():
    """Tracks C and D both type against this shape, so an absent key is a
    silently-undefined field on someone else's HUD rather than an error."""
    r = await _armed()
    st = r.status()
    for key in ("state", "recording", "episode_index", "invalidated_reason",
                "fps_declared", "fps_measured", "skipped_frames", "drops",
                "alerts", "record_rate_tolerance"):
        assert key in st, key
    assert st["drops"] == {"cameras": {}, "arms": {}}
    assert st["recording"] is (st["state"] == RECORDING)


async def test_recording_is_exactly_the_state_and_cannot_drift_from_it():
    """The shipped boolean and the new three-valued state are two spellings of
    one fact, so `recording` is DERIVED. Two fields would be two things to keep
    in step, and the first stop path that updated one and not the other would
    leave the HUD reading `recording` over a closed episode."""
    r = await _armed()
    assert r.status()["recording"] is False
    await r.roll()
    assert r.status()["recording"] is True
    await r.stop_episode(save=False)
    assert r.status()["recording"] is False
    with pytest.raises(AttributeError):
        r._state.recording = True          # there is no setter to drift through


# --- the uid on real parquet ---------------------------------------------

async def test_the_uid_survives_a_round_trip_through_parquet(tmp_path):
    """The column is only worth stamping if it comes back, so this is the real
    lerobot path — create, write, save, finalize, reload — rather than a fake.

    int64 specifically: microseconds since 1970 overflow float32's
    exact-integer range by nine orders of magnitude, so a uid stored as a float
    would come back rounded and no longer compare equal to itself.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    arms = _FakeArms({"left": _FakeArm(SIX), "right": _FakeArm(SIX)})
    r = DatasetRecorder(
        tick_bus=_measured_bus(), telemetry=_FakeTelemetry(arms, hz=20.0),
        human_teleop=_FakeHumanTeleop(dict(STOPPED)),
        cameras=_FakeCameras([]), root=str(tmp_path / "ds"),
    )
    await r.arm("smoke/uid", "lift the cube")
    uid = r._state.episode_uid
    await r.roll()
    for _ in range(4):
        r._dataset.add_frame({
            "observation.state": np.zeros(12, dtype=np.float32),
            "action": np.zeros(12, dtype=np.float32),
            "observation.effort": np.zeros(12, dtype=np.float32),
            "observation.base": np.zeros(2, dtype=np.float32),
            "observation.wall_clock": np.zeros(1, dtype=np.float32),
            EPISODE_UID_FEATURE: np.asarray([uid], dtype=np.int64),
            "task": "lift the cube",
        })
        r._state.episode_frames += 1
    await r.stop_episode(save=True)
    r.close()

    reloaded = LeRobotDataset("smoke/uid", root=str(tmp_path / "ds"))
    assert EPISODE_UID_FEATURE in reloaded.meta.features
    assert reloaded.meta.features[EPISODE_UID_FEATURE]["dtype"] == "int64"
    values = {int(reloaded[i][EPISODE_UID_FEATURE]) for i in range(len(reloaded))}
    assert values == {uid}, "the uid must come back bit-for-bit, not rounded"
