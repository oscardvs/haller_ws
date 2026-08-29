# hmi/backend/tests/test_policy_bridge.py
"""The production callables behind the policy ingest, asserted from C0-C4.

Harness modelled on `tests/test_policy_ingest.py`: where the wire matters, the
CHILD'S OWN `IngestClient`/`decode_observation`/`_decode_image` are the judges,
because a fake reader built from this module's assumptions is the blind spot
Track B's blind criteria exist to avoid.

What the bridge adds on top of the ingest, and what is pinned here:

* `bus_conflict` finally gives `lease.bus_conflict` its production caller (C1),
  plus the calibration branch lease cannot see.
* `observe` produces a coherent sample — canonical state order, images keyed by
  `dataset_feature_key` — or NOTHING (invariant 9: withheld, never padded).
* `submit` rides `ArmHandle.send_goal` with the DEFAULT speed cap — a policy is
  not an operator's hand — and refuses a STOP arm, disabled torque, and a side
  that names no arm, without ever enabling anything itself.
* the tick-bus producer claim stands the `IdleSampler` down for the run — the
  2026-08-29 half-duplex collision, kept fixed.
"""
from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from haller_hmi.config import CameraConfig
from haller_hmi.policy_bridge import (
    PRODUCER_NAME,
    FiniteActionIngest,
    PolicyBridge,
    ingest_port_from_env,
)
from haller_hmi.policy_ingest import ACTION_TYPE, PolicyIngest, PolicyRefusal
from haller_hmi.recorder import SO101_JOINT_ORDER
from haller_hmi.safety import Mode, ModeGuard
from haller_hmi.tick import IdleSampler, TickBus

# --- fakes ----------------------------------------------------------------

class FakeArmHandle:
    """Just enough of `ArmHandle` for the bridge: limits, guard, snapshots,
    and a `send_goal` that records its kwargs — the speed-cap assertion needs
    to see exactly what was passed, not what a default filled in."""

    def __init__(self, positions=None, mode=Mode.MANUAL):
        self.joint_limits_deg = {
            j: ((0.0, 100.0) if j == "gripper" else (-110.0, 110.0))
            for j in SO101_JOINT_ORDER
        }
        self.guard = ModeGuard(mode)
        self.torque_enabled = True
        self.goal_calls: list[tuple[dict, dict]] = []
        self._pos = dict(positions
                         or {j: 0.0 for j in SO101_JOINT_ORDER})

    def state_snapshot(self):
        return {
            "mode": self.guard.mode.value,
            "torque": self.torque_enabled,
            "joints": {
                j: {"pos": p,
                    "min": self.joint_limits_deg[j][0],
                    "max": self.joint_limits_deg[j][1],
                    "torque": True}
                for j, p in self._pos.items()
            },
        }

    def send_goal(self, goal_deg, **kwargs):
        # Same order as the real one: the mode guard refuses BEFORE anything
        # is recorded as sent.
        self.guard.assert_manual()
        self.goal_calls.append((dict(goal_deg), dict(kwargs)))
        out = {}
        for j, v in goal_deg.items():
            if j not in self.joint_limits_deg:
                continue
            lo, hi = self.joint_limits_deg[j]
            out[j] = max(lo, min(hi, float(v)))
        return out


class FakeArms:
    def __init__(self, handles):
        self._h = dict(handles)

    def keys(self):
        return self._h.keys()

    def __getitem__(self, key):
        if key not in self._h:
            raise KeyError(f"unknown arm id {key!r}")
        return self._h[key]

    def values(self):
        return list(self._h.values())


def _jpeg_bytes():
    import cv2
    import numpy as np

    frame = np.full((8, 8, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


class FakeCamera:
    def __init__(self, cfg: CameraConfig, jpeg: bytes | None):
        self.cfg = cfg
        self.active = True
        self._jpeg = jpeg

    def latest_rgb(self, max_age_ms: int = 500):  # presence is the filter
        return None

    def latest_jpeg(self, max_age_ms: int = 500):
        return self._jpeg


class FakeCameraHub:
    """The recorder's camera hub surface: keys/__getitem__/is_recorded."""

    def __init__(self, cams):
        self._cams = dict(cams)

    def keys(self):
        return self._cams.keys()

    def __getitem__(self, cam_id):
        return self._cams[cam_id]

    def is_recorded(self, cam_id):
        return bool(getattr(self._cams[cam_id].cfg, "record", True))


def _top_camera(jpeg=None):
    cfg = CameraConfig(id="base_front", role="base", source="opencv",
                       dataset_key="top")
    return FakeCameraHub({"base_front": FakeCamera(
        cfg, _jpeg_bytes() if jpeg is None else jpeg)})


def _bridge(*, arms, cameras=None, recorder=None, calibration=None,
            sessions=(), devices=(), bus=None):
    return PolicyBridge(
        arms=arms,
        tick_bus=bus if bus is not None else TickBus(),
        get_cameras=lambda: cameras,
        get_recorder=lambda: recorder,
        calibration=calibration or SimpleNamespace(current=None),
        sessions=sessions,
        devices=list(devices),
    )


def _solo_positions():
    """Distinct value per joint, so order errors cannot cancel out."""
    return {j: float(10 * i) for i, j in enumerate(SO101_JOINT_ORDER)}


# --- observe: the solo layout the kit ACTs were trained on ----------------

def test_observe_carries_solo_state_in_canonical_order_and_a_top_image():
    """One arm -> the 6-dim UNPREFIXED layout (`RigSpec` classifies unprefixed
    columns as rig "solo" — so101_pick_cube's own shape), joints in
    SO101_JOINT_ORDER; the image rides under the camera's dataset_feature_key.

    Judged by the CHILD's own readers end to end: `decode_observation` on the
    wire message and `_decode_image` on the payload, so "observation-ready"
    means the inference loop can consume it, not that a dict has keys.
    """
    from haller_hmi.runners.rollout_runner import _decode_image, decode_observation

    bridge = _bridge(arms=FakeArms({"left": FakeArmHandle(_solo_positions())}),
                     cameras=_top_camera())
    payload = bridge.observe()
    assert payload is not None
    assert payload["state"] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert payload["state_names"] == list(SO101_JOINT_ORDER)

    ing = PolicyIngest(bus_conflict=lambda: None, submit=lambda *a: {},
                       observe=lambda: None, port=0)
    decoded = decode_observation(ing.observation_message(0, payload))
    assert decoded is not None, "the child could not decode the observation"
    assert decoded["state"] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    image = _decode_image(decoded["images"]["top"])
    assert image.shape == (8, 8, 3)


def test_observe_on_a_bimanual_rig_is_twelve_dim_and_side_prefixed():
    """Two arms -> `recorder._state_names` semantics exactly: sides left then
    right, `{side}_{joint}`, 12 dims."""
    left = FakeArmHandle({j: 1.0 for j in SO101_JOINT_ORDER})
    right = FakeArmHandle({j: 2.0 for j in SO101_JOINT_ORDER})
    bridge = _bridge(arms=FakeArms({"left": left, "right": right}))
    payload = bridge.observe()
    assert payload is not None
    assert payload["state_names"] == (
        [f"left_{j}" for j in SO101_JOINT_ORDER]
        + [f"right_{j}" for j in SO101_JOINT_ORDER])
    assert payload["state"] == [1.0] * 6 + [2.0] * 6


def test_observe_withholds_the_sample_rather_than_padding_a_hole():
    """Invariant 9. A dropped joint or a stale recorded camera is a WITHHELD
    observation — the child times out on a named stall — never a fabricated
    state the rig was not in."""
    # A read that lost the elbow: the snapshot simply has no such joint.
    positions = _solo_positions()
    del positions["elbow_flex"]
    bridge = _bridge(arms=FakeArms({"left": FakeArmHandle(positions)}),
                     cameras=_top_camera())
    assert bridge.observe() is None

    # A recorded camera with no fresh frame.
    hub = FakeCameraHub({"base_front": FakeCamera(
        CameraConfig(id="base_front", role="base", source="opencv",
                     dataset_key="top"), jpeg=None)})
    bridge = _bridge(arms=FakeArms({"left": FakeArmHandle(_solo_positions())}),
                     cameras=hub)
    assert bridge.observe() is None


def test_observe_skips_cameras_outside_the_recorded_set():
    """`record: false` views exist to teleop from, not to feed a policy —
    same filter as the recorder's `_active_camera_specs`."""
    hud_only = FakeCamera(
        CameraConfig(id="wrist_left", role="wrist", source="opencv",
                     record=False), _jpeg_bytes())
    top = FakeCamera(
        CameraConfig(id="base_front", role="base", source="opencv",
                     dataset_key="top"), _jpeg_bytes())
    hub = FakeCameraHub({"wrist_left": hud_only, "base_front": top})
    bridge = _bridge(arms=FakeArms({"left": FakeArmHandle(_solo_positions())}),
                     cameras=hub)
    payload = bridge.observe()
    assert payload is not None
    assert sorted(payload["images"]) == ["top"]


# --- submit: the commit chain's front door, at the DEFAULT cap ------------

def test_submit_routes_to_send_goal_WITHOUT_a_speed_cap_override():
    """A policy is not an operator's hand: no `speed_cap_deg_s` is passed, so
    `motion.max_speed_deg_s` — the discrete-move default — governs. Asserted
    on the recorded kwargs, because a wrong default would satisfy any
    assertion made on the committed values alone."""
    handle = FakeArmHandle()
    bridge = _bridge(arms=FakeArms({"left": handle}))
    out = bridge.submit("r1", 3, {"left": {"shoulder_pan": 12.5, "gripper": 40.0}})
    assert len(handle.goal_calls) == 1
    goal, kwargs = handle.goal_calls[0]
    assert goal == {"shoulder_pan": 12.5, "gripper": 40.0}
    assert "speed_cap_deg_s" not in kwargs, (
        "the submit path must keep the default discrete-move cap")
    side = out["sides"]["left"]
    assert side["arm"] == "left"
    assert side["committed"] == {"shoulder_pan": 12.5, "gripper": 40.0}
    assert out["seq"] == 3


def test_submit_attributes_an_alteration_to_a_named_stage():
    """C3's second clause: a committed value that differs from the sent one is
    attributed to the stage that imposed it, not merely 'different'."""
    handle = FakeArmHandle()
    bridge = _bridge(arms=FakeArms({"left": handle}))
    out = bridge.submit("r1", 0, {"left": {"shoulder_pan": 500.0}})
    side = out["sides"]["left"]
    assert side["committed"]["shoulder_pan"] == 110.0     # the limit, imposed
    assert side["altered"] == {"shoulder_pan": "clamp"}


def test_submit_refuses_a_side_with_no_arm_rather_than_guessing():
    """The two arms are 40 cm apart; a rollout aimed at the wrong one is a
    collision. On the solo rig only 'left' exists (config.solo-real.yaml)."""
    handle = FakeArmHandle()
    bridge = _bridge(arms=FakeArms({"left": handle}))
    out = bridge.submit("r1", 0, {"right": {"shoulder_pan": 1.0}})
    assert "refused" in out["sides"]["right"]
    assert "right" in out["sides"]["right"]["refused"]
    assert handle.goal_calls == []


def test_submit_never_writes_to_a_STOP_arm_and_never_enables_torque():
    """The mode guard STAYS: an E-STOPPED arm refuses every policy frame. And
    torque is never enabled from this path — a rollout that silently
    re-energised an arm someone made limp is the lunge incident again."""
    stopped = FakeArmHandle(mode=Mode.STOP)
    bridge = _bridge(arms=FakeArms({"left": stopped}))
    out = bridge.submit("r1", 0, {"left": {"shoulder_pan": 1.0}})
    assert "manual required" in out["sides"]["left"]["refused"]
    assert stopped.goal_calls == []

    limp = FakeArmHandle()
    limp.torque_enabled = False
    bridge = _bridge(arms=FakeArms({"left": limp}))
    out = bridge.submit("r1", 0, {"left": {"shoulder_pan": 1.0}})
    assert "torque disabled" in out["sides"]["left"]["refused"]
    assert limp.goal_calls == []
    assert limp.torque_enabled is False    # looked at, never flipped


# --- bus_conflict: lease's first production caller ------------------------

def test_bus_conflict_refuses_teleop_calibration_and_an_open_episode():
    """Each branch by the sentence the operator will actually read. The
    calibration branch is the bridge's own — a sweep owns the serial line the
    way a session owns the tick, and `lease.bus_conflict` cannot see it."""
    rec_box = {"r": SimpleNamespace(
        status=lambda: {"recording": False, "repo_id": ""},
        _episode_open=False)}
    session = SimpleNamespace(running=False)
    calibration = SimpleNamespace(current=None)
    bridge = PolicyBridge(
        arms=FakeArms({"left": FakeArmHandle()}),
        tick_bus=TickBus(),
        get_cameras=lambda: None,
        get_recorder=lambda: rec_box["r"],
        calibration=calibration,
        sessions=(session,),
        devices=[],
    )
    assert bridge.bus_conflict() is None

    session.running = True
    conflict = bridge.bus_conflict()
    assert conflict is not None and "teleop session is driving" in conflict
    session.running = False

    calibration.current = SimpleNamespace(arm_id="left")
    conflict = bridge.bus_conflict()
    assert conflict is not None and "calibrated" in conflict
    calibration.current = None

    rec_box["r"] = SimpleNamespace(
        status=lambda: {"recording": True, "repo_id": "local/x"},
        _episode_open=True)
    conflict = bridge.bus_conflict()
    assert conflict is not None and "episode is being recorded" in conflict

    rec_box["r"] = SimpleNamespace(
        status=lambda: {"recording": False, "repo_id": ""},
        _episode_open=False)
    assert bridge.bus_conflict() is None


def test_bus_conflict_refuses_while_a_discrete_move_ramps():
    """A MoveExecutor ramp is its own OS thread writing the bus — the same
    two-writers hazard as a session, invisible to lease.bus_conflict. The
    reverse half (the /arm routes 409ing mid-rollout) is pinned in the
    operator-doors test below."""
    handle = FakeArmHandle()
    handle.executor = SimpleNamespace(is_running=True)
    bridge = _bridge(arms=FakeArms({"left": handle}))
    conflict = bridge.bus_conflict()
    assert conflict is not None and "discrete move in progress" in conflict

    handle.executor.is_running = False
    assert bridge.bus_conflict() is None


def test_a_session_that_reports_running_through_status_alone_still_refuses():
    """`SimLeaderTeleop` has no top-level `.running`; the status() fallback is
    what keeps it from being invisible to the refusal."""
    session = SimpleNamespace(status=lambda: {"running": True})
    bridge = _bridge(arms=FakeArms({"left": FakeArmHandle()}),
                     sessions=(session,))
    conflict = bridge.bus_conflict()
    assert conflict is not None and "teleop session is driving" in conflict


# --- the producer claim: the idle sampler stands down ---------------------

def test_the_claim_stands_the_idle_sampler_down_and_release_hands_back():
    """The 2026-08-29 collision, kept fixed: while a policy drives, the idle
    sampler must not read beside it on the half-duplex line. `tick_once`
    checks the producer BEFORE it samples, so the assertion is that the
    sample callable was never invoked — the serial line was never touched."""
    bus = TickBus()
    reads: list[int] = []

    def sample():
        reads.append(1)
        return {"arms": {}, "goal_deg": {}}

    sampler = IdleSampler(bus, sample=sample, hz=20.0)
    assert sampler.tick_once() is not None            # idle: sampler owns it

    bridge = _bridge(arms=FakeArms({"left": FakeArmHandle(_solo_positions())}),
                     bus=bus)
    assert bridge.observe() is not None
    assert bus.producer_name == PRODUCER_NAME
    before = len(reads)
    assert sampler.tick_once() is None                # stood down
    assert len(reads) == before

    bridge.release()
    assert bus.producer_name is None
    assert sampler.tick_once() is not None            # handed back


def test_observations_publish_ticks_with_the_policy_goals_as_goal_deg():
    """Telemetry keeps painting during a rollout, and the tick's `goal_deg`
    is what the POLICY commanded — the HUD tells the truth about the leader."""
    bus = TickBus()
    bridge = _bridge(arms=FakeArms({"left": FakeArmHandle(_solo_positions())}),
                     bus=bus)
    bridge.submit("r1", 0, {"left": {"shoulder_pan": 33.0}})
    assert bridge.observe() is not None
    sample = bus.latest()
    assert sample is not None
    assert dict(sample.goal_deg["left"]) == {"shoulder_pan": 33.0}
    assert "left" in sample.arms


# --- the wire, end to end against the child's own client ------------------

async def test_a_solo_bridge_serves_the_real_child_client_end_to_end():
    """The integration seam in one test: the REAL `IngestClient` handshakes
    against a `PolicyIngest` wired with the REAL bridge callables, receives an
    observation the child's own decoders accept — state 6-dim in canonical
    order, image keyed "top" — and a submitted action lands on `send_goal`
    with no speed-cap override."""
    import asyncio

    from haller_hmi.runners.rollout_runner import (
        IngestClient,
        _decode_image,
        action_message,
        decode_observation,
    )

    handle = FakeArmHandle(_solo_positions())
    bridge = _bridge(arms=FakeArms({"left": handle}), cameras=_top_camera())
    ing = PolicyIngest(bus_conflict=bridge.bus_conflict, submit=bridge.submit,
                       observe=bridge.observe, on_session_end=bridge.release,
                       port=0)
    bridge.attach_ingest(ing)
    await ing.start()
    client = IngestClient("127.0.0.1", ing.port)
    try:
        await asyncio.to_thread(client.connect)
        hello = {"type": "policy_hello", "run_id": "r-e2e", "unit": "deg",
                 "control_hz_declared": 30.0, "rig": "solo"}
        ack = await asyncio.to_thread(client.handshake, hello)
        assert ack["server_pid"] == os.getpid()

        msg = await asyncio.to_thread(client.recv, 5.0)
        assert msg is not None and msg["type"] == "policy_observation"
        decoded = decode_observation(msg)
        assert decoded is not None
        assert decoded["state"] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        assert msg["state_names"] == list(SO101_JOINT_ORDER)
        assert _decode_image(decoded["images"]["top"]).shape == (8, 8, 3)

        await asyncio.to_thread(
            client.send,
            action_message("r-e2e", 0, {"left": {"shoulder_pan": 12.0}}))
        for _ in range(100):
            if ing.actions_committed >= 1:
                break
            await asyncio.sleep(0.02)
        assert ing.actions_committed == 1
        goal, kwargs = handle.goal_calls[0]
        assert goal == {"shoulder_pan": 12.0}
        assert "speed_cap_deg_s" not in kwargs
        assert ing.last_commit["sides"]["left"]["committed"] == {"shoulder_pan": 12.0}
    finally:
        client.close()
        await ing.stop()
        bridge.release()


# --- the doors, kept shut in BOTH directions ------------------------------
#
# The 2026-08-29 review's findings, pinned. `bus_conflict` always refused a
# rollout while an operator path ran; everything below is the reverse
# direction — and the two teardown paths that used to leave a run's machinery
# standing after the run was gone.

def _wired_solo(handle=None):
    """A real `FiniteActionIngest` over a real bridge on a solo FakeArm rig,
    port 0. The wire tests below all start from here."""
    handle = handle if handle is not None else FakeArmHandle(_solo_positions())
    bus = TickBus()
    bridge = _bridge(arms=FakeArms({"left": handle}), bus=bus)
    ing = FiniteActionIngest(
        bus_conflict=bridge.bus_conflict, submit=bridge.submit,
        observe=bridge.observe, on_session_end=bridge.release, port=0)
    bridge.attach_ingest(ing)
    return handle, bus, bridge, ing


def _hello(run_id):
    return {"type": "policy_hello", "run_id": run_id, "unit": "deg",
            "control_hz_declared": 30.0, "rig": "solo"}


async def test_shutdown_with_a_child_STILL_CONNECTED_completes_and_frees_the_bus():
    """The lifespan hang, kept fixed. On 3.12 `Server.wait_closed()` waits for
    the connection handlers, and the handler's stream loop runs until the
    active run is nulled — which `PolicyIngest.stop()` does only AFTER the
    wait. With a child connected that is a circular wait: Ctrl-C/SIGTERM
    (or --reload) hung at the lifespan's first step while the policy kept
    committing goals, and on --reload the new process took the arms beside
    the old one still writing. `PolicyBridge.shutdown` nulls the run FIRST.

    The elapsed bound is the assertion, not decoration: at 2 s the internal
    drain belt has fired, which means the loop did not exit on the nulled run
    and the deadlock is back. The outer wait_for turns a full regression into
    a failure rather than a hung suite.
    """
    from haller_hmi.runners.rollout_runner import IngestClient, action_message

    _handle, bus, bridge, ing = _wired_solo()
    await ing.start()
    client = IngestClient("127.0.0.1", ing.port)
    try:
        await asyncio.to_thread(client.connect)
        await asyncio.to_thread(client.handshake, _hello("r-shutdown"))
        await asyncio.to_thread(
            client.send,
            action_message("r-shutdown", 0, {"left": {"shoulder_pan": 5.0}}))
        for _ in range(100):
            if ing.actions_committed >= 1:
                break
            await asyncio.sleep(0.02)
        assert ing.actions_committed == 1
        assert bus.producer_name == PRODUCER_NAME     # mid-run, claim held

        t0 = time.perf_counter()
        await asyncio.wait_for(bridge.shutdown(), 5.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.9, (
            f"shutdown took {elapsed:.2f}s — the 2 s drain belt fired, so the "
            f"stream loop no longer exits on the nulled run")
        assert ing.active_run_id is None
        assert ing.status()["listening"] is False
        assert bus.producer_name is None              # the claim went with it
    finally:
        client.close()


async def test_a_malformed_frame_no_longer_wedges_the_operator_paths():
    """The leaked producer claim, healed. A wrong-run_id frame refuses the RUN
    on the ingest path that nulls `active_run_id` before the handler's finally
    compares it — so `on_session_end` never fires and the "policy-rollout"
    claim outlives the run: the idle sampler stays stood down, `measured_hz`
    decays until arming refuses (invariant 10), and every session start 409s
    against a rollout that no longer exists, until another handshake or a
    backend restart. The watchdog bounds that: the bus frees on its own, and
    a session's own `attach_producer` succeeds again."""
    from haller_hmi.runners.rollout_runner import IngestClient

    handle, bus, bridge, ing = _wired_solo()
    await ing.start()
    client = IngestClient("127.0.0.1", ing.port)
    try:
        await asyncio.to_thread(client.connect)
        await asyncio.to_thread(client.handshake, _hello("r-leak"))

        # The bad frame: right shape, wrong run. The refusal must cross the
        # wire (skipping the observations streaming beside it)...
        await asyncio.to_thread(client.send, {
            "type": ACTION_TYPE, "seq": 0, "t_ms": 1, "run_id": "someone-else",
            "action": {"left": {"shoulder_pan": 1.0}}})
        detail = None
        for _ in range(200):
            msg = await asyncio.to_thread(client.recv, 0.05)
            if msg is not None and msg.get("type") == "policy_refused":
                detail = msg["detail"]
                break
        assert detail is not None and "admitted run" in detail
        assert ing.active_run_id is None              # the run is over...

        # ...and the claim it leaked frees WITHOUT another handshake.
        for _ in range(150):
            if bus.producer_name is None:
                break
            await asyncio.sleep(0.02)
        assert bus.producer_name is None, (
            "the watchdog never released the dead run's tick-bus claim")
        token = bus.attach_producer("human-teleop")   # the wedge, gone
        token.detach()
        assert handle.goal_calls == []                # nothing ever committed
    finally:
        client.close()
        await ing.stop()
        bridge.release()


def test_rollout_conflict_names_the_run_and_heals_a_stale_claim():
    """The reverse half of `bus_conflict`, unit-level. While a run streams,
    the sentence names it — the operator reads WHAT to stop. Once the run is
    gone, the same call releases a claim that outlived it before answering,
    so a dead rollout never costs an operator start a 409."""
    bus = TickBus()
    bridge = _bridge(arms=FakeArms({"left": FakeArmHandle(_solo_positions())}),
                     bus=bus)
    ing = SimpleNamespace(active_run_id="r-live")
    bridge.attach_ingest(ing)
    assert bridge.observe() is not None               # takes the claim
    conflict = bridge.rollout_conflict()
    assert conflict is not None and "r-live" in conflict
    assert bus.producer_name == PRODUCER_NAME         # a live run keeps it

    ing.active_run_id = None                          # the leak shape
    assert bridge.rollout_conflict() is None
    assert bus.producer_name is None                  # healed on the way out


def test_non_finite_targets_refuse_the_run_by_name():
    """stdlib `json.loads` accepts NaN/Infinity literals and the frozen decode
    path's bare `float(v)` passes them, so an unguarded chain resolves them at
    the clamp — `max(lo, min(hi, nan))` is the UPPER limit — and a NaN-emitting
    checkpoint (the classic fresh-training failure) becomes "slew every NaN
    joint to its limit at the full default cap, every tick" instead of a
    refused run. The VR wire refuses exactly this class
    (`human_teleop._usable_side`); `FiniteActionIngest` is the policy wire's
    same gate, riding the module's own malformed-frame rule."""
    ing = FiniteActionIngest(bus_conflict=lambda: None, submit=lambda *a: {},
                             observe=lambda: None, port=0)
    ing.active_run_id = "r1"
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(PolicyRefusal, match="non-finite"):
            ing.decode_action({
                "type": ACTION_TYPE, "seq": 0, "t_ms": 1, "run_id": "r1",
                "action": {"left": {"shoulder_pan": bad, "gripper": 10.0}}})

    # A finite frame decodes exactly as the base class would.
    seq, act = ing.decode_action({
        "type": ACTION_TYPE, "seq": 7, "t_ms": 1, "run_id": "r1",
        "action": {"left": {"shoulder_pan": 1.5}}})
    assert (seq, act) == (7, {"left": {"shoulder_pan": 1.5}})


async def test_a_NaN_frame_crosses_the_wire_as_a_refusal_and_commits_nothing():
    """End to end with the child's own client: stdlib json EMITS the NaN
    literal (`json.dumps` does not refuse it), stdlib json parses it back, and
    the run is refused with the sentence on the wire — the child's own
    `_next_observation` turns exactly this message into its exit sentence. No
    goal reaches `send_goal`, and the C3 counter stays honest at zero."""
    from haller_hmi.runners.rollout_runner import IngestClient, action_message

    handle, _bus, bridge, ing = _wired_solo()
    await ing.start()
    client = IngestClient("127.0.0.1", ing.port)
    try:
        await asyncio.to_thread(client.connect)
        await asyncio.to_thread(client.handshake, _hello("r-nan"))
        await asyncio.to_thread(
            client.send,
            action_message("r-nan", 0,
                           {"left": {"shoulder_pan": float("nan"),
                                     "gripper": 40.0}}))
        detail = None
        for _ in range(200):
            msg = await asyncio.to_thread(client.recv, 0.05)
            if msg is not None and msg.get("type") == "policy_refused":
                detail = msg["detail"]
                break
        assert detail is not None and "non-finite" in detail
        assert "shoulder_pan" in detail               # named, not generic
        assert handle.goal_calls == []                # nothing reached the arm
        assert ing.actions_committed == 0
        assert ing.actions_received == 1              # received != committed
        assert ing.active_run_id is None              # the RUN was refused
    finally:
        client.close()
        await ing.stop()
        bridge.release()


# --- the lifespan wiring, through the real app ----------------------------

def _prime_lifespan(monkeypatch, srv):
    """Patch the heavy lifespan collaborators so the REAL `_lifespan` runs the
    real ingest wiring against the conftest mock rig."""
    srv.arms.world.return_value = None
    monkeypatch.setattr(srv.cfg, "arms", [])

    hub = MagicMock()
    hub.keys.return_value = []
    monkeypatch.setattr(srv, "CameraManager", MagicMock(return_value=hub))

    telemetry = MagicMock()
    telemetry.stop = AsyncMock()
    monkeypatch.setattr(srv, "TelemetryBroadcaster",
                        MagicMock(return_value=telemetry))
    monkeypatch.setattr(srv, "IdleSampler", MagicMock())

    recorder = MagicMock()
    recorder.status.return_value = {"recording": False, "repo_id": ""}
    recorder._episode_open = False        # lease reads this attr verbatim
    recorder.stop_episode = AsyncMock()
    monkeypatch.setattr(srv, "DatasetRecorder", MagicMock(return_value=recorder))

    # The three sessions must answer "not running" the way the real ones do —
    # a bare Mock's truthy attributes would refuse every handshake.
    srv.teleop.running = False
    srv.human_teleop.running = False
    srv.human_teleop.tick_bus = TickBus()


def test_the_lifespan_starts_the_ingest_and_a_child_handshakes_through_it(
        app_with_mocks, monkeypatch):
    """Deliverable-1 evidence: entering the app's REAL lifespan constructs and
    starts `PolicyIngest` with the bridge's callables, a real `IngestClient`
    completes the handshake against it, and the wired `bus_conflict` still
    refuses (C2) — the sentence crossing the wire verbatim. Shutdown then
    stops it cleanly before the arms drain (the `with` exit is the proof: it
    runs the whole teardown and nothing hangs or raises)."""
    import haller_hmi.server as srv
    from haller_hmi.runners.rollout_runner import IngestClient

    # Port 0 via the same env var the child dials by, so the suite never
    # squats the real 8781 (`test_policy_ingest` pins that constant).
    monkeypatch.setenv("HALLER_POLICY_INGEST", "tcp://127.0.0.1:0")
    assert ingest_port_from_env(8781) == 0
    _prime_lifespan(monkeypatch, srv)

    with app_with_mocks:
        assert srv.policy_ingest is not None
        status = srv.policy_ingest.status()
        assert status["listening"] is True
        port = srv.policy_ingest.port
        assert port > 0

        hello = {"type": "policy_hello", "run_id": "r-lifespan", "unit": "deg",
                 "control_hz_declared": 30.0, "rig": "solo"}

        client = IngestClient("127.0.0.1", port)
        try:
            client.connect()
            ack = client.handshake(hello)
            assert ack["server_pid"] == os.getpid()
            assert srv.policy_ingest.active_run_id == "r-lifespan"
        finally:
            client.close()
        for _ in range(100):                    # the goodbye is the close
            if srv.policy_ingest.active_run_id is None:
                break
            time.sleep(0.02)
        assert srv.policy_ingest.active_run_id is None

        # C2 through the whole stack: a live teleop session refuses the next
        # child, with lease's sentence intact on the wire.
        srv.human_teleop.running = True
        client = IngestClient("127.0.0.1", port)
        try:
            client.connect()
            with pytest.raises(SystemExit) as e:
                client.handshake(hello)
            assert "teleop session is driving" in str(e.value)
        finally:
            client.close()
    assert srv.policy_ingest is None            # shutdown nulled it


def test_operator_start_routes_409_while_a_rollout_streams(
        app_with_mocks, monkeypatch):
    """The reverse doors, shut through the real app. While a run is admitted,
    leader-follower teleop, sim teleop, calibration and VR teleop starts all
    409 with the sentence naming the run — none of the first three attaches a
    tick-bus producer (their loops write from their own OS threads; a sweep
    cuts torque), so without the route check nothing refuses them and the
    result is two writers interleaving on one half-duplex Feetech line. And
    the door REOPENS once the child is gone: a closed stream ends the run,
    and the same check heals rather than 409ing against a dead rollout."""
    import haller_hmi.server as srv
    from haller_hmi.runners.rollout_runner import IngestClient

    monkeypatch.setenv("HALLER_POLICY_INGEST", "tcp://127.0.0.1:0")
    _prime_lifespan(monkeypatch, srv)

    with app_with_mocks as http:
        # The lifespan constructs the GATED ingest, not the base class — the
        # finiteness gate exists only if the wiring says so.
        assert isinstance(srv.policy_ingest, FiniteActionIngest)

        child = IngestClient("127.0.0.1", srv.policy_ingest.port)
        try:
            child.connect()
            child.handshake({"type": "policy_hello", "run_id": "r-doors",
                             "unit": "deg", "control_hz_declared": 30.0,
                             "rig": "solo"})
            assert srv.policy_ingest.active_run_id == "r-doors"

            starts = [
                ("/teleop/start", {"leader": "right", "follower": "right"}),
                ("/teleop/sim/start", {"follower": "right", "hz": 30.0,
                                       "leader": {"source": "mouse",
                                                  "arm_name": "left"}}),
                ("/calibration/right/start", None),
                ("/teleop/human/start", {"left_arm": "right"}),
                # Discrete moves are writers too: a MoveExecutor ramp thread
                # beside the ingest's commits is the same half-duplex
                # corruption as a session. Gate checked BEFORE the preset
                # lookup, so even an unknown preset 409s here, not 404s.
                ("/arm/right/goal", {"shoulder_pan": 1.0}),
                ("/arm/right/home", None),
                ("/arm/right/preset", {"name": "anything"}),
            ]
            for path, body in starts:
                r = http.post(path, json=body) if body is not None \
                    else http.post(path)
                assert r.status_code == 409, f"{path}: {r.status_code} {r.text}"
                detail = r.json()["detail"]
                assert "rollout" in detail and "r-doors" in detail, (path, detail)
        finally:
            child.close()

        for _ in range(100):                    # the goodbye is the close
            if srv.policy_ingest.active_run_id is None:
                break
            time.sleep(0.02)
        assert srv.policy_ingest.active_run_id is None

        r = http.post("/calibration/right/start")     # the door reopens
        assert r.status_code == 200, r.text
        http.post("/calibration/right/abort")


def test_the_lifespan_teardown_survives_a_child_that_never_hung_up(
        app_with_mocks, monkeypatch):
    """Ctrl-C during a rollout, through the real teardown. Exiting the app
    runs the REAL lifespan shutdown while a child is still connected — the
    exact state that used to deadlock `wait_closed()` against the stream loop
    at the teardown's FIRST step, before the recorder close, the session
    stops and `arms.disconnect_all()`, with the policy still committing
    goals. Bounded now, and the bound is the assertion: past ~2 s the drain
    belt fired, which means the nulled run no longer ends the loop."""
    import haller_hmi.server as srv
    from haller_hmi.runners.rollout_runner import IngestClient

    monkeypatch.setenv("HALLER_POLICY_INGEST", "tcp://127.0.0.1:0")
    _prime_lifespan(monkeypatch, srv)

    ctx = app_with_mocks
    ctx.__enter__()
    child = IngestClient("127.0.0.1", srv.policy_ingest.port)
    exited = False
    try:
        child.connect()
        child.handshake({"type": "policy_hello", "run_id": "r-hangup",
                         "unit": "deg", "control_hz_declared": 30.0,
                         "rig": "solo"})
        assert srv.policy_ingest.active_run_id == "r-hangup"

        t0 = time.perf_counter()
        ctx.__exit__(None, None, None)
        exited = True
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.9, (
            f"lifespan teardown took {elapsed:.2f}s with a connected child — "
            f"the drain belt fired instead of the stream loop exiting")
        assert srv.policy_ingest is None
        assert srv.policy_bridge is None
    finally:
        child.close()
        if not exited:
            ctx.__exit__(None, None, None)
