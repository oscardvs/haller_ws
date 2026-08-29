# hmi/backend/haller_hmi/policy_bridge.py
"""Production callables for `PolicyIngest`: bus_conflict, observe, submit.

`policy_ingest.py` owns the wire and takes three zero-config callables at
construction; this module is where they get their production bodies. The
contract (`docs/port/trackb-lab-contract.md`) names no home for them — its
module map ends at `runners/*`, and the ingest postdates the freeze — so this
is a new module beside the ingest rather than a tenant in `lab/` (which is
banned from touching arms) or in `server.py` (which is wiring, not behaviour).

The ruled architecture, restated because every choice below hangs off it
(contract addendum "rollout: the child owns the policy, never the bus",
2026-08-27): the child runs inference and streams target degrees; THIS process
keeps the Feetech bus and commits those targets through the same machinery as
every other leader. "This is the teleop architecture with a different leader."

## Who reads the serial line while a policy streams

The one real design decision in this file. `arm.py` serialises bus access by
architecture, not by locks: at any moment exactly one thread transacts. While
idle that thread is the `IdleSampler`'s; while a session runs it is the
session's, which STANDS THE SAMPLER DOWN by claiming the `TickBus` producer
(`tick.py::IdleSampler.tick_once` checks the owner before it reads — the
2026-08-29 "no status packet" storm is what happens when two threads share the
half-duplex line).

A rollout has the same shape: `submit` writes goals and `observe` reads state,
both from the ingest's asyncio task. So the bridge claims the producer for the
run's lifetime — the sampler stands down, and every read and write rides ONE
thread (the event loop; telemetry's own fallback reads already run there for
exactly this reason). The claim makes the exclusion bilateral for the VR
session, whose own `attach_producer` hits `ProducerConflict` and 409s — the
mirror of `bus_conflict` refusing a rollout while a session runs. The other
operator paths attach no producer (`TeleopSession` and `SimLeaderTeleop` write
from their own OS threads; a calibration sweep owns the line without a claim),
so their start routes ask `rollout_conflict` — the reverse half of
`bus_conflict` — and 409 while a run streams. The one-tick overlap at the
handover is the same one-shot race `tick.py` documents and the session already
tolerates.

Each observation is also PUBLISHED as a tick (state verbatim, `goal_deg` = the
last committed policy targets), so telemetry keeps painting the HUD during a
rollout, the bus keeps a measured rate, and the moment the child inferred from
is the moment every other consumer saw — invariant 8, one sampler per moment.

## The observation layout

`observe()` is zero-arg — the hello's `rig` never reaches it — so the layout
comes from THIS rig's arm set, by the same derivation the contract fixes for
datasets (`RigSpec`): two arms -> 12-dim, sides left then right, names
`{side}_{joint}` (exactly `recorder._state_names()`); ONE arm -> the 6-dim
UNPREFIXED solo layout, joints in `SO101_JOINT_ORDER` — the layout the kit
datasets carry (`local/so101_pick_cube`: unprefixed `shoulder_pan.pos` ...
`gripper.pos`) and the one `RigSpec.from_info` classifies as `rig == "solo"`.
A rig whose arm count differs from the policy's training dim cannot be served
through this signature; the wire stays self-describing via `state_names`, and
the honest fix is the hello-declares-the-order escalation already recorded in
`policy_ingest._canonical_state_names`.

## What `submit` does and refuses

Targets land on `ArmHandle.send_goal` with the DEFAULT speed cap — no
`speed_cap_deg_s` is passed, so `motion.max_speed_deg_s` governs. The teleop
session brings its own ceiling because a practised hand earned one; a policy
is not an operator's hand, and the discrete-move cap is the conservative
default the contract's commit chain implies, not a downgrade. The mode guard
stays (`send_goal` raises on a non-MANUAL arm — an E-STOPPED arm refuses every
frame), torque is NEVER enabled here, and a side that names no arm on this rig
is refused per action rather than guessed at — the two arms are 40 cm apart.

Refusals are returned in the outcome, never raised: the ingest treats an
exception from `submit` as a session failure and closes without a sentence,
and the outcome dict is the C3 artefact (`status()["last_commit"]`) that
distinguishes committed from merely received.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import math
import time
from collections.abc import Callable
from typing import Any

from .lab import lease
from .policy_ingest import ACTION_TYPE, PolicyIngest, PolicyRefusal
from .recorder import SO101_JOINT_ORDER
from .safety import ModeError
from .tick import ProducerConflict, ProducerToken, TickBus

logger = logging.getLogger(__name__)

#: The bus producer name a rollout claims. Shows up in `ProducerConflict`
#: sentences when a teleop session tries to start mid-rollout, so it names the
#: thing the operator must stop.
PRODUCER_NAME = "policy-rollout"

#: How often the claim watchdog re-checks that the tick-bus claim still has a
#: live run behind it. One ingest path ends a run without firing
#: `on_session_end` (a malformed frame nulls `active_run_id` inside `_stream`
#: before the handler's finally compares it), and `PolicyIngest` is ground
#: truth here, not editable — so the heal has to run on this side. Half a
#: second bounds how long the idle sampler stays stood down for a run that no
#: longer exists; the route-side `rollout_conflict` heals INSTANTLY for the
#: paths an operator actually pushes on.
STALE_CLAIM_SWEEP_S = 0.5

#: The environment variable the CHILD dials by (`rollout_runner.INGEST_URL_ENV`
#: spells the same string). Spelled here too, rather than imported, because the
#: serving process must not import `runners/*` — "Nothing here is imported by
#: the serving process" is that module's own first promise — so a source grep
#: on this name is the only instrument pinning the two equal, exactly like the
#: port constant in `policy_ingest`. The server honours it for its BIND PORT
#: (host stays loopback, always): children inherit this process's environment
#: via `lab/runs.launch`, so one setting moves both halves together, and a
#: server that ignored it would strand every child at a port nothing listens on.
INGEST_URL_ENV = "HALLER_POLICY_INGEST"


def ingest_port_from_env(default: int) -> int:
    """The port to bind, from `$HALLER_POLICY_INGEST` when set, else `default`.

    Only the PORT is taken. The bind host is `policy_ingest._BIND_HOST` and is
    "LOOPBACK ONLY, and never configurable to anything else" — an env var must
    not be the thing that quietly widens an unauthenticated motion socket. A
    value that does not parse falls back LOUDLY: binding the default while the
    child dials the configured value is a handshake that can never happen, and
    the log line is the only witness.
    """
    import os
    from urllib.parse import urlparse

    raw = os.environ.get(INGEST_URL_ENV)
    if not raw:
        return default
    try:
        parts = urlparse(raw.strip())
        if parts.scheme != "tcp" or parts.port is None:
            raise ValueError(f"need tcp://host:port, got {raw!r}")
        return int(parts.port)
    except ValueError as e:
        logger.error(
            "ignoring $%s (%s): binding the default port %d instead — the "
            "rollout child reads the same variable, so fix it or unset it, or "
            "no child will ever reach this server", INGEST_URL_ENV, e, default)
        return default


def _session_running(session: Any) -> bool:
    """Is this teleop-ish session driving? Tolerant of all three shapes.

    `HumanTeleopSession` and `TeleopSession` expose `.running`;
    `SimLeaderTeleop` only says so through `status()`. A session that cannot
    be read answers True — refusing a rollout is the safe direction, and a
    broken status call is not a licence to drive past it.
    """
    try:
        if bool(getattr(session, "running", False)):
            return True
        status = getattr(session, "status", None)
        if callable(status):
            return bool((status() or {}).get("running"))
        return False
    except Exception:  # noqa: BLE001 — unreadable session -> refuse, fail closed
        return True


class FiniteActionIngest(PolicyIngest):
    """`PolicyIngest` plus the finiteness gate the wire itself is missing.

    stdlib `json.loads` accepts `NaN`/`Infinity` literals and `decode_action`'s
    bare `float(v)` passes them, so a NaN-emitting policy — the classic
    fresh-checkpoint failure — would reach `send_goal`, where
    `clamp_joint_goal`'s `max(lo, min(hi, v))` resolves NaN and +inf to the
    UPPER joint limit and -inf to the lower. Nothing unbounded reaches the bus,
    but "slew every NaN joint to its limit at the full default cap, every tick"
    is a run silently reinterpreted, not a run refused. The VR wire refuses
    exactly this class (`human_teleop._usable_side`, whose docstring names the
    `max(lo, min(hi, nan))` hazard); the policy wire gets the same gate here.

    A subclass rather than an edit because `policy_ingest.py` is this port's
    ground truth for the wire and is not editable from this side. Raising
    `PolicyRefusal` from `decode_action` rides the module's own
    malformed-frame rule — the RUN is refused, and the sentence reaches the
    child as its exit message rather than dying in a server log.
    """

    def decode_action(self, msg: dict) -> tuple[int, dict]:
        seq, action = super().decode_action(msg)
        bad = [f"{side}.{joint}={value!r}"
               for side, joints in action.items()
               for joint, value in joints.items()
               if not math.isfinite(value)]
        if bad:
            raise PolicyRefusal(
                f"{ACTION_TYPE} carries a non-finite target "
                f"({', '.join(sorted(bad))}). Clamping would resolve NaN/+inf "
                f"to the joint's UPPER limit and -inf to its lower — a pose "
                f"the policy never asked for, driven at the full default cap. "
                f"A policy emitting non-finite targets is a broken checkpoint, "
                f"and the run is refused rather than reinterpreted.")
        return seq, action


class PolicyBridge:
    """The serving process's half of a rollout: refuse, observe, commit.

    Everything is injected (`lab/routes.py`'s pattern): `get_cameras` and
    `get_recorder` are zero-arg because both are lifespan globals, and the
    sessions/calibration/arms are the live objects so every answer is about
    the instant the child asks.
    """

    def __init__(
        self,
        *,
        arms,
        tick_bus: TickBus,
        get_cameras: Callable[[], Any],
        get_recorder: Callable[[], Any],
        calibration,
        sessions: tuple = (),
        devices: list[str] | None = None,
        left_arm_id: str = "left",
        right_arm_id: str = "right",
    ) -> None:
        self._arms = arms
        self._tick_bus = tick_bus
        self._get_cameras = get_cameras
        self._get_recorder = get_recorder
        self._calibration = calibration
        self._sessions = tuple(sessions)
        self._devices = list(devices or [])
        #: Side -> arm id, the recorder's own mapping (`DatasetRecorder`
        #: defaults). The action message's keys are SIDES; the arm manager is
        #: keyed by id; on every config in this repo the two spell the same.
        self._side_arms = {"left": left_arm_id, "right": right_arm_id}
        self._ingest = None
        self._token: ProducerToken | None = None
        #: The stale-claim watchdog task, spawned beside the claim whenever the
        #: claim is taken on a running event loop. See `_watch_claim`.
        self._watchdog: asyncio.Task | None = None
        #: {side: {joint: deg}} — last committed targets, published as the
        #: tick's `goal_deg` so the HUD shows what the policy is commanding.
        self._goals: dict[str, dict[str, float]] = {}
        #: Last logged fault sentence, so a stalled camera at 30 Hz is one log
        #: line, not thirty a second. Cleared on the next good sample.
        self._fault_msg: str | None = None

    def attach_ingest(self, ingest) -> None:
        """Give the bridge the ingest, for the stale-claim heal in
        `bus_conflict`. Set after construction because each needs the other."""
        self._ingest = ingest

    # ---- bus_conflict ----------------------------------------------------

    def bus_conflict(self) -> str | None:
        """Why a policy source must not be admitted right now, or None.

        C1's refusal half: `lease.bus_conflict` composes the sentences and this
        is the production caller its docstring promised ("no production caller
        yet ... This one's is Track A's ingest"). Three live inputs plus one
        this module adds:

        * the recorder (episode open / dataset still held),
        * every teleop session (human, leader-follower, sim leader),
        * a foreign holder of each real arm's serial device,
        * a CALIBRATION session — `lease.bus_conflict` predates the wizard's
          bus ownership and cannot see it, but a sweep owns the serial line
          the way a session owns the tick (`server.py`'s idle-sampler note),
          so admitting a policy over it corrupts the sweep AND the rollout.
        """
        self._release_if_stale()
        current = getattr(self._calibration, "current", None)
        if current is not None:
            arm = getattr(current, "arm_id", "?")
            return (f"cannot start a rollout: arm {arm!r} is being calibrated. "
                    "Finish or abort the calibration first.")
        # A discrete move's ramp runs on its own MoveExecutor thread — the
        # same two-writers hazard as a teleop session, invisible to
        # `lease.bus_conflict`. The reverse half lives on the /arm routes
        # (`server._refuse_during_rollout`).
        for arm_id in self._arms.keys():  # noqa: SIM118  (ArmManager, not a dict)
            executor = getattr(self._arms[arm_id], "executor", None)
            if getattr(executor, "is_running", False):
                return (f"cannot start a rollout: arm {arm_id!r} has a "
                        "discrete move in progress; wait for it to finish.")
        teleop_running = any(_session_running(s) for s in self._sessions)
        recorder = self._get_recorder()
        for device in (self._devices or [""]):
            conflict = lease.bus_conflict(
                device, recorder=recorder, teleop_running=teleop_running)
            if conflict:
                return conflict
        return None

    def rollout_conflict(self) -> str | None:
        """Why an operator path must not take the arms right now, or None.

        THE REVERSE HALF of `bus_conflict`, and the routes' one question:
        `bus_conflict` refuses a policy while an operator path runs; this
        refuses a teleop or calibration START while a policy streams. The VR
        session already gets this exclusion from its own `attach_producer`
        (`ProducerConflict` -> 409), but `TeleopSession`, `SimLeaderTeleop`
        and `CalibrationManager` attach no producer — their loops write from
        their own OS threads beside the ingest's asyncio commits, two writers
        interleaving packets on one half-duplex Feetech line — so their start
        routes ask here instead.

        Race-free by construction, not by lock: the ingest admits a run on the
        event loop, and every start route that calls this is `async def` with
        no await between the check and the session claiming its arms — the
        same single-thread argument `/teleop/sim/start` already documents.

        Heals before it answers: a claim whose run is gone (the leaked
        malformed-frame path) is released here, so a dead rollout never costs
        the operator a 409.
        """
        self._release_if_stale()
        run_id = getattr(self._ingest, "active_run_id", None)
        if run_id is not None:
            return (f"a policy rollout is streaming (run {run_id!r}). Stop it "
                    f"— or let it finish — before taking the arms: two "
                    f"writers on one half-duplex bus corrupt each other's "
                    f"packets, in both directions.")
        return None

    # ---- observe ---------------------------------------------------------

    def observe(self) -> dict | None:
        """One coherent sample: state + images from this instant, or None.

        None is the honest answer to any hole — a failed arm read, a dropped
        joint, a recorded camera with no fresh frame. Invariant 9: a dropped
        observation is a stall the child times out on and NAMES; a fabricated
        or partial one is a state the rig was never in, fed to the thing
        driving the arm. The withheld reason lands in the log, once per
        distinct fault.
        """
        if not self._claim():
            return None
        sides = self._sides()
        if not sides:
            self._fault("no arms are configured; a rollout has nothing to observe")
            return None
        solo = len(sides) == 1
        state: list[float] = []
        names: list[str] = []
        snaps: dict[str, dict] = {}
        for side, arm_id in sides:
            try:
                handle = self._arms[arm_id]
                snap = handle.state_snapshot()
            except Exception as e:  # noqa: BLE001 — any bus fault is a hole
                self._fault(f"arm {arm_id!r} state read failed: {e}")
                return None
            snaps[arm_id] = snap
            joints = dict(snap.get("joints") or {})
            for joint in self._joint_order(handle):
                cell = joints.get(joint)
                if not isinstance(cell, dict) or "pos" not in cell:
                    self._fault(
                        f"arm {arm_id!r} read dropped joint {joint!r}; "
                        "observation withheld rather than padded")
                    return None
                state.append(float(cell["pos"]))
                names.append(joint if solo else f"{side}_{joint}")
        images = self._images()
        if images is None:
            return None
        # Stamped AFTER the reads, like `idle_sample`: a clock taken before a
        # bus round trip is stale by however long the round trip took.
        t_mono = time.perf_counter()
        t_unix = time.time()
        self._publish_tick(t_mono, t_unix, snaps)
        self._fault_msg = None
        return {"state": state, "state_names": names, "images": images}

    def _images(self) -> dict[str, str] | None:
        """base64 JPEG per recorded camera, keyed by `dataset_feature_key`.

        The same hub and the same three filters as the recorder's
        `_active_camera_specs` — connected, RGB-capable, in the RUNTIME
        recorded set — so the policy sees exactly the views its training data
        carried and the device is never opened twice. The payload is what the
        child's `_decode_image` takes without the raw-list fallback.
        """
        cameras = self._get_cameras()
        if cameras is None:
            return {}
        images: dict[str, str] = {}
        is_recorded = getattr(cameras, "is_recorded", None)
        for cam_id in cameras.keys():  # noqa: SIM118 — CameraManager, not a dict
            handle = cameras[cam_id]
            if not getattr(handle, "active", False):
                continue
            if not hasattr(handle, "latest_rgb"):
                continue
            cfg = getattr(handle, "cfg", None)
            recorded = (is_recorded(cam_id) if callable(is_recorded)
                        else bool(getattr(cfg, "record", True)))
            if not recorded:
                continue
            key = getattr(cfg, "dataset_key", None) or cam_id
            jpeg = handle.latest_jpeg()
            if jpeg is None:
                self._fault(
                    f"camera {cam_id!r} has no fresh frame; observation "
                    "withheld — a policy fed a stale view drives blind")
                return None
            images[str(key)] = base64.b64encode(jpeg).decode("ascii")
        return images

    # ---- submit ----------------------------------------------------------

    def submit(self, run_id: str, seq: int, action: dict) -> dict:
        """Commit one action frame; say per joint what the chain did (C3).

        The outcome names the stage that altered each target — `clamp` (joint
        limits) or `rate_cap` (the per-tick step budget at the DEFAULT
        `motion.max_speed_deg_s`) — because "the committed value differs in the
        direction the chain imposes" is C3's evidence and a bare diff is
        indistinguishable from a bypass. Joints `send_goal` dropped (unknown to
        this arm, or unmeasurable this tick) are listed, not silently absent.
        """
        outcome: dict[str, Any] = {"run_id": run_id, "seq": int(seq), "sides": {}}
        if not self._claim():
            for side in action:
                outcome["sides"][str(side)] = {"refused": (
                    "the tick bus has another producer; a policy must not "
                    "interleave writes with it")}
            return outcome
        for side, joints in action.items():
            outcome["sides"][str(side)] = self._submit_side(str(side), joints)
        return outcome

    def _submit_side(self, side: str, joints: dict) -> dict:
        arm_id = self._side_arms.get(side, side)
        try:
            handle = self._arms[arm_id]
        except KeyError:
            sentence = (f"no arm for side {side!r} on this rig (arms: "
                        f"{sorted(self._arms.keys())}); refused rather than "
                        "guessed — the two arms are 40 cm apart")
            self._fault(sentence)
            return {"refused": sentence}
        if not getattr(handle, "torque_enabled", False):
            sentence = (f"arm {arm_id!r} has torque disabled; a rollout never "
                        "enables torque on its own. Enable it from the cockpit "
                        "first.")
            self._fault(sentence)
            return {"refused": sentence}
        try:
            # DEFAULT speed cap, on purpose: no `speed_cap_deg_s` is passed.
            committed = handle.send_goal(dict(joints))
        except ModeError as e:
            # The mode guard staying is the point — an E-STOPPED arm refuses
            # every frame and never re-energises from here.
            sentence = f"arm {arm_id!r}: {e}"
            self._fault(sentence)
            return {"refused": sentence}
        self._goals.setdefault(side, {}).update(committed)
        altered: dict[str, str] = {}
        dropped: list[str] = []
        limits = getattr(handle, "joint_limits_deg", {}) or {}
        for joint, target in joints.items():
            if joint not in committed:
                dropped.append(joint)
                continue
            got = float(committed[joint])
            want = float(target)
            if abs(got - want) <= 1e-9:
                continue
            lo_hi = limits.get(joint)
            clamped = want if lo_hi is None else max(lo_hi[0], min(lo_hi[1], want))
            altered[joint] = "rate_cap" if abs(got - clamped) > 1e-9 else "clamp"
        out: dict[str, Any] = {"arm": arm_id, "committed": dict(committed)}
        if altered:
            out["altered"] = altered
        if dropped:
            out["dropped"] = dropped
        return out

    # ---- the producer claim ----------------------------------------------

    def _claim(self) -> bool:
        """Hold the tick bus for the run, standing the idle sampler down.

        Lazy — the ingest has no session-start hook, and the first `observe`
        follows the ack within one loop turn — and idempotent, so a claim that
        outlived a run (see `_release_if_stale`) is reused, never doubled.
        """
        if self._token is not None and self._token.live:
            return True
        try:
            self._token = self._tick_bus.attach_producer(PRODUCER_NAME)
        except ProducerConflict as e:
            self._fault(f"cannot drive the arms for a policy: {e}")
            return False
        self._spawn_watchdog()
        return True

    def _spawn_watchdog(self) -> None:
        """Put a bound on how long a claim can outlive its run.

        Spawned beside the claim, on the loop the ingest already runs on —
        `_claim` only ever fires inside `observe`/`submit`, which the ingest
        calls from its own coroutine. Callers with no running loop (unit
        tests driving the bridge directly) get no watchdog and need none:
        they hold the claim, so they release it.
        """
        if self._watchdog is not None and not self._watchdog.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._watchdog = loop.create_task(self._watch_claim())

    async def _watch_claim(self) -> None:
        """Release the claim once its run is gone, whoever ended it.

        The routes heal instantly through `rollout_conflict`, but the idle
        sampler asks nobody's permission — it just finds the bus owned and
        stands down — so a leaked claim silently decays `measured_hz` until
        recording arming refuses (invariant 10). This loop is what puts the
        sampler back without waiting for an operator to push on a route.
        Exits on its own once the claim is released, and `release()` never
        needs to know it exists.
        """
        while self._token is not None and self._token.live:
            await asyncio.sleep(STALE_CLAIM_SWEEP_S)
            self._release_if_stale()

    def release(self) -> None:
        """End-of-run teardown: hand the tick back to the idle sampler.

        Wired as the ingest's `on_session_end`, and called again from the
        lifespan's shutdown — detaching twice is a no-op by design. The
        watchdog goes with the claim it watches; cancelling it from inside
        its own `_release_if_stale` call is benign (the coroutine returns
        before it ever awaits again).
        """
        token, self._token = self._token, None
        if token is not None and token.live:
            token.detach()
        watchdog, self._watchdog = self._watchdog, None
        if watchdog is not None and not watchdog.done():
            watchdog.cancel()
        self._goals = {}
        self._fault_msg = None

    async def shutdown(self, *, timeout_s: float = 2.0) -> None:
        """Stop the ingest without deadlocking on its own stream loop.

        THE ORDER IS THE FIX. On Python 3.12 `asyncio.Server.wait_closed()`
        waits for connection handlers to finish, and the handler's `_stream`
        loop runs `while active_run_id is not None` — so a `stop()` that nulls
        `active_run_id` only AFTER `wait_closed()` is a circular wait whenever
        a child is still connected: shutdown hangs at its first step while the
        policy keeps committing goals, and on a --reload the new process
        connects the arms beside the old one still writing — two processes on
        one Feetech bus, the exact corruption `foreign_port_holders` exists to
        refuse. `PolicyIngest.stop()` is ground truth and not editable from
        here, so the safe sequence lives on this side: null the run FIRST
        (the stream loop exits within one poll, ≤ a quarter of the declared
        period), then let `stop()` close the listener and drain the handlers.

        The timeout is a belt for the one handler `active_run_id` cannot
        reach: a peer that connected and never spoke sits in the 10 s
        handshake read, and shutdown must not inherit that wait. On timeout
        the listener is already closed (close() precedes wait_closed()), the
        cancelled handler dies with the loop, and the arms drain on schedule.

        The handler's finally then SKIPS `on_session_end` (the run id it
        compares is already nulled), which is why this ends by releasing the
        claim itself — the same cover the lifespan always provided.
        """
        ing = self._ingest
        if ing is not None:
            ing.active_run_id = None
            try:
                await asyncio.wait_for(ing.stop(), timeout_s)
            except TimeoutError:
                logger.warning(
                    "policy ingest did not drain its connections in %.1fs; "
                    "the listener is closed and shutdown proceeds", timeout_s)
            # A handshake that slipped into the scheduling gap between the
            # null above and the listener closing could have re-armed the
            # stream loop; nulling again after stop() — when no new admission
            # is possible — ends it rather than letting it commit goals into
            # the arm teardown below the lifespan's yield.
            ing.active_run_id = None
        self.release()

    def _release_if_stale(self) -> None:
        """Heal a claim that outlived its run.

        One ingest path ends a run without firing `on_session_end` (a
        malformed frame nulls `active_run_id` before the handler's finally
        compares it), and `PolicyIngest` is not editable from here. Three
        callers, three latencies: the next handshake (`bus_conflict`) and any
        operator start route (`rollout_conflict`) heal at the moment anything
        cares, and `_watch_claim` bounds the quiet case — no route pushed on —
        at `STALE_CLAIM_SWEEP_S`, so the idle sampler is never stood down
        indefinitely for a run that no longer exists.
        """
        if self._token is None or not self._token.live:
            return
        if self._ingest is not None and self._ingest.active_run_id is None:
            logger.info("policy bridge: releasing a tick-bus claim that "
                        "outlived its run")
            self.release()

    def _publish_tick(self, t_mono: float, t_unix: float,
                      snaps: dict[str, dict]) -> None:
        """One tick per observation, so the HUD and the rate stay live.

        Never lets a publish failure cost the observation — the tick is a
        courtesy to telemetry; the observation is the job.
        """
        if self._token is None or not self._token.live:
            return
        try:
            self._token.publish(
                t_mono=t_mono, t_unix=t_unix, arms=snaps, arm_errors={},
                goal_deg={side: dict(g) for side, g in self._goals.items()},
                base={}, degraded=False)
        except Exception:
            logger.exception("policy bridge: tick publish failed")

    # ---- layout ----------------------------------------------------------

    def _sides(self) -> list[tuple[str, str]]:
        """(side, arm_id) pairs this rig actually has, left first —
        `recorder._sides` semantics against the same arm manager."""
        out: list[tuple[str, str]] = []
        for side in ("left", "right"):
            arm_id = self._side_arms[side]
            try:
                self._arms[arm_id]
            except KeyError:
                continue
            out.append((side, arm_id))
        return out

    def _joint_order(self, handle) -> list[str]:
        """Joints present on this arm, in canonical SO-101 order —
        `recorder._joint_order` semantics."""
        present = set(getattr(handle, "joint_limits_deg", {}) or {})
        return [j for j in SO101_JOINT_ORDER if j in present]

    def _fault(self, sentence: str) -> None:
        if sentence != self._fault_msg:
            self._fault_msg = sentence
            logger.warning("policy bridge: %s", sentence)
