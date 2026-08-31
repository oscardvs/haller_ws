# hmi/backend/haller_hmi/server.py
"""FastAPI app. The only place that ties lerobot, ROS, presets, and HTTP together."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from .arm import ArmManager
from .calibration import (
    CalibrationManager,
    CalibrationState,
    ConflictError,
    RecenterError,
    SweepWrapError,
    UnmovedJointsError,
    WrongStateError,
    _calibration_paths,
)
from .cameras import CameraManager
from .config import load_config
from .human_teleop import HumanTeleopSession
from . import motion as motion_policy
from .motion import MoveRefused
from .presets import PresetNotFound, PresetStore
from .ros_bridge import RosBridge
from .safety import Mode, ModeError
from .teleop import TeleopSession
from .telemetry import TelemetryBroadcaster
from .tick import IdleSampler
from .recorder import DatasetRecorder, lerobot_home
from .policy_bridge import FiniteActionIngest, PolicyBridge, ingest_port_from_env
from .policy_ingest import INGEST_PORT, PolicyIngest
from .lab.routes import build_lab_router
from .sim.scene import RandomSpec, SceneController
from .sim.task import InsertionMonitor, TaskMonitor
from .sim.teleop import SimLeaderTeleop
from .vr_teleop import wire as vr_wire

logger = logging.getLogger(__name__)

VERSION = "0.1.0"

# Globals — wired in lifespan
cfg = load_config()
arms = ArmManager(cfg.arms, motion=cfg.motion, sim_cubes=cfg.sim_cubes,
                  sim_task=cfg.sim_task)
cameras: CameraManager | None = None   # constructed in lifespan, after arms.connect_all()
ros = RosBridge(cfg.ros)
presets = PresetStore()
teleop = TeleopSession(arms)
# The collision guard needs a mount pose for every enabled arm; an arm it has
# no geometry for would silently pass every check, which is the fail-open a
# guard must not have. Missing mounts therefore make it UNAVAILABLE — a state
# no runtime toggle can leave.
#
# It is constructed even when `collision.enabled` is false, which it was not
# before. Two things follow, and both are the point of the change: the guard
# can be switched on and off mid-session from the headset or the panel
# (`POST /teleop/human/collision`), and a guard that is switched OFF still
# publishes live clearance. An operator who turned it off because it bit too
# early — the common complaint, and the reason the switch exists — should
# still be able to see on the HUD how close the arm actually is.
from .collision import CollisionGuard

_unmounted = [a.id for a in cfg.arms
              if a.enabled and a.id not in cfg.collision.mounts]
if _unmounted:
    logger.warning(
        "collision guard UNAVAILABLE: no mounts configured for arms %s "
        "(add them under collision.mounts in config.yaml)", _unmounted)
_collision_guard = CollisionGuard(cfg.collision, available=not _unmounted)
if not _collision_guard.enabled:
    logger.warning("collision guard starts DISABLED (config collision.enabled="
                   "%s, available=%s)", cfg.collision.enabled,
                   _collision_guard.available)


def _wrist_floor_m() -> float:
    """Lowest wrist height the old demand-shaping floor used, mount frame.

    Derived from the collision config's bench geometry. The kit-faithful
    driven path carries no demand floor (the kit has none; the collision
    guard, when on, is the backstop) — these still seed the config so the
    settings panel keeps echoing the fields it always carried.
    """
    if cfg.collision.table_z_m is None:
        return 0.02
    return cfg.collision.table_z_m + cfg.collision.wrist_min_m + 0.005


def _tip_floor_m() -> float:
    """Lowest fingertip height, same story as `_wrist_floor_m`."""
    if cfg.collision.table_z_m is None:
        return 0.005
    return cfg.collision.table_z_m + cfg.collision.tip_min_m + 0.005


def _make_vr_teleop_config():
    """THE VR mapping config — one instance, session-and-socket shared.

    The clutch anchors moved into the session (`KitSideTeleop` per driven
    side), so the knobs that steer them live beside the session too: every
    teleop socket writes `config_update`s onto this same instance, and the
    running session's adapters read it on their next tick. A per-connection
    copy — the old converter's model — would leave a reconnecting headset
    tuning a config nothing was driving with.
    """
    from .vr_teleop.config import QuestTeleopConfig, apply_update
    qcfg = QuestTeleopConfig(min_tip_z=_tip_floor_m(),
                             min_wrist_z=_wrist_floor_m())
    if cfg.teleop:
        # The YAML `teleop:` section, through the same clamps a headset
        # write gets.
        applied = apply_update(qcfg, cfg.teleop)
        logger.info("teleop config seeded from YAML: %s", applied)
    return qcfg


vr_cfg = _make_vr_teleop_config()
human_teleop = HumanTeleopSession(arms, collision_guard=_collision_guard,
                                  lpf_tau_s=cfg.motion.lpf_tau_s,
                                  vr_config=vr_cfg)
sim_teleop = SimLeaderTeleop(arms)
# Bilateral session lock between every pair — any one running blocks the others.
teleop.attach_peer(human_teleop)
teleop.attach_peer(sim_teleop)
human_teleop.attach_peer(teleop)
human_teleop.attach_peer(sim_teleop)
sim_teleop.attach_peer(teleop)
sim_teleop.attach_peer(human_teleop)
calibration = CalibrationManager()
telemetry: TelemetryBroadcaster | None = None
recorder: DatasetRecorder | None = None  # constructed in lifespan, after telemetry.start()
# Both need a live MuJoCo world, which only exists once ArmManager has built one
# for a source: sim arm — so both stay None on an all-real rig. Constructed in
# lifespan; the /sim/* routes 409 when there is no world and 503 in the narrow
# window before lifespan has run.
scene: SceneController | None = None
task: TaskMonitor | None = None
# The server half of the policy wire: the rollout child dials this and streams
# target degrees; the bridge supplies the three injected callables. Constructed
# in lifespan — bus_conflict/observe/submit need the recorder, the cameras and
# the sessions to exist first.
policy_ingest: PolicyIngest | None = None
policy_bridge: PolicyBridge | None = None

#: How long POST /sim/scene/reset waits for a requested homing ramp to finish
#: before dealing cubes. motion.home() only SCHEDULES the ramp, so without a
#: wait the cubes land under an arm still swinging through them. Generous
#: enough for a full-travel home at the configured 60°/s, bounded so a wedged
#: executor can't hang the route.
_HOME_WAIT_S = 6.0


# ---- request schemas -----------------------------------------------------

class CmdVel(BaseModel):
    linear: float
    angular: float


class ArmModeBody(BaseModel):
    mode: str


class PresetBody(BaseModel):
    name: str


class TorqueBody(BaseModel):
    enabled: bool


class TeleopStartBody(BaseModel):
    leader: str
    follower: str
    hz: float = 60.0


class HumanTeleopStartBody(BaseModel):
    # Either side may be null for a SINGLE-ARM session: that hand's
    # controller is ignored and nothing is ever written to that arm. This is
    # what a first hardware bring-up wants, and what a rig with one working
    # servo board is left with.
    left_arm: str | None = None
    right_arm: str | None = None
    hz: float = 60.0


class HumanTeleopReattachBody(BaseModel):
    token: str


class SimTeleopStartBody(BaseModel):
    follower: str
    hz: float = 60.0
    # Body of the leader source — one of:
    #   {"source": "mouse", "arm_name": "left"}
    #   {"source": "replay", "dataset_path": "/path/to/lerobot/dataset"}
    leader: dict


# `extra="forbid"` on all four: a body field the model has never heard of is a 422
# rather than a silently-dropped key. `{save}` alone stays valid, so both shipped
# desktop callers are unaffected — forbid rejects UNKNOWN fields, not absent ones.
class RecordStartBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_id: str          # e.g. "oscardvs/haller_pick_cube"
    task: str             # natural-language instruction logged with every frame


class RecordArmBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_id: str
    task: str


class RecordRollBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecordStopBody(BaseModel):
    # `rearm` and `extra="forbid"` LAND TOGETHER, and the order is load-bearing.
    # Forbid arriving first would 422 the headset's `{save:true, rearm:true}` —
    # a keep — and the take the operator just drove would be LOST, which is
    # strictly worse than the silent 200 that dropping `rearm` used to give.
    model_config = ConfigDict(extra="forbid")
    save: bool = True     # False -> discard the episode buffer (bad take)
    rearm: bool = False   # OPTIONAL: `{save}` alone keeps its shipped meaning


class SimSceneResetBody(BaseModel):
    # None -> fresh entropy. Pass a seed to make an episode reproducible.
    seed: int | None = None
    randomize: bool = True
    # Reflect the bench about x = 0, the plane between the two arm mounts, so
    # the part that was in front of the left arm starts in front of the right.
    # The insertion parts' home slots are not symmetric — the pin lies outboard
    # of the right arm and is 0.537 m from the left arm's base — so the
    # mirrored arm assignment ("right holds, left inserts") is out of reach
    # without this. Same seed + mirror is the exact mirror image of that seed.
    mirror: bool = False
    # Send the arms back to their calibrated home first, via the same bounded
    # motion path as /arm/{id}/home — so an oversize move is refused rather
    # than swept blind. Off by default: a reset in the middle of a teleop
    # session should move the bench, not the robot.
    home_arms: bool = False


# ---- lifespan ------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global telemetry, cameras, recorder, scene, task
    logger.info("haller-hmi backend starting (version %s)", VERSION)
    # Wire the ownership guard here, not as separate attach_peer calls below:
    # see ArmManager.connect_all's docstring and A6 in the plan.
    arms.connect_all(teleop_peers=[teleop, human_teleop, sim_teleop])
    # There is no global "sim mode" flag — the world exists iff some arm is
    # source: sim, so this is the test, here and in every /sim/* route.
    _world = arms.world()
    if _world is not None:
        # `cfg.sim_random` is key-checked at load (config._sim_random_from), so
        # an empty dict here means "RandomSpec's defaults" and a populated one
        # cannot carry a key RandomSpec would reject. Without this the headset
        # path had no way to reach the jitter that `sim/record.py --xy-jitter-m`
        # reaches on the scripted path.
        scene = SceneController(
            _world, RandomSpec(**cfg.sim_random) if cfg.sim_random else None)
        # The monitor follows the scene, from the one config key. Wiring the
        # cube predicate to an insertion scene would score every episode a
        # failure and the dataset would look like a operator problem rather
        # than a config one, so these are never chosen independently.
        task = (InsertionMonitor(_world) if cfg.sim_task == "insertion"
                else TaskMonitor(_world))
        if cfg.sim_seed is not None:
            scene.reset(seed=cfg.sim_seed)
            logger.info("sim scene seeded from config: sim_seed=%d", cfg.sim_seed)
    cameras = CameraManager(cfg.cameras, world=arms.world())
    cameras.connect_all()
    ros.start()
    human_teleop.set_base_source(ros)
    telemetry = TelemetryBroadcaster(arms, ros, hz=cfg.telemetry.hz,
                                     teleop=teleop, human_teleop=human_teleop,
                                     calibration=calibration,
                                     tick_bus=human_teleop.tick_bus)
    telemetry.start()
    # The idle sampler owns the tick whenever no session does, so ONE moment is
    # published at every moment. Not cosmetic: arming refuses against a MEASURED
    # rate (invariant 10), and with no producer while idle `measured_hz()` is
    # None exactly when the gate first reads it — which refuses. So this is what
    # makes arming possible on a fresh backend, not what makes idle look tidy.
    # A local, not a global: `_lifespan` is one async generator, so a name bound
    # before `yield` is still in scope after it.
    # Measured on config.bimanual-sim: `measured_hz()` is None for ~0.98 s / 30
    # samples after start, then settles at 29.9. So arming REFUSES for about a
    # second after the backend comes up, which is the safe direction and is far
    # below the time it takes an operator to reach for A/X — but it is a real
    # window, not zero, and a caller that arms programmatically on boot will see
    # it.
    # A calibration sweep owns the serial line the way a session owns the
    # tick, but it attaches no bus producer — its reads run inside
    # telemetry's frame building on the asyncio thread. Stand the sampler
    # down for the sweep's whole lifetime, or its thread reads beside the
    # sweep on a lock-free half-duplex bus and the wizard's single-error
    # abort dies on the first "Port is in use!". Telemetry then falls back
    # to direct reads, which run on the SAME asyncio thread as the sweep
    # and therefore cannot collide with it.
    idle_sampler = IdleSampler(human_teleop.tick_bus,
                               sample=lambda: (
                                   None if calibration.current is not None
                                   else human_teleop.idle_sample()),
                               hz=cfg.telemetry.hz)
    idle_sampler.start()
    # `task` is None on an all-real rig, and the recorder treats that as "these
    # episodes are unscored" rather than "these episodes failed": no
    # next.reward/next.done columns at all, and an info.json block that says so.
    # See recorder.py's module docstring.
    # `tick_bus` is REQUIRED in practice: `_freeze_fps` raises without it, so a
    # recorder built without one 409s every take (invariant 10 failing closed).
    # It was omitted here from 2c until 2d and recording was dead the whole time —
    # see `test_every_bus_consumer_is_wired_to_the_session_bus`, which checks the
    # argument PER CALL because a substring over this function cannot say which
    # of three call sites got it.
    recorder = DatasetRecorder(telemetry=telemetry, human_teleop=human_teleop,
                               cameras=cameras, task_monitor=task,
                               tick_bus=human_teleop.tick_bus)
    # The policy wire's server half. The bridge holds the three production
    # callables (contract C1-C3: lease.bus_conflict's first caller, the
    # observation sampler, the commit front door); the ingest owns the socket.
    # Zero-arg getters for cameras/recorder for the lab router's stated reason:
    # both are module globals assigned lines above, and a rebind must be seen.
    global policy_ingest, policy_bridge
    policy_bridge = PolicyBridge(
        arms=arms,
        tick_bus=human_teleop.tick_bus,
        get_cameras=lambda: cameras,
        get_recorder=lambda: recorder,
        calibration=calibration,
        sessions=(teleop, human_teleop, sim_teleop),
        # Only real arms have a serial device a foreign process could hold.
        devices=[a.port for a in cfg.arms if a.enabled and a.source == "real"],
    )
    # `FiniteActionIngest`, not the base class: json.loads accepts NaN/Infinity
    # literals and the frozen decode path would pass them into the clamp, which
    # resolves non-finite to a joint LIMIT — see the subclass docstring.
    policy_ingest = FiniteActionIngest(
        bus_conflict=policy_bridge.bus_conflict,
        submit=policy_bridge.submit,
        observe=policy_bridge.observe,
        on_session_end=policy_bridge.release,
        # The module's own default port, unless $HALLER_POLICY_INGEST moves it —
        # the child dials by the same variable (it inherits this environment via
        # lab/runs.launch), so one setting moves both halves. Host is loopback
        # only, and no env can widen it.
        port=ingest_port_from_env(INGEST_PORT),
    )
    policy_bridge.attach_ingest(policy_ingest)
    try:
        await policy_ingest.start()
    except OSError:
        # A stale holder of the port must not cost the operator teleop: the
        # HMI's first job is the arms. Every rollout child then refuses loudly
        # ("no policy ingest at ..."), which names this exact condition.
        logger.exception(
            "policy ingest could not bind %s — rollouts will be refused at "
            "the handshake until it can", policy_ingest.status()["url"])
    yield
    logger.info("haller-hmi backend shutting down")
    # FIRST, before anything below drains the arms: stop admitting and
    # streaming policy actions. NOT a bare `policy_ingest.stop()`: on 3.12
    # `wait_closed()` waits for the connection handlers, whose stream loop
    # runs until the active run is nulled — which stop() does only AFTER the
    # wait. With a child connected that is a circular wait: shutdown (Ctrl-C,
    # SIGTERM, --reload) hangs at this exact step while the policy keeps
    # committing goals, and on --reload the new process takes the arms beside
    # the old one still writing. `PolicyBridge.shutdown` nulls the run first,
    # bounds the drain, and releases the tick-bus claim itself (the handler's
    # finally skips `on_session_end` once the run id is nulled from outside).
    if policy_bridge is not None:
        await policy_bridge.shutdown()
        policy_bridge = None
        policy_ingest = None
    elif policy_ingest is not None:      # bridge never built; belt only
        policy_ingest.active_run_id = None
        await policy_ingest.stop()
        policy_ingest = None
    if recorder is not None:
        if recorder.status()["recording"]:
            await recorder.stop_episode(save=True)
        recorder.close()
    # BEFORE arms.disconnect_all(): a daemon thread doing per-arm reads, and
    # disconnecting underneath it throws on every remaining tick. Caught and
    # logged rather than fatal, so the cost is a shutdown full of noise that
    # reads as a fault. Synchronous — a bounded thread join, not an asyncio task.
    idle_sampler.stop()
    if telemetry is not None:
        await telemetry.stop()
    # Signal every in-flight ramp to stop FIRST, before the three sessions'
    # .stop() calls below — each joins its own thread for up to ~1-2s, during
    # which an unsignalled ramp would keep commanding. Same ordering fix as
    # /estop, and for the same reason: a mid-ramp shutdown would otherwise
    # null `handle.robot` under a live daemon thread once disconnect_all()
    # runs (MoveExecutor._play degrades safely via its broad except, but this
    # is the honest fix, not a reason to leave the window open longer than it
    # has to be).
    for handle in arms.values():
        handle.executor.request_stop()
    teleop.stop()
    human_teleop.stop()
    sim_teleop.stop()
    for handle in arms.values():
        handle.executor.wait(timeout=2.0)
    cameras.disconnect_all()
    arms.disconnect_all()
    ros.stop()


app = FastAPI(title="haller-hmi", version=VERSION, lifespan=_lifespan)

# The Lab router takes zero-arg callables, not values: routers mount at import
# time but cameras/recorder are module globals assigned in _lifespan — a router
# closing over the values would capture None and 503 forever.
#
# One router, not three: /lab/runs/** and /lab/system join this same builder as
# they land, so this line never changes again.
app.include_router(build_lab_router(
    get_cameras=lambda: cameras,
    get_recorder=lambda: recorder,
    lerobot_home=lerobot_home,
    # () -> bool. None reads HALLER_ALLOW_REMOTE_CONTROL. Mutating Lab routes
    # are loopback-only by default: --host 0.0.0.0 is how the Quest reaches the
    # page, and reaching the page must not also mean deleting a dataset.
    allow_remote_control=None,
))

# Permissive CORS — the HMI is intended for trusted local networks (Wi-Fi or AP).
# Add an env-flagged origin whitelist when we add auth.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- helpers -------------------------------------------------------------

def _arm_or_404(arm_id: str):
    try:
        return arms[arm_id]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _require_calibration_session(arm_id: str) -> None:
    if calibration.current is None or calibration.current.arm_id != arm_id:
        raise HTTPException(status_code=409, detail="no active session for this arm")


def _refuse_during_rollout() -> None:
    """409 any operator start path while a policy rollout streams.

    The REVERSE half of the bridge's `bus_conflict`: that one refuses a
    rollout while an operator drives; without this, `/teleop/start`,
    `/teleop/sim/start` and `/calibration/{arm}/start` would all succeed
    mid-rollout — none of them attaches a tick-bus producer, so nothing else
    refuses them — putting a session's OS thread (or a torque-cutting sweep)
    beside the ingest's asyncio commits on one half-duplex serial line.

    Race-free where it is called from an `async def` route with no await
    between this check and the session claiming its arms: rollout admission
    happens on this same event loop. Callable before the lifespan has run
    (`policy_bridge` still None) because routes are mountable without it —
    then there is no ingest, so there is nothing to refuse.
    """
    if policy_bridge is None:
        return
    conflict = policy_bridge.rollout_conflict()
    if conflict:
        raise HTTPException(status_code=409, detail=conflict)


# ---- routes --------------------------------------------------------------

@app.get("/health")
def get_health():
    return {"status": "ok", "arms_online": len(list(arms.keys())), "base_online": True}


@app.get("/config")
def get_config():
    return {
        "version": VERSION,
        "arms": [
            {
                "id": h.config.id,
                "model": h.config.model,
                "port": h.config.port,
                "mode": h.guard.mode.value,
                # Reported rather than inferred. The cockpit gates its
                # sim-leader preset on this; it used to sniff `port ==
                # "(sim)"`, which is a convention this module happens to
                # follow, not something it promises.
                "source": h.config.source,
                "sim_arm_name": h.config.sim_arm_name,
            }
            for h in arms.values()
        ],
        "cameras": [c.__dict__ for c in cfg.cameras],
    }


@app.post("/base/cmd_vel")
def post_cmd_vel(body: CmdVel):
    sent = ros.publish_cmd_vel(body.linear, body.angular)
    return {"ok": True, "linear": sent[0], "angular": sent[1]}


@app.post("/arm/{arm_id}/goal")
async def post_arm_goal(arm_id: str, body: dict[str, float]):
    """The jog channel: `JointSlider` debounces at 50ms, so a drag can post up
    to 20 Hz. This deliberately stays on `handle.send_goal` — bounded by
    elapsed time via `step_budget_deg` (see arm.py) — rather than
    `motion.move_to`'s ramp-and-refuse policy, which is sized for a single
    discrete pose change, not a stream: comparing a fast drag's goal against
    measured position while the previous call is still ramping accumulates
    error until it crosses `large_move_deg` and starts 409-ing mid-drag, and
    each call would additionally cost a blocking serial read plus a
    `Thread.join()` on the event loop. `/home` and `/preset` stay on
    `move_to` — they command an absolute pose a recalibration can invalidate,
    which is the incident this plan exists to prevent; a bounded jog cannot
    reproduce it the same way.
    """
    _refuse_during_rollout()
    handle = _arm_or_404(arm_id)
    if not handle.torque_enabled:
        raise HTTPException(
            status_code=409,
            detail=f"arm {arm_id!r} has torque disabled; enable it before sending a goal",
        )
    try:
        clamped = handle.send_goal(body)
    except ModeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "sent": clamped}


@app.post("/arm/{arm_id}/mode")
async def post_arm_mode(arm_id: str, body: ArmModeBody):
    handle = _arm_or_404(arm_id)
    if calibration.current is not None and calibration.current.arm_id == arm_id:
        raise HTTPException(status_code=409,
                            detail=f"arm {arm_id!r} is being calibrated")
    try:
        new_mode = Mode(body.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid mode {body.mode!r}")
    handle.guard.set(new_mode)
    if new_mode is Mode.STOP:
        handle.disable_torque()
    else:
        # Leaving STOP (or staying in auto/manual): make sure torque is engaged so
        # subsequent goals actually move the arm.
        if not handle.torque_enabled:
            handle.enable_torque()
    return {"ok": True, "mode": new_mode.value}


@app.post("/arm/{arm_id}/home")
async def post_arm_home(arm_id: str):
    _refuse_during_rollout()
    handle = _arm_or_404(arm_id)
    try:
        sent = motion_policy.home(handle)
    except ModeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except MoveRefused as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "sent": sent}


@app.post("/arm/{arm_id}/torque")
async def post_arm_torque(arm_id: str, body: TorqueBody):
    handle = _arm_or_404(arm_id)
    if body.enabled:
        handle.enable_torque()
    else:
        handle.disable_torque()
    return {"ok": True, "torque": handle.torque_enabled}


def _preflight_payload(arm_id: str) -> dict:
    """One arm's last preflight verdict, in the shape the HMI can act on."""
    report = arms.preflight_reports().get(arm_id)
    if report is None:
        return {"arm_id": arm_id, "ran": False, "ok": None, "message":
                "no preflight has run for this arm"}
    return {
        "arm_id": arm_id,
        "ran": True,
        "ok": report.ok(),
        "skipped": report.skipped,
        "message": report.message(),
        "out_of_range": list(report.out_of_range),
        "calibration_problems": list(report.calibration_problems),
        "calibration_warnings": list(report.calibration_warnings),
        "torque_dropped": report.torque_dropped,
        "torque_refused": list(report.torque_refused),
        "mode": arms[arm_id].guard.mode.value,
    }


@app.get("/arm/{arm_id}/preflight")
async def get_arm_preflight(arm_id: str):
    """Why an arm is refusing, in the operator's own words.

    Preflight has always run at connect and always logged its verdict; it has
    never been READABLE from the HMI, so the first sign of a failed one was a
    session that started, drove an arm outside its limits, and tripped a servo
    into overload. The verdict is the thing that should have been on the
    screen. 2026-08-28.
    """
    _arm_or_404(arm_id)
    return _preflight_payload(arm_id)


@app.post("/arm/{arm_id}/preflight")
async def post_arm_preflight(arm_id: str):
    """Re-run preflight, and clear the STOP if the arm now passes.

    This is the "until an operator clears it" the failure log has always
    promised and nothing has ever provided. The only way to re-check an arm
    was to restart the backend — which drops torque, lets the arm sag, and
    hands the next preflight a worse pose than the one it just refused. That
    loop is unwinnable one joint at a time.

    Re-running is NOT a way to dismiss the refusal: it re-reads the arm and
    the verdict is whatever the hardware now says. A pose that still fails
    still fails, and the arm stays in STOP.
    """
    handle = _arm_or_404(arm_id)
    arms.rerun_preflight(arm_id)
    payload = _preflight_payload(arm_id)
    if payload.get("ok"):
        # Preflight dropped torque on the way in; an arm that now passes is
        # one the operator has physically fixed, so give it back the state a
        # clean connect would have left it in.
        handle.enable_torque()
        handle.guard.set(Mode.MANUAL)
        payload["mode"] = handle.guard.mode.value
        payload["torque"] = handle.torque_enabled
    return payload


@app.get("/arm/{arm_id}/presets")
async def get_arm_presets(arm_id: str):
    _arm_or_404(arm_id)  # 404 if arm unknown
    return {"names": presets.list(arm_id)}


@app.delete("/arm/{arm_id}/preset/{name}")
async def delete_arm_preset(arm_id: str, name: str):
    _arm_or_404(arm_id)
    try:
        presets.delete(name, arm_id)
    except PresetNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


@app.post("/arm/{arm_id}/preset")
async def post_arm_preset(arm_id: str, body: PresetBody):
    _refuse_during_rollout()
    handle = _arm_or_404(arm_id)
    try:
        goal = presets.get(body.name, arm_id)
    except PresetNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        clamped = motion_policy.move_to(handle, goal)
    except ModeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except MoveRefused as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "sent": clamped}


@app.post("/arm/{arm_id}/preset/record")
async def post_arm_preset_record(arm_id: str, body: PresetBody):
    handle = _arm_or_404(arm_id)
    snap = handle.state_snapshot()
    current = {j: v["pos"] for j, v in snap["joints"].items()}
    presets.save(body.name, arm_id, current)
    return {"ok": True, "saved": current}


@app.get("/cameras")
async def get_cameras():
    return {"cameras": cameras.list()}


@app.get("/cameras/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: str):
    try:
        handle = cameras[camera_id]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    jpeg = handle.latest_jpeg()
    if jpeg is None:
        raise HTTPException(
            status_code=503,
            detail=f"no frame available for {camera_id!r} (placeholder source or capture failure)",
        )
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/cameras/{camera_id}/stream")
async def get_camera_stream(camera_id: str):
    try:
        cameras[camera_id]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return StreamingResponse(
        cameras.mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/estop")
async def post_estop():
    logger.warning("E-STOP triggered")
    # Signal every in-flight ramp to stop FIRST — before anything below that
    # blocks the event loop. calibration.abort() and each session's .stop()
    # (including sim_teleop, whose _loop otherwise catches ModeError and
    # keeps looping instead of exiting — see below) can each join a thread
    # for up to ~1-2s; an unsignalled ramp would keep commanding for however
    # long all of those take before it even hears the request to stop.
    # handle.executor.cancel() would also join (up to 2s) under the same lock
    # is_running()/wait() take, so calling it per-arm here — instead of
    # signal-all-then-join-all — would delay disable_torque() on arm 2 behind
    # arm 1's join. See MoveExecutor.request_stop and A6 in the plan.
    for handle in arms.values():
        handle.executor.request_stop()
    calibration.abort()
    teleop.stop()
    human_teleop.stop()
    # Mode.STOP alone does not stop a running SimLeaderTeleop: its _loop
    # catches send_goal's resulting ModeError inside a broad `except
    # Exception`, sleeps 50ms, and ticks again forever — nothing sets
    # `_stop`. Without this, the normal recovery (POST /arm/{id}/mode
    # {"mode":"manual"}, which also re-enables torque) would let the
    # still-live loop resume driving the arm with no further operator
    # action. _lifespan teardown already calls this; /estop must too.
    sim_teleop.stop()
    for handle in arms.values():
        handle.disable_torque()
        handle.guard.set(Mode.STOP)
    for handle in arms.values():
        handle.executor.wait(timeout=2.0)
    ros.zero_cmd_vel()
    return {"ok": True}


@app.get("/teleop")
async def get_teleop():
    return teleop.status()


@app.post("/teleop/start")
async def post_teleop_start(body: TeleopStartBody):
    _arm_or_404(body.leader)
    _arm_or_404(body.follower)
    # No await between this and teleop.start(): the check and the session's
    # claim are one atomic stretch on the event loop the ingest admits on.
    _refuse_during_rollout()
    try:
        teleop.start(body.leader, body.follower, hz=body.hz)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, **teleop.status()}


@app.post("/teleop/stop")
async def post_teleop_stop():
    teleop.stop()
    return {"ok": True, **teleop.status()}


@app.get("/teleop/human")
async def get_human_teleop():
    return human_teleop.status()


@app.post("/teleop/human/start")
async def post_human_teleop_start(body: HumanTeleopStartBody):
    if not body.left_arm and not body.right_arm:
        raise HTTPException(
            status_code=400,
            detail="at least one of left_arm/right_arm is required",
        )
    for arm_id in (body.left_arm, body.right_arm):
        if arm_id:
            _arm_or_404(arm_id)
    # The session's own attach_producer would 409 anyway (ProducerConflict);
    # asking first buys the sentence that names the RUN — and heals a claim
    # whose run is already gone, so a dead rollout never costs a start.
    _refuse_during_rollout()
    try:
        human_teleop.start(
            left_arm=body.left_arm, right_arm=body.right_arm, hz=body.hz,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, **human_teleop.status()}


@app.post("/teleop/human/stop")
async def post_human_teleop_stop():
    human_teleop.stop()
    return {"ok": True, **human_teleop.status()}


@app.post("/teleop/human/reattach")
async def post_human_teleop_reattach(body: HumanTeleopReattachBody):
    """The driver's page is back after a reload; hold the session for it.

    The SECOND door into `HumanTeleopSession.reattach`, and it exists because
    the first one cannot cover a reload. `VRTeleopPanel` opens the pose socket
    only on entering XR, so a page that has just reloaded has no socket at all
    to say `attach` on — and it cannot get one until the operator clicks Enter
    VR, which is the very thing the window is meant to buy time for. A page
    that reloads with a token in `sessionStorage` calls this on mount instead.

    Never a 4xx: a token that no longer matches (the session really did end, or
    someone else's is running now) is an ANSWER, not a failure — the page uses
    it to drop the stale token and offer a fresh start.
    """
    return {"ok": human_teleop.reattach(body.token), **human_teleop.status()}


@app.post("/teleop/human/home")
async def post_human_teleop_home():
    """Park every non-driving side at home, inside the running session.

    The discrete `/arm/{id}/home` is refused while a session owns the arms;
    this is the in-session counterpart the headset's hold-the-left-stick
    reset uses. See HumanTeleopSession.request_home for the semantics.
    """
    if not human_teleop.running:
        raise HTTPException(status_code=409, detail="no teleop session running")
    sides = human_teleop.request_home()
    return {"ok": True, "sides": sides}


class CollisionEnableBody(BaseModel):
    enabled: bool


@app.post("/teleop/human/collision")
async def post_human_teleop_collision(body: CollisionEnableBody):
    """Switch the collision/workspace guard on or off, live.

    A runtime switch rather than config-only, because the decision is made
    with the arm in front of you: the guard's margins are sized for a mount
    geometry that is still a placeholder on this rig, and an operator who
    finds it clamping while they are plainly nowhere near anything needs to
    be able to turn it off and keep working — not stop, edit YAML and
    restart. Switching it off leaves the measurement running, so
    `status().collision.slack_m` keeps telling them how close they really
    are.

    What this does NOT switch off: the teleop's own workspace floor (see
    `vr_teleop.config.QuestTeleopConfig.min_tip_z`), the per-joint limits,
    the rate caps or the motion envelope. Those are separate on purpose —
    turning off the bimanual guard should not also remove the bench.
    """
    if _collision_guard is None:
        raise HTTPException(status_code=409, detail="no collision guard wired")
    try:
        _collision_guard.enabled = body.enabled
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    logger.warning("collision guard %s by operator request",
                   "ENABLED" if _collision_guard.enabled else "DISABLED")
    # Read back off the GUARD, not off the session status. The session only
    # publishes a clearance read-out once its loop has run a tick, so
    # answering from there would report `null` to an operator who flipped the
    # switch before starting a session — which reads as "the toggle did
    # nothing".
    return {"ok": True, "collision": {
        "enabled": _collision_guard.enabled,
        "available": _collision_guard.available,
        "margin_m": _collision_guard.cfg.margin_m,
    }}


@app.post("/teleop/sim/start")
async def teleop_sim_start(body: SimTeleopStartBody):
    # async, not a plain def: a sync def route runs in Starlette's threadpool,
    # genuinely concurrent with the event loop — so it could race move_to()
    # (called from /home and /preset, both async def with synchronous bodies)
    # across the ~5-20ms window between move_to's read_joints_deg()/plan_ramp
    # and handle.executor.run(). Both configured arms are source: real, so
    # that race reaches hardware: two threads writing Goal_Position on one
    # unlocked PortHandler. async def puts this route on the same event loop
    # thread as move_to's callers, where neither body has an await point, so
    # they can't interleave — PROVIDED nothing in this function's own body
    # awaits in the middle of claiming the arm. See the next comment: this is
    # exactly what src.prepare() would have broken if left inline.
    # Checked twice, and both are load-bearing: here so a mid-rollout request
    # refuses before the replay path loads a whole LeRobotDataset for nothing,
    # and again after the await below, because src.prepare() yields the loop
    # and a rollout could be admitted while it runs.
    _refuse_during_rollout()
    leader_cfg = body.leader
    src_kind = leader_cfg.get("source")
    if src_kind == "mouse":
        from .sim.sources import MouseDragSource
        world = arms.world()
        if world is None:
            raise HTTPException(status_code=409, detail="sim world not active")
        src = MouseDragSource(world=world, arm_name=leader_cfg["arm_name"])
    elif src_kind == "replay":
        from .sim.sources import DatasetReplaySource
        src = DatasetReplaySource(dataset_path=leader_cfg["dataset_path"])
    else:
        raise HTTPException(status_code=400, detail=f"unknown leader source {src_kind!r}")
    # The slow half of getting a source ready — a "replay" source loads a
    # LeRobotDataset here, which can be a network round trip if it resolves
    # against the Hub (see sim/sources.py) — runs off the event loop, and
    # strictly BEFORE any arm state is touched. Everything from here on is
    # synchronous with no further await, so it stays one atomic, non-yielding
    # stretch on the event loop, serialised against move_to() the same way
    # two async def routes already are relative to each other. This is what
    # keeps that property from depending on "nothing else in this file ever
    # awaits" as an invariant to remember — the one await here is placed
    # before sim_teleop.start() claims the follower, not interleaved with it.
    await asyncio.to_thread(src.prepare)
    # The recheck after the one await — from here to start() nothing yields,
    # so this is the atomic stretch the comment above promises.
    _refuse_during_rollout()
    try:
        sim_teleop.start(follower_id=body.follower, source=src, hz=body.hz)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return sim_teleop.status()


@app.post("/teleop/sim/stop")
def teleop_sim_stop():
    sim_teleop.stop()
    return sim_teleop.status()


@app.get("/teleop/sim/status")
def teleop_sim_status():
    return sim_teleop.status()


# ---- sim scene reset + task success --------------------------------------

def _require_sim_world() -> None:
    """409 unless a MuJoCo world is live.

    `arms.world()` is the only honest test: there is no global sim flag, just
    per-arm `source: "sim"` driving ArmManager's lazy world construction.
    """
    if arms.world() is None:
        raise HTTPException(status_code=409, detail="sim world not active")


def _require_scene() -> SceneController:
    _require_sim_world()
    if scene is None:  # world exists but lifespan hasn't wired us yet
        raise HTTPException(status_code=503, detail="sim scene not ready")
    return scene


def _wait_for_arm_ramps(timeout: float) -> None:
    for handle in arms.values():
        handle.executor.wait(timeout=timeout)


@app.post("/sim/scene/reset")
async def post_sim_scene_reset(body: SimSceneResetBody):
    # async def for the reason spelled out at /teleop/sim/start: it keeps this
    # body on the event loop thread, where it cannot interleave with move_to()'s
    # read-then-run window. The one `await` below is placed AFTER homing has
    # been claimed and before anything else, not in the middle of it.
    ctl = _require_scene()
    if body.home_arms:
        # An open episode means the recorder is already writing frames. Sending
        # the arms home underneath it would splice a move nobody demonstrated
        # into the middle of the take — and unlike the cube reset, that lands in
        # the action column, not just the observation.
        #
        # getattr rather than an import: recorder.py is being edited
        # concurrently, and this guard must not depend on its internals.
        _rec_status = getattr(recorder, "status", None)
        if callable(_rec_status) and bool(_rec_status().get("recording")):
            raise HTTPException(
                status_code=409,
                detail="stop the recording episode before homing the arms")
        try:
            for handle in arms.values():
                motion_policy.home(handle)
        except ModeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except MoveRefused as e:
            raise HTTPException(status_code=409, detail=str(e))
        # home() returns as soon as the ramp is scheduled. Deal the cubes only
        # once the arms have actually arrived, or they get swept off the slots
        # they were just placed on.
        await asyncio.to_thread(_wait_for_arm_ramps, _HOME_WAIT_S)
    snapshot = ctl.reset(seed=body.seed, randomize=body.randomize,
                         mirror=body.mirror)
    if task is not None:
        # A cube that was still sitting on the pad when the last episode ended
        # would otherwise carry its qualifying streak into this one.
        task.reset()
    return {"ok": True, **snapshot}


@app.get("/sim/scene")
async def get_sim_scene():
    return _require_scene().snapshot()


@app.get("/sim/task/status")
async def get_sim_task_status():
    _require_sim_world()
    if task is None:
        raise HTTPException(status_code=503, detail="sim task monitor not ready")
    return task.poll()


# ---- dataset recording (HMI-integrated bimanual recorder, v0) -------------

@app.get("/record/status")
def get_record_status():
    if recorder is None:
        raise HTTPException(status_code=503, detail="recorder not ready")
    return recorder.status()


@app.get("/record/schema")
def get_record_schema():
    """The feature set a take started right now would write.

    Exists for `haller_hmi.dataset_migrate`, which has to migrate an older
    dataset to the schema `_open_dataset` will actually compare against. The
    recorded camera set is runtime state, so the running rig is the only thing
    that knows it — a migration computed from config.yaml would target a schema
    the recorder still refuses. Shapes go out as lists; JSON has no tuple.
    """
    if recorder is None:
        raise HTTPException(status_code=503, detail="recorder not ready")
    features = {k: {**v, "shape": list(v["shape"])}
                for k, v in recorder.features().items()}
    return {"features": features}


@app.post("/record/start")
async def post_record_start(body: RecordStartBody):
    if recorder is None:
        raise HTTPException(status_code=503, detail="recorder not ready")
    try:
        await recorder.start_episode(repo_id=body.repo_id, task=body.task)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, **recorder.status()}


@app.post("/record/arm")
async def post_record_arm(body: RecordArmBody):
    # Returns a BARE status, not `{ok: true, **status}` — the headset types
    # `recordArm`/`recordRoll` as `RecordStatus` and `recordStart`/`recordStop`
    # as `{ok} & RecordStatus`. Two shapes on one surface, matching the client.
    if recorder is None:
        raise HTTPException(status_code=503, detail="recorder not ready")
    try:
        return await recorder.arm(repo_id=body.repo_id, task=body.task)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/record/roll")
async def post_record_roll(body: RecordRollBody):
    if recorder is None:
        raise HTTPException(status_code=503, detail="recorder not ready")
    try:
        return await recorder.roll()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/record/stop")
async def post_record_stop(body: RecordStopBody):
    if recorder is None:
        raise HTTPException(status_code=503, detail="recorder not ready")
    status = await recorder.stop_episode(save=body.save, rearm=body.rearm)
    return {"ok": True, **status}


@app.get("/calibration/status")
def get_calibration_status():
    out_arms = []
    for h in arms.values():
        paths = _calibration_paths(h.config.calibration_id)
        target = paths[0]
        in_session = (
            calibration.current is not None
            and calibration.current.arm_id == h.config.id
        )
        out_arms.append({
            "id": h.config.id,
            "has_file": target.exists(),
            "path": str(target),
            "mtime": target.stat().st_mtime if target.exists() else None,
            "in_session": in_session,
        })
    session = calibration.current
    current = None
    if session is not None:
        current = {"arm_id": session.arm_id, "state": session.state.value}
        if session.state is CalibrationState.REVIEW:
            current["proposed"] = session.proposed
            current["current"] = session.current_on_disk
    return {"arms": out_arms, "current_session": current}


@app.post("/calibration/{arm_id}/start")
async def post_calibration_start(arm_id: str):
    _arm_or_404(arm_id)
    # The bridge refuses a rollout while a sweep runs (its calibration
    # branch); this is the missing reverse half. Without it a sweep cuts
    # torque on the target arm mid-rollout while the policy keeps driving the
    # other one, and after save the still-streaming policy resumes under a
    # freshly redefined zero.
    _refuse_during_rollout()
    try:
        session = calibration.start(arms, arm_id)
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "state": session.state.value}


@app.post("/calibration/{arm_id}/capture_neutral")
async def post_calibration_capture_neutral(arm_id: str):
    handle = _arm_or_404(arm_id)
    _require_calibration_session(arm_id)
    try:
        calibration.current.capture_neutral(handle)
    except RecenterError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except WrongStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "state": calibration.current.state.value,
            "homing_offsets": calibration.current.homing_offsets}


@app.post("/calibration/{arm_id}/finish_sweep")
async def post_calibration_finish_sweep(arm_id: str):
    handle = _arm_or_404(arm_id)
    _require_calibration_session(arm_id)
    try:
        proposed = calibration.current.finish_sweep(handle)
    except (UnmovedJointsError, SweepWrapError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except WrongStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {
        "ok": True,
        "state": calibration.current.state.value,
        "proposed": proposed,
        "current": calibration.current.current_on_disk,
    }


@app.post("/calibration/{arm_id}/save")
async def post_calibration_save(arm_id: str):
    _arm_or_404(arm_id)
    _require_calibration_session(arm_id)
    try:
        target, backup = calibration.save(arms)
    except WrongStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"save failed: {e}")
    return {"ok": True, "state": "done", "path": str(target),
            "backup_path": str(backup) if backup else None}


@app.post("/calibration/{arm_id}/abort")
async def post_calibration_abort(arm_id: str):
    _arm_or_404(arm_id)
    calibration.abort()
    return {"ok": True, "state": "aborted"}


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    assert telemetry is not None
    sub = telemetry.subscribe()
    try:
        async for frame in sub:
            await ws.send_json(frame)
    except WebSocketDisconnect:
        return


# A headset publishes continuously (~30 Hz) for as long as it is driving, so
# silence on the teleop socket means the connection is dead — e.g. wifi
# dropped without a FIN, where `receive` would otherwise sit blocked until the
# OS gives up, minutes later. Naming that silence turns the session's existing
# WS-disconnect grace (freeze now, auto-stop after 5 s) into something that
# actually runs.
WS_IDLE_TIMEOUT_S = 2.0

#: How often the socket pushes `ik_state` back to the client. The headset uses
#: it to drive haptics and paint the HUD, neither of which needs the frame
#: rate; the pose stream flows the other way at display rate.
IK_STATE_HZ = 20.0


async def _receive_or_idle_timeout(ws: WebSocket):
    """Await one JSON frame; return `None` if the socket stays silent for
    `WS_IDLE_TIMEOUT_S`. `WebSocketDisconnect` is left to propagate to the
    caller's handler."""
    try:
        return await asyncio.wait_for(ws.receive_json(),
                                       timeout=WS_IDLE_TIMEOUT_S)
    except TimeoutError:
        return None


@app.websocket("/ws/teleop/vr/in")
async def ws_vr_teleop_in(ws: WebSocket):
    """The one teleop socket: WebXR frames in, IK state and settings back.

    Frames are stored RAW on the session, latest-wins per side, and SOLVED
    there — once per 60 Hz tick, by the vendored kit adapter
    (`vr_teleop.kit_teleop.KitSideTeleop`). This socket converts nothing: a
    per-frame solve seeded from the session's throttled committed pose is
    the structure that bled hand-to-tool correspondence away, so the door's
    whole job is normalising the two wire spellings
    (`vr_teleop.wire.normalize_frame`) and handing the result over.

    Message types, client → server:
      `vr_keypoints` / `xr_frame`  a pose frame (both shapes normalised at the
                                   door, see `vr_teleop.wire.normalize_frame`)
      `config_update`              live tuning, clamped by QuestTeleopConfig
      `request_settings`           ask for the current config

    and server → client: `ik_state` (at IK_STATE_HZ while frames flow,
    assembled from the session's per-side `KitSideTeleop.diag()`),
    `config_applied`, `settings`. The config is ONE shared instance
    (`vr_cfg`): the anchors it steers live on the session, not on this
    connection.
    """
    await ws.accept()
    last_state = 0.0
    state_period = 1.0 / IK_STATE_HZ
    # Whether this client has ever streamed a pose. The idle timeout exists to
    # catch a headset that WAS driving and went quiet; a page that is merely
    # open — parked on the landing screen, tuning sliders, not yet in XR — is
    # not an operator who has stopped, and tearing it down every two seconds
    # just makes it reconnect. Measured: 62 pointless reconnects across one
    # smoke run, each one a log line.
    streamed = False
    # Send the settings unprompted, so the client's sliders start on the
    # robot's actual values rather than on their own defaults — one message,
    # and it removes a window where the two disagree.
    try:
        await ws.send_json(_settings_message())
    except Exception:
        pass
    try:
        while True:
            msg = await _receive_or_idle_timeout(ws)
            if msg is None:
                if not streamed:
                    continue        # never drove; nothing has stopped
                logger.warning("vr teleop socket idle %.1fs; treating as "
                               "disconnected", WS_IDLE_TIMEOUT_S)
                human_teleop.notify_ws_disconnected()
                await ws.close()
                return
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("type")
            if mtype == "config_update":
                # Echo the CLAMPED values back, so a slider that asked for
                # something out of range snaps to what the robot actually took
                # instead of silently disagreeing with it.
                from .vr_teleop.config import apply_update
                applied = apply_update(vr_cfg, msg.get("config") or {})
                if applied:
                    logger.info("vr teleop config_update: %s", applied)
                await ws.send_json({"type": "config_applied", "config": applied})
                continue
            if mtype == "request_settings":
                await ws.send_json(_settings_message())
                continue
            if mtype == "attach":
                # A client that drove this session before the stream went
                # quiet, coming back. See `HumanTeleopSession.reattach` for why
                # the token is the discriminator and why the window it buys is
                # one-shot.
                ok = human_teleop.reattach(str(msg.get("token") or ""))
                await ws.send_json({"type": "attach_result", "ok": ok})
                continue
            first_frame = not streamed
            streamed = True
            try:
                human_teleop.ingest_frame(vr_wire.normalize_frame(msg))
            except Exception:
                # One malformed frame must never drop the operator's
                # connection mid-session.
                logger.exception("vr teleop ingest_frame failed")
            if first_frame:
                # The pose frame IS the proof: this connection is driving, so
                # it — and only it — is told the session's token, to present
                # on the way back from a reload. A page parked on the landing
                # screen never streams and so never sees one.
                token = human_teleop.driver_token()
                if token:
                    await ws.send_json({"type": "session", "token": token})
            now = asyncio.get_running_loop().time()
            if now - last_state >= state_period:
                last_state = now
                try:
                    await ws.send_json({"type": "ik_state",
                                        "config": vr_cfg.to_dict(),
                                        "sides": human_teleop.ik_sides()})
                except Exception:
                    pass
    except WebSocketDisconnect:
        # Same rule as the idle path: only a client that was actually
        # streaming poses has an operator whose disappearance should start the
        # session's grace window. A watcher page closing is not an operator
        # leaving.
        if streamed:
            human_teleop.notify_ws_disconnected()
        return


def _settings_message() -> dict:
    """The full live-tunable config, as its own message type — the reply to
    `request_settings` and the greeting on connect. `ik_state` carries the same
    config block, but only starts flowing once frames do, and a client parked
    on the tuning panel needs the values before it drives anything."""
    return {"type": "settings", "config": vr_cfg.to_dict()}


def run() -> None:
    """Entry point for the `haller-hmi` console script."""
    import uvicorn
    uvicorn.run("haller_hmi.server:app", host="0.0.0.0", port=8000, log_level="info")
