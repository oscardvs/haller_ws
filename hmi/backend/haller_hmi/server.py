# hmi/backend/haller_hmi/server.py
"""FastAPI app. The only place that ties lerobot, ROS, presets, and HTTP together."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .arm import ArmManager
from .calibration import (
    CalibrationManager,
    CalibrationState,
    ConflictError,
    UnmovedJointsError,
    WrongStateError,
    _calibration_paths,
)
from .cameras import CameraManager
from .config import load_config
from .human_teleop import ClutchSource, HumanTeleopSession
from .vr_input import vr_frame_to_keypoint_frame
from . import motion as motion_policy
from .motion import MoveRefused
from .presets import PresetNotFound, PresetStore
from .ros_bridge import RosBridge
from . import safety
from .safety import Mode, ModeError
from .teleop import TeleopSession
from .telemetry import TelemetryBroadcaster
from .recorder import DatasetRecorder
from .sim.teleop import SimLeaderTeleop

logger = logging.getLogger(__name__)

VERSION = "0.1.0"

# Globals — wired in lifespan
cfg = load_config()
arms = ArmManager(cfg.arms, motion=cfg.motion)
cameras: CameraManager | None = None   # constructed in lifespan, after arms.connect_all()
ros = RosBridge(cfg.ros)
presets = PresetStore()
teleop = TeleopSession(arms)
# The bimanual collision guard needs a mount pose for every enabled arm; an
# arm it has no geometry for would silently pass every check, which is the
# fail-open a guard must not have. Missing mounts therefore disable it loudly.
_collision_guard = None
if cfg.collision.enabled:
    _unmounted = [a.id for a in cfg.arms
                  if a.enabled and a.id not in cfg.collision.mounts]
    if _unmounted:
        logger.warning(
            "collision guard DISABLED: no mounts configured for arms %s "
            "(add them under collision.mounts in config.yaml)", _unmounted)
    else:
        from .collision import CollisionGuard
        _collision_guard = CollisionGuard(cfg.collision)
human_teleop = HumanTeleopSession(arms, collision_guard=_collision_guard)
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


# ---- request schemas -----------------------------------------------------

class CmdVel(BaseModel):
    linear: float
    angular: float


class ArmGoal(BaseModel):
    model_config = ConfigDict(extra="allow")  # any subset of joint names

    # No declared fields — the joint dict comes through as `model_extra`
    # so we read it via `.__dict__`.


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
    left_arm: str
    right_arm: str
    swap: bool = False
    hz: float = 60.0
    # Typed, not a bare str: the source selected here is the session's sole
    # dead-man authority for its whole life, so an unrecognised value must be
    # rejected at the door rather than silently falling through to spacebar.
    clutch_source: ClutchSource = "spacebar"


class HumanTeleopSwapBody(BaseModel):
    swap: bool


class HumanPinchCalibSide(BaseModel):
    min_m: float
    max_m: float


class HumanMouthCalib(BaseModel):
    # Sustained levels over a hold window, not peaks — see safety.py. The
    # browser never derives these itself; it records traces and posts them to
    # /teleop/human/mouth/analyze, which returns exactly this shape.
    talk_hold: float
    open_hold: float
    talk_peak: float | None = None


class MouthTraceBody(BaseModel):
    """Recorded jawOpen traces as [t_ms, score] pairs, in arrival order.

    Stateless on purpose: the analyser is a pure function of the traces, so the
    same request can be replayed, unit-tested, and reasoned about without a
    live session. `verify` is checked against `calib` — the calibration the
    operator is about to trust — which is why it is supplied rather than read
    off the session.
    """

    talk: list[tuple[float, float]] | None = None
    open: list[tuple[float, float]] | None = None
    verify: list[tuple[float, float]] | None = None
    calib: HumanMouthCalib | None = None


class HumanTeleopCalibrateBody(BaseModel):
    left: HumanPinchCalibSide | None = None
    right: HumanPinchCalibSide | None = None
    mouth: HumanMouthCalib | None = None


class SimTeleopStartBody(BaseModel):
    follower: str
    hz: float = 60.0
    # Body of the leader source — one of:
    #   {"source": "mouse", "arm_name": "left"}
    #   {"source": "replay", "dataset_path": "/path/to/lerobot/dataset"}
    leader: dict


class RecordStartBody(BaseModel):
    repo_id: str          # e.g. "oscardvs/haller_pick_cube"
    task: str             # natural-language instruction logged with every frame


class RecordStopBody(BaseModel):
    save: bool = True     # False -> discard the episode buffer (bad take)


# ---- lifespan ------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global telemetry, cameras, recorder
    logger.info("haller-hmi backend starting (version %s)", VERSION)
    # Wire the ownership guard here, not as separate attach_peer calls below:
    # see ArmManager.connect_all's docstring and A6 in the plan.
    arms.connect_all(teleop_peers=[teleop, human_teleop, sim_teleop])
    cameras = CameraManager(cfg.cameras, world=arms.world())
    cameras.connect_all()
    ros.start()
    telemetry = TelemetryBroadcaster(arms, ros, hz=cfg.telemetry.hz,
                                     teleop=teleop, human_teleop=human_teleop, calibration=calibration)
    telemetry.start()
    recorder = DatasetRecorder(telemetry=telemetry, human_teleop=human_teleop, cameras=cameras)
    yield
    logger.info("haller-hmi backend shutting down")
    if recorder is not None:
        if recorder.status()["recording"]:
            await recorder.stop_episode(save=True)
        recorder.close()
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
    if body.clutch_source == "mouth" and not human_teleop.mouth_calib_is_valid():
        raise HTTPException(
            status_code=400,
            detail=("mouth clutch has no valid calibration: capture talk/open "
                    "and ensure their separation is at least 0.25"),
        )
    _arm_or_404(body.left_arm)
    _arm_or_404(body.right_arm)
    try:
        human_teleop.start(
            left_arm=body.left_arm, right_arm=body.right_arm,
            swap=body.swap, hz=body.hz, clutch_source=body.clutch_source,
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


@app.post("/teleop/human/swap")
async def post_human_teleop_swap(body: HumanTeleopSwapBody):
    human_teleop.set_swap(body.swap)
    return {"ok": True, **human_teleop.status()}


@app.post("/teleop/human/calibrate")
async def post_human_teleop_calibrate(body: HumanTeleopCalibrateBody):
    human_teleop.set_pinch_calib(
        left=body.left.model_dump() if body.left else None,
        right=body.right.model_dump() if body.right else None,
    )
    if body.mouth is not None:
        human_teleop.set_mouth_calib(body.mouth.model_dump())
    return {"ok": True}


@app.post("/teleop/human/mouth/analyze")
async def post_human_teleop_mouth_analyze(body: MouthTraceBody):
    """Derive a mouth calibration from recorded traces, and/or replay one.

    Two jobs, one route, because they are the same measurement seen twice:
      - talk+open  → the thresholds the operator would run with.
      - verify     → what those thresholds would have done to a fresh recording
                     of them speaking normally. `engaged: false` there is the
                     one safety claim this feature rests on, and it is the only
                     way to check it against a real person rather than against
                     an assumption about jaws in general.
    """
    out: dict = {}
    if body.talk is not None or body.open is not None:
        out.update(safety.analyze_mouth_capture(body.talk or [], body.open or []))
    if body.verify is not None:
        if body.calib is None:
            raise HTTPException(
                status_code=400,
                detail="verify needs a calib to check against",
            )
        out["verify"] = safety.replay_mouth_clutch(
            body.verify,
            safety.MouthClutchCalib(
                talk_hold=body.calib.talk_hold,
                open_hold=body.calib.open_hold,
                talk_peak=body.calib.talk_peak,
            ),
        )
    return out


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


# ---- dataset recording (HMI-integrated bimanual recorder, v0) -------------

@app.get("/record/status")
def get_record_status():
    if recorder is None:
        raise HTTPException(status_code=503, detail="recorder not ready")
    return recorder.status()


@app.post("/record/start")
async def post_record_start(body: RecordStartBody):
    if recorder is None:
        raise HTTPException(status_code=503, detail="recorder not ready")
    try:
        await recorder.start_episode(repo_id=body.repo_id, task=body.task)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, **recorder.status()}


@app.post("/record/stop")
async def post_record_stop(body: RecordStopBody):
    if recorder is None:
        raise HTTPException(status_code=503, detail="recorder not ready")
    status = await recorder.stop_episode(save=body.save)
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
    except UnmovedJointsError as e:
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


@app.websocket("/ws/teleop/human/in")
async def ws_human_teleop_in(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            frame = await ws.receive_json()
            try:
                human_teleop.ingest_frame(frame)
            except Exception:
                # Don't kill the socket over a bad frame — log and continue.
                logger.exception("human teleop ingest_frame failed")
    except WebSocketDisconnect:
        human_teleop.notify_ws_disconnected()
        return


@app.websocket("/ws/teleop/vr/in")
async def ws_vr_teleop_in(ws: WebSocket):
    """Raw WebXR frames from a headset browser.

    A separate socket from the MediaPipe one, but only at the door: the frame
    is converted to the same `KeypointFrame` and handed to the same
    `ingest_frame`. Two converters exist:

      - position mode (default): clutch-relative hand tracking through IK on
        the real chain — see vr_pose_mode.py for why angle copying cannot
        feel natural on this kinematics. Stateful per CONNECTION (the anchors
        live exactly as long as the operator's socket), hence constructed
        here and not module-level.
      - `vr_mode: "joints"`: the original angle-copying adapter
        (vr_input.py), kept for comparison and for the body-angle purists.
    """
    await ws.accept()
    from .vr_pose_mode import VRPoseMode
    pose_mode = VRPoseMode(human_teleop, arms)
    try:
        while True:
            frame = await ws.receive_json()
            try:
                if frame.get("vr_mode") == "joints":
                    kp = vr_frame_to_keypoint_frame(frame)
                else:
                    kp = pose_mode.convert(frame)
                human_teleop.ingest_frame(kp)
            except Exception:
                # Same rule as the MediaPipe socket: one malformed frame must
                # not drop the operator's connection mid-session.
                logger.exception("vr teleop ingest_frame failed")
    except WebSocketDisconnect:
        human_teleop.notify_ws_disconnected()
        return


def run() -> None:
    """Entry point for the `haller-hmi` console script."""
    import uvicorn
    uvicorn.run("haller_hmi.server:app", host="0.0.0.0", port=8000, log_level="info")
