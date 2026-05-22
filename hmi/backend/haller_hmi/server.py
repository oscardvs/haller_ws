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
from .human_teleop import HumanTeleopSession
from .presets import PresetNotFound, PresetStore
from .ros_bridge import RosBridge
from .safety import Mode, ModeError
from .teleop import TeleopSession
from .telemetry import TelemetryBroadcaster

logger = logging.getLogger(__name__)

VERSION = "0.1.0"

# Globals — wired in lifespan
cfg = load_config()
arms = ArmManager(cfg.arms)
cameras = CameraManager(cfg.cameras)
ros = RosBridge(cfg.ros)
presets = PresetStore()
teleop = TeleopSession(arms)
human_teleop = HumanTeleopSession(arms)
teleop.attach_peer(human_teleop)
human_teleop.attach_peer(teleop)
calibration = CalibrationManager()
telemetry: TelemetryBroadcaster | None = None


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


# ---- lifespan ------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global telemetry
    logger.info("haller-hmi backend starting (version %s)", VERSION)
    arms.connect_all()
    cameras.connect_all()
    ros.start()
    telemetry = TelemetryBroadcaster(arms, ros, hz=cfg.telemetry.hz,
                                     teleop=teleop, calibration=calibration)
    telemetry.start()
    yield
    logger.info("haller-hmi backend shutting down")
    if telemetry is not None:
        await telemetry.stop()
    teleop.stop()
    human_teleop.stop()
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
    handle = _arm_or_404(arm_id)
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
        sent = handle.home()
    except ModeError as e:
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
        clamped = handle.send_goal(goal)
    except ModeError as e:
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
    calibration.abort()
    teleop.stop()
    human_teleop.stop()
    for handle in arms.values():
        handle.disable_torque()
        handle.guard.set(Mode.STOP)
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


def run() -> None:
    """Entry point for the `haller-hmi` console script."""
    import uvicorn
    uvicorn.run("haller_hmi.server:app", host="0.0.0.0", port=8000, log_level="info")
