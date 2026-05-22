# hmi/backend/haller_hmi/server.py
"""FastAPI app. The only place that ties lerobot, ROS, presets, and HTTP together."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .arm import ArmManager
from .config import load_config
from .presets import PresetNotFound, PresetStore
from .ros_bridge import RosBridge
from .safety import Mode, ModeError
from .telemetry import TelemetryBroadcaster

logger = logging.getLogger(__name__)

VERSION = "0.1.0"

# Globals — wired in lifespan
cfg = load_config()
arms = ArmManager(cfg.arms)
ros = RosBridge(cfg.ros)
presets = PresetStore()
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


# ---- lifespan ------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global telemetry
    logger.info("haller-hmi backend starting (version %s)", VERSION)
    arms.connect_all()
    ros.start()
    telemetry = TelemetryBroadcaster(arms, ros, hz=cfg.telemetry.hz)
    telemetry.start()
    yield
    logger.info("haller-hmi backend shutting down")
    if telemetry is not None:
        await telemetry.stop()
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


@app.post("/estop")
async def post_estop():
    logger.warning("E-STOP triggered")
    for handle in arms.values():
        handle.disable_torque()
        handle.guard.set(Mode.STOP)
    ros.zero_cmd_vel()
    return {"ok": True}


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
