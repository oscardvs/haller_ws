"""The relay: a broadcast WebSocket hub plus the WebXR page the headset loads.

Ported from the reference stack's `relay/`, with its central property intact:
for pose and state messages the relay is a **pure broadcast hub**. Anything a
client sends goes to every other client, so any number of listeners — a second
headset, a desktop viewer, a smoke-test harness pretending to be a Quest — can
watch the same stream without the teleop knowing or caring.

What is different, and deliberately:

* **It is mounted inside the existing FastAPI app, not run as its own server.**
  WebXR only exists in a secure context, an HTTPS page needs WSS, and a
  certificate warning *cannot be accepted for a WebSocket* — which is why this
  rig serves page, API and sockets from ONE origin behind Caddy. A relay on
  its own port would be a second origin and would reintroduce exactly the
  failure the single-origin setup was built to remove. Mounted here it
  inherits the origin for free, and `adb reverse tcp:8444 tcp:8444` over USB
  keeps working too.

* **The teleop consumer is in-process.** The reference has a separate teleop
  process subscribe to `/ws`; here `QuestTeleoperator` is constructed per
  connection and fed directly, because the thing that drives the arms is
  `HumanTeleopSession` in this same process and a loopback WebSocket between
  them would buy nothing but latency. The message schema is unchanged, so an
  external subscriber still works.

* **Two frame shapes are accepted.** The reference client sends
  `{type: "xr_frame", controllers: {left, right}, viewer: {...}}`; this repo's
  existing Next.js page sends `{type: "vr_keypoints", left, right, head}`.
  Both are normalised at the door, so the ported client and the existing page
  can drive the same backend — which is what makes the old page usable as a
  fallback rather than something that has to be kept in lockstep.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

#: How often the relay pushes `ik_state` back to the clients. The headset
#: uses it to drive haptics and paint the HUD, neither of which needs the
#: frame rate; the pose stream flows the other way at display rate.
STATE_HZ = 20.0

#: Idle timeout on the socket. Matches the existing teleop sockets: a headset
#: that has gone quiet for this long is treated as disconnected so the
#: session's grace window starts, rather than the arms sitting live behind a
#: socket nobody is on the other end of.
IDLE_TIMEOUT_S = 2.0


class Hub:
    """Broadcast fan-out over the connected clients.

    Deliberately unaware of message meaning: the teleop reads what it cares
    about on the way past, and everything else is simply relayed. Sends are
    best-effort — one wedged client must not stall the operator's, so a
    failed send drops that client rather than raising.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def join(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def leave(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    @property
    def count(self) -> int:
        return len(self._clients)

    async def broadcast(self, message: dict, exclude: WebSocket | None = None) -> None:
        async with self._lock:
            targets = [c for c in self._clients if c is not exclude]
        for client in targets:
            try:
                await client.send_json(message)
            except Exception:
                await self.leave(client)


def normalize_frame(msg: dict) -> dict:
    """Accept either client's frame shape and return this repo's.

    The reference client nests controllers under `controllers` and the head
    pose under `viewer`, and reports buttons as an indexed array following the
    WebXR `xr-standard` gamepad mapping. Translating here rather than in two
    clients means the converter, the session and the recorder only ever see
    one shape.
    """
    if msg.get("type") != "xr_frame":
        return msg
    ctrls = msg.get("controllers") or {}
    out: dict = {
        "type": "vr_keypoints",
        "ts_ms": int(msg.get("ts_ms") or 0),
        "stance": msg.get("stance"),
        "vr_mode": msg.get("vr_mode"),
    }
    viewer = msg.get("viewer") or {}
    if viewer.get("orientation") is not None:
        out["head"] = {"position": viewer.get("position") or [0.0, 0.0, 0.0],
                       "orientation": viewer["orientation"]}
    else:
        out["head"] = None
    dead_man = False
    for side in ("left", "right"):
        raw = ctrls.get(side)
        if not isinstance(raw, dict):
            out[side] = None
            continue
        buttons = raw.get("buttons") or []
        squeeze = bool(_button(buttons, _BUTTON_SQUEEZE, "p", False))
        dead_man = dead_man or squeeze
        out[side] = {
            "tracked": bool(raw.get("tracked", True)),
            "position": raw.get("position") or [0.0, 0.0, 0.0],
            "orientation": raw.get("orientation") or [0.0, 0.0, 0.0, 1.0],
            "trigger": float(_button(buttons, _BUTTON_TRIGGER, "v", 0.0) or 0.0),
            "squeeze": squeeze,
            "precision": bool(_button(buttons, _BUTTON_AX, "p", False)),
        }
    out["dead_man"] = dead_man
    return out


# `xr-standard` gamepad indices. Index, not name, is what the WebXR spec
# guarantees — the same constants the Next.js client uses.
_BUTTON_TRIGGER = 0
_BUTTON_SQUEEZE = 1
_BUTTON_AX = 4


def _button(buttons: list, index: int, field: str, default):
    """One entry of a gamepad button array, tolerating a short one.

    Some runtimes report fewer buttons than the xr-standard mapping promises,
    so a missing index has to read as "not pressed" rather than raise — a
    controller with an unexpected button count must not take the session down.
    """
    if len(buttons) > index and isinstance(buttons[index], dict):
        return buttons[index].get(field, default)
    return default


def build_router(*, make_teleoperator, ingest, on_disconnect,
                 hub: Hub | None = None) -> APIRouter:
    """Wire the relay into an app.

    Injected rather than imported so this module never reaches back into
    `server.py` (which imports it) and so the tests can drive a hub with no
    arms attached:

        make_teleoperator() -> QuestTeleoperator   one per connection
        ingest(keypoint_frame)                     hand a converted frame on
        on_disconnect()                            the socket went away
    """
    router = APIRouter()
    hub = hub or Hub()
    router.state_hub = hub          # type: ignore[attr-defined]  (for tests)

    # `no-store`, deliberately. These two files are edited between bench runs
    # and reloaded in a headset that is already on the operator's face, and a
    # browser that serves a cached `client.js` looks exactly like a change
    # that did not take — the tuning you just added is simply absent, with no
    # error anywhere. Caught during bring-up, where a stale client kept
    # calling an endpoint that had been renamed. The files are ~30 kB on a
    # LAN or a USB cable; there is nothing to save here.
    _NO_STORE = {"cache-control": "no-store, must-revalidate"}

    @router.get("/vr", include_in_schema=False)
    async def vr_page_slash() -> Response:
        """Send `/vr` to `/vr/`; do NOT serve the page from both.

        Everything this page fetches is resolved relative to its own URL —
        `<script src="client.js">` in the markup, and `api()`'s `../config`,
        `../teleop/human/start`, `../estop` in the client. One trailing
        slash decides what those mean. Served from `/api/vr`, `client.js`
        resolves to `/api/client.js` and 404s, so the page arrives with no
        JavaScript at all: the arm selectors stay empty and Start does
        nothing, with no error anywhere to say why. Served from `/api/vr/`
        the same markup resolves to `/api/vr/client.js` and `/api/config`,
        which is what the proxy actually routes to the backend.

        Relative Location on purpose. Behind Caddy the browser asked for
        `/api/vr` while this app only ever sees `/vr`, so an absolute
        `/vr/` would send the headset to a path Next.js owns. `vr/`
        resolves against whatever the browser actually requested, under the
        proxy or straight at :8000.

        307, not 308: a permanent redirect is cacheable, and this is the
        one page that must never be served from a Quest's stale cache —
        the same reason _NO_STORE exists above.
        """
        return RedirectResponse(url="vr/", status_code=307, headers=_NO_STORE)

    @router.get("/vr/", include_in_schema=False)
    async def vr_page() -> Response:
        return FileResponse(WEB_DIR / "index.html",
                            media_type="text/html; charset=utf-8",
                            headers=_NO_STORE)

    @router.get("/vr/client.js", include_in_schema=False)
    async def vr_client() -> Response:
        return FileResponse(WEB_DIR / "client.js",
                            media_type="text/javascript; charset=utf-8",
                            headers=_NO_STORE)

    @router.websocket("/vr/ws")
    async def vr_ws(ws: WebSocket) -> None:
        await ws.accept()
        await hub.join(ws)
        teleop = make_teleoperator()
        last_state = 0.0
        state_period = 1.0 / STATE_HZ
        # Whether this client has ever streamed a pose. The idle timeout
        # exists to catch a headset that WAS driving and went quiet; a page
        # that is merely open — parked on the landing screen, tuning sliders,
        # not yet in XR — is not an operator who has stopped, and tearing it
        # down every two seconds just makes it reconnect. Measured: 62
        # pointless reconnects across one smoke run, each one a log line.
        streamed = False
        # Send the current settings straight away so the panel's sliders
        # start on the robot's actual values rather than on their own
        # defaults — the reference does this with a `request_settings`
        # round trip; sending unprompted costs one message and removes a
        # window where the two disagree.
        try:
            await ws.send_json(teleop.state())
        except Exception:
            pass
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive_json(),
                                                 timeout=IDLE_TIMEOUT_S)
                except TimeoutError:
                    if not streamed:
                        continue        # never drove; nothing has stopped
                    logger.warning("vr relay socket idle %.1fs; treating as "
                                   "disconnected", IDLE_TIMEOUT_S)
                    on_disconnect()
                    await ws.close()
                    return
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if mtype in ("xr_frame", "vr_keypoints"):
                    streamed = True
                    frame = normalize_frame(msg)
                    try:
                        ingest(teleop.convert(frame))
                    except Exception:
                        # One malformed frame must never drop the operator's
                        # connection mid-session — the same rule the existing
                        # teleop sockets follow.
                        logger.exception("vr relay: frame conversion failed")
                    now = time.monotonic()
                    if now - last_state >= state_period:
                        last_state = now
                        try:
                            await ws.send_json(teleop.state())
                        except Exception:
                            pass
                elif mtype == "config_update":
                    applied = teleop.apply_config_update(msg.get("config") or {})
                    # Echo the CLAMPED values back, so a slider that asked
                    # for something out of range snaps to what the robot
                    # actually took instead of silently disagreeing with it.
                    await hub.broadcast({"type": "config_applied",
                                         "config": applied})
                    try:
                        await ws.send_json(teleop.state())
                    except Exception:
                        pass
                elif mtype == "request_settings":
                    await ws.send_json(teleop.state())
                else:
                    await hub.broadcast(msg, exclude=ws)
        except WebSocketDisconnect:
            # Same rule as the idle path: only a client that was actually
            # streaming poses has an operator whose disappearance should
            # start the session's grace window. A watcher page closing is not
            # an operator leaving.
            if streamed:
                on_disconnect()
        finally:
            await hub.leave(ws)

    return router
