"""Idle-timeout detection on the teleop websocket.

A half-open TCP connection (wifi drop without a FIN) leaves `receive_json`
blocked forever; nothing else in the stack notices, so the session's
WS-disconnect grace — the thing that auto-stops an abandoned session — never
runs. The handler now treats `WS_IDLE_TIMEOUT_S` of silence as a disconnect,
but only for a client that was actually streaming: see the `streamed` guard
in `ws_vr_teleop_in`, and `test_a_page_that_never_streamed_survives_idle` in
test_routes_vr_teleop.py.
"""
from __future__ import annotations

import asyncio

import pytest


class _SilentWS:
    """A websocket whose client vanished without closing."""

    def __init__(self):
        self.closed = False

    async def receive_json(self):
        await asyncio.sleep(3600)  # never delivers

    async def close(self):
        self.closed = True


class _TalkingWS:
    def __init__(self, frames):
        self._frames = list(frames)

    async def receive_json(self):
        return self._frames.pop(0)


async def test_silent_socket_returns_none(monkeypatch):
    import haller_hmi.server as srv
    monkeypatch.setattr(srv, "WS_IDLE_TIMEOUT_S", 0.05)
    ws = _SilentWS()
    t0 = asyncio.get_event_loop().time()
    frame = await srv._receive_or_idle_timeout(ws)
    assert frame is None
    assert asyncio.get_event_loop().time() - t0 < 1.0


async def test_frame_within_timeout_is_returned(monkeypatch):
    import haller_hmi.server as srv
    monkeypatch.setattr(srv, "WS_IDLE_TIMEOUT_S", 0.5)
    ws = _TalkingWS([{"hello": 1}])
    frame = await srv._receive_or_idle_timeout(ws)
    assert frame == {"hello": 1}


def test_endpoint_idle_disconnect_notifies_the_session(app_with_mocks, monkeypatch):
    """Full-path: a client that WAS streaming and then went quiet is dropped
    and the session is told, so the existing 5 s grace auto-stop can fire."""
    import haller_hmi.server as srv

    monkeypatch.setattr(srv, "WS_IDLE_TIMEOUT_S", 0.1)
    client = app_with_mocks
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/teleop/vr/in") as ws:
            ws.receive_json()               # the unprompted settings message
            ws.send_json({"type": "vr_keypoints", "ts_ms": 1,
                          "left": None, "right": None})
            # Then say nothing. The server must close us after the timeout.
            while True:
                ws.receive_json()
    assert srv.human_teleop.notify_ws_disconnected.called
