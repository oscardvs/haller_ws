"""The relay: the WebXR page, the broadcast hub, and the frame normaliser.

`normalize_frame` is where the two client shapes meet, so it gets most of the
attention — the reference stack's `xr_frame` (controllers nested, buttons as
an indexed gamepad array) has to land on exactly what the existing Next.js
page already sends, or the two headset pages diverge and one of them silently
stops being a usable fallback.
"""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from haller_hmi.vr_teleop import relay as vr_relay


def _button(pressed=False, value=0.0):
    return {"p": pressed, "v": value}


def _xr_frame(**kw):
    return {
        "type": "xr_frame",
        "ts_ms": 1234,
        "stance": "mirror",
        "viewer": {"position": [0, 1.5, 0], "orientation": [0, 0, 0, 1]},
        "controllers": {
            "right": {
                "position": [0.1, 1.2, -0.3],
                "orientation": [0, 0, 0, 1],
                "buttons": [_button(value=0.7),          # 0 trigger
                            _button(pressed=True),        # 1 grip
                            _button(),                    # 2
                            _button(),                    # 3 stick
                            _button(pressed=True)],       # 4 A/X
            },
            "left": None,
        },
        **kw,
    }


# ---- frame normalisation -------------------------------------------------

def test_reference_frame_shape_is_translated():
    out = vr_relay.normalize_frame(_xr_frame())
    assert out["type"] == "vr_keypoints"
    assert out["ts_ms"] == 1234
    assert out["stance"] == "mirror"
    assert out["head"]["orientation"] == [0, 0, 0, 1]
    assert out["left"] is None
    right = out["right"]
    assert right["tracked"] is True
    assert right["position"] == [0.1, 1.2, -0.3]
    assert right["trigger"] == pytest.approx(0.7)
    assert right["squeeze"] is True
    assert right["precision"] is True
    assert out["dead_man"] is True


def test_this_repos_frame_shape_passes_through_untouched():
    native = {"type": "vr_keypoints", "ts_ms": 7,
              "right": {"tracked": True, "position": [0, 0, 0],
                        "orientation": [0, 0, 0, 1], "trigger": 0.2,
                        "squeeze": False}}
    assert vr_relay.normalize_frame(native) is native


def test_a_short_button_array_does_not_explode():
    """Some runtimes report fewer buttons than the xr-standard mapping
    promises; a missing index must read as 'not pressed', not raise."""
    frame = _xr_frame()
    frame["controllers"]["right"]["buttons"] = [_button(value=0.4)]
    out = vr_relay.normalize_frame(frame)
    assert out["right"]["trigger"] == pytest.approx(0.4)
    assert out["right"]["squeeze"] is False
    assert out["right"]["precision"] is False
    assert out["dead_man"] is False


def test_missing_viewer_pose_yields_no_head():
    frame = _xr_frame()
    frame["viewer"] = {}
    assert vr_relay.normalize_frame(frame)["head"] is None


# ---- the router ----------------------------------------------------------

class _StubTeleop:
    def __init__(self):
        self.frames = []
        self.cfg = {"scale_translation": 1.0}

    def convert(self, frame):
        self.frames.append(frame)
        return {"type": "keypoints", "ts_ms": frame.get("ts_ms", 0)}

    def apply_config_update(self, update):
        self.cfg.update({k: v for k, v in update.items() if k in self.cfg})
        return dict(self.cfg)

    def state(self):
        return {"type": "ik_state", "config": dict(self.cfg), "sides": {}}


@pytest.fixture()
def relay_app():
    made, ingested, disconnects = [], [], []

    def make():
        t = _StubTeleop()
        made.append(t)
        return t

    app = FastAPI()
    app.include_router(vr_relay.build_router(
        make_teleoperator=make,
        ingest=ingested.append,
        on_disconnect=lambda: disconnects.append(1),
    ))
    return TestClient(app), made, ingested, disconnects


def test_serves_the_webxr_page_and_its_script(relay_app):
    client, *_ = relay_app
    page = client.get("/vr/")
    assert page.status_code == 200
    assert "Haller VR teleop" in page.text
    # The page loads the module by a RELATIVE path, which is what makes it
    # work behind Caddy's /api prefix and over `adb reverse` alike.
    assert 'src="client.js"' in page.text
    script = client.get("/vr/client.js")
    assert script.status_code == 200
    assert "immersive-ar" in script.text


def test_frames_reach_the_session(relay_app):
    client, made, ingested, _ = relay_app
    with client.websocket_connect("/vr/ws") as ws:
        assert ws.receive_json()["type"] == "ik_state"   # sent unprompted
        ws.send_json(_xr_frame())
        ws.send_json({"type": "request_settings"})
        assert ws.receive_json()["type"] in ("ik_state", "config_applied")
    assert len(ingested) == 1
    # ...and the teleoperator saw the NORMALISED frame, not the raw one.
    assert made[0].frames[0]["type"] == "vr_keypoints"


def test_config_update_is_echoed_clamped(relay_app):
    client, made, _, _ = relay_app
    with client.websocket_connect("/vr/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "config_update", "config": {"scale_translation": 2.5}})
        msgs = [ws.receive_json(), ws.receive_json()]
    kinds = {m["type"] for m in msgs}
    assert "config_applied" in kinds
    assert made[0].cfg["scale_translation"] == 2.5


def test_one_client_per_connection_gets_its_own_clutch_state(relay_app):
    """The anchors are connection state; two headsets must not share them."""
    client, made, _, _ = relay_app
    with client.websocket_connect("/vr/ws") as a:
        a.receive_json()
        with client.websocket_connect("/vr/ws") as b:
            b.receive_json()
    assert len(made) == 2
    assert made[0] is not made[1]


def test_a_bad_frame_does_not_drop_the_socket(relay_app):
    client, made, ingested, _ = relay_app

    def boom(_frame):
        raise ValueError("synthetic")

    with client.websocket_connect("/vr/ws") as ws:
        ws.receive_json()
        made[0].convert = boom
        ws.send_json(_xr_frame())
        # Still alive: a later well-formed message is still answered.
        ws.send_json({"type": "request_settings"})
        assert ws.receive_json()["type"] == "ik_state"
    assert ingested == []


def test_unknown_messages_are_relayed_to_other_clients(relay_app):
    """The hub is a hub: anything it does not understand still reaches the
    other subscribers."""
    client, *_ = relay_app
    with client.websocket_connect("/vr/ws") as a:
        a.receive_json()
        with client.websocket_connect("/vr/ws") as b:
            b.receive_json()
            a.send_json({"type": "note", "text": "hello"})
            assert b.receive_json() == {"type": "note", "text": "hello"}


def test_a_page_that_never_streamed_is_not_torn_down_when_idle(relay_app):
    """The idle timeout catches a headset that WAS driving and went quiet. A
    page merely open — parked on the landing screen, tuning sliders, not yet
    in XR — is not an operator who stopped, and closing it just makes it
    reconnect in a loop (62 times across one smoke run, before this)."""
    client, _, _, disconnects = relay_app
    with client.websocket_connect("/vr/ws") as ws:
        ws.receive_json()
        # Long enough to cross the idle timeout several times over.
        time.sleep(vr_relay.IDLE_TIMEOUT_S * 2.5)
        # Still alive and still answering.
        ws.send_json({"type": "request_settings"})
        assert ws.receive_json()["type"] == "ik_state"
        assert disconnects == []
    # ...and closing it does not start a grace window either: a watcher page
    # going away is not an operator leaving.
    assert disconnects == []


def test_a_client_that_streamed_and_then_went_quiet_IS_torn_down(relay_app):
    """The other half: once frames have flowed, silence means the operator's
    headset stopped talking and the session's grace window must start."""
    client, _, ingested, disconnects = relay_app
    with client.websocket_connect("/vr/ws") as ws:
        ws.receive_json()
        ws.send_json(_xr_frame())
        ws.receive_json()
        time.sleep(vr_relay.IDLE_TIMEOUT_S * 1.6)
        with pytest.raises(Exception):
            # The relay closed it; the next receive fails.
            while True:
                ws.receive_json()
    assert len(ingested) == 1
    assert disconnects == [1]
