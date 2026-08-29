"""The VR teleop surface: the start/collision routes, and the one socket.

`app_with_mocks` stubs the session itself, so what these check is the
CONTRACT — validation, status codes, what actually reaches the session, and
the socket's message protocol. The session's own behaviour is pinned in
`test_human_teleop*.py` against a real session and real arms; the converter's
in `tests/vr_teleop/`.
"""
from __future__ import annotations

import time

import pytest

import haller_hmi.server as srv


def test_single_arm_start_reaches_the_session_with_one_side_null(app_with_mocks):
    client = app_with_mocks
    res = client.post("/teleop/human/start", json={"right_arm": "right"})
    assert res.status_code == 200, res.text
    srv.human_teleop.start.assert_called_once()
    kwargs = srv.human_teleop.start.call_args.kwargs
    assert kwargs["left_arm"] is None
    assert kwargs["right_arm"] == "right"


def test_start_with_no_arms_is_a_400(app_with_mocks):
    client = app_with_mocks
    res = client.post("/teleop/human/start", json={})
    assert res.status_code == 400
    assert "at least one" in res.json()["detail"]


def test_start_still_404s_an_unknown_arm(app_with_mocks):
    client = app_with_mocks
    res = client.post("/teleop/human/start", json={"right_arm": "nope"})
    assert res.status_code == 404


def test_collision_toggle_round_trip(app_with_mocks):
    client = app_with_mocks
    off = client.post("/teleop/human/collision", json={"enabled": False})
    assert off.status_code == 200, off.text
    assert off.json()["collision"]["enabled"] is False
    assert srv._collision_guard.enabled is False

    on = client.post("/teleop/human/collision", json={"enabled": True})
    assert on.status_code == 200, on.text
    assert on.json()["collision"]["enabled"] is True
    assert srv._collision_guard.enabled is True


def test_toggle_answers_before_a_session_has_ever_run(app_with_mocks):
    """The reply is read off the guard, not off the session's clearance
    read-out — which does not exist until the commit loop has ticked. An
    operator flipping the switch before starting a session would otherwise
    get `null` back and read it as 'the toggle did nothing'."""
    client = app_with_mocks
    body = client.post("/teleop/human/collision", json={"enabled": False}).json()
    assert body["collision"]["available"] is True
    assert isinstance(body["collision"]["margin_m"], float)
    client.post("/teleop/human/collision", json={"enabled": True})


def test_enabling_an_unavailable_guard_is_refused(app_with_mocks, monkeypatch):
    """A guard with no mount geometry would pass every check for the arm it
    has none for — the fail-open the module exists to prevent. Refuse loudly
    rather than accept a toggle that silently does nothing."""
    client = app_with_mocks
    monkeypatch.setattr(srv._collision_guard, "available", False)
    res = client.post("/teleop/human/collision", json={"enabled": True})
    assert res.status_code == 409
    assert "mount geometry" in res.json()["detail"]


# ---- the one teleop socket ----------------------------------------------
#
# `/ws/teleop/vr/in` absorbed the relay: pose frames in (either wire shape),
# `ik_state` back at 20 Hz, and the live-tuning round trip. There is no other
# teleop socket and no `vr_mode` dispatch — every frame takes the ik path.

def _xr_ctrl(squeeze=True):
    return {"tracked": True, "position": [0.1, 1.2, -0.3],
            "orientation": [0, 0, 0, 1], "trigger": 0.0, "squeeze": squeeze}


def _kp_frame(**kw):
    # `dead_man` included because the native page sends it (derived from the
    # squeezes) — and the pin below is that the frame passes the door RAW.
    return {"type": "vr_keypoints", "ts_ms": 1234, "dead_man": True,
            "head": {"position": [0, 1.5, 0], "orientation": [0, 0, 0, 1]},
            "left": None, "right": _xr_ctrl(), **kw}


def test_settings_arrive_unprompted_on_connect(app_with_mocks):
    """One message, and it removes the window where the client's sliders and
    the robot's actual config disagree."""
    with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "settings"
    assert msg["config"]["scale_translation"] == pytest.approx(1.0)


def test_a_frame_reaches_the_session_raw(app_with_mocks):
    """The socket converts NOTHING: the session stores the raw frame and
    solves it at its own 60 Hz tick (the kit's loop shape). A per-frame
    solve here — seeded from the session's throttled committed pose — is the
    structure the audit found bleeding hand-to-tool correspondence away."""
    with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as ws:
        ws.receive_json()
        ws.send_json(_kp_frame())
        ws.receive_json()               # the ik_state that follows the frame
    frame = srv.human_teleop.ingest_frame.call_args.args[0]
    assert frame["type"] == "vr_keypoints"
    assert frame["dead_man"] is True
    assert frame["left"] is None
    assert frame["right"]["squeeze"] is True
    assert frame["right"]["position"] == [0.1, 1.2, -0.3]
    assert "joint_goal" not in (frame["right"] or {})


def test_the_xr_standard_frame_shape_is_accepted_too(app_with_mocks):
    """Normalised at the door, so the session and the recorder only ever see
    one shape — and a client written against the WebXR gamepad mapping still
    drives the rig."""
    with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as ws:
        ws.receive_json()
        ws.send_json({
            "type": "xr_frame", "ts_ms": 7,
            "viewer": {"position": [0, 1.5, 0], "orientation": [0, 0, 0, 1]},
            "controllers": {"left": None, "right": {
                "position": [0.1, 1.2, -0.3], "orientation": [0, 0, 0, 1],
                "buttons": [{"p": False, "v": 0.0}, {"p": True, "v": 1.0}],
            }},
        })
        ws.receive_json()
    frame = srv.human_teleop.ingest_frame.call_args.args[0]
    assert frame["type"] == "vr_keypoints"
    assert frame["ts_ms"] == 7
    assert frame["right"]["squeeze"] is True
    assert frame["left"] is None


def test_ik_state_is_pushed_while_frames_flow(app_with_mocks):
    """The HUD's telemetry channel: per-side conditioning and the orientation
    residual, at a rate that does not need to be the frame rate."""
    with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as ws:
        ws.receive_json()
        ws.send_json(_kp_frame())
        msg = ws.receive_json()
    assert msg["type"] == "ik_state"
    assert set(msg["sides"]) == {"left", "right"}
    assert "config" in msg


def test_config_update_is_echoed_clamped(app_with_mocks):
    """A slider that asks for something out of range must snap to what the
    robot actually took, rather than silently disagreeing with it."""
    with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as ws:
        ws.receive_json()
        ws.send_json({"type": "config_update",
                      "config": {"scale_translation": 99.0}})
        msg = ws.receive_json()
    assert msg["type"] == "config_applied"
    assert msg["config"]["scale_translation"] == pytest.approx(4.0)


def test_request_settings_answers_with_the_whole_config(app_with_mocks):
    with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as ws:
        ws.receive_json()
        ws.send_json({"type": "config_update",
                      "config": {"scale_translation": 2.5}})
        ws.receive_json()
        ws.send_json({"type": "request_settings"})
        msg = ws.receive_json()
    assert msg["type"] == "settings"
    assert msg["config"]["scale_translation"] == pytest.approx(2.5)


def test_the_config_is_one_shared_instance_across_connections(app_with_mocks):
    """What replaced the per-connection converter, deliberately.

    The clutch anchors moved INTO the session (`KitSideTeleop` per driven
    side) — they live exactly as long as the session, not the socket — so
    the config that steers them must be the session's one shared instance.
    A per-connection copy (the old model this test used to pin) would leave
    a reconnecting headset tuning a config nothing was driving with: a
    slider written on one socket must be the value every other socket reads
    back."""
    with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as a:
        a.receive_json()
        a.send_json({"type": "config_update",
                     "config": {"scale_translation": 2.5}})
        assert a.receive_json()["type"] == "config_applied"
        with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as b:
            greeting = b.receive_json()
    assert greeting["type"] == "settings"
    assert greeting["config"]["scale_translation"] == pytest.approx(2.5)


def test_a_bad_frame_does_not_drop_the_socket(app_with_mocks):
    """One malformed frame must never end the operator's session."""
    srv.human_teleop.ingest_frame.side_effect = ValueError("synthetic")
    try:
        with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as ws:
            ws.receive_json()
            ws.send_json(_kp_frame())
            ws.receive_json()
            # Still alive: a later well-formed message is still answered.
            ws.send_json({"type": "request_settings"})
            assert ws.receive_json()["type"] == "settings"
    finally:
        srv.human_teleop.ingest_frame.side_effect = None


def test_a_page_that_never_streamed_survives_idle(app_with_mocks, monkeypatch):
    """The idle timeout catches a headset that WAS driving and went quiet. A
    page merely open — parked on the landing screen, tuning sliders, not yet
    in XR — is not an operator who stopped, and closing it just makes it
    reconnect in a loop (62 times across one smoke run, before this)."""
    monkeypatch.setattr(srv, "WS_IDLE_TIMEOUT_S", 0.05)
    with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as ws:
        ws.receive_json()
        time.sleep(0.2)                 # several timeouts' worth
        ws.send_json({"type": "request_settings"})
        assert ws.receive_json()["type"] == "settings"
    # ...and closing it does not start a grace window either: a watcher page
    # going away is not an operator leaving.
    assert not srv.human_teleop.notify_ws_disconnected.called


def test_closing_after_streaming_starts_the_grace_window(app_with_mocks):
    with app_with_mocks.websocket_connect("/ws/teleop/vr/in") as ws:
        ws.receive_json()
        ws.send_json(_kp_frame())
        ws.receive_json()
    assert srv.human_teleop.notify_ws_disconnected.called


def test_the_legacy_teleop_sockets_and_pages_are_gone(app_with_mocks):
    """One input path. A route that still answered here would be a second one
    nobody maintains — which is what this refactor removed."""
    client = app_with_mocks
    for path in ("/vr", "/vr/", "/vr/client.js"):
        assert client.get(path).status_code == 404, path
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/teleop/human/in"):
            pass
