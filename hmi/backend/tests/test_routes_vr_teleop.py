"""HTTP surface for the two things a bench session needs to change live:
which arms a session runs on, and whether the collision guard clamps.

`app_with_mocks` stubs the session itself, so what these check is the HTTP
contract — validation, status codes, and what actually reaches the session.
The session's own single-arm behaviour is pinned in
`test_human_teleop_single_arm.py`, against a real session and real arms.
"""
from __future__ import annotations

import haller_hmi.server as srv


def test_single_arm_start_reaches_the_session_with_one_side_null(app_with_mocks):
    client = app_with_mocks
    res = client.post("/teleop/human/start",
                      json={"right_arm": "right", "clutch_source": "vr_grip"})
    assert res.status_code == 200, res.text
    srv.human_teleop.start.assert_called_once()
    kwargs = srv.human_teleop.start.call_args.kwargs
    assert kwargs["left_arm"] is None
    assert kwargs["right_arm"] == "right"


def test_start_with_no_arms_is_a_400(app_with_mocks):
    client = app_with_mocks
    res = client.post("/teleop/human/start", json={"clutch_source": "vr_grip"})
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


def test_the_relay_page_is_mounted_on_the_app(app_with_mocks):
    """Same origin as the API and the sockets. A separate origin would bring
    back the certificate interstitial that cannot be accepted for a
    WebSocket, which is the whole reason this rig has one origin."""
    client = app_with_mocks
    assert client.get("/vr/").status_code == 200
    assert client.get("/vr/client.js").status_code == 200


def test_bare_vr_redirects_to_the_trailing_slash(app_with_mocks):
    """`/vr` must redirect, never serve the page in place.

    Bench, 2026-08-21, Quest on `https://<host>:8444/api/vr`: both arm
    selectors empty and Start inert. The page had been served from the bare
    path, so `<script src="client.js">` resolved to `/api/client.js` and
    404'd — the markup arrived with no client at all, and nothing in the UI
    could say so.
    """
    client = app_with_mocks
    res = client.get("/vr", follow_redirects=False)
    assert res.status_code == 307
    # Relative: behind the proxy the browser asked for /api/vr, and an
    # absolute /vr/ would land on the Next.js side of the origin.
    assert res.headers["location"] == "vr/"
    assert "no-store" in res.headers.get("cache-control", "")


def test_bare_vr_followed_lands_on_the_page(app_with_mocks):
    client = app_with_mocks
    res = client.get("/vr", follow_redirects=True)
    assert res.status_code == 200
    assert "client.js" in res.text
