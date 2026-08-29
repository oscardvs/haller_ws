# hmi/backend/tests/lab/test_gate.py
"""`require_local` and the exception ladder — the two things every `/lab` route
inherits without restating.

The gate exists because `--host 0.0.0.0` is how the Quest reaches the HMI, and
reaching the HMI must not also mean deleting a dataset or launching a training
job. Everything here is driven through a THROWAWAY two-route app rather than a
hand-built ASGI scope: what is under test is `request.client`, and a synthetic
scope would only be asserting about the scope the test itself wrote. The one
exception is the `client=None` case, which has no socket by definition.

The ladder is tested over the wire for the same reason — `{"detail": ...}` and
nothing else is the frozen contract with Track C, and a status that is right in
Python but re-wrapped by an exception handler on the way out is still a page
that cannot read its own errors.

No dataset, no hardware, no `HF_LEROBOT_HOME`. Nothing here touches disk.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from haller_hmi.api import errors, gate

#: A plausible client on Oscar's LAN — the Quest, or the laptop.
LAN_HOST = "192.168.1.50"


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    """`HALLER_ALLOW_REMOTE_CONTROL` set in the shell that ran pytest would
    turn every 403 assertion below into a 200. Clear it, per test."""
    monkeypatch.delenv(gate.REMOTE_CONTROL_ENV, raising=False)


def _gate_app(allow_remote_control=None) -> FastAPI:
    """One gated route and one ungated one.

    Both, because "the LAN client got a 403" only means the GATE refused if the
    same client can still reach the route beside it.
    """
    require_local = gate.build_require_local(allow_remote_control)
    app = FastAPI()

    @app.get("/gated")
    def gated(_: None = Depends(require_local)) -> dict:
        return {"ran": "gated"}

    @app.get("/open")
    def open_route() -> dict:
        return {"ran": "open"}

    return app


def _client(app: FastAPI, host: str) -> TestClient:
    return TestClient(app, client=(host, 51000))


# ---- who gets through ----

@pytest.mark.parametrize("host", sorted(gate.LOOPBACK))
def test_a_loopback_client_passes(host):
    """`localhost` is in the set alongside the two addresses because a proxy in
    front of the app can hand the NAME through rather than the address."""
    assert _client(_gate_app(), host).get("/gated").status_code == 200


def test_a_lan_client_is_refused_and_told_where_it_came_from():
    """The detail has to name the host and the escape hatch: this 403 is read
    inside a headset, where "forbidden" alone is indistinguishable from a
    broken route."""
    response = _client(_gate_app(), LAN_HOST).get("/gated")
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert LAN_HOST in detail
    assert gate.REMOTE_CONTROL_ENV in detail
    # The frozen shape — one key, no second envelope.
    assert set(response.json()) == {"detail"}


def test_the_ungated_route_still_answers_the_same_lan_client():
    """Triage from the headset is the job the HUD exists for: every GET, and
    `mark` / `bulk` / `autoclass/preview`, stay LAN-writable deliberately."""
    client = _client(_gate_app(), LAN_HOST)
    assert client.get("/open").status_code == 200
    assert client.get("/gated").status_code == 403


def test_allow_remote_control_true_lets_the_lan_client_through():
    app = _gate_app(lambda: True)
    assert _client(app, LAN_HOST).get("/gated").status_code == 200


def test_allow_remote_control_false_refuses_even_with_the_env_var_set(monkeypatch):
    """An explicit callable is the answer; it is not consulted alongside the
    environment. Two sources for one permission is how the one that is actually
    read stops being obvious."""
    monkeypatch.setenv(gate.REMOTE_CONTROL_ENV, "1")
    assert _client(_gate_app(lambda: False), LAN_HOST).get("/gated").status_code == 403


def test_the_env_var_is_read_per_call_not_captured(monkeypatch):
    """ONE gate, ONE client, three requests, and the answer follows the
    environment each time.

    Caching the variable at build time would mean a restart is the only way to
    change the answer — and the moment it matters is the one where Oscar is
    already holding the headset and wants a job launched from the couch. That
    is the bug this guards, and it is invisible to any test that builds a fresh
    gate per assertion.
    """
    client = _client(_gate_app(None), LAN_HOST)

    assert client.get("/gated").status_code == 403
    monkeypatch.setenv(gate.REMOTE_CONTROL_ENV, "1")
    assert client.get("/gated").status_code == 200
    monkeypatch.delenv(gate.REMOTE_CONTROL_ENV)
    assert client.get("/gated").status_code == 403


@pytest.mark.parametrize(
    ("value", "allowed"),
    [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("  yes  ", True),
        ("0", False), ("false", False), ("no", False), ("", False), ("maybe", False),
    ],
)
def test_remote_control_allowed_reads_the_environment(monkeypatch, value, allowed):
    monkeypatch.setenv(gate.REMOTE_CONTROL_ENV, value)
    assert gate.remote_control_allowed() is allowed


def test_remote_control_allowed_is_false_when_unset():
    assert gate.remote_control_allowed() is False


def test_a_clientless_request_is_local_because_refusing_would_403_every_in_process_test():
    """A hand-built ASGI scope carries no `client` key, and neither does
    `httpx.ASGITransport(client=None)`.

    Refusing there would 403 every in-process caller while changing NOTHING for
    a real remote client: one arrives over a socket and always has a peer
    address. So the gate treats the absence as local — which is a decision
    about test plumbing, not a hole, and is worth a test saying so.
    """
    require_local = gate.build_require_local(None)
    request = Request(
        {"type": "http", "method": "GET", "path": "/gated",
         "headers": [], "query_string": b""}
    )
    assert request.client is None
    assert require_local(request) is None


def test_testclients_default_client_is_NOT_loopback():
    """`TestClient` defaults to `client=("testclient", 50000)`, so it gets the
    403 — pinned here because every future Lab route test that wants past this
    gate has to pass `client=("127.0.0.1", 0)` or set the env var, and
    discovering that from a 403 in an unrelated test is an afternoon."""
    assert TestClient(_gate_app()).get("/gated").status_code == 403


# ---- the exception ladder ----
# One rung per row: the exception a route body raises, the status the contract
# gives it, and the `detail` string it must carry.

_LADDER = [
    pytest.param(
        lambda: errors.DataDependencyError("reading datasets needs pandas + pyarrow"),
        503, "reading datasets needs pandas + pyarrow", id="dependency-503",
    ),
    pytest.param(
        lambda: errors.DatasetBusyError("a recording session is most likely still running"),
        409, "a recording session is most likely still running", id="busy-409",
    ),
    pytest.param(
        lambda: FileNotFoundError("no dataset at /home/odesha/robot-data/lerobot/local/nope"),
        404, "no dataset at /home/odesha/robot-data/lerobot/local/nope", id="missing-404",
    ),
    pytest.param(
        # `str(KeyError(...))` keeps the key's repr QUOTES. That is the kit's
        # wording and the fixtures compare against it, so it is not tidied away.
        lambda: KeyError("observation.state"),
        404, "'observation.state'", id="keyerror-404",
    ),
    pytest.param(
        lambda: ValueError("filter_mark must be one of keep, reject, unset, got 'kept'"),
        400, "filter_mark must be one of keep, reject, unset, got 'kept'", id="badinput-400",
    ),
    pytest.param(
        # The catch-all names the TYPE, because a 500 from here reaches Oscar as
        # a toast in the headset and not as a server log he is reading.
        lambda: RuntimeError("the servo bus went away"),
        500, "RuntimeError: the servo bus went away", id="unexpected-500",
    ),
]


@pytest.mark.parametrize(("make", "status", "detail"), _LADDER)
def test_each_exception_maps_to_its_documented_status(make, status, detail):
    """Over the wire, and compared as a WHOLE body: `{"detail": ...}` and
    nothing else. A route that grew an `{"ok": false, "error": ...}` envelope
    would still pass a status-only assertion, and would make the page's error
    handling depend on which route it called."""
    app = FastAPI()

    @app.get("/boom")
    def boom() -> dict:
        with errors.as_http():
            raise make()

    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert response.status_code == status
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(("make", "status", "detail"), _LADDER)
def test_wrap_applies_the_same_ladder_as_the_context_manager(make, status, detail):
    """`wrap(fn, ...)` and `as_http()` are two spellings of one ladder; a rung
    that only exists in one of them is a route answering differently depending
    on how its author chose to call it."""
    def raiser():
        raise make()

    with pytest.raises(HTTPException) as caught:
        errors.wrap(raiser)
    assert caught.value.status_code == status
    assert caught.value.detail == detail


def test_an_httpexception_passes_through_untouched():
    """A route that chose a status the ladder has no rung for — the gate's 403,
    a hand-written 400 on a missing body field — must not have it rewritten on
    the way out. Same object, same status, same detail."""
    chosen = HTTPException(status_code=403, detail="limited to the machine running the HMI")
    with pytest.raises(HTTPException) as caught, errors.as_http():
        raise chosen
    assert caught.value is chosen
    assert caught.value.status_code == 403
    assert caught.value.detail == "limited to the machine running the HMI"


def test_the_gates_403_survives_the_ladder_over_the_wire():
    """The two modules meet here: a gated route whose body runs under
    `as_http()` still answers a LAN client with the gate's 403 and the gate's
    wording, not with a 500 the ladder manufactured."""
    require_local = gate.build_require_local(None)
    app = FastAPI()

    @app.get("/gated")
    def gated(request: Request) -> dict:
        with errors.as_http():
            require_local(request)
            return {"ran": "gated"}

    response = _client(app, LAN_HOST).get("/gated")
    assert response.status_code == 403
    assert LAN_HOST in response.json()["detail"]
    assert set(response.json()) == {"detail"}
