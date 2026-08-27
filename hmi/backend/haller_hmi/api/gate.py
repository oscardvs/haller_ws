# hmi/backend/haller_hmi/api/gate.py
"""`require_local` — the reason `--host 0.0.0.0` is not also a remote console.

The HMI binds every interface because that is how the Quest reaches it over the
LAN. Reaching the HMI must not also mean deleting a dataset or launching a
training job, so the destructive and process-starting endpoints refuse a
non-loopback client:

    autoclass/apply, autoclass/revert, prune,
    runs/train, runs/{id}/stop, DELETE runs/{id}

Everything else stays LAN-writable **deliberately** — every GET,
`datasets/mark`, `datasets/bulk`, and `autoclass/preview` (which writes
nothing). Triage is the thing Oscar actually does from inside the headset: mark
a fumbled take reject the moment he sees it, before taking the headset off.
Gating those would push him back to the desk to do the one job the HUD exists
for.

The escape hatch is the environment variable `HALLER_ALLOW_REMOTE_CONTROL`.
"""
from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import HTTPException, Request

#: Hosts that count as "this machine". `localhost` is in the set because a
#: proxy in front of the app can hand the name through rather than the address.
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

REMOTE_CONTROL_ENV = "HALLER_ALLOW_REMOTE_CONTROL"

_TRUTHY = frozenset({"1", "true", "yes"})


def remote_control_allowed() -> bool:
    """`HALLER_ALLOW_REMOTE_CONTROL`, read from the environment RIGHT NOW.

    Not cached at import: caching it means the only way to change the answer is
    to restart the server, and the moment that matters is the one where Oscar
    is holding the headset and wants a job launched from the couch.
    """
    return os.environ.get(REMOTE_CONTROL_ENV, "").strip().lower() in _TRUTHY


def build_require_local(
    allow_remote_control: Callable[[], bool] | None,
) -> Callable[[Request], None]:
    """Build the gate the destructive routes call.

        allow_remote_control()  -> bool
        None                    -> read the environment variable each call

    Zero-arg callable rather than a bool for the same reason the rest of the
    injected handles are (see `deps.LabDeps`): the routers are built once, at
    import time, and a bool captured then is a bool for the life of the
    process.

    The returned callable takes a `Request` and returns `None`, so a route can
    either call it (`require_local(request)`) or mount it as a FastAPI
    dependency — the signature satisfies both.
    """
    allowed = allow_remote_control or remote_control_allowed

    def require_local(request: Request) -> None:
        if allowed():
            return
        # `request.client` is None whenever the ASGI scope carries no `client`
        # key — a hand-built scope, `httpx.ASGITransport(client=None)`. Treat
        # that as LOCAL: a real remote client arrives over a socket and ALWAYS
        # has a peer address, so refusing a None here buys no safety and only
        # 403s in-process callers. (Measured: `fastapi.testclient.TestClient`
        # is NOT one of them — it defaults to `client=("testclient", 50000)`
        # and so gets the 403 below. A test that wants past this gate passes
        # `TestClient(app, client=("127.0.0.1", 0))`, or sets the env var.)
        host = request.client.host if request.client else None
        if host is None or host in LOOPBACK:
            return
        raise HTTPException(
            status_code=403,
            detail=(
                f"this action is limited to the machine running the HMI "
                f"(request came from {host}). Set {REMOTE_CONTROL_ENV}=1 in the "
                f"server's environment to permit it from the LAN."
            ),
        )

    return require_local
