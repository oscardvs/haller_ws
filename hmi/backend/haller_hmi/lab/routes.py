# hmi/backend/haller_hmi/lab/routes.py
"""The one thing `server.py` imports from the Lab.

`build_lab_router` composes every Lab sub-router into a single router and
returns it, so mounting the whole surface is one line that never changes again
as sub-routers are added:

    from .lab.routes import build_lab_router

    app.include_router(build_lab_router(
        get_cameras=lambda: cameras,          # CameraManager | None
        get_recorder=lambda: recorder,        # DatasetRecorder | None
        lerobot_home=lerobot_home,            # () -> Path
        allow_remote_control=None,            # () -> bool; None reads the env
    ))

It REPLACES `build_data_router(...)` outright rather than mounting beside it:
the four legacy `/record` and `/cameras` paths live on the datasets router at
their existing URLs with their existing response shapes, proven by
`tests/lab/test_routes_compat.py` (which mounts both routers over the same
fakes and asserts equal responses) and by `tests/test_routes_data.py`'s own 31
tests run unmodified against this router.

**The arguments are ZERO-ARG CALLABLES, and that is load-bearing rather than
stylistic.** `server.py` mounts its routers at IMPORT time but builds the
`CameraManager` and `DatasetRecorder` inside `lifespan`. A router that closed
over the values would capture `None` for the entire life of the process and 503
forever. `routes_data.build_router` documents the same constraint at line 84;
it bit the 08-22 unification once already.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter

from ..api.deps import LabDeps
from .routes_datasets import build_datasets_router


def build_lab_router(
    *,
    get_cameras: Callable[[], object | None],
    get_recorder: Callable[[], object | None],
    lerobot_home: Callable[[], Path],
    allow_remote_control: Callable[[], bool] | None = None,
) -> APIRouter:
    """Every Lab route on one router.

    `allow_remote_control=None` means "read `HALLER_ALLOW_REMOTE_CONTROL` per
    call" — see `api/gate.py`. Per call, not once at import: caching it would
    mean a restart to change it.
    """
    deps = LabDeps(
        get_cameras=get_cameras,
        get_recorder=get_recorder,
        lerobot_home=lerobot_home,
        allow_remote_control=allow_remote_control,
    )

    router = APIRouter()
    # Sub-routers are included in the order a reader would look for them.
    # `/lab/runs/**` and `/lab/system` join here as they land; nothing about
    # the mount in `server.py` changes when they do, which is the point of
    # this file existing at all rather than server.py including three routers.
    router.include_router(build_datasets_router(deps))
    return router
