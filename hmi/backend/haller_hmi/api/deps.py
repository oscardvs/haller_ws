# hmi/backend/haller_hmi/api/deps.py
"""The handles every `/lab` route resolves per request.

Four zero-arg callables, and their being callables is load-bearing rather than
stylistic. `server.py` mounts its routers at IMPORT time but builds the
`CameraManager` and `DatasetRecorder` inside `lifespan`; a router that closed
over the values would capture `None` for the entire life of the process and
503 forever. `haller_hmi/routes_data.py:84` (`build_router`) already documents
this — this is the same rule, carried into `build_lab_router`, not a second
derivation of it.

    get_cameras()   -> CameraManager | None
    get_recorder()  -> DatasetRecorder | None
    lerobot_home()  -> Path             the dataset cache root to scan
    allow_remote_control  -> () -> bool, or None meaning "read
                            HALLER_ALLOW_REMOTE_CONTROL each call"
                            (see `gate.build_require_local`)

The concrete types are named in prose only. Annotating them means importing
`recorder.py`, and that module imports lerobot — the one import this package is
forbidden to make, because `/lab` runs in the serving process, which is the
teleop latency path.

Two flavours of accessor, because the routes genuinely split:

* `cameras()` / `recorder()` raise 503 for a route whose whole answer depends
  on live hardware.
* `cameras_or_none()` / `recorder_or_none()` for a route that reads the
  dataset off disk and only consults the recorder to decorate the answer.
  Those must still work with the arms unplugged — reviewing yesterday's
  episodes does not need a robot attached.

`require_local` is NOT built here. `lab/routes.py` builds it once from
`deps.allow_remote_control` and hands it to the routes modules that need it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException


@dataclass(frozen=True)
class LabDeps:
    """What `build_lab_router` is handed, passed down to each routes module."""

    get_cameras: Callable[[], Any]
    get_recorder: Callable[[], Any]
    lerobot_home: Callable[[], Path]
    allow_remote_control: Callable[[], bool] | None = None

    # ---- resolved, or a clear 503 ----------------------------------------

    def cameras(self) -> Any:
        cameras = self.get_cameras()
        if cameras is None:
            raise HTTPException(status_code=503, detail="cameras not ready")
        return cameras

    def recorder(self) -> Any:
        recorder = self.get_recorder()
        if recorder is None:
            # This exact string, because the four legacy `/record`, `/cameras`
            # paths keep their existing response shapes when the lab router
            # replaces `build_data_router`, and `routes_data._require_recorder`
            # answers with it today.
            raise HTTPException(status_code=503, detail="recorder not ready")
        return recorder

    # ---- resolved, absence tolerated -------------------------------------

    def cameras_or_none(self) -> Any | None:
        return self.get_cameras()

    def recorder_or_none(self) -> Any | None:
        return self.get_recorder()

    def home(self) -> Path:
        """The dataset cache root as a `Path`, whatever the injector returned.

        Not `.resolve()`d here: `lab/catalog.hf_home()` owns that, and doing it
        in two places is how one of them ends up resolving and the other not —
        which is exactly the `relative_to` "is not in the subpath of" failure a
        symlinked `HF_LEROBOT_HOME` produces.
        """
        return Path(self.lerobot_home())
