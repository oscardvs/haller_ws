# hmi/backend/haller_hmi/routes_data.py
"""Camera-recording control and dataset management, as a mountable router.

Two jobs the cockpit's collection workspace needs and `server.py` had no home
for:

* **Which cameras record.** A view the operator drives from is not
  automatically a view the policy should see, and that call now belongs to
  whoever is running the session rather than to a config edit and a restart.
  `POST /cameras/{id}/record` moves a camera in and out of the recorded set;
  `GET /cameras` already reports it (see `CameraManager.list`).

* **What has been collected so far.** Episode counts, per-take frame counts and
  durations, disk usage, and popping a fumbled take back off — read straight
  from the dataset's own metadata on disk.

Everything here reads the dataset with plain parquet/json (see
`recorder.read_episode_rows`) rather than constructing a `LeRobotDataset`.
That is deliberate: opening one pulls from the Hub, builds writers and costs
seconds, all so a panel can print "12 episodes, 4.2 GB". A listing must be
cheap enough to poll.

Injected through `build_router` rather than importing the server's globals, for
the reason `vr_teleop.relay` does the same: `server.py` imports this module, so
this module must never import back, and the tests then get to mount the router
on their own app with fakes.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .recorder import read_episode_rows

logger = logging.getLogger(__name__)

#: Deepest nesting we look for a dataset under the lerobot home. A repo_id is
#: `<owner>/<name>`, so datasets sit two levels down; the extra level is slack
#: for anyone who nests further by hand. Bounded because the alternative is
#: walking a cache that also holds video files and model checkpoints.
REPO_SCAN_DEPTH = 3


class CameraRecordBody(BaseModel):
    record: bool


def _dir_size_bytes(path: Path) -> int:
    """Bytes on disk under `path`. Broken symlinks and races are worth 0.

    `scandir` rather than `Path.rglob` + `stat`: a dataset is thousands of
    video and parquet files and this runs on every listing, so the entry's
    already-fetched stat is worth having.
    """
    total = 0
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat().st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _read_info(root: Path) -> dict:
    try:
        return json.loads((root / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return {}


def build_router(*, get_cameras, get_recorder, lerobot_home) -> APIRouter:
    """Wire the data routes into an app.

        get_cameras()   -> CameraManager    runtime recorded-set owner
        get_recorder()  -> DatasetRecorder | None
        lerobot_home()  -> Path             where `GET /record/repos` scans

    All three are ZERO-ARG CALLABLES, resolved per request, and that is
    load-bearing rather than stylistic: `server.py` mounts its routers at
    import time, but builds the `CameraManager` and `DatasetRecorder` inside
    `lifespan`. A router that closed over the values would capture `None` for
    the entire life of the process and 503 forever. (Contrast `relay.py`,
    which can inject bound methods because its `HumanTeleopSession` is built at
    import time.)
    """
    router = APIRouter()

    def _require_recorder():
        recorder = get_recorder()
        if recorder is None:
            raise HTTPException(status_code=503, detail="recorder not ready")
        return recorder

    def _episode_is_open() -> bool:
        """Is a take in progress? Either flag is enough to refuse a change.

        `recording` is the loop's own view and `_episode_open` the writer's;
        they differ for exactly as long as the save/discard tail runs, which is
        precisely a window where the camera set must not move.
        """
        recorder = get_recorder()
        if recorder is None:
            return False
        return bool(recorder.status().get("recording")
                    or getattr(recorder, "_episode_open", False))

    def _resolve_repo(repo_id: str | None) -> str:
        """Fall back to the repo the recorder is on, or was last on."""
        if repo_id:
            return repo_id
        recorder = get_recorder()
        current = (recorder.status().get("repo_id") if recorder is not None else None)
        if not current:
            raise HTTPException(
                status_code=400,
                detail="no repo_id given and the recorder has not opened one yet",
            )
        return current

    def _root_for(repo_id: str) -> Path:
        root = Path(_require_recorder().dataset_root(repo_id))
        if not (root / "meta" / "info.json").exists():
            raise HTTPException(
                status_code=404, detail=f"no dataset for repo_id {repo_id!r} at {root}")
        return root

    # ---- which cameras record --------------------------------------------

    @router.post("/cameras/{camera_id}/record")
    async def post_camera_record(camera_id: str, body: CameraRecordBody):
        """Move a camera in or out of the recorded set.

        409 while an episode is open, and not out of caution: `start_episode`
        freezes the camera set because it freezes the dataset SCHEMA with it —
        every frame of a take carries exactly the image columns the take
        opened with. A toggle accepted mid-take could not take effect until
        the next one, so accepting it would report a change that did not
        happen.
        """
        if _episode_is_open():
            raise HTTPException(
                status_code=409,
                detail="an episode is being recorded; the camera set is frozen "
                       "until it stops",
            )
        try:
            record = get_cameras().set_record(camera_id, body.record)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"id": camera_id, "record": record}

    # ---- what has been collected -----------------------------------------

    @router.get("/record/episodes")
    async def get_record_episodes(repo_id: str | None = Query(default=None)):
        """Every episode in one dataset: index, frames, task, duration.

        `length_s` is frames/fps — the recorded rate, which is what the take
        actually lasted. It is not read from the video's own duration, which
        would disagree the moment a tick was skipped.
        """
        _require_recorder()
        repo = _resolve_repo(repo_id)
        root = _root_for(repo)
        info = _read_info(root)
        fps = float(info.get("fps") or 0) or None
        episodes = []
        for row in read_episode_rows(root):
            tasks = row.get("tasks") or []
            frames = int(row.get("length") or 0)
            episodes.append({
                "index": int(row["episode_index"]),
                "frames": frames,
                # One string, not the list: a take is driven against one
                # instruction here. The column is a list because lerobot
                # allows several, so the rest are not invented away — they
                # simply have no operator-facing slot yet.
                "task": (next(iter(tasks)) if len(tasks) else None),
                "length_s": (round(frames / fps, 3) if fps else None),
            })
        episodes.sort(key=lambda e: e["index"])
        return {
            "repo_id": repo,
            "root": str(root),
            "episodes": episodes,
            "total_frames": int(info.get("total_frames") or 0),
            "size_bytes": _dir_size_bytes(root),
        }

    @router.get("/record/repos")
    async def get_record_repos():
        """Every dataset under the lerobot home, cheapest-possible scan.

        A directory is a dataset iff it has `meta/info.json`; the repo_id is
        its path relative to the home, which is exactly how it was written.
        """
        home = Path(lerobot_home())
        repos = []
        if home.is_dir():
            seen: set[Path] = set()

            def walk(directory: Path, depth: int) -> None:
                if depth > REPO_SCAN_DEPTH:
                    return
                try:
                    children = sorted(p for p in directory.iterdir() if p.is_dir())
                except OSError:
                    return
                for child in children:
                    if (child / "meta" / "info.json").exists():
                        # A dataset root: never descend into it. `videos/` and
                        # `data/` below it are chunk directories, not repos.
                        if child not in seen:
                            seen.add(child)
                            info = _read_info(child)
                            repos.append({
                                "repo_id": child.relative_to(home).as_posix(),
                                "episodes": int(info.get("total_episodes") or 0),
                                "frames": int(info.get("total_frames") or 0),
                                "size_bytes": _dir_size_bytes(child),
                            })
                        continue
                    walk(child, depth + 1)

            walk(home, 1)
        repos.sort(key=lambda r: r["repo_id"])
        return {"root": str(home), "repos": repos}

    @router.delete("/record/episodes/last")
    async def delete_last_episode(repo_id: str | None = Query(default=None)):
        """Undo the last take: pop the highest-numbered episode off the dataset.

        See `DatasetRecorder.delete_last_episode` for why this is an in-place
        pop and why it is last-only. The refusals it raises are all operator
        errors or datasets in a shape it will not guess at, so they are 409s
        rather than 500s.
        """
        rec = _require_recorder()
        repo = _resolve_repo(repo_id)
        _root_for(repo)
        try:
            return rec.delete_last_episode(repo)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))

    return router
