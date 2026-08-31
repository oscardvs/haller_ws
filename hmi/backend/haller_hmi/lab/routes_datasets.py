# hmi/backend/haller_hmi/lab/routes_datasets.py
"""`/lab/datasets/**`, plus the four legacy `/record` and `/cameras` paths.

Three properties shape every handler below, and none of them is a preference.

* **Every handler is a plain `def`, never `async def`.** FastAPI runs plain
  defs in a worker thread, so a multi-megabyte `read_parquet` can never stall
  the event loop that forwards teleop frames to the arms. The kit does this for
  the same reason (`data/api.py`); it matters more here because this process
  also owns the Feetech bus. Nothing below awaits anything, so the rule is
  applied uniformly rather than per-route — a handler that grew a disk read
  later would otherwise be the one that is still `async`.

* **Catalog's key names are NOT the wire's.** `lab/catalog.py` keeps the kit's
  spellings (`seconds`, `status`, `tasks`, `total_episodes`, `total_frames`,
  `review`, `review_stale`) because the kit's own captured output is what the
  fixtures compare against; the wire says `duration_s`, `mark`, `task`,
  `episodes`, `frames`, `marks`, `stale`. Translating is this module's whole
  job, and it happens in exactly two functions — `_episode_wire` and
  `_dataset_wire` — so no route can invent a third spelling.

  `episode_index` is the ONE catalog name that survives onto the wire
  unchanged, and `_episode_wire` documents why. The contract originally spelled
  it `index`; that was overridden 2026-08-27 once LeRobot's v3.0 parquet was
  found to use `index` for the GLOBAL FRAME INDEX, a different quantity, in the
  same feature set. `docs/port/trackb-lab-contract.md` now agrees.

* **Repo-ids are QUERY parameters, never path segments.** They contain a slash,
  and a `{repo_id:path}` route would shadow every sub-resource under it.

`require_local` is applied HERE, in the phase that introduces the routes,
rather than bolted on later. `--host 0.0.0.0` is how the Quest reaches the HMI
and how Oscar triages from inside the headset; reaching it must not also mean
deleting a dataset. Gated: `autoclass/apply`, `autoclass/revert`, `prune`,
`DELETE /lab/datasets`. Ungated deliberately: every GET, `mark`, `bulk` and
`autoclass/preview`, which writes nothing.

## Bodies are dicts, not models

POST bodies arrive as a plain `dict` (`JsonBody`) and are validated by hand,
because the frozen error contract is "400 bad input" and a pydantic model
answers a missing field with 422. The one exception is `CameraRecordBody` on
the legacy camera toggle, whose 422 is asserted by `tests/test_routes_data.py`.

## The legacy four

`POST /cameras/{id}/record`, `GET /record/episodes`, `GET /record/repos` and
`DELETE /record/episodes/last` keep their existing URLs and their existing
response shapes byte for byte: `tests/test_routes_data.py` (31 tests) and
Track C's shipped code both depend on them, and `build_lab_router` REPLACES
`build_data_router` rather than mounting beside it. Those 31 tests were run
unmodified against THIS router, with only `build_router` swapped for
`build_datasets_router`: 31 passed.

They are reimplemented here rather than imported from `routes_data.py` for one
hard reason: that module does `from .recorder import read_episode_rows`, and
`recorder.py` imports `lerobot` at module scope. Importing it from `lab/` would
drag CUDA and a Hub client into the serving process's import graph — the one
thing this package may never do. `read_episode_rows` itself is pure pyarrow, so
`_episode_meta_rows` below is that function's body, skip-on-unreadable included.

`lab/catalog.py` is not substituted into those four either, and not out of
tidiness: `catalog.list_datasets` reads `HF_LEROBOT_HOME` and globs two levels,
where `GET /record/repos` walks the INJECTED `lerobot_home()` to depth 3 and
sorts by repo_id; `catalog._load_episode_meta` fails whole-dataset where
`read_episode_rows` skips per FILE, which is exactly what keeps the listing
alive across the open-writer window every recording session has (takes 10+ live
in a footerless parquet). Same question, different answers, so the answers are
not swapped.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..api.deps import LabDeps
from ..api.errors import DataDependencyError, as_http
from ..api.gate import build_require_local
from . import autoclass, catalog, lease
from . import review as review_mod
from . import runs as runs_mod

logger = logging.getLogger(__name__)

#: Deepest nesting `GET /record/repos` looks for a dataset under the lerobot
#: home. Kept at `routes_data.REPO_SCAN_DEPTH`'s value, and kept BOUNDED, for
#: its reason: the cache also holds video files and model checkpoints.
REPO_SCAN_DEPTH = 3

#: How far back a busy-check reads the run store. The kit's `_refuse_if_busy`
#: number: only `running` runs can hold a dataset, and a store with fifty
#: finished runs above the running one is a store nobody has cleaned in months.
BUSY_RUN_SCAN = 50


class CameraRecordBody(BaseModel):
    record: bool


#: A required JSON object body. Spelled through `Annotated` rather than as a
#: `Body(...)` default because a call in a default argument is a lint error
#: (B008) that ruff exempts `Query` and `Depends` from and `Body` not at all.
#: The shape is deliberately `dict` and not a model — see the module docstring
#: on why a missing field here answers 400 rather than 422.
JsonBody = Annotated[dict, Body()]


# ---- parquet, without the recorder ----

def _pq():
    """`pyarrow.parquet`, or a 503 that names the venv.

    Guarded the way `catalog._pandas` is rather than imported at module scope:
    a missing pyarrow is a broken serving venv, and the honest answer to a
    request that would have worked on a correct one is 503 on that request, not
    an ImportError that takes the whole server down at startup.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - a broken serving venv
        raise DataDependencyError(
            "reading datasets needs pandas + pyarrow in the serving venv "
            "(~/venvs/haller-hmi)"
        ) from e
    return pq


def _warm_pandas() -> None:
    """Import pandas HERE, on the thread that builds the router.

    Not a style choice and not premature: `catalog._pandas()` defers the first
    `import pandas` to its first call, every handler in this module is a plain
    `def`, and FastAPI runs plain defs on anyio's worker threads. So on a cold
    process the first `GET /lab/datasets` performs the process's first pandas
    (and therefore pyarrow) import on a WORKER thread, and the next read from a
    different worker segfaults. Measured on this box, no haller code involved:

        thread A:  import pandas; pd.read_parquet(meta/tasks.parquet)
        thread B:  pd.read_parquet(meta/episodes/chunk-000/file-00*.parquet)
        -> Fatal Python error: Segmentation fault, in pyarrow.parquet.core.read

    A segfault in this process is a segfault in the teleop path — the arms go
    down with the page. Building the router happens once, on the main thread,
    before any request, so doing the import there costs a startup fraction of a
    second and removes the window entirely.

    `ImportError` is swallowed on purpose: a serving venv without pandas must
    still mount, and answer 503 per request through `catalog._pandas` /
    `_pq()`, rather than failing to start.
    """
    try:
        import pandas  # noqa: F401 - imported for its side effect, see above
    except ImportError:  # pragma: no cover - a broken serving venv
        pass


def _episode_meta_files(root: Path) -> list[Path]:
    """Every episode-metadata parquet, in write order.

    There is more than one and that is not an edge case: on RESUME lerobot
    starts a fresh metadata file rather than appending, so a dataset collects
    one file per recording SESSION. Reading only `chunk-000/file-000.parquet`
    reports the first session and silently loses every later one.
    """
    directory = root / "meta" / "episodes"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("chunk-*/file-*.parquet"))


def _episode_meta_rows(root: Path) -> list[dict]:
    """Episode metadata off disk, one dict per episode, `stats/*` dropped.

    `recorder.read_episode_rows`, reproduced because importing it would import
    lerobot (see the module docstring). The skip is the load-bearing part: a
    `pq.ParquetWriter` only lays down its footer on close, so the session's own
    metadata file is unreadable until finalize, and those episodes are covered
    by the session overlay below. A listing that 503s from the tenth take of
    every session is a worse answer than one that is an episode short.
    """
    pq = _pq()
    rows: list[dict] = []
    for path in _episode_meta_files(root):
        try:
            names = pq.read_schema(path).names
            cols = [c for c in names if not c.startswith("stats/")]
            rows += pq.read_table(path, columns=cols).to_pylist()
        except Exception as e:  # noqa: BLE001 - see above; also the truncated
            logger.warning(     # file a crash leaves, which lerobot cannot read
                "lab: episode metadata at %s is not readable (%s); skipping it",
                path, e)
    return rows


def _dir_size_bytes(path: Path) -> int:
    """Bytes on disk under `path`. Broken symlinks and races are worth 0.

    `scandir` rather than `rglob` + `stat`: a dataset is thousands of video and
    parquet files and this runs on every listing, so the entry's already-fetched
    stat is worth having.
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


# ---- catalog's names -> the frozen wire names ----

def _first_task(tasks) -> str | None:
    """One string, not the list.

    A take is driven against one instruction here; the column is a list because
    lerobot allows several, so the rest are not invented away — they simply have
    no operator-facing slot yet.
    """
    for task in tasks or ():
        return str(task)
    return None


def _episode_wire(ep: dict) -> dict:
    """One episode row, in Track C's spelling.

    `seconds -> duration_s`, `status -> mark`, `tasks[0] -> task`. `label` is
    the 1-BASED number Oscar uses in conversation and `episode_index` the
    stored one; the UI shows both ("Ep 4 (idx 3)") because that off-by-one is
    how the wrong demonstration gets deleted.

    `episode_index` is NOT shortened to `index`, and that is a correction, not
    a stutter left in place: LeRobot's own v3.0 parquet carries BOTH as
    DIFFERENT columns, where `index` is the GLOBAL FRAME INDEX across the whole
    dataset. Verified on both real datasets — episode 1's first three frames
    read episode_index [1,1,1], frame_index [0,1,2], index [855,856,857].
    Spelling an episode index `index` here would collide with an existing
    column meaning something else, on the surface most likely to be read next
    to frame data. If a global frame index is ever exposed, it is `index`.
    """
    return {
        "episode_index": ep["episode_index"],
        "label": ep["label"],
        "frames": ep["frames"],
        "duration_s": ep["seconds"],
        "share": ep["share"],
        "task": _first_task(ep.get("tasks")),
        "verdict": ep["verdict"],
        "reasons": ep["reasons"],
        "arms": ep["arms"],
        "mark": ep.get("status", review_mod.UNSET),
        "note": ep.get("note", ""),
        "tags": ep.get("tags") or [],
        "videos": ep.get("videos") or {},
    }


def _dataset_wire(row: dict) -> dict:
    """One `GET /lab/datasets` card.

    `total_episodes -> episodes`, `total_frames -> frames`,
    `seconds -> duration_s`, `review -> marks`, `review_stale -> stale`. The
    card deliberately carries no episode list: this endpoint is POLLED, and
    `catalog.list_datasets` opens no parquet to answer it.

    `units` is `catalog._units_summary`, THREE SCALARS AND NOT THE FULL BLOCK,
    and the difference is the same polling rule: the card gets
    `declared`/`state_unit`/`convertible`, while the joint-name lists and the
    operator-facing sentence stay on `_detail_wire`, one click away. Carried at
    all because a card that cannot say "units unknown" is a card that shows a
    foreign dataset exactly as it shows one of ours, and the two are
    indistinguishable by inspection: both are small signed numbers with
    joint-shaped trajectories (`catalog.dataset_units`).
    """
    return {
        "repo_id": row["repo_id"],
        "task": _first_task(row.get("tasks")),
        "episodes": row["total_episodes"],
        "frames": row["total_frames"],
        "duration_s": row["seconds"],
        "size_bytes": row["size_bytes"],
        "marks": row["review"],
        "is_backup": row["is_backup"],
        "rig": row["rig"],
        "stale": row["review_stale"],
        "units": row["units"],
    }


def _detail_wire(detail: dict) -> dict:
    """`catalog.dataset_detail`, renamed onto the wire.

    Everything the catalog computed is carried through rather than trimmed to
    the eight keys the contract enumerates, because two of the extras are read
    by the page and a missing key breaks a UI where an extra one cannot:
    `joints` is what labels a sweep bar (the alternative is the page assuming
    the kit's five joint names, which is wrong on every bimanual dataset), and
    `stale_episodes` is what flags an individual row — the episode shape has no
    per-row staleness field.

    `episode_frames` is the one thing dropped: it is `{index: frames}`, JSON
    would stringify its keys, and every value is already on its episode.

    `units` is the FULL block here, not the listing's three scalars: this is
    the response that carries `features`, and `features` can say a column is
    `float32[12]` while having no slot at all for what those twelve numbers
    MEAN. So the two travel together, and the extra fields the detail carries
    (`uncalibrated`, `reason`, `note`) are the ones an operator needs after
    reading "units unknown" on a card: which joints, why, and the sentence
    saying the values must not be read as this robot's degrees.
    """
    return {
        "repo_id": detail["repo_id"],
        "root": detail["root"],
        "fps": detail["fps"],
        "robot_type": detail["robot_type"],
        "codebase_version": detail["codebase_version"],
        "video_keys": detail["video_keys"],
        "features": detail["features"],
        # Which of those columns the launcher ticks. Sent rather than derived
        # in the browser: the same rule validates the choice on the way back
        # in, and a second implementation of it would drift into a policy
        # trained on an observation space the form never showed.
        "policy_inputs_default": detail["policy_inputs_default"],
        "rig": detail["rig"],
        "units": detail["units"],
        "joints": detail["joints"],
        "tasks": detail["tasks"],
        "total_episodes": detail["total_episodes"],
        "total_frames": detail["total_frames"],
        "duration_s": detail["seconds"],
        "marks": detail["review"],
        "stale": detail["review_stale"],
        "stale_episodes": detail["stale_episodes"],
        "keep_list": detail["keep_list"],
        "episodes": [_episode_wire(e) for e in detail["episodes"]],
    }


# ---- request bodies ----

def _need(payload: dict, key: str):
    """A required body field, or a 400 that names it."""
    value = (payload or {}).get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise HTTPException(status_code=400, detail=f"{key} is required")
    return value


def _need_int(payload: dict, key: str) -> int:
    """A required integer field. `0` is a valid episode index, so the presence
    test is `is None` and never a truth test."""
    raw = (payload or {}).get(key)
    if raw is None:
        raise HTTPException(status_code=400, detail=f"{key} is required")
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400, detail=f"{key} must be an integer, got {raw!r}") from None


def _need_indices(payload: dict, key: str) -> list[int]:
    raw = (payload or {}).get(key)
    if not isinstance(raw, (list, tuple)):
        raise HTTPException(
            status_code=400, detail=f"{key} must be a list of episode indices")
    try:
        return [int(x) for x in raw]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400, detail=f"{key} must be a list of episode indices") from None


def _known(episode: int, episode_frames: dict[int, int]) -> int:
    """Refuse a mark on an index the dataset does not have.

    A mark is stored against an index and validated against the episode NOW at
    it, so marking an index that does not exist writes an entry that is stale
    the instant it lands — and reads, on the page, as a mark that was lost to a
    prune. 404 rather than a silent no-op: the caller believed it was marking a
    demonstration.
    """
    if episode not in episode_frames:
        raise KeyError(f"episode {episode} is not in this dataset")
    return episode


def build_datasets_router(deps: LabDeps) -> APIRouter:
    """Wire `/lab/datasets/**` and the four legacy paths onto one router.

    `deps` carries zero-arg callables resolved per request, for the reason
    `routes_data.build_router` documents at line 84 and `api/deps.py` repeats:
    routers mount at import time, `cameras` and `recorder` are assigned in
    `lifespan`, and a router closing over the VALUES would capture `None` for
    the life of the process.

    The gate is built ONCE here, from `deps.allow_remote_control`, and mounted
    as a route dependency on the four destructive endpoints.
    """
    _warm_pandas()
    router = APIRouter()
    require_local = build_require_local(deps.allow_remote_control)
    local_only = [Depends(require_local)]

    # No handler below carries a `-> dict` return annotation, and that is not
    # an oversight: FastAPI turns one into `response_model=dict`, which
    # revalidates and re-encodes the whole payload on the way out. On a
    # 400-episode `detail` — every episode's `arms`, `reasons` and video slice
    # — that is a second full pass over the exact response this page exists to
    # deliver over a LAN to a headset. `routes_data.py` carries none either.

    def _refuse_if_busy(repo_id: str, verb: str) -> None:
        """409 while a run or the recorder still owns the dataset.

        Two questions, both answered by `lab/lease.py` in SENTENCES rather than
        bools, because this 409 is read in a headset with both hands full of
        arm: it has to say what is holding the dataset, which one, and what to
        do about it.
        """
        reason = lease.dataset_busy(
            repo_id, runs_mod.list_runs(limit=BUSY_RUN_SCAN), verb=verb)
        if reason is None:
            reason = lease.recorder_busy(
                deps.recorder_or_none(), repo_id, verb=verb)
        if reason:
            raise HTTPException(status_code=409, detail=reason)

    # ---- the dataset list and one dataset --------------------------------

    @router.get("/lab/datasets")
    def get_datasets():
        """Every dataset under the lerobot home, newest first.

        Opens no parquet: `rig` and the mark counts come from `info.json` and
        the `review.json` sidecar, so this stays cheap enough to poll.
        """
        with as_http():
            return {"datasets": [_dataset_wire(d) for d in catalog.list_datasets()]}

    @router.get("/lab/datasets/detail")
    def get_detail(repo_id: str = Query(...)):
        """One dataset with every episode graded, marked and video-sliced."""
        with as_http():
            return _detail_wire(catalog.dataset_detail(repo_id))

    @router.get("/lab/datasets/episodes")
    def get_episodes(
        repo_id: str = Query(...),
        sort: str | None = Query(default=None),
        order: str = Query(default="asc"),
        filter_mark: str | None = Query(default=None),
        filter_verdict: str | None = Query(default=None),
        tag: str | None = Query(default=None),
        q: str | None = Query(default=None),
        offset: int = Query(default=0),
        limit: int | None = Query(default=None),
    ):
        """One page of episodes, filtered and sorted SERVER-side.

        `total` is the count after filtering and before `offset`/`limit`, which
        is what a pager needs. Sorting in the browser instead would mean
        shipping 400 episodes' `arms` and `reasons` over the LAN to a headset to
        reorder them, which is the request that makes the page feel broken on
        the only network it has to work on.
        """
        with as_http():
            page = catalog.query_episodes(
                repo_id,
                sort=sort,
                order=order,
                filter_mark=filter_mark,
                filter_verdict=filter_verdict,
                tag=tag,
                q=q,
                offset=offset,
                limit=limit,
            )
            return {
                "total": page["total"],
                "episodes": [_episode_wire(e) for e in page["episodes"]],
            }

    @router.get("/lab/datasets/trace")
    def get_trace(repo_id: str = Query(...), episode: int = Query(...)):
        """One episode's action/state/gripper series for the timeline chart.

        The gripper block carries the exact `closed_below`/`open_above` the
        verdict beside the chart was reached with, so the guides cannot
        disagree with the words.

        `seconds -> duration_s` is the one rename: the trace is the third place
        a duration reaches the wire, and one API that spells the same quantity
        two ways makes every reader ask which one this route uses. The episode
        index passes through as `episode_index`, which is already the wire
        spelling everywhere else.
        """
        with as_http():
            trace = dict(catalog.episode_trace(repo_id, episode))
            # Inside the ladder, not after it: a `seconds` that ever went
            # missing would raise a KeyError past `as_http` and answer with
            # FastAPI's own 500 body instead of the frozen `{"detail": ...}`.
            trace["duration_s"] = trace.pop("seconds")
            return trace

    @router.get("/lab/datasets/video")
    def get_video(
        repo_id: str = Query(...),
        key: str = Query(...),
        episode: int = Query(...),
    ):
        """The PACKED v3.0 mp4 holding an episode, resolved server-side.

        v3.0 puts MANY episodes in ONE file — measured on
        `local/so101_pick_cube`: 46 episodes in 7 mp4s, with episodes 2-6 all
        inside `file-001.mp4` at 0.0 / 15.53 / 33.70 / 48.60 / 60.57. So the
        episode has to be resolved to its chunk/file HERE, from the same
        `videos` block `detail` returns, and no client ever builds a chunk path.
        Serving a per-episode cut instead would mean re-encoding around the
        hole, which is the same minutes-of-AV1 cost that makes pruning a
        background job.

        The offsets ride back as headers so a player can seek without a second
        round trip, and `X-Video-Chunk-Index`/`X-Video-File-Index` are what tell
        a seek inside the current file from a re-buffer of a new one.
        Starlette's `FileResponse` answers Range itself, which is what makes
        that seek a partial fetch instead of a whole-file download.
        """
        with as_http():
            detail = catalog.dataset_detail(repo_id)
            slices = {e["episode_index"]: e.get("videos") or {}
                      for e in detail["episodes"]}
            if int(episode) not in slices:
                raise KeyError(f"episode {episode} is not in {repo_id}")
            block = slices[int(episode)].get(key)
            if block is None:
                raise KeyError(
                    f"episode {episode} of {repo_id} has no video slice for "
                    f"{key!r} — known keys: {', '.join(detail['video_keys']) or 'none'}")
            path = catalog.video_path(
                repo_id, key, block["chunk_index"], block["file_index"])
            return FileResponse(
                path,
                media_type="video/mp4",
                headers={
                    "X-Video-Chunk-Index": str(block["chunk_index"]),
                    "X-Video-File-Index": str(block["file_index"]),
                    "X-Episode-From-Timestamp": repr(float(block["from_timestamp"])),
                    "X-Episode-To-Timestamp": repr(float(block["to_timestamp"])),
                },
            )

    @router.get("/lab/datasets/split")
    def get_split(
        repo_id: str = Query(...),
        eval_split: float = Query(default=0.0),
        seed: int = Query(default=42),
        mode: str = Query(default="random"),
    ):
        """Which kept episodes the trainer will hold out for eval loss.

        Computed here rather than in the page so there is exactly one
        implementation of the rule the trainer actually follows. `order` is the
        list LeRobot receives and it is deliberately NOT sorted — see
        `lab/split.py`; `train_episodes` and `eval_episodes` are reports.
        """
        with as_http():
            detail = catalog.dataset_detail(repo_id)
            plan = catalog.plan_eval_split(
                detail["episodes"], detail["keep_list"], eval_split, seed, mode)
            return {
                "order": plan["order"],
                "train_episodes": plan["train"],
                "eval_episodes": plan["eval"],
                "mode": plan["mode"],
                "seed": plan["seed"],
                "eval_split": plan["eval_split"],
            }

    # ---- marking ---------------------------------------------------------
    # Ungated on purpose. Triage is the thing Oscar actually does from inside
    # the headset — mark a fumbled take reject the moment he sees it — and
    # gating it would push him back to the desk to do the one job the page
    # exists for.

    @router.post("/lab/datasets/mark")
    def post_mark(payload: JsonBody):
        """Mark one episode keep / reject / unset, with an optional note."""
        repo_id = _need(payload, "repo_id")
        episode = _need_int(payload, "episode")
        status = _need(payload, "status")
        with as_http():
            detail = catalog.dataset_detail(repo_id)
            _known(episode, detail["episode_frames"])
            review_mod.set_status(
                detail["root"],
                episode,
                str(status),
                payload.get("note"),
                fingerprint=detail["fingerprint"],
                # Stamp the episode's own length onto the mark. A mark that
                # does not record what it was made about cannot be told from
                # one that survived a prune and now names a different take.
                episode_frames=detail["episode_frames"].get(episode),
            )
            return {"ok": True}

    @router.post("/lab/datasets/bulk")
    def post_bulk(payload: JsonBody):
        """A status and/or tags across many episodes, in ONE write.

        `updated` is how many stored entries actually CHANGED, not how many
        were named: adding a tag every selected episode already carries reports
        0. "12 updated" after a no-op is how a selection that missed its rows
        goes unnoticed.
        """
        repo_id = _need(payload, "repo_id")
        episodes = _need_indices(payload, "episodes")
        if not episodes:
            raise HTTPException(
                status_code=400,
                detail="episodes is empty — nothing was selected to update")
        with as_http():
            detail = catalog.dataset_detail(repo_id)
            for episode in episodes:
                _known(episode, detail["episode_frames"])
            updated = review_mod.bulk_update(
                detail["root"],
                episodes,
                status=payload.get("status"),
                note=payload.get("note"),
                tags_add=payload.get("tags_add"),
                tags_remove=payload.get("tags_remove"),
                episode_frames=detail["episode_frames"],
            )
            return {"updated": updated}

    # ---- autoclassify ----------------------------------------------------

    @router.post("/lab/datasets/autoclass/preview")
    def post_autoclass_preview(payload: JsonBody):
        """Compute a diff and write nothing.

        Ungated by the frozen contract, and that is exactly why `rules` goes
        through `lab/rules.py`'s hand-written parser and never `eval`: this call
        is reachable from the LAN, on the machine that owns the servo bus.
        """
        repo_id = _need(payload, "repo_id")
        mode = _need(payload, "mode")
        with as_http():
            return autoclass.preview(repo_id, str(mode), payload.get("params"))

    @router.post("/lab/datasets/autoclass/apply", dependencies=local_only)
    def post_autoclass_apply(payload: JsonBody):
        """Apply a previewed diff, recording an undo batch first.

        409 on a stale token rather than a silent re-run: the operator
        confirmed a diff computed against a dataset STATE, and applying it to a
        different state applies decisions they never saw. `StaleTokenError` is
        caught here because `api/errors.as_http` has no rung for it — it is a
        `RuntimeError` on purpose, so it cannot fall through to 400 and tell the
        operator their well-formed request was malformed.
        """
        repo_id = _need(payload, "repo_id")
        token = _need(payload, "token")
        with as_http():
            try:
                return autoclass.apply(repo_id, str(token))
            except autoclass.StaleTokenError as e:
                raise HTTPException(status_code=409, detail=str(e)) from e

    @router.post("/lab/datasets/autoclass/revert", dependencies=local_only)
    def post_autoclass_revert(payload: JsonBody):
        """Restore every mark a batch overwrote, absence included."""
        repo_id = _need(payload, "repo_id")
        batch = _need(payload, "batch")
        with as_http():
            return autoclass.revert(repo_id, str(batch))

    # ---- prune -----------------------------------------------------------

    @router.post("/lab/datasets/prune", dependencies=local_only)
    def post_prune(payload: JsonBody):
        """Drop the rejected episodes — as a background export run, not inline.

        A v3.0 dataset packs every episode into one mp4, so dropping one means
        re-encoding around the hole: minutes of AV1, not milliseconds. This
        route therefore launches `runners/export` through `lab/runs.py` and
        returns its id; the page follows the run's log like any other job.

        `expect_episodes` is what the client believes it is deleting. If the
        rejected set moved since the page loaded, REFUSE — deleting a different
        set than the one that was confirmed is the failure this endpoint exists
        to make impossible. It is required when `backup` is false, which is the
        one path here with nothing to fall back on.
        """
        repo_id = _need(payload, "repo_id")
        backup = bool(payload.get("backup", True))
        expect = payload.get("expect_episodes")
        with as_http():
            detail = catalog.dataset_detail(repo_id)
            keep = set(detail["keep_list"])
            drop = [e["episode_index"] for e in detail["episodes"]
                    if e["episode_index"] not in keep]
            if not drop:
                raise ValueError(
                    "no episodes are rejected — nothing would be deleted")
            if not keep:
                # A dataset with no episodes is not a dataset. Pointing at the
                # whole-dataset delete is the honest answer: it is the route
                # that asks for the name typed back.
                raise ValueError(
                    "every episode is rejected — a dataset cannot be emptied "
                    "this way. Delete the whole dataset instead "
                    "(DELETE /lab/datasets?repo_id=...&confirm=...)."
                )
            if expect is None:
                if not backup:
                    raise ValueError(
                        "expect_episodes is required when backup is false — "
                        "that is the one prune with nothing to fall back on."
                    )
            else:
                if not isinstance(expect, (list, tuple)):
                    raise ValueError(
                        "expect_episodes must be a list of episode indices")
                try:
                    confirmed = sorted(int(e) for e in expect)
                except (TypeError, ValueError):
                    raise ValueError(
                        "expect_episodes must be a list of episode indices"
                    ) from None
                if confirmed != sorted(drop):
                    raise ValueError(
                        f"the rejected set changed since you confirmed (now "
                        f"{sorted(drop)}). Reload the page and check the marks "
                        f"before pruning."
                    )
            _refuse_if_busy(repo_id, "prune")
            record = runs_mod.launch(
                "export",
                {
                    "mode": "in_place",
                    "repo_id": repo_id,
                    "delete_episodes": drop,
                    "keep_backup": backup,
                },
                name="prune-" + str(repo_id).replace("/", "-"),
            )
            return {"run_id": record["id"], "delete_episodes": drop}

    # ---- whole-dataset delete --------------------------------------------

    @router.delete("/lab/datasets", dependencies=local_only)
    def delete_lab_dataset(
        repo_id: str = Query(...),
        confirm: str = Query(default=""),
    ):
        """Remove a dataset directory outright. There is no undo.

        `confirm` must equal `repo_id` byte for byte — not stripped, not
        case-folded — so a stray click cannot satisfy it. The route exists
        because the alternative is `rm -rf` against a path from memory, on a box
        with NO BACKUP OF ANY KIND (verified 2026-08-26: one NVMe, no external
        media, no sync, and the 500G NTFS partition is on the same disk).

        It does NOT touch the `<name>_old` sibling a prune leaves. That is a
        separate dataset with its own row and its own delete, and removing it
        as a side effect would throw away the only copy of the episodes the
        prune dropped.
        """
        if confirm != repo_id:
            raise HTTPException(
                status_code=400,
                detail="confirm must repeat the dataset name exactly")
        with as_http():
            _refuse_if_busy(repo_id, "delete")
            return catalog.delete_dataset(repo_id)

    # ---- legacy: the recorder's own view ---------------------------------
    # Three helpers the four compat paths share, lifted from
    # `routes_data.build_router` unchanged in behaviour.

    def _episode_is_open() -> bool:
        """Is a take in progress? Either flag is enough to refuse a change.

        `recording` is the loop's own view and `_episode_open` the writer's;
        they differ for exactly as long as the save/discard tail runs, which is
        precisely a window where the camera set must not move.
        """
        recorder = deps.recorder_or_none()
        if recorder is None:
            return False
        return bool(recorder.status().get("recording")
                    or getattr(recorder, "_episode_open", False))

    def _resolve_repo(repo_id: str | None) -> str:
        """Fall back to the repo the recorder is on, or was last on."""
        if repo_id:
            return repo_id
        recorder = deps.recorder_or_none()
        current = (recorder.status().get("repo_id") if recorder is not None else None)
        if not current:
            raise HTTPException(
                status_code=400,
                detail="no repo_id given and the recorder has not opened one yet",
            )
        return current

    def _root_for(repo_id: str) -> Path:
        """The recorder's own root for a repo — NOT `catalog.dataset_root`.

        The recorder is constructed with its own `root`, which is the lerobot
        home only by default; resolving through the catalog would answer about
        a different directory the moment a test or a second cache is in play.
        """
        root = Path(deps.recorder().dataset_root(repo_id))
        if not (root / "meta" / "info.json").exists():
            raise HTTPException(
                status_code=404,
                detail=f"no dataset for repo_id {repo_id!r} at {root}")
        return root

    # ---- legacy: which cameras record ------------------------------------

    @router.post("/cameras/{camera_id}/record")
    def post_camera_record(camera_id: str, body: CameraRecordBody):
        """Move a camera in or out of the recorded set.

        409 while an episode is open, and not out of caution: `start_episode`
        freezes the camera set because it freezes the dataset SCHEMA with it —
        every frame of a take carries exactly the image columns the take opened
        with. A toggle accepted mid-take could not take effect until the next
        one, so accepting it would report a change that did not happen.
        """
        if _episode_is_open():
            raise HTTPException(
                status_code=409,
                detail="an episode is being recorded; the camera set is frozen "
                       "until it stops",
            )
        try:
            record = deps.cameras().set_record(camera_id, body.record)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"id": camera_id, "record": record}

    # ---- legacy: what has been collected ---------------------------------

    @router.get("/record/episodes")
    def get_record_episodes(repo_id: str | None = Query(default=None)):
        """Every episode in one dataset: index, frames, task, duration.

        `length_s` is frames/fps — the recorded rate, which is what the take
        actually lasted. It is not read from the video's own duration, which
        would disagree the moment a tick was skipped.
        """
        recorder = deps.recorder()
        repo = _resolve_repo(repo_id)
        root = _root_for(repo)
        info = _read_info(root)
        fps = float(info.get("fps") or 0) or None

        def _entry(index: int, frames: int, task) -> dict:
            return {
                "index": int(index),
                "frames": int(frames),
                "task": task,
                "length_s": (round(frames / fps, 3) if fps else None),
            }

        by_index: dict[int, dict] = {}
        for row in _episode_meta_rows(root):
            index = int(row["episode_index"])
            by_index[index] = _entry(
                index, int(row.get("length") or 0), _first_task(row.get("tasks")))

        # Overlay what this process has saved but lerobot has not yet made
        # readable. Disk wins on conflict — it is the durable record, and the
        # session log is only ever filling a gap it left. Without this the
        # browser shows NOTHING for the first nine takes of a session and then
        # nothing again until finalize.
        for saved in recorder.session_episodes(repo):
            by_index.setdefault(
                saved["index"], _entry(saved["index"], saved["frames"], saved["task"]))

        return {
            "repo_id": repo,
            "root": str(root),
            "episodes": [by_index[i] for i in sorted(by_index)],
            "total_frames": int(info.get("total_frames") or 0),
            "size_bytes": _dir_size_bytes(root),
        }

    @router.get("/record/repos")
    def get_record_repos():
        """Every dataset under the lerobot home, cheapest-possible scan.

        A directory is a dataset iff it has `meta/info.json`; the repo_id is its
        path relative to the home, which is exactly how it was written.
        """
        home = deps.home()
        repos: list[dict] = []
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
    def delete_last_episode(repo_id: str | None = Query(default=None)):
        """Undo the last take: pop the highest-numbered episode off the dataset.

        Delegated to the recorder, which owns both the in-place pop and the
        session log the pop has to drop with it. See
        `DatasetRecorder.delete_last_episode` for why this is last-only. The
        refusals it raises are all operator errors or datasets in a shape it
        will not guess at, so they are 409s rather than 500s.
        """
        recorder = deps.recorder()
        repo = _resolve_repo(repo_id)
        _root_for(repo)
        try:
            return recorder.delete_last_episode(repo)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    return router
