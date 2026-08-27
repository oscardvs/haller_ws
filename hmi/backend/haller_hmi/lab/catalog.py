# hmi/backend/haller_hmi/lab/catalog.py
"""Reading recorded LeRobotDatasets with parquet and JSON alone.

This module is imported by the SERVING process — the one forwarding teleop
frames to the arms — so the package ban applies at its hardest here: no
`lerobot`, no `torch`. Both drag CUDA, a Hub client and seconds of import time
into the latency path to answer questions that are, all of them, plain parquet
and JSON reads. Every heavy read below is a plain `def`, never `async`: a
multi-megabyte `read_parquet` on the event loop would stall the websocket the
arms are being driven over.

Dataset layout (LeRobot v3.0), which is what shapes the whole API:

    <root>/meta/info.json                     features, fps, totals, calibration
    <root>/meta/tasks.parquet                 task strings
    <root>/meta/episodes/chunk-*/file-*.parquet
                                              per-episode index ranges, video
                                              timestamps, stats
    <root>/data/chunk-*/file-*.parquet        ALL episodes' frames
    <root>/videos/<key>/chunk-*/file-*.mp4    ALL episodes' video
    <root>/review.json                        the marks sidecar (lab/review.py)

v3.0 packs MANY episodes into ONE parquet and ONE mp4 — an episode is a SLICE,
identified by `dataset_from_index`/`to_index` for the frames and
`from_timestamp`/`to_timestamp` for the video. That is why the player can show a
single episode without transcoding: it seeks.

What is NOT the kit's, and why:

* Grading goes through `RigSpec` + `grade_episode`, so a bimanual dataset is
  graded per arm instead of through the kit's `GRIPPER_IDX = 5`. The spec is
  built ONCE per dataset — it comes from `info.json`, which cannot differ
  between two episodes of the same parquet.
* An episode carries `verdict`, `reasons` and `arms` where the kit had a flat
  `why` plus one arm's measurements.
* `query_episodes` filters and sorts HERE rather than in the browser.

Field names stay the kit's (`episode_index`, `seconds`, `status`, `tasks`); the
frozen HTTP contract spells four of them differently (`index`, `duration_s`,
`mark`, `task`) and the routes layer renames on the way out. The sort keys
`query_episodes` accepts are the HTTP spellings, because they arrive from a
query string and never pass through Python.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np

from ..api.errors import DataDependencyError, DatasetBusyError
from . import review as review_mod
from .grade import VERDICTS, grade_episode
from .schema import RigSpec
from .split import plan_eval_split

# Re-exported, not redeclared. `DataDependencyError` / `DatasetBusyError` are
# DEFINED in `api/errors.py` so that module can name them in its `except`
# clauses without importing `lab/` back (see its docstring); `plan_eval_split`
# lives in `split.py` so a tidy-up cannot reach the shuffle that makes it work.
# Both are re-exported so `catalog.DatasetBusyError` and
# `catalog.plan_eval_split` still resolve for anyone porting from the kit.
__all__ = [
    "TRACE_MAX_POINTS",
    "DataDependencyError",
    "DatasetBusyError",
    "dataset_detail",
    "dataset_root",
    "delete_dataset",
    "episode_trace",
    "hf_home",
    "list_datasets",
    "plan_eval_split",
    "query_episodes",
    "rename_dataset",
    "validate_repo_id",
    "video_path",
]

#: Directories under HF_LEROBOT_HOME that are not datasets.
_NOT_DATASETS = {"calibration", "hub"}

#: Cap on the number of points sent to the browser for one episode's trace. A
#: 60 s episode at 30 fps is 1800 samples; drawing more pixels than the chart
#: has is wasted bandwidth on every episode click.
TRACE_MAX_POINTS = 600


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - a broken serving venv
        raise DataDependencyError(
            "reading datasets needs pandas + pyarrow in the serving venv "
            "(~/venvs/haller-hmi)"
        ) from e
    return pd


def hf_home() -> Path:
    """The dataset cache root, always fully resolved.

    Resolving here rather than at each call site is load-bearing, and it broke
    the page once: `~/.cache/huggingface/lerobot` is a SYMLINK to
    `~/robot-data/lerobot` on this box. A resolved dataset root then has no
    common prefix with an unresolved base, and `Path.relative_to` fails with
    "is not in the subpath of" on two spellings of one directory.
    """
    base = os.environ.get("HF_LEROBOT_HOME")
    if base:
        return Path(base).expanduser().resolve()
    return (Path.home() / ".cache/huggingface/lerobot").resolve()


def dataset_root(repo_id: str) -> Path:
    """Resolve a repo-id to its directory, refusing to escape the cache.

    `repo_id` arrives from a URL, so `../` traversal is a real concern: without
    this check the video endpoint would serve any mp4 on the machine.
    """
    base = hf_home()
    root = (base / repo_id).resolve()
    if not (root == base or base in root.parents):
        raise ValueError(f"repo_id escapes the dataset cache: {repo_id!r}")
    return root


# ---- discovery ----

def _info(root: Path) -> dict | None:
    """The dataset's own metadata, or None if it cannot be read.

    Anything at all — missing, truncated mid-write, not JSON — means "not a
    dataset I can show", and the listing skips it. A page that 500s because one
    directory under the cache is malformed hides the twenty that are fine.
    """
    try:
        return json.loads((root / "meta" / "info.json").read_text())
    except Exception:  # noqa: BLE001 - see above; any failure means "skip it"
        return None


def _dir_size(root: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def _fingerprint(info: dict) -> dict:
    return {
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", 0)),
    }


def _video_keys(info: dict) -> list[str]:
    return [
        k for k, f in (info.get("features") or {}).items()
        if f.get("dtype") == "video"
    ]


def list_datasets() -> list[dict]:
    """Every dataset under HF_LEROBOT_HOME, newest first.

    Both `name/meta/info.json` and `namespace/name/meta/info.json` are matched:
    `local/<name>` is what the recorder writes, but a bare directory is still a
    valid dataset.

    Nothing here opens a parquet. `rig` comes from `info.json` alone, so the
    card can say "bimanual" without the listing paying a grading pass — the
    listing is polled.
    """
    base = hf_home()
    if not base.exists():
        return []
    seen: set[Path] = set()
    out: list[dict] = []
    for pattern in ("*/meta/info.json", "*/*/meta/info.json"):
        for path in base.glob(pattern):
            root = path.parent.parent
            if root in seen or root.name in _NOT_DATASETS:
                continue
            if root.parent != base and root.parent.name in _NOT_DATASETS:
                continue
            seen.add(root)
            info = _info(root)
            if info is None:
                continue
            repo_id = str(root.relative_to(base))
            total_eps = int(info.get("total_episodes", 0))
            fps = int(info.get("fps", 30)) or 30
            rev = review_mod.load(root)
            out.append({
                "repo_id": repo_id,
                "root": str(root),
                "total_episodes": total_eps,
                "total_frames": int(info.get("total_frames", 0)),
                "fps": fps,
                "seconds": int(info.get("total_frames", 0)) / fps,
                "robot_type": info.get("robot_type"),
                "codebase_version": info.get("codebase_version"),
                "video_keys": _video_keys(info),
                "rig": RigSpec.from_info(info).rig,
                "tasks": _tasks(root),
                "size_bytes": _dir_size(root),
                "modified": path.stat().st_mtime,
                "review": review_mod.counts(rev, total_eps),
                # The card only has totals to work with; the precise per-mark
                # check runs when the dataset is opened.
                "review_stale": review_mod.is_stale(rev, _fingerprint(info)),
                # A dataset whose backup twin exists was pruned in place by
                # `lerobot-edit-dataset`; the UI de-emphasises the leftover.
                "is_backup": repo_id.endswith("_old"),
            })
    out.sort(key=lambda d: d["modified"], reverse=True)
    return out


def _tasks(root: Path) -> list[str]:
    pd = _pandas()
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        return []
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - pyarrow's own hierarchy; no task strings
        return []
    # The task string is the INDEX in v3.0's tasks.parquet.
    if df.index.name == "task":
        return [str(t) for t in df.index.tolist()]
    if "task" in df.columns:
        return [str(t) for t in df["task"].tolist()]
    return []


# ---- per-dataset detail ----

def _parquets(root: Path, *parts: str) -> list[str]:
    return sorted(glob.glob(str(root.joinpath(*parts) / "**" / "*.parquet"), recursive=True))


def _stamp(root: Path) -> tuple:
    """Cheap change-detector for the caches: every data/meta parquet's size and
    mtime. Recording appends files, so this moves whenever the dataset does."""
    files = _parquets(root, "data") + _parquets(root, "meta", "episodes")
    out = []
    for f in files:
        try:
            st = os.stat(f)
            out.append((f, st.st_size, st.st_mtime))
        except OSError:
            pass
    return tuple(out)


_detail_cache: dict[str, tuple[tuple, dict]] = {}

# Loaded frame tables, keyed by dataset root. Clicking through episodes asks for
# one trace at a time, and re-reading the whole parquet on every click would
# make a 46-episode review crawl. Bounded to a couple of datasets so a long
# browsing session cannot grow without limit.
_frames_cache: dict[str, tuple[tuple, object]] = {}
_FRAMES_CACHE_MAX = 2

# The only three columns grading and tracing need. The video is the bulk of a
# dataset and is never read here.
_GRADE_COLUMNS = ["action", "observation.state", "episode_index"]


def _load_frames(root: Path, columns: list[str] | None = None):
    pd = _pandas()
    files = _parquets(root, "data")
    if not files:
        raise FileNotFoundError(f"no data parquet under {root}/data")
    try:
        return pd.concat(
            [pd.read_parquet(f, columns=columns) for f in files], ignore_index=True
        )
    except Exception as e:
        raise DatasetBusyError(
            "this dataset cannot be read yet — a recording session is most likely "
            "still running. Stop it from the cockpit; the parquet is only complete "
            "once the session finishes."
        ) from e


def _frames(root: Path):
    """Cached action/state/episode_index table for one dataset."""
    key = str(root)
    stamp = _stamp(root)
    hit = _frames_cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    df = _load_frames(root, columns=_GRADE_COLUMNS)
    if len(_frames_cache) >= _FRAMES_CACHE_MAX:
        _frames_cache.pop(next(iter(_frames_cache)))
    _frames_cache[key] = (stamp, df)
    return df


def _load_episode_meta(root: Path):
    pd = _pandas()
    files = _parquets(root, "meta", "episodes")
    if not files:
        return None
    try:
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    except Exception:  # noqa: BLE001 - unreadable meta costs tasks and video
        return None      # slices, not the grade; the data parquet is the one
                         # whose failure is a DatasetBusyError.


def dataset_detail(repo_id: str, use_cache: bool = True) -> dict:
    """Info plus one row per episode: length, task, video slice, grade, mark.

    Grading reads the whole data parquet (action + state only), which is a few
    MB even for a long session, and is cached against `_stamp`. Review marks are
    NOT cached: they are one small JSON file, they change on every click, and a
    mark that appeared lost behind a cached grade would be re-applied by an
    operator who has no reason to believe the first click worked.
    """
    root = dataset_root(repo_id)
    info = _info(root)
    if info is None:
        raise FileNotFoundError(f"no dataset at {root}")

    stamp = _stamp(root)
    cached = _detail_cache.get(repo_id)
    if use_cache and cached and cached[0] == stamp:
        detail = dict(cached[1])
    else:
        detail = _build_detail(root, info)
        _detail_cache[repo_id] = (stamp, detail)
        detail = dict(detail)

    rev = review_mod.load(root)
    episodes = []
    for ep in detail["episodes"]:
        ep = dict(ep)
        ep["status"] = review_mod.status_of(rev, ep["episode_index"])
        ep["note"] = review_mod.note_of(rev, ep["episode_index"])
        ep["tags"] = review_mod.tags_of(rev, ep["episode_index"])
        episodes.append(ep)
    detail["episodes"] = episodes
    detail["review"] = review_mod.counts(rev, detail["total_episodes"])
    # Validate each mark against the episode NOW at its index rather than
    # against the dataset's totals: `--resume` appends without renumbering, so
    # totals move on every session while nothing a mark names has changed.
    frames_by_index = {e["episode_index"]: e["frames"] for e in episodes}
    detail["episode_frames"] = frames_by_index
    detail["stale_episodes"] = review_mod.stale_marks(rev, frames_by_index)
    detail["review_stale"] = bool(detail["stale_episodes"])
    detail["keep_list"] = review_mod.keep_list(rev, detail["total_episodes"])
    detail["fingerprint"] = _fingerprint(info)
    return detail


def _build_detail(root: Path, info: dict) -> dict:
    fps = int(info.get("fps", 30)) or 30
    total_frames = int(info.get("total_frames", 0))
    video_keys = _video_keys(info)
    # ONE spec for the whole dataset: it is derived from metadata that cannot
    # differ between two episodes of the same parquet, and per-episode
    # derivation would be 46 identical passes over the same names.
    rig = RigSpec.from_info(info)
    ep_meta = _load_episode_meta(root)

    df = _frames(root)
    total = len(df)

    meta_by_ep: dict[int, dict] = {}
    if ep_meta is not None:
        for _, row in ep_meta.iterrows():
            meta_by_ep[int(row["episode_index"])] = row

    episodes = []
    for ep, sub in df.groupby("episode_index"):
        ep = int(ep)
        state = np.stack(sub["observation.state"].to_numpy())
        action = np.stack(sub["action"].to_numpy())
        r = grade_episode(state, action, rig, fps, total or total_frames)
        row = meta_by_ep.get(ep)
        tasks = []
        videos = {}
        if row is not None:
            raw_tasks = row.get("tasks")
            if raw_tasks is not None:
                tasks = [str(t) for t in list(raw_tasks)]
            for key in video_keys:
                col = f"videos/{key}"
                if f"{col}/from_timestamp" in row:
                    videos[key] = {
                        "chunk_index": int(row[f"{col}/chunk_index"]),
                        "file_index": int(row[f"{col}/file_index"]),
                        "from_timestamp": float(row[f"{col}/from_timestamp"]),
                        "to_timestamp": float(row[f"{col}/to_timestamp"]),
                    }
        episodes.append({
            "episode_index": ep,
            # Oscar counts episodes 1-based in conversation; the UI shows BOTH
            # ("Ep 4 (idx 3)") because that off-by-one is how the wrong
            # demonstration gets deleted.
            "label": ep + 1,
            "frames": r["frames"],
            "seconds": r["seconds"],
            "share": r["share"],
            "verdict": r["verdict"],
            "reasons": r["reasons"],
            "arms": r["arms"],
            "tasks": tasks,
            "videos": videos,
        })
    episodes.sort(key=lambda e: e["episode_index"])

    return {
        "repo_id": str(root.relative_to(hf_home())),
        "root": str(root),
        "fps": fps,
        "total_episodes": int(info.get("total_episodes", len(episodes))),
        "total_frames": total_frames,
        "seconds": total_frames / fps,
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "video_keys": video_keys,
        "rig": rig.rig,
        # Keyed by the same `side` an episode's `arms` entries carry, so the
        # chart labelling a sweep bar looks its joint name up rather than
        # assuming the kit's five. "" is the unprefixed solo arm.
        "joints": {arm.side: list(arm.joint_names) for arm in rig.arms},
        "tasks": _tasks(root),
        "features": {
            k: {"dtype": f.get("dtype"), "shape": f.get("shape"), "names": f.get("names")}
            for k, f in (info.get("features") or {}).items()
        },
        "episodes": episodes,
    }


# ---- episode query ----
# Filtering and sorting run HERE, not in the browser. A 46-episode dataset is
# fine either way, but this page is also the thing you point at a 400-episode
# dataset over a LAN from inside a headset, and shipping 400 episodes' arms and
# reasons to sort them client-side is the request that makes the page feel
# broken on the only network it has to work on.

_VERDICT_ORDER = {v: i for i, v in enumerate(VERDICTS)}

# Ascending puts the UNDECIDED episodes first: sorting by mark is how an
# operator finds the work still to do, not how they admire finished decisions.
_MARK_ORDER = {review_mod.UNSET: 0, review_mod.KEEP: 1, review_mod.REJECT: 2}

#: Sort keys are the HTTP contract's spellings — they arrive from a query
#: string and never pass through Python — so two of them map onto differently
#: named episode fields (`duration_s` -> `seconds`, `mark` -> `status`).
_SORT_KEYS = {
    "index": lambda e: e["episode_index"],
    "frames": lambda e: e["frames"],
    "duration_s": lambda e: e["seconds"],
    "share": lambda e: e["share"],
    "verdict": lambda e: _VERDICT_ORDER.get(e["verdict"], len(_VERDICT_ORDER)),
    "mark": lambda e: _MARK_ORDER.get(e["status"], len(_MARK_ORDER)),
}


def _norm_filter(value: str | None, allowed, label: str) -> str | None:
    """One closed-vocabulary filter, case-folded. Empty means NO filter.

    An absent query parameter arrives as None and a cleared one as `""`; both
    have to mean "everything", or clearing a filter in the UI returns an empty
    list. A value that is neither is a ValueError the routes layer turns into a
    400: silently returning zero episodes for a typo reads as a lost dataset.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in allowed:
        if candidate.lower() == text.lower():
            return candidate
    raise ValueError(f"{label} must be one of {', '.join(allowed)}, got {value!r}")


def _haystack(ep: dict) -> str:
    """What `q` searches: the text an operator can actually see on the row."""
    parts = [str(ep.get("verdict") or ""), str(ep.get("note") or "")]
    parts += [str(t) for t in ep.get("tasks") or ()]
    parts += [str(r) for r in ep.get("reasons") or ()]
    parts += [str(t) for t in ep.get("tags") or ()]
    return " ".join(parts).lower()


def query_episodes(
    repo_id: str,
    *,
    sort: str | None = None,
    order: str = "asc",
    filter_mark: str | None = None,
    filter_verdict: str | None = None,
    tag: str | None = None,
    q: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """One page of `dataset_detail`'s episodes, filtered and sorted server-side.

    `total` is the count AFTER filtering and BEFORE `offset`/`limit` — that is
    the number a pager needs, and the count of everything is already in
    `dataset_detail`.

    `sort=None` is the stored order, so `order` still applies to it. An unknown
    sort key is a ValueError; an unknown `order` is not, and the asymmetry is
    deliberate — falling back to ascending still answers the question that was
    asked, while falling back to a different column answers a different one.
    """
    episodes = dataset_detail(repo_id)["episodes"]

    key = _SORT_KEYS.get(sort or "index")
    if key is None:
        raise ValueError(
            f"unknown sort key {sort!r} — use one of {', '.join(_SORT_KEYS)}"
        )
    mark = _norm_filter(filter_mark, review_mod.STATUSES, "filter_mark")
    verdict = _norm_filter(filter_verdict, VERDICTS, "filter_verdict")

    rows = list(episodes)
    if mark is not None:
        rows = [e for e in rows if e["status"] == mark]
    if verdict is not None:
        rows = [e for e in rows if e["verdict"] == verdict]
    if tag and str(tag).strip():
        wanted = str(tag).strip().lower()
        rows = [e for e in rows if any(str(t).lower() == wanted for t in e.get("tags") or ())]
    if q and str(q).strip():
        needle = str(q).strip().lower()
        rows = [e for e in rows if needle in _haystack(e)]

    # Stable, and the input is already in episode order, so ties stay in stored
    # order in both directions — a page that reshuffles equal rows between two
    # polls is a page nobody can read.
    rows.sort(key=key, reverse=str(order or "").strip().lower() == "desc")

    total = len(rows)
    start = max(0, int(offset or 0))
    page = rows[start:] if limit is None else rows[start:start + max(0, int(limit))]
    return {"total": total, "episodes": page}


# ---- dataset lifecycle ----
# Rename is a directory move: LeRobot v3.0's meta/info.json carries no repo_id
# of its own, so the path IS the identity. Delete is an rmtree. Both are cheap
# enough to run inline, unlike a prune (which re-encodes video).

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def validate_repo_id(repo_id: str) -> str:
    """Accept `name` or `namespace/name`, nothing that could climb out."""
    repo_id = (repo_id or "").strip().strip("/")
    if not repo_id:
        raise ValueError("dataset name cannot be empty")
    if not _REPO_ID_RE.match(repo_id):
        raise ValueError(
            f"{repo_id!r} is not a valid dataset name — use letters, digits, "
            ". _ - and at most one / (e.g. local/so101_pick_cube)"
        )
    return repo_id


def _forget(root: Path) -> None:
    """Drop cached reads for a dataset whose files just moved or vanished."""
    _frames_cache.pop(str(root), None)
    for key in [k for k, v in _detail_cache.items() if v[1].get("root") == str(root)]:
        _detail_cache.pop(key, None)


def rename_dataset(repo_id: str, new_repo_id: str) -> dict:
    new_repo_id = validate_repo_id(new_repo_id)
    root = dataset_root(repo_id)
    new_root = dataset_root(new_repo_id)
    if not root.exists():
        raise FileNotFoundError(f"no dataset at {root}")
    if new_root == root:
        raise ValueError("that is already the dataset's name")
    if new_root.exists():
        raise ValueError(f"{new_repo_id} already exists — pick another name")

    new_root.parent.mkdir(parents=True, exist_ok=True)
    root.rename(new_root)
    _forget(root)
    # An in-place prune leaves a `<name>_old` sibling; keep the pair together so
    # the backup does not end up orphaned under the previous name.
    backup = root.with_name(root.name + "_old")
    moved_backup = None
    if backup.exists():
        new_backup = new_root.with_name(new_root.name + "_old")
        if not new_backup.exists():
            backup.rename(new_backup)
            _forget(backup)
            moved_backup = str(new_backup)
    return {"repo_id": new_repo_id, "root": str(new_root), "moved_backup": moved_backup}


def delete_dataset(repo_id: str) -> dict:
    """Remove a dataset directory outright. There is no undo."""
    import shutil

    root = dataset_root(repo_id)
    if not root.exists():
        raise FileNotFoundError(f"no dataset at {root}")
    if root == hf_home():
        raise ValueError("refusing to delete the dataset cache root")
    size = _dir_size(root)
    shutil.rmtree(root)
    _forget(root)
    return {"repo_id": repo_id, "root": str(root), "freed_bytes": size}


# ---- per-episode trace ----

def episode_trace(repo_id: str, episode: int, max_points: int = TRACE_MAX_POINTS) -> dict:
    """Downsampled action/state per column for the timeline chart.

    Downsampling is by STRIDE, not by averaging: the point of this chart is
    whether the gripper closed and whether the arm followed, and an average
    would smear exactly the short transitions that answer both.
    """
    root = dataset_root(repo_id)
    info = _info(root)
    if info is None:
        raise FileNotFoundError(f"no dataset at {root}")
    fps = int(info.get("fps", 30)) or 30
    rig = RigSpec.from_info(info)

    df = _frames(root)
    sub = df[df["episode_index"] == int(episode)]
    if sub.empty:
        raise KeyError(f"episode {episode} not in {repo_id}")

    state = np.stack(sub["observation.state"].to_numpy())
    action = np.stack(sub["action"].to_numpy())
    n = len(state)
    stride = max(1, n // max_points)
    idx = np.arange(0, n, stride)

    names = rig.state_names or tuple(f"j{i}" for i in range(state.shape[1]))

    # The jaw trace is the one series a reviewer actually reads, and its
    # thresholds are what the verdict beside the chart was reached with. Pulling
    # both out per arm — rather than leaving the page to find the gripper column
    # by name — is what stops a bimanual dataset drawing the left jaw twice.
    gripper = [
        {
            "side": arm.side,
            "name": arm.gripper_name,
            "index": arm.gripper_idx,
            "closed_below": arm.closed_below,
            "open_above": arm.open_above,
            "values": [float(v) for v in state[idx, arm.gripper_idx]],
        }
        for arm in rig.arms
        if arm.gripper_idx is not None and arm.gripper_idx < state.shape[1]
    ]

    return {
        "episode_index": int(episode),
        "fps": fps,
        "frames": n,
        "seconds": n / fps,
        "rig": rig.rig,
        "names": [str(x) for x in names],
        "t": [float(i) / fps for i in idx],
        "state": [[float(v) for v in state[idx, j]] for j in range(state.shape[1])],
        "action": [[float(v) for v in action[idx, j]] for j in range(action.shape[1])],
        "gripper": gripper,
    }


# ---- video ----

def video_path(repo_id: str, key: str, chunk_index: int = 0, file_index: int = 0) -> Path:
    """Path of the mp4 holding a given video key's chunk/file.

    The template lives in info.json (`video_path`), so this follows the
    dataset's own convention rather than assuming the default layout.
    """
    root = dataset_root(repo_id)
    info = _info(root)
    if info is None:
        raise FileNotFoundError(f"no dataset at {root}")
    if key not in _video_keys(info):
        raise KeyError(f"{key!r} is not a video feature of {repo_id}")
    template = (
        info.get("video_path")
        or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    rel = template.format(video_key=key, chunk_index=int(chunk_index), file_index=int(file_index))
    path = (root / rel).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("video path escapes the dataset root")
    if not path.exists():
        raise FileNotFoundError(f"no video at {path}")
    return path
