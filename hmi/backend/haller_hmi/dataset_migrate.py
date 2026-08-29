# hmi/backend/haller_hmi/dataset_migrate.py
"""Bring a dataset recorded under an OLDER schema up to the one this rig writes.

WHY THIS EXISTS
    `DatasetRecorder._open_dataset` refuses to resume a dataset whose feature
    set is not the rig's, and that refusal is right: `add_frame` validates every
    frame against the dataset's frozen `info.json` features and rejects a
    mismatch in either direction, so appending would run the whole take and then
    save an empty episode. But "record into a NEW repo_id" is not always the
    answer an operator wants — 86 episodes driven by hand before the schema grew
    a column are 86 episodes that should not have to become a second dataset
    just because `observation.effort` was added afterwards.

    So this module is the other branch the refusal names: migrate the older
    dataset to the current schema, once, offline, and then resume into it
    normally. It only ever ADDS columns.

WHAT IT WILL NOT DO — the whole design is in the refusals
    A migration that can invent any column is a migration that can quietly
    fabricate the dataset's content, so only four columns have a fill rule at
    all (`BACKFILL`), and every one of them is a column whose absence has an
    unambiguous meaning. Everything else is refused:

      - a missing CAMERA (`observation.images.*`): there is no honest fill for
        a view that was never pointed at the bench.
      - a missing `next.reward`/`next.done`: a zero reward column is
        indistinguishable from a dataset where every episode failed. This is
        the same reason the recorder omits the pair on an unscored rig rather
        than writing zeros (see recorder's module docstring).
      - a STALE column — one the dataset has and the rig does not produce.
        Removing it would destroy recorded data, and this tool is additive.
      - a SHARED column whose dtype or shape differs. A 6-joint dataset and a
        12-joint rig are not the same robot, and `validate_frame` checks shape
        per frame, so a migration that "succeeded" here would leave the resume
        failing anyway — later, and less legibly.

WHAT IS FABRICATED, AND HOW A CONSUMER TELLS
    The backfilled values are synthetic and the dataset says so on disk: every
    run appends a record to `info.json`'s `haller_migration` list naming the
    columns, the fill, and the episode range that was migrated. Per-frame, the
    marker is `episode_uid`: migrated episodes get NEGATIVE uids, real ones are
    microseconds since the Unix epoch and so are always positive. `episode_uid
    < 0` is exactly "this episode predates the column".

      observation.effort  -> zeros. Already the recorder's own encoding for a
                             take with no load channel: "0.0 is also what an
                             unreadable load register writes, so a flat-zero
                             column means 'no effort channel on that take', not
                             'no contact'".
      observation.base    -> zeros, i.e. a base that never moved. True of every
                             rig that has no base, which is every rig that
                             recorded a dataset without the column.
      observation.wall_clock -> the dataset's own `timestamp` column, which is
                             synthetic (`frame_index / fps`). This one is a
                             REAL loss and not just a gap in the record: the
                             column exists to expose sampling holes, and a
                             migrated episode can no longer show one. It is
                             filled rather than left NaN because a NaN column
                             poisons `aggregate_stats` and any normalisation
                             computed over the merged dataset; the honest
                             signal is the provenance block plus the negative
                             uid, not a value that breaks arithmetic downstream.
      episode_uid         -> `episode_index - total_episodes`, so migrated
                             episodes are unique, negative, and SORT IN
                             RECORDING ORDER before every real uid. Order is
                             the property the column was added for.

WHY IT IS NOT AUTOMATIC
    Nothing calls this from the record path. The recorder's refusal stays a
    refusal, and rewriting an operator's dataset is an explicit act with a
    backup, not a side effect of pressing Start Recording.

    python -m haller_hmi.dataset_migrate local/so101_pick_cube --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .recorder import (
    DONE_FEATURE,
    EPISODE_UID_FEATURE,
    REWARD_FEATURE,
    episode_meta_files,
    lerobot_home,
)

logger = logging.getLogger(__name__)

#: Provenance for every migration this dataset has been through, a LIST because
#: the schema can grow more than once. Namespaced like the recorder's other
#: `info.json` blocks so it cannot collide with a future LeRobot field.
MIGRATION_INFO_KEY = "haller_migration"

#: Where the pre-migration `data/` and `meta/` are copied. Dot-prefixed and
#: INSIDE the dataset root: `lab.catalog.list_datasets` scans for
#: `*/meta/info.json` and `*/*/meta/info.json` under the LeRobot home, and a
#: backup one level deeper than the dataset matches neither — a backup that
#: showed up in the cockpit as a second dataset would be its own bug.
BACKUP_PREFIX = ".pre-migration-"


class MigrationRefused(RuntimeError):
    """This dataset cannot be migrated additively. The message says why."""


# ---- fills ---------------------------------------------------------------

def _fill_zeros(spec: dict, table, ctx: dict) -> np.ndarray:
    n = table.num_rows
    width = int(spec["shape"][0])
    if spec["shape"] == (1,):
        return np.zeros(n, dtype=spec["dtype"])
    return np.zeros((n, width), dtype=spec["dtype"])


def _fill_from_timestamp(spec: dict, table, ctx: dict) -> np.ndarray:
    """LeRobot's own `timestamp`, which is `frame_index / fps` — see the module
    docstring for why a synthetic value beats a NaN here."""
    return table.column("timestamp").to_numpy(zero_copy_only=False).astype(
        spec["dtype"], copy=False)


def _fill_synthetic_uid(spec: dict, table, ctx: dict) -> np.ndarray:
    """Negative, unique, and ascending with `episode_index`: sorts before every
    real (positive, microsecond) uid, which is the ordering the column exists
    for, and marks the frame as pre-dating it."""
    ep = table.column("episode_index").to_numpy(zero_copy_only=False)
    return (ep.astype(np.int64) - int(ctx["total_episodes"])).astype(spec["dtype"])


#: The ONLY columns that can be backfilled. A missing feature outside this
#: table is a refusal, not a zero — see the module docstring.
BACKFILL = {
    "observation.effort": _fill_zeros,
    "observation.base": _fill_zeros,
    "observation.wall_clock": _fill_from_timestamp,
    EPISODE_UID_FEATURE: _fill_synthetic_uid,
}

#: What each fill wrote, recorded into `info.json` so the fabrication is on disk
#: and not only in this file.
FILL_NOTES = {
    "observation.effort": "zeros — no load channel was recorded on these takes",
    "observation.base": "zeros — these takes were recorded on a rig with no base",
    "observation.wall_clock": (
        "copied from the synthetic `timestamp` column (frame_index / fps); "
        "these episodes CANNOT reveal dropped ticks"),
    EPISODE_UID_FEATURE: (
        "synthetic negative ids (episode_index - total_episodes): unique and in "
        "recording order, and negative so they never collide with, and always "
        "sort before, a real microsecond uid"),
}


# ---- planning ------------------------------------------------------------

def _default_feature_keys() -> set[str]:
    from lerobot.datasets.utils import DEFAULT_FEATURES
    return set(DEFAULT_FEATURES)


def _shape(spec: dict) -> tuple:
    return tuple(spec["shape"])


def plan_migration(root: Path, features: dict) -> dict:
    """What migrating `root` to `features` would do, without touching anything.

    Pure and cheap — `info.json` only, no parquet — so a caller can show the
    plan before asking for the rewrite. `ready` is the one field a caller has
    to look at; the rest is what to print.
    """
    from lerobot.datasets.io_utils import load_info

    info = load_info(root)
    have = info["features"]
    want = {k: {**v, "shape": _shape(v)} for k, v in features.items()}
    defaults = _default_feature_keys()

    missing = [k for k in want if k not in have]
    stale = [k for k in have if k not in want and k not in defaults]
    conflicts = [
        {"feature": k,
         "on_disk": {"dtype": have[k]["dtype"], "shape": list(_shape(have[k]))},
         "this_rig": {"dtype": want[k]["dtype"], "shape": list(want[k]["shape"])}}
        for k in want if k in have
        and (have[k]["dtype"] != want[k]["dtype"]
             or _shape(have[k]) != want[k]["shape"])
    ]
    unfillable = [k for k in missing if k not in BACKFILL]

    return {
        "root": str(root),
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", 0)),
        "add": missing,
        "stale": stale,
        "conflicts": conflicts,
        "unfillable": unfillable,
        "ready": bool(missing) and not (stale or conflicts or unfillable),
        "already_current": not missing and not stale and not conflicts,
    }


def _refuse_unless_ready(plan: dict) -> None:
    if plan["already_current"]:
        raise MigrationRefused(
            f"{plan['root']} already has exactly the schema this rig records — "
            "nothing to migrate. If recording still refuses, the mismatch is in "
            "a dtype or shape; run with --dry-run to see the comparison.")
    if plan["stale"]:
        raise MigrationRefused(
            f"{plan['root']} has {plan['stale']} that this rig does not record. "
            "This tool only ADDS columns; dropping those would destroy recorded "
            "data, so it refuses. Record into a new repo_id instead — or, if "
            f"the stale set is {REWARD_FEATURE}/{DONE_FEATURE}, resume it on the "
            "rig that can score (the sim), which is the rig it was recorded on.")
    if plan["conflicts"]:
        detail = "; ".join(
            f"{c['feature']}: on disk {c['on_disk']['dtype']}{c['on_disk']['shape']}, "
            f"this rig {c['this_rig']['dtype']}{c['this_rig']['shape']}"
            for c in plan["conflicts"])
        raise MigrationRefused(
            f"{plan['root']} disagrees with this rig about a column that BOTH "
            f"have ({detail}). That is a different robot, not an older schema — "
            "a 6-joint dataset cannot take 12-joint frames, and `validate_frame` "
            "checks the shape of every frame. Record into a new repo_id.")
    if plan["unfillable"]:
        raise MigrationRefused(
            f"{plan['root']} is missing {plan['unfillable']}, and there is no "
            "honest value to write there. A camera that never saw the bench and "
            "a task outcome nothing scored cannot be invented — a zero reward "
            "column reads exactly like a dataset where every episode failed. "
            "Record into a new repo_id.")


def _assert_readable(root: Path) -> tuple[list[Path], list[Path]]:
    """Every parquet must be readable before anything is rewritten.

    The failure this catches is a LIVE RECORDING SESSION: `pq.ParquetWriter`
    lays its footer down on close, so the file the open dataset is writing has
    no footer and cannot be read. Migrating around it would drop that session's
    episodes on the floor. Same check catches the truncated file a crash leaves.
    """
    data = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    meta = episode_meta_files(root)
    if not data:
        raise MigrationRefused(f"{root} has no data parquet to migrate")
    for path in data + meta:
        try:
            pq.read_schema(path)
        except Exception as e:
            raise MigrationRefused(
                f"{path} cannot be read ({e}). Almost always a recording "
                "session still holding the dataset open — stop the session and "
                "run this again. If no session is running the file is truncated, "
                "and that is a repair, not a migration.") from e
    return data, meta


# ---- the rewrite ---------------------------------------------------------

def _backup(root: Path) -> Path:
    dest = root / f"{BACKUP_PREFIX}{time.strftime('%Y%m%dT%H%M%S')}"
    dest.mkdir(parents=True)
    for name in ("data", "meta"):
        if (root / name).is_dir():
            shutil.copytree(root / name, dest / name)
    return dest


def _migrated_table(table, new_features: dict, ctx: dict):
    """One data parquet, rewritten with the added columns.

    Built through `datasets.Dataset.from_dict` with LeRobot's own HF features
    rather than by hand in pyarrow, because that is the exact call
    `DatasetWriter._save_episode_data` makes for every new episode. Same call,
    same arrow types and same column order, so the file this writes and the file
    the next take appends are one schema — which `load_nested_dataset` requires,
    since it hands every parquet in the directory to one `Dataset.from_parquet`.
    """
    import datasets
    from lerobot.datasets.feature_utils import get_hf_features_from_features

    hf_features = get_hf_features_from_features(new_features)
    cols: dict = {}
    for key in hf_features:
        if key in table.schema.names:
            arr = table.column(key).to_numpy(zero_copy_only=False)
            # A fixed-size-list column comes back as an object array of rows;
            # stack it so `from_dict` sees one 2-D block per feature.
            cols[key] = (np.stack(arr) if arr.dtype == object else arr)
        else:
            cols[key] = BACKFILL[key](new_features[key], table, ctx)
    ds = datasets.Dataset.from_dict(cols, features=hf_features, split="train")
    return ds.with_format("arrow")[:]


def _episode_stats_for(added: list[str], new_features: dict,
                       data_paths: list[Path]) -> dict[int, dict]:
    """Per-episode stats for the ADDED columns only, keyed by episode_index.

    Only the new columns: the existing ones already have stats on disk that were
    computed from these same frames, and recomputing them would rewrite numbers
    this migration did not change.
    """
    from lerobot.datasets.compute_stats import compute_episode_stats

    by_episode: dict[int, list] = {}
    for path in data_paths:
        table = pq.read_table(path, columns=["episode_index"] + added)
        eps = table.column("episode_index").to_numpy(zero_copy_only=False)
        for key in added:
            arr = table.column(key).to_numpy(zero_copy_only=False)
            arr = np.stack(arr) if arr.dtype == object else arr
            for ep in np.unique(eps):
                sel = arr[eps == ep]
                by_episode.setdefault(int(ep), {}).setdefault(key, []).append(sel)

    out: dict[int, dict] = {}
    for ep, per_key in by_episode.items():
        buf = {}
        for key, chunks in per_key.items():
            block = np.concatenate(chunks)
            # `compute_episode_stats` reduces over axis 0 and keeps dims only
            # for a 1-D column, which is the layout a shape-(1,) feature has.
            buf[key] = block
        out[ep] = compute_episode_stats(
            buf, {k: new_features[k] for k in buf})
    return out


def _rewrite_episode_meta(path: Path, stats_by_episode: dict[int, dict]) -> None:
    import pyarrow as pa

    table = pq.read_table(path)
    eps = [int(e) for e in table.column("episode_index").to_pylist()]
    columns = {name: table.column(name) for name in table.schema.names}
    for key, stat_names in _stat_layout(stats_by_episode).items():
        for stat in stat_names:
            values = [np.asarray(stats_by_episode[e][key][stat]).ravel().tolist()
                      for e in eps]
            columns[f"stats/{key}/{stat}"] = pa.array(values)
    pq.write_table(pa.table(columns), path, compression="snappy",
                   use_dictionary=True)


def _stat_layout(stats_by_episode: dict[int, dict]) -> dict[str, list[str]]:
    any_episode = next(iter(stats_by_episode.values()))
    return {key: list(stat.keys()) for key, stat in any_episode.items()}


def _fold_into_aggregate(root: Path, stats_by_episode: dict[int, dict]) -> None:
    """Add the new columns to `meta/stats.json`, leaving the old ones alone.

    `aggregate_stats` takes the UNION of keys, so an aggregate that is one
    column short still loads — but the recorder folds every new episode into
    this file incrementally, and a column that only ever appears from the next
    take onwards would carry statistics for a fraction of the dataset while
    looking like statistics for all of it.
    """
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import load_stats, write_stats

    stats = load_stats(root) or {}
    per_episode = list(stats_by_episode.values())
    if not per_episode:
        return
    stats.update(aggregate_stats(per_episode))
    write_stats(stats, root)


def _write_info(root: Path, new_features: dict, added: list[str],
                plan: dict) -> None:
    from lerobot.datasets.io_utils import write_info

    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"] = {
        k: {**v, "shape": list(v["shape"])} for k, v in new_features.items()}
    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "added": list(added),
        "episodes": [0, max(plan["total_episodes"] - 1, 0)],
        "frames": plan["total_frames"],
        "fills": {k: FILL_NOTES[k] for k in added},
        "note": (
            "These columns were BACKFILLED, not recorded. Every frame of "
            f"episodes 0..{max(plan['total_episodes'] - 1, 0)} carries a "
            "synthetic value for them; frames recorded after this migration are "
            f"genuine. Per-frame, `{EPISODE_UID_FEATURE}` < 0 marks a migrated "
            "episode."),
    }
    info.setdefault(MIGRATION_INFO_KEY, []).append(record)
    write_info(info, root)


def _inherit_state_names(new_features: dict, added: list[str]) -> None:
    """A backfilled `observation.effort` is named like the dataset's OWN state.

    The column's contract is "same names and same layout as state, so a consumer
    can zip the three columns joint-for-joint" — and the state column being
    zipped against is the one already on disk. A dataset written by
    `lerobot-record` calls its joints `shoulder_pan.pos`; taking this rig's
    `left_shoulder_pan` for the new column would put two naming conventions in
    one dataset and break exactly the zip the names exist for.
    """
    state = new_features.get("observation.state", {})
    names = state.get("names")
    if "observation.effort" not in added or not names:
        return
    effort = new_features["observation.effort"]
    if len(names) == effort["shape"][0]:
        effort["names"] = list(names)


def migrate_dataset(root: Path, features: dict, *, backup: bool = True) -> dict:
    """Rewrite `root` in place so it carries exactly `features`.

    ORDER IS CRASH SAFETY. `info.json` is the schema DECLARATION and is written
    last: a crash after the parquets are rewritten leaves a dataset with columns
    its info does not mention, which loads; the reverse leaves an info promising
    columns that are not there, which does not.
    """
    root = Path(root)
    plan = plan_migration(root, features)
    _refuse_unless_ready(plan)
    data_paths, meta_paths = _assert_readable(root)

    from lerobot.datasets.io_utils import load_info

    added = list(plan["add"])
    on_disk = load_info(root)["features"]
    new_features = {**on_disk}
    for key, spec in features.items():
        new_features.setdefault(key, {**spec, "shape": _shape(spec)})
    _inherit_state_names(new_features, added)
    ctx = {"total_episodes": plan["total_episodes"]}

    backup_dir = _backup(root) if backup else None
    logger.info("dataset_migrate: %s adding %s (%d episodes, %d frames)%s",
                root, added, plan["total_episodes"], plan["total_frames"],
                f"; backup at {backup_dir}" if backup_dir else "")

    for path in data_paths:
        table = _migrated_table(pq.read_table(path), new_features, ctx)
        pq.write_table(table, path, compression="snappy", use_dictionary=True)

    stats_by_episode = _episode_stats_for(added, new_features, data_paths)
    for path in meta_paths:
        _rewrite_episode_meta(path, stats_by_episode)
    _fold_into_aggregate(root, stats_by_episode)
    _write_info(root, new_features, added, plan)

    return {**plan, "added": added, "backup": str(backup_dir) if backup_dir else None}


# ---- CLI -----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m haller_hmi.dataset_migrate",
        description="Add this rig's newer columns to an older LeRobot dataset "
                    "so recording can resume into it.")
    parser.add_argument("repo_id", help="e.g. local/so101_pick_cube")
    parser.add_argument("--root", help="dataset directory, if not under the "
                                       "LeRobot home")
    parser.add_argument("--features", help="JSON file holding the feature dict "
                                           "to migrate TO; defaults to asking "
                                           "the running HMI at --hmi")
    parser.add_argument("--hmi", default="http://127.0.0.1:8000",
                        help="running backend to ask for the rig's schema")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and change nothing")
    parser.add_argument("--no-backup", action="store_true",
                        help="skip the pre-migration copy of data/ and meta/")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(args.root) if args.root else lerobot_home() / args.repo_id
    if not (root / "meta" / "info.json").exists():
        print(f"no dataset at {root}", file=sys.stderr)
        return 2

    try:
        features = _load_features(args)
    except Exception as e:
        print(f"could not determine this rig's schema: {e}", file=sys.stderr)
        return 2

    plan = plan_migration(root, features)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    try:
        result = migrate_dataset(root, features, backup=not args.no_backup)
    except MigrationRefused as e:
        print(f"\nrefused: {e}", file=sys.stderr)
        return 1
    print(f"\nmigrated {root}: added {result['added']}")
    if result["backup"]:
        print(f"pre-migration copy kept at {result['backup']}")
    return 0


def _load_features(args) -> dict:
    """The schema to migrate TO: a JSON file, or the running backend's own."""
    if args.features:
        return json.loads(Path(args.features).read_text())
    import urllib.request
    with urllib.request.urlopen(f"{args.hmi}/record/schema", timeout=5) as r:
        return json.loads(r.read())["features"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
