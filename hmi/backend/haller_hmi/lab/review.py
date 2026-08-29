# hmi/backend/haller_hmi/lab/review.py
"""Keep/reject marks, tags and undo batches for recorded episodes — a sidecar,
not a rewrite.

Deciding an episode is bad and *destroying* it are two different acts,
and a review UI that fuses them is one misclick away from losing a
demonstration that took a minute of your life to record. So marks live
in `review.json` at the dataset root and change nothing else: training
simply passes the kept set as `--dataset.episodes`, and writing a
pruned dataset is a separate, explicit export.

The file sits at the dataset ROOT, not under `meta/` — that directory
belongs to LeRobot's own loaders and is not ours to litter.

Layout (v2):

    {
      "version": 2,
      "updated": "2026-08-26T18:40:00+00:00",
      "fingerprint": {"total_episodes": 46, "total_frames": 29500},
      "episodes": {"1": {"status": "reject", "note": "missed grasp",
                         "frames": 640, "tags": ["blurry"]}},
      "batches": [{"id": "...", "at": "...Z", "mode": "grade",
                   "before": {"1": {"status": "keep"}, "2": null}}]
    }

Only marked episodes are stored; anything absent is `unset`, which
counts as KEEP. That way a fresh dataset trains on everything, and the
file stays a record of decisions you actually made.

Each mark also records the LENGTH of the episode it was made about,
which is what makes a mark verifiable later. `delete_episodes` renumbers
the survivors 0..n-1, so marks written before a prune silently describe
*different* episodes afterwards — a "reject" can become a keep-worthy
demonstration. Comparing each mark's recorded length against the episode
now at that index catches it per mark.

Comparing dataset TOTALS cannot do this job, and the earlier version
that tried got it wrong in both directions: appending episodes with
`--resume` changes the totals without renumbering anything (a false
alarm on every recording session), while a single later click would
overwrite the stored totals and hide a real renumbering. Lengths are
per-episode and are not disturbed by an append.

Version 1 files are read as they are and are never upgraded on load —
the real 46-mark review on `local/so101_pick_cube` carries neither
`frames` nor `tags` and must keep reading as 35 keep / 11 reject with
no stale marks. `save` stamps `REVIEW_VERSION`, so a v1 file becomes a
v2 file the first time something is actually marked, and not before.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

REVIEW_FILENAME = "review.json"
REVIEW_VERSION = 2

KEEP = "keep"
REJECT = "reject"
UNSET = "unset"
STATUSES = (KEEP, REJECT, UNSET)

#: How many autoclass batches stay undoable. The review page re-reads this
#: whole file on every poll, so an unbounded undo log is a slow leak in the
#: hot read path; 20 batches is more autoclass runs than one session makes.
MAX_BATCHES = 20


# ---- file ----

def review_path(root: Path | str) -> Path:
    return Path(root) / REVIEW_FILENAME


def _empty() -> dict:
    return {
        "version": REVIEW_VERSION,
        "updated": None,
        "fingerprint": {},
        "episodes": {},
        "batches": [],
    }


def load(root: Path | str) -> dict:
    """Read the sidecar. A missing or corrupt file reads as empty — a
    review file is an annotation, never a thing worth failing a page load
    over."""
    path = review_path(root)
    if not path.exists():
        return _empty()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return _empty()
    # setdefault, never assignment: a v1 file on disk keeps `version: 1`.
    # Filling absent keys in memory is not an upgrade — only `save` writes,
    # and only `save` stamps REVIEW_VERSION.
    data.setdefault("version", REVIEW_VERSION)
    data.setdefault("episodes", {})
    data.setdefault("fingerprint", {})
    data.setdefault("updated", None)
    data.setdefault("batches", [])
    return data


def save(root: Path | str, data: dict) -> None:
    """Atomic write — a half-written review.json read by a concurrent
    page load would look like corruption and reset every mark."""
    path = review_path(root)
    data["version"] = REVIEW_VERSION
    data["updated"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".review-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---- entries ----

def _norm_tags(values: Iterable[str] | None) -> list[str]:
    """Deduplicated, order-preserving, stripped.

    A tag is a filter key in a query string (`/lab/datasets/episodes?tag=`),
    so `" blurry"` must not become a second, invisible tag next to `"blurry"`.
    Order is preserved because the page renders tags in the order they were
    added and a set would reshuffle the chips on every write.
    """
    out: list[str] = []
    for value in values or ():
        tag = str(value).strip()
        if tag and tag not in out:
            out.append(tag)
    return out


def _is_empty(entry: dict) -> bool:
    """True for an entry that holds no decision, which is not stored.

    `unset` with no note is the kit's rule and it survives tags: tags ARE a
    decision, so an entry carrying only tags stays: dropping it would delete
    them. `frames` is metadata ABOUT a decision and never one itself, so it
    does not keep an otherwise empty entry alive.
    """
    return (
        entry.get("status", UNSET) == UNSET
        and not entry.get("note")
        and not entry.get("tags")
    )


def set_status(
    root: Path | str,
    episode: int,
    status: str,
    note: str | None = None,
    fingerprint: dict | None = None,
    episode_frames: int | None = None,
) -> dict:
    """Mark one episode. `unset` with no note removes the entry entirely
    so the file only ever holds real decisions."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    data = load(root)
    key = str(int(episode))
    entry = dict(data["episodes"].get(key) or {})
    entry["status"] = status
    if note is not None:
        entry["note"] = note
    if episode_frames is not None:
        entry["frames"] = int(episode_frames)
    if _is_empty(entry):
        data["episodes"].pop(key, None)
    else:
        data["episodes"][key] = entry
    if fingerprint:
        data["fingerprint"] = fingerprint
    save(root, data)
    return data


def set_many(
    root: Path | str,
    marks: dict[int, str],
    fingerprint: dict | None = None,
    note: str | None = None,
    episode_frames: dict[int, int] | None = None,
) -> dict:
    """Apply several marks in one write (the auto-grade bulk action)."""
    data = load(root)
    for episode, status in marks.items():
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        key = str(int(episode))
        entry = dict(data["episodes"].get(key) or {})
        entry["status"] = status
        if note is not None:
            entry["note"] = note
        if episode_frames and int(episode) in episode_frames:
            entry["frames"] = int(episode_frames[int(episode)])
        if _is_empty(entry):
            data["episodes"].pop(key, None)
        else:
            data["episodes"][key] = entry
    if fingerprint:
        data["fingerprint"] = fingerprint
    save(root, data)
    return data


def bulk_update(
    root: Path | str,
    episodes: Iterable[int],
    status: str | None = None,
    note: str | None = None,
    tags_add: Iterable[str] | None = None,
    tags_remove: Iterable[str] | None = None,
    episode_frames: dict[int, int] | None = None,
) -> int:
    """Apply a status and/or tags to a LIST of episodes in ONE write.

    Returns how many episodes' stored entries actually CHANGED, not how many
    were named: adding a tag every selected episode already carries reports 0.
    The page shows that number back to the operator, and "12 updated" after a
    no-op is how a selection that missed its rows goes unnoticed.

    Removals happen before additions, so passing a tag in both `tags_remove`
    and `tags_add` leaves it present rather than depending on argument order.
    """
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    add = _norm_tags(tags_add)
    drop = set(_norm_tags(tags_remove))
    if status is None and note is None and not add and not drop:
        raise ValueError("nothing to apply: pass status, note, tags_add or tags_remove")

    data = load(root)
    changed = 0
    for episode in episodes:
        key = str(int(episode))
        before = data["episodes"].get(key)
        entry = dict(before or {})
        if status is not None:
            entry["status"] = status
        if note is not None:
            entry["note"] = note
        if add or drop:
            tags = [t for t in _norm_tags(entry.get("tags")) if t not in drop]
            tags += [t for t in add if t not in tags]
            if tags:
                entry["tags"] = tags
            else:
                entry.pop("tags", None)
        if episode_frames and int(episode) in episode_frames:
            entry["frames"] = int(episode_frames[int(episode)])
        if _is_empty(entry):
            data["episodes"].pop(key, None)
            after = None
        else:
            data["episodes"][key] = entry
            after = entry
        if after != before:
            changed += 1
    # No write when nothing changed: this file is polled, and rewriting it to
    # move only `updated` would upgrade a v1 file for a no-op.
    if changed:
        save(root, data)
    return changed


def clear(root: Path | str, fingerprint: dict | None = None) -> dict:
    data = load(root)
    data["episodes"] = {}
    if fingerprint:
        data["fingerprint"] = fingerprint
    save(root, data)
    return data


# ---- reads ----

def status_of(data: dict, episode: int) -> str:
    return (data.get("episodes", {}).get(str(int(episode))) or {}).get("status", UNSET)


def note_of(data: dict, episode: int) -> str:
    return (data.get("episodes", {}).get(str(int(episode))) or {}).get("note", "")


def tags_of(data: dict, episode: int) -> list[str]:
    """A fresh normalised list, so a caller cannot mutate the loaded entry."""
    entry = data.get("episodes", {}).get(str(int(episode))) or {}
    return _norm_tags(entry.get("tags"))


def keep_list(data: dict, total_episodes: int) -> list[int]:
    """Episode indices that training should see: everything not rejected."""
    return [i for i in range(total_episodes) if status_of(data, i) != REJECT]


def stale_marks(data: dict, episode_frames: dict[int, int]) -> list[int]:
    """Marked episodes that are no longer the episode that was marked.

    `episode_frames` maps the dataset's CURRENT episode indices to their
    lengths. A mark is suspect when the episode it names has a different
    length than it did when marked, or has vanished off the end — both
    are what renumbering after a prune looks like from here.

    Marks written before lengths were recorded carry no `frames` and can
    only be checked for pointing past the end.
    """
    suspect = []
    for key, entry in (data.get("episodes") or {}).items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        current = episode_frames.get(idx)
        if current is None:
            suspect.append(idx)          # marked an episode that is gone
            continue
        recorded = entry.get("frames")
        if recorded is not None and int(recorded) != int(current):
            suspect.append(idx)
    return sorted(suspect)


def is_stale(data: dict, fingerprint: dict, episode_frames: dict[int, int] | None = None) -> bool:
    """True when at least one mark can no longer be trusted.

    Deliberately NOT "the dataset changed size": recording more episodes
    with `--resume` appends them, leaving every existing index exactly
    where it was, and warning about that on every session would train
    people to ignore the warning that matters.
    """
    if not data.get("episodes"):
        return False
    if episode_frames is not None:
        return bool(stale_marks(data, episode_frames))
    # Legacy fallback for review files written before per-mark lengths:
    # only a SHRINK implies episodes were removed and renumbered.
    stored = data.get("fingerprint") or {}
    if not stored:
        return False
    return any(
        k in stored and fingerprint.get(k, 0) < stored[k]
        for k in ("total_episodes", "total_frames")
    )


def counts(data: dict, total_episodes: int) -> dict:
    kept = rejected = unset = 0
    for i in range(total_episodes):
        s = status_of(data, i)
        if s == REJECT:
            rejected += 1
        elif s == KEEP:
            kept += 1
        else:
            unset += 1
    return {"keep": kept, "reject": rejected, "unset": unset, "train": kept + unset}


# ---- autoclass batches ----

def record_batch(root: Path | str, batch_id: str, mode: str, before: dict) -> dict:
    """Append the marks an autoclass apply is about to overwrite.

    `before` maps episode index -> the entry that was there, or None when the
    episode had no entry at all. None has to mean DELETE on the way back:
    "there was no mark" is a state the operator saw, and restoring it as an
    `unset` entry would leave the file holding a decision nobody made.

    Append-only, capped at the most recent `MAX_BATCHES`.
    """
    data = load(root)
    batch = {
        "id": str(batch_id),
        # Z, per the contract. `updated` keeps the kit's `+00:00`; both are
        # UTC and both parse in Python and in the browser.
        "at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": str(mode),
        "before": {
            str(int(ep)): (dict(entry) if entry else None)
            for ep, entry in (before or {}).items()
        },
    }
    batches = list(data.get("batches") or [])
    batches.append(batch)
    data["batches"] = batches[-MAX_BATCHES:]
    save(root, data)
    return batch


def revert_batch(root: Path | str, batch_id: str) -> int:
    """Restore every entry a batch overwrote, and REMOVE the batch.

    Returns the number of episodes restored; raises KeyError for an id that is
    not in the list — including one that was already reverted. The batch is
    dropped rather than kept because its `before` is only true of the state
    immediately after the apply: replaying it a second time would silently
    undo whatever was marked by hand since, and still report success.
    """
    data = load(root)
    batches = list(data.get("batches") or [])
    wanted = str(batch_id)
    for i, batch in enumerate(batches):
        if str(batch.get("id")) == wanted:
            break
    else:
        raise KeyError(wanted)
    restored = 0
    for key, entry in (batch.get("before") or {}).items():
        key = str(key)
        if entry:
            data["episodes"][key] = dict(entry)
        else:
            data["episodes"].pop(key, None)
        restored += 1
    batches.pop(i)
    data["batches"] = batches
    save(root, data)
    return restored
