# hmi/backend/tests/lab/test_review.py
"""`review.json` — the sidecar that decides which episodes train.

Three properties carry the weight here.

STALENESS IS PER MARK. The earlier attempt compared dataset TOTALS and was
wrong in both directions: `--resume` appends episodes without renumbering
anything, so it cried wolf on every recording session, while a single later
click overwrote the stored totals and hid a real prune. Both directions are
tested, because a staleness warning that fires every session is a staleness
warning nobody reads.

V1 FILES ARE NOT REWRITTEN ON LOAD. The real 46-mark review on
`local/so101_pick_cube` carries no `tags`, no `batches` and no per-mark
`frames`, and it has to keep reading as it is until something is actually
marked.

ABSENCE IS A STATE. `unset` with no note removes the entry rather than storing
a decision nobody made, and a reverted batch restores absence rather than
manufacturing an `unset`.

Every test builds its own tree with `_dataset.make_dataset`, whose episodes
have DISTINCT lengths (90, 91, 92, ...) — equal lengths would make a
renumbering invisible, which is the failure the per-mark check exists to see.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from haller_hmi.lab import review

from . import _dataset


@pytest.fixture()
def root(tmp_path) -> Path:
    ds = tmp_path / "local" / "smoke"
    _dataset.make_dataset(ds, n_episodes=4)
    return ds


def _frames(root: Path) -> dict[int, int]:
    """The dataset's CURRENT episode index -> length, off its own meta."""
    meta = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return {int(r.episode_index): int(r.length) for r in meta.itertuples()}


# ---- v1 files ----

def test_a_v1_file_loads_and_counts_without_being_rewritten(root):
    """A v1 file is READ as it is. `load` fills absent keys in memory with
    `setdefault`, which is not an upgrade — only `save` stamps a version, and
    only a real mark calls `save`."""
    path = _dataset.write_review(
        root, {0: "keep", 1: "reject", 3: {"status": "reject", "note": "still arm"}},
        version=1)
    on_disk = path.read_text()

    data = review.load(root)

    assert data["version"] == 1
    assert review.counts(data, 4) == {"keep": 1, "reject": 2, "unset": 1, "train": 2}
    assert review.keep_list(data, 4) == [0, 2]
    assert review.note_of(data, 3) == "still arm"
    assert path.read_text() == on_disk
    assert json.loads(path.read_text())["version"] == 1


def test_an_unset_episode_counts_as_train(root):
    """A fresh dataset trains on everything: only a REJECT is excluded, so the
    file stays a record of decisions actually made."""
    _dataset.write_review(root, {}, version=1)
    data = review.load(root)

    assert review.counts(data, 4)["unset"] == 4
    assert review.keep_list(data, 4) == [0, 1, 2, 3]


def test_a_missing_review_file_reads_as_empty_rather_than_failing(root):
    data = review.load(root)

    assert data["episodes"] == {}
    assert review.counts(data, 4) == {"keep": 0, "reject": 0, "unset": 4, "train": 4}


def test_a_corrupt_review_file_reads_as_empty(root):
    """An annotation file is never worth failing a page load over."""
    review.review_path(root).write_text("{not json")

    assert review.load(root)["episodes"] == {}


def test_a_bulk_update_that_changes_nothing_does_not_upgrade_a_v1_file(root):
    """This file is polled. Rewriting it to move only `updated` would silently
    turn a v1 file into a v2 one for a no-op."""
    path = _dataset.write_review(root, {0: "keep"}, version=1)
    on_disk = path.read_text()

    changed = review.bulk_update(root, [0], status=review.KEEP)

    assert changed == 0
    assert path.read_text() == on_disk


# ---- staleness, per mark ----

def test_appending_an_episode_flags_no_mark(root):
    """`--resume` leaves every existing index exactly where it was. Warning
    here on every recording session would train people to ignore the warning
    that matters."""
    frames = _frames(root)
    review.set_status(root, 1, review.REJECT, episode_frames=frames[1])
    data = review.load(root)

    appended = dict(frames)
    appended[4] = 120

    assert review.stale_marks(data, appended) == []
    assert review.is_stale(data, {}, appended) is False


def test_changing_the_length_at_a_marked_index_flags_only_that_index(root):
    """A prune renumbers the survivors, so the episode at a marked index is a
    DIFFERENT episode afterwards — and the only evidence is that its length
    changed."""
    frames = _frames(root)
    review.set_status(root, 1, review.REJECT, episode_frames=frames[1])
    review.set_status(root, 3, review.KEEP, episode_frames=frames[3])
    data = review.load(root)

    pruned = dict(frames)
    pruned[3] = 999

    assert review.stale_marks(data, pruned) == [3]
    assert review.is_stale(data, {}, pruned) is True


def test_a_mark_pointing_past_the_end_is_stale(root):
    frames = _frames(root)
    review.set_status(root, 3, review.REJECT, episode_frames=frames[3])
    data = review.load(root)

    shrunk = {k: v for k, v in frames.items() if k != 3}

    assert review.stale_marks(data, shrunk) == [3]
    assert review.is_stale(data, {}, shrunk) is True


def test_an_unmarked_review_is_never_stale(root):
    data = review.load(root)

    assert review.is_stale(data, {}, {}) is False
    assert review.stale_marks(data, {}) == []


# ---- entries: absence is a state ----

def test_unset_with_no_note_removes_the_entry(root):
    review.set_status(root, 2, review.REJECT)
    assert "2" in review.load(root)["episodes"]

    review.set_status(root, 2, review.UNSET)

    assert "2" not in review.load(root)["episodes"]


def test_an_entry_with_tags_survives_an_unset(root):
    """Tags ARE a decision. Dropping the entry on unset would delete them."""
    review.bulk_update(root, [2], tags_add=["blurry"])

    review.set_status(root, 2, review.UNSET)
    data = review.load(root)

    assert review.tags_of(data, 2) == ["blurry"]
    assert review.status_of(data, 2) == review.UNSET


def test_an_entry_with_a_note_survives_an_unset(root):
    review.set_status(root, 2, review.UNSET, note="looked at it, not sure")
    data = review.load(root)

    assert review.note_of(data, 2) == "looked at it, not sure"


def test_an_unknown_status_is_refused_rather_than_stored(root):
    with pytest.raises(ValueError):
        review.set_status(root, 0, "maybe")


# ---- tags ----

def test_tags_dedupe_and_preserve_the_order_they_were_added(root):
    """A tag is a filter key in a query string, so `" blurry"` must not become
    a second, invisible tag next to `"blurry"`; order is kept because the page
    renders the chips in it."""
    review.bulk_update(root, [1], tags_add=[" blurry", "blurry", "dark", "blurry", ""])
    data = review.load(root)

    assert review.tags_of(data, 1) == ["blurry", "dark"]


def test_bulk_update_applies_status_and_tags_in_one_write(root, monkeypatch):
    review.bulk_update(root, [0, 1, 2], tags_add=["stale-tag"])
    writes = _count_saves(monkeypatch)

    changed = review.bulk_update(root, [0, 1, 2], status=review.REJECT,
                                 tags_add=["blurry"], tags_remove=["stale-tag"])
    data = review.load(root)

    assert changed == 3
    assert writes == [1], "three episodes, three changes, ONE write"
    for episode in (0, 1, 2):
        assert review.status_of(data, episode) == review.REJECT
        assert review.tags_of(data, episode) == ["blurry"]


def test_bulk_update_reports_how_many_entries_actually_changed(root):
    """A count of what was NAMED rather than what changed is how a selection
    that missed its rows goes unnoticed: the page shows the number back."""
    review.bulk_update(root, [0, 1], tags_add=["blurry"])

    assert review.bulk_update(root, [0, 1], tags_add=["blurry"]) == 0
    assert review.bulk_update(root, [1, 2], tags_add=["blurry"]) == 1


def test_bulk_update_with_nothing_to_apply_is_refused(root):
    with pytest.raises(ValueError):
        review.bulk_update(root, [0, 1])


def test_a_tag_in_both_lists_survives_regardless_of_argument_order(root):
    review.bulk_update(root, [0], tags_add=["blurry"])

    review.bulk_update(root, [0], tags_add=["blurry"], tags_remove=["blurry"])

    assert review.tags_of(review.load(root), 0) == ["blurry"]


def test_tags_of_hands_back_a_copy_the_caller_cannot_mutate(root):
    review.bulk_update(root, [0], tags_add=["blurry"])
    data = review.load(root)

    review.tags_of(data, 0).append("injected")

    assert review.tags_of(data, 0) == ["blurry"]


# ---- autoclass batches ----

def test_a_batch_round_trips_including_an_episode_that_had_no_entry(root):
    """`None` in `before` means DELETE on the way back: "there was no mark" is
    a state the operator saw, and restoring it as an `unset` entry would leave
    the file holding a decision nobody made."""
    review.set_status(root, 1, review.REJECT, note="missed grasp")
    prior = review.load(root)["episodes"]
    before = {1: prior.get("1"), 2: prior.get("2")}
    assert before[2] is None

    review.record_batch(root, "abc123def456", "grade", before)
    review.set_status(root, 1, review.KEEP)
    review.set_status(root, 2, review.REJECT)

    restored = review.revert_batch(root, "abc123def456")
    data = review.load(root)

    assert restored == 2
    assert data["episodes"]["1"] == {"status": "reject", "note": "missed grasp"}
    assert "2" not in data["episodes"], "an absent entry must go back to absent"


def test_reverting_a_batch_twice_raises_rather_than_replaying_it(root):
    """A batch's `before` is only true of the state immediately after the
    apply. Replaying it would silently undo whatever was marked by hand since,
    and still report success."""
    review.record_batch(root, "once", "grade", {0: {"status": "keep"}})
    review.revert_batch(root, "once")

    with pytest.raises(KeyError):
        review.revert_batch(root, "once")


def test_reverting_an_unknown_batch_raises(root):
    with pytest.raises(KeyError):
        review.revert_batch(root, "never-recorded")


def test_a_reverted_batch_is_dropped_from_the_file(root):
    review.record_batch(root, "keepme", "grade", {0: None})
    review.record_batch(root, "dropme", "grade", {1: None})

    review.revert_batch(root, "dropme")

    assert [b["id"] for b in review.load(root)["batches"]] == ["keepme"]


def test_the_batch_list_is_capped_at_twenty(root):
    """The review page re-reads this whole file on every poll, so an unbounded
    undo log is a slow leak in the hot read path."""
    assert review.MAX_BATCHES == 20
    for i in range(25):
        review.record_batch(root, f"b{i:02d}", "grade", {0: None})

    batches = review.load(root)["batches"]

    assert len(batches) == 20
    assert [b["id"] for b in batches] == [f"b{i:02d}" for i in range(5, 25)]


# ---- atomic writes ----

def test_save_leaves_no_temp_file_in_the_dataset_root(root):
    """`review.json` is written through a `.review-*.json` temp and an
    `os.replace`; a leftover temp would show up in the dataset tree that gets
    uploaded and pruned."""
    review.set_status(root, 0, review.KEEP)
    review.bulk_update(root, [1, 2], status=review.REJECT, tags_add=["blurry"])
    review.record_batch(root, "abc", "grade", {0: None})

    assert sorted(p.name for p in root.glob(".review-*")) == []
    assert json.loads(review.review_path(root).read_text())["version"] == 2


def test_a_failed_save_leaves_neither_a_temp_file_nor_a_truncated_review(root, monkeypatch):
    """A half-written file read by a concurrent page load would look like
    corruption, and `load` resets every mark on corruption — so a failed write
    must leave the previous file exactly where it was."""
    review.set_status(root, 0, review.KEEP)
    intact = review.review_path(root).read_text()

    def boom(*_args, **_kwargs):
        raise ValueError("disk full")

    monkeypatch.setattr("haller_hmi.lab.review.json.dump", boom)
    with pytest.raises(ValueError):
        review.set_status(root, 1, review.REJECT)

    assert review.review_path(root).read_text() == intact
    assert sorted(p.name for p in root.glob(".review-*")) == []
    assert review.status_of(review.load(root), 0) == review.KEEP


# ---- helpers ----

def _count_saves(monkeypatch) -> list[int]:
    """A list that grows by one every time `review.save` reaches the disk."""
    calls: list[int] = []
    real = review.save

    def counted(root, data):
        calls.append(1)
        return real(root, data)

    monkeypatch.setattr(review, "save", counted)
    return calls
