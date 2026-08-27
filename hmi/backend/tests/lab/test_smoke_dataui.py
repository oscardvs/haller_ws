# hmi/backend/tests/lab/test_smoke_dataui.py
"""The kit's `tools/smoke_test_dataui.py`, ported with every assertion intact.

38 checks in the kit, 38 here: 14 over the dataset layer run TWICE — once
against a direct `HF_LEROBOT_HOME` and once through a symlinked one — plus 10
over `plan_eval_split`.

**The symlink pass is the point, not thoroughness.** Moving datasets out of
`~/.cache` and leaving a compatibility symlink behind is the obvious way to do
it, and it broke the page once: `hf_home()` returned the unresolved link while
`dataset_root()` returned the resolved target, so `relative_to` raised "is not
in the subpath of" on two spellings of one directory. Nothing about that is
visible until someone actually has a symlinked home — and on this box
`~/.cache/huggingface/lerobot` IS a symlink to `~/robot-data/lerobot`.

Three shapes moved in the port, and only three. An episode's flat `why` is now
`reasons[0]` (one entry per arm); `catalog` and `review` live under
`haller_hmi.lab`; and `plan_eval_split` is re-exported by `catalog` rather than
defined in it. The mark is still spelled `status` on a catalog episode — the
frozen HTTP contract renames it to `mark`, but that rename happens at the
routes layer, which is not what this file reads.

Every tree is built under `tmp_path`. Nothing here reads `~/robot-data`: those
are the equivalence anchors, and a test that marks or prunes one destroys a
recording that cost an evening to capture.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from haller_hmi.lab import catalog, review

from ._dataset import make_dataset

#: Frames in episode 0 of the kit's fixture; the trace must return all of them
#: undownsampled (`TRACE_MAX_POINTS` is 600).
EPISODE_0_FRAMES = 90


@pytest.fixture(autouse=True)
def cold_caches():
    """Empty the catalog's module-level caches, and prove the test refilled them.

    `hf_home()` is re-read on every call, but `_detail_cache` is keyed by
    repo-id ALONE and `_frames_cache` by dataset root — so the symlink pass,
    which resolves to the same root under the same repo-id, could be answered
    entirely out of the direct pass's entries. A symlink regression test that
    never re-read is worse than no test: it passes for the same reason the bug
    is invisible. The kit sidestepped this by deleting `vr_teleop_kit.data.*`
    out of `sys.modules` between runs.

    Cleared BEFORE and checked non-empty AFTER, which together are the proof:
    an entry can only be present at teardown if this test's own read put it
    there.
    """
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()
    yield
    if not catalog._detail_cache or not catalog._frames_cache:
        pytest.fail(
            "catalog caches are still empty — this pass answered from nothing "
            "it read, so it proved nothing about the home it was given"
        )


def _lerobot_home(tmp_path: Path, spelling: str) -> Path:
    """A `local/smoke` dataset, and the spelling of its home under test."""
    real = tmp_path / "real" / "lerobot"
    make_dataset(real / "local" / "smoke")
    if spelling == "direct":
        return real
    link = tmp_path / "cache" / "lerobot"
    link.parent.mkdir(parents=True)
    link.symlink_to(real)
    return link


def _refuses(repo_id: str) -> bool:
    """True when `dataset_root` rejects a repo-id instead of resolving it.

    A plain bool feeding a plain `assert` rather than `pytest.raises`, so the
    failure names WHICH spelling was let through.
    """
    try:
        catalog.dataset_root(repo_id)
    except ValueError:
        return True
    return False


@pytest.mark.parametrize("spelling", ["direct", "symlink"])
def test_dataset_layer(monkeypatch, tmp_path, spelling):
    """The 14 checks that must hold whatever spelling of the home is used."""
    home = _lerobot_home(tmp_path, spelling)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))

    listed = catalog.list_datasets()
    assert [d["repo_id"] for d in listed] == ["local/smoke"]

    # The regression: `repo_id` is derived by `relative_to(hf_home())`, so an
    # unresolved home and a resolved root have no common prefix.
    detail = catalog.dataset_detail("local/smoke")
    assert detail["repo_id"] == "local/smoke"
    assert [e["verdict"] for e in detail["episodes"]] == ["PASS", "FAIL"]
    assert "never moved" in detail["episodes"][1]["reasons"][0]
    # 1-based, because Oscar counts episodes 1-based in conversation and that
    # off-by-one is how the wrong demonstration gets deleted.
    assert [e["label"] for e in detail["episodes"]] == [1, 2]

    trace = catalog.episode_trace("local/smoke", 0)
    assert len(trace["t"]) == EPISODE_0_FRAMES

    assert catalog.video_path("local/smoke", "observation.images.top", 0, 0).exists()

    # `repo_id` arrives from a URL. The second spelling is inside the cache
    # until it is normalised, so only a check AFTER the join catches it.
    for bad in ("../../etc", "local/../../etc"):
        assert _refuses(bad), f"{bad!r} resolved instead of being refused"

    root = catalog.dataset_root("local/smoke")
    review.set_status(root, 1, review.REJECT, note="still arm",
                      episode_frames=detail["episode_frames"][1])
    reread = catalog.dataset_detail("local/smoke")
    assert reread["episodes"][1]["status"] == "reject"
    assert reread["keep_list"] == [0]

    # Appending episodes must NOT invalidate marks: `--resume` leaves every
    # existing index exactly where it was, and a warning on every recording
    # session trains people to ignore the warning that matters.
    frames = dict(reread["episode_frames"])
    frames[2] = 120
    marks = review.load(root)
    assert not review.is_stale(marks, {}, frames)

    # A prune renumbers the survivors, so the episode at a marked index is a
    # different one. Its LENGTH is the only evidence of that, per mark.
    pruned = {0: reread["episode_frames"][0], 1: 999}
    assert review.is_stale(marks, {}, pruned)
    assert review.stale_marks(marks, pruned) == [1]


def test_eval_split(monkeypatch, tmp_path):
    """The eval split must be a random SAMPLE, not the newest episodes.

    LeRobot holds out the tail of the episode list it is handed and never sorts
    it, and episode order is not neutral: operator skill improves across a
    session (Oscar's first 20: 3/10 kept in the first half, 7/10 in the second)
    and a resumed dataset ends with a different day's conditions. Holding out
    the last N therefore validates on the best, most recent demonstrations
    while training on the sloppiest — so the ORDER passed to LeRobot is
    shuffled instead, and the tail of that order is the holdout.
    """
    home = tmp_path / "split" / "lerobot"
    make_dataset(home / "local" / "many", n_episodes=10)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))

    d = catalog.dataset_detail("local/many")
    keep = d["keep_list"]

    recent = catalog.plan_eval_split(d["episodes"], keep, 0.2, 0, "recent")
    assert recent["eval"] == [8, 9]

    rnd = catalog.plan_eval_split(d["episodes"], keep, 0.2, 42, "random")
    assert len(rnd["eval"]) == 2
    assert rnd["eval"] != [8, 9]
    assert not (set(rnd["train"]) & set(rnd["eval"]))
    assert sorted(rnd["train"] + rnd["eval"]) == sorted(keep)
    assert sorted(rnd["order"]) == sorted(keep) and len(rnd["order"]) == len(keep)
    # The property that makes the shuffle work at all.
    assert sorted(rnd["order"][-len(rnd["eval"]):]) == rnd["eval"], \
        f"order={rnd['order']} eval={rnd['eval']}"

    same = catalog.plan_eval_split(d["episodes"], keep, 0.2, 42, "random")
    assert same["eval"] == rnd["eval"]
    draws = {tuple(catalog.plan_eval_split(d["episodes"], keep, 0.2, s, "random")["eval"])
             for s in range(8)}
    assert len(draws) > 1, f"{len(draws)} distinct"

    assert catalog.plan_eval_split(d["episodes"], keep, 0.0, 42, "random")["eval"] == []
