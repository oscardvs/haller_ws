# hmi/backend/tests/lab/test_split.py
"""`plan_eval_split` — the properties whose loss is invisible.

The ported smoke test (`test_smoke_dataui.py`) carries the kit's ten split
assertions and this file does not repeat them. What it adds is the set of
properties that a tidy-up would break WITHOUT breaking any of those: the
returned `order` really is unsorted, the tail is held out per TASK rather than
across the dataset, the holdout rounds up, and a rejected episode is gone from
every one of the three lists.

The first one is the whole reason this function lives in its own module.
LeRobot holds out the tail of the list it is handed and never sorts it, so the
shuffle IS the split. Sorting `order` still yields a train/eval split of the
right size — it is just the chronological one, validating on the best and most
recent demonstrations while training on the sloppiest, and nothing downstream
can tell. So it is asserted directly.

Episode dicts are built inline: the function reads `episode_index` and `tasks`
and nothing else, and building a parquet tree to supply two keys would only
put a reader between the test and the thing being tested.
"""
from __future__ import annotations

import pytest

from haller_hmi.lab.split import plan_eval_split

#: A seed whose shuffle demonstrably moves episodes off their chronological
#: positions on the ten-episode dataset below. A test for "not sorted" is
#: worthless on a draw that happens to come back sorted.
MOVING_SEED = 42


def _episodes(n: int, task: str = "Test task") -> list[dict]:
    return [{"episode_index": i, "tasks": [task]} for i in range(n)]


# ---- the order must not be sorted ----

def test_the_random_order_is_not_sorted():
    """The property whose loss is invisible: `order` is what LeRobot receives,
    and a sorted `order` is a chronological split wearing a random one's name.

    `train` and `eval` next to it ARE sorted, deliberately — they are reports
    for the UI. That asymmetry is exactly what makes sorting `order` look like
    a cleanup.
    """
    episodes = _episodes(10)
    out = plan_eval_split(episodes, list(range(10)), 0.2, MOVING_SEED, "random")

    assert out["order"] != sorted(out["order"])
    assert sorted(out["order"]) == list(range(10)), "still a permutation of the kept set"
    assert out["train"] == sorted(out["train"])
    assert out["eval"] == sorted(out["eval"])


def test_a_random_order_holds_out_something_other_than_the_newest():
    episodes = _episodes(10)

    out = plan_eval_split(episodes, list(range(10)), 0.2, MOVING_SEED, "random")

    assert out["eval"] != [8, 9]
    assert len(out["eval"]) == 2


def test_recent_mode_keeps_the_chronological_order_it_was_given():
    """`mode="recent"` exists for when validating on the newest conditions IS
    the intent, so its order is the input order untouched."""
    episodes = _episodes(10)

    out = plan_eval_split(episodes, list(range(10)), 0.2, MOVING_SEED, "recent")

    assert out["order"] == list(range(10))
    assert out["eval"] == [8, 9]


def test_an_unknown_mode_does_not_shuffle():
    """Anything that is not `random` leaves the order alone rather than
    silently picking a shuffle the caller did not ask for."""
    episodes = _episodes(10)

    out = plan_eval_split(episodes, list(range(10)), 0.2, MOVING_SEED, "chronological")

    assert out["order"] == list(range(10))


# ---- grouping is by task ----

def test_each_task_holds_out_its_own_tail():
    """LeRobot groups the list by task before slicing, so a two-task dataset
    gets two holdouts. Slicing the tail of the whole list instead would take
    every eval episode from whichever task happens to be last, and validate a
    multi-task policy on one task.
    """
    episodes = [{"episode_index": i, "tasks": ["pick" if i < 6 else "place"]}
                for i in range(10)]

    out = plan_eval_split(episodes, list(range(10)), 0.5, 0, "recent")

    assert out["eval"] == [3, 4, 5, 8, 9]
    assert out["train"] == [0, 1, 2, 6, 7]


def test_each_task_holds_out_its_own_tail_of_the_shuffled_order():
    """The same grouping, after the shuffle: each task's eval set is the tail
    of THAT task's subsequence of `order`, not of `order` itself."""
    episodes = [{"episode_index": i, "tasks": ["pick" if i < 6 else "place"]}
                for i in range(10)]

    out = plan_eval_split(episodes, list(range(10)), 0.5, MOVING_SEED, "random")
    held = set(out["eval"])

    for task, size, n_eval in (("pick", 6, 3), ("place", 4, 2)):
        members = [e["episode_index"] for e in episodes if e["tasks"][0] == task]
        assert len(members) == size
        group = [ep for ep in out["order"] if ep in members]
        assert set(group[-n_eval:]) == held & set(members)


def test_an_episode_with_no_task_is_grouped_with_the_other_untasked_ones():
    """`tasks` missing or empty groups under `""` rather than raising: a
    hand-written or partially migrated meta must not fail a split."""
    episodes = [{"episode_index": i} for i in range(4)]

    out = plan_eval_split(episodes, list(range(4)), 0.5, 0, "recent")

    assert out["eval"] == [2, 3]


# ---- ceil, not floor ----

def test_the_holdout_rounds_up_so_a_small_dataset_still_gets_an_eval_episode():
    """`floor(3 * 0.2)` is 0, and an eval split that silently held out nothing
    would report a training run with no validation curve as configured."""
    out = plan_eval_split(_episodes(3), [0, 1, 2], 0.2, 0, "recent")

    assert out["eval"] == [2]
    assert out["train"] == [0, 1]


@pytest.mark.parametrize("n,eval_split,expected", [
    pytest.param(3, 0.2, 1, id="3-at-0.2-rounds-up"),
    pytest.param(10, 0.2, 2, id="10-at-0.2-is-exact"),
    pytest.param(11, 0.2, 3, id="11-at-0.2-rounds-up"),
    pytest.param(4, 0.5, 2, id="half"),
])
def test_the_holdout_size_is_ceil_of_the_group_size(n, eval_split, expected):
    out = plan_eval_split(_episodes(n), list(range(n)), eval_split, 0, "recent")

    assert len(out["eval"]) == expected


def test_an_eval_split_of_zero_holds_out_nothing():
    out = plan_eval_split(_episodes(10), list(range(10)), 0.0, MOVING_SEED, "random")

    assert out["eval"] == []
    assert sorted(out["train"]) == list(range(10))


# ---- rejected episodes ----

def test_a_rejected_episode_never_appears_in_order_train_or_eval():
    """`keep_list` is the only gate: an episode the operator rejected must not
    reach LeRobot at all, in any of the three lists."""
    episodes = _episodes(10)
    keep = [i for i in range(10) if i != 4]

    out = plan_eval_split(episodes, keep, 0.2, MOVING_SEED, "random")

    assert 4 not in out["order"]
    assert 4 not in out["train"]
    assert 4 not in out["eval"]
    assert sorted(out["order"]) == keep
    assert sorted(out["train"] + out["eval"]) == keep


def test_rejecting_everything_leaves_an_empty_plan():
    out = plan_eval_split(_episodes(4), [], 0.2, MOVING_SEED, "random")

    assert out["order"] == []
    assert out["train"] == []
    assert out["eval"] == []


def test_a_keep_list_naming_an_episode_the_dataset_does_not_have_is_ignored():
    """The kept set is intersected with the episodes actually present, so a
    review file written before a prune cannot inject a phantom index into the
    list LeRobot is handed."""
    out = plan_eval_split(_episodes(3), [0, 1, 2, 99], 0.5, 0, "recent")

    assert 99 not in out["order"]
    assert sorted(out["order"]) == [0, 1, 2]


# ---- the plan reports what it was asked for ----

def test_the_plan_echoes_the_mode_seed_and_split_it_used():
    """The run spec summary carries these, and a split whose parameters were
    not recorded cannot be reproduced from the run record."""
    out = plan_eval_split(_episodes(10), list(range(10)), 0.25, 7, "random")

    assert out["mode"] == "random"
    assert out["seed"] == 7
    assert out["eval_split"] == 0.25
    assert type(out["seed"]) is int
