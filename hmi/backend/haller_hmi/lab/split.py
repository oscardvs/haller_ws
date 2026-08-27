# hmi/backend/haller_hmi/lab/split.py
"""Which kept episodes LeRobot holds out for eval loss.

This is one function in its own module because the thing it does is not
visible in its output, and burying it in `catalog.py` next to the parquet
readers invites exactly the tidy-up that breaks it.

The trick, in full:
    LeRobot's `make_train_eval_datasets` groups the episode list it is
    handed BY TASK and holds out the LAST `ceil(n * eval_split)` of each
    group. It NEVER sorts that list — `LeRobotDataset` stores
    `self.episodes = episodes` exactly as given. So the ORDER passed in
    `--dataset.episodes` is what decides the split, and shuffling it here
    with `random.Random(seed)` randomises the holdout with NO LeRobot
    patch at all.

    **Any code that sorts, dedupes or set-ifies that order silently
    destroys the split.** The failure is invisible: you still get a
    train/eval split of the right size, it is just the wrong one, and
    nothing downstream can tell.

Why a chronological holdout is the wrong one: operator skill improves
across a session — Oscar's first 20 episodes kept 3 of the first 10 and 7
of the second 10 — and a resumed dataset ends with a different day's
lighting and object placement. Holding out the last N therefore validates
on the best, most recent demonstrations while training on the sloppiest.
The two halves are not samples from the same distribution, and the eval
curve stops meaning what it looks like it means.

`mode="recent"` keeps the chronological behaviour, for when validating on
the newest conditions IS the intent.

pandas/pyarrow-free and numpy-free by construction: this module runs in
the serving process, which is the teleop latency path.
"""
from __future__ import annotations

import math
import random


def plan_eval_split(
    episodes: list[dict],
    keep_list: list[int],
    eval_split: float,
    seed: int = 42,
    mode: str = "random",
) -> dict:
    """Decide which kept episodes are held out for eval loss.

    `episodes` is `catalog.dataset_detail`'s episode list; only
    `episode_index` and `tasks` are read. Returns the episode ORDER to
    pass to LeRobot, plus the train and eval sets that order will produce.

    See the module docstring: the returned `order` is the load-bearing
    value and it must stay unsorted.
    """
    kept = set(keep_list)
    order = [e["episode_index"] for e in episodes if e["episode_index"] in kept]
    if mode == "random":
        random.Random(int(seed)).shuffle(order)

    # Mirror factory.py exactly: group by task in the order given, then
    # slice each group's tail.
    task_of = {e["episode_index"]: (e["tasks"][0] if e.get("tasks") else "") for e in episodes}
    groups: dict[str, list[int]] = {}
    for ep in order:
        groups.setdefault(task_of.get(ep, ""), []).append(ep)

    train, val = [], []
    for eps in groups.values():
        n_eval = math.ceil(len(eps) * eval_split) if eval_split > 0 else 0
        train.extend(eps[: len(eps) - n_eval])
        val.extend(eps[len(eps) - n_eval:])

    # `train` and `eval` are sorted and `order` is not, and that asymmetry
    # is deliberate: these two are REPORTS, for the UI and the run spec
    # summary. `order` is the list LeRobot actually receives. Sorting it to
    # match would look like a cleanup and would throw the split away.
    return {
        "order": order,
        "train": sorted(train),
        "eval": sorted(val),
        "mode": mode,
        "seed": int(seed),
        "eval_split": eval_split,
    }
