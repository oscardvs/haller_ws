# hmi/backend/tests/lab/test_compare.py
"""`lab/compare.py` — the four ways a comparison chart lies.

The chart this module feeds is read to answer "did that run diverge, and did
this one beat it". Four failures make it answer wrong while still drawing:

* an **invented x-axis** — a row plotted at its position in the file rather
  than at its `steps`, so a resumed run or a changed log interval draws evenly
  spaced points over unevenly spaced training;
* a **swallowed spike** — the one logged step where the loss blew up, dropped
  by the downsampler, leaving a chart that shows a smooth run that stopped;
* **lost endpoints** — a first or last point dropped, so the axis misreports
  where the run started or finished;
* **one absent run taking the rest down** — an id that no longer exists
  answering with an exception instead of an empty series.

Every test writes `metrics.jsonl` by hand into `tmp_path` with `HALLER_RUNS`
pointed at it. `runs.launch` is deliberately not used: this module only ever
READS those files, and a real child would make the tests depend on a runner
that does not exist yet to prove something about parsing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from haller_hmi.lab import compare, runs


@pytest.fixture()
def store(tmp_path, monkeypatch) -> Path:
    base = tmp_path / "runs"
    base.mkdir()
    monkeypatch.setenv(runs.RUNS_DIR_ENV, str(base))
    return base


def _write(store: Path, run_id: str, rows: list[dict]) -> str:
    """A run directory holding nothing but the metrics file this module reads.

    No `run.json`: `compare` never calls `runs.load`, and a fixture that wrote
    one would hide it if it started to.
    """
    rdir = store / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "metrics.jsonl", "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return run_id


def _train(steps: int, **metrics) -> dict:
    """A row shaped like the one `MetricsTracker.to_dict()` produces."""
    return {"kind": "train", "steps": steps, "samples": steps * 8, **metrics}


# ---- assembly ----

def test_two_runs_one_key_come_back_under_their_own_ids(store):
    _write(store, "train-a", [_train(0, loss=1.0), _train(100, loss=0.5)])
    _write(store, "train-b", [_train(0, loss=2.0), _train(100, loss=1.5)])

    out = compare.series(["train-a", "train-b"], ["loss"])

    assert out == {"runs": {
        "train-a": {"loss": [[0.0, 1.0], [100.0, 0.5]]},
        "train-b": {"loss": [[0.0, 2.0], [100.0, 1.5]]},
    }}


def test_a_key_one_run_never_logged_is_an_empty_list_not_a_missing_key(store):
    """The page draws its legend from what it ASKED for. A key that silently
    vanishes for one run leaves a series with no entry and no explanation;
    an empty list says "this run has no eval loss" in the shape every other
    series already renders through."""
    _write(store, "train-a", [
        _train(0, loss=1.0),
        {"kind": "eval", "steps": 0, "eval_loss": 0.9},
    ])
    _write(store, "train-b", [_train(0, loss=2.0)])

    out = compare.series(["train-a", "train-b"], ["loss", "eval_loss"])

    assert out["runs"]["train-a"]["eval_loss"] == [[0.0, 0.9]]
    assert out["runs"]["train-b"]["eval_loss"] == []
    assert set(out["runs"]["train-b"]) == {"loss", "eval_loss"}


def test_train_and_eval_rows_split_by_key_alone_never_by_kind(store):
    """`loss` and `eval_loss` are two series over ONE file, and nothing asks
    which `kind` a row was. A row carries the key or it does not — which is
    what keeps the caller from having to name a kind it would then have to
    keep in sync with the runner."""
    _write(store, "train-a", [
        _train(0, loss=1.0),
        {"kind": "eval", "steps": 0, "eval_loss": 0.9},
        _train(100, loss=0.5),
        {"kind": "eval", "steps": 100, "eval_loss": 0.4},
    ])

    out = compare.series(["train-a"], ["loss", "eval_loss"])

    assert out["runs"]["train-a"]["loss"] == [[0.0, 1.0], [100.0, 0.5]]
    assert out["runs"]["train-a"]["eval_loss"] == [[0.0, 0.9], [100.0, 0.4]]


def test_an_unknown_run_is_an_empty_series_and_takes_nobody_with_it(store):
    """The whole point of the view is the comparison; a run deleted between
    the page loading its list and the chart asking for its metrics must not
    blank the other three."""
    _write(store, "train-a", [_train(0, loss=1.0)])

    out = compare.series(["train-a", "train-gone"], ["loss"])

    assert out["runs"]["train-gone"] == {"loss": []}
    assert out["runs"]["train-a"]["loss"] == [[0.0, 1.0]]


def test_a_run_directory_with_no_metrics_file_is_an_empty_series(store):
    """A rollout or an export run has no `metrics.jsonl` at all, and one can be
    dragged onto the chart beside a training run."""
    (store / "rollout-x").mkdir()

    out = compare.series(["rollout-x"], ["loss"])

    assert out["runs"]["rollout-x"] == {"loss": []}


@pytest.mark.parametrize("run_id", ["..", "../elsewhere", "train a", ""])
def test_an_id_that_cannot_name_a_run_reads_as_absent_not_as_an_error(store, run_id):
    """`runs.run_dir` refuses these — `..` names the parent of the store and
    the others fail `RUN_ID_RE`. That refusal stays; what must not happen is it
    reaching the route, where the ladder would turn it into a 400 and one bad
    id in a list of four would take the chart down. A traversal attempt also
    has to answer exactly like a typo, or the difference is a probe."""
    out = compare.series([run_id], ["loss"])

    assert out["runs"][run_id] == {"loss": []}
    with pytest.raises(ValueError):
        runs.run_dir(run_id)  # the refusal itself is still in place


def test_the_same_id_twice_is_read_once(store, monkeypatch):
    _write(store, "train-a", [_train(0, loss=1.0)])
    reads = []
    real = runs.read_metrics
    monkeypatch.setattr(
        runs, "read_metrics",
        lambda run_id, *a, **k: (reads.append(run_id), real(run_id, *a, **k))[1])

    out = compare.series(["train-a", "train-a"], ["loss"])

    assert reads == ["train-a"]
    assert out["runs"]["train-a"]["loss"] == [[0.0, 1.0]]


def test_no_key_asked_for_reads_no_file(store, monkeypatch):
    """Twelve files opened to answer nothing."""
    _write(store, "train-a", [_train(0, loss=1.0)])
    monkeypatch.setattr(
        runs, "read_metrics",
        lambda *a, **k: pytest.fail("read a metrics file for zero keys"))

    assert compare.series(["train-a"], []) == {"runs": {"train-a": {}}}


# ---- the x-axis ----

def test_rows_without_steps_are_skipped_not_plotted_at_their_position(store):
    """`{"kind": "split", ...}` is a real row the training runner writes near
    the top of every file and it carries no `steps`. Plotting rows by ordinal
    would put it on the chart AND shift every point after it."""
    _write(store, "train-a", [
        {"kind": "split", "train_episodes": 40, "eval_episodes": 6, "loss": 9.9},
        _train(0, loss=1.0),
        {"kind": "train", "loss": 0.7},                 # tracker row, no steps
        _train(200, loss=0.5),
    ])

    out = compare.series(["train-a"], ["loss"])

    assert out["runs"]["train-a"]["loss"] == [[0.0, 1.0], [200.0, 0.5]]


def test_x_is_the_step_count_and_never_the_row_number(store):
    """A run that logged every 200 steps and then every 10 — the shape a
    changed log interval or a resume produces. An ordinal x would draw those
    six points evenly spaced and the chart would say the loss fell at a
    constant rate over the whole run."""
    _write(store, "train-a", [_train(s, loss=1.0) for s in
                              (0, 200, 400, 410, 420, 430)])

    xs = [x for x, _ in compare.series(["train-a"], ["loss"])["runs"]
          ["train-a"]["loss"]]

    assert xs == [0.0, 200.0, 400.0, 410.0, 420.0, 430.0]


def test_a_resumed_run_folds_back_rather_than_being_sorted(store):
    """Resuming from an earlier checkpoint really does go backwards in steps.
    Sorting would interleave two training trajectories into one line that
    never happened; file order draws the fold, which is the truth."""
    _write(store, "train-a", [_train(s, loss=v) for s, v in
                              [(0, 1.0), (100, 0.6), (200, 0.4),
                               (100, 0.55), (200, 0.3)]])

    out = compare.series(["train-a"], ["loss"])

    assert out["runs"]["train-a"]["loss"] == [
        [0.0, 1.0], [100.0, 0.6], [200.0, 0.4], [100.0, 0.55], [200.0, 0.3]]


@pytest.mark.parametrize("bad", [None, "1200", True, [1], {"a": 1}])
def test_a_steps_that_is_not_a_number_is_not_an_x(store, bad):
    """`True` is the one that has to be excluded by hand: `bool` is an `int`
    subclass, so a row carrying it would otherwise plot at x = 1 among real
    steps."""
    _write(store, "train-a", [{"kind": "train", "steps": bad, "loss": 0.1},
                              _train(10, loss=0.2)])

    out = compare.series(["train-a"], ["loss"])

    assert out["runs"]["train-a"]["loss"] == [[10.0, 0.2]]


def test_a_non_finite_value_is_dropped_rather_than_sent(store):
    """`json` round-trips `NaN` in both directions and a diverged run is
    exactly what writes one — but `NaN` on the wire is JSON that `JSON.parse`
    rejects, so one bad row would take the whole chart down instead of costing
    it a point. The rise before the divergence still draws."""
    path = store / "train-a"
    path.mkdir()
    (path / "metrics.jsonl").write_text(
        '{"kind": "train", "steps": 0, "loss": 1.0}\n'
        '{"kind": "train", "steps": 100, "loss": NaN}\n'
        '{"kind": "train", "steps": 200, "loss": Infinity}\n'
        '{"kind": "train", "steps": NaN, "loss": 0.3}\n'
        '{"kind": "train", "steps": 300, "loss": 0.2}\n'
    )

    out = compare.series(["train-a"], ["loss"])

    assert out["runs"]["train-a"]["loss"] == [[0.0, 1.0], [300.0, 0.2]]
    assert all(math.isfinite(y) for _, y in out["runs"]["train-a"]["loss"])


# ---- downsampling ----

@pytest.mark.parametrize("n", [0, 1, 2, 3, 599, 600, 601, 1000, 5000])
@pytest.mark.parametrize("budget", [2, 3, 7, 600])
def test_the_downsample_never_exceeds_max_points(n, budget):
    points = [[float(i), math.sin(i / 13.0)] for i in range(n)]

    out = compare.downsample(points, budget)

    assert len(out) <= budget
    assert len(out) <= n  # never padded either, so a short series stays short


@pytest.mark.parametrize("n", [3, 601, 1000, 10_000])
def test_the_first_and_last_points_survive(n):
    """A chart that loses its endpoints misreports where the run started and
    where it finished — both numbers the operator reads straight off the
    axis."""
    points = [[float(i), math.sin(i / 13.0)] for i in range(n)]

    out = compare.downsample(points, 600)

    assert out[0] == points[0]
    assert out[-1] == points[-1]


def test_the_endpoints_survive_through_the_route_shape_too(store):
    """The property has to hold on the assembled payload, not only on the
    helper: `series` is what the route calls."""
    rows = [_train(s * 10, loss=1.0 / (s + 1)) for s in range(5000)]
    _write(store, "train-a", rows)

    out = compare.series(["train-a"], ["loss"], max_points=600)["runs"]["train-a"]

    assert len(out["loss"]) <= 600
    assert out["loss"][0] == [0.0, 1.0]
    assert out["loss"][-1] == [49990.0, 1.0 / 5000]


def test_a_spike_survives_the_downsample(store):
    """The property an averaging downsample loses and a stride loses 94 times
    in 100 (measured: 624 of 9998 positions kept at 10 000 -> 600). One logged
    step orders of magnitude above its neighbours IS the divergence, and a
    chart that smooths it shows a run that trained cleanly and stopped."""
    rows = [_train(s * 10, loss=0.5) for s in range(5000)]
    rows[1234] = _train(12340, loss=900.0)
    _write(store, "train-a", rows)

    points = compare.series(["train-a"], ["loss"])["runs"]["train-a"]["loss"]

    assert [12340.0, 900.0] in points


@pytest.mark.parametrize("spike_at", [1, 7, 499, 2500, 4321, 4998])
def test_the_spike_survives_wherever_it_falls_in_the_series(spike_at):
    """Once, at one index, would pass on a downsampler that happens to keep
    that bucket's maximum. The spike is walked across the series instead."""
    points = [[float(i), 0.5] for i in range(5000)]
    points[spike_at] = [float(spike_at), 900.0]

    out = compare.downsample(points, 600)

    assert [float(spike_at), 900.0] in out


def test_downsampling_never_synthesises_a_point():
    """Every `[x, y]` on the chart is a step the trainer actually logged. An
    averaged point would carry an x no row ever had, so hovering it would name
    a step that does not exist."""
    points = [[float(i), math.sin(i / 7.0)] for i in range(5000)]

    out = compare.downsample(points, 600)

    originals = {id(p) for p in points}
    assert all(id(p) in originals for p in out)


def test_a_series_that_fits_is_returned_unchanged(store):
    """A 100k-step run logging every 200 steps is 500 points into a 600 px
    chart. Downsampling it at all would cost fidelity for nothing."""
    points = [[float(i), float(i)] for i in range(500)]

    assert compare.downsample(points, 600) == points


def test_max_points_is_clamped_rather_than_refused_from_above():
    """Unlike the two caps, a large budget costs response bytes only — the
    file has already been read by then — so there is no fan-out to refuse."""
    points = [[float(i), 0.5] for i in range(compare.MAX_POINTS + 500)]

    out = compare.downsample(points, 10_000_000)

    assert len(out) == compare.MAX_POINTS


# ---- the caps ----

def test_too_many_runs_raises_value_error(store):
    """An unbounded fan-out reads every `metrics.jsonl` in the store on one
    GET."""
    ids = [f"train-{i}" for i in range(compare.MAX_RUNS + 1)]

    with pytest.raises(ValueError, match="too many runs"):
        compare.series(ids, ["loss"])

    assert "runs" in compare.series(ids[:compare.MAX_RUNS], ["loss"])


def test_too_many_keys_raises_value_error(store):
    keys = [f"k{i}" for i in range(compare.MAX_KEYS + 1)]

    with pytest.raises(ValueError, match="too many keys"):
        compare.series(["train-a"], keys)

    assert "runs" in compare.series(["train-a"], keys[:compare.MAX_KEYS])


@pytest.mark.parametrize("budget", [1, 0, -5])
def test_a_budget_that_cannot_hold_both_endpoints_is_refused(store, budget):
    """The first and last points are non-negotiable, so a budget below two is
    one this module cannot honour — refused rather than silently answered with
    a single point that misreports one end of the run."""
    with pytest.raises(ValueError, match="max_points"):
        compare.downsample([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], budget)
    with pytest.raises(ValueError, match="max_points"):
        compare.series(["train-a"], ["loss"], max_points=budget)


def test_the_caps_are_the_only_value_error_series_raises(store):
    """The route renders `ValueError` as 400. Anything else that can go wrong
    in here — a missing run, an unreadable file, an id that escapes the store —
    is an empty series, so a 400 from this route always means "you asked for
    too much" and never "one of your ids was odd"."""
    _write(store, "train-a", [_train(0, loss=1.0)])
    (store / "train-b").mkdir()

    out = compare.series(
        ["train-a", "train-b", "train-gone", "..", "no/such"],
        ["loss", "eval_loss"],
    )

    assert len(out["runs"]) == 5
    assert out["runs"]["train-a"]["loss"] == [[0.0, 1.0]]
    for run_id in ("train-b", "train-gone", "..", "no/such"):
        assert out["runs"][run_id] == {"loss": [], "eval_loss": []}


# ---- the process this runs in ----

def test_compare_imports_nothing_heavy():
    """`lab/**` runs in the serving process, which owns the Feetech bus. This
    module reads JSONL with the stdlib, so unlike `routes_datasets` it has no
    deferred heavy import to warm on the main thread (invariant 5c) and no
    lerobot/torch to keep out of the latency path."""
    source = Path(compare.__file__).read_text()

    for banned in ("import lerobot", "import torch", "import pandas",
                   "import numpy", "import pyarrow"):
        assert banned not in source
