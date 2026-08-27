# hmi/backend/haller_hmi/lab/compare.py
"""Several runs' metric series, on one chart, downsampled before they are sent.

Backs `GET /lab/runs/metrics?ids&keys&max_points=600` — the "is this run better
than that one" view, where four training runs' `loss` and `eval_loss` are drawn
over each other. Every series is read through `runs.read_metrics`: stdlib
`json` over an append-only file, no pandas, no pyarrow, no lerobot. This module
therefore adds NOTHING to the serving process's import graph and needs no
main-thread warm-up of the kind `routes_datasets._warm_pandas` exists to do —
invariant 5c has no surface here because there is no heavy import to defer.

## The x-axis is `steps`, and a row without one is dropped

Rows are what the training runner appends: `{"kind": "train", "steps": N,
"loss": ..., ...}` straight out of `MetricsTracker.to_dict()`, `{"kind":
"eval", "steps": N, "eval_loss": ...}`, and `{"kind": "split",
"train_episodes": ..., "eval_episodes": ...}` — which carries no `steps` at
all.

X is that `steps` value and NEVER the row's position in the file. A run that
was stopped and resumed, or that logged every 200 steps and then every 10,
would otherwise draw evenly spaced points over unevenly spaced training: an
x-axis that is a lie. The `split` row is the concrete case rather than the
hypothetical one — one row without `steps` near the top of every training
file, and an ordinal x would shift every point after it.

So a row whose `steps` is missing, non-numeric or non-finite is SKIPPED, and
so is a point whose y is. Non-finite matters twice over: `json` round-trips
`NaN` happily, a diverged run is exactly what writes one, and `NaN` on the
wire is JSON that `JSON.parse` rejects — one bad row would take the whole
chart down rather than cost it a point.

Order is the order the file was written and is never re-sorted. A run resumed
from an earlier checkpoint really does go backwards in steps, and a line that
folds back on itself says precisely that; sorting would interleave two
training trajectories into one line that never happened.

## Largest-triangle-three-buckets, and never an average

The chart is ~600 px wide. A 100k-step run logging every 200 steps is 500
points and needs no downsampling at all; the same run logging every 10 is
10 000 points into 600 slots, and WHICH point survives each bucket is then the
whole question.

`catalog.episode_trace` refuses averaging for the reason this module inherits
("an average would smear exactly the short transitions") and strides instead,
which is right for it: its x is uniform 30 fps, its decimation is 3:1 on a 60 s
episode (1800 samples), and its series are joint angles — continuous, with no
one-sample feature to lose.

Neither holds here. A diverging loss is ONE logged step orders of magnitude
above its neighbours. Measured on this shape — 10 000 points, budget 600, a
single outlier walked through all 9 998 interior positions:

    stride (episode_trace's rule, stride 16)   keeps it at   624 / 9998  =  6 %
    LTTB                                        keeps it at  9998 / 9998  = 100 %

A stride chart therefore shows a run that trained smoothly and stopped, 94
times out of 100 — the most expensive wrong answer this view can give. LTTB
keeps, per bucket, the point forming the largest triangle with the previous
kept point and the next bucket's mean; an outlier is the vertex furthest from
that baseline, so it is the one that survives, and the first and last points
are kept by definition, so the chart cannot misreport where a run started or
finished either. Same refusal of averaging as the trace, one rung stronger
because the feature being preserved here is one sample wide.

## Caps, and the one thing a ValueError means

12 runs, 8 keys. A request naming every id in the store reads every
`metrics.jsonl` on disk on a single GET, which is an unbounded fan-out behind
one URL.

**`ValueError` out of `series` means a cap was exceeded and nothing else** —
that is what the route renders as 400. A run id that does not exist, has no
`metrics.jsonl`, is unreadable, or does not even name a legal run directory
reads as an EMPTY LIST for every key, never as an exception: this view is a
comparison, and one absent run must not take the other three down with it. A
requested key no row carries is `[]` too, and never a missing key, so the page
renders every requested series through one shape.
"""
from __future__ import annotations

import math

from . import runs as runs_mod

__all__ = [
    "MAX_KEYS",
    "MAX_POINTS",
    "MAX_RUNS",
    "X_KEY",
    "downsample",
    "series",
]

#: The row field every point's x comes from. See the module docstring: a row
#: without it is skipped rather than plotted at its ordinal.
X_KEY = "steps"

#: Runs per request. Twelve series is already more than a chart can be read
#: with; the cap exists for the fan-out, not for the legibility.
MAX_RUNS = 12

#: Metric keys per request. `loss`, `eval_loss`, `grad_norm`, `lr` and their
#: neighbours — eight is every series `MetricsTracker` reports at once.
MAX_KEYS = 8

#: Upper clamp on `max_points`, applied silently rather than refused. Past a
#: few thousand the extra points are invisible on a 600 px chart, and unlike
#: the two caps above a large budget costs response bytes only — the file has
#: already been read by then, so there is no fan-out to refuse.
MAX_POINTS = 10_000


def _finite(value: object) -> float | None:
    """A plottable number, or None for anything that is not one.

    `bool` is excluded explicitly because it is an `int` subclass: a row
    carrying `"steps": true` would otherwise plot at x = 1, silently, next to
    real steps.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def downsample(points: list[list[float]], max_points: int) -> list[list[float]]:
    """Largest-triangle-three-buckets, preserving spikes and both endpoints.

    Returns at most `max_points` of the ORIGINAL points — no point is
    synthesised, so every `[x, y]` on the chart is a step the trainer actually
    logged. `points[0]` and `points[-1]` are always in the result: an endpoint
    dropped by a downsampler misreports where the run started or finished, and
    that is a number the operator reads off the axis.

    Buckets are index-based, one per output slot, and within each the point
    kept is the one whose triangle with the previously kept point and the next
    bucket's mean has the largest area. A lone outlier is the vertex furthest
    from that baseline, so it is the point that survives — which is the whole
    reason this is not a stride (measured: keeps a one-sample spike 6 % of the
    time at 10 000 → 600, against LTTB's 9998/9998) and not an average
    (certainty of erasing it).

    `max_points < 2` is a `ValueError`: the two endpoints are non-negotiable,
    so a budget that cannot hold both is one this function cannot honour.
    """
    if int(max_points) < 2:
        raise ValueError(
            f"max_points must be at least 2, got {max_points} — the first and "
            "last points are always kept"
        )
    budget = min(int(max_points), MAX_POINTS)
    n = len(points)
    if n <= budget:
        return list(points)
    if budget == 2:
        return [points[0], points[n - 1]]

    every = (n - 2) / (budget - 2)
    kept = [points[0]]
    anchor = 0
    for i in range(budget - 2):
        # The NEXT bucket's mean is the triangle's third vertex. Clamped to the
        # end of the series: the last bucket's look-ahead runs past it.
        ahead_lo = int((i + 1) * every) + 1
        ahead_hi = min(int((i + 2) * every) + 1, n)
        ahead = points[ahead_lo:ahead_hi]
        if ahead:
            ahead_x = sum(p[0] for p in ahead) / len(ahead)
            ahead_y = sum(p[1] for p in ahead) / len(ahead)
        else:
            ahead_x, ahead_y = points[n - 1][0], points[n - 1][1]

        lo = int(i * every) + 1
        hi = min(int((i + 1) * every) + 1, n - 1)
        if lo >= hi:
            continue  # empty bucket: only reachable through float rounding
        anchor_x, anchor_y = points[anchor][0], points[anchor][1]
        best, best_area = lo, -1.0
        for j in range(lo, hi):
            px, py = points[j][0], points[j][1]
            area = abs((anchor_x - ahead_x) * (py - anchor_y)
                       - (anchor_x - px) * (ahead_y - anchor_y))
            if area > best_area:
                best_area, best = area, j
        kept.append(points[best])
        anchor = best

    kept.append(points[n - 1])
    return kept


def _rows(run_id: str) -> list[dict]:
    """Every metric row of one run, or `[]` for a run that cannot be read.

    `read_metrics` raises `ValueError` on an id that fails `RUN_ID_RE` or that
    escapes the run store, and `OSError` on a file that exists but cannot be
    opened. Both are swallowed HERE rather than propagated, because `series`
    reserves `ValueError` for the caps: letting one bad id out of a list of
    four turn into a 400 would be the absent run taking the chart down, and a
    traversal attempt would answer differently from a typo, which is a probe.
    """
    try:
        return runs_mod.read_metrics(run_id).get("rows") or []
    except (ValueError, OSError):
        return []


def _series_for(rows: list[dict], keys: list[str],
                max_points: int) -> dict[str, list[list[float]]]:
    """One pass over a run's rows, every requested key at once.

    A pass per key would re-test `steps` eight times on a 10 000 row file, and
    worse, would put the decision "is this row plottable" in eight places. A
    row has one x or none, decided once, here.
    """
    out: dict[str, list[list[float]]] = {key: [] for key in keys}
    for row in rows:
        if not isinstance(row, dict):
            continue
        x = _finite(row.get(X_KEY))
        if x is None:
            continue  # a `split` row, or anything else logged without a step
        for key, points in out.items():
            y = _finite(row.get(key))
            if y is not None:
                points.append([x, y])
    return {key: downsample(pts, max_points) for key, pts in out.items()}


def series(run_ids: list[str], keys: list[str],
           max_points: int = 600) -> dict:
    """Assemble `{"runs": {run_id: {key: [[x, y], ...]}}}` for one chart.

    Every requested id appears in `runs` and every requested key appears under
    it, whether or not there was anything to plot — the page draws a legend
    from what it asked for, so an absent run is an empty series and never a
    missing entry. Ids repeated in the request are read once.

    Raises `ValueError` ONLY for the caps (`MAX_RUNS`, `MAX_KEYS`, and a
    `max_points` below 2), which the route renders as 400.
    """
    ids = [str(r) for r in run_ids]
    wanted = [str(k) for k in keys]
    if len(ids) > MAX_RUNS:
        raise ValueError(
            f"too many runs: {len(ids)} — at most {MAX_RUNS} per request"
        )
    if len(wanted) > MAX_KEYS:
        raise ValueError(
            f"too many keys: {len(wanted)} — at most {MAX_KEYS} per request"
        )
    if int(max_points) < 2:
        raise ValueError(
            f"max_points must be at least 2, got {max_points} — the first and "
            "last points are always kept"
        )

    out: dict[str, dict[str, list[list[float]]]] = {}
    for run_id in ids:
        if run_id in out:
            continue
        if not wanted:
            # No key can produce a point, so every read would be a file opened
            # to answer nothing — on up to twelve runs.
            out[run_id] = {}
            continue
        out[run_id] = _series_for(_rows(run_id), wanted, max_points)
    return {"runs": out}
