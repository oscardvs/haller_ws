# hmi/backend/haller_hmi/lab/routes_runs.py
"""`/lab/runs/**` — start a training job, watch it, stop it, delete it.

Four properties shape every handler below, and none of them is a preference.

* **Every handler is a plain `def`, never `async def`.** FastAPI runs plain
  defs on an anyio worker thread, so `stop()`'s 20-second SIGINT grace and
  `train`'s parquet read can never stall the event loop that forwards teleop
  frames to the arms. `routes_datasets.py` applies the same rule for the same
  reason; it matters here twice over, because two of these handlers block on
  purpose.

* **`_warm_pandas()` runs at ROUTER-BUILD time**, on the main thread. `POST
  /lab/runs/train` validates its spec through `catalog.dataset_detail`, which
  is the process's first `import pandas` on a cold server — and a first pandas
  import on a worker thread segfaults the next parquet read from a DIFFERENT
  worker (invariant 5c, reproduced 3/3 on this box). SIGSEGV runs no `finally`,
  no `atexit` and no `_release_torque_per_motor`, and `/estop` is unreachable
  because the server is gone: the arms stay energised on their last goal until
  someone reaches the bench PSU. This module warms it even though
  `routes_datasets` also does, because either router must be mountable alone —
  a build order is not a safety mechanism.

* **`lab/runs.py`'s records are NOT the wire's.** `load()` returns `alive`,
  `pid`, `cwd`, `log_size`, `metrics_size`, `output_dir` and `runner_python`
  alongside what Track C froze. Trimming happens in exactly one function,
  `_run_wire`, the way `routes_datasets._episode_wire` owns its rename, so no
  route can invent a second spelling of a run.

* **The COLLECTION `/lab/runs/metrics` is declared before the `{run_id}`
  routes**, and the comment at that route says so too. Starlette matches in
  declaration order, `metrics` satisfies `RUN_ID_RE`, and the failure of
  getting it wrong is a cross-run chart answering `404 no run metrics`.

`require_local` gates EXACTLY four routes — `POST /lab/runs/train`, `POST
/lab/runs/rollout`, `POST /lab/runs/{id}/stop`, `DELETE /lab/runs/{id}` — and
no GET. `--host 0.0.0.0` is
how the Quest reaches the HMI; reaching it must not also mean launching an
hours-long GPU job or killing one. Watching a run from the headset is exactly
what the LAN is for.

## What this module does not launch

`train` and `rollout`. There is no `record` kind anywhere in this port —
recording owns the Feetech bus and the bus stays in the serving process
(`lab/runs.py` carries the 2026-08-21 incident that closed that path).

**`rollout` is a `/lab/runs` route like any other and NOT a bus handover.** The
contract's rollout addendum rules that the detached child loads the checkpoint
and runs INFERENCE ONLY, streaming target joint angles to the server, which
keeps the bus and commits them through the same LPF → rate cap → clamp →
collision guard → floors → E-STOP chain every other input goes through. So
nothing here ever hands a resource to a child, and `stop` never escalates past
SIGTERM.

The rollout launcher carries the one check neither side can do alone: the
DECLARED control rate against the rate the policy was TRAINED at. The child is
handed `control_hz` and never opens `info.json`, so it can record both numbers
but cannot compare them. See `post_rollout`.
"""
from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from ..api.deps import LabDeps
from ..api.errors import as_http
from ..api.gate import build_require_local
from . import catalog, compare
from . import runs as runs_mod

#: How deep a listing reads the run store before filtering. `list_runs` loads
#: EVERY run directory whatever number it is handed — the argument only trims
#: the sorted result — so this costs nothing extra and buys the filter its
#: correctness: filtering a pre-trimmed head would answer "no train runs" for a
#: store whose hundred newest runs happen to be exports.
RUN_SCAN = 1000

#: Rows one listing answers with, after filtering. More than the table shows at
#: once; a store deeper than this is one nobody has cleaned in months.
RUN_LIST_LIMIT = 200

#: `max_points` for the cross-run chart when the caller does not say. The
#: frozen default; `lab/compare.py` clamps the upper end.
DEFAULT_MAX_POINTS = 600

# Spec defaults. These mirror `runners/train_runner.build_argv`'s fallbacks on
# purpose rather than by omission: `spec.json` is the run's own record of what
# it was asked for, and a spec that left fields out would describe itself
# differently depending on which runner version happened to launch it.
DEFAULT_POLICY_TYPE = "act"
DEFAULT_DEVICE = "cuda"
DEFAULT_STEPS = 100_000
DEFAULT_BATCH_SIZE = 8
DEFAULT_SAVE_FREQ = 20_000
DEFAULT_NUM_WORKERS = 4
DEFAULT_EVAL_SEED = 42
DEFAULT_EVAL_MODE = "random"

#: How long a rollout runs when the caller does not say. A bounded default
#: rather than "until stopped", because this loop moves a real arm from a
#: browser button.
#:
#: The CEILING is deliberately NOT mirrored here. `rollout_runner` owns it
#: (`MAX_DURATION_S`) and refuses past it, and `lab/` must not import
#: `runners/` — so a copy of that number in this module would be a second
#: fact that must agree with the first and would eventually not. The cost is
#: that an over-long duration dies in the child instead of arriving as a 400.
DEFAULT_ROLLOUT_DURATION_S = 60.0

#: Points the loss chart should end up with whatever the run's length, which is
#: what `log_freq` is scaled to. LeRobot's fixed default of 200 means a 40-step
#: debug run logs nothing at all and the chart stays blank the whole way
#: through. The kit's rule, kept.
LOG_POINTS = 200

#: A required JSON object body, spelled through `Annotated` because a call in a
#: default argument is a lint error that ruff exempts `Query` and `Depends`
#: from and `Body` not at all. A `dict` and not a model, for
#: `routes_datasets.py`'s reason: the frozen error contract is "400 bad input"
#: and a pydantic model answers a missing field with 422.
JsonBody = Annotated[dict, Body()]

#: A query parameter that may be repeated (`?ids=a&ids=b`). Spelled through
#: `Annotated` rather than as a `Query(default=None)` default because ruff's
#: B008 exemption for FastAPI does not cover a `list[...]` annotation — the
#: same reason `JsonBody` above is written this way.
RepeatableQuery = Annotated[list[str] | None, Query()]


def _warm_pandas() -> None:
    """Import pandas HERE, on the thread that builds the router.

    See the module docstring and `routes_datasets._warm_pandas`, which carries
    the measurement: a first `import pandas` on an anyio worker followed by a
    read from a second worker is a SIGSEGV in the process that owns the arms.
    `POST /lab/runs/train` reaches pandas through `catalog.dataset_detail`, so
    this router has the same window and closes it the same way.

    `ImportError` is swallowed on purpose: a serving venv without pandas must
    still mount and answer 503 per request through `catalog._pandas`, rather
    than failing to start.
    """
    try:
        import pandas  # noqa: F401 - imported for its side effect, see above
    except ImportError:  # pragma: no cover - a broken serving venv
        pass


# ---- lab/runs.py's records -> the frozen wire shapes ----

def _run_wire(record: dict, *, detail: bool) -> dict:
    """One run, in Track C's spelling. The only place either shape is written.

    Two shapes, one function, because the two overlap on six fields and a
    second function would be a second chance to spell `started_at` or a status
    differently. `detail=False` is a listing row and `detail=True` the single-run
    view; neither is a subset of the other:

    * a row carries `tags` and `spec_summary` — what the table renders, and the
      only place a run remembers what it was asked for once its spec has been
      superseded;
    * the detail carries `spec`, `argv`, `exit_code` and `error`, which are
      per-run and would be fifty full training specs on a listing nobody reads
      them from.

    Everything `load()` adds beyond those is dropped: `alive` (already resolved
    into `status`), `pid`, `cwd`, `runner_python`, `log_size`, `metrics_size`
    and `output_dir` are the server's own bookkeeping, and a page that started
    reading them would be depending on a shape nobody froze.
    """
    wire = {
        "id": record["id"],
        "kind": record.get("kind", ""),
        "name": record.get("name", ""),
        "status": record.get("status", ""),
        "started_at": record.get("started_at"),
        # Absent while a run is alive: `load()` only fills it from the
        # `result.json` the runner writes in its `finally`.
        "finished_at": record.get("finished_at"),
    }
    if not detail:
        wire["tags"] = record.get("tags") or []
        wire["spec_summary"] = record.get("spec_summary") or ""
        return wire
    wire["spec"] = record.get("spec") or {}
    wire["argv"] = record.get("argv") or []
    wire["exit_code"] = record.get("exit_code")
    wire["error"] = record.get("error") or ""
    return wire


def _checkpoint_wire(entry: dict) -> dict:
    """One checkpoint row: `{name, step, path, is_link, modified}` -> `{step,
    path, has_model}`.

    `has_model` is structurally `True` for every row, and that is not a stub:
    `runs.checkpoints` already SKIPS a checkpoint whose `pretrained_model`
    directory is missing, because offering one would hand a rollout half a file
    mid-write. The field stays on the wire because it is frozen with Track C,
    and because the day this surface learns to list a still-writing checkpoint
    it is the field that says so — re-stat'ing the path here instead would be a
    second answer to a question `runs.checkpoints` already owns.

    `step` is `None` for LeRobot's `last` symlink, which is how a client tells
    it from the numbered checkpoint it points at; `name` and `is_link` are
    dropped because the wire has no slot for them.
    """
    return {
        "step": entry["step"],
        "path": entry["path"],
        "has_model": True,
    }


# ---- request validation ----
# `_need` mirrors `routes_datasets._need` deliberately rather than importing
# it: a 400 that names the missing field is the frozen behaviour, and a
# cross-module import of a private helper would couple two routers that only
# happen to agree.

def _need(payload: dict, key: str):
    """A required body field, or a 400 that names it."""
    value = (payload or {}).get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise HTTPException(status_code=400, detail=f"{key} is required")
    return value


def _positive_int(payload: dict, key: str, default: int) -> int:
    """A positive integer, or a 400 naming the field and what arrived.

    `steps=0` and `batch_size=0` both launch a run that dies in the child's
    first second with a traceback in `run.log` that nobody is watching yet.
    Refusing now costs the operator one corrected field; refusing later costs a
    run directory and the walk over to read why it is empty.
    """
    raw = (payload or {}).get(key)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be a positive integer, got {raw!r}") from None
    if value <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be a positive integer, got {value}")
    return value


def _non_negative_int(payload: dict, key: str, default: int) -> int:
    """A count where 0 means "off" — `eval_steps`, `max_eval_samples`."""
    raw = (payload or {}).get(key)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be an integer, got {raw!r}") from None
    if value < 0:
        raise HTTPException(
            status_code=400, detail=f"{key} cannot be negative, got {value}")
    return value


def _eval_split(payload: dict) -> float:
    """The held-out fraction, in `[0, 1)`.

    1.0 holds out everything and trains on nothing, and a negative fraction
    reaches `math.ceil` as a negative count and quietly means zero. Both are
    specs whose result is not what was asked for, which is the class of thing
    this route refuses before a run directory exists.
    """
    raw = (payload or {}).get("eval_split")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"eval_split must be a fraction, got {raw!r}") from None
    if not 0.0 <= value < 1.0:
        raise HTTPException(
            status_code=400,
            detail=(f"eval_split must be at least 0 and below 1, got {value} — "
                    "1 would hold out every episode and train on none"))
    return value


def _extra_args(payload: dict) -> list[str]:
    """`extra_args` passes through to `lerobot-train` verbatim.

    A list, so `"--steps=1"` as a bare string cannot arrive as 11 single-
    character arguments. This route is `require_local`, which is what makes a
    passthrough acceptable at all — it is the escape hatch that keeps every
    LeRobot flag reachable without this form growing a field for each one.
    """
    raw = (payload or {}).get("extra_args")
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise HTTPException(
            status_code=400, detail="extra_args must be a list of strings")
    return [str(arg) for arg in raw]


def _requested_episodes(payload: dict, detail: dict) -> list[int] | None:
    """The caller's explicit episode set, validated, or `None` for "the review".

    Validated against the dataset that was just read rather than left to the
    child: an index that does not exist is dropped silently by LeRobot, so a
    typo'd list trains on fewer episodes than the operator chose and nothing
    ever says so.

    A 400 and not the 404 `routes_datasets` answers for the same condition on
    `/lab/datasets/mark`: the dataset resolved, so a 404 here would name the
    wrong missing thing. What is wrong is the list in the request.
    """
    raw = (payload or {}).get("episodes")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise HTTPException(
            status_code=400,
            detail="episodes must be a list of episode indices")
    try:
        wanted = [int(episode) for episode in raw]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="episodes must be a list of episode indices") from None
    if not wanted:
        raise HTTPException(
            status_code=400,
            detail="episodes is empty — there would be nothing to train on")
    known = detail["episode_frames"]
    unknown = sorted({e for e in wanted if e not in known})
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(f"episodes {unknown} are not in {detail['repo_id']} — it has "
                    f"{len(known)} ({min(known, default=0)}..{max(known, default=0)})"))
    return wanted


def _positive_float(payload: dict, key: str, default: float) -> float:
    """A positive float, or a 400 naming the field and what arrived.

    `None` and a missing key take the default; anything present and unusable is
    refused rather than coerced, because the coercion nobody sees is the one
    that runs the arm at a rate the operator did not choose.
    """
    raw = (payload or {}).get(key)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be a number, not {raw!r}") from None
    # `isfinite` first: NaN fails every comparison, so `value <= 0` alone lets
    # it through and it would reach the arm as a period of nan seconds.
    if not math.isfinite(value) or value <= 0:
        raise HTTPException(
            status_code=400, detail=f"{key} must be greater than 0, not {raw!r}")
    return value


def _spec_of(payload: dict, marker: str = "repo_id") -> dict:
    """The job spec out of the request body.

    The frozen line reads `POST /lab/runs/train {spec}` and does not say
    whether the spec IS the body or sits under a `spec` key, so both are
    accepted. A client that guessed the other way would otherwise be told
    `repo_id is required` by a request that plainly carries one, which is the
    least actionable 400 this route could produce.

    `marker` is the field that tells a flat spec from a wrapper: whatever the
    route's own required key is (`repo_id` to train, `policy_path` to roll
    out). Passing the wrong one would make `{"spec": {...}}` unwrap correctly
    and a flat body unwrap to nothing, so it names the route's key rather than
    defaulting to a shape.
    """
    inner = (payload or {}).get("spec")
    if isinstance(inner, dict) and marker not in (payload or {}):
        return inner
    return payload or {}


def _trained_rate(policy_path: str) -> dict:
    """The rate the checkpoint at `policy_path` was TRAINED at.

    `{fps, repo_id, source, reason}`. `fps` is None exactly when a link of the
    chain could not be read and `reason` then names which one — see
    `runs.trained_dataset` for why no link is ever guessed around, and for the
    check that nothing in a checkpoint carries a rate directly.

    Both halves are here because this is the only place both are in scope. The
    rollout child is handed `control_hz` in its spec and never opens
    `info.json`, so it can record the two numbers but cannot compare them —
    which makes a divergence reconstructible after the fact rather than
    detected before the arm moves.
    """
    found = runs_mod.trained_dataset(policy_path)
    repo_id = found["repo_id"]
    if not repo_id:
        return {"fps": None, "repo_id": None, "source": None,
                "reason": found["reason"]}
    try:
        fps = catalog.dataset_fps(repo_id)
    except ValueError as e:
        # `dataset_root` refuses a repo-id that escapes the cache. Arriving from
        # a file on disk rather than from a URL makes that stranger, not safer.
        return {"fps": None, "repo_id": repo_id,
                "source": found["config_path"], "reason": str(e)}
    if fps is None:
        return {
            "fps": None, "repo_id": repo_id, "source": found["config_path"],
            "reason": (
                f"this policy was trained on {repo_id}, which cannot be read "
                "here — renamed, pruned or deleted — so the rate it was trained "
                "at cannot be recovered from it"
            ),
        }
    return {"fps": fps, "repo_id": repo_id, "source": found["config_path"],
            "reason": ""}


def _rate_matches(declared: float, trained_fps: int) -> bool:
    """Does a declared control rate match the rate the policy was trained at?

    **Exact, not a tolerance, and two-sided.**

    Two-sided because declaring is a free choice in both directions and both
    directions are the same error: a policy trained at 30 Hz and stepped at 15
    applies action deltas sized for 33 ms over 67 ms, and stepped at 60 applies
    them over 17. The child's run-time gate is one-sided only because a loop
    that sleeps to its declared period cannot MEASURE faster than it declared,
    which is physics rather than policy.

    Exact because there is no noise source between these two numbers. Gate
    (b)'s 0.9 absorbs the real physical gap between an intended period and an
    achieved one; here `fps` is an `int` in lerobot's own `DatasetInfo` and
    `control_hz` is a declared spec value, so nothing between them can produce
    27 or 33 by accident. A +/-10% band would admit exactly two things — typos,
    and deliberate choices. A typo should be refused, and a deliberate choice
    belongs in the override where it is stamped into the run record rather than
    slipping under a threshold that leaves no evidence. **Reading
    `MIN_RATE_FRACTION` here would make one constant answer two questions**,
    and the day it is tuned for jitter it would silently widen a typo gate.

    Structural note, so this is not revisited blindly: invariant 10 requires
    `fps` to be MEASURED, and a fractional fps would make exact match refuse
    every rollout. LeRobot's schema types the field `int`, so a measured rate
    is rounded before it is stored — the constraint holds by the format, not by
    anyone remembering it here.
    """
    return float(declared) == float(trained_fps)


def _query_list(values: list[str] | None) -> list[str]:
    """`?ids=a&ids=b` and `?ids=a,b`, both.

    The frozen line is `GET /lab/runs/metrics?ids&keys&max_points` and does not
    say which spelling, and the two are indistinguishable from the server's
    side until a page picks one. Accepting both costs a split; picking one and
    being wrong costs an empty chart with a 200 beside it, which reads as "this
    run logged nothing".
    """
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in str(value).split(","))
    return [part for part in out if part]


def _kind_filter(kind: str | None) -> str | None:
    """A `?kind=` value, or None for "every kind".

    Validated against `runs.RUNNERS`, which is the authority on what kinds
    exist, because answering a typo with an empty list reads as a lost run.
    An absent parameter arrives as None and a cleared one as `""`; both have to
    mean "everything" or clearing the filter in the UI empties the table.
    """
    if kind is None:
        return None
    text = str(kind).strip()
    if not text:
        return None
    for candidate in runs_mod.RUNNERS:
        if candidate.lower() == text.lower():
            return candidate
    raise HTTPException(
        status_code=400,
        detail=f"unknown kind {text!r} — one of {', '.join(runs_mod.RUNNERS)}")


def _status_filter(status: str | None) -> str | None:
    """A `?status=` value, case-folded, or None for "every status".

    Deliberately NOT validated against a vocabulary, unlike `kind` above. The
    statuses are written in two other places — `runners/_common.run_guarded`
    picks `done` / `failed` / `stopped`, `runs.load` resolves `running` /
    `died` and `launch` writes `launch_failed` — and a third copy in a routes
    module is the one that goes stale, at which point this filter starts
    refusing a status the store actually contains.
    """
    if status is None:
        return None
    text = str(status).strip().lower()
    return text or None


def build_runs_router(deps: LabDeps) -> APIRouter:
    """Wire `/lab/runs/**` onto one router.

    `deps` carries zero-arg callables resolved per request, for the reason
    `api/deps.py` documents: routers mount at import time and the handles are
    assigned in `lifespan`. The gate is built ONCE here, from
    `deps.allow_remote_control`, and mounted as a route dependency on the three
    endpoints that start or end a process.
    """
    _warm_pandas()
    router = APIRouter()
    require_local = build_require_local(deps.allow_remote_control)
    local_only = [Depends(require_local)]

    # No handler below carries a `-> dict` return annotation, for
    # `routes_datasets.build_datasets_router`'s reason: FastAPI turns one into
    # `response_model=dict` and revalidates and re-encodes the whole payload on
    # the way out — on a log tail that is a second pass over 200 KB of text.

    # ---- the listing ------------------------------------------------------

    @router.get("/lab/runs")
    def get_runs(
        kind: str | None = Query(default=None),
        status: str | None = Query(default=None),
    ):
        """Every run, newest first, optionally one kind and/or one status.

        Filtering happens here rather than in the page for the same reason
        `/lab/datasets/episodes` sorts server-side: this table is read over a
        LAN from inside a headset, and `status` is a resolved value the client
        cannot recompute — a dead pid with no `result.json` is `died`, and only
        `runs.load` knows that.
        """
        with as_http():
            wanted_kind = _kind_filter(kind)
            wanted_status = _status_filter(status)
            rows = []
            for record in runs_mod.list_runs(limit=RUN_SCAN):
                if wanted_kind and record.get("kind") != wanted_kind:
                    continue
                if wanted_status and str(
                        record.get("status", "")).lower() != wanted_status:
                    continue
                rows.append(_run_wire(record, detail=False))
                if len(rows) >= RUN_LIST_LIMIT:
                    break
            return {"runs": rows}

    # ---- the cross-run chart ---------------------------------------------
    # DECLARED BEFORE THE `{run_id}` ROUTES ON PURPOSE. Starlette matches in
    # declaration order and `metrics` satisfies `RUN_ID_RE`, so moving this
    # below `GET /lab/runs/{run_id}/...` would capture it as a run id and
    # answer the comparison chart with `404 no run metrics`. Invisible until
    # someone tidies the file into alphabetical order.

    @router.get("/lab/runs/metrics")
    def get_runs_metrics(
        ids: RepeatableQuery = None,
        keys: RepeatableQuery = None,
        max_points: int = Query(default=DEFAULT_MAX_POINTS),
    ):
        """Several runs' metric series on one chart, downsampled server-side.

        `lab/compare.py` raises `ValueError` ONLY for its caps (12 runs, 8
        keys, `max_points` below 2), which the ladder renders as 400. A run id
        that does not exist, has no `metrics.jsonl` or is unreadable is an
        EMPTY series and never an error: this view is a comparison, and one
        absent run must not take the other three down with it.
        """
        with as_http():
            return compare.series(
                _query_list(ids), _query_list(keys), max_points)

    # ---- launching --------------------------------------------------------

    @router.post("/lab/runs/train", dependencies=local_only)
    def post_train(payload: JsonBody):
        """Validate a training spec, then launch it as a detached child.

        Every check below happens BEFORE `launch`, because a 400 now beats a
        run directory that dies in two seconds: the operator is looking at the
        form when the 400 arrives and is looking at a table of runs when the
        traceback does.

        **The episode list is the review's keep set, in `plan_eval_split`'s
        ORDER.** That order IS the eval split — LeRobot groups the list it is
        handed by task and holds out each group's TAIL without ever sorting it
        — so passing a sorted set here would hold out the newest episodes and
        nothing downstream could tell. It matters because operator skill
        improves across a session (Oscar's first 20: 3/10 kept in the first
        half, 7/10 in the second), so a chronological holdout validates on the
        best demonstrations and trains on the sloppiest.

        A caller-supplied `episodes` chooses the SET and never the order, for
        that same reason: a hand-ordered list would be a holdout the operator
        never saw. `eval_mode="recent"` is how a chronological split is asked
        for deliberately.
        """
        spec_in = _spec_of(payload)
        repo_id = str(_need(spec_in, "repo_id")).strip()
        steps = _positive_int(spec_in, "steps", DEFAULT_STEPS)
        batch_size = _positive_int(spec_in, "batch_size", DEFAULT_BATCH_SIZE)
        save_freq = _positive_int(spec_in, "save_freq", DEFAULT_SAVE_FREQ)
        num_workers = _non_negative_int(spec_in, "num_workers", DEFAULT_NUM_WORKERS)
        eval_steps = _non_negative_int(spec_in, "eval_steps", 0)
        max_eval_samples = _non_negative_int(spec_in, "max_eval_samples", 0)
        eval_split = _eval_split(spec_in)
        eval_seed = _non_negative_int(spec_in, "eval_seed", DEFAULT_EVAL_SEED)
        eval_mode = str(spec_in.get("eval_mode") or DEFAULT_EVAL_MODE)
        policy_type = str(spec_in.get("policy_type") or DEFAULT_POLICY_TYPE)
        device = str(spec_in.get("device") or DEFAULT_DEVICE)
        job_name = str(spec_in.get("job_name") or "")
        extra_args = _extra_args(spec_in)
        # Scaled to the run so the loss chart has ~200 points whatever its
        # length; LeRobot's fixed 200 means a 40-step debug run logs nothing.
        log_freq = _positive_int(
            spec_in, "log_freq", max(1, min(LOG_POINTS, steps // LOG_POINTS)))

        with as_http():
            # 404 from here when the repo_id names nothing, which is the whole
            # "must resolve to a real dataset" check — and it is also the read
            # that makes the two below possible at all.
            detail = catalog.dataset_detail(repo_id)
            chosen = _requested_episodes(spec_in, detail)
            keep_list = detail["keep_list"] if chosen is None else chosen
            plan = catalog.plan_eval_split(
                detail["episodes"], keep_list, eval_split, eval_seed, eval_mode)
            episodes = plan["order"]
            if not episodes:
                raise ValueError(
                    f"no episodes left to train on in {repo_id} — every one is "
                    "rejected. Keep some on the review page first."
                )
            if eval_split > 0 and not plan["train"]:
                raise ValueError(
                    f"eval_split={eval_split} would hold out all "
                    f"{len(episodes)} kept episodes and leave none to train on"
                )
            spec = {
                "repo_id": repo_id,
                # NOT sorted, NOT deduped, NOT set-ified. See the docstring.
                "episodes": [int(e) for e in episodes],
                "total_episodes": detail["total_episodes"],
                # Carried so a later rollout can prefill its camera mapping and
                # task wording from what the policy was actually trained on: a
                # rollout whose observation space differs from the recording is
                # a policy being shown a world it has never seen.
                "camera_keys": detail["video_keys"],
                "task_text": (detail["tasks"] or [""])[0],
                "policy_type": policy_type,
                "steps": steps,
                "batch_size": batch_size,
                "eval_split": eval_split,
                "eval_mode": eval_mode,
                "eval_seed": eval_seed,
                "eval_episodes": plan["eval"],
                "train_episodes": plan["train"],
                "eval_steps": eval_steps,
                "max_eval_samples": max_eval_samples,
                "save_freq": save_freq,
                "log_freq": log_freq,
                "num_workers": num_workers,
                "device": device,
                "job_name": job_name,
                "extra_args": extra_args,
            }
            record = runs_mod.launch(
                "train",
                spec,
                name=job_name or policy_type,
                # One line, stored once, never recomputed: the run's own answer
                # to "what was this?" after its dataset has been pruned and its
                # spec superseded. `x of y episodes` and not `x episodes`
                # because a run that trained on 35 of 46 and one that trained
                # on 35 of 35 are different runs.
                spec_summary=(
                    f"train · {repo_id} · {len(episodes)} of "
                    f"{detail['total_episodes']} episodes · {policy_type} · "
                    f"{steps} steps"
                ),
            )
            # The id ONLY. The page follows the run through the routes below,
            # so returning the whole record here would be a second spelling of
            # `GET /lab/runs/{id}` that no rename could ever reach.
            return {"id": record["id"]}

    @router.post("/lab/runs/rollout", dependencies=local_only)
    def post_rollout(payload: JsonBody):
        """Run a trained policy, refusing a control rate it was not trained for.

        **The child owns the policy and never the bus.** It loads the
        checkpoint, runs inference, and streams target joint angles to this
        server over loopback; the server keeps the bus and commits them through
        the same LPF -> rate cap -> clamp -> collision guard -> workspace floors
        -> E-STOP chain every other input goes through. Nothing is handed to a
        child here. This route writes a spec and starts a process.

        **Check (a) — declared `control_hz` against the rate the policy was
        TRAINED at — lives here because this is the only place both numbers are
        in scope.** The child is handed `control_hz` in its spec and never opens
        `info.json`, so it can stamp both numbers but cannot compare them: that
        makes a divergence reconstructible after the arm has moved, not detected
        before it does. Its own gate (b), measured-vs-declared, is a different
        check on a different event and both exist.

        The chain is `<checkpoint>/train_config.json` -> `dataset.repo_id` ->
        that dataset's `meta/info.json` -> `fps`, and it is the ONLY route to
        the number: nothing in a checkpoint records a rate directly. So a broken
        link REFUSES rather than falling back — inferring the dataset from the
        run directory or from whatever the operator has selected would compare
        the declared rate against the wrong dataset's fps and report agreement,
        which is worse than no check because it reassures.

        `control_hz` defaults to the trained rate, so the correct value is the
        one you get by not choosing, and the gate fires only on a divergence
        somebody typed. Both numbers are stamped into the spec either way,
        override or not.

        What this route deliberately does NOT re-check: the child's own spec
        contract — the duration ceiling, the ingest scheme, the rig/side
        agreement. Those are `rollout_runner`'s, and `lab/` cannot import
        `runners/`, so every one of them re-checked here would be a second copy
        of a rule that has to agree with the first.
        """
        spec_in = _spec_of(payload, marker="policy_path")
        policy_path = str(_need(spec_in, "policy_path")).strip()
        duration_s = _positive_float(
            spec_in, "duration_s", DEFAULT_ROLLOUT_DURATION_S)
        if duration_s > runs_mod.MAX_ROLLOUT_DURATION_S:
            # The SAME constant the child refuses on, read rather than copied,
            # so the two can never disagree. Refused here as well as there
            # because a browser button that launches a doomed run and reports it
            # dead two minutes later is worse than a refusal at the door.
            raise HTTPException(status_code=400, detail=(
                f"duration_s {duration_s:g} exceeds the "
                f"{runs_mod.MAX_ROLLOUT_DURATION_S:g} s ceiling: a policy loop "
                "started from a browser button must not be able to run until "
                "someone notices."
            ))
        override = bool(spec_in.get("allow_rate_mismatch"))
        action_names = spec_in.get("action_names")

        with as_http():
            trained = _trained_rate(policy_path)
            trained_fps = trained["fps"]

            if trained_fps is None and spec_in.get("control_hz") is None:
                raise HTTPException(status_code=400, detail=(
                    f"cannot tell what rate this policy was trained at, and the "
                    f"request declares no control_hz: {trained['reason']}. Pass "
                    "control_hz to say what rate to run it at."
                ))
            # The default is the trained rate itself, so the only way to reach
            # the gate below is to have asked for something else. Unreachable
            # when `trained_fps` is None — the guard above required the key.
            declared_by = (
                "request" if spec_in.get("control_hz") is not None
                else "trained_fps")
            declared = _positive_float(
                spec_in, "control_hz", float(trained_fps or 0.0))

            if trained_fps is None:
                matches = False
                refusal = (
                    f"refusing to roll out at {declared:g} Hz: "
                    f"{trained['reason']}. Nothing in a checkpoint records the "
                    "rate it was trained at, so there is no second source to "
                    "check against."
                )
            else:
                matches = _rate_matches(declared, trained_fps)
                refusal = (
                    f"refusing to roll out at {declared:g} Hz a policy trained "
                    f"at {trained_fps:g} Hz on {trained['repo_id']}: that is a "
                    "different dynamical system, not a faster or slower one — "
                    "the action deltas are sized for "
                    f"{1000.0 / trained_fps:.0f} ms steps and would be applied "
                    f"over {1000.0 / declared:.0f} ms. Declare "
                    f"{trained_fps:g} Hz, or leave control_hz out to get it."
                )
            if not matches and not override:
                raise HTTPException(
                    status_code=400,
                    detail=refusal + " Set allow_rate_mismatch to launch anyway.")

            # ONE source for the joint layout — and the predicate is whether
            # the dataset could be READ, not whether the checkpoint names one.
            # A checkpoint naming a dataset that has since been pruned or
            # deleted has no rig source either, so `action_names` is required
            # there rather than forbidden. Keying this off `repo_id` refused the
            # only spec that case can produce.
            readable = trained["fps"] is not None
            # The child prefers `action_names` over `repo_id`, so accepting both
            # would let the rig come from one dataset and the rate from another
            # with nothing comparing them.
            if readable and action_names:
                raise HTTPException(status_code=400, detail=(
                    f"this checkpoint records the dataset it was trained on "
                    f"({trained['repo_id']}), so the joint layout comes from "
                    "there; passing action_names as well would give the rig and "
                    "the rate two different sources. Drop action_names."
                ))
            # Not a duplicate of the child's rig check but a check that the spec
            # about to be written is runnable at all: with neither key the child
            # cannot know what the action vector holds, and launching would buy
            # a run directory that dies in its first second.
            if not readable and not action_names:
                raise HTTPException(status_code=400, detail=(
                    "nothing says what joints this policy's action vector holds: "
                    f"{trained['reason']}, and the request carries no "
                    "action_names. Pass action_names to name them explicitly."
                ))

            spec = {
                "policy_path": policy_path,
                "control_hz": declared,
                # Both numbers, stamped either way — the ruling is that an
                # override is recorded, not that it is unrecorded. A run whose
                # rates disagreed says so in its own spec forever.
                "control_hz_trained": trained_fps,
                # WHERE the declared rate came from. Without it every run reads
                # as a deliberate agreement between two numbers and a later
                # reader cannot tell whether the operator chose 30 or got it for
                # free. Same instinct as stamping the measured rate either way.
                "control_hz_declared_by": declared_by,
                "control_hz_trained_repo_id": trained["repo_id"],
                "control_hz_trained_source": trained["source"],
                "control_hz_trained_reason": trained["reason"],
                "control_hz_mismatch_override": override,
                "duration_s": duration_s,
                "device": str(spec_in.get("device") or DEFAULT_DEVICE),
                "side": str(spec_in.get("side") or ""),
                # Gate (b)'s override, which is the child's and a different
                # decision: this one says the DECLARED rate may differ from the
                # trained one, that one says the MEASURED rate may fall under
                # the declared one. Passed through untouched.
                "allow_slow": bool(spec_in.get("allow_slow")),
            }
            # The rig is derived from the dataset the policy was TRAINED on,
            # never from whatever the operator has open — a rollout whose
            # observation space differs from the recording is a policy being
            # shown a world it has never seen.
            if readable:
                spec["repo_id"] = trained["repo_id"]
            if action_names:
                spec["action_names"] = [str(n) for n in action_names]
            for key in ("task", "robot_type", "ingest_url", "port"):
                if spec_in.get(key) is not None:
                    spec[key] = str(spec_in[key])

            bits = ["rollout"]
            # `_policy_label` rather than a second copy of its rule: every
            # checkpoint directory is named `pretrained_model`, so identifying
            # one takes the two segments above it.
            label = trained["repo_id"] or runs_mod._policy_label(spec)
            if label:
                bits.append(str(label))
            bits.append(f"{declared:g} Hz")
            bits.append(f"{duration_s:g} s")
            if override:
                # Visible in the listing forever, because the whole failure this
                # gate exists for is a run reported as a success with a rate
                # attached to nothing.
                bits.append(
                    f"rate override ({trained_fps:g} Hz trained)"
                    if trained_fps is not None else "rate override (rate unknown)")

            record = runs_mod.launch(
                "rollout",
                spec,
                name=str(trained["repo_id"] or "").split("/")[-1],
                spec_summary=" · ".join(bits),
            )
            return {"id": record["id"]}

    # ---- one run ----------------------------------------------------------

    @router.get("/lab/runs/{run_id}")
    def get_run(run_id: str):
        """One run with its status resolved: what it is, what it ran, how it
        ended.

        `status` is never inferred from a dead pid alone — after a restart the
        server is not the child's parent and cannot reap it, so a pid that is
        gone with no `result.json` is `died` and never `done`.
        """
        with as_http():
            return _run_wire(runs_mod.load(run_id), detail=True)

    @router.get("/lab/runs/{run_id}/metrics")
    def get_run_metrics(run_id: str, offset: int = Query(default=0)):
        """Metric rows appended since `offset`, WHOLE LINES ONLY.

        A byte offset and not a row count: the file is being appended to by
        another process while this reads it, so a count would have to re-read
        from the start to mean anything. A record caught mid-write leaves the
        offset where it was and is picked up whole on the next poll.
        """
        with as_http():
            tail = runs_mod.read_metrics(run_id, offset)
            return {"offset": tail["offset"], "rows": tail["rows"]}

    @router.get("/lab/runs/{run_id}/log")
    def get_run_log(run_id: str, offset: int = Query(default=0)):
        """stdout + stderr appended since `offset`.

        The page passes back the offset it last saw, so a multi-hour log is
        never re-sent; a client that fell far behind gets the TAIL rather than
        a 50 MB response, and the returned `offset` is where it should resume
        from — which is not `offset + len(text)` when that clamp fired.
        """
        with as_http():
            tail = runs_mod.tail_log(run_id, offset)
            return {"offset": tail["offset"], "text": tail["text"]}

    @router.get("/lab/runs/{run_id}/checkpoints")
    def get_run_checkpoints(run_id: str):
        """Saved checkpoints, newest step first, half-written ones omitted."""
        with as_http():
            return {
                "checkpoints": [
                    _checkpoint_wire(c) for c in runs_mod.checkpoints(run_id)
                ]
            }

    # ---- ending a run -----------------------------------------------------

    @router.post("/lab/runs/{run_id}/stop", dependencies=local_only)
    def post_run_stop(run_id: str):
        """Ask the job to wind down: SIGINT, 20 s, then SIGTERM. Never SIGKILL.

        SIGINT first because LeRobot's training loop treats it as "wind down"
        and saves a checkpoint, and a rollout returns the arm to its initial
        pose. This handler therefore BLOCKS for up to `runs.STOP_GRACE_S` — it
        is a plain `def`, so that wait happens on a worker thread and never on
        the event loop that forwards teleop frames to the arms.

        `{"ok": true}` and not the run record: the run's status a quarter of a
        second after SIGINT is not the status the operator is asking about, and
        shipping it would invite a page that renders it as final.
        """
        with as_http():
            runs_mod.stop(run_id)
            return {"ok": True}

    @router.delete("/lab/runs/{run_id}", dependencies=local_only)
    def delete_run(run_id: str):
        """Remove a run directory. There is no undo, and no force flag.

        409 while the run is alive. `runs.delete_run` raises `RuntimeError` for
        exactly that and for nothing else, so it is caught here — `api/errors`
        has no rung for it and would answer 500, which reads as a broken server
        rather than as "stop it first". `DataDependencyError` and
        `DatasetBusyError` are also `RuntimeError`s but cannot arrive from this
        call, which touches no parquet.

        There is deliberately no "delete anyway": the process is detached and
        the server is not its parent, so removing the directory would leave a
        live training job writing into an unlinked file, holding the GPU, with
        nothing left on disk to say it exists.
        """
        with as_http():
            try:
                runs_mod.delete_run(run_id)
            except RuntimeError as e:
                raise HTTPException(status_code=409, detail=str(e)) from e
            return {"ok": True}

    return router
