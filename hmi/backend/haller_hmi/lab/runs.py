# hmi/backend/haller_hmi/lab/runs.py
"""Detached job directories: launch, status, log and metric tails, checkpoints.

Training an ACT policy takes hours; a rollout moves a real arm. Neither belongs
inside a web request, and neither may die because the HMI was restarted to pick
up a code change. So every job is a DETACHED subprocess (`start_new_session=True`,
its own process group) that owns a directory:

    <runs_dir>/<run_id>/
        spec.json      what was asked for
        run.json       pid, argv, tags, spec_summary, status, timestamps
        run.log        stdout + stderr, appended live
        metrics.jsonl  one JSON object per logged step (training only)
        train/         the child's own output_dir (checkpoints)
        result.json    written by the RUNNER in a `finally`

The page reattaches by reading those files, so a run survives a backend restart
with its full log and metric history intact.

Exit status is written by the runner itself into `result.json`, and is never
inferred here: after a restart the server is no longer the process's parent and
cannot reap it, so "pid is gone" alone cannot tell a clean finish from a crash.
A dead pid with no `result.json` is reported as `died`, which is exactly what it
means — never as `done`.

Ported from the kit's `data/runs.py`. Three things are Haller's, and the first
is the one a reader will otherwise "restore":

* **There is no `record` kind.** Recording stays IN-PROCESS. Forced on
  2026-08-21, when an overloaded shoulder aborted lerobot's bulk
  `disable_torque()` mid-sweep and left four joints energised: `/estop` walks
  every motor in-process, and a child process cannot be allowed to own the
  Feetech bus. The kit's `RUNNERS["record"]` entry is deliberately absent below.
* **The child runs under a different interpreter** (`runner_python()`). The
  serving venv is lerobot 0.5.1 on purpose; `~/venvs/haller-lab` is lerobot
  0.6.1 + torch 2.11.0+cu130. The serving venv cannot import
  `lerobot.scripts.lerobot_rollout` at all, and that one module is the entire
  reason the second venv exists.
* **`launch` records `tags` and a `spec_summary`** — the one-line string the UI
  prints verbatim, and the only place a run remembers what it was asked for once
  its spec is superseded.

Nothing here imports lerobot or torch, and nothing here imports
`haller_hmi.runners`: this module runs in the serving process, the runner
modules are named as STRINGS and only ever reached through `-m` in a child.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

__all__ = [
    "MAX_ROLLOUT_DURATION_S",
    "RUNNERS",
    "RUN_ID_RE",
    "STOP_GRACE_S",
    "checkpoints",
    "delete_run",
    "launch",
    "list_runs",
    "load",
    "new_run_id",
    "read_metrics",
    "run_dir",
    "runner_python",
    "runs_dir",
    "stop",
    "tail_log",
    "trained_dataset",
    "write_result",
]

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

#: Runner modules, by job kind. Each takes a single argument: the path to its
#: spec.json.
#:
#: The kit had a fifth entry here, `"record"`, and Haller does NOT. Recording
#: owns the Feetech bus and the bus stays in the serving process, because
#: `/estop` walks every motor in-process — see the module docstring for the
#: 2026-08-21 incident that closed that path. A recording job launched as a
#: detached child would take the bus with it and leave `/estop` with nothing to
#: talk to. Do not add it back.
#: These are the IMPORT PATHS the child is launched with, so each one must name
#: a module that actually exists and is runnable as `-m`. They carry the
#: `_runner` suffix because the files do; dropping it here is not a cosmetic
#: mismatch, it is `No module named 'haller_hmi.runners.train'` on every launch.
#: `test_every_runner_target_is_importable` pins them, because the route and
#: launch tests point `HALLER_LAB_PYTHON` at `/bin/true`, which ignores its
#: arguments and exits 0 — so a launch "succeeds" and its run directory appears
#: whether or not the module resolves.
RUNNERS = {
    "train": "haller_hmi.runners.train_runner",
    "eval": "haller_hmi.runners.eval_runner",
    "rollout": "haller_hmi.runners.rollout_runner",
    "export": "haller_hmi.runners.export_runner",
}

#: Ceiling on how long one rollout may run. A policy loop started from a
#: browser button must not be able to run until somebody notices.
#:
#: It lives HERE, in `lab/`, rather than beside the loop it bounds, so that the
#: launch route and the child read ONE number. `runners/` already imports
#: `lab/` (`lab.lease`, `lab.schema`, `lab.catalog`) and nothing under `lab/`
#: imports `runners/`, so the dependency runs this way already and this costs no
#: new coupling. The alternative was a copy in the route, which is the failure
#: mode where two numbers must agree and one day do not — and the route needs it
#: to refuse an over-long duration AT THE DOOR, rather than launching a run that
#: is already doomed and reporting it dead two minutes later.
MAX_ROLLOUT_DURATION_S = 900.0

#: Seconds to wait for a clean SIGINT shutdown before escalating to SIGTERM.
#: LeRobot's `ProcessSignalHandler` and its training loop both handle
#: KeyboardInterrupt, and training needs a moment to finish writing a
#: checkpoint.
STOP_GRACE_S = 20.0

#: The lab venv's interpreter, resolved at import so it can be pointed
#: elsewhere in a test without moving `$HOME`. `runner_python()` falls through
#: to `sys.executable` when it is missing, so a box with one venv still runs.
LAB_PYTHON = Path.home() / "venvs" / "haller-lab" / "bin" / "python"

#: Environment variable overriding `LAB_PYTHON`.
LAB_PYTHON_ENV = "HALLER_LAB_PYTHON"

#: Environment variable overriding the run store's location.
RUNS_DIR_ENV = "HALLER_RUNS"


def runs_dir() -> Path:
    """Where run directories live.

    `$HALLER_RUNS` overrides; otherwise `./outputs/runs` relative to the
    process's working directory, which is the repo root in every documented way
    of starting the HMI. The resolved path is reported by `/lab/system` so it is
    never a guess.

    (The kit reads `VR_TELEOP_RUNS`. Renamed rather than kept: the two backends
    are installed side by side on this box, and sharing one run store would let
    the kit's UI offer to stop a Haller training run.)
    """
    env = os.environ.get(RUNS_DIR_ENV)
    base = Path(env).expanduser() if env else Path.cwd() / "outputs" / "runs"
    return base


def runner_python() -> Path:
    """The interpreter a detached child is launched with.

    `$HALLER_LAB_PYTHON`, else `~/venvs/haller-lab/bin/python`, else this
    process's own interpreter. The lab venv carries lerobot 0.6.1 and torch
    2.11.0+cu130 (CUDA, RTX 4080 SUPER); the serving venv is deliberately
    lerobot 0.5.1 and cannot import `lerobot.scripts.lerobot_rollout` at all.

    A PATH CHECK ONLY — this never imports anything to verify the interpreter.
    Importing torch to prove torch is importable would cost the serving process
    seconds and a CUDA context, in the latency path, to answer a question the
    child is about to answer for real. A wrong interpreter surfaces as the
    child's own traceback in `run.log`, which is where it is readable.
    """
    env = os.environ.get(LAB_PYTHON_ENV)
    if env:
        return Path(env).expanduser()
    if LAB_PYTHON.exists():
        return LAB_PYTHON
    return Path(sys.executable)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def new_run_id(kind: str, name: str = "") -> str:
    """`<kind>-<local timestamp>[-<slug>]`.

    The one LOCAL timestamp in this module, and deliberately so: the id is what
    an operator reads off a directory listing to find the run they started ten
    minutes ago, and a UTC id is two hours off that comparison all summer.
    Every timestamp a machine compares — `started_at`, `finished_at` — is UTC.
    """
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 - see above
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return f"{kind}-{stamp}" + (f"-{slug}" if slug else "")


def run_dir(run_id: str) -> Path:
    """The directory a run id names, refusing anything that leaves the store.

    Two guards, because the kit's regex alone is not enough once something
    DELETES a run directory: `".."` matches `RUN_ID_RE` (dots are legal in a run
    id) and names the parent of the run store. The containment check is what
    makes `delete_run("..")` a refusal instead of an rmtree of `outputs/`.
    """
    if not RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"bad run id: {run_id!r}")
    base = runs_dir()
    path = (base / run_id).resolve()
    if base.resolve() not in path.parents:
        raise ValueError(f"run id escapes the run store: {run_id!r}")
    return base / run_id


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - missing, truncated mid-write, not JSON:
        return None    # all of them mean "no record yet", never a 500


def _pid_alive(pid: int, run_id: str) -> bool:
    """Alive AND still the process we launched.

    Checking only `kill(pid, 0)` would call a recycled pid our training job — on
    a workstation that spawns processes all day, that is a real way to show a
    finished run as running forever. `/proc/<pid>/cmdline` carries the spec path,
    which carries the run id, so the id appearing in it is the identity check.
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return True  # no procfs: fall back to the kill(0) answer
    return run_id in cmdline


def _one_line(text: str) -> str:
    """Collapse to a single line. `spec_summary` is printed verbatim in a table
    row, so a newline in it breaks the row rather than the string."""
    return " ".join(str(text or "").split())


def _policy_label(spec: dict) -> str:
    """What policy this run is about, in as few characters as still identify it.

    A training run names its architecture (`act`). A rollout names a checkpoint
    directory, and `Path(...).name` on one of those is `pretrained_model` — the
    same string for every checkpoint LeRobot has ever written. The two
    components that do identify it are the step and the run it came out of, so
    the generic path segments are dropped and the last two kept.
    """
    policy = spec.get("policy_type")
    if policy:
        return str(policy)
    raw = spec.get("policy_path")
    if not raw:
        return ""
    parts = [p for p in Path(str(raw)).parts
             if p not in ("/", "train", "checkpoints", "pretrained_model")]
    return "/".join(parts[-2:]) if parts else str(raw)


def _default_spec_summary(kind: str, spec: dict) -> str:
    """What this run was asked for, in one line, from the spec's own fields.

    Built ONCE at launch and stored. Re-deriving it in the UI would mean the
    page owning a second copy of every spec's shape, and a run whose spec has
    since been superseded — a checkpoint retrained, a dataset pruned — would
    then describe itself as whatever the current shape says.
    """
    bits: list[str] = [kind]
    repo = spec.get("new_repo_id") or spec.get("repo_id")
    if repo:
        bits.append(str(repo))
    episodes = spec.get("episodes")
    if isinstance(episodes, (list, tuple, set)):
        bits.append(f"{len(episodes)} episodes")
    policy = _policy_label(spec)
    if policy:
        bits.append(policy)
    steps = spec.get("steps")
    if steps:
        bits.append(f"{int(steps)} steps")
    return _one_line(" · ".join(bits))


def launch(
    kind: str,
    spec: dict,
    name: str = "",
    env: dict | None = None,
    *,
    tags: list[str] | None = None,
    spec_summary: str = "",
) -> dict:
    """Start a job. Returns its run record.

    `tags` and `spec_summary` are stored and never recomputed; see
    `_default_spec_summary`. `env` adds to the server's own environment rather
    than replacing it — the child needs `HF_LEROBOT_HOME` and the CUDA
    variables it inherited.
    """
    if kind not in RUNNERS:
        raise ValueError(f"unknown job kind {kind!r} — one of {', '.join(RUNNERS)}")
    run_id = new_run_id(kind, name)
    rdir = run_dir(run_id)
    rdir.mkdir(parents=True, exist_ok=True)

    spec = dict(spec)
    spec["run_id"] = run_id
    spec["run_dir"] = str(rdir)
    spec_path = rdir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    python = runner_python()
    argv = [str(python), "-m", RUNNERS[kind], str(spec_path)]
    log_path = rdir / "run.log"

    record = {
        "id": run_id,
        "kind": kind,
        "name": name,
        "spec": spec,
        "spec_summary": _one_line(spec_summary) or _default_spec_summary(kind, spec),
        "tags": [str(t) for t in (tags or [])],
        "argv": argv,
        "runner_python": str(python),
        "cwd": str(Path.cwd()),
        "started_at": _now(),
        "status": "running",
        "pid": None,
    }

    child_env = dict(os.environ)
    if env:
        child_env.update({k: str(v) for k, v in env.items()})
    # Unbuffered, so the log pane shows progress as it happens instead of in
    # 4 KB bursts — an hour into a training run, a buffered child looks hung.
    child_env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(log_path, "ab") as log:
            log.write(f"$ {' '.join(argv)}\n".encode())
            log.flush()
            proc = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(Path.cwd()),
                env=child_env,
                # Own session/process group: the job outlives the HMI, and
                # stop() can signal the whole group without ever reaching the
                # server itself.
                start_new_session=True,
            )
    except Exception as e:
        record["status"] = "launch_failed"
        record["error"] = str(e)
        (rdir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        raise

    record["pid"] = proc.pid
    (rdir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def load(run_id: str) -> dict:
    """Run record with its live status resolved."""
    rdir = run_dir(run_id)
    record = _read_json(rdir / "run.json")
    if record is None:
        raise FileNotFoundError(f"no run {run_id}")
    result = _read_json(rdir / "result.json")
    alive = _pid_alive(int(record.get("pid") or 0), run_id)

    if result is not None:
        record["status"] = result.get("status", "done")
        record["exit_code"] = result.get("exit_code")
        record["finished_at"] = result.get("finished_at")
        record["error"] = result.get("error") or record.get("error")
    elif alive:
        record["status"] = "running"
    elif record.get("status") == "running":
        # Pid gone, no result written: killed hard, OOM, or a crash the runner
        # could not survive long enough to report. NOT "done" — the difference
        # between a finished run and a dead one is the whole reason the runner
        # writes result.json rather than the server guessing.
        record["status"] = "died"

    record["alive"] = alive
    # Defaults for a run.json written before these fields existed, so the
    # listing renders every row through one shape instead of the UI checking.
    record.setdefault("tags", [])
    record.setdefault("spec_summary", "")
    log = rdir / "run.log"
    record["log_size"] = log.stat().st_size if log.exists() else 0
    metrics = rdir / "metrics.jsonl"
    record["metrics_size"] = metrics.stat().st_size if metrics.exists() else 0
    record["output_dir"] = str(rdir / "train")
    return record


def list_runs(limit: int = 100) -> list[dict]:
    """Every readable run, newest first. Filtering by kind/status is the route's
    job — this is the whole store."""
    base = runs_dir()
    if not base.exists():
        return []
    out = []
    for rdir in base.iterdir():
        if not rdir.is_dir() or not (rdir / "run.json").exists():
            continue
        try:
            out.append(load(rdir.name))
        except Exception:  # noqa: BLE001, S112 - one unreadable run must not
            continue       # hide the twenty that are fine
    # Newest first by START TIME, not by directory name: run ids are prefixed
    # with the kind, so sorting by name would group every rollout above every
    # train regardless of when they ran.
    out.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return out[:limit]


def stop(run_id: str) -> dict:
    """SIGINT the job's process group, escalating to SIGTERM.

    SIGINT first because LeRobot's training loop and `ProcessSignalHandler` both
    treat it as "wind down": training saves a checkpoint, a rollout returns the
    arm to its initial pose. **SIGKILL is never sent from here** — a half-killed
    rollout leaves an arm under torque in an unknown pose, and a half-killed
    training run throws away the hours since the last checkpoint.
    """
    record = load(run_id)
    pid = int(record.get("pid") or 0)
    if not record.get("alive"):
        return record
    try:
        os.killpg(os.getpgid(pid), signal.SIGINT)
    except (ProcessLookupError, PermissionError, OSError) as e:
        record["error"] = f"could not signal: {e}"
        return record

    deadline = time.time() + STOP_GRACE_S
    while time.time() < deadline:
        time.sleep(0.25)
        if not _pid_alive(pid, run_id):
            break
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass

    record = load(run_id)
    record["stop_requested"] = True
    return record


def tail_log(run_id: str, offset: int = 0, max_bytes: int = 200_000) -> dict:
    """Incremental read. The page passes back the offset it last saw, so a
    multi-hour log is never re-sent."""
    path = run_dir(run_id) / "run.log"
    if not path.exists():
        return {"offset": 0, "text": "", "size": 0}
    size = path.stat().st_size
    offset = max(0, min(int(offset), size))
    # A client that fell far behind gets the TAIL, not a 50 MB response.
    if size - offset > max_bytes:
        offset = size - max_bytes
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(max_bytes)
    return {"offset": offset + len(data), "size": size, "text": data.decode("utf-8", "replace")}


def read_metrics(run_id: str, offset: int = 0) -> dict:
    """Incremental JSONL read. **Only whole lines are consumed**, so a record
    caught mid-write is picked up on the NEXT poll rather than dropped or
    half-parsed.

    Byte offsets, not line counts: the file is appended to by another process
    while this reads it, and a line count would have to re-read from the start
    to mean anything.
    """
    path = run_dir(run_id) / "metrics.jsonl"
    if not path.exists():
        return {"offset": 0, "rows": [], "size": 0}
    size = path.stat().st_size
    offset = max(0, min(int(offset), size))
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    text = data.decode("utf-8", "replace")
    consumed = len(data)
    if text and not text.endswith("\n"):
        cut = text.rfind("\n")
        if cut == -1:
            # Nothing complete yet. The offset does not move, so the partial
            # record is re-read whole on the next poll.
            return {"offset": offset, "rows": [], "size": size}
        consumed = len(text[: cut + 1].encode("utf-8"))
        text = text[: cut + 1]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001, S112 - a corrupt row costs one point
            continue       # on a chart; raising would cost the whole chart
    return {"offset": offset + consumed, "size": size, "rows": rows}


def checkpoints(run_id: str) -> list[dict]:
    """Saved checkpoints of a training run, newest step first.

    LeRobot writes `<output_dir>/checkpoints/<step>/pretrained_model` plus a
    `last` symlink; the rollout launcher wants the model directory, so that is
    what is returned. An entry with no `pretrained_model` is a checkpoint still
    being written and is skipped — offering it would hand a rollout half a file.
    """
    out_dir = run_dir(run_id) / "train" / "checkpoints"
    if not out_dir.exists():
        return []
    found = []
    for entry in out_dir.iterdir():
        model = entry / "pretrained_model"
        if not model.is_dir():
            continue
        found.append({
            "name": entry.name,
            "step": int(entry.name) if entry.name.isdigit() else None,
            "path": str(model),
            "is_link": entry.is_symlink(),
            "modified": entry.stat().st_mtime,
        })
    # Numeric steps descend; `last` and anything else non-numeric sorts after
    # them by name, so the newest real checkpoint is always row one.
    found.sort(key=lambda c: (c["step"] is None, -(c["step"] or 0), c["name"]))
    return found


#: LeRobot writes the training job's fully-resolved config beside the weights,
#: and it is the ONLY record of what a checkpoint was trained on. Verified
#: against the kit's real ACT run 2026-08-27 by walking every key of both
#: files: `config.json` carries the policy ARCHITECTURE and the safetensors
#: carry weights, and **nothing anywhere in a checkpoint records an fps or a
#: control rate**. So the rate a policy was trained at is reachable by exactly
#: one route —
#:
#:     <checkpoint>/train_config.json -> dataset.repo_id
#:                                    -> <that dataset>/meta/info.json -> fps
#:
#: which is a chain and not a preference. That is why a broken link is reported
#: rather than worked around: there is no second source to fall back to, so any
#: fallback would be an invention.
TRAIN_CONFIG = "train_config.json"


def trained_dataset(policy_path: str | Path) -> dict:
    """What the checkpoint at `policy_path` was trained on.

    Returns `{repo_id, episodes, config_path, reason}`. `repo_id` is None
    exactly when the link could not be read, and `reason` then names WHICH link
    broke in the operator's own terms — no checkpoint there, a checkpoint with
    no `train_config.json`, JSON that will not parse, or a config naming no
    dataset.

    **A broken link is not an exception.** A hand-copied checkpoint, or one
    from a run older than this metadata, is a legitimate thing to be holding;
    what the caller does about it is the caller's decision. This function's one
    job is to refuse to invent the answer. Inferring the dataset from the run
    directory or from whatever the operator happens to have selected would
    compare a declared rate against the WRONG dataset's fps and report
    agreement — worse than no check at all, because it reassures (ruled
    2026-08-27).

    `episodes` is LeRobot's own training list and is returned in its original
    ORDER, unsorted and undeduped: the eval split is that list's tail, so
    sorting it here would silently describe a holdout that never happened.
    """
    model_dir = Path(policy_path)
    config_path = model_dir / TRAIN_CONFIG
    unknown = {"repo_id": None, "episodes": None, "config_path": str(config_path)}

    if not model_dir.exists():
        return {**unknown, "reason": f"no checkpoint at {model_dir}"}
    try:
        config = json.loads(config_path.read_text())
    except FileNotFoundError:
        return {**unknown, "reason": (
            f"{model_dir} carries no {TRAIN_CONFIG}, so nothing records which "
            "dataset this policy was trained on"
        )}
    except (OSError, ValueError) as e:
        return {**unknown, "reason": f"{config_path} could not be read: {e}"}

    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return {**unknown, "reason": f"{config_path} carries no 'dataset' block"}
    repo_id = str(dataset.get("repo_id") or "").strip()
    if not repo_id:
        return {**unknown, "reason": f"{config_path} names no dataset.repo_id"}

    episodes = dataset.get("episodes")
    return {
        "repo_id": repo_id,
        "episodes": [int(e) for e in episodes] if isinstance(episodes, list) else None,
        "config_path": str(config_path),
        "reason": "",
    }


def delete_run(run_id: str) -> dict:
    """Remove a run directory. There is no undo.

    RAISES `RuntimeError` while the run is alive (the route turns that into a
    409): deleting the directory out from under a live child leaves it writing
    into an unlinked file and takes its log with it, so there would be nothing
    left to explain what happened. Stop it first.
    """
    rdir = run_dir(run_id)              # RUN_ID_RE + containment
    base = runs_dir().resolve()
    resolved = rdir.resolve()
    # Re-checked here rather than trusted from run_dir: this is the one call
    # that removes a tree, and the cost of the second check is a stat.
    if resolved == base or base not in resolved.parents:
        raise ValueError(f"refusing to delete outside the run store: {run_id!r}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"no run {run_id}")

    record = _read_json(rdir / "run.json") or {}
    if _pid_alive(int(record.get("pid") or 0), run_id):
        raise RuntimeError(
            f"{run_id} is still running — stop it before deleting it"
        )
    shutil.rmtree(resolved)
    return {"id": run_id, "path": str(rdir), "deleted": True}


def write_result(run_dir_path: str | Path, status: str, exit_code: int = 0,
                 error: str = "") -> None:
    """Called by a runner in its `finally` — this is what makes a finished run
    distinguishable from a killed one.

    Lives here, not in `runners/`, because both sides need it: the child writes
    the file and `load()` reads it, and a second spelling of these four keys is
    a run that reads as `died` after a clean finish.
    """
    payload = {
        "status": status,
        "exit_code": exit_code,
        "error": error,
        "finished_at": _now(),
    }
    Path(run_dir_path, "result.json").write_text(json.dumps(payload, indent=2) + "\n")
