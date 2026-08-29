# hmi/backend/haller_hmi/lab/routes_system.py
"""`/lab/system` — the preflight numbers, and the one import this route refuses.

One GET, and most of its design is about what it must NOT do.

    {disk_free_bytes, lerobot_home, runs_dir, runner_python,
     runner_python_exists, torch_available, lerobot_version,
     compare_max_runs, compare_max_keys}

**`torch_available` is answered by a CHILD PROCESS, never by importing torch.**
This module runs in the serving process, which owns the Feetech bus and the
teleop latency path, and `import torch` there is banned outright — it is the
whole reason `~/venvs/haller-lab` exists. Importing it to answer a page's
preflight question would also pull a CUDA context into the process holding the
arms, and cost seconds on a worker thread doing it. So the LAB interpreter is
asked about itself, over a pipe, and the answer is CACHED. `runs.runner_python`
documents the same refusal from the other side: it is a path check and never an
import.

## The compare caps are published, not guessed

`compare_max_runs` and `compare_max_keys` are `lab/compare.py`'s own
`MAX_RUNS` / `MAX_KEYS`, READ here rather than re-declared. `series()` raises
above either, so a client that does not know them can only discover the cliff
by being refused: a real 60k-step ACT run logs 13 numeric keys per row — 12
chartable once `steps` is spent as the x-axis, and 15 distinct across the run —
against a cap of 8, so Compare's first honest request is a 400.

The alternative was the frontend hardcoding 8, which is the copied-constant
failure this port has already paid for twice (the rate gate, and the HUD
comment that was true when written). A published cap lets the page batch
`ceil(keys / cap)` requests and stay correct at any future value, including one
nobody remembers to tell it about.

Importing `compare` costs nothing this module was avoiding: it is `math` and
`lab/runs` and no more, so the stdlib-only property below is intact.

There is no `_warm_pandas()` here and none is needed — invariant 5c is closed by
this module importing nothing outside the stdlib at all, at module scope or
anywhere else. That is not an accident to be optimised away: the one heavy
import this route could plausibly make is exactly the one that must never happen
in this process, so it happens in a different interpreter or not at all.

## Why every failure is `False` and never a status

A box with no lab venv is a NORMAL state — a fresh checkout has one until
`scripts/setup_lab_venv.sh` runs — and this is the route that tells the operator
so. A 500 there makes the page look broken instead, which is the opposite of the
job. A missing interpreter, a spawn error, a timeout, a non-zero exit and
unparseable output therefore all mean `torch_available: False`, with the reason
logged at WARNING for whoever is reading the server log. `runner_python_exists`
is reported beside the path for the same reason: a missing lab venv is the most
likely reason a train run dies in the first second, and a path with no
"...and it is there" beside it answers the wrong half of the question.

## The cache is a TTL, not process-lifetime

A probe per poll on a route the page polls is its own problem, so the answer is
cached. It is a TTL rather than forever because the recovery this route exists
to point at — running `scripts/setup_lab_venv.sh` — happens while the HMI is
UP, and a process-lifetime cache would keep the page saying "no lab venv" until
someone restarted the server that owns the servo bus. This page must never be
the reason for that restart.

Hence the asymmetry: a good answer is held for `PROBE_TTL_S` (the lab venv does
not change under a running server), a bad one only for `PROBE_RETRY_S` (it is
about to). `exists`, `torch_available` and `lerobot_version` are cached as ONE
observation so the response can never mix a fresh `exists: true` with a stale
`torch_available: false` and read as a broken venv rather than a missing one.

The cache is keyed by the interpreter PATH: `runner_python()` re-reads
`$HALLER_LAB_PYTHON` on every call, and reporting interpreter A's torch under
interpreter B's path is a wrong answer, not a stale one.

Nothing is probed at router-build time. `build_lab_router` is called at import
in `server.py`, and a torch import spawned there would spend a second of CPU
during the bring-up that energises the arms, to pre-answer a question no client
has asked yet.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter

from ..api.deps import LabDeps
from ..api.errors import as_http
from . import compare
from . import runs as runs_mod

logger = logging.getLogger(__name__)

#: What the LAB interpreter is asked, as a single `-c` program. It imports
#: torch for real — "can this interpreter import torch" has no cheaper honest
#: answer, and a broken CUDA install is a torch that is INSTALLED and does not
#: import, which is precisely the state a train run dies of.
#:
#: lerobot's version comes from the distribution METADATA rather than from
#: `import lerobot`, which answers "which lerobot would the runner get" even
#: when importing it is what is broken — and that is the failure worth naming.
PROBE_SOURCE = """\
import json
report = {"torch": False, "lerobot_version": None}
try:
    import torch
    report["torch"] = True
except BaseException:
    pass
try:
    from importlib.metadata import version
    report["lerobot_version"] = version("lerobot")
except BaseException:
    pass
print(json.dumps(report))
"""

#: Seconds before the probe is abandoned. `import torch` in `~/venvs/haller-lab`
#: measures 0.98 s warm on this box (2.11.0+cu130, RTX 4080 SUPER); the number
#: to size against is a COLD page cache, which is the same first-import cost a
#: training child pays. 20 s is far above that and far below the point where a
#: hung interpreter has cost anything but one worker thread, once — the lock
#: below is what keeps a poll loop from spending one thread per poll on it.
PROBE_TIMEOUT_S = 20.0

#: How long a successful probe is trusted. The lab venv does not change under a
#: running server; five minutes is "effectively cached" without being forever.
PROBE_TTL_S = 300.0

#: How long a FAILED probe is trusted. Short on purpose: this is the window
#: between `scripts/setup_lab_venv.sh` finishing and the page saying so.
PROBE_RETRY_S = 15.0

#: One probe at a time. Without it a page polling every 2 s through a 20 s
#: timeout spawns ten interpreters before the first one caches its answer.
_probe_lock = threading.Lock()

#: `(interpreter path, monotonic expiry, answer)`, or None. Read outside the
#: lock and rechecked inside it; a tuple assignment is atomic, so a racing
#: reader sees the old answer or the new one and never half of either.
_probe_cache: tuple[str, float, dict] | None = None


def _reset_probe_cache() -> None:
    """Forget the cached probe. For tests; nothing in the server calls it."""
    global _probe_cache
    with _probe_lock:
        _probe_cache = None


def _runnable(python: Path) -> str | None:
    """What `runs.launch` would actually execute, or None if nothing would.

    `Path.exists()` alone is the wrong question, and reachably so:
    `runner_python()` returns `$HALLER_LAB_PYTHON` verbatim, and that variable
    is ALSO read by `scripts/setup_lab_venv.sh:26` with a different meaning —
    the base interpreter to build the venv FROM, defaulting to the bare name
    `python3.12`. A shell that exported it to build the venv and then started
    the HMI hands `runner_python()` a bare name. `Popen` resolves that on
    `$PATH` and launches it happily (measured here: `python3.12` resolves to the
    SERVING venv), so reporting "the interpreter is missing" would name the
    wrong defect — the interpreter is there, it is the wrong one, and
    `torch_available` is what says so.
    """
    if python.exists():
        return str(python)
    return shutil.which(str(python))


def _probe(python: Path) -> dict:
    """Ask `python` about itself in a child. NEVER raises, never returns None.

    Returns `{"exists", "torch_available", "lerobot_version"}` — one
    observation of one interpreter, cached as a unit by `_probe_cached`.
    """
    answer = {"exists": False, "torch_available": False, "lerobot_version": None}
    runnable = _runnable(python)
    if runnable is None:
        # The cheapest possible "no lab venv": no spawn at all. This is the
        # fresh-checkout state and the most common non-answer, so it costs a
        # stat rather than a process that cannot start.
        return answer
    answer["exists"] = True

    try:
        done = subprocess.run(
            [runnable, "-c", PROBE_SOURCE],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
            check=False,
            # The environment is INHERITED, exactly as `runs.launch` gives it to
            # a real child. A `PYTHONPATH` that shadows the lab venv's torch is
            # then reported rather than hidden — sanitising only the probe would
            # make it answer about an interpreter no run will ever have.
        )
    except (OSError, subprocess.SubprocessError) as e:
        # TimeoutExpired is a SubprocessError; so is anything else Popen can
        # raise. All of them mean False, and all of them are worth a log line
        # because the wire says only False.
        logger.warning("lab: torch probe under %s did not complete (%s)", python, e)
        return answer

    if done.returncode != 0:
        logger.warning(
            "lab: torch probe under %s exited %s (%s)",
            python, done.returncode, (done.stderr or "").strip()[-400:])
        return answer

    try:
        # LAST line only: a venv can print on startup (this box shadows Debian's
        # numpy-1 builds precisely to stop one such wall of text), and a banner
        # ahead of the payload must not read as a broken probe.
        report = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        logger.warning("lab: torch probe under %s printed no JSON (%s)", python, e)
        return answer

    answer["torch_available"] = bool(report.get("torch"))
    version = report.get("lerobot_version")
    answer["lerobot_version"] = str(version) if version else None
    return answer


def _probe_cached(python: Path) -> dict:
    """`_probe`, at most once per `PROBE_TTL_S` per interpreter path."""
    global _probe_cache
    key = str(python)

    cached = _probe_cache
    if cached is not None and cached[0] == key and time.monotonic() < cached[1]:
        return cached[2]

    with _probe_lock:
        # Rechecked under the lock: every request that queued behind a 20 s
        # timeout is here, and each one spawning its own child afterwards would
        # defeat the lock it just waited on.
        cached = _probe_cache
        if cached is not None and cached[0] == key and time.monotonic() < cached[1]:
            return cached[2]
        answer = _probe(python)
        ttl = PROBE_TTL_S if answer["torch_available"] else PROBE_RETRY_S
        _probe_cache = (key, time.monotonic() + ttl, answer)
    return answer


def _disk_free_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding `path`, or its nearest live ancestor.

    The walk up is the fresh-checkout case rather than an edge case:
    `$HF_LEROBOT_HOME` can name a directory nothing has created yet, and
    `shutil.disk_usage` raises `FileNotFoundError` on it — a 500 on the route
    whose job is to say "this box is not set up yet". Free space is a property
    of the MOUNT, so the nearest existing ancestor answers the same question,
    and answers it correctly for the recording about to create that directory.

    The number is what answers "can I record another session": `so101_pick_cube`
    is 46 episodes / 29 500 frames of 30 fps 640×480 in 709 MiB, i.e. ~24 minutes
    of recording per GiB, against 653 GiB free here.
    """
    for candidate in (path, *path.parents):
        try:
            return int(shutil.disk_usage(candidate).free)
        except OSError:
            continue
    return 0  # pragma: no cover - `/` would have to be unstattable


def build_system_router(deps: LabDeps) -> APIRouter:
    """Wire `GET /lab/system` onto its own router.

    UNGATED, deliberately: it is a GET that starts nothing and writes nothing,
    and it is the first thing Oscar wants to see from inside the headset when a
    run dies on launch. `require_local` is for the destructive and
    process-starting routes (`api/gate.py`).
    """
    router = APIRouter()

    # No `-> dict` annotation, for `routes_datasets.build_datasets_router`'s
    # reason: FastAPI turns one into `response_model=dict` and revalidates the
    # payload on the way out.

    @router.get("/lab/system")
    def get_system():
        """Disk, dataset root, runner interpreter, and what it can import.

        A plain `def`, so the probe's subprocess wait happens on a worker thread
        and never on the event loop that forwards teleop frames to the arms.
        """
        with as_http():
            # Resolved here and not in `deps.home()`, which documents why it
            # leaves that to its callers: `~/.cache/huggingface/lerobot` is a
            # symlink to `~/robot-data/lerobot` on this box, and the page
            # printing one spelling while `catalog.hf_home()` uses the other is
            # how a path that is obviously right gets compared and rejected.
            home = deps.home().resolve()
            python = runs_mod.runner_python()
            probe = _probe_cached(python)
            return {
                "disk_free_bytes": _disk_free_bytes(home),
                "lerobot_home": str(home),
                # Resolved for `lerobot_home`'s reason — one spelling, so the
                # page and the server never compare two names for one
                # directory. `$HALLER_RUNS` moves it, so it is not derivable
                # from anything else the payload carries.
                "runs_dir": str(runs_mod.runs_dir().resolve()),
                "runner_python": str(python),
                "runner_python_exists": probe["exists"],
                "torch_available": probe["torch_available"],
                "lerobot_version": probe["lerobot_version"],
                # `lab/compare.py`'s own caps. Re-declaring them here would be
                # a third copy of a number two surfaces already have to agree on.
                "compare_max_runs": compare.MAX_RUNS,
                "compare_max_keys": compare.MAX_KEYS,
            }

    return router
