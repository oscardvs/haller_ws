# hmi/backend/tests/lab/test_routes_system.py
"""`/lab/system` — the route that must answer on a box where nothing is set up.

Two things are under test and they pull in opposite directions.

**It must never import torch into THIS process.** That is the whole design:
the serving process owns the Feetech bus, `import torch` is banned there, and
the answer comes from a child interpreter. So no assertion below ever states
whether torch is importable in the test process — the value of
`torch_available` under a real interpreter is a fact about which venv pytest
happens to be running in, and a test asserting it would be asserting about the
box rather than about the route. (Measured 2026-08-27: `~/venvs/haller-hmi`
carries torch 2.10.0+cu128 and `~/venvs/haller-lab` 2.11.0+cu130, so the
serving venv answers `True` here — see the report; the shapes are what is
pinned.) What IS asserted is that answering the route leaves `sys.modules`
alone.

**It must be a 200 in every broken state.** A fresh checkout has no
`~/venvs/haller-lab` and possibly no `$HF_LEROBOT_HOME` either, and this is the
page that says so. Missing interpreter, hung interpreter, non-zero exit,
garbage on stdout, absent dataset root: every one of them is a 200 below, and a
500 in any of them makes the page look broken instead of telling the operator
to run `scripts/setup_lab_venv.sh`.

The stand-in interpreters are `/bin/sh` scripts under `tmp_path`, never
mocks of `subprocess.run`. The probe's spawn, its timeout, its exit code and
its stdout parse are four separate things that can be wrong, and a fake that
returns a `CompletedProcess` proves only that the code calls the function it
calls. Spawns are counted by the script APPENDING TO A FILE, so the cache test
counts real processes.

`$HALLER_LAB_PYTHON` is pointed somewhere harmless by an autouse fixture: the
real `~/venvs/haller-lab` exists on this box, and a test that fell through to
it would spend a second importing CUDA to assert something it did not mean.

The app is built here rather than imported from `server.py`, as
`test_routes_datasets.py` does, so a failure names this routes module.
"""
from __future__ import annotations

import ast
import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from haller_hmi.api.deps import LabDeps
from haller_hmi.lab import routes_system, runs
from haller_hmi.lab.routes_system import build_system_router

#: The response, EXACTLY. The first four are the frozen contract; the last two
#: are the additions this chunk was asked for — `runner_python_exists` because
#: a path without "and it is there" answers the wrong half of "why did my run
#: die instantly", and `lerobot_version` because the probe is already running.
SYSTEM_KEYS = frozenset({
    "disk_free_bytes", "lerobot_home", "runner_python",
    "runner_python_exists", "torch_available", "lerobot_version",
})

#: A plausible client on Oscar's LAN — the Quest, or the laptop.
LAN_HOST = "192.168.1.50"

#: `TestClient` defaults to `client=("testclient", 50000)`, which is NOT
#: loopback (`api/gate.py` documents this). Every client below says which it is
#: so the ungated assertion means something.
LOOPBACK_HOST = "127.0.0.1"

#: What a healthy lab venv's probe prints.
GOOD_PROBE = '{"torch": true, "lerobot_version": "0.6.1"}'


# ---- app ----

def _client(
    home: Path,
    *,
    host: str = LOOPBACK_HOST,
    allow_remote_control: bool = False,
) -> TestClient:
    """`build_system_router` on a throwaway app, driven from `host`."""
    deps = LabDeps(
        get_cameras=lambda: None,
        get_recorder=lambda: None,
        lerobot_home=lambda: home,
        allow_remote_control=lambda: allow_remote_control,
    )
    app = FastAPI()
    app.include_router(build_system_router(deps))
    return TestClient(app, client=(host, 51000))


def _system(client: TestClient) -> dict:
    response = client.get("/lab/system")
    assert response.status_code == 200, response.text
    return response.json()


# ---- stand-in interpreters ----

def _interpreter(tmp_path: Path, name: str, script: str) -> Path:
    """An executable `/bin/sh` stand-in for a Python interpreter.

    The probe invokes it as `<path> -c <source>`; a shell script ignores both
    arguments, which is what makes it a usable stand-in without a Python in it.
    """
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(0o755)
    return path


def _counting_interpreter(
    tmp_path: Path, name: str, stdout: str = GOOD_PROBE,
) -> tuple[Path, Path]:
    """A stand-in that records every invocation, and the tally file to read.

    A file rather than a patched `subprocess.run`: what the cache has to
    prevent is a PROCESS per poll, so the count has to come from processes.
    """
    tally = tmp_path / f"{name}.calls"
    path = _interpreter(tmp_path, name, (
        f"echo run >> '{tally}'\n"
        f"cat <<'PROBE_EOF'\n{stdout}\nPROBE_EOF\n"
    ))
    return path, tally


def _spawns(tally: Path) -> int:
    return len(tally.read_text().splitlines()) if tally.exists() else 0


# ---- fixtures ----

@pytest.fixture(autouse=True)
def cold_probe_cache():
    """No probe answer survives into or out of a test.

    The cache is a module global with a TTL of five minutes, so without this
    the second test in the file would answer out of the first one's
    interpreter — and the cache-keying test would pass for the wrong reason.
    """
    routes_system._reset_probe_cache()
    yield
    routes_system._reset_probe_cache()


@pytest.fixture(autouse=True)
def lab_python(tmp_path, monkeypatch) -> Path:
    """`$HALLER_LAB_PYTHON` at a path that does not exist, for every test.

    Autouse and unconditional: `runs.runner_python()` falls back to
    `~/venvs/haller-lab/bin/python`, which is REAL on this box, so a test that
    forgot to point it somewhere would spawn the actual lab interpreter and
    import CUDA. Returned as well as set, because "the interpreter is missing"
    is itself one of the cases under test.
    """
    missing = tmp_path / "venvs" / "haller-lab" / "bin" / "python"
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(missing))
    return missing


@pytest.fixture
def home(tmp_path) -> Path:
    """An existing, empty dataset cache root."""
    root = tmp_path / "lerobot"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def client(home) -> TestClient:
    return _client(home)


@pytest.fixture
def no_spawn(monkeypatch):
    """Make any subprocess spawn a test failure.

    Used only where the point is that NOTHING was spawned; every other test
    lets the real spawn happen.
    """
    def _refuse(*args, **kwargs):
        raise AssertionError(f"the route spawned a process: {args!r}")

    monkeypatch.setattr(routes_system.subprocess, "run", _refuse)


# ============================================================================
# the happy path
# ============================================================================

def test_the_response_carries_the_frozen_keys_and_only_those(
    home, client, monkeypatch,
):
    """`$HALLER_LAB_PYTHON` at the CURRENT interpreter: a real spawn, a real
    parse, a real answer.

    `torch_available` is checked for TYPE and never for value — see the module
    docstring. Whether pytest's own venv can import torch is a fact about the
    box, and this route exists precisely so that fact is never established by
    importing it here.
    """
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, sys.executable)

    body = _system(client)

    assert set(body) == set(SYSTEM_KEYS), (
        f"unexpected {sorted(set(body) - SYSTEM_KEYS)}, "
        f"missing {sorted(SYSTEM_KEYS - set(body))}"
    )
    assert body["runner_python"] == sys.executable
    assert body["runner_python_exists"] is True
    assert body["lerobot_home"] == str(home.resolve())
    assert isinstance(body["torch_available"], bool)
    assert body["lerobot_version"] is None or isinstance(body["lerobot_version"], str)
    assert body["disk_free_bytes"] > 1024


def test_answering_the_route_never_imports_torch_into_this_process(
    client, monkeypatch,
):
    """The reason the probe is a subprocess at all.

    Compared before/after rather than asserted absent: another test in the run
    may legitimately have imported torch already, and what this guards is that
    THIS request is not what does it. `sys.executable` is used on purpose — an
    interpreter that really has torch, probed for real.
    """
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, sys.executable)
    before = "torch" in sys.modules

    _system(client)

    assert ("torch" in sys.modules) is before


def test_the_handler_is_a_plain_def(home):
    """A coroutine handler would do the probe's subprocess wait ON the event
    loop that forwards teleop frames to the arms. Plain defs run on a worker
    thread; `routes_datasets.py` applies the same rule to every handler.

    Read off the ROUTER rather than off an app it is mounted on: this FastAPI
    keeps an included router as one opaque `_IncludedRouter` entry in
    `app.routes` and never flattens it, so the app has no `/lab/system` path to
    find.
    """
    router = build_system_router(LabDeps(
        get_cameras=lambda: None,
        get_recorder=lambda: None,
        lerobot_home=lambda: home,
    ))

    endpoints = [r.endpoint for r in router.routes
                 if getattr(r, "path", "") == "/lab/system"]

    assert endpoints, "the router has no /lab/system route"
    assert not any(asyncio.iscoroutinefunction(fn) for fn in endpoints)


def test_the_route_is_ungated_from_the_lan(home):
    """A GET that starts nothing and writes nothing. It is the first thing
    wanted from inside the headset when a run dies on launch, so it is not on
    the `require_local` list."""
    remote = _client(home, host=LAN_HOST, allow_remote_control=False)

    assert remote.get("/lab/system").status_code == 200


# ============================================================================
# the probe: every failure is False, never a status
# ============================================================================

def test_a_missing_lab_venv_is_a_200_that_says_the_interpreter_is_gone(
    client, lab_python, no_spawn,
):
    """The fresh-checkout state, and the single most likely reason a train run
    dies in its first second. It must read as "not set up", not as an error.

    `no_spawn` is the second half: a path that does not exist is answered by one
    `stat`, not by a process that cannot start.
    """
    body = _system(client)

    assert body["runner_python"] == str(lab_python)
    assert body["runner_python_exists"] is False
    assert body["torch_available"] is False
    assert body["lerobot_version"] is None


def test_a_probe_that_hangs_is_false_rather_than_a_hung_request(
    tmp_path, client, monkeypatch,
):
    """`exec sleep`, so the timeout kills ONE process and leaves no orphan.

    The elapsed bound is the assertion that matters: without the timeout this
    request blocks a worker thread for thirty seconds, and the page polls.
    """
    sleeper = _interpreter(tmp_path, "sleeper", "exec sleep 30\n")
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(sleeper))
    monkeypatch.setattr(routes_system, "PROBE_TIMEOUT_S", 0.5)

    started = time.monotonic()
    body = _system(client)
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, f"the timeout did not bound the probe ({elapsed:.1f}s)"
    assert body["runner_python_exists"] is True
    assert body["torch_available"] is False
    assert body["lerobot_version"] is None


def test_a_probe_that_exits_nonzero_is_false(tmp_path, client, monkeypatch):
    """A lab venv whose torch is installed and does not import — a CUDA/driver
    mismatch — exits non-zero with a traceback. False, and the traceback goes
    to the server log rather than onto the wire."""
    broken = _interpreter(
        tmp_path, "broken", "echo 'ImportError: libcudart.so.13' >&2\nexit 1\n")
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(broken))

    body = _system(client)

    assert body["runner_python_exists"] is True
    assert body["torch_available"] is False
    assert body["lerobot_version"] is None


def test_a_probe_that_prints_no_json_is_false(tmp_path, client, monkeypatch):
    """Exit 0 is not the same as an answer. Something on `$PATH` named `python`
    that is not one still gets a False, not a 500 out of `json.loads`."""
    chatty = _interpreter(tmp_path, "chatty", "echo 'Python 3.12.3'\n")
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(chatty))

    body = _system(client)

    assert body["runner_python_exists"] is True
    assert body["torch_available"] is False


def test_a_banner_ahead_of_the_payload_still_parses(tmp_path, client, monkeypatch):
    """Only the LAST line is the payload. A venv can print on startup — this
    box shadows Debian's numpy-1 builds specifically to stop one such wall of
    text — and a banner must not read as a broken probe."""
    noisy, _ = _counting_interpreter(
        tmp_path, "noisy", stdout=f"a numpy warning\n{GOOD_PROBE}")
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(noisy))

    body = _system(client)

    assert body["torch_available"] is True
    assert body["lerobot_version"] == "0.6.1"


def test_a_bare_interpreter_name_on_the_path_counts_as_present(
    tmp_path, client, monkeypatch,
):
    """`$HALLER_LAB_PYTHON` can hold a bare NAME, and reachably so:
    `scripts/setup_lab_venv.sh:26` reads the same variable to mean the base
    interpreter to build the venv FROM, defaulting to `python3.12`. `Popen`
    resolves that on `$PATH` and `runs.launch` would spawn it, so
    `Path.exists()` alone would report a runnable interpreter as missing and
    point at the wrong defect — it is present, it is the WRONG one, and
    `torch_available` is the field that says so."""
    python, tally = _counting_interpreter(
        tmp_path, "python3.12", stdout='{"torch": false, "lerobot_version": null}')
    monkeypatch.setenv("PATH", f"{tmp_path}:{python.parent}")
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, python.name)

    body = _system(client)

    assert body["runner_python"] == python.name       # what is configured
    assert body["runner_python_exists"] is True       # ...and it is runnable
    assert body["torch_available"] is False           # ...and it is not the lab venv
    assert _spawns(tally) == 1


# ============================================================================
# the cache
# ============================================================================

def test_the_probe_runs_once_across_two_requests(tmp_path, client, monkeypatch):
    """The page polls this route. One spawn per poll is its own problem, and
    the count comes from real processes rather than from a patched
    `subprocess.run` — a process per poll is exactly what is being prevented."""
    python, tally = _counting_interpreter(tmp_path, "python")
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(python))

    first = _system(client)
    second = _system(client)

    assert _spawns(tally) == 1, f"probed {_spawns(tally)} times, expected 1"
    # Only the PROBE-derived fields are expected to be identical. `disk_free_bytes`
    # is a live reading of a disk three other sessions are writing to, and
    # comparing whole responses made this fail on a 12 KB drift between two
    # calls — a flake that says nothing about whether the probe was cached.
    probed = ("runner_python", "runner_python_exists", "torch_available",
              "lerobot_version")
    assert {k: first[k] for k in probed} == {k: second[k] for k in probed}
    assert first["torch_available"] is True
    assert first["lerobot_version"] == "0.6.1"


def test_the_cache_is_keyed_by_the_interpreter_path(tmp_path, client, monkeypatch):
    """`runner_python()` re-reads `$HALLER_LAB_PYTHON` every call, so a cache
    keyed by nothing would report interpreter A's torch under interpreter B's
    path — a wrong answer, not a stale one."""
    first_py, first_tally = _counting_interpreter(tmp_path, "first")
    second_py, second_tally = _counting_interpreter(
        tmp_path, "second", stdout='{"torch": false, "lerobot_version": "0.5.1"}')

    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(first_py))
    before = _system(client)
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(second_py))
    after = _system(client)

    assert _spawns(first_tally) == 1
    assert _spawns(second_tally) == 1
    assert before["runner_python"] == str(first_py)
    assert before["torch_available"] is True
    assert after["runner_python"] == str(second_py)
    assert after["torch_available"] is False
    assert after["lerobot_version"] == "0.5.1"


def test_a_failed_probe_is_retried_sooner_than_a_good_one(
    tmp_path, client, monkeypatch,
):
    """`scripts/setup_lab_venv.sh` is run WHILE the HMI is up — that is the
    recovery this route exists to point at. A process-lifetime cache would keep
    the page saying "no lab venv" until someone restarted the server that owns
    the servo bus, so a False is held only briefly and a True for far longer."""
    assert routes_system.PROBE_RETRY_S < routes_system.PROBE_TTL_S

    python, tally = _counting_interpreter(
        tmp_path, "python", stdout='{"torch": false, "lerobot_version": null}')
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(python))
    monkeypatch.setattr(routes_system, "PROBE_RETRY_S", 0.0)

    assert _system(client)["torch_available"] is False
    assert _system(client)["torch_available"] is False

    assert _spawns(tally) == 2, "a failed probe was cached like a good one"


# ============================================================================
# disk and the dataset root
# ============================================================================

def test_disk_free_bytes_is_a_positive_int_the_filesystem_agrees_with(
    home, client,
):
    """The number that answers "can I record another session":
    `so101_pick_cube` is 46 episodes / 29 500 frames in 709 MiB, about 24
    minutes of recording per GiB."""
    body = _system(client)

    free = body["disk_free_bytes"]

    assert isinstance(free, int)
    assert free > 1024                                   # no bool sneaks through
    assert free <= shutil.disk_usage(home).total


def test_a_dataset_root_that_does_not_exist_still_reports_free_space(tmp_path):
    """`shutil.disk_usage` raises `FileNotFoundError` on a path nothing has
    created, and a fresh checkout is exactly that. Free space is a property of
    the mount, so the nearest live ancestor answers the same question — for the
    recording that is about to create the directory."""
    absent = tmp_path / "never" / "created" / "lerobot"
    body = _system(_client(absent))

    assert body["lerobot_home"] == str(absent)
    assert body["disk_free_bytes"] > 1024


# ============================================================================
# the package ban
# ============================================================================

def test_importing_routes_system_pulls_in_neither_lerobot_nor_torch():
    """`lab/` is imported by the serving process, which is the teleop latency
    path. A subprocess, because pytest has already imported half the world into
    this one and `sys.modules` here would prove nothing.

    This module is the one that has a REASON to import torch and must not.
    """
    probe = ("import sys; import haller_hmi.lab.routes_system as m; "
             "print('torch' in sys.modules, 'lerobot' in sys.modules, bool(m))")
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True, timeout=120,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert out.stdout.strip() == "False False True", out.stderr


def test_no_import_statement_in_the_module_names_torch_or_lerobot():
    """The neighbours scan their source for the substring `import torch`. This
    module CANNOT be checked that way and must not be "fixed" to be: the
    substring is there on purpose, inside `PROBE_SOURCE`, which is the program
    the OTHER interpreter runs. So the check is on the parse tree — every
    `import` this module actually performs — and both halves are asserted, so
    the day someone replaces the subprocess with a real import the test that
    catches it is this one."""
    source = Path(routes_system.__file__).read_text()
    assert "import torch" in source, "PROBE_SOURCE stopped asking about torch"

    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])

    assert not imported & {"torch", "lerobot"}, sorted(imported)


def test_the_lerobot_home_is_resolved(tmp_path):
    """`~/.cache/huggingface/lerobot` is a SYMLINK to `~/robot-data/lerobot` on
    this box, and `catalog.hf_home()` resolves. A page printing the other
    spelling is how a path that is obviously right gets compared and rejected —
    the `relative_to` "is not in the subpath of" failure, one surface up."""
    real = tmp_path / "robot-data" / "lerobot"
    real.mkdir(parents=True)
    link = tmp_path / "link-to-lerobot"
    link.symlink_to(real)

    body = _system(_client(link))

    assert body["lerobot_home"] == str(real.resolve())
