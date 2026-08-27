# hmi/backend/tests/lab/test_routes_compose.py
"""`build_lab_router` — the COMPOSITION, which nothing else covers.

Every sub-router has its own suite and each one mounts itself on a throwaway
`FastAPI()`, so each proves its own routes work in isolation. None of them can
notice the two failures that live in `lab/routes.py` alone:

* **A sub-router silently missing from the compose.** `server.py` mounts one
  router and never names the three, which is the point of that file — and the
  cost is that dropping an `include_router` line produces a smaller app that
  starts cleanly, answers every route it still has, and 404s the rest. So the
  expected `(method, path)` set is written out IN FULL below and compared for
  EQUALITY, not for containment: a missing block fails, and so does a route
  that appeared without anyone updating the contract.

* **The same path mounted twice.** Two routers claiming one path is not an
  error in Starlette — the first declaration wins, silently, and which one that
  is depends on the order of three lines in a file nobody re-reads. A duplicate
  is therefore asserted against directly rather than hoped about.

Two behaviours the docstring in `lab/routes.py` argues for are pinned here
because the argument is the only thing currently holding them:

* **The zero-arg callables are resolved PER REQUEST.** `server.py` mounts at
  IMPORT time and builds the `CameraManager` and `DatasetRecorder` inside
  `lifespan`; a router that closed over the VALUES would capture `None` and 503
  forever. The tests below build the app once with a handle that is `None`, get
  the 503, then fill the handle and get a 200 from the SAME app — which is
  exactly the startup sequence, and which no test that passes a ready fake at
  build time can distinguish from the broken version.

* **`allow_remote_control=None` falls through to the environment per call.**
  Caching it would mean a restart to change the answer, and the moment that
  matters is the one where Oscar is holding the headset. So the env var is
  flipped BETWEEN two requests on one app, in both directions.

Everything is built with `_dataset.make_dataset` under `tmp_path`, and
`HALLER_RUNS` / `HALLER_LAB_PYTHON` are redirected for the whole file: this
file deletes a dataset and a run, and the real trees are the equivalence
anchors on a box with no backup of any kind.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from haller_hmi.api import gate
from haller_hmi.lab import catalog, routes_system, runs
from haller_hmi.lab.routes import build_lab_router

from ._dataset import make_dataset

#: The one dataset these tests build.
REPO = "local/smoke"

#: A plausible client on Oscar's LAN — the Quest, or the laptop.
LAN_HOST = "192.168.1.50"

#: `TestClient` defaults to `client=("testclient", 50000)`, which is NOT
#: loopback and so trips `require_local` (`api/gate.py` documents this). Every
#: client below says which it is, so an ungated assertion means something.
LOOPBACK_HOST = "127.0.0.1"

# ---- the expected surface, written out rather than derived -----------------
# Derived from the sub-routers themselves this would assert that the compose
# equals the compose. These four blocks are the frozen HTTP contract in
# `docs/port/trackb-lab-contract.md`, transcribed, so a route that moves
# between sub-routers still passes and a route that vanishes does not.

DATASET_ROUTES = frozenset({
    ("GET", "/lab/datasets"),
    ("GET", "/lab/datasets/detail"),
    ("GET", "/lab/datasets/episodes"),
    ("GET", "/lab/datasets/trace"),
    ("GET", "/lab/datasets/video"),
    ("GET", "/lab/datasets/split"),
    ("POST", "/lab/datasets/mark"),
    ("POST", "/lab/datasets/bulk"),
    ("POST", "/lab/datasets/autoclass/preview"),
    ("POST", "/lab/datasets/autoclass/apply"),
    ("POST", "/lab/datasets/autoclass/revert"),
    ("POST", "/lab/datasets/prune"),
    ("DELETE", "/lab/datasets"),
})

#: The four compat paths `build_lab_router` carries because it REPLACES
#: `build_data_router` outright. They are at their old URLs on purpose; the
#: page that calls them is the recorder HUD and it was not ported.
LEGACY_ROUTES = frozenset({
    ("POST", "/cameras/{camera_id}/record"),
    ("GET", "/record/episodes"),
    ("GET", "/record/repos"),
    ("DELETE", "/record/episodes/last"),
})

RUN_ROUTES = frozenset({
    ("GET", "/lab/runs"),
    ("GET", "/lab/runs/metrics"),
    ("POST", "/lab/runs/train"),
    ("GET", "/lab/runs/{run_id}"),
    ("GET", "/lab/runs/{run_id}/metrics"),
    ("GET", "/lab/runs/{run_id}/log"),
    ("GET", "/lab/runs/{run_id}/checkpoints"),
    ("POST", "/lab/runs/{run_id}/stop"),
    ("DELETE", "/lab/runs/{run_id}"),
})

SYSTEM_ROUTES = frozenset({
    ("GET", "/lab/system"),
})

EXPECTED_ROUTES = DATASET_ROUTES | LEGACY_ROUTES | RUN_ROUTES | SYSTEM_ROUTES

#: Named blocks, so a dropped `include_router` fails a test that says WHICH
#: sub-router went missing instead of diffing 27 tuples.
ROUTE_BLOCKS = (
    ("datasets", DATASET_ROUTES),
    ("legacy", LEGACY_ROUTES),
    ("runs", RUN_ROUTES),
    ("system", SYSTEM_ROUTES),
)


# ---- the handles `server.py` fills in `lifespan` ----------------------------

class _Slot:
    """A handle that starts empty and is filled later — and is its own callable.

    This IS the shape `build_lab_router` is handed: `server.py` passes
    `lambda: cameras` at import time, when `cameras` is still `None`, and
    assigns the real object inside `lifespan`. A test that passed a ready fake
    at build time could not tell a per-request lookup from a captured value,
    which is the entire failure `api/deps.py` exists to prevent.
    """

    def __init__(self, value=None) -> None:
        self.value = value

    def __call__(self):
        return self.value


class _FakeRecorder:
    """Only what the legacy `/record/episodes` path touches.

    A real `DatasetRecorder` cannot be constructed in this process at all —
    building one imports lerobot, which `haller_hmi.lab` is banned from, and a
    test that imported it would be running the ban's own violation.
    """

    def __init__(self, home: Path) -> None:
        self._home = home

    def status(self) -> dict:
        return {"recording": False, "repo_id": REPO}

    def dataset_root(self, repo_id: str) -> Path:
        return self._home / repo_id

    def session_episodes(self, repo_id: str) -> list[dict]:
        return []


class _FakeCameras:
    """`set_record` and nothing else — the only method the legacy camera path
    calls, and the only one a `CameraManager` could offer without a camera."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def set_record(self, camera_id: str, record: bool) -> bool:
        self.calls.append((camera_id, record))
        return record


# ---- fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def cold_caches():
    """Empty `catalog`'s module-level caches around every test.

    `_detail_cache` is keyed by repo-id ALONE and every test here builds
    `local/smoke` at a fresh `tmp_path`. Cleared after as well, so nothing
    leaks into the suites that read the real datasets.
    """
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()
    yield
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()


@pytest.fixture(autouse=True)
def cold_probe():
    """Forget `/lab/system`'s cached interpreter probe around every test: it is
    keyed by path and this file repoints `$HALLER_LAB_PYTHON`."""
    routes_system._reset_probe_cache()
    yield
    routes_system._reset_probe_cache()


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    """A shell that exported `HALLER_ALLOW_REMOTE_CONTROL` must not turn every
    403 assertion below into a 200."""
    monkeypatch.delenv(gate.REMOTE_CONTROL_ENV, raising=False)


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """An empty dataset cache and an empty run store, both under `tmp_path`.

    The runner interpreter is pointed at `true` for the whole file: nothing
    here means to launch a child, and this is what keeps a route that gets past
    its refusals from being `~/venvs/haller-lab/bin/python -m
    haller_hmi.runners.*` on a machine that has a real lab venv.
    """
    home = tmp_path / "lerobot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    monkeypatch.setenv(runs.RUNS_DIR_ENV, str(tmp_path / "runs"))
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, shutil.which("true") or "/bin/true")
    return home


# ---- helpers ----------------------------------------------------------------

def _router(
    home: Path,
    *,
    get_cameras=None,
    get_recorder=None,
    allow_remote_control=False,
) -> APIRouter:
    """`build_lab_router` with the same argument shape `server.py` uses.

    `allow_remote_control=None` is passed THROUGH, not defaulted away: that is
    the value that means "read the environment per call", and one test below
    turns on exactly that path.
    """
    allow = (
        None if allow_remote_control is None
        else (lambda: bool(allow_remote_control))
    )
    return build_lab_router(
        get_cameras=get_cameras or _Slot(),
        get_recorder=get_recorder or _Slot(),
        lerobot_home=lambda: home,
        allow_remote_control=allow,
    )


def _app(home: Path, *, host: str = LOOPBACK_HOST, **kwargs) -> TestClient:
    """The composed router on a throwaway app, driven from `host`."""
    app = FastAPI()
    app.include_router(_router(home, **kwargs))
    return TestClient(app, client=(host, 51000))


def _served_pairs(home: Path) -> set[tuple[str, str]]:
    """Every `(METHOD, path)` the mounted app actually SERVES.

    Read off `app.openapi()` rather than off the router, because that is the
    public surface and it is prefix-aware: `include_router(prefix=...)` would
    be invisible to a walk over route objects and is not invisible here.
    FastAPI's four own doc routes are not in the schema, so nothing has to be
    subtracted.
    """
    app = FastAPI()
    app.include_router(_router(home))
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }


def _declared_pairs(router: APIRouter) -> list[tuple[str, str]]:
    """Every `(METHOD, path)` in DECLARATION ORDER, duplicates included.

    A list and not a set, because two assertions below are about multiplicity
    and order — the two properties a set throws away and the two a compose can
    break. `app.openapi()` cannot answer either: it is a dict keyed by path, so
    a duplicate mount collapses into one entry there.

    The walk recurses because `include_router` is LAZY in FastAPI 0.140:
    `router.routes` holds a node wrapping the included router rather than that
    router's routes, so the composed router is a TREE. Duck-typed on
    `original_router` rather than on the private class, so an eager FastAPI
    (which yields the routes directly) walks correctly too.
    """
    def walk(route) -> list[tuple[str, str]]:
        included = getattr(route, "original_router", None)
        if included is not None:
            return [pair for child in included.routes for pair in walk(child)]
        return [(m, route.path) for m in sorted(getattr(route, "methods", ()) or ())]

    return [pair for route in router.routes for pair in walk(route)]


def _finished_run(kind: str = "train", run_id: str = "train-20260827-000000") -> str:
    """A run directory on disk that has already ended, written BY HAND.

    Not `runs.launch`: that spawns a detached child, and what these tests need
    is a run id `DELETE /lab/runs/{id}` will actually remove. `pid: 0` is
    `runs._pid_alive`'s "no pid" short circuit, so the run is never `alive` and
    the delete is never a 409; `result.json` is what stops `load()` reporting
    it as `died`, which would be a corpse rather than a finished run.
    """
    rdir = runs.runs_dir() / run_id
    rdir.mkdir(parents=True)
    (rdir / "run.json").write_text(json.dumps({
        "id": run_id,
        "kind": kind,
        "name": "",
        "pid": 0,
        "argv": ["/bin/true"],
        "status": "running",
        "started_at": "2026-08-27T00:00:00Z",
        "spec": {"repo_id": REPO},
    }))
    (rdir / "result.json").write_text(json.dumps({
        "status": "done",
        "exit_code": 0,
        "error": "",
        "finished_at": "2026-08-27T00:01:00Z",
    }))
    return run_id


# ============================================================================
# the compose carries every sub-router
# ============================================================================

@pytest.mark.parametrize(("name", "block"), ROUTE_BLOCKS, ids=[n for n, _ in ROUTE_BLOCKS])
def test_no_sub_router_is_dropped_from_the_compose(home, name, block):
    """Each block, named, so a deleted `include_router` says which one.

    Containment here and equality in the next test are not the same assertion:
    this one fails with a readable "the runs router is missing" and that one
    catches a route nobody declared in the contract.
    """
    served = _served_pairs(home)
    assert block <= served, f"{name} routes missing: {sorted(block - served)}"


def test_the_compose_carries_exactly_the_frozen_surface(home):
    """EQUALITY, not containment.

    `server.py` mounts one router and never names the three, so a dropped
    `include_router` line produces an app that starts cleanly and 404s a third
    of the Lab. An added route that no contract mentions is the same failure
    read from the other end: Track C codes against the document, and a surface
    the document does not describe is one nobody can call.

    Asserted against what the app SERVES and against what the router DECLARES,
    because those two can differ: a prefix on an `include_router` moves every
    served path and leaves the declared ones exactly where they were.
    """
    assert _served_pairs(home) == EXPECTED_ROUTES
    assert set(_declared_pairs(_router(home))) == EXPECTED_ROUTES


def test_no_path_is_registered_twice(home):
    """A duplicate mount is not an error, and that is why it is asserted.

    Starlette matches in declaration order and the FIRST match wins, so two
    routers claiming one path answer from whichever `include_router` line came
    first — a coin flip a reader cannot see, and one that flips when the three
    lines in `lab/routes.py` are reordered for tidiness.
    """
    counted = Counter(_declared_pairs(_router(home)))
    duplicated = sorted(pair for pair, n in counted.items() if n > 1)
    assert duplicated == []


def test_the_metrics_collection_still_precedes_the_run_id_wildcard(home):
    """`include_router` APPENDS, and the runs router relies on that.

    `metrics` satisfies `RUN_ID_RE`, so `GET /lab/runs/{run_id}` declared first
    would capture it and answer the cross-run comparison chart with `404 no run
    metrics`. The order is `routes_runs.py`'s to declare and this file's to
    keep: a compose that sorted its routes would break a route it never
    touched.
    """
    pairs = _declared_pairs(_router(home))
    assert pairs.index(("GET", "/lab/runs/metrics")) < pairs.index(
        ("GET", "/lab/runs/{run_id}"))


# ============================================================================
# the zero-arg callables are resolved per request
# ============================================================================

def test_the_recorder_is_looked_up_per_request_not_captured_at_build(home):
    """503 before `lifespan` filled the handle, 200 after — SAME app.

    This is the startup sequence, not a hypothetical: `server.py` mounts at
    import time with `recorder is None` and assigns it in `lifespan`. A router
    that closed over the value would 503 for the life of the process, and the
    only symptom is a Lab page that never works until a restart that also never
    fixes it.
    """
    make_dataset(home / REPO, n_episodes=2)
    slot = _Slot()
    client = _app(home, get_recorder=slot)

    before = client.get("/record/episodes", params={"repo_id": REPO})
    assert before.status_code == 503, before.text
    # The exact string `routes_data._require_recorder` answers with: the four
    # legacy paths keep their existing response shapes, 503 body included.
    assert before.json()["detail"] == "recorder not ready"

    slot.value = _FakeRecorder(home)

    after = client.get("/record/episodes", params={"repo_id": REPO})
    assert after.status_code == 200, after.text
    assert [e["index"] for e in after.json()["episodes"]] == [0, 1]


def test_the_cameras_handle_is_looked_up_per_request_too(home):
    """The same flip on the other handle.

    Both are asserted rather than one standing in for the other: they are two
    arguments, resolved by two methods on `LabDeps`, and a compose that passed
    one through and captured the other would leave half the surface dead.
    """
    slot = _Slot()
    client = _app(home, get_cameras=slot)

    before = client.post("/cameras/top/record", json={"record": True})
    assert before.status_code == 503, before.text
    assert before.json()["detail"] == "cameras not ready"

    cameras = _FakeCameras()
    slot.value = cameras

    after = client.post("/cameras/top/record", json={"record": True})
    assert after.status_code == 200, after.text
    assert after.json() == {"id": "top", "record": True}
    assert cameras.calls == [("top", True)]


# ============================================================================
# the gate reaches every sub-router, and reads the env per call
# ============================================================================

def test_the_gate_is_built_from_deps_for_every_gated_sub_router(home):
    """One `allow_remote_control` argument, two routers that gate on it.

    `build_lab_router` hands `deps` to all three sub-routers and each builds its
    own `require_local` from it. A compose that dropped the argument on the way
    into one of them would leave that router's destructive routes LAN-open with
    no test failing anywhere else — which on `POST /lab/runs/train` means the
    LAN can start an hours-long GPU job.
    """
    make_dataset(home / REPO, n_episodes=2)
    run_id = _finished_run()
    client = _app(home, host=LAN_HOST)

    refused = client.delete(
        "/lab/datasets", params={"repo_id": REPO, "confirm": REPO})
    assert refused.status_code == 403, refused.text
    assert client.post("/lab/runs/train", json={"repo_id": REPO}).status_code == 403
    assert client.delete(f"/lab/runs/{run_id}").status_code == 403

    # Ungated on purpose, from the same LAN client: triage from inside the
    # headset is the job the HUD exists for. One GET per sub-router, so a gate
    # accidentally mounted router-wide fails here rather than in the field.
    assert client.get("/lab/datasets").status_code == 200
    assert client.get("/lab/runs").status_code == 200
    assert client.get("/lab/system").status_code == 200


def test_allow_remote_control_none_reads_the_environment_per_call(home, monkeypatch):
    """The env var is flipped BETWEEN requests on ONE app, both directions.

    `allow_remote_control=None` is what `server.py` passes, and `api/gate.py`
    refuses to cache it: caching would mean a restart is the only way to change
    the answer, and the moment that matters is the one where Oscar is holding
    the headset and wants a job launched from the couch. Rebuilding the app
    between the two calls would pass against a value read once at build time,
    which is the bug.
    """
    make_dataset(home / REPO, n_episodes=2)
    first = _finished_run(run_id="train-20260827-000001")
    second = _finished_run(run_id="train-20260827-000002")
    client = _app(home, host=LAN_HOST, allow_remote_control=None)

    assert client.delete(f"/lab/runs/{first}").status_code == 403

    monkeypatch.setenv(gate.REMOTE_CONTROL_ENV, "1")
    allowed = client.delete(f"/lab/runs/{first}")
    assert allowed.status_code == 200, allowed.text
    assert allowed.json() == {"ok": True}
    assert not (runs.runs_dir() / first).exists()

    # The datasets router reads the same env through its own gate instance, so
    # the flip has to be visible there too — and a real 200, not "past the
    # gate": this route removes a directory with no undo.
    deleted = client.delete(
        "/lab/datasets", params={"repo_id": REPO, "confirm": REPO})
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["repo_id"] == REPO
    assert not (home / REPO).exists()

    # And back off again. A gate that latched on the first truthy read would
    # pass every assertion above and leave the LAN holding the delete button.
    monkeypatch.delenv(gate.REMOTE_CONTROL_ENV)
    assert client.delete(f"/lab/runs/{second}").status_code == 403
    assert (runs.runs_dir() / second).exists()
