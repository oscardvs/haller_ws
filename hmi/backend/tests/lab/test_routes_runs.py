# hmi/backend/tests/lab/test_routes_runs.py
"""`/lab/runs/**` as a CONTRACT, not as a router.

Track C is writing its runs table against `docs/port/trackb-lab-contract.md`
right now, so what is under test here is the WIRE. `lab/runs.py`,
`lab/compare.py` and `lab/split.py` have their own suites and nothing below
re-tests a downsampler, a pid check or a shuffle.

**The trim is the whole point.** `runs.load()` returns `alive`, `pid`, `cwd`,
`runner_python`, `log_size`, `metrics_size` and `output_dir` alongside what
Track C froze, and `routes_runs._run_wire` drops them. Every key-set assertion
below therefore comes in two halves: the frozen names are present AND the
internal ones are absent. A response carrying both passes a "does it have
`status`?" test forever, and then one day starts carrying only `alive` — and
the page that broke is in a different repo, written by someone who read the
contract and not this code.

**Nothing here launches a trainer.** `HALLER_LAB_PYTHON` is pointed at `true`
for the whole file, so `POST /lab/runs/train` really does write `spec.json`,
really does fork a detached child, and that child exits immediately without
importing lerobot or touching the GPU. A real launch would occupy the 4080
SUPER for hours. The one place a LIVE process is needed — `DELETE` while the
run is running — spawns a `time.sleep` under this interpreter with the run id
as an extra argv, which is exactly the `/proc/<pid>/cmdline` identity check
`runs._pid_alive` performs, and the fixture kills it.

Datasets are built with `_dataset.make_dataset` under `tmp_path`. Nothing here
reads `~/robot-data/lerobot`.

The app is built HERE rather than imported from `server.py`. `build_lab_router`
is what `server.py` mounts; this file mounts `build_runs_router(deps)` on a
throwaway `FastAPI()` with fake handles, so a failure names the routes module
and not the composition above it — `tests/lab/test_routes_compose.py` owns that
question.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from haller_hmi.api import gate
from haller_hmi.api.deps import LabDeps
from haller_hmi.lab import catalog, compare, runs
from haller_hmi.lab.routes_runs import build_runs_router

from ._dataset import make_dataset, write_review

#: The dataset almost every train test builds.
REPO = "local/smoke"

#: A plausible client on Oscar's LAN — the Quest, or the laptop.
LAN_HOST = "192.168.1.50"

#: `TestClient` defaults to `client=("testclient", 50000)`, which is NOT
#: loopback and so trips `require_local`. Every "local" client below has to say
#: so explicitly (`api/gate.py` documents this).
LOOPBACK_HOST = "127.0.0.1"

#: A `/lab/runs` row, EXACTLY, from the frozen line in the contract.
RUN_ROW_KEYS = frozenset({
    "id", "kind", "name", "status", "started_at", "finished_at",
    "tags", "spec_summary",
})

#: A `/lab/runs/{id}` body, EXACTLY.
RUN_DETAIL_KEYS = frozenset({
    "id", "kind", "name", "status", "spec", "argv", "started_at",
    "finished_at", "exit_code", "error", "tags",
})

#: What `runs.load()` adds and the wire must NOT carry, on either shape. These
#: are the server's own bookkeeping: `alive` is already resolved into `status`,
#: and the four sizes and paths are how this process finds files on ITS disk.
INTERNAL_RUN_KEYS = (
    "alive", "pid", "cwd", "runner_python", "log_size", "metrics_size",
    "output_dir",
)

#: Ceiling on every wait. A child that has not exec'd in ten seconds is a
#: failure, not a slow box.
WAIT_S = 10.0

#: Long enough that the one live child in this file cannot finish inside a test.
FOREVER_S = 120.0


# ---- app ----

def _client(
    home: Path,
    *,
    host: str = LOOPBACK_HOST,
    allow_remote_control: bool = False,
) -> TestClient:
    """`build_runs_router` on a throwaway app, driven from `host`.

    `allow_remote_control` is an explicit callable rather than the environment
    variable so a shell that exported `HALLER_ALLOW_REMOTE_CONTROL` cannot turn
    every 403 assertion in the gate matrix into a 200.
    """
    deps = LabDeps(
        get_cameras=lambda: None,
        get_recorder=lambda: None,
        lerobot_home=lambda: home,
        allow_remote_control=lambda: allow_remote_control,
    )
    app = FastAPI()
    app.include_router(build_runs_router(deps))
    return TestClient(app, client=(host, 51000))


# ---- fixtures ----

@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    monkeypatch.delenv(gate.REMOTE_CONTROL_ENV, raising=False)


@pytest.fixture(autouse=True)
def cold_caches():
    """Empty `catalog`'s module-level caches around every test.

    `_detail_cache` is keyed by repo-id ALONE, and every test below builds
    `local/smoke` at a fresh `tmp_path`. The `_stamp` check would catch that
    anyway, but a route test answering out of the previous test's dataset would
    be asserting about a tree it never wrote.
    """
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()
    yield
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()


@pytest.fixture
def store(tmp_path, monkeypatch) -> Path:
    """An empty run store, and an interpreter that cannot train.

    `true` exits 0 immediately and writes no `result.json`, so a launched run
    resolves to `died` — which is correct and is what makes these tests
    deterministic: the pid is gone by the first poll, and a zombie's
    `/proc/<pid>/cmdline` is EMPTY, so `_pid_alive`'s run-id check fails on it
    rather than racing.

    Pointing this at the real `~/venvs/haller-lab/bin/python` would launch
    `-m haller_hmi.runners.train` for real, against the GPU, for hours.
    """
    monkeypatch.setenv(runs.RUNS_DIR_ENV, str(tmp_path / "runs"))
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, shutil.which("true") or "/bin/true")
    return tmp_path / "runs"


@pytest.fixture
def home(tmp_path, monkeypatch, store) -> Path:
    """An empty dataset cache. `catalog.hf_home()` reads the environment, so
    the env var is the load-bearing half and `deps.lerobot_home` agrees with
    it rather than the other way round."""
    home = tmp_path / "lerobot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    return home


@pytest.fixture
def client(home) -> TestClient:
    return _client(home)


@pytest.fixture
def lan(home) -> TestClient:
    return _client(home, host=LAN_HOST)


@pytest.fixture
def reaper():
    """Kills every live stand-in child, whatever the test did."""
    children: list[subprocess.Popen] = []
    yield children
    for child in children:
        child.kill()
        child.wait()


# ---- helpers ----

def _fabricate(
    store: Path,
    run_id: str,
    *,
    kind: str = "train",
    name: str = "",
    started_at: str = "2026-08-27T10:00:00+00:00",
    pid: int = 0,
    tags: list[str] | None = None,
    spec_summary: str = "",
    spec: dict | None = None,
    argv: list[str] | None = None,
    finished: str | None = None,
    exit_code: int = 0,
    error: str = "",
) -> Path:
    """A run directory written by hand.

    Faster than launching for the states a listing test is about, and the only
    way to reach some of them at all: a `done` run needs the `result.json` a
    runner writes in its `finally`, and a run whose start time is yesterday
    needs a timestamp this process cannot produce.

    `finished=None` leaves `run.json` saying `running` with a dead pid, which
    `runs.load` resolves to `died` — never to `done`. That distinction is the
    whole reason `result.json` exists.
    """
    rdir = store / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": run_id,
        "kind": kind,
        "name": name,
        "status": "running",
        "started_at": started_at,
        "pid": pid,
        "tags": list(tags or []),
        "spec_summary": spec_summary,
        "spec": spec if spec is not None else {"repo_id": REPO},
        "argv": argv if argv is not None else ["/bin/true", "-m", "x", "spec.json"],
        "runner_python": "/bin/true",
        "cwd": str(store),
    }
    (rdir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    if finished is not None:
        runs.write_result(rdir, finished, exit_code, error)
    return rdir


def _live_run(store: Path, run_id: str, reaper: list) -> Path:
    """A fabricated run whose pid is a process that is really alive.

    The run id is passed as an extra argv on purpose: `runs._pid_alive` checks
    that the id appears in `/proc/<pid>/cmdline`, because `kill(pid, 0)` alone
    would call a recycled pid our training job. This is the cheapest process
    that satisfies that check honestly.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({FOREVER_S})", run_id])
    reaper.append(child)
    cmdline = Path(f"/proc/{child.pid}/cmdline")

    def exec_done() -> bool:
        try:
            return run_id in cmdline.read_bytes().decode("utf-8", "replace")
        except OSError:
            return False

    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline and not exec_done():
        time.sleep(0.02)
    assert exec_done(), "the stand-in child never reached exec"
    return _fabricate(store, run_id, pid=child.pid)


def _rows(client: TestClient, **params) -> list[dict]:
    response = client.get("/lab/runs", params=params or None)
    assert response.status_code == 200, response.text
    return response.json()["runs"]


def _ids(client: TestClient, **params) -> list[str]:
    return [row["id"] for row in _rows(client, **params)]


def _launch_train(client: TestClient, **spec) -> str:
    """POST a training spec that must succeed, and return the run id."""
    response = client.post("/lab/runs/train", json=spec)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"id"}, body
    return body["id"]


def _launch_train_eval(client: TestClient, **spec) -> str:
    """POST an eval spec that must succeed, and return the run id."""
    response = client.post("/lab/runs/eval", json=spec)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _launch_simeval(client: TestClient, **spec) -> str:
    """POST a sim-eval spec that must succeed, and return the run id."""
    response = client.post("/lab/runs/simeval", json=spec)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _spec_on_disk(store: Path, run_id: str) -> dict:
    """`spec.json` as the CHILD would read it — never the route's response.

    The route answers `{"id"}` and nothing else, so the only evidence about
    what was actually asked for is the file the runner is handed.
    """
    return json.loads((store / run_id / "spec.json").read_text())


# ============================================================================
# GET /lab/runs
# ============================================================================

def test_a_listing_row_carries_the_wire_names_and_only_those(store, client):
    """The exact key set, because Track C is writing `row.spec_summary` today."""
    _fabricate(store, "train-a", name="act", tags=["nightly"],
               spec_summary="train · local/smoke · 35 of 46 episodes",
               finished="done")

    row = _rows(client)[0]

    assert set(row) == set(RUN_ROW_KEYS), (
        f"unexpected {sorted(set(row) - RUN_ROW_KEYS)}, "
        f"missing {sorted(RUN_ROW_KEYS - set(row))}"
    )
    # The other half. `set(row) == RUN_ROW_KEYS` already implies it; spelled out
    # so a failure names the leak rather than a set difference.
    for internal in INTERNAL_RUN_KEYS:
        assert internal not in row, f"the listing leaked {internal!r}"
    assert row["id"] == "train-a"
    assert row["kind"] == "train"
    assert row["name"] == "act"
    assert row["status"] == "done"
    assert row["tags"] == ["nightly"]
    assert row["finished_at"]


def test_a_row_that_never_finished_carries_finished_at_as_null(store, client):
    """`finished_at` is a KEY on every row and a value on some. A row that
    dropped it while running would make the column's absence mean two things —
    "still going" and "this build forgot" — and the table renders one dash for
    both."""
    _fabricate(store, "train-a")

    row = _rows(client)[0]

    assert "finished_at" in row
    assert row["finished_at"] is None
    # Pid gone with no result.json: killed hard, OOM, or a crash the runner
    # could not survive long enough to report. NOT `done`.
    assert row["status"] == "died"


def test_the_listing_is_newest_first_by_start_time(store, client):
    """By START TIME, not by directory name: run ids are prefixed with the
    kind, so a name sort would group every rollout above every train whatever
    the clock said."""
    _fabricate(store, "train-old", started_at="2026-08-26T09:00:00+00:00")
    _fabricate(store, "export-new", kind="export",
               started_at="2026-08-27T18:00:00+00:00")
    _fabricate(store, "train-mid", started_at="2026-08-27T08:00:00+00:00")

    assert _ids(client) == ["export-new", "train-mid", "train-old"]


def test_the_listing_filters_by_kind(store, client):
    _fabricate(store, "train-a")
    _fabricate(store, "train-b")
    _fabricate(store, "export-a", kind="export")
    _fabricate(store, "rollout-a", kind="rollout")

    assert sorted(_ids(client, kind="train")) == ["train-a", "train-b"]
    assert _ids(client, kind="export") == ["export-a"]
    assert _ids(client, kind="rollout") == ["rollout-a"]
    assert len(_ids(client)) == 4


def test_the_listing_filters_by_status(store, client):
    """`status` is a RESOLVED value the client cannot recompute — a dead pid
    with no `result.json` is `died`, and only `runs.load` knows that — so the
    filter has to be here."""
    _fabricate(store, "train-done", finished="done")
    _fabricate(store, "train-failed", finished="failed", exit_code=1)
    _fabricate(store, "train-stopped", finished="stopped", exit_code=130)
    _fabricate(store, "train-died")

    assert _ids(client, status="done") == ["train-done"]
    assert _ids(client, status="failed") == ["train-failed"]
    assert _ids(client, status="stopped") == ["train-stopped"]
    assert _ids(client, status="died") == ["train-died"]


def test_the_status_filter_is_case_folded(store, client):
    """The UI's chips are capitalised and the store's statuses are not."""
    _fabricate(store, "train-done", finished="done")

    assert _ids(client, status="Done") == ["train-done"]


def test_kind_and_status_filter_together(store, client):
    _fabricate(store, "train-done", finished="done")
    _fabricate(store, "export-done", kind="export", finished="done")
    _fabricate(store, "train-died")

    assert _ids(client, kind="train", status="done") == ["train-done"]


def test_a_cleared_filter_means_every_run_not_no_runs(store, client):
    """An absent parameter arrives as None and one the UI just cleared arrives
    as `""`. Both have to mean "everything", or clearing the filter empties the
    table."""
    _fabricate(store, "train-a")
    _fabricate(store, "export-a", kind="export")

    assert len(_ids(client, kind="", status="")) == 2


def test_an_unknown_kind_is_a_400_and_not_an_empty_list(store, client):
    """`runs.RUNNERS` is the authority on what kinds exist. Answering a typo
    with 200 and no rows reads as "the run I started is gone"."""
    _fabricate(store, "train-a")

    response = client.get("/lab/runs", params={"kind": "finetune"})

    assert response.status_code == 400, response.text
    assert "finetune" in response.json()["detail"]


def test_an_unknown_status_filters_to_nothing_rather_than_refusing(store, client):
    """Deliberately NOT symmetrical with `kind`. The statuses are written in
    `runners/_common.run_guarded`, `runs.load` and `runs.launch`; a fourth copy
    in a routes module is the one that goes stale, and it would then refuse a
    status the store actually contains."""
    _fabricate(store, "train-a", finished="done")

    response = client.get("/lab/runs", params={"status": "pending"})

    assert response.status_code == 200, response.text
    assert response.json()["runs"] == []


def test_spec_summary_is_a_string_and_never_an_object(store, client):
    """Track C renders it VERBATIM and never re-derives it, so a dict arriving
    where a line was promised renders as `{'repo_id': ...}` in the table. Built
    once at launch, because a run whose dataset has since been pruned must
    still say what it was asked for."""
    summary = "train · local/smoke · 35 of 46 episodes · act · 100000 steps"
    _fabricate(store, "train-a", spec_summary=summary)

    value = _rows(client)[0]["spec_summary"]

    assert isinstance(value, str)
    assert value == summary


def test_a_run_written_before_tags_existed_still_renders(store, client):
    """`tags` and `spec_summary` default rather than being absent, so the table
    renders every row through one shape instead of the page checking."""
    rdir = store / "train-old"
    rdir.mkdir(parents=True)
    (rdir / "run.json").write_text(json.dumps(
        {"id": "train-old", "kind": "train", "name": "", "status": "running",
         "started_at": "2026-08-26T09:00:00+00:00", "pid": 0}))

    row = _rows(client)[0]

    assert row["tags"] == []
    assert row["spec_summary"] == ""


# ============================================================================
# POST /lab/runs/train
# ============================================================================

def test_train_answers_with_the_id_and_nothing_else(home, store, client):
    """The page follows the run through `GET /lab/runs/{id}`. Returning the
    whole record here would be a second spelling of that route which no rename
    could ever reach."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/runs/train", json={"repo_id": REPO})

    assert response.status_code == 200, response.text
    assert set(response.json()) == {"id"}
    assert (store / response.json()["id"] / "spec.json").exists()


def test_a_missing_repo_id_is_a_400_that_names_the_field(home, client):
    response = client.post("/lab/runs/train", json={"steps": 100})

    assert response.status_code == 400, response.text
    assert "repo_id" in response.json()["detail"]


def test_a_repo_id_that_does_not_exist_is_a_404(home, client):
    """404 and not 400: the request is well formed and the dataset is what is
    missing. `catalog.dataset_detail` raises `FileNotFoundError` and the
    `api/errors` ladder renders that rung."""
    response = client.post("/lab/runs/train", json={"repo_id": "local/nope"})

    assert response.status_code == 404, response.text
    assert "local/nope" in response.json()["detail"]


# ============================================================================
# POST /lab/runs/eval
# ============================================================================

def _eval_checkpoint(store: Path, run_id: str, step: str) -> Path:
    """A checkpoint directory shaped the way LeRobot writes one.

    Distinct from `_checkpoint` further down, which takes a run DIRECTORY and
    writes no `config.json` — `eval_runner` refuses a directory without one, so
    these tests need the file to exist.
    """
    model = store / run_id / "train" / "checkpoints" / step / "pretrained_model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}")
    return model


def test_eval_takes_the_repo_and_the_HELD_OUT_episodes_from_a_training_run(
        home, store, client):
    """Per-episode loss only ranks anything when it is measured on episodes the
    checkpoint did NOT train on. `eval_episodes` and not `episodes`, which is
    the whole kept set."""
    make_dataset(home / REPO, n_episodes=6)
    train_id = _launch_train(client, repo_id=REPO, eval_split=0.34)
    _eval_checkpoint(store, train_id, "010000")

    run_id = _launch_train_eval(client, source_run=train_id)

    spec = _spec_on_disk(store, run_id)
    train_spec = _spec_on_disk(store, train_id)
    assert spec["repo_id"] == REPO
    assert spec["episodes"] == train_spec["eval_episodes"]
    assert spec["episodes"] != train_spec["episodes"]
    assert spec["source_run"] == train_id


def test_eval_defaults_to_the_NEWEST_checkpoint_of_the_source_run(
        home, store, client):
    """`runs.checkpoints()` sorts newest first, so "the checkpoint" is row 0.
    Taking the last row would score the OLDEST one and quietly rank a
    half-trained policy."""
    make_dataset(home / REPO, n_episodes=6)
    train_id = _launch_train(client, repo_id=REPO, eval_split=0.34)
    _eval_checkpoint(store, train_id, "010000")
    newest = _eval_checkpoint(store, train_id, "090000")

    run_id = _launch_train_eval(client, source_run=train_id)

    assert _spec_on_disk(store, run_id)["checkpoint"] == str(newest)


def test_eval_refuses_a_source_run_that_has_no_checkpoint_yet(home, store, client):
    make_dataset(home / REPO, n_episodes=6)
    train_id = _launch_train(client, repo_id=REPO, eval_split=0.34)

    response = client.post(
        "/lab/runs/eval", json={"source_run": train_id})

    assert response.status_code == 400, response.text
    assert "no checkpoint" in response.json()["detail"]


def test_an_explicit_checkpoint_and_episodes_beat_the_source_run(
        home, store, client):
    """So the same checkpoint can be scored over a different set without
    unpicking `source_run`."""
    make_dataset(home / REPO, n_episodes=6)
    train_id = _launch_train(client, repo_id=REPO, eval_split=0.34)
    _eval_checkpoint(store, train_id, "010000")
    other = _eval_checkpoint(store, train_id, "020000")

    run_id = _launch_train_eval(
        client, source_run=train_id, checkpoint=str(other), episodes=[0, 1])

    spec = _spec_on_disk(store, run_id)
    assert spec["checkpoint"] == str(other)
    assert spec["episodes"] == [0, 1]


def test_eval_without_a_source_run_needs_a_checkpoint_and_a_repo(home, client):
    make_dataset(home / REPO, n_episodes=3)

    missing_ckpt = client.post("/lab/runs/eval", json={"repo_id": REPO})
    assert missing_ckpt.status_code == 400, missing_ckpt.text
    assert "checkpoint" in missing_ckpt.json()["detail"]

    missing_repo = client.post("/lab/runs/eval", json={"checkpoint": "/tmp/x"})
    assert missing_repo.status_code == 400, missing_repo.text
    assert "repo_id" in missing_repo.json()["detail"]


def test_eval_episodes_must_be_whole_numbers(home, client):
    """Shape only — whether the episode EXISTS is the child's check. A bare
    string would otherwise be written into the spec as its characters."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/runs/eval", json={
        "repo_id": REPO, "checkpoint": "/tmp/x", "episodes": "0,1"})

    assert response.status_code == 400, response.text
    assert "episodes" in response.json()["detail"]


def test_eval_launches_the_eval_runner_and_not_the_trainer(home, store, client):
    """The kind is what picks the module out of `runs.RUNNERS`; getting it
    wrong would start a training run against an eval spec."""
    make_dataset(home / REPO, n_episodes=6)
    train_id = _launch_train(client, repo_id=REPO, eval_split=0.34)
    _eval_checkpoint(store, train_id, "010000")

    run_id = _launch_train_eval(client, source_run=train_id)

    record = json.loads((store / run_id / "run.json").read_text())
    assert record["kind"] == "eval"
    assert record["argv"][2] == "haller_hmi.runners.eval_runner"


# ============================================================================
# POST /lab/runs/simeval
# ============================================================================
#
# The counterpart to the eval route above, and the reason it is a second route
# rather than a flag on that one: `eval` measures per-episode training LOSS over
# recorded frames and `simeval` measures SUCCESS RATE against `sim/task.py`'s
# predicate on a MuJoCo bench. The two are not comparable numbers, so they do
# not share a run kind, a spec or a run row.

#: The 12 columns Haller's recorder writes for the bimanual bench. Passed as
#: `action_names` wherever a test has no dataset to read them from.
SIM_ACTION_NAMES = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper")
]


def test_simeval_takes_the_policy_and_the_REPO_from_a_training_run(
        home, store, client):
    """`repo_id` is not decoration on this route. It is what tells the child
    which column of the policy's action vector is which joint, and a sim
    evaluation with that mapping guessed scores a policy whose wrist is driven
    by the shoulder's number."""
    make_dataset(home / REPO, n_episodes=6)
    train_id = _launch_train(client, repo_id=REPO, eval_split=0.34)
    checkpoint = _eval_checkpoint(store, train_id, "010000")

    run_id = _launch_simeval(client, source_run=train_id)

    spec = _spec_on_disk(store, run_id)
    assert spec["policy_path"] == str(checkpoint)
    assert spec["repo_id"] == REPO
    assert spec["source_run"] == train_id


def test_simeval_defaults_to_the_NEWEST_checkpoint_of_the_source_run(
        home, store, client):
    """`runs.checkpoints()` sorts newest first, so "the checkpoint" is row 0.
    Row -1 would score the oldest and report a half-trained policy's rate."""
    make_dataset(home / REPO, n_episodes=6)
    train_id = _launch_train(client, repo_id=REPO, eval_split=0.34)
    _eval_checkpoint(store, train_id, "010000")
    newest = _eval_checkpoint(store, train_id, "090000")

    run_id = _launch_simeval(client, source_run=train_id)

    assert _spec_on_disk(store, run_id)["policy_path"] == str(newest)


def test_simeval_refuses_a_source_run_that_has_no_checkpoint_yet(
        home, store, client):
    make_dataset(home / REPO, n_episodes=6)
    train_id = _launch_train(client, repo_id=REPO, eval_split=0.34)

    response = client.post("/lab/runs/simeval", json={"source_run": train_id})

    assert response.status_code == 400, response.text
    assert "no checkpoint" in response.json()["detail"]


def test_an_explicit_policy_path_and_repo_beat_the_source_run(
        home, store, client):
    """So the checkpoint from one run can be scored against another dataset's
    column layout without unpicking `source_run`."""
    make_dataset(home / REPO, n_episodes=6)
    train_id = _launch_train(client, repo_id=REPO, eval_split=0.34)
    _eval_checkpoint(store, train_id, "010000")
    other = _eval_checkpoint(store, train_id, "020000")

    run_id = _launch_simeval(
        client, source_run=train_id, policy_path=str(other),
        repo_id="local/other")

    spec = _spec_on_disk(store, run_id)
    assert spec["policy_path"] == str(other)
    assert spec["repo_id"] == "local/other"


def test_simeval_without_a_source_run_needs_a_policy_path(home, client):
    response = client.post(
        "/lab/runs/simeval", json={"repo_id": REPO})

    assert response.status_code == 400, response.text
    assert "policy_path" in response.json()["detail"]


def test_simeval_refuses_to_guess_the_action_layout(home, client):
    """Neither `repo_id` nor `action_names` means nothing on disk says which
    column is which joint. Guessing it does not fail loudly: it scores zero and
    reads as a bad policy."""
    response = client.post(
        "/lab/runs/simeval", json={"policy_path": "/tmp/ckpt"})

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "repo_id" in detail and "action_names" in detail


def test_action_names_are_an_accepted_substitute_for_a_repo(home, store, client):
    """A checkpoint whose dataset has since been pruned still has a layout, and
    naming it explicitly is the only way to say so."""
    run_id = _launch_simeval(
        client, policy_path="/tmp/ckpt", action_names=SIM_ACTION_NAMES)

    spec = _spec_on_disk(store, run_id)
    assert spec["action_names"] == SIM_ACTION_NAMES
    assert "repo_id" not in spec or not spec["repo_id"]
    # The one-line summary still reads, with no empty segment where the missing
    # repo would have been.
    summary = json.loads((store / run_id / "run.json").read_text())["spec_summary"]
    assert summary.startswith("sim eval")
    assert " ·  · " not in summary


def test_action_names_must_be_a_list(home, client):
    """A bare string accepted as a list would arrive at the child as 18
    single-character column names."""
    response = client.post("/lab/runs/simeval", json={
        "policy_path": "/tmp/ckpt", "action_names": "left_shoulder_pan"})

    assert response.status_code == 400, response.text
    assert "action_names" in response.json()["detail"]


def test_simeval_passes_the_seed_list_through_verbatim(home, store, client):
    """Seeds ARE the experiment. Replaying exactly the layouts a previous run
    scored is the comparison this route exists for, so the list is not sorted,
    de-duplicated or trimmed on the way to the spec."""
    run_id = _launch_simeval(
        client, policy_path="/tmp/ckpt", action_names=SIM_ACTION_NAMES,
        seeds=[9, 2, 9])

    assert _spec_on_disk(store, run_id)["seeds"] == [9, 2, 9]


def test_simeval_seeds_must_be_whole_numbers(home, client):
    """Shape only. Whether the seed produces a solvable bench is the sim's
    business, not this route's."""
    response = client.post("/lab/runs/simeval", json={
        "policy_path": "/tmp/ckpt", "action_names": SIM_ACTION_NAMES,
        "seeds": "0,1"})

    assert response.status_code == 400, response.text
    assert "seeds" in response.json()["detail"]


def test_the_bench_settings_reach_the_spec(home, store, client):
    """`spec.json` is the run's own record of what it was asked for, so every
    knob that changes the number has to be in it - including the rig config,
    which is NOT resolved here: `simeval_runner._config_path` applies the
    `$HALLER_HMI_CONFIG`-then-default rule, and resolving it in this process
    would answer with the SERVER's rig on a box where the two differ."""
    run_id = _launch_simeval(
        client, policy_path="/tmp/ckpt", action_names=SIM_ACTION_NAMES,
        config="/rigs/bimanual-sim.yaml", control_hz=20, max_episode_s=45,
        episodes=4, seed_start=100, randomize=False, mirror=True,
        task="put the red cube on the pad", robot_type="so101_bimanual",
        device="cpu")

    spec = _spec_on_disk(store, run_id)
    assert spec["config"] == "/rigs/bimanual-sim.yaml"
    assert spec["control_hz"] == 20.0
    assert spec["max_episode_s"] == 45.0
    assert spec["episodes"] == 4
    assert spec["seed_start"] == 100
    assert spec["randomize"] is False
    assert spec["mirror"] is True
    assert spec["task"] == "put the red cube on the pad"
    assert spec["robot_type"] == "so101_bimanual"
    assert spec["device"] == "cpu"


def test_an_explicit_null_step_cap_reaches_the_spec_as_null(home, store, client):
    """A run with no per-tick step cap is a different experiment, not a missing
    field, so it must not be defaulted away between the browser and the child."""
    run_id = _launch_simeval(
        client, policy_path="/tmp/ckpt", action_names=SIM_ACTION_NAMES,
        max_speed_deg_s=None)

    spec = _spec_on_disk(store, run_id)
    assert "max_speed_deg_s" in spec
    assert spec["max_speed_deg_s"] is None


def test_a_zero_step_cap_is_refused_rather_than_read_as_off(home, client):
    response = client.post("/lab/runs/simeval", json={
        "policy_path": "/tmp/ckpt", "action_names": SIM_ACTION_NAMES,
        "max_speed_deg_s": 0})

    assert response.status_code == 400, response.text
    assert "max_speed_deg_s" in response.json()["detail"]


def test_simeval_launches_the_simeval_runner_and_not_the_eval_runner(
        home, store, client):
    """The kind is what picks the module out of `runs.RUNNERS`. Getting it wrong
    here would start a training-loss job against a spec that names no dataset
    episodes, and report its number under this run's id."""
    run_id = _launch_simeval(
        client, policy_path="/tmp/ckpt", action_names=SIM_ACTION_NAMES)

    record = json.loads((store / run_id / "run.json").read_text())
    assert record["kind"] == "simeval"
    assert record["argv"][2] == "haller_hmi.runners.simeval_runner"
    assert record["argv"][2] == runs.RUNNERS["simeval"]


def test_simeval_is_local_only(home, lan):
    """`--host 0.0.0.0` is how the Quest reaches the HMI. Reaching it must not
    also mean starting a GPU job from the LAN."""
    response = lan.post("/lab/runs/simeval", json={
        "policy_path": "/tmp/ckpt", "action_names": SIM_ACTION_NAMES})

    assert response.status_code == 403, response.text


def test_policy_inputs_reach_the_spec_resolved_to_shapes(home, store, client):
    """The spec carries the RESOLVED map, not the names that were ticked. The
    dataset can grow a column the next day — that is exactly what happened on
    2026-08-29 — and a spec holding only names would re-resolve to a different
    observation space on a relaunch while looking identical on the page."""
    make_dataset(home / REPO, n_episodes=3)

    run_id = _launch_train(
        client, repo_id=REPO,
        policy_inputs=["observation.state", "observation.images.top"])

    assert _spec_on_disk(store, run_id)["policy_input_features"] == {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.top": {"type": "VISUAL", "shape": [480, 640, 3]},
    }


def test_a_column_left_out_of_policy_inputs_stays_out_of_the_spec(home, store, client):
    """The point of the field. LeRobot takes every `observation.*` column when
    it derives the space itself, so a run that means to exclude one has to say
    so here or the exclusion never happens."""
    make_dataset(home / REPO, n_episodes=3)
    info = json.loads((home / REPO / "meta" / "info.json").read_text())
    info["features"]["observation.wall_clock"] = {
        "dtype": "float32", "shape": [1], "names": ["t"]}
    (home / REPO / "meta" / "info.json").write_text(json.dumps(info))

    run_id = _launch_train(
        client, repo_id=REPO,
        policy_inputs=["observation.state", "observation.images.top"])

    spec = _spec_on_disk(store, run_id)
    assert "observation.wall_clock" not in spec["policy_input_features"]


def test_no_policy_inputs_leaves_the_key_off_the_spec_entirely(home, store, client):
    """Absent means "LeRobot's own rule", which is what every run before this
    field did and what an API caller that has not been updated still gets. A
    key defaulted to the narrow set here would change those runs silently."""
    make_dataset(home / REPO, n_episodes=3)

    run_id = _launch_train(client, repo_id=REPO)

    assert "policy_input_features" not in _spec_on_disk(store, run_id)


def test_policy_inputs_naming_a_column_the_dataset_lacks_is_a_400(home, client):
    """Same rule as `episodes`: the dataset resolved, the request is what is
    wrong. Silently dropping the name would train on a smaller observation
    space than was ticked and nothing would say so."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/runs/train", json={
        "repo_id": REPO, "policy_inputs": ["observation.state", "observation.nope"]})

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "observation.nope" in detail
    assert "observation.state" in detail


def test_an_empty_policy_inputs_list_is_a_400(home, client):
    make_dataset(home / REPO, n_episodes=3)

    response = client.post(
        "/lab/runs/train", json={"repo_id": REPO, "policy_inputs": []})

    assert response.status_code == 400, response.text


def test_policy_inputs_that_is_not_a_list_is_a_400(home, client):
    """A bare string would otherwise resolve as its characters."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.post("/lab/runs/train", json={
        "repo_id": REPO, "policy_inputs": "observation.state"})

    assert response.status_code == 400, response.text
    assert "policy_inputs" in response.json()["detail"]


def test_episodes_naming_an_episode_the_dataset_does_not_have_is_a_400(home, client):
    """A 400 and not a 404, because the dataset resolved — what is wrong is the
    list in the request. LeRobot drops an unknown index SILENTLY, so a typo'd
    list would otherwise train on fewer episodes than the operator chose and
    nothing would ever say so."""
    make_dataset(home / REPO, n_episodes=3)

    response = client.post(
        "/lab/runs/train", json={"repo_id": REPO, "episodes": [0, 99]})

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "99" in detail and REPO in detail


def test_an_empty_episode_list_is_a_400(home, client):
    make_dataset(home / REPO, n_episodes=3)

    response = client.post(
        "/lab/runs/train", json={"repo_id": REPO, "episodes": []})

    assert response.status_code == 400, response.text


def test_with_no_episodes_the_launched_spec_is_the_review_keep_set(
        home, store, client):
    """THE test this whole surface exists for.

    Reject one episode on the review page, launch a training run without naming
    any episodes, and read `spec.json` OFF DISK — the file the detached child is
    actually handed. The rejected episode must not be in it.

    Nothing else in the suite closes this loop. `review.py` proves a mark is
    stored, `catalog` proves `keep_list` excludes it, `runs.launch` proves a
    spec is written — and a route that read `keep_list` and then passed the full
    range would satisfy all three while training on the take Oscar threw away.
    """
    make_dataset(home / REPO, n_episodes=4)
    write_review(home / REPO, {1: "reject"})

    run_id = _launch_train(client, repo_id=REPO, job_name="keepset")

    spec = _spec_on_disk(store, run_id)
    assert 1 not in spec["episodes"], (
        f"the rejected episode reached the trainer: {spec['episodes']}")
    assert sorted(spec["episodes"]) == [0, 2, 3]
    # `total_episodes` is the dataset's, not the keep set's: a run that trained
    # on 3 of 4 and one that trained on 3 of 3 are different runs.
    assert spec["total_episodes"] == 4


def test_the_spec_summary_says_how_many_of_how_many(home, store, client):
    """`x of y episodes`, for the reason above — and stored, so it survives the
    dataset being pruned out from under it."""
    make_dataset(home / REPO, n_episodes=4)
    write_review(home / REPO, {1: "reject"})

    run_id = _launch_train(client, repo_id=REPO, job_name="summary")

    row = next(r for r in _rows(client) if r["id"] == run_id)

    assert "3 of 4 episodes" in row["spec_summary"]
    assert REPO in row["spec_summary"]


def test_an_explicit_episode_list_overrides_the_review(home, store, client):
    """A caller-supplied list chooses the SET. It is the escape hatch for
    training on a subset without re-marking 46 episodes, and it deliberately
    does not consult the review at all."""
    make_dataset(home / REPO, n_episodes=4)
    write_review(home / REPO, {1: "reject"})

    run_id = _launch_train(client, repo_id=REPO, episodes=[1, 3], job_name="explicit")

    assert sorted(_spec_on_disk(store, run_id)["episodes"]) == [1, 3]


def test_a_dataset_with_every_episode_rejected_is_a_400(home, client):
    """Before a run directory exists. The operator is looking at the form when
    the 400 arrives and at a table of runs when the traceback does."""
    make_dataset(home / REPO, n_episodes=2)
    write_review(home / REPO, {0: "reject", 1: "reject"})

    response = client.post("/lab/runs/train", json={"repo_id": REPO})

    assert response.status_code == 400, response.text
    assert "reject" in response.json()["detail"]


def test_the_eval_split_order_survives_into_the_spec_unsorted(home, store, client):
    """The trick, end to end, and the one that fails SILENTLY.

    LeRobot groups the episode list it is handed by task and holds out each
    group's TAIL, and it NEVER sorts it — `LeRobotDataset` stores
    `self.episodes = episodes` exactly as given. So the ORDER in `spec.json` IS
    the split. Any tidy-up on the way here — `sorted()`, a `set`, a dedupe —
    still produces a train/eval split of the right SIZE, just the wrong one,
    and nothing downstream can tell.

    Eight episodes, seed 42: `random.Random(42).shuffle` gives
    `[3, 4, 6, 7, 2, 5, 0, 1]`, so a 0.25 holdout is episodes 0 and 1 — NOT the
    chronological tail 6 and 7. That inequality is the assertion that matters:
    a `sorted()` anywhere on the path makes the two sets identical, and then
    the eval curve is measured on the newest, best-executed demonstrations
    while training on the sloppiest (Oscar's first 20 kept 3 of the first 10
    and 7 of the second 10).
    """
    make_dataset(home / REPO, n_episodes=8)

    run_id = _launch_train(
        client, repo_id=REPO, eval_split=0.25, eval_seed=42, job_name="shuffled")

    spec = _spec_on_disk(store, run_id)
    assert spec["episodes"] == [3, 4, 6, 7, 2, 5, 0, 1]
    assert spec["episodes"] != sorted(spec["episodes"]), (
        "the order reached the spec sorted — the holdout is now chronological")
    assert sorted(spec["episodes"]) == list(range(8))
    # The tail of the ORDER, which is not the tail of the numbering.
    assert spec["eval_episodes"] == [0, 1]
    assert spec["eval_episodes"] != [6, 7]
    assert spec["train_episodes"] == [2, 3, 4, 5, 6, 7]


def test_eval_mode_recent_is_how_a_chronological_holdout_is_asked_for(
        home, store, client):
    """The contrast that proves the shuffle above is a choice and not an
    accident: `recent` leaves the order chronological, and then the holdout IS
    the last two episodes. That is a legitimate intent — validating on the
    newest lighting and object placement — and it has to be asked for."""
    make_dataset(home / REPO, n_episodes=8)

    run_id = _launch_train(
        client, repo_id=REPO, eval_split=0.25, eval_seed=42,
        eval_mode="recent", job_name="recent")

    spec = _spec_on_disk(store, run_id)
    assert spec["episodes"] == list(range(8))
    assert spec["eval_episodes"] == [6, 7]


def test_an_eval_split_that_would_leave_nothing_to_train_on_is_a_400(
        home, client):
    make_dataset(home / REPO, n_episodes=2)

    response = client.post(
        "/lab/runs/train", json={"repo_id": REPO, "eval_split": 0.9})

    assert response.status_code == 400, response.text


@pytest.mark.parametrize(
    ("field", "value"),
    [("steps", 0), ("batch_size", 0), ("save_freq", -1), ("num_workers", -1),
     ("eval_split", 1.0), ("eval_split", -0.1), ("steps", "lots")],
)
def test_a_spec_field_that_would_die_in_the_childs_first_second_is_a_400(
        home, client, field, value):
    """Refusing now costs one corrected field. Refusing later costs a run
    directory and the walk over to read why it is empty."""
    make_dataset(home / REPO, n_episodes=2)

    response = client.post(
        "/lab/runs/train", json={"repo_id": REPO, field: value})

    assert response.status_code == 400, response.text
    assert field in response.json()["detail"]


# ============================================================================
# GET /lab/runs/{id}
# ============================================================================

def test_a_run_detail_carries_the_frozen_keys_and_only_those(store, client):
    _fabricate(store, "train-a", name="act", spec={"repo_id": REPO, "steps": 10},
               argv=["/bin/true", "-m", "haller_hmi.runners.train", "spec.json"],
               finished="failed", exit_code=1, error="CUDA out of memory")

    response = client.get("/lab/runs/train-a")

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == set(RUN_DETAIL_KEYS), (
        f"unexpected {sorted(set(body) - RUN_DETAIL_KEYS)}, "
        f"missing {sorted(RUN_DETAIL_KEYS - set(body))}"
    )
    for internal in INTERNAL_RUN_KEYS:
        assert internal not in body, f"the detail leaked {internal!r}"
    assert body["spec"]["repo_id"] == REPO
    assert body["argv"][0] == "/bin/true"
    assert body["status"] == "failed"
    assert body["exit_code"] == 1
    assert body["error"] == "CUDA out of memory"


def test_the_detail_omits_spec_summary_but_not_tags(store, client):
    """Kept and NARROWED rather than deleted, because half of it was right.

    `spec_summary` is a listing shape: the detail is showing the spec itself,
    so a one-line rendering of it beside the real thing is a second answer to
    one question, and nothing in the detail view reads it.

    `tags` never belonged in that sentence. They are set at launch, live on the
    record, and appear nowhere in `spec` — so there was nothing for them to
    duplicate, and omitting them silently dropped a field the detail screen
    renders. The old version of this test asserted `"tags" not in body` and so
    DEFENDED the defect: it was written from the branch structure rather than
    from the claim, and `Run = RunSummary & {...}` had said otherwise all along.
    """
    _fabricate(store, "train-a", tags=["nightly"], spec_summary="train · x")

    body = client.get("/lab/runs/train-a").json()

    assert body["tags"] == ["nightly"]
    assert "spec_summary" not in body


def test_the_detail_carries_every_tag_the_listing_carries(store, client):
    """Payload against payload, not either against a fixture.

    The two shapes are built by one function down two branches, so the only
    thing that can prove they agree about tags is reading both for the same run
    and comparing. A fixture-based assertion would pass against a detail branch
    that hardcoded the same list.
    """
    _fabricate(store, "train-a", tags=["nightly", "sweep"])
    _fabricate(store, "train-b", tags=[])

    rows = {r["id"]: r for r in client.get("/lab/runs").json()["runs"]}
    for run_id, row in rows.items():
        detail = client.get(f"/lab/runs/{run_id}").json()
        assert detail["tags"] == row["tags"], run_id


def test_a_run_recorded_before_tags_existed_reads_as_empty_on_the_detail(
        store, client):
    """A `run.json` with no `tags` key renders as `[]` and never as `null`.

    Real case, not hypothetical: the kit-written `run.json` files have no
    `tags` key at all.

    **This test pins the PROPERTY and cannot isolate the mechanism, which is
    worth saying rather than leaving for someone to discover.** Two guards
    independently prevent a `null` here — `runs.load()` defaults the key to
    `[]` before the wire ever sees the record, and `_run_wire` applies `or []`
    on top. Mutating either one alone leaves this green, so a surviving
    single-point mutation here is an EQUIVALENT MUTANT rather than a gap in the
    assertion. Measured: `load()` on a record with no `tags` key already
    returns `tags: []`.

    Both are kept. Removing `_run_wire`'s default would make it silently
    dependent on a behaviour of `load()` that nothing states, and this is a
    default rather than a safety check — two agreeing defaults cost nothing,
    where two agreeing safety checks would hide which one fires.
    """
    _fabricate(store, "train-a")
    (store / "train-a" / "run.json").write_text(json.dumps(
        {k: v for k, v in
         json.loads((store / "train-a" / "run.json").read_text()).items()
         if k != "tags"}))

    assert client.get("/lab/runs/train-a").json()["tags"] == []


def test_an_unknown_run_id_is_a_404(store, client):
    response = client.get("/lab/runs/train-nope")

    assert response.status_code == 404, response.text
    assert "train-nope" in response.json()["detail"]


def test_a_run_id_that_is_not_one_directory_in_the_store_is_refused(
        store, client):
    """Two guards, and the second is the one `RUN_ID_RE` alone misses: `".."`
    matches the regex (dots are legal in a run id) and names the PARENT of the
    run store. `run_dir`'s containment check is what makes that a refusal.

    A literal slash never reaches the handler at all — Starlette's path
    converter stops at `/` — so it 404s as an unrouted URL, and a percent-
    encoded one decodes to a path segment with a slash in it and does the same.
    """
    answers = {
        # No route: `{run_id}` cannot span a slash.
        "/lab/runs/a/b": 404,
        "/lab/runs/a%2Fb": 404,
        # Reaches the handler as the literal string "..", and is refused there.
        "/lab/runs/%2e%2e": 400,
    }
    for url, expected in answers.items():
        response = client.get(url)
        assert response.status_code == expected, f"{url} -> {response.text}"

    assert "escapes the run store" in client.get("/lab/runs/%2e%2e").json()["detail"]


# ============================================================================
# GET /lab/runs/{id}/metrics and /log
# ============================================================================

def test_the_metrics_offset_round_trips(store, client):
    """Poll, keep the offset, poll again with it, and get ONLY what is new.

    A byte offset and not a row count: the file is appended to by another
    process while this reads it, so a count would have to re-read from the
    start to mean anything.
    """
    rdir = _fabricate(store, "train-a")
    (rdir / "metrics.jsonl").write_text(
        '{"steps": 1, "loss": 2.0}\n{"steps": 2, "loss": 1.0}\n')

    first = client.get("/lab/runs/train-a/metrics").json()
    assert [row["steps"] for row in first["rows"]] == [1, 2]
    assert first["offset"] > 0

    # Nothing new: the same offset, no rows, and no re-send.
    idle = client.get("/lab/runs/train-a/metrics",
                      params={"offset": first["offset"]}).json()
    assert idle["rows"] == []
    assert idle["offset"] == first["offset"]

    with open(rdir / "metrics.jsonl", "a") as f:
        f.write('{"steps": 3, "loss": 0.5}\n')

    second = client.get("/lab/runs/train-a/metrics",
                        params={"offset": first["offset"]}).json()
    assert [row["steps"] for row in second["rows"]] == [3]
    assert second["offset"] > first["offset"]


def test_a_half_written_metric_line_is_withheld_until_its_newline_arrives(
        store, client):
    """WHOLE LINES ONLY, and the offset does not move past the fragment.

    The trainer is appending to this file from another process while the page
    polls it. A poll that landed mid-`write` and parsed what it found would
    either raise on a truncated JSON object or — worse — drop the row and never
    come back for it, leaving one silent hole in the loss curve.
    """
    rdir = _fabricate(store, "train-a")
    (rdir / "metrics.jsonl").write_text('{"steps": 1, "loss": 2.0}\n')

    first = client.get("/lab/runs/train-a/metrics").json()
    assert [row["steps"] for row in first["rows"]] == [1]

    with open(rdir / "metrics.jsonl", "a") as f:
        f.write('{"steps": 2, "loss')          # torn mid-write

    torn = client.get("/lab/runs/train-a/metrics",
                      params={"offset": first["offset"]}).json()
    assert torn["rows"] == []
    assert torn["offset"] == first["offset"], (
        "the offset moved past a fragment — that row is now lost forever")

    with open(rdir / "metrics.jsonl", "a") as f:
        f.write('": 1.0}\n')                   # the rest of the same row

    whole = client.get("/lab/runs/train-a/metrics",
                       params={"offset": torn["offset"]}).json()
    assert whole["rows"] == [{"steps": 2, "loss": 1.0}]


def test_the_log_offset_round_trips(store, client):
    rdir = _fabricate(store, "train-a")
    (rdir / "run.log").write_text("epoch 1\n")

    first = client.get("/lab/runs/train-a/log").json()
    assert first["text"] == "epoch 1\n"

    with open(rdir / "run.log", "a") as f:
        f.write("epoch 2\n")

    second = client.get("/lab/runs/train-a/log",
                        params={"offset": first["offset"]}).json()
    assert second["text"] == "epoch 2\n"
    assert second["offset"] == first["offset"] + len("epoch 2\n")


def test_a_tail_carries_the_offset_and_the_payload_and_nothing_else(
        store, client):
    """`runs.tail_log` and `read_metrics` both also return `size`, which the
    frozen lines do not carry. A page that started rendering a progress bar off
    it would be depending on a shape nobody froze."""
    rdir = _fabricate(store, "train-a")
    (rdir / "run.log").write_text("x\n")
    (rdir / "metrics.jsonl").write_text('{"steps": 1}\n')

    assert set(client.get("/lab/runs/train-a/log").json()) == {"offset", "text"}
    assert set(client.get("/lab/runs/train-a/metrics").json()) == {"offset", "rows"}


def test_a_run_that_has_logged_nothing_yet_reads_as_empty(store, client):
    """The first poll happens before the child has written a byte. Empty, at
    offset zero — never a 404, which the page would render as "this run is
    gone" a second after starting it."""
    _fabricate(store, "train-a")

    assert client.get("/lab/runs/train-a/log").json() == {"offset": 0, "text": ""}
    assert client.get("/lab/runs/train-a/metrics").json() == {"offset": 0, "rows": []}


# ============================================================================
# GET /lab/runs/{id}/checkpoints
# ============================================================================

def _checkpoint(rdir: Path, step: str, *, with_model: bool = True) -> Path:
    entry = rdir / "train" / "checkpoints" / step
    (entry / "pretrained_model" if with_model else entry).mkdir(parents=True)
    return entry


def test_checkpoints_carry_step_path_and_has_model_newest_first(store, client):
    """`path` is the `pretrained_model` directory, because that is what a
    rollout is handed — not the step directory above it."""
    rdir = _fabricate(store, "train-a")
    _checkpoint(rdir, "100")
    _checkpoint(rdir, "200")

    body = client.get("/lab/runs/train-a/checkpoints").json()

    assert set(body) == {"checkpoints"}
    for entry in body["checkpoints"]:
        assert set(entry) == {"step", "path", "has_model"}
    assert [c["step"] for c in body["checkpoints"]] == [200, 100]
    assert body["checkpoints"][0]["path"].endswith("/checkpoints/200/pretrained_model")
    assert body["checkpoints"][0]["has_model"] is True


def test_the_last_symlink_has_no_step_and_sorts_below_the_numbered_ones(
        store, client):
    """LeRobot writes a `last` symlink beside the numbered checkpoints. `step`
    is `None` for it, which is how a client tells the alias from the checkpoint
    it points at — the two `path`s differ, so without that a rollout form would
    offer the same checkpoint twice under two names."""
    rdir = _fabricate(store, "train-a")
    _checkpoint(rdir, "100")
    step_200 = _checkpoint(rdir, "200")
    (rdir / "train" / "checkpoints" / "last").symlink_to(step_200)

    steps = [c["step"] for c in client.get("/lab/runs/train-a/checkpoints").json()["checkpoints"]]

    assert steps == [200, 100, None]


def test_a_checkpoint_still_being_written_is_omitted_entirely(store, client):
    """A step directory with no `pretrained_model` inside is a checkpoint the
    trainer is still writing, and offering it would hand a rollout half a file.

    **It is ABSENT from the list, not present with `has_model: false`.**
    `runs.checkpoints` skips it before this route ever sees it, so `has_model`
    is structurally `True` on every row this surface can produce. The field is
    on the wire because Track C froze it and because it is where "we listed one
    anyway" would eventually be said — but no request to this route returns
    `false` today, and a page that filters on it is filtering nothing.
    Reported to the integrator; the contract line reads as though `false` were
    reachable here.
    """
    rdir = _fabricate(store, "train-a")
    _checkpoint(rdir, "100")
    _checkpoint(rdir, "200", with_model=False)

    checkpoints = client.get("/lab/runs/train-a/checkpoints").json()["checkpoints"]

    assert [c["step"] for c in checkpoints] == [100]
    assert all(c["has_model"] is True for c in checkpoints)


def test_a_run_with_no_checkpoints_is_an_empty_list(store, client):
    _fabricate(store, "train-a")

    assert client.get("/lab/runs/train-a/checkpoints").json() == {"checkpoints": []}


# ============================================================================
# GET /lab/runs/metrics — the cross-run chart
# ============================================================================

def test_the_cross_run_chart_is_keyed_by_run_then_key(store, client):
    """`{"runs": {id: {key: [[x, y], ...]}}}`. x is the row's `steps` and never
    its position in the file."""
    for run_id, loss in (("train-a", 2.0), ("train-b", 3.0)):
        rdir = _fabricate(store, run_id)
        (rdir / "metrics.jsonl").write_text(
            json.dumps({"steps": 10, "loss": loss}) + "\n")

    body = client.get("/lab/runs/metrics",
                      params=[("ids", "train-a"), ("ids", "train-b"),
                              ("keys", "loss")]).json()

    assert set(body) == {"runs"}
    assert set(body["runs"]) == {"train-a", "train-b"}
    assert body["runs"]["train-a"] == {"loss": [[10.0, 2.0]]}
    assert body["runs"]["train-b"] == {"loss": [[10.0, 3.0]]}


def test_ids_are_accepted_both_repeated_and_comma_separated(store, client):
    """The frozen line is `?ids&keys&max_points` and does not say which
    spelling. Picking one and being wrong costs an empty chart with a 200
    beside it, which reads as "this run logged nothing"."""
    rdir = _fabricate(store, "train-a")
    (rdir / "metrics.jsonl").write_text('{"steps": 1, "loss": 1.0}\n')
    _fabricate(store, "train-b")

    body = client.get("/lab/runs/metrics",
                      params={"ids": "train-a,train-b", "keys": "loss"}).json()

    assert set(body["runs"]) == {"train-a", "train-b"}


def test_a_run_with_no_metrics_is_an_empty_series_and_not_an_error(store, client):
    """This view is a COMPARISON. One absent run must not take the other three
    down with it, so an id that does not exist, has no `metrics.jsonl` or is
    unreadable is `[]` — and it still appears, because the page draws its
    legend from what it asked for."""
    rdir = _fabricate(store, "train-a")
    (rdir / "metrics.jsonl").write_text('{"steps": 1, "loss": 1.0}\n')
    _fabricate(store, "train-empty")

    body = client.get("/lab/runs/metrics",
                      params=[("ids", "train-a"), ("ids", "train-empty"),
                              ("ids", "train-never-existed"),
                              ("keys", "loss")]).json()

    assert body["runs"]["train-a"]["loss"] == [[1.0, 1.0]]
    assert body["runs"]["train-empty"] == {"loss": []}
    assert body["runs"]["train-never-existed"] == {"loss": []}


def test_a_key_no_row_carries_is_an_empty_series_and_not_a_missing_key(
        store, client):
    rdir = _fabricate(store, "train-a")
    (rdir / "metrics.jsonl").write_text('{"steps": 1, "loss": 1.0}\n')

    body = client.get("/lab/runs/metrics",
                      params=[("ids", "train-a"), ("keys", "loss"),
                              ("keys", "eval_loss")]).json()

    assert body["runs"]["train-a"]["eval_loss"] == []


def test_more_runs_than_the_cap_is_a_400(store, client):
    """A request naming every id in the store reads every `metrics.jsonl` on
    disk on ONE GET — an unbounded fan-out behind one URL."""
    ids = [("ids", f"train-{i}") for i in range(compare.MAX_RUNS + 1)]

    response = client.get("/lab/runs/metrics", params=[*ids, ("keys", "loss")])

    assert response.status_code == 400, response.text
    assert str(compare.MAX_RUNS) in response.json()["detail"]


def test_more_keys_than_the_cap_is_a_400(store, client):
    keys = [("keys", f"k{i}") for i in range(compare.MAX_KEYS + 1)]

    response = client.get("/lab/runs/metrics", params=[("ids", "train-a"), *keys])

    assert response.status_code == 400, response.text
    assert str(compare.MAX_KEYS) in response.json()["detail"]


def test_a_max_points_below_two_is_a_400(store, client):
    """The two endpoints are non-negotiable — a downsampler that dropped one
    would misreport where the run started or finished."""
    response = client.get("/lab/runs/metrics",
                          params={"ids": "train-a", "keys": "loss",
                                  "max_points": 1})

    assert response.status_code == 400, response.text


def test_the_collection_metrics_route_is_not_captured_as_a_run_id(store, client):
    """ROUTE ORDER, tested directly — the failure that only appears when
    someone tidies this file into alphabetical order.

    Starlette matches in DECLARATION order, and `metrics` satisfies
    `RUN_ID_RE`, so `GET /lab/runs/metrics` declared below `GET
    /lab/runs/{run_id}` would answer the cross-run chart with `404 no run
    metrics` — or, once a run directory called `metrics` exists, with that
    run's detail, at 200, and the chart would render an empty comparison
    forever.

    So the store below really does contain a run named `metrics`. That is the
    adversarial case: a reordered file passes any test that only checks the
    status code.
    """
    _fabricate(store, "metrics", name="the decoy", spec_summary="not a chart")
    rdir = _fabricate(store, "train-a")
    (rdir / "metrics.jsonl").write_text('{"steps": 1, "loss": 1.0}\n')

    body = client.get("/lab/runs/metrics",
                      params={"ids": "train-a", "keys": "loss"}).json()

    assert set(body) == {"runs"}, f"the run detail answered instead: {body}"
    assert body["runs"]["train-a"]["loss"] == [[1.0, 1.0]]
    # And the decoy is still reachable as a run, so this is route ORDER and not
    # a run id that got refused.
    assert client.get("/lab/runs/metrics/log").status_code == 200


# ============================================================================
# POST /lab/runs/{id}/stop and DELETE /lab/runs/{id}
# ============================================================================

def test_stop_answers_ok(store, client):
    """`{"ok": true}` and not the run record: the status a quarter of a second
    after SIGINT is not the status the operator is asking about, and shipping
    it would invite a page that renders it as final."""
    _fabricate(store, "train-a")

    response = client.post("/lab/runs/train-a/stop")

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True}


def test_stopping_an_unknown_run_is_a_404(store, client):
    assert client.post("/lab/runs/train-nope/stop").status_code == 404


def test_delete_answers_ok_and_removes_the_directory(store, client):
    rdir = _fabricate(store, "train-a")
    (rdir / "run.log").write_text("x\n")

    response = client.delete("/lab/runs/train-a")

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True}
    assert not rdir.exists()
    assert _ids(client) == []


def test_delete_while_the_run_is_alive_is_a_409(store, client, reaper):
    """There is deliberately no "delete anyway". The child is DETACHED and the
    server is not its parent, so removing the directory would leave a live
    training job writing into an unlinked file, holding the GPU, with nothing
    on disk left to say it exists.

    409 and not 500: `runs.delete_run` raises `RuntimeError` for exactly this,
    and the `api/errors` ladder has no rung for it — an uncaught one reads as a
    broken server rather than as "stop it first".
    """
    rdir = _live_run(store, "train-live", reaper)

    response = client.delete("/lab/runs/train-live")

    assert response.status_code == 409, response.text
    assert "still running" in response.json()["detail"]
    assert rdir.exists()


def test_delete_of_an_unknown_id_is_a_404(store, client):
    response = client.delete("/lab/runs/train-nope")

    assert response.status_code == 404, response.text
    assert "train-nope" in response.json()["detail"]


def test_delete_cannot_escape_the_run_store(store, client):
    """This is the one call that removes a TREE. `".."` matches `RUN_ID_RE`, so
    the containment check is what stands between it and an rmtree of
    `outputs/`."""
    _fabricate(store, "train-a")

    response = client.delete("/lab/runs/%2e%2e")

    assert response.status_code == 400, response.text
    assert "escapes the run store" in response.json()["detail"]
    assert store.exists()
    assert (store / "train-a").exists()


# ============================================================================
# require_local — the matrix
# ============================================================================

def test_from_the_lan_exactly_the_three_process_routes_are_refused(
        home, store, lan):
    """Both halves, in one test, because the interesting failure is one-sided.

    A gate that 403s everything is a headset that cannot triage — and triage
    from inside the headset is the entire reason `--host 0.0.0.0` is on. A gate
    that 403s nothing is a LAN that can start an hours-long GPU job and kill a
    running one. So the assertion is the SHAPE of the matrix: exactly three
    refusals, and every GET still answered.
    """
    make_dataset(home / REPO, n_episodes=2)
    _fabricate(store, "train-a")

    refused = [
        ("POST", "/lab/runs/train", {"repo_id": REPO}),
        ("POST", "/lab/runs/train-a/stop", None),
        ("DELETE", "/lab/runs/train-a", None),
    ]
    for method, url, body in refused:
        response = lan.request(method, url, json=body)
        assert response.status_code == 403, f"{method} {url} -> {response.text}"
        assert LAN_HOST in response.json()["detail"]

    for url in ("/lab/runs",
                "/lab/runs/train-a",
                "/lab/runs/train-a/metrics",
                "/lab/runs/train-a/log",
                "/lab/runs/train-a/checkpoints",
                "/lab/runs/metrics"):
        response = lan.get(url)
        assert response.status_code == 200, f"GET {url} -> {response.text}"

    # And nothing was launched or removed on the way through.
    assert _ids(lan) == ["train-a"]


def test_the_gate_refuses_before_the_spec_is_even_read(home, store, lan):
    """403 and not 404, on a repo-id that does not exist. The gate is a route
    DEPENDENCY, so it runs before the handler — a remote caller cannot use the
    error code to find out which datasets are on this machine."""
    response = lan.post("/lab/runs/train", json={"repo_id": "local/nope"})

    assert response.status_code == 403, response.text


def test_allow_remote_control_opens_the_three(home, store):
    """The escape hatch, for the evening Oscar wants a job started from the
    couch. Asserted so the 403s above are known to be the gate and not a
    routing accident."""
    make_dataset(home / REPO, n_episodes=2)
    _fabricate(store, "train-a")
    remote = _client(home, host=LAN_HOST, allow_remote_control=True)

    assert remote.post("/lab/runs/train",
                       json={"repo_id": REPO, "job_name": "remote"}
                       ).status_code == 200
    assert remote.post("/lab/runs/train-a/stop").status_code == 200
    assert remote.delete("/lab/runs/train-a").status_code == 200
