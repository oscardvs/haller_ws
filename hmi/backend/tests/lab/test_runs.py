# hmi/backend/tests/lab/test_runs.py
"""`lab/runs.py` — the four properties a detached job store exists for.

Every test here launches a REAL child. A mocked `Popen` would prove that
`launch` calls `Popen`, which is not the thing that breaks; what breaks is a
run that reads `done` after being killed, a status inferred from a recycled
pid, a metric row torn in half by a poll that arrived mid-write, and a log tail
that re-sends an hour of output. All four need a process that actually runs,
actually dies, and actually appends to a file while it is being read.

The stand-in runner is written into `tmp_path` and put on `PYTHONPATH`, never
added to the repo: a runner module living in `haller_hmi/runners/` that exists
only for tests is one an operator can launch from the UI. It imports
`write_result` from the module under test, because the child writing that file
and `load()` reading it are two halves of one contract — a test that wrote
`result.json` by hand would agree with itself about the key names and prove
nothing.

Waits are bounded (`_wait_for`) and every launched process group is killed by
the fixture, so a failing assertion cannot leave a sleeper behind.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from haller_hmi.lab import runs

#: `hmi/backend`, so a subprocess can import `haller_hmi` from a clean interpreter.
BACKEND = Path(__file__).resolve().parents[2]

#: The stand-in runner. Spec keys drive it: `say` (a line on stdout), `metrics`
#: (rows appended to metrics.jsonl), `checkpoints` (step directories),
#: `sleep_s` (stay alive), `ignore_signals`, `write_result`.
#:
#: `print()` is deliberately NOT flushed: stdout redirected to a file is block
#: buffered, so text appearing in run.log while the child still sleeps is proof
#: that `PYTHONUNBUFFERED=1` reached the child rather than proof that the test
#: flushed. The `finally` is the shape every real runner must have.
#:
#: It touches `ready` LAST, inside the try, and `_Lab.sleeper` waits for that
#: file rather than for the pid: a pid exists from `exec`, seconds before the
#: interpreter has imported anything, and a test that signalled at that point
#: was signalling a process that had installed no handler and entered no `try`.
#: Both stop tests failed that way first — the same race a real runner has, and
#: the reason a real one must install its handler before it does any work.
FAKE_RUNNER = '''\
import json
import sys
import time
from pathlib import Path

from haller_hmi.lab.runs import write_result

spec = json.loads(Path(sys.argv[1]).read_text())
rdir = Path(spec["run_dir"])
status, code = "done", 0

if spec.get("ignore_signals"):
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
    _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)

try:
    print(spec.get("say") or ("start " + spec["run_id"]))
    for row in spec.get("metrics") or []:
        with open(rdir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\\n")
    for step in spec.get("checkpoints") or []:
        (rdir / "train" / "checkpoints" / str(step) / "pretrained_model").mkdir(
            parents=True, exist_ok=True)
    (rdir / "ready").write_text("1")
    if spec.get("sleep_s"):
        time.sleep(float(spec["sleep_s"]))
except KeyboardInterrupt:
    status, code = "stopped", 130
finally:
    if spec.get("write_result", True):
        write_result(rdir, status, code)
'''

#: A second stand-in, reporting what `launch` gave it rather than doing work:
#: the `PYTHONPATH` the child was started with, and where its own `haller_hmi`
#: came from. Written from INSIDE the child on purpose — a path the launcher
#: computed and then failed to pass would satisfy an assertion on `child_env`
#: and still kill every real run.
PATH_RUNNER = '''\
import json
import os
import sys
from pathlib import Path

import haller_hmi
from haller_hmi.lab.runs import write_result

spec = json.loads(Path(sys.argv[1]).read_text())
rdir = Path(spec["run_dir"])
(rdir / "seen.json").write_text(json.dumps({
    "pythonpath": os.environ.get("PYTHONPATH", ""),
    "haller_hmi": haller_hmi.__file__,
}))
write_result(rdir, "done", 0)
'''

#: Long enough that nothing finishes on its own inside a test.
FOREVER_S = 120.0

#: Ceiling on every wait. A child that has not started in ten seconds is a
#: failure, not a slow box.
WAIT_S = 10.0


def _wait_for(predicate, timeout: float = WAIT_S, interval: float = 0.02) -> bool:
    """Poll `predicate` until it holds or the deadline passes. Bounded, always."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class _Lab:
    """A private run store plus a launcher for the stand-in runner."""

    def __init__(self, store: Path, pythonpath: str):
        self.store = store
        prior = os.environ.get("PYTHONPATH", "")
        self.env = {"PYTHONPATH": os.pathsep.join([pythonpath, prior]) if prior else pythonpath}
        self.pids: list[int] = []

    def launch(self, *, name: str = "", tags=None, spec_summary: str = "", **spec) -> dict:
        record = runs.launch("train", spec, name=name, env=self.env,
                             tags=tags, spec_summary=spec_summary)
        self.pids.append(int(record["pid"]))
        return record

    def sleeper(self, **spec) -> dict:
        """A child that is RUNNING ITS OWN CODE by the time this returns — its
        signal handlers installed and its `try` entered, not merely `exec`ed."""
        spec.setdefault("sleep_s", FOREVER_S)
        record = self.launch(**spec)
        ready = self.store / record["id"] / "ready"
        assert _wait_for(ready.exists), "child never reached its work"
        return record

    def kill_all(self) -> None:
        for pid in self.pids:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except OSError:
                pass


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    store = tmp_path / "runs"
    monkeypatch.setenv(runs.RUNS_DIR_ENV, str(store))
    # The env branch of `runner_python()`, pointed at the interpreter running
    # the tests: `~/venvs/haller-lab` has lerobot and torch in it and a test
    # must never pay a CUDA context to start a `time.sleep`.
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, sys.executable)
    (tmp_path / "fake_runner.py").write_text(FAKE_RUNNER)
    monkeypatch.setitem(runs.RUNNERS, "train", "fake_runner")
    harness = _Lab(store, str(tmp_path))
    yield harness
    harness.kill_all()


def _fabricate(store: Path, run_id: str, **fields) -> Path:
    """A run.json written by hand, for the states a real child cannot produce
    on demand (a stale pid, a start time in the past)."""
    rdir = store / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    record = {"id": run_id, "kind": "train", "name": "", "status": "running",
              "started_at": "2026-08-27T10:00:00+00:00", "pid": 0}
    record.update(fields)
    (rdir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return rdir


# ---- the store and the interpreter ----

def test_runs_dir_follows_the_env(lab, tmp_path):
    assert runs.runs_dir() == tmp_path / "runs"


def test_runs_dir_defaults_under_the_working_directory(monkeypatch, tmp_path):
    """No env set: `<cwd>/outputs/runs`, which is the repo root in every
    documented way of starting the HMI."""
    monkeypatch.delenv(runs.RUNS_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    assert runs.runs_dir() == tmp_path / "outputs" / "runs"


def test_runner_python_prefers_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, str(tmp_path / "elsewhere" / "python"))

    assert runs.runner_python() == tmp_path / "elsewhere" / "python"


def test_runner_python_falls_back_to_the_lab_venv(monkeypatch, tmp_path):
    lab_python = tmp_path / "haller-lab" / "bin" / "python"
    lab_python.parent.mkdir(parents=True)
    lab_python.write_text("")
    monkeypatch.delenv(runs.LAB_PYTHON_ENV, raising=False)
    monkeypatch.setattr(runs, "LAB_PYTHON", lab_python)

    assert runs.runner_python() == lab_python


def test_runner_python_falls_back_to_this_interpreter(monkeypatch, tmp_path):
    monkeypatch.delenv(runs.LAB_PYTHON_ENV, raising=False)
    monkeypatch.setattr(runs, "LAB_PYTHON", tmp_path / "no-such-venv" / "python")

    assert runs.runner_python() == Path(sys.executable)


def test_runner_python_never_imports_anything_to_check_the_interpreter(monkeypatch):
    """A PATH CHECK ONLY. Verifying the interpreter by importing torch would
    cost the serving process a CUDA context in the teleop latency path, so a
    path that does not exist is returned rather than raised on."""
    monkeypatch.setenv(runs.LAB_PYTHON_ENV, "/nonexistent/venv/bin/python")

    resolved = runs.runner_python()

    assert resolved == Path("/nonexistent/venv/bin/python")
    assert not resolved.exists()


def test_the_child_is_launched_with_the_lab_interpreter(lab):
    record = lab.launch(name="interp")

    assert record["argv"][0] == sys.executable
    assert record["runner_python"] == sys.executable
    assert record["argv"][1:3] == ["-m", "fake_runner"]


def test_backend_dir_is_the_directory_haller_hmi_lives_in():
    """The `parents[2]` walk in `runs.py`, checked from a file that counts its
    own — the two arithmetics are independent, so moving either module makes
    this fail rather than making every run fail."""
    assert runs.BACKEND_DIR == BACKEND
    assert (runs.BACKEND_DIR / "haller_hmi" / "__init__.py").exists()


def test_the_child_is_given_the_path_to_import_haller_hmi(lab, tmp_path, monkeypatch):
    """`runner_python()` is a venv with no `haller_hmi` installed in it.

    The lab venv is lerobot 0.6.1 + torch and nothing of this package, so
    `-m haller_hmi.runners.train_runner` resolves ONLY because `launch` puts
    `BACKEND_DIR` on the child's `PYTHONPATH`. It was missing once, and every
    kind — train, eval, rollout, export — died on `ModuleNotFoundError` after
    the UI had reported the run started.

    FIRST, so a run imports the checkout that launched it and not an older copy
    further down an inherited path, and the rest kept: `env` is how the caller
    hands the child a module of its own, and this box exports a long ROS
    `PYTHONPATH` the child inherits.
    """
    (tmp_path / "path_runner.py").write_text(PATH_RUNNER)
    monkeypatch.setitem(runs.RUNNERS, "train", "path_runner")

    record = lab.launch(name="path")
    seen_path = lab.store / record["id"] / "seen.json"
    assert _wait_for(seen_path.exists), "child never reached its work"
    seen = json.loads(seen_path.read_text())
    parts = seen["pythonpath"].split(os.pathsep)

    assert parts[0] == str(runs.BACKEND_DIR)
    assert Path(seen["haller_hmi"]).resolve() == BACKEND / "haller_hmi" / "__init__.py"
    assert str(tmp_path) in parts[1:], "the caller's own entry was dropped"


def test_there_is_no_record_kind(lab):
    """The single biggest divergence from the kit. Recording owns the Feetech
    bus and the bus stays in the serving process, because `/estop` walks every
    motor in-process — a detached child cannot be allowed to own it."""
    assert "record" not in runs.RUNNERS
    assert set(runs.RUNNERS) == {"train", "eval", "rollout", "export"}

    with pytest.raises(ValueError, match="unknown job kind"):
        runs.launch("record", {"repo_id": "local/x"})


def test_an_unknown_kind_is_refused_before_any_directory_is_written(lab):
    with pytest.raises(ValueError, match="unknown job kind"):
        runs.launch("finetune", {})

    assert not lab.store.exists() or list(lab.store.iterdir()) == []


# ---- detachment ----

def test_the_child_gets_its_own_session_so_it_outlives_the_server(lab):
    """`start_new_session=True`: the job survives an HMI restart, and `stop()`
    can signal the whole group without ever reaching the server itself."""
    record = lab.sleeper(name="detached")
    pid = record["pid"]

    assert os.getpgid(pid) == pid, "child is not its own process-group leader"
    assert os.getpgid(pid) != os.getpgid(os.getpid())


def test_the_log_is_appended_live_because_the_child_runs_unbuffered(lab):
    """The child prints WITHOUT flushing and then sleeps. Text arriving in
    run.log while it is still alive is `PYTHONUNBUFFERED=1` doing its job;
    without it the same line sits in an 8 KB buffer for the whole run and the
    log pane looks hung."""
    record = lab.sleeper(name="live", say="epoch 1 loss 0.42")
    log = lab.store / record["id"] / "run.log"

    assert _wait_for(lambda: "epoch 1 loss 0.42" in log.read_text())
    assert runs.load(record["id"])["alive"], "the log arrived only because it exited"
    assert log.read_text().splitlines()[0].startswith("$ "), "the argv header is missing"


# ---- status is written by the runner, never inferred ----

def test_a_clean_run_reaches_done_through_result_json(lab):
    record = lab.launch(name="clean", metrics=[{"step": 1, "loss": 0.5}],
                        checkpoints=[100])
    run_id = record["id"]

    assert _wait_for(lambda: (lab.store / run_id / "result.json").exists())
    loaded = runs.load(run_id)

    assert loaded["status"] == "done"
    assert loaded["exit_code"] == 0
    assert loaded["finished_at"]
    assert loaded["alive"] is False
    assert runs.read_metrics(run_id)["rows"] == [{"step": 1, "loss": 0.5}]
    assert [c["step"] for c in runs.checkpoints(run_id)] == [100]


def test_a_hard_killed_child_with_no_result_reads_died_not_done(lab):
    """SIGKILL leaves no `finally` to run and so no `result.json`. After a
    backend restart the server is not the process's parent and cannot reap it,
    so a dead pid alone cannot tell a clean finish from a crash — and calling
    that `done` would report a training run that never trained as finished."""
    record = lab.sleeper(name="doomed")
    run_id = record["id"]

    os.killpg(os.getpgid(record["pid"]), signal.SIGKILL)

    assert _wait_for(lambda: not runs.load(run_id)["alive"])
    loaded = runs.load(run_id)

    assert not (lab.store / run_id / "result.json").exists()
    assert loaded["status"] == "died"
    assert loaded["status"] != "done"


def test_a_runner_that_reports_a_failure_keeps_that_status(lab):
    """`result.json` is authoritative in both directions: a dead pid with one
    is whatever the runner said, not `died`."""
    rdir = _fabricate(lab.store, "train-20260827-101500-boom", pid=0)
    runs.write_result(rdir, "failed", exit_code=3, error="CUDA out of memory")

    loaded = runs.load("train-20260827-101500-boom")

    assert loaded["status"] == "failed"
    assert loaded["exit_code"] == 3
    assert loaded["error"] == "CUDA out of memory"


def test_a_recycled_pid_is_not_the_run_that_claimed_it(lab):
    """`os.kill(pid, 0)` alone would call this very test process a training job.
    The cmdline check is what stops a finished run showing as running forever on
    a workstation that spawns processes all day."""
    run_id = "train-20260827-090000-recycled"
    _fabricate(lab.store, run_id, pid=os.getpid())

    assert runs._pid_alive(os.getpid(), run_id) is False
    loaded = runs.load(run_id)

    assert loaded["alive"] is False
    assert loaded["status"] == "died"


def test_a_live_child_answers_only_to_its_own_run_id(lab):
    record = lab.sleeper(name="mine")

    assert runs._pid_alive(record["pid"], record["id"]) is True
    assert runs._pid_alive(record["pid"], "train-20260101-000000-other") is False


def test_pid_zero_is_never_alive(lab):
    """A run.json written by the launch-failure path carries `pid: null`, and
    `kill(0, 0)` addresses the caller's entire process group."""
    assert runs._pid_alive(0, "train-20260827-090000-x") is False


# ---- metrics: whole lines only ----

def _write_metrics(store: Path, run_id: str, text: str) -> Path:
    rdir = _fabricate(store, run_id)
    with open(rdir / "metrics.jsonl", "a") as f:
        f.write(text)
    return rdir


def test_read_metrics_holds_back_a_half_written_line_and_returns_it_next_poll(lab):
    """The runner appends while the page polls. A record caught mid-write is
    picked up on the NEXT poll rather than dropped or half-parsed, which is
    what byte offsets buy over line counts."""
    run_id = "train-20260827-110000-partial"
    _write_metrics(lab.store, run_id,
                   '{"step": 1, "loss": 0.9}\n{"step": 2, "lo')

    first = runs.read_metrics(run_id)

    assert first["rows"] == [{"step": 1, "loss": 0.9}]
    assert first["offset"] < first["size"], "the partial line was consumed"

    _write_metrics(lab.store, run_id, 'ss": 0.8}\n')
    second = runs.read_metrics(run_id, first["offset"])

    assert second["rows"] == [{"step": 2, "loss": 0.8}]
    assert second["offset"] == second["size"]


def test_read_metrics_returns_nothing_while_the_only_line_is_incomplete(lab):
    run_id = "train-20260827-110100-nothing"
    _write_metrics(lab.store, run_id, '{"step": 1, "lo')

    out = runs.read_metrics(run_id)

    assert out["rows"] == []
    assert out["offset"] == 0, "the offset must not move past an unread record"


def test_read_metrics_never_returns_the_same_row_twice(lab):
    run_id = "train-20260827-110200-once"
    _write_metrics(lab.store, run_id, '{"step": 1}\n{"step": 2}\n')

    first = runs.read_metrics(run_id)
    again = runs.read_metrics(run_id, first["offset"])

    assert first["rows"] == [{"step": 1}, {"step": 2}]
    assert again["rows"] == []
    assert again["offset"] == first["offset"]


def test_read_metrics_skips_a_corrupt_row_rather_than_losing_the_chart(lab):
    run_id = "train-20260827-110300-corrupt"
    _write_metrics(lab.store, run_id, '{"step": 1}\nnot json at all\n{"step": 3}\n')

    out = runs.read_metrics(run_id)

    assert out["rows"] == [{"step": 1}, {"step": 3}]


def test_read_metrics_on_a_run_that_logged_nothing(lab):
    _fabricate(lab.store, "train-20260827-110400-quiet")

    assert runs.read_metrics("train-20260827-110400-quiet") == {
        "offset": 0, "rows": [], "size": 0}


# ---- log tail ----

def test_tail_log_returns_only_what_is_new(lab):
    run_id = "train-20260827-120000-tail"
    rdir = _fabricate(lab.store, run_id)
    (rdir / "run.log").write_text("line one\n")

    first = runs.tail_log(run_id)
    (rdir / "run.log").open("a").write("line two\n")
    second = runs.tail_log(run_id, first["offset"])

    assert first["text"] == "line one\n"
    assert second["text"] == "line two\n"
    assert second["offset"] == second["size"]


def test_a_client_far_behind_gets_the_tail_not_the_whole_log(lab):
    """The page reattaches to a run that has been printing for an hour. Sending
    it every byte since offset 0 is a 50 MB response to draw one screen."""
    run_id = "train-20260827-120100-behind"
    rdir = _fabricate(lab.store, run_id)
    (rdir / "run.log").write_text("".join(f"{i:09d}\n" for i in range(1000)))
    size = (rdir / "run.log").stat().st_size

    out = runs.tail_log(run_id, offset=0, max_bytes=100)

    assert out["size"] == size
    assert len(out["text"]) == 100
    assert out["offset"] == size
    assert out["text"].endswith("000000999\n"), "the tail, not the head"


def test_tail_log_clamps_an_offset_past_the_end(lab):
    """A client holding an offset from before the log was rotated or truncated
    must get an empty read, not a seek past EOF."""
    run_id = "train-20260827-120200-clamp"
    rdir = _fabricate(lab.store, run_id)
    (rdir / "run.log").write_text("short\n")

    out = runs.tail_log(run_id, offset=10_000)

    assert out["text"] == ""
    assert out["offset"] == out["size"] == 6


def test_tail_log_on_a_run_with_no_log_yet(lab):
    _fabricate(lab.store, "train-20260827-120300-silent")

    assert runs.tail_log("train-20260827-120300-silent") == {
        "offset": 0, "text": "", "size": 0}


# ---- checkpoints ----

def _checkpoint(store: Path, run_id: str, name: str, *, model: bool = True) -> Path:
    entry = store / run_id / "train" / "checkpoints" / name
    (entry / "pretrained_model" if model else entry).mkdir(parents=True)
    return entry


def test_checkpoints_are_newest_step_first(lab):
    run_id = "train-20260827-130000-ckpt"
    _fabricate(lab.store, run_id)
    for name in ("000100", "020000", "005000"):
        _checkpoint(lab.store, run_id, name)

    found = runs.checkpoints(run_id)

    assert [c["step"] for c in found] == [20000, 5000, 100]
    assert found[0]["path"].endswith("020000/pretrained_model")


def test_the_last_symlink_sorts_after_every_numbered_step(lab):
    """`last` is LeRobot's alias for the newest step, so it must not displace
    the real checkpoint the rollout launcher wants at row one."""
    run_id = "train-20260827-130100-last"
    _fabricate(lab.store, run_id)
    _checkpoint(lab.store, run_id, "000100")
    _checkpoint(lab.store, run_id, "last")

    found = runs.checkpoints(run_id)

    assert [c["step"] for c in found] == [100, None]
    assert found[-1]["name"] == "last"


def test_a_checkpoint_still_being_written_is_not_offered(lab):
    """No `pretrained_model` yet means LeRobot is mid-save. Offering it would
    hand a rollout half a file."""
    run_id = "train-20260827-130200-half"
    _fabricate(lab.store, run_id)
    _checkpoint(lab.store, run_id, "000100")
    _checkpoint(lab.store, run_id, "000200", model=False)

    assert [c["step"] for c in runs.checkpoints(run_id)] == [100]


def test_checkpoints_of_a_run_that_never_trained(lab):
    _fabricate(lab.store, "train-20260827-130300-none")

    assert runs.checkpoints("train-20260827-130300-none") == []


# ---- listing ----

def test_list_runs_is_newest_first_by_start_time_not_by_name(lab):
    """Run ids are prefixed with the kind, so sorting by directory name would
    group every rollout above every train regardless of when they ran."""
    _fabricate(lab.store, "aaa-20260827-100000", started_at="2026-08-27T10:00:00+00:00")
    _fabricate(lab.store, "zzz-20260827-120000", started_at="2026-08-27T12:00:00+00:00")
    _fabricate(lab.store, "mmm-20260827-110000", started_at="2026-08-27T11:00:00+00:00")

    listed = [r["id"] for r in runs.list_runs()]

    assert listed == ["zzz-20260827-120000", "mmm-20260827-110000", "aaa-20260827-100000"]


def test_a_directory_without_a_run_json_is_not_a_run(lab):
    _fabricate(lab.store, "train-20260827-140000-real")
    (lab.store / "scratch").mkdir()

    assert [r["id"] for r in runs.list_runs()] == ["train-20260827-140000-real"]


def test_list_runs_on_a_store_that_does_not_exist_yet(lab):
    assert runs.list_runs() == []


def test_an_old_run_json_still_renders_with_tags_and_a_summary(lab):
    """`tags` and `spec_summary` are Haller's additions; a run.json written
    before them must not make the listing render `undefined`."""
    _fabricate(lab.store, "train-20260827-140100-old")

    loaded = runs.load("train-20260827-140100-old")

    assert loaded["tags"] == []
    assert loaded["spec_summary"] == ""


# ---- what a run remembers it was asked for ----

def test_launch_records_the_tags_and_a_one_line_summary(lab):
    record = lab.launch(name="tagged", tags=["overnight", "act"],
                        spec_summary="retrain\nafter the prune")

    assert record["tags"] == ["overnight", "act"]
    assert record["spec_summary"] == "retrain after the prune", "must stay one line"
    assert runs.load(record["id"])["spec_summary"] == "retrain after the prune"


def test_a_default_summary_names_the_dataset_episodes_policy_and_steps(lab):
    """The only place a run remembers what it was asked for once its spec is
    superseded, so it is built ONCE at launch and never re-derived in the UI."""
    record = lab.launch(name="sum", repo_id="local/so101_pick_cube",
                        episodes=[0, 1, 2], policy_type="act", steps=100_000)

    assert record["spec_summary"] == (
        "train · local/so101_pick_cube · 3 episodes · act · 100000 steps")


def test_a_summary_names_the_checkpoint_a_rollout_was_pointed_at(lab):
    """`Path(policy_path).name` is `pretrained_model` for every checkpoint
    LeRobot has ever written, so the step and the run it came from are what get
    printed instead."""
    record = lab.launch(
        name="roll",
        policy_path="/home/odesha/outputs/runs/train-20260827-090000-act"
                    "/train/checkpoints/020000/pretrained_model")

    assert record["spec_summary"] == "train · train-20260827-090000-act/020000"


def test_a_summary_of_only_whitespace_falls_back_to_the_default(lab):
    record = lab.launch(name="blank", spec_summary="   ", repo_id="local/x")

    assert record["spec_summary"] == "train · local/x"


def test_the_spec_is_stored_with_the_run_id_and_directory_the_child_needs(lab):
    record = lab.launch(name="spec", repo_id="local/x")
    stored = json.loads((lab.store / record["id"] / "spec.json").read_text())

    assert stored["run_id"] == record["id"]
    assert stored["run_dir"] == str(lab.store / record["id"])
    assert stored["repo_id"] == "local/x"


# ---- stop ----

def test_stop_sigints_the_group_and_the_runner_records_the_wind_down(lab, monkeypatch):
    """SIGINT, because LeRobot's training loop treats it as "wind down" and
    saves a checkpoint. The runner's own `finally` is what writes the status."""
    monkeypatch.setattr(runs, "STOP_GRACE_S", 5.0)
    record = lab.sleeper(name="windable")

    stopped = runs.stop(record["id"])

    assert stopped["stop_requested"] is True
    assert stopped["status"] == "stopped"
    assert stopped["exit_code"] == 130
    assert stopped["alive"] is False


def test_stop_never_sends_sigkill(lab, monkeypatch):
    """A half-killed rollout leaves an arm under torque in an unknown pose, so
    the escalation stops at SIGTERM. A child ignoring both is left alive and
    reported as running — the operator's problem to see, not ours to solve with
    a signal that cannot be caught."""
    monkeypatch.setattr(runs, "STOP_GRACE_S", 1.0)
    record = lab.sleeper(name="stubborn", ignore_signals=True)

    stopped = runs.stop(record["id"])

    assert stopped["alive"] is True
    assert stopped["status"] == "running"


def test_stop_on_a_finished_run_is_a_no_op(lab):
    record = lab.launch(name="over")
    assert _wait_for(lambda: not runs.load(record["id"])["alive"])

    stopped = runs.stop(record["id"])

    assert stopped["status"] == "done"
    assert "stop_requested" not in stopped


# ---- delete ----

def test_delete_run_refuses_while_the_run_is_alive(lab):
    """The route turns this into a 409. Unlinking the directory under a live
    child leaves it writing into a file nobody can read and takes its log with
    it, so there would be nothing left to explain what happened."""
    record = lab.sleeper(name="busy")

    with pytest.raises(RuntimeError, match="still running"):
        runs.delete_run(record["id"])

    assert (lab.store / record["id"]).is_dir()


def test_delete_run_removes_the_directory_once_it_is_finished(lab):
    record = lab.launch(name="finished")
    assert _wait_for(lambda: not runs.load(record["id"])["alive"])

    out = runs.delete_run(record["id"])

    assert out["id"] == record["id"]
    assert not (lab.store / record["id"]).exists()
    assert runs.list_runs() == []


def test_delete_run_on_a_run_that_is_not_there(lab):
    lab.store.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        runs.delete_run("train-20260827-150000-ghost")


# ---- run ids that would leave the store ----

@pytest.mark.parametrize("run_id", [
    pytest.param("", id="empty"),
    pytest.param("../evil", id="traversal"),
    pytest.param("a/b", id="separator"),
    pytest.param("train 1", id="space"),
    pytest.param("..", id="parent"),
    pytest.param(".", id="self"),
])
def test_a_run_id_that_would_leave_the_store_is_refused(lab, run_id):
    """`..` matches the kit's `RUN_ID_RE` — dots are legal in a run id — and
    names the parent of the store, which is why the containment check exists on
    top of the regex."""
    with pytest.raises(ValueError):
        runs.run_dir(run_id)


def test_delete_run_refuses_to_remove_the_store_itself(lab, tmp_path):
    lab.store.mkdir(parents=True)
    (tmp_path / "keepme.txt").write_text("not a run")

    for run_id in ("..", "."):
        with pytest.raises(ValueError):
            runs.delete_run(run_id)

    assert lab.store.is_dir()
    assert (tmp_path / "keepme.txt").exists()


def test_a_legal_run_id_resolves_inside_the_store(lab):
    assert runs.run_dir("train-20260827-160000-ok") == lab.store / "train-20260827-160000-ok"


def test_new_run_id_slugs_the_name_into_something_run_dir_accepts(lab):
    run_id = runs.new_run_id("train", "local/so101 pick cube")

    assert runs.RUN_ID_RE.match(run_id)
    assert run_id.startswith("train-")
    assert run_id.endswith("-local-so101-pick-cube")
    assert runs.run_dir(run_id).parent == lab.store


# ---- result.json is one contract, written by one function ----

def test_write_result_is_what_load_reads(lab, tmp_path):
    rdir = _fabricate(lab.store, "train-20260827-170000-result")

    runs.write_result(rdir, "done", exit_code=0)
    payload = json.loads((rdir / "result.json").read_text())

    assert set(payload) == {"status", "exit_code", "error", "finished_at"}
    assert payload["finished_at"].endswith("+00:00"), "UTC, so two boxes agree"


def test_every_runner_target_is_importable():
    """Each `RUNNERS` value must name a module that EXISTS and runs as `-m`.

    This is the test that was missing, and its absence hid a real defect: every
    entry read `haller_hmi.runners.train` where the file is `train_runner.py`,
    so every launch would have died instantly with `No module named
    'haller_hmi.runners.train'`.

    Nothing else could catch it. The route and launch tests point
    `$HALLER_LAB_PYTHON` at `/bin/true`, which ignores its arguments and exits
    0 — so the child "runs", `result.json` never appears, the run reads `died`
    exactly as a crashed job would, and the run directory is created either way.
    A launch test built that way cannot distinguish a working target from a
    misspelled one, which is why this asserts the import directly.

    In a SUBPROCESS: importing the runners here would pull lerobot into the
    serving-venv test process, and `runners/` is the one package allowed to
    import it. `-c "import X"` proves the module path resolves without this
    interpreter keeping it.
    """
    for kind, module in runs.RUNNERS.items():
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True, text=True, cwd=str(BACKEND), timeout=120,
            check=False,
        )
        assert result.returncode == 0, (
            f"RUNNERS[{kind!r}] = {module!r} does not import: {result.stderr.strip()}"
        )


def test_every_runner_target_has_a_main_guard():
    """`launch` runs the child as `python -m <module>`, which executes the module
    body and nothing else — a runner without `if __name__ == "__main__"` would
    import cleanly, do nothing, exit 0, and report a successful run that never
    ran."""
    for kind, module in runs.RUNNERS.items():
        path = Path(BACKEND, *module.split(".")).with_suffix(".py")
        assert path.exists(), f"RUNNERS[{kind!r}] names {path}, which is not there"
        assert '__name__ == "__main__"' in path.read_text(), (
            f"{path.name} has no main guard, so `-m` would exit 0 having done nothing"
        )
