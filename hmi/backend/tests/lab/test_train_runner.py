# hmi/backend/tests/lab/test_train_runner.py
"""`runners/train_runner.py` and `runners/_common.py`, WITHOUT lerobot.

Every test here runs under the SERVING venv (lerobot 0.5.1), which cannot run
the thing being tested — the real child runs under `~/venvs/haller-lab` at
lerobot 0.6.1 + torch 2.11.0+cu130. That is the point rather than a limitation:
the module keeps every heavy import inside `main()`'s call tree, so the argv it
builds, the metrics it taps and its `--dry-run` are all reachable from a process
that has no trainer at all. If a `import lerobot` ever migrates to module scope,
`test_importing_the_runner_pulls_in_neither_lerobot_nor_torch` fails here rather
than an operator discovering it as a 3-second stall on the teleop path.

Nothing here starts a training run. Training occupies the 4080 SUPER for hours,
and a runner whose only testable surface was "run it" would have no tests.

The metrics handler gets a FAKE `MetricsTracker`, and the test that matters is
`test_the_step_recorded_is_11500_not_the_rounded_12K`: LeRobot logs the tracker
OBJECT and its `__str__` rounds through `format_big_number` (verified under the
lab venv: 11500 -> "12K", 12499 -> "12K"), so a handler that scraped the string
would draw the loss curve against an x-axis it made up.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from haller_hmi.runners import _common, train_runner

BACKEND = Path(__file__).resolve().parents[2]


# ---- helpers --------------------------------------------------------------

def make_spec(tmp_path: Path, **over) -> dict:
    """The minimum `lab/runs.launch` puts on disk, plus whatever a test needs.

    `run_id` and `run_dir` are stamped in by `launch`, so every real spec has
    them and `build_argv` may read them without a default.
    """
    run_dir = tmp_path / "train-20260827-120000"
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "run_id": "train-20260827-120000",
        "run_dir": str(run_dir),
        "repo_id": "local/so101_pick_cube",
    }
    spec.update(over)
    return spec


def flag(argv: list[str], name: str) -> str | None:
    """The value of `--name=value`, or None. Whole-flag match, so looking for
    `--eval_steps` never answers with `--steps`."""
    for item in argv:
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FakeTracker:
    """Stands in for `lerobot.utils.logging_utils.MetricsTracker`.

    `to_dict` is the real one's shape; `__str__` reproduces what
    `format_big_number` does to the step count, which is the whole reason the
    handler must not read it.
    """

    def __init__(self, steps: int, loss: float = 0.0431) -> None:
        self.steps = steps
        self.loss = loss

    def to_dict(self) -> dict:
        return {"steps": self.steps, "samples": self.steps * 8,
                "epochs": 1.5, "loss": self.loss}

    def __str__(self) -> str:
        return f"step:{self.steps / 1000:.0f}K smpl:{self.steps * 8 / 1000:.0f}K"


class ExplodingTracker(FakeTracker):
    def to_dict(self) -> dict:
        raise RuntimeError("the metric this run reports has moved")


class Tap:
    """A handler wired to a real logger, so records arrive the way LeRobot
    sends them (`logging.info(<object>)`) rather than hand-built."""

    def __init__(self, tmp_path: Path, tracker_cls: type | tuple = FakeTracker) -> None:
        self.path = tmp_path / "metrics.jsonl"
        self.handler = train_runner.JsonlMetricsHandler(self.path, tracker_cls=tracker_cls)
        self.logger = logging.getLogger(f"tap-{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.addHandler(self.handler)

    def log(self, msg) -> None:
        self.logger.info(msg)

    def rows(self) -> list[dict]:
        return rows(self.path)


@pytest.fixture
def tap(tmp_path):
    t = Tap(tmp_path)
    yield t
    t.handler.close()


@pytest.fixture
def root_logger():
    """Snapshot and restore the ROOT logger. The re-attach machinery mutates it
    by design, and a test that leaked a closed file handler onto the root logger
    would break every later test in the session with a ValueError on a closed
    file."""
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    yield root
    root.handlers[:] = before
    root.setLevel(level)


def result_of(run_dir: Path) -> dict:
    return json.loads((run_dir / "result.json").read_text())


# ---- build_argv -----------------------------------------------------------

def test_a_minimal_spec_builds_the_kits_argv(tmp_path):
    """Ported flag for flag. The kit's defaults are what trained the policy that
    ran, so they are evidence rather than taste."""
    spec = make_spec(tmp_path)

    argv = train_runner.build_argv(spec)

    assert argv == [
        "lerobot-train",
        "--dataset.repo_id=local/so101_pick_cube",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--output_dir={Path(spec['run_dir']) / 'train'}",
        "--job_name=train-20260827-120000",
        "--steps=100000",
        "--batch_size=8",
        "--log_freq=200",
        "--save_freq=20000",
        "--num_workers=4",
        "--wandb.enable=false",
        "--env_eval_freq=0",
    ]


def test_every_knob_the_ui_offers_reaches_the_trainer(tmp_path):
    spec = make_spec(
        tmp_path,
        episodes=[4, 1, 9],
        eval_split=0.2,
        eval_steps=500,
        max_eval_samples=200,
        policy_type="smolvla",
        device="cpu",
        job_name="cube-v2",
        steps=20_000,
        batch_size=16,
        log_freq=50,
        save_freq=5_000,
        num_workers=2,
        extra_args=["--policy.n_action_steps=50"],
    )

    argv = train_runner.build_argv(spec)

    assert flag(argv, "--dataset.episodes") == "[4,1,9]"
    assert flag(argv, "--dataset.eval_split") == "0.2"
    assert flag(argv, "--eval_steps") == "500"
    assert flag(argv, "--max_eval_samples") == "200"
    assert flag(argv, "--policy.type") == "smolvla"
    assert flag(argv, "--policy.device") == "cpu"
    assert flag(argv, "--job_name") == "cube-v2"
    assert flag(argv, "--steps") == "20000"
    assert flag(argv, "--batch_size") == "16"
    assert flag(argv, "--log_freq") == "50"
    assert flag(argv, "--save_freq") == "5000"
    assert flag(argv, "--num_workers") == "2"
    assert argv[-1] == "--policy.n_action_steps=50"


def test_the_kept_set_is_compact_json_of_ints(tmp_path):
    """Compact because it is one shell argument: a space after each comma would
    make LeRobot's parser see four arguments where there is one."""
    spec = make_spec(tmp_path, episodes=[0, 3, "7", 12])

    value = flag(train_runner.build_argv(spec), "--dataset.episodes")

    assert value == "[0,3,7,12]"
    assert " " not in value
    assert json.loads(value) == [0, 3, 7, 12]
    assert all(isinstance(e, int) for e in json.loads(value))


def test_the_kept_order_is_passed_through_unsorted(tmp_path):
    """**The single most breakable line in this file.** LeRobot groups this list
    by task and holds out the TAIL of each group without ever sorting it, so the
    order `lab/split.py` chose IS the eval split. Sorting it here would still
    produce a split of the right size — just the wrong one, silently."""
    spec = make_spec(tmp_path, episodes=[9, 2, 7, 0, 4])

    assert flag(train_runner.build_argv(spec), "--dataset.episodes") == "[9,2,7,0,4]"


def test_an_empty_kept_set_is_still_passed(tmp_path):
    """`[]` and "not specified" mean opposite things to the trainer — every
    episode versus none of them — so `episodes: []` must not be dropped as
    falsy. Refusing it is the route's job; the runner's job is not to lie."""
    spec = make_spec(tmp_path, episodes=[])

    assert flag(train_runner.build_argv(spec), "--dataset.episodes") == "[]"


def test_env_eval_freq_is_always_zero(tmp_path):
    """No simulator on this box, and LeRobot's 20k default would stall the run
    at step 20,000 waiting for an env that never arrives."""
    minimal = train_runner.build_argv(make_spec(tmp_path))
    loaded = train_runner.build_argv(
        make_spec(tmp_path, eval_split=0.2, eval_steps=100, extra_args=["--seed=7"]))

    assert "--env_eval_freq=0" in minimal
    assert "--env_eval_freq=0" in loaded


def test_eval_flags_need_an_eval_split_to_hold_out_anything(tmp_path):
    """`--eval_steps` without a held-out set is an eval pass over nothing."""
    spec = make_spec(tmp_path, eval_steps=500, max_eval_samples=200)

    argv = train_runner.build_argv(spec)

    assert flag(argv, "--eval_steps") is None
    assert flag(argv, "--max_eval_samples") is None
    assert flag(argv, "--dataset.eval_split") is None


def test_the_eval_sample_cap_is_optional(tmp_path):
    """Uncapped, every eval pass walks the whole held-out set and decodes video
    for each frame — longer than the training steps between passes."""
    spec = make_spec(tmp_path, eval_split=0.2, eval_steps=500)

    argv = train_runner.build_argv(spec)

    assert flag(argv, "--eval_steps") == "500"
    assert flag(argv, "--max_eval_samples") is None


def test_a_pinned_observation_space_reaches_lerobot_as_json(tmp_path):
    """`--policy.input_features` is the only way to stop LeRobot deriving the
    observation space from the dataset — and taking every `observation.*`
    column with it. Compact JSON because draccus parses the value as one
    argument and a space in it would split the flag."""
    spec = make_spec(tmp_path, policy_input_features={
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
    })

    value = flag(train_runner.build_argv(spec), "--policy.input_features")

    assert value is not None
    assert " " not in value
    assert json.loads(value) == {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
    }


def test_no_pin_leaves_lerobot_deriving_the_space_itself(tmp_path):
    """Absent is not the same as empty. `policies/factory.py` fills the space
    in `if not cfg.input_features`, so emitting nothing is what keeps a spec
    written before this field behaving exactly as it did."""
    argv = train_runner.build_argv(make_spec(tmp_path))

    assert flag(argv, "--policy.input_features") is None


def test_extra_args_pass_through_verbatim_and_last(tmp_path):
    """Verbatim so anything LeRobot supports stays reachable; last so an
    operator can override a default this function emitted."""
    spec = make_spec(tmp_path, extra_args=["--policy.chunk_size=100", "--seed=1000"])

    argv = train_runner.build_argv(spec)

    assert argv[-2:] == ["--policy.chunk_size=100", "--seed=1000"]


def test_the_output_dir_is_the_runs_own_train_subdirectory(tmp_path):
    """`lab/runs.checkpoints()` reads `<run_dir>/train/checkpoints/<step>/
    pretrained_model`, so this path and that one are one contract."""
    spec = make_spec(tmp_path)

    assert flag(train_runner.build_argv(spec), "--output_dir") == \
        str(Path(spec["run_dir"]) / "train")


def test_the_job_name_falls_back_to_the_run_id(tmp_path):
    spec = make_spec(tmp_path, job_name="")

    assert flag(train_runner.build_argv(spec), "--job_name") == "train-20260827-120000"


# ---- --dry-run ------------------------------------------------------------

def test_dry_run_prints_the_argv_and_returns_zero(tmp_path, monkeypatch, capsys):
    spec = make_spec(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["train_runner", str(spec_path), "--dry-run"])

    code = train_runner.main()

    assert code == 0
    assert capsys.readouterr().out.strip() == " ".join(train_runner.build_argv(spec))
    assert not (Path(spec["run_dir"]) / "result.json").exists()


def test_a_dry_run_can_be_asked_for_in_the_spec(tmp_path, monkeypatch, capsys):
    """`lab/runs.launch` builds the child's argv itself and has no way to pass a
    flag, so the spec key is the only route a dry run has from the UI."""
    spec = make_spec(tmp_path, dry_run=True)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["train_runner", str(spec_path)])

    assert train_runner.main() == 0
    assert capsys.readouterr().out.startswith("lerobot-train ")


def test_dry_run_imports_neither_lerobot_nor_torch(tmp_path):
    """A subprocess, because pytest has already imported half the world into
    this one and `sys.modules` here would prove nothing.

    This is what makes `--dry-run` usable as a preflight from a box with no GPU
    free: it answers "what exactly would you run" without a CUDA context, a
    checkpoint or an hour."""
    spec = make_spec(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    probe = textwrap.dedent(f"""
        import sys
        sys.argv = ["train_runner", {str(spec_path)!r}, "--dry-run"]
        from haller_hmi.runners import train_runner
        code = train_runner.main()
        print("EXIT", code)
        print("HEAVY", "torch" in sys.modules, "lerobot" in sys.modules)
    """)

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    lines = out.stdout.strip().splitlines()
    assert lines[0] == " ".join(train_runner.build_argv(spec)), out.stderr
    assert lines[1] == "EXIT 0"
    assert lines[2] == "HEAVY False False"


def test_importing_the_runner_pulls_in_neither_lerobot_nor_torch():
    """The two-interpreter rule, from the runner's side. `lab/runs.py` names
    this module as a STRING and never imports it — but `_common` imports
    `lab.runs` back, and a module-scope `import torch` here would ride that edge
    into the serving process the moment anyone tidied the string away."""
    probe = ("import sys; from haller_hmi.runners import train_runner as m; "
             "print('torch' in sys.modules, 'lerobot' in sys.modules, bool(m))")

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    assert out.stdout.strip() == "False False True", out.stderr


def test_no_spec_path_is_a_usage_message_and_exit_two(monkeypatch, capsys):
    """Exit 2 and NO `result.json`, which is right rather than an omission:
    without a spec there is no run directory to write one into."""
    monkeypatch.setattr(sys, "argv", ["train_runner", "--dry-run"])

    with pytest.raises(SystemExit) as excinfo:
        train_runner.main()

    assert excinfo.value.code == 2
    assert capsys.readouterr().out.strip() == _common.USAGE


# ---- the metrics handler --------------------------------------------------

def test_the_tracker_object_is_recorded_through_to_dict(tap):
    tap.log(FakeTracker(steps=1200, loss=0.0431))

    assert tap.rows() == [{"kind": "train", "steps": 1200, "samples": 9600,
                           "epochs": 1.5, "loss": 0.0431}]


def test_the_step_recorded_is_11500_not_the_rounded_12K(tap):
    """**The bug this file exists to prevent.**

    LeRobot logs the `MetricsTracker` OBJECT, and its `__str__` rounds every
    count through `format_big_number` — verified under `~/venvs/haller-lab`:
    11500 -> "12K", and so does 12499. A handler that scraped stdout would plot
    the run's losses against an x-axis it invented, and the chart would look
    entirely plausible while being wrong by up to 999 steps.
    """
    tracker = FakeTracker(steps=11_500)
    assert "step:12K" in str(tracker)  # what the terminal shows

    tap.log(tracker)

    assert tap.rows()[0]["steps"] == 11_500
    text = tap.path.read_text()
    assert "11500" in text
    assert "12000" not in text
    assert "12K" not in text


def test_the_eval_line_is_parsed_out_of_the_text(tap):
    """`lerobot_train.py:673` — held-out loss is the one metric reported as a
    formatted string rather than through the tracker."""
    tap.log("step 1200: eval_loss=0.0431")

    assert tap.rows() == [{"kind": "eval", "steps": 1200, "eval_loss": 0.0431}]


def test_the_eval_regex_searches_rather_than_anchoring(tap):
    """`.search`, so a line the trainer decorates still parses; and the loss goes
    through `float`, so scientific notation late in a run is a point on the chart
    rather than a dropped row."""
    tap.log("rank0: step 40000: eval_loss=1.2e-05")

    assert tap.rows() == [{"kind": "eval", "steps": 40_000, "eval_loss": 1.2e-05}]


def test_the_split_line_is_parsed_out_of_the_text(tap):
    """`lerobot/datasets/factory.py:182`, verbatim. This is the only place a run
    states how `plan_eval_split`'s shuffled order actually landed."""
    tap.log("Train/eval split: 4 train, 1 eval (eval_split=0.2, 1 tasks)")

    assert tap.rows() == [{"kind": "split", "train_episodes": 4, "eval_episodes": 1}]


def test_ordinary_log_lines_write_nothing(tap):
    for line in ("Output dir: outputs/runs/train-1/train",
                 "Logs will be saved locally.",
                 "Resume training from step 200"):
        tap.log(line)

    assert not tap.path.read_text()


def test_emit_swallows_a_tracker_whose_to_dict_raises(tmp_path):
    """A logging handler must never take the training run down with it: hours of
    GPU time are not worth a chart feed's exception. The run keeps going and the
    NEXT record is still recorded."""
    tap = Tap(tmp_path, tracker_cls=FakeTracker)
    try:
        tap.log(ExplodingTracker(steps=800))
        tap.log(FakeTracker(steps=1000))
    finally:
        tap.handler.close()

    assert [r["steps"] for r in tap.rows()] == [1000]


def test_the_tracker_class_is_guarded_when_lerobot_has_moved(tmp_path, monkeypatch):
    """`()` rather than a raise: a lerobot layout change must cost the chart, not
    the training run. `isinstance(x, ())` is a constant False, so `emit` needs no
    special case for it."""
    monkeypatch.setitem(sys.modules, "lerobot", None)

    handler = train_runner.JsonlMetricsHandler(tmp_path / "metrics.jsonl")
    try:
        assert handler._tracker_cls == ()
        handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1,
                                       FakeTracker(steps=5), None, None))
        handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1,
                                       "step 5: eval_loss=0.5", None, None))
    finally:
        handler.close()

    assert rows(tmp_path / "metrics.jsonl") == [
        {"kind": "eval", "steps": 5, "eval_loss": 0.5}]


def test_the_file_is_appended_to_not_truncated(tmp_path):
    """A run resumed after a restart keeps the metrics it already reported —
    `read_metrics` hands the page a byte offset, and a truncation would make
    every stored offset point into the middle of a different row."""
    first = train_runner.JsonlMetricsHandler(tmp_path / "metrics.jsonl",
                                             tracker_cls=FakeTracker)
    first.emit(logging.LogRecord("t", logging.INFO, __file__, 1,
                                 FakeTracker(steps=200), None, None))
    first.close()

    second = train_runner.JsonlMetricsHandler(tmp_path / "metrics.jsonl",
                                              tracker_cls=FakeTracker)
    second.emit(logging.LogRecord("t", logging.INFO, __file__, 1,
                                  FakeTracker(steps=400), None, None))
    second.close()

    assert [r["steps"] for r in rows(tmp_path / "metrics.jsonl")] == [200, 400]


# ---- surviving init_logging ----------------------------------------------

def test_the_handler_is_reattached_after_the_root_logger_is_cleared(tap, root_logger):
    """`lerobot.utils.utils.init_logging()` does `logger.handlers.clear()` on the
    ROOT logger and installs its own, and `train()` calls it 264 lines in — long
    after the handler was attached. A dropped handler does not raise: it shows up
    as a BLANK CHART AN HOUR INTO TRAINING."""
    def init_logging(**kwargs):
        root_logger.handlers.clear()   # exactly what lerobot's does
        root_logger.addHandler(logging.NullHandler())
        return "initialised"

    patched = train_runner._reattacher(tap.handler, init_logging)
    root_logger.addHandler(tap.handler)

    assert patched(console_level="INFO") == "initialised"
    assert tap.handler in root_logger.handlers


def test_reattaching_twice_does_not_stack_the_handler(tap, root_logger):
    """Two copies on the root logger would write every metric row twice, and a
    duplicated x value is a chart that reads as a stall."""
    patched = train_runner._reattacher(tap.handler, lambda: None)

    patched()
    patched()

    assert root_logger.handlers.count(tap.handler) == 1


def test_the_reattacher_returns_what_init_logging_returned(tap, root_logger):
    """It wraps `init_logging`, so it has to be substitutable for it."""
    patched = train_runner._reattacher(tap.handler, lambda *a, **k: (a, k))

    assert patched(1, display_pid=True) == ((1,), {"display_pid": True})


def test_install_is_a_no_op_when_lerobot_is_absent(tmp_path, monkeypatch, root_logger):
    """The serving venv reaching this function would be a bug, but a hard
    ImportError inside a `finally`-guarded run is a worse one."""
    monkeypatch.setitem(sys.modules, "lerobot", None)
    handler = train_runner.JsonlMetricsHandler(tmp_path / "metrics.jsonl",
                                               tracker_cls=())
    before = list(root_logger.handlers)
    try:
        train_runner._install_handler_after_init_logging(handler)
    finally:
        handler.close()

    assert root_logger.handlers == before


# ---- run_guarded: the five endings ---------------------------------------

def test_a_clean_return_is_done(tmp_path):
    assert _common.run_guarded(tmp_path, lambda: None) == 0

    written = result_of(tmp_path)
    assert written["status"] == "done"
    assert written["exit_code"] == 0
    assert written["error"] == ""


def test_the_functions_return_value_is_ignored(tmp_path):
    """Returning at all is `done, 0`. A runner that wants a non-zero exit raises
    `SystemExit`, so one place decides the exit code rather than every caller's
    return statement being a second one."""
    assert _common.run_guarded(tmp_path, lambda: 7) == 0
    assert result_of(tmp_path)["status"] == "done"


def test_an_interrupt_is_stopped_not_failed(tmp_path, capsys):
    """SIGINT is how `runs.stop()` asks for a wind-down — LeRobot catches it and
    saves a checkpoint. Reporting that as `failed` would tell the operator their
    deliberate stop broke something."""
    def fn():
        raise KeyboardInterrupt

    assert _common.run_guarded(tmp_path, fn) == 130
    assert result_of(tmp_path)["status"] == "stopped"
    assert result_of(tmp_path)["error"] == "interrupted"
    assert "interrupted" in capsys.readouterr().out


def test_a_zero_system_exit_is_done(tmp_path):
    def fn():
        raise SystemExit(0)

    assert _common.run_guarded(tmp_path, fn) == 0
    assert result_of(tmp_path)["status"] == "done"
    assert result_of(tmp_path)["error"] == ""


def test_a_nonzero_system_exit_is_failed_with_its_code(tmp_path):
    def fn():
        raise SystemExit(3)

    assert _common.run_guarded(tmp_path, fn) == 3
    assert result_of(tmp_path)["status"] == "failed"
    assert result_of(tmp_path)["exit_code"] == 3
    assert result_of(tmp_path)["error"] == "exited with 3"


def test_a_string_system_exit_is_printed_and_its_first_line_is_the_error(tmp_path, capsys):
    """A preflight refusal. The kit raises these with the whole explanation
    attached — `runs.stop it first, two processes on one Feetech bus corrupt each
    other's packets`. All of it belongs in run.log; `error` is one table cell."""
    message = ("/dev/ttyACM0 is already open by:\n  pid 4211 python\n"
               "Stop it first — two processes on one Feetech bus corrupt each other.")

    def fn():
        raise SystemExit(message)

    assert _common.run_guarded(tmp_path, fn) == 1
    assert result_of(tmp_path)["status"] == "failed"
    assert result_of(tmp_path)["error"] == "/dev/ttyACM0 is already open by:"
    assert message in capsys.readouterr().out


def test_a_bare_system_exit_is_done(tmp_path):
    """`SystemExit()` carries `code = None`, which the interpreter itself treats
    as exit 0."""
    def fn():
        raise SystemExit

    assert _common.run_guarded(tmp_path, fn) == 0
    assert result_of(tmp_path)["status"] == "done"


def test_an_exception_is_failed_with_its_type_and_message(tmp_path, capsys):
    def fn():
        raise ValueError("no checkpoint at that step")

    assert _common.run_guarded(tmp_path, fn) == 1
    assert result_of(tmp_path)["status"] == "failed"
    assert result_of(tmp_path)["exit_code"] == 1
    assert result_of(tmp_path)["error"] == "ValueError: no checkpoint at that step"
    assert "Traceback" in capsys.readouterr().err


def test_result_json_is_written_even_when_fn_raises(tmp_path):
    """The whole reason `run_guarded` exists. `lab/runs.load()` reports a dead
    pid with NO `result.json` as `died` and never infers `done` from it — after a
    backend restart the server is not the child's parent and cannot reap it, so
    this file IS the exit status."""
    def fn():
        raise RuntimeError("CUDA out of memory")

    _common.run_guarded(tmp_path, fn)

    written = result_of(tmp_path)
    assert set(written) == {"status", "exit_code", "error", "finished_at"}
    assert written["finished_at"].endswith("+00:00")


# ---- load_spec ------------------------------------------------------------

def test_load_spec_reads_the_file_and_the_flag(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"repo_id": "local/x"}))

    assert _common.load_spec([str(spec_path)]) == ({"repo_id": "local/x"}, False)
    assert _common.load_spec([str(spec_path), "--dry-run"])[1] is True
    assert _common.load_spec(["--dry-run", str(spec_path)])[1] is True


def test_load_spec_honours_dry_run_in_the_spec(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"repo_id": "local/x", "dry_run": True}))

    spec, dry_run = _common.load_spec([str(spec_path)])

    assert dry_run is True
    assert spec["dry_run"] is True


def test_load_spec_without_a_path_exits_two(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _common.load_spec([])

    assert excinfo.value.code == 2
    assert capsys.readouterr().out.strip() == _common.USAGE
