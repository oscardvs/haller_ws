# hmi/backend/tests/lab/test_export_runner.py
"""`runners/export_runner.py` and `runners/eval_runner.py`, WITHOUT lerobot.

Every test here runs under the SERVING venv (lerobot 0.5.1). Neither runner can
actually run there — the real children run under `~/venvs/haller-lab` at lerobot
0.6.1 + torch 2.11.0+cu130 — and that is the point rather than a limitation:
both modules keep every heavy import inside `main()`'s call tree, so the plan
they build, every refusal they make and both `--dry-run` paths are reachable
from a process with no trainer at all.

**Nothing here exports or evaluates anything.** An export re-encodes video
(measured: `local/so101_pick_cube` is 46 episodes in 7 files, 707 MB) and one
of its modes deletes a dataset on a box with no backup of any kind; an
evaluation occupies the 4080 SUPER. What is tested is the preflight — which is
where the refusals have to live anyway, because a refusal that arrives after
the re-encode has already cost what it was meant to save.

Two things are pinned across a module boundary, both because a package the
serving process imports cannot be imported by the child:

* `export_runner._hf_home` / `_dataset_root` are `lab/catalog`'s rule spelled
  again — `catalog` imports `api/errors`, which imports `fastapi`, which is not
  installed in `~/venvs/haller-lab`. `test_the_home_it_resolves_is_the_catalogs`
  and `test_the_traversal_refusal_is_the_catalogs` hold the copies together,
  including on the SYMLINKED home this box actually has.
* `eval_runner.EPISODE_LOSS_FILENAME` and the row shape it writes are what
  `lab/autoclass.py`'s `policy-loss` mode reads.
  `test_the_rows_this_runner_writes_are_the_rows_autoclass_ranks` runs a file of
  those rows through `autoclass.preview` — that round trip IS the contract
  between the two files, and it is the reason this file tests a second runner.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from haller_hmi.lab import autoclass, catalog, review, runs
from haller_hmi.runners import eval_runner, export_runner

from . import _dataset

BACKEND = Path(__file__).resolve().parents[2]


# ---- fixtures -------------------------------------------------------------

def _forget() -> None:
    """Drop the catalog's caches; they key on file size and mtime."""
    catalog._detail_cache.clear()
    catalog._frames_cache.clear()


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    base = tmp_path / "lerobot"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(base))
    monkeypatch.setenv("HALLER_RUNS", str(tmp_path / "runs"))
    _forget()
    yield base
    _forget()


@pytest.fixture()
def source(home) -> Path:
    """Three episodes at `local/graded`, the fixture the autoclass tests use."""
    root = home / "local" / "graded"
    _dataset.make_dataset(root, n_episodes=3)
    return root


def run_dir(tmp_path: Path, run_id: str) -> Path:
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_spec(tmp_path, **over) -> dict:
    """The minimum `lab/runs.launch` puts on disk, plus whatever a test needs."""
    spec = {
        "run_id": "export-20260827-120000",
        "run_dir": str(run_dir(tmp_path, "export-20260827-120000")),
        "repo_id": "local/graded",
        "delete_episodes": [1],
    }
    spec.update(over)
    return spec


def eval_spec(tmp_path, **over) -> dict:
    checkpoint = tmp_path / "train" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text('{"type": "act"}')
    spec = {
        "run_id": "eval-20260827-120000",
        "run_dir": str(run_dir(tmp_path, "eval-20260827-120000")),
        "repo_id": "local/graded",
        "checkpoint": str(checkpoint),
    }
    spec.update(over)
    return spec


def write_spec(tmp_path: Path, spec: dict, name: str = "spec.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(spec))
    return path


def run_main(module, tmp_path, spec, monkeypatch, *, dry: bool = True):
    """Drive a runner's `main()` the way its child is driven."""
    path = write_spec(tmp_path, spec)
    argv = [module.__name__, str(path)] + (["--dry-run"] if dry else [])
    monkeypatch.setattr(sys, "argv", argv)
    return module.main()


def refuse(module, tmp_path, spec, monkeypatch) -> str:
    """The message a `--dry-run` refuses with."""
    with pytest.raises(SystemExit) as excinfo:
        run_main(module, tmp_path, spec, monkeypatch)
    return str(excinfo.value)


# ---- export: the copy plan ------------------------------------------------

def test_a_copy_dry_run_names_both_datasets_and_every_dropped_episode(
        source, tmp_path, monkeypatch, capsys):
    spec = export_spec(tmp_path, new_repo_id="local/graded_kept",
                       delete_episodes=[1, 2])

    assert run_main(export_runner, tmp_path, spec, monkeypatch) == 0

    out = capsys.readouterr().out
    assert "export local/graded -> local/graded_kept" in out
    # BOTH spellings of every index. Oscar numbers episodes 1-based in
    # conversation and the parquet stores them 0-based; this is the last line
    # printed before minutes of irreversible re-encoding.
    assert "Ep 2 (idx 1)" in out
    assert "Ep 3 (idx 2)" in out
    assert "is not touched" in out


def test_a_copy_dry_run_carries_the_two_operator_facing_facts(
        source, tmp_path, monkeypatch, capsys):
    """Renumbering and the missing marks are one fact with two halves: index 3
    still exists afterwards, it is just a different demonstration."""
    spec = export_spec(tmp_path, new_repo_id="local/graded_kept")

    run_main(export_runner, tmp_path, spec, monkeypatch)

    out = capsys.readouterr().out
    assert "RENUMBERED 0..n-1" in out
    assert "NO review marks" in out


def test_the_dry_run_prints_exactly_what_the_real_run_would(
        source, tmp_path, monkeypatch, capsys):
    """One `describe()` behind both paths — a preflight that describes something
    other than what runs is worse than none."""
    spec = export_spec(tmp_path, new_repo_id="local/graded_kept")

    run_main(export_runner, tmp_path, spec, monkeypatch)

    expected = export_runner.describe(export_runner.build_plan(spec))
    assert capsys.readouterr().out.strip() == "\n".join(expected)


def test_a_dry_run_writes_no_result_json(source, tmp_path, monkeypatch):
    """No run happened, so there is no exit status to record. `lab/runs.load`
    reads a run.json-less directory as nothing at all, which is correct here."""
    spec = export_spec(tmp_path, new_repo_id="local/graded_kept")

    run_main(export_runner, tmp_path, spec, monkeypatch)

    assert not (Path(spec["run_dir"]) / "result.json").exists()


def test_a_dry_run_can_be_asked_for_in_the_spec(source, tmp_path, monkeypatch, capsys):
    """`lab/runs.launch` builds the child's argv itself and has no way to pass a
    flag, so the spec key is the only route a dry run has from the UI."""
    spec = export_spec(tmp_path, new_repo_id="local/graded_kept", dry_run=True)

    assert run_main(export_runner, tmp_path, spec, monkeypatch, dry=False) == 0
    assert "export local/graded -> local/graded_kept" in capsys.readouterr().out


# ---- export: the in-place plan --------------------------------------------

def test_an_in_place_dry_run_says_permanently_and_names_the_backup(
        source, tmp_path, monkeypatch, capsys):
    spec = export_spec(tmp_path, mode="in_place", delete_episodes=[0])

    assert run_main(export_runner, tmp_path, spec, monkeypatch) == 0

    out = capsys.readouterr().out
    assert "PERMANENTLY deleting 1 episode(s) from local/graded" in out
    assert "Ep 1 (idx 0)" in out
    assert "moved to graded_old first" in out


def test_an_in_place_dry_run_says_the_marks_are_cleared(
        source, tmp_path, monkeypatch, capsys):
    """A prune that kept the marks would attach decisions made about episode 4
    to whichever demonstration lands at index 3 afterwards."""
    spec = export_spec(tmp_path, mode="in_place")

    run_main(export_runner, tmp_path, spec, monkeypatch)

    out = capsys.readouterr().out
    assert "review marks are CLEARED" in out
    assert "RENUMBERED 0..n-1" in out


def test_dropping_the_backup_is_called_unrecoverable(
        source, tmp_path, monkeypatch, capsys):
    """`keep_backup: false` is the only genuinely unrecoverable path in this
    runner, and the log has to say so before it takes it."""
    spec = export_spec(tmp_path, mode="in_place", keep_backup=False)

    run_main(export_runner, tmp_path, spec, monkeypatch)

    out = capsys.readouterr().out
    assert "graded_old will be DELETED" in out
    assert "NOT recoverable" in out
    assert "no backup of any kind" in out


def test_every_plan_says_the_video_is_re_encoded(source, tmp_path, monkeypatch, capsys):
    """The reason this is a job and not a request. A v3.0 dataset packs many
    episodes into one mp4, so a hole in the middle is re-encoded around."""
    for spec in (export_spec(tmp_path, new_repo_id="local/graded_kept"),
                 export_spec(tmp_path, mode="in_place")):
        run_main(export_runner, tmp_path, spec, monkeypatch)
        assert "re-encodes video" in capsys.readouterr().out


# ---- export: every refusal ------------------------------------------------

def test_an_export_that_drops_nothing_is_refused(source, tmp_path, monkeypatch):
    """Not a no-op: it would re-encode every video file to produce a dataset
    identical to its source."""
    spec = export_spec(tmp_path, new_repo_id="local/graded_kept",
                       delete_episodes=[])

    assert "nothing to drop" in refuse(export_runner, tmp_path, spec, monkeypatch)


def test_a_missing_delete_episodes_key_is_the_same_refusal(
        source, tmp_path, monkeypatch):
    spec = export_spec(tmp_path, new_repo_id="local/graded_kept")
    spec.pop("delete_episodes")

    assert "nothing to drop" in refuse(export_runner, tmp_path, spec, monkeypatch)


def test_copy_without_a_new_repo_id_is_refused(source, tmp_path, monkeypatch):
    spec = export_spec(tmp_path)

    assert "needs a new_repo_id" in refuse(export_runner, tmp_path, spec, monkeypatch)


def test_copying_onto_the_source_is_refused(source, tmp_path, monkeypatch):
    spec = export_spec(tmp_path, new_repo_id="local/graded")

    message = refuse(export_runner, tmp_path, spec, monkeypatch)
    assert "onto itself" in message
    assert "mode=in_place" in message


def test_a_second_spelling_of_the_source_is_refused_too(source, tmp_path, monkeypatch):
    """`local//graded` is not `local/graded` as a string and IS as a path. The
    check compares resolved paths for exactly this reason: a string compare
    would let a self-overwrite through under a different spelling."""
    spec = export_spec(tmp_path, new_repo_id="local//graded")

    assert "onto itself" in refuse(export_runner, tmp_path, spec, monkeypatch)


def test_an_output_that_already_exists_is_refused(source, home, tmp_path, monkeypatch):
    (home / "local" / "graded_kept").mkdir(parents=True)
    spec = export_spec(tmp_path, new_repo_id="local/graded_kept")

    assert "already exists" in refuse(export_runner, tmp_path, spec, monkeypatch)


def test_an_unknown_mode_is_refused_by_name(source, tmp_path, monkeypatch):
    spec = export_spec(tmp_path, mode="in-place")

    message = refuse(export_runner, tmp_path, spec, monkeypatch)
    assert "unknown export mode 'in-place'" in message
    assert "copy, in_place" in message


def test_a_missing_repo_id_is_refused(tmp_path, home, monkeypatch):
    spec = export_spec(tmp_path, repo_id="", new_repo_id="local/graded_kept")

    assert "no repo_id" in refuse(export_runner, tmp_path, spec, monkeypatch)


def test_a_source_that_is_not_on_disk_is_refused_here(home, tmp_path, monkeypatch):
    """Named locally rather than left to LeRobot: with no local root,
    `LeRobotDataset` goes to the Hub looking for a `local/...` repo that was
    only ever a typo, and the run fails as a network error minutes later."""
    spec = export_spec(tmp_path, repo_id="local/typo",
                       new_repo_id="local/graded_kept")

    assert "no dataset at" in refuse(export_runner, tmp_path, spec, monkeypatch)


def test_a_repo_id_that_climbs_out_of_the_cache_is_refused(home, tmp_path, monkeypatch):
    """These ids reach this process from an HTTP body. Without the containment
    check, `mode=in_place` would rmtree a directory of the caller's choosing."""
    spec = export_spec(tmp_path, repo_id="../../etc", mode="in_place")

    assert "escapes the dataset cache" in refuse(
        export_runner, tmp_path, spec, monkeypatch)


def test_a_new_repo_id_that_climbs_out_of_the_cache_is_refused(
        source, tmp_path, monkeypatch):
    spec = export_spec(tmp_path, new_repo_id="../../etc/graded")

    message = refuse(export_runner, tmp_path, spec, monkeypatch)
    assert "new_repo_id" in message
    assert "escapes the dataset cache" in message


def test_delete_episodes_that_are_not_indices_are_refused(
        source, tmp_path, monkeypatch):
    spec = export_spec(tmp_path, new_repo_id="local/graded_kept",
                       delete_episodes=["two"])

    assert "list of episode indices" in refuse(
        export_runner, tmp_path, spec, monkeypatch)


def test_a_refused_export_writes_a_result_json_rather_than_reading_as_died(
        source, tmp_path, monkeypatch, capsys):
    """A refusal on the REAL path goes through `run_guarded`, so it is a
    `failed` run carrying its own sentence. Returning without a `result.json`
    would make `lab/runs.load` report the deliberate refusal as `died`, which
    is what it reports for an OOM kill."""
    spec = export_spec(tmp_path, delete_episodes=[])

    assert run_main(export_runner, tmp_path, spec, monkeypatch, dry=False) == 1

    result = json.loads((Path(spec["run_dir"]) / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["exit_code"] == 1
    assert "nothing to drop" in result["error"]
    assert "nothing to drop" in capsys.readouterr().out


def test_no_spec_path_is_a_usage_message_and_exit_two(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["export_runner", "--dry-run"])

    with pytest.raises(SystemExit) as excinfo:
        export_runner.main()

    assert excinfo.value.code == 2


# ---- export: the paths are the catalog's ----------------------------------

def test_the_home_it_resolves_is_the_catalogs(tmp_path, monkeypatch):
    """Through a SYMLINK, which is the case this box actually has:
    `~/.cache/huggingface/lerobot` is a symlink to `~/robot-data/lerobot`. An
    unresolved base has no common prefix with a resolved root, and the
    containment check below would then refuse every real repo-id."""
    real = tmp_path / "robot-data"
    real.mkdir()
    link = tmp_path / "cache-link"
    link.symlink_to(real)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(link))

    assert export_runner._hf_home() == catalog.hf_home()
    assert export_runner._hf_home() == real.resolve()


def test_the_dataset_root_it_resolves_is_the_catalogs(home):
    assert (export_runner._dataset_root("local/graded")
            == catalog.dataset_root("local/graded"))


def test_the_traversal_refusal_is_the_catalogs(home):
    """Both copies refuse, and refuse with the same sentence — the drift that
    matters here is one of them quietly starting to allow it."""
    with pytest.raises(ValueError) as ours:
        export_runner._dataset_root("../../etc")
    with pytest.raises(ValueError) as theirs:
        catalog.dataset_root("../../etc")

    assert str(ours.value) == str(theirs.value)


# ---- export: no heavy imports ---------------------------------------------

def test_a_dry_run_imports_neither_lerobot_nor_torch(source, home, tmp_path):
    """A subprocess, because pytest has already imported half the world into
    this one and `sys.modules` here would prove nothing.

    This is what makes `--dry-run` usable as a preflight: it answers "what
    exactly would you do to my dataset" without loading a trainer."""
    import os

    spec = export_spec(tmp_path, new_repo_id="local/graded_kept")
    spec_path = write_spec(tmp_path, spec)
    probe = textwrap.dedent(f"""
        import sys
        sys.argv = ["export_runner", {str(spec_path)!r}, "--dry-run"]
        from haller_hmi.runners import export_runner
        code = export_runner.main()
        print("EXIT", code)
        print("HEAVY", "torch" in sys.modules, "lerobot" in sys.modules)
    """)

    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        check=True, timeout=120, cwd=str(BACKEND),
        env={**os.environ, "HF_LEROBOT_HOME": str(home)})

    lines = out.stdout.strip().splitlines()
    assert lines[-2] == "EXIT 0", out.stderr
    assert lines[-1] == "HEAVY False False"


@pytest.mark.parametrize("module", ["export_runner", "eval_runner"])
def test_importing_a_runner_pulls_in_neither_lerobot_nor_torch(module):
    """The two-interpreter rule, from the runner's side. `lab/runs.py` names
    these modules as STRINGS and never imports them — but `_common` imports
    `lab.runs` back, and a module-scope `import torch` here would ride that edge
    into the serving process the moment anyone tidied the string away."""
    probe = (f"import sys; from haller_hmi.runners import {module} as m; "
             "print('torch' in sys.modules, 'lerobot' in sys.modules, bool(m))")

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    assert out.stdout.strip() == "False False True", out.stderr


def test_neither_runner_imports_a_package_the_lab_venv_lacks():
    """`~/venvs/haller-lab` has lerobot, torch, pandas, pyarrow and numpy — and
    NO fastapi (verified 2026-08-27). So `lab.catalog`, `lab.autoclass` and
    every `lab.routes_*` are unimportable in the interpreter that runs these
    children, and a runner that imported one would die before its first line.
    `lab.review` and `lab.runs` are stdlib-only and are fine."""
    banned = {"haller_hmi.lab.catalog", "haller_hmi.lab.autoclass",
              "haller_hmi.api.errors", "fastapi"}
    probe = ("import sys; from haller_hmi.runners import export_runner, eval_runner; "
             "print(sorted(m for m in sys.modules if m in "
             f"{sorted(banned)!r}))")

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    assert out.stdout.strip() == "[]", out.stderr


# ---- eval: the plan -------------------------------------------------------

def test_an_eval_dry_run_prints_the_plan_and_returns_zero(tmp_path, monkeypatch, capsys):
    spec = eval_spec(tmp_path, episodes=[0, 2], device="cpu", batch_size=4)

    assert run_main(eval_runner, tmp_path, spec, monkeypatch) == 0

    out = capsys.readouterr().out
    assert "pretrained_model" in out
    assert "dataset local/graded, 2 episode(s): Ep 1 (idx 0), Ep 3 (idx 2)" in out
    assert "device cpu, batch_size 4" in out
    assert eval_runner.EPISODE_LOSS_FILENAME in out


def test_an_eval_dry_run_says_every_episode_when_none_are_named(
        tmp_path, monkeypatch, capsys):
    """The total is not knowable without lerobot, so the plan says what it
    means rather than a count it would have to invent."""
    assert run_main(eval_runner, tmp_path, eval_spec(tmp_path), monkeypatch) == 0

    assert "every episode in local/graded" in capsys.readouterr().out


def test_the_eval_plan_says_this_is_a_sort_order_and_never_a_mark(
        tmp_path, monkeypatch, capsys):
    """The one sentence that keeps the number honest. A high loss is as often a
    rare-but-correct demonstration as a bad one, so nothing downstream — not
    this runner, not `autoclass.apply` — is allowed to turn it into a mark."""
    run_main(eval_runner, tmp_path, eval_spec(tmp_path), monkeypatch)

    out = capsys.readouterr().out
    assert "sort order" in out
    assert "never a mark" in out


def test_an_eval_dry_run_writes_neither_a_result_nor_a_loss_file(
        tmp_path, monkeypatch):
    spec = eval_spec(tmp_path)

    run_main(eval_runner, tmp_path, spec, monkeypatch)

    assert not (Path(spec["run_dir"]) / "result.json").exists()
    assert not (Path(spec["run_dir"]) / eval_runner.EPISODE_LOSS_FILENAME).exists()


def test_an_eval_dry_run_imports_neither_lerobot_nor_torch(tmp_path):
    spec_path = write_spec(tmp_path, eval_spec(tmp_path))
    probe = textwrap.dedent(f"""
        import sys
        sys.argv = ["eval_runner", {str(spec_path)!r}, "--dry-run"]
        from haller_hmi.runners import eval_runner
        code = eval_runner.main()
        print("EXIT", code)
        print("HEAVY", "torch" in sys.modules, "lerobot" in sys.modules)
    """)

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    lines = out.stdout.strip().splitlines()
    assert lines[-2] == "EXIT 0", out.stderr
    assert lines[-1] == "HEAVY False False"


# ---- eval: every refusal --------------------------------------------------

def test_an_eval_without_a_repo_id_is_refused(tmp_path, monkeypatch):
    spec = eval_spec(tmp_path, repo_id="")

    assert "no repo_id" in refuse(eval_runner, tmp_path, spec, monkeypatch)


def test_an_eval_without_a_checkpoint_is_refused(tmp_path, monkeypatch):
    spec = eval_spec(tmp_path, checkpoint="")

    message = refuse(eval_runner, tmp_path, spec, monkeypatch)
    assert "no checkpoint" in message
    assert "pretrained_model" in message


def test_a_checkpoint_that_is_not_on_disk_is_refused(tmp_path, monkeypatch):
    spec = eval_spec(tmp_path, checkpoint=str(tmp_path / "nope"))

    assert "no checkpoint directory at" in refuse(
        eval_runner, tmp_path, spec, monkeypatch)


def test_a_checkpoint_directory_one_level_too_high_is_refused(tmp_path, monkeypatch):
    """`<step>/` holds `pretrained_model/`; it is not itself one. Named here
    rather than left to LeRobot, which answers a missing local config.json by
    treating the path as a HUB repo id and 404ing about a repository nobody
    asked for."""
    spec = eval_spec(tmp_path)
    parent = Path(spec["checkpoint"]).parent
    spec["checkpoint"] = str(parent)

    message = refuse(eval_runner, tmp_path, spec, monkeypatch)
    assert "holds no config.json" in message
    assert "not the pretrained_model directory inside it" in message


def test_an_empty_episode_list_is_refused(tmp_path, monkeypatch):
    """`episodes: null` means every episode; `episodes: []` means the caller
    computed a set and it came out empty, which is a different thing and worth
    saying out loud."""
    spec = eval_spec(tmp_path, episodes=[])

    assert "nothing to score" in refuse(eval_runner, tmp_path, spec, monkeypatch)


def test_episodes_that_are_not_indices_are_refused(tmp_path, monkeypatch):
    spec = eval_spec(tmp_path, episodes=["one"])

    assert "list of episode indices" in refuse(
        eval_runner, tmp_path, spec, monkeypatch)


def test_a_batch_size_below_one_is_refused(tmp_path, monkeypatch):
    spec = eval_spec(tmp_path, batch_size=0)

    assert "batch_size must be at least 1" in refuse(
        eval_runner, tmp_path, spec, monkeypatch)


def test_a_refused_eval_writes_a_result_json(tmp_path, monkeypatch):
    spec = eval_spec(tmp_path, checkpoint="")

    assert run_main(eval_runner, tmp_path, spec, monkeypatch, dry=False) == 1

    result = json.loads((Path(spec["run_dir"]) / "result.json").read_text())
    assert result["status"] == "failed"
    assert "no checkpoint" in result["error"]


# ---- eval -> autoclass: the round trip ------------------------------------

def test_the_loss_filename_is_the_one_autoclass_looks_for(tmp_path):
    """Two spellings of one filename because the child cannot import the module
    that owns it — `autoclass` reaches `fastapi` through `catalog`, and the lab
    venv has no fastapi. This assertion is the join."""
    assert eval_runner.EPISODE_LOSS_FILENAME == autoclass.EPISODE_LOSS_FILENAME


def test_the_rows_this_runner_writes_are_the_rows_autoclass_ranks(source, tmp_path):
    """THE contract between `eval_runner` and `lab/autoclass.py`.

    The runner writes `{"episode": i, "loss": x, "frames": n}` per line and the
    `policy-loss` mode reads that file and ranks it. Neither side can import the
    other, so nothing but this test holds the two ends of a JSON row together —
    and the failure it prevents is silent: a key renamed on one side produces an
    empty ranking and `available: false`, which reads exactly like a run that
    was never evaluated.
    """
    rdir = runs.run_dir("eval-20260827-120000")
    rdir.mkdir(parents=True, exist_ok=True)
    rows = [{"episode": 0, "loss": 0.11, "frames": 300},
            {"episode": 1, "loss": 0.93, "frames": 250},
            {"episode": 2, "loss": 0.42, "frames": 310}]
    (rdir / eval_runner.EPISODE_LOSS_FILENAME).write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    out = autoclass.preview("local/graded", "policy-loss",
                            {"run_id": "eval-20260827-120000"})

    assert out["available"] is True
    # Hardest to fit first.
    assert out["ranking"] == [
        {"episode": 1, "score": 0.93, "rank": 1},
        {"episode": 2, "score": 0.42, "rank": 2},
        {"episode": 0, "score": 0.11, "rank": 3},
    ]
    # ALWAYS empty, in every mode of this ranking's existence.
    assert out["diff"] == []


def test_a_loss_file_cut_off_half_way_still_ranks_what_it_measured(source, tmp_path):
    """The runner writes a row per episode as it lands, line buffered, so a
    stopped run leaves a partial file. It has to rank the part it measured —
    that is the whole reason the rows are appended rather than dumped at the
    end."""
    rdir = runs.run_dir("eval-20260827-120000")
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / eval_runner.EPISODE_LOSS_FILENAME).write_text(
        '{"episode": 0, "loss": 0.11, "frames": 300}\n'
        '{"episode": 1, "loss": 0.93, "fra')

    out = autoclass.preview("local/graded", "policy-loss",
                            {"run_id": "eval-20260827-120000"})

    assert out["available"] is True
    assert [r["episode"] for r in out["ranking"]] == [0]


def test_this_runner_writes_no_marks(source, tmp_path, monkeypatch):
    """A loss ranking is not a review. Nothing in `eval_runner` imports
    `lab.review`, and a dry run over a marked dataset leaves the sidecar
    byte-identical."""
    review.set_status(source, 1, "reject", "wobbly")
    before = (source / "review.json").read_bytes()

    run_main(eval_runner, tmp_path, eval_spec(tmp_path), monkeypatch)

    assert (source / "review.json").read_bytes() == before
    assert "review" not in sys.modules.get(
        "haller_hmi.runners.eval_runner").__dict__
