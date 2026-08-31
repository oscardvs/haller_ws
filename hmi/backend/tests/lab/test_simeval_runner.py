"""`runners/simeval_runner.py` as a SPEC CONTRACT, not as a policy run.

Nothing here loads a checkpoint, imports lerobot or builds a MuJoCo world. What
is under test is the half that decides whether a run should happen at all:
`build_plan`'s refusals, the seed list, the summary arithmetic, and the two
import rules that make this runner launchable from a venv the serving process
cannot import.

The episode loop itself is `tests/sim/test_episode.py`, against the real bench.
The one thing neither file can cover is `PolicyDriver.act`, which needs lerobot
0.6.1, a checkpoint and a GPU: it is the same untested surface
`rollout_runner._rollout` and `train_runner._train` have, for the same reason,
and it is stated in the report rather than papered over with a mock that would
only prove the mock.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from haller_hmi.runners import simeval_runner as sr

BACKEND = Path(__file__).resolve().parents[2]

#: The rig an evaluation actually runs against. Used rather than a fixture YAML
#: because a test rig that drifted from it would pass while the real one stopped
#: naming a sim arm.
RIG = BACKEND / "config.bimanual-sim.yaml"

#: The 12 columns Haller's recorder writes for this bench, in its own order.
BIMANUAL_NAMES = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper")
]

#: The kit's unprefixed solo layout, for the rig-mismatch refusal.
SOLO_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


# ---- fixtures ------------------------------------------------------------

@pytest.fixture
def checkpoint(tmp_path) -> Path:
    """A `pretrained_model` directory shaped the way LeRobot writes one.

    The `config.json` is what makes it one: without it,
    `PreTrainedConfig.from_pretrained` treats the path as a Hub repo id and
    fails with a 404 about a repository nobody asked for, which is why
    `build_plan` names the omission itself.
    """
    model = tmp_path / "train" / "checkpoints" / "010000" / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    return model


@pytest.fixture
def spec(tmp_path, checkpoint) -> dict:
    """A spec that must build a plan, so every refusal test can name its edit."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return {
        "run_id": "simeval-test",
        "run_dir": str(run_dir),
        "policy_path": str(checkpoint),
        "config": str(RIG),
        "action_names": list(BIMANUAL_NAMES),
        "episodes": 3,
    }


# ---- the two-interpreter rule -------------------------------------------

def test_importing_the_runner_pulls_in_neither_lerobot_nor_torch_nor_mujoco():
    """`_common` imports `lab.runs` back, so a module-scope heavy import here
    would ride that edge into the serving process, which owns the bus and the
    teleop latency path.

    mujoco is on the list as well as torch, which is one more than
    `rollout_runner` has to answer for: this runner is the only one that reaches
    a simulator, and `--dry-run` has to stay usable as a preflight from a box
    with no display.
    """
    probe = ("import sys; from haller_hmi.runners import simeval_runner as m; "
             "print('torch' in sys.modules, 'lerobot' in sys.modules, "
             "'mujoco' in sys.modules, bool(m))")

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    assert out.stdout.strip() == "False False False True", out.stderr


def test_dry_run_describes_the_run_and_imports_nothing_heavy(tmp_path, spec):
    """A subprocess, because pytest has already imported half the world into
    this one and `sys.modules` here would prove nothing.

    Both halves matter: no heavy import, so this is a preflight from a box with
    the GPU busy and no display; and no `result.json`, because nothing ran.
    """
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    probe = textwrap.dedent(f"""
        import sys
        sys.argv = ["simeval_runner", {str(spec_path)!r}, "--dry-run"]
        from haller_hmi.runners import simeval_runner
        code = simeval_runner.main()
        print("EXIT", code)
        print("HEAVY", "torch" in sys.modules, "lerobot" in sys.modules,
              "mujoco" in sys.modules)
    """)

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, timeout=120, cwd=str(BACKEND))

    lines = out.stdout.strip().splitlines()
    assert lines[-2] == "EXIT 0"
    assert lines[-1] == "HEAVY False False False"
    assert any("sim success rate" in line for line in lines), out.stdout
    assert any("seeds 0, 1, 2" in line for line in lines), out.stdout
    assert not (Path(spec["run_dir"]) / "result.json").exists()


def test_the_dry_run_can_be_asked_for_by_the_spec(tmp_path, spec, capsys):
    """`lab/runs.launch` builds the child's argv itself and cannot pass a flag,
    so `"dry_run": true` in the spec is the only route a dry run has from the
    UI. `_common.load_spec` owns that; this pins that THIS runner honours it."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({**spec, "dry_run": True}))
    argv = sys.argv
    sys.argv = ["simeval_runner", str(spec_path)]
    try:
        assert sr.main() == 0
    finally:
        sys.argv = argv

    assert "sim success rate" in capsys.readouterr().out
    assert not (Path(spec["run_dir"]) / "result.json").exists()


# ---- the seed list -------------------------------------------------------

def test_the_seed_list_defaults_to_the_boring_one():
    assert sr.seed_list({}) == list(range(sr.DEFAULT_EPISODES))
    assert sr.seed_list({"episodes": 3}) == [0, 1, 2]
    assert sr.seed_list({"episodes": 3, "seed_start": 100}) == [100, 101, 102]


def test_an_explicit_seed_list_wins_and_is_taken_verbatim():
    """Including its order and its repeats. Replaying exactly the layouts a
    previous run scored is the comparison this runner exists for, and a list
    quietly sorted or de-duplicated would not be that."""
    assert sr.seed_list({"seeds": [9, 2, 9], "episodes": 50}) == [9, 2, 9]


@pytest.mark.parametrize("bad, match", [
    ({"seeds": "0,1"}, "list of whole numbers"),
    ({"seeds": ["x"]}, "list of whole numbers"),
    ({"seeds": []}, "nothing to score"),
    ({"episodes": 0}, "at least 1"),
    ({"episodes": "many"}, "whole number"),
    ({"seed_start": "x"}, "whole number"),
])
def test_a_malformed_seed_request_is_refused(bad, match):
    with pytest.raises(SystemExit, match=match):
        sr.seed_list(bad)


# ---- build_plan ----------------------------------------------------------

def test_the_plan_carries_the_rig_the_seeds_and_where_the_output_goes(spec):
    plan = sr.build_plan(spec)

    assert plan["policy_path"] == spec["policy_path"]
    assert plan["config_path"] == str(RIG)
    assert plan["arms"] == ["left", "right"]
    assert plan["rig"] == "bimanual"
    assert plan["seeds"] == [0, 1, 2]
    assert plan["camera_keys"] == ["top", "left_wrist", "right_wrist"]
    # The rig's own telemetry rate, which is the rate its datasets are recorded
    # at. A policy run at a rate it was not trained at is a different dynamical
    # system, which is what `post_rollout` refuses on.
    assert plan["control_hz"] == 30.0
    assert plan["max_episode_s"] == sr.DEFAULT_MAX_EPISODE_S
    assert plan["max_speed_deg_s"] == 60.0
    assert plan["randomize"] is True
    assert plan["mirror"] is False
    assert plan["out_path"] == Path(spec["run_dir"]) / sr.SIM_EVAL_FILENAME
    assert plan["summary_path"] == (
        Path(spec["run_dir"]) / sr.SIM_EVAL_SUMMARY_FILENAME)


def test_the_output_filenames_are_the_documented_ones():
    """Nothing reads them yet, so this is the definition rather than a check
    against a reader. The day one appears it takes the names from here."""
    assert sr.SIM_EVAL_FILENAME == "sim_eval.jsonl"
    assert sr.SIM_EVAL_SUMMARY_FILENAME == "sim_eval_summary.json"


def test_an_explicit_null_step_cap_means_uncapped_and_not_the_default(spec):
    """A different experiment, not a missing field. `spec.get(...) or default`
    would turn it back into 60 deg/s silently."""
    plan = sr.build_plan({**spec, "max_speed_deg_s": None})

    assert plan["max_speed_deg_s"] is None
    assert "uncapped" in " ".join(sr.describe(plan))


def test_a_zero_step_cap_is_refused_rather_than_read_as_off(spec):
    with pytest.raises(SystemExit, match="max_speed_deg_s must be greater"):
        sr.build_plan({**spec, "max_speed_deg_s": 0})


@pytest.mark.parametrize("edit, match", [
    ({"policy_path": ""}, "no policy_path"),
    ({"policy_path": "/nope/here"}, "no checkpoint directory"),
    ({"config": "/nope/rig.yaml"}, "no rig config"),
    ({"control_hz": 0}, "control_hz must be greater"),
    ({"control_hz": "fast"}, "control_hz must be a number"),
    ({"max_episode_s": -1}, "max_episode_s must be greater"),
])
def test_a_bad_spec_is_refused_on_the_spec(spec, edit, match):
    """Every refusal below is reachable from `--dry-run`, which is what makes it
    a refusal on the SPEC rather than three minutes into a CUDA context."""
    with pytest.raises(SystemExit, match=match):
        sr.build_plan({**spec, **edit})


def test_a_checkpoint_directory_without_a_config_is_named_as_such(spec, tmp_path):
    bare = tmp_path / "checkpoints" / "010000"
    bare.mkdir(parents=True)

    with pytest.raises(SystemExit, match="holds no config.json"):
        sr.build_plan({**spec, "policy_path": str(bare)})


def test_a_rig_with_no_sim_arm_is_refused_before_anything_is_built(spec, tmp_path):
    rig = tmp_path / "real.yaml"
    rig.write_text(
        "arms:\n"
        "  - id: right\n"
        "    model: so101_follower\n"
        "    port: /dev/ttyACM0\n"
        "    calibration_id: haller_right\n"
        "    source: real\n"
        "    enabled: true\n"
        "cameras: []\n"
    )

    with pytest.raises(SystemExit, match="no enabled `source: sim` arm"):
        sr.build_plan({**spec, "config": str(rig)})


def test_the_action_layout_is_required_and_never_guessed(spec):
    """Which element of the policy's vector is the left elbow is a property of
    the DATASET; which joint this bench's index 2 is, is a property of the RIG.
    Assuming they agree scores a policy whose wrist is driven by the shoulder's
    number, and that does not fail loudly - it scores zero."""
    without = {k: v for k, v in spec.items() if k != "action_names"}

    with pytest.raises(SystemExit, match="neither 'action_names' nor 'repo_id'"):
        sr.build_plan(without)


def test_a_solo_policy_pointed_at_a_bimanual_bench_is_refused(spec):
    """Not silently run with the right arm holding its start pose all episode:
    a rate measured that way is a rate for a task the policy was never asked to
    do."""
    with pytest.raises(SystemExit, match=r"drives \['right'\].*has \['left', 'right'\]"):
        sr.build_plan({**spec, "action_names": SOLO_NAMES, "side": "right"})


def test_the_plan_carries_what_the_driver_needs_to_resolve_the_rig_again(spec):
    """`build_plan` resolves the rig and throws the object away rather than
    putting a mutable `RigSpec` into the plan dict, which is
    `rollout_runner._rollout`'s choice. So the two keys it resolves FROM have to
    survive into the plan, or the driver cannot repeat the resolution."""
    plan = sr.build_plan(spec)

    assert plan["action_names"] == BIMANUAL_NAMES
    assert sr.plan_spec(plan)["action_names"] == BIMANUAL_NAMES

    from haller_hmi.runners.rollout_runner import resolve_rig
    assert resolve_rig(sr.plan_spec(plan)).rig == "bimanual"


def test_an_empty_action_names_is_a_missing_one_and_not_a_zero_column_rig(spec):
    """`resolve_rig` treats a falsy value as "not given" and falls through to
    `repo_id`, so `[]` has to reach the plan as `None`. Carried through as `[]`
    it would resolve to a rig with no columns: one that drives nothing, refuses
    nothing, and scores every episode a failure."""
    stripped = {**spec, "action_names": []}

    with pytest.raises(SystemExit, match="neither 'action_names' nor 'repo_id'"):
        sr.build_plan(stripped)

    assert sr.build_plan(spec)["action_names"] == BIMANUAL_NAMES


# ---- describe ------------------------------------------------------------

def test_describe_states_the_limitation_it_would_be_easiest_to_omit(spec):
    """A sim success rate is evidence about a sim-trained policy. Saying so in
    the run log is the cheapest place it cannot be missed."""
    lines = " ".join(sr.describe(sr.build_plan(spec)))

    assert "sim/task.py" in lines
    assert "only success authority" in lines
    assert "will not transfer" in lines
    assert str(RIG) in lines
    assert sr.SIM_EVAL_FILENAME in lines
    assert sr.SIM_EVAL_SUMMARY_FILENAME in lines


def test_describe_abbreviates_a_long_seed_list_without_hiding_its_length(spec):
    lines = " ".join(sr.describe(sr.build_plan({**spec, "episodes": 40})))

    assert "40 episode(s)" in lines
    assert "(40 seeds)" in lines


# ---- the summary ---------------------------------------------------------

class _Record:
    """Only the fields `summarise` reads. The real one is
    `sim/episode.EpisodeRecord`, which this file must not need mujoco to build.
    """

    def __init__(self, seed: int, success: bool, steps: int = 10,
                 sim_s: float = 1.0, reason: str = "timeout") -> None:
        self.seed = seed
        self.success = success
        self.steps = steps
        self.sim_s = sim_s
        self.reason = "success" if success else reason


def test_the_summary_carries_the_rate_the_seeds_and_the_predicate(spec):
    plan = sr.build_plan(spec)
    records = [_Record(0, True), _Record(1, False), _Record(2, True)]
    provenance = {"predicate": {"predicate": "haller_hmi.sim.task.cube_placed"},
                  "thresholds": {"settle_s": 0.5}}

    summary = sr.summarise(plan, records, provenance, wall_s=6.0)

    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["successes"] == 2
    assert summary["n"] == 3
    assert summary["seeds"] == [0, 1, 2]
    assert summary["seeds_run"] == [0, 1, 2]
    assert summary["complete"] is True
    assert summary["reasons"] == {"success": 2, "timeout": 1}
    # The number and the thing that decided it, in one file.
    assert summary["provenance"] is provenance
    assert summary["control_ticks_per_s"] == pytest.approx(30 / 6.0)
    assert json.loads(json.dumps(summary))["success_rate"] == pytest.approx(2 / 3)


def test_a_stopped_run_reports_the_rate_over_what_it_actually_ran(spec):
    """And says so. Reporting 1 success out of 5 requested seeds when only 2
    were played is a different and false claim; reporting nothing throws away
    work that was done."""
    plan = sr.build_plan({**spec, "episodes": 5})
    records = [_Record(0, True), _Record(1, False)]

    summary = sr.summarise(plan, records, {}, wall_s=1.0)

    assert summary["n"] == 2
    assert summary["requested"] == 5
    assert summary["complete"] is False
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["seeds"] == [0, 1, 2, 3, 4]
    assert summary["seeds_run"] == [0, 1]


def test_a_run_that_scored_nothing_is_a_zero_and_not_a_crash(spec):
    summary = sr.summarise(sr.build_plan(spec), [], {}, wall_s=0.0)

    assert summary["n"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["control_ticks_per_s"] == 0.0
    assert summary["complete"] is False
