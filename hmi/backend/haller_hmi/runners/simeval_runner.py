# hmi/backend/haller_hmi/runners/simeval_runner.py
"""Detached child that scores a checkpoint by SUCCESS RATE in the MuJoCo sim.

Launched by `lab/runs.launch("simeval", spec)` as

    ~/venvs/haller-lab/bin/python -m haller_hmi.runners.simeval_runner SPEC.json

It writes two files into the run directory:

    sim_eval.jsonl          one row per episode
    sim_eval_summary.json   the rate, the seed list, and the predicate

A row is exactly:

    {"episode": 3, "seed": 3, "success": true, "steps": 214,
     "sim_s": 7.133, "reason": "success"}

## Why this exists next to `eval_runner`, which already "scores" a checkpoint

`runners/eval_runner.py` measures per-episode TRAINING LOSS, and its own
docstring is emphatic that this is not a quality score: a high loss is as often
a rare-but-correct demonstration as a bad one. It answers "how well does this
policy fit that demonstration", which ranks episodes and cannot rank policies.
Nothing in the Lab answered "does this policy do the task", because answering it
needs a simulator to run the policy IN, and `train_runner.py:201` hardcodes
`--env_eval_freq=0` with the comment "Environment-based policy evaluation needs
a simulator. There isn't one here."

There is one. `sim/` has been running the teleop bench for months, with seeded
scene reset (`sim/scene.py:327`) and contact-based success predicates
(`sim/task.py`) both tested. What was missing was a loop joining them to a
policy, which is now `sim/episode.py`. This runner is the thin part: load a
checkpoint, wrap it as an `EpisodeDriver`, run the seed list, write the rows.

## The predicate is recorded next to the number, always

`sim_eval_summary.json` carries `TaskMonitor.provenance()` AND the monitor's
threshold values AND the scene's `RandomSpec`. A success rate is a statement
about a predicate, and one whose predicate is not written down is not
reproducible: "60%" against `settle_s=0.5` and "60%" against `settle_s=0.1` are
different claims, and six months later nothing else on disk says which was run.
The seed list is recorded for the same reason one level out: the layouts ARE the
experiment, and re-running the list re-creates them exactly.

## What this number is, and what it is not

It is the fraction of seeded sim episodes in which `sim/task.py`'s predicate
fired. That is a real measurement of a policy against a task, which loss is not.

**It is not a statement about the real bench.** A policy trained on REAL `top`
frames will not transfer to MuJoCo renders, and this runner does nothing to
close that gap: see `sim/episode.py`'s module docstring, which states the
limitation rather than papering over it. Score a sim-trained checkpoint on sim
and a real-trained one on the bench.

**It is not a rollout.** Nothing here opens a serial port, touches the Feetech
bus or reaches `policy_ingest`. The arms this drives are `data.ctrl` indices in
a `mjModel` this process built and owns, which is why - unlike
`rollout_runner.py`, which is careful to own the policy and never the bus -
there is no bus to be careful about.

## Where the heavy imports are

`lerobot` and `torch` are imported inside `main()`'s call tree, and so is
`mujoco`: `build_plan` and `describe` reach `haller_hmi.config` and nothing
else, so `--dry-run` prints exactly what a real run would do from a box with no
GPU, no display and no MuJoCo. That is `eval_runner`'s rule, extended by one
package because this runner has one more heavy dependency than that one does.

The checkpoint is loaded by `rollout_runner._load_policy`, IMPORTED rather than
copied. Two spellings of "config -> policy class -> weights -> pre/post
processors" would eventually disagree about whether the returned action is
un-normalised, and the answer to that question is the difference between degrees
and a plausible-looking number that is not degrees.

    python -m haller_hmi.runners.simeval_runner SPEC.json
    python -m haller_hmi.runners.simeval_runner SPEC.json --dry-run
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from ..config import DEFAULT_CONFIG_PATH, load_config
from ._common import load_spec, run_guarded
from .rollout_runner import action_from_vector, resolve_rig

__all__ = [
    "CHECKPOINT_CONFIG",
    "DEFAULT_EPISODES",
    "DEFAULT_MAX_EPISODE_S",
    "SIM_EVAL_FILENAME",
    "SIM_EVAL_SUMMARY_FILENAME",
    "PolicyDriver",
    "build_plan",
    "describe",
    "main",
    "seed_list",
    "summarise",
]

#: One row per episode. Spelled here rather than imported from a reader,
#: because nothing reads it yet: `sim_eval.jsonl` is this runner's output and
#: the Lab UI's future input, and the day a reader appears it must take the name
#: from here rather than the other way round (see
#: `eval_runner.EPISODE_LOSS_FILENAME`, which carries the same note for the
#: opposite reason).
SIM_EVAL_FILENAME = "sim_eval.jsonl"

#: The rate, the seeds and the predicate that produced them.
SIM_EVAL_SUMMARY_FILENAME = "sim_eval_summary.json"

#: `huggingface_hub.constants.CONFIG_NAME`. Same check `eval_runner` makes and
#: for the same reason: `PreTrainedConfig.from_pretrained` answers a directory
#: with no local config by going to the HUB with the path as a repo name, and
#: fails with a 404 about a repository nobody asked for.
CHECKPOINT_CONFIG = "config.json"

#: Episodes when the spec names neither `seeds` nor `episodes`. Ten is enough
#: for a rate to mean anything at 10% resolution and cheap enough to run while
#: someone waits: measured 2026-08-31 on this box, 10 x 20 s at 30 Hz is about
#: 50 s of wall clock with a trivial driver, plus inference.
DEFAULT_EPISODES = 10

#: First seed of the generated list. Seeds are the experiment, so the default
#: is the boring one: 0, 1, 2, ... reproduces across boxes and across people.
DEFAULT_SEED_START = 0

#: Ceiling on one episode, in SIM seconds. 20 s at 30 Hz is 600 control ticks,
#: comfortably longer than a pick-and-place demonstration on this bench.
DEFAULT_MAX_EPISODE_S = 20.0


def seed_list(spec: dict) -> list[int]:
    """The seed of every episode this run will play, in order.

    Two ways in, and `seeds` wins: an explicit list is how a run re-plays
    exactly the layouts a previous run scored, which is the comparison anyone
    actually wants when they change a checkpoint. `episodes` + `seed_start`
    generates one, and generating is all it does - the generated list is written
    into the summary just the same, so a run is never described by a rule that
    has to be re-applied to be understood.
    """
    raw = spec.get("seeds")
    if raw is not None:
        if not isinstance(raw, (list, tuple)):
            raise SystemExit("seeds must be a list of whole numbers")
        try:
            seeds = [int(s) for s in raw]
        except (TypeError, ValueError) as e:
            raise SystemExit(f"seeds must be a list of whole numbers: {e}") from e
        if not seeds:
            raise SystemExit("seeds is empty - there is nothing to score")
        return seeds

    raw_n = spec.get("episodes")
    try:
        episodes = DEFAULT_EPISODES if raw_n is None else int(raw_n)
    except (TypeError, ValueError) as e:
        raise SystemExit(f"episodes must be a whole number: {e}") from e
    if episodes < 1:
        raise SystemExit(f"episodes must be at least 1, not {episodes}")
    try:
        start = DEFAULT_SEED_START if spec.get("seed_start") is None \
            else int(spec["seed_start"])
    except (TypeError, ValueError) as e:
        raise SystemExit(f"seed_start must be a whole number: {e}") from e
    return list(range(start, start + episodes))


def build_plan(spec: dict) -> dict:
    """Validate a spec into everything the evaluation needs, or refuse.

    Refusals are `SystemExit("<sentence>")`, which `_common.run_guarded` records
    as a `failed` run whose `error` is that sentence.

    Spec: `policy_path` (a `pretrained_model` directory), `config` (the rig YAML
    to evaluate on), `seeds` or `episodes`, and one of `action_names` /
    `repo_id` to say what the policy's action vector holds. Optional:
    `control_hz`, `max_episode_s`, `randomize`, `mirror`, `max_speed_deg_s`,
    `device`, `task`, `robot_type`. `run_id` / `run_dir` are stamped in by
    `lab/runs.launch`.

    **The action column layout is required and never guessed.** The policy emits
    a vector; which element is the left elbow is a property of the DATASET it
    was trained on, and this loop's own state layout is a property of the RIG.
    Assuming they agree would produce a success rate computed with the wrist
    driven by the shoulder's number, which does not fail loudly - it just scores
    zero and reads as a bad policy. `rollout_runner.resolve_rig` is the one
    resolver for this, reused rather than re-derived.

    **The rig is checked against the policy HERE**, where both are cheap to
    read, so a bimanual checkpoint pointed at a solo config is refused on the
    spec rather than three episodes into a run.

    Imports `haller_hmi.config` and `rollout_runner` and nothing heavier, so
    every refusal below is reachable from `--dry-run` with no GPU, no display
    and no MuJoCo.
    """
    raw_policy = str(spec.get("policy_path") or "").strip()
    if not raw_policy:
        raise SystemExit(
            "no policy_path - point at a pretrained_model directory, e.g. "
            "<run_dir>/train/checkpoints/last/pretrained_model")
    policy_path = Path(raw_policy).expanduser()
    if not policy_path.is_dir():
        raise SystemExit(f"no checkpoint directory at {policy_path}")
    if not (policy_path / CHECKPOINT_CONFIG).is_file():
        raise SystemExit(
            f"{policy_path} holds no {CHECKPOINT_CONFIG} - that is a checkpoint "
            "directory, not the pretrained_model directory inside it")

    config_path = _config_path(spec)
    cfg = load_config(config_path)
    arm_ids = [a.sim_arm_name for a in cfg.arms
               if a.enabled and a.source == "sim" and a.sim_arm_name]
    if not arm_ids:
        raise SystemExit(
            f"{config_path} has no enabled `source: sim` arm, so there is no "
            "bench to evaluate on. Point `config` at a sim rig, e.g. "
            "config.bimanual-sim.yaml.")
    camera_keys = [c.dataset_feature_key for c in cfg.cameras if c.record]

    rig = resolve_rig(spec)
    side = str(spec.get("side") or "")
    # Built once here so a layout mismatch is refused now, on the spec, rather
    # than on the first action of the first episode. `action_from_vector` is
    # also what the driver uses per tick, so this is the same mapping the run
    # will actually apply, not a check that resembles it.
    mapped = action_from_vector([0.0] * rig.dim, rig, side=side)
    if sorted(mapped) != sorted(arm_ids):
        raise SystemExit(
            f"this policy drives {sorted(mapped)} and {config_path.name} has "
            f"{sorted(arm_ids)}. A success rate measured with one arm's targets "
            "landing on the other, or on nothing, is not a measurement.")

    seeds = seed_list(spec)
    control_hz = _positive_float(spec, "control_hz", float(cfg.telemetry.hz))
    max_episode_s = _positive_float(spec, "max_episode_s", DEFAULT_MAX_EPISODE_S)
    # `in spec` and not `.get(...) or`: an explicit null is how a caller asks
    # for NO rate cap, and `or` would turn it into the default silently.
    if "max_speed_deg_s" in spec:
        raw_cap = spec["max_speed_deg_s"]
        max_speed = None if raw_cap is None else _positive_float(
            spec, "max_speed_deg_s", 0.0)
    else:
        max_speed = float(cfg.motion.max_speed_deg_s)

    return {
        "run_id": str(spec.get("run_id") or ""),
        "policy_path": str(policy_path),
        "config_path": str(config_path),
        "device": str(spec.get("device") or "cuda"),
        "task": str(spec.get("task") or ""),
        "robot_type": str(spec.get("robot_type") or ""),
        "repo_id": str(spec.get("repo_id") or ""),
        "side": side,
        "rig": rig.rig,
        # Carried so `PolicyDriver` can resolve the same rig from the plan
        # alone. `None` rather than `[]` when absent: `resolve_rig` treats a
        # falsy value as "not given" and falls through to `repo_id`, and an
        # empty list would resolve to a zero-column rig instead.
        "action_names": ([str(n) for n in spec["action_names"]]
                         if spec.get("action_names") else None),
        "arms": list(arm_ids),
        # Preview only. `sim/episode._recorded_cameras` is the authority and
        # additionally REFUSES a recorded camera this loop cannot render, which
        # this list deliberately does not do: importing it would drag mujoco
        # into the dry run.
        "camera_keys": camera_keys,
        "seeds": seeds,
        "control_hz": control_hz,
        "max_episode_s": max_episode_s,
        "max_speed_deg_s": max_speed,
        "randomize": bool(spec.get("randomize", True)),
        "mirror": bool(spec.get("mirror", False)),
        "out_path": Path(spec["run_dir"]) / SIM_EVAL_FILENAME,
        "summary_path": Path(spec["run_dir"]) / SIM_EVAL_SUMMARY_FILENAME,
    }


def _config_path(spec: dict) -> Path:
    """The rig YAML, resolved the way the HMI resolves its own.

    `spec['config']`, else `$HALLER_HMI_CONFIG`, else `config.py`'s default -
    the same order `load_config` applies for the server, so a sim evaluation and
    a sim cockpit disagree about the rig only when someone points them at
    different files on purpose. Checked for existence HERE because
    `load_config`'s own failure is a `FileNotFoundError` traceback, and a run
    that names the wrong config should say so in one sentence.
    """
    raw = spec.get("config")
    if raw:
        path = Path(str(raw)).expanduser()
    else:
        path = Path(os.environ.get("HALLER_HMI_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if not path.is_file():
        raise SystemExit(f"no rig config at {path}")
    # Resolved on the way out, so the summary records an absolute path. A run
    # launched with a relative `config` would otherwise record a name that only
    # means anything alongside the cwd in `run.json`, and the summary is the
    # file that gets copied out of the run directory.
    return path.resolve()


def _positive_float(spec: dict, key: str, default: float) -> float:
    """A number greater than zero, or a refusal naming what arrived."""
    raw = spec.get(key)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as e:
        raise SystemExit(f"{key} must be a number: {e}") from e
    if value <= 0:
        raise SystemExit(f"{key} must be greater than 0, not {value:g}")
    return value


def describe(plan: dict) -> list[str]:
    """The lines a run prints before it starts.

    Printed by `--dry-run` and by the real run from the same function, so the
    preflight describes what actually runs.
    """
    seeds = plan["seeds"]
    shown = ", ".join(str(s) for s in seeds[:12])
    if len(seeds) > 12:
        shown += f", ... ({len(seeds)} seeds)"
    cap = ("uncapped" if plan["max_speed_deg_s"] is None
           else f"{plan['max_speed_deg_s']:g} deg/s")
    return [
        f"sim success rate for {plan['policy_path']}",
        f"rig {plan['config_path']}: arms {plan['arms']}, "
        f"cameras {plan['camera_keys']}",
        f"{len(seeds)} episode(s), seeds {shown}",
        f"{plan['control_hz']:g} Hz control, {plan['max_episode_s']:g} sim s "
        f"per episode, step cap {cap}, "
        f"randomize={plan['randomize']} mirror={plan['mirror']}",
        f"policy rig {plan['rig']} -> arms {plan['arms']}, "
        f"device {plan['device']}",
        f"writing {plan['out_path']} and {plan['summary_path']}",
        ("the predicate in sim/task.py is the only success authority here: an "
         "episode it does not fire on is a failure, and the thresholds it fired "
         "against are written into the summary beside the rate"),
        ("a policy trained on REAL camera frames will not transfer to these "
         "renders. This number is evidence about a sim-trained policy."),
    ]


class PolicyDriver:
    """A trained checkpoint, wrapped as a `sim/episode.EpisodeDriver`.

    Holds the policy and the two processor pipelines, plus the permutation from
    the DATASET's action column order to the RIG's joint order. Both halves
    matter: the processors are where un-normalisation happens (so the returned
    numbers are degrees rather than plausible-looking normalised ones), and the
    permutation is what stops the left elbow being driven by whatever column
    happened to sit at index 2.
    """

    def __init__(self, plan: dict, state_names: list[str]) -> None:
        self.plan = plan
        self.rig = resolve_rig(plan_spec(plan))
        #: `(side, plain joint)` per element of the RIG's state vector, derived
        #: from `sim/episode.state_names` (`left_shoulder_pan` -> left,
        #: shoulder_pan). This is the order actions must come back in.
        self.targets = [tuple(name.split("_", 1)) for name in state_names]
        self.policy, self.preprocessor, self.postprocessor = _load_checkpoint(plan)

    def reset(self, seed: int) -> None:
        """Clear the policy's own state between episodes.

        An action-chunking policy (ACT, pi0) holds a queue of future actions
        from its last inference. Carried across an episode boundary, the first
        moves of episode N+1 are the tail of episode N's chunk, executed against
        a bench that has just been re-dealt. That is not scoring N episodes, it
        is scoring one episode N times with a discontinuity in it.

        `seed` is accepted and unused: the policy is deterministic given its
        observation, and the bench's randomness is the scene's. It is in the
        protocol because a scripted or stochastic driver needs it.
        """
        del seed
        self.policy.reset()

    def act(self, obs: dict) -> list[float]:
        """One inference step, in the RIG's joint order, in degrees.

        Mirrors `rollout_runner._infer` step for step: prepare, preprocess,
        `select_action`, postprocess. That is lerobot's own sync engine's order,
        and a hand-rolled forward pass would skip the un-normalisation and emit
        normalised numbers that look like degrees. The one difference is the
        input: that one decodes camera frames off a wire, this one is handed
        live `np.ndarray` renders by the episode loop, so there is nothing to
        decode.
        """
        import numpy as np
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        # A FRESH dict, because `prepare_observation_for_inference` rebinds
        # every value in the mapping it is handed (lerobot policies/utils.py:127)
        # and handing it the loop's own observation would leave torch tensors
        # where the driver protocol promises numpy renders. The camera arrays
        # are passed by reference and are not written to; only the state is
        # rebuilt, because that one arrives as a list of Python floats and
        # `torch.from_numpy` needs an array.
        frame: dict = {}
        for key, value in obs.items():
            frame[key] = (np.asarray(value, dtype=np.float32)
                          if key == "observation.state" else value)
        with torch.inference_mode():
            prepared = prepare_observation_for_inference(
                frame, torch.device(self.plan["device"]),
                self.plan["task"] or None,
                # The robot type belongs to the DATASET, not to this file, for
                # `rollout_runner._infer`'s reason: a literal here could not be
                # right for every rig.
                self.plan["robot_type"] or None,
            )
            action = self.postprocessor(
                self.policy.select_action(self.preprocessor(prepared)))
        values = action.squeeze(0).cpu().tolist()
        # Indexed by the DATASET's action column names, so it is reordered onto
        # the rig's layout by NAME rather than by position.
        mapped = action_from_vector(values, self.rig, side=self.plan["side"])
        return [float(mapped[side][joint]) for side, joint in self.targets]


def plan_spec(plan: dict) -> dict:
    """The subset of the original spec `resolve_rig` reads.

    `build_plan` has already resolved the rig once and thrown the object away
    rather than putting a mutable `RigSpec` into the plan dict - the same choice
    `rollout_runner._rollout` makes, and for the same reason. This rebuilds the
    two keys it needs so the driver can resolve it again from the plan alone.
    """
    return {"action_names": plan.get("action_names"),
            "repo_id": plan.get("repo_id")}


def _load_checkpoint(plan: dict):
    """`(policy, preprocessor, postprocessor)`, through `rollout_runner`'s loader.

    Imported, not copied. `rollout_runner._load_policy` reads `plan["device"]`
    and `plan["policy_path"]`, which this plan carries under the same names on
    purpose: two loaders would eventually disagree about whether the pre/post
    processor pipeline was applied, and that disagreement is the difference
    between an action in degrees and one that is normalised and looks fine.
    """
    from .rollout_runner import _load_policy

    return _load_policy(plan)


def summarise(plan: dict, records: list, provenance: dict, *,
              wall_s: float) -> dict:
    """The run's one number, with everything needed to reproduce it.

    `success_rate` is over the episodes that ACTUALLY RAN, and `complete` says
    whether that was all of them. A run stopped half way has measured a real
    rate over a real subset; reporting it as the rate over the requested seed
    list would be a different and false claim, and reporting nothing would throw
    away work that was done.
    """
    n = len(records)
    successes = sum(1 for r in records if r.success)
    steps = sum(r.steps for r in records)
    sim_s = sum(r.sim_s for r in records)
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    return {
        "run_id": plan["run_id"],
        "policy_path": plan["policy_path"],
        "config_path": plan["config_path"],
        "repo_id": plan["repo_id"],
        "device": plan["device"],
        "task": plan["task"],
        # The number, and the two things that qualify it.
        "success_rate": (successes / n) if n else 0.0,
        "successes": successes,
        "n": n,
        "requested": len(plan["seeds"]),
        "complete": n == len(plan["seeds"]),
        # The layouts ARE the experiment. Both lists, because a stopped run's
        # rate is over the first `n` of them.
        "seeds": [int(s) for s in plan["seeds"]],
        "seeds_run": [int(r.seed) for r in records],
        "reasons": reasons,
        # What decided every one of those verdicts.
        "provenance": provenance,
        "steps": steps,
        "sim_s": sim_s,
        "wall_s": wall_s,
        "control_ticks_per_s": (steps / wall_s) if wall_s > 0 else 0.0,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def _simeval(plan: dict) -> None:
    """Run the seed list, writing each row as it lands and a summary at the end.

    Rows are appended per episode and line buffered, for `eval_runner._evaluate`'s
    reason: this walks a policy through hundreds of inference steps per episode,
    and a run stopped half way should still report the half it measured.

    The summary is written in a `finally`, which is the same argument one level
    up. It carries `complete` so a partial rate can never be read as a full one.
    """
    # HERE and not at module scope: `sim/episode` imports mujoco, and
    # `--dry-run` must stay runnable from a box with no MuJoCo, no GPU and no
    # display. Same rule `eval_runner` applies to torch.
    from ..sim.episode import EpisodeRunner, EpisodeSpec

    for line in describe(plan):
        print(line, flush=True)

    cfg = load_config(Path(plan["config_path"]))
    spec = EpisodeSpec(
        control_hz=plan["control_hz"],
        max_episode_s=plan["max_episode_s"],
        randomize=plan["randomize"],
        mirror=plan["mirror"],
        max_speed_deg_s=plan["max_speed_deg_s"],
    )
    records: list = []
    started = time.perf_counter()
    with EpisodeRunner(cfg, spec) as runner:
        provenance = runner.provenance()
        print(f"loading {plan['policy_path']} on {plan['device']}", flush=True)
        driver = PolicyDriver(plan, runner.state_names)
        out = open(plan["out_path"], "w", buffering=1)  # noqa: SIM115 - held
        try:                                            # across every episode
            for record in runner.run(plan["seeds"], driver):
                records.append(record)
                out.write(json.dumps(record.row()) + "\n")
                done = sum(1 for r in records if r.success)
                print(f"seed {record.seed}: "
                      f"{'SUCCESS' if record.success else record.reason} "
                      f"after {record.steps} steps / {record.sim_s:.2f} sim s "
                      f"({done}/{len(records)} so far)", flush=True)
        finally:
            out.close()
            wall_s = time.perf_counter() - started
            summary = summarise(plan, records, provenance, wall_s=wall_s)
            Path(plan["summary_path"]).write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"success rate {summary['success_rate']:.3f} "
          f"({summary['successes']}/{summary['n']}) over seeds "
          f"{summary['seeds_run']}", flush=True)
    print(f"{summary['control_ticks_per_s']:.1f} control ticks/s "
          f"({summary['steps']} steps in {summary['wall_s']:.1f} s wall, "
          f"{summary['sim_s']:.1f} s sim)", flush=True)
    print(f"wrote {plan['out_path']} and {plan['summary_path']}", flush=True)


def main() -> int:
    spec, dry_run = load_spec(sys.argv[1:])
    run_dir = Path(spec["run_dir"])

    if dry_run:
        # Every refusal `build_plan` makes fires here too, and nothing heavy is
        # imported: no torch, no CUDA context, no MuJoCo model, no checkpoint
        # read. A refusal leaves this as `SystemExit` and writes no
        # `result.json`, because no run happened.
        for line in describe(build_plan(spec)):
            print(line)
        return 0

    return run_guarded(run_dir, lambda: _simeval(build_plan(spec)))


if __name__ == "__main__":
    raise SystemExit(main())
