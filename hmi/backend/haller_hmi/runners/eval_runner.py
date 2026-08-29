# hmi/backend/haller_hmi/runners/eval_runner.py
"""Detached child that scores a trained checkpoint EPISODE BY EPISODE.

Launched by `lab/runs.launch("eval", spec)` as

    ~/venvs/haller-lab/bin/python -m haller_hmi.runners.eval_runner SPEC.json

It writes `<run_dir>/episode_loss.jsonl`, one row per episode:

    {"episode": 3, "loss": 0.0412, "frames": 640}

That file is the ONLY thing `lab/autoclass.py`'s `policy-loss` mode reads, and
until this runner existed that mode answered `available: false` with a reason
saying so. The key spellings here are the ones its reader already accepts —
`tests/lab/test_export_runner.py` feeds a hand-written file through
`autoclass.preview` to hold the two sides to it.

There is no kit equivalent. Per-episode loss is not something LeRobot logs: the
trainer reports one held-out `eval_loss` per eval step, averaged over whatever
the split happened to hold out, which cannot rank one demonstration against
another.

## What this number is, and what it is not

It is the mean per-frame training loss of a fixed policy over one episode's
frames — how badly the policy fits that demonstration. **It is not a quality
score.** A high loss is as often a rare-but-correct demonstration the dataset
has only one of as it is a bad one, and deleting the tail of the loss
distribution is how a policy loses the only examples of the case it fails.

That is why `autoclass` surfaces it as a SORT ORDER and never as an auto-mark,
why `apply` on that mode is a 400, and why **this runner writes no marks
either**. It imports `lab.review` for nothing and touches `review.json` never.
The output is a ranking of what to WATCH.

Nothing about it is a rollout, either: no policy action reaches a servo here.
`policy.forward(batch)` is the training loss on recorded frames, on the GPU,
with the arm powered down.

## The measurement follows `lerobot_train`'s own eval branch

`lerobot/scripts/lerobot_train.py:654-667`: `policy.eval()`, uint8 camera
tensors scaled to float, `preprocessor(batch)`, then `policy.forward(batch)`.
Reproduced step for step rather than reinvented, so a per-episode loss is
comparable with the `eval_loss` on the same run's chart instead of being a
second, subtly different number wearing the same name. The pipeline's own
device step does the placement, exactly as it does there.

One episode at a time, through a `LeRobotDataset(..., episodes=[i])`, so every
frame in a batch belongs to the episode being scored and the batch mean is a
mean over its frames. Batches are weighted by their length before averaging,
because the last batch of an episode is short.

`lerobot`/`torch` are imported INSIDE `main()`'s call tree. `--dry-run` prints
the plan and imports neither, which is what makes any of this testable from the
serving venv (lerobot 0.5.1, no trainer).

    python -m haller_hmi.runners.eval_runner SPEC.json
    python -m haller_hmi.runners.eval_runner SPEC.json --dry-run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ._common import load_spec, run_guarded

__all__ = ["EPISODE_LOSS_FILENAME", "build_plan", "describe", "main"]

#: What `lab/autoclass.py` looks for in a run directory. Spelled here as well as
#: there because `autoclass` imports `catalog`, which imports `api/errors`,
#: which imports `fastapi` — and fastapi is NOT installed in
#: `~/venvs/haller-lab` (verified 2026-08-27), so this module cannot import the
#: constant from the one place that also uses it. The test asserts they match.
EPISODE_LOSS_FILENAME = "episode_loss.jsonl"

#: `huggingface_hub.constants.CONFIG_NAME`. A `pretrained_model` directory
#: without it is not one, and `PreTrainedConfig.from_pretrained` answers a
#: missing local config by going to the HUB with the path as a repo name.
CHECKPOINT_CONFIG = "config.json"

#: Dataloader workers. Not a spec key: the cost here is video decode, the
#: number is not tuned, and a form field for it would imply it was.
DATALOADER_WORKERS = 4


def build_plan(spec: dict) -> dict:
    """Validate a spec into everything the evaluation needs, or refuse.

    Refusals are `SystemExit("<sentence>")`, which `_common.run_guarded` records
    as a `failed` run whose `error` is that sentence.

    Spec: `repo_id`, `checkpoint` (a `pretrained_model` directory — exactly what
    `lab/runs.checkpoints()` returns as `path`), `episodes` (optional; every
    episode when absent), `device`, `batch_size`. `run_id` / `run_dir` are
    stamped in by `lab/runs.launch`.

    **The dataset root is not resolved here.** `repo_id` goes to LeRobot
    untouched, so this run reads the dataset from wherever `lerobot-train` read
    it — one `HF_LEROBOT_HOME` rule, LeRobot's, rather than a second copy of it
    that can disagree with the trainer about which dataset was scored.
    """
    repo_id = str(spec.get("repo_id") or "").strip().strip("/")
    if not repo_id:
        raise SystemExit("no repo_id — there is nothing to evaluate")

    raw = str(spec.get("checkpoint") or "").strip()
    if not raw:
        raise SystemExit(
            "no checkpoint — point at a pretrained_model directory, e.g. "
            "<run_dir>/train/checkpoints/last/pretrained_model"
        )
    checkpoint = Path(raw).expanduser()
    if not checkpoint.is_dir():
        raise SystemExit(f"no checkpoint directory at {checkpoint}")
    if not (checkpoint / CHECKPOINT_CONFIG).is_file():
        # Named rather than left to LeRobot: with no local config it treats the
        # path as a Hub repo id and fails with a 404 about a repository nobody
        # asked for.
        raise SystemExit(
            f"{checkpoint} holds no {CHECKPOINT_CONFIG} — that is a checkpoint "
            "directory, not the pretrained_model directory inside it"
        )

    episodes = spec.get("episodes")
    if episodes is not None:
        try:
            episodes = [int(e) for e in episodes]
        except (TypeError, ValueError) as e:
            raise SystemExit(f"episodes must be a list of episode indices: {e}") from e
        if not episodes:
            raise SystemExit("episodes is empty — there is nothing to score")

    # `or 8` would turn a `batch_size: 0` into 8 and run the job the caller
    # asked it not to. 0 is a value someone can type; it has to refuse.
    raw_batch = spec.get("batch_size")
    try:
        batch_size = 8 if raw_batch is None else int(raw_batch)
    except (TypeError, ValueError) as e:
        raise SystemExit(f"batch_size must be a whole number: {e}") from e
    if batch_size < 1:
        raise SystemExit(f"batch_size must be at least 1, not {batch_size}")

    return {
        "repo_id": repo_id,
        "checkpoint": checkpoint,
        "episodes": episodes,
        "device": str(spec.get("device") or "cuda"),
        "batch_size": batch_size,
        "out_path": Path(spec["run_dir"]) / EPISODE_LOSS_FILENAME,
    }


def describe(plan: dict) -> list[str]:
    """The lines a run prints before it starts.

    Printed by `--dry-run` and by the real run from the same function, so the
    preflight describes what actually runs.
    """
    episodes = plan["episodes"]
    which = (f"{len(episodes)} episode(s): "
             + ", ".join(f"Ep {e + 1} (idx {e})" for e in episodes)
             if episodes else f"every episode in {plan['repo_id']}")
    return [
        f"per-episode loss for {plan['checkpoint']}",
        f"dataset {plan['repo_id']}, {which}",
        f"device {plan['device']}, batch_size {plan['batch_size']}",
        f"writing {plan['out_path']}",
        ("this ranks episodes the policy fits badly — a high loss is as often a "
         "rare-but-correct demonstration as a bad one, so it is a sort order "
         "and never a mark"),
    ]


def _load_policy(plan: dict):
    """`(policy, preprocessor, dataset metadata, delta_timestamps)`.

    Built once and reused across every episode: the checkpoint is hundreds of MB
    and the dataset's stats are the same for all of them.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    path = str(plan["checkpoint"])
    cfg = PreTrainedConfig.from_pretrained(path)
    # `pretrained_path` is what makes `make_policy` LOAD the weights instead of
    # initialising a fresh net — a policy scored without it would report the
    # loss of random parameters, uniformly high, and rank nothing.
    cfg.pretrained_path = path
    cfg.device = plan["device"]

    meta = LeRobotDatasetMetadata(plan["repo_id"])
    policy = make_policy(cfg, ds_meta=meta)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        cfg, pretrained_path=path, dataset_stats=meta.stats, dataset_meta=meta,
    )
    # An action-chunking policy is trained against a WINDOW of future actions;
    # without the same delta timestamps the trainer used, `forward` would score
    # a differently-shaped target and the number would not be the training loss
    # at all.
    return policy, preprocessor, meta, resolve_delta_timestamps(cfg, meta)


def _episode_loss(plan: dict, episode: int, policy, preprocessor, meta,
                  delta_timestamps) -> tuple[float, int]:
    """`(mean per-frame loss, frames)` for one episode."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.collate import lerobot_collate_fn

    dataset = LeRobotDataset(
        plan["repo_id"], episodes=[episode], delta_timestamps=delta_timestamps,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=plan["batch_size"],
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=plan["device"].startswith("cuda"),
        drop_last=False,
        # `lerobot_train.py:500` picks the collate the same way. A dataset with
        # language columns hands `default_collate` a list of strings, which it
        # cannot stack; the wrong choice here is a crash on the first batch of
        # any dataset carrying a task instruction.
        collate_fn=lerobot_collate_fn if meta.has_language_columns else None,
    )

    total, frames = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:  # lerobot_collate_fn drops empty batches
                continue
            # Read off the batch itself, before the preprocessor reshapes
            # anything. `loss.item()` is a mean over the batch and the LAST
            # batch of an episode is short, so weighting by the real length is
            # what makes the division below a mean over FRAMES rather than over
            # batches.
            n = next((int(v.shape[0]) for v in batch.values()
                      if torch.is_tensor(v) and v.ndim), plan["batch_size"])
            for cam_key in meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            total += float(loss.item()) * n
            frames += n
    return (total / frames if frames else 0.0), frames


def _evaluate(plan: dict) -> None:
    """Score every requested episode, writing each row as it lands.

    Line buffered and appended per episode rather than dumped at the end: this
    walks every frame of every episode through a policy, and a run stopped half
    way should still rank the half it measured. `lab/autoclass` reads whole JSON
    lines and skips anything it cannot parse, so a file cut mid-row costs one
    episode's rank.
    """
    for line in describe(plan):
        print(line, flush=True)

    policy, preprocessor, meta, delta_timestamps = _load_policy(plan)
    episodes = plan["episodes"]
    if episodes is None:
        episodes = list(range(meta.total_episodes))

    out = open(plan["out_path"], "w", buffering=1)  # noqa: SIM115 - held across
    try:                                            # every episode, closed below
        for episode in episodes:
            loss, frames = _episode_loss(
                plan, episode, policy, preprocessor, meta, delta_timestamps)
            out.write(json.dumps(
                {"episode": episode, "loss": loss, "frames": frames}) + "\n")
            print(f"Ep {episode + 1} (idx {episode}): loss {loss:.4f} "
                  f"over {frames} frames", flush=True)
    finally:
        out.close()

    print(f"wrote {plan['out_path']} — hand this run's id to the Lab page's "
          "policy-loss ranking", flush=True)


def main() -> int:
    spec, dry_run = load_spec(sys.argv[1:])
    run_dir = Path(spec["run_dir"])

    if dry_run:
        # Refusals fire here too, and NOTHING heavy is imported: no torch, no
        # CUDA context, no checkpoint read. A refusal leaves this function as
        # `SystemExit` and writes no `result.json`, because no run happened.
        for line in describe(build_plan(spec)):
            print(line)
        return 0

    return run_guarded(run_dir, lambda: _evaluate(build_plan(spec)))


if __name__ == "__main__":
    raise SystemExit(main())
