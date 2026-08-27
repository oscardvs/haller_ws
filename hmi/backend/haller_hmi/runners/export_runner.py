# hmi/backend/haller_hmi/runners/export_runner.py
"""Detached child that prunes rejected episodes out of a dataset.

Launched by `lab/runs.launch("export", spec)` as

    ~/venvs/haller-lab/bin/python -m haller_hmi.runners.export_runner SPEC.json

Two modes, both driven by LeRobot's own `delete_episodes`:

  copy      (default) write a NEW dataset holding only the kept episodes. The
            source is never touched, so a review decision can be revisited.
  in_place  rewrite the dataset itself. Delegates to `lerobot-edit-dataset`'s
            own `handle_delete_episodes` so the backup-then-replace dance
            matches that CLI exactly, including moving the original to
            `<name>_old` first. `keep_backup: false` then deletes that too and
            is the ONLY genuinely unrecoverable path here — on a box with no
            backup of any kind.

**This is a job rather than a request because it RE-ENCODES VIDEO.** A v3.0
dataset packs many episodes into one mp4, so dropping one from the middle means
re-encoding the file around the hole, at the dataset's own crf/preset — minutes
of AV1, not milliseconds. Measured on `local/so101_pick_cube`: 46 episodes in 7
files totalling 707 MB.

Two facts about the result are operator-facing and every run PRINTS both:

* episodes are RENUMBERED 0..n-1, so "Ep 4" before the prune is not "Ep 4"
  after it;
* review marks are therefore NOT carried over — a copy starts with none, and an
  in-place prune CLEARS the source's, because keeping them would silently
  attach old decisions to new episodes.

## Why the paths are resolved here and not by `lab/catalog.py`

`catalog.hf_home` / `catalog.dataset_root` answer exactly this question, and
this module CANNOT import them: `catalog` imports `..api.errors`, which imports
`fastapi`, and **fastapi is not installed in `~/venvs/haller-lab`** (verified
2026-08-27 — that venv is lerobot 0.6.1 + torch 2.11.0+cu130 and nothing web).
An `import catalog` here would make every export die at import time in the only
interpreter that can run one. So `_hf_home` / `_dataset_root` below are catalog's
four lines again, and `tests/lab/test_export_runner.py` asserts the two agree —
on the symlinked-home case and on the traversal refusal — rather than trusting
that they still do.

`lab/review.py` IS imported directly: it is stdlib-only and imports fine over
there, and `review.clear` writing the sidecar is the same act on both sides.

    python -m haller_hmi.runners.export_runner SPEC.json
    python -m haller_hmi.runners.export_runner SPEC.json --dry-run
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ..lab import review
from ._common import load_spec, run_guarded

__all__ = ["MODES", "build_plan", "describe", "main"]

#: Write a new dataset; the source survives untouched.
COPY = "copy"
#: Rewrite the dataset itself, through `lerobot-edit-dataset`'s own handler.
IN_PLACE = "in_place"

MODES = (COPY, IN_PLACE)

#: Printed by every run, in both modes. The renumbering is the fact that makes
#: a stale review mark dangerous rather than merely wrong: index 3 still exists
#: afterwards, it is just a different demonstration.
RENUMBER_NOTE = (
    'episodes are RENUMBERED 0..n-1 — "Ep 4" before this prune is not '
    '"Ep 4" after it'
)


def _hf_home() -> Path:
    """The dataset cache root, fully resolved.

    Deliberately a copy of `lab/catalog.hf_home` — see the module docstring for
    why this module cannot import it. The `.resolve()` is load-bearing and is
    what broke the page once: `~/.cache/huggingface/lerobot` is a SYMLINK to
    `~/robot-data/lerobot` on this box, and an unresolved base has no common
    prefix with a resolved root.
    """
    base = os.environ.get("HF_LEROBOT_HOME")
    if base:
        return Path(base).expanduser().resolve()
    return (Path.home() / ".cache/huggingface/lerobot").resolve()


def _dataset_root(repo_id: str) -> Path:
    """Resolve a repo-id to its directory, refusing to escape the cache.

    `repo_id` and `new_repo_id` reach this process from an HTTP body, so `../`
    is a real concern: without this check `mode=in_place` would rmtree a
    directory of the caller's choosing.
    """
    base = _hf_home()
    root = (base / repo_id).resolve()
    if not (root == base or base in root.parents):
        raise ValueError(f"repo_id escapes the dataset cache: {repo_id!r}")
    return root


def _resolve(repo_id: str, what: str) -> Path:
    """`_dataset_root`, with the traversal refusal wearing the same shape as
    every other refusal below."""
    try:
        return _dataset_root(repo_id)
    except ValueError as e:
        raise SystemExit(f"{what}: {e}") from e


def _labels(drop: list[int]) -> str:
    """`Ep 4 (idx 3), Ep 7 (idx 6)`.

    BOTH spellings, in full, however long the list gets. Oscar numbers episodes
    1-based in conversation and the parquet stores them 0-based; this line is
    the last thing printed before minutes of irreversible re-encoding, and that
    off-by-one is how the wrong demonstration gets deleted.
    """
    return ", ".join(f"Ep {e + 1} (idx {e})" for e in drop)


def build_plan(spec: dict) -> dict:
    """Validate a spec into everything the export needs, or refuse.

    Every refusal is `SystemExit("<sentence>")`, which `_common.run_guarded`
    records as a `failed` run whose `error` is that sentence — the shape the
    runs table renders. A refusal must reach here BEFORE lerobot is imported,
    because the whole value of a preflight on this job is that it costs a
    millisecond where being wrong costs minutes of video and, in one mode, the
    dataset.

    Spec keys (the kit's, kept): `repo_id`, `mode`, `new_repo_id`,
    `delete_episodes`, `keep_backup`. `run_id` / `run_dir` are stamped in by
    `lab/runs.launch`.
    """
    repo_id = str(spec.get("repo_id") or "").strip().strip("/")
    if not repo_id:
        raise SystemExit("no repo_id — there is nothing to export")

    mode = str(spec.get("mode") or COPY)
    if mode not in MODES:
        raise SystemExit(f"unknown export mode {mode!r} — one of {', '.join(MODES)}")

    try:
        drop = sorted({int(e) for e in spec.get("delete_episodes") or []})
    except (TypeError, ValueError) as e:
        raise SystemExit(f"delete_episodes must be a list of episode indices: {e}") from e
    if not drop:
        raise SystemExit(
            "nothing to drop — every episode is kept. A prune that removes "
            "nothing still re-encodes every video file to produce a dataset "
            "identical to its source."
        )

    root = _resolve(repo_id, "repo_id")
    if not root.is_dir():
        # Without this, `LeRobotDataset(repo_id, root=<missing>)` goes to the
        # Hub looking for a `local/...` repo that was only ever a typo, and the
        # run fails as a network error minutes later.
        raise SystemExit(f"no dataset at {root} — check the repo-id")

    plan = {
        "mode": mode,
        "repo_id": repo_id,
        "root": root,
        "drop": drop,
        "new_repo_id": None,
        "output_dir": None,
        "keep_backup": True,
        "backup": root.with_name(root.name + "_old"),
    }

    if mode == COPY:
        new_repo_id = str(spec.get("new_repo_id") or "").strip().strip("/")
        if not new_repo_id:
            raise SystemExit(
                "copy mode needs a new_repo_id — a copy whose whole point is "
                "that the source survives it has to be written somewhere else"
            )
        output_dir = _resolve(new_repo_id, "new_repo_id")
        # Compared as PATHS, not as strings: `local/x`, `local//x` and
        # `./local/x` are three spellings of one directory, and a string
        # compare would let two of them through into a self-overwrite.
        if new_repo_id == repo_id or output_dir == root:
            raise SystemExit(
                f"refusing to export {repo_id} onto itself — pick another "
                "repo-id, or ask for mode=in_place and mean it"
            )
        if output_dir.exists():
            raise SystemExit(
                f"{output_dir} already exists — choose a different name rather "
                "than overwriting a dataset"
            )
        plan["new_repo_id"] = new_repo_id
        plan["output_dir"] = output_dir
    else:
        plan["keep_backup"] = bool(spec.get("keep_backup", True))

    return plan


def describe(plan: dict) -> list[str]:
    """The lines a run prints before it starts, one per fact.

    Printed by `--dry-run` and by the real run, from the same function: a
    preflight that describes something other than what runs is worse than none.
    """
    lines: list[str] = []
    if plan["mode"] == COPY:
        lines += [
            f"export {plan['repo_id']} -> {plan['new_repo_id']}",
            f"dropping {len(plan['drop'])} episode(s): {_labels(plan['drop'])}",
            f"output: {plan['output_dir']}",
            (f"the source at {plan['root']} is not touched — this review "
             "decision can be revisited"),
            RENUMBER_NOTE,
            ("the copy starts with NO review marks — carrying them over would "
             "attach old decisions to new episodes"),
        ]
    else:
        lines.append(
            f"PERMANENTLY deleting {len(plan['drop'])} episode(s) from "
            f"{plan['repo_id']}: {_labels(plan['drop'])}"
        )
        if plan["keep_backup"]:
            lines.append(f"the original is moved to {plan['backup'].name} first")
        else:
            lines.append(
                f"{plan['backup'].name} will be DELETED afterwards — this "
                "prune is NOT recoverable, and this box has no backup of any kind"
            )
        lines += [
            RENUMBER_NOTE,
            (f"{plan['repo_id']}'s review marks are CLEARED — keeping them "
             "would attach old decisions to new episodes"),
        ]
    lines.append(
        "this re-encodes video around every hole it makes and takes minutes, "
        "not seconds"
    )
    return lines


def _export_copy(plan: dict) -> None:
    """Write a new dataset holding only the kept episodes."""
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(plan["repo_id"], root=str(plan["root"]))
    new_dataset = delete_episodes(
        dataset,
        episode_indices=plan["drop"],
        output_dir=plan["output_dir"],
        repo_id=plan["new_repo_id"],
    )
    total = new_dataset.meta.total_episodes
    print(
        f"wrote {plan['new_repo_id']}: {total} episodes, "
        f"{new_dataset.meta.total_frames} frames -> {plan['output_dir']}",
        flush=True,
    )
    print(f"episodes are now 0..{total - 1}, and the copy has no review marks.",
          flush=True)


def _prune_in_place(plan: dict) -> None:
    """Rewrite the dataset without the rejected episodes.

    Delegates to `lerobot-edit-dataset`'s own in-place handler rather than
    calling `delete_episodes` and moving directories here: `get_output_path`
    moves the original to `<name>_old` BEFORE the rewrite and points the source
    dataset at it, so a failure half way leaves the backup whole. Reimplementing
    that ordering is how the two would drift, and the direction it drifts is the
    one that loses the dataset.
    """
    import shutil

    from lerobot.scripts.lerobot_edit_dataset import (
        DeleteEpisodesConfig,
        EditDatasetConfig,
        handle_delete_episodes,
    )

    root = plan["root"]
    cfg = EditDatasetConfig(
        operation=DeleteEpisodesConfig(episode_indices=plan["drop"]),
        repo_id=plan["repo_id"],
        root=str(root),
        # Identical to `repo_id`/`root`, which is what `_is_in_place` tests for.
        new_repo_id=plan["repo_id"],
        new_root=str(root),
    )
    handle_delete_episodes(cfg)

    # The rewrite moved the ORIGINAL — review.json included — aside to `_old`,
    # so what stands at `root` now is a fresh dataset with no sidecar. Clearing
    # it anyway is belt and braces and costs one small write: the marks the
    # operator can still see are the ones in the backup, where they still
    # describe the episodes they were made about.
    review.clear(root)
    print("review marks cleared — the episodes they described were renumbered",
          flush=True)

    backup = plan["backup"]
    if plan["keep_backup"]:
        if backup.exists():
            print(f"previous version kept at {backup}", flush=True)
        return
    if backup.exists():
        # Only reached once the rewrite above returned, so this discards a
        # backup of a dataset that has already been replaced intact.
        shutil.rmtree(backup)
        print(f"backup {backup.name} deleted — this prune is not recoverable",
              flush=True)


def _export(spec: dict) -> None:
    """The guarded half: validate, say what is about to happen, then do it.

    `build_plan` is called HERE, inside `run_guarded`, and not before it: a
    refusal outside the guard would return without writing `result.json`, and
    `lab/runs.load` reports that as `died` — a crash — rather than as the
    deliberate refusal it was.
    """
    plan = build_plan(spec)
    for line in describe(plan):
        print(line, flush=True)
    if plan["mode"] == COPY:
        _export_copy(plan)
    else:
        _prune_in_place(plan)


def main() -> int:
    spec, dry_run = load_spec(sys.argv[1:])
    run_dir = Path(spec["run_dir"])

    if dry_run:
        # Every refusal fires here too — a dry run that passes while the real
        # run refuses is a preflight that lies — and NOTHING heavy is imported:
        # no lerobot, no re-encode, no dataset moved. A refusal leaves this
        # function as `SystemExit`, which exits non-zero with its own message
        # and writes no `result.json`, because no run happened.
        for line in describe(build_plan(spec)):
            print(line)
        return 0

    return run_guarded(run_dir, lambda: _export(spec))


if __name__ == "__main__":
    raise SystemExit(main())
