#!/usr/bin/env python
"""Offline policy evaluation: replay a recorded dataset through a policy.

For each frame of one recorded episode, this script:
  1. feeds the observation (state + camera images) to the policy,
  2. collects the predicted action,
  3. compares it against the action the human teleoperator actually used.

Outputs (under --output-dir):
  - actions.csv       per-frame predicted vs ground-truth, every joint
  - summary.json      per-joint MAE / RMSE + meta (policy, dataset, episode, gpu)
  - joints.png        matplotlib figure: left arm in one column, right arm in the
                      other, one row per joint, pred + ground-truth overlaid

This is NOT a substitute for closed-loop evaluation on the real robot —
the policy never gets to see how the scene responds to its actions. But it's
a quick, hardware-free way to answer "would this policy have done something
reasonable?" before you commit time to a live deployment.

--- HALLER IS BIMANUAL -------------------------------------------------------
Two SO-101 arms => 12-dim state/action: left arm's 6 joints
(left_shoulder_pan … left_gripper) then the right arm's 6, in DEGREES. π0.5 pads
state/action to 32 dims internally and unpads the output back to the dataset's
real width, so nothing here needs a bimanual switch. The per-joint stats below
are derived from `dataset.features["action"]["names"]`, so they follow whatever
your dataset actually declares — 6 joints, 12, or anything else.

--- WHICH CAMERA KEYS DOES THE POLICY WANT? (read this before using --rename) ---
Earlier versions of this file told you to map your cameras onto
`observation.images.cam0` / `cam1`. That was wrong twice over: no such naming
convention exists anywhere in lerobot, and renaming is usually the thing that
BREAKS a working setup. What actually decides the expected keys is how the
policy's `input_features` get populated — and there are exactly two modes,
mirroring the two forms lerobot-train accepts:

  MODE A — features from YOUR dataset          (default here; `--from-dataset`)
    The lerobot-train analogue is:
        --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base
    make_policy() sees empty input_features and fills them from the dataset
    metadata, so the policy expects YOUR key names verbatim. A rename map here is
    unnecessary and actively harmful: you would be renaming your keys to names
    nothing is looking for.

  MODE B — features from the CHECKPOINT        (`--from-policy`)
    The lerobot-train analogue is:
        --policy.path=lerobot/pi05_base
    which loads the saved config wholesale, camera names included (pi05_base
    ships base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb). make_policy() does
    NOT overwrite non-empty input_features, so your dataset's keys will not
    match, and you get:
        ValueError: All image features are missing from the batch
    THIS is where a rename map belongs. Use --rename (repeatable), which maps
    DATASET key -> POLICY key and is fed to lerobot's own
    RenameObservationsProcessorStep — the same mechanism as lerobot-train's
    top-level `--rename_map='{"observation.images.left": "observation.images.camera1"}'`.

Note that output_features (the action) is ALWAYS taken from the dataset by
make_policy, in both modes. Only the inputs differ.

--- STATS ---
π0.5 normalizes state/action with QUANTILES, so your dataset's meta/stats.json
must carry q01/q99. Older recordings raise "QUANTILES normalization mode
requires q01 and q99 stats" on the very first frame. Fix once:
    lerobot-edit-dataset --repo_id <ds> --operation.type recompute_stats --push_to_hub true

This script always overrides the checkpoint's saved normalizer with YOUR
dataset's stats. That matters: make_pre_post_processors() loads the pipelines
verbatim when given a pretrained_path and silently ignores a `dataset_stats=`
kwarg, so a naive call would normalize your degrees against pi05_base's
pretraining statistics and produce meaningless error numbers.

--- PRECISION ---
--dtype sets config.dtype, which π0.5 applies selectively (it keeps the vision
tower and layernorms in float32 on purpose). Observations stay float32: π0.5's
prompt step calls state.cpu().numpy(), and torch cannot hand bfloat16 to numpy.

Usage:
    # Default: features from your dataset, eval episode 0
    python scripts/runpod/replay_eval.py \\
        --dataset-repo $HF_USER/haller_bimanual_pick_red_cube

    # Pick a specific policy + episode:
    python scripts/runpod/replay_eval.py \\
        --dataset-repo $HF_USER/haller_bimanual_pick_red_cube \\
        --policy-repo $HF_USER/pi05_haller_bimanual_pick_red_cube_lora \\
        --episode 3

    # MODE B: keep the checkpoint's own camera names, and map yours onto them.
    python scripts/runpod/replay_eval.py \\
        --dataset-repo $HF_USER/haller_bimanual_pick_red_cube \\
        --from-policy \\
        --rename observation.images.base=observation.images.base_0_rgb \\
        --rename observation.images.left_wrist=observation.images.left_wrist_0_rgb \\
        --rename observation.images.right_wrist=observation.images.right_wrist_0_rgb
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy


def _resolve_policy_class(policy_repo: str) -> type[PreTrainedPolicy]:
    """Pick the right policy class for a known generalist checkpoint."""
    n = policy_repo.lower()
    if "pi05" in n:
        from lerobot.policies.pi05 import PI05Policy
        return PI05Policy
    if "pi0_fast" in n or "pi0fast" in n:
        from lerobot.policies.pi0_fast import PI0FastPolicy
        return PI0FastPolicy
    if "pi0" in n:
        from lerobot.policies.pi0 import PI0Policy
        return PI0Policy
    if "smolvla" in n:
        from lerobot.policies.smolvla import SmolVLAPolicy
        return SmolVLAPolicy
    if "act" in n:
        from lerobot.policies.act import ACTPolicy
        return ACTPolicy
    raise SystemExit(
        f"don't know which policy class to use for {policy_repo!r} — "
        "extend _resolve_policy_class()."
    )


def _parse_rename(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            raise SystemExit(f"--rename expects DATASET_KEY=POLICY_KEY, got {v!r}")
        k, _, dst = v.partition("=")
        out[k.strip()] = dst.strip()
    return out


@dataclass
class JointStats:
    name: str
    mae: float
    rmse: float
    max_err: float


def _add_batch_dim(value):
    """LeRobotDataset returns single frames; policies expect batched input.

    The preprocessor's AddBatchDimensionProcessorStep would do this too, and it
    is idempotent, so doing it here is belt-and-braces for the keys it does not
    cover. Note what we do NOT do: cast dtypes. π0.5's state-discretization step
    calls state.cpu().numpy(), which raises on bfloat16.
    """
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    if isinstance(value, str):
        return [value]
    return value


def _arm_of(name: str) -> str | None:
    """Return 'left'/'right' for a bimanual joint name, else None."""
    for side in ("left", "right"):
        if name.startswith(f"{side}_"):
            return side
    return None


def _plot_joints(df, action_names: list[str], title: str, png_path: Path) -> None:
    """One column per arm, one row per joint, so the two arms line up visually.

    A flat 2-column grid (what this used to do) puts left_wrist_flex next to
    left_wrist_roll, which tells you nothing. For a bimanual robot the
    interesting comparison is left vs right at the same joint, so that is the
    axis we put side by side. Falls back to the old flat grid for datasets whose
    joints are not left_/right_ prefixed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left = [n for n in action_names if _arm_of(n) == "left"]
    right = [n for n in action_names if _arm_of(n) == "right"]
    bimanual = bool(left) and bool(right) and len(left) + len(right) == len(action_names)

    if bimanual:
        columns = [("left arm", left), ("right arm", right)]
    else:
        # Fallback: chunk the joints into two columns of equal-ish height.
        half = (len(action_names) + 1) // 2
        columns = [("", action_names[:half]), ("", action_names[half:])]

    n_rows = max(len(col) for _, col in columns)
    n_cols = len(columns)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 2.0 * n_rows), sharex=True, squeeze=False,
    )

    first_ax = None
    for c, (col_title, names) in enumerate(columns):
        for r in range(n_rows):
            ax = axes[r][c]
            if r >= len(names):
                ax.set_visible(False)
                continue
            name = names[r]
            ax.plot(df["t_s"], df[f"gt_{name}"], label="ground truth", linewidth=1.2)
            ax.plot(df["t_s"], df[f"pred_{name}"], label="predicted", linewidth=1.0, alpha=0.85)
            # The column header already names the arm, so drop the redundant
            # left_/right_ prefix from each panel and keep the joint legible.
            label = name[len(_arm_of(name)) + 1:] if bimanual else name
            ax.set_title(label, fontsize=9)
            # Grippers are a normalized position, not an angle. Test the SUFFIX:
            # with bimanual names the joint is `left_gripper`, so the old
            # `name != "gripper"` test never matched and BOTH grippers were
            # mislabelled "deg".
            ax.set_ylabel("deg" if not name.endswith("gripper") else "pos")
            ax.grid(alpha=0.25)
            if first_ax is None:
                first_ax = ax
        if names:
            if col_title:
                # Column header sits above the first panel of the column.
                axes[0][c].annotate(
                    col_title, xy=(0.5, 1.35), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                )
            axes[len(names) - 1][c].set_xlabel("time (s)")

    if first_ax is not None:
        first_ax.legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-repo", default="lerobot/pi05_base")
    p.add_argument("--dataset-repo", required=True,
                   help="HF repo id of the LeRobotDataset to evaluate against")
    p.add_argument("--episode", type=int, default=0, help="which episode index to replay")
    p.add_argument("--output-dir", default=None,
                   help="where to write outputs (default: outputs/eval/<timestamp>_<policy>_<dataset>)")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="sets config.dtype; π0.5 keeps its vision path in float32 regardless. "
                        "float16 is not supported by π0.5.")
    p.add_argument("--from-policy", dest="from_policy", action="store_true",
                   help="MODE B: keep the checkpoint's own input_features (camera key names "
                        "and state width), like lerobot-train's --policy.path. You will "
                        "almost certainly need --rename with this. Default is MODE A: take "
                        "input_features from the dataset, like --policy.pretrained_path.")
    p.add_argument("--rename", action="append", default=[], metavar="DATASET_KEY=POLICY_KEY",
                   help="remap observation keys, dataset name -> policy name (repeatable). "
                        "Only meaningful with --from-policy; in the default mode the policy "
                        "already expects your dataset's names and renaming will break it. "
                        "Equivalent to lerobot-train's --rename_map. "
                        "e.g. --rename observation.images.base=observation.images.base_0_rgb")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap frames per episode (default: full episode)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — this script needs a GPU.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    rename = _parse_rename(args.rename)
    if rename and not args.from_policy:
        print("WARNING: --rename without --from-policy. In the default mode the policy's "
              "input_features come from your dataset, so it already expects your key names "
              "and this map will hide them. See the module docstring.", file=sys.stderr)

    # --- output dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    slug = f"{args.policy_repo.split('/')[-1]}__{args.dataset_repo.replace('/', '_')}__ep{args.episode}"
    out_dir = Path(args.output_dir or f"outputs/eval/{timestamp}_{slug}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    # --- dataset
    print(f"loading dataset: {args.dataset_repo}")
    dataset = LeRobotDataset(args.dataset_repo, episodes=[args.episode])
    print(f"  episode {args.episode}: {dataset.num_frames} frames @ {dataset.fps} fps")
    action_names: list[str] = list(dataset.features["action"].get("names") or [])
    if not action_names:
        # Fall back to indexed names so the CSV stays readable.
        action_dim = dataset.features["action"]["shape"][-1]
        action_names = [f"j{i}" for i in range(action_dim)]
    print(f"  action joints ({len(action_names)}): {action_names}")

    # --- policy config: this is where MODE A / MODE B is decided.
    print(f"loading policy: {args.policy_repo}")
    config = PreTrainedConfig.from_pretrained(args.policy_repo)
    config.device = device.type
    if hasattr(config, "dtype"):
        # π0.5 / π0 read config.dtype at construction and apply their own
        # selective cast. Do NOT `.to(dtype=...)` the policy afterwards — that
        # would drag the deliberately-float32 vision tower down with it.
        config.dtype = args.dtype

    ds_features = dataset_to_policy_features(dataset.meta.features)
    # output_features always comes from the dataset — this is what make_policy
    # does, and it is what makes π0.5 unpad its 32-dim action back to your 12.
    config.output_features = {k: ft for k, ft in ds_features.items() if ft.type is FeatureType.ACTION}
    if not args.from_policy:
        config.input_features = {k: ft for k, ft in ds_features.items()
                                 if k not in config.output_features}
    if hasattr(config, "action_feature_names"):
        config.action_feature_names = list(action_names)

    print(f"  mode: {'B (features from checkpoint)' if args.from_policy else 'A (features from dataset)'}")
    print(f"  policy expects images: "
          f"{[k for k, ft in config.input_features.items() if ft.type is FeatureType.VISUAL]}")
    print(f"  policy action width:   {config.output_features['action'].shape[0]}")

    PolicyCls = _resolve_policy_class(args.policy_repo)
    policy = PolicyCls.from_pretrained(args.policy_repo, config=config).eval()

    # Load the checkpoint's saved processor pipelines, but override the pieces
    # that must follow THIS dataset. Mirrors what scripts/lerobot_train.py does;
    # without the overrides, make_pre_post_processors() ignores dataset_stats
    # whenever a pretrained_path is given and you silently normalize against the
    # checkpoint's pretraining statistics.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.policy_repo,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**config.input_features, **config.output_features},
                "norm_map": config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": rename},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
            },
        },
    )

    # --- rollout
    n_frames = min(dataset.num_frames, args.max_frames) if args.max_frames else dataset.num_frames
    print(f"replaying {n_frames} frames…")

    rows: list[dict] = []
    policy.reset()
    inference_ms_sum = 0.0
    for i in range(n_frames):
        frame = dataset[i]
        # Split observation vs ground-truth action.
        gt_action: torch.Tensor = frame["action"]
        obs = {k: v for k, v in frame.items() if k.startswith("observation.") or k == "task"}
        # If the dataset has no language field but the policy needs one, fall back to
        # the dataset-level single_task (recorded on first push). Better than feeding
        # an empty string, which language-conditioned VLAs handle poorly.
        if "task" not in obs:
            single_task = getattr(dataset.meta, "single_task", None) or ""
            obs["task"] = single_task
        # Batch only. Renaming, normalization, tokenization and the device move
        # all happen inside the preprocessor, in that order.
        obs_batched = {k: _add_batch_dim(v) for k, v in obs.items()}

        obs_batched = preprocessor(obs_batched)
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred_action = policy.select_action(obs_batched)
        inference_ms_sum += (time.perf_counter() - t0) * 1000
        pred_action = postprocessor(pred_action)
        # Drop batch dim, keep first action of the chunk if the policy returns a chunk.
        pred = pred_action[0].detach().to("cpu", dtype=torch.float32).numpy()
        if pred.ndim == 2:  # (chunk, action_dim)
            pred = pred[0]
        gt = gt_action.detach().to("cpu", dtype=torch.float32).numpy()
        if gt.ndim == 2:
            gt = gt[0]

        row = {"frame": i, "t_s": i / dataset.fps}
        for j, name in enumerate(action_names):
            row[f"pred_{name}"] = float(pred[j])
            row[f"gt_{name}"] = float(gt[j])
            row[f"err_{name}"] = float(pred[j] - gt[j])
        rows.append(row)

    print(f"  mean inference: {inference_ms_sum / n_frames:.1f} ms/frame")

    # --- write CSV
    df = pd.DataFrame(rows)
    csv_path = out_dir / "actions.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    # --- per-joint stats (dataset-derived, so this is arm-count agnostic)
    per_joint: list[JointStats] = []
    for name in action_names:
        err = df[f"err_{name}"].to_numpy()
        per_joint.append(JointStats(
            name=name,
            mae=float(np.mean(np.abs(err))),
            rmse=float(np.sqrt(np.mean(err ** 2))),
            max_err=float(np.max(np.abs(err))),
        ))

    summary = {
        "policy_repo": args.policy_repo,
        "dataset_repo": args.dataset_repo,
        "episode": args.episode,
        "n_frames": n_frames,
        "fps": float(dataset.fps),
        "dtype": args.dtype,
        "feature_mode": "policy" if args.from_policy else "dataset",
        "device": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "mean_inference_ms": inference_ms_sum / n_frames,
        "joints": [vars(s) for s in per_joint],
        "global_mae": float(np.mean([s.mae for s in per_joint])),
        "rename": rename,
    }
    # Per-arm rollup: the number you actually compare between runs when one arm
    # is doing the work and the other is parked.
    for side in ("left", "right"):
        arm = [s.mae for s in per_joint if _arm_of(s.name) == side]
        if arm:
            summary[f"{side}_arm_mae"] = float(np.mean(arm))
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    print(f"  global mean-absolute-error: {summary['global_mae']:.3f}")
    for side in ("left", "right"):
        if f"{side}_arm_mae" in summary:
            print(f"  {side} arm MAE: {summary[f'{side}_arm_mae']:.3f}")

    # --- plot joints (matplotlib import deferred so a CSV-only run on a server without
    # X11 / Agg-friendly install doesn't fail before writing the data)
    try:
        png_path = out_dir / "joints.png"
        _plot_joints(
            df,
            action_names,
            f"{args.policy_repo} on {args.dataset_repo} · ep {args.episode}",
            png_path,
        )
        print(f"wrote {png_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
