#!/usr/bin/env python
"""Offline policy evaluation: replay a recorded dataset through a policy.

For each frame of one recorded episode, this script:
  1. feeds the observation (state + camera images) to the policy,
  2. collects the predicted action,
  3. compares it against the action the human teleoperator actually used.

Outputs (under --output-dir):
  - actions.csv       per-frame predicted vs ground-truth, every joint
  - summary.json      per-joint MAE / RMSE + meta (policy, dataset, episode, gpu)
  - joints.png        matplotlib figure: 6 joint traces, pred + ground-truth

This is NOT a substitute for closed-loop evaluation on the real robot —
the policy never gets to see how the scene responds to its actions. But it's
a quick, hardware-free way to answer "would this policy have done something
reasonable?" before you commit time to a live deployment.

Usage:
    # Use the default pi05_base against your dataset, eval episode 0:
    python scripts/runpod/replay_eval.py \\
        --dataset-repo $HF_USER/so101_pick_red_cube

    # Pick a specific policy + episode:
    python scripts/runpod/replay_eval.py \\
        --dataset-repo $HF_USER/so101_pick_red_cube \\
        --policy-repo lerobot/pi05_base \\
        --episode 3

    # If the policy's camera key names don't match your dataset
    # (e.g. policy wants observation.images.cam0 but dataset has observation.images.base):
    python scripts/runpod/replay_eval.py \\
        --dataset-repo $HF_USER/so101_pick_red_cube \\
        --rename observation.images.base=observation.images.cam0 \\
        --rename observation.images.wrist=observation.images.cam1
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
            raise SystemExit(f"--rename expects key=value, got {v!r}")
        k, _, dst = v.partition("=")
        out[k.strip()] = dst.strip()
    return out


def _rename_keys(obs: dict, rename: dict[str, str]) -> dict:
    if not rename:
        return obs
    return {rename.get(k, k): v for k, v in obs.items()}


@dataclass
class JointStats:
    name: str
    mae: float
    rmse: float
    max_err: float


def _to_device_dtype(value, device, dtype):
    """Move a tensor to (device, dtype) for floats; ints/strings pass through."""
    if isinstance(value, torch.Tensor):
        if value.dtype.is_floating_point:
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    return value


def _add_batch_dim(value):
    """LeRobotDataset returns single frames; policies expect batched input."""
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    if isinstance(value, str):
        return [value]
    return value


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-repo", default="lerobot/pi05_base")
    p.add_argument("--dataset-repo", required=True,
                   help="HF repo id of the LeRobotDataset to evaluate against")
    p.add_argument("--episode", type=int, default=0, help="which episode index to replay")
    p.add_argument("--output-dir", default=None,
                   help="where to write outputs (default: outputs/eval/<timestamp>_<policy>_<dataset>)")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--rename", action="append", default=[], metavar="DATASET_KEY=POLICY_KEY",
                   help="remap observation keys from dataset names to what the policy expects "
                        "(repeatable). e.g. --rename observation.images.base=observation.images.cam0")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap frames per episode (default: full episode)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — this script needs a GPU.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    rename = _parse_rename(args.rename)

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
    print(f"  action joints: {action_names}")

    # --- policy
    print(f"loading policy: {args.policy_repo}")
    PolicyCls = _resolve_policy_class(args.policy_repo)
    policy = PolicyCls.from_pretrained(args.policy_repo).to(device=device, dtype=dtype).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_repo,
        dataset_stats=dataset.meta.stats,
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
        obs = _rename_keys(obs, rename)
        # Batch + move.
        obs_batched = {k: _add_batch_dim(_to_device_dtype(v, device, dtype)) for k, v in obs.items()}

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

    # --- per-joint stats
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
        "device": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "mean_inference_ms": inference_ms_sum / n_frames,
        "joints": [vars(s) for s in per_joint],
        "global_mae": float(np.mean([s.mae for s in per_joint])),
        "rename": rename,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    print(f"  global mean-absolute-error: {summary['global_mae']:.3f}")

    # --- plot joints (matplotlib import deferred so a CSV-only run on a server without
    # X11 / Agg-friendly install doesn't fail before writing the data)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(action_names)
        cols = 2
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(11, 2.3 * rows_n), sharex=True)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, name in zip(axes_flat, action_names):
            ax.plot(df["t_s"], df[f"gt_{name}"], label="ground truth", linewidth=1.2)
            ax.plot(df["t_s"], df[f"pred_{name}"], label="predicted",   linewidth=1.0, alpha=0.85)
            ax.set_title(name, fontsize=10)
            ax.set_ylabel("deg" if name != "gripper" else "pos")
            ax.grid(alpha=0.25)
        for ax in axes_flat[n:]:
            ax.set_visible(False)
        axes_flat[0].legend(loc="best", fontsize=8)
        axes_flat[-1].set_xlabel("time (s)") if n > 1 else None
        fig.suptitle(f"{args.policy_repo} on {args.dataset_repo} · ep {args.episode}", fontsize=11)
        fig.tight_layout()
        png_path = out_dir / "joints.png"
        fig.savefig(png_path, dpi=110)
        print(f"wrote {png_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
