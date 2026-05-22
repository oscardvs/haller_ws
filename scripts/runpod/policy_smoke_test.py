#!/usr/bin/env python
"""Smoke test: load a generalist policy checkpoint and run one inference step.

Validates that:
  - the model downloads from the Hub,
  - it loads onto CUDA,
  - it accepts a plausible synthetic SO-101-shaped observation,
  - it produces an action of the expected dimensionality.

Failing here means there's no point trying replay_eval.py — fix the env first.

Usage:
    python scripts/runpod/policy_smoke_test.py                       # default: lerobot/pi05_base
    python scripts/runpod/policy_smoke_test.py --policy-repo lerobot/pi0_base
    python scripts/runpod/policy_smoke_test.py --dtype float32       # if your GPU lacks bfloat16
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _resolve_policy_class(policy_repo: str):
    """Pick the right policy class for a known generalist checkpoint.

    Hardcoding the mapping is fine for a smoke test — the goal is to fail fast
    on a clearly-wrong checkpoint name. For replay_eval.py we use the
    config-driven factory instead.
    """
    name = policy_repo.lower()
    if "pi05" in name:
        from lerobot.policies.pi05 import PI05Policy
        return PI05Policy
    if "pi0_fast" in name or "pi0fast" in name:
        from lerobot.policies.pi0_fast import PI0FastPolicy
        return PI0FastPolicy
    if "pi0" in name:
        from lerobot.policies.pi0 import PI0Policy
        return PI0Policy
    if "smolvla" in name:
        from lerobot.policies.smolvla import SmolVLAPolicy
        return SmolVLAPolicy
    raise SystemExit(
        f"don't know which policy class to use for {policy_repo!r} — "
        "extend _resolve_policy_class() or use replay_eval.py (config-driven)."
    )


def _make_synthetic_batch(action_dim: int, num_images: int, dtype: torch.dtype, device: torch.device):
    """Build an observation batch shaped like a typical SO-101 inference call.

    Real observations come from `lerobot.processor.make_default_processors`
    applied to a robot's `get_observation()` — see hmi/backend/haller_hmi/arm.py
    for the production path. Here we mimic the *shape* with random data so we
    can confirm forward + decode without needing a real robot.
    """
    batch = {
        "observation.state": torch.randn(1, action_dim, dtype=dtype, device=device),
        "task": ["pick up the red cube"],
    }
    for i in range(num_images):
        # H, W match the default lerobot expectation; the policy will resize internally.
        batch[f"observation.images.cam{i}"] = torch.rand(
            1, 3, 224, 224, dtype=dtype, device=device,
        )
    return batch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-repo", default="lerobot/pi05_base")
    p.add_argument("--num-images", type=int, default=2,
                   help="how many synthetic camera streams to send the policy")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — this script needs a GPU.", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading {args.policy_repo} (this can take 3–10 min on first run)…")
    t0 = time.perf_counter()
    PolicyCls = _resolve_policy_class(args.policy_repo)
    policy = PolicyCls.from_pretrained(args.policy_repo).to(device=device, dtype=dtype).eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    free, total = torch.cuda.mem_get_info()
    print(f"  vram after load: {(total - free) / 2**30:.2f} GiB used / {total / 2**30:.1f} GiB total")

    batch = _make_synthetic_batch(
        action_dim=len(SO101_JOINTS),
        num_images=args.num_images,
        dtype=dtype,
        device=device,
    )

    print("Running one inference step on a synthetic batch…")
    t0 = time.perf_counter()
    with torch.inference_mode():
        action = policy.select_action(batch)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  action shape: {tuple(action.shape)}  dtype: {action.dtype}")
    print(f"  inference:    {elapsed_ms:.1f} ms")

    # A reasonable VLA emits a chunk per call; we just confirm the *last* dim
    # matches the joint count. Different policies use different chunk lengths.
    assert action.shape[-1] == len(SO101_JOINTS), (
        f"unexpected action_dim {action.shape[-1]} — policy expects something "
        f"other than 6-DOF SO-101? Likely needs a feature-rename map."
    )
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
