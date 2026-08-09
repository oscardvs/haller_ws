#!/usr/bin/env python
"""Smoke test: load a generalist policy checkpoint and run one inference step.

Validates that:
  - the model downloads from the Hub (including the GATED PaliGemma tokenizer),
  - it loads onto CUDA at the precision the policy actually manages itself,
  - the full preprocessor pipeline accepts a Haller-shaped observation,
  - it produces an action of the expected dimensionality,
  - and — the open question this script exists to answer — how many camera
    channels π0.5 will actually swallow (see THE 5-CAMERA QUESTION below).

Failing here means there's no point trying replay_eval.py — fix the env first.

--- HALLER IS BIMANUAL -------------------------------------------------------
Two SO-101 arms => a 12-dim state/action vector: left arm's 6 joints, then the
right arm's 6, recorded in DEGREES. π0.5 pads state and action to 32 dims
internally (max_state_dim / max_action_dim in configuration_pi05.py) and unpads
its output back to your dataset's real width in predict_action_chunk(). So 12
dims needs NO architectural change and NO config surgery. This script proves it
by declaring 12-dim features and checking that 12 dims come back out.

--- WHY THIS GOES THROUGH THE PREPROCESSOR ----------------------------------
An earlier version called `policy.select_action(batch)` with a raw `"task"`
string. That can never work: predict_action_chunk() reads
`batch["observation.language.tokens"]` and `["observation.language.attention_mask"]`,
which only exist after TokenizerProcessorStep has run. For π0.5 the pipeline
also folds the *discretized state* into the text prompt
(Pi05PrepareStateTokenizerProcessorStep) — bypass it and the model never sees
the robot's pose at all. So: always build the observation via the processors.

Two consequences worth knowing:
  * The state tensor must stay float32 through the pipeline. The π0.5 prompt
    step calls `state.cpu().numpy()`, and torch refuses to hand bfloat16 to
    numpy ("Got unsupported ScalarType BFloat16"). We therefore never cast the
    observation; precision is a *model* setting (below), not an input setting.
  * π0.5 manages its own mixed precision: config.dtype="bfloat16" casts most of
    the net but deliberately keeps the vision tower, projector and layernorms in
    float32. Calling `.to(dtype=torch.bfloat16)` on the policy from outside
    flattens that distinction. Use --dtype, which sets config.dtype. (float16 is
    not offered: PaliGemmaWithExpertModel only accepts "bfloat16" or "float32".)

--- CAMERA KEYS ARE NOT cam0/cam1 -------------------------------------------
There is no `observation.images.cam0` convention anywhere in lerobot. Which keys
a policy expects depends on how you loaded it:
  * `PI05Policy.from_pretrained("lerobot/pi05_base")` bare  -> the keys baked
    into that checkpoint's config (base_0_rgb / left_wrist_0_rgb /
    right_wrist_0_rgb at the time of writing — this script prints what it
    actually found, so trust the output over this comment).
  * training with `--policy.type=pi05 --policy.pretrained_path=...` -> the keys
    come from YOUR dataset (make_policy fills empty input_features from ds_meta).
Sending a key the config doesn't declare is silently ignored, and if NONE of the
declared keys are present you get
`ValueError: All image features are missing from the batch`.

--- THE 5-CAMERA QUESTION ---------------------------------------------------
π0.5 was pretrained with 3 camera slots. Haller can produce 5. Is that legal?

Architecturally, yes-by-construction: PI05Pytorch.embed_prefix() just iterates
`zip(images, img_masks)` and pushes every view through ONE shared SigLIP tower,
concatenating the resulting embeddings into the attention prefix. There is no
per-camera parameter and no hard-coded 3 anywhere — the only cost of another
camera is a longer prefix (and the VRAM that implies). Missing declared cameras
are padded with -1 images and a zero mask, so declaring more than you send costs
you compute without adding information.

Statistically, 5 is out-of-distribution: the pretraining mixture used 3 views,
and no published 5-camera π0.5 run was found. "It runs" is NOT "it works".

So this script actually TRIES it. By default it runs the 3-camera case and then
probes 5, and prints a verdict. Anything it reports is a claim about *shape and
memory*, never about policy quality — only replay_eval.py and a real robot can
speak to that.

Usage:
    python scripts/runpod/policy_smoke_test.py                       # 3 cams, then probe 5
    python scripts/runpod/policy_smoke_test.py --max-cameras 0       # skip the probe
    python scripts/runpod/policy_smoke_test.py --num-images 5        # 5 as the main pass
    python scripts/runpod/policy_smoke_test.py --bare-features       # use the ckpt's own dims
    python scripts/runpod/policy_smoke_test.py --policy-repo lerobot/pi0_base
    python scripts/runpod/policy_smoke_test.py --dtype float32       # if bfloat16 misbehaves
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


# Haller = two SO-101 arms. Order matters: it is the order the recorder writes
# into the 12-dim state/action vector, left arm first.
SO101_JOINT_SUFFIXES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
HALLER_JOINTS = [f"{side}_{j}" for side in ("left", "right") for j in SO101_JOINT_SUFFIXES]

# π0.5's internal pad width (configuration_pi05.py: max_state_dim / max_action_dim).
# A checkpoint loaded without dataset-derived features reports this as its action
# width, which is why the final assert accepts it alongside the real 12.
PI05_PAD_DIM = 32

# π0.5's pretraining used this many camera slots. Not a hard limit — see the
# module docstring — but it is the in-distribution count.
PI05_PRETRAIN_CAMERAS = 3


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


def _synthetic_quantile_stats(state_dim: int, action_dim: int) -> dict:
    """Fabricate the normalization stats the preprocessor demands.

    π0.5 normalizes STATE and ACTION with QUANTILES, i.e. it wants q01/q99 in
    meta/stats.json and raises if they're absent (that's the error real Haller
    recordings hit until you run `lerobot-edit-dataset --operation.type
    recompute_stats`). A smoke test has no dataset, so we invent a plausible
    envelope: Haller records DEGREES, and no SO-101 joint leaves ±180.

    These numbers are FAKE. They are enough to exercise the pipeline's shapes
    and the tokenizer's 256-bin state discretization; they say nothing about
    whether the resulting action is sensible. That's replay_eval.py's job.
    """
    def envelope(dim: int) -> dict:
        lo = torch.full((dim,), -180.0)
        hi = torch.full((dim,), 180.0)
        return {
            "q01": lo,
            "q99": hi,
            "min": lo,
            "max": hi,
            "mean": torch.zeros(dim),
            "std": torch.full((dim,), 90.0),
        }

    from lerobot.utils.constants import ACTION, OBS_STATE
    return {OBS_STATE: envelope(state_dim), ACTION: envelope(action_dim)}


def _set_camera_count(config, num_images: int, height: int, width: int) -> list[str]:
    """Force the policy config to declare exactly `num_images` camera features.

    Legitimate because π0.5 has no per-camera parameters — embed_prefix() runs
    every view through the same SigLIP tower (see the module docstring). Growing
    or shrinking this list changes only the attention prefix length, never a
    weight shape, so the loaded checkpoint stays valid.

    Returns the camera keys now declared, in order.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import OBS_IMAGES

    existing = [k for k, ft in config.input_features.items() if ft.type is FeatureType.VISUAL]
    keys = list(existing[:num_images])
    # Synthesize extra slots if we want more views than the checkpoint declares.
    # The names are arbitrary — nothing keys off them but our own batch.
    for i in range(len(keys), num_images):
        keys.append(f"{OBS_IMAGES}.smoke_cam_{i}")

    for k in existing:
        config.input_features.pop(k, None)
    for k in keys:
        config.input_features[k] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
    return keys


def _make_synthetic_observation(
    camera_keys: list[str], state_dim: int, height: int, width: int, task: str,
) -> dict:
    """Build a raw, UNBATCHED, float32 observation — processor input, not model input.

    Real observations come from `lerobot.processor.make_default_processors`
    applied to a robot's `get_observation()` — see hmi/backend/haller_hmi/arm.py
    for the production path. Here we mimic the *shape* with random data.

    Deliberately float32 and unbatched: the preprocessor's
    AddBatchDimensionProcessorStep adds the leading dim (it is idempotent, so a
    pre-batched tensor also survives), and DeviceProcessorStep does the device
    move. Casting to bfloat16 here would crash π0.5's state-discretization step.
    """
    from lerobot.utils.constants import OBS_STATE

    obs = {
        OBS_STATE: torch.randn(state_dim, dtype=torch.float32),
        "task": task,
    }
    for key in camera_keys:
        # lerobot images are float32 in [0, 1], CHW. The policy resizes and
        # rescales to PaliGemma's [-1, 1] internally.
        obs[key] = torch.rand(3, height, width, dtype=torch.float32)
    return obs


def _run_once(policy, config, num_images: int, state_dim: int, task: str, stats: dict):
    """Set up `num_images` cameras and run one inference step.

    Returns (action, elapsed_ms, peak_vram_gib, camera_keys).
    """
    from lerobot.policies.factory import make_pre_post_processors

    height, width = config.image_resolution if hasattr(config, "image_resolution") else (224, 224)
    camera_keys = _set_camera_count(config, num_images, height, width)

    # For a PI05Config this dispatches to make_pi05_pre_post_processors, the
    # factory that installs TokenizerProcessorStep — the step that creates
    # observation.language.tokens / .attention_mask. No pretrained_path: we want
    # processors built from THIS config with OUR stats, not the checkpoint's
    # saved pipeline (which would silently ignore the stats we pass).
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=stats,
    )

    obs = _make_synthetic_observation(camera_keys, state_dim, height, width, task)

    torch.cuda.reset_peak_memory_stats()
    policy.reset()
    t0 = time.perf_counter()
    with torch.inference_mode():
        action = policy.select_action(preprocessor(obs))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    action = postprocessor(action)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return action, elapsed_ms, peak_gib, camera_keys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-repo", default="lerobot/pi05_base")
    p.add_argument("--num-images", type=int, default=PI05_PRETRAIN_CAMERAS,
                   help="camera streams for the main pass. Default 3 = π0.5's pretraining "
                        "slot count, i.e. the in-distribution choice.")
    p.add_argument("--max-cameras", type=int, default=5,
                   help="after the main pass, also probe this many cameras and print a "
                        "verdict. Haller can produce 5 and nobody has published that "
                        "configuration. 0 disables the probe.")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="sets config.dtype (π0.5 keeps the vision path in float32 "
                        "regardless). float16 is not supported by π0.5.")
    p.add_argument("--bare-features", action="store_true",
                   help="do NOT reshape the checkpoint's state/action features to Haller's "
                        "12 dims; use whatever the checkpoint declares (bare pi05_base "
                        "reports its 32-dim pad width).")
    p.add_argument("--task", default="pick up the red cube with the left arm")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — this script needs a GPU.", file=sys.stderr)
        return 1

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE

    print(f"Loading {args.policy_repo} (this can take 3–10 min on first run)…")
    print("  (first run also fetches the GATED google/paligemma-3b-pt-224 tokenizer —")
    print("   a GatedRepoError here means you have not accepted its licence on the Hub)")
    t0 = time.perf_counter()

    # Resolve (and therefore IMPORT) the policy class first. PreTrainedConfig is a
    # draccus ChoiceRegistry: subclasses only register when their configuration
    # module is imported, so calling from_pretrained() on a fresh interpreter
    # fails with "invalid choice: 'pi05'" until something has imported
    # lerobot.policies.pi05. Verified: get_known_choices() is [] before the import.
    PolicyCls = _resolve_policy_class(args.policy_repo)

    config = PreTrainedConfig.from_pretrained(args.policy_repo)
    config.device = "cuda"
    if hasattr(config, "dtype"):
        # π0.5 / π0 read this at construction and apply their own selective cast.
        config.dtype = args.dtype

    if not args.bare_features:
        # Simulate what make_policy() does from a real Haller LeRobotDataset:
        # overwrite the feature dims with the dataset's true widths. 12 in, 12 out.
        config.input_features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE, shape=(len(HALLER_JOINTS),)
        )
        config.output_features[ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(len(HALLER_JOINTS),)
        )
        if hasattr(config, "action_feature_names"):
            config.action_feature_names = list(HALLER_JOINTS)

    policy = PolicyCls.from_pretrained(args.policy_repo, config=config).eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    state_dim = config.input_features[OBS_STATE].shape[0]
    action_dim = config.output_features[ACTION].shape[0]
    print(f"  state dim:  {state_dim}   action dim: {action_dim}")
    print(f"  camera keys declared by the checkpoint: "
          f"{[k for k, ft in config.input_features.items() if ft.type is FeatureType.VISUAL]}")

    free, total = torch.cuda.mem_get_info()
    print(f"  vram after load: {(total - free) / 2**30:.2f} GiB used / {total / 2**30:.1f} GiB total")

    stats = _synthetic_quantile_stats(state_dim, action_dim)

    # --- main pass -----------------------------------------------------------
    print(f"\nRunning one inference step with {args.num_images} camera(s)…")
    action, elapsed_ms, peak_gib, camera_keys = _run_once(
        policy, config, args.num_images, state_dim, args.task, stats
    )
    print(f"  camera keys sent: {camera_keys}")
    print(f"  action shape: {tuple(action.shape)}  dtype: {action.dtype}")
    print(f"  inference:    {elapsed_ms:.1f} ms   peak vram: {peak_gib:.2f} GiB")

    # A VLA emits a chunk per call; select_action() pops one step off that chunk,
    # so we only check the last dim. Two widths are legitimate:
    #   len(HALLER_JOINTS) == 12  -> features were shaped from a real (or, here,
    #                                simulated) bimanual dataset; π0.5 unpadded
    #                                its 32-dim internal action back down.
    #   PI05_PAD_DIM      == 32  -> loaded bare, so output_features[ACTION] is
    #                                still the max_action_dim pad width and there
    #                                is nothing to unpad to.
    # Anything else means the feature plumbing is wrong, not the robot.
    assert action.shape[-1] in (len(HALLER_JOINTS), PI05_PAD_DIM), (
        f"unexpected action_dim {action.shape[-1]} — expected {len(HALLER_JOINTS)} "
        f"(bimanual Haller: {', '.join(HALLER_JOINTS)}) or {PI05_PAD_DIM} "
        f"(π0.5's internal pad width, seen when the checkpoint is loaded without "
        f"dataset-derived features)."
    )

    # --- the 5-camera probe --------------------------------------------------
    if args.max_cameras and args.max_cameras != args.num_images:
        n = args.max_cameras
        print(f"\n--- probing {n} cameras (Haller's maximum; out-of-distribution) ---")
        try:
            action_n, ms_n, peak_n, keys_n = _run_once(
                policy, config, n, state_dim, args.task, stats
            )
            print(f"  action shape: {tuple(action_n.shape)}")
            print(f"  inference:    {ms_n:.1f} ms   peak vram: {peak_n:.2f} GiB")
            print(f"\n  VERDICT ({n} cameras): RUNS.")
            print(f"    Shapes and memory are fine on this GPU: {ms_n:.0f} ms/step, "
                  f"{peak_n:.2f} GiB peak vs {peak_gib:.2f} GiB at {args.num_images} cameras.")
            print(f"    This is a SHAPE result only. π0.5 pretrained with "
                  f"{PI05_PRETRAIN_CAMERAS} views, so {n} is out-of-distribution and "
                  f"no published {n}-camera run was found.")
            print("    Whether the policy is any GOOD with 5 views is unanswered here —")
            print("    finetune, then compare replay_eval.py MAE at 3 vs 5 cameras.")
        except torch.cuda.OutOfMemoryError:
            print(f"\n  VERDICT ({n} cameras): OUT OF MEMORY on this GPU.")
            print(f"    Each extra view adds a full SigLIP prefix to every forward pass. "
                  f"{args.num_images} cameras fit in {peak_gib:.2f} GiB peak; {n} did not.")
            print("    Options: a bigger card, or drop to 3 views for training.")
        except Exception as e:  # noqa: BLE001 - we want the verdict, not a traceback
            print(f"\n  VERDICT ({n} cameras): FAILED — {type(e).__name__}: {e}")
            print("    Report this: it would contradict the architectural reading that "
                  "π0.5 is camera-count-flexible.")

    print("\nSMOKE TEST PASSED")
    print("  What this did NOT prove: that the policy's outputs are meaningful. "
          "The state/action stats above are fabricated. Run replay_eval.py against "
          "a real dataset next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
