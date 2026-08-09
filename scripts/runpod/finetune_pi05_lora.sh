#!/usr/bin/env bash
# scripts/runpod/finetune_pi05_lora.sh — LoRA-finetune pi05_base on your dataset.
#
# Wraps `lerobot-train` with the parameters that matter for a first finetune:
# the pi05_base starting point, a fresh LoRA adapter via the top-level --peft.*
# block, gradient checkpointing, and bfloat16.
#
# --- MEMORY, HONESTLY -------------------------------------------------------
# This used to claim "sized for a 24 GB GPU". That was optimistic. Known numbers:
#
#   * openpi's own README (the upstream π0/π0.5 implementation lerobot ports)
#     gives:  inference > 8 GB · LoRA finetune > 22.5 GB · full finetune > 70 GB.
#     So a 24 GB card (RTX 4090 / A10G) has ~1.5 GB of headroom for LoRA. That
#     is a floor, not a comfortable target.
#   * LeRobot's own π0/π0.5 guide (huggingface.co/docs/lerobot/en/pi0) trains at
#     --batch_size=32 and never mentions LoRA or PEFT at all. The LoRA path used
#     here is a lerobot *library* feature, not something the guide walks through.
#   * openpi's 22.5 GB figure is for its own JAX stack with 1 camera at batch 1.
#     Haller RECORDS 3 cameras (top / left_wrist / right_wrist) — deliberately
#     matching pi05's own base_0_rgb + 2-wrist pretraining slot count. The sim
#     RENDERS 5, but the extra two are operator views and are never recorded, so
#     training cost is set by the 3 in the dataset. Each camera adds a full
#     SigLIP prefix of image tokens to every forward pass.
#
# Read the default below as: 24 GB = LoRA + batch 1–2 + gradient checkpointing
# at 3 cameras. If you OOM, in order of least to most damaging:
#
#   1. BATCH_SIZE=1                      (there is no gradient-accumulation flag
#                                         in lerobot-train 0.5.1 — checked; the
#                                         old comment here promising one was wrong)
#   2. FREEZE_VISION_ENCODER=true        --policy.freeze_vision_encoder
#   3. TRAIN_EXPERT_ONLY=true            --policy.train_expert_only  (freezes the
#                                         whole VLM; trains only the action expert
#                                         + projections. Biggest single saving,
#                                         and the lever LeRobot's own guide names.)
#   4. Fewer cameras in the dataset, or rent an A100/H100 for an evening.
#
# --- THE PEFT SURFACE (this is what the old version got wrong) --------------
# `--policy.peft_config.use_peft=true` DOES NOT EXIST in lerobot 0.5.1. Argument
# parsing dies before a single byte is downloaded. The real surface is two things:
#
#   --policy.use_peft            (configs/policies.py)  ... ADAPTER-RESUME ONLY
#   --peft.<field>               (configs/train.py -> configs/default.py:PeftConfig)
#
# and they are NOT interchangeable:
#
#   * pretrained_path + use_peft  -> policies/factory.py routes to
#     `PeftConfig.from_pretrained(cfg.pretrained_path)`, i.e. it reads
#     adapter_config.json out of the path and treats it as an ALREADY-TRAINED
#     adapter to resume. `lerobot/pi05_base` has no adapter_config.json, so a
#     first LoRA run down this path fails.
#   * pretrained_path + --peft.*  -> factory loads the base policy normally, then
#     scripts/lerobot_train.py calls `policy.wrap_with_peft(...)` to attach a
#     FRESH adapter. This is the path we want, and the one below.
#
# We deliberately do NOT pass --policy.use_peft. lerobot sets it for you inside
# wrap_with_peft() so the checkpoint saves correctly.
#
# π0.5 supplies its own default LoRA target_modules (gemma_expert q/v projections
# plus state/action projections), so --peft.r + --peft.method_type is enough.
#
# --- BIMANUAL -----------------------------------------------------------------
# Haller is two SO-101 arms: 12-dim state/action (left arm 6 joints, then right
# arm 6), degrees, 3 recorded RGB channels. π0.5 pads state/action to 32 dims
# internally (max_state_dim/max_action_dim) and unpads its output back to your
# dataset's real width. 12-dim therefore needs NO architectural change, no config
# override, and no rename map. Do not go looking for a "bimanual" flag.
#
# Nor do you need --rename_map here: with `--policy.type=pi05
# --policy.pretrained_path=...` the policy's input_features start empty and are
# filled from YOUR dataset's keys. (--rename_map is only for the --policy.path
# form, which inherits pi05_base's own camera names. See replay_eval.py.)
#
# Usage:
#     scripts/runpod/finetune_pi05_lora.sh <dataset_repo> [steps]
#
# Examples:
#     scripts/runpod/finetune_pi05_lora.sh osrdevos/haller_bimanual_pick_red_cube
#     scripts/runpod/finetune_pi05_lora.sh osrdevos/haller_bimanual_pick_red_cube 5000
#     TRAIN_EXPERT_ONLY=true BATCH_SIZE=1 scripts/runpod/finetune_pi05_lora.sh osrdevos/haller_bimanual_handover
#
# Prerequisites:
#     - `lerobot[pi,peft]` installed (setup.sh does this; plain `lerobot[pi]` has NO peft)
#     - google/paligemma-3b-pt-224 licence accepted on your HF account (gated tokenizer)
#     - dataset meta/stats.json contains q01/q99 (π0.5 normalizes with quantiles):
#         lerobot-edit-dataset --repo_id <ds> --operation.type recompute_stats --push_to_hub true
#
# Tunables (env vars):
#     HF_USER                 HF user/org to push the trained policy under.
#                             Default: resolved from `hf auth whoami`.
#     POLICY_REPO             Where the trained policy is uploaded.
#                             Default: ${HF_USER}/pi05_<slug>_lora
#     BATCH_SIZE              Default: 2. See the memory notes above.
#     LR                      Learning rate. Default: 5e-5 (typical for LoRA).
#     LORA_RANK               --peft.r. Default: 16 (lerobot's PeftConfig default).
#     TRAIN_EXPERT_ONLY       true/false. Default: false. Freeze the whole VLM.
#     FREEZE_VISION_ENCODER   true/false. Default: false. Freeze just the vision tower.
#     RESUME_ADAPTER          Repo/path of an EXISTING adapter to continue training.
#                             Switches to the --policy.use_peft route described above.
set -euo pipefail

if [ $# -lt 1 ]; then
    cat <<EOF
Usage: $0 <dataset_repo> [steps]
   e.g. $0 osrdevos/haller_bimanual_pick_red_cube 5000
EOF
    exit 64
fi

DATASET_REPO="$1"
STEPS="${2:-5000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-5}"
LORA_RANK="${LORA_RANK:-16}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
RESUME_ADAPTER="${RESUME_ADAPTER:-}"

# Resolve HF user from token if not pinned.
if [ -z "${HF_USER:-}" ]; then
    if HF_USER=$(NO_COLOR=1 hf auth whoami 2>/dev/null | awk -F': *' 'NR==1 {print $2}') && [ -n "$HF_USER" ]; then
        :
    else
        echo "Cannot resolve HF_USER. Run: hf auth login   (or pass HF_USER=...)" >&2
        exit 1
    fi
fi

# Fail early and legibly if peft is missing, rather than 10 minutes into a
# model download inside policies/pretrained.py's `from peft import get_peft_model`.
if ! python -c "import peft" 2>/dev/null; then
    echo "peft is not installed — 'lerobot[pi]' does not pull it in." >&2
    echo "Fix: pip install 'lerobot[pi,peft]>=0.5,<0.6'   (or re-run setup.sh)" >&2
    exit 1
fi

# Derive a slug from the dataset name for the output repo.
dataset_slug=$(printf "%s" "$DATASET_REPO" | sed -E 's|^[^/]+/||; s/[^a-zA-Z0-9]+/_/g')
POLICY_REPO="${POLICY_REPO:-${HF_USER}/pi05_${dataset_slug}_lora}"
OUTPUT_DIR="outputs/train/pi05_${dataset_slug}_lora"

# Two mutually exclusive PEFT routes — see the header.
if [ -n "$RESUME_ADAPTER" ]; then
    BASE_DESC="$RESUME_ADAPTER (resuming an existing adapter)"
    peft_args=(
        --policy.pretrained_path="$RESUME_ADAPTER"
        --policy.use_peft=true
    )
else
    BASE_DESC="lerobot/pi05_base (fresh LoRA adapter)"
    peft_args=(
        --policy.pretrained_path=lerobot/pi05_base
        --peft.method_type=LORA
        --peft.r="$LORA_RANK"
    )
fi

cat <<EOF
=== finetune pi05_base (LoRA) ===
  dataset:  $DATASET_REPO
  base:     $BASE_DESC
  steps:    $STEPS
  batch:    $BATCH_SIZE  (no gradient-accumulation flag exists in lerobot-train 0.5.1)
  lr:       $LR
  lora r:   $LORA_RANK
  freeze:   train_expert_only=$TRAIN_EXPERT_ONLY  freeze_vision_encoder=$FREEZE_VISION_ENCODER
  output:   $OUTPUT_DIR
  push to:  $POLICY_REPO

  Reminder: 24 GB is the *floor* for π0.5 LoRA (openpi: >22.5 GB) and that
  figure predates 5-camera bimanual input. Expect to tune the levers above.
EOF
ans=""
read -r -p "Proceed? [y/N] " ans || true
[ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "Aborted."; exit 0; }

exec lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=pi05 \
    "${peft_args[@]}" \
    --policy.repo_id="$POLICY_REPO" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=true \
    --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
    --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER" \
    --policy.optimizer_lr="$LR" \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --policy.scheduler_decay_steps="$STEPS" \
    --save_freq=1000 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="pi05_${dataset_slug}_lora" \
    --wandb.enable=false
