#!/usr/bin/env bash
# scripts/runpod/finetune_pi05_lora.sh — LoRA-finetune pi05_base on your dataset.
#
# Wraps `lerobot-train` with the parameters that matter for a first finetune:
# the pi05_base starting point, PEFT/LoRA enabled, gradient checkpointing, and
# bfloat16. Sized for a 24 GB GPU (RTX 4090 / A10G); bump --batch-size and
# disable gradient checkpointing on an A100 if you have one.
#
# Usage:
#     scripts/runpod/finetune_pi05_lora.sh <dataset_repo> [steps]
#
# Examples:
#     scripts/runpod/finetune_pi05_lora.sh osrdevos/so101_pick_red_cube
#     scripts/runpod/finetune_pi05_lora.sh osrdevos/so101_pick_red_cube 5000
#
# Tunables (env vars):
#     HF_USER        HF user/org to push the trained policy under.
#                    Default: resolved from `hf auth whoami`.
#     POLICY_REPO    Where the trained policy is uploaded.
#                    Default: ${HF_USER}/pi05_<slug>_lora
#     BATCH_SIZE     Default: 4. Use gradient accumulation for effective batch>=16.
#     LR             Learning rate. Default: 5e-5 (typical for LoRA).
set -eo pipefail

if [ $# -lt 1 ]; then
    cat <<EOF
Usage: $0 <dataset_repo> [steps]
   e.g. $0 osrdevos/so101_pick_red_cube 5000
EOF
    exit 64
fi

DATASET_REPO="$1"
STEPS="${2:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-5e-5}"

# Resolve HF user from token if not pinned.
if [ -z "${HF_USER:-}" ]; then
    if HF_USER=$(NO_COLOR=1 hf auth whoami 2>/dev/null | awk -F': *' 'NR==1 {print $2}') && [ -n "$HF_USER" ]; then
        :
    else
        echo "Cannot resolve HF_USER. Run: hf auth login   (or pass HF_USER=...)" >&2
        exit 1
    fi
fi

# Derive a slug from the dataset name for the output repo.
dataset_slug=$(printf "%s" "$DATASET_REPO" | sed -E 's|^[^/]+/||; s/[^a-zA-Z0-9]+/_/g')
POLICY_REPO="${POLICY_REPO:-${HF_USER}/pi05_${dataset_slug}_lora}"
OUTPUT_DIR="outputs/train/pi05_${dataset_slug}_lora"

cat <<EOF
=== finetune pi05_base (LoRA) ===
  dataset:  $DATASET_REPO
  base:     lerobot/pi05_base
  steps:    $STEPS
  batch:    $BATCH_SIZE  (use gradient_accumulation for effective batch >= 16)
  lr:       $LR
  output:   $OUTPUT_DIR
  push to:  $POLICY_REPO
EOF
read -r -p "Proceed? [y/N] " ans
[ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "Aborted."; exit 0; }

exec lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.repo_id="$POLICY_REPO" \
    --policy.peft_config.use_peft=true \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=true \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --policy.scheduler_decay_steps="$STEPS" \
    --save_freq=1000 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="pi05_${dataset_slug}_lora" \
    --wandb.enable=false
