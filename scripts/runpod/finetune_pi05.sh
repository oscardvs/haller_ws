#!/usr/bin/env bash
# scripts/runpod/finetune_pi05.sh — finetune pi05_base on a Haller dataset.
#
# DEFAULT IS A FULL FINE-TUNE. This script used to be finetune_pi05_lora.sh and
# to default to `--peft.method_type=LORA --peft.r=16`. That default was wrong,
# and not marginally:
#
#   lerobot 0.5.1's DEFAULT PEFT TARGET SET FOR pi05 FREEZES THE VISION TOWER
#   AND THE LLM.
#
# `Pi05Policy._get_default_peft_targets()` (modeling_pi05.py:1285) targets only
# `gemma_expert.*.self_attn.(q|v)_proj` plus the state/action projections, with
# an empty `modules_to_save`; `pretrained.py:301` then does
# `for p in self.parameters(): p.requires_grad_(False)` before attaching the
# adapters. So SigLIP and Gemma-2B are frozen solid and only the action
# expert's q/v get low-rank deltas.
#
# That is the configuration measured as catastrophic for adapting to a new
# embodiment. arXiv:2607.10172 (pi0, UR5e, 4 assembly tasks, 200 demos each,
# Average Task Progress over 20 rollouts):
#
#     full fine-tune ................................. 0.76
#     LoRA r=32, SigLIP TRAINABLE .................... 0.74   (p > 0.05 vs FFT)
#     SigLIP frozen .................................. 0.14
#     VLM frozen, action expert only ................. 0.15
#
# Their SVD of the full-FT weight deltas puts SigLIP's MLPs at rank ~597 for
# 95% of the spectral energy — the highest normalised rank usage in the model.
# The visual shift cannot be absorbed at low rank. Haller's case is HARDER than
# that paper's, not easier: the data is MuJoCo renders, so the domain shift
# from web+real-robot pretraining is larger than a new real workspace.
#
# Physical Intelligence say the same thing about their own model, in
# openpi/examples/droid/README_train.md:
#   "We have experimented with LoRA for cheaper finetuning, but haven't found
#    the policies to perform well so far."
# and openpi ships SIX pi05 configs, every one of them a full fine-tune.
#
# The cost difference is ~$20-35 for the whole run (A100 80GB PCIe is
# $1.19/h community / $1.39/h secure, vs $0.34/h for a 4090). This was never a
# budget decision.
#
# MODES
#   full   (default) Full fine-tune. Needs an 80 GB card. ~18 h on an A100.
#   hybrid           openpi's ACTUAL LoRA recipe: LoRA on the LLM + action
#                    expert, vision tower fully trainable. The 0.74 row above.
#                    Should fit a 48 GB L40S. Overrides the broken default
#                    target set explicitly. UNVERIFIED — see the warning it
#                    prints.
#   expert-lora      The old default, kept only so the failure mode is
#                    reproducible and named. Refuses to run without
#                    I_KNOW_THIS_IS_THE_BAD_ONE=1.
#
# SIZING, from the actual artifact: lerobot/pi05_base is 3,616,757,520 params
# stored as F32 = 13.47 GiB. Full-FT static footprint (params + grads + two
# fp32 AdamW moments) is ~43-58 GiB BEFORE activations. 80 GB is tight, not
# roomy: start at batch 16 with gradient checkpointing and raise it only if it
# fits.
#
# DISK. `save_checkpoint()` writes model.safetensors AND
# optimizer_state.safetensors every time, and lerobot 0.5.1 has no checkpoint
# pruning. For a full FT that is ~20-43 GiB per checkpoint. The old
# --save_freq=1000 over 20k steps would write 400-860 GB — harmless for a LoRA
# adapter, a disk bomb here. Hence save_freq 5000, and a 300 GB volume.
#
# LICENSING. pi05_base is `license: gemma`, NOT Apache-2.0. Commercial use is
# permitted, but PUBLISHING the finetuned weights is "Distribution" under the
# Gemma Terms and pulls in notice + pass-through obligations. So POLICY_REPO
# defaults to a PRIVATE repo. See docs/setup/runpod-inference.md.
#
# WHAT THE FREEZE LEVERS ACTUALLY DO
#   --policy.freeze_vision_encoder / --policy.train_expert_only work on the
#   FULL path and are no-ops on any PEFT path (`_set_requires_grad` runs at
#   construction; `wrap_with_peft` freezes everything afterwards). They are not
#   memory levers to reach for: turning either on IS the 0.14/0.15 failure.
#
# USAGE
#     scripts/runpod/finetune_pi05.sh <dataset_repo> [steps]
#     MODE=hybrid scripts/runpod/finetune_pi05.sh osrdevos/haller_insertion
#
# ENV
#     MODE           full | hybrid | expert-lora     (default: full)
#     BATCH_SIZE     default 16 (full/hybrid), 2 (expert-lora)
#     LR             default 2.5e-5 — lerobot's pi05 preset, mirroring openpi's
#                    CosineDecaySchedule peak_lr. The old 5e-5 was "typical for
#                    LoRA" and is too hot for a full FT without warmup.
#     LORA_RANK      default 32 (the rank the paper measured at 0.74)
#     POLICY_REPO    default ${HF_USER}/pi05_<slug>  (PRIVATE)
#     PUSH_PUBLIC=1  make the pushed repo public — read the licensing note first
#     RESUME_ADAPTER resume an existing adapter (uses --policy.use_peft)
#     WANDB_ENABLE   default false. true needs a real API key, offline mode does not help.
#     EVAL_SPLIT     default 0.0 (no held-out eval). Set WITH EVAL_STEPS or it raises.
#     EVAL_STEPS     default 0. Set WITH EVAL_SPLIT.
#     SAVE_FREQ      default 5000. Lower it on short runs, or you will miss the optimum.
#     OUTPUT_DIR     default outputs/train/pi05_<slug>_<mode>
#
# PREREQS
#     - `lerobot[pi,peft]` (setup.sh does this; plain `lerobot[pi]` has NO peft)
#     - `lerobot-edit-dataset --operation.type recompute_stats` on your dataset
#       first. pi05 normalises STATE/ACTION by quantiles (q01/q99), so degrees
#       are fine — but the stats must be YOUR dataset's, not pretraining's.
set -euo pipefail

if [ $# -lt 1 ]; then
    cat <<EOF
Usage: $0 <dataset_repo> [steps]
   e.g. $0 osrdevos/haller_bimanual_insertion 20000
   MODE=hybrid $0 osrdevos/haller_bimanual_insertion
EOF
    exit 64
fi

DATASET_REPO="$1"
MODE="${MODE:-full}"
STEPS="${2:-20000}"
LR="${LR:-2.5e-5}"
LORA_RANK="${LORA_RANK:-32}"
RESUME_ADAPTER="${RESUME_ADAPTER:-}"
# wandb demands an API key even under WANDB_MODE=offline, because lerobot passes
# the mode explicitly and overrides the env. Defaulting this to true is what
# blocked a launch. MetricsTracker still prints losses to the console, which is
# where every loss curve in this project actually came from.
WANDB_ENABLE="${WANDB_ENABLE:-false}"
# HELD-OUT EVAL. Both 2026-08-31 runs set neither of these and produced nothing
# but training losses. In lerobot 0.6.1 `eval_steps > 0` RAISES unless
# `eval_split > 0.0`, so the two are coupled and are validated together below.
EVAL_SPLIT="${EVAL_SPLIT:-0.0}"
EVAL_STEPS="${EVAL_STEPS:-0}"
# 5000 was hardcoded, and on a 6k-step schedule it never wrote a checkpoint near
# the held-out optimum, which lands inside the first epoch.
SAVE_FREQ="${SAVE_FREQ:-5000}"

case "$MODE" in
    full)         BATCH_SIZE="${BATCH_SIZE:-16}" ;;
    hybrid)       BATCH_SIZE="${BATCH_SIZE:-16}" ;;
    expert-lora)  BATCH_SIZE="${BATCH_SIZE:-2}" ;;
    *) echo "unknown MODE=$MODE (expected: full | hybrid | expert-lora)" >&2; exit 64 ;;
esac

if [ "$MODE" = "expert-lora" ] && [ "${I_KNOW_THIS_IS_THE_BAD_ONE:-}" != "1" ]; then
    cat >&2 <<'EOF'
MODE=expert-lora freezes SigLIP and the Gemma LLM (lerobot 0.5.1's default
pi05 PEFT targets). That is the ATP 0.14 configuration in the header — it is
kept for reproducing the failure, not for training a policy.

If you really mean it:  I_KNOW_THIS_IS_THE_BAD_ONE=1 MODE=expert-lora $0 ...
EOF
    exit 64
fi

# Resolve HF user from token if not pinned.
if [ -z "${HF_USER:-}" ]; then
    # hub 1.x prints a two-line block, and the username is NOT on line 1:
    #     ✓ Logged in
    #       user: Oskrt
    # The old `NR==1 {print $2}` read "Logged in" and resolved empty. Match the
    # `user:` line by name, and tolerate `=` as the separator too.
    if HF_USER=$(NO_COLOR=1 hf auth whoami 2>/dev/null \
            | sed -nE 's/^[[:space:]]*user[:=][[:space:]]*([^[:space:]]+).*/\1/p' \
            | head -1) && [ -n "$HF_USER" ]; then
        :
    else
        echo "Cannot resolve HF_USER. Run: hf auth login   (or pass HF_USER=...)" >&2
        exit 1
    fi
fi

# Fail early and legibly if peft is missing, rather than 10 minutes into a
# model download inside policies/pretrained.py's `from peft import get_peft_model`.
if [ "$MODE" != "full" ] || [ -n "$RESUME_ADAPTER" ]; then
    if ! python -c "import peft" 2>/dev/null; then
        echo "peft is not installed — 'lerobot[pi]' does not pull it in." >&2
        echo "Fix: pip install 'lerobot[pi,peft]>=0.6.1,<0.7'   (or re-run setup.sh)" >&2
        exit 1
    fi
fi

dataset_slug=$(printf "%s" "$DATASET_REPO" | sed -E 's|^[^/]+/||; s/[^a-zA-Z0-9]+/_/g')
POLICY_REPO="${POLICY_REPO:-${HF_USER}/pi05_${dataset_slug}}"
# Indirection matters: without it a second concurrent run writes its checkpoints
# into the first run's directory. POLICY_REPO was already overridable; this was not.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/pi05_${dataset_slug}_${MODE}}"

# openpi's real LoRA freeze filter, spelled out because lerobot's default is
# not it. `full_training_modules` is lerobot's name for PEFT's
# `modules_to_save` (renamed in _preprocess_peft_cli_overrides).
HYBRID_TARGETS='(.*\.(gemma_expert|language_model)\..*\.self_attn\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'

if [ -n "$RESUME_ADAPTER" ]; then
    BASE_DESC="$RESUME_ADAPTER (resuming an existing adapter)"
    mode_args=(--policy.pretrained_path="$RESUME_ADAPTER" --policy.use_peft=true)
else
    case "$MODE" in
        full)
            BASE_DESC="lerobot/pi05_base (FULL fine-tune, no adapters)"
            mode_args=(--policy.pretrained_path=lerobot/pi05_base)
            ;;
        hybrid)
            BASE_DESC="lerobot/pi05_base (LoRA r=$LORA_RANK on LLM+expert, vision tower TRAINABLE)"
            mode_args=(
                --policy.pretrained_path=lerobot/pi05_base
                --peft.method_type=LORA
                --peft.r="$LORA_RANK"
                --peft.target_modules="$HYBRID_TARGETS"
                --peft.full_training_modules="['vision_tower']"
            )
            ;;
        expert-lora)
            BASE_DESC="lerobot/pi05_base (expert-only LoRA — the KNOWN-BAD config)"
            mode_args=(
                --policy.pretrained_path=lerobot/pi05_base
                --peft.method_type=LORA
                --peft.r="$LORA_RANK"
            )
            ;;
    esac
fi

private_flag=true
[ "${PUSH_PUBLIC:-}" = "1" ] && private_flag=false

cat <<EOF
=== finetune pi05_base ===
  mode:     $MODE
  dataset:  $DATASET_REPO
  base:     $BASE_DESC
  steps:    $STEPS
  batch:    $BATCH_SIZE  (no gradient-accumulation flag exists in lerobot-train 0.5.1)
  lr:       $LR
  output:   $OUTPUT_DIR
  push to:  $POLICY_REPO   (private=$private_flag)

  $STEPS steps x batch $BATCH_SIZE = $((STEPS * BATCH_SIZE)) sample-presentations.
  openpi's own pi05_droid_finetune recipe is 20000 x 32 = 640000. A 100-episode
  Haller dataset is ~90000 frames, so that is ~3.5 epochs. If this number is
  under ~200000 you are not fine-tuning, you are warming up.

  Disk: checkpoints are model + optimizer state, ~20-43 GiB EACH for a full FT,
  and lerobot 0.5.1 never prunes them. At save_freq=5000 over $STEPS steps that
  is $((STEPS / 5000)) checkpoints. Use a 300 GB volume.
EOF
if [ "$MODE" = "hybrid" ]; then
    cat <<'EOF'

  WARNING, hybrid is UNVERIFIED on this box. Two things to check in the first
  minute of the log before you leave it running overnight:
    1. draccus parsed the target_modules regex (target_modules: list[str]|str)
    2. `trainable_params` is in the hundreds of millions, not the low millions.
       Low millions means the vision tower did NOT come along and you are back
       in the 0.14 configuration.
EOF
fi
# Enforce lerobot's coupling here rather than letting it raise minutes in.
eval_args=()
if [ "$EVAL_STEPS" != "0" ] || [ "$EVAL_SPLIT" != "0.0" ]; then
    if [ "$EVAL_STEPS" = "0" ] || [ "$EVAL_SPLIT" = "0.0" ]; then
        echo "EVAL_SPLIT and EVAL_STEPS must be set together (lerobot raises otherwise)." >&2
        echo "Got EVAL_SPLIT=$EVAL_SPLIT EVAL_STEPS=$EVAL_STEPS. Example: EVAL_SPLIT=0.15 EVAL_STEPS=250" >&2
        exit 64
    fi
    eval_args=(--dataset.eval_split="$EVAL_SPLIT" --eval_steps="$EVAL_STEPS")
else
    echo "WARNING: no held-out eval. Every number this run reports will be a"
    echo "         TRAINING loss. Set EVAL_SPLIT and EVAL_STEPS to change that."
fi

ans=""
read -r -p "Proceed? [y/N] " ans || true
[ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "Aborted."; exit 0; }

exec lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=pi05 \
    "${mode_args[@]}" \
    --policy.repo_id="$POLICY_REPO" \
    --policy.private="$private_flag" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=false \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --policy.optimizer_lr="$LR" \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --policy.scheduler_decay_steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    "${eval_args[@]}" \
    --num_workers=8 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="pi05_${dataset_slug}_${MODE}" \
    --wandb.enable="$WANDB_ENABLE"
