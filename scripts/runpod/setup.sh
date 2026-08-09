#!/usr/bin/env bash
# scripts/runpod/setup.sh — first-boot setup on a RunPod pod.
#
# Designed for the standard `runpod/pytorch:2.x-py3.12-cuda12.x-devel-*` images.
# NOTE THE PYTHON VERSION: lerobot 0.5.1 declares `Requires-Python: >=3.12`, so
# the older py3.11 RunPod templates cannot install it at all — pip resolves to
# nothing and you lose an hour working out why. Step 0 below checks this first.
# Idempotent: re-running it is harmless.
#
# Usage:
#     # On the pod (after `git clone https://github.com/oscardvs/haller_ws.git`):
#     cd haller_ws && bash scripts/runpod/setup.sh
#
#     # Then authenticate with the Hub interactively:
#     hf auth login
#
# Tunables (env vars):
#     LEROBOT_EXTRAS  Extras to install with lerobot. Default: "pi,peft".
#                     "pi"   -> pi0 / pi0-fast / pi05  (resolves to transformers-dep + scipy-dep only)
#                     "peft" -> transformers-dep + peft>=0.18  — REQUIRED for finetune_pi05.sh.
#                     `lerobot[pi]` alone does NOT pull peft in, so a LoRA run would
#                     ImportError deep inside lerobot (policies/pretrained.py, `from peft import
#                     get_peft_model`). Verified against lerobot 0.5.1 package metadata.
#                     Add "smolvla" for SmolVLA, "groot" for GR00T N1.x, "feetech" for SO-101
#                     hardware (not needed on a cloud pod — no serial bus there).
set -euo pipefail

LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-pi,peft}"

echo "=== runpod setup ==="
echo "lerobot extras: $LEROBOT_EXTRAS"

# 0. Fail on the wrong interpreter before spending five minutes on apt.
echo "[0/4] python version…"
python - <<'PY'
import sys
v = sys.version_info
print(f"  python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
if (v.major, v.minor) < (3, 12):
    raise SystemExit(
        "lerobot 0.5.1 requires Python >= 3.12 (Requires-Python: >=3.12).\n"
        "Pick a runpod/pytorch template with py3.12 or newer and start over — "
        "installing on 3.11 fails in confusing ways rather than cleanly."
    )
PY

# 1. OS deps. The runpod/pytorch images already include git+python+pip but
# usually skip ffmpeg, which lerobot needs for video decoding.
echo "[1/4] apt deps (ffmpeg, build basics)…"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git-lfs >/dev/null
git lfs install --skip-repo

# 2. Python deps. pip --upgrade first because runpod images sometimes pin old pip.
echo "[2/4] python deps…"
python -m pip install --upgrade pip wheel
# NOTE: no `[cli]` extra any more. lerobot 0.5.1 requires huggingface-hub>=1.0,
# and in hub 1.x the `hf` command is a plain console_script — `huggingface_hub[cli]`
# is an unknown extra that pip merely warns about, which is easy to miss.
python -m pip install \
    "huggingface_hub>=1.0,<2.0" \
    "lerobot[${LEROBOT_EXTRAS}]>=0.5,<0.6" \
    "matplotlib>=3.9" \
    "pandas>=2.2"

# 3. GPU sanity: confirm torch sees the GPU before we waste time on a model download.
echo "[3/4] gpu sanity…"
python - <<'PY'
import torch
assert torch.cuda.is_available(), (
    "CUDA not available. Check the pod template — needs an NVIDIA-enabled image."
)
print(f"  torch {torch.__version__}")
print(f"  cuda  {torch.version.cuda}")
print(f"  gpus  {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info()
print(f"  vram  {free / 2**30:.1f} / {total / 2**30:.1f} GiB free")
PY

# 4. lerobot sanity: confirm the install resolved + version reports, and that peft
# actually landed (the single most common cause of a LoRA run dying 10 minutes in).
echo "[4/4] lerobot sanity…"
python - <<'PY'
import lerobot
print(f"  lerobot {lerobot.__version__}")
try:
    import peft
    print(f"  peft    {peft.__version__}")
except ImportError:
    print("  peft    MISSING — finetune_pi05.sh will fail. "
          "Re-run with LEROBOT_EXTRAS='pi,peft'.")
PY

cat <<'EOF'

=== ready ===

--- 1. Hub authentication (two separate things) ---

  hf auth login          # paste a WRITE-access token

That covers pulling `lerobot/pi05_base` and pushing your finetuned policy.

It does NOT cover the tokenizer. π0.5's preprocessor builds its prompt with the
PaliGemma tokenizer (`google/paligemma-3b-pt-224`, hardcoded in lerobot's
policies/pi05/processor_pi05.py). That repo is GATED: "To access PaliGemma on
Hugging Face, you're required to review and agree to Google's usage license."
Until you accept it *with the same account as your token*, every π0.5 script
here dies with a GatedRepoError on the first tokenizer fetch.

  -> Open https://huggingface.co/google/paligemma-3b-pt-224 while logged in,
     accept the license, THEN come back. One-time, per account.

--- 2. Dataset stats: π0.5 needs QUANTILES ---

π0.5 normalizes state and action with q01/q99 quantiles, not mean/std
(lerobot/policies/pi05/configuration_pi05.py: STATE and ACTION -> QUANTILES).
If meta/stats.json in your recording predates lerobot's quantile support, the
first training/eval batch raises:

  "QUANTILES normalization mode requires q01 and q99 stats"

Fix it once, then push, so the pod and your laptop agree:

  lerobot-edit-dataset \
      --repo_id $HF_USER/haller_bimanual_<your_task> \
      --operation.type recompute_stats \
      --push_to_hub true

--- 3. Next steps ---

  python scripts/runpod/policy_smoke_test.py          # ~3 min: download + warm pi05_base
  python scripts/runpod/policy_smoke_test.py --num-images 5   # does 5 cameras work?
  python scripts/runpod/replay_eval.py \
      --dataset-repo $HF_USER/haller_bimanual_<your_task>     # offline eval

Haller is BIMANUAL: 12-dim state/action (left arm 6 joints, then right arm 6),
recorded in degrees, 3 recorded RGB channels (top, left_wrist, right_wrist). π0.5 pads state/action to 32 dims
internally and unpads its output back to your dataset's real width, so 12-dim
needs no architectural change and no config edit.

See docs/setup/runpod-inference.md for the full workflow.
EOF
