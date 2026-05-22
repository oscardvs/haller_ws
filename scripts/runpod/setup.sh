#!/usr/bin/env bash
# scripts/runpod/setup.sh — first-boot setup on a RunPod pod.
#
# Designed for the standard `runpod/pytorch:2.x-py3.11-cuda12.x-devel-*` images.
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
#     LEROBOT_EXTRAS  Extras to install with lerobot. Default: "pi" (covers pi0/pi0-fast/pi05).
#                     Add "smolvla" for SmolVLA, "feetech" for SO-101 hardware (not needed for inference-only).
set -euo pipefail

LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-pi}"

echo "=== runpod setup ==="
echo "lerobot extras: $LEROBOT_EXTRAS"

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
python -m pip install \
    "huggingface_hub[cli]>=0.27" \
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

# 4. lerobot sanity: confirm the install resolved + version reports.
echo "[4/4] lerobot sanity…"
python -c "import lerobot; print(f'  lerobot {lerobot.__version__}')"

cat <<EOF

=== ready ===

Next steps:
  hf auth login                                       # paste a write-access token
  python scripts/runpod/policy_smoke_test.py          # ~3 min: download + warm pi05_base
  python scripts/runpod/replay_eval.py \\
      --dataset-repo \$HF_USER/so101_<your_task>      # offline eval against your dataset

See docs/setup/runpod-inference.md for the full workflow.
EOF
