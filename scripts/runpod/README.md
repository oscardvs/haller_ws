# scripts/runpod/

Cloud-GPU recipes for running and finetuning generalist VLA policies (π0.5, π0,
SmolVLA, …) against SO-101 datasets recorded with this repo. Designed for
[RunPod](https://www.runpod.io) pods on the `runpod/pytorch:2.x-cuda12.x` image,
but nothing here is RunPod-specific — anywhere with a CUDA GPU and Python 3.11+
will work.

| File | Purpose |
|------|---------|
| [`setup.sh`](./setup.sh) | First-boot setup on a fresh pod: apt deps, `lerobot[pi]`, GPU sanity. Idempotent. |
| [`policy_smoke_test.py`](./policy_smoke_test.py) | Load `lerobot/pi05_base` and run one inference step on a synthetic SO-101 observation. Fail-fast check that the pod is correctly provisioned. |
| [`replay_eval.py`](./replay_eval.py) | Replay one episode of a recorded `LeRobotDataset` through a policy; output per-joint MAE/RMSE + a matplotlib plot of predicted vs ground-truth joint traces. Hardware-free way to ask "would this policy have done something reasonable?". |
| [`finetune_pi05_lora.sh`](./finetune_pi05_lora.sh) | Wrapper around `lerobot-train` that LoRA-finetunes `pi05_base` on your dataset. Sized for a 24 GB GPU. |

Full end-to-end guide: [`docs/setup/runpod-inference.md`](../../docs/setup/runpod-inference.md).

Prerequisite (one-time, on your dev machine): record an SO-101 dataset and push
it to the Hugging Face Hub — see
[`docs/setup/dataset-collection.md`](../../docs/setup/dataset-collection.md).

## Quick path

```bash
# On the pod, after git clone:
bash scripts/runpod/setup.sh
hf auth login

python scripts/runpod/policy_smoke_test.py
python scripts/runpod/replay_eval.py --dataset-repo $HF_USER/so101_<your_task>

# Once replay-eval shows the model is at least reacting to your data:
scripts/runpod/finetune_pi05_lora.sh $HF_USER/so101_<your_task> 5000
```
