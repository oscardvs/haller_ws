# RunPod inference and finetuning (π0.5, GR00T, etc.)

This guide takes you from "I have an SO-101 dataset on Hugging Face" to
"I've seen what π0.5 predicts on it" and "I've LoRA-finetuned it on my data,"
all on a rented cloud GPU.

You don't need this guide for ACT or any small policy that fits on a laptop
GPU — train those locally. Use RunPod when the model needs ≥10 GB VRAM,
which is every interesting generalist VLA (π0, π0.5, π0-FAST, SmolVLA-450M,
Wall-X, X-VLA, NVIDIA GR00T).

> Companion guides:
> - [`dataset-collection.md`](./dataset-collection.md) — how to record the
>   SO-101 dataset you'll evaluate against.
> - The scripts referenced below all live in
>   [`scripts/runpod/`](../../scripts/runpod/).

## Why offline / replay eval first

The natural urge is to deploy a policy in closed loop on the real arm. Don't
start there. The arm at home is connected over the internet to the pod, the
camera streams have to be tunneled, and any policy mistake can damage the
hardware. Instead, start with **replay-style evaluation**:

> _Take an episode I already recorded, feed the policy each frame's
> observation, and compare its predicted action to the one I actually used._

That answers "would this policy have done something sensible" without any
network, latency, or hardware-damage risk. It's the cheapest possible signal
that the model is reasonable on your data. Closed-loop comes later.

`scripts/runpod/replay_eval.py` does exactly this.

## Picking a pod

| Goal                                     | Recommended GPU             | Approx. price (community/secure) |
|------------------------------------------|-----------------------------|----------------------------------|
| Inference + replay eval on π0.5 / pi0    | RTX 4090 24 GB              | ~$0.34 / $0.69 per hour          |
| LoRA finetune π0.5 on a small dataset    | RTX 4090 24 GB              | ~$0.34 / $0.69 per hour          |
| Full finetune π0.5 or X-VLA              | A100 40 GB                  | ~$1.20 / $1.90 per hour          |
| Production deploy (later, on Jetson)     | Jetson Thor / Orin AGX      | own-hardware                     |

Template: **`runpod/pytorch:2.4.0-py3.11-cuda12.4-devel-ubuntu22.04`** (or the
equivalent latest 2.x / CUDA-12 image). The PyTorch image is fastest to
provision; LeRobot pip-installs on top in ~2 minutes. **Volume:** 50 GB
minimum (π0.5 itself is ~9 GB, plus dataset cache + intermediate checkpoints).

Cost rule of thumb: a complete cycle of "spin up pod → setup → smoke test →
replay-eval one episode → shut down" costs ~$0.30 on a 4090. Don't agonize.

## On-pod setup

After the pod boots, open a shell from the RunPod UI and run:

```bash
git clone https://github.com/oscardvs/haller_ws.git
cd haller_ws
bash scripts/runpod/setup.sh
```

`setup.sh` is idempotent. It:
1. Installs `ffmpeg`, `libgl1`, `git-lfs`,
2. Pip-installs `lerobot[pi]`, `huggingface_hub[cli]`, `matplotlib`, `pandas`,
3. Confirms `torch.cuda.is_available()` sees the GPU and prints VRAM,
4. Confirms `import lerobot` resolves.

Then authenticate with the Hub (one-time, interactive):

```bash
hf auth login
# Paste a write-access token from https://huggingface.co/settings/tokens
# Required only if you want to push artifacts back (finetuned weights, eval datasets).
# For read-only eval against your own dataset, a read token is enough.
```

> **Tip.** You can also bake the token into the pod via the RunPod "Environment
> variables" UI as `HF_TOKEN=...`; the `hf` CLI picks it up automatically and
> you skip the interactive login.

## Smoke test the policy

Before paying for a full dataset download + replay, confirm π0.5 actually
loads and runs on this pod:

```bash
python scripts/runpod/policy_smoke_test.py
# Or pick a different generalist:
python scripts/runpod/policy_smoke_test.py --policy-repo lerobot/pi0_base
```

The first run downloads ~9 GB of weights and warms the GPU. Subsequent runs
take ~10 s. A pass means the model loads, accepts an SO-101-shaped synthetic
observation, and emits a 6-DOF action chunk. A fail at this stage means
something's wrong with the env or the checkpoint — don't move on until it
passes.

## Replay-eval against your dataset

Once you've recorded a dataset (see [`dataset-collection.md`](./dataset-collection.md))
and pushed it to the Hub, evaluate it:

```bash
python scripts/runpod/replay_eval.py \
    --dataset-repo $HF_USER/so101_pick_red_cube \
    --policy-repo  lerobot/pi05_base \
    --episode 0
```

Outputs land in `outputs/eval/<timestamp>_<policy>__<dataset>__ep<n>/`:

- `actions.csv` — every frame, every joint, predicted + ground-truth + error
- `summary.json` — per-joint MAE / RMSE / max error, mean inference latency,
  the GPU it ran on, the seed
- `joints.png` — six-panel matplotlib plot, predicted (orange) overlaid on
  ground-truth (blue) for every joint over time

### Reading the output

**Low MAE on every joint** doesn't mean the policy would succeed on the real
arm — open-loop replay can't capture closed-loop dynamics. But **wild
divergence** (predicted actions jumping orders of magnitude away from the
human-teleoperated ones) is a strong "this policy doesn't understand the
task in this setting" signal, and saves you a closed-loop deployment.

**π0.5 is generalist, not magic.** Out of the box it will probably get the
*shape* of your task vaguely right (move in the rough direction) but not
match the human teleop trace closely — that's expected, you haven't
finetuned it yet. The replay eval mostly tells you whether the model is
*reacting to the observation at all* vs emitting noise.

### Observation-key mismatches

If your dataset's camera key names don't match what the policy was trained
with, you'll get a runtime error. Remap them:

```bash
python scripts/runpod/replay_eval.py \
    --dataset-repo $HF_USER/so101_pick_red_cube \
    --rename observation.images.base=observation.images.cam0 \
    --rename observation.images.wrist=observation.images.cam1
```

For π0.5 the convention is `observation.images.cam{i}`. Your dataset
probably uses the names you configured in `hmi/backend/config.yaml`
(`base_front`, `wrist_right` → `observation.images.base_front`,
`observation.images.wrist_right`). The error message will tell you exactly
which key the policy expected.

## LoRA-finetune π0.5 on your dataset

When replay eval confirms the pipeline works, finetune:

```bash
scripts/runpod/finetune_pi05_lora.sh $HF_USER/so101_pick_red_cube 5000
```

Defaults:
- `--policy.pretrained_path=lerobot/pi05_base`
- `--policy.peft_config.use_peft=true` (LoRA — keeps trainable params small)
- `--policy.gradient_checkpointing=true` (fits a 4 GB activation budget on 24 GB)
- `--policy.dtype=bfloat16`
- `--batch_size=4`, `--steps=5000`

5 000 steps on a 4090 with a 50-episode dataset is roughly 1.5–2 hours
(~$0.50–0.70). Output goes to
`outputs/train/pi05_<your-dataset>_lora/checkpoints/` and the final
checkpoint pushes to `${HF_USER}/pi05_<your-dataset>_lora` on the Hub.

Override the trainable params, batch size, or push target via env vars:

```bash
HF_USER=myteam \
POLICY_REPO=myteam/pi05_pickplace_v2 \
BATCH_SIZE=8 \
scripts/runpod/finetune_pi05_lora.sh myteam/so101_pickplace 8000
```

After the finetune, re-run `replay_eval.py` against your new policy and
compare — the joint traces should track much closer to ground truth on the
training episodes (and that's the basic sanity check that finetuning did
*something*).

## Retrieving results to your laptop

Two easy paths:

**1. Push back to the Hub.** Trained policies push automatically. For eval
outputs:

```bash
hf upload $HF_USER/eval_pi05_so101_pick_red_cube \
    outputs/eval/<timestamp>_<run> \
    --repo-type=dataset
```

**2. Pull directly from the pod.** If the pod exposes SSH (RunPod does by
default for paid pods), `scp` works:

```bash
# From your laptop:
scp -r runpod-pod:haller_ws/outputs/eval/<run> .
```

## Cost discipline

- **Stop the pod when idle.** RunPod bills per-second. A pod left running
  overnight at $0.34/h is $8.20 / day. Stop after each session unless you're
  actively iterating.
- **Use community-cloud GPUs for exploration.** Half the price of secure
  cloud; the only practical difference is interruptibility, and inference /
  short finetunes finish well within typical uptime.
- **Cache models on a network volume.** If you do this regularly, mount a
  RunPod Network Volume at `/workspace` and put the HF cache there
  (`HF_HOME=/workspace/hf_cache`) so π0.5's 9 GB only downloads once.

## What this guide doesn't cover (yet)

- **Closed-loop deployment** (live policy → real arm over the network).
  Needs a websocket policy server, a local bridge owning the USB + cameras,
  and Tailscale to keep latency predictable. See the Phase 3 roadmap entry
  in [`dataset-collection.md`](./dataset-collection.md).
- **GR00T N1.7 deployment.** Uses NVIDIA's separate `Isaac-GR00T` runtime
  rather than `lerobot.policies.*`. Will be a sibling script set.
- **Multi-task / X-Embodiment finetuning.** Same shape as the single-task
  recipe above with a multi-dataset config; coverage TBD.
