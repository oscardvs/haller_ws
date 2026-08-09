# scripts/runpod/

Cloud-GPU recipes for running and finetuning generalist VLA policies (π0.5, π0,
SmolVLA, GR00T, …) against **bimanual Haller** datasets recorded with this repo.
Designed for [RunPod](https://www.runpod.io) pods on the
`runpod/pytorch:2.x-cuda12.x` image, but nothing here is RunPod-specific —
anywhere with a CUDA GPU and Python 3.12+ will work.

Verified against **lerobot 0.5.1** (the version this repo pins in
`hmi/backend/pyproject.toml`). CLI surfaces in lerobot move fast; if you bump the
pin, re-check the flags in `finetune_pi05_lora.sh` first.

| File | Purpose |
|------|---------|
| [`setup.sh`](./setup.sh) | First-boot setup on a fresh pod: apt deps, `lerobot[pi,peft]`, GPU sanity, and the two Hub gates you must clear. Idempotent. |
| [`policy_smoke_test.py`](./policy_smoke_test.py) | Load `lerobot/pi05_base`, push one synthetic 12-dim bimanual observation through the real preprocessor, and probe how many cameras the model will take. Fail-fast check that the pod is correctly provisioned. |
| [`replay_eval.py`](./replay_eval.py) | Replay one episode of a recorded `LeRobotDataset` through a policy; output per-joint and per-arm MAE/RMSE plus a left-arm/right-arm plot of predicted vs ground-truth traces. Hardware-free way to ask "would this policy have done something reasonable?". |
| [`finetune_pi05_lora.sh`](./finetune_pi05_lora.sh) | Wrapper around `lerobot-train` that attaches a fresh LoRA adapter to `pi05_base`. See the memory section below before you rent a card. |

Full end-to-end guide: [`docs/setup/runpod-inference.md`](../../docs/setup/runpod-inference.md).

Prerequisite (one-time, on your dev machine): record a bimanual dataset and push
it to the Hugging Face Hub — see
[`docs/setup/dataset-collection.md`](../../docs/setup/dataset-collection.md).

## What "bimanual" costs you here: nothing

Haller is two SO-101 arms, so state and action are **12-dim** — the left arm's
six joints (`left_shoulder_pan … left_gripper`) followed by the right arm's six —
recorded in **degrees**, alongside 3 recorded RGB camera channels (`top`, `left_wrist`, `right_wrist`).

π0.5 pads state and action to **32 dims internally**
(`max_state_dim` / `max_action_dim` in lerobot's `configuration_pi05.py`) and
unpads its output back to your dataset's real width inside
`predict_action_chunk()`. A 12-dim bimanual dataset therefore needs **no
architectural change, no config override, and no "bimanual" flag**. Do not go
looking for one; it does not exist because it is not needed.

Two visible consequences of the 32-dim pad, so they don't surprise you:

* A checkpoint loaded *bare* (no dataset attached) reports an action width of
  **32**, not 12 — there is nothing to unpad to yet. `policy_smoke_test.py`
  accepts both widths for exactly this reason.
* Nothing in the stack knows "left" from "right". The arms are just the first
  and last six columns of one vector. `replay_eval.py` splits them apart for
  reporting by name prefix, which is a presentation choice, not a model one.

## Quick path

```bash
# On the pod, after git clone:
bash scripts/runpod/setup.sh
hf auth login

# One-time, and easy to forget — see "Two Hub gates" below.
#   -> accept https://huggingface.co/google/paligemma-3b-pt-224

# One-time per dataset: π0.5 needs quantile stats.
lerobot-edit-dataset \
    --repo_id $HF_USER/haller_bimanual_<your_task> \
    --operation.type recompute_stats \
    --push_to_hub true

python scripts/runpod/policy_smoke_test.py
python scripts/runpod/replay_eval.py --dataset-repo $HF_USER/haller_bimanual_<your_task>

# Once replay-eval shows the model is at least reacting to your data:
scripts/runpod/finetune_pi05_lora.sh $HF_USER/haller_bimanual_<your_task> 5000
```

## Two Hub gates, not one

`hf auth login` gets you `lerobot/pi05_base` and lets you push results. It does
**not** get you the tokenizer. π0.5's preprocessor builds its prompt with
`google/paligemma-3b-pt-224` (hardcoded in lerobot's `policies/pi05/processor_pi05.py`),
and that repo is gated: *"To access PaliGemma on Hugging Face, you're required to
review and agree to Google's usage license."* Accept it, logged in as the same
account your token belongs to, or every π0.5 script here dies with a
`GatedRepoError` on the first tokenizer fetch.

## Dataset stats: π0.5 wants quantiles

π0.5 sets `STATE` and `ACTION` normalization to `QUANTILES`
(`configuration_pi05.py`), which needs `q01` and `q99` in your dataset's
`meta/stats.json`. Recordings made before lerobot added quantile stats do not
have them, and you find out on the **first batch**:

```
ValueError: QUANTILES normalization mode requires q01 and q99 stats
```

Fix it once and push, so the pod and your laptop agree:

```bash
lerobot-edit-dataset --repo_id <ds> --operation.type recompute_stats --push_to_hub true
```

(If you also plan to train with `--policy.use_relative_actions=true`, add
`--operation.relative_action true --operation.chunk_size 50
--operation.relative_exclude_joints "['left_gripper','right_gripper']"` — note
both gripper names, since the bimanual dataset has two.)

## GPU memory: what "24 GB" really buys

This README used to say the LoRA script was "sized for a 24 GB GPU". Treat that
as a floor, not a spec:

| Source | Claim |
|---|---|
| [openpi README](https://github.com/Physical-Intelligence/openpi) (upstream π0/π0.5) | inference > 8 GB · **LoRA finetune > 22.5 GB** · full finetune > 70 GB |
| [LeRobot π0/π0.5 guide](https://huggingface.co/docs/lerobot/en/pi0) | trains at `--batch_size=32`; **never mentions LoRA or PEFT at all** |

So a 24 GB card (RTX 4090 / A10G) clears openpi's LoRA floor by about 1.5 GB —
and that floor is for a single camera at batch 1 on openpi's own JAX stack. Haller
records **3** cameras — `top` / `left_wrist` / `right_wrist`, chosen to match π0.5's
own `base_0_rgb` + two-wrist pretraining slot count — each adding a full SigLIP
prefix of image tokens to every forward pass. The bimanual sim *renders* 5, but the
other two are operator views that are never recorded, so training cost is set by
the 3 in the dataset. Read the script's default as: *24 GB = LoRA + batch 1–2 +
gradient checkpointing at 3 cameras*, still unverified end-to-end on this stack.

Memory levers, least to most damaging to quality:

1. `BATCH_SIZE=1`. Note there is **no gradient-accumulation flag** in
   `lerobot-train` 0.5.1 — an earlier version of this directory claimed one.
2. `FREEZE_VISION_ENCODER=true` → `--policy.freeze_vision_encoder`.
3. `TRAIN_EXPERT_ONLY=true` → `--policy.train_expert_only`. Freezes the whole
   VLM and trains only the action expert plus projections. Biggest single saving,
   and the lever LeRobot's own guide names.
4. Fewer cameras, or rent an A100/H100 for an evening.

Also worth knowing: **LoRA is not part of LeRobot's official π0.5 workflow.** It
is a library feature (`--peft.*` on `lerobot-train`) that the guide does not walk
through. `finetune_pi05_lora.sh` documents the exact flag surface, including the
trap that `--policy.use_peft` means *resume an existing adapter*, not *start a new
one*.

## How many cameras can π0.5 take?

There is **no documented maximum**. Reading lerobot's implementation,
`PI05Pytorch.embed_prefix()` simply iterates `zip(images, img_masks)` and pushes
every view through one shared SigLIP tower, concatenating the results into the
attention prefix. There is no per-camera parameter and no hard-coded 3, so the
architecture is N-flexible; extra cameras cost prefix length and VRAM, nothing
else.

But π0.5 was **pretrained with 3 camera slots**, and no published 5-camera run
was found. 5 views is out-of-distribution. "It runs" is not "it works".

`policy_smoke_test.py` is where that gets settled empirically rather than
argued: it runs 3 cameras, then probes 5, and prints a verdict with timings and
peak VRAM. That verdict is a claim about **shape and memory only**. Whether 5
views help or hurt the policy is a question for `replay_eval.py` MAE at 3 vs 5
after a finetune.

## Licensing: unresolved for commercial use

**Do not assume π0.5 is Apache-2.0.** The picture is genuinely split:

* openpi's repository **code** is Apache-2.0.
* lerobot's π0.5 **source** is Apache-2.0, and LeRobot's own docs page states
  "This model follows the Apache 2.0 License, consistent with the original
  OpenPI repository" — <https://huggingface.co/docs/lerobot/en/pi0>.
* But the **weights** you actually download declare something else. The
  `lerobot/pi05_base` model card frontmatter says `license: gemma`
  (<https://huggingface.co/lerobot/pi05_base>), because π0.5 is built on a
  PaliGemma backbone — and PaliGemma itself is gated behind Google's licence.

Gemma Terms of Use permit commercial use, but attach a Prohibited Use Policy and
obligations that follow the model downstream (you must pass the terms on with
any redistribution, and keep the use restrictions attached to derivatives —
including a LoRA adapter trained on top).

**Flagging this as UNRESOLVED, not settled.** If Haller output is ever going to
be commercial, the "π0.5 is Apache-2.0" line is the wrong one to rely on; someone
needs to read the Gemma terms against the intended use. Sources:

* <https://huggingface.co/lerobot/pi05_base> (frontmatter: `license: gemma`)
* <https://github.com/Physical-Intelligence/openpi> (repo code: Apache-2.0)
* <https://huggingface.co/google/paligemma-3b-pt-224> (the gated backbone)
* <https://ai.google.dev/gemma/terms>

## Worth an A/B: GR00T N1.7

If π0.5's licence position or its 3-camera pretraining bias becomes a problem,
NVIDIA's GR00T is the natural comparison, and it is **cheaper to try than you
would expect**:

* `--policy.type=groot` is **already registered in the installed lerobot 0.5.1**
  (`policies/groot/configuration_groot.py`). No separate Isaac-GR00T runtime, no
  second training stack — it consumes a LeRobotDataset v3.0 directly, same as
  π0.5, and `lerobot-train` drives it with the same flags.
* GR00T **N1.7** ships under the NVIDIA Open Model License, which permits
  commercial use with attribution. The non-commercial restriction people
  remember applied to **N1.5**, not N1.7.
* Like π0.5 it zero-pads short vectors (`max_state_dim=64`, `max_action_dim=32`),
  so 12-dim bimanual is a non-issue there too.
* It has its own LoRA knobs (`--policy.lora_rank`, `--policy.lora_alpha`) rather
  than the shared `--peft.*` block.

Two caveats before you budget a day for it (**neither verified here — no GPU on
the machine this was written on**):

* lerobot 0.5.1's `GrootConfig` defaults to `base_model_path="nvidia/GR00T-N1.5-3B"`,
  i.e. the **N1.5** weights. Getting the commercially-licensed N1.7 means
  overriding `--policy.base_model_path=nvidia/GR00T-N1.7-3B`.
* The companion `tokenizer_assets_repo` default is
  `lerobot/eagle2hg-processor-groot-n1p5`, which is N1.5-specific. Whether it is
  compatible with N1.7 weights is untested.

This is a pointer, not an implementation. Install it with `lerobot[groot]` (which
does pull `peft` in, unlike `lerobot[pi]`).

## Things that used to be documented here and were wrong

Kept deliberately, because each one cost real debugging time:

| Claim | Reality |
|---|---|
| `--policy.peft_config.use_peft=true` | Does not exist in 0.5.1. Dies in argument parsing. Real surface: `--policy.use_peft` (adapter *resume*) and a top-level `--peft.*` block (fresh adapter). |
| `--policy.pretrained_path` + `use_peft` starts a LoRA run | It **resumes** one. lerobot reads `adapter_config.json` from that path; `lerobot/pi05_base` has none, so it fails. |
| `lerobot[pi]` is enough for LoRA | It resolves to `transformers-dep` + `scipy-dep` only. No `peft`. Use `lerobot[pi,peft]`. |
| Map your cameras onto `observation.images.cam0` / `cam1` | No such convention exists anywhere in lerobot. Expected keys come from your dataset (with `--policy.pretrained_path`) or from the checkpoint's own config (with `--policy.path`). Renaming in the first case is what *breaks* it. |
| `--batch-size` plus "use gradient accumulation for effective batch ≥ 16" | `lerobot-train` 0.5.1 has no gradient-accumulation flag. |
| π0.5 is Apache-2.0 | See "Licensing" above. |
