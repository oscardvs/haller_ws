# Dataset collection (SO-101)

This guide walks through recording a LeRobot dataset on the Haller SO-101 arms
and pushing it to the Hugging Face Hub. It is the prerequisite for training or
fine-tuning a policy on your own task.

It assumes the [LeRobot environment](./lerobot-environment.md) and the
[SO-101 arm bring-up](./so101-arm.md) are already done, and that you can
teleoperate via the [HMI](../../hmi/README.md).

The collection path documented here is **Phase 1**: recording happens
out-of-process via `scripts/record_dataset.sh`, which wraps `lerobot-record`.
The HMI gets a Recording panel that builds the command for you, but launching
recording from inside the HMI process is a separate piece of work (Phase 2);
see the roadmap at the end.

---

## Why a custom dataset

You can run policy *inference* against any SO-101 dataset on the Hub
(e.g. [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)).
That's a useful smoke test of the inference stack.

You cannot, however, expect a published policy to *succeed at your task* on
*your hardware* without finetuning. Different cameras, different lighting,
different calibration midpoints, different objects — generalist VLAs like
π0.5 and GR00T N1.7 narrow the gap considerably but don't close it. So:
record demonstrations on your arm, of your task, then either train ACT on
~50 episodes or LoRA-finetune a generalist VLA on ~20–50 episodes.

The dataset format is [LeRobotDataset](https://huggingface.co/docs/lerobot/datasets)
(parquet shards + per-camera MP4s), the same shape that every policy in
`lerobot-train` consumes.

---

## Prerequisites

### Hardware

- Follower arm at `/dev/haller_arm_follower`, leader arm at
  `/dev/haller_arm_leader` (udev symlinks from `scripts/99-haller-devices.rules`).
- Both arms calibrated (see [`so101-arm.md`](./so101-arm.md)). The HMI's
  calibration bootstrap auto-copies the leader's teleoperator calibration into
  the follower directory so both can be driven as follower-style robots.
- **At least one camera plugged in.** A 2-camera setup (wrist + base) is the
  norm in published SO-101 datasets and the recommended target — most
  generalist VLAs are trained with multiple camera streams. A single base
  camera is fine for ACT smoke testing but suboptimal for VLA finetuning.

### Software

- HMI venv installed (`~/venvs/haller-hmi/`) with `lerobot[feetech]>=0.5,<0.6`
  and `opencv-python>=4.x` — both pulled in by `hmi/backend/pyproject.toml`.
- Hugging Face CLI authenticated:

  ```bash
  hf auth login --token "$HUGGINGFACE_TOKEN"
  ```

- (Optional, recommended) write-access HF repo where the dataset will land.
  The script defaults to `<your-hf-user>/so101_<slug>` and creates the repo on
  first push.

---

## 1. Wire your cameras

Cameras are declared in [`hmi/backend/config.yaml`](../../hmi/backend/config.yaml).
The same `(index_or_path, width, height, fps)` tuple feeds both the HMI's
live view *and* `lerobot-record`, so configuring it once is enough.

Find your camera devices:

```bash
# Each USB webcam typically registers 2 nodes (capture + metadata). The
# capture node is the smaller-numbered one in each pair.
for d in /dev/video*; do
    echo "=== $d ==="
    udevadm info --query=property --name="$d" | grep -E "ID_(MODEL|VENDOR|V4L_PRODUCT)="
done

# Or use the v4l2-utils package if installed:
#   v4l2-ctl --list-devices
```

Test a candidate device with `ffplay`:

```bash
ffplay -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video2
# Ctrl-C to close.
```

Then edit `hmi/backend/config.yaml`:

```yaml
cameras:
  - id: wrist_right
    role: wrist
    arm_id: right            # binds this camera to the "right" arm card in the HMI
    source: opencv
    index_or_path: /dev/video2
    width: 640
    height: 480
    fps: 30
  - id: base_front
    role: base
    source: opencv
    index_or_path: /dev/video0
    width: 640
    height: 480
    fps: 30
```

Restart the HMI to pick up the new config:

```bash
# Dev laptop: stop scripts/run_hmi.sh and restart it
# Jetson:    sudo systemctl restart haller-hmi.service
```

Open the dashboard. You should see live thumbnails in the new **Cameras** strip
and a real wrist-camera feed inside each arm card.

> **Tip.** A camera marked `source: placeholder` still appears in the HMI but
> renders the dashed-border "no feed" placeholder. That's the right state for
> a slot reserved for hardware that hasn't arrived yet.

---

## 2. Smoke-test the camera pipeline

From the HMI dashboard:

- The **Cameras** strip should show `N/N live` for every configured `opencv`
  camera.
- Each tile should show a real image with the configured resolution + FPS
  printed in the lower-right corner.

From the command line:

```bash
# Snapshot a single JPEG from each camera
curl -o /tmp/base.jpg http://localhost:8000/cameras/base_front/snapshot

# Watch the live stream in a browser
xdg-open http://localhost:8000/cameras/base_front/stream
```

If a camera 503s the snapshot endpoint, check the HMI logs — the most common
cause is "device busy" because another process (`cheese`, a previous
`scripts/run_hmi.sh` that didn't shut down cleanly, etc.) is holding it open.

---

## 3. Stop the HMI before recording

`lerobot-record` needs exclusive control of:

- the **leader serial port** (`/dev/haller_arm_leader`) — to read the
  operator-driven joint positions,
- the **follower serial port** (`/dev/haller_arm_follower`) — to send actions,
- every configured **camera device** — to capture frames in lockstep with the
  control loop.

The HMI holds all of these while it's running, so you must stop it first.

```bash
# Dev laptop: Ctrl-C the scripts/run_hmi.sh process
# Jetson:    sudo systemctl stop haller-hmi.service
```

`scripts/record_dataset.sh` will refuse to start if anything else has those
device nodes open, and tells you which PID is holding them.

---

## 4. Record

The script lives at `scripts/record_dataset.sh` and wraps
`lerobot-record` with sensible defaults.

```bash
# Required arg: the task description (1 sentence — used as the language
# instruction and slugified for the dataset name).
# Optional arg: number of episodes (default 50).
scripts/record_dataset.sh "Grab the red cube and place it in the box" 20
```

What happens:

1. Activates the HMI venv (`~/venvs/haller-hmi/`).
2. Resolves your HF username from `hf auth whoami` (or `HF_USER=...` env override).
3. Refuses to start if any required device is in use; prints a confirmation
   banner with the resolved task, dataset name, cameras, and ports.
4. Runs `lerobot-record` with `--display_data=true` so `rerun` opens a live
   viewer of joint positions + camera feeds.
5. For each episode: leader → follower teleop for `EPISODE_TIME_SEC` (default
   30 s), then a `RESET_TIME_SEC` (default 5 s) pause to reset the scene.
6. On the final episode, encodes videos, writes parquet shards, and pushes
   the dataset to `<your-hf-user>/so101_<slug>`.

### Tunables (env vars)

| Variable            | Default | Notes |
|---------------------|---------|-------|
| `HF_USER`           | from `hf auth whoami` | Override if your HF org differs from your username. |
| `DATASET_REPO`      | `${HF_USER}/so101_<slug>` | Set explicitly to pick a custom repo. |
| `FPS`               | 30      | Capture + control rate. Matches most public SO-101 datasets. |
| `EPISODE_TIME_SEC`  | 30      | Max time per episode. |
| `RESET_TIME_SEC`    | 5       | Pause between episodes for scene reset. |
| `CAMERAS_JSON`      | base camera on `/dev/video0` | Full `lerobot --robot.cameras` dict. Override when you have multiple cameras. |

Example: two cameras, 60 episodes of a fruit-sorting task into a custom repo:

```bash
HF_USER=myteam \
DATASET_REPO=myteam/so101_fruit_sort_v1 \
CAMERAS_JSON='{ wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, base: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' \
scripts/record_dataset.sh "Sort the fruit by color into the matching bowl" 60
```

### Keyboard controls during recording

`lerobot-record` watches the keyboard:

| Key            | Effect |
|----------------|--------|
| `→` (right)    | end current episode early, save, move on |
| `←` (left)     | end current episode early, discard, re-record |
| `ESC`          | stop the run; videos written so far are encoded and pushed |

---

## 5. Sanity-check the dataset

Visualize the dataset locally before training:

```bash
hf download <your-hf-user>/so101_<slug> --repo-type=dataset --local-dir ~/lerobot_data/so101_<slug>
lerobot-visualize-dataset \
    --repo-id="<your-hf-user>/so101_<slug>" \
    --episode-index=0
```

Or open the auto-generated dataset card on the Hub — for v2.1+ datasets
HuggingFace renders an interactive episode viewer right on the page.

Things to check:

- **Episode count matches** what you intended.
- **Camera videos play** without dropped frames.
- **Joint state and action arrays are non-zero and varied** — flat traces
  usually mean the leader wasn't being driven, or the follower was stuck in
  STOP mode.
- **The language instruction is what you intended** (parquet column `task`).

---

## 6. What to do with the dataset

| Goal                                | Path |
|-------------------------------------|------|
| Train ACT from scratch (single task, ≥50 episodes) | `lerobot-train --policy.type=act --dataset.repo_id=...` — fits on a laptop GPU. |
| LoRA-finetune SmolVLA-base          | `--policy.path=lerobot/smolvla_base --policy.peft_config.use_peft=true` on a 16 GB+ cloud GPU. |
| LoRA-finetune π0.5 (recommended VLA path) | See [`runpod-inference.md`](./runpod-inference.md) — `scripts/runpod/finetune_pi05_lora.sh <your-dataset>` is the one-liner. |
| Replay-eval an existing policy on your data | See [`runpod-inference.md`](./runpod-inference.md) — `scripts/runpod/replay_eval.py` runs π0.5 / pi0 against your dataset and dumps per-joint error + plots. |
| Finetune NVIDIA GR00T N1.7          | Follow [Post-Training Isaac GR00T N1.5 for LeRobot SO-101 Arm](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) — official guide, ports cleanly to N1.7. |

Live closed-loop evaluation on the real arm is out of scope of this guide;
see the [LeRobot imitation-learning tutorial](https://huggingface.co/docs/lerobot/il_robots)
once you have a trained policy.

---

## Roadmap

**Phase 2 — HMI-integrated recorder.** Recording from inside the HMI without
having to stop it. Requires the HMI to multiplex serial-port access (currently
each `SO101Follower` owns its port exclusively) and to own the camera capture
loop (which it already does). Tracked as future work; nothing in `config.yaml`
needs to change when it lands.

**Phase 3 — Closed-loop policy evaluation in the HMI.** A policy path field,
a "deploy" button, the existing E-STOP wired into the policy loop, and a
small reward-labeling UI for SARM-style stage-aware reward modeling. See
the [LeRobot v0.5.0 release notes](https://huggingface.co/blog/lerobot-release-v050)
for what's available upstream.
