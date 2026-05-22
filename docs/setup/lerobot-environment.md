# LeRobot environment setup

This guide reproduces the Python environment used by Haller's SO-101 arms. It follows the official LeRobot install ([huggingface.co/docs/lerobot/installation](https://huggingface.co/docs/lerobot/installation)) with two Haller-specific patches that protect the env from this machine's ROS 2 install and from the user-site site-packages directory.

**Target host:** Ubuntu 24.04, ROS 2 Jazzy already installed at `/opt/ros/jazzy`.
**Result:** a conda env named `lerobot` with PyTorch 2.11 (CUDA 13), LeRobot 0.5.x, and the Feetech servo SDK.

---

## 1. Install Miniforge (conda)

```bash
cd /tmp
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
exec $SHELL          # pick up the new ~/.bashrc hooks
```

Miniforge is preferred over full Anaconda: it's smaller, defaults to `conda-forge`, and is MIT-licensed.

## 2. Create the env

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y ffmpeg -c conda-forge
```

Verify `ffmpeg` ships the `libsvtav1` encoder (LeRobot's video pipeline expects it):

```bash
ffmpeg -encoders | grep svtav1
# →  V..... libsvtav1            SVT-AV1 ... encoder
```

## 3. Isolate the env from ROS and user-site (Haller-specific)

ROS 2 Jazzy adds `/opt/ros/jazzy/lib/python3.12/site-packages` to `PYTHONPATH` when sourced from `~/.bashrc`. That path lands **first** in `sys.path` once you `conda activate`, so ROS's Python packages shadow the env's. Separately, `~/.local/lib/python3.12/site-packages` (where `pip install --user` writes) is added by Python's `site.py` ahead of the env, and on this host it contains an outdated `nvidia-nccl-cu12` that breaks PyTorch's CUDA load (`undefined symbol: ncclDevCommDestroy`).

Two activate-hooks fix both problems for this env only:

```bash
mkdir -p ~/miniforge3/envs/lerobot/etc/conda/activate.d
mkdir -p ~/miniforge3/envs/lerobot/etc/conda/deactivate.d

cat > ~/miniforge3/envs/lerobot/etc/conda/activate.d/zz_unset_pythonpath.sh <<'EOF'
#!/usr/bin/env bash
if [ -n "${PYTHONPATH:-}" ]; then
    export _LEROBOT_SAVED_PYTHONPATH="$PYTHONPATH"
    unset PYTHONPATH
fi
if [ -n "${PYTHONNOUSERSITE:-}" ]; then
    export _LEROBOT_SAVED_PYTHONNOUSERSITE="$PYTHONNOUSERSITE"
fi
export PYTHONNOUSERSITE=1
EOF

cat > ~/miniforge3/envs/lerobot/etc/conda/deactivate.d/zz_restore_pythonpath.sh <<'EOF'
#!/usr/bin/env bash
if [ -n "${_LEROBOT_SAVED_PYTHONPATH:-}" ]; then
    export PYTHONPATH="$_LEROBOT_SAVED_PYTHONPATH"
    unset _LEROBOT_SAVED_PYTHONPATH
fi
if [ -n "${_LEROBOT_SAVED_PYTHONNOUSERSITE:-}" ]; then
    export PYTHONNOUSERSITE="$_LEROBOT_SAVED_PYTHONNOUSERSITE"
    unset _LEROBOT_SAVED_PYTHONNOUSERSITE
else
    unset PYTHONNOUSERSITE
fi
EOF
```

Reactivate the env to load the hooks:

```bash
conda deactivate && conda activate lerobot
echo "PYTHONPATH=[$PYTHONPATH]"          # → PYTHONPATH=[]
echo "PYTHONNOUSERSITE=[$PYTHONNOUSERSITE]"  # → PYTHONNOUSERSITE=[1]
```

`sys.path` from inside the env should contain only the env's own directories plus `~/lerobot/src` (added by the editable install in the next step), and no `/opt/ros/...` or `~/.local/...` entries.

## 4. Install LeRobot with the Feetech extra

```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
pip install -e ".[feetech]"
```

The `[feetech]` extra pulls in `feetech-servo-sdk` (the Python module name is `scservo_sdk`), which talks to the STS3215 servos on the SO-101.

## 5. Verify

```bash
python - <<'PY'
import lerobot, torch, torchvision, scservo_sdk, huggingface_hub
print("lerobot           OK")
print("torch            ", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("torchvision      ", torchvision.__version__)
print("scservo_sdk      OK")
print("huggingface_hub  ", huggingface_hub.__version__)
PY
```

You should see something like:

```
lerobot            OK
torch             2.11.0+cu130 | CUDA: True
torchvision       0.26.0+cu130
scservo_sdk       OK
huggingface_hub   1.16.1
```

Also confirm the LeRobot CLI entry points are on PATH:

```bash
which lerobot-find-port lerobot-setup-motors lerobot-calibrate lerobot-teleoperate
```

## Next

Once the env is verified, move on to [`so101-arm.md`](./so101-arm.md) to configure the motors and calibrate the arm.

## Notes & troubleshooting

- **`pip check` reports unrelated conflicts.** `pip check` walks every Python install on the box, not just the active env. Conflicts about `kubernetes`, `google-genai`, `markitdown`, etc. come from `~/.local/` or `/usr/lib/python3.12/dist-packages/` and don't affect this env.
- **`torch.cuda.is_available()` returns `False`.** Confirm the NVIDIA driver is installed (`nvidia-smi`). The env brings its own CUDA runtime libraries via `nvidia-*` pip wheels, so a system CUDA install is not required.
- **`ImportError: undefined symbol: ncclDevCommDestroy` when importing torch.** The activate hooks above (`PYTHONNOUSERSITE=1`) must be in place. If you skipped step 3 you'll hit this every time.
- **Forgetting to activate the env.** A naked `pip install` from a non-activated shell on this host will sometimes silently satisfy deps from `~/.local/`. Always `conda activate lerobot` first.
