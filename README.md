# Haller

Haller is an open-source mobile-manipulation robot: a three-wheeled differential-drive base (two driven front wheels + a rear caster) that carries **two SO-101 arms** for bimanual manipulation. This repository (`haller_ws`) is the umbrella codebase — ROS 2 stack for the base, LeRobot integration for the arms, scripts for deployment, and documentation for reproducing the build.

> **Status (August 2026):** the working rig is the arms. One or both SO-101 arms run through the unified HMI (FastAPI + Next.js + shadcn): per-arm controls (joint sliders, home, free-drive, pose presets) and an in-browser calibration wizard (homing + range-of-motion sweep + save, with automatic backup). **Teleop is WebXR from a Meta Quest** — one input path since the 2026-08-22 unification, which deleted the MediaPipe webcam pipeline and the body-angle modes; see [`hmi/README.md`](./hmi/README.md). Live MJPEG camera streams and an **HMI-integrated dataset recorder** are wired — the recorder runs inside the HMI (start/stop from the cockpit or from inside the headset), because Haller's two arms are both *followers* and stock `lerobot-record` structurally cannot capture a two-arm demo on this hardware. See [`docs/setup/dataset-collection.md`](./docs/setup/dataset-collection.md). Three MuJoCo sim presets (solo, bimanual, leader+follower) drop into the same HMI surface, with seeded per-episode scene reset, domain randomization and an automatic task-success predicate, so a rehearsal needs no hardware; see [`docs/setup/sim.md`](./docs/setup/sim.md). The mobile base's ROS 2 stack is in the tree but is mid-migration to Jazzy / JetPack 7 and is not part of that rig today. Public datasets to bootstrap from: [`docs/setup/public-datasets.md`](./docs/setup/public-datasets.md).

## Hardware overview

| Subsystem        | Components                                                              |
| ---------------- | ----------------------------------------------------------------------- |
| Compute          | NVIDIA Jetson Orin Nano                                                 |
| Mobile base      | Differential drive — 2 driven front wheels + rear caster, LK-TECH MF5010 BLDC motors over CAN |
| Perception       | Slamtec RPLIDAR A1M8 (2D LiDAR), camera modules                          |
| Arms             | 2× SO-ARM101 ("SO-101") follower arms with Feetech STS3215 servos        |
| Servo bus        | Feetech bus servo adapter board (USB ↔ TTL half-duplex daisy-chain)      |
| Networking       | Wi-Fi access point fallback (`scripts/setup_ap.sh`)                      |

The SO-101 hardware design is from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100). All other hardware references live in [`docs/`](./docs).

## Repository layout

```
haller_ws/
├── README.md                        ← you are here
├── CLAUDE.md                        ← repo-level rules for the Claude Code agent
├── docs/                            ← vendor datasheets + setup guides
│   ├── setup/
│   │   ├── lerobot-environment.md   ← Python/conda env for the arms
│   │   ├── so101-arm.md             ← SO-101 motor configuration + calibration
│   │   ├── dataset-collection.md    ← record bimanual datasets + push to HF Hub
│   │   ├── public-datasets.md       ← public SO-101 datasets to bootstrap from
│   │   └── runpod-inference.md      ← cloud-GPU inference + LoRA finetune (π0.5, GR00T, …)
│   └── *.pdf                        ← LK-TECH MF5010 manuals (drive motors)
├── scripts/                         ← provisioning, services, udev rules
│   ├── install.sh
│   ├── haller_bringup.sh
│   ├── 99-haller-devices.rules      ← udev rules (stable device names)
│   ├── haller-robot.service         ← systemd unit, robot bringup on boot
│   ├── haller-ap.service            ← systemd unit, Wi-Fi AP fallback
│   ├── record_dataset.sh            ← wrapper around lerobot-record (Phase 1 data collection)
│   ├── runpod/                      ← cloud-GPU recipes (setup, smoke test, replay eval, LoRA finetune)
│   └── setup_ap.sh
├── src/                             ← ROS 2 colcon workspace
│   ├── haller_ros/                  ← core ROS 2 packages
│   │   ├── haller_common/           ←   controllers, description, hw iface, msgs, utils
│   │   ├── haller_robot/            ←   robot-specific hardware drivers
│   │   └── haller_simulator/        ←   Gazebo simulation
│   ├── haller_navigation/           ← Nav2 stack
│   ├── haller_scanning/             ← scanning / perception
│   ├── sllidar_ros2/                ← submodule: Slamtec LiDAR driver
│   └── README.md                    ← detailed ROS 2 workspace notes
└── test.py, can_test.py             ← CAN bench scripts for the drive motors
```

## Getting started

The project has two largely independent software stacks. You can bring them up in either order.

### 1. Mobile base (ROS 2)

See [`src/README.md`](./src/README.md) for the full nav-stack bringup. Quick path:

```bash
# clone with submodules
git clone --recurse-submodules https://github.com/oscardvs/haller_ws.git
cd haller_ws

# install ROS 2 deps (ROS 2 Jazzy on Ubuntu 24.04; the older src/README references Humble — that's stale)
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install
source install/setup.bash
ros2 launch haller_gazebo haller_sim.launch.py   # simulation
```

### 2. Arms (LeRobot + SO-101)

Two guides, run them in order:

1. **[`docs/setup/lerobot-environment.md`](./docs/setup/lerobot-environment.md)** — install Miniforge, create the `lerobot` conda env, install LeRobot with the Feetech extra, and patch the env so it isn't poisoned by ROS's `PYTHONPATH` or the user-site directory.
2. **[`docs/setup/so101-arm.md`](./docs/setup/so101-arm.md)** — find the bus servo adapter's serial port, configure each motor's ID and baud rate one-by-one, wire the arm, calibrate, and run a smoke test.
3. **[`hmi/README.md`](./hmi/README.md)** — bring up the unified HMI (FastAPI backend + Next.js + shadcn frontend) that replaces the legacy `web_teleop.py`.
4. **[`docs/setup/dataset-collection.md`](./docs/setup/dataset-collection.md)** — wire your cameras, record a 12-dim bimanual teleop dataset with the HMI recorder, push to the Hugging Face Hub. Prerequisite for training or finetuning a policy on your own task. See also [`docs/setup/public-datasets.md`](./docs/setup/public-datasets.md) for public SO-101 datasets you can train on before you have your own.
5. **[`docs/setup/runpod-inference.md`](./docs/setup/runpod-inference.md)** — rent a cloud GPU on RunPod, run π0.5 / GR00T inference against your dataset, and LoRA-finetune on top. The whole "see what a generalist VLA does on my data" flow.

## License

Apache-2.0. See `src/README.md` for the source attribution; the SO-101 mechanical design is licensed under its own terms by TheRobotStudio.

## Authors

- Oscar Devos
