# Haller

Haller is an open-source mobile-manipulation robot: a four-wheeled differential-drive base that carries **two SO-101 arms** for bimanual manipulation. This repository (`haller_ws`) is the umbrella codebase — ROS 2 stack for the base, LeRobot integration for the arms, scripts for deployment, and documentation for reproducing the build.

> **Status (May 2026):** mobile base operational under ROS 2; first SO-101 arm assembled and being brought up via LeRobot.

## Hardware overview

| Subsystem        | Components                                                              |
| ---------------- | ----------------------------------------------------------------------- |
| Compute          | NVIDIA Jetson Orin Nano                                                 |
| Mobile base      | 4-wheel differential drive, LK-TECH MF5010 BLDC motors over CAN          |
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
│   │   └── so101-arm.md             ← SO-101 motor configuration + calibration
│   └── *.pdf                        ← LK-TECH MF5010 manuals (drive motors)
├── scripts/                         ← provisioning, services, udev rules
│   ├── install.sh
│   ├── haller_bringup.sh
│   ├── 99-haller-devices.rules      ← udev rules (stable device names)
│   ├── haller-robot.service         ← systemd unit, robot bringup on boot
│   ├── haller-ap.service            ← systemd unit, Wi-Fi AP fallback
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

## License

Apache-2.0. See `src/README.md` for the source attribution; the SO-101 mechanical design is licensed under its own terms by TheRobotStudio.

## Authors

- Oscar Devos
