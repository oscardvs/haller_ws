# Haller Robot Workspace

A ROS2 workspace for the Haller mobile robot platform - a 3-wheeled differential-drive robot (2 driven front wheels + rear caster) equipped with camera modules and Slamtech 2D LiDAR, running on NVIDIA Jetson Orin Nano.

## Hardware Specifications

- **Compute**: NVIDIA Jetson Orin Nano (developer kit)
- **Drive**: differential drive: 2 driven front wheels (MF5010 BLDC over CAN) + rear caster (3 wheels total)
- **Sensors**:
  - Slamtech 2D LiDAR (RPLIDAR A1M8)
  - Camera modules (IMX219)
- **Power**: Hilti B 22-195 Nuron battery (21.6 V nominal, 9.0 Ah, 194.4 Wh)
- **Communication**: ROS2 Humble

Hardware documentation:

- [`docs/hardware_inventory.md`](../docs/hardware_inventory.md): parts list,
  what is confirmed vs unknown, and the shopping list
- [`docs/power_system.md`](../docs/power_system.md): power architecture,
  battery specifications, connector pinout, load budget, DC-DC requirements,
  fusing, and state-of-charge thresholds

## Package Structure

```
haller_ws/src/
├── haller_ros/                    # Core ROS2 packages
│   ├── haller_common/             # Shared packages
│   │   ├── haller_controllers/    # Controller configurations
│   │   ├── haller_description/    # URDF/Xacro robot description
│   │   ├── haller_hardware_interface/  # ros2_control hardware interface
│   │   ├── haller_msgs/           # Custom message definitions
│   │   └── haller_utils/          # Utility nodes and tools
│   ├── haller_robot/              # Robot-specific packages
│   │   └── haller_hardware/       # Hardware driver nodes
│   └── haller_simulator/          # Simulation packages
│       └── haller_gazebo/         # Gazebo simulation
├── haller_navigation/             # Navigation stack (Nav2)
└── haller_scanning/               # Scanning and perception
```

## Prerequisites

- ROS2 Humble Hawksbill
- Gazebo Fortress (for simulation)
- NVIDIA JetPack SDK (for Jetson deployment)

### Install Dependencies

```bash
# Install ROS2 dependencies
sudo apt update
sudo apt install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-gazebo-ros2-control \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-xacro \
    ros-humble-joint-state-publisher-gui

# Install rosdep dependencies
cd ~/haller_ws
rosdep install --from-paths src --ignore-src -r -y
```

### LiDAR Setup (Real Robot Only)

The Slamtec RPLIDAR A1M8 requires udev rules for USB permissions. Run this once:

```bash
# Copy udev rules
sudo cp ~/haller_ws/src/sllidar_ros2/scripts/rplidar.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

# Verify the lidar is detected (after connecting)
ls -la /dev/ttyUSB*
```

## Building

```bash
cd ~/haller_ws
colcon build --symlink-install
source install/setup.bash
```

## Usage

### Simulation

In simulation, the LiDAR is provided by a Gazebo plugin - no physical hardware needed.

```bash
# Launch Gazebo simulation with robot (includes simulated LiDAR)
ros2 launch haller_gazebo haller_sim.launch.py

# In another terminal - launch navigation
ros2 launch haller_navigation navigation.launch.py use_sim_time:=true
```

### Real Robot

On the Jetson Orin with hardware connected:

```bash
# Launch hardware drivers (includes RPLIDAR A1M8)
ros2 launch haller_hardware haller_bringup.launch.py

# In another terminal - launch navigation
ros2 launch haller_navigation navigation.launch.py
```

To test only the LiDAR:

```bash
# Launch just the lidar node
ros2 launch sllidar_ros2 sllidar_a1_launch.py

# Verify scan data
ros2 topic echo /scan --once
```

### Visualization

```bash
# View robot model in RViz (requires display)
ros2 launch haller_description display.launch.py

# For headless Jetson, run RViz on a remote machine on the same network
```

## Development

### Code Style

- Follow ROS2 coding guidelines
- Use `ament_cmake` for C++ packages
- Use `ament_python` for Python packages

### Testing

```bash
colcon test --packages-select <package_name>
colcon test-result --verbose
```

## License

Apache-2.0

## Authors

- Oscar Devos

