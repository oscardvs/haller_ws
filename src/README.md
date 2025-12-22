# Haller Robot Workspace

A ROS2 workspace for the Haller mobile robot platform - a 4-wheeled differential drive robot equipped with camera modules and Slamtech 2D LiDAR, running on NVIDIA Jetson Orin Nano.

## Hardware Specifications

- **Compute**: NVIDIA Jetson Orin Nano
- **Drive**: 4-wheel differential drive
- **Sensors**:
  - Slamtech 2D LiDAR (RPLIDAR)
  - Camera modules
- **Communication**: ROS2 Humble

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
    ros-humble-rplidar-ros \
    ros-humble-xacro \
    ros-humble-joint-state-publisher-gui

# Install rosdep dependencies
cd ~/haller_ws
rosdep install --from-paths src --ignore-src -r -y
```

## Building

```bash
cd ~/haller_ws
colcon build --symlink-install
source install/setup.bash
```

## Usage

### Simulation

```bash
# Launch Gazebo simulation with robot
ros2 launch haller_gazebo haller_sim.launch.py

# In another terminal - launch navigation
ros2 launch haller_navigation navigation.launch.py use_sim_time:=true
```

### Real Robot

```bash
# Launch hardware drivers
ros2 launch haller_hardware haller_bringup.launch.py

# Launch navigation
ros2 launch haller_navigation navigation.launch.py
```

### Visualization

```bash
# View robot model in RViz
ros2 launch haller_description display.launch.py
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

