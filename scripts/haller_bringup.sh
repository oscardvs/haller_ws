#!/bin/bash
# Haller Robot Bringup Script
# Starts all ROS nodes for the robot

set -e

# Source ROS
source /opt/ros/humble/setup.bash
source /home/orin/haller_ws/install/setup.bash

# Wait for USB devices
echo "[haller] Waiting for devices..."
for i in $(seq 1 10); do
    if [ -e /dev/haller_lidar ] && [ -e /dev/haller_can ]; then
        echo "[haller] All USB devices found"
        break
    fi
    sleep 1
done

# Launch the robot
echo "[haller] Launching ROS nodes..."
exec ros2 launch haller_hardware haller_bringup.launch.py enable_vision:=false
