"""
Simulation Camera Launch

In simulation mode, the Gazebo camera plugin already publishes to /camera/* topics.
This launch file is a placeholder that ensures the camera interface is consistent.

No nodes are started here - Gazebo handles camera simulation.
"""

from launch import LaunchDescription
from launch.actions import LogInfo


def generate_launch_description():
    return LaunchDescription([
        LogInfo(
            msg="[haller_vision] Simulation mode: Camera provided by Gazebo plugin"
        ),
        # No camera node needed - Gazebo plugin publishes to:
        # - /camera/image_raw
        # - /camera/camera_info
    ])

