"""
Hardware Camera Launch (IMX219 on Jetson Orin)

This launch file starts the V4L2 camera driver for the IMX219 CSI camera.
Used internally by camera.launch.py when use_sim:=false
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_vision = get_package_share_directory('haller_vision')
    
    # Camera configuration
    camera_config = os.path.join(
        pkg_vision, 'config', 'camera', 'imx219_hardware.yaml'
    )
    
    return LaunchDescription([
        # V4L2 Camera Node for IMX219
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='camera',
            namespace='',
            parameters=[camera_config],
            remappings=[
                ('image_raw', '/camera/image_raw'),
                ('camera_info', '/camera/camera_info'),
            ],
            output='screen',
        ),
    ])

