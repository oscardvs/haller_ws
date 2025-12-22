"""
Hardware Camera Launch (IMX219 on Jetson Orin via GStreamer)

This launch file starts the gscam driver for the IMX219 CSI camera.
Uses GStreamer pipeline for Bayer-to-RGB conversion.

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
        # GStreamer Camera Node for IMX219
        Node(
            package='gscam',
            executable='gscam_node',
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

