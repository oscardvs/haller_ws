"""
Unified Camera Launch

Automatically selects between hardware (IMX219) and simulation camera
based on the use_sim argument.

Usage:
    # Hardware camera
    ros2 launch haller_vision camera.launch.py use_sim:=false
    
    # Simulation (Gazebo provides camera)
    ros2 launch haller_vision camera.launch.py use_sim:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_vision = get_package_share_directory('haller_vision')
    
    # Launch arguments
    use_sim = LaunchConfiguration('use_sim')
    
    # Camera configuration for hardware
    camera_config = os.path.join(
        pkg_vision, 'config', 'camera', 'imx219_hardware.yaml'
    )
    
    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Use simulation mode (Gazebo camera)'
        ),
        
        # Log which mode we're in
        LogInfo(
            condition=UnlessCondition(use_sim),
            msg="[haller_vision] Starting hardware camera (IMX219)"
        ),
        LogInfo(
            condition=IfCondition(use_sim),
            msg="[haller_vision] Simulation mode: using Gazebo camera"
        ),
        
        # Hardware camera node (only when use_sim=false)
        Node(
            condition=UnlessCondition(use_sim),
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
        
        # In simulation mode, Gazebo plugin provides the camera
        # No additional nodes needed
    ])

