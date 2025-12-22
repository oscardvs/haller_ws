"""
Full Vision Pipeline Launch

Launches the complete vision pipeline including camera, detection,
segmentation, and traversability analysis.

This launch file provides modular control via arguments:
- use_sim: Switch between hardware and simulation camera
- enable_detection: Enable/disable object detection
- enable_segmentation: Enable/disable semantic segmentation
- enable_traversability: Enable/disable traversability costmap

Usage:
    # Full pipeline on hardware
    ros2 launch haller_vision vision_pipeline.launch.py use_sim:=false

    # Full pipeline in simulation
    ros2 launch haller_vision vision_pipeline.launch.py use_sim:=true

    # Detection only (no segmentation)
    ros2 launch haller_vision vision_pipeline.launch.py enable_segmentation:=false
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
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
    enable_detection = LaunchConfiguration('enable_detection')
    enable_segmentation = LaunchConfiguration('enable_segmentation')
    enable_traversability = LaunchConfiguration('enable_traversability')
    
    # Config file paths
    camera_config = os.path.join(pkg_vision, 'config', 'camera', 'imx219_hardware.yaml')
    detection_config = os.path.join(pkg_vision, 'config', 'detection', 'yolov8_config.yaml')
    segmentation_config = os.path.join(pkg_vision, 'config', 'segmentation', 'segformer_config.yaml')
    traversability_config = os.path.join(pkg_vision, 'config', 'traversability', 'traversability.yaml')
    
    return LaunchDescription([
        # ==================== Launch Arguments ====================
        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Use simulation mode (Gazebo camera instead of hardware)'
        ),
        DeclareLaunchArgument(
            'enable_detection',
            default_value='true',
            description='Enable object detection node'
        ),
        DeclareLaunchArgument(
            'enable_segmentation',
            default_value='true',
            description='Enable semantic segmentation node'
        ),
        DeclareLaunchArgument(
            'enable_traversability',
            default_value='true',
            description='Enable traversability costmap generation'
        ),
        
        # ==================== Logging ====================
        LogInfo(
            condition=UnlessCondition(use_sim),
            msg="[haller_vision] Starting HARDWARE vision pipeline"
        ),
        LogInfo(
            condition=IfCondition(use_sim),
            msg="[haller_vision] Starting SIMULATION vision pipeline"
        ),
        
        # ==================== Camera Layer ====================
        # Hardware camera (only when use_sim=false)
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
        # Note: In simulation, Gazebo plugin provides camera topics
        
        # ==================== Detection Node ====================
        Node(
            condition=IfCondition(enable_detection),
            package='haller_vision',
            executable='detection_node.py',
            name='detection_node',
            parameters=[detection_config],
            output='screen',
        ),
        
        # ==================== Segmentation Node ====================
        Node(
            condition=IfCondition(enable_segmentation),
            package='haller_vision',
            executable='segmentation_node.py',
            name='segmentation_node',
            parameters=[segmentation_config],
            output='screen',
        ),
        
        # ==================== Traversability Node ====================
        Node(
            condition=IfCondition(enable_traversability),
            package='haller_vision',
            executable='traversability_node.py',
            name='traversability_node',
            parameters=[traversability_config],
            output='screen',
        ),
    ])

