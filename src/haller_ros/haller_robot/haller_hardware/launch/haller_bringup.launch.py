"""
Haller Robot Hardware Bringup

Launches all hardware components:
- Robot state publisher
- ros2_control with diff drive controller
- RPLIDAR LiDAR sensor
- Vision pipeline (IMX219 camera, detection, segmentation, traversability)

Usage:
    # Full robot bringup
    ros2 launch haller_hardware haller_bringup.launch.py

    # Without vision (for debugging)
    ros2 launch haller_hardware haller_bringup.launch.py enable_vision:=false

    # With fake hardware (simulation mode without Gazebo)
    ros2 launch haller_hardware haller_bringup.launch.py use_fake_hardware:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Get package directories
    pkg_hardware = get_package_share_directory('haller_hardware')
    pkg_description = get_package_share_directory('haller_description')
    pkg_controllers = get_package_share_directory('haller_controllers')
    pkg_vision = get_package_share_directory('haller_vision')

    # Launch arguments
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    enable_vision = LaunchConfiguration('enable_vision')
    enable_detection = LaunchConfiguration('enable_detection')
    enable_segmentation = LaunchConfiguration('enable_segmentation')

    # Robot description
    xacro_file = os.path.join(pkg_description, 'urdf', 'haller.urdf.xacro')
    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file,
            ' use_sim:=false',
            ' use_fake_hardware:=', use_fake_hardware,
            ' sim_gazebo:=false'
        ]),
        value_type=str
    )

    # Controller config
    controller_config = os.path.join(pkg_controllers, 'config', 'haller_controllers.yaml')

    # LiDAR config
    lidar_config = os.path.join(pkg_hardware, 'config', 'rplidar.yaml')

    return LaunchDescription([
        # ==================== Launch Arguments ====================
        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='false',
            description='Use fake hardware for testing'
        ),
        DeclareLaunchArgument(
            'enable_vision',
            default_value='true',
            description='Enable vision pipeline (camera, detection, segmentation)'
        ),
        DeclareLaunchArgument(
            'enable_detection',
            default_value='true',
            description='Enable object detection (requires enable_vision:=true)'
        ),
        DeclareLaunchArgument(
            'enable_segmentation',
            default_value='true',
            description='Enable semantic segmentation (requires enable_vision:=true)'
        ),

        # ==================== Logging ====================
        LogInfo(msg="[haller_hardware] Starting robot bringup..."),

        # ==================== Robot State Publisher ====================
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': False
            }]
        ),

        # ==================== ros2_control ====================
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                {'robot_description': robot_description},
                controller_config
            ],
            output='screen',
        ),

        # Joint State Broadcaster
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
            output='screen',
        ),

        # Diff Drive Controller
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
            output='screen',
        ),

        # ==================== LiDAR ====================
        Node(
            package='sllidar_ros2',
            executable='sllidar_node',
            name='sllidar_node',
            parameters=[lidar_config],
            output='screen',
        ),

        # ==================== Vision Pipeline ====================
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_vision, 'launch', 'vision_pipeline.launch.py')
            ]),
            launch_arguments={
                'use_sim': 'false',  # Hardware mode
                'enable_detection': enable_detection,
                'enable_segmentation': enable_segmentation,
                'enable_traversability': 'true',
            }.items(),
            condition=IfCondition(enable_vision),
        ),
    ])
