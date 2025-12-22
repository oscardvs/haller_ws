"""
Haller Robot Gazebo Simulation

Launches complete robot simulation including:
- Gazebo with configurable world
- Robot state publisher
- ros2_control with diff drive controller
- Vision pipeline (detection, segmentation, traversability)

The Gazebo camera plugin provides camera images, and the same vision
processing nodes run as on hardware for consistent behavior.

Usage:
    # Default simulation
    ros2 launch haller_gazebo haller_sim.launch.py

    # With specific world
    ros2 launch haller_gazebo haller_sim.launch.py world:=/path/to/world.world

    # Without vision (faster for testing locomotion)
    ros2 launch haller_gazebo haller_sim.launch.py enable_vision:=false
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
    pkg_gazebo = get_package_share_directory('haller_gazebo')
    pkg_description = get_package_share_directory('haller_description')
    pkg_controllers = get_package_share_directory('haller_controllers')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_vision = get_package_share_directory('haller_vision')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    enable_vision = LaunchConfiguration('enable_vision')
    enable_detection = LaunchConfiguration('enable_detection')
    enable_segmentation = LaunchConfiguration('enable_segmentation')

    # Robot description
    xacro_file = os.path.join(pkg_description, 'urdf', 'haller.urdf.xacro')
    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file,
            ' use_sim:=true',
            ' use_fake_hardware:=false',
            ' sim_gazebo:=true'
        ]),
        value_type=str
    )

    # Controller config
    controller_config = os.path.join(pkg_controllers, 'config', 'haller_controllers.yaml')

    # World file
    default_world = os.path.join(pkg_gazebo, 'worlds', 'empty.world')

    return LaunchDescription([
        # ==================== Launch Arguments ====================
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='World file to load'
        ),
        DeclareLaunchArgument(
            'enable_vision',
            default_value='true',
            description='Enable vision pipeline (detection, segmentation)'
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
        LogInfo(msg="[haller_gazebo] Starting Gazebo simulation..."),

        # ==================== Gazebo ====================
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
            ]),
            launch_arguments={
                'world': world,
                'verbose': 'true',
            }.items(),
        ),

        # ==================== Robot State Publisher ====================
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': use_sim_time
            }]
        ),

        # ==================== Spawn Robot ====================
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_entity',
            output='screen',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'haller',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.1',
            ]
        ),

        # ==================== Controllers ====================
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
            output='screen',
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
            output='screen',
        ),

        # ==================== Vision Pipeline ====================
        # In simulation, Gazebo camera plugin provides /camera/image_raw
        # The vision pipeline processes these images identically to hardware
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_vision, 'launch', 'vision_pipeline.launch.py')
            ]),
            launch_arguments={
                'use_sim': 'true',  # Simulation mode - no camera node needed
                'enable_detection': enable_detection,
                'enable_segmentation': enable_segmentation,
                'enable_traversability': 'true',
            }.items(),
            condition=IfCondition(enable_vision),
        ),
    ])
