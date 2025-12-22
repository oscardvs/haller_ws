import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_scanning = get_package_share_directory('haller_scanning')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),

        # Scan Processor
        Node(
            package='haller_scanning',
            executable='scan_processor.py',
            name='scan_processor',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'min_range': 0.15,
                'max_range': 12.0,
                'filter_window': 3,
            }]
        ),

        # Obstacle Detector
        Node(
            package='haller_scanning',
            executable='obstacle_detector.py',
            name='obstacle_detector',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'warning_distance': 0.5,
                'critical_distance': 0.3,
                'front_angle': 1.0,
            }]
        ),
    ])

