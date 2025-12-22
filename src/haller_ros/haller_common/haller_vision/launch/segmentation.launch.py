"""
Segmentation-Only Launch

Launches camera and semantic segmentation without detection.
Useful for testing segmentation and traversability.

Usage:
    # Hardware
    ros2 launch haller_vision segmentation.launch.py use_sim:=false

    # Simulation
    ros2 launch haller_vision segmentation.launch.py use_sim:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_vision = get_package_share_directory('haller_vision')
    
    use_sim = LaunchConfiguration('use_sim')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Use simulation mode'
        ),
        
        # Include full pipeline with only segmentation enabled
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_vision, 'launch', 'vision_pipeline.launch.py')
            ]),
            launch_arguments={
                'use_sim': use_sim,
                'enable_detection': 'false',
                'enable_segmentation': 'true',
                'enable_traversability': 'true',
            }.items(),
        ),
    ])

