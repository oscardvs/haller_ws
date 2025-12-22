import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    pkg_navigation = get_package_share_directory('haller_navigation')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')

    # Config files
    default_params = os.path.join(pkg_navigation, 'config', 'nav2_params.yaml')
    default_map = os.path.join(pkg_navigation, 'maps', 'map.yaml')
    rviz_config = os.path.join(pkg_navigation, 'rviz', 'navigation.rviz')

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Full path to map yaml file'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Full path to Nav2 params file'
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Automatically start Nav2 stack'
        ),

        # Nav2 Bringup
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_nav2_bringup, 'launch', 'bringup_launch.py')
            ]),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'map': map_file,
                'params_file': params_file,
                'autostart': autostart,
            }.items(),
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),
    ])

