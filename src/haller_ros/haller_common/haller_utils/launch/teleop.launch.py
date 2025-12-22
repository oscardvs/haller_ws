from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'linear_speed',
            default_value='0.5',
            description='Maximum linear speed'
        ),
        DeclareLaunchArgument(
            'angular_speed',
            default_value='1.0',
            description='Maximum angular speed'
        ),

        Node(
            package='haller_utils',
            executable='teleop_keyboard.py',
            name='teleop_keyboard',
            output='screen',
            prefix='xterm -e',
            parameters=[{
                'linear_speed': LaunchConfiguration('linear_speed'),
                'angular_speed': LaunchConfiguration('angular_speed'),
            }]
        ),
    ])

