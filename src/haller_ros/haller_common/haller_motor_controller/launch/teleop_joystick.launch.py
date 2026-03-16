import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('haller_motor_controller')
    joystick_config = os.path.join(pkg_dir, 'config', 'joystick.yaml')

    return LaunchDescription([
        # Joystick driver
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'deadzone': 0.1,
                'autorepeat_rate': 20.0,
            }],
            output='screen',
        ),

        # Twist mux from joystick
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_twist_joy_node',
            parameters=[joystick_config],
            output='screen',
        ),

        # Motor controller
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'motor_controller.launch.py')
            ),
        ),
    ])
