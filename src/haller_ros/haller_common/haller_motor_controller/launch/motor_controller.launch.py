import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('haller_motor_controller')
    params_file = os.path.join(pkg_dir, 'config', 'motor_params.yaml')

    return LaunchDescription([
        Node(
            package='haller_motor_controller',
            executable='motor_controller_node',
            name='motor_controller',
            parameters=[params_file],
            output='screen',
        ),
    ])
