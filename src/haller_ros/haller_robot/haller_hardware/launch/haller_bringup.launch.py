import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Get package directories
    pkg_hardware = get_package_share_directory('haller_hardware')
    pkg_description = get_package_share_directory('haller_description')
    pkg_controllers = get_package_share_directory('haller_controllers')

    # Launch arguments
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')

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
        # Launch arguments
        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='false',
            description='Use fake hardware for testing'
        ),

        # Robot State Publisher
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

        # Controller Manager
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

        # RPLIDAR Node (Slamtec A1M8)
        Node(
            package='sllidar_ros2',
            executable='sllidar_node',
            name='sllidar_node',
            parameters=[lidar_config],
            output='screen',
        ),
    ])

