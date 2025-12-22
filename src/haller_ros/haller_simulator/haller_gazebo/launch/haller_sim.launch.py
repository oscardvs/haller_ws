import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
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

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')

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
        # Launch arguments
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

        # Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
            ]),
            launch_arguments={
                'world': world,
                'verbose': 'true',
            }.items(),
        ),

        # Robot State Publisher
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

        # Spawn robot in Gazebo
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
    ])

