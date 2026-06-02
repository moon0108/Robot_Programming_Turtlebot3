import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.actions import SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def spawn_burger_cam(context, burger_cam_sdf):
    start_pose = context.launch_configurations.get('start_pose', 'center')
    y_override = context.launch_configurations.get('y_pose', '')
    x_pose = context.launch_configurations.get('x_pose', '-0.5')
    preset_y = {
        'left': '0.2',
        'right': '-0.2',
        'center': '0.0',
    }
    y_pose = y_override if y_override else preset_y[start_pose]

    return [
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'burger_cam',
                '-file', burger_cam_sdf,
                '-x', x_pose,
                '-y', y_pose,
                '-z', '0.01',
            ],
            output='screen',
        )
    ]


def generate_launch_description():
    package_share = get_package_share_directory('marker_comm')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    world_path = os.path.join(package_share, 'worlds', 'aruco_alignment.world')
    model_path = os.path.join(package_share, 'models')
    turtlebot3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
    turtlebot3_source_share = '/home/min/colcon_ws/src/turtlebot3_simulations/turtlebot3_gazebo'
    turtlebot3_model_path = first_existing_path(
        os.path.join(turtlebot3_gazebo_share, 'models'),
        os.path.join(turtlebot3_source_share, 'models'),
    )
    burger_cam_sdf = os.path.join(model_path, 'turtlebot3_burger_cam_aruco', 'model.sdf')
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_path}.items(),
    )

    aruco_pose = Node(
        package='marker_comm',
        executable='aruco_pose',
        output='screen',
        condition=IfCondition(LaunchConfiguration('run_aruco_pose')),
        parameters=[{
            'image_topic': '/camera/image_raw',
            'camera_info_topic': '/camera/camera_info',
            'marker_size': 0.03,
            'marker_ids': '0,1',
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_pose',
            default_value='center',
            choices=['left', 'center', 'right'],
        ),
        DeclareLaunchArgument('x_pose', default_value='0.5'),
        DeclareLaunchArgument('y_pose', default_value=''),
        DeclareLaunchArgument('run_aruco_pose', default_value='true'),
        SetEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=os.pathsep.join(
                path for path in [model_path, turtlebot3_model_path, existing_model_path] if path
            ),
        ),
        gazebo,
        OpaqueFunction(function=spawn_burger_cam, args=[burger_cam_sdf]),
        aruco_pose,
    ])
