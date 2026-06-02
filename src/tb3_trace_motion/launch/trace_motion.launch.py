from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    motion_file = LaunchConfiguration('motion_file')
    odom_topic = LaunchConfiguration('odom_topic')
    use_ui = LaunchConfiguration('use_ui')

    executor = Node(
        package='tb3_trace_motion',
        executable='motion_executor',
        name='tb3_motion_executor',
        output='screen',
        parameters=[{
            'motion_file': motion_file,
            'odom_topic': odom_topic,
        }],
    )

    ui = Node(
        package='tb3_trace_motion',
        executable='motion_ui',
        name='tb3_motion_ui',
        output='screen',
        parameters=[{
            'motion_file': motion_file,
        }],
        condition=IfCondition(use_ui),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'motion_file',
            default_value='/root/maps/tb3_motion_paths.yaml',
            description='YAML file storing trace motion paths.',
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Odometry topic used for distance and yaw feedback.',
        ),
        DeclareLaunchArgument(
            'use_ui',
            default_value='true',
            description='Launch the Qt editor.',
        ),
        executor,
        ui,
    ])
