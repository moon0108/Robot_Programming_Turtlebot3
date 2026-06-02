from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    waypoint_file = LaunchConfiguration('waypoint_file')
    pose_topic = LaunchConfiguration('pose_topic')
    map_frame = LaunchConfiguration('map_frame')
    ignore_goal_yaw = LaunchConfiguration('ignore_goal_yaw')

    return LaunchDescription([
        DeclareLaunchArgument(
            'waypoint_file',
            default_value='/root/maps/waypoints.yaml',
            description='YAML file used to store named waypoints.',
        ),
        DeclareLaunchArgument(
            'pose_topic',
            default_value='/amcl_pose',
            description='Current robot pose topic.',
        ),
        DeclareLaunchArgument(
            'map_frame',
            default_value='map',
            description='Frame used for navigation goals.',
        ),
        DeclareLaunchArgument(
            'ignore_goal_yaw',
            default_value='false',
            description=(
                'Use the current yaw instead of the saved waypoint yaw.'
            ),
        ),
        Node(
            package='waypoint_nav',
            executable='waypoint_gui',
            name='waypoint_gui',
            output='screen',
            parameters=[{
                'waypoint_file': waypoint_file,
                'pose_topic': pose_topic,
                'map_frame': map_frame,
                'ignore_goal_yaw': ParameterValue(
                    ignore_goal_yaw,
                    value_type=bool,
                ),
            }],
        ),
    ])
