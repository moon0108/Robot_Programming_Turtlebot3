from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ros_domain_id = LaunchConfiguration('ros_domain_id')
    ros_localhost_only = LaunchConfiguration('ros_localhost_only')
    compressed_image_topic = LaunchConfiguration('compressed_image_topic')
    raw_image_topic = LaunchConfiguration('raw_image_topic')
    model_path = LaunchConfiguration('model_path')
    input_size = LaunchConfiguration('input_size')
    timer_period = LaunchConfiguration('timer_period')
    device = LaunchConfiguration('device')
    conf_threshold = LaunchConfiguration('conf_threshold')
    iou_threshold = LaunchConfiguration('iou_threshold')
    exclude_class_names = LaunchConfiguration('exclude_class_names')
    launch_rqt = LaunchConfiguration('launch_rqt')

    return LaunchDescription([
        DeclareLaunchArgument(
            'ros_domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='15'),
        ),
        DeclareLaunchArgument(
            'ros_localhost_only',
            default_value=EnvironmentVariable('ROS_LOCALHOST_ONLY', default_value='0'),
        ),
        DeclareLaunchArgument(
            'compressed_image_topic',
            default_value='/camera/image_raw/compressed',
        ),
        DeclareLaunchArgument('raw_image_topic', default_value='/camera/image_decompressed'),
        DeclareLaunchArgument(
            'model_path',
            default_value='/root/robotpro.pt',
        ),
        DeclareLaunchArgument('input_size', default_value='640'),
        DeclareLaunchArgument('timer_period', default_value='0.2'),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('conf_threshold', default_value='0.85'),
        DeclareLaunchArgument('iou_threshold', default_value='0.45'),
        DeclareLaunchArgument('exclude_class_names', default_value='Pepero'),
        DeclareLaunchArgument('launch_rqt', default_value='false'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', ros_domain_id),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', ros_localhost_only),

        Node(
            package='image_transport',
            executable='republish',
            name='republish_compressed_to_raw',
            output='screen',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', compressed_image_topic),
                ('out', raw_image_topic),
            ],
        ),

        Node(
            package='yolo_v8_ros',
            executable='yolo_node',
            name='yolo_v8_ros_node',
            output='screen',
            parameters=[
                {'image_topic': raw_image_topic},
                {'model_path': model_path},
                {'input_size': ParameterValue(input_size, value_type=int)},
                {'timer_period': ParameterValue(timer_period, value_type=float)},
                {'device': ParameterValue(device, value_type=str)},
                {'conf_threshold': ParameterValue(conf_threshold, value_type=float)},
                {'iou_threshold': ParameterValue(iou_threshold, value_type=float)},
                {'exclude_class_names': ParameterValue(exclude_class_names, value_type=str)},
                {'display_window': False},
                {'publish_annotated_image': True},
                {'annotated_image_topic': '/yolo/image'},
            ],
        ),

        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_image_view',
            output='screen',
            condition=IfCondition(launch_rqt),
        ),
    ])
