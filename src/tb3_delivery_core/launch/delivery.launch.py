import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_gui = LaunchConfiguration('use_gui')
    use_motion_ui = LaunchConfiguration('use_motion_ui')
    launch_aruco = LaunchConfiguration('launch_aruco')
    manage_aruco_processes = LaunchConfiguration('manage_aruco_processes')
    launch_yolo = LaunchConfiguration('launch_yolo')
    yolo_continuous_detection = LaunchConfiguration('yolo_continuous_detection')
    launch_yolo_view = LaunchConfiguration('launch_yolo_view')
    launch_camera_compressed_to_raw = LaunchConfiguration('launch_camera_compressed_to_raw')
    launch_nav2 = LaunchConfiguration('launch_nav2')
    launch_motion_executor = LaunchConfiguration('launch_motion_executor')
    motion_file = LaunchConfiguration('motion_file')
    waypoints_file = LaunchConfiguration('waypoints_file')
    mission_params_file = LaunchConfiguration('mission_params_file')
    aruco_params_file = LaunchConfiguration('aruco_params_file')
    map_file = LaunchConfiguration('map')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    camera_compressed_topic = LaunchConfiguration('camera_compressed_topic')
    camera_raw_topic = LaunchConfiguration('camera_raw_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    yolo_conf_threshold = LaunchConfiguration('yolo_conf_threshold')
    yolo_iou_threshold = LaunchConfiguration('yolo_iou_threshold')
    yolo_exclude_class_names = LaunchConfiguration('yolo_exclude_class_names')

    tb3_delivery_share = get_package_share_directory('tb3_delivery_core')
    marker_comm_share = get_package_share_directory('marker_comm')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')
    turtlebot3_navigation_share = get_package_share_directory(
        'turtlebot3_navigation2'
    )

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_share, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_file,
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_file,
        }.items(),
        condition=IfCondition(launch_nav2),
    )

    compressed_to_raw_republish = Node(
        package='image_transport',
        executable='republish',
        name='republish_compressed_to_raw',
        output='screen',
        arguments=['compressed', 'raw'],
        remappings=[
            ('in/compressed', camera_compressed_topic),
            ('out', camera_raw_topic),
        ],
        condition=IfCondition(launch_camera_compressed_to_raw),
    )

    yolo_detector = Node(
        package='yolo_v8_ros',
        executable='yolo_node',
        name='yolo_v8_ros_node',
        output='screen',
        parameters=[{
            'image_topic': camera_raw_topic,
            'model_path': '/root/robotpro.pt',
            'device': 'cpu',
            'conf_threshold': ParameterValue(yolo_conf_threshold, value_type=float),
            'iou_threshold': ParameterValue(yolo_iou_threshold, value_type=float),
            'exclude_class_names': ParameterValue(yolo_exclude_class_names, value_type=str),
            'display_window': False,
            'publish_annotated_image': True,
            'annotated_image_topic': '/yolo/image',
            'publish_best_object': True,
            'continuous_detection': ParameterValue(
                yolo_continuous_detection,
                value_type=bool,
            ),
            'keep_image_subscription': ParameterValue(
                yolo_continuous_detection,
                value_type=bool,
            ),
            'snapshot_wait_for_image_sec': 2.0,
            'snapshot_duration_sec': 3.0,
            'snapshot_sample_period_sec': 0.2,
            'log_detections_period_sec': 0.0,
            'snapshot_service_name': '/yolo/snapshot',
            'cmd_vel_topic': '/cmd_vel',
            'drive_on_snapshot_detection': True,
            'snapshot_forward_linear_x': 0.2,
            'snapshot_forward_duration_sec': 2.0,
            'snapshot_reverse_linear_x': -0.05,
            'snapshot_reverse_duration_sec': 3.0,
        }],
        condition=IfCondition(launch_yolo),
    )

    yolo_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        output='screen',
        condition=IfCondition(launch_yolo_view),
    )

    delivery_master = Node(
        package='tb3_delivery_core',
        executable='delivery_master',
        name='delivery_master',
        output='screen',
        parameters=[{
            'waypoints_file': waypoints_file,
            'mission_params_file': mission_params_file,
        }],
    )

    delivery_gui = Node(
        package='tb3_delivery_gui',
        executable='delivery_gui_node',
        name='delivery_gui_node',
        output='screen',
        condition=IfCondition(use_gui),
    )

    motion_executor = Node(
        package='tb3_trace_motion',
        executable='motion_executor',
        name='tb3_motion_executor',
        output='screen',
        parameters=[{
            'motion_file': motion_file,
        }],
        condition=IfCondition(launch_motion_executor),
    )

    motion_ui = Node(
        package='tb3_trace_motion',
        executable='motion_ui',
        name='tb3_motion_ui',
        output='screen',
        parameters=[{
            'motion_file': motion_file,
        }],
        condition=IfCondition(use_motion_ui),
    )

    aruco_process_manager = Node(
        package='tb3_delivery_core',
        executable='aruco_process_manager',
        name='aruco_process_manager',
        output='screen',
        parameters=[{
            'aruco_params_file': aruco_params_file,
            'image_topic': camera_raw_topic,
            'camera_info_topic': camera_info_topic,
            'dictionary': 'DICT_4X4_50',
            'marker_ids': '0,1',
            'marker_size': 0.015,
            'cmd_vel_topic': '/cmd_vel',
            'done_topic': '/aruco_align/done',
            'zero_cmd_publish_count': 5,
            'zero_cmd_publish_period_sec': 0.03,
        }],
        condition=IfCondition(manage_aruco_processes),
    )

    # Direct launch is kept only for marker_comm standalone debugging.
    # In the integrated system, aruco_process_manager starts/stops these nodes
    # from /aruco_align/active so launch does not leave them idle forever.
    aruco_pose = Node(
        package='marker_comm',
        executable='aruco_pose',
        name='aruco_pose',
        output='screen',
        parameters=[aruco_params_file, {
            'image_topic': camera_raw_topic,
            'camera_info_topic': camera_info_topic,
            'dictionary': 'DICT_4X4_50',
            'marker_ids': '0,1',
            'marker_size': 0.015,
        }],
        condition=IfCondition(launch_aruco),
    )

    aruco_align_controller = Node(
        package='marker_comm',
        executable='aruco_align_controller',
        name='aruco_align_controller',
        output='screen',
        parameters=[aruco_params_file],
        condition=IfCondition(launch_aruco),
    )

    return LaunchDescription([
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0'),
        DeclareLaunchArgument(
            'use_gui',
            default_value='false',
            description='Launch the optional delivery GUI with camera views.',
        ),
        DeclareLaunchArgument(
            'use_motion_ui',
            default_value='false',
            description='Launch the optional trace motion editor.',
        ),
        DeclareLaunchArgument(
            'launch_motion_executor',
            default_value='true',
            description='Launch odom-based trace motion executor.',
        ),
        DeclareLaunchArgument(
            'motion_file',
            default_value='/root/maps/tb3_motion_paths.yaml',
            description='YAML file storing trace motion paths.',
        ),
        DeclareLaunchArgument(
            'waypoints_file',
            default_value=os.path.join(
                tb3_delivery_share,
                'config',
                'waypoints.yaml',
            ),
            description='Delivery waypoint YAML file.',
        ),
        DeclareLaunchArgument(
            'mission_params_file',
            default_value=os.path.join(
                tb3_delivery_share,
                'config',
                'mission_params.yaml',
            ),
            description='Delivery mission parameter YAML file.',
        ),
        DeclareLaunchArgument(
            'aruco_params_file',
            default_value=os.path.join(
                marker_comm_share,
                'config',
                'aruco_alignment_params.yaml',
            ),
            description='ArUco pose and alignment controller parameter YAML file.',
        ),
        DeclareLaunchArgument(
            'launch_aruco',
            default_value='false',
            description='Directly launch marker_comm ArUco nodes for standalone debugging.',
        ),
        DeclareLaunchArgument(
            'manage_aruco_processes',
            default_value='true',
            description='Start/stop marker_comm ArUco nodes on /aruco_align/active.',
        ),
        DeclareLaunchArgument(
            'launch_yolo',
            default_value='true',
            description='Launch YOLO node for snapshot/debug detection.',
        ),
        DeclareLaunchArgument(
            'yolo_continuous_detection',
            default_value='false',
            description='Run YOLO continuously instead of only on /yolo/snapshot.',
        ),
        DeclareLaunchArgument(
            'launch_yolo_view',
            default_value='false',
            description='Launch rqt_image_view for YOLO annotated image.',
        ),
        DeclareLaunchArgument(
            'launch_camera_compressed_to_raw',
            default_value='true',
            description='Decompress /camera/image_raw/compressed locally for remote camera stability.',
        ),
        DeclareLaunchArgument(
            'camera_compressed_topic',
            default_value='/camera/image_raw/compressed',
            description='Compressed camera topic from robot or remote bridge.',
        ),
        DeclareLaunchArgument(
            'camera_raw_topic',
            default_value='/camera/image_decompressed',
            description='Raw image topic used by ArUco and YOLO nodes after optional decompression.',
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera/camera_info',
            description='CameraInfo topic used by marker_comm ArUco pose node.',
        ),
        DeclareLaunchArgument(
            'yolo_conf_threshold',
            default_value='0.85',
            description='YOLO confidence threshold.',
        ),
        DeclareLaunchArgument(
            'yolo_iou_threshold',
            default_value='0.45',
            description='YOLO IoU threshold.',
        ),
        DeclareLaunchArgument(
            'yolo_exclude_class_names',
            default_value='Pepero',
            description='Comma-separated YOLO class names to ignore.',
        ),
        DeclareLaunchArgument(
            'launch_nav2',
            default_value='false',
            description='Launch Nav2 with tb3_delivery_core tolerance params.',
        ),
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(
                turtlebot3_navigation_share,
                'map',
                'map.yaml',
            ),
            description='Full path to the map YAML file for Nav2.',
        ),
        DeclareLaunchArgument(
            'nav2_params_file',
            default_value=os.path.join(
                tb3_delivery_share,
                'config',
                'nav2_delivery_params.yaml',
            ),
            description='Nav2 params file with delivery goal tolerances.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock.',
        ),
        nav2_bringup,
        compressed_to_raw_republish,
        yolo_detector,
        yolo_view,
        motion_executor,
        delivery_master,
        delivery_gui,
        motion_ui,
        aruco_process_manager,
        aruco_pose,
        aruco_align_controller,
    ])
