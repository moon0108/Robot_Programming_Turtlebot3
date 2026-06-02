# Robot Programming TurtleBot3 Delivery Project

This project is a ROS 2 Humble based indoor delivery robot system using TurtleBot3 Burger. After the user selects an item through the GUI, the robot performs object detection, movement, ArUco-based alignment, and lift operation in sequence.

## Workspace Structure

```text
turtlebot3_ws/
├── src/                  # ROS 2 source packages
├── build/                # colcon build output
├── install/              # installed ROS 2 packages
├── log/                  # build/runtime logs
└── README.md
```

## Package Structure

```text
src/
├── tb3_delivery_core/        # main delivery system, launch files, and config files
├── tb3_delivery_gui/         # GUI package
├── tb3_delivery_interfaces/  # custom action/interface definitions
├── marker_comm/              # ArUco marker alignment package
├── yolo_v8_ros/              # YOLO object detection package
├── yolo_remote_bringup/      # remote camera/image bringup helper
├── waypoint_nav/             # waypoint navigation helper
├── tb3_trace_motion/         # recorded/trace motion helper
├── turtlebot3/               # TurtleBot3 base packages
├── turtlebot3_msgs/          # TurtleBot3 message/service definitions
├── turtlebot3_simulations/   # Gazebo simulation packages
└── DynamixelSDK/             # Dynamixel SDK
```

## Package Roles

| Package | Role |
| --- | --- |
| `tb3_delivery_core` | Main delivery system package containing the primary launch file and configuration files. |
| `tb3_delivery_gui` | GUI package for user interaction. |
| `tb3_delivery_interfaces` | Custom interface definitions used in the project. |
| `marker_comm` | ArUco marker based alignment package. |
| `yolo_v8_ros` | YOLO based object detection package. |
| `yolo_remote_bringup` | Helper package for receiving and converting remote camera images. |
| `waypoint_nav` | Helper package for waypoint based navigation. |
| `tb3_trace_motion` | Helper package for executing recorded motion paths. |
| `turtlebot3`, `turtlebot3_msgs`, `turtlebot3_simulations` | TurtleBot3 base, message, and simulation packages. |
| `DynamixelSDK` | SDK for Dynamixel motor control. |

## Build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run


```bash
cd /root/turtlebot3_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger

ros2 launch tb3_delivery_core delivery.launch.py use_gui:=true
```

