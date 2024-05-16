import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def load_yaml(package_name, file_name):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_name)
    with open(absolute_file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    realsense_args = {
        "enable_rgbd": "true",
        "enable_sync": "true",
        "align_depth.enable": "true",
        "enable_color": "true",
        "enable_depth": "true",
        "depth_module.depth_profile": "640x480x30",
        "depth_module.infra_profile": "640x480x30",
    }
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("realsense2_camera"),
                         "launch", "rs_launch.py")
        ]),
        launch_arguments=realsense_args.items())

    calibration_tf_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("easy_handeye2"),
                         "launch", "publish.launch.py")
        ]),
        launch_arguments={"name": "ar4_calibration"}.items())

    delay_calibration_tf_publisher = TimerAction(
        actions=[calibration_tf_publisher],
        period=2.0,
    )

    ar_moveit_launch = PythonLaunchDescriptionSource([
        os.path.join(get_package_share_directory("ar_moveit_config"), "launch",
                     "ar_moveit.launch.py")
    ])
    ar_moveit_args = {
        "include_gripper": "True",
        "rviz_config_file": "moveit_with_camera.rviz"
    }.items()
    ar_moveit = IncludeLaunchDescription(ar_moveit_launch,
                                         launch_arguments=ar_moveit_args)

    tabletop_handybot_node = Node(
        package="tabletop_handybot",
        executable="tabletop_handybot_node",
        name="tabletop_handybot_node",
        output="screen",
    )

    audio_prompt_node = Node(
        package="tabletop_handybot",
        executable="audio_prompt_node",
        name="audio_prompt_node",
        output="screen",
    )

    return LaunchDescription([
        realsense, delay_calibration_tf_publisher, ar_moveit,
        audio_prompt_node, tabletop_handybot_node
    ])
