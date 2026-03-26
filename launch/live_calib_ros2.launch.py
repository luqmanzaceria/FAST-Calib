"""
live_calib_ros2.launch.py — ROS2 launch file for FAST-Calib live calibration.

Usage:
    ros2 launch fast_calib live_calib_ros2.launch.py

Override any argument:
    ros2 launch fast_calib live_calib_ros2.launch.py \\
        image_topic:=/camera/color/image_raw \\
        lidar_topic:=/ouster/points \\
        calib_interval:=10.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("fast_calib")

    return LaunchDescription([
        DeclareLaunchArgument(
            "image_topic", default_value="/camera/image_raw",
            description="Camera image topic"),
        DeclareLaunchArgument(
            "lidar_topic", default_value="",
            description="LiDAR PointCloud2 topic (empty = read from yaml)"),
        DeclareLaunchArgument(
            "calib_interval", default_value="5.0",
            description="Seconds between calibration attempts"),
        DeclareLaunchArgument(
            "config_path", default_value="",
            description="Path to qr_params.yaml (empty = use package default)"),
        DeclareLaunchArgument(
            "output_dir", default_value="",
            description="Output directory (empty = <package>/output)"),

        Node(
            package="fast_calib",
            executable="live_calib_ros2",
            name="live_calib_ros2",
            output="screen",
            parameters=[
                PathJoinSubstitution([pkg_share, "config", "qr_params.yaml"]),
                {
                    "image_topic":    LaunchConfiguration("image_topic"),
                    "lidar_topic":    LaunchConfiguration("lidar_topic"),
                    "calib_interval": LaunchConfiguration("calib_interval"),
                    "config_path":    LaunchConfiguration("config_path"),
                    "output_dir":     LaunchConfiguration("output_dir"),
                },
            ],
        ),
    ])
