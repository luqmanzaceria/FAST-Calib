"""
Launch marker box + passthrough-filtered PointCloud2 for RViz.

Runs:
  • publish_filter_box_rviz.py  → /filter_box_marker (default)
  • publish_scan_in_box_rviz.py → /sensor_scan_in_box (default)

Usage:
    ros2 launch fast_calib publish_filter_visualization.launch.py
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("fast_calib")
    launch_dir = Path(__file__).resolve().parent
    root = launch_dir.parent
    marker_script = root / "scripts" / "publish_filter_box_rviz.py"
    scan_script = root / "scripts" / "publish_scan_in_box_rviz.py"

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_path",
            default_value=PathJoinSubstitution(
                [pkg_share, "config", "qr_params.yaml"]
            ),
        ),
        DeclareLaunchArgument("frame_id", default_value=""),
        DeclareLaunchArgument("cloud_topic", default_value=""),
        DeclareLaunchArgument("marker_topic", default_value="/filter_box_marker"),
        DeclareLaunchArgument("filtered_cloud_topic", default_value="/sensor_scan_in_box"),
        DeclareLaunchArgument("marker_rate", default_value="2.0"),

        ExecuteProcess(
            cmd=[
                "python3",
                str(marker_script),
                "--config",
                LaunchConfiguration("config_path"),
                "--frame-id",
                LaunchConfiguration("frame_id"),
                "--cloud-topic",
                LaunchConfiguration("cloud_topic"),
                "--topic",
                LaunchConfiguration("marker_topic"),
                "--rate",
                LaunchConfiguration("marker_rate"),
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                "python3",
                str(scan_script),
                "--config",
                LaunchConfiguration("config_path"),
                "--in-topic",
                LaunchConfiguration("cloud_topic"),
                "--out-topic",
                LaunchConfiguration("filtered_cloud_topic"),
            ],
            output="screen",
        ),
    ])
