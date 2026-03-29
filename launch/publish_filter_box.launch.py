"""
publish_filter_box.launch.py — Run publish_filter_box_rviz.py from this repo layout.

The script path is resolved relative to this launch file: <parent>/scripts/publish_filter_box_rviz.py
(works when the repo has launch/ and scripts/ under the same root).

Usage:
    ros2 launch fast_calib publish_filter_box.launch.py
    ros2 launch fast_calib publish_filter_box.launch.py frame_id:=os_sensor
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("fast_calib")
    launch_dir = Path(__file__).resolve().parent
    script_path = launch_dir.parent / "scripts" / "publish_filter_box_rviz.py"

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_path",
            default_value=PathJoinSubstitution(
                [pkg_share, "config", "qr_params.yaml"]
            ),
            description="Path to qr_params.yaml",
        ),
        DeclareLaunchArgument(
            "frame_id",
            default_value="",
            description="Marker frame (empty = sniff from first PointCloud2)",
        ),
        DeclareLaunchArgument(
            "cloud_topic",
            default_value="",
            description="PointCloud2 topic for sniffing (empty = yaml lidar_topic or /sensor_scan)",
        ),
        DeclareLaunchArgument(
            "marker_topic",
            default_value="/filter_box_marker",
            description="Published Marker topic",
        ),
        DeclareLaunchArgument(
            "rate",
            default_value="2.0",
            description="Publish rate (Hz)",
        ),

        ExecuteProcess(
            cmd=[
                "python3",
                str(script_path),
                "--config",
                LaunchConfiguration("config_path"),
                "--frame-id",
                LaunchConfiguration("frame_id"),
                "--cloud-topic",
                LaunchConfiguration("cloud_topic"),
                "--topic",
                LaunchConfiguration("marker_topic"),
                "--rate",
                LaunchConfiguration("rate"),
            ],
            output="screen",
        ),
    ])
