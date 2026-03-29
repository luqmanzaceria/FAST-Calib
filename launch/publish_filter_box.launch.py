"""
publish_filter_box.launch.py — Run publish_filter_box_rviz.py from this repo layout.

Expects: launch/ and scripts/ under the same package root (typical git clone).

Usage:
    ros2 launch fast_calib publish_filter_box.launch.py
    ros2 launch fast_calib publish_filter_box.launch.py frame_id:=lidar_frame
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("fast_calib")
    # share/.. is install prefix; share/../lib/... or for source overlays use path via substitution
    # Resolve script next to share: many workspaces put share at install/fast_calib/share/fast_calib
    # Fallback: path relative to this file when launch is run from source (launch beside scripts)
    launch_dir = Path(__file__).resolve().parent
    repo_scripts = launch_dir.parent / "scripts" / "publish_filter_box_rviz.py"

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_path",
            default_value="",
            description="qr_params.yaml (empty = <pkg share>/config/qr_params.yaml)",
        ),
        DeclareLaunchArgument(
            "frame_id",
            default_value="",
            description="Marker frame (empty = sniff from cloud)",
        ),
        DeclareLaunchArgument(
            "cloud_topic",
            default_value="",
            description="PointCloud2 topic for sniffing (empty = from yaml /sensor_scan)",
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
                str(repo_scripts),
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
            shell=False,
        ),
    ])
