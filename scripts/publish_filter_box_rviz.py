#!/usr/bin/env python3
"""
Publish a visualization_msgs/Marker CUBE for the passthrough filter box defined in
qr_params.yaml (x_min … z_max), so RViz can show it aligned with the LiDAR cloud.

Requires: ROS 2 (rclpy), visualization_msgs, sensor_msgs.

Usage:
    source /opt/ros/<distro>/setup.bash
    python3 scripts/publish_filter_box_rviz.py --config config/qr_params.yaml

If --frame-id is omitted, the node subscribes once to the PointCloud2 topic and uses
that message's header.frame_id (default cloud topic: lidar_topic from yaml, else /sensor_scan).

In RViz: add a Marker display subscribed to the marker topic; set Fixed Frame to match
the cloud frame (or TF so the marker frame is reachable).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from fast_calib_core import load_config

_BOUNDS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _read_bounds(cfg: dict) -> tuple[float, float, float, float, float, float]:
    missing = [k for k in _BOUNDS if k not in cfg]
    if missing:
        raise SystemExit(f"Config missing passthrough keys: {missing}")
    return tuple(float(cfg[k]) for k in _BOUNDS)


class FilterBoxMarkerPublisher(Node):
    def __init__(
        self,
        bounds: tuple[float, float, float, float, float, float],
        frame_id: str,
        cloud_topic: str,
        marker_topic: str,
        rate_hz: float,
    ) -> None:
        super().__init__("publish_filter_box_rviz")
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        self._cx = (x_min + x_max) / 2.0
        self._cy = (y_min + y_max) / 2.0
        self._cz = (z_min + z_max) / 2.0
        self._sx = max(float(x_max - x_min), 1e-6)
        self._sy = max(float(y_max - y_min), 1e-6)
        self._sz = max(float(z_max - z_min), 1e-6)

        self._frame_id = frame_id
        self._ready = bool(frame_id)
        self._cloud_sub = None
        if not self._ready:
            self._cloud_sub = self.create_subscription(
                PointCloud2,
                cloud_topic,
                self._on_cloud,
                qos_profile_sensor_data,
            )
            self.get_logger().info(
                f"Waiting for frame_id from PointCloud2 on '{cloud_topic}' …"
            )

        self._pub = self.create_publisher(Marker, marker_topic, 10)
        period = 1.0 / rate_hz if rate_hz > 0 else 0.5
        self._timer = self.create_timer(period, self._publish)

    def _on_cloud(self, msg: PointCloud2) -> None:
        if self._ready:
            return
        fid = (msg.header.frame_id or "").strip()
        if not fid:
            return
        self._frame_id = fid
        self._ready = True
        if self._cloud_sub is not None:
            self.destroy_subscription(self._cloud_sub)
            self._cloud_sub = None
        self.get_logger().info(f"Using frame_id '{self._frame_id}' from point cloud")

    def _publish(self) -> None:
        if not self._ready:
            return
        now = self.get_clock().now().to_msg()
        m = Marker()
        m.header.stamp = now
        m.header.frame_id = self._frame_id
        m.ns = "fast_calib_filter_box"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = self._cx
        m.pose.position.y = self._cy
        m.pose.position.z = self._cz
        m.pose.orientation.w = 1.0
        m.scale.x = self._sx
        m.scale.y = self._sy
        m.scale.z = self._sz
        m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.35)
        m.lifetime.sec = 0
        m.lifetime.nanosec = 0
        self._pub.publish(m)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish RViz marker for qr_params passthrough filter box."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to qr_params.yaml",
    )
    parser.add_argument(
        "--frame-id",
        default="",
        help="Marker frame_id (if empty, taken from first PointCloud2 on cloud topic)",
    )
    parser.add_argument(
        "--cloud-topic",
        default="",
        help="PointCloud2 topic for frame sniffing (default: lidar_topic from yaml or /sensor_scan)",
    )
    parser.add_argument(
        "--topic",
        default="/filter_box_marker",
        help="Marker publication topic",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="Publish rate (Hz)",
    )
    args = parser.parse_args(remove_ros_args(sys.argv))

    cfg = load_config(args.config)
    bounds = _read_bounds(cfg)
    cloud_topic = (args.cloud_topic or "").strip() or cfg.get("lidar_topic") or "/sensor_scan"
    frame_id = (args.frame_id or "").strip()

    rclpy.init()
    try:
        node = FilterBoxMarkerPublisher(
            bounds=bounds,
            frame_id=frame_id,
            cloud_topic=cloud_topic,
            marker_topic=args.topic,
            rate_hz=args.rate,
        )
        rclpy.spin(node)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
