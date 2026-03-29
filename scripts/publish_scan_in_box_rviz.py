#!/usr/bin/env python3
"""
Republish sensor_msgs/PointCloud2 containing only points inside the passthrough box
from qr_params.yaml (same bounds as fast_calib_core.passthrough_filter).

Use with publish_filter_box_rviz.py: RViz PointCloud2 → /sensor_scan_in_box,
Marker → /filter_box_marker, Fixed Frame = cloud frame.

Requires: ROS 2 (rclpy), sensor_msgs, numpy.

Usage:
    python3 scripts/publish_scan_in_box_rviz.py --config config/qr_params.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2, PointField

from fast_calib_core import decode_pointcloud2, load_config, passthrough_filter

_BOUNDS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _ensure_bounds(cfg: dict) -> None:
    missing = [k for k in _BOUNDS if k not in cfg]
    if missing:
        raise SystemExit(f"Config missing passthrough keys: {missing}")


def _xyz_pointcloud2(header, xyz: np.ndarray) -> PointCloud2:
    """Build PointCloud2 with x,y,z float32 fields only."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N, 3)")
    n = int(xyz.shape[0])
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * max(n, 1)
    msg.is_dense = True
    msg.data = xyz.tobytes() if n > 0 else b""
    if n == 0:
        msg.row_step = 0
    return msg


class ScanInBoxPublisher(Node):
    def __init__(self, cfg: dict, in_topic: str, out_topic: str) -> None:
        super().__init__("publish_scan_in_box_rviz")
        self._cfg = cfg
        self._pub = self.create_publisher(
            PointCloud2, out_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2,
            in_topic,
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Passthrough cloud: '{in_topic}' → '{out_topic}' "
            f"(x∈[{cfg['x_min']},{cfg['x_max']}], …)"
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        pts = decode_pointcloud2(msg)
        if pts.shape[0] == 0:
            self._pub.publish(_xyz_pointcloud2(msg.header, np.zeros((0, 3), np.float32)))
            return
        filt = passthrough_filter(pts, self._cfg)
        self._pub.publish(_xyz_pointcloud2(msg.header, filt[:, :3]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Republish PointCloud2 cropped to qr_params passthrough box."
    )
    parser.add_argument("--config", required=True, help="Path to qr_params.yaml")
    parser.add_argument(
        "--in-topic",
        default="",
        help="Input PointCloud2 (default: lidar_topic from yaml or /sensor_scan)",
    )
    parser.add_argument(
        "--out-topic",
        default="/sensor_scan_in_box",
        help="Output PointCloud2 topic",
    )
    argv = remove_ros_args(sys.argv)
    args = parser.parse_args(argv[1:] if len(argv) > 1 else [])

    cfg = load_config(args.config)
    _ensure_bounds(cfg)
    in_topic = (args.in_topic or "").strip() or cfg.get("lidar_topic") or "/sensor_scan"

    rclpy.init()
    try:
        node = ScanInBoxPublisher(cfg, in_topic, args.out_topic)
        rclpy.spin(node)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
