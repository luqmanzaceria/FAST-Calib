#!/usr/bin/env python3
"""
Subscribe to a PointCloud2, apply the qr_params.yaml passthrough box (same as
fast_calib_core.passthrough_filter), and write the filtered clouds to a ROS 2 bag.

Requires: ROS 2 (rclpy, rosbag2_py), sensor_msgs, numpy.

Usage (system Python matching ROS, e.g. 3.10 on Humble):
    source /opt/ros/humble/setup.bash
    ros2 bag play /path/to/input_bag   # or live driver
    python3 scripts/record_scan_in_box_bag.py --config config/qr_params.yaml \\
        --output ~/bags/in_box_scan

Playback in RViz:
    ros2 bag play ~/bags/in_box_scan
    # Add PointCloud2 display on /sensor_scan_in_box (or --record-topic name)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np
import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2, PointField

from fast_calib_core import decode_pointcloud2, load_config, passthrough_filter

_BOUNDS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _ensure_bounds(cfg: dict) -> None:
    missing = [k for k in _BOUNDS if k not in cfg]
    if missing:
        raise SystemExit(f"Config missing passthrough keys: {missing}")


def _xyz_pointcloud2(header, xyz: np.ndarray) -> PointCloud2:
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


def _topic_metadata(topic_name: str) -> rosbag2_py.TopicMetadata:
    kwargs = dict(
        name=topic_name,
        type="sensor_msgs/msg/PointCloud2",
        serialization_format="cdr",
    )
    try:
        return rosbag2_py.TopicMetadata(**kwargs, offered_qos_profiles="")
    except TypeError:
        return rosbag2_py.TopicMetadata(**kwargs)


def _stamp_to_ns(msg: PointCloud2, fallback_ns: int) -> int:
    st = msg.header.stamp
    ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)
    return ns if ns > 0 else fallback_ns


class ScanInBoxRecorder(Node):
    def __init__(
        self,
        cfg: dict,
        in_topic: str,
        record_topic: str,
        output_uri: Path,
        storage_id: str,
    ) -> None:
        super().__init__("record_scan_in_box_bag")
        self._cfg = cfg
        self._record_topic = record_topic
        self._count = 0

        self._writer = rosbag2_py.SequentialWriter()
        self._writer.open(
            rosbag2_py.StorageOptions(
                uri=str(output_uri),
                storage_id=storage_id,
            ),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )
        self._writer.create_topic(_topic_metadata(record_topic))

        self.create_subscription(
            PointCloud2,
            in_topic,
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Recording '{record_topic}' ← '{in_topic}' → {output_uri} "
            f"(storage={storage_id}); Ctrl+C to stop"
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        pts = decode_pointcloud2(msg)
        if pts.shape[0] == 0:
            out = _xyz_pointcloud2(msg.header, np.zeros((0, 3), np.float32))
        else:
            filt = passthrough_filter(pts, self._cfg)
            out = _xyz_pointcloud2(msg.header, filt[:, :3])

        now_ns = self.get_clock().now().nanoseconds
        ts_ns = _stamp_to_ns(out, now_ns)
        self._writer.write(
            self._record_topic,
            serialize_message(out),
            ts_ns,
        )
        self._count += 1
        if self._count == 1 or self._count % 200 == 0:
            self.get_logger().info(f"Wrote {self._count} messages")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record passthrough-filtered PointCloud2 to a ROS 2 bag."
    )
    parser.add_argument("--config", required=True, help="Path to qr_params.yaml")
    parser.add_argument(
        "--in-topic",
        default="",
        help="Input PointCloud2 (default: lidar_topic from yaml or /sensor_scan)",
    )
    parser.add_argument(
        "--record-topic",
        default="/sensor_scan_in_box",
        help="Topic name stored in the output bag",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output bag path: directory for sqlite3, or .mcap file for mcap",
    )
    parser.add_argument(
        "--storage",
        choices=("sqlite3", "mcap"),
        default="sqlite3",
        help="Bag storage format (default: sqlite3 bag directory)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output path before recording",
    )
    argv = remove_ros_args(sys.argv)
    args = parser.parse_args(argv[1:] if len(argv) > 1 else [])

    cfg = load_config(args.config)
    _ensure_bounds(cfg)
    in_topic = (args.in_topic or "").strip() or cfg.get("lidar_topic") or "/sensor_scan"
    out_path = Path(args.output).expanduser().resolve()

    if args.storage == "mcap" and out_path.suffix.lower() != ".mcap":
        raise SystemExit("For --storage mcap, --output should end with .mcap")

    if out_path.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {out_path} (use --overwrite)")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # sqlite3: SequentialWriter creates the bag directory; do not mkdir(out_path) here.

    rclpy.init()
    node: ScanInBoxRecorder | None = None
    try:
        node = ScanInBoxRecorder(
            cfg,
            in_topic,
            args.record_topic,
            out_path,
            args.storage,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            n_msg = node._count
            logger = node.get_logger()
            w = getattr(node, "_writer", None)
            node._writer = None
            if w is not None:
                del w
            logger.info(f"Bag closed; wrote {n_msg} messages")
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
