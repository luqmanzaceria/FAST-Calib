#!/usr/bin/env python3
"""
live_calib_ros2.py — FAST-Calib live LiDAR-camera calibration for ROS2 Humble.

Subscribes to camera and LiDAR topics, accumulates data, then computes
T_cam_lidar using the same algorithm as the offline pipeline.  Shuts down
automatically on success.

Usage (standalone):
    python3 scripts/live_calib_ros2.py \\
        --config config/qr_params.yaml \\
        --image-topic  /camera/image_raw \\
        --lidar-topic  /livox/lidar \\
        --output-dir   output/

Usage via launch file:
    ros2 launch fast_calib live_calib_ros2.launch.py

ROS2 params (set via launch args or --ros-args):
    config_path    – path to qr_params.yaml
    image_topic    – camera image topic
    lidar_topic    – LiDAR PointCloud2 topic
    output_dir     – where to write results
    calib_interval – seconds between calibration attempts (default 5.0)

Dependencies:
    pip install numpy "opencv-python>=4.5" open3d scipy pyyaml
    ROS2 Humble: rclpy sensor_msgs cv_bridge (or decode manually)
"""

from __future__ import annotations
import argparse
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from fast_calib_core import (
    load_config, decode_pointcloud2, run_calibration
)

# ════════════════════════════════════════════════════════════════════════════
# ROS2 node
# ════════════════════════════════════════════════════════════════════════════

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False


def _decode_ros2_image(msg: "Image") -> np.ndarray | None:
    """Convert sensor_msgs.msg.Image to BGR ndarray without cv_bridge."""
    enc = msg.encoding
    h, w = msg.height, msg.width
    raw  = bytes(msg.data)
    try:
        if enc in ("bgr8", "rgb8"):
            arr = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            return arr.copy() if enc == "bgr8" else arr[:, :, ::-1].copy()
        if enc == "mono8":
            return cv2.cvtColor(
                np.frombuffer(raw, np.uint8).reshape(h, w),
                cv2.COLOR_GRAY2BGR)
        if enc in ("16UC1", "mono16"):
            arr16 = np.frombuffer(raw, np.uint16).reshape(h, w)
            return cv2.cvtColor((arr16 >> 8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # Fallback
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    except Exception:
        return None


def _decode_ros2_pc2(msg: "PointCloud2") -> np.ndarray:
    """Convert sensor_msgs.msg.PointCloud2 to Nx4 float32 [x,y,z,ring]."""
    # Re-use the rosbags-compatible decoder – the field layout is the same
    # for both rclpy and rosbags deserialized messages.
    return decode_pointcloud2(msg)


class LiveCalibNode(Node):
    """
    ROS2 node that accumulates image + LiDAR data and periodically
    attempts LiDAR-camera extrinsic calibration.
    """

    def __init__(self, cfg: dict, output_dir: str,
                 image_topic: str, lidar_topic: str,
                 calib_interval: float):
        super().__init__("live_calib")
        self._cfg         = cfg
        self._output_dir  = output_dir
        self._lock        = threading.Lock()

        # State
        self._best_image       = None
        self._best_score       = -1.0
        self._best_n_markers   = 0
        self._cloud_parts: list[np.ndarray] = []
        self._done = False

        # ArUco detector for selecting best image
        import cv2.aruco as aruco
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        try:
            det_params = aruco.DetectorParameters()
            self._aruco_detector = aruco.ArucoDetector(aruco_dict, det_params)
            self._aruco_old_api  = False
        except AttributeError:
            self._aruco_params   = aruco.DetectorParameters_create()
            self._aruco_dict     = aruco_dict
            self._aruco_old_api  = True

        # Subscribers
        self._img_sub = self.create_subscription(
            Image, image_topic, self._image_cb, 10)
        self._pc2_sub = self.create_subscription(
            PointCloud2, lidar_topic, self._cloud_cb, 50)

        # Calibration timer
        self._timer = self.create_timer(calib_interval, self._calib_cb)

        self.get_logger().info(
            f"LiveCalib ready — image: '{image_topic}', "
            f"lidar: '{lidar_topic}', "
            f"interval: {calib_interval:.1f}s")

    # ── callbacks ────────────────────────────────────────────────────────────

    def _count_markers(self, gray: np.ndarray) -> int:
        if self._aruco_old_api:
            import cv2.aruco as aruco
            _, ids, _ = aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._aruco_params)
        else:
            _, ids, _ = self._aruco_detector.detectMarkers(gray)
        return len(ids) if ids is not None else 0

    def _image_cb(self, msg: "Image") -> None:
        if self._done:
            return
        bgr = _decode_ros2_image(msg)
        if bgr is None:
            return
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        n    = self._count_markers(gray)
        if n >= int(self._cfg["min_detected_markers"]):
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            with self._lock:
                if n > self._best_n_markers or \
                   (n == self._best_n_markers and sharpness > self._best_score):
                    self._best_image     = bgr.copy()
                    self._best_score     = sharpness
                    self._best_n_markers = n
                    self.get_logger().info(
                        f"Best image updated: {n} markers, "
                        f"sharpness={sharpness:.1f}")

    def _cloud_cb(self, msg: "PointCloud2") -> None:
        if self._done:
            return
        pts = _decode_ros2_pc2(msg)
        if len(pts):
            with self._lock:
                self._cloud_parts.append(pts)

    # ── periodic calibration attempt ─────────────────────────────────────────

    def _calib_cb(self) -> None:
        if self._done:
            return

        with self._lock:
            image  = self._best_image
            parts  = list(self._cloud_parts)

        if image is None:
            self.get_logger().warn(
                "Waiting for image with sufficient ArUco markers …")
            return

        if not parts:
            self.get_logger().warn("Waiting for LiDAR data …")
            return

        cloud = np.concatenate(parts, axis=0)
        self.get_logger().info(
            f"Attempting calibration: {len(cloud):,} LiDAR pts …")

        T, rmse = run_calibration(image, cloud, self._cfg,
                                  self._output_dir, tag="live")

        if T is None:
            self.get_logger().warn(
                "Calibration attempt failed — clearing cloud buffer, retrying.")
            with self._lock:
                self._cloud_parts.clear()
            return

        self.get_logger().info(
            f"Calibration SUCCEEDED  RMSE={rmse:.4f} m  "
            f"Results: {self._output_dir}/")
        self._done = True
        self._timer.cancel()
        rclpy.shutdown()


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FAST-Calib live calibration ROS2 node")
    p.add_argument("--config",         default=None)
    p.add_argument("--image-topic",    default="/camera/image_raw")
    p.add_argument("--lidar-topic",    default=None)
    p.add_argument("--output-dir",     default=None)
    p.add_argument("--calib-interval", type=float, default=5.0,
                   help="Seconds between calibration attempts")
    # Allow unknown args so --ros-args ... doesn't break argparse
    return p.parse_known_args()[0]


def main():
    if not _HAS_RCLPY:
        sys.exit(
            "[ERROR] rclpy not found.\n"
            "  Source your ROS2 Humble workspace:\n"
            "    source /opt/ros/humble/setup.bash\n"
            "  and make sure the fast_calib workspace is also sourced.")

    args = parse_args()
    repo_root   = _HERE.parent
    config_path = args.config     or str(repo_root / "config" / "qr_params.yaml")
    output_dir  = args.output_dir or str(repo_root / "output")
    os.makedirs(output_dir, exist_ok=True)

    if not Path(config_path).exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")

    cfg = load_config(config_path)

    lidar_topic = args.lidar_topic or cfg.get("lidar_topic", "/livox/lidar")

    rclpy.init()
    node = LiveCalibNode(
        cfg            = cfg,
        output_dir     = output_dir,
        image_topic    = args.image_topic,
        lidar_topic    = lidar_topic,
        calib_interval = args.calib_interval,
    )

    # Also read ROS2 params if the node was launched with --ros-args
    try:
        for pname, default, attr in [
            ("config_path",    config_path,     None),
            ("image_topic",    args.image_topic, None),
            ("lidar_topic",    lidar_topic,      None),
            ("output_dir",     output_dir,       None),
            ("calib_interval", args.calib_interval, None),
        ]:
            node.declare_parameter(pname, default)
    except Exception:
        pass

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
