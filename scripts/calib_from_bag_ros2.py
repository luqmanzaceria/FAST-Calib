#!/usr/bin/env python3
"""
calib_from_bag_ros2.py — Offline LiDAR-camera extrinsic calibration
                          from a ROS2 .db3 bag (or a directory containing one).

No ROS installation required.  Install dependencies with:
    pip install rosbags numpy "opencv-python>=4.5" open3d pyyaml scipy

Usage:
    python3 scripts/calib_from_bag_ros2.py \\
        --bag    /path/to/bag_dir   \\
        --config config/qr_params.yaml \\
        [--image  /path/to/image.png]   \\
        [--image-topic  /camera/image_raw] \\
        [--lidar-topic  /livox/lidar]   \\
        [--output-dir   output/]

The bag path may be:
  • the bag DIRECTORY  (contains metadata.yaml + *.db3)
  • the *.db3 FILE itself  (parent directory is used as bag root)
  • a ROS1 *.bag file      (AnyReader handles both)

If --image is omitted the script scans the image topic for the sharpest
frame that has >= min_detected_markers ArUco markers visible.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

# Make scripts/ importable as a package sibling
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from fast_calib_core import (
    load_config, decode_pointcloud2, decode_image_msg, run_calibration
)


# ════════════════════════════════════════════════════════════════════════════
# Bag reading helpers
# ════════════════════════════════════════════════════════════════════════════

def _bag_root(path: Path) -> Path:
    """Return the bag directory regardless of whether user gave the .db3 file
    or the containing directory."""
    if path.is_file() and path.suffix == ".db3":
        return path.parent
    return path


def _open_reader(bag_root: Path):
    """Open a rosbags AnyReader on the bag directory."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        sys.exit("[ERROR] rosbags not installed.  Run: pip install rosbags")
    return AnyReader([bag_root])


def list_topics(bag_root: Path):
    """Print all topics and message types in the bag."""
    with _open_reader(bag_root) as reader:
        for conn in reader.connections:
            print(f"  {conn.topic:<50s}  {conn.msgtype}")


def _best_image(bag_root: Path, image_topic: str,
                min_markers: int, aruco_dict) -> np.ndarray | None:
    """
    Scan the bag for the sharpest frame that has >= min_markers ArUco markers.
    Returns a BGR numpy image or None.
    """
    best_img   = None
    best_score = -1.0
    best_n     = 0

    try:
        det_params = aruco.DetectorParameters()
        detector   = aruco.ArucoDetector(aruco_dict, det_params)
        def _detect(gray):
            corners, ids, _ = detector.detectMarkers(gray)
            return len(ids) if ids is not None else 0
    except AttributeError:
        det_params = aruco.DetectorParameters_create()
        def _detect(gray):
            corners, ids, _ = aruco.detectMarkers(
                gray, aruco_dict, parameters=det_params)
            return len(ids) if ids is not None else 0

    with _open_reader(bag_root) as reader:
        conns = [c for c in reader.connections if c.topic == image_topic]
        if not conns:
            return None
        for conn, ts, raw in reader.messages(connections=conns):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                bgr = decode_image_msg(msg)
            except Exception:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            n    = _detect(gray)
            if n >= min_markers:
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if n > best_n or (n == best_n and sharpness > best_score):
                    best_img   = bgr.copy()
                    best_score = sharpness
                    best_n     = n
                    print(f"  [Image] new best: {n} markers, "
                          f"sharpness={sharpness:.1f}", flush=True)

    return best_img


def _read_cloud(bag_root: Path, lidar_topic: str,
                max_msgs: int = 0) -> np.ndarray:
    """
    Accumulate all (or up to max_msgs) PointCloud2 messages from lidar_topic.
    Returns Nx4 float32 [x, y, z, ring].
    """
    all_pts = []
    count   = 0
    with _open_reader(bag_root) as reader:
        conns = [c for c in reader.connections if c.topic == lidar_topic]
        if not conns:
            return np.zeros((0, 4), dtype=np.float32)
        for conn, ts, raw in reader.messages(connections=conns):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                pts = decode_pointcloud2(msg)
                if len(pts):
                    all_pts.append(pts)
                    count += 1
                    if max_msgs and count >= max_msgs:
                        break
            except Exception as e:
                print(f"  [Cloud] decode error: {e}", file=sys.stderr)

    if not all_pts:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(all_pts, axis=0)


def _auto_lidar_topic(bag_root: Path) -> str | None:
    """Return the first PointCloud2 topic found in the bag."""
    keywords = ["lidar", "points", "scan", "cloud"]
    with _open_reader(bag_root) as reader:
        for conn in reader.connections:
            t = conn.msgtype.lower()
            if "pointcloud2" in t:
                return conn.topic
        # Fallback: topic name heuristic
        for conn in reader.connections:
            for kw in keywords:
                if kw in conn.topic.lower():
                    return conn.topic
    return None


def _auto_image_topic(bag_root: Path) -> str | None:
    """Return the first Image or CompressedImage topic found."""
    with _open_reader(bag_root) as reader:
        for conn in reader.connections:
            t = conn.msgtype.lower()
            if "image" in t and "compressed" not in t:
                return conn.topic
    return None


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FAST-Calib offline calibration from a ROS2 .db3 bag.")
    p.add_argument("--bag",         required=True,
                   help="Bag directory or path to the .db3 file")
    p.add_argument("--config",      default=None,
                   help="Path to qr_params.yaml "
                        "(default: <repo>/config/qr_params.yaml)")
    p.add_argument("--image",       default=None,
                   help="Camera image file (PNG/JPEG). "
                        "If omitted, extracted from --image-topic.")
    p.add_argument("--image-topic", default=None,
                   help="Camera topic for image extraction (auto-detected if omitted)")
    p.add_argument("--lidar-topic", default=None,
                   help="LiDAR topic (auto-detected if omitted)")
    p.add_argument("--output-dir",  default=None,
                   help="Where to write results (default: <repo>/output)")
    p.add_argument("--list-topics", action="store_true",
                   help="Print all topics in the bag and exit")
    return p.parse_args()


def main():
    args = parse_args()

    # Locate repo root for default paths
    repo_root  = _HERE.parent
    bag_root   = _bag_root(Path(args.bag).resolve())
    config_path = args.config or str(repo_root / "config" / "qr_params.yaml")
    output_dir  = args.output_dir or str(repo_root / "output")

    if not bag_root.exists():
        sys.exit(f"[ERROR] Bag not found: {bag_root}")
    if not Path(config_path).exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")

    if args.list_topics:
        print(f"\nTopics in {bag_root}:")
        list_topics(bag_root)
        return

    cfg = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── topic resolution ─────────────────────────────────────────────────────
    lidar_topic = (args.lidar_topic
                   or cfg.get("lidar_topic")
                   or _auto_lidar_topic(bag_root))
    if not lidar_topic:
        sys.exit("[ERROR] No LiDAR topic found. Use --lidar-topic.")

    image_topic = (args.image_topic
                   or _auto_image_topic(bag_root)
                   or "/camera/image_raw")

    print(f"[Bag]   root        : {bag_root}")
    print(f"[Bag]   lidar_topic : {lidar_topic}")
    print(f"[Bag]   image_topic : {image_topic}")
    print(f"[Config] {config_path}")
    print(f"[Output] {output_dir}\n")

    # ── image ─────────────────────────────────────────────────────────────────
    if args.image:
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            sys.exit(f"[ERROR] Cannot read image: {args.image}")
        print(f"[Image] Loaded from file: {args.image}")
    else:
        print(f"[Image] Scanning '{image_topic}' for best ArUco frame …")
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        image = _best_image(bag_root, image_topic,
                            int(cfg["min_detected_markers"]), aruco_dict)
        if image is None:
            sys.exit(
                f"[ERROR] No suitable image found on '{image_topic}'.\n"
                f"  Check the topic name with --list-topics, or supply "
                f"--image /path/to/image.png")
        # Save the extracted image for reference
        extracted_path = os.path.join(output_dir, "extracted_image.png")
        cv2.imwrite(extracted_path, image)
        print(f"[Image] Saved extracted image: {extracted_path}")

    # ── point cloud ───────────────────────────────────────────────────────────
    print(f"\n[Cloud] Reading all messages on '{lidar_topic}' …")
    pts_N4 = _read_cloud(bag_root, lidar_topic)
    if len(pts_N4) == 0:
        sys.exit(
            f"[ERROR] No point cloud data on '{lidar_topic}'.\n"
            f"  Check the topic name with --list-topics.")
    print(f"[Cloud] Total accumulated points: {len(pts_N4):,}")

    # ── calibration ───────────────────────────────────────────────────────────
    print("\n[Calib] Running calibration pipeline …\n")
    T, rmse = run_calibration(image, pts_N4, cfg, output_dir, tag="single")

    if T is None:
        sys.exit("[ERROR] Calibration failed.  Check your parameters "
                 "(filter bounds, marker size, circle dimensions).")

    print(f"\n[Done] RMSE = {rmse:.4f} m   Results in: {output_dir}/")


if __name__ == "__main__":
    main()
