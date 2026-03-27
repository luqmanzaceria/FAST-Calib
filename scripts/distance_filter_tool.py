#!/usr/bin/env python3
"""
distance_filter_tool.py — Visualise the passthrough filter bounds on a
                           point cloud extracted from a ROS2 bag.

Requires ROS2 Humble to be sourced:
    source /opt/ros/humble/setup.bash

Additional Python deps:
    pip install numpy pyyaml matplotlib

Usage:
    python3 scripts/distance_filter_tool.py \\
        --bag     /path/to/lidar_bag         \\
        --config  config/qr_params.yaml      \\
        [--lidar-topic /sensor_scan]         \\
        [--max-frames  30]                   \\
        [--output      output/filter_check.png]

Three 2-D projections are generated (top-down XY, side XZ, front YZ).
Each shows the full cloud (grey) and the filter box (orange rectangle).
Points INSIDE the box are coloured by depth for easier visual inspection.
"""

from __future__ import annotations
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; change to "TkAgg" if you want a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2 as Pc2Msg

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from fast_calib_core import load_config, passthrough_filter


# ─────────────────────────────────────────────────────────────────────────────
# Bag helpers
# ─────────────────────────────────────────────────────────────────────────────

_LIDAR_TYPE     = "sensor_msgs/msg/PointCloud2"
_LIDAR_KEYWORDS = ["/sensor_scan", "/lidar", "/points", "/velodyne", "/ouster"]


def _make_reader(bag_path: Path) -> rosbag2_py.SequentialReader:
    """Open a ROS2 bag (directory, .db3, or .mcap) and return a SequentialReader."""
    if bag_path.is_dir():
        uri        = str(bag_path)
        storage_id = ""              # inferred from metadata.yaml
    elif bag_path.suffix == ".mcap":
        uri        = str(bag_path)
        storage_id = "mcap"
    else:
        uri        = str(bag_path)
        storage_id = "sqlite3"

    storage_opts   = rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id)
    converter_opts = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_opts, converter_opts)
    return reader


def _list_topics(bag_path: Path) -> dict[str, str]:
    """Return {topic_name: type_string} for every topic in the bag."""
    reader = _make_reader(bag_path)
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def _auto_lidar_topic(topics: dict[str, str]) -> str | None:
    """Pick the most likely LiDAR topic from a {name: type} dict."""
    pc2_topics = [n for n, t in topics.items() if t == _LIDAR_TYPE]
    for kw in _LIDAR_KEYWORDS:
        for name in pc2_topics:
            if kw in name:
                return name
    return pc2_topics[0] if pc2_topics else None


# ─────────────────────────────────────────────────────────────────────────────
# PointCloud2 → Nx3 numpy
# ─────────────────────────────────────────────────────────────────────────────

def _pc2_to_xyz(msg: Pc2Msg) -> np.ndarray:
    """
    Decode a sensor_msgs/PointCloud2 message to an Nx3 float32 array.
    Looks for fields named 'x', 'y', 'z' (datatype 7 = FLOAT32).
    """
    field_map: dict[str, int] = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"}.issubset(field_map):
        return np.empty((0, 3), dtype=np.float32)

    n_pts = msg.width * msg.height
    if n_pts == 0:
        return np.empty((0, 3), dtype=np.float32)

    step  = msg.point_step
    fmt   = "<" if not msg.is_bigendian else ">"
    data  = bytes(msg.data)

    ox, oy, oz = field_map["x"], field_map["y"], field_map["z"]
    xyz = np.empty((n_pts, 3), dtype=np.float32)
    for i in range(n_pts):
        base = i * step
        xyz[i, 0] = struct.unpack_from(fmt + "f", data, base + ox)[0]
        xyz[i, 1] = struct.unpack_from(fmt + "f", data, base + oy)[0]
        xyz[i, 2] = struct.unpack_from(fmt + "f", data, base + oz)[0]
    return xyz


def _pc2_to_xyz_fast(msg: Pc2Msg) -> np.ndarray:
    """
    Fast vectorised decoder — uses numpy structured array when fields
    are packed as 3 consecutive float32 starting at offset 0.
    Falls back to the slow loop version otherwise.
    """
    field_map: dict[str, int] = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"}.issubset(field_map):
        return np.empty((0, 3), dtype=np.float32)

    ox, oy, oz = field_map["x"], field_map["y"], field_map["z"]
    step = msg.point_step
    n    = msg.width * msg.height

    # Fast path: x/y/z are three consecutive float32s at the start
    if ox == 0 and oy == 4 and oz == 8:
        raw = np.frombuffer(bytes(msg.data), dtype=np.float32)
        pts = raw.reshape(-1, step // 4)[:, :3].copy()
        return pts.astype(np.float32)

    return _pc2_to_xyz(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_cloud(bag_path: Path, topic: str, max_frames: int) -> np.ndarray:
    """Return Nx3 (x,y,z) numpy array from up to *max_frames* PointCloud2 msgs."""
    print(f"[Cloud] Reading up to {max_frames} frames on '{topic}' …", flush=True)

    reader  = _make_reader(bag_path)
    filter_ = rosbag2_py.StorageFilter(topics=[topic])
    reader.set_filter(filter_)

    chunks: list[np.ndarray] = []
    n_frames = 0

    while reader.has_next():
        _, raw, _ = reader.read_next()
        msg = deserialize_message(raw, Pc2Msg)
        xyz = _pc2_to_xyz_fast(msg)
        if len(xyz) == 0:
            continue
        # Drop NaN / Inf
        valid = np.isfinite(xyz).all(axis=1)
        xyz   = xyz[valid]
        if len(xyz) == 0:
            continue
        chunks.append(xyz)
        n_frames += 1
        if max_frames > 0 and n_frames >= max_frames:
            print(f"[Cloud] stopped after {n_frames} frames", flush=True)
            break

    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _draw_rect(ax, lo1, hi1, lo2, hi2, color="darkorange", lw=2, label=None):
    """Draw an axis-aligned rectangle on *ax* given axis-span pairs."""
    rect = mpatches.FancyBboxPatch(
        (lo1, lo2), hi1 - lo1, hi2 - lo2,
        boxstyle="square,pad=0", linewidth=lw,
        edgecolor=color, facecolor="none", label=label, zorder=5,
    )
    ax.add_patch(rect)


def visualise(xyz_all: np.ndarray, xyz_in: np.ndarray,
              cfg: dict, output_path: str) -> None:
    """
    Three 2-D projection plots:
      - XY  : top-down   (x = forward, y = lateral)
      - XZ  : side view  (x = forward, z = vertical)
      - YZ  : front view (y = lateral, z = vertical)
    """
    x_min, x_max = cfg["x_min"], cfg["x_max"]
    y_min, y_max = cfg["y_min"], cfg["y_max"]
    z_min, z_max = cfg["z_min"], cfg["z_max"]

    MAX_PLOT_PTS = 200_000
    if len(xyz_all) > MAX_PLOT_PTS:
        idx    = np.random.choice(len(xyz_all), MAX_PLOT_PTS, replace=False)
        xyz_bg = xyz_all[idx]
    else:
        xyz_bg = xyz_all

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Distance Filter Visualiser", fontsize=14, fontweight="bold")

    projections = [
        (axes[0],
         xyz_bg[:, 0], xyz_bg[:, 1],
         xyz_in[:, 0] if len(xyz_in) else np.array([]),
         xyz_in[:, 1] if len(xyz_in) else np.array([]),
         xyz_in[:, 0] if len(xyz_in) else np.array([]),
         x_min, x_max, y_min, y_max,
         "x (forward) [m]", "y (lateral) [m]", "XY  top-down"),
        (axes[1],
         xyz_bg[:, 0], xyz_bg[:, 2],
         xyz_in[:, 0] if len(xyz_in) else np.array([]),
         xyz_in[:, 2] if len(xyz_in) else np.array([]),
         xyz_in[:, 0] if len(xyz_in) else np.array([]),
         x_min, x_max, z_min, z_max,
         "x (forward) [m]", "z (up) [m]", "XZ  side view"),
        (axes[2],
         xyz_bg[:, 1], xyz_bg[:, 2],
         xyz_in[:, 1] if len(xyz_in) else np.array([]),
         xyz_in[:, 2] if len(xyz_in) else np.array([]),
         xyz_in[:, 0] if len(xyz_in) else np.array([]),
         y_min, y_max, z_min, z_max,
         "y (lateral) [m]", "z (up) [m]", "YZ  front view"),
    ]

    for (ax,
         bg_h, bg_v,
         in_h, in_v, in_c,
         box_h_lo, box_h_hi, box_v_lo, box_v_hi,
         xlabel, ylabel, title) in projections:

        ax.scatter(bg_h, bg_v, s=0.2, c="#aaaaaa", rasterized=True, label="all pts")

        if len(in_h):
            sc = ax.scatter(in_h, in_v, s=0.8, c=in_c, cmap="plasma",
                            rasterized=True, label="inside filter", zorder=4)
            plt.colorbar(sc, ax=ax, label="x (forward) [m]", fraction=0.04, pad=0.02)

        _draw_rect(ax, box_h_lo, box_h_hi, box_v_lo, box_v_hi,
                   color="darkorange", lw=1.5, label="filter box")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", markerscale=5, fontsize=7)
        ax.grid(True, alpha=0.3)

    pct   = 100.0 * len(xyz_in) / max(len(xyz_all), 1)
    stats = (
        f"Total pts: {len(xyz_all):,}    Inside filter: {len(xyz_in):,}  ({pct:.1f} %)\n"
        f"Filter  x=[{x_min}, {x_max}]   y=[{y_min}, {y_max}]   z=[{z_min}, {z_max}]\n"
    )
    if len(xyz_in):
        stats += (
            f"Filtered cloud bbox:\n"
            f"  x=[{xyz_in[:,0].min():.2f}, {xyz_in[:,0].max():.2f}]   "
            f"  y=[{xyz_in[:,1].min():.2f}, {xyz_in[:,1].max():.2f}]   "
            f"  z=[{xyz_in[:,2].min():.2f}, {xyz_in[:,2].max():.2f}]"
        )
    fig.text(0.5, 0.01, stats, ha="center", va="bottom",
             fontsize=8, family="monospace",
             bbox=dict(facecolor="#ffffcc", alpha=0.8, edgecolor="#cccc00"))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[Filter] Saved visualisation → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise passthrough filter bounds on a ROS2 LiDAR bag.")
    p.add_argument("--bag",         required=True,
                   help="LiDAR bag — directory (with metadata.yaml), .db3, or .mcap file")
    p.add_argument("--config",      default=None,
                   help="Path to qr_params.yaml  (default: config/qr_params.yaml)")
    p.add_argument("--lidar-topic", default=None,
                   help="LiDAR topic  (auto-detected if omitted)")
    p.add_argument("--max-frames",  type=int, default=30, metavar="N",
                   help="LiDAR frames to accumulate (default: 30)")
    p.add_argument("--output",      default=None,
                   help="Output PNG path  (default: output/filter_check.png)")
    p.add_argument("--list-topics", action="store_true",
                   help="Print available topics and exit")
    return p.parse_args()


def main():
    args  = parse_args()
    _root = _HERE.parent

    bag_path    = Path(args.bag).expanduser().resolve()
    config_path = args.config or str(_root / "config" / "qr_params.yaml")
    output_path = args.output or str(_root / "output" / "filter_check.png")

    if not bag_path.exists():
        sys.exit(f"[ERROR] Bag not found: {bag_path}")

    topics = _list_topics(bag_path)

    if args.list_topics:
        print(f"\nTopics in {bag_path}:")
        for name, typ in sorted(topics.items()):
            print(f"  {name:<55} {typ}")
        return

    if not Path(config_path).exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")

    cfg = load_config(config_path)
    print(f"[Filter] Config: {config_path}")
    print(f"[Filter] Bounds: "
          f"x=[{cfg['x_min']}, {cfg['x_max']}]  "
          f"y=[{cfg['y_min']}, {cfg['y_max']}]  "
          f"z=[{cfg['z_min']}, {cfg['z_max']}]")

    lidar_topic = args.lidar_topic or _auto_lidar_topic(topics)
    if not lidar_topic:
        sys.exit(
            f"[ERROR] Cannot detect LiDAR topic.\n"
            f"  Available topics: {list(topics.keys())}\n"
            f"  Specify one with --lidar-topic"
        )
    print(f"[Filter] LiDAR topic: {lidar_topic}")

    xyz_all = _accumulate_cloud(bag_path, lidar_topic, args.max_frames)
    if len(xyz_all) == 0:
        sys.exit(f"[ERROR] No point cloud data found on '{lidar_topic}'.")
    print(f"[Cloud] Total accumulated: {len(xyz_all):,} pts")

    xyz_in = passthrough_filter(xyz_all, cfg)
    print(f"[Filter] Points inside filter: {len(xyz_in):,}  "
          f"({100.*len(xyz_in)/max(len(xyz_all),1):.1f} %)")

    if len(xyz_in) == 0:
        print("[WARNING] No points inside filter box — box may be too tight "
              "or in the wrong coordinate frame.  Check x/y/z min/max in config.")
    else:
        print(f"[Filter] Filtered cloud bbox:\n"
              f"  x=[{xyz_in[:,0].min():.3f}, {xyz_in[:,0].max():.3f}]\n"
              f"  y=[{xyz_in[:,1].min():.3f}, {xyz_in[:,1].max():.3f}]\n"
              f"  z=[{xyz_in[:,2].min():.3f}, {xyz_in[:,2].max():.3f}]")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    visualise(xyz_all, xyz_in, cfg, output_path)


if __name__ == "__main__":
    main()
