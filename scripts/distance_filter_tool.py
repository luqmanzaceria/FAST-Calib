#!/usr/bin/env python3
"""
distance_filter_tool.py — Visualise the passthrough filter bounds on a
                           point cloud extracted from a ROS2 bag.

Install dependencies:
    pip install "rosbags[mcap]" numpy "opencv-python>=4.5" pyyaml matplotlib

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
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; change to "TkAgg" if you want a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fast_calib_core import load_config, decode_pointcloud2, passthrough_filter

# Import bag-reading helpers from calib_from_bag_ros2 (protected by __main__ guard)
from calib_from_bag_ros2 import (
    _open_bag,
    _open_reader,
    _topics_from_reader,
    _iter_topic,
    _auto_lidar_topic,
)


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate_cloud(bag_path: Path, bag_files: list[Path],
                      topic: str, max_frames: int) -> np.ndarray:
    """Return Nx3 (x,y,z) numpy array from up to *max_frames* PointCloud2 msgs."""
    chunks: list[np.ndarray] = []
    n_frames = 0
    print(f"[Cloud] Reading up to {max_frames} frames on '{topic}' …", flush=True)
    with _open_reader(bag_path, bag_files) as reader:
        for msg in _iter_topic(reader, topic):
            pts = decode_pointcloud2(msg)        # Nx4 [x,y,z,ring]
            if pts is None or len(pts) == 0:
                continue
            chunks.append(pts[:, :3].astype(np.float32))
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
        edgecolor=color, facecolor="none", label=label, zorder=5
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

    # Subsample for plotting speed
    MAX_PLOT_PTS = 200_000
    if len(xyz_all) > MAX_PLOT_PTS:
        idx = np.random.choice(len(xyz_all), MAX_PLOT_PTS, replace=False)
        xyz_bg = xyz_all[idx]
    else:
        xyz_bg = xyz_all

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Distance Filter Visualiser", fontsize=14, fontweight="bold")

    projections = [
        # (ax, horiz_axis, vert_axis, horiz_label, vert_label,
        #  bg_h, bg_v, in_h, in_v,
        #  box_h_lo, box_h_hi, box_v_lo, box_v_hi, title)
        (axes[0],
         xyz_bg[:, 0], xyz_bg[:, 1],  # background XY
         xyz_in[:, 0], xyz_in[:, 1],  # filtered  XY  (colour by x=depth)
         xyz_in[:, 0],                 # colour value
         x_min, x_max, y_min, y_max,
         "x (forward) [m]", "y (lateral) [m]",
         "XY  top-down"),
        (axes[1],
         xyz_bg[:, 0], xyz_bg[:, 2],
         xyz_in[:, 0], xyz_in[:, 2],
         xyz_in[:, 0],
         x_min, x_max, z_min, z_max,
         "x (forward) [m]", "z (up) [m]",
         "XZ  side view"),
        (axes[2],
         xyz_bg[:, 1], xyz_bg[:, 2],
         xyz_in[:, 1], xyz_in[:, 2],
         xyz_in[:, 0],
         y_min, y_max, z_min, z_max,
         "y (lateral) [m]", "z (up) [m]",
         "YZ  front view"),
    ]

    for (ax,
         bg_h, bg_v,
         in_h, in_v, in_c,
         box_h_lo, box_h_hi, box_v_lo, box_v_hi,
         xlabel, ylabel, title) in projections:

        # Background cloud
        ax.scatter(bg_h, bg_v, s=0.2, c="#aaaaaa", rasterized=True, label="all pts")

        # Filtered points coloured by forward depth
        if len(in_h):
            sc = ax.scatter(in_h, in_v, s=0.8, c=in_c, cmap="plasma",
                            rasterized=True, label="inside filter", zorder=4)
            plt.colorbar(sc, ax=ax, label="x (forward) [m]", fraction=0.04, pad=0.02)

        # Filter box
        _draw_rect(ax, box_h_lo, box_h_hi, box_v_lo, box_v_hi,
                   color="darkorange", lw=1.5, label="filter box")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", markerscale=5, fontsize=7)
        ax.grid(True, alpha=0.3)

    # Stats summary as figure text
    pct = 100.0 * len(xyz_in) / max(len(xyz_all), 1)
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
                   help="LiDAR bag — directory, .db3 file, or .mcap file")
    p.add_argument("--config",      default=None,
                   help="Path to qr_params.yaml  (default: config/qr_params.yaml)")
    p.add_argument("--lidar-topic", default=None,
                   help="LiDAR topic  (auto-detected if omitted)")
    p.add_argument("--max-frames",  type=int, default=30, metavar="N",
                   help="Number of LiDAR frames to accumulate (default: 30)")
    p.add_argument("--output",      default=None,
                   help="Output PNG path  (default: output/filter_check.png)")
    return p.parse_args()


def main():
    args  = parse_args()
    _root = _HERE.parent

    config_path = args.config or str(_root / "config" / "qr_params.yaml")
    output_path = args.output or str(_root / "output" / "filter_check.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not Path(config_path).exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")

    cfg = load_config(config_path)
    print(f"[Filter] Config: {config_path}")
    print(f"[Filter] Bounds: x=[{cfg['x_min']}, {cfg['x_max']}]  "
          f"y=[{cfg['y_min']}, {cfg['y_max']}]  z=[{cfg['z_min']}, {cfg['z_max']}]")

    bag_path, bag_files = _open_bag(args.bag, "LiDAR bag")

    with _open_reader(bag_path, bag_files) as r:
        topics = _topics_from_reader(r)

    lidar_topic = args.lidar_topic or _auto_lidar_topic(topics)
    if not lidar_topic:
        sys.exit(f"[ERROR] Cannot detect LiDAR topic.  "
                 f"Available topics: {topics}\n"
                 f"Specify one with --lidar-topic")
    print(f"[Filter] LiDAR topic: {lidar_topic}")

    xyz_all = _accumulate_cloud(bag_path, bag_files, lidar_topic, args.max_frames)
    if len(xyz_all) == 0:
        sys.exit(f"[ERROR] No point cloud data found on '{lidar_topic}'.")
    print(f"[Cloud] Total accumulated: {len(xyz_all):,} pts")

    # Wrap Nx3 as Nx4 (fast_calib_core passthrough_filter expects Nx4 column order
    # but only uses [:, :3], so padding a zero 4th column is fine)
    xyz_in = passthrough_filter(xyz_all, cfg)
    print(f"[Filter] Points inside filter: {len(xyz_in):,}  "
          f"({100.*len(xyz_in)/max(len(xyz_all),1):.1f} %)")

    if len(xyz_in) == 0:
        print("[WARNING] No points inside filter box — the box may be too tight "
              "or in the wrong coordinate frame.  Check x/y/z min/max in config.")
    else:
        print(f"[Filter] Filtered cloud bbox:\n"
              f"  x=[{xyz_in[:,0].min():.3f}, {xyz_in[:,0].max():.3f}]\n"
              f"  y=[{xyz_in[:,1].min():.3f}, {xyz_in[:,1].max():.3f}]\n"
              f"  z=[{xyz_in[:,2].min():.3f}, {xyz_in[:,2].max():.3f}]")

    visualise(xyz_all, xyz_in, cfg, output_path)


if __name__ == "__main__":
    main()
