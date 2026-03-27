#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
1) 自动检测 rosbag 中雷达点云类型：
   - sensor_msgs/msg/PointCloud2  (如 /hesai/pandar)
   - livox_ros_driver2/msg/CustomMsg (如 /livox/lidar)
2) 按各自的解析方式把点云导出成一个带 intensity 的 PCD 文件 (x y z intensity, ASCII)
3) 使用 Open3D 对该 PCD 进行交互选点（至少 4 个点），并根据 4 个点计算包围范围，
   保存为同名 .txt 文件。

依赖：
    - rosbag2_py
    - rclpy
    - sensor_msgs_py
    - open3d, numpy

用法示例：
    python distance_filter_tool.py
    python distance_filter_tool.py /path/to/data /path/to/output_dir
"""

import os
import sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2 as pc2
import open3d as o3d


# ===================== rosbag2 helper =====================

def open_bag_reader(bag_path):
    """Open a rosbag2 SequentialReader, trying common storage plug-ins."""
    for storage_id in ['', 'sqlite3', 'mcap']:
        try:
            storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr',
            )
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
            return reader
        except Exception:
            continue
    raise RuntimeError(f"[ERROR] 无法打开 bag: {bag_path}")


def build_type_map(reader):
    """Return {topic_name: msg_type_str} from bag metadata."""
    return {ti.name: ti.type for ti in reader.get_all_topics_and_types()}

# ===================== 通用：保存 PCD =====================

def save_pcd_with_intensity(points, intensities, output_path):
    """
    保存点云为带 intensity 字段的 PCD 文件 (ASCII 格式)
    points: ndarray shape (N, 3)
    intensities: ndarray shape (N,)
    """
    points = np.asarray(points, dtype=np.float32)
    intensities = np.asarray(intensities, dtype=np.float32).reshape(-1, 1)
    data = np.hstack([points, intensities])
    N = len(data)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {N}\n"
        "HEIGHT 1\n"
        f"POINTS {N}\n"
        "DATA ascii\n"
    )
    print(f"[PCD] 写入 {N} 个点到: {output_path}", flush=True)
    with open(output_path, 'w') as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.6f")
    print(f"[PCD] 保存完成: {output_path}", flush=True)

# ===================== 情况 1：PointCloud2 =====================

def find_intensity_field(msg):
    """
    在 PointCloud2 的 fields 中自动检测强度字段名称。
    优先匹配常见名称；找不到时返回 None。
    """
    candidates = ["intensity", "reflectivity", "i", "ref", "ring", "time", "t"]
    field_names = [f.name for f in msg.fields]
    for name in field_names:
        if name.lower() in candidates:
            return name
    print(f"[Bag] 可用字段: {field_names}，无匹配强度字段")
    return None


def convert_pointcloud2_bag_to_pcd(
    bag_file,
    output_dir,
    topic_name="/hesai/pandar",
    pcd_name="sensor_PointCloud2_inten_ascii.pcd",
    max_frames=None
):
    """
    将 rosbag2 中 PointCloud2 类型的点云合并导出为一个 PCD 文件。
    保持原始雷达坐标，不做坐标变换。
    """
    print(f"[Bag] 打开 rosbag2: {bag_file}")
    reader = open_bag_reader(bag_file)
    type_map = build_type_map(reader)

    if topic_name not in type_map:
        print(f"[ERROR] bag 中未找到 topic '{topic_name}'", file=sys.stderr)
        print("[ERROR] bag 中可用 topics:", file=sys.stderr)
        for t, mt in sorted(type_map.items()):
            print(f"  {t}  [{mt}]", file=sys.stderr)
        return None

    msg_class = get_message(type_map[topic_name])

    # 1) 先检测强度字段（只读第一条匹配消息，检查一次即止）
    intensity_field = None
    reader2 = open_bag_reader(bag_file)
    while reader2.has_next():
        topic, data, _ = reader2.read_next()
        if topic == topic_name:
            msg = deserialize_message(data, msg_class)
            intensity_field = find_intensity_field(msg)
            if intensity_field:
                print(f"[Bag] 检测到 intensity 字段: {intensity_field}")
            else:
                print("[WARN] 未找到强度字段，将以强度=0 导出纯 XYZ 点云。")
            break  # 只检查第一条消息

    # 2) 读取指定 topic 的所有点云（用 numpy 批量操作，避免逐点循环）
    chunks_xyz = []
    chunks_inten = []
    msg_count = 0

    limit_str = f"（最多 {max_frames} 帧）" if max_frames else ""
    print(f"[Bag] 开始从 topic '{topic_name}' 读取 PointCloud2 点云{limit_str}...", flush=True)

    reader3 = open_bag_reader(bag_file)
    while reader3.has_next():
        if max_frames and msg_count >= max_frames:
            break
        topic, data, _ = reader3.read_next()
        if topic != topic_name:
            continue
        try:
            msg = deserialize_message(data, msg_class)

            if intensity_field:
                # columns: x=0  y=1  z=2  intensity=3
                arr = pc2.read_points_numpy(
                    msg, field_names=["x", "y", "z", intensity_field], skip_nans=True
                )
                if arr is not None and len(arr) > 0:
                    chunks_xyz.append(arr[:, 0:3].astype(np.float32))
                    chunks_inten.append(arr[:, 3].astype(np.float32))
            else:
                # columns: x=0  y=1  z=2
                arr = pc2.read_points_numpy(
                    msg, field_names=["x", "y", "z"], skip_nans=True
                )
                if arr is not None and len(arr) > 0:
                    chunks_xyz.append(arr[:, 0:3].astype(np.float32))
                    chunks_inten.append(np.zeros(len(arr), dtype=np.float32))

            msg_count += 1
            if msg_count % 50 == 0:
                total = sum(len(c) for c in chunks_xyz)
                print(f"[Bag]   已处理 {msg_count} 帧，累计 {total} 个点...", flush=True)

        except Exception as e:
            print(f"[ERROR] 读取第 {msg_count+1} 帧时出错: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            continue

    print(f"[Bag] 共读取 {msg_count} 帧", flush=True)

    if not chunks_xyz:
        print("[ERROR] 未找到 PointCloud2 点云数据！", file=sys.stderr)
        return None

    all_points = np.vstack(chunks_xyz)
    all_intensities = np.concatenate(chunks_inten)
    print(f"[Bag] 总点数: {len(all_points)}", flush=True)

    output_path = os.path.join(output_dir, pcd_name)
    save_pcd_with_intensity(all_points, all_intensities, output_path)
    return output_path

# ===================== 情况 2：Livox CustomMsg =====================

def parse_livox_custom_msg(msg):
    """
    从 livox_ros_driver/CustomMsg 中解析 x, y, z, reflectivity
    假设 msg.points 是 CustomPoint 对象列表，字段为 x, y, z, reflectivity
    """
    points = []
    intensities = []

    for pt in msg.points:
        points.append([pt.x, pt.y, pt.z])
        intensities.append(pt.reflectivity)

    return points, intensities

def convert_livox_custom_bag_to_pcd(
    bag_file,
    output_dir,
    topic_name="/livox/lidar",
    pcd_name="livox_CustomMsg_inten_ascii.pcd"
):
    """
    将 rosbag2 中 livox_ros_driver2/msg/CustomMsg 类型的点云合并导出为一个 PCD 文件。
    保持原始雷达坐标，不做坐标变换。
    """
    print(f"[Bag] 打开 rosbag2: {bag_file}")
    reader = open_bag_reader(bag_file)
    type_map = build_type_map(reader)

    if topic_name not in type_map:
        print(f"[ERROR] bag 中未找到 topic '{topic_name}'", file=sys.stderr)
        print("[ERROR] bag 中可用 topics:", file=sys.stderr)
        for t, mt in sorted(type_map.items()):
            print(f"  {t}  [{mt}]", file=sys.stderr)
        return None

    msg_class = get_message(type_map[topic_name])

    all_points = []
    all_intensities = []

    print(f"[Bag] 开始从 topic '{topic_name}' 读取 CustomMsg 点云...")

    reader2 = open_bag_reader(bag_file)
    while reader2.has_next():
        topic, data, _ = reader2.read_next()
        if topic != topic_name:
            continue
        try:
            msg = deserialize_message(data, msg_class)
            pts, intens = parse_livox_custom_msg(msg)
            all_points.extend(pts)
            all_intensities.extend(intens)
        except Exception as e:
            print(f"[ERROR] 读取错误: {str(e)}", file=sys.stderr)
            continue

    if not all_points:
        print("[ERROR] 未找到 Livox CustomMsg 点云数据!", file=sys.stderr)
        return None

    output_path = os.path.join(output_dir, pcd_name)
    intensities = np.array(all_intensities, dtype=np.float32)
    save_pcd_with_intensity(all_points, intensities, output_path)
    return output_path

# ===================== 自动检测：这个 bag 用哪种方式 =====================

def detect_lidar_msg_type(bag_file):
    """
    通过 bag 的 topic 元数据检测是否有 PointCloud2 或 Livox CustomMsg。
    返回：
        ("PointCloud2", topic_name), ("CustomMsg", topic_name), 或 (None, None)
    如果两种都有，默认优先 PointCloud2 并打印提示。
    """
    print(f"[Detect] 扫描 bag: {bag_file}")
    reader = open_bag_reader(bag_file)
    type_map = build_type_map(reader)

    print("[Detect] bag 中的 topics:")
    for t, mt in sorted(type_map.items()):
        print(f"  {t}  [{mt}]")

    pc2_topics = [t for t, mt in type_map.items() if mt == "sensor_msgs/msg/PointCloud2"]
    livox_topics = [t for t, mt in type_map.items() if "CustomMsg" in mt]

    if pc2_topics and livox_topics:
        print("[Detect] 同时检测到 PointCloud2 和 Livox CustomMsg, 默认使用 PointCloud2。")
        return "PointCloud2", pc2_topics[0]
    elif pc2_topics:
        print(f"[Detect] 检测到 PointCloud2 点云，使用 topic: {pc2_topics[0]}")
        return "PointCloud2", pc2_topics[0]
    elif livox_topics:
        print(f"[Detect] 检测到 Livox CustomMsg 点云，使用 topic: {livox_topics[0]}")
        return "CustomMsg", livox_topics[0]
    else:
        print("[Detect] 未检测到 PointCloud2 或 Livox CustomMsg 点云。")
        return None, None

# ===================== Open3D 交互选点 & 保存范围 =====================

def color_by_height(pcd):
    """Color points by z-height using a jet-like gradient for easier scene reading."""
    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max > z_min:
        t = (z - z_min) / (z_max - z_min)
    else:
        t = np.zeros_like(z)
    # jet: blue -> cyan -> green -> yellow -> red
    r = np.clip(1.5 - np.abs(t * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(t * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(t * 4 - 1), 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(np.stack([r, g, b], axis=1))
    return pcd


def select_and_save_points(pcd_folder, target_pcd_name, voxel_size=0.05, crop_distance=None):
    """
    在给定目录中读取指定 PCD 文件，用 Open3D 交互式选点并保存范围。
    voxel_size:     可视化前体素降采样分辨率（米），0 表示不降采样
    crop_distance:  只显示距离原点 <= N 米的点（None 表示不裁剪）
    """
    pcd_path = os.path.join(pcd_folder, target_pcd_name)
    if not os.path.isfile(pcd_path):
        print(f"[ERROR] 指定的 PCD 文件不存在: {pcd_path}", file=sys.stderr)
        return

    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"[ERROR] {target_pcd_name} 中没有点云数据，已跳过", file=sys.stderr)
        return

    pts = np.asarray(pcd.points)
    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    print(f"\n正在处理: {target_pcd_name}  ({len(pcd.points):,} 个点)")
    print(f"  空间范围  X: [{bb_min[0]:.2f}, {bb_max[0]:.2f}]  "
          f"Y: [{bb_min[1]:.2f}, {bb_max[1]:.2f}]  "
          f"Z: [{bb_min[2]:.2f}, {bb_max[2]:.2f}]")

    pcd_vis = pcd

    # 距离裁剪（只保留近处点，便于找到标定板）
    if crop_distance and crop_distance > 0:
        dists = np.linalg.norm(pts, axis=1)
        mask = dists <= crop_distance
        pcd_vis = pcd_vis.select_by_index(np.where(mask)[0])
        print(f"[Crop] 距离 <= {crop_distance}m：{len(pcd_vis.points):,} 个点")
        if not pcd_vis.has_points():
            print("[WARN] 裁剪后无点，改用全部点云", file=sys.stderr)
            pcd_vis = pcd

    # 体素降采样
    if voxel_size > 0:
        pcd_vis = pcd_vis.voxel_down_sample(voxel_size)
        print(f"[VoxelDown] 降采样后: {len(pcd_vis.points):,} 个点 (voxel={voxel_size}m)")

    # 按高度着色，方便识别场景结构
    pcd_vis = color_by_height(pcd_vis)

    print("\n操作说明:")
    print("  旋转:  左键拖拽")
    print("  平移:  中键拖拽  或  Ctrl + 左键拖拽")
    print("  缩放:  滚轮")
    print("  选点:  Shift + 左键单击（至少选 4 个标定板角点）")
    print("  完成:  按 Q 关闭窗口\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"选择点 - {target_pcd_name}", width=1280, height=720)
    vis.add_geometry(pcd_vis)
    # 添加坐标轴，帮助定向
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # 等待用户交互（Shift+左键选点, Q 退出）
    vis.run()
    vis.destroy_window()

    # 获取用户选择的点的索引
    selected_indices = vis.get_picked_points()

    if not selected_indices:
        print(f"[ERROR] 未选择任何点，{target_pcd_name} 没有保存文件", file=sys.stderr)
        return

    if len(selected_indices) < 4:
        print(f"[ERROR] 只选中了 {len(selected_indices)} 个点，少于 4 个，跳过该文件", file=sys.stderr)
        return

    selected_indices = selected_indices[:4]

    # 选点来自降采样后的点云，直接取其坐标
    all_points = np.asarray(pcd_vis.points)
    selected_points = all_points[selected_indices, :]  # 形状 (4, 3)

    # 计算四个点在各轴上的最小值和最大值
    mins = selected_points.min(axis=0)  # [x_min_raw, y_min_raw, z_min_raw]
    maxs = selected_points.max(axis=0)  # [x_max_raw, y_max_raw, z_max_raw]

    # 按你的定义扩展 0.2m
    x_min = mins[0] - 0.2
    x_max = maxs[0] + 0.2
    y_min = mins[1] - 0.2
    y_max = maxs[1] + 0.2
    z_min = mins[2] - 0.2
    z_max = maxs[2] + 0.2

    # 生成保存文件名 (与 PCD 文件同名，改为 txt)
    base_name = os.path.splitext(target_pcd_name)[0]
    save_file = os.path.join(pcd_folder, f"{base_name}.txt")

    with open(save_file, 'w') as f:
        f.write("# 4 selected points (x y z)\n")
        for p in selected_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        f.write("# range values in order:\n")
        f.write(f"x_min: {x_min:.1f}\n")
        f.write(f"x_max: {x_max:.1f}\n")
        f.write(f"y_min: {y_min:.1f}\n")
        f.write(f"y_max: {y_max:.1f}\n")
        f.write(f"z_min: {z_min:.1f}\n")
        f.write(f"z_max: {z_max:.1f}\n")

    print(f"[Save] 已保存选点与范围到: {save_file}")
    print("点云处理完成。")

# ===================== main =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="从 rosbag2 中提取点云并交互式选点生成距离过滤范围"
    )
    parser.add_argument(
        "--bag", "-b",
        default=None,
        help="rosbag2 路径（目录）；与 --pcd 二选一"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.getcwd(),
        help="输出目录（默认：当前目录）"
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="指定点云 topic（默认：PointCloud2 用 /hesai/pandar，CustomMsg 用 /livox/lidar）"
    )
    parser.add_argument(
        "--pcd",
        default=None,
        help="直接使用已有的 PCD 文件，跳过 bag 读取（仅做交互选点）"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="最多读取 N 帧点云（减少点数，加快处理；默认读取全部帧）"
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.05,
        metavar="M",
        help="可视化前体素降采样分辨率（米），0 表示不降采样（默认 0.05）"
    )
    parser.add_argument(
        "--crop",
        type=float,
        default=None,
        metavar="M",
        help="只显示距离原点 <= M 米的点，便于定位标定板（例如 --crop 10）"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="（保留参数，暂未使用）配置文件路径"
    )
    args = parser.parse_args()

    if not args.bag and not args.pcd:
        parser.error("必须提供 --bag 或 --pcd 之一")

    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] 输出目录不存在，已创建: {output_dir}")

    # -- 直接使用已有 PCD，跳过 bag 读取 --
    if args.pcd:
        pcd_path = args.pcd
        if not os.path.isfile(pcd_path):
            print(f"[ERROR] PCD 文件不存在: {pcd_path}", file=sys.stderr)
            sys.exit(1)
        select_and_save_points(
            pcd_folder=os.path.dirname(os.path.abspath(pcd_path)),
            target_pcd_name=os.path.basename(pcd_path),
            voxel_size=args.voxel,
            crop_distance=args.crop,
        )
        sys.exit(0)

    # -- 从 bag 读取 --
    bag_file = args.bag
    if not os.path.exists(bag_file):
        print(f"[ERROR] bag 路径 '{bag_file}' 不存在", file=sys.stderr)
        sys.exit(1)

    msg_type, detected_topic = detect_lidar_msg_type(bag_file)
    if msg_type is None:
        print("[ERROR] 未检测到支持的雷达消息类型，退出。", file=sys.stderr)
        sys.exit(1)

    if msg_type == "PointCloud2":
        topic = args.topic or detected_topic
        pcd_path = convert_pointcloud2_bag_to_pcd(
            bag_file=bag_file,
            output_dir=output_dir,
            topic_name=topic,
            pcd_name="sensor_PointCloud2_inten_ascii.pcd",
            max_frames=args.max_frames,
        )
    else:  # "CustomMsg"
        topic = args.topic or detected_topic
        pcd_path = convert_livox_custom_bag_to_pcd(
            bag_file=bag_file,
            output_dir=output_dir,
            topic_name=topic,
            pcd_name="livox_CustomMsg_inten_ascii.pcd",
        )

    if pcd_path is None:
        print("[ERROR] PCD 生成失败，退出。", file=sys.stderr)
        sys.exit(1)

    select_and_save_points(
        pcd_folder=output_dir,
        target_pcd_name=os.path.basename(pcd_path),
        voxel_size=args.voxel,
        crop_distance=args.crop,
    )
