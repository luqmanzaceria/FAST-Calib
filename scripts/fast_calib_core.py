#!/usr/bin/env python3
"""
fast_calib_core.py — Pure-Python port of the FAST-Calib detection pipeline.

No ROS required.  Dependencies (pip install):
    numpy opencv-python open3d pyyaml

Implements the same algorithms as the C++ nodes:
  • QR / ArUco board detection  → 4 circle centers in camera frame
  • LiDAR circle detection       → 4 circle centers in LiDAR frame
    – solid-state path (Livox / generic, no ring field)
    – mechanical path  (ring-based edge detection)
  • SVD rigid-transform estimation
  • Geometric consistency check  (Square.is_valid)
  • sort_pattern_centers
  • save_calibration_results
"""

from __future__ import annotations
import os
import struct
import yaml
import numpy as np
import cv2
import cv2.aruco as aruco
import open3d as o3d
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# Constants (mirror C++ #defines)
# ────────────────────────────────────────────────────────────────────────────
TARGET_NUM_CIRCLES  = 4
GEOMETRY_TOLERANCE  = 0.08

# ════════════════════════════════════════════════════════════════════════════
# 1.  Config
# ════════════════════════════════════════════════════════════════════════════

_DEFAULTS = dict(
    fx=1215.318, fy=1214.730, cx=1047.866, cy=745.068,
    k1=-0.33575, k2=0.10997, p1=1.573e-4, p2=5.449e-4,
    marker_size=0.16,
    delta_width_qr_center=0.55, delta_height_qr_center=0.35,
    delta_width_circles=0.50,   delta_height_circles=0.40,
    circle_radius=0.10, min_detected_markers=3,
    x_min=2.0, x_max=6.0, y_min=-1.0, y_max=4.0, z_min=-0.5, z_max=2.5,
    lidar_topic="/livox/lidar",
    image_path="", bag_path="", output_path="./output",
)


def load_config(yaml_path: str) -> dict:
    """Load qr_params.yaml; fall back to hard-coded defaults for missing keys."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = dict(_DEFAULTS)
    for k, v in raw.items():
        if isinstance(v, (int, float, str, bool)):
            cfg[k] = v
    return cfg


# ════════════════════════════════════════════════════════════════════════════
# 2.  Message decoders (work on rosbags-deserialized messages)
# ════════════════════════════════════════════════════════════════════════════

# PointCloud2 datatype codes → numpy dtype strings
_PC2_DTYPE = {1: "i1", 2: "u1", 3: "i2", 4: "u2",
              5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def decode_pointcloud2(msg) -> np.ndarray:
    """
    Decode a sensor_msgs/PointCloud2 (rosbags or rclpy) message to an Nx4
    float32 array  [x, y, z, ring].  ring = 65535 if the field is absent.
    """
    step = int(msg.point_step)
    n    = int(msg.width) * int(msg.height)

    # Build structured dtype with per-field offsets
    fields_needed = ["x", "y", "z"]
    has_ring = any(f.name == "ring" for f in msg.fields)
    if has_ring:
        fields_needed.append("ring")

    field_map = {f.name: (int(f.offset), _PC2_DTYPE.get(int(f.datatype), "f4"))
                 for f in msg.fields}

    names   = [nm for nm in fields_needed if nm in field_map]
    formats = [field_map[nm][1] for nm in names]
    offsets = [field_map[nm][0] for nm in names]

    dt = np.dtype({"names": names, "formats": formats,
                   "offsets": offsets, "itemsize": step})

    raw = msg.data if isinstance(msg.data, (bytes, bytearray)) else bytes(msg.data)
    pts = np.frombuffer(raw, dtype=dt)

    x = pts["x"].astype(np.float32)
    y = pts["y"].astype(np.float32)
    z = pts["z"].astype(np.float32)
    ring = pts["ring"].astype(np.float32) if has_ring \
           else np.full(n, 0xFFFF, np.float32)

    out = np.column_stack([x, y, z, ring])
    valid = np.isfinite(out[:, :3]).all(axis=1)
    return out[valid]


def decode_image_msg(msg) -> np.ndarray:
    """Convert sensor_msgs/Image or sensor_msgs/CompressedImage to BGR uint8."""
    raw = msg.data if isinstance(msg.data, (bytes, bytearray)) else bytes(msg.data)

    # CompressedImage has no 'encoding' field; Image always does.
    # (Don't use hasattr(msg, 'format') — every Python object inherits __format__.)
    if not hasattr(msg, 'encoding'):
        arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            fmt = getattr(msg, 'format', '?')
            raise ValueError(f"cv2.imdecode failed for compressed image (format={fmt})")
        return bgr

    h, w = int(msg.height), int(msg.width)
    enc  = msg.encoding

    if enc in ("bgr8", "rgb8"):
        arr = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        return arr.copy() if enc == "bgr8" else arr[:, :, ::-1].copy()
    if enc == "mono8":
        return cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(h, w),
                            cv2.COLOR_GRAY2BGR)
    if enc in ("16UC1", "mono16"):
        arr16 = np.frombuffer(raw, np.uint16).reshape(h, w)
        return cv2.cvtColor((arr16 >> 8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Fallback – try bgr8
    try:
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    except Exception:
        raise ValueError(f"Unsupported image encoding: {enc}")


# ════════════════════════════════════════════════════════════════════════════
# 3.  QR / ArUco detection  (mirrors qr_detect.hpp)
# ════════════════════════════════════════════════════════════════════════════

def detect_qr_centers(image: np.ndarray, cfg: dict):
    """
    Detect the 4 circle-center 3-D positions on the calibration target
    using ArUco board pose estimation.

    Returns:
        centers_cam : (4, 3) float64  – circle centers in camera frame
        annotated   : BGR image with drawn detections
    On failure returns (None, annotated_image).
    """
    K    = np.array([[cfg["fx"],        0, cfg["cx"]],
                     [       0, cfg["fy"], cfg["cy"]],
                     [       0,        0,          1]], dtype=np.float64)
    dist = np.array([cfg["k1"], cfg["k2"], cfg["p1"], cfg["p2"], 0.0])

    ms   = cfg["marker_size"]
    dw   = cfg["delta_width_qr_center"]
    dh   = cfg["delta_height_qr_center"]
    cw   = cfg["delta_width_circles"]  / 2.0
    ch   = cfg["delta_height_circles"] / 2.0
    min_m = int(cfg.get("min_detected_markers", 3))

    # ── Board geometry (mirrors C++ boardCorners / boardCircleCenters) ──────
    # Marker layout: [ID1 TL, ID2 TR, ID4 BR, ID3 BL]
    qr_cx = [-dw, +dw, +dw, -dw]
    qr_cy = [+dh, +dh, -dh, -dh]
    half  = ms / 2.0

    board_corners = []
    for xc, yc in zip(qr_cx, qr_cy):
        board_corners.append(np.array([
            [xc - half, yc + half, 0],   # j=0 top-left
            [xc + half, yc + half, 0],   # j=1 top-right
            [xc + half, yc - half, 0],   # j=2 bottom-right
            [xc - half, yc - half, 0],   # j=3 bottom-left
        ], dtype=np.float32))

    board_ids   = np.array([1, 2, 4, 3])
    circ_board  = np.array([[-cw, +ch, 0], [+cw, +ch, 0],
                             [+cw, -ch, 0], [-cw, -ch, 0]], dtype=np.float32)

    gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    annotated  = image.copy()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # Detect markers – handle old (≤4.6) and new (≥4.7) OpenCV API
    try:
        det_params = aruco.DetectorParameters()
        det_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector  = aruco.ArucoDetector(aruco_dict, det_params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        det_params = aruco.DetectorParameters_create()
        det_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=det_params)

    if ids is None or len(ids) < min_m:
        n = len(ids) if ids is not None else 0
        print(f"[QR] {n} markers found, need >= {min_m}")
        return None, annotated

    aruco.drawDetectedMarkers(annotated, corners, ids)

    # Build board object
    try:
        board = aruco.Board(board_corners, aruco_dict, board_ids)
    except (TypeError, AttributeError):
        board = aruco.Board_create(board_corners, aruco_dict, board_ids)  # type: ignore

    # Initial pose guess from single markers (average)
    rvec = np.zeros(3, np.float64)
    tvec = np.zeros(3, np.float64)
    try:
        rvecs_s, tvecs_s, _ = aruco.estimatePoseSingleMarkers(corners, ms, K, dist)
        tvec = np.mean([t[0] for t in tvecs_s], axis=0)
        sin_r = np.mean([np.sin(r[0]) for r in rvecs_s], axis=0)
        cos_r = np.mean([np.cos(r[0]) for r in rvecs_s], axis=0)
        rvec  = np.arctan2(sin_r, cos_r)
    except Exception:
        pass

    # Board pose estimation
    n_valid = 0
    try:
        n_valid, rvec, tvec = aruco.estimatePoseBoard(
            corners, ids, board, K, dist, rvec, tvec, True)
    except Exception:
        try:
            n_valid, rvec, tvec = aruco.estimatePoseBoard(
                corners, ids, board, K, dist, rvec, tvec)
        except Exception as e:
            # Newer OpenCV: use board.matchImagePoints + solvePnP
            try:
                obj_pts, img_pts = board.matchImagePoints(corners, ids)
                if obj_pts is not None and len(obj_pts) >= 4:
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
                    n_valid = 1 if ok else 0
            except Exception as e2:
                print(f"[QR] pose estimation failed: {e2}")
                return None, annotated

    if not n_valid:
        print("[QR] estimatePoseBoard: 0 valid markers")
        return None, annotated

    # Draw board axis
    try:
        cv2.drawFrameAxes(annotated, K, dist, rvec, tvec, 0.2)
    except AttributeError:
        try:
            aruco.drawAxis(annotated, K, dist, rvec, tvec, 0.2)  # type: ignore
        except Exception:
            pass

    # Transform circle centers: board frame → camera frame
    R_board, _ = cv2.Rodrigues(rvec)
    centers_cam = (R_board @ circ_board.T).T + tvec.reshape(1, 3)

    # Draw projected centers
    for c3 in centers_cam:
        uv, _ = cv2.projectPoints(c3.reshape(1, 1, 3).astype(np.float32),
                                   np.zeros(3), np.zeros(3), K, np.zeros(5))
        u, v = int(uv[0, 0, 0]), int(uv[0, 0, 1])
        if 0 <= u < annotated.shape[1] and 0 <= v < annotated.shape[0]:
            cv2.circle(annotated, (u, v), 6, (0, 255, 0), -1)

    return centers_cam.astype(np.float64), annotated


# ════════════════════════════════════════════════════════════════════════════
# 4.  Geometry helpers
# ════════════════════════════════════════════════════════════════════════════

def passthrough_filter(pts: np.ndarray, cfg: dict) -> np.ndarray:
    """Keep points inside the bounding box specified in cfg."""
    m = ((pts[:, 0] >= cfg["x_min"]) & (pts[:, 0] <= cfg["x_max"]) &
         (pts[:, 1] >= cfg["y_min"]) & (pts[:, 1] <= cfg["y_max"]) &
         (pts[:, 2] >= cfg["z_min"]) & (pts[:, 2] <= cfg["z_max"]))
    return pts[m]


def _rodrigues_rotation(n_from: np.ndarray, n_to: np.ndarray) -> np.ndarray:
    """Return 3×3 rotation matrix rotating unit vector n_from → n_to."""
    n_from = n_from / np.linalg.norm(n_from)
    n_to   = n_to   / np.linalg.norm(n_to)
    axis   = np.cross(n_from, n_to)
    s      = np.linalg.norm(axis)
    c      = np.dot(n_from, n_to)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    axis /= s
    K  = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def align_plane_to_z0(pts_xyz: np.ndarray, plane_normal: np.ndarray):
    """
    Rotate pts_xyz so that plane_normal aligns with +Z.
    Returns (pts_xy: Nx2, R_align: 3×3, avg_z: float)
    """
    n  = np.array(plane_normal, float)
    n /= np.linalg.norm(n)
    R  = _rodrigues_rotation(n, np.array([0., 0., 1.]))
    aligned = (R @ pts_xyz.T).T
    avg_z   = float(aligned[:, 2].mean())
    return aligned[:, :2], R, avg_z


def _voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Grid voxel downsample — pure numpy, no open3d PointCloud needed."""
    coords = np.floor(xyz / voxel_size).astype(np.int64)
    # Use a structured dtype for fast np.unique over 3-tuples
    dt = np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64)])
    keys = np.empty(len(coords), dtype=dt)
    keys['x'] = coords[:, 0]
    keys['y'] = coords[:, 1]
    keys['z'] = coords[:, 2]
    _, first = np.unique(keys, return_index=True)
    return xyz[first]


def _plane_fit_ransac(pts_xyz: np.ndarray,
                      dist_thr: float = 0.02,
                      n_iter: int = 500) -> tuple:
    """
    Pure-numpy RANSAC plane fit — no open3d required.
    Returns (normal 3-vec, inlier index array).
    """
    n = len(pts_xyz)
    # Random subsample to keep each iteration fast
    _MAX = 20_000
    if n > _MAX:
        pts_xyz = pts_xyz[np.random.choice(n, _MAX, replace=False)]
        n = _MAX

    best_normal  = np.array([0., 0., 1.])
    best_inliers = np.empty(0, dtype=np.intp)
    rng = np.random.default_rng(0)

    for _ in range(n_iter):
        s = rng.choice(n, 3, replace=False)
        v1 = pts_xyz[s[1]] - pts_xyz[s[0]]
        v2 = pts_xyz[s[2]] - pts_xyz[s[0]]
        nrm = np.cross(v1, v2)
        length = np.linalg.norm(nrm)
        if length < 1e-10:
            continue
        nrm /= length
        d = -float(nrm @ pts_xyz[s[0]])
        dists = np.abs(pts_xyz @ nrm + d)
        inliers = np.where(dists < dist_thr)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal  = nrm.copy()

    return best_normal, best_inliers


def _boundary_indices(pts_2d: np.ndarray, radius: float = 0.03,
                      min_gap: float = np.pi / 4) -> np.ndarray:
    """
    Return indices of boundary points using the max-angular-gap criterion.
    Pure numpy (squared-distance brute force) — no open3d Vector3dVector.
    Fast enough for the typical few-thousand-point plane cloud.
    """
    if len(pts_2d) < 3:
        return np.arange(len(pts_2d))

    r2 = radius * radius
    n_pts       = len(pts_2d)
    is_boundary = np.ones(n_pts, dtype=bool)

    for i, p in enumerate(pts_2d):
        diff = pts_2d - p                          # (N, 2)
        sq   = (diff * diff).sum(axis=1)
        nbrs = np.where((sq < r2) & (sq > 0))[0]
        if len(nbrs) < 2:
            continue
        angles = np.arctan2(diff[nbrs, 1], diff[nbrs, 0])
        a      = np.sort(angles)
        gaps   = np.diff(a)
        wrap   = 2 * np.pi + a[0] - a[-1]
        is_boundary[i] = float(max(gaps.max(), wrap)) > min_gap

    return np.where(is_boundary)[0]


def _circle_from_3(p1, p2, p3):
    """Circle centre through three 2-D points, or None if degenerate."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-10:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
          + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
          + (cx**2 + cy**2) * (bx - ax)) / D
    return (ux, uy)


def ransac_circle_2d(pts_2d: np.ndarray, target_r: float,
                     dist_thr: float = 0.02,
                     max_iter: int = 500,
                     min_inliers: int = 5):
    """
    RANSAC 2-D circle fitting.
    Returns (center_xy, inlier_indices) or (None, []).
    """
    n = len(pts_2d)
    if n < 3:
        return None, []
    best_inliers: list = []
    best_center           = None
    rng = np.random.default_rng(42)

    for _ in range(max_iter):
        idx    = rng.choice(n, 3, replace=False)
        center = _circle_from_3(pts_2d[idx[0]], pts_2d[idx[1]], pts_2d[idx[2]])
        if center is None:
            continue
        dists    = np.hypot(pts_2d[:, 0] - center[0], pts_2d[:, 1] - center[1])
        # Skip if estimated radius is wildly wrong
        if abs(float(np.median(dists)) - target_r) > target_r * 0.5:
            continue
        inliers  = np.where(np.abs(dists - target_r) < dist_thr)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = list(inliers)
            best_center  = center

    if len(best_inliers) < min_inliers:
        return None, []
    return best_center, best_inliers


def sort_pattern_centers(pts: np.ndarray, mode: str = "camera") -> np.ndarray:
    """
    Sort 4 3-D points by CCW angle around their centroid.
    mode="lidar" first converts LiDAR → camera convention, sorts,
    then converts back (mirrors C++ sortPatternCenters).
    """
    pts = np.array(pts, dtype=np.float64)
    if mode == "lidar":
        work = np.column_stack([-pts[:, 1], -pts[:, 2], pts[:, 0]])
    else:
        work = pts.copy()

    centroid = work.mean(axis=0)
    angles   = np.arctan2(work[:, 1] - centroid[1],
                          work[:, 0] - centroid[0])
    order    = np.argsort(angles)
    sw       = work[order]

    # Ensure CCW orientation
    v01 = sw[1, :2] - sw[0, :2]
    v12 = sw[2, :2] - sw[1, :2]
    if float(np.cross(v01, v12)) > 0:
        sw[[1, 3]] = sw[[3, 1]]

    if mode == "lidar":
        result = np.column_stack([sw[:, 2], -sw[:, 0], -sw[:, 1]])
        return result
    return sw


def is_valid_square(pts: np.ndarray, width: float, height: float) -> bool:
    """Geometric consistency check (mirrors C++ Square::is_valid)."""
    pts = np.array(pts, dtype=np.float64)
    if len(pts) != 4:
        return False
    diag     = np.hypot(width, height)
    centroid = pts.mean(axis=0)
    for p in pts:
        d = np.linalg.norm(p - centroid)
        if abs(d - diag / 2) / (diag / 2) > GEOMETRY_TOLERANCE * 2.0:
            return False
    sp = sort_pattern_centers(pts, "camera")

    def sd(i, j):
        return float(np.linalg.norm(sp[i] - sp[j]))

    s = [sd(i, (i + 1) % 4) for i in range(4)]
    t  = GEOMETRY_TOLERANCE
    p1 = (abs(s[0]-width)/width   < t and abs(s[1]-height)/height < t and
          abs(s[2]-width)/width   < t and abs(s[3]-height)/height < t)
    p2 = (abs(s[0]-height)/height < t and abs(s[1]-width)/width   < t and
          abs(s[2]-height)/height < t and abs(s[3]-width)/width   < t)
    if not (p1 or p2):
        return False
    perim = sum(s)
    ideal = 2 * (width + height)
    return abs(perim - ideal) / ideal <= GEOMETRY_TOLERANCE


def _pick_best_4(centers_3d: np.ndarray, cfg: dict):
    """
    Choose the group of 4 points (from potentially more candidates) that
    passes the geometric consistency check.  Returns 4×3 or None.
    """
    from itertools import combinations
    w  = cfg["delta_width_circles"]
    h  = cfg["delta_height_circles"]
    n  = len(centers_3d)
    if n < TARGET_NUM_CIRCLES:
        return None
    for group in combinations(range(n), TARGET_NUM_CIRCLES):
        pts = centers_3d[list(group)]
        if is_valid_square(pts, w, h):
            return pts
    return None


# ════════════════════════════════════════════════════════════════════════════
# 5.  LiDAR detection
# ════════════════════════════════════════════════════════════════════════════

def detect_solid_lidar(pts_N4: np.ndarray, cfg: dict):
    """
    Solid-state LiDAR circle detection (mirrors detect_solid_lidar in C++).
    pts_N4: Nx4 [x, y, z, ring]
    Returns: (4, 3) float64 circle centers in LiDAR frame, or None.
    """
    # 1. Passthrough filter
    xyz = passthrough_filter(pts_N4[:, :3], cfg)
    print(f"[LiDAR-solid] after filter: {len(xyz)} pts")
    if len(xyz) < 20:
        return None

    # 2. Voxel downsample
    xyz = _voxel_downsample(xyz, 0.02)
    print(f"[LiDAR-solid] after voxel: {len(xyz)} pts")

    # 3. RANSAC plane
    normal, inliers = _plane_fit_ransac(xyz)
    plane_pts       = xyz[inliers]
    print(f"[LiDAR-solid] plane inliers: {len(plane_pts)} pts")

    # 4. Align to Z=0
    pts_2d, R_align, avg_z = align_plane_to_z0(plane_pts, normal)

    # 5. Boundary detection
    bdry_idx = _boundary_indices(pts_2d, radius=0.03, min_gap=np.pi / 4)
    bdry_2d  = pts_2d[bdry_idx]
    print(f"[LiDAR-solid] boundary pts: {len(bdry_2d)}")
    if len(bdry_2d) < 20:
        return None

    # 6. DBSCAN clustering of boundary points
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(
        np.column_stack([bdry_2d, np.zeros(len(bdry_2d))]))
    labels = np.asarray(pcd_b.cluster_dbscan(
        eps=0.05, min_points=10, print_progress=False))
    print(f"[LiDAR-solid] DBSCAN clusters: {labels.max() + 1 if labels.max() >= 0 else 0}")

    # 7. Per-cluster circle fitting
    r_target = cfg["circle_radius"]
    centers_2d = []
    for lbl in range(int(labels.max()) + 1):
        clust = bdry_2d[labels == lbl]
        if len(clust) < 5:
            continue
        center, inliers_c = ransac_circle_2d(clust, r_target, dist_thr=0.02)
        if center is None:
            continue
        dists = np.hypot(clust[inliers_c, 0] - center[0],
                         clust[inliers_c, 1] - center[1])
        err   = float(np.mean(np.abs(dists - r_target)))
        if err < 0.030:
            centers_2d.append(center)

    print(f"[LiDAR-solid] circles found: {len(centers_2d)}")
    if len(centers_2d) < TARGET_NUM_CIRCLES:
        return None

    # 8. Transform circle centres back to LiDAR frame
    R_inv = np.linalg.inv(R_align)
    centers_3d = np.array([R_inv @ np.array([cx, cy, avg_z])
                            for cx, cy in centers_2d])
    return _pick_best_4(centers_3d, cfg)


def detect_mech_lidar(pts_N4: np.ndarray, cfg: dict):
    """
    Mechanical (ring-based) LiDAR circle detection (mirrors detect_mech_lidar).
    pts_N4: Nx4 [x, y, z, ring]
    Returns: (4, 3) float64 circle centers in LiDAR frame, or None.
    """
    # 1. Passthrough filter
    filt = passthrough_filter(pts_N4, cfg)
    print(f"[LiDAR-mech] after filter: {len(filt)} pts")
    if len(filt) < 20:
        return None

    # 2. RANSAC plane
    normal, inliers = _plane_fit_ransac(filt[:, :3])
    plane_pts  = filt[inliers]
    a, b, c    = normal
    norm_n     = float(np.linalg.norm(normal))
    print(f"[LiDAR-mech] plane inliers: {len(plane_pts)} pts")

    # 3. Ring-based edge detection
    GAP_THR   = 0.10   # C++: neighbor_gap_threshold
    MIN_PTS   = 10     # C++: min_points_per_ring
    PLANE_THR = 0.03   # C++: dist_plane < 0.03

    rings     = {}
    for pt in filt:
        r = int(pt[3])
        rings.setdefault(r, []).append(pt)

    edge_pts = []
    for pts_ring in rings.values():
        if len(pts_ring) < MIN_PTS:
            continue
        pr = np.array(pts_ring)
        for k in range(1, len(pr) - 1):
            px, py, pz = pr[k, :3]
            # Only keep points close to plane
            dp = abs(a * px + b * py + c * pz
                     + (a * plane_pts[0, 0] + b * plane_pts[0, 1]
                        + c * plane_pts[0, 2])) / norm_n
            # Recompute plane dist correctly with plane eq ax+by+cz+d=0
            # Use simpler: dist from fitted plane cloud centroid plane
            if dp >= PLANE_THR:
                continue
            d_prev = float(np.linalg.norm(pr[k, :3] - pr[k-1, :3]))
            d_next = float(np.linalg.norm(pr[k, :3] - pr[k+1, :3]))
            if d_prev > GAP_THR or d_next > GAP_THR:
                edge_pts.append(pr[k, :3])

    print(f"[LiDAR-mech] edge pts: {len(edge_pts)}")
    if len(edge_pts) < 12:
        return None

    edge_arr = np.array(edge_pts)

    # 4. Align to Z=0
    pts_2d, R_align, avg_z = align_plane_to_z0(edge_arr, normal)

    # 5. Iterative RANSAC circle (remove inliers between iterations)
    r_target    = cfg["circle_radius"]
    remaining   = pts_2d.copy()
    centers_2d  = []
    MIN_INLIERS = 5

    while len(remaining) > 3 and len(centers_2d) < TARGET_NUM_CIRCLES + 2:
        center, inliers = ransac_circle_2d(remaining, r_target,
                                           dist_thr=0.02, max_iter=500,
                                           min_inliers=MIN_INLIERS)
        if center is None:
            break
        centers_2d.append(center)
        remaining = np.delete(remaining, inliers, axis=0)

    print(f"[LiDAR-mech] circles found: {len(centers_2d)}")
    if len(centers_2d) < TARGET_NUM_CIRCLES:
        return None

    # 6. Transform back to LiDAR frame
    R_inv = np.linalg.inv(R_align)
    centers_3d = np.array([R_inv @ np.array([cx, cy, avg_z])
                            for cx, cy in centers_2d])
    return _pick_best_4(centers_3d, cfg)


def detect_lidar(pts_N4: np.ndarray, cfg: dict):
    """Auto-detect lidar type and run appropriate pipeline."""
    has_ring = np.any(pts_N4[:, 3] != 0xFFFF)
    if has_ring:
        print("[LiDAR] Ring field detected → mechanical pipeline")
        result = detect_mech_lidar(pts_N4, cfg)
        if result is not None:
            return result, "mech"
        print("[LiDAR] Mechanical detection failed, trying solid-state pipeline")
    result = detect_solid_lidar(pts_N4, cfg)
    return result, "solid"


# ════════════════════════════════════════════════════════════════════════════
# 6.  SVD transform estimation
# ════════════════════════════════════════════════════════════════════════════

def estimate_rigid_transform(lidar_pts: np.ndarray,
                              cam_pts:   np.ndarray) -> np.ndarray:
    """
    Estimate 4×4 T_cam_lidar such that cam_pts ≈ T @ lidar_pts (homogeneous)
    via SVD (mirrors pcl::TransformationEstimationSVD).
    """
    L  = np.array(lidar_pts, float)
    C  = np.array(cam_pts,   float)
    mL = L.mean(axis=0)
    mC = C.mean(axis=0)
    Sigma = (L - mL).T @ (C - mC)
    U, _, Vt = np.linalg.svd(Sigma)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        D = np.diag([1., 1., -1.])
        R = Vt.T @ D @ U.T
    t = mC - R @ mL
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T.astype(np.float32)


def compute_rmse(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    diff = np.array(pts_a, float) - np.array(pts_b, float)
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# ════════════════════════════════════════════════════════════════════════════
# 7.  Save results (mirrors saveCalibrationResults)
# ════════════════════════════════════════════════════════════════════════════

def project_to_image(pts_xyz: np.ndarray, T: np.ndarray,
                     K: np.ndarray, dist: np.ndarray,
                     image: np.ndarray) -> np.ndarray:
    """Color point cloud by projecting into image. Returns Nx6 [x,y,z,r,g,b]."""
    img_ud = cv2.undistort(image, K, dist)
    h, w   = img_ud.shape[:2]
    R34    = T[:3, :4]
    pts_h  = np.column_stack([pts_xyz,
                               np.ones(len(pts_xyz))]).T   # 4×N
    pts_cam = (R34 @ pts_h).T                              # N×3
    mask_front = pts_cam[:, 2] > 0
    pts_cam    = pts_cam[mask_front]

    K32 = K.astype(np.float32)
    pts_proj, _ = cv2.projectPoints(pts_cam.astype(np.float32),
                                     np.zeros(3), np.zeros(3),
                                     K32, np.zeros(5))
    uvs   = pts_proj.reshape(-1, 2)
    ui    = uvs[:, 0].astype(int)
    vi    = uvs[:, 1].astype(int)
    valid = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

    colored = []
    for k, (u, v, ok) in enumerate(zip(ui, vi, valid)):
        if ok:
            bgr = img_ud[v, u]
            colored.append([*pts_cam[k], float(bgr[2]),
                             float(bgr[1]), float(bgr[0])])
    return np.array(colored, dtype=np.float32) if colored else np.zeros((0, 6), np.float32)


def save_results(T: np.ndarray, cfg: dict, output_dir: str,
                 annotated_image: np.ndarray,
                 cloud_xyz: np.ndarray | None = None,
                 tag: str = "single") -> None:
    """Save calibration result, colored PCD, and annotated image."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, tag)

    # ── text result ──────────────────────────────────────────────────────────
    txt_path = prefix + "_calib_result.txt"
    R = T[:3, :3]
    t = T[:3, 3]
    with open(txt_path, "w") as f:
        f.write("# FAST-LIVO2 calibration format\n")
        f.write("cam_model: Pinhole\n")
        for k in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"):
            f.write(f"cam_{k}: {cfg[k]}\n")
        f.write(f"\nRcl: [{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f},\n")
        f.write(f"      {R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f},\n")
        f.write(f"      {R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}]\n")
        f.write(f"Pcl: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n")
    print(f"[Result] Saved: {txt_path}")

    # ── annotated image ───────────────────────────────────────────────────────
    img_path = prefix + "_qr_detect.png"
    cv2.imwrite(img_path, annotated_image)
    print(f"[Result] Saved: {img_path}")

    # ── colored PCD ──────────────────────────────────────────────────────────
    if cloud_xyz is not None and len(cloud_xyz) > 0:
        K_mat  = np.array([[cfg["fx"],0,cfg["cx"]],[0,cfg["fy"],cfg["cy"]],[0,0,1]], np.float32)
        d_mat  = np.array([cfg["k1"],cfg["k2"],cfg["p1"],cfg["p2"],0.], np.float32)
        colored = project_to_image(cloud_xyz, T, K_mat, d_mat, annotated_image)
        if len(colored):
            pcd_path = prefix + "_colored.pcd"
            pcd      = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(colored[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(colored[:, 3:] / 255.0)
            o3d.io.write_point_cloud(pcd_path, pcd)
            print(f"[Result] Saved: {pcd_path}")

    # ── print to stdout ───────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════╗")
    print("║       FAST-Calib: T_cam_lidar result          ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  R:\n{R}")
    print(f"  t: {t}")
    print("──────────────────────────────────────────────")


# ════════════════════════════════════════════════════════════════════════════
# 8.  Top-level calibration runner
# ════════════════════════════════════════════════════════════════════════════

def run_calibration(image: np.ndarray,
                    pts_N4: np.ndarray,
                    cfg: dict,
                    output_dir: str,
                    tag: str = "single"):
    """
    Full pipeline: image + cloud → T_cam_lidar.
    Returns (T 4×4, rmse) or (None, None) on failure.
    """
    # Camera
    qr_centers, annotated = detect_qr_centers(image, cfg)
    if qr_centers is None:
        print("[Calib] QR detection failed.")
        return None, None

    # LiDAR
    lidar_centers, _ = detect_lidar(pts_N4, cfg)
    if lidar_centers is None:
        print("[Calib] LiDAR detection failed.")
        return None, None

    # Sort consistently
    qr_sorted    = sort_pattern_centers(qr_centers,    "camera")
    lidar_sorted = sort_pattern_centers(lidar_centers, "lidar")

    # SVD
    T    = estimate_rigid_transform(lidar_sorted, qr_sorted)
    T_l  = lidar_sorted @ T[:3, :3].T + T[:3, 3]
    rmse = compute_rmse(T_l, qr_sorted)
    print(f"[Calib] RMSE: {rmse:.4f} m")

    save_results(T, cfg, output_dir, annotated, pts_N4[:, :3], tag=tag)
    return T, rmse
