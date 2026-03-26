#!/usr/bin/env python3
"""
calib_from_bag_ros2.py — Offline LiDAR-camera extrinsic calibration
                          from a ROS2 bag (.db3 or .mcap).

No ROS installation required.  Install dependencies with:
    pip install "rosbags[mcap]" numpy "opencv-python>=4.5" open3d pyyaml

Usage:
    python3 scripts/calib_from_bag_ros2.py \\
        --bag        /path/to/lidar_bag   \\
        --camera-bag /path/to/camera_bag  \\   # omit if same bag
        --config     config/qr_params.yaml \\
        [--image         /path/to/image.png]   \\
        [--image-topic   /camera/image_raw]    \\
        [--lidar-topic   /sensor_scan]         \\
        [--output-dir    output/]

The bag path may be:
  • a bag DIRECTORY containing *.db3 or *.mcap files
  • the *.db3 FILE itself  (works even without metadata.yaml)
  • the *.mcap FILE itself

If --image is omitted the script scans the image topic for the sharpest
frame that has >= min_detected_markers ArUco markers visible.
"""

from __future__ import annotations
import argparse
import os
import sqlite3
import struct
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from fast_calib_core import (
    load_config, decode_pointcloud2, decode_image_msg, run_calibration
)

# ════════════════════════════════════════════════════════════════════════════
# Minimal CDR parser — no rosbags required
# Handles sensor_msgs/msg/Image and sensor_msgs/msg/PointCloud2
# ════════════════════════════════════════════════════════════════════════════

class _CdrParser:
    """Walk through a CDR-encoded ROS2 message byte-by-byte."""

    def __init__(self, raw: bytes) -> None:
        if len(raw) < 4:
            raise ValueError("CDR data too short")
        self._raw    = raw
        self._pos    = 4                         # skip 4-byte encapsulation header
        self._endian = '<' if (raw[1] & 1) else '>'

    def _align(self, n: int) -> None:
        r = self._pos % n
        if r:
            self._pos += n - r

    def uint8(self) -> int:
        v = self._raw[self._pos]; self._pos += 1; return v

    def int32(self) -> int:
        self._align(4)
        v, = struct.unpack_from(self._endian + 'i', self._raw, self._pos)
        self._pos += 4; return v

    def uint32(self) -> int:
        self._align(4)
        v, = struct.unpack_from(self._endian + 'I', self._raw, self._pos)
        self._pos += 4; return v

    def string(self) -> str:
        length = self.uint32()           # includes null terminator
        if length == 0:
            return ''
        s = self._raw[self._pos:self._pos + length - 1].decode('utf-8', errors='replace')
        self._pos += length; return s

    def read_bytes(self, n: int) -> bytes:
        v = self._raw[self._pos:self._pos + n]; self._pos += n; return v


class _FakeImageMsg:
    __slots__ = ('height', 'width', 'encoding', 'data')
    def __init__(self, height, width, encoding, data):
        self.height = height; self.width = width
        self.encoding = encoding; self.data = data


class _FakeField:
    __slots__ = ('name', 'offset', 'datatype')
    def __init__(self, name, offset, datatype):
        self.name = name; self.offset = offset; self.datatype = datatype


class _FakePc2Msg:
    __slots__ = ('height', 'width', 'fields', 'point_step', 'row_step', 'data')
    def __init__(self, height, width, fields, point_step, row_step, data):
        self.height = height; self.width = width; self.fields = fields
        self.point_step = point_step; self.row_step = row_step; self.data = data


def _cdr_parse_image(raw: bytes):
    """Decode sensor_msgs/msg/Image without rosbags."""
    try:
        p = _CdrParser(raw)
        p.int32(); p.uint32()        # Header.stamp (sec, nanosec)
        p.string()                   # Header.frame_id
        height   = p.uint32()
        width    = p.uint32()
        encoding = p.string()
        p.uint8()                    # is_bigendian
        p.uint32()                   # step (auto-aligns past is_bigendian)
        data = p.read_bytes(p.uint32())
        return _FakeImageMsg(height, width, encoding, data)
    except Exception:
        return None


def _cdr_parse_pc2(raw: bytes):
    """Decode sensor_msgs/msg/PointCloud2 without rosbags."""
    try:
        p = _CdrParser(raw)
        p.int32(); p.uint32()        # Header.stamp
        p.string()                   # Header.frame_id
        height = p.uint32()
        width  = p.uint32()
        fields = []
        for _ in range(p.uint32()):  # PointField[] sequence
            name     = p.string()
            offset   = p.uint32()
            datatype = p.uint8()
            p.uint32()               # count (auto-aligns past datatype)
            fields.append(_FakeField(name, offset, datatype))
        p.uint8()                    # is_bigendian
        point_step = p.uint32()      # auto-aligns
        row_step   = p.uint32()
        data = p.read_bytes(p.uint32())
        return _FakePc2Msg(height, width, fields, point_step, row_step, data)
    except Exception:
        return None


class _FakeCompressedImageMsg:
    __slots__ = ('format', 'data')
    def __init__(self, fmt, data):
        self.format = fmt; self.data = data


def _cdr_parse_compressed_image(raw: bytes):
    """Decode sensor_msgs/msg/CompressedImage without rosbags."""
    try:
        p = _CdrParser(raw)
        p.int32(); p.uint32()        # Header.stamp
        p.string()                   # Header.frame_id
        fmt  = p.string()            # format, e.g. "jpeg", "png"
        data = p.read_bytes(p.uint32())
        return _FakeCompressedImageMsg(fmt, data)
    except Exception:
        return None


_CDR_FALLBACK = {
    "sensor_msgs/msg/Image":            _cdr_parse_image,
    "sensor_msgs/msg/CompressedImage":  _cdr_parse_compressed_image,
    "sensor_msgs/msg/PointCloud2":      _cdr_parse_pc2,
}

# ════════════════════════════════════════════════════════════════════════════
# Path resolution — find bag file(s) (.db3 or .mcap) from user-supplied path
# ════════════════════════════════════════════════════════════════════════════

_BAG_SUFFIXES = (".db3", ".mcap")


def _resolve_bag_files(path: Path) -> list[Path]:
    """
    Return all bag files (.db3 or .mcap) to read, given a user-supplied path.
    Prefers .mcap over .db3 when both are present in the same directory.
    Searches one level deep into sub-directories.
    """
    path = path.resolve()
    if path.is_file() and path.suffix in _BAG_SUFFIXES:
        return [path]
    if path.is_dir():
        for suffix in _BAG_SUFFIXES:
            hits = sorted(path.glob(f"*{suffix}"))
            if hits:
                return hits
        # One level deeper
        for sub in sorted(path.iterdir()):
            if sub.is_dir():
                for suffix in _BAG_SUFFIXES:
                    hits = sorted(sub.glob(f"*{suffix}"))
                    if hits:
                        return hits
    return []


def _bag_format(bag_files: list[Path]) -> str:
    """Return 'mcap' or 'db3' based on the first file's suffix."""
    return bag_files[0].suffix.lstrip(".") if bag_files else "unknown"


def _has_metadata(bag_files: list[Path]) -> bool:
    return bool(bag_files) and (bag_files[0].parent / "metadata.yaml").exists()

# ════════════════════════════════════════════════════════════════════════════
# SQLite reader — works even without metadata.yaml
# ════════════════════════════════════════════════════════════════════════════

class _SqliteReader:
    """
    Direct SQLite reader for ROS2 .db3 files.
    Does NOT require metadata.yaml.
    Uses rosbags for CDR deserialization with automatic typestore selection.
    """

    def __init__(self, db3_files: list[Path]):
        self._db3_files = db3_files
        self._typestore  = None
        self._topics: dict[str, tuple[int, str]] = {}  # name → (topic_id, typename)
        self._conns: list[sqlite3.Connection] = []

    def __enter__(self):
        self._conns = [sqlite3.connect(str(p)) for p in self._db3_files]
        # Collect topic names from the first file (metadata is duplicated)
        for conn in self._conns:
            for tid, name, typename in conn.execute(
                    "SELECT id, name, type FROM topics"):
                self._topics.setdefault(name, (tid, typename))
        self._typestore = self._pick_typestore()
        if self._typestore is None:
            print("  [Bag] rosbags not installed — using built-in CDR decoder "
                  "(Image + PointCloud2 only).", flush=True)
        return self

    def __exit__(self, *_):
        for c in self._conns:
            c.close()

    def _pick_typestore(self):
        try:
            from rosbags.typesys import get_typestore, Stores
        except ImportError:
            return None
        for store in [
            "ROS2_HUMBLE", "ROS2_IRON", "ROS2_GALACTIC", "ROS2_FOXY", "EMPTY"
        ]:
            try:
                return get_typestore(getattr(Stores, store))
            except Exception:
                continue
        return None

    @property
    def topic_names(self) -> list[str]:
        return list(self._topics.keys())

    def typename_of(self, topic: str) -> str:
        return self._topics.get(topic, (None, ""))[1]

    def iter_messages(self, topic: str):
        """Yield deserialized messages for a single topic (all db3 files)."""
        typename = self.typename_of(topic)
        if not typename:
            return
        for conn in self._conns:
            tid_row = conn.execute(
                "SELECT id FROM topics WHERE name = ?", (topic,)).fetchone()
            if not tid_row:
                continue
            tid = tid_row[0]
            for (ts, raw) in conn.execute(
                    "SELECT timestamp, data FROM messages "
                    "WHERE topic_id = ? ORDER BY timestamp", (tid,)):
                msg = self._deserialize(bytes(raw), typename)
                if msg is not None:
                    yield ts, msg

    def _deserialize(self, raw: bytes, typename: str):
        # Try rosbags typestore first (richer; supports all message types)
        if self._typestore is not None:
            try:
                return self._typestore.deserialize_cdr(raw, typename)
            except Exception:
                try:
                    return self._typestore.deserialize_cdr(raw[4:], typename)
                except Exception:
                    pass
        # Fall back to built-in CDR parser for the two types we need
        fallback = _CDR_FALLBACK.get(typename)
        if fallback:
            return fallback(raw)
        return None


# ════════════════════════════════════════════════════════════════════════════
# High-level readers (try AnyReader first, fall back to SQLite for .db3)
# ════════════════════════════════════════════════════════════════════════════

def _get_typestore():
    """Return the best available rosbags typestore, or None."""
    try:
        from rosbags.typesys import get_typestore, Stores
    except ImportError:
        return None
    for name in ["ROS2_HUMBLE", "ROS2_IRON", "ROS2_GALACTIC", "ROS2_FOXY"]:
        try:
            return get_typestore(getattr(Stores, name))
        except Exception:
            continue
    return None


def _open_any_reader(bag_path: Path, bag_files: list[Path]):
    """
    Try rosbags.highlevel.AnyReader.
    • For .mcap files: pass them directly (no metadata.yaml needed).
    • For .db3 files: pass the directory if metadata.yaml exists.
    Always provides a default_typestore to avoid 'no type definitions' error.
    Returns an un-entered reader, or None if rosbags is unavailable.
    """
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        return None

    typestore = _get_typestore()
    kwargs = {"default_typestore": typestore} if typestore else {}

    fmt = _bag_format(bag_files)
    try:
        if fmt == "mcap":
            return AnyReader(bag_files, **kwargs)
        # db3: need the directory (AnyReader reads metadata.yaml from there)
        meta = bag_path / "metadata.yaml"
        if meta.exists():
            return AnyReader([bag_path], **kwargs)
    except Exception:
        pass
    return None


def _open_reader(bag_path: Path, bag_files: list[Path]):
    """
    Return a context manager that can iterate bag messages.
    Priority:
      1. rosbags AnyReader  — works for .mcap and .db3-with-metadata
      2. _SqliteReader      — direct SQLite fallback for .db3-without-metadata
    Exits with an error if .mcap files are present but rosbags is not installed.
    """
    any_r = _open_any_reader(bag_path, bag_files)
    if any_r is not None:
        return any_r

    fmt = _bag_format(bag_files)
    if fmt == "mcap":
        sys.exit(
            "[ERROR] MCAP bags require rosbags with mcap support.\n"
            "  Install with: pip install 'rosbags[mcap]'")

    # .db3 without metadata.yaml → direct SQLite reader
    return _SqliteReader(bag_files)


def _topics_from_reader(reader) -> list[str]:
    if isinstance(reader, _SqliteReader):
        return reader.topic_names
    # AnyReader
    return [c.topic for c in reader.connections]


def _iter_topic(reader, topic: str):
    """Yield (timestamp, deserialized_msg) for a topic, regardless of reader type."""
    if isinstance(reader, _SqliteReader):
        yield from reader.iter_messages(topic)
    else:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return   # topic not in bag — don't fall through to all-messages
        for conn, ts, raw in reader.messages(connections=conns):
            try:
                yield ts, reader.deserialize(raw, conn.msgtype)
            except Exception:
                continue


# ════════════════════════════════════════════════════════════════════════════
# Data extraction
# ════════════════════════════════════════════════════════════════════════════

def list_topics(bag_path: Path, bag_files: list[Path]) -> None:
    reader = _open_reader(bag_path, bag_files)
    print(f"\nTopics in {bag_path}:")
    if isinstance(reader, _SqliteReader):
        with reader:
            for name in reader.topic_names:
                print(f"  {name:<50s}  {reader.typename_of(name)}")
    else:
        with reader:
            for c in reader.connections:
                print(f"  {c.topic:<50s}  {c.msgtype}")


def _best_image_from_bag(bag_path: Path, bag_files: list[Path],
                          image_topic: str, min_markers: int,
                          aruco_dict,
                          save_any_frame: str | None = None) -> np.ndarray | None:
    best_img,     best_score,     best_n     = None, -1.0, 0
    best_aruco,   best_aruco_sh,  best_aruco_n = None, -1.0, 0  # best with ANY aruco
    n_iter = n_dec = n_aruco = 0
    first_enc: str | None = None
    first_dec_err: str | None = None

    try:
        det_params = aruco.DetectorParameters()
        detector   = aruco.ArucoDetector(aruco_dict, det_params)
        def _detect(gray):
            _, ids, _ = detector.detectMarkers(gray)
            return len(ids) if ids is not None else 0
    except AttributeError:
        det_params = aruco.DetectorParameters_create()
        def _detect(gray):
            _, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=det_params)
            return len(ids) if ids is not None else 0

    reader = _open_reader(bag_path, bag_files)
    with reader:
        for _ts, msg in _iter_topic(reader, image_topic):
            n_iter += 1
            try:
                bgr = decode_image_msg(msg)
                n_dec += 1
                if first_enc is None:
                    first_enc = getattr(msg, 'encoding', '?')
            except Exception as exc:
                if first_dec_err is None:
                    first_dec_err = str(exc)
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            n    = _detect(gray)
            if n > 0:
                n_aruco += 1
                sh = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if n > best_aruco_n or (n == best_aruco_n and sh > best_aruco_sh):
                    best_aruco, best_aruco_sh, best_aruco_n = bgr.copy(), sh, n
            if n >= min_markers:
                sh = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if n > best_n or (n == best_n and sh > best_score):
                    best_img, best_score, best_n = bgr.copy(), sh, n
                    print(f"  [Image] best so far: {n} markers, sharpness={sh:.1f}",
                          flush=True)

    # --save-any-frame: prefer the best ArUco frame, fall back to first decoded
    if save_any_frame:
        frame_to_save = best_aruco if best_aruco is not None else best_img
        if frame_to_save is not None:
            cv2.imwrite(save_any_frame, frame_to_save)
            label = (f"{best_aruco_n} ArUco markers"
                     if best_aruco is not None else "no ArUco")
            print(f"  [Image] saved best frame ({label}) → {save_any_frame}",
                  flush=True)

    if best_img is None:
        if n_iter == 0:
            print(f"  [Image] WARNING: 0 messages on topic '{image_topic}'.",
                  flush=True)
            print(f"          Check topic name with --list-topics.", flush=True)
        elif n_dec == 0:
            print(f"  [Image] WARNING: iterated {n_iter} msgs but decoded 0 images.",
                  flush=True)
            if first_dec_err:
                print(f"          decode error: {first_dec_err}", flush=True)
            print(f"          Try: pip install rosbags", flush=True)
        elif n_aruco == 0:
            print(f"  [Image] WARNING: decoded {n_dec}/{n_iter} frames "
                  f"(encoding={first_enc}) but no ArUco markers found in any.",
                  flush=True)
            print(f"          • Is the calibration board visible in the bag?",
                  flush=True)
            print(f"          • Use --save-any-frame output/raw_frame.png to inspect.",
                  flush=True)
        else:
            print(f"  [Image] WARNING: {n_aruco} frames had ArUco markers "
                  f"(best={best_aruco_n}) but none reached "
                  f"min_detected_markers={min_markers}.",
                  flush=True)
            print(f"          • Add --min-markers {best_aruco_n} to use the best "
                  f"available frame.", flush=True)
            if save_any_frame and best_aruco is not None:
                print(f"          • Inspect the frame saved to {save_any_frame}.",
                      flush=True)

    return best_img


def _read_cloud(bag_path: Path, bag_files: list[Path],
                lidar_topic: str,
                max_frames: int = 50) -> np.ndarray:
    """
    Read at most `max_frames` PointCloud2 messages and concatenate them.
    50 frames at 10 Hz = 5 s of data; more than enough for calibration.
    Use --max-cloud-frames 0 to read everything (may be very slow/large).
    """
    parts = []
    n_frames = 0
    reader = _open_reader(bag_path, bag_files)
    with reader:
        for _ts, msg in _iter_topic(reader, lidar_topic):
            if max_frames and n_frames >= max_frames:
                break
            try:
                pts = decode_pointcloud2(msg)
                if len(pts):
                    parts.append(pts)
                    n_frames += 1
            except Exception as e:
                print(f"  [Cloud] decode error: {e}", file=sys.stderr)
    if max_frames and n_frames >= max_frames:
        print(f"  [Cloud] stopped after {n_frames} frames "
              f"(use --max-cloud-frames 0 to read all)", flush=True)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 4), np.float32)


def _auto_lidar_topic(topics: list[str]) -> str | None:
    keywords = ["lidar", "points", "scan", "cloud"]
    # Prefer exact PointCloud2 topics by keyword
    for kw in keywords:
        for t in topics:
            if kw in t.lower():
                return t
    return topics[0] if topics else None


def _auto_image_topic(topics: list[str]) -> str | None:
    # 1. Prefer raw (uncompressed) color image topics
    for t in topics:
        tl = t.lower()
        if "image" in tl and "depth" not in tl and "compressed" not in tl:
            return t
    # 2. Fall back to compressed color image topics
    for t in topics:
        tl = t.lower()
        if "image" in tl and "depth" not in tl:
            return t
    # 3. Any image topic at all
    for t in topics:
        if "image" in t.lower():
            return t
    return None


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def _open_bag(path_str: str, label: str) -> tuple[Path, list[Path]]:
    """Resolve a user-supplied bag path; print a summary; exit on failure."""
    p = Path(path_str).resolve()
    bag_files = _resolve_bag_files(p)
    if not bag_files:
        sys.exit(
            f"[ERROR] No .db3 or .mcap files found at: {p}\n"
            f"  Make sure the path is the bag directory or the bag file itself.")
    bag_path = bag_files[0].parent
    fmt      = _bag_format(bag_files)
    has_meta = _has_metadata(bag_files)
    print(f"[{label}] found {len(bag_files)} {fmt} file(s): "
          f"{[f.name for f in bag_files]}")
    print(f"[{label}] directory  : {bag_path}")
    if fmt == "db3":
        print(f"[{label}] metadata.yaml: "
              f"{'YES' if has_meta else 'NO — using direct SQLite reader'}")
    return bag_path, bag_files


def parse_args():
    p = argparse.ArgumentParser(
        description="FAST-Calib offline calibration from a ROS2 .db3 bag.",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bag",          required=True,
                   help="LiDAR bag — directory or .db3 file\n"
                        "(also used for the camera if --camera-bag is not given)")
    p.add_argument("--camera-bag",   default=None,
                   help="Separate camera bag — directory or .db3 file\n"
                        "(use when camera and LiDAR were recorded in different bags)")
    p.add_argument("--config",       default=None,
                   help="Path to qr_params.yaml "
                        "(default: <repo>/config/qr_params.yaml)")
    p.add_argument("--image",        default=None,
                   help="Camera image (PNG/JPEG). Skips bag image extraction.")
    p.add_argument("--image-topic",  default=None,
                   help="Camera topic name (auto-detected if omitted)")
    p.add_argument("--lidar-topic",  default=None,
                   help="LiDAR PointCloud2 topic (auto-detected if omitted)")
    p.add_argument("--output-dir",   default=None,
                   help="Output directory (default: <repo>/output)")
    p.add_argument("--list-topics",  action="store_true",
                   help="Print all topics from both bags and exit")
    p.add_argument("--min-markers",  type=int, default=None,
                   help="Override min_detected_markers from config")
    p.add_argument("--save-any-frame", default=None, metavar="PATH",
                   help="Save best ArUco frame to PATH for visual inspection")
    p.add_argument("--max-cloud-frames", type=int, default=50, metavar="N",
                   help="Max LiDAR frames to accumulate (default: 50 ≈ 5 s at 10 Hz).\n"
                        "Use 0 to read all frames (may use a lot of RAM).")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root   = _HERE.parent
    config_path = args.config    or str(repo_root / "config" / "qr_params.yaml")
    output_dir  = args.output_dir or str(repo_root / "output")

    if not Path(config_path).exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")

    # ── open bag(s) ───────────────────────────────────────────────────────────
    lidar_bag_path, lidar_bag_files = _open_bag(args.bag, "LiDAR bag")

    if args.camera_bag:
        cam_bag_path, cam_bag_files = _open_bag(args.camera_bag, "Camera bag")
    else:
        cam_bag_path, cam_bag_files = lidar_bag_path, lidar_bag_files

    # ── list-topics ───────────────────────────────────────────────────────────
    if args.list_topics:
        print("\nTopics in LiDAR bag:")
        list_topics(lidar_bag_path, lidar_bag_files)
        if args.camera_bag:
            print("\nTopics in camera bag:")
            list_topics(cam_bag_path, cam_bag_files)
        return

    cfg = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── topic resolution — LiDAR bag ──────────────────────────────────────────
    with _open_reader(lidar_bag_path, lidar_bag_files) as r:
        lidar_topics = _topics_from_reader(r)
    print(f"[LiDAR bag] topics: {lidar_topics}")

    lidar_topic = (args.lidar_topic
                   or cfg.get("lidar_topic")
                   or _auto_lidar_topic(lidar_topics))
    if not lidar_topic:
        sys.exit("[ERROR] No LiDAR topic found. Use --lidar-topic.")
    print(f"[LiDAR bag] lidar_topic : {lidar_topic}")

    # ── topic resolution — camera bag ─────────────────────────────────────────
    if args.camera_bag:
        with _open_reader(cam_bag_path, cam_bag_files) as r:
            cam_topics = _topics_from_reader(r)
        print(f"[Camera bag] topics: {cam_topics}")
    else:
        cam_topics = lidar_topics

    image_topic = (args.image_topic
                   or _auto_image_topic(cam_topics)
                   or "/camera/image_raw")
    print(f"[Camera bag] image_topic : {image_topic}")

    print(f"[Config] {config_path}")
    print(f"[Output] {output_dir}\n")

    # ── image ─────────────────────────────────────────────────────────────────
    if args.image:
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            sys.exit(f"[ERROR] Cannot read image: {args.image}")
        print(f"[Image] Loaded from file: {args.image}")
    else:
        min_markers = (args.min_markers if args.min_markers is not None
                       else int(cfg["min_detected_markers"]))
        print(f"[Image] Scanning '{image_topic}' for best ArUco frame "
              f"(min_markers={min_markers}) …")
        adict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        image = _best_image_from_bag(cam_bag_path, cam_bag_files, image_topic,
                                     min_markers, adict,
                                     save_any_frame=args.save_any_frame)
        if image is None:
            sys.exit(
                f"[ERROR] No suitable image found on '{image_topic}'.\n"
                f"  Run --list-topics to see available topics, or use "
                f"--image /path/to/image.png")
        saved = os.path.join(output_dir, "extracted_image.png")
        cv2.imwrite(saved, image)
        print(f"[Image] Saved extracted image: {saved}")

    # ── point cloud ───────────────────────────────────────────────────────────
    max_frames = args.max_cloud_frames
    print(f"\n[Cloud] Reading {'all' if not max_frames else f'up to {max_frames}'} "
          f"frames on '{lidar_topic}' …")
    pts_N4 = _read_cloud(lidar_bag_path, lidar_bag_files, lidar_topic,
                         max_frames=max_frames)
    if len(pts_N4) == 0:
        sys.exit(
            f"[ERROR] No point cloud data found on '{lidar_topic}'.\n"
            f"  Run --list-topics to see available topics.")
    print(f"[Cloud] Total accumulated points: {len(pts_N4):,}")

    # ── calibration ───────────────────────────────────────────────────────────
    print("\n[Calib] Running calibration pipeline …\n")
    T, rmse = run_calibration(image, pts_N4, cfg, output_dir, tag="single")

    if T is None:
        sys.exit(
            "[ERROR] Calibration failed.\n"
            "  Common causes:\n"
            "  • Filter bounds (x_min/max etc.) don't cover the calibration board\n"
            "  • Image doesn't show all 4 ArUco markers clearly\n"
            "  • circle_radius / delta_width_circles don't match your target\n"
            "  Run with --list-topics and --image to diagnose.")

    print(f"\n[Done] RMSE = {rmse:.4f} m   Results in: {output_dir}/")


if __name__ == "__main__":
    main()
