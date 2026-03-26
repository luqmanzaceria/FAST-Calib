#!/usr/bin/env bash
# calib_from_bag.sh — Run FAST-Calib offline from a rosbag and print T_cam_lidar
#
# Usage:
#   ./calib_from_bag.sh <bag> [image_file] [OPTIONS]
#
# Arguments:
#   bag         Path to either:
#                 • ROS1  .bag file
#                 • ROS2  .db3 file  (the .db3 inside the bag directory, OR the
#                                     bag directory itself that contains metadata.yaml)
#   image_file  (optional) Path to camera image. If omitted, the best ArUco
#               frame is automatically extracted from the bag.
#
# Options:
#   --lidar-topic TOPIC   LiDAR topic in the bag (default: value in qr_params.yaml)
#   --image-topic TOPIC   Camera topic used for image extraction (default: /camera/image_raw)
#   --config YAML         Path to qr_params.yaml (default: <package>/config/qr_params.yaml)
#   --output-dir DIR      Where to write results (default: <package>/output)
#   --timeout SECS        Seconds to wait for calibration node (default: 60)
#
# Output:
#   <output_dir>/single_calib_result.txt   — T_cam_lidar in FAST-LIVO2 format
#   <output_dir>/colored_cloud.pcd         — coloured point cloud
#   <output_dir>/qr_detect.png             — annotated image
#
# Requirements:
#   ROS 1 (Noetic) environment: rospack, rosrun, roscore, rosparam — this package
#   is a catkin/ROS1 node; ROS 2 (Humble/Jazzy) alone is not enough.
#   For .db3 / ROS2 bags: pip install rosbags  (provides rosbags-convert)
#
#   macOS: ROS1 is not installed under /opt/ros by default. Typical options are
#   a Linux machine/VM, Docker (e.g. osrf/ros:noetic-desktop-full), or remote SSH.

set -euo pipefail

# ─── helpers ──────────────────────────────────────────────────────────────────

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

die()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[calib_from_bag] $*"; }

ros_missing_help() {
    echo "[ERROR] ROS 1 tools not found (need: rospack, rosrun, …)." >&2
    echo "" >&2
    echo "  FAST-Calib runs the \`fast_calib\` ROS1 node. Sourcing ROS 2 (e.g. Humble)" >&2
    echo "  does not provide \`rospack\`; you need a ROS Noetic (or older) workspace" >&2
    echo "  with this package built and sourced." >&2
    echo "" >&2
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "  On macOS, official ROS1 desktop packages are not in /opt/ros like on Ubuntu." >&2
        echo "  Use Docker with osrf/ros:noetic-desktop-full, a Linux VM, or run on a Linux robot PC." >&2
    else
        echo "  On Ubuntu: sudo apt install ros-noetic-desktop-full" >&2
        echo "  Then: source /opt/ros/noetic/setup.bash && source <ws>/devel/setup.bash" >&2
    fi
    exit 1
}

# ─── defaults ─────────────────────────────────────────────────────────────────

BAG_INPUT=""
IMAGE_FILE=""
LIDAR_TOPIC=""
IMAGE_TOPIC="/camera/image_raw"
CONFIG_YAML=""
OUTPUT_DIR=""
TIMEOUT_SECS=60

# Variables populated during execution — used in cleanup
CONVERTED_BAG=""      # temp ROS1 .bag created from a db3/ROS2 bag
TMP_IMG_DIR=""        # temp dir for extracted image
NODE_PID=""
ROSCORE_PID=""
STARTED_ROSCORE=false

# ─── single cleanup handler ───────────────────────────────────────────────────

cleanup() {
    [ -n "$NODE_PID"    ] && kill "$NODE_PID"    2>/dev/null || true
    $STARTED_ROSCORE     && kill "$ROSCORE_PID"  2>/dev/null || true
    [ -n "$CONVERTED_BAG" ] && rm -f "$CONVERTED_BAG"
    [ -n "$TMP_IMG_DIR"   ] && rm -rf "$TMP_IMG_DIR"
}
trap cleanup EXIT

# ─── argument parsing ─────────────────────────────────────────────────────────

if [ $# -lt 1 ]; then usage; fi

BAG_INPUT=$(realpath "$1"); shift

# Second positional arg is image_file if it doesn't start with --
if [ $# -gt 0 ] && [[ "$1" != --* ]] && [ -f "$1" ]; then
    IMAGE_FILE=$(realpath "$1"); shift
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --lidar-topic)  LIDAR_TOPIC="$2";            shift 2 ;;
        --image-topic)  IMAGE_TOPIC="$2";             shift 2 ;;
        --config)       CONFIG_YAML=$(realpath "$2"); shift 2 ;;
        --output-dir)   OUTPUT_DIR=$(realpath "$2");  shift 2 ;;
        --timeout)      TIMEOUT_SECS="$2";            shift 2 ;;
        -h|--help)      usage ;;
        *) die "Unknown option: $1" ;;
    esac
done

[ -e "$BAG_INPUT" ] || die "Bag not found: $BAG_INPUT"

# ─── locate package ───────────────────────────────────────────────────────────

command -v rospack >/dev/null 2>&1 || ros_missing_help
PKG_DIR=$(rospack find fast_calib 2>/dev/null) \
    || die "fast_calib package not found. Build and source your workspace."

[ -z "$CONFIG_YAML" ] && CONFIG_YAML="$PKG_DIR/config/qr_params.yaml"
[ -f "$CONFIG_YAML" ] || die "Config file not found: $CONFIG_YAML"

[ -z "$OUTPUT_DIR"  ] && OUTPUT_DIR="$PKG_DIR/output"
mkdir -p "$OUTPUT_DIR"

RESULT_FILE="$OUTPUT_DIR/single_calib_result.txt"
SCRIPT_DIR="$PKG_DIR/scripts"

# ─── step 1: convert ROS2 db3 bag → ROS1 .bag if needed ──────────────────────
#
# Accepted input forms:
#   a) /path/to/bag_dir/          — ROS2 bag directory (contains metadata.yaml)
#   b) /path/to/bag_dir/foo_0.db3 — the .db3 file inside a ROS2 bag directory
#   c) /path/to/recording.bag     — ROS1 bag, use directly

ROS2_BAG_DIR=""

if [[ "$BAG_INPUT" == *.db3 ]]; then
    # The user pointed at the .db3 file itself; the bag root is the parent dir.
    PARENT=$(dirname "$BAG_INPUT")
    if [ -f "$PARENT/metadata.yaml" ]; then
        ROS2_BAG_DIR="$PARENT"
    else
        die "Cannot find metadata.yaml next to $BAG_INPUT. " \
            "Point to the bag directory instead of the .db3 file, or ensure metadata.yaml is present."
    fi
elif [ -d "$BAG_INPUT" ] && [ -f "$BAG_INPUT/metadata.yaml" ]; then
    ROS2_BAG_DIR="$BAG_INPUT"
fi

if [ -n "$ROS2_BAG_DIR" ]; then
    info "Detected ROS2 bag: $ROS2_BAG_DIR"
    command -v rosbags-convert >/dev/null 2>&1 \
        || die "rosbags-convert not found. Install it with: pip install rosbags"

    CONVERTED_BAG=$(mktemp --suffix=".bag")
    info "Converting to ROS1 .bag (this may take a moment)…"
    rosbags-convert --src "$ROS2_BAG_DIR" --dst "$CONVERTED_BAG" \
        || die "Conversion failed. Check that the ROS2 bag is valid."
    info "Conversion complete: $CONVERTED_BAG"
    BAG_FILE="$CONVERTED_BAG"
else
    # Treat as ROS1 .bag
    [ -f "$BAG_INPUT" ] || die "Expected a .bag file but got: $BAG_INPUT"
    BAG_FILE="$BAG_INPUT"
fi

# ─── step 2: extract image if not supplied ────────────────────────────────────

if [ -z "$IMAGE_FILE" ]; then
    info "No image provided — extracting best ArUco frame from bag on topic '$IMAGE_TOPIC'…"

    TMP_IMG_DIR=$(mktemp -d)

    python3 "$SCRIPT_DIR/extract_images.py" "$BAG_FILE" "$TMP_IMG_DIR" "$IMAGE_TOPIC" \
        || die "Image extraction failed. Is '$IMAGE_TOPIC' a valid camera topic in the bag?"

    IMAGE_FILE=$(find "$TMP_IMG_DIR" -name "*.png" | head -1)
    [ -n "$IMAGE_FILE" ] \
        || die "extract_images.py produced no image. Check the topic or add --image-topic."
    info "Extracted image: $IMAGE_FILE"
fi

[ -f "$IMAGE_FILE" ] || die "Image file not found: $IMAGE_FILE"

# ─── step 3: ensure roscore is running ────────────────────────────────────────

if ! rostopic list >/dev/null 2>&1; then
    info "Starting roscore…"
    roscore &
    ROSCORE_PID=$!
    STARTED_ROSCORE=true
    for i in $(seq 1 10); do
        rostopic list >/dev/null 2>&1 && break
        sleep 1
    done
    rostopic list >/dev/null 2>&1 || die "roscore failed to start."
fi

# ─── step 4: load config params, then override bag/image paths ───────────────

info "Loading config: $CONFIG_YAML"
rosparam load "$CONFIG_YAML"

rosparam set bag_path    "$BAG_FILE"
rosparam set image_path  "$IMAGE_FILE"
rosparam set output_path "$OUTPUT_DIR"
[ -n "$LIDAR_TOPIC" ] && rosparam set lidar_topic "$LIDAR_TOPIC"

info "bag_path  : $BAG_FILE"
info "image_path: $IMAGE_FILE"
info "output_dir: $OUTPUT_DIR"

# ─── step 5: remove stale result so we can detect when it's written ──────────

rm -f "$RESULT_FILE"

# ─── step 6: launch calibration node ─────────────────────────────────────────

info "Launching fast_calib node…"
rosrun fast_calib fast_calib &
NODE_PID=$!

# ─── step 7: wait for result file ────────────────────────────────────────────

info "Waiting up to ${TIMEOUT_SECS}s for calibration result…"
ELAPSED=0
while [ ! -f "$RESULT_FILE" ] && [ "$ELAPSED" -lt "$TIMEOUT_SECS" ]; do
    if ! kill -0 "$NODE_PID" 2>/dev/null; then
        die "Calibration node exited before writing results. Check ROS logs."
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ ! -f "$RESULT_FILE" ]; then
    die "Timed out after ${TIMEOUT_SECS}s without a result. Try --timeout or check bag/image/params."
fi

# ─── step 8: display result ───────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         FAST-Calib: T_cam_lidar result        ║"
echo "╚══════════════════════════════════════════════╝"
cat "$RESULT_FILE"
echo "──────────────────────────────────────────────"
info "Results saved to: $OUTPUT_DIR"
