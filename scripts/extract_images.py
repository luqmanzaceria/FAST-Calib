import cv2
import cv2.aruco as aruco
import rosbag
from cv_bridge import CvBridge
import sys 
import os

bridge = CvBridge()

if len(sys.argv) < 3:
    print("Usage: python extract_images.py <bag_name.bag> <output_dir> [image_topic]")
    sys.exit(1)

bag_name = sys.argv[1]
output_dir = sys.argv[2]
image_topic = sys.argv[3] if len(sys.argv) >= 4 else '/camera/image_raw'

best_image = None
best_score = -1

with rosbag.Bag(bag_name, 'r') as bag:
    count = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        detector_params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        num_markers = len(corners) if ids is not None else 0
        if num_markers == 4:
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score>best_score:
                best_score = score
                best_image = cv_img.copy()
if best_image is not None:
    bag_basename = os.path.splitext(os.path.basename(bag_name))[0]
    out_path = os.path.join(output_dir, f"{bag_basename}_best.png")
    success = cv2.imwrite(out_path, best_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if success:
        print(f"Saved best image (sharpness={best_score:.1f}): {out_path}")
    else:
        print(f"[ERROR] cv2.imwrite failed for path: {out_path}")
else:
    print("No frames found with all 4 markers visible.")
    sys.exit(1)
                
           