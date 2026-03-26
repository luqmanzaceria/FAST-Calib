/*
Developer: FAST-Calib live calibration node

Subscribes to live camera and LiDAR topics, accumulates data over a
configurable window, and computes T_cam_lidar using the same ArUco +
LiDAR circle detection pipeline as the offline fast_calib node.

Launch with:
  roslaunch fast_calib live_calib.launch

Key ROS params (set in qr_params.yaml or via launch args):
  image_topic   (string)  camera image topic         [default: /camera/image_raw]
  lidar_topic   (string)  LiDAR point-cloud topic    [default: /livox/lidar]
  calib_interval (double) seconds between attempts   [default: 5.0]
  (all other params from qr_params.yaml are also used)

On success:
  • Prints T_cam_lidar to stdout
  • Saves output/single_calib_result.txt  (FAST-LIVO2 format)
  • Saves output/colored_cloud.pcd
  • Saves output/qr_detect.png
  • Shuts down
*/

#include "qr_detect.hpp"
#include "lidar_detect.hpp"
#include "data_preprocess.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <mutex>

class LiveCalib
{
public:
    LiveCalib(ros::NodeHandle &nh, Params &params)
        : nh_(nh),
          params_(params),
          accumulated_cloud_(new pcl::PointCloud<Common::Point>),
          qrDetect_(new QRDetect(nh, params)),
          lidarDetect_(new LidarDetect(nh, params)),
          lidar_type_(LiDARType::Unknown),
          calibration_done_(false),
          best_image_score_(-1.0),
          best_marker_count_(0)
    {
        nh_.param<std::string>("image_topic", image_topic_, "/camera/image_raw");
        nh_.param("calib_interval", calib_interval_, 5.0);

        image_sub_ = nh_.subscribe(image_topic_, 5,
                                   &LiveCalib::imageCallback, this);

        // Subscribe to both PointCloud2 and Livox CustomMsg on the same topic.
        // ROS only delivers messages whose type matches the callback signature,
        // so exactly one of these will receive data depending on what the sensor
        // publishes.
        lidar_sub_pc2_   = nh_.subscribe(params_.lidar_topic, 50,
                                         &LiveCalib::lidarCallbackPC2, this);
        lidar_sub_livox_ = nh_.subscribe(params_.lidar_topic, 50,
                                         &LiveCalib::lidarCallbackLivox, this);

        calib_timer_ = nh_.createTimer(ros::Duration(calib_interval_),
                                       &LiveCalib::attemptCalibration, this);

        ROS_INFO("[LiveCalib] image_topic   : %s", image_topic_.c_str());
        ROS_INFO("[LiveCalib] lidar_topic   : %s", params_.lidar_topic.c_str());
        ROS_INFO("[LiveCalib] calib_interval: %.1f s", calib_interval_);
        ROS_INFO("[LiveCalib] Waiting for sensor data…");
    }

private:
    ros::NodeHandle  &nh_;
    Params            params_;
    std::string       image_topic_;
    double            calib_interval_;

    std::mutex cloud_mutex_, image_mutex_;

    pcl::PointCloud<Common::Point>::Ptr accumulated_cloud_;
    cv::Mat    best_image_;
    double     best_image_score_;
    int        best_marker_count_;

    std::shared_ptr<QRDetect>    qrDetect_;
    std::shared_ptr<LidarDetect> lidarDetect_;
    LiDARType lidar_type_;
    bool      calibration_done_;

    ros::Subscriber image_sub_;
    ros::Subscriber lidar_sub_pc2_;
    ros::Subscriber lidar_sub_livox_;
    ros::Timer      calib_timer_;

    // ── image callback ──────────────────────────────────────────────────────
    void imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        if (calibration_done_) return;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR_THROTTLE(5.0, "[LiveCalib] cv_bridge: %s", e.what());
            return;
        }

        cv::Mat gray;
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::aruco::Dictionary> dict =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        std::vector<int>                          ids;
        std::vector<std::vector<cv::Point2f>>     corners;
        cv::aruco::detectMarkers(gray, dict, corners, ids);

        int n = static_cast<int>(ids.size());
        if (n >= params_.min_detected_markers) {
            double sharpness = cv::Laplacian(gray, cv::CV_64F).var();
            std::lock_guard<std::mutex> lk(image_mutex_);
            if (n > best_marker_count_ ||
                (n == best_marker_count_ && sharpness > best_image_score_))
            {
                best_image_        = cv_ptr->image.clone();
                best_image_score_  = sharpness;
                best_marker_count_ = n;
                ROS_INFO_THROTTLE(2.0,
                    "[LiveCalib] Best image updated: %d markers, sharpness=%.1f",
                    n, sharpness);
            }
        }
    }

    // ── LiDAR callback: standard PointCloud2 (mechanical or generic solid) ──
    void lidarCallbackPC2(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        if (calibration_done_) return;

        bool has_ring = false;
        for (const auto &f : msg->fields)
            if (f.name == "ring") { has_ring = true; break; }

        sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");

        std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<std::uint16_t>> it_ring;
        if (has_ring) {
            it_ring.reset(
                new sensor_msgs::PointCloud2ConstIterator<std::uint16_t>(*msg, "ring"));
            lidar_type_ = LiDARType::Mech;
        } else {
            lidar_type_ = LiDARType::Solid;
        }

        const size_t n = static_cast<size_t>(msg->width) * msg->height;
        std::lock_guard<std::mutex> lk(cloud_mutex_);
        for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z) {
            Common::Point p;
            p.x = *it_x;  p.y = *it_y;  p.z = *it_z;
            if (has_ring) { p.ring = **it_ring;  ++(*it_ring); }
            else            p.ring = 0xFFFF;
            accumulated_cloud_->push_back(p);
        }
    }

    // ── LiDAR callback: Livox CustomMsg (solid-state) ───────────────────────
    void lidarCallbackLivox(const livox_ros_driver::CustomMsg::ConstPtr &msg)
    {
        if (calibration_done_) return;
        lidar_type_ = LiDARType::Solid;

        std::lock_guard<std::mutex> lk(cloud_mutex_);
        accumulated_cloud_->reserve(accumulated_cloud_->size() + msg->point_num);
        for (uint32_t i = 0; i < msg->point_num; ++i) {
            Common::Point p;
            p.x    = msg->points[i].x;
            p.y    = msg->points[i].y;
            p.z    = msg->points[i].z;
            p.ring = static_cast<std::uint16_t>(msg->points[i].line);
            accumulated_cloud_->push_back(p);
        }
    }

    // ── periodic calibration attempt ────────────────────────────────────────
    void attemptCalibration(const ros::TimerEvent &)
    {
        if (calibration_done_) return;

        // Snapshot the best image
        cv::Mat image;
        {
            std::lock_guard<std::mutex> lk(image_mutex_);
            if (best_image_.empty()) {
                ROS_WARN_THROTTLE(5.0,
                    "[LiveCalib] Waiting for image with >= %d ArUco markers on '%s'…",
                    params_.min_detected_markers, image_topic_.c_str());
                return;
            }
            image = best_image_.clone();
        }

        // Snapshot the accumulated cloud
        pcl::PointCloud<Common::Point>::Ptr cloud(new pcl::PointCloud<Common::Point>);
        {
            std::lock_guard<std::mutex> lk(cloud_mutex_);
            if (accumulated_cloud_->empty()) {
                ROS_WARN_THROTTLE(5.0,
                    "[LiveCalib] Waiting for LiDAR data on '%s'…",
                    params_.lidar_topic.c_str());
                return;
            }
            *cloud = *accumulated_cloud_;
        }

        ROS_INFO("[LiveCalib] Attempting calibration: %zu LiDAR pts (type=%s)…",
                 cloud->size(),
                 lidar_type_ == LiDARType::Mech ? "Mech" : "Solid");

        // ── QR / camera detection ──
        pcl::PointCloud<pcl::PointXYZ>::Ptr qr_center_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
        qrDetect_->detect_qr(image, qr_center_cloud);

        if (qr_center_cloud->size() != 4) {
            ROS_WARN("[LiveCalib] QR: found %zu/4 centers — retrying next cycle.",
                     qr_center_cloud->size());
            return;
        }

        // ── LiDAR detection ──
        // Re-create LidarDetect to reset its internal state between attempts.
        lidarDetect_.reset(new LidarDetect(nh_, params_));
        pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_center_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);

        if (lidar_type_ == LiDARType::Mech)
            lidarDetect_->detect_mech_lidar(cloud, lidar_center_cloud);
        else
            lidarDetect_->detect_solid_lidar(cloud, lidar_center_cloud);

        if (lidar_center_cloud->size() != 4) {
            ROS_WARN("[LiveCalib] LiDAR: found %zu/4 centers — clearing cloud and accumulating more.",
                     lidar_center_cloud->size());
            std::lock_guard<std::mutex> lk(cloud_mutex_);
            accumulated_cloud_->clear();
            return;
        }

        // ── sort and align ──
        pcl::PointCloud<pcl::PointXYZ>::Ptr qr_centers(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_centers(
            new pcl::PointCloud<pcl::PointXYZ>);
        sortPatternCenters(qr_center_cloud,    qr_centers,    "camera");
        sortPatternCenters(lidar_center_cloud, lidar_centers, "lidar");

        saveTargetHoleCenters(lidar_centers, qr_centers, params_);

        // ── SVD extrinsic estimation ──
        Eigen::Matrix4f transformation;
        pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
        svd.estimateRigidTransformation(*lidar_centers, *qr_centers, transformation);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
        aligned->reserve(lidar_centers->size());
        alignPointCloud(lidar_centers, aligned, transformation);
        double rmse = computeRMSE(qr_centers, aligned);

        // ── print result ──
        std::cout << BOLDYELLOW << "[Result] RMSE: " << BOLDRED
                  << std::fixed << std::setprecision(4) << rmse << " m"
                  << RESET << std::endl;
        std::cout << BOLDYELLOW
                  << "[Result] Live calibration: extrinsic parameters T_cam_lidar = "
                  << RESET << std::endl;
        std::cout << BOLDCYAN << std::fixed << std::setprecision(6)
                  << transformation << RESET << std::endl;

        // ── save outputs ──
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        projectPointCloudToImage(cloud, transformation,
                                 qrDetect_->cameraMatrix_, qrDetect_->distCoeffs_,
                                 image, colored_cloud);
        saveCalibrationResults(params_, transformation, colored_cloud,
                               qrDetect_->imageCopy_);

        calibration_done_ = true;
        ROS_INFO("[LiveCalib] Calibration complete. Shutting down.");
        ros::shutdown();
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "live_calib");
    ros::NodeHandle nh;

    Params params = loadParameters(nh);

    LiveCalib calib(nh, params);
    ros::spin();

    return 0;
}
