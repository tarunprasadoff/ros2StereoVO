#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

class VisualOdometryNode : public rclcpp::Node
{
public:
    VisualOdometryNode() : Node("visual_odometry_node")
    {
        // Set camera intrinsics (fx, fy, cx, cy, and baseline)
        fx_ = 348.925;
        fy_ = 351.135;
        cx_ = 339.075;
        cy_ = 177.45;
        baseline_ = 0.120;  // 12 cm baseline

        // Camera matrix using intrinsics
        camera_matrix_ = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);

        // Message filters subscribers for synchronization
        left_image_sub_.subscribe(this, "/camera/left/image_raw");
        depth_image_sub_.subscribe(this, "/camera/depth/image_raw");

        // Synchronizer with approximate time policy
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), left_image_sub_, depth_image_sub_);
        sync_->registerCallback(std::bind(&VisualOdometryNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2));

        // Publisher for robot's pose
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/robot/pose", 10);

        // Open the CSV file for writing
        pose_file_.open("/tmp/pose_data.csv", std::ios::app);
        if (!pose_file_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file for writing.");
        } else {
            // Write the header if the file is empty
            pose_file_ << "timestamp_sec,timestamp_nsec,position_x,position_y,position_z,orientation_x,orientation_y,orientation_z,orientation_w\n";
        }
    }

    ~VisualOdometryNode() {
        // Close the CSV file when the node is destroyed
        if (pose_file_.is_open()) {
            pose_file_.close();
        }
    }

private:
    // Synchronizer policy type
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;

    // Callback for synchronized image data
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
    {
        // Convert left image from ROS to OpenCV format
        cv::Mat left_image = cv_bridge::toCvCopy(left_msg, "bgr8")->image;

        // Convert depth image from ROS to OpenCV format
        depth_image_ = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;

        // Check if the images are empty
        if (left_image.empty() || depth_image_.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received an empty image.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Received synchronized images of size: %dx%d", left_image.cols, left_image.rows);

        // Process only if we have a previous image to compare to
        if (!prev_left_image_.empty())
        {
            RCLCPP_INFO(this->get_logger(), "Processing current and previous images.");
            RCLCPP_INFO(this->get_logger(), "Depth image size: %dx%d", depth_image_.cols, depth_image_.rows);

            // Convert the previous image to grayscale
            cv::Mat gray_prev_image;
            cv::cvtColor(prev_left_image_, gray_prev_image, cv::COLOR_BGR2GRAY);
            RCLCPP_INFO(this->get_logger(), "Converted previous image to grayscale.");

            // Create ORB detector with adjusted parameters
            cv::Ptr<cv::ORB> orb = cv::ORB::create(3000, 1.1f, 8);

            // Perform feature detection on the grayscale previous image
            std::vector<cv::KeyPoint> keypoints_prev;
            cv::Mat descriptors_prev;
            orb->detectAndCompute(gray_prev_image, cv::noArray(), keypoints_prev, descriptors_prev);
            if (keypoints_prev.empty()) {
                RCLCPP_ERROR(this->get_logger(), "No features detected in the previous grayscale image.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Detected %zu keypoints in the previous grayscale image.", keypoints_prev.size());

            // Convert the current image to grayscale
            cv::Mat gray_current_image;
            cv::cvtColor(left_image, gray_current_image, cv::COLOR_BGR2GRAY);

            // Detect features in the current image
            std::vector<cv::KeyPoint> keypoints_curr;
            cv::Mat descriptors_curr;
            orb->detectAndCompute(gray_current_image, cv::noArray(), keypoints_curr, descriptors_curr);
            if (keypoints_curr.empty()) {
                RCLCPP_ERROR(this->get_logger(), "No features detected in the current grayscale image.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Detected %zu keypoints in the current grayscale image.", keypoints_curr.size());

            // Set the number of nearest neighbors for knnMatch
            int k = 3;

            // Perform feature matching using knnMatch with variable k
            cv::BFMatcher matcher(cv::NORM_HAMMING);
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(descriptors_prev, descriptors_curr, knn_matches, k);

            // Filter matches based on a generalized ratio test
            std::vector<cv::DMatch> good_matches;
            const float ratio_thresh = 0.75f;  // Ratio threshold for filtering

            for (size_t i = 0; i < knn_matches.size(); ++i) {
                if (knn_matches[i].size() >= k) {
                    bool is_good_match = true;
                    float best_distance = knn_matches[i][0].distance;

                    // Check the distance ratio for k-1 neighbors
                    for (int j = 1; j < k; ++j) {
                        if (best_distance >= ratio_thresh * knn_matches[i][j].distance) {
                            is_good_match = false;
                            break;
                        }
                    }

                    if (is_good_match) {
                        good_matches.push_back(knn_matches[i][0]);
                    }
                }
            }

            RCLCPP_INFO(this->get_logger(), "Filtered down to %zu good matches after ratio test.", good_matches.size());

            if (good_matches.empty()) {
                RCLCPP_ERROR(this->get_logger(), "No good matches found between the previous and current images.");
                return;
            }

            // Recover 3D points from matches using the depth image
            std::vector<cv::Point3f> prev_points_3d;
            std::vector<cv::Point2f> curr_points_2d;
            for (const auto& match : good_matches) {
                cv::Point2f prev_pt = keypoints_prev[match.queryIdx].pt;
                cv::Point2f curr_pt = keypoints_curr[match.trainIdx].pt;

                // Ensure the previous point is within the bounds of the depth image
                if (prev_pt.x < 0 || prev_pt.x >= depth_image_.cols || prev_pt.y < 0 || prev_pt.y >= depth_image_.rows) {
                    RCLCPP_WARN(this->get_logger(), "Previous keypoint (%f, %f) is out of depth image bounds.", prev_pt.x, prev_pt.y);
                    continue;
                }

                // Get the depth value from the depth image
                float depth = depth_image_.at<uint16_t>(static_cast<int>(prev_pt.y), static_cast<int>(prev_pt.x));
                if (depth == 0) {
                    RCLCPP_WARN(this->get_logger(), "Depth value is zero at point (%f, %f).", prev_pt.x, prev_pt.y);
                    continue;
                }

                // Convert depth to meters and compute 3D coordinates
                float z = depth * 0.3125/1000;
                float x = (prev_pt.x - cx_) * z / fx_;
                float y = (prev_pt.y - cy_) * z / fy_;
                prev_points_3d.emplace_back(x, y, z);
                curr_points_2d.push_back(curr_pt);
            }

            // Check if we have enough 3D points for pose estimation
            if (prev_points_3d.size() < 4 || curr_points_2d.size() < 4) {
                RCLCPP_ERROR(this->get_logger(), "Not enough valid 3D-2D point correspondences for pose estimation.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Recovered %zu 3D-2D point correspondences for pose estimation.", prev_points_3d.size());

            // Estimate the pose using PnPRANSAC
            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            bool success = cv::solvePnPRansac(
                prev_points_3d,        // 3D points from previous frame
                curr_points_2d,        // 2D points from current frame
                camera_matrix_,        // Camera intrinsic matrix
                cv::noArray(),         // No distortion coefficients
                rvec,                  // Output rotation vector
                tvec,                  // Output translation vector
                false,                 // Use initial guess (false)
                100,                   // Number of iterations
                8.0,                   // Reprojection error threshold
                0.99,                  // Confidence level
                inliers                // Output inlier indices
            );

            // Check if pose estimation was successful
            if (!success || inliers.size() < 4) {
                RCLCPP_ERROR(this->get_logger(), "PnPRANSAC failed or insufficient inliers.");
                return;
            }

            RCLCPP_INFO(this->get_logger(), "PnPRANSAC successful with %zu inliers.", inliers.size());

            // Convert rotation vector to a rotation matrix
            cv::Mat R;
            cv::Rodrigues(rvec, R);

            // Publish the pose
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp = left_msg->header.stamp;
            pose_msg.pose.position.x = tvec.at<double>(0);
            pose_msg.pose.position.y = tvec.at<double>(1);
            pose_msg.pose.position.z = tvec.at<double>(2);
            pose_msg.pose.orientation.x = R.at<double>(0, 0);
            pose_msg.pose.orientation.y = R.at<double>(1, 0);
            pose_msg.pose.orientation.z = R.at<double>(2, 0);
            pose_msg.pose.orientation.w = 1.0;

            pose_pub_->publish(pose_msg);
            RCLCPP_INFO(this->get_logger(), "Pose published.");

            // Save pose to CSV file
            if (pose_file_.is_open()) {
                pose_file_ << left_msg->header.stamp.sec << "," << left_msg->header.stamp.nanosec << ","
                           << tvec.at<double>(0) << "," << tvec.at<double>(1) << "," << tvec.at<double>(2) << ","
                           << R.at<double>(0, 0) << "," << R.at<double>(1, 0) << "," << R.at<double>(2, 0) << ",1.0\n";
            }
        }

        // Update the previous image
        prev_left_image_ = left_image.clone();
        RCLCPP_INFO(this->get_logger(), "Updated the previous image for the next callback.");
    }

    // Message filter subscribers
    message_filters::Subscriber<sensor_msgs::msg::Image> left_image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_image_sub_;

    // Synchronizer
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

    // Image storage
    cv::Mat prev_left_image_;
    cv::Mat depth_image_;

    // Camera intrinsics
    double fx_, fy_, cx_, cy_, baseline_;

    // Camera matrix
    cv::Mat camera_matrix_;

    // File for saving pose data
    std::ofstream pose_file_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisualOdometryNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
