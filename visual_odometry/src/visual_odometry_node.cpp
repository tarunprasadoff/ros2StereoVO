#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include <vector>
#include <Eigen/Geometry>

using namespace std::placeholders;

class VisualOdometryNode : public rclcpp::Node {

public:

    VisualOdometryNode() : Node("visual_odometry_node"), first_frame(true) {
        
        depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/camera/depth/image_raw", 10, std::bind(&VisualOdometryNode::depth_image_callback, this, _1));
        left_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/camera/left/image_raw", 10, std::bind(&VisualOdometryNode::left_image_callback, this, _1));
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/robot/pose", 10);
    
    }

private:

    void left_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {

        if (depth_image_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Depth image is not available yet.");
            return;
        }

        try {

            cv::Mat left_image = cv_bridge::toCvShare(msg, "bgr8")->image;
            cv::Mat left_image_gray;
            cv::cvtColor(left_image, left_image_gray, cv::COLOR_BGR2GRAY);

            // RCLCPP_INFO(this->get_logger(), "Left Image Callback");
            
            if (first_frame) {
                initialize_tracking(left_image_gray);
                first_frame = false;
            } else {
                process_image(left_image_gray);
            }

            image_pts_prev_ = image_pts_curr_;
            left_image_prev_ = left_image_gray.clone();

        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }

    }


    void depth_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        
        try {
            
            depth_image_ = cv_bridge::toCvShare(msg, "mono16")->image;
            depth_image_.convertTo(depth_image_, CV_32FC1, 0.3125 / 1000.0); // Apply scale factor

            // RCLCPP_INFO(this->get_logger(), "Depth Image Callback");
        
        } catch (cv_bridge::Exception &e) {
            
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        
        }

    }
    
    void initialize_tracking(cv::Mat &gray_image) {

        RCLCPP_INFO(this->get_logger(), "Tracking Initializing");
        
        cv::Mat mask = cv::Mat::ones(gray_image.size(), CV_8U);
        int ignore_height = gray_image.rows * 0.2;
        cv::Rect roi(0, 0, gray_image.cols, gray_image.rows - ignore_height);
        mask(cv::Rect(0, gray_image.rows - ignore_height, gray_image.cols, ignore_height)) = 0;
        
        // RCLCPP_INFO(this->get_logger(), "Tracking Masked");
        
        cv::goodFeaturesToTrack(gray_image, image_pts_curr_, max_corners, 0.01, 10, mask, 3, false, 0.04);
        cv::cornerSubPix(gray_image, image_pts_curr_, subPixWinSize, cv::Size(-1, -1), termcrit);
        get_3d_pts(depth_image_, image_pts_curr_, image_pts_trunc_, image_pts_3d_);
        
        image_pts_curr_.clear();

        for (unsigned int i = 0; i < image_pts_trunc_.size(); i++) {
            
            cv::Point2f valid_pt = image_pts_trunc_[i];
            image_pts_curr_.push_back(valid_pt);

        }

        RCLCPP_INFO(this->get_logger(), "Tracking Initialized");

    }

    void process_image(cv::Mat &left_image_gray) {

        // RCLCPP_INFO(this->get_logger(), "Process Initializing");

        if (image_pts_prev_.empty()) {
            RCLCPP_WARN(this->get_logger(), "Previous points are empty, skipping frame.");
            return;
        }

        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(left_image_prev_, left_image_gray, image_pts_prev_, image_pts_curr_, status, err, winSize, 3, termcrit, 0, 0.001);
        filter_good_points(status);

        if (!image_pts_prev_.empty() && !image_pts_curr_.empty() && !image_pts_3d_.empty()) {

            cv::Mat rvec, tvec;
            cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);
            std::vector<cv::Point2f> image_pts_curr_trunc = image_pts_curr_;
            std::vector<cv::Point3f> image_pts_3d_trunc = image_pts_3d_;

            remove_bad_pts1(left_image_gray, image_pts_curr_trunc, image_pts_3d_trunc, left_image_gray.cols, left_image_gray.rows);

            if (image_pts_curr_trunc.size() >= min_pts_pnp_ && image_pts_3d_trunc.size() >= min_pts_pnp_) {

                try {

                    cv::solvePnPRansac(image_pts_3d_trunc, image_pts_curr_trunc, cam_matrix_, dist_coeffs, rvec, tvec);
                    publish_pose(rvec, tvec);

                } catch (const std::exception &e) {

                    RCLCPP_ERROR(this->get_logger(), "solvePnPRansac failed: %s", e.what());

                }

            } else {

                RCLCPP_WARN(this->get_logger(), "Not enough points for pose estimation.");

            }

        }

        // RCLCPP_INFO(this->get_logger(), "Process Initialized");

    }

    void filter_good_points(const std::vector<uchar> &status) {

        std::vector<cv::Point2f> image_pts_0_good;
        std::vector<cv::Point2f> image_pts_1_good;
        std::vector<cv::Point3f> image_pts_3d_good;

        for (size_t i = 0; i < status.size(); i++) {

            if (status[i]) {

                image_pts_0_good.push_back(image_pts_prev_[i]);
                image_pts_1_good.push_back(image_pts_curr_[i]);
                image_pts_3d_good.push_back(image_pts_3d_[i]);

            }

        }

        image_pts_prev_ = image_pts_0_good;
        image_pts_curr_ = image_pts_1_good;
        image_pts_3d_ = image_pts_3d_good;

    }
    
    void publish_pose(const cv::Mat &rvec, const cv::Mat &tvec) {
    
        geometry_msgs::msg::PoseStamped pose_msg;

        pose_msg.header.stamp = this->now();
        pose_msg.header.frame_id = "map";

        // Translation
        pose_msg.pose.position.x = tvec.at<double>(0);
        pose_msg.pose.position.y = tvec.at<double>(1);
        pose_msg.pose.position.z = tvec.at<double>(2);

        // Convert rotation vector to rotation matrix
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);

        // Manually convert OpenCV rotation matrix to Eigen matrix
        Eigen::Matrix3d rotation_eigen;
        rotation_eigen(0, 0) = rotation_matrix.at<double>(0, 0);
        rotation_eigen(0, 1) = rotation_matrix.at<double>(0, 1);
        rotation_eigen(0, 2) = rotation_matrix.at<double>(0, 2);
        rotation_eigen(1, 0) = rotation_matrix.at<double>(1, 0);
        rotation_eigen(1, 1) = rotation_matrix.at<double>(1, 1);
        rotation_eigen(1, 2) = rotation_matrix.at<double>(1, 2);
        rotation_eigen(2, 0) = rotation_matrix.at<double>(2, 0);
        rotation_eigen(2, 1) = rotation_matrix.at<double>(2, 1);
        rotation_eigen(2, 2) = rotation_matrix.at<double>(2, 2);

        // Convert to quaternion
        Eigen::Quaterniond quaternion(rotation_eigen);

        // Set orientation
        pose_msg.pose.orientation.x = quaternion.x();
        pose_msg.pose.orientation.y = quaternion.y();
        pose_msg.pose.orientation.z = quaternion.z();
        pose_msg.pose.orientation.w = quaternion.w();

        // Publish the pose message
        pose_pub_->publish(pose_msg);

    }
    
    void get_3d_pts(cv::Mat image_depth_float, std::vector<cv::Point2f> image_pts, std::vector<cv::Point2f> &image_pts_trunc, std::vector<cv::Point3f> &image_pts_3d) {
        
        // Camera intrinsic parameters (focal lengths and principal point)
        float fx = 348.925, fy = 351.135, cx = 339.075, cy = 177.45;

        // Loop through all the input 2D image points
        for (auto &kp : image_pts) {

            // Compute normalized image coordinates from pixel coordinates
            float x = (kp.x - cx) / fx;
            float y = (kp.y - cy) / fy;

            // Create a point in normalized 3D space
            cv::Point3f unprojected_pt(x, y, 1.0f);

            // RCLCPP_INFO(this->get_logger(), "No Seg Fault yet 1");

            // Multiply by the corresponding depth value to get the actual 3D point
            // unprojected_pt *= image_depth_float.at<float>(round(kp.y), round(kp.x));

            // RCLCPP_INFO(this->get_logger(), "No Seg Fault yet 2");

            // // Check if the depth (z-coordinate) is valid (non-zero)
            // if (unprojected_pt.z != 0) {

            //     // If valid, add this 3D point to the output list
            //     image_pts_3d.push_back(unprojected_pt);

            //     RCLCPP_INFO(this->get_logger(), "No Seg Fault yet 3");

            //     // Also store the corresponding 2D point in the truncated list
            //     image_pts_trunc.push_back(cv::Point2f(kp.x, kp.y));

            //     RCLCPP_INFO(this->get_logger(), "No Seg Fault yet 4");

            // }

            if (!image_depth_float.empty() && 
                round(kp.x) >= 0 && round(kp.x) < image_depth_float.cols && 
                round(kp.y) >= 0 && round(kp.y) < image_depth_float.rows) {
                
                unprojected_pt *= image_depth_float.at<float>(round(kp.y), round(kp.x));
                
                if (unprojected_pt.z != 0) {
                    image_pts_3d.push_back(unprojected_pt);
                    image_pts_trunc.push_back(cv::Point2f(kp.x, kp.y));
                }
            } else {
                RCLCPP_INFO(this->get_logger(), "Depth image size: %d x %d", image_depth_float.cols, image_depth_float.rows);
                float depth_value = image_depth_float.at<float>(round(kp.y), round(kp.x));
                RCLCPP_INFO(this->get_logger(), "Depth value at keypoint (%f, %f): %f", kp.x, kp.y, depth_value);
                RCLCPP_WARN(this->get_logger(), "Invalid depth image or keypoint out of bounds.");
            }

        }

    }
    
    void remove_bad_pts1(cv::Mat image_left, std::vector<cv::Point2f> &image_pts_left,
                                         std::vector<cv::Point3f> &image_pts_3d, int ncols, int nrows) {

        int margin = 10;  // Margin around edges of the image
        int image_width = ncols;
        int image_height = nrows;

        // Create temporary storage for valid points
        std::vector<cv::Point2f> image_pts_left_valid;
        std::vector<cv::Point3f> image_pts_3d_valid;

        int num_pts_removed = 0;

        // Iterate over all points
        for (unsigned int i = 0; i < image_pts_left.size(); i++) {

            cv::Point2f &image_pt_left = image_pts_left[i];
            cv::Point3f &image_pt_3d = image_pts_3d[i];

            // Check if the point lies within the valid region (excluding margin)
            if (image_pt_left.x > margin && image_pt_left.x < image_width - margin &&
                image_pt_left.y > margin && image_pt_left.y < image_height - margin) {
                
                // Check for the validity of the corresponding 3D point
                if (std::isfinite(image_pt_3d.x) && std::isfinite(image_pt_3d.y) && std::isfinite(image_pt_3d.z) &&
                    cv::norm(image_pt_3d) > 1e-6) {

                    // Add valid points to the new list
                    image_pts_left_valid.push_back(image_pt_left);
                    image_pts_3d_valid.push_back(image_pt_3d);

                } else {

                    // If 3D point is invalid, increase the count of removed points
                    num_pts_removed++;

                }

            } else {

                // If the 2D point is outside the margins, increase the count of removed points
                num_pts_removed++;

            }

        }

        // Replace the original point vectors with the valid points
        image_pts_left = image_pts_left_valid;
        image_pts_3d = image_pts_3d_valid;

        // Log the number of points removed
        RCLCPP_INFO(this->get_logger(), "Removed %d points out of bounds or with invalid 3D data.", num_pts_removed);

    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub_, depth_image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

    cv::Mat depth_image_, left_image_prev_;
    std::vector<cv::Point2f> image_pts_prev_, image_pts_curr_, image_pts_trunc_;
    std::vector<cv::Point3f> image_pts_3d_;

    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::Size subPixWinSize = cv::Size(10, 10), winSize = cv::Size(31, 31);

    cv::Mat cam_matrix_ = (cv::Mat_<double>(3, 3) << 348.925, 0, 339.075, 0, 351.135, 177.45, 0, 0, 1);
    int max_corners = 500, min_pts_pnp_ = 10;
    bool first_frame;

};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisualOdometryNode>());
    rclcpp::shutdown();
    return 0;
}
