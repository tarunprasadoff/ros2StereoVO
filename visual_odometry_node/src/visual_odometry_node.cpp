#include <eigen3/Eigen/Dense>   // Make sure Eigen is included first
#include <opencv2/core/eigen.hpp>  // Then include OpenCV Eigen utilities
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class VisualOdometryNode : public rclcpp::Node
{
public:
    VisualOdometryNode() : Node("visual_odometry_node")
    {
        // Subscriptions
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/left/image_raw", 10, std::bind(&VisualOdometryNode::imageCallback, this, std::placeholders::_1));
        depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", 10, std::bind(&VisualOdometryNode::depthCallback, this, std::placeholders::_1));
        
        // Publisher for pose
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/robot/pose", 10);
    }

private:
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub_;

    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

    // Variables to store image and depth information
    cv::Mat curr_image_, curr_depth_;
    std::vector<cv::Point2f> curr_image_pts_;
    std::vector<cv::Point3f> curr_image_pts_3d_;

    // Minimum points for PnP
    const int MIN_PTS_PNP = 4;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        // Convert to OpenCV image
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        curr_image_ = cv_ptr->image;

        // Convert the image to grayscale before passing to goodFeaturesToTrack
        cv::Mat gray_image;
        cv::cvtColor(curr_image_, gray_image, cv::COLOR_BGR2GRAY);

        // Use the grayscale image in the visual odometry process
        curr_image_ = gray_image;

        // Process visual odometry
        processVisualOdometry();
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
}


    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            curr_depth_ = cv_ptr->image;
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void processVisualOdometry() {
        
        // Check if we have valid image and depth map
        if (curr_image_.empty() || curr_depth_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Image or depth data is missing.");
            return;
        }

        // Detect feature points in the current image
        cv::goodFeaturesToTrack(curr_image_, curr_image_pts_, 100, 0.01, 10);

        // Convert depth map to 3D points
        convertDepthTo3DPoints();

        // Calculate how many points to drop
        int num_2d_pts = static_cast<int>(curr_image_pts_.size());
        int num_3d_pts = static_cast<int>(curr_image_pts_3d_.size());
        int num_points_to_drop = num_2d_pts - num_3d_pts;

        // Drop every Nth point while preserving the order
        if (num_points_to_drop > 0)
        {
            int drop_interval = num_2d_pts / num_points_to_drop;
            std::vector<cv::Point2f> filtered_image_pts;

            for (int i = 0; i < num_2d_pts; ++i)
            {
                if (i % drop_interval != 0 || num_points_to_drop <= 0)
                {
                    filtered_image_pts.push_back(curr_image_pts_[i]);
                }
                else
                {
                    num_points_to_drop--;
                }
            }

            // Replace original points with filtered set
            curr_image_pts_ = filtered_image_pts;
        }

        // Check if we have enough points for PnP after filtering
        if (static_cast<int>(curr_image_pts_.size()) < MIN_PTS_PNP || static_cast<int>(curr_image_pts_3d_.size()) < MIN_PTS_PNP)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough points for solvePnPRansac. Skipping frame.");
            return;
        }

        // Log the number of valid points
        RCLCPP_INFO(this->get_logger(), "Number of 2D points: %ld", curr_image_pts_.size());
        RCLCPP_INFO(this->get_logger(), "Number of 3D points: %ld", curr_image_pts_3d_.size());

        // Use provided camera calibration parameters
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
            348.925, 0, 339.075,
            0, 351.135, 177.45,
            0, 0, 1);

        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F); // Assuming no lens distortion

        // R and t to store rotation and translation vectors
        cv::Mat rvec, tvec;

        try
        {
            // Call solvePnPRansac
            cv::solvePnPRansac(curr_image_pts_3d_, curr_image_pts_, camera_matrix, dist_coeffs, rvec, tvec);
            publishPose(rvec, tvec);
        }
        catch (const cv::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "OpenCV solvePnPRansac failed: %s", e.what());
        }
    }

    void convertDepthTo3DPoints()
    {
        // Convert the depth image into 3D points
        curr_image_pts_3d_.clear();
        for (const auto &pt : curr_image_pts_)
        {
            float depth = curr_depth_.at<float>(pt.y, pt.x); // Access depth at feature point
            if (depth > 0) // Valid depth
            {
                float x = (pt.x - 607.1928) * depth / 718.856;
                float y = (pt.y - 185.2157) * depth / 718.856;
                float z = depth;
                curr_image_pts_3d_.push_back(cv::Point3f(x, y, z));
            }
        }
    }

    void publishPose(const cv::Mat &R, const cv::Mat &t)
    {
        // Convert the rotation vector (rvec) to a rotation matrix
        cv::Mat R_mat;
        cv::Rodrigues(R, R_mat);

        // Eigen conversion for orientation (quaternion)
        Eigen::Matrix3f eigen_R;
        cv::cv2eigen(R_mat, eigen_R);
        Eigen::Quaternionf quaternion(eigen_R);

        // Prepare the PoseStamped message
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = this->get_clock()->now();
        pose_msg.header.frame_id = "map"; // Set your frame id

        // Set translation (position)
        pose_msg.pose.position.x = t.at<double>(0);
        pose_msg.pose.position.y = t.at<double>(1);
        pose_msg.pose.position.z = t.at<double>(2);

        // Set orientation (quaternion)
        pose_msg.pose.orientation.x = quaternion.x();
        pose_msg.pose.orientation.y = quaternion.y();
        pose_msg.pose.orientation.z = quaternion.z();
        pose_msg.pose.orientation.w = quaternion.w();

        // Publish the pose
        pose_pub_->publish(pose_msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisualOdometryNode>());
    rclcpp::shutdown();
    return 0;
}
