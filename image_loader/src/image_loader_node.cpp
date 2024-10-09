#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>
#include <filesystem>

namespace fs = std::filesystem;

class ImageLoaderNode : public rclcpp::Node
{
public:
    ImageLoaderNode() : Node("image_loader_node"), image_index_(0)
    {
        // Declare parameters for directories and the rate of publishing
        this->declare_parameter<std::string>("left_image_dir", "/home/tarun/rainier/assignmentData/left/");
        this->declare_parameter<std::string>("depth_image_dir", "/home/tarun/rainier/assignmentData/depth/");
        this->declare_parameter<double>("publish_rate", 1.0);

        // Load parameters
        this->get_parameter("left_image_dir", left_image_dir_);
        this->get_parameter("depth_image_dir", depth_image_dir_);
        this->get_parameter("publish_rate", publish_rate_);

        // Create publishers for left and depth images
        left_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/left/image_raw", 10);
        depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/depth/image_raw", 10);

        // Create a timer to publish images at the specified rate
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
            std::bind(&ImageLoaderNode::publishImages, this));
    }

private:
    void publishImages()
    {
        // Load the current left and depth images
        std::string left_image_path = left_image_dir_ + std::to_string(image_index_) + ".png";
        std::string depth_image_path = depth_image_dir_ + std::to_string(image_index_) + ".png";

        // Check if the files exist
        if (!fs::exists(left_image_path) || !fs::exists(depth_image_path)) {
            RCLCPP_WARN(this->get_logger(), "Image files not found: %s or %s", left_image_path.c_str(), depth_image_path.c_str());
            return;
        }

        // Load images using OpenCV
        cv::Mat left_image = cv::imread(left_image_path, cv::IMREAD_UNCHANGED);  // Keep original channels (RGBA)
        cv::Mat depth_image = cv::imread(depth_image_path, cv::IMREAD_UNCHANGED);  // Depth image as 16-bit

        if (left_image.empty() || depth_image.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load images: %s or %s", left_image_path.c_str(), depth_image_path.c_str());
            return;
        }

        // Check if the left image is RGBA (4 channels) and convert to BGR
        if (left_image.channels() == 4) {
            cv::Mat bgr_image;
            cv::cvtColor(left_image, bgr_image, cv::COLOR_RGBA2BGR);  // Convert from RGBA to BGR
            left_image = bgr_image;
        } else if (left_image.channels() != 3) {
            RCLCPP_WARN(this->get_logger(), "Unexpected number of channels in left image: %d", left_image.channels());
            return;
        }

        // Convert OpenCV images to ROS2 messages
        auto left_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", left_image).toImageMsg();
        auto depth_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono16", depth_image).toImageMsg();  // 16-bit depth

        // Publish the images
        left_image_pub_->publish(*left_msg);
        depth_image_pub_->publish(*depth_msg);

        RCLCPP_INFO(this->get_logger(), "Published images: %s and %s", left_image_path.c_str(), depth_image_path.c_str());

        // Increment the image index
        image_index_++;
    }

    // Parameters
    std::string left_image_dir_;
    std::string depth_image_dir_;
    double publish_rate_;
    int image_index_;

    // ROS2 publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_;

    // Timer for publishing at a fixed rate
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImageLoaderNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}