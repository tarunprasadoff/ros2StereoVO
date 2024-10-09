#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <octomap/octomap.h>
#include <pangolin/pangolin.h>
#include <GL/glut.h>
#include <mutex>

class PointCloudStitcher : public rclcpp::Node
{
public:
    PointCloudStitcher() : Node("point_cloud_stitcher"), octree_(0.1)
    {
        // Subscribing to depth image and pose topics
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", 10, std::bind(&PointCloudStitcher::depthCallback, this, std::placeholders::_1));

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/robot/pose", 10, std::bind(&PointCloudStitcher::poseCallback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000),
            std::bind(&PointCloudStitcher::stitchAndGenerateMap, this));

        RCLCPP_INFO(this->get_logger(), "PointCloudStitcher node has been started.");
    }

private:
    std::mutex data_mutex_;  // Mutex to prevent concurrent access to shared data

    // Callback for depth image data
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (msg->data.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Received empty depth image.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Received depth image of size: %zu", msg->data.size());

        // Log depth image info for first few points
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        cv::Mat depth_image = cv_ptr->image;
        uint16_t depth_sample = depth_image.at<uint16_t>(0, 0);  // First depth value
        RCLCPP_INFO(this->get_logger(), "First depth value: %u", depth_sample);

        auto cloud = depthToPointCloud(msg);
        accumulated_clouds_.push_back(cloud);
    }

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        current_pose_ = msg;

        // Log current pose
        RCLCPP_INFO(this->get_logger(), "Received pose: x = %f, y = %f, z = %f, qw = %f, qx = %f, qy = %f, qz = %f",
                    msg->pose.position.x, msg->pose.position.y, msg->pose.position.z,
                    msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    }

    // Convert depth image to point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr depthToPointCloud(const sensor_msgs::msg::Image::SharedPtr depth_msg)
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        cv::Mat depth_image = cv_ptr->image;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        // Camera intrinsics
        float fx = 348.925;
        float fy = 351.135;
        float cx = 339.075;
        float cy = 177.45;
        float depth_scale = 0.001;  // Adjust based on your depth image scale

        for (int v = 0; v < depth_image.rows; ++v)
        {
            for (int u = 0; u < depth_image.cols; ++u)
            {
                uint16_t depth = depth_image.at<uint16_t>(v, u);
                if (depth == 0) continue;  // Skip invalid depth points

                float z = depth * depth_scale;

                // Sanity check to avoid division by zero
                if (fx == 0.0 || fy == 0.0)
                {
                    RCLCPP_ERROR(this->get_logger(), "Invalid camera intrinsic values. fx or fy is zero.");
                    continue;
                }

                float x = (u - cx) * z / fx;
                float y = (v - cy) * z / fy;

                // Additional sanity check for NaN or infinity values
                if (std::isnan(x) || std::isnan(y) || std::isnan(z) || 
                    std::isinf(x) || std::isinf(y) || std::isinf(z))
                {
                    RCLCPP_ERROR(this->get_logger(), "Invalid point detected: x = %f, y = %f, z = %f", x, y, z);
                    continue;
                }

                cloud->points.emplace_back(x, y, z);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Converted depth image to point cloud with %zu points.", cloud->points.size());

        return cloud;
    }

    // Transform point cloud using robot pose
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const geometry_msgs::msg::PoseStamped::SharedPtr pose_msg)
    {
        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        transform.translation() << pose_msg->pose.position.x, pose_msg->pose.position.y, pose_msg->pose.position.z;
        Eigen::Quaterniond q(pose_msg->pose.orientation.w, pose_msg->pose.orientation.x, pose_msg->pose.orientation.y, pose_msg->pose.orientation.z);
        
        // Sanity check for valid quaternion
        if (q.norm() == 0.0)
        {
            RCLCPP_ERROR(this->get_logger(), "Invalid quaternion in pose data.");
            return cloud;  // Return the original cloud if quaternion is invalid
        }

        transform.rotate(q);

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform.matrix());

        RCLCPP_INFO(this->get_logger(), "Transformed point cloud with pose data.");

        return transformed_cloud;
    }

    // Stitch point clouds and generate an Octomap
    void stitchAndGenerateMap()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        try
        {
            if (current_pose_ == nullptr || accumulated_clouds_.empty())
            {
                RCLCPP_WARN(this->get_logger(), "No pose or depth clouds available, skipping stitching.");
                return;
            }

            // Check if the pose is effectively "empty" (no movement)
            if (current_pose_->pose.position.x == 0.0 && current_pose_->pose.position.y == 0.0 && current_pose_->pose.position.z == 0.0 &&
                current_pose_->pose.orientation.w == 1.0 && current_pose_->pose.orientation.x == 0.0 && current_pose_->pose.orientation.y == 0.0 &&
                current_pose_->pose.orientation.z == 0.0)
            {
                RCLCPP_WARN(this->get_logger(), "Received pose is identity (no movement), skipping map update.");
                return; // Skip updating the map if there is no movement
            }

            RCLCPP_INFO(this->get_logger(), "Stitching and generating Octomap.");

            for (auto& cloud : accumulated_clouds_)
            {
                auto transformed_cloud = transformPointCloud(cloud, current_pose_);

                // Skip if the transformed cloud is empty
                if (transformed_cloud->points.empty())
                {
                    RCLCPP_WARN(this->get_logger(), "Transformed point cloud is empty, skipping this cloud.");
                    continue;
                }

                for (const auto& point : transformed_cloud->points)
                {
                    // Sanity check for invalid points
                    if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z) ||
                        std::isinf(point.x) || std::isinf(point.y) || std::isinf(point.z))
                    {
                        RCLCPP_ERROR(this->get_logger(), "Invalid point detected after transformation: x = %f, y = %f, z = %f", point.x, point.y, point.z);
                        continue;  // Skip invalid points
                    }

                    // Ensure the point is within a reasonable range (to avoid out-of-bounds errors)
                    if (std::abs(point.x) > 1000 || std::abs(point.y) > 1000 || std::abs(point.z) > 1000)
                    {
                        RCLCPP_WARN(this->get_logger(), "Point is out of bounds: x = %f, y = %f, z = %f", point.x, point.y, point.z);
                        continue;
                    }

                    // Insert valid points into octree
                    RCLCPP_INFO(this->get_logger(), "Inserting point into Octomap: x = %f, y = %f, z = %f", point.x, point.y, point.z);
                    octree_.updateNode(octomap::point3d(point.x, point.y, point.z), true);
                }
            }

            octree_.updateInnerOccupancy();
            octree_.writeBinary("map.bt");

            RCLCPP_INFO(this->get_logger(), "Octomap generated and saved to map.bt.");

            accumulated_clouds_.clear();  // Clear buffer after stitching
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Exception caught during map stitching: %s", e.what());
        }
        catch (...)
        {
            RCLCPP_ERROR(this->get_logger(), "Unknown exception caught during map stitching.");
        }
    }

    // Display Octomap using Pangolin
    void displayOctomap(octomap::OcTree& octree)
    {
        pangolin::CreateWindowAndBind("Octomap Viewer", 640, 480);
        glEnable(GL_DEPTH_TEST);

        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin::ModelViewLookAt(-1, -1, -1, 0, 0, 0, pangolin::AxisY)
        );

        pangolin::Handler3D handler(s_cam);
        pangolin::View& d_cam = pangolin::CreateDisplay()
                               .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
                               .SetHandler(&handler);

        while (!pangolin::ShouldQuit())
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);
            glColor3f(1.0, 1.0, 1.0);

            for (auto it = octree.begin(), end = octree.end(); it != end; ++it)
            {
                if (octree.isNodeOccupied(*it))
                {
                    octomap::point3d point = it.getCoordinate();
                    double size = it.getSize() / 2.0;

                    glPushMatrix();
                    glTranslatef(point.x(), point.y(), point.z());
                    glutWireCube(size);
                    glPopMatrix();
                }
            }

            pangolin::FinishFrame();
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    geometry_msgs::msg::PoseStamped::SharedPtr current_pose_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
    octomap::OcTree octree_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudStitcher>());
    rclcpp::shutdown();
    return 0;
}
