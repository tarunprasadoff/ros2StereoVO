#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "octomap/octomap.h"
#include "octomap/OcTree.h"
#include "octomap_msgs/msg/octomap.hpp"
#include "octomap_msgs/conversions.h"
#include <pangolin/pangolin.h>
#include <vector>

class PointCloudStitcher : public rclcpp::Node {
public:
    PointCloudStitcher() : Node("point_cloud_stitcher"), frame_count(0) {
        // Subscription to depth image
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", 10,
            std::bind(&PointCloudStitcher::depthImageCallback, this, std::placeholders::_1));

        // Subscription to robot pose
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/robot/pose", 10,
            std::bind(&PointCloudStitcher::poseCallback, this, std::placeholders::_1));

        // Create an Octomap with a resolution of 0.1 meters
        octree_ = std::make_shared<octomap::OcTree>(0.1);  // 0.1 meter resolution

        // Initialize Pangolin for visualization
        initPangolin();
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    geometry_msgs::msg::PoseStamped current_pose_;
    std::vector<geometry_msgs::msg::Point> accumulated_point_cloud;
    int frame_count;
    const int N = 10;  // Number of frames to accumulate
    std::shared_ptr<octomap::OcTree> octree_;  // Octree for Octomap
    pangolin::OpenGlRenderState s_cam_;
    pangolin::View* d_cam_;

    // Camera intrinsics
    const double fx = 348.925, fy = 351.135;
    const double cx = 339.075, cy = 177.45;

    // Callback for receiving depth images
    void depthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Convert depth image to OpenCV format
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Unproject the depth image to a point cloud
        std::vector<geometry_msgs::msg::Point> local_cloud = unprojectDepthImage(cv_ptr->image);
        
        // Transform the point cloud to the world frame using the current pose
        std::vector<geometry_msgs::msg::Point> world_cloud = transformToWorldFrame(local_cloud, current_pose_);
        
        // Accumulate the point cloud
        accumulated_point_cloud.insert(accumulated_point_cloud.end(), world_cloud.begin(), world_cloud.end());
        frame_count++;

        // If we have accumulated N frames, process the point cloud
        if (frame_count >= N) {
            // Process the accumulated point cloud (stitching and generating Octomap)
            stitchAndGenerateOctomap();
            
            // Clear the accumulated point cloud and reset the frame counter
            accumulated_point_cloud.clear();
            frame_count = 0;
        }
    }

    // Callback for receiving pose data
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        current_pose_ = *msg;  // Store the current pose for transformations
    }

    // Function to unproject a depth image into a 3D point cloud
    std::vector<geometry_msgs::msg::Point> unprojectDepthImage(const cv::Mat& depth_image) {
        std::vector<geometry_msgs::msg::Point> point_cloud;

        for (int v = 0; v < depth_image.rows; ++v) {
            for (int u = 0; u < depth_image.cols; ++u) {
                float Z = depth_image.at<float>(v, u);
                if (Z > 0) {  // Only consider points with a valid depth
                    geometry_msgs::msg::Point point;
                    point.z = Z;
                    point.x = (u - cx) * Z / fx;
                    point.y = (v - cy) * Z / fy;
                    point_cloud.push_back(point);
                }
            }
        }

        return point_cloud;
    }

    // Function to transform the point cloud to the world frame using the pose
    std::vector<geometry_msgs::msg::Point> transformToWorldFrame(
        const std::vector<geometry_msgs::msg::Point>& local_cloud,
        const geometry_msgs::msg::PoseStamped& pose) {

        std::vector<geometry_msgs::msg::Point> world_cloud;

        // Create transformation matrix from pose (rotation and translation)
        tf2::Quaternion q(pose.pose.orientation.x, pose.pose.orientation.y,
                          pose.pose.orientation.z, pose.pose.orientation.w);
        tf2::Matrix3x3 rotation_matrix(q);

        for (const auto& point : local_cloud) {
            geometry_msgs::msg::Point world_point;
            
            // Apply rotation
            world_point.x = rotation_matrix[0][0] * point.x + rotation_matrix[0][1] * point.y + rotation_matrix[0][2] * point.z;
            world_point.y = rotation_matrix[1][0] * point.x + rotation_matrix[1][1] * point.y + rotation_matrix[1][2] * point.z;
            world_point.z = rotation_matrix[2][0] * point.x + rotation_matrix[2][1] * point.y + rotation_matrix[2][2] * point.z;
            
            // Apply translation (robot's position in the world)
            world_point.x += pose.pose.position.x;
            world_point.y += pose.pose.position.y;
            world_point.z += pose.pose.position.z;

            world_cloud.push_back(world_point);
        }

        return world_cloud;
    }

    // Initialize Pangolin for visualization
    void initPangolin() {
        // Create OpenGL window in single line
        pangolin::CreateWindowAndBind("Octomap Viewer", 640, 480);

        // Enable depth
        glEnable(GL_DEPTH_TEST);

        // Adjust camera projection and modelview matrix
        s_cam_ = pangolin::OpenGlRenderState(
           pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),  // Adjust near and far planes
           pangolin::ModelViewLookAt(0, 0, -10, 0, 0, 0, pangolin::AxisY)  // Adjust the look-at position
        );

        // Create Interactive View in window
        d_cam_ = &pangolin::CreateDisplay()
           .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
           .SetHandler(new pangolin::Handler3D(s_cam_));
    }

    // Function to render the Octomap in Pangolin
    void renderOctomap() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam_->Activate(s_cam_);

        for (octomap::OcTree::leaf_iterator it = octree_->begin_leafs(),
            end = octree_->end_leafs(); it != end; ++it) {
           
            if (octree_->isNodeOccupied(*it)) {
                double voxel_size = it.getSize();
                octomap::point3d pt = it.getCoordinate();
                
                // Debugging: Log inserted voxel information
                RCLCPP_INFO(this->get_logger(), "Voxel at x: %f, y: %f, z: %f with size: %f", 
                            pt.x(), pt.y(), pt.z(), voxel_size);

                // Set color and draw the voxel as a cube
                glColor3f(0.0f, 0.5f, 1.0f);  // Set a color

                glPushMatrix();
                glTranslatef(pt.x(), pt.y(), pt.z());
                glScalef(voxel_size, voxel_size, voxel_size);  // Scale the cube to voxel size
                pangolin::glDrawColouredCube(-0.5f, 0.5f);  // Draw the cube
                glPopMatrix();
            }
        }

        pangolin::FinishFrame();  // Render frame
    }

    // Function to stitch the point clouds and generate an Octomap
    void stitchAndGenerateOctomap() {
        // Insert accumulated points into Octree
        for (const auto& point : accumulated_point_cloud) {
            octree_->updateNode(octomap::point3d(point.x, point.y, point.z), true);
            
            // Debugging: Log accumulated points
            RCLCPP_INFO(this->get_logger(), "Accumulated point at x: %f, y: %f, z: %f", 
                        point.x, point.y, point.z);
        }

        // After updating the octree, clear the accumulated points
        accumulated_point_cloud.clear();

        // Render the updated Octomap
        renderOctomap();
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    // Create and run the ROS2 node
    rclcpp::spin(std::make_shared<PointCloudStitcher>());

    rclcpp::shutdown();
    return 0;
}
