
# ROS2 Stereo Visual Odometry and 3D Map Generation

This repository contains a multi-node ROS2 system that processes stereo camera images, estimates visual odometry, and generates a 3D map using depth data. The system consists of four ROS2 packages:
1. Image Loader: Loads and publishes stereo camera images.
2. Visual Odometry: Estimates the robot's pose from the stereo images.
3. Point Cloud Stitcher & Octomap Generator: Unprojects depth images into point clouds, stitches them, and generates a 3D map.
4. Multi Node Launcher: Launches the three nodes together.

## Prerequisites
- ROS2 Humble (or any ROS2 distribution)
- OpenCV: For image processing
- PCL (Point Cloud Library): For point cloud generation
- Octomap: For 3D map generation
- Pangolin: For map visualization

## Installation Instructions

### 1. Set Up a ROS2 Workspace
If you don’t have a ROS2 workspace already, follow these steps to create one:

```
# Create a new directory for the ROS2 workspace
mkdir -p ~/ros2_ws/src

# Navigate into the workspace
cd ~/ros2_ws
```

### 2. Clone the Repository

Inside the `src` folder of your workspace, clone this repository:

```
# Go to the src folder of the workspace
cd ~/ros2_ws/src

# Clone the repository
git clone https://github.com/tarunprasadoff/ros2StereoVO.git
```

### 3. Build the Workspace

Once the repository is cloned, go back to the workspace root and build the packages using `colcon`:

```
# Navigate back to the root of the workspace
cd ~/ros2_ws

# Build all the packages
colcon build
```

### 4. Source the Workspace

Before running any ROS2 commands, source the workspace:

```
source ~/ros2_ws/install/setup.bash
```

### 5. Run the Launch File

Now you can run all the nodes using the provided launch file:

```
ros2 launch multi_node_launcher multi_node_launch.py
```

This command will:
- Launch the `image_loader_node` to publish the stereo images.
- Start the `visual_odometry_node` to estimate the camera pose.
- Run the `point_cloud_stitcher` to convert depth images into point clouds and generate a 3D map.

## Approach

### 1. Image Loader Node
This node reads left and depth camera images from directories, converts them to ROS2 image messages, and publishes them on the appropriate topics. The node uses OpenCV to load the images and `cv_bridge` to convert them for ROS2. The publish rate is configurable via a parameter.

### 2. Visual Odometry Node
The visual odometry node subscribes to the stereo camera images and depth data, using stereo visual odometry techniques to estimate the robot's pose. The node OpenCV’s `solvePnPRansac` to calculate the pose, publishing it as a `PoseStamped` message. The implementation is based on the following stereoVO repository: https://github.com/jaychak/stereoVO/

### 3. Point Cloud Stitcher & Octomap Generator Node
This node subscribes to the depth images and estimated poses, converting depth data into point clouds. These point clouds are stitched together using the pose estimates, and an Octomap is generated. The map is visualized using Pangolin. Due to reliance on the odometry node, the map accuracy depends on the pose estimates.

## Assumptions

- Assumptions:
    - The camera intrinsics provided in the code are accurate.
    - Depth data is reliable enough for point cloud generation.
    - The Pangolin viewer is used for 3D map visualization.
