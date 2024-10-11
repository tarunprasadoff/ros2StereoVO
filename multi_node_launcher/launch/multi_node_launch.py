from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_loader',
            executable='image_loader_node',
            name='image_loader',
            output='screen'
        ),
        Node(
            package='visual_odometry',
            executable='visual_odometry_node',
            name='visual_odometry',
            output='screen'
        ),
        Node(
            package='point_cloud_stitcher',
            executable='point_cloud_stitcher',
            name='point_cloud_stitcher',
            output='screen'
        )
    ])
