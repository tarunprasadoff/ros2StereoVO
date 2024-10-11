cmake_minimum_required(VERSION 3.8)
project(point_cloud_stitcher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(Pangolin REQUIRED)

# Add the executable
add_executable(point_cloud_stitcher src/point_cloud_stitcher.cpp)

# Link against the required libraries
ament_target_dependencies(point_cloud_stitcher
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_geometry_msgs
  cv_bridge
  OpenCV
  octomap
  octomap_msgs
  Pangolin
)

target_link_libraries(point_cloud_stitcher
  ${Pangolin_LIBRARIES}
)

# Install the target
install(TARGETS point_cloud_stitcher
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
