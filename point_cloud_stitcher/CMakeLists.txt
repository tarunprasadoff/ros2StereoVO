# cmake_policy(SET CMP0074 NEW)  # Add this at the top of the CMakeLists.txt
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
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(octomap REQUIRED)
find_package(Pangolin REQUIRED)
find_package(GLUT REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${octomap_INCLUDE_DIRS}
  # ${octomath_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIRS}
)

# Create the executable
add_executable(point_cloud_stitcher_node src/point_cloud_stitcher.cpp)

# Link the required libraries
target_link_libraries(point_cloud_stitcher_node 
  ${OpenCV_LIBRARIES} 
  ${PCL_LIBRARIES} 
  ${octomap_LIBRARIES}
  # ${octomath_LIBRARIES}
  octomap
  octomath
  ${Pangolin_LIBRARIES}
  ${GLUT_LIBRARIES}
  # pcl_conversions
  # pcl_ros
)

# Add dependencies for ament
ament_target_dependencies(point_cloud_stitcher_node 
  rclcpp 
  sensor_msgs 
  geometry_msgs 
  cv_bridge 
  PCL 
  octomap 
  Pangolin
)

install(TARGETS
  point_cloud_stitcher_node
  DESTINATION lib/${PROJECT_NAME})

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
