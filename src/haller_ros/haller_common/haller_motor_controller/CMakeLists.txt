cmake_minimum_required(VERSION 3.8)
project(haller_motor_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Motor controller executable
add_executable(motor_controller_node
  src/main.cpp
  src/slcan_interface.cpp
  src/lktech_motor.cpp
  src/diff_drive_node.cpp
)

target_include_directories(motor_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(motor_controller_node
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Install executable
install(TARGETS motor_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install config, launch, scripts
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/set_motor_id.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
