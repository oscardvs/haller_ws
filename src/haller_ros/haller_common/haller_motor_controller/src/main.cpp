#include "haller_motor_controller/diff_drive_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<haller_motor_controller::DiffDriveNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
