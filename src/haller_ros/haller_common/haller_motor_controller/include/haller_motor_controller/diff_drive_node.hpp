#pragma once

#include "haller_motor_controller/lktech_motor.hpp"
#include "haller_motor_controller/slcan_interface.hpp"

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <memory>

namespace haller_motor_controller {

class DiffDriveNode : public rclcpp::Node {
public:
  explicit DiffDriveNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~DiffDriveNode() override;

private:
  // Callbacks
  void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void controlLoop();
  void diagnosticsTick();

  // Service
  void motorEnableCallback(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response);

  // Motor control
  bool initMotors();
  void shutdownMotors();

  // Parameters
  std::string can_device_;
  int can_bitrate_;
  int left_motor_id_;
  int right_motor_id_;
  double wheel_radius_;
  double wheel_separation_;
  double max_linear_speed_;
  double max_angular_speed_;
  double cmd_vel_timeout_;
  double control_rate_;
  std::string odom_frame_;
  std::string base_frame_;

  // Hardware
  SLCANInterface slcan_;
  std::unique_ptr<LKTechMotor> left_motor_;
  std::unique_ptr<LKTechMotor> right_motor_;
  bool motors_enabled_ = false;

  // ROS interfaces
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr motor_enable_srv_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;

  // State
  geometry_msgs::msg::Twist last_cmd_vel_;
  rclcpp::Time last_cmd_vel_time_;
  double odom_x_ = 0.0;
  double odom_y_ = 0.0;
  double odom_theta_ = 0.0;
  double left_wheel_pos_ = 0.0;   // radians (cumulative)
  double right_wheel_pos_ = 0.0;  // radians (cumulative)
  double prev_left_angle_ = 0.0;  // previous motor reading
  double prev_right_angle_ = 0.0;
  bool first_angle_read_ = true;
};

} // namespace haller_motor_controller
