#include "haller_motor_controller/diff_drive_node.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <cmath>

namespace haller_motor_controller {

DiffDriveNode::DiffDriveNode(const rclcpp::NodeOptions &options)
    : Node("motor_controller", options) {
  // Declare parameters
  this->declare_parameter("can_device", "/dev/ttyACM0");
  this->declare_parameter("can_bitrate", 1000000);
  this->declare_parameter("left_motor_id", 4);
  this->declare_parameter("right_motor_id", 1);
  this->declare_parameter("wheel_radius", 0.05);
  this->declare_parameter("wheel_separation", 0.35);
  this->declare_parameter("max_linear_speed", 1.0);
  this->declare_parameter("max_angular_speed", 2.0);
  this->declare_parameter("cmd_vel_timeout", 0.5);
  this->declare_parameter("control_rate", 50.0);
  this->declare_parameter("odom_frame", "odom");
  this->declare_parameter("base_frame", "base_link");

  // Read parameters
  can_device_ = this->get_parameter("can_device").as_string();
  can_bitrate_ = this->get_parameter("can_bitrate").as_int();
  left_motor_id_ = this->get_parameter("left_motor_id").as_int();
  right_motor_id_ = this->get_parameter("right_motor_id").as_int();
  wheel_radius_ = this->get_parameter("wheel_radius").as_double();
  wheel_separation_ = this->get_parameter("wheel_separation").as_double();
  max_linear_speed_ = this->get_parameter("max_linear_speed").as_double();
  max_angular_speed_ = this->get_parameter("max_angular_speed").as_double();
  cmd_vel_timeout_ = this->get_parameter("cmd_vel_timeout").as_double();
  control_rate_ = this->get_parameter("control_rate").as_double();
  odom_frame_ = this->get_parameter("odom_frame").as_string();
  base_frame_ = this->get_parameter("base_frame").as_string();

  RCLCPP_INFO(this->get_logger(),
              "Diff drive controller: device=%s, bitrate=%d, "
              "left_id=%d, right_id=%d, wheel_r=%.3f, wheel_sep=%.3f",
              can_device_.c_str(), can_bitrate_, left_motor_id_,
              right_motor_id_, wheel_radius_, wheel_separation_);

  // Create publishers
  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
  joint_state_pub_ =
      this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // Create subscriber
  cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 10,
      std::bind(&DiffDriveNode::cmdVelCallback, this, std::placeholders::_1));

  // Create service
  motor_enable_srv_ = this->create_service<std_srvs::srv::SetBool>(
      "~/motor_enable",
      std::bind(&DiffDriveNode::motorEnableCallback, this,
                std::placeholders::_1, std::placeholders::_2));

  // Initialize last_cmd_vel_time to now (so watchdog doesn't fire immediately)
  last_cmd_vel_time_ = this->now();

  // Initialize motors
  if (!initMotors()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Failed to initialize motors! Node will retry on cmd_vel.");
  }

  // Control loop timer
  auto period =
      std::chrono::duration<double>(1.0 / control_rate_);
  control_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&DiffDriveNode::controlLoop, this));

  // Diagnostics timer (1 Hz)
  diagnostics_timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&DiffDriveNode::diagnosticsTick, this));

  RCLCPP_INFO(this->get_logger(), "Diff drive node initialized at %.0f Hz",
              control_rate_);
}

DiffDriveNode::~DiffDriveNode() { shutdownMotors(); }

bool DiffDriveNode::initMotors() {
  if (!slcan_.open(can_device_, can_bitrate_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open SLCAN device: %s",
                 can_device_.c_str());
    return false;
  }

  left_motor_ = std::make_unique<LKTechMotor>(
      static_cast<uint8_t>(left_motor_id_), slcan_);
  right_motor_ = std::make_unique<LKTechMotor>(
      static_cast<uint8_t>(right_motor_id_), slcan_);

  // Turn on both motors
  if (!left_motor_->motorOn()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to turn on left motor (ID %d)",
                 left_motor_id_);
    return false;
  }
  if (!right_motor_->motorOn()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to turn on right motor (ID %d)",
                 right_motor_id_);
    return false;
  }

  motors_enabled_ = true;
  first_angle_read_ = true;

  RCLCPP_INFO(this->get_logger(),
              "Motors initialized: left(ID %d, CAN 0x%03X), "
              "right(ID %d, CAN 0x%03X)",
              left_motor_id_, left_motor_->canId(), right_motor_id_,
              right_motor_->canId());
  return true;
}

void DiffDriveNode::shutdownMotors() {
  if (left_motor_) {
    left_motor_->motorStop();
    left_motor_->motorOff();
  }
  if (right_motor_) {
    right_motor_->motorStop();
    right_motor_->motorOff();
  }
  motors_enabled_ = false;
  slcan_.close();
  RCLCPP_INFO(this->get_logger(), "Motors shut down");
}

void DiffDriveNode::cmdVelCallback(
    const geometry_msgs::msg::Twist::SharedPtr msg) {
  last_cmd_vel_ = *msg;
  last_cmd_vel_time_ = this->now();
}

void DiffDriveNode::controlLoop() {
  if (!motors_enabled_ || !slcan_.isOpen()) {
    return;
  }

  auto now = this->now();

  // Check cmd_vel watchdog
  double time_since_cmd = (now - last_cmd_vel_time_).seconds();
  double linear_vel = 0.0;
  double angular_vel = 0.0;

  if (time_since_cmd < cmd_vel_timeout_) {
    // Clamp velocities
    linear_vel = std::clamp(last_cmd_vel_.linear.x, -max_linear_speed_,
                            max_linear_speed_);
    angular_vel = std::clamp(last_cmd_vel_.angular.z, -max_angular_speed_,
                             max_angular_speed_);
  }
  // else: timeout → velocities stay at 0.0

  // Differential drive kinematics: cmd_vel → wheel speeds (rad/s)
  double v_left_rad =
      (linear_vel - angular_vel * wheel_separation_ / 2.0) / wheel_radius_;
  double v_right_rad =
      (linear_vel + angular_vel * wheel_separation_ / 2.0) / wheel_radius_;

  // Convert rad/s → degrees/s for motor commands
  // Right motor is mounted mirrored, so negate its direction
  double v_left_dps = v_left_rad * 180.0 / M_PI;
  double v_right_dps = -(v_right_rad * 180.0 / M_PI);

  // Send speed commands to motors
  if (!left_motor_->setSpeed(v_left_dps)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Failed to send speed to left motor");
  }
  if (!right_motor_->setSpeed(v_right_dps)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Failed to send speed to right motor");
  }

  // Read wheel angles for odometry
  double left_angle_rad = 0.0;
  double right_angle_rad = 0.0;
  bool left_ok = left_motor_->readMultiAngle(left_angle_rad);
  bool right_ok = right_motor_->readMultiAngle(right_angle_rad);
  right_angle_rad = -right_angle_rad;  // Right motor is mirrored

  if (left_ok && right_ok) {
    if (first_angle_read_) {
      // Initialize previous angles on first read
      prev_left_angle_ = left_angle_rad;
      prev_right_angle_ = right_angle_rad;
      first_angle_read_ = false;
    } else {
      // Compute wheel angle deltas
      double delta_left = left_angle_rad - prev_left_angle_;
      double delta_right = right_angle_rad - prev_right_angle_;

      // Update cumulative wheel positions (for joint_states)
      left_wheel_pos_ += delta_left;
      right_wheel_pos_ += delta_right;

      // Compute linear and angular displacement
      double d_left = delta_left * wheel_radius_;
      double d_right = delta_right * wheel_radius_;
      double d_center = (d_left + d_right) / 2.0;
      double d_theta = (d_right - d_left) / wheel_separation_;

      // Update odometry pose (midpoint integration)
      double mid_theta = odom_theta_ + d_theta / 2.0;
      odom_x_ += d_center * cos(mid_theta);
      odom_y_ += d_center * sin(mid_theta);
      odom_theta_ += d_theta;

      // Normalize theta to [-pi, pi]
      odom_theta_ = atan2(sin(odom_theta_), cos(odom_theta_));

      prev_left_angle_ = left_angle_rad;
      prev_right_angle_ = right_angle_rad;
    }

    // Compute current velocities from wheel speed commands (for odom twist)
    double vel_linear = (v_left_rad + v_right_rad) / 2.0 * wheel_radius_;
    double vel_angular =
        (v_right_rad - v_left_rad) / wheel_separation_ * wheel_radius_;

    // Publish odometry
    auto odom_msg = nav_msgs::msg::Odometry();
    odom_msg.header.stamp = now;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = base_frame_;

    // Pose
    odom_msg.pose.pose.position.x = odom_x_;
    odom_msg.pose.pose.position.y = odom_y_;
    odom_msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, odom_theta_);
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();

    // Twist (in base_link frame)
    odom_msg.twist.twist.linear.x = vel_linear;
    odom_msg.twist.twist.angular.z = vel_angular;

    // Covariance (diagonal)
    odom_msg.pose.covariance[0] = 0.001;  // x
    odom_msg.pose.covariance[7] = 0.001;  // y
    odom_msg.pose.covariance[14] = 1e6;   // z (not measured)
    odom_msg.pose.covariance[21] = 1e6;   // roll
    odom_msg.pose.covariance[28] = 1e6;   // pitch
    odom_msg.pose.covariance[35] = 0.01;  // yaw

    odom_pub_->publish(odom_msg);

    // Publish TF: odom → base_link
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = now;
    tf_msg.header.frame_id = odom_frame_;
    tf_msg.child_frame_id = base_frame_;
    tf_msg.transform.translation.x = odom_x_;
    tf_msg.transform.translation.y = odom_y_;
    tf_msg.transform.translation.z = 0.0;
    tf_msg.transform.rotation = odom_msg.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf_msg);

    // Publish joint states for both front wheels
    auto js_msg = sensor_msgs::msg::JointState();
    js_msg.header.stamp = now;
    js_msg.name = {"left_wheel_joint", "right_wheel_joint"};
    js_msg.position = {left_wheel_pos_, right_wheel_pos_};
    js_msg.velocity = {v_left_rad, v_right_rad};
    joint_state_pub_->publish(js_msg);
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Failed to read motor angles (left=%s, right=%s)",
                         left_ok ? "ok" : "FAIL", right_ok ? "ok" : "FAIL");
  }
}

void DiffDriveNode::diagnosticsTick() {
  if (!motors_enabled_ || !slcan_.isOpen()) {
    return;
  }

  // Read error/voltage from both motors
  int8_t temp;
  double voltage;
  uint8_t error_state;

  if (left_motor_->readError(temp, voltage, error_state)) {
    if (error_state != 0) {
      RCLCPP_WARN(this->get_logger(),
                  "Left motor error: 0x%02X, temp=%d°C, voltage=%.1fV",
                  error_state, temp, voltage);
    } else {
      RCLCPP_DEBUG(this->get_logger(),
                   "Left motor: temp=%d°C, voltage=%.1fV", temp, voltage);
    }
  }

  if (right_motor_->readError(temp, voltage, error_state)) {
    if (error_state != 0) {
      RCLCPP_WARN(this->get_logger(),
                  "Right motor error: 0x%02X, temp=%d°C, voltage=%.1fV",
                  error_state, temp, voltage);
    } else {
      RCLCPP_DEBUG(this->get_logger(),
                   "Right motor: temp=%d°C, voltage=%.1fV", temp, voltage);
    }
  }
}

void DiffDriveNode::motorEnableCallback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
  if (request->data) {
    // Enable motors
    if (motors_enabled_) {
      response->success = true;
      response->message = "Motors already enabled";
      return;
    }
    if (initMotors()) {
      response->success = true;
      response->message = "Motors enabled";
    } else {
      response->success = false;
      response->message = "Failed to enable motors";
    }
  } else {
    // Disable motors
    shutdownMotors();
    response->success = true;
    response->message = "Motors disabled";
  }
}

} // namespace haller_motor_controller
