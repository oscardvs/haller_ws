#include "haller_hardware_interface/haller_hardware_interface.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace haller_hardware_interface
{

hardware_interface::CallbackReturn HallerHardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Get hardware parameters from URDF
  serial_port_ = info_.hardware_parameters.count("serial_port") ?
    info_.hardware_parameters.at("serial_port") : "/dev/ttyUSB0";
  baud_rate_ = info_.hardware_parameters.count("baud_rate") ?
    std::stoi(info_.hardware_parameters.at("baud_rate")) : 115200;

  // Initialize storage for 4 wheels
  hw_positions_.resize(info_.joints.size(), 0.0);
  hw_velocities_.resize(info_.joints.size(), 0.0);
  hw_commands_.resize(info_.joints.size(), 0.0);

  // Validate joint configuration
  for (const hardware_interface::ComponentInfo & joint : info_.joints) {
    // Each joint should have one command interface (velocity)
    if (joint.command_interfaces.size() != 1) {
      RCLCPP_FATAL(
        rclcpp::get_logger("HallerHardwareInterface"),
        "Joint '%s' has %zu command interfaces. Expected 1.",
        joint.name.c_str(), joint.command_interfaces.size());
      return hardware_interface::CallbackReturn::ERROR;
    }

    if (joint.command_interfaces[0].name != hardware_interface::HW_IF_VELOCITY) {
      RCLCPP_FATAL(
        rclcpp::get_logger("HallerHardwareInterface"),
        "Joint '%s' has command interface '%s'. Expected '%s'.",
        joint.name.c_str(), joint.command_interfaces[0].name.c_str(),
        hardware_interface::HW_IF_VELOCITY);
      return hardware_interface::CallbackReturn::ERROR;
    }

    // Each joint should have two state interfaces (position, velocity)
    if (joint.state_interfaces.size() != 2) {
      RCLCPP_FATAL(
        rclcpp::get_logger("HallerHardwareInterface"),
        "Joint '%s' has %zu state interfaces. Expected 2.",
        joint.name.c_str(), joint.state_interfaces.size());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  RCLCPP_INFO(
    rclcpp::get_logger("HallerHardwareInterface"),
    "Initialized with %zu joints", info_.joints.size());

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn HallerHardwareInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(
    rclcpp::get_logger("HallerHardwareInterface"),
    "Configuring hardware interface...");

  // Reset states
  for (size_t i = 0; i < hw_positions_.size(); i++) {
    hw_positions_[i] = 0.0;
    hw_velocities_[i] = 0.0;
    hw_commands_[i] = 0.0;
  }

  RCLCPP_INFO(
    rclcpp::get_logger("HallerHardwareInterface"),
    "Hardware interface configured");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn HallerHardwareInterface::on_cleanup(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(
    rclcpp::get_logger("HallerHardwareInterface"),
    "Cleaning up hardware interface...");

  disconnect_from_hardware();

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn HallerHardwareInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(
    rclcpp::get_logger("HallerHardwareInterface"),
    "Activating hardware interface...");

  if (!connect_to_hardware()) {
    RCLCPP_ERROR(
      rclcpp::get_logger("HallerHardwareInterface"),
      "Failed to connect to hardware");
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Reset commands
  for (size_t i = 0; i < hw_commands_.size(); i++) {
    hw_commands_[i] = 0.0;
  }

  RCLCPP_INFO(
    rclcpp::get_logger("HallerHardwareInterface"),
    "Hardware interface activated");

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn HallerHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(
    rclcpp::get_logger("HallerHardwareInterface"),
    "Deactivating hardware interface...");

  // Stop all motors
  for (size_t i = 0; i < hw_commands_.size(); i++) {
    hw_commands_[i] = 0.0;
  }
  send_velocity_commands(hw_commands_);

  disconnect_from_hardware();

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
HallerHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  for (size_t i = 0; i < info_.joints.size(); i++) {
    state_interfaces.emplace_back(
      hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]));
    state_interfaces.emplace_back(
      hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
HallerHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  for (size_t i = 0; i < info_.joints.size(); i++) {
    command_interfaces.emplace_back(
      hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_commands_[i]));
  }

  return command_interfaces;
}

hardware_interface::return_type HallerHardwareInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  // Read encoder values from hardware
  // TODO: Implement actual hardware communication
  // For now, simulate wheel motion based on commands

  for (size_t i = 0; i < hw_positions_.size(); i++) {
    // Simulate position update based on velocity
    hw_positions_[i] += hw_velocities_[i] * period.seconds();

    // Simulate velocity approaching command velocity
    hw_velocities_[i] = hw_commands_[i];
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type HallerHardwareInterface::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Send velocity commands to hardware
  // TODO: Implement actual hardware communication

  return hardware_interface::return_type::OK;
}

bool HallerHardwareInterface::connect_to_hardware()
{
  // TODO: Implement serial connection to motor controllers
  RCLCPP_INFO(
    rclcpp::get_logger("HallerHardwareInterface"),
    "Connecting to hardware on %s at %d baud (simulated)",
    serial_port_.c_str(), baud_rate_);
  return true;
}

void HallerHardwareInterface::disconnect_from_hardware()
{
  // TODO: Implement serial disconnection
  RCLCPP_INFO(
    rclcpp::get_logger("HallerHardwareInterface"),
    "Disconnected from hardware");
}

bool HallerHardwareInterface::send_velocity_commands(
  const std::vector<double> & /*velocities*/)
{
  // TODO: Implement command sending
  return true;
}

bool HallerHardwareInterface::read_encoder_values(
  std::vector<double> & /*positions*/,
  std::vector<double> & /*velocities*/)
{
  // TODO: Implement encoder reading
  return true;
}

}  // namespace haller_hardware_interface

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  haller_hardware_interface::HallerHardwareInterface,
  hardware_interface::SystemInterface)

