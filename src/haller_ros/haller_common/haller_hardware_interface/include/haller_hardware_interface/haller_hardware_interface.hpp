#ifndef HALLER_HARDWARE_INTERFACE__HALLER_HARDWARE_INTERFACE_HPP_
#define HALLER_HARDWARE_INTERFACE__HALLER_HARDWARE_INTERFACE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace haller_hardware_interface
{

class HallerHardwareInterface : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(HallerHardwareInterface)

  /// Initialize the hardware interface from URDF data
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  /// Configure the hardware interface
  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  /// Cleanup the hardware interface
  hardware_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;

  /// Activate the hardware interface
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  /// Deactivate the hardware interface
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  /// Export state interfaces
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  /// Export command interfaces
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  /// Read hardware state
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  /// Write commands to hardware
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // Hardware parameters
  std::string serial_port_;
  int baud_rate_;

  // Joint state storage
  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
  std::vector<double> hw_commands_;

  // Hardware communication (to be implemented)
  bool connect_to_hardware();
  void disconnect_from_hardware();
  bool send_velocity_commands(const std::vector<double> & velocities);
  bool read_encoder_values(std::vector<double> & positions, std::vector<double> & velocities);
};

}  // namespace haller_hardware_interface

#endif  // HALLER_HARDWARE_INTERFACE__HALLER_HARDWARE_INTERFACE_HPP_

