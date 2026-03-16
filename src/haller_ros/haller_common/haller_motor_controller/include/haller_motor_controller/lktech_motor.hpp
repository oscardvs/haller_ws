#pragma once

#include "haller_motor_controller/slcan_interface.hpp"
#include <cstdint>

namespace haller_motor_controller {

/// Driver for a single LK-TECH motor using CAN protocol V2.35.
/// CAN ID = 0x140 + motor_id (motor_id is 1-based).
class LKTechMotor {
public:
  /// @param motor_id 1-based motor ID (CAN ID = 0x140 + motor_id)
  /// @param bus Shared SLCAN interface (must outlive this object)
  LKTechMotor(uint8_t motor_id, SLCANInterface &bus);

  /// Turn motor on (0x88). Enables the motor driver.
  bool motorOn();

  /// Turn motor off (0x80). Disables the motor driver, clears state.
  bool motorOff();

  /// Stop motor (0x81). Stops motion but keeps driver enabled.
  bool motorStop();

  /// Set speed in degrees per second (0xA2).
  /// The motor's internal controller runs closed-loop speed control.
  /// @param dps Target speed in degrees/second
  bool setSpeed(double dps);

  /// Read cumulative multi-turn angle (0x92).
  /// @param angle_rad Output angle in radians (cumulative, wraps after ~2^63
  /// counts)
  bool readMultiAngle(double &angle_rad);

  /// Read motor state (0x9C): temperature, torque current, speed, encoder.
  bool readState(int8_t &temp_c, int16_t &torque_current, int16_t &speed_dps,
                 uint16_t &encoder);

  /// Read error/voltage state (0x9A).
  bool readError(int8_t &temp_c, double &voltage, uint8_t &error_state);

  uint8_t motorId() const { return motor_id_; }
  uint16_t canId() const { return can_id_; }

private:
  /// Send 8-byte command and receive response.
  /// @param cmd Command byte (data[0])
  /// @param payload Bytes 4-7 (bytes 1-3 are zero-padded)
  /// @param resp Output response data (8 bytes)
  bool sendCommand(uint8_t cmd, const uint8_t *payload, uint8_t *resp);

  uint8_t motor_id_;
  uint16_t can_id_;   // 0x140 + motor_id
  SLCANInterface &bus_;
};

} // namespace haller_motor_controller
