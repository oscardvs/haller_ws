#include "haller_motor_controller/lktech_motor.hpp"
#include <cmath>
#include <cstring>

namespace haller_motor_controller {

LKTechMotor::LKTechMotor(uint8_t motor_id, SLCANInterface &bus)
    : motor_id_(motor_id), can_id_(0x140 + motor_id), bus_(bus) {}

bool LKTechMotor::sendCommand(uint8_t cmd, const uint8_t *payload,
                              uint8_t *resp) {
  uint8_t data[8] = {};
  data[0] = cmd;
  // data[1..3] = 0 (reserved)
  if (payload) {
    data[4] = payload[0];
    data[5] = payload[1];
    data[6] = payload[2];
    data[7] = payload[3];
  }

  // Response comes back on same CAN ID with same command byte
  return bus_.sendAndReceive(can_id_, data, 8, can_id_, cmd, resp, 50);
}

bool LKTechMotor::motorOn() {
  uint8_t resp[8] = {};
  return sendCommand(0x88, nullptr, resp);
}

bool LKTechMotor::motorOff() {
  uint8_t resp[8] = {};
  return sendCommand(0x80, nullptr, resp);
}

bool LKTechMotor::motorStop() {
  uint8_t resp[8] = {};
  return sendCommand(0x81, nullptr, resp);
}

bool LKTechMotor::setSpeed(double dps) {
  // Speed control command 0xA2
  // Bytes 4-7: int32_t speed in 0.01 dps/LSB (little-endian)
  int32_t speed_val = static_cast<int32_t>(dps * 100.0);
  uint8_t payload[4];
  memcpy(payload, &speed_val, 4); // little-endian on ARM

  uint8_t resp[8] = {};
  return sendCommand(0xA2, payload, resp);
}

bool LKTechMotor::readMultiAngle(double &angle_rad) {
  // Multi-turn angle command 0x92
  // Response bytes 1-7: int64_t angle in 0.01 degrees/LSB (7 bytes, byte 0 is
  // cmd)
  uint8_t data[8] = {0x92, 0, 0, 0, 0, 0, 0, 0};
  uint8_t resp[8] = {};

  if (!bus_.sendAndReceive(can_id_, data, 8, can_id_, 0x92, resp, 50)) {
    return false;
  }

  // Response format: resp[0]=0x92, resp[1..7] = 7 bytes of int64 angle
  // The angle is a 7-byte signed integer (bytes 1-7), with byte 7 being MSB
  int64_t raw_angle = 0;
  memcpy(&raw_angle, &resp[1], 7);
  // Sign-extend from 56 bits
  if (raw_angle & (1LL << 55)) {
    raw_angle |= static_cast<int64_t>(0xFF) << 56;
  }

  // Convert: 0.01 degrees → radians
  angle_rad = (static_cast<double>(raw_angle) * 0.01) * M_PI / 180.0;
  return true;
}

bool LKTechMotor::readState(int8_t &temp_c, int16_t &torque_current,
                            int16_t &speed_dps, uint16_t &encoder) {
  uint8_t resp[8] = {};
  if (!sendCommand(0x9C, nullptr, resp)) {
    return false;
  }

  // Response: [0]=cmd, [1]=temp, [2-3]=torque(int16), [4-5]=speed(int16),
  // [6-7]=encoder(uint16)
  temp_c = static_cast<int8_t>(resp[1]);
  memcpy(&torque_current, &resp[2], 2);
  memcpy(&speed_dps, &resp[4], 2);
  memcpy(&encoder, &resp[6], 2);
  return true;
}

bool LKTechMotor::readError(int8_t &temp_c, double &voltage,
                            uint8_t &error_state) {
  uint8_t resp[8] = {};
  if (!sendCommand(0x9A, nullptr, resp)) {
    return false;
  }

  // Response: [0]=cmd, [1]=temp, [2]=0, [3-4]=voltage(uint16, 0.1V/LSB),
  // [5-6]=0, [7]=error
  temp_c = static_cast<int8_t>(resp[1]);
  uint16_t raw_voltage;
  memcpy(&raw_voltage, &resp[3], 2);
  voltage = raw_voltage * 0.1;
  error_state = resp[7];
  return true;
}

} // namespace haller_motor_controller
