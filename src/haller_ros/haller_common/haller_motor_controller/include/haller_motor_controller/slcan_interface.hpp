#pragma once

#include <cstdint>
#include <mutex>
#include <string>

namespace haller_motor_controller {

/// Low-level SLCAN ASCII protocol over POSIX serial.
/// Thread-safe: all bus operations are mutex-protected.
class SLCANInterface {
public:
  SLCANInterface() = default;
  ~SLCANInterface();

  SLCANInterface(const SLCANInterface &) = delete;
  SLCANInterface &operator=(const SLCANInterface &) = delete;

  /// Open serial device and initialize SLCAN (C, S8, O commands).
  /// @param device Serial device path (e.g. "/dev/ttyACM0")
  /// @param bitrate CAN bitrate code: 1000000 → S8
  /// @return true on success
  bool open(const std::string &device, int bitrate);

  /// Close SLCAN channel and serial port.
  void close();

  /// @return true if the serial port is open
  bool isOpen() const;

  /// Send a CAN frame (fire-and-forget).
  bool send(uint16_t can_id, const uint8_t *data, uint8_t len);

  /// Send a CAN frame and wait for a response matching resp_id and resp_cmd.
  /// @param resp_data Output buffer, must be at least 8 bytes
  /// @param timeout_ms Read timeout in milliseconds
  /// @return true if a matching response was received
  bool sendAndReceive(uint16_t can_id, const uint8_t *data, uint8_t len,
                      uint16_t resp_id, uint8_t resp_cmd, uint8_t *resp_data,
                      int timeout_ms = 50);

private:
  /// Parse one SLCAN ASCII frame into CAN ID + data.
  /// @return true if parsing succeeded
  bool parseFrame(const std::string &frame, uint16_t &id, uint8_t *data,
                  uint8_t &dlc);

  /// Send a raw string to the serial port.
  bool writeRaw(const std::string &s);

  /// Read available data from serial port with timeout.
  /// @return bytes read into buf
  int readWithTimeout(char *buf, int max_len, int timeout_ms);

  int fd_ = -1;
  std::mutex mutex_;
};

} // namespace haller_motor_controller
