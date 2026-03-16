#include "haller_motor_controller/slcan_interface.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

namespace haller_motor_controller {

SLCANInterface::~SLCANInterface() { close(); }

bool SLCANInterface::open(const std::string &device, int bitrate) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }

  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    return false;
  }

  // Configure serial port for raw mode
  struct termios tty {};
  if (tcgetattr(fd_, &tty) != 0) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  // Raw mode - no echo, no canonical, no signal chars
  cfmakeraw(&tty);

  // CANable2 USB-serial uses its own baud rate internally;
  // set host serial to a high speed (the actual CAN bitrate is set via SLCAN
  // commands)
  cfsetispeed(&tty, B115200);
  cfsetospeed(&tty, B115200);

  // 8N1, no flow control
  tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
  tty.c_cflag |= (CS8 | CLOCAL | CREAD);

  // Non-blocking reads
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;

  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  // Flush any stale data
  tcflush(fd_, TCIOFLUSH);

  // SLCAN initialization sequence:
  // 1. Close any existing channel
  writeRaw("C\r");
  usleep(100000); // 100ms

  // Flush after close command
  tcflush(fd_, TCIFLUSH);

  // 2. Set CAN bitrate
  // S0=10k, S1=20k, S2=50k, S3=100k, S4=125k, S5=250k, S6=500k, S7=800k,
  // S8=1M
  std::string bitrate_cmd;
  switch (bitrate) {
  case 10000:
    bitrate_cmd = "S0\r";
    break;
  case 20000:
    bitrate_cmd = "S1\r";
    break;
  case 50000:
    bitrate_cmd = "S2\r";
    break;
  case 100000:
    bitrate_cmd = "S3\r";
    break;
  case 125000:
    bitrate_cmd = "S4\r";
    break;
  case 250000:
    bitrate_cmd = "S5\r";
    break;
  case 500000:
    bitrate_cmd = "S6\r";
    break;
  case 800000:
    bitrate_cmd = "S7\r";
    break;
  case 1000000:
    bitrate_cmd = "S8\r";
    break;
  default:
    bitrate_cmd = "S8\r";
    break;
  }
  writeRaw(bitrate_cmd);
  usleep(50000); // 50ms

  // 3. Open CAN channel
  writeRaw("O\r");
  usleep(200000); // 200ms

  // Flush any init responses
  tcflush(fd_, TCIFLUSH);

  return true;
}

void SLCANInterface::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (fd_ >= 0) {
    writeRaw("C\r");
    usleep(50000);
    ::close(fd_);
    fd_ = -1;
  }
}

bool SLCANInterface::isOpen() const { return fd_ >= 0; }

bool SLCANInterface::send(uint16_t can_id, const uint8_t *data, uint8_t len) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (fd_ < 0 || len > 8) {
    return false;
  }

  // Build SLCAN frame: t{ID:03X}{DLC}{DATA_HEX}\r
  char frame[32];
  int pos =
      snprintf(frame, sizeof(frame), "t%03X%d", can_id, static_cast<int>(len));
  for (int i = 0; i < len; i++) {
    pos += snprintf(frame + pos, sizeof(frame) - pos, "%02X", data[i]);
  }
  frame[pos++] = '\r';
  frame[pos] = '\0';

  ssize_t written = ::write(fd_, frame, pos);
  return written == pos;
}

bool SLCANInterface::sendAndReceive(uint16_t can_id, const uint8_t *data,
                                    uint8_t len, uint16_t resp_id,
                                    uint8_t resp_cmd, uint8_t *resp_data,
                                    int timeout_ms) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (fd_ < 0 || len > 8) {
    return false;
  }

  // Flush input before sending
  tcflush(fd_, TCIFLUSH);

  // Build and send SLCAN frame
  char frame[32];
  int pos =
      snprintf(frame, sizeof(frame), "t%03X%d", can_id, static_cast<int>(len));
  for (int i = 0; i < len; i++) {
    pos += snprintf(frame + pos, sizeof(frame) - pos, "%02X", data[i]);
  }
  frame[pos++] = '\r';
  frame[pos] = '\0';

  ssize_t written = ::write(fd_, frame, pos);
  if (written != pos) {
    return false;
  }

  // Read response with timeout
  // Accumulate data and look for matching response frame
  char read_buf[512];
  std::string accum;
  int elapsed_ms = 0;
  const int poll_interval_ms = 5;

  while (elapsed_ms < timeout_ms) {
    int n = readWithTimeout(read_buf, sizeof(read_buf) - 1, poll_interval_ms);
    if (n > 0) {
      accum.append(read_buf, n);

      // Try to parse complete frames (delimited by \r)
      size_t start = 0;
      while (true) {
        size_t cr = accum.find('\r', start);
        if (cr == std::string::npos) {
          break;
        }

        std::string one_frame = accum.substr(start, cr - start);
        start = cr + 1;

        uint16_t rx_id = 0;
        uint8_t rx_data[8] = {};
        uint8_t rx_dlc = 0;

        if (parseFrame(one_frame, rx_id, rx_data, rx_dlc)) {
          if (rx_id == resp_id && rx_dlc > 0 && rx_data[0] == resp_cmd) {
            memcpy(resp_data, rx_data, rx_dlc);
            return true;
          }
        }
      }

      // Keep unparsed tail
      if (start > 0) {
        accum = accum.substr(start);
      }
    }
    elapsed_ms += poll_interval_ms;
  }

  return false;
}

bool SLCANInterface::parseFrame(const std::string &frame, uint16_t &id,
                                uint8_t *data, uint8_t &dlc) {
  // Standard frame format: t{ID:3hex}{DLC:1}{DATA:2*DLC hex}
  if (frame.size() < 5 || frame[0] != 't') {
    return false;
  }

  // Parse 3-digit hex CAN ID
  char *end = nullptr;
  unsigned long parsed_id = strtoul(frame.substr(1, 3).c_str(), &end, 16);
  if (end == nullptr || *end != '\0') {
    return false;
  }
  id = static_cast<uint16_t>(parsed_id);

  // Parse DLC
  int parsed_dlc = frame[4] - '0';
  if (parsed_dlc < 0 || parsed_dlc > 8) {
    return false;
  }
  dlc = static_cast<uint8_t>(parsed_dlc);

  // Check we have enough hex chars for data
  if (frame.size() < 5 + dlc * 2u) {
    return false;
  }

  // Parse data bytes
  for (int i = 0; i < dlc; i++) {
    unsigned long byte_val =
        strtoul(frame.substr(5 + i * 2, 2).c_str(), &end, 16);
    if (end == nullptr || *end != '\0') {
      return false;
    }
    data[i] = static_cast<uint8_t>(byte_val);
  }

  return true;
}

bool SLCANInterface::writeRaw(const std::string &s) {
  if (fd_ < 0) {
    return false;
  }
  ssize_t written = ::write(fd_, s.c_str(), s.size());
  return written == static_cast<ssize_t>(s.size());
}

int SLCANInterface::readWithTimeout(char *buf, int max_len, int timeout_ms) {
  struct pollfd pfd {};
  pfd.fd = fd_;
  pfd.events = POLLIN;

  int ret = poll(&pfd, 1, timeout_ms);
  if (ret > 0 && (pfd.revents & POLLIN)) {
    ssize_t n = ::read(fd_, buf, max_len);
    if (n > 0) {
      return static_cast<int>(n);
    }
  }
  return 0;
}

} // namespace haller_motor_controller
