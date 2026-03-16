#!/usr/bin/env python3
"""
LK-TECH Motor Identification Utility

Standalone script (not a ROS node) to identify and test motors on the CAN bus.
Scans for motors, reads their state, and helps verify which physical motor
corresponds to which CAN ID.

Usage:
    python3 set_motor_id.py                    # Scan for all motors
    python3 set_motor_id.py --spin 1           # Briefly spin motor ID 1 to identify it
    python3 set_motor_id.py --device /dev/ttyACM1  # Use alternate serial device

Note: To actually CHANGE a motor's CAN ID, use the LK-TECH motor tool software.
This script only identifies/tests motors.
"""

import argparse
import serial
import struct
import sys
import time


DEVICE_DEFAULT = '/dev/ttyACM0'
CAN_BASE_ID = 0x140


def parse_frames(raw):
    """Parse SLCAN ASCII frames from raw serial data."""
    frames = []
    for part in raw.decode(errors='ignore').split('\r'):
        part = part.strip()
        if part.startswith('t') and len(part) >= 5:
            try:
                can_id = int(part[1:4], 16)
                dlc = int(part[4])
                if len(part) >= 5 + dlc * 2:
                    data = bytes.fromhex(part[5:5 + dlc * 2])
                    frames.append((can_id, data))
            except (ValueError, IndexError):
                pass
    return frames


def send_cmd(ser, can_id, data, timeout=0.15):
    """Send CAN command and return matching response frames."""
    ser.reset_input_buffer()
    hex_data = ''.join(f'{b:02X}' for b in data)
    frame = f"t{can_id:03X}{len(data)}{hex_data}"
    ser.write(frame.encode() + b'\r')
    time.sleep(timeout)
    raw = ser.read(4096)
    frames = parse_frames(raw)

    cmd = data[0]
    matched = [(cid, d) for cid, d in frames if len(d) > 0 and d[0] == cmd]
    return matched


def init_slcan(ser):
    """Initialize SLCAN interface."""
    ser.write(b'C\r')
    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write(b'S8\r')
    time.sleep(0.1)
    ser.write(b'O\r')
    time.sleep(0.3)
    ser.reset_input_buffer()


def close_slcan(ser):
    """Close SLCAN interface."""
    ser.write(b'C\r')
    time.sleep(0.1)
    ser.close()


def scan_motors(ser, id_range=range(1, 33)):
    """Scan for motors on the CAN bus."""
    found = []
    for motor_id in id_range:
        can_id = CAN_BASE_ID + motor_id
        responses = send_cmd(ser, can_id, [0x9C, 0, 0, 0, 0, 0, 0, 0], timeout=0.1)
        if responses:
            _, data = responses[0]
            temp = struct.unpack('b', bytes([data[1]]))[0]
            speed = struct.unpack('<h', data[4:6])[0]
            print(f"  Motor ID {motor_id} (CAN 0x{can_id:03X}): "
                  f"temp={temp}°C, speed={speed} dps")
            found.append(motor_id)
    return found


def read_motor_info(ser, motor_id):
    """Read detailed motor information."""
    can_id = CAN_BASE_ID + motor_id
    print(f"\n--- Motor ID {motor_id} (CAN 0x{can_id:03X}) ---")

    # Read error/voltage state (0x9A)
    responses = send_cmd(ser, can_id, [0x9A, 0, 0, 0, 0, 0, 0, 0])
    if responses:
        _, data = responses[0]
        temp = struct.unpack('b', bytes([data[1]]))[0]
        voltage = struct.unpack('<H', data[3:5])[0] * 0.1
        error = data[7]
        print(f"  Temperature: {temp}°C")
        print(f"  Voltage: {voltage:.1f}V")
        print(f"  Error state: 0x{error:02X} ({'OK' if error == 0 else 'ERROR'})")
    else:
        print("  Failed to read error state")

    # Read motor state (0x9C)
    responses = send_cmd(ser, can_id, [0x9C, 0, 0, 0, 0, 0, 0, 0])
    if responses:
        _, data = responses[0]
        temp = struct.unpack('b', bytes([data[1]]))[0]
        torque = struct.unpack('<h', data[2:4])[0]
        speed = struct.unpack('<h', data[4:6])[0]
        encoder = struct.unpack('<H', data[6:8])[0]
        print(f"  Torque current: {torque}")
        print(f"  Speed: {speed} dps")
        print(f"  Encoder: {encoder}")

    # Read multi-angle (0x92)
    responses = send_cmd(ser, can_id, [0x92, 0, 0, 0, 0, 0, 0, 0])
    if responses:
        _, data = responses[0]
        raw = int.from_bytes(data[1:8], byteorder='little', signed=True)
        angle_deg = raw * 0.01
        print(f"  Multi-angle: {angle_deg:.2f}°")


def spin_motor(ser, motor_id, speed_dps=180, duration=1.0):
    """Briefly spin a motor for physical identification."""
    can_id = CAN_BASE_ID + motor_id
    print(f"\nSpinning motor ID {motor_id} at {speed_dps} dps for {duration}s...")

    # Motor ON
    send_cmd(ser, can_id, [0x88, 0, 0, 0, 0, 0, 0, 0])
    time.sleep(0.2)

    # Speed command
    speed_val = int(speed_dps * 100)
    spd_bytes = list(struct.pack('<i', speed_val))
    send_cmd(ser, can_id, [0xA2, 0, 0, 0] + spd_bytes)
    time.sleep(duration)

    # Stop
    send_cmd(ser, can_id, [0x81, 0, 0, 0, 0, 0, 0, 0])
    time.sleep(0.2)

    # Motor OFF
    send_cmd(ser, can_id, [0x80, 0, 0, 0, 0, 0, 0, 0])
    print("  Done.")


def main():
    parser = argparse.ArgumentParser(
        description='LK-TECH Motor Identification Utility')
    parser.add_argument('--device', default=DEVICE_DEFAULT,
                        help=f'Serial device (default: {DEVICE_DEFAULT})')
    parser.add_argument('--spin', type=int, default=None,
                        help='Spin motor with given ID briefly for identification')
    parser.add_argument('--speed', type=float, default=180.0,
                        help='Spin speed in dps (default: 180)')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Spin duration in seconds (default: 1.0)')
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.device, baudrate=115200, timeout=0.1)
    except serial.SerialException as e:
        print(f"Error opening {args.device}: {e}", file=sys.stderr)
        sys.exit(1)

    time.sleep(0.2)
    init_slcan(ser)

    print("=" * 50)
    print("  LK-TECH Motor Identification Utility")
    print("=" * 50)

    if args.spin is not None:
        # Spin a specific motor for identification
        spin_motor(ser, args.spin, args.speed, args.duration)
    else:
        # Scan for all motors
        print("\nScanning for motors (ID 1-32)...")
        found = scan_motors(ser)

        if not found:
            print("\n  No motors found! Check wiring and power.")
        else:
            print(f"\nFound {len(found)} motor(s). Reading details...")
            for motor_id in found:
                read_motor_info(ser, motor_id)

            print(f"\nTo identify a motor physically, run:")
            for motor_id in found:
                print(f"  python3 {sys.argv[0]} --spin {motor_id}")

    close_slcan(ser)
    print("\nDone.")


if __name__ == '__main__':
    main()
