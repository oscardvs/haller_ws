#!/usr/bin/env python3
"""LK-TECH Motor Test - no drain loop, fixed"""
import serial
import time
import struct

DEVICE = '/dev/ttyACM0'
MOTOR_ID = 0x141

ser = serial.Serial(DEVICE, baudrate=115200, timeout=0.1)
time.sleep(0.2)

def parse_frames(raw):
    frames = []
    for part in raw.decode(errors='ignore').split('\r'):
        part = part.strip()
        if part.startswith('t') and len(part) >= 21:
            try:
                can_id = int(part[1:4], 16)
                data = bytes.fromhex(part[5:5 + int(part[4]) * 2])
                frames.append((can_id, data))
            except:
                pass
    return frames

def send(data, label=""):
    ser.reset_input_buffer()  # just one reset, no loop
    hex_data = ''.join(f'{b:02X}' for b in data)
    frame = f"t{MOTOR_ID:03X}{len(data)}{hex_data}"
    ser.write(frame.encode() + b'\r')
    time.sleep(0.15)
    raw = ser.read(4096)  # read fixed chunk, don't wait for all
    frames = parse_frames(raw)

    # Find response matching our command byte
    cmd = data[0]
    matched = [d for _, d in frames if d[0] == cmd]
    other = [d for _, d in frames if d[0] != cmd]

    print(f"  {label}: got {len(frames)} frames, {len(matched)} match cmd 0x{cmd:02X}")
    for d in matched[:3]:
        print(f"    -> {' '.join(f'{b:02X}' for b in d)}", end="")
        if cmd in (0x9C, 0xA2, 0xA1, 0xA0):
            temp = struct.unpack('b', bytes([d[1]]))[0]
            iq = struct.unpack('<h', d[2:4])[0]
            spd = struct.unpack('<h', d[4:6])[0]
            enc = struct.unpack('<H', d[6:8])[0]
            print(f"  (T={temp}°C iq={iq} spd={spd}dps enc={enc})", end="")
        if cmd == 0x9A:
            temp = struct.unpack('b', bytes([d[1]]))[0]
            volt = struct.unpack('<H', d[3:5])[0] * 0.1
            err = d[7]
            print(f"  (T={temp}°C V={volt:.1f}V err=0x{err:02X})", end="")
        print()
    if other[:1]:
        print(f"    (also got cmd 0x{other[0][0]:02X} ...)")
    return frames

# Init SLCAN
ser.write(b'C\r')
time.sleep(0.3)
ser.reset_input_buffer()
ser.write(b'S8\r')
time.sleep(0.1)
ser.write(b'O\r')
time.sleep(0.5)
ser.reset_input_buffer()

print("=" * 55)
print("  LK-TECH Motor Test")
print("=" * 55)

# Check state
print("\n[1] Read State:")
send([0x9A, 0, 0, 0, 0, 0, 0, 0], "Error state")
time.sleep(0.3)
send([0x9C, 0, 0, 0, 0, 0, 0, 0], "Motor state")
time.sleep(0.3)

# Motor ON
print("\n[2] Motor ON:")
send([0x88, 0, 0, 0, 0, 0, 0, 0], "Motor ON")
time.sleep(1)

# Spin forward 360 dps
print("\n[3] FORWARD 360 dps:")
spd = list(struct.pack('<i', 36000))
send([0xA2, 0, 0, 0] + spd, "Speed fwd")
for i in range(3):
    time.sleep(1)
    send([0x9C, 0, 0, 0, 0, 0, 0, 0], f"State t+{i+1}s")

# Stop
print("\n[4] STOP:")
send([0x81, 0, 0, 0, 0, 0, 0, 0], "Stop")
time.sleep(1)

# Spin reverse
print("\n[5] REVERSE 360 dps:")
spd = list(struct.pack('<i', -36000))
send([0xA2, 0, 0, 0] + spd, "Speed rev")
for i in range(3):
    time.sleep(1)
    send([0x9C, 0, 0, 0, 0, 0, 0, 0], f"State t+{i+1}s")

# Stop + OFF
print("\n[6] STOP + OFF:")
send([0x81, 0, 0, 0, 0, 0, 0, 0], "Stop")
time.sleep(0.3)
send([0x80, 0, 0, 0, 0, 0, 0, 0], "OFF")

ser.write(b'C\r')
ser.close()
print("\nDone!")
