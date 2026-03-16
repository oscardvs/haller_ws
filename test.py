                                                                                       
import can

bus = can.interface.Bus(channel='/dev/ttyACM0', interface='slcan', bitrate=1000000)

print('Scanning for motors...')
for i in range(0x141, 0x161):
    bus.send(can.Message(arbitration_id=i, data=[0x9C,0,0,0,0,0,0,0]))
    resp = bus.recv(timeout=0.1)
    if resp:
        print(f'Motor found at ID 0x{i:03X} (motor #{i - 0x140}): {resp}')

bus.shutdown()
print('Scan complete.')
