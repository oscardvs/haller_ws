# Overall robot status message

std_msgs/Header header

# Motor status
bool motors_enabled
bool motor_fault

# System status
uint8 system_state
uint8 STATE_IDLE = 0
uint8 STATE_RUNNING = 1
uint8 STATE_EMERGENCY_STOP = 2
uint8 STATE_ERROR = 3

# Sensor status
bool lidar_ok
bool camera_ok
bool imu_ok

# Error codes
uint32 error_code
string error_message

