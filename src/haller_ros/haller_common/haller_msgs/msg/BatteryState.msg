# Battery status message

std_msgs/Header header

float32 voltage          # Battery voltage in volts
float32 current          # Current draw in amps (positive = discharging)
float32 percentage       # State of charge (0.0 - 100.0)
float32 temperature      # Battery temperature in Celsius

bool is_charging
bool low_battery_warning

