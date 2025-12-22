#!/usr/bin/env python3
"""
Battery monitoring node for the Haller robot.
Publishes battery status and warnings.
"""

import rclpy
from rclpy.node import Node
from haller_msgs.msg import BatteryState


class BatteryMonitor(Node):
    def __init__(self):
        super().__init__('battery_monitor')

        # Parameters
        self.declare_parameter('low_battery_threshold', 20.0)
        self.declare_parameter('critical_battery_threshold', 10.0)
        self.declare_parameter('publish_rate', 1.0)

        self.low_threshold = self.get_parameter('low_battery_threshold').value
        self.critical_threshold = self.get_parameter('critical_battery_threshold').value
        publish_rate = self.get_parameter('publish_rate').value

        # Publisher
        self.battery_pub = self.create_publisher(
            BatteryState,
            'battery_state',
            10
        )

        # Timer
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)

        self.get_logger().info('Battery monitor started')

    def timer_callback(self):
        """Read battery status and publish."""
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()

        # TODO: Read actual battery data from hardware
        # For now, use placeholder values
        msg.voltage = 12.6
        msg.current = 0.5
        msg.percentage = 85.0
        msg.temperature = 25.0
        msg.is_charging = False

        # Check for low battery
        msg.low_battery_warning = msg.percentage < self.low_threshold

        if msg.percentage < self.critical_threshold:
            self.get_logger().warn('CRITICAL: Battery level critical!')
        elif msg.low_battery_warning:
            self.get_logger().warn('Battery level low')

        self.battery_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BatteryMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

