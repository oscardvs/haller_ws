#!/usr/bin/env python3
"""
Scan processor node for the Haller robot.
Filters and processes laser scan data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class ScanProcessor(Node):
    def __init__(self):
        super().__init__('scan_processor')

        # Parameters
        self.declare_parameter('min_range', 0.15)
        self.declare_parameter('max_range', 12.0)
        self.declare_parameter('filter_window', 3)

        self.min_range = self.get_parameter('min_range').value
        self.max_range = self.get_parameter('max_range').value
        self.filter_window = self.get_parameter('filter_window').value

        # Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        # Publisher
        self.filtered_scan_pub = self.create_publisher(
            LaserScan,
            'scan_filtered',
            10
        )

        self.get_logger().info('Scan processor started')

    def scan_callback(self, msg: LaserScan):
        """Process incoming scan data."""
        filtered_msg = LaserScan()
        filtered_msg.header = msg.header
        filtered_msg.angle_min = msg.angle_min
        filtered_msg.angle_max = msg.angle_max
        filtered_msg.angle_increment = msg.angle_increment
        filtered_msg.time_increment = msg.time_increment
        filtered_msg.scan_time = msg.scan_time
        filtered_msg.range_min = self.min_range
        filtered_msg.range_max = self.max_range

        # Convert to numpy array for processing
        ranges = np.array(msg.ranges)

        # Replace inf and nan with max_range
        ranges = np.where(np.isinf(ranges) | np.isnan(ranges), self.max_range, ranges)

        # Clip to valid range
        ranges = np.clip(ranges, self.min_range, self.max_range)

        # Apply median filter
        if self.filter_window > 1:
            ranges = self.median_filter(ranges, self.filter_window)

        filtered_msg.ranges = ranges.tolist()
        filtered_msg.intensities = list(msg.intensities) if msg.intensities else []

        self.filtered_scan_pub.publish(filtered_msg)

    def median_filter(self, data: np.ndarray, window: int) -> np.ndarray:
        """Apply median filter to scan data."""
        filtered = np.zeros_like(data)
        half_window = window // 2

        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            filtered[i] = np.median(data[start:end])

        return filtered


def main(args=None):
    rclpy.init(args=args)
    node = ScanProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

