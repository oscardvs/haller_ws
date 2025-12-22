#!/usr/bin/env python3
"""
Simple obstacle detector node for the Haller robot.
Detects nearby obstacles and publishes warnings.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
import numpy as np


class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')

        # Parameters
        self.declare_parameter('warning_distance', 0.5)
        self.declare_parameter('critical_distance', 0.3)
        self.declare_parameter('front_angle', 1.0)  # radians

        self.warning_distance = self.get_parameter('warning_distance').value
        self.critical_distance = self.get_parameter('critical_distance').value
        self.front_angle = self.get_parameter('front_angle').value

        # Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        # Publishers
        self.obstacle_warning_pub = self.create_publisher(
            Bool,
            'obstacle_warning',
            10
        )
        self.obstacle_critical_pub = self.create_publisher(
            Bool,
            'obstacle_critical',
            10
        )
        self.closest_obstacle_pub = self.create_publisher(
            Vector3,
            'closest_obstacle',
            10
        )

        self.get_logger().info('Obstacle detector started')

    def scan_callback(self, msg: LaserScan):
        """Process scan and detect obstacles."""
        ranges = np.array(msg.ranges)
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)

        # Handle array size mismatch
        if len(angles) > len(ranges):
            angles = angles[:len(ranges)]
        elif len(ranges) > len(angles):
            ranges = ranges[:len(angles)]

        # Replace inf/nan
        valid_mask = ~(np.isinf(ranges) | np.isnan(ranges))
        ranges = np.where(valid_mask, ranges, msg.range_max)

        # Find minimum range in front sector
        front_mask = np.abs(angles) < self.front_angle
        front_ranges = ranges[front_mask]
        front_angles = angles[front_mask]

        if len(front_ranges) > 0:
            min_idx = np.argmin(front_ranges)
            min_range = front_ranges[min_idx]
            min_angle = front_angles[min_idx]

            # Publish closest obstacle position (in robot frame)
            closest = Vector3()
            closest.x = float(min_range * np.cos(min_angle))
            closest.y = float(min_range * np.sin(min_angle))
            closest.z = 0.0
            self.closest_obstacle_pub.publish(closest)

            # Check warning/critical thresholds
            warning = Bool()
            warning.data = bool(min_range < self.warning_distance)
            self.obstacle_warning_pub.publish(warning)

            critical = Bool()
            critical.data = bool(min_range < self.critical_distance)
            self.obstacle_critical_pub.publish(critical)

            if critical.data:
                self.get_logger().warn(f'CRITICAL: Obstacle at {min_range:.2f}m!')
            elif warning.data:
                self.get_logger().info(f'Warning: Obstacle at {min_range:.2f}m')


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

