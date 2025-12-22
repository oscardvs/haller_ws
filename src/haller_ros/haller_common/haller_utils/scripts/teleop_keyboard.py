#!/usr/bin/env python3
"""
Keyboard teleop node for the Haller robot.
Uses arrow keys or WASD for control.
"""

import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


INSTRUCTIONS = """
Haller Robot Keyboard Teleop
-----------------------------
Moving around:
   w
a  s  d

w/s : increase/decrease linear velocity
a/d : increase/decrease angular velocity
space : stop

q : quit
"""


class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')

        # Parameters
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 1.0)
        self.declare_parameter('linear_step', 0.1)
        self.declare_parameter('angular_step', 0.2)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.linear_step = self.get_parameter('linear_step').value
        self.angular_step = self.get_parameter('angular_step').value

        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Current velocities
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        self.get_logger().info('Teleop keyboard node started')

    def get_key(self):
        """Get a single keypress from terminal."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def run(self):
        """Main loop for keyboard input."""
        print(INSTRUCTIONS)
        print(f'Linear: {self.linear_vel:.2f}, Angular: {self.angular_vel:.2f}')

        try:
            while rclpy.ok():
                key = self.get_key()

                if key == 'w':
                    self.linear_vel = min(self.linear_vel + self.linear_step, self.linear_speed)
                elif key == 's':
                    self.linear_vel = max(self.linear_vel - self.linear_step, -self.linear_speed)
                elif key == 'a':
                    self.angular_vel = min(self.angular_vel + self.angular_step, self.angular_speed)
                elif key == 'd':
                    self.angular_vel = max(self.angular_vel - self.angular_step, -self.angular_speed)
                elif key == ' ':
                    self.linear_vel = 0.0
                    self.angular_vel = 0.0
                elif key == 'q' or key == '\x03':  # q or Ctrl+C
                    break

                # Publish velocity
                twist = Twist()
                twist.linear.x = self.linear_vel
                twist.angular.z = self.angular_vel
                self.cmd_vel_pub.publish(twist)

                print(f'\rLinear: {self.linear_vel:.2f}, Angular: {self.angular_vel:.2f}   ', end='')

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

        finally:
            # Stop robot
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            print('\nTeleop stopped')


def main(args=None):
    rclpy.init(args=args)
    node = TeleopKeyboard()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

