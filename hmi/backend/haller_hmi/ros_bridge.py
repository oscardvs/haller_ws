# hmi/backend/haller_hmi/ros_bridge.py
"""ROS 2 bridge that runs in a background thread.

Responsibilities:
  - publish geometry_msgs/Twist on /cmd_vel from POST /base/cmd_vel
  - subscribe to /odom and /scan, expose latest snapshot for telemetry frames
  - keep the rclpy executor spinning without blocking the FastAPI event loop

The executor lives on a dedicated thread. `latest()` is a thread-safe read of the
last odom and scan messages (held as plain dicts so the JSON serializer is happy).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from .config import RosConfig

logger = logging.getLogger(__name__)


@dataclass
class BaseSnapshot:
    linear: float = 0.0
    angular: float = 0.0
    odom: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "yaw": 0.0})
    scan_min_range: float | None = None


class _HmiNode(Node):
    def __init__(self, cfg: RosConfig, snap: BaseSnapshot, lock: threading.Lock):
        super().__init__("haller_hmi")
        self._cfg = cfg
        self._snap = snap
        self._lock = lock
        self._pub = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self.create_subscription(Odometry, cfg.odom_topic, self._on_odom, 10)
        self.create_subscription(LaserScan, cfg.scan_topic, self._on_scan, 10)

    def publish_cmd_vel(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._pub.publish(msg)
        with self._lock:
            self._snap.linear = float(linear)
            self._snap.angular = float(angular)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # quaternion → yaw (small util to avoid pulling tf_transformations)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        import math
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self._lock:
            self._snap.odom = {"x": float(p.x), "y": float(p.y), "yaw": float(yaw)}

    def _on_scan(self, msg: LaserScan) -> None:
        # min finite range; +inf if all infinite
        rng = [r for r in msg.ranges if (r > 0.0 and r < float("inf"))]
        with self._lock:
            self._snap.scan_min_range = float(min(rng)) if rng else None


class RosBridge:
    def __init__(self, cfg: RosConfig):
        self._cfg = cfg
        self._snap = BaseSnapshot()
        self._lock = threading.Lock()
        self._node: _HmiNode | None = None
        self._exec: SingleThreadedExecutor | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        rclpy.init(args=None)
        self._node = _HmiNode(self._cfg, self._snap, self._lock)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, name="haller-hmi-ros", daemon=True)
        self._thread.start()
        logger.info("ROS bridge started; node=%s", self._node.get_name())

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._exec is not None:
            self._exec.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        # Idempotent on purpose. rclpy installs its own SIGINT/SIGTERM handlers
        # and shuts the global context down itself, so on a signalled exit
        # rcl_shutdown has usually already run by the time the FastAPI lifespan
        # reaches here, and a second call raises:
        #
        #   RCLError: failed to shutdown: rcl_shutdown already called
        #
        # Uncaught, that turns every normal Ctrl-C/SIGTERM shutdown into
        # "ERROR: Application shutdown failed" — which is not just noise, it
        # makes a REAL shutdown failure (one that skipped closing the dataset
        # or de-energising the arms) indistinguishable from the routine case.
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError as e:  # RCLError subclasses RuntimeError
            logger.debug("rclpy already shut down: %s", e)

    def _spin(self) -> None:
        assert self._exec is not None
        while not self._stop.is_set():
            try:
                self._exec.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                # rclpy's signal handler tore the global context down out from
                # under this thread. That is the normal way a signalled exit
                # looks from in here, not a fault — returning quietly beats
                # dumping a traceback into every shutdown log.
                logger.debug("ROS context shut down externally; spin thread exiting")
                return

    # public API used by FastAPI routes / telemetry

    def publish_cmd_vel(self, linear: float, angular: float) -> tuple[float, float]:
        # clamp to configured maxes
        linear = max(-self._cfg.max_linear, min(self._cfg.max_linear, float(linear)))
        angular = max(-self._cfg.max_angular, min(self._cfg.max_angular, float(angular)))
        assert self._node is not None
        self._node.publish_cmd_vel(linear, angular)
        return (linear, angular)

    def zero_cmd_vel(self) -> None:
        self.publish_cmd_vel(0.0, 0.0)

    def snapshot(self) -> BaseSnapshot:
        with self._lock:
            # return a copy so the reader can't see torn state
            return BaseSnapshot(
                linear=self._snap.linear,
                angular=self._snap.angular,
                odom=dict(self._snap.odom),
                scan_min_range=self._snap.scan_min_range,
            )
