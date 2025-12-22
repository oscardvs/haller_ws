#!/usr/bin/env python3
"""
Traversability Analysis Node

Combines segmentation and detection outputs to generate a traversability costmap
for Nav2 navigation. Projects camera-based traversability analysis to a 2D costmap.

This node is hardware-agnostic - works with both real camera and Gazebo simulation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional, Tuple
import math
import tf2_ros


class TraversabilityNode(Node):
    """Generates traversability costmap from vision inputs."""
    
    def __init__(self):
        super().__init__('traversability_node')
        
        # Declare parameters
        self.declare_parameter('segmentation_topic', '/segmentation/traversability')
        self.declare_parameter('detection_topic', '/detections')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('costmap_topic', '/traversability/costmap')
        self.declare_parameter('annotated_topic', '/camera/traversability_annotated')
        
        # Costmap parameters
        self.declare_parameter('costmap_resolution', 0.05)
        self.declare_parameter('costmap_width', 10.0)
        self.declare_parameter('costmap_height', 10.0)
        self.declare_parameter('costmap_frame', 'base_link')
        
        # Cost values
        self.declare_parameter('free_cost', 0)
        self.declare_parameter('inscribed_cost', 253)
        self.declare_parameter('lethal_cost', 254)
        self.declare_parameter('unknown_cost', -1)
        
        # Detection inflation
        self.declare_parameter('detection_inflation_radius', 0.5)
        self.declare_parameter('person_safety_radius', 1.0)
        
        # Camera parameters
        self.declare_parameter('camera_height', 0.15)
        self.declare_parameter('camera_pitch', 0.0)
        self.declare_parameter('camera_fov_h', 2.0944)  # 120 degrees
        
        # Temporal filtering
        self.declare_parameter('costmap_decay_time', 2.0)
        self.declare_parameter('use_temporal_filter', True)
        
        # Publishing rate
        self.declare_parameter('publish_rate', 10.0)
        
        # Get parameters
        self.costmap_resolution = self.get_parameter('costmap_resolution').value
        self.costmap_width = self.get_parameter('costmap_width').value
        self.costmap_height = self.get_parameter('costmap_height').value
        self.costmap_frame = self.get_parameter('costmap_frame').value
        
        self.free_cost = self.get_parameter('free_cost').value
        self.lethal_cost = self.get_parameter('lethal_cost').value
        self.unknown_cost = self.get_parameter('unknown_cost').value
        
        self.detection_inflation = self.get_parameter('detection_inflation_radius').value
        self.person_safety = self.get_parameter('person_safety_radius').value
        
        self.camera_height = self.get_parameter('camera_height').value
        self.camera_pitch = self.get_parameter('camera_pitch').value
        self.camera_fov_h = self.get_parameter('camera_fov_h').value
        
        self.decay_time = self.get_parameter('costmap_decay_time').value
        self.use_temporal = self.get_parameter('use_temporal_filter').value
        
        # Calculate costmap dimensions in cells
        self.costmap_width_cells = int(self.costmap_width / self.costmap_resolution)
        self.costmap_height_cells = int(self.costmap_height / self.costmap_resolution)
        
        # Initialize costmap data
        self.costmap_data = np.full(
            (self.costmap_height_cells, self.costmap_width_cells),
            self.unknown_cost,
            dtype=np.int8
        )
        self.last_update_time = np.zeros(
            (self.costmap_height_cells, self.costmap_width_cells)
        )
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera info storage
        self.camera_info: Optional[CameraInfo] = None
        self.camera_matrix: Optional[np.ndarray] = None
        
        # Latest inputs
        self.latest_seg_mask: Optional[np.ndarray] = None
        self.latest_detections: Optional[Detection2DArray] = None
        
        # TF2 buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.seg_sub = self.create_subscription(
            Image,
            self.get_parameter('segmentation_topic').value,
            self.segmentation_callback,
            sensor_qos
        )
        
        self.det_sub = self.create_subscription(
            Detection2DArray,
            self.get_parameter('detection_topic').value,
            self.detection_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            sensor_qos
        )
        
        # Publishers
        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            self.get_parameter('costmap_topic').value,
            reliable_qos
        )
        
        # Timer for publishing costmap
        publish_rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_costmap)
        
        self.get_logger().info("Traversability node initialized")
        self.get_logger().info(f"  Costmap size: {self.costmap_width}x{self.costmap_height}m")
        self.get_logger().info(f"  Resolution: {self.costmap_resolution}m/cell")
        self.get_logger().info(f"  Grid size: {self.costmap_width_cells}x{self.costmap_height_cells}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics."""
        self.camera_info = msg
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
    
    def segmentation_callback(self, msg: Image):
        """Process segmentation mask."""
        try:
            self.latest_seg_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self._update_costmap_from_segmentation()
        except Exception as e:
            self.get_logger().error(f"Segmentation callback error: {e}")
    
    def detection_callback(self, msg: Detection2DArray):
        """Process object detections."""
        self.latest_detections = msg
        self._update_costmap_from_detections()
    
    def _update_costmap_from_segmentation(self):
        """Project segmentation mask to costmap."""
        if self.latest_seg_mask is None:
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        mask = self.latest_seg_mask
        h, w = mask.shape
        
        # Project pixels to ground plane
        # For simplicity, we assume a flat ground and use a simple projection
        # based on image row (depth) and column (lateral offset)
        
        # Camera field of view determines lateral extent
        fov_half = self.camera_fov_h / 2
        
        for row in range(h):
            # Estimate depth from image row (simple pinhole model approximation)
            # Rows closer to top = farther away
            # Rows closer to bottom = closer
            
            # Normalize row to [0, 1] where 0 is bottom, 1 is top
            row_normalized = 1.0 - (row / h)
            
            # Skip top portion (sky)
            if row_normalized > 0.6:
                continue
            
            # Estimate depth (simple linear model)
            # Bottom of image is ~0.5m, middle is ~5m
            depth = 0.5 + row_normalized * 8.0
            
            # Calculate lateral range at this depth
            lateral_range = depth * math.tan(fov_half)
            
            for col in range(w):
                # Get traversability value
                trav_value = mask[row, col]
                
                # Map pixel column to lateral offset
                col_normalized = (col / w) - 0.5  # [-0.5, 0.5]
                lateral_offset = col_normalized * 2 * lateral_range
                
                # Convert to costmap coordinates
                # costmap origin is at robot's back-left
                cx = int((depth + self.costmap_height / 2) / self.costmap_resolution)
                cy = int((lateral_offset + self.costmap_width / 2) / self.costmap_resolution)
                
                # Bounds check
                if 0 <= cx < self.costmap_height_cells and 0 <= cy < self.costmap_width_cells:
                    # Convert traversability to cost
                    if trav_value == 0:  # Traversable
                        cost = self.free_cost
                    elif trav_value == 254:  # Obstacle
                        cost = self.lethal_cost
                    else:  # Unknown
                        cost = self.unknown_cost
                    
                    # Update with max cost (obstacles override free)
                    if cost > self.costmap_data[cx, cy] or self.costmap_data[cx, cy] == self.unknown_cost:
                        self.costmap_data[cx, cy] = cost
                        self.last_update_time[cx, cy] = current_time
    
    def _update_costmap_from_detections(self):
        """Add detected objects to costmap as obstacles."""
        if self.latest_detections is None or self.camera_info is None:
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        for detection in self.latest_detections.detections:
            if not detection.results:
                continue
            
            # Get bounding box center
            cx = detection.bbox.center.position.x
            cy = detection.bbox.center.position.y
            
            # Get class
            class_id = int(detection.results[0].hypothesis.class_id)
            
            # Determine safety radius
            if class_id == 0:  # Person
                safety_radius = self.person_safety
            else:
                safety_radius = self.detection_inflation
            
            # Estimate depth from bounding box (simple heuristic)
            # Larger boxes = closer objects
            bbox_height = detection.bbox.size_y
            estimated_depth = 500.0 / max(bbox_height, 10)  # Simple inverse relationship
            estimated_depth = np.clip(estimated_depth, 0.5, 10.0)
            
            # Estimate lateral offset from bbox center
            image_center_x = self.camera_info.width / 2
            lateral_offset = (cx - image_center_x) / image_center_x * estimated_depth * math.tan(self.camera_fov_h / 2)
            
            # Convert to costmap coordinates
            map_x = int((estimated_depth + self.costmap_height / 2) / self.costmap_resolution)
            map_y = int((lateral_offset + self.costmap_width / 2) / self.costmap_resolution)
            
            # Inflate obstacle
            inflation_cells = int(safety_radius / self.costmap_resolution)
            
            for dx in range(-inflation_cells, inflation_cells + 1):
                for dy in range(-inflation_cells, inflation_cells + 1):
                    nx, ny = map_x + dx, map_y + dy
                    
                    if 0 <= nx < self.costmap_height_cells and 0 <= ny < self.costmap_width_cells:
                        # Distance from center
                        dist = math.sqrt(dx * dx + dy * dy) * self.costmap_resolution
                        
                        if dist <= safety_radius:
                            # Lethal at center, inscribed at edge
                            if dist < safety_radius * 0.5:
                                cost = self.lethal_cost
                            else:
                                cost = int(self.lethal_cost * (1 - dist / safety_radius))
                            
                            if cost > self.costmap_data[nx, ny]:
                                self.costmap_data[nx, ny] = cost
                                self.last_update_time[nx, ny] = current_time
    
    def _apply_temporal_decay(self):
        """Decay old obstacle data."""
        if not self.use_temporal:
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        age = current_time - self.last_update_time
        
        # Cells older than decay_time become unknown
        old_cells = age > self.decay_time
        self.costmap_data[old_cells] = self.unknown_cost
    
    def publish_costmap(self):
        """Publish the current costmap."""
        # Apply temporal decay
        self._apply_temporal_decay()
        
        # Create OccupancyGrid message
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.costmap_frame
        
        # Map metadata
        msg.info.resolution = self.costmap_resolution
        msg.info.width = self.costmap_width_cells
        msg.info.height = self.costmap_height_cells
        
        # Origin is at robot's back-left corner
        msg.info.origin.position.x = -self.costmap_height / 2
        msg.info.origin.position.y = -self.costmap_width / 2
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Flatten and convert costmap data
        # OccupancyGrid uses row-major order
        msg.data = self.costmap_data.flatten().tolist()
        
        self.costmap_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

