#!/usr/bin/env python3
"""
YOLOv8 Object Detection Node

Performs real-time object detection using YOLOv8 with TensorRT acceleration.
Publishes Detection2DArray messages and optionally annotated images.

This node is hardware-agnostic - works with both real camera and Gazebo simulation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class DetectionNode(Node):
    """Object detection node using YOLOv8."""
    
    # COCO class names
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(self):
        super().__init__('detection_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_type', 'yolov8n')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 480)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('target_classes', [0, 2, 5, 7, 14, 15, 16, 24, 25, 26, 27])
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('detection_topic', '/detections')
        self.declare_parameter('annotated_image_topic', '/camera/image_annotated')
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('use_tensorrt', True)
        self.declare_parameter('fp16_inference', True)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.model_type = self.get_parameter('model_type').value
        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.target_classes = self.get_parameter('target_classes').value
        self.publish_annotated = self.get_parameter('publish_annotated_image').value
        self.max_detections = self.get_parameter('max_detections').value
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Model (lazy initialization)
        self.model = None
        self.model_loaded = False
        
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            sensor_qos
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            self.get_parameter('detection_topic').value,
            10
        )
        
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image,
                self.get_parameter('annotated_image_topic').value,
                sensor_qos
            )
        
        self.get_logger().info(f"Detection node initialized")
        self.get_logger().info(f"  Model: {self.model_type}")
        self.get_logger().info(f"  Input size: {self.input_width}x{self.input_height}")
        self.get_logger().info(f"  Confidence threshold: {self.conf_threshold}")
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            # Try to import ultralytics
            from ultralytics import YOLO
            
            if self.model_path and os.path.exists(self.model_path):
                # Load custom model/engine
                self.model = YOLO(self.model_path)
                self.get_logger().info(f"Loaded model from: {self.model_path}")
            else:
                # Download and use default model
                self.model = YOLO(f'{self.model_type}.pt')
                self.get_logger().info(f"Using default {self.model_type} model")
            
            self.model_loaded = True
            
        except ImportError:
            self.get_logger().warn(
                "ultralytics not installed. Running in passthrough mode. "
                "Install with: pip install ultralytics"
            )
            self.model_loaded = False
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def image_callback(self, msg: Image):
        """Process incoming image for object detection."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run detection
            detections, annotated_image = self._detect(cv_image)
            
            # Create and publish detection message
            detection_msg = self._create_detection_msg(detections, msg.header)
            self.detection_pub.publish(detection_msg)
            
            # Publish annotated image if enabled
            if self.publish_annotated and annotated_image is not None:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
                
        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")
    
    def _detect(self, image: np.ndarray) -> Tuple[List[dict], Optional[np.ndarray]]:
        """
        Run object detection on image.
        
        Returns:
            List of detection dictionaries and annotated image
        """
        detections = []
        annotated = image.copy() if self.publish_annotated else None
        
        if not self.model_loaded:
            # Passthrough mode - no detections
            return detections, annotated
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.nms_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    
                    # Filter by target classes if specified
                    if self.target_classes and cls_id not in self.target_classes:
                        continue
                    
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection = {
                        'class_id': cls_id,
                        'class_name': self.COCO_CLASSES[cls_id] if cls_id < len(self.COCO_CLASSES) else 'unknown',
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                    
                    # Draw on annotated image
                    if annotated is not None:
                        self._draw_detection(annotated, detection)
            
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
        
        return detections, annotated
    
    def _draw_detection(self, image: np.ndarray, detection: dict):
        """Draw detection bounding box and label on image."""
        x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        
        # Color based on class (simple hash)
        color_idx = detection['class_id'] % 10
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0)
        ]
        color = colors[color_idx]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _create_detection_msg(self, detections: List[dict], header: Header) -> Detection2DArray:
        """Create ROS Detection2DArray message from detections."""
        msg = Detection2DArray()
        msg.header = header
        
        for det in detections:
            detection = Detection2D()
            detection.header = header
            
            # Bounding box
            x1, y1, x2, y2 = det['bbox']
            detection.bbox.center.position.x = (x1 + x2) / 2
            detection.bbox.center.position.y = (y1 + y2) / 2
            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1
            
            # Class hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det['class_id'])
            hypothesis.hypothesis.score = det['confidence']
            detection.results.append(hypothesis)
            
            msg.detections.append(detection)
        
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

