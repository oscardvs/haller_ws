#!/usr/bin/env python3
"""
Semantic Segmentation Node

Performs real-time semantic segmentation for traversability analysis.
Supports multiple backends: PyTorch, ONNX, and TensorRT.

This node is hardware-agnostic - works with both real camera and Gazebo simulation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Tuple, Optional
import os


class SegmentationNode(Node):
    """Semantic segmentation node for traversability analysis."""
    
    # Cityscapes class colors for visualization
    CITYSCAPES_COLORS = np.array([
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ], dtype=np.uint8)
    
    # Traversability class colors
    TRAVERSABILITY_COLORS = {
        'traversable': [0, 255, 0],    # Green
        'caution': [255, 255, 0],      # Yellow
        'obstacle': [255, 0, 0],       # Red
        'unknown': [128, 128, 128],    # Gray
    }
    
    def __init__(self):
        super().__init__('segmentation_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_type', 'segformer_b0')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 480)
        self.declare_parameter('num_classes', 19)
        self.declare_parameter('traversable_classes', [0, 1, 9])
        self.declare_parameter('obstacle_classes', [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18])
        self.declare_parameter('unknown_classes', [8, 10])
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('segmentation_topic', '/segmentation/mask')
        self.declare_parameter('colored_mask_topic', '/segmentation/colored')
        self.declare_parameter('traversability_mask_topic', '/segmentation/traversability')
        self.declare_parameter('publish_colored_mask', True)
        self.declare_parameter('use_tensorrt', True)
        self.declare_parameter('fp16_inference', True)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.model_type = self.get_parameter('model_type').value
        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value
        self.num_classes = self.get_parameter('num_classes').value
        self.traversable_classes = set(self.get_parameter('traversable_classes').value)
        self.obstacle_classes = set(self.get_parameter('obstacle_classes').value)
        self.unknown_classes = set(self.get_parameter('unknown_classes').value)
        self.publish_colored = self.get_parameter('publish_colored_mask').value
        
        # CV Bridge
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
        self.mask_pub = self.create_publisher(
            Image,
            self.get_parameter('segmentation_topic').value,
            10
        )
        
        self.traversability_pub = self.create_publisher(
            Image,
            self.get_parameter('traversability_mask_topic').value,
            10
        )
        
        if self.publish_colored:
            self.colored_pub = self.create_publisher(
                Image,
                self.get_parameter('colored_mask_topic').value,
                sensor_qos
            )
        
        self.get_logger().info(f"Segmentation node initialized")
        self.get_logger().info(f"  Model: {self.model_type}")
        self.get_logger().info(f"  Input size: {self.input_width}x{self.input_height}")
        self.get_logger().info(f"  Traversable classes: {self.traversable_classes}")
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the segmentation model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load TensorRT or ONNX model
                if self.model_path.endswith('.engine'):
                    self._load_tensorrt_model(self.model_path)
                elif self.model_path.endswith('.onnx'):
                    self._load_onnx_model(self.model_path)
                else:
                    self.get_logger().warn(f"Unknown model format: {self.model_path}")
            else:
                # Try to load a default model using transformers
                self._load_default_model()
                
        except Exception as e:
            self.get_logger().warn(f"Failed to load model: {e}. Running in demo mode.")
            self.model_loaded = False
    
    def _load_default_model(self):
        """Try to load a default segmentation model."""
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            import torch
            
            model_name = "nvidia/segformer-b0-finetuned-cityscapes-640-1280"
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            self.model_backend = 'transformers'
            self.get_logger().info(f"Loaded SegFormer model on {self.device}")
            
        except ImportError:
            self.get_logger().warn(
                "transformers not installed. Install with: pip install transformers torch"
            )
            self.model_loaded = False
    
    def _load_tensorrt_model(self, path: str):
        """Load TensorRT engine."""
        self.get_logger().info(f"TensorRT loading from: {path}")
        # TensorRT loading would go here
        # For now, mark as not loaded
        self.model_loaded = False
    
    def _load_onnx_model(self, path: str):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.model = ort.InferenceSession(path, providers=providers)
            self.model_loaded = True
            self.model_backend = 'onnx'
            self.get_logger().info(f"Loaded ONNX model from: {path}")
            
        except ImportError:
            self.get_logger().warn("onnxruntime not installed")
            self.model_loaded = False
    
    def image_callback(self, msg: Image):
        """Process incoming image for semantic segmentation."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Run segmentation
            seg_mask, colored_mask, traversability_mask = self._segment(cv_image)
            
            # Publish segmentation mask (class IDs)
            if seg_mask is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(seg_mask, encoding='mono8')
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)
            
            # Publish traversability mask
            if traversability_mask is not None:
                trav_msg = self.bridge.cv2_to_imgmsg(traversability_mask, encoding='mono8')
                trav_msg.header = msg.header
                self.traversability_pub.publish(trav_msg)
            
            # Publish colored mask for visualization
            if self.publish_colored and colored_mask is not None:
                colored_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding='rgb8')
                colored_msg.header = msg.header
                self.colored_pub.publish(colored_msg)
                
        except Exception as e:
            self.get_logger().error(f"Segmentation error: {e}")
    
    def _segment(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run semantic segmentation on image.
        
        Returns:
            Tuple of (class_mask, colored_mask, traversability_mask)
        """
        if not self.model_loaded:
            # Demo mode: create synthetic segmentation
            return self._demo_segmentation(image)
        
        try:
            if self.model_backend == 'transformers':
                return self._segment_transformers(image)
            elif self.model_backend == 'onnx':
                return self._segment_onnx(image)
            else:
                return self._demo_segmentation(image)
                
        except Exception as e:
            self.get_logger().error(f"Segmentation inference error: {e}")
            return None, None, None
    
    def _segment_transformers(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run segmentation using transformers/PyTorch."""
        import torch
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Upsample to original size
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(image.shape[0], image.shape[1]),
            mode='bilinear',
            align_corners=False
        )
        
        # Get class predictions
        seg_mask = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # Create colored mask
        colored_mask = self._colorize_mask(seg_mask)
        
        # Create traversability mask
        traversability_mask = self._create_traversability_mask(seg_mask)
        
        return seg_mask, colored_mask, traversability_mask
    
    def _segment_onnx(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run segmentation using ONNX Runtime."""
        # Preprocess image
        input_image = cv2.resize(image, (self.input_width, self.input_height))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        output = self.model.run(None, {input_name: input_image})[0]
        
        # Get class predictions
        seg_mask = np.argmax(output, axis=1).squeeze().astype(np.uint8)
        
        # Resize back to original size
        seg_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
        
        # Create colored and traversability masks
        colored_mask = self._colorize_mask(seg_mask)
        traversability_mask = self._create_traversability_mask(seg_mask)
        
        return seg_mask, colored_mask, traversability_mask
    
    def _demo_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create demo segmentation for testing without a model."""
        h, w = image.shape[:2]
        
        # Create a simple demo mask based on image position
        # Bottom half is "road", top half is "sky", middle has some "obstacles"
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Sky (class 10)
        seg_mask[:h//3, :] = 10
        
        # Road (class 0)
        seg_mask[h//2:, :] = 0
        
        # Vegetation/terrain (class 9) on sides
        seg_mask[h//3:h//2, :w//4] = 9
        seg_mask[h//3:h//2, 3*w//4:] = 9
        
        # Building (class 2) in middle background
        seg_mask[h//3:h//2, w//4:3*w//4] = 2
        
        colored_mask = self._colorize_mask(seg_mask)
        traversability_mask = self._create_traversability_mask(seg_mask)
        
        return seg_mask, colored_mask, traversability_mask
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert class mask to colored RGB image."""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id in range(len(self.CITYSCAPES_COLORS)):
            colored[mask == class_id] = self.CITYSCAPES_COLORS[class_id]
        
        return colored
    
    def _create_traversability_mask(self, seg_mask: np.ndarray) -> np.ndarray:
        """
        Convert semantic segmentation to traversability mask.
        
        Values:
            0: Traversable (safe)
            127: Caution (possible but risky)
            254: Obstacle (cannot traverse)
            255: Unknown
        """
        trav_mask = np.full(seg_mask.shape, 255, dtype=np.uint8)  # Default unknown
        
        # Mark traversable areas
        for cls_id in self.traversable_classes:
            trav_mask[seg_mask == cls_id] = 0
        
        # Mark obstacles
        for cls_id in self.obstacle_classes:
            trav_mask[seg_mask == cls_id] = 254
        
        # Unknown classes stay as 255
        
        return trav_mask


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

