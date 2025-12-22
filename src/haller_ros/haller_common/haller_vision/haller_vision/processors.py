"""
Vision Processing Classes

Shared processing classes for the Haller robot vision pipeline.
These classes provide reusable components for detection, segmentation,
and traversability analysis.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TraversabilityClass(Enum):
    """Traversability classification for semantic segmentation."""
    TRAVERSABLE = 0      # Safe to drive
    CAUTION = 127        # Possible but risky
    OBSTACLE = 254       # Cannot traverse
    UNKNOWN = 255        # Unknown/unmapped


@dataclass
class Detection:
    """Represents a single object detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get bounding box area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class SegmentationResult:
    """Represents segmentation output."""
    class_mask: np.ndarray        # HxW array of class IDs
    confidence_map: Optional[np.ndarray] = None  # HxW array of confidences
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.class_mask.shape[:2]
    
    def get_class_pixels(self, class_id: int) -> np.ndarray:
        """Get mask of pixels belonging to a specific class."""
        return self.class_mask == class_id


class ImagePreprocessor:
    """Handles image preprocessing for inference."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 480),
                 normalize: bool = True,
                 mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: (width, height) to resize images to
            normalize: Whether to apply ImageNet normalization
            mean: Channel means for normalization
            std: Channel stds for normalization
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def preprocess(self, image: np.ndarray, 
                   for_model: str = 'detection') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess image for inference.
        
        Args:
            image: BGR image from OpenCV
            for_model: 'detection' or 'segmentation'
        
        Returns:
            Preprocessed image and metadata for postprocessing
        """
        original_shape = image.shape[:2]
        
        # Resize
        resized = cv2.resize(image, self.target_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization if requested
        if self.normalize:
            normalized = (normalized - self.mean) / self.std
        
        # Convert to NCHW format for PyTorch/TensorRT
        nchw = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(nchw, axis=0)
        
        metadata = {
            'original_shape': original_shape,
            'target_size': self.target_size,
            'scale_x': self.target_size[0] / original_shape[1],
            'scale_y': self.target_size[1] / original_shape[0],
        }
        
        return batched.astype(np.float32), metadata
    
    def scale_bbox(self, bbox: Tuple[float, float, float, float],
                   metadata: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Scale bounding box back to original image coordinates."""
        x1, y1, x2, y2 = bbox
        scale_x = metadata['scale_x']
        scale_y = metadata['scale_y']
        
        return (
            x1 / scale_x,
            y1 / scale_y,
            x2 / scale_x,
            y2 / scale_y
        )


class DetectionPostprocessor:
    """Handles detection model output postprocessing."""
    
    def __init__(self, confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.45,
                 class_names: Optional[List[str]] = None):
        """
        Initialize postprocessor.
        
        Args:
            confidence_threshold: Minimum detection confidence
            nms_threshold: NMS IoU threshold
            class_names: List of class names
        """
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names or []
    
    def process(self, model_output: np.ndarray,
                metadata: Dict[str, Any]) -> List[Detection]:
        """
        Process model output into detections.
        
        Args:
            model_output: Raw model output tensor
            metadata: Preprocessing metadata
        
        Returns:
            List of Detection objects
        """
        # This is a placeholder - actual implementation depends on model format
        # YOLO outputs are typically [batch, num_boxes, 5 + num_classes]
        # where 5 = [x, y, w, h, objectness]
        
        detections = []
        # Implementation would parse model_output here
        
        return detections
    
    def apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply non-maximum suppression to detections."""
        if not detections:
            return []
        
        # Convert to OpenCV NMS format
        boxes = np.array([[d.bbox[0], d.bbox[1], 
                          d.bbox[2] - d.bbox[0], 
                          d.bbox[3] - d.bbox[1]] for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.conf_threshold, 
            self.nms_threshold
        )
        
        if len(indices) == 0:
            return []
        
        return [detections[i] for i in indices.flatten()]


class TraversabilityMapper:
    """Maps segmentation classes to traversability costs."""
    
    def __init__(self, 
                 traversable_classes: List[int],
                 obstacle_classes: List[int],
                 caution_classes: Optional[List[int]] = None):
        """
        Initialize mapper.
        
        Args:
            traversable_classes: Class IDs that are safe to traverse
            obstacle_classes: Class IDs that are obstacles
            caution_classes: Class IDs that require caution
        """
        self.traversable = set(traversable_classes)
        self.obstacles = set(obstacle_classes)
        self.caution = set(caution_classes or [])
    
    def map_segmentation(self, seg_mask: np.ndarray) -> np.ndarray:
        """
        Convert segmentation mask to traversability mask.
        
        Args:
            seg_mask: HxW array of class IDs
        
        Returns:
            HxW array of traversability values
        """
        trav_mask = np.full(seg_mask.shape, TraversabilityClass.UNKNOWN.value, 
                           dtype=np.uint8)
        
        for cls_id in self.traversable:
            trav_mask[seg_mask == cls_id] = TraversabilityClass.TRAVERSABLE.value
        
        for cls_id in self.caution:
            trav_mask[seg_mask == cls_id] = TraversabilityClass.CAUTION.value
        
        for cls_id in self.obstacles:
            trav_mask[seg_mask == cls_id] = TraversabilityClass.OBSTACLE.value
        
        return trav_mask
    
    def to_costmap(self, trav_mask: np.ndarray) -> np.ndarray:
        """
        Convert traversability mask to Nav2 costmap values.
        
        Nav2 costmap uses:
            0 = FREE
            1-252 = cost gradient
            253 = INSCRIBED
            254 = LETHAL
            255 = UNKNOWN (or -1 for signed)
        
        Args:
            trav_mask: Traversability mask
        
        Returns:
            Costmap-compatible array
        """
        costmap = np.full(trav_mask.shape, -1, dtype=np.int8)  # Unknown
        
        costmap[trav_mask == TraversabilityClass.TRAVERSABLE.value] = 0  # Free
        costmap[trav_mask == TraversabilityClass.CAUTION.value] = 100   # Medium cost
        costmap[trav_mask == TraversabilityClass.OBSTACLE.value] = 254  # Lethal
        
        return costmap


class VisualizationHelper:
    """Helpers for visualizing vision outputs."""
    
    # Default color palette for segmentation
    CITYSCAPES_COLORS = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
        (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
        (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]
    
    @staticmethod
    def draw_detections(image: np.ndarray, 
                       detections: List[Detection],
                       colors: Optional[Dict[int, Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        Draw detection boxes on image.
        
        Args:
            image: BGR image
            detections: List of detections to draw
            colors: Optional dict mapping class_id to BGR color
        
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            
            # Get color
            if colors and det.class_id in colors:
                color = colors[det.class_id]
            else:
                color = (0, 255, 0)  # Default green
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    @classmethod
    def colorize_segmentation(cls, seg_mask: np.ndarray,
                             colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        Convert class mask to colored RGB image.
        
        Args:
            seg_mask: HxW array of class IDs
            colors: Optional list of RGB colors per class
        
        Returns:
            HxWx3 RGB image
        """
        colors = colors or cls.CITYSCAPES_COLORS
        colored = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(colors):
            if class_id < len(colors):
                colored[seg_mask == class_id] = color
        
        return colored
    
    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray, 
                    alpha: float = 0.5) -> np.ndarray:
        """
        Overlay colored mask on image.
        
        Args:
            image: BGR image
            mask: RGB colored mask
            alpha: Blend factor (0 = only image, 1 = only mask)
        
        Returns:
            Blended image
        """
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(image, 1 - alpha, mask_bgr, alpha, 0)

