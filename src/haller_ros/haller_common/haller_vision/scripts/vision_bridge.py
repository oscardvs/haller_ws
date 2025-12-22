#!/usr/bin/env python3
"""
Vision Bridge Utilities

Shared utilities for the vision pipeline nodes.
Provides common functions for image processing, coordinate transforms, etc.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import math


class CameraProjection:
    """Handles camera projection and coordinate transforms."""
    
    def __init__(self, camera_matrix: np.ndarray, camera_height: float, 
                 camera_pitch: float = 0.0):
        """
        Initialize camera projection.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            camera_height: Camera height above ground (meters)
            camera_pitch: Camera pitch angle (radians, positive = looking down)
        """
        self.K = camera_matrix
        self.camera_height = camera_height
        self.camera_pitch = camera_pitch
        
        # Extract focal lengths and principal point
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
    
    def pixel_to_ground(self, u: float, v: float, 
                        ground_height: float = 0.0) -> Optional[Tuple[float, float]]:
        """
        Project a pixel to 3D ground plane coordinates.
        
        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            ground_height: Height of ground plane (default 0)
        
        Returns:
            (x, y) in robot frame, or None if ray doesn't hit ground
        """
        # Convert pixel to normalized camera coordinates
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        
        # Ray direction in camera frame (Z forward)
        ray = np.array([x_norm, y_norm, 1.0])
        ray = ray / np.linalg.norm(ray)
        
        # Apply camera pitch rotation
        cos_p = math.cos(self.camera_pitch)
        sin_p = math.sin(self.camera_pitch)
        R_pitch = np.array([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ])
        ray = R_pitch @ ray
        
        # Check if ray points downward
        if ray[2] >= 0:
            return None  # Ray pointing up, no ground intersection
        
        # Intersect with ground plane
        t = (ground_height - self.camera_height) / ray[2]
        
        if t < 0:
            return None  # Ground is behind camera
        
        # Compute intersection point
        ground_x = ray[0] * t
        ground_y = ray[1] * t
        
        return (ground_x, ground_y)
    
    def ground_to_pixel(self, x: float, y: float, 
                        ground_height: float = 0.0) -> Optional[Tuple[int, int]]:
        """
        Project a ground point to pixel coordinates.
        
        Args:
            x: Ground x coordinate (forward)
            y: Ground y coordinate (left)
            ground_height: Height of ground plane
        
        Returns:
            (u, v) pixel coordinates, or None if behind camera
        """
        # 3D point in world frame
        point = np.array([x, y, ground_height - self.camera_height])
        
        # Apply inverse pitch rotation
        cos_p = math.cos(-self.camera_pitch)
        sin_p = math.sin(-self.camera_pitch)
        R_pitch = np.array([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ])
        point = R_pitch @ point
        
        # Check if point is in front of camera
        if point[2] <= 0:
            return None
        
        # Project to pixel
        u = int(self.fx * point[0] / point[2] + self.cx)
        v = int(self.fy * point[1] / point[2] + self.cy)
        
        return (u, v)


class ImageProcessor:
    """Common image processing utilities."""
    
    @staticmethod
    def resize_for_inference(image: np.ndarray, target_width: int, 
                              target_height: int) -> Tuple[np.ndarray, float, float]:
        """
        Resize image for inference while preserving aspect ratio.
        
        Returns:
            Resized image, scale_x, scale_y for coordinate mapping
        """
        h, w = image.shape[:2]
        scale_x = target_width / w
        scale_y = target_height / h
        
        resized = cv2.resize(image, (target_width, target_height))
        
        return resized, scale_x, scale_y
    
    @staticmethod
    def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO detection."""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # NCHW format
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image
    
    @staticmethod
    def preprocess_for_segmentation(image: np.ndarray, 
                                     mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
        """Preprocess image for segmentation with ImageNet normalization."""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(mean)) / np.array(std)
        
        # NCHW format
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)


class CostmapUtils:
    """Utilities for costmap operations."""
    
    @staticmethod
    def world_to_map(x: float, y: float, origin_x: float, origin_y: float,
                     resolution: float) -> Tuple[int, int]:
        """Convert world coordinates to map cell indices."""
        mx = int((x - origin_x) / resolution)
        my = int((y - origin_y) / resolution)
        return mx, my
    
    @staticmethod
    def map_to_world(mx: int, my: int, origin_x: float, origin_y: float,
                     resolution: float) -> Tuple[float, float]:
        """Convert map cell indices to world coordinates."""
        x = mx * resolution + origin_x + resolution / 2
        y = my * resolution + origin_y + resolution / 2
        return x, y
    
    @staticmethod
    def inflate_costmap(costmap: np.ndarray, inflation_radius: float,
                        resolution: float, inscribed_cost: int = 253) -> np.ndarray:
        """
        Apply inflation to obstacles in costmap.
        
        Args:
            costmap: Input costmap array
            inflation_radius: Inflation radius in meters
            resolution: Costmap resolution in meters/cell
            inscribed_cost: Cost value for inscribed region
        
        Returns:
            Inflated costmap
        """
        inflated = costmap.copy()
        radius_cells = int(inflation_radius / resolution)
        
        # Find lethal cells
        lethal_mask = costmap >= 254
        lethal_indices = np.argwhere(lethal_mask)
        
        for idx in lethal_indices:
            y, x = idx
            
            # Apply circular inflation
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    ny, nx = y + dy, x + dx
                    
                    if 0 <= ny < costmap.shape[0] and 0 <= nx < costmap.shape[1]:
                        dist = math.sqrt(dx * dx + dy * dy) * resolution
                        
                        if dist <= inflation_radius:
                            # Calculate cost based on distance
                            cost_factor = 1.0 - (dist / inflation_radius)
                            cost = int(inscribed_cost * cost_factor)
                            
                            if cost > inflated[ny, nx]:
                                inflated[ny, nx] = cost
        
        return inflated


def main():
    """Test utilities."""
    print("Vision Bridge Utilities")
    
    # Test camera projection
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    proj = CameraProjection(K, camera_height=0.15, camera_pitch=0.1)
    
    # Test pixel to ground
    result = proj.pixel_to_ground(320, 400)
    if result:
        print(f"Pixel (320, 400) -> Ground ({result[0]:.2f}, {result[1]:.2f})")
    
    print("Tests passed!")


if __name__ == '__main__':
    main()

