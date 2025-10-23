"""
Perspective Transform Module
"""

import cv2
import json
import numpy as np

from data_class import CalibrationPoint
from typing import Tuple, List, Optional

class PerspectiveTransformer:
    """
    Handles perspective transformation from camera view to BEV
    """
    
    def __init__(self, meters_per_pixel: float = 0.055):
        self.homography_matrix = None
        self.inverse_homography = None
        self.calibration_points = []
        
        # BEV scale calibration
        self.meters_per_pixel = meters_per_pixel  # meters per pixel in BEV
        
        # Store vehicle dimensions (in BEV pixels)
        # These can be calibrated based on known vehicle sizes
        self.default_vehicle_length = 40  # pixels
        self.default_vehicle_width = 20   # pixels
        
        # Real-world vehicle dimensions (width x length in meters) based on typical vehicle sizes
        self.vehicle_dimensions = {
            'car': (2.0, 4.5),
            'truck': (2.5, 8.0),
            'bus': (2.5, 12.0),
            'motorcycle': (0.8, 2.0),
            'bicycle': (0.6, 1.8),
            'van': (2.2, 5.5),
            'semi': (2.6, 16.0),
            'default': (2.0, 4.5)  # Default to car size
        }
    
    def add_calibration_point(self, camera_point: Tuple[float, float], 
                             bev_point: Tuple[float, float],
                             description: str = ""):
        """Add a correspondence point"""
        self.calibration_points.append(
            CalibrationPoint(camera_point, bev_point, description)
        )
    
    def load_calibration_from_file(self, filepath: str):
        """Load calibration points from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.calibration_points = []
            for point in data['points']:
                self.add_calibration_point(
                    tuple(point['camera']),
                    tuple(point['bev']),
                    point.get('description', '')
                )
    
    def save_calibration(self, filepath: str):
        """Save calibration points to JSON file"""
        data = {
            'points': [
                {
                    'camera': list(p.camera_point),
                    'bev': list(p.bev_point),
                    'description': p.description
                }
                for p in self.calibration_points
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def compute_homography(self, method=cv2.RANSAC):
        """
        Compute homography matrix from calibration points
        Minimum 4 points required
        """
        if len(self.calibration_points) < 4:
            raise ValueError("At least 4 calibration points required")
        
        src_points = np.float32([p.camera_point for p in self.calibration_points])
        dst_points = np.float32([p.bev_point for p in self.calibration_points])
        
        self.homography_matrix, mask = cv2.findHomography(
            src_points, dst_points, method, ransacReprojThreshold=5.0
        )
        
        self.inverse_homography = np.linalg.inv(self.homography_matrix)
        
        return self.homography_matrix
    
    def camera_to_bev(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from camera view to BEV
        
        Args:
            points: Nx2 array of (x, y) coordinates
        
        Returns:
            Nx2 array of transformed coordinates
        """
        if self.homography_matrix is None:
            raise ValueError("Homography not computed. Call compute_homography first.")
        
        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        points = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.homography_matrix)
        
        return transformed.reshape(-1, 2)
    
    def bev_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Transform points from BEV to camera view"""
        if self.inverse_homography is None:
            raise ValueError("Homography not computed.")
        
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.inverse_homography)
        
        return transformed.reshape(-1, 2)
    
    def estimate_heading_from_trajectory(self, 
                                        trajectory: List[Tuple[float, float]], 
                                        min_points: int = 3) -> Optional[float]:
        """
        Estimate vehicle heading (angle) from its trajectory in BEV
        
        Args:
            trajectory: List of (x, y) positions in BEV (most recent last)
            min_points: Minimum number of points needed for estimation
            
        Returns:
            Angle in radians (0 = pointing right, increases counter-clockwise)
            Returns None if not enough points
        """
        if len(trajectory) < min_points:
            return None
        
        # Use recent trajectory points (last 5-10 frames)
        recent_points = trajectory[-min(10, len(trajectory)):]
        points = np.array(recent_points, dtype=np.float32)
        
        # Fit line using least squares
        if len(points) >= 2:
            # Calculate mean
            mean = np.mean(points, axis=0)
            
            # Calculate covariance matrix
            centered = points - mean
            
            # Use simple linear regression for heading
            if len(points) >= 3:
                # SVD-based approach (more robust)
                _, _, vt = np.linalg.svd(centered)
                direction = vt[0]  # First principal component
            else:
                # Simple two-point direction
                direction = points[-1] - points[0]
                direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            # Convert to angle
            angle = np.arctan2(direction[1], direction[0])
            return angle
        
        return None
    
    def create_oriented_rectangle(self, 
                                 center: np.ndarray,
                                 angle: float,
                                 length: float = None,
                                 width: float = None) -> np.ndarray:
        """
        Create an oriented rectangle in BEV
        
        Args:
            center: (x, y) center position
            angle: Orientation angle in radians
            length: Vehicle length (default uses self.default_vehicle_length)
            width: Vehicle width (default uses self.default_vehicle_width)
            
        Returns:
            4x2 array of corner points
        """
        if length is None:
            length = self.default_vehicle_length
        if width is None:
            width = self.default_vehicle_width
        
        # Create rectangle centered at origin
        half_l = length / 2
        half_w = width / 2
        corners = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ], dtype=np.float32)
        
        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Rotate and translate
        rotated = corners @ rotation.T
        translated = rotated + center
        
        return translated
    
    def get_vehicle_size_from_class(self, class_name: str) -> Tuple[float, float]:
        """
        Get vehicle size in BEV pixels based on class name
        
        Args:
            class_name: Vehicle class (e.g., 'car', 'bus', 'truck')
            
        Returns:
            (length, width) in BEV pixels
        """
        # Normalize class name (handle variations)
        class_name_lower = class_name.lower().strip()
        
        # Get dimensions in meters
        if class_name_lower in self.vehicle_dimensions:
            width_m, length_m = self.vehicle_dimensions[class_name_lower]
        else:
            # Fallback to default (car)
            width_m, length_m = self.vehicle_dimensions['default']
        
        # Convert to BEV pixels
        width_px = width_m / self.meters_per_pixel
        length_px = length_m / self.meters_per_pixel
        
        return length_px, width_px
    
    def estimate_vehicle_size_from_bbox(self, 
                                       bbox: Tuple[int, int, int, int],
                                       scale_factor: float = 0.8) -> Tuple[float, float]:
        """
        Estimate vehicle size in BEV from camera bbox
        This provides a rough approximation based on bbox corners transformation
        
        Args:
            bbox: (x1, y1, x2, y2) in camera view
            scale_factor: Scale down factor to account for bbox being larger than vehicle
            
        Returns:
            (length, width) in BEV pixels
        """
        x1, y1, x2, y2 = bbox
        
        # Transform bottom edge (closer to camera, more reliable)
        bottom_left = np.array([[[x1, y2]]], dtype=np.float32)
        bottom_right = np.array([[[x2, y2]]], dtype=np.float32)
        
        bl_bev = cv2.perspectiveTransform(bottom_left, self.homography_matrix)[0][0]
        br_bev = cv2.perspectiveTransform(bottom_right, self.homography_matrix)[0][0]
        
        # Estimate width from bottom edge
        width = np.linalg.norm(br_bev - bl_bev) * scale_factor
        
        # Estimate length (use default or scale from height)
        # This is less reliable due to perspective, so we use a ratio
        length = width * 2.0  # Typical car length/width ratio
        
        return length, width
    
    def transform_bbox(self, 
                      bbox: Tuple[int, int, int, int], 
                      transform_corners: bool = False,
                      trajectory: Optional[List[Tuple[float, float]]] = None,
                      use_oriented_rect: bool = True,
                      class_name: str = 'car',
                      use_class_size: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform bounding box from camera view to BEV with proper orientation
        
        Args:
            bbox: (x1, y1, x2, y2) in camera view
            transform_corners: If True and no trajectory, uses direct transformation
            trajectory: List of historical BEV positions for heading estimation
            use_oriented_rect: If True, generates oriented rectangle based on trajectory
            class_name: Vehicle class name (e.g., 'car', 'bus', 'truck')
            use_class_size: If True, use class-based size; otherwise estimate from bbox
            
        Returns:
            (center, corners) where:
                - center: 2D array of center position in BEV
                - corners: 4x2 array of oriented rectangle corners, or None
        """
        x1, y1, x2, y2 = bbox
        
        # Transform bottom center point (vehicle ground position)
        bottom_center = np.array([[[(x1 + x2) / 2, y2]]], dtype=np.float32)
        transformed_center = cv2.perspectiveTransform(
            bottom_center, self.homography_matrix
        )[0][0]
        
        # Generate oriented rectangle if trajectory is available
        if use_oriented_rect and trajectory is not None and len(trajectory) >= 2:
            # Estimate heading from trajectory
            angle = self.estimate_heading_from_trajectory(trajectory)
            
            if angle is not None:
                # Get vehicle size based on preference
                if use_class_size:
                    length, width = self.get_vehicle_size_from_class(class_name)
                else:
                    length, width = self.estimate_vehicle_size_from_bbox(bbox)
                
                # Create oriented rectangle
                oriented_corners = self.create_oriented_rectangle(
                    transformed_center, angle, length, width
                )
                
                return transformed_center, oriented_corners
        
        # Fallback: either use direct transformation or default rectangle
        if transform_corners:
            # Direct transformation (will be distorted)
            corners = np.array([
                [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
            ], dtype=np.float32)
            
            transformed_corners = cv2.perspectiveTransform(
                corners, self.homography_matrix
            ).reshape(-1, 2)
            
            return transformed_center, transformed_corners
        else:
            # Default rectangle with class-based size
            if use_class_size:
                length, width = self.get_vehicle_size_from_class(class_name)
            else:
                length = self.default_vehicle_length
                width = self.default_vehicle_width
            
            half_l = length / 2
            half_w = width / 2
            cx, cy = transformed_center
            default_corners = np.array([
                [cx - half_l, cy - half_w],
                [cx + half_l, cy - half_w],
                [cx + half_l, cy + half_w],
                [cx - half_l, cy + half_w]
            ], dtype=np.float32)
            
            return transformed_center, default_corners
