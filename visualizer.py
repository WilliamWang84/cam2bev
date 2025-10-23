"""
Visualizer Module
"""
import cv2
import numpy as np

from typing import List, Tuple

class BEVVisualizer:
    """
    Visualizes tracked vehicles on BEV image
    """
    
    def __init__(self, bev_image_path: str, scale_factor: float = 1.0):
        self.bev_image = cv2.imread(bev_image_path)
        if self.bev_image is None:
            raise ValueError(f"Could not load BEV image: {bev_image_path}")
        
        self.scale_factor = scale_factor
        self.original_bev = self.bev_image.copy()
    
    def reset(self):
        """Reset to clean BEV image"""
        self.bev_image = self.original_bev.copy()
    
    def draw_vehicle(self, 
                     center: Tuple[int, int],
                     corners: np.ndarray = None,
                     vehicle_id: int = None,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     class_name: str = "vehicle",
                     size: int = 20,
                     font_scale: float = 1.0
                     ):
        """
        Draw vehicle on BEV image
        
        Args:
            center: (x, y) position in BEV
            corners: Optional 4x2 array of bbox corners
            vehicle_id: Optional tracking ID
            color: RGB color tuple
            class_name: Vehicle class name
            size: Size of marker if corners not provided
            font_scale: Font scale for label (default 1.0)
        """
        center = (int(center[0]), int(center[1]))
        
        if corners is not None:
            # Draw oriented bounding box
            corners = corners.astype(np.int32)
            cv2.polylines(self.bev_image, [corners], True, color, 2)
            
            # Fill semi-transparent
            overlay = self.bev_image.copy()
            cv2.fillPoly(overlay, [corners], color)
            self.bev_image = cv2.addWeighted(self.bev_image, 0.7, overlay, 0.3, 0)
        else:
            # Draw simple box
            half_size = size // 2
            x1 = center[0] - half_size
            y1 = center[1] - half_size
            x2 = center[0] + half_size
            y2 = center[1] + half_size
            cv2.rectangle(self.bev_image, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(self.bev_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Draw center point
        cv2.circle(self.bev_image, center, 3, (255, 255, 255), -1)
        
        # Draw label
        if vehicle_id is not None:
            label = f"ID:{vehicle_id} {class_name}"
            font_thickness = 2

            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Draw text background
            text_x = center[0] + 10
            text_y = center[1] - 10
            cv2.rectangle(self.bev_image, (text_x - 2, text_y - h - baseline), (text_x + w + 2, text_y + baseline), color, -1)
            
            # Draw text
            cv2.putText(self.bev_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

            # cv2.rectangle(self.bev_image, 
            #              (center[0] - w//2, center[1] - 25),
            #              (center[0] + w//2, center[1] - 15),
            #              (0, 0, 0), -1)
            # cv2.putText(self.bev_image, label,
            #            (center[0] - w//2, center[1] - 18),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_trajectory(self, 
                       points: List[Tuple[int, int]], 
                       color: Tuple[int, int, int] = (0, 255, 255)):
        """Draw trajectory line"""
        if len(points) < 2:
            return
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(self.bev_image, [points], False, color, 2)
    
    def add_legend(self):
        """Add legend/info overlay"""
        # Add semi-transparent background
        h, w = self.bev_image.shape[:2]
        overlay = self.bev_image.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        self.bev_image = cv2.addWeighted(self.bev_image, 0.7, overlay, 0.3, 0)
        
        # Add text
        cv2.putText(self.bev_image, "Bird's Eye View", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.bev_image, "Vehicle Tracking", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def get_image(self) -> np.ndarray:
        """Get current BEV image"""
        return self.bev_image.copy()
    
    def save(self, output_path: str):
        """Save BEV image"""
        cv2.imwrite(output_path, self.bev_image)
