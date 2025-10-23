"""
Define data classes
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class Vehicle:
    """Represents a detected/tracked vehicle"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]
    score: float # confidence score of the detection
    class_name: str
    color: Tuple[int, int, int]


@dataclass
class CalibrationPoint:
    """Correspondence point between camera view and BEV"""
    camera_point: Tuple[float, float]  # (x, y) in camera view
    bev_point: Tuple[float, float]     # (x, y) in BEV (meters or pixels)
    description: str = ""