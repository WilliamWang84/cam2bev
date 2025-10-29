import copy
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple


def centroid_distances(boxes_a: torch.Tensor, boxes_b: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Calculate L1 or L2 distances between centroids of two groups of boxes.
    
    Args:
        boxes_a: tensor of shape (N, 4) in xyxy format
        boxes_b: tensor of shape (M, 4) in xyxy format
        p: 1 for L1 (Manhattan), 2 for L2 (Euclidean)
    
    Returns:
        distances: tensor of shape (N, M)
    """
    centroids_a = (boxes_a[:, :2] + boxes_a[:, 2:]) / 2
    centroids_b = (boxes_b[:, :2] + boxes_b[:, 2:]) / 2
    return torch.cdist(centroids_a, centroids_b, p=p)


class CentroidTracker_Track:
    """Represents a single tracked object using centroid tracking."""
    def __init__(self, track_id: int, bbox: np.ndarray, score: float, class_name: str):
        self.id = track_id
        self.bbox = bbox
        self.score = score
        self.class_name = class_name
        self.time_since_update = 0
        self.hits = 1

    def update(self, bbox: np.ndarray, score: float, class_name: str):
        """Updates the track with a new detection."""
        self.bbox = bbox
        self.score = score
        self.class_name = class_name
        self.time_since_update = 0
        self.hits += 1


class CentroidTracker:
    """
    Simplified tracker using centroid distances for matching instead of Kalman Filter + IoU.
    """
    def __init__(self, max_age: int = 50, min_hits: int = 3, max_distance: float = 50.0, distance_metric: str = 'l2'):
        """
        Args:
            max_age: Maximum number of frames to keep alive a track without matching
            min_hits: Minimum number of hits before a track is confirmed (not used in basic version)
            max_distance: Maximum centroid distance threshold for matching
            distance_metric: 'l1' for Manhattan distance, 'l2' for Euclidean distance
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.distance_metric = distance_metric
        self.distance_p = 1 if distance_metric == 'l1' else 2
        self.tracks = []
        self.next_id = 1
        self.colors = {}

    def _generate_color(self, obj_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for object ID"""
        np.random.seed(obj_id)
        return tuple(np.random.randint(50, 255, 3).tolist())

    def _associate(self, tracks: List[CentroidTracker_Track], detections: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Performs data association using centroid distances and Hungarian algorithm."""
        if not tracks or detections.shape[0] == 0:
            return [], list(range(len(tracks))), list(range(detections.shape[0]))

        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes = detections[:, :4]
        
        # Calculate centroid distances
        distance_matrix = centroid_distances(
            torch.from_numpy(track_bboxes).float(), 
            torch.from_numpy(det_bboxes).float(),
            p=self.distance_p
        ).numpy()
        
        # Hungarian algorithm to find optimal assignment
        track_indices, det_indices = linear_sum_assignment(distance_matrix)
        
        matches = []
        unmatched_track_indices = set(range(len(tracks)))
        unmatched_det_indices = set(range(len(detections)))
        
        # Filter matches based on distance threshold
        for track_idx, det_idx in zip(track_indices, det_indices):
            if distance_matrix[track_idx, det_idx] < self.max_distance:
                matches.append((track_idx, det_idx))
                unmatched_track_indices.discard(track_idx)
                unmatched_det_indices.discard(det_idx)
                
        return matches, list(unmatched_track_indices), list(unmatched_det_indices)

    def update(self, detections: List[Dict]) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """
        Update tracker with new detections.
        
        Args:
            detections (List[Dict] or Dict): A list or dictionary of detection dictionaries.
                Each detection should have: {'bbox': [x1, y1, x2, y2], 'score': float, 'class_name': str}
        
        Returns:
            Tuple[Dict[int, Dict], Dict[int, Dict]]: 
                - Current active tracks (time_since_update == 0)
                - Previous frame's tracks (for tracking lost objects)
        """
        old_tracks = copy.deepcopy(self.tracks)

        # --- Step 1: Increment age for all tracks ---
        for track in self.tracks:
            track.time_since_update += 1

        # --- Step 2: Format Detections ---
        detection_list = list(detections.values()) if isinstance(detections, dict) else detections

        if len(detection_list) == 0:
            det_array = np.empty((0, 5))
        else:
            det_array = np.array([[*d['bbox'], d['score']] for d in detection_list])

        # Only match against recently seen tracks
        active_tracks = [t for t in self.tracks if t.time_since_update <= 1]

        # --- Step 3: Matching between Tracks and Detections ---
        matches, _, unmatched_dets = self._associate(active_tracks, det_array)

        # Update matched tracks
        for track_idx, det_idx in matches:
            track = active_tracks[track_idx]       
            det = detection_list[det_idx]
            track.update(np.array(det['bbox']), det['score'], det['class_name'])

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # --- Step 4: Initialize new tracks from unmatched detections ---
        for det_idx in unmatched_dets:
            det = detection_list[det_idx]
            new_track = CentroidTracker_Track(
                self.next_id, 
                np.array(det['bbox']), 
                det['score'], 
                det['class_name']
            )
            self.colors[self.next_id] = self._generate_color(self.next_id)
            self.tracks.append(new_track)
            self.next_id += 1

        # --- Step 5: Format outputs ---
        result = {}
        for track in self.tracks:
            if track.time_since_update == 0:
                result[track.id] = {
                    'id': track.id,
                    'bbox': track.bbox,
                    'color': self.colors.get(track.id, (0, 0, 255)),
                    'class_name': track.class_name,
                    'score': track.score
                }
        
        # Result old is for getting vehicle type 'class_name' for lost vehicles
        result_old = {}
        for track_old in old_tracks:
            result_old[track_old.id] = {
                'id': track_old.id,
                'bbox': track_old.bbox,
                'color': self.colors.get(track_old.id, (0, 0, 255)),
                'class_name': track_old.class_name,
                'score': track_old.score
            }
        
        return result, result_old
