"""
ByteTrack Tracker Module
"""

import copy
import numpy as np
import torch

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from typing import List, Dict, Tuple

# =============================================================================
# ByteTracker Implementation
# =============================================================================

def bbox_xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """Converts [x1, y1, x2, y2] to [center_x, center_y, width, height]."""
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    cx = xyxy[0] + w / 2
    cy = xyxy[1] + h / 2
    return np.array([cx, cy, w, h])

def bbox_xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Converts [center_x, center_y, width, height] to [x1, y1, x2, y2]."""
    x1 = xywh[0] - xywh[2] / 2
    y1 = xywh[1] - xywh[3] / 2
    x2 = xywh[0] + xywh[2] / 2
    y2 = xywh[1] + xywh[3] / 2
    return np.array([x1, y1, x2, y2])

class KalmanFilterTracker:
    """A Kalman Filter wrapper for a single track."""
    def __init__(self, bbox: np.ndarray):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # State transition matrix
        self.kf.F = np.array([[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
                               [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
        # Measurement matrix
        self.kf.H = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]])
        
        # Measurement uncertainty
        self.kf.R[2:,2:] *= 10.
        # Process uncertainty
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        
        # Initial state [cx, cy, w, h, vx, vy, vw, vh]
        self.kf.x[:4] = bbox_xyxy_to_xywh(bbox).reshape(4, 1)

    def predict(self) -> np.ndarray:
        """Predicts the next state. Returns the predicted bbox in [x1, y1, x2, y2] format."""
        self.kf.predict()
        return bbox_xywh_to_xyxy(self.kf.x[:4].flatten())

    def update(self, bbox: np.ndarray):
        """Updates the filter with a new measurement."""
        self.kf.update(bbox_xyxy_to_xywh(bbox).reshape(4, 1))

class ByteTracker_Track:
    """Represents a single tracked object with its Kalman Filter."""
    def __init__(self, track_id: int, bbox: np.ndarray, score: float, class_name: str):
        self.id = track_id
        self.kf = KalmanFilterTracker(bbox)
        self.score = score
        self.class_name = class_name
        self.time_since_update = 0
        self.hits = 1

    @property
    def bbox(self) -> np.ndarray:
        """Returns the current bounding box estimate from the Kalman Filter."""
        return bbox_xywh_to_xyxy(self.kf.kf.x[:4].flatten())

    def predict(self):
        """Advances the state vector and returns the predicted bounding box."""
        return self.kf.predict()

    def update(self, bbox: np.ndarray, score: float, class_name: str):
        """Updates the Kalman Filter with a new detection."""
        self.kf.update(bbox)
        self.score = score
        self.class_name = class_name
        self.time_since_update = 0
        self.hits += 1

class ByteTracker:
    """The full ByteTrack algorithm with Kalman Filter and Hungarian matching."""
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1, iou_thresh: float = 0.7, max_age: int = 100):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1
        self.colors = {}

    def _generate_color(self, obj_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for object ID"""
        np.random.seed(obj_id)
        return tuple(np.random.randint(50, 255, 3).tolist())

    def _associate(self, tracks: List[ByteTracker_Track], detections: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Performs data association using the Hungarian algorithm."""
        if not tracks or detections.shape[0] == 0:
            return [], list(range(len(tracks))), list(range(detections.shape[0]))

        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes = detections[:, :4]
        
        iou_matrix = box_iou(torch.from_numpy(track_bboxes), torch.from_numpy(det_bboxes)).numpy()
        cost_matrix = 1 - iou_matrix
        
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_track_indices = set(range(len(tracks)))
        unmatched_det_indices = set(range(len(detections)))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] < (1 - self.iou_thresh):
                matches.append((track_idx, det_idx))
                unmatched_track_indices.discard(track_idx)
                unmatched_det_indices.discard(det_idx)
                
        return matches, list(unmatched_track_indices), list(unmatched_det_indices)

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections (List[Dict] or Dict): A list or dictionary of detection dictionaries.
        
        Returns:
            Dict[int, Dict]: A dictionary mapping track_id to detection info.
        """
        old_tracks = copy.deepcopy(self.tracks)

        # --- Step 1: Predict new locations and increment age ---
        for track in self.tracks:
            track.predict()
            track.time_since_update += 1

        # --- Step 2: Format detections and separate them ---
        # Handle both list of dicts and dict of dicts as input
        detection_list = list(detections.values()) if isinstance(detections, dict) else detections

        if len(detection_list) == 0:
            det_array = np.empty((0, 5))
        else:
            det_array = np.array([[*d['bbox'], d['score']] for d in detection_list])

        high_score_indices = [i for i, d in enumerate(detection_list) if d['score'] >= self.high_thresh]
        low_score_indices = [i for i, d in enumerate(detection_list) if self.low_thresh <= d['score'] < self.high_thresh]
        
        high_score_dets = det_array[high_score_indices]
        low_score_dets = det_array[low_score_indices]

        active_tracks = [t for t in self.tracks if t.time_since_update <= 1]
        lost_tracks = [t for t in self.tracks if t.time_since_update > 1]
        
        # --- Step 3: First Association (High-score detections) ---
        matches_high, _, unmatched_dets_high_indices_rel = self._associate(active_tracks, high_score_dets)
        
        for track_idx, det_idx_rel in matches_high:
            track = active_tracks[track_idx]
            original_det_idx = high_score_indices[det_idx_rel]
            det = detection_list[original_det_idx]
            track.update(np.array(det['bbox']), det['score'], det['class_name'])

        # --- Step 4: Second Association (Low-score detections with remaining lost tracks) ---
        updated_track_ids = {active_tracks[t_idx].id for t_idx, _ in matches_high}
        unmatched_lost_tracks = [t for t in lost_tracks if t.id not in updated_track_ids]

        matches_low, _, _ = self._associate(unmatched_lost_tracks, low_score_dets)

        for track_idx, det_idx_rel in matches_low:
            track = unmatched_lost_tracks[track_idx]
            original_det_idx = low_score_indices[det_idx_rel]
            det = detection_list[original_det_idx]
            track.update(np.array(det['bbox']), det['score'], det['class_name'])

        # --- Step 5: Finalize Tracks ---
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Initialize new tracks from unmatched high-score detections
        for det_idx_rel in unmatched_dets_high_indices_rel:
            original_det_idx = high_score_indices[det_idx_rel]
            det = detection_list[original_det_idx]
            new_track = ByteTracker_Track(self.next_id, np.array(det['bbox']), det['score'], det['class_name'])
            self.colors[self.next_id] = self._generate_color(self.next_id)
            self.tracks.append(new_track)
            self.next_id += 1
            
        # --- Step 6: Format outputs ---
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
        
        # Result old is for getting vehicle type 'class_name' for lost vehicles -> to be stored to DB at _finalize_lost_tracks()
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

if __name__ == '__main__':
    print("--- Byte Tracker Test ---")

    byte_tracker = ByteTracker()
    
    # Frame 1: Object appears
    print("\n--- Frame 1 ---")
    detections_f1 = [{'bbox': [50, 50, 100, 100], 'score': 0.9, 'class_name': 'car'}]
    tracked_objects = byte_tracker.update(detections_f1)
    print(tracked_objects)

    # Frame 2: Object moves right
    print("\n--- Frame 2 ---")
    detections_f2 = [{'bbox': [60, 50, 110, 100], 'score': 0.9, 'class_name': 'car'}]
    tracked_objects = byte_tracker.update(detections_f2)
    print(tracked_objects)

    # Frame 3: No detection (occlusion).
    print("\n--- Frame 3 (No Detection) ---")
    detections_f3 = []
    tracked_objects = byte_tracker.update(detections_f3)
    print(f"  Output: {tracked_objects}")
    print(f"  Tracker has {len(byte_tracker.tracks)} track(s) internally.")
    for obj in byte_tracker.tracks: print(f"  Predicted BBox for Track {obj.id}: {obj.bbox.astype(int)}")

    # Frame 4: Object reappears with a low score.
    print("\n--- Frame 4 (Re-appears with low score) ---")
    detections_f4 = [{'bbox': [85, 50, 135, 100], 'score': 0.4, 'class_name': 'car'}]
    tracked_objects = byte_tracker.update(detections_f4)
    print(tracked_objects)
