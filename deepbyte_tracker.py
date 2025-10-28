"""
DeepByte Tracker Module - Batch ops optimized, Supports a few feature extraction model options
"""

import copy
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from typing import List, Dict, Tuple, Optional

# =============================================================================
# DeepByteTracker Implementation (combination of Deepsort and ByteTracker)
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

def cosine_distance(a: np.ndarray, b: np.ndarray, data_is_normalized: bool = False) -> np.ndarray:
    """
    Computes cosine distance between two feature vectors or matrices.
    Returns a distance matrix of shape (len(a), len(b))
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    
    if not data_is_normalized:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    
    return 1. - np.dot(a, b.T)


# --- Feature Extractor (Multiple Model Support) ---
class FeatureExtractor:
    """
    Multi-model feature extractor supporting ResNet50, MobileNetV3, and EfficientNet-B0.
    
    Performance comparison (approximate):
    - ResNet50:        100-200ms/batch on CPU, 10-20ms on GPU, 2048 features (speed benchmark)
    - MobileNetV3:     20-40ms/batch on CPU, 2-5ms on GPU, 1280 features  (5-10x faster)
    - EfficientNet-B0: 30-60ms/batch on CPU, 3-8ms on GPU, 1280 features  (3-7x faster)
    """
    
    SUPPORTED_MODELS = ['resnet50', 'mobilenetv3', 'efficientnet_b0']
    
    def __init__(self, model_name: str = 'mobilenetv3', device: Optional[str] = None):
        """
        Initialize feature extractor with specified model.
        
        Args:
            model_name: Model to use ('resnet50', 'mobilenetv3', 'efficientnet_b0')
            device: Device to run model on ('cuda' or 'cpu'). Auto-detects if None.
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        # Device support priority: Linux (or orchestrated docker container) | Windows (GPU / cuda) >= Mac (mps) >= Others (cpu) 
        self.device = device if device else ('cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu'))
        
        # Load model based on selection
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.feature_extractor = torch.nn.Sequential(*(list(self.model.children())[:-1]))
            self.feature_dim = 2048
            img_size = 224
            
        elif model_name == 'mobilenetv3':
            self.model = models.mobilenet_v3_large(pretrained=True)
            # MobileNetV3: features are from classifier[0] (avgpool output)
            self.feature_extractor = torch.nn.Sequential(
                self.model.features,
                self.model.avgpool,
            )
            self.feature_dim = 960  # MobileNetV3-Large feature dimension
            img_size = 224
            
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            # EfficientNet: features are from avgpool output
            self.feature_extractor = torch.nn.Sequential(
                self.model.features,
                self.model.avgpool,
            )
            self.feature_dim = 1280  # EfficientNet-B0 feature dimension
            img_size = 224
        
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        
        # Preprocessing pipeline (standard ImageNet normalization)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"[FeatureExtractor] Initialized {model_name} on {self.device} (feature_dim={self.feature_dim})")
    
    def extract(self, img_pil: Image) -> np.ndarray:
        """Extract features from PIL image."""
        img_tensor = self.preprocess(img_pil)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        
        # Flatten features to 1D vector
        features = features.squeeze().cpu().numpy()
        if features.ndim == 0:  # Handle scalar case
            features = features.reshape(1)
        
        return features

    def extract_cv(self, img_cv: np.ndarray) -> np.ndarray:
        """Extract features from OpenCV image (BGR format)."""
        cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(cv_rgb)
        return self.extract(img_pil)
    
    def extract_batch(self, img_cv_list: List[np.ndarray]) -> np.ndarray:
        """
        Extract features for multiple images in batch for better GPU utilization.
        
        Args:
            img_cv_list: List of OpenCV images (BGR format)
            
        Returns:
            numpy array of shape (N, feature_dim)
        """
        if len(img_cv_list) == 0:
            return np.array([])
        
        # Convert all images to PIL and preprocess
        tensors = []
        for img_cv in img_cv_list:
            cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(cv_rgb)
            tensors.append(self.preprocess(img_pil))
        
        # Stack into batch
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(batch)
        
        # Flatten features and convert to numpy
        features = features.view(features.size(0), -1)  # Flatten to (N, feature_dim)
        features = features.cpu().numpy()
        
        return features


# Global feature extractor instance
# Options: 'resnet50', 'mobilenetv3', 'efficientnet_b0'
# Default to MobileNetV3 for best speed/accuracy tradeoff
feature_extractor = FeatureExtractor(model_name='mobilenetv3')


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


class DeepByteTracker_Track:
    """Represents a single tracked object with its Kalman Filter."""
    
    MAX_FEATURE_HISTORY = 100  # Limit feature gallery size for memory efficiency
    
    def __init__(self, track_id: int, bbox: np.ndarray, score: float, class_name: str, features: np.ndarray):
        self.id = track_id
        self.kf = KalmanFilterTracker(bbox)
        self.score = score
        self.class_name = class_name
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.state = 'tentative'
        
        # Store normalized features for efficient cosine distance computation
        normalized_features = features / (np.linalg.norm(features) + 1e-8)
        self.features = [normalized_features]

    @property
    def bbox(self) -> np.ndarray:
        """Returns the current bounding box estimate from the Kalman Filter."""
        return bbox_xywh_to_xyxy(self.kf.kf.x[:4].flatten())

    def predict(self):
        """Advances the state vector and returns the predicted bounding box."""
        self.age += 1
        return self.kf.predict()

    def update(self, bbox: np.ndarray, score: float, class_name: str, features: np.ndarray, min_hits: int = 3):
        """Updates the Kalman Filter with a new detection."""
        self.kf.update(bbox)
        self.score = score
        self.class_name = class_name
        self.time_since_update = 0
        self.hits += 1
        
        # Store normalized features
        normalized_features = features / (np.linalg.norm(features) + 1e-8)
        self.features.append(normalized_features)
        
        # Limit feature history to save memory
        if len(self.features) > self.MAX_FEATURE_HISTORY:
            self.features.pop(0)
        
        # Update state
        if self.state == 'tentative' and self.hits >= min_hits:
            self.state = 'confirmed'


class DeepByteTracker:
    """The full ByteTrack algorithm with Kalman Filter, Hungarian matching, and DeepSORT appearance features."""
    
    def __init__(self, 
                 high_thresh: float = 0.5, 
                 low_thresh: float = 0.1, 
                 iou_thresh: float = 0.7, 
                 max_age: int = 100, 
                 appearance_thresh: float = 0.4,
                 iou_weight: float = 0.7,
                 appearance_weight: float = 0.3,
                 feature_extractor: Optional[FeatureExtractor] = None):
        """
        Initialize DeepByteTracker.
        
        Args:
            high_thresh: High confidence threshold for detections
            low_thresh: Low confidence threshold for detections
            iou_thresh: IoU threshold for matching
            max_age: Maximum frames to keep track without detection
            appearance_thresh: Appearance cost threshold for matching
            iou_weight: Weight for IoU cost in combined matching
            appearance_weight: Weight for appearance cost in combined matching
            feature_extractor: Custom feature extractor (uses global default if None)
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.appearance_thresh = appearance_thresh
        self.iou_weight = iou_weight
        self.appearance_weight = appearance_weight
        self.tracks = []
        self.next_id = 1
        self.colors = {}
        
        # Use provided feature extractor or global default
        self.feature_extractor = feature_extractor if feature_extractor else feature_extractor

    def _generate_color(self, obj_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for object ID"""
        np.random.seed(obj_id)
        return tuple(np.random.randint(50, 255, 3).tolist())

    def _associate_deep(self, 
                       tracks: List[DeepByteTracker_Track], 
                       detections: np.ndarray,
                       detection_features: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Perform data association with weighted combination of IoU and appearance costs.
        
        Args:
            tracks: List of tracks to match
            detections: Detection array (N, 5) with [x1, y1, x2, y2, score]
            detection_features: Pre-extracted features for detections (N, feature_dim)
            
        Returns:
            matches, unmatched_track_indices, unmatched_detection_indices
        """
        if not tracks or detections.shape[0] == 0:
            return [], list(range(len(tracks))), list(range(detections.shape[0]))

        # Compute IoU matrix
        track_bboxes = torch.from_numpy(np.array([t.bbox for t in tracks])).float()
        det_bboxes = torch.from_numpy(detections[:, :4]).float()
        iou_matrix = box_iou(track_bboxes, det_bboxes).numpy()

        # Compute appearance cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for t_idx, track in enumerate(tracks):
            # Use the most recent feature (already normalized)
            track_feature = track.features[-1].reshape(1, -1)
            
            for d_idx in range(len(detections)):
                # Detection features are already normalized
                det_feature = detection_features[d_idx].reshape(1, -1)
                
                # Compute cosine distance (already normalized, so faster)
                app_cost = cosine_distance(track_feature, det_feature, data_is_normalized=True)[0, 0]
                
                # Combined cost: weighted sum of IoU and appearance
                cost_matrix[t_idx, d_idx] = (1 - iou_matrix[t_idx, d_idx]) * self.iou_weight + app_cost * self.appearance_weight

        # Hungarian algorithm for assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_track_indices = set(range(len(tracks)))
        unmatched_det_indices = set(range(len(detections)))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] < self.appearance_thresh:
                matches.append((track_idx, det_idx))
                unmatched_track_indices.discard(track_idx)
                unmatched_det_indices.discard(det_idx)

        return matches, list(unmatched_track_indices), list(unmatched_det_indices)

    def _associate(self, 
                  tracks: List[DeepByteTracker_Track], 
                  detections: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Performs data association using only IoU and the Hungarian algorithm.
        
        Args:
            tracks: List of tracks to match
            detections: Detection array (N, 5) with [x1, y1, x2, y2, score]
            
        Returns:
            matches, unmatched_track_indices, unmatched_detection_indices
        """
        if not tracks or detections.shape[0] == 0:
            return [], list(range(len(tracks))), list(range(detections.shape[0]))

        track_bboxes = torch.from_numpy(np.array([t.bbox for t in tracks])).float()
        det_bboxes = torch.from_numpy(detections[:, :4]).float()
        
        iou_matrix = box_iou(track_bboxes, det_bboxes).numpy()
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

    def update(self, detections: List[Dict]) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries or dict of dicts with keys:
                       'bbox': [x1, y1, x2, y2]
                       'score': confidence score
                       'class_name': object class
                       'cv_img': OpenCV image crop (should be 224x224)
        
        Returns:
            Tuple of (current_tracks, old_tracks) where each is Dict[track_id -> track_info]
        """
        # Store old track info before updates (only what's needed for lost track finalization)
        old_track_info = {
            t.id: {
                'id': t.id,
                'bbox': t.bbox.copy(),
                'color': self.colors.get(t.id, (0, 0, 255)),
                'class_name': t.class_name,
                'score': t.score
            }
            for t in self.tracks
        }

        # --- Step 1: Predict new locations ---
        for track in self.tracks:
            track.predict()
            track.time_since_update += 1

        # --- Step 2: Format detections and extract features once ---
        detection_list = list(detections.values()) if isinstance(detections, dict) else detections

        if len(detection_list) == 0:
            det_array = np.empty((0, 5))
            all_features = np.empty((0, self.feature_extractor.feature_dim))
        else:
            # Build detection array (no image data, just bbox + score)
            det_array = np.array([[*d['bbox'], d['score']] for d in detection_list])
            
            # Batch extract features for all detections - MUCH FASTER
            all_features = self.feature_extractor.extract_batch([d['cv_img'] for d in detection_list])
            
            # Normalize all features once
            all_features = all_features / (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8)

        # Separate high and low score detections
        high_score_mask = det_array[:, 4] >= self.high_thresh
        low_score_mask = (det_array[:, 4] >= self.low_thresh) & (det_array[:, 4] < self.high_thresh)
        
        high_score_indices = np.where(high_score_mask)[0].tolist()
        low_score_indices = np.where(low_score_mask)[0].tolist()
        
        high_score_dets = det_array[high_score_mask]
        low_score_dets = det_array[low_score_mask]
        high_score_features = all_features[high_score_mask] if len(all_features) > 0 else np.empty((0, self.feature_extractor.feature_dim))

        # Separate tracks by state and activity
        confirmed_tracks = [t for t in self.tracks if t.state == 'confirmed']
        unconfirmed_tracks = [t for t in self.tracks if t.state == 'tentative']
        active_tracks = [t for t in self.tracks if t.time_since_update <= 1]
        lost_tracks = [t for t in self.tracks if t.time_since_update > 1]
        
        # --- Step 3: First Association (High-score detections with active tracks using deep matching) ---
        matches_high, _, unmatched_dets_high_indices_rel = self._associate_deep(
            active_tracks, high_score_dets, high_score_features
        )
        
        for track_idx, det_idx_rel in matches_high:
            track = active_tracks[track_idx]
            original_det_idx = high_score_indices[det_idx_rel]
            det = detection_list[original_det_idx]
            track.update(
                np.array(det['bbox']), 
                det['score'], 
                det['class_name'], 
                all_features[original_det_idx]
            )

        # --- Step 4: Second Association (Low-score detections with remaining lost tracks) ---
        updated_track_ids = {active_tracks[t_idx].id for t_idx, _ in matches_high}
        unmatched_lost_tracks = [t for t in lost_tracks if t.id not in updated_track_ids]

        matches_low, _, _ = self._associate(unmatched_lost_tracks, low_score_dets)

        for track_idx, det_idx_rel in matches_low:
            track = unmatched_lost_tracks[track_idx]
            original_det_idx = low_score_indices[det_idx_rel]
            det = detection_list[original_det_idx]
            track.update(
                np.array(det['bbox']), 
                det['score'], 
                det['class_name'], 
                all_features[original_det_idx]
            )

        # --- Step 5: Third Association (Unconfirmed tracks with remaining high-score detections) ---
        # Get unmatched high score detections from step 3
        unmatched_high_score_dets = high_score_dets[unmatched_dets_high_indices_rel]
        unmatched_high_score_features = high_score_features[unmatched_dets_high_indices_rel]
        
        matches_unconfirmed, _, unmatched_dets_final_rel = self._associate(
            unconfirmed_tracks, unmatched_high_score_dets
        )

        for track_idx, det_idx_rel in matches_unconfirmed:
            track = unconfirmed_tracks[track_idx]
            # Map back to original detection index
            original_det_idx = high_score_indices[unmatched_dets_high_indices_rel[det_idx_rel]]
            det = detection_list[original_det_idx]
            track.update(
                np.array(det['bbox']), 
                det['score'], 
                det['class_name'], 
                all_features[original_det_idx]
            )

        # --- Step 6: Finalize Tracks ---
        # Remove tracks that are too old
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Initialize new tracks from remaining unmatched high-score detections
        for det_idx_rel in unmatched_dets_final_rel:
            # Map back to original detection index
            original_det_idx = high_score_indices[unmatched_dets_high_indices_rel[det_idx_rel]]
            det = detection_list[original_det_idx]
            
            new_track = DeepByteTracker_Track(
                self.next_id, 
                np.array(det['bbox']), 
                det['score'], 
                det['class_name'], 
                all_features[original_det_idx]
            )
            self.colors[self.next_id] = self._generate_color(self.next_id)
            self.tracks.append(new_track)
            self.next_id += 1
            
        # --- Step 7: Format outputs ---
        result = {}
        for track in self.tracks:
            if track.time_since_update == 0:
                result[track.id] = {
                    'id': track.id,
                    'bbox': track.bbox,
                    'color': self.colors.get(track.id, (0, 0, 255)),
                    'class_name': track.class_name,
                    'score': track.score,
                }
        
        return result, old_track_info


if __name__ == '__main__':
    print("--- Deep Byte Tracker Test ---")
    print("\nTesting with different feature extractors...\n")
    
    # Test 1: MobileNetV3 (Default - Fastest)
    print("=" * 60)
    print("Test 1: MobileNetV3 (Recommended for real-time)")
    print("=" * 60)
    mobilenet_fe = FeatureExtractor(model_name='mobilenetv3')
    tracker_mobile = DeepByteTracker(feature_extractor=mobilenet_fe)
    
    detections_f1 = [{'bbox': [50, 50, 100, 100], 'score': 0.9, 'class_name': 'car', 
                      'cv_img': np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)}]
    tracked_objects, _ = tracker_mobile.update(detections_f1)
    print(f"Frame 1 - Tracked objects: {list(tracked_objects.keys())}")
    
    detections_f2 = [{'bbox': [60, 50, 110, 100], 'score': 0.9, 'class_name': 'car', 
                      'cv_img': np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)}]
    tracked_objects, _ = tracker_mobile.update(detections_f2)
    print(f"Frame 2 - Tracked objects: {list(tracked_objects.keys())}")
    
    # Test 2: EfficientNet-B0 (Good balance)
    print("\n" + "=" * 60)
    print("Test 2: EfficientNet-B0 (Good speed/accuracy balance)")
    print("=" * 60)
    efficientnet_fe = FeatureExtractor(model_name='efficientnet_b0')
    tracker_efficient = DeepByteTracker(feature_extractor=efficientnet_fe)
    
    tracked_objects, _ = tracker_efficient.update(detections_f1)
    print(f"Frame 1 - Tracked objects: {list(tracked_objects.keys())}")
    
    tracked_objects, _ = tracker_efficient.update(detections_f2)
    print(f"Frame 2 - Tracked objects: {list(tracked_objects.keys())}")
    
    # Test 3: ResNet50 (Most accurate but slowest)
    print("\n" + "=" * 60)
    print("Test 3: ResNet50 (Highest accuracy, slower)")
    print("=" * 60)
    resnet_fe = FeatureExtractor(model_name='resnet50')
    tracker_resnet = DeepByteTracker(feature_extractor=resnet_fe)
    
    tracked_objects, _ = tracker_resnet.update(detections_f1)
    print(f"Frame 1 - Tracked objects: {list(tracked_objects.keys())}")
    
    tracked_objects, _ = tracker_resnet.update(detections_f2)
    print(f"Frame 2 - Tracked objects: {list(tracked_objects.keys())}")
    
    # Test occlusion handling
    print("\n" + "=" * 60)
    print("Test 4: Occlusion handling (using MobileNetV3)")
    print("=" * 60)
    detections_f3 = []
    tracked_objects, _ = tracker_mobile.update(detections_f3)
    print(f"Frame 3 (no detection) - Active tracks: {len(tracker_mobile.tracks)}")
    
    detections_f4 = [{'bbox': [85, 50, 135, 100], 'score': 0.4, 'class_name': 'car', 
                      'cv_img': np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)}]
    tracked_objects, _ = tracker_mobile.update(detections_f4)
    print(f"Frame 4 (low score reappear) - Tracked objects: {list(tracked_objects.keys())}")
    
    print("\n" + "=" * 60)
    print("Recommendation:")
    print("  - Real-time (30+ FPS): Use 'mobilenetv3'")
    print("  - Balanced (15-30 FPS): Use 'efficientnet_b0'")
    print("  - Max accuracy (<15 FPS): Use 'resnet50'")
    print("=" * 60)
    print("\n--- Test Complete ---")
