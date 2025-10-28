"""
Detector module, primarily use YOLO (ultralytics)
"""
import cv2
import numpy as np

from typing import List, Dict


class YOLODetector:
    """
    YOLO-based vehicle detector
    Supports YOLOv8, YOLOv5, or YOLO with OpenCV DNN
    """
    
    def __init__(self, model_path: str = None, use_ultralytics: bool = True):
        self.use_ultralytics = use_ultralytics
        self.model = None
        
        if use_ultralytics:
            # Using Ultralytics YOLOv8 (recommended)
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path or 'yolov8n.pt')
            except ImportError:
                print("Ultralytics not installed. Install: pip install ultralytics")
                self.use_ultralytics = False
        
        if not self.use_ultralytics and model_path:
            # Fallback to OpenCV DNN
            self.net = cv2.dnn.readNet(model_path)
            
        self.load_class_names()
    
    def load_class_names(self):
        """Load COCO class names"""
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog'
        ]
        # Vehicle classes we care about
        self.vehicle_classes = {'car', 'bus', 'motorcycle'}
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect vehicles in frame
        
        Returns:
            List of detections with format:
            {'bbox': [x1, y1, x2, y2], 'score': float, 'class_name': str}
        """
        if self.use_ultralytics and self.model:
            return self._detect_ultralytics(frame, conf_threshold)
        else:
            return self._detect_opencv(frame, conf_threshold)
    
    def _detect_ultralytics(self, frame: np.ndarray, conf_threshold: float) -> List[Dict]:
        """Detect using Ultralytics YOLO"""
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            
            # Filter for vehicles only
            if class_name in self.vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                frame_crop = frame[y1:y2, x1:x2]
                frame_crop = cv2.resize(frame_crop, (224, 224))
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': conf,
                    'class_name': class_name,
                    'cv_img': frame_crop
                })

        return detections
    
    def _detect_opencv(self, frame: np.ndarray, conf_threshold: float) -> List[Dict]:
        """Detect using OpenCV DNN (fallback)"""
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        # Process outputs (simplified)
        detections = []
        h, w = frame.shape[:2]
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                score = scores[class_id]
                
                if score > conf_threshold and self.classes[class_id] in self.vehicle_classes:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    
                    x1 = int(center_x - width / 2)
                    y1 = int(center_y - height / 2)
                    
                    detections.append({
                        'bbox': [x1, y1, x1 + width, y1 + height],
                        'score': score,
                        'class_name': self.classes[class_id]
                    })
        
        return detections
