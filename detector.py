"""
Detector module, primarily use YOLO (ultralytics)
Enhanced with GPU/MPS device support
"""
import cv2
import numpy as np

from typing import List, Dict, Optional


class YOLODetector:
    """
    YOLO-based vehicle detector
    Supports YOLOv8, YOLOv5, or YOLO with OpenCV DNN
    Auto-detects GPU (CUDA), Apple Silicon (MPS), or falls back to CPU
    """
    
    def __init__(
        self, 
        model_path: str = None, 
        use_ultralytics: bool = True,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to model weights (None for default yolov8n.pt)
            use_ultralytics: Use ultralytics YOLO (recommended) vs OpenCV DNN
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            verbose: Print device information
        """
        self.use_ultralytics = use_ultralytics
        self.model = None
        self.device = device
        self.verbose = verbose
        
        if use_ultralytics:
            self._init_ultralytics(model_path)
        elif model_path:
            self._init_opencv_dnn(model_path)
        else:
            raise ValueError("model_path required when use_ultralytics=False")
            
        self.load_class_names()
    
    def _init_ultralytics(self, model_path: str):
        """Initialize Ultralytics YOLO with device detection"""
        try:
            from ultralytics import YOLO
            import torch
            
            # Detect and set device
            detected_device = self._detect_device(torch)
            
            # Use specified device or auto-detected one
            self.device = self.device or detected_device
            
            # Load model
            self.model = YOLO(model_path or 'yolov8n.pt')
            
            # Move model to device
            self.model.to(self.device)
            
            if self.verbose:
                print(f"✓ Ultralytics YOLO initialized on: {self._get_device_name(torch)}")
                
        except ImportError as e:
            print(f"✗ Ultralytics not installed: {e}")
            print("  Install with: pip install ultralytics")
            self.use_ultralytics = False
        except Exception as e:
            print(f"✗ Error initializing Ultralytics YOLO: {e}")
            self.use_ultralytics = False
    
    def _init_opencv_dnn(self, model_path: str):
        """Initialize OpenCV DNN with GPU support if available"""
        try:
            self.net = cv2.dnn.readNet(model_path)
            
            # Try to enable CUDA backend
            cuda_available = False
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                cuda_available = True
                if self.verbose:
                    print("✓ OpenCV DNN initialized with CUDA backend")
            except:
                if self.verbose:
                    print("✓ OpenCV DNN initialized with CPU backend")
                    
        except Exception as e:
            print(f"✗ Error loading OpenCV DNN model: {e}")
            raise
    
    def _detect_device(self, torch) -> str:
        """Detect best available device"""
        if self.device:
            # User specified device - validate it
            if self.device == 'cuda' and not torch.cuda.is_available():
                print(f"⚠ CUDA requested but not available, falling back to CPU")
                return 'cpu'
            if self.device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print(f"⚠ MPS requested but not available, falling back to CPU")
                return 'cpu'
            return self.device
        
        # Auto-detect best device
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _get_device_name(self, torch) -> str:
        """Get human-readable device name"""
        if self.device == 'cuda':
            try:
                gpu_name = torch.cuda.get_device_name(0)
                return f"CUDA GPU ({gpu_name})"
            except:
                return "CUDA GPU"
        elif self.device == 'mps':
            return "Apple MPS (Metal Performance Shaders)"
        else:
            return "CPU"
    
    def get_device_info(self) -> Dict[str, any]:
        """
        Get detailed device information
        
        Returns:
            Dictionary with device capabilities and current device
        """
        info = {
            'current_device': self.device,
            'ultralytics_mode': self.use_ultralytics
        }
        
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            info['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if info['cuda_available']:
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['cuda_device_count'] = torch.cuda.device_count()
            
            if info['mps_available']:
                info['mps_device_name'] = "Apple Silicon GPU"
                
        except ImportError:
            info['torch_available'] = False
            
        return info
    
    def load_class_names(self):
        """Load COCO class names"""
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket'
        ]
        # Vehicle classes we care about
        self.vehicle_classes = {'car', 'bus', 'truck', 'motorcycle'}
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input image/frame (numpy array)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detections with format:
            {
                'bbox': [x1, y1, x2, y2], 
                'score': float, 
                'class_name': str,
                'cv_img': cropped vehicle image (224x224)
            }
        """
        if self.use_ultralytics and self.model:
            return self._detect_ultralytics(frame, conf_threshold)
        else:
            return self._detect_opencv(frame, conf_threshold)
    
    def _detect_ultralytics(self, frame: np.ndarray, conf_threshold: float) -> List[Dict]:
        """Detect using Ultralytics YOLO with GPU/MPS support"""
        # Run inference - device is already set in model
        results = self.model(
            frame, 
            conf=conf_threshold, 
            verbose=False,
            device=self.device  # Explicitly pass device
        )[0]
        
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            
            # Filter for vehicles only
            if class_name in self.vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Crop and resize vehicle image
                frame_crop = frame[y1:y2, x1:x2]
                if frame_crop.size > 0:  # Ensure valid crop
                    frame_crop = cv2.resize(frame_crop, (224, 224))
                else:
                    frame_crop = np.zeros((224, 224, 3), dtype=np.uint8)
                
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': conf,
                    'class_name': class_name,
                    'cv_img': frame_crop
                })

        return detections
    
    def _detect_opencv(self, frame: np.ndarray, conf_threshold: float) -> List[Dict]:
        """Detect using OpenCV DNN (fallback with CUDA support if available)"""
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        # Process outputs
        detections = []
        h, w = frame.shape[:2]
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                score = scores[class_id]
                
                if score > conf_threshold and class_id < len(self.classes):
                    if self.classes[class_id] in self.vehicle_classes:
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        width = int(detection[2] * w)
                        height = int(detection[3] * h)
                        
                        x1 = int(center_x - width / 2)
                        y1 = int(center_y - height / 2)
                        x2 = x1 + width
                        y2 = y1 + height
                        
                        # Crop and resize vehicle image
                        frame_crop = frame[y1:y2, x1:x2]
                        if frame_crop.size > 0:
                            frame_crop = cv2.resize(frame_crop, (224, 224))
                        else:
                            frame_crop = np.zeros((224, 224, 3), dtype=np.uint8)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': float(score),
                            'class_name': self.classes[class_id],
                            'cv_img': frame_crop
                        })
        
        return detections
    
    def benchmark_device(self, frame: np.ndarray, iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark detection speed on current device
        
        Args:
            frame: Test frame
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.detect(frame)
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'fps': 1.0 / np.mean(times),
            'device': self.device
        }


# Example usage
if __name__ == "__main__":
    # Auto-detect best device
    detector = YOLODetector()
    
    # Print device info
    print("\n=== Device Information ===")
    info = detector.get_device_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Or explicitly specify device
    # detector = YOLODetector(device='cuda')  # Force CUDA
    # detector = YOLODetector(device='mps')   # Force MPS (Apple Silicon)
    # detector = YOLODetector(device='cpu')   # Force CPU
    
    # Test detection
    test_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    detections = detector.detect(test_frame)
    print(f"\n=== Test Detection ===")
    print(f"Found {len(detections)} vehicles")
    
    # Benchmark
    print("\n=== Performance Benchmark ===")
    benchmark = detector.benchmark_device(test_frame, iterations=10)
    print(f"Device: {benchmark['device']}")
    print(f"Mean inference time: {benchmark['mean_time']*1000:.2f}ms")
    print(f"FPS: {benchmark['fps']:.2f}")
