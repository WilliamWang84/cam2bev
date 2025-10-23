"""
Traffic Camera to Bird's-Eye-View (BEV) Transformation System

This system detects and tracks vehicles in a traffic camera feed and projects
their positions onto a satellite/bird's-eye-view image.

Pipeline:
1. Vehicle Detection (YOLO)
2. Vehicle Tracking (ByteTrack)
3. Perspective Transformation (Homography)
4. BEV Visualization with bounding boxes
"""

import cv2
import json
import numpy as np

from byte_tracker import ByteTracker
from data_class import Vehicle, CalibrationPoint
from detector import YOLODetector
from transformer import PerspectiveTransformer
from visualizer import BEVVisualizer

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
from traffic_output_module import TrafficBEVOutput

class TrafficBEVSystem:
    """
    Main system integrating all components
    """
    
    def __init__(self, 
                 bev_image_path: str,
                 yolo_model_path: str = 'yolov8n.pt',
                 calibration_file: str = None,
                 camera_config: Dict = None, 
                 enable_output: bool = False,
                 db_path: str = 'traffic_tracks.db'                 
                 ):
        
        self.detector = YOLODetector(yolo_model_path)
        self.tracker = ByteTracker()
        self.transformer = PerspectiveTransformer()
        self.visualizer = BEVVisualizer(bev_image_path)
        
        # Load calibration if provided
        if calibration_file and Path(calibration_file).exists():
            self.transformer.load_calibration_from_file(calibration_file)
            self.transformer.compute_homography()
        
        # Store trajectories
        self.trajectories = defaultdict(list)
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0

        self.output_handler = None
        self.previous_tracks = set()  # Track which vehicles left the scene
        if enable_output and camera_config:
            self.output_handler = TrafficBEVOutput(
                camera_config, 
                db_path=db_path
            )

    def _finalize_lost_tracks(self, current_tracks: set, tracked_objects_old: Dict):
        """Save tracks that left the scene to database"""
        if not self.output_handler:
            return
        
        lost_tracks = self.previous_tracks - current_tracks

        for lost_id in lost_tracks:
            if lost_id in self.trajectories and len(self.trajectories[lost_id]) > 5:
                
                # Get vehicle type from last known data (you'll need to store this)
                vehicle_type = tracked_objects_old[lost_id]['class_name']
                
                # Save to database
                frame_numbers = list(range(len(self.trajectories[lost_id])))
                self.output_handler.save_track_history(
                    track_id=lost_id,
                    trajectory_points=self.trajectories[lost_id],
                    vehicle_type=vehicle_type,
                    frame_numbers=frame_numbers
                )
        
        self.previous_tracks = current_tracks

    # Method to generate GeoJSON output
    def generate_geojson_output(self, tracked_objects: Dict, output_file: str = None) -> Dict:
        """
        Generate GeoJSON from currently tracked vehicles
        
        Args:
            tracked_objects: Current tracked objects from self.tracker
            output_file: Optional path to save GeoJSON
        """
        if not self.output_handler:
            print("Output handler not initialized. Set enable_output=True")
            return None
        
        # Convert tracked objects to format expected by output handler
        tracks_data = []
        for obj_id, obj_data in tracked_objects.items():    
            if obj_id not in self.trajectories or len(self.trajectories[obj_id]) == 0:
                continue
            
            # Get latest BEV position
            bev_x, bev_y = self.trajectories[obj_id][-1]
            
            # Calculate speed and heading from trajectory
            speed = 0.0
            heading = 0.0
            if len(self.trajectories[obj_id]) >= 2:   
                prev = self.trajectories[obj_id][-2]
                curr = self.trajectories[obj_id][-1]
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                distance_pix = np.sqrt(dx**2 + dy**2)
                distance_mtr = distance_pix * self.output_handler.bev_meters_per_pixel
                time_diff = 1.0 / self.output_handler.video_fps
                speed = distance_mtr / time_diff
                heading = np.degrees(np.arctan2(dx, -dy)) % 360
            
            tracks_data.append({
                'id': obj_id,
                'bev_x': bev_x,
                'bev_y': bev_y,
                'class': obj_data['class_name'],
                'confidence': obj_data.get('confidence', 0.9),
                'speed': speed,
                'heading': heading
            })
        
        return self.output_handler.generate_geojson(tracks_data, output_file)
    
    # Method to get statistics
    def get_statistics(self, time_window_minutes: int = 60) -> Dict:
        """Get traffic statistics"""
        if not self.output_handler:
            print("Output handler not initialized")
            return {}
        
        return self.output_handler.get_statistics(time_window_minutes)
    

    def process_frame(self, 
                     frame: np.ndarray,
                     conf_threshold: float = 0.5,
                     visualize_camera: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process single frame
        
        Returns:
            camera_view, bev_view, tracked_objects
        """
        self.frame_count += 1
        
        # 1. Detect vehicles
        detections = self.detector.detect(frame, conf_threshold)
        self.total_detections += len(detections)
        
        # 2. Track vehicles
        tracked_objects, tracked_objects_old = self.tracker.update(detections)
        
        # 3. Transform to BEV and visualize
        self.visualizer.reset()
        
        camera_viz = frame.copy() if visualize_camera else None
        
        for obj_id, obj_data in tracked_objects.items():
            bbox = obj_data['bbox']
            color = obj_data['color']
            class_name = obj_data['class_name']
            
            # Draw on camera view
            if visualize_camera:
                [x1, y1, x2, y2] = [int(x) for x in bbox]
                 
                cv2.rectangle(camera_viz, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{obj_id} {class_name}"
                cv2.putText(camera_viz, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Transform to BEV
            try:
                # Get current trajectory for this vehicle
                current_trajectory = self.trajectories.get(obj_id, [])

                # Transform with trajectory-based orientation
                bev_center, bev_corners = self.transformer.transform_bbox(
                    bbox, 
                    transform_corners=False,
                    trajectory=current_trajectory,
                    use_oriented_rect=True
                )

                # Store trajectory
                self.trajectories[obj_id].append(tuple(bev_center.astype(int)))
                if len(self.trajectories[obj_id]) > 50:  # Keep last 50 points
                    self.trajectories[obj_id].pop(0)
                
                # Draw on BEV
                self.visualizer.draw_vehicle(
                    bev_center,
                    bev_corners,
                    obj_id,
                    color,
                    class_name
                )
                
                # Draw trajectory
                if len(self.trajectories[obj_id]) > 1:
                    self.visualizer.draw_trajectory(
                        self.trajectories[obj_id], color
                    )
            
            except Exception as e:
                # If transformation fails, skip this vehicle
                print(f"Warning: vehicle {obj_id} tranformation failed, skipped")
                continue

        current_tracks = set(tracked_objects.keys())
        self._finalize_lost_tracks(current_tracks, tracked_objects_old)

        self.visualizer.add_legend()
        bev_viz = self.visualizer.get_image()
        
        return camera_viz, bev_viz, tracked_objects
    
    def process_video(self, 
                     video_path: str,
                     output_path: str = 'output_bev.mp4',
                     conf_threshold: float = 0.5,
                     show_preview: bool = False):
        """
        Process entire video
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                             (self.visualizer.bev_image.shape[1], 
                              self.visualizer.bev_image.shape[0]))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Process frame
            camera_viz, bev_viz, tracked_objects = self.process_frame(frame, conf_threshold)
            
            # Write output
            out.write(bev_viz)
            
            # Show preview
            if show_preview:
                # Resize for display
                display_camera = cv2.resize(camera_viz, (640, 480))
                display_bev = cv2.resize(bev_viz, (640, 480))
                combined = np.hstack([display_camera, display_bev])
                
                cv2.imshow('Camera View | BEV', combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
                geojson = self.generate_geojson_output(
                    tracked_objects, 
                    output_file=f'./output/result_frame_{frame_idx}.geojson'
                )
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Output saved to: {output_path}")
        print(f"Average detections per frame: {self.total_detections / frame_idx:.2f}")
    
    def calibrate_interactive(self, sample_frame: np.ndarray):
        """
        Interactive calibration tool
        User clicks corresponding points in camera and BEV views
        """
        print("Interactive Calibration")
        print("Click corresponding points in both images")
        print("Press 'c' to compute homography, 'r' to reset, 'q' to quit")
        
        camera_points = []
        bev_points = []
        
        def camera_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                camera_points.append((x, y))
                cv2.circle(sample_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Camera View', sample_frame)
        
        def bev_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                bev_points.append((x, y))
                cv2.circle(self.visualizer.bev_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('BEV View', self.visualizer.bev_image)
        
        cv2.imshow('Camera View', sample_frame)
        cv2.imshow('BEV View', self.visualizer.bev_image)
        cv2.setMouseCallback('Camera View', camera_click)
        cv2.setMouseCallback('BEV View', bev_click)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                if len(camera_points) >= 4 and len(camera_points) == len(bev_points):
                    for cp, bp in zip(camera_points, bev_points):
                        self.transformer.add_calibration_point(cp, bp)
                    self.transformer.compute_homography()
                    print(f"Homography computed with {len(camera_points)} points")
                    break
                else:
                    print(f"Need at least 4 point pairs. Current: {len(camera_points)}")
            
            elif key == ord('r'):
                camera_points.clear()
                bev_points.clear()
                sample_frame = sample_frame.copy()
                self.visualizer.reset()
                cv2.imshow('Camera View', sample_frame)
                cv2.imshow('BEV View', self.visualizer.bev_image)
            
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()


# Example usage and calibration helper
def run():

    # Define camera configuration
    camera_config = {
        'id': 'cam_001',
        'location': {
            'lat': -33.8688,  # Your camera GPS location
            'lon': 151.2093,
            'alt': 15.0
        },
        # Optional: Add GPS transformation matrix 
        'bev_to_gps_transform': np.array([[-1.9604, -1.0042, 575.4273],[-2.1582, -7.2994, 3775.0186],[-0.0014, -0.0056, 1]]), # Matrix obtained from calibration_tools validate mode
        'bev_meters_per_pixel': 0.055, # Calibrated using the data provided (average of 7 segments) by the calibration_tools meter2pixel mode 
        'fps': 24
    }    

    # Initialize system WITH output enabled
    system = TrafficBEVSystem(
        bev_image_path='imageC.png',
        yolo_model_path='yolov8n.pt',
        calibration_file='calibration.json',
        camera_config=camera_config, 
        enable_output=True  
    )    
    
    # If calibration doesn't exist, run interactive calibration
    if not Path('calibration.json').exists():
        print("No calibration file found. Starting interactive calibration...")
        
        # Load a sample frame from video
        cap = cv2.VideoCapture('traffic_video.mp4')
        ret, sample_frame = cap.read()
        cap.release()
        
        if ret:
            system.calibrate_interactive(sample_frame)
            system.transformer.save_calibration('calibration.json')
        else:
            print("Could not load sample frame for calibration")
            return
    
    # Process video or rtsp stream
    system.process_video(
        video_path='traffic_video.mp4', # Or rtsp_link 
        output_path='output_bev.mp4',
        conf_threshold=0.05,
        show_preview=True
    )
    
    # Get statistics
    time_window_minutes=1
    stats = system.get_statistics(time_window_minutes=time_window_minutes)
    print(f"\nTraffic Statistics (last {time_window_minutes} min):")
    print(f"  Total vehicles: {stats['total_vehicles']}")
    print(f"  Vehicle counts: {stats['vehicle_counts']}")
    print(f"  Avg speed: {stats['avg_speed_kmh']:.1f} km/h")  
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    run()

