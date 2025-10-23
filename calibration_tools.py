"""
Calibration Helper Tools for Camera-to-BEV Transformation

This module provides tools for:
1. Extracting measurements from Google Earth KML
2. Interactive calibration point selection
3. Homography quality assessment
4. Automatic calibration refinement
"""

import cv2
import json
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from xml.etree import ElementTree as ET


class KMLParser:
    """Parse KML files from Google Earth to extract measurements"""
    
    def __init__(self, kml_path: str):
        self.kml_path = kml_path
        self.placemarks = []
    
    def parse(self) -> List[Dict]:
        """
        Parse KML file and extract placemark coordinates
        
        Returns:
            List of placemarks with names and coordinates
        """
        tree = ET.parse(self.kml_path)
        root = tree.getroot()
        
        # Define namespace
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        placemarks = []
        
        for placemark in root.findall('.//kml:Placemark', ns):
            name_elem = placemark.find('kml:name', ns)
            name = name_elem.text if name_elem is not None else "Unnamed"
            
            # Try to find coordinates
            coords_elem = placemark.find('.//kml:coordinates', ns)
            if coords_elem is not None:
                coords_text = coords_elem.text.strip()
                coords_list = []
                
                # Parse coordinates (format: lon,lat,alt)
                for coord in coords_text.split():
                    parts = coord.split(',')
                    if len(parts) >= 2:
                        lon, lat = float(parts[0]), float(parts[1])
                        coords_list.append((lon, lat))
                
                if coords_list:
                    placemarks.append({
                        'name': name,
                        'coordinates': coords_list,
                        'type': 'point' if len(coords_list) == 1 else 'line'
                    })
        
        self.placemarks = placemarks
        return placemarks
    
    def get_distance_measurements(self) -> List[Dict]:
        """
        Extract distance measurements from line placemarks
        Useful for scale calibration
        """
        distances = []
        
        for pm in self.placemarks:
            if pm['type'] == 'line' and len(pm['coordinates']) >= 2:
                # Calculate distance using Haversine formula
                coords = pm['coordinates']
                total_distance = 0
                
                for i in range(len(coords) - 1):
                    lon1, lat1 = coords[i]
                    lon2, lat2 = coords[i + 1]
                    
                    # Simplified distance (for small distances)
                    # For accurate results, use proper Haversine
                    dx = (lon2 - lon1) * 111320 * np.cos(np.radians(lat1))
                    dy = (lat2 - lat1) * 110540
                    dist = np.sqrt(dx**2 + dy**2)
                    total_distance += dist
                
                distances.append({
                    'name': pm['name'],
                    'distance_meters': total_distance,
                    'start': coords[0],
                    'end': coords[-1]
                })
        
        return distances


class CalibrationTool:
    """
    Interactive tool for selecting calibration points
    """
    
    def __init__(self, camera_frame: np.ndarray, bev_image: np.ndarray):
        self.camera_frame = camera_frame.copy()
        self.bev_image = bev_image.copy()
        self.camera_display = camera_frame.copy()
        self.bev_display = bev_image.copy()
        
        self.camera_points = []
        self.bev_points = []
        self.point_descriptions = []
        
        self.current_mode = 'camera'  # 'camera' or 'bev'
        self.point_counter = 0
    
    def run(self) -> Tuple[List[Tuple], List[Tuple], List[str]]:
        """
        Run interactive calibration
        
        Returns:
            camera_points, bev_points, descriptions
        """
        print("\n" + "="*60)
        print("INTERACTIVE CALIBRATION TOOL")
        print("="*60)
        print("\nInstructions:")
        print("1. Click on a landmark in the CAMERA view")
        print("2. Click the SAME landmark in the BEV view")
        print("3. Repeat for at least 4 corresponding points")
        print("\nGood landmarks: road intersections, lane markings,")
        print("                building corners, sign posts")
        print("\nControls:")
        print("  [Space] - Switch between camera and BEV windows")
        print("  [u]     - Undo last point")
        print("  [c]     - Compute homography (needs 4+ points)")
        print("  [s]     - Save calibration")
        print("  [r]     - Reset all points")
        print("  [q]     - Quit without saving")
        print("="*60 + "\n")
        
        cv2.namedWindow('Camera View')
        cv2.namedWindow('BEV View')
        cv2.setMouseCallback('Camera View', self._camera_click)
        cv2.setMouseCallback('BEV View', self._bev_click)
        
        self._update_displays()
        
        while True:
            cv2.imshow('Camera View', self.camera_display)
            cv2.imshow('BEV View', self.bev_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                self.current_mode = 'bev' if self.current_mode == 'camera' else 'camera'
                print(f"Switched to {self.current_mode.upper()} mode")
            
            elif key == ord('u'):
                self._undo_last()
            
            elif key == ord('c'):
                if self._validate_points():
                    print("\n✓ Calibration points valid")
                    return self.camera_points, self.bev_points, self.point_descriptions
            
            elif key == ord('s'):
                self._save_calibration('calibration_backup.json')
            
            elif key == ord('r'):
                self._reset()
            
            elif key == ord('q'):
                print("Calibration cancelled")
                cv2.destroyAllWindows()
                return [], [], []
        
        cv2.destroyAllWindows()
    
    def _camera_click(self, event, x, y, flags, param):
        """Handle mouse clicks on camera view"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.camera_points) == len(self.bev_points):
                self.camera_points.append((x, y))
                self.point_counter += 1
                print(f"Point {self.point_counter}: Camera ({x}, {y})")
                self._update_displays()
    
    def _bev_click(self, event, x, y, flags, param):
        """Handle mouse clicks on BEV view"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.bev_points) < len(self.camera_points):
                self.bev_points.append((x, y))
                print(f"Point {self.point_counter}: BEV ({x}, {y})")
                
                # Ask for description
                desc = input(f"Description for point {self.point_counter} (or press Enter): ")
                self.point_descriptions.append(desc if desc else f"Point {self.point_counter}")
                
                self._update_displays()
    
    def _update_displays(self):
        """Update display images with marked points"""
        self.camera_display = self.camera_frame.copy()
        self.bev_display = self.bev_image.copy()
        
        # Draw camera points
        for i, pt in enumerate(self.camera_points):
            color = (0, 255, 0) if i < len(self.bev_points) else (0, 165, 255)
            cv2.circle(self.camera_display, pt, 5, color, -1)
            cv2.circle(self.camera_display, pt, 7, (255, 255, 255), 2)
            cv2.putText(self.camera_display, str(i+1), 
                       (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw BEV points
        for i, pt in enumerate(self.bev_points):
            color = (0, 255, 0)
            cv2.circle(self.bev_display, pt, 5, color, -1)
            cv2.circle(self.bev_display, pt, 7, (255, 255, 255), 2)
            cv2.putText(self.bev_display, str(i+1),
                       (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw connecting lines if we have pairs
        if len(self.camera_points) > 1 and len(self.camera_points) == len(self.bev_points):
            cam_pts = np.array(self.camera_points, dtype=np.int32)
            bev_pts = np.array(self.bev_points, dtype=np.int32)
            cv2.polylines(self.camera_display, [cam_pts], False, (255, 0, 0), 1)
            cv2.polylines(self.bev_display, [bev_pts], False, (255, 0, 0), 1)
        
        # Add status text
        status_cam = f"Camera: {len(self.camera_points)} points"
        status_bev = f"BEV: {len(self.bev_points)} points"
        
        cv2.putText(self.camera_display, status_cam, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.bev_display, status_bev, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _undo_last(self):
        """Undo last point pair"""
        if self.bev_points:
            self.bev_points.pop()
            self.point_descriptions.pop()
            self.camera_points.pop()
            self.point_counter -= 1
            print(f"Undone point {self.point_counter + 1}")
            self._update_displays()
    
    def _reset(self):
        """Reset all points"""
        self.camera_points.clear()
        self.bev_points.clear()
        self.point_descriptions.clear()
        self.point_counter = 0
        print("All points cleared")
        self._update_displays()
    
    def _validate_points(self) -> bool:
        """Validate calibration points"""
        if len(self.camera_points) < 4:
            print(f"✗ Need at least 4 points, have {len(self.camera_points)}")
            return False
        
        if len(self.camera_points) != len(self.bev_points):
            print("✗ Mismatch in number of camera/BEV points")
            return False
        
        return True
    
    def _save_calibration(self, filepath: str):
        """Save calibration to file"""
        data = {
            'points': [
                {
                    'camera': list(self.camera_points[i]),
                    'bev': list(self.bev_points[i]),
                    'description': self.point_descriptions[i]
                }
                for i in range(len(self.camera_points))
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Calibration saved to {filepath}")


class HomographyValidator:
    """
    Validate and assess quality of homography transformation
    """
    
    def __init__(self, camera_points: np.ndarray, bev_points: np.ndarray,
                 homography: np.ndarray):
        self.camera_points = camera_points
        self.bev_points = bev_points
        self.homography = homography
    
    def compute_reprojection_error(self) -> float:
        """
        Compute mean reprojection error
        Lower is better (< 5 pixels is good)
        """
        camera_pts = self.camera_points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(camera_pts, self.homography)
        transformed = transformed.reshape(-1, 2)
        
        errors = np.linalg.norm(transformed - self.bev_points, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"\nReprojection Error:")
        print(f"  Mean: {mean_error:.2f} pixels")
        print(f"  Max:  {max_error:.2f} pixels")
        print(f"  Std:  {np.std(errors):.2f} pixels")
        
        return mean_error
    
    def check_homography_validity(self) -> bool:
        """
        Check if homography is valid (not degenerate)
        """
        # Check determinant
        det = np.linalg.det(self.homography)
        
        if abs(det) < 1e-6:
            print("✗ Homography is degenerate (determinant ~0)")
            return False
        
        # Check for reasonable scale
        scale = np.sqrt(abs(det))
        if scale < 0.1 or scale > 10:
            print(f"⚠ Warning: Unusual scale factor: {scale:.2f}")
        
        # Check condition number
        cond = np.linalg.cond(self.homography)
        if cond > 1000:
            print(f"⚠ Warning: High condition number: {cond:.2f}")
            print("   Homography may be poorly conditioned")
        
        print(f"✓ Homography determinant: {det:.4f}")
        return True
    
    def visualize_grid_warp(self, camera_frame: np.ndarray, 
                           bev_image: np.ndarray) -> np.ndarray:
        """
        Visualize how a grid warps from camera to BEV
        Useful for debugging
        """
        h, w = camera_frame.shape[:2]
        
        # Create grid in camera view
        grid_spacing = 50
        grid_img = camera_frame.copy()
        
        # Draw grid lines
        for i in range(0, w, grid_spacing):
            cv2.line(grid_img, (i, 0), (i, h), (0, 255, 0), 1)
        for j in range(0, h, grid_spacing):
            cv2.line(grid_img, (0, j), (w, j), (0, 255, 0), 1)
        
        # Warp grid to BEV
        bev_h, bev_w = bev_image.shape[:2]
        warped_grid = cv2.warpPerspective(
            grid_img, self.homography, (bev_w, bev_h)
        )
        
        # Overlay on BEV
        result = cv2.addWeighted(bev_image, 0.7, warped_grid, 0.3, 0)
        
        return result


class AutoCalibration:
    """
    Automatic calibration using lane detection and feature matching
    (Advanced - for future improvement)
    """
    
    def __init__(self):
        self.lane_detector = None
    
    def detect_lane_markings(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detect lane markings in camera view
        Can be used as automatic calibration points
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 50,
            minLineLength=30, maxLineGap=10
        )
        
        return lines if lines is not None else []
    
    def extract_road_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract key road features for matching
        """
        # Use ORB or SIFT for feature detection
        orb = cv2.ORB_create(1000)
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        
        return keypoints, descriptors

def create_sample_calibration():
    """
    Create a sample calibration file for reference
    """
    sample_data = {
        'metadata': {
            'description': 'Sample calibration for traffic camera',
            'date': '2025-10-21',
            'camera_resolution': [1920, 1080],
            'bev_resolution': [2048, 2048]
        },
        'points': [
            {
                'camera': [856, 423],
                'bev': [1024, 800],
                'description': 'Top-left intersection corner'
            },
            {
                'camera': [1245, 445],
                'bev': [1400, 820],
                'description': 'Top-right intersection corner'
            },
            {
                'camera': [1567, 789],
                'bev': [1450, 1200],
                'description': 'Bottom-right intersection corner'
            },
            {
                'camera': [645, 756],
                'bev': [950, 1180],
                'description': 'Bottom-left intersection corner'
            }
        ]
    }
    
    with open('sample_calibration.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Sample calibration created: sample_calibration.json")


def print_usage():
    """
    Helper function to print tool usage
    """
    print("Usage:")
    print("  python calibration_tools.py calibrate <camera_img> <bev_img>")
    print("  python calibration_tools.py parse_kml <kml_file>")
    print("  python calibration_tools.py validate <calibration.json> <camera_img> <bev_img>")
    print("  python calibration_tools.py sample")
    print("  python calibration_tools.py finetunebev <calibration.json> <x_offset> <y_offset>")
    sys.exit(1)


def is_int_str(s: str) -> bool:
    """
    Test if input string is an integer
    """
    if s and s[0] == '-':  # Check for negative sign
        return s[1:].isdigit()
    return s.isdigit()

def calcualte_bev_meters_per_pix(start_pts: List[Tuple[int, int]], end_pts: List[Tuple[int, int]], measure_mtr: List[float], distance_type: str = 'L2') -> float:
    ls_dists = []
    # Calculate meters per pixel for each measurement
    for start_pt, end_pt, real_distance_m in zip(start_pts, end_pts, measure_mtr):
        # Calculate pixel distance
        if distance_type == 'L1':
            # L1 (Manhattan distance): |x2-x1| + |y2-y1|
            pixel_distance = abs(end_pt[0] - start_pt[0]) + abs(end_pt[1] - start_pt[1])
        elif distance_type == 'L2':
            # L2 (Euclidean distance): sqrt((x2-x1)^2 + (y2-y1)^2)
            dx = end_pt[0] - start_pt[0]
            dy = end_pt[1] - start_pt[1]
            pixel_distance = np.sqrt(dx**2 + dy**2)
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}. Use 'L1' or 'L2'")
        
        # Calculate meters per pixel for this measurement
        meters_per_pixel = real_distance_m / pixel_distance
        ls_dists.append(meters_per_pixel)
    return np.mean(np.array(ls_dists))


# Main execution example
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print_usage()
    
    command = sys.argv[1]
    
    if command == 'calibrate':
        camera_img = cv2.imread(sys.argv[2])
        bev_img = cv2.imread(sys.argv[3])
        
        tool = CalibrationTool(camera_img, bev_img)
        cam_pts, bev_pts, descs = tool.run()
        
        if len(cam_pts) >= 4:
            # Compute and validate homography
            cam_pts_np = np.float32(cam_pts)
            bev_pts_np = np.float32(bev_pts)
            H, _ = cv2.findHomography(cam_pts_np, bev_pts_np, cv2.RANSAC)
            
            validator = HomographyValidator(cam_pts_np, bev_pts_np, H)
            validator.compute_reprojection_error()
            validator.check_homography_validity()
            
            # Save
            data = {
                'points': [
                    {
                        'camera': list(cam_pts[i]),
                        'bev': list(bev_pts[i]),
                        'description': descs[i]
                    }
                    for i in range(len(cam_pts))
                ]
            }
            
            with open('calibration.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            print("\n✓ Calibration saved to calibration.json")
    
    elif command == 'parse_kml':
        parser = KMLParser(sys.argv[2])
        placemarks = parser.parse()
        distances = parser.get_distance_measurements()
        
        print(f"\nFound {len(placemarks)} placemarks:")
        for pm in placemarks:
            print(f"  - {pm['name']} ({pm['type']})")
        
        print(f"\nDistance measurements:")
        for d in distances:
            print(f"  - {d['name']}: {d['distance_meters']:.2f} meters")
    
    elif command == 'sample':
        create_sample_calibration()
    
    elif command == 'validate':
        # Load calibration
        with open(sys.argv[2], 'r') as f:
            calib = json.load(f)
        
        cam_pts = np.float32([p['camera'] for p in calib['points']])
        bev_pts = np.float32([p['bev'] for p in calib['points']])
        
        H, _ = cv2.findHomography(cam_pts, bev_pts, cv2.RANSAC)
        
        print(H)

        validator = HomographyValidator(cam_pts, bev_pts, H)
        validator.compute_reprojection_error()
        validator.check_homography_validity()
        
        # Visualize
        if len(sys.argv) >= 5:
            camera_img = cv2.imread(sys.argv[3])
            bev_img = cv2.imread(sys.argv[4])
            result = validator.visualize_grid_warp(camera_img, bev_img)
            cv2.imwrite('calibration_validation.jpg', result)
            print("\n✓ Validation visualization saved")
    
    elif command == 'finetunebev':
        # Load calibration
        with open(sys.argv[2], 'r') as f:
            calib = json.load(f)
        
        if is_int_str(sys.argv[3]) and is_int_str(sys.argv[4]):
            x_offset, y_offset = int(sys.argv[3]), int(sys.argv[4])

            for i, p in enumerate(calib['points']):
                calib['points'][i]['bev'][0] += x_offset
                calib['points'][i]['bev'][1] += y_offset

            with open('finetuned.json', 'w') as f:
                json.dump(calib, f, indent=2)

            print("Coordinate finetune for BEV completed, ", (i+1), " coordinates changed")
        else:
            raise Exception("Incorrect input for BEV offsets, must be integers")

    elif command == 'meter2pixel':
        # Can use a json / kml file for the pixel to meter data loading.
        mpp = calcualte_bev_meters_per_pix([(520,522),(261,1056),(395,997),(610,880),(535,1048),(414,1107),(540,1241)], 
                                  [(545,395),(307,1035),(605,872),(714,1115),(572,1030),(448,1089),(544,1318)], 
                                  [6.7594, 2.9872, 13.4526, 13.0655, 2.265, 2.2516, 4.1646])
        print(mpp)

    else:
        print("Unknown command: ", command)
        print_usage()


