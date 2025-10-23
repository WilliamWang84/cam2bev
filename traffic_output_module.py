"""
Output Module for Traffic BEV System
"""

import csv
import json
import numpy as np
import sqlite3

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class TrafficBEVOutput:
    """
    Handles output operations: GeoJSON export, SQLite3 storage, statistics
    """
    
    def __init__(self, camera_config: Dict, db_path: str = 'traffic_tracks.db'):
        """
        Args:
            camera_config: {
                'id': 'cam_001',
                'location': {'lat': -33.8688, 'lon': 151.2093, 'alt': 15.0},
                'bev_to_gps_transform': np.ndarray (3x3 homography matrix) - optional
            }
        """
        self.camera_id = camera_config['id']
        self.camera_location = camera_config['location']
        self.transform_matrix = camera_config.get('bev_to_gps_transform')
        
        self.db_path = db_path
        self._init_database()

        self.bev_meters_per_pixel = camera_config.get('bev_meters_per_pixel', 0.055)
        self.video_fps = camera_config.get('fps', 25)
    
    def _init_database(self):
        """Initialize SQLite3 database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Track summaries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                track_id INTEGER PRIMARY KEY,
                camera_id TEXT NOT NULL,
                vehicle_type TEXT,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                duration_seconds REAL,
                total_frames INTEGER,
                avg_speed_kmh REAL,
                distance_traveled_m REAL,
                start_lat REAL,
                start_lon REAL,
                end_lat REAL,
                end_lon REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trajectory points
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trajectory_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                timestamp TIMESTAMP,
                frame_number INTEGER,
                bev_x REAL,
                bev_y REAL,
                lat REAL,
                lon REAL,
                speed_mps REAL,
                heading_deg REAL,
                confidence REAL,
                FOREIGN KEY (track_id) REFERENCES tracks(track_id)
            )
        ''')
        
        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_camera ON tracks(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trajectory_track ON trajectory_points(track_id)')
        
        conn.commit()
        conn.close()
    
    def bev_to_gps(self, bev_x: float, bev_y: float) -> Tuple[float, float]:
        """Convert BEV coordinates to GPS (simplified for small areas)"""
        if self.transform_matrix is None:
            return self.camera_location['lat'], self.camera_location['lon']
        
        bev_point = np.array([bev_x, bev_y, 1])
        world_coords = self.transform_matrix @ bev_point
        world_coords = world_coords[:2] / world_coords[2]
        
        lat = self.camera_location['lat'] + world_coords[1] / 111320.0
        lon = self.camera_location['lon'] + world_coords[0] / (111320.0 * np.cos(np.radians(lat)))
        
        return lat, lon
    
    def generate_geojson(self, tracks: List[Dict], output_file: Optional[str] = None) -> Dict:
        """
        Generate GeoJSON from current tracks
        
        Args:
            tracks: List of dicts with keys: id, bev_x, bev_y, class, confidence, speed, heading
            output_file: Optional path to save JSON file
        """
        features = []
        
        for track in tracks:
            lat, lon = self.bev_to_gps(track['bev_x'], track['bev_y'])
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "track_id": track['id'],
                    "vehicle_type": track.get('class', 'unknown'),
                    "confidence": float(track.get('confidence', 0.0)),
                    "velocity": {
                        "speed_mps": float(track.get('speed', 0.0)),
                        "heading_deg": float(track.get('heading', 0.0)),
                        "speed_kmh": float(track.get('speed', 0.0) * 3.6)
                    },
                    "last_updated": datetime.utcnow().isoformat() + 'Z'
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "metadata": {
                "camera_id": self.camera_id,
                "camera_location": self.camera_location,
                "vehicle_count": len(tracks)
            },
            "features": features
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(geojson, f, indent=2)
            print(f"GeoJSON saved to {output_file}")
        
        return geojson
    
    def save_track_history(self, track_id: int, trajectory_points: List[Tuple], 
                        vehicle_type: str, frame_numbers: List[int],
                        start_timestamp: Optional[datetime] = None):
        """
        Save track to database
        
        Args:
            track_id: Unique ID
            trajectory_points: List of (bev_x, bev_y) tuples from self.trajectories[track_id]
            vehicle_type: Vehicle class name
            frame_numbers: Frame numbers for each point
            start_timestamp: Optional timestamp of first frame (defaults to now)
        """
        if len(trajectory_points) == 0:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Use start_timestamp or current time as base
        if start_timestamp is None:
            start_timestamp = datetime.utcnow()
        
        # Convert trajectory to database format
        db_trajectory = []
        speeds = []
        
        for i, (bev_x, bev_y) in enumerate(trajectory_points):
            lat, lon = self.bev_to_gps(bev_x, bev_y)
            
            # Calculate timestamp based on frame number and FPS
            frame_offset_seconds = frame_numbers[i] / self.video_fps
            point_timestamp = start_timestamp + timedelta(seconds=frame_offset_seconds)
            
            # Calculate speed from trajectory
            speed = 0.0
            heading = 0.0
            if i > 0:
                prev_x, prev_y = trajectory_points[i-1]
                dx = bev_x - prev_x  # last frame to current frame pix difference x 
                dy = bev_y - prev_y  # last frame to current frame pix difference y
                distance_pix = np.sqrt(dx**2 + dy**2)  # Euclidean L2 distance last to current frame
                distance_mtr = distance_pix * self.bev_meters_per_pixel
                time_diff = (frame_numbers[i] - frame_numbers[i-1]) / self.video_fps
                if time_diff > 0:
                    speed = distance_mtr / time_diff

                heading = np.degrees(np.arctan2(dx, -dy)) % 360
                speeds.append(speed)
            
            db_trajectory.append({
                'track_id': track_id,
                'timestamp': point_timestamp,
                'frame_number': frame_numbers[i] if i < len(frame_numbers) else i,
                'bev_x': float(bev_x),
                'bev_y': float(bev_y),
                'lat': lat,
                'lon': lon,
                'speed_mps': speed,
                'heading_deg': heading,
                'confidence': 0.9
            })
        
        # Calculate statistics
        first = db_trajectory[0]
        last = db_trajectory[-1]

        duration = (frame_numbers[-1] - frame_numbers[0]) / self.video_fps
        avg_speed_kmh = np.mean(speeds) * 3.6 if speeds else 0.0
        distance = self._calculate_distance(db_trajectory)
        
        # Insert track summary
        cursor.execute('''
            INSERT OR REPLACE INTO tracks (
                track_id, camera_id, vehicle_type,
                first_seen, last_seen, duration_seconds, total_frames,
                avg_speed_kmh, distance_traveled_m,
                start_lat, start_lon, end_lat, end_lon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            track_id, self.camera_id, vehicle_type,
            first['timestamp'], last['timestamp'],
            duration, len(trajectory_points),
            avg_speed_kmh, distance,
            first['lat'], first['lon'], last['lat'], last['lon']
        ))
        
        # Insert trajectory points
        cursor.executemany('''
            INSERT INTO trajectory_points (
                track_id, timestamp, frame_number,
                bev_x, bev_y, lat, lon,
                speed_mps, heading_deg, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            (p['track_id'], p['timestamp'], p['frame_number'],
            p['bev_x'], p['bev_y'], p['lat'], p['lon'],
            p['speed_mps'], p['heading_deg'], p['confidence'])
            for p in db_trajectory
        ])
        
        conn.commit()
        conn.close()
        
    
    def _calculate_distance(self, trajectory: List[Dict]) -> float:
        """Calculate distance using Haversine"""
        distance = 0.0
        for i in range(1, len(trajectory)):
            lat1, lon1 = trajectory[i-1]['lat'], trajectory[i-1]['lon']
            lat2, lon2 = trajectory[i]['lat'], trajectory[i]['lon']
            distance += self._haversine_distance(lat1, lon1, lat2, lon2)
        return distance
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Distance between GPS points in meters"""
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def get_statistics(self, time_window_minutes: int = 60) -> Dict:
        """Get traffic statistics for recent time window"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = datetime.utcnow().timestamp() - (time_window_minutes * 60)
        cutoff_datetime = datetime.fromtimestamp(cutoff)
        
        # Count by vehicle type
        cursor.execute('''
            SELECT vehicle_type, COUNT(*) as count
            FROM tracks
            WHERE camera_id = ? AND first_seen >= ?
            GROUP BY vehicle_type
        ''', (self.camera_id, cutoff_datetime))
        vehicle_counts = dict(cursor.fetchall())
        
        # Speed statistics
        cursor.execute('''
            SELECT AVG(avg_speed_kmh) as avg_speed
            FROM tracks
            WHERE camera_id = ? AND first_seen >= ?
        ''', (self.camera_id, cutoff_datetime))
        speed_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'time_window_minutes': time_window_minutes,
            'camera_id': self.camera_id,
            'total_vehicles': sum(vehicle_counts.values()),
            'vehicle_counts': vehicle_counts,
            'avg_speed_kmh': speed_stats[0] if speed_stats[0] else 0.0,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
