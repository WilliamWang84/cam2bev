# Simple Cam2Bev (camera to satellite / bird's eye view) System 

A Simple camera view to satellite / bird's eye view perspective transform system

## Description

This project implements a Camera view (CAM) to Bird's Eye View (BEV) transform system. The system workflow is intuitive:
  - 1. Given a camera view, and its corresponding satellite view, do the followings:
    - a. Identify / mark solid landmark points for both CAM and BEV (fixation points that do not change over time)
    - b. Perform Perspective Transform (calcualte homography matrix - H) from CAM to BEV (and optionally its inverse) using landmarks obtained from 1a.
  - 2. Run Mainstream object detection / tracking on the CAM view, common generic / robust detection / tracking pipeline includes but not limited to YOLO / MaskRCNN / vLLM (for detection) + ByteTrack / SORT / DeepSORT (for tracking)
    (TODO)
    - a. Use transfer learning to fine-tune the pre-trained YOLO / vLLM to further boost detector performance
    - b. Fine-tune tracking algorithm parameters, obtain accurate centroid (and corner) points, as well as their trajectories for objects detected / tracked in CAM 
  - 3. With the results (object centroids, corners and trajectories) from 2b, together with the results (homography transformation matrix) from 1b, do the followings:  
    - a. Perform perspective transform (CAM objects ---(H)---> BEV objects)
    - b. Plot the transformed results from 3a to a clean BEV image canvas 

## Getting Started

### Dependencies

Required Libraries: 
 - filterpy
 - json (part of python)
 - numpy
 - opencv-python (cv2)
 - pathlib
 - pytorch (torch)
 - scikit-learn (scipy)
 - sqlite3 (part of python)
 - torchvision
 - ultralytics

### Installing

- 1. clone this repo
- 2. create virtual environment (strongly recommended)
- 3. install dependencies (note the cu118 after /whl, choose a version that is compatible with your GPU Driver)
     ```
     pip install filterpy numpy opencv-python pathlib scikit-learn ultralytics
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```
- 4. prepare the following files:
   - a. your camera image (with solid landmarks clearly visible) - e.g. imageA.jpg
   - b. your satellite image (with solid landmarks clearly visible) - e.g. imageB.jpg
- 5. run calibration_tools.py to obtain calibration.json file - e.g.
     ```
     python calibration_tools.py calibrate imageA.jpg imageB.jpg
     ```
- 6. run calibration_tools.py to obtain the homography matrix - e.g.
     ```
     python calibration_tools.py validate calibration.json imageA.jpg imageB.jpg
     ```
- 7. prepare the camera_config dictionary in bev_traffic_system.py, the dictionary should contain:
   - a. 'id' - the camera id
   - b. 'location' - a dictionary of 'lat','lon','alt' representing the floating point representations of your camera's latitude, longtitude and altitude
   - c. 'bev_to_gps_transform' - the homography matrix obtained from 6
   - d. 'bev_meters_per_pixel' - the meters per pixel mapping from your BEV, can be measured / provided or estimated from map view
   - e. 'fps' - the video / stream fps 
- 8. check and confirm correctness for the following parameters in bev_traffic_system.py:
   - a. bev_image_path - e.g. imageB.jpg
   - b. yolo_model_path - e.g. yolo12n.pt (ultralytics will handle the downloading of the model automatically)
   - c. calibration_file - e.g. calibration.json
   - d. camera_config - <from 7>
   - e. enable_output - True if you like to save geojson files and tracks, False otherwise 

### Executing program

* Ensure the dependancies are available
* Run the following 
```
python bev_traffic_system.py
```

## Authors

Sheng WANG (William)
wangshengcom@gmail.com

## Version History

* 0.1
    * Initial Release
    * Added DeepSort

## Acknowledgments

* [ultralytics](https://github.com/ultralytics)
* [ByteTrack](https://github.com/FoundationVision/ByteTrack)
* [SQLite](https://sqlite.org/download.html)
