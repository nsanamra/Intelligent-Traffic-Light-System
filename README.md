# Intelligent Traffic System using YOLO

This project implements an intelligent traffic management system using YOLO for real-time vehicle detection. The system dynamically adjusts traffic signals based on vehicle density in different lanes to optimize traffic flow.

## Features
- Uses YOLO for vehicle detection in specified lane regions.
- Dynamically switches traffic signals based on vehicle density.
- Implements an adaptive signal timing mechanism.
- Includes a `roi_coords.py` script to select lane regions by clicking on an image.

## Requirements
Ensure you have the following installed:
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- PyTorch
- Ultralytics YOLO (`ultralytics`)

Install dependencies using:
```bash
pip install opencv-python numpy torch ultralytics
