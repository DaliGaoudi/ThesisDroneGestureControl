# Drone Gesture Control

This project implements hand gesture control for a DJI Tello drone using computer vision and OpenCV.

## Requirements

- Python 3.8 or higher
- DJI Tello Drone
- Webcam
- Required Python packages (listed in requirements.txt)

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Connect to your Tello drone's WiFi network

3. Run the basic test script:
```bash
python drone_gesture_control_gui_pyqt
```

## Project Structure

- `test_drone.py`: Basic drone connection and control test
- `gestures.py`: Hand gesture recognition and drone control implementation
- 'drone_gesture_control_gui_pyqt.py': GUI
- `utils/`: Utility functions and helper modules

## Features

- Real-time hand gesture recognition using webcam
- Basic drone control (takeoff, landing, movement)
- Gesture-based drone navigation 