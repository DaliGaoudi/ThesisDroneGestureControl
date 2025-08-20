import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QCheckBox, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import cv2
from gestures import HandGestureDetector
from djitellopy import Tello
import threading
import time
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

class DroneGestureControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Gesture Control System (PyQt5)")
        self.setGeometry(100, 100, 1400, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Drone and gesture detector
        self.drone = Tello()
        self.webcam = cv2.VideoCapture(0)
        self.gesture_detector = HandGestureDetector()

        # State
        self.is_connected = False
        self.is_flying = False
        self.video_running = False
        self.trajectory_points = []
        self.override_gesture = None

        # UI Elements
        self.status_label = QLabel("Status: Gesture Detection Active")
        self.status_label.setStyleSheet("color: #4CAF50; font-size: 16px;")
        self.battery_label = QLabel("Battery: --%")
        self.gesture_label = QLabel("Gesture: None")
        self.connect_btn = QPushButton("Connect to Drone")
        self.connect_btn.clicked.connect(self.connect_drone)
        self.performance_checkbox = QCheckBox("Fast Mode")
        self.performance_checkbox.setChecked(True)

        # Gesture override buttons
        self.gesture_names = [
            "Palm", "Fist", "Thumbs Up", "Thumbs Down", "Peace", "Okay", "Point", "Stop", "Wave", "Rock"
        ]
        self.gesture_buttons = []
        for gesture in self.gesture_names:
            btn = QPushButton(gesture)
            btn.clicked.connect(lambda checked, g=gesture: self.override_gesture_action(g))
            self.gesture_buttons.append(btn)

        # Video display
        self.drone_video_label = QLabel()
        self.drone_video_label.setFixedSize(960, 720)
        self.drone_video_label.setStyleSheet("background-color: black;")
        self.webcam_video_label = QLabel()
        self.webcam_video_label.setFixedSize(960, 720)
        self.webcam_video_label.setStyleSheet("background-color: black;")

        # 3D Trajectory Viewer
        self.trajectory_widget = gl.GLViewWidget()
        self.trajectory_widget.setFixedSize(480, 480)
        self.trajectory_widget.setWindowTitle('Drone Trajectory')
        self.trajectory_widget.setCameraPosition(distance=20)
        self.trajectory_plot = gl.GLLinePlotItem()
        self.trajectory_widget.addItem(self.trajectory_plot)

        # Layouts
        self._setup_layout()

        # Timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)
        self.video_running = True

    def update_trajectory(self, x, y, z):
        self.trajectory_points.append([x, y, z])
        pts = np.array(self.trajectory_points)
        self.trajectory_plot.setData(pos=pts, color=(1, 0, 0, 1), width=2, antialias=True)

    def move_drone(self, dx, dy, dz):
        # This should be called when you send a movement command to the drone
        if self.trajectory_points:
            last = self.trajectory_points[-1]
        else:
            last = [0, 0, 0]
        new_pos = [last[0] + dx, last[1] + dy, last[2] + dz]
        self.update_trajectory(*new_pos)

    def _setup_layout(self):
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        video_layout = QHBoxLayout()
        extra_layout = QHBoxLayout()
        gesture_btn_layout = QHBoxLayout()

        # Control panel
        control_layout.addWidget(self.connect_btn)
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.battery_label)
        control_layout.addWidget(self.gesture_label)
        control_layout.addWidget(self.performance_checkbox)

        # Gesture override buttons
        for btn in self.gesture_buttons:
            gesture_btn_layout.addWidget(btn)

        # Video panels
        drone_frame = QFrame()
        drone_frame.setFrameShape(QFrame.StyledPanel)
        drone_layout = QVBoxLayout()
        drone_layout.addWidget(QLabel("Drone View"))
        drone_layout.addWidget(self.drone_video_label)
        drone_frame.setLayout(drone_layout)

        webcam_frame = QFrame()
        webcam_frame.setFrameShape(QFrame.StyledPanel)
        webcam_layout = QVBoxLayout()
        webcam_layout.addWidget(QLabel("Gesture Detection (Mirrored)"))
        webcam_layout.addWidget(self.webcam_video_label)
        webcam_frame.setLayout(webcam_layout)

        video_layout.addWidget(drone_frame)
        video_layout.addWidget(webcam_frame)

        # Add 3D trajectory widget
        extra_layout.addWidget(self.trajectory_widget)

        main_layout.addLayout(control_layout)
        main_layout.addLayout(gesture_btn_layout)
        main_layout.addLayout(video_layout)
        main_layout.addLayout(extra_layout)
        self.central_widget.setLayout(main_layout)
    def override_gesture_action(self, gesture_name):
        self.override_gesture = gesture_name
        self.gesture_label.setText(f"Gesture: {gesture_name} (Override)")
        # Here you can trigger the corresponding gesture action, e.g.:
        # self.handle_gesture(gesture_name)

    def connect_drone(self):
        if not self.is_connected:
            self.connect_btn.setText("Connecting...")
            self.status_label.setText("Status: Connecting...")
            self.status_label.setStyleSheet("color: orange;")
            threading.Thread(target=self._connect_drone_thread, daemon=True).start()

    def _connect_drone_thread(self):
        try:
            self.drone.connect()
            battery = self.drone.get_battery()
            self.drone.streamon()
            time.sleep(2)
            self.is_connected = True
            self.status_label.setText("Status: Connected & Ready")
            self.status_label.setStyleSheet("color: #4CAF50;")
            self.battery_label.setText(f"Battery: {battery}%")
            self.connect_btn.setText("Disconnect")
            QMessageBox.information(self, "Success", "Successfully connected to drone!")
        except Exception as e:
            self.status_label.setText("Status: Connection Failed")
            self.status_label.setStyleSheet("color: #ff6b6b;")
            self.connect_btn.setText("Connect to Drone")
            QMessageBox.critical(self, "Connection Error", f"Failed to connect to drone:\n{e}")

    def update_video(self):
        # Webcam video
        ret, webcam_frame = self.webcam.read()
        if ret:
            gesture_frame = cv2.flip(webcam_frame, 1)
            if self.override_gesture:
                gestures = self.override_gesture
            else:
                gesture_frame, gestures = self.gesture_detector.detect_gesture(gesture_frame)
            # Show gesture label
            if gestures:
                if isinstance(gestures, list):
                    gesture_text = ", ".join(gestures)
                else:
                    gesture_text = gestures
                if self.override_gesture:
                    self.gesture_label.setText(f"Gesture: {gesture_text} (Override)")
                else:
                    self.gesture_label.setText(f"Gesture: {gesture_text}")
            else:
                self.gesture_label.setText("Gesture: None")
            # Convert to QImage
            gesture_frame = cv2.resize(gesture_frame, (960, 720))
            rgb_image = cv2.cvtColor(gesture_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.webcam_video_label.setPixmap(pixmap)
        # Drone video (if connected)
        if self.is_connected:
            try:
                drone_frame = self.drone.get_frame_read().frame
                if drone_frame is not None:
                    drone_frame = cv2.resize(drone_frame, (960, 720))
                    rgb_image = cv2.cvtColor(drone_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.drone_video_label.setPixmap(pixmap)
            except Exception as e:
                pass

    def closeEvent(self, event):
        self.video_running = False
        self.webcam.release()
        if self.is_connected:
            try:
                self.drone.streamoff()
            except:
                pass
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = DroneGestureControlWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 