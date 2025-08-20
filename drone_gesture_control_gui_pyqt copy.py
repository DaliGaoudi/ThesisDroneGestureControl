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

        # UI Elements
        self.status_label = QLabel("Status: Gesture Detection Active")
        self.status_label.setStyleSheet("color: #4CAF50; font-size: 16px;")
        self.battery_label = QLabel("Battery: --%")
        self.gesture_label = QLabel("Gesture: None")
        self.connect_btn = QPushButton("Connect to Drone")
        self.connect_btn.clicked.connect(self.connect_drone)
        self.performance_checkbox = QCheckBox("Fast Mode")
        self.performance_checkbox.setChecked(True)

        # Video display
        self.drone_video_label = QLabel()
        self.drone_video_label.setFixedSize(960, 720)
        self.drone_video_label.setStyleSheet("background-color: black;")
        self.webcam_video_label = QLabel()
        self.webcam_video_label.setFixedSize(960, 720)
        self.webcam_video_label.setStyleSheet("background-color: black;")

        # Layouts
        self._setup_layout()

        # Timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)
        self.video_running = True

    def _setup_layout(self):
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        video_layout = QHBoxLayout()

        # Control panel
        control_layout.addWidget(self.connect_btn)
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.battery_label)
        control_layout.addWidget(self.gesture_label)
        control_layout.addWidget(self.performance_checkbox)

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

        main_layout.addLayout(control_layout)
        main_layout.addLayout(video_layout)
        self.central_widget.setLayout(main_layout)

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
            gesture_frame, gestures = self.gesture_detector.detect_gesture(gesture_frame)
            # Show gesture label
            if gestures:
                if isinstance(gestures, list):
                    gesture_text = ", ".join(gestures)
                else:
                    gesture_text = gestures
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