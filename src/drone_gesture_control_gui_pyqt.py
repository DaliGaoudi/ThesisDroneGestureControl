import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QCheckBox, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QPixmap, QImage
import cv2
from gestures import HandGestureDetector
from djitellopy import Tello
import threading
import time
import numpy as np

class GestureWorker(QObject):
    """Runs gesture detection in a separate thread to avoid blocking the UI."""
    new_frame_and_gesture = pyqtSignal(np.ndarray, str)

    def __init__(self, webcam, gesture_detector):
        super().__init__()
        self.webcam = webcam
        self.gesture_detector = gesture_detector
        self._is_running = True

    def run(self):
        """Continuously captures frames and detects gestures."""
        while self._is_running:
            ret, frame = self.webcam.read()
            if not ret:
                time.sleep(0.01)  # Wait a bit if frame capture fails
                continue

            # Perform detection
            gesture_frame = cv2.flip(frame, 1)
            processed_frame, gestures = self.gesture_detector.detect_gesture(gesture_frame)

            current_gesture = "Fist"  # Default to hover
            if gestures:
                if isinstance(gestures, list):
                    current_gesture = gestures[0] if gestures else "Fist"
                else:
                    current_gesture = gestures
            
            # Emit the results to the main thread
            self.new_frame_and_gesture.emit(processed_frame, current_gesture)

    def stop(self):
        self._is_running = False


class DroneGestureControlWindow(QMainWindow):
    connection_result = pyqtSignal(bool, str, int)
    disconnection_complete = pyqtSignal()
    log_message_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Gesture Control System (PyQt5)")
        self.setGeometry(100, 100, 1400, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.drone = None
        self.webcam = cv2.VideoCapture(0)
        self.gesture_detector = HandGestureDetector()

        self.is_connected = False
        self.is_flying = False
        self.override_gesture = None
        self.last_gesture = "None"

        # New state variables for handling dynamic gestures
        self.dynamic_gesture_lock = None
        self.dynamic_gesture_timer = QTimer()
        self.dynamic_gesture_timer.setSingleShot(True)
        self.dynamic_gesture_timer.timeout.connect(self.release_dynamic_gesture_lock)

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
            "Open", "Fist", "thumbs_up", "thumbs_down", "Pointing Right", "Pointing Left", "Pointing Up", "Pointing Down", "rotate_clockwise", "rotate_counterclockwise", "OK", "Clear Override"
        ]
        self.gesture_buttons = []
        for gesture in self.gesture_names:
            btn = QPushButton(gesture)
            btn.clicked.connect(lambda checked, g=gesture: self.override_gesture_action(g))
            self.gesture_buttons.append(btn)

        self.drone_video_label = QLabel()
        self.drone_video_label.setFixedSize(960, 720)
        self.drone_video_label.setStyleSheet("background-color: black;")
        self.webcam_video_label = QLabel()
        self.webcam_video_label.setFixedSize(960, 720)
        self.webcam_video_label.setStyleSheet("background-color: black;")
        
        # Drone Console
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("background-color: #2b2b2b; color: #f0f0f0; font-family: Consolas, monaco, monospace;")
        self.console_output.setPlaceholderText("Drone command and status log will appear here...")
        
        self._setup_layout()

        # This timer is now ONLY for the drone video feed, which is fast.
        self.drone_video_timer = QTimer(self)
        self.drone_video_timer.timeout.connect(self.update_drone_video)
        self.drone_video_timer.start(100)

        # Set up and start the background thread for gesture detection
        self._setup_gesture_thread()

        self.connection_result.connect(self.show_connection_result)
        self.disconnection_complete.connect(self.on_disconnection_complete)
        self.log_message_signal.connect(self.append_log_message)
        self.log_message("Welcome to the Drone Gesture Control System.")

    def _setup_gesture_thread(self):
        """Initializes and starts the gesture detection worker and thread."""
        self.gesture_thread = QThread()
        self.gesture_worker = GestureWorker(self.webcam, self.gesture_detector)
        self.gesture_worker.moveToThread(self.gesture_thread)

        # Connect signals: When the worker has a result, the main thread will process it.
        self.gesture_thread.started.connect(self.gesture_worker.run)
        self.gesture_worker.new_frame_and_gesture.connect(self.on_new_gesture_data)
        
        # Start the thread. The 'run' method will be executed in the background.
        self.gesture_thread.start()
        self.log_message("Gesture detection thread started.")

    def log_message(self, message):
        self.log_message_signal.emit(f"[{time.strftime('%H:%M:%S')}] {message}")

    def append_log_message(self, message):
        self.console_output.append(message)
        self.console_output.verticalScrollBar().setValue(self.console_output.verticalScrollBar().maximum())

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
        
        console_frame = QFrame()
        console_frame.setFrameShape(QFrame.StyledPanel)
        console_layout = QVBoxLayout()
        console_layout.addWidget(QLabel("Drone Console"))
        console_layout.addWidget(self.console_output)
        console_frame.setLayout(console_layout)

        extra_layout.addWidget(console_frame)
        
        main_layout.addLayout(control_layout)
        main_layout.addLayout(gesture_btn_layout)
        main_layout.addLayout(video_layout)
        main_layout.addLayout(extra_layout)
        self.central_widget.setLayout(main_layout)

    def _execute_flight_command(self, action):
        """Executes blocking flight commands in a separate thread."""
        try:
            if action == "takeoff":
                self.log_message("Takeoff command initiated in background thread...")
                self.drone.takeoff()
                self.is_flying = True
                self.log_message("Takeoff complete. Drone is now flying.")
            elif action == "land":
                self.log_message("Land command initiated in background thread...")
                self.drone.land()
                self.is_flying = False
                self.log_message("Landing complete. Drone is no longer flying.")
        except Exception as e:
            error_msg = f"Error during '{action}': {e}"
            self.log_message(error_msg)
            # If takeoff fails, we are not flying.
            if action == "takeoff":
                self.is_flying = False

    def handle_one_shot_command(self, gesture):
        """Handles commands that are only executed once, like takeoff and land."""
        if not self.is_connected or not self.drone:
            return
        
        action = None
        if gesture == "Open" and not self.is_flying:
            action = "takeoff"
        elif gesture == "OK" and self.is_flying:
            action = "land"

        if action:
            self.log_message(f"Gesture '{gesture}' -> Action: '{action}'")
            # Run blocking commands in a separate thread to avoid freezing the GUI
            threading.Thread(target=self._execute_flight_command, args=(action,), daemon=True).start()

    def override_gesture_action(self, gesture_name):
        if gesture_name == "Clear Override":
            self.override_gesture = None
            self.gesture_label.setText("Gesture: None")
            self.log_message("Gesture override cleared.")
            # Send a stop command just in case
            if self.is_flying and self.drone:
                self.drone.send_rc_control(0, 0, 0, 0)
            return

        self.override_gesture = gesture_name
        self.gesture_label.setText(f"Gesture: {gesture_name} (Override)")
        self.log_message(f"Gesture override set to '{gesture_name}'.")
        
        # Immediately handle one-shot commands for responsiveness
        self.handle_one_shot_command(gesture_name)

    def connect_drone(self):
        self.connect_btn.setEnabled(False)
        if self.is_connected:
            self.log_message("Disconnect button clicked. Starting disconnection...")
            self.status_label.setText("Status: Disconnecting...")
            self.status_label.setStyleSheet("color: orange;")
            threading.Thread(target=self._disconnect_drone_thread, daemon=True).start()
        else:
            self.log_message("Connect button clicked. Initializing Tello and attempting to connect...")
            self.drone = Tello()
            self.connect_btn.setText("Connecting...")
            self.status_label.setText("Status: Connecting...")
            self.status_label.setStyleSheet("color: orange;")
            threading.Thread(target=self._connect_drone_thread, daemon=True).start()

    def _disconnect_drone_thread(self):
        self.log_message("Disconnect thread started.")
        if self.is_flying:
            try:
                self.log_message("Drone is flying. Attempting to land...")
                self.drone.land()
                self.log_message("Land command sent.")
            except Exception as e:
                self.log_message(f"Exception during landing on disconnect: {e}")
        try:
            self.log_message("Turning off video stream...")
            self.drone.streamoff()
        except Exception as e:
            self.log_message(f"Exception during streamoff on disconnect: {e}")
        try:
            self.log_message("Ending connection to drone...")
            self.drone.end()
        except Exception as e:
            self.log_message(f"Exception during end on disconnect: {e}")
        
        self.log_message("Disconnection process finished.")
        self.disconnection_complete.emit()

    def on_disconnection_complete(self):
        self.is_connected = False
        self.is_flying = False
        self.drone = None
        self.status_label.setText("Status: Disconnected")
        self.status_label.setStyleSheet("color: #ff6b6b;")
        self.battery_label.setText("Battery: --%")
        self.connect_btn.setText("Connect to Drone")
        self.connect_btn.setEnabled(True)
        self.drone_video_label.clear()
        self.drone_video_label.setStyleSheet("background-color: black;")
        self.log_message("UI updated to disconnected state.")

    def show_connection_result(self, success, message, battery):
        """This slot handles UI updates after the connection attempt."""
        self.connect_btn.setEnabled(True)
        if success:
            self.is_connected = True
            self.status_label.setText("Status: Connected & Ready")
            self.status_label.setStyleSheet("color: #4CAF50;")
            self.battery_label.setText(f"Battery: {battery}%")
            self.connect_btn.setText("Disconnect")
            self.log_message(f"Successfully connected to drone. Battery at {battery}%.")
            QMessageBox.information(self, "Success", "Successfully connected to drone!")
        else:
            self.status_label.setText("Status: Connection Failed")
            self.status_label.setStyleSheet("color: #ff6b6b;")
            self.connect_btn.setText("Connect to Drone")
            self.drone = None
            self.log_message(f"Failed to connect: {message}")
            QMessageBox.critical(self, "Connection Error", f"Failed to connect to drone:\n{message}")

    def _connect_drone_thread(self):
        self.log_message("Connection thread started.")
        try:
            self.drone.connect()
            self.log_message("Socket connection successful.")
            battery = self.drone.get_battery()
            self.log_message(f"Retrieved battery: {battery}%")
            self.drone.streamon()
            self.log_message("Video stream turned on.")
            time.sleep(2)
            self.connection_result.emit(True, "Successfully connected", battery)
        except Exception as e:
            self.log_message(f"An exception occurred during connection: {e}")
            self.connection_result.emit(False, str(e), -1)

    def on_new_gesture_data(self, processed_frame, detected_gesture):
        """This slot is executed in the main thread when the gesture worker emits its signal."""
        
        # 1. Update the webcam video label with the processed frame
        processed_frame = cv2.resize(processed_frame, (960, 720))
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.webcam_video_label.setPixmap(pixmap)

        # 2. Handle dynamic gesture locking
        # If a dynamic gesture is currently locked, we ignore new detections and force the locked gesture.
        if self.dynamic_gesture_lock:
            current_gesture = self.dynamic_gesture_lock
        else:
            # Otherwise, use the override if it exists, or the newly detected gesture.
            current_gesture = self.override_gesture if self.override_gesture else detected_gesture
            
            # Check if the new gesture is a dynamic one and should trigger a lock.
            if current_gesture in ['rotate_clockwise', 'rotate_counterclockwise']:
                self.dynamic_gesture_lock = current_gesture
                self.dynamic_gesture_timer.start(700) # Lock for 700ms
                self.log_message(f"Dynamic gesture '{current_gesture}' locked for 0.7 seconds.")

        # 3. Update gesture label in the UI
        gesture_text = current_gesture
        if self.override_gesture:
            self.gesture_label.setText(f"Gesture: {gesture_text} (Override)")
        elif self.dynamic_gesture_lock:
            self.gesture_label.setText(f"Gesture: {gesture_text} (Action!)")
        else:
            self.gesture_label.setText(f"Gesture: {gesture_text}")

        # 4. Handle one-shot commands (takeoff/land) only when the gesture changes
        if current_gesture != self.last_gesture:
            self.log_message(f"New gesture detected: '{current_gesture}'")
            # We don't want to trigger takeoff/land with a dynamic gesture
            if not self.dynamic_gesture_lock:
                self.handle_one_shot_command(current_gesture)

        # 5. Handle continuous flight commands
        self.send_rc_command_for_gesture(current_gesture)
        
        # 6. Store the gesture for the next comparison
        self.last_gesture = current_gesture

    def release_dynamic_gesture_lock(self):
        """Called by the QTimer to release the lock on a dynamic gesture."""
        if self.dynamic_gesture_lock:
            self.log_message(f"Dynamic lock on '{self.dynamic_gesture_lock}' released.")
            self.dynamic_gesture_lock = None
            # Send a stop command to ensure the drone doesn't keep rotating
            if self.is_flying and self.drone:
                self.log_message("Sending stop command after dynamic gesture.")
                self.drone.send_rc_control(0, 0, 0, 0)

    def send_rc_command_for_gesture(self, gesture):
        """Calculates and sends the appropriate RC command for a given gesture."""
        if not self.is_flying or not self.drone:
            return

        SPEED = 40
        lr, fb, ud, yv = 0, 0, 0, 0

        if gesture == "thumbs_up": fb = SPEED
        elif gesture == "thumbs_down": fb = -SPEED
        elif gesture == "Pointing Right": lr = SPEED
        elif gesture == "Pointing Left": lr = -SPEED
        elif gesture == "Pointing Up": ud = SPEED
        elif gesture == "Pointing Down": ud = -SPEED
        elif gesture == "rotate_clockwise": yv = 60
        elif gesture == "rotate_counterclockwise": yv = 60
        
        if any([lr, fb, ud, yv]):
            self.log_message(f"Sending RC command: [lr:{lr}, fb:{fb}, ud:{ud}, yv:{yv}]")
        
        try:
            self.drone.send_rc_control(lr, fb, ud, yv)
        except Exception as e:
            self.log_message(f"Error sending RC control: {e}")

    def update_drone_video(self):
        """Handles updating the drone's video feed."""
        if self.is_connected and self.drone:
            try:
                drone_frame = self.drone.get_frame_read().frame
                if drone_frame is not None:
                    # --- FIX: Ensure the drone frame is also resized ---
                    drone_frame = cv2.resize(drone_frame, (960, 720))
                    rgb_image = cv2.cvtColor(drone_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.drone_video_label.setPixmap(pixmap)
            except Exception:
                # This can happen during connection/disconnection, so we can ignore it.
                pass

    def closeEvent(self, event):
        self.log_message("Close button clicked. Shutting down.")
        
        # Cleanly stop the gesture detection thread
        self.gesture_worker.stop()
        self.gesture_thread.quit()
        self.gesture_thread.wait() # Wait for thread to finish
        self.log_message("Gesture detection thread stopped.")

        self.webcam.release()

        if self.is_connected and self.drone:
            self.log_message("Disconnecting from drone on close...")
            if self.is_flying:
                try:
                    self.log_message("Drone is flying. Attempting to land...")
                    self.drone.land()
                    self.log_message("Land command sent on close.")
                except Exception as e:
                    self.log_message(f"Error landing on close: {e}")
            try:
                self.log_message("Turning off video stream...")
                self.drone.streamoff()
                self.log_message("Streamoff command sent on close.")
            except Exception as e:
                self.log_message(f"Error on streamoff: {e}")
            try:
                self.log_message("Ending connection to drone...")
                self.drone.end()
                self.log_message("End command sent on close.")
            except Exception as e:
                self.log_message(f"Error on end: {e}")
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = DroneGestureControlWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()