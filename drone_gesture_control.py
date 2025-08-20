from djitellopy import Tello
import cv2
import time
import threading
from gestures import HandGestureDetector

class DroneGestureControl:
    def __init__(self):
        # Initialize the drone
        self.drone = Tello()
        
        # Initialize webcam for gesture detection
        self.webcam = cv2.VideoCapture(0)
        
        # Initialize gesture detector
        self.gesture_detector = HandGestureDetector()
        
        # RC Control parameters for smooth movement
        self.rc_speed = 50  # Speed for RC control (0-100)
        self.rc_duration = 0.1  # Duration for each RC command
        
        # Movement state tracking
        self.current_movement = {
            'left_right': 0,    # -100 to 100
            'forward_back': 0,  # -100 to 100
            'up_down': 0,       # -100 to 100
            'yaw': 0           # -100 to 100
        }
        
        # State variables
        self.is_flying = False
        self.last_gesture = None
        self.gesture_cooldown = 0.1  # Reduced to 0.1 seconds for immediate response
        self.last_gesture_time = 0
        self.is_hovering = False
        
        # Safety and validation
        self.gesture_hold_time = 0.8  # How long to hold a gesture before executing
        self.last_gesture_start_time = 0
        self.current_gesture = None
        self.gesture_count = 0  # Count consecutive detections of same gesture
        self.min_gesture_count = 3  # Minimum consecutive detections before executing
        
        # Critical gestures that need extra validation
        self.critical_gestures = ['OK', 'fist']  # Landing and emergency gestures
        
        # Takeoff and landing state management
        self.is_taking_off = False
        self.is_landing = False
        self.takeoff_start_time = 0
        self.landing_start_time = 0
        self.takeoff_timeout = 10  # seconds
        self.landing_timeout = 15  # seconds
        
        # RC control thread
        self.rc_thread = None
        self.rc_running = False
        
        # Crash detection variables
        self.last_battery_check = 0
        self.battery_check_interval = 2.0  # Check battery every 2 seconds
        self.last_connection_check = 0
        self.connection_check_interval = 1.0  # Check connection every 1 second
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        self.crash_detected = False
        self.last_successful_command = time.time()
        self.command_timeout = 5.0  # If no successful command for 5 seconds, consider crashed
        
    def connect_drone(self):
        """Connect to the drone and start video stream"""
        try:
            print("Connecting to drone...")
            self.drone.connect()
            print(f"Battery level: {self.drone.get_battery()}%")
            
            # Start video stream
            print("Starting video stream...")
            self.drone.streamon()
            
            # Wait a moment for stream to stabilize
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Failed to connect to drone: {str(e)}")
            return False
    
    def detect_crash(self):
        """Detect if the drone has crashed or is unresponsive"""
        current_time = time.time()
        
        # Check battery periodically
        if current_time - self.last_battery_check > self.battery_check_interval:
            try:
                battery = self.drone.get_battery()
                if battery is not None:
                    self.last_battery_check = current_time
                    self.consecutive_errors = 0  # Reset error count on successful battery check
                    
                    # If battery is critically low, consider it a crash
                    if battery < 10:
                        print(f"Critical battery level: {battery}% - treating as crash")
                        return True
            except Exception as e:
                self.consecutive_errors += 1
                print(f"Battery check failed: {e}")
        
        # Check connection periodically
        if current_time - self.last_connection_check > self.connection_check_interval:
            try:
                # Try to get a simple status command
                status = self.drone.get_battery()
                if status is not None:
                    self.last_connection_check = current_time
                    self.last_successful_command = current_time
                    self.consecutive_errors = 0
                else:
                    self.consecutive_errors += 1
            except Exception as e:
                self.consecutive_errors += 1
                print(f"Connection check failed: {e}")
        
        # Check if too many consecutive errors
        if self.consecutive_errors >= self.max_consecutive_errors:
            print(f"Too many consecutive errors ({self.consecutive_errors}) - crash detected")
            return True
        
        # Check if no successful commands for too long
        if current_time - self.last_successful_command > self.command_timeout:
            if self.is_flying:
                print("No successful commands for too long - crash detected")
                return True
        
        return False
    
    def reset_crash_state(self):
        """Reset all state variables after a crash"""
        print("Resetting crash state...")
        self.is_flying = False
        self.is_hovering = False
        self.is_taking_off = False
        self.is_landing = False
        self.crash_detected = False
        self.consecutive_errors = 0
        self.last_successful_command = time.time()
        self.stop_rc_control()
        self.stop_movement()
        print("Crash state reset complete - drone ready for takeoff")
    
    def start_rc_control(self):
        """Start the RC control thread for smooth movement"""
        if not self.rc_running:
            self.rc_running = True
            self.rc_thread = threading.Thread(target=self._rc_control_loop)
            self.rc_thread.daemon = True
            self.rc_thread.start()
            print("RC control started")
    
    def stop_rc_control(self):
        """Stop the RC control thread"""
        self.rc_running = False
        if self.rc_thread:
            self.rc_thread.join(timeout=1)
        print("RC control stopped")
    
    def _rc_control_loop(self):
        """Continuous RC control loop for smooth movement"""
        while self.rc_running and self.is_flying:
            try:
                # Send RC commands
                self.drone.send_rc_control(
                    self.current_movement['left_right'],
                    self.current_movement['forward_back'],
                    self.current_movement['up_down'],
                    self.current_movement['yaw']
                )
                self.last_successful_command = time.time()  # Update successful command time
                time.sleep(self.rc_duration)
            except Exception as e:
                print(f"RC control error: {e}")
                self.consecutive_errors += 1
                break
    
    def set_movement(self, left_right=0, forward_back=0, up_down=0, yaw=0):
        """Set movement values for RC control"""
        self.current_movement['left_right'] = left_right
        self.current_movement['forward_back'] = forward_back
        self.current_movement['up_down'] = up_down
        self.current_movement['yaw'] = yaw
    
    def stop_movement(self):
        """Stop all movement (hover)"""
        self.set_movement(0, 0, 0, 0)
    
    def takeoff_async(self):
        """Non-blocking takeoff in a separate thread"""
        def takeoff_thread():
            try:
                print("Starting takeoff...")
                self.drone.takeoff()
                print("Takeoff completed successfully")
                self.is_flying = True
                self.is_taking_off = False
                self.last_successful_command = time.time()
                # Start RC control after takeoff
                self.start_rc_control()
            except Exception as e:
                print(f"Takeoff failed: {str(e)}")
                self.is_taking_off = False
                self.is_flying = False
                self.consecutive_errors += 1
        
        # Start takeoff in a separate thread
        takeoff_thread = threading.Thread(target=takeoff_thread)
        takeoff_thread.daemon = True
        takeoff_thread.start()
    
    def land_async(self):
        """Non-blocking landing in a separate thread"""
        def landing_thread():
            try:
                print("Starting landing...")
                # Stop RC control before landing
                self.stop_rc_control()
                self.stop_movement()
                self.drone.land()
                print("Landing completed successfully")
                self.is_flying = False
                self.is_landing = False
                self.is_hovering = False
                self.last_successful_command = time.time()
            except Exception as e:
                print(f"Landing failed: {str(e)}")
                self.is_landing = False
                self.consecutive_errors += 1
        
        # Start landing in a separate thread
        landing_thread = threading.Thread(target=landing_thread)
        landing_thread.daemon = True
        landing_thread.start()
    
    def emergency_land_async(self):
        """Non-blocking emergency landing in a separate thread"""
        def emergency_thread():
            try:
                print("Emergency landing...")
                # Stop RC control before emergency landing
                self.stop_rc_control()
                self.stop_movement()
                self.drone.emergency()
                print("Emergency landing completed")
                self.is_flying = False
                self.is_landing = False
                self.is_hovering = False
                self.last_successful_command = time.time()
            except Exception as e:
                print(f"Emergency landing failed: {str(e)}")
                self.is_landing = False
                self.consecutive_errors += 1
        
        # Start emergency landing in a separate thread
        emergency_thread = threading.Thread(target=emergency_thread)
        emergency_thread.daemon = True
        emergency_thread.start()
    
    def validate_gesture(self, gesture):
        """Validate if a gesture should be executed"""
        current_time = time.time()
        
        # Don't process gestures during takeoff/landing
        if self.is_taking_off or self.is_landing:
            return False
        
        # For critical gestures (landing, emergency), require longer hold time
        if gesture in self.critical_gestures:
            required_hold_time = 1.5  # 1.5 seconds for critical gestures
        else:
            required_hold_time = self.gesture_hold_time
        
        # Check if this is the same gesture as before
        if gesture == self.current_gesture:
            self.gesture_count += 1
            
            # Check if gesture has been held long enough
            if current_time - self.last_gesture_start_time >= required_hold_time:
                if self.gesture_count >= self.min_gesture_count:
                    return True
        else:
            # New gesture detected
            self.current_gesture = gesture
            self.last_gesture_start_time = current_time
            self.gesture_count = 1
        
        return False
    
    def get_drone_status(self):
        """Get actual drone status to verify our state"""
        try:
            # Try to get battery level as a status check
            battery = self.drone.get_battery()
            if battery is None:
                return "Unknown"
            
            # Check if drone is responding to commands
            # This is a simple way to check if drone is still connected
            return "Connected"
        except:
            return "Disconnected"
    
    def process_gesture(self, gestures):
        """Process the detected gestures and control the drone - IMMEDIATE EXECUTION"""
        current_time = time.time()
        
        # Check for crash first
        if self.detect_crash():
            if not self.crash_detected:
                print("CRASH DETECTED! Resetting drone state...")
                self.crash_detected = True
                self.reset_crash_state()
            return  # Don't process gestures after crash
        
        # Minimal cooldown to prevent command spam
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        # Handle both single gesture and list of gestures
        if isinstance(gestures, list):
            # With priority system, dynamic gestures take precedence
            # So we process the first gesture in the list (highest priority)
            gesture = gestures[0] if gestures else None
        else:
            gesture = gestures
        
        # Skip if no gesture
        if not gesture:
            return
        
        # Execute the gesture immediately
        self.last_gesture = gesture
        self.last_gesture_time = current_time
        
        print(f"Executing gesture: {gesture}")
        
        try:
            # Map gestures from gestures.py to drone actions
            if gesture == 'Open':
                if not self.is_flying and not self.is_taking_off:
                    print("Initiating takeoff...")
                    self.is_taking_off = True
                    self.takeoff_start_time = current_time
                    self.takeoff_async()
                
            elif gesture == 'OK' and self.is_flying and not self.is_landing:
                print("Initiating landing...")
                self.is_landing = True
                self.landing_start_time = current_time
                self.land_async()
                
            elif gesture == 'fist' and self.is_flying and not self.is_landing:
                print("Initiating emergency landing...")
                self.is_landing = True
                self.landing_start_time = current_time
                self.emergency_land_async()
                
            elif self.is_flying and not self.is_taking_off and not self.is_landing:
                # Use RC control for smooth movement
                if gesture == 'Pointing Up':
                    print("Ascending...")
                    self.set_movement(up_down=self.rc_speed)
                    
                elif gesture == 'Pointing Down':
                    print("Descending...")
                    self.set_movement(up_down=-self.rc_speed)
                    
                elif gesture == 'Pointing Left':
                    print("Moving left...")
                    self.set_movement(left_right=-self.rc_speed)
                    
                elif gesture == 'Pointing Right':
                    print("Moving right...")
                    self.set_movement(left_right=self.rc_speed)
                    
                elif gesture == 'thumbs_up':
                    print("Moving forward...")
                    self.set_movement(forward_back=self.rc_speed)
                    
                elif gesture == 'thumbs_down':
                    print("Moving backward...")
                    self.set_movement(forward_back=-self.rc_speed)
                    
                elif gesture == 'rotate_clockwise':
                    print("Rotating clockwise...")
                    self.set_movement(yaw=self.rc_speed)
                    
                elif gesture == 'rotate_counterclockwise':
                    print("Rotating counterclockwise...")
                    self.set_movement(yaw=-self.rc_speed)
                
        except Exception as e:
            print(f"Error executing drone command: {str(e)}")
            self.consecutive_errors += 1
            # Try to reset state if there's an error
            try:
                status = self.get_drone_status()
                if status == "Disconnected":
                    self.reset_crash_state()
            except:
                pass
    
    def check_timeouts(self):
        """Check for takeoff/landing timeouts"""
        current_time = time.time()
        
        # Check takeoff timeout
        if self.is_taking_off and (current_time - self.takeoff_start_time) > self.takeoff_timeout:
            print("Takeoff timeout - resetting state")
            self.is_taking_off = False
            self.is_flying = False
        
        # Check landing timeout
        if self.is_landing and (current_time - self.landing_start_time) > self.landing_timeout:
            print("Landing timeout - resetting state")
            self.is_landing = False
    
    def run(self):
        """Main control loop"""
        if not self.connect_drone():
            return
        
        try:
            while True:
                # Check for timeouts
                self.check_timeouts()
                
                # Get frame from drone
                drone_frame = self.drone.get_frame_read().frame
                if drone_frame is None:
                    print("No drone frame received, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Get frame from webcam
                ret, webcam_frame = self.webcam.read()
                if not ret:
                    print("No webcam frame received, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Resize frames for better display
                drone_frame = cv2.resize(drone_frame, (640, 480))
                # Make webcam window bigger and flip it for natural view
                webcam_frame = cv2.resize(webcam_frame, (640, 480))
                webcam_frame = cv2.flip(webcam_frame, 1)  # Flip horizontally for mirror effect
                
                # Detect gesture from webcam
                webcam_frame, gestures = self.gesture_detector.detect_gesture(webcam_frame)
                
                # Process gesture if detected - IMMEDIATE EXECUTION
                if gestures:
                    self.process_gesture(gestures)
                    
                    # Display gesture and drone status
                    status = "Flying" if self.is_flying else "Landed"
                    if self.crash_detected:
                        status = "CRASHED - Reset Ready"
                    elif self.is_taking_off:
                        status = "Taking Off..."
                    elif self.is_landing:
                        status = "Landing..."
                    elif self.is_hovering:
                        status += " (Hovering)"
                    
                    # Handle multiple gestures display
                    if isinstance(gestures, list):
                        gesture_text = ", ".join(gestures)
                    else:
                        gesture_text = gestures
                    
                    cv2.putText(
                        drone_frame,
                        f"Gesture: {gesture_text} | Status: {status}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if not self.crash_detected else (0, 0, 255),
                        2
                    )
                
                # Display battery level
                try:
                    battery = self.drone.get_battery()
                    cv2.putText(
                        drone_frame,
                        f"Battery: {battery}%",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                except:
                    cv2.putText(
                        drone_frame,
                        "Battery: Unknown",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                
                # Display crash detection info
                if self.crash_detected:
                    cv2.putText(
                        drone_frame,
                        "CRASH DETECTED - Use 'Open' gesture to takeoff again",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
                elif self.is_flying:
                    # Display current movement values
                    movement_text = f"RC: LR:{self.current_movement['left_right']:3d} FB:{self.current_movement['forward_back']:3d} UD:{self.current_movement['up_down']:3d} Y:{self.current_movement['yaw']:3d}"
                    cv2.putText(
                        drone_frame,
                        movement_text,
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
                
                # Display both frames - webcam is now bigger and flipped
                cv2.imshow('Drone View', drone_frame)
                cv2.imshow('Gesture Detection (Mirrored)', webcam_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
        finally:
            # Stop RC control and land the drone if it's still flying
            if self.is_flying and not self.is_landing:
                try:
                    print("Emergency landing on exit...")
                    self.stop_rc_control()
                    self.stop_movement()
                    self.drone.land()
                    time.sleep(3)
                except:
                    pass
            
            # Clean up
            self.stop_rc_control()
            self.drone.streamoff()
            self.webcam.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = DroneGestureControl()
    controller.run() 