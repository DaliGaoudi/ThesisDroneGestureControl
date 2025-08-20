import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from djitellopy import Tello
from gestures import HandGestureDetector

class DroneGestureControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Gesture Control System")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize drone and gesture detector
        self.drone = Tello()
        
        # Initialize webcam for gesture detection with optimized settings
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.webcam.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize gesture detector with optimized settings
        self.gesture_detector = HandGestureDetector(
            max_hands=1,  # Only detect one hand for faster processing
            min_detection_confidence=0.6,  # Lower threshold for faster detection
            min_tracking_confidence=0.5
        )
        
        # State variables
        self.is_connected = False
        self.is_flying = False
        self.is_taking_off = False
        self.is_landing = False
        self.crash_detected = False
        self.consecutive_errors = 0
        self.last_successful_command = time.time()
        
        # RC Control parameters
        self.rc_speed = 50
        self.rc_duration = 0.1
        self.current_movement = {
            'left_right': 0,
            'forward_back': 0,
            'up_down': 0,
            'yaw': 0
        }
        
        # Threading variables
        self.rc_thread = None
        self.rc_running = False
        self.video_thread = None
        self.video_running = False
        
        # Video display variables
        self.drone_frame_tk = None
        self.webcam_frame_tk = None
        
        # Performance mode toggle
        self.performance_mode = tk.BooleanVar(value=True)
        
        self.setup_gui()
        
        # Start webcam gesture detection immediately
        self.video_running = True
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Drone Gesture Control System", 
            font=("Arial", 20, "bold"),
            bg='#2b2b2b',
            fg='white'
        )
        title_label.pack(pady=(0, 20))
        
        # Control panel frame
        control_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Connection button
        self.connect_btn = tk.Button(
            control_frame,
            text="Connect to Drone",
            command=self.connect_drone,
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='white',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10
        )
        self.connect_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Status: Gesture Detection Active",
            font=("Arial", 12),
            bg='#3b3b3b',
            fg='#4CAF50'
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Battery label
        self.battery_label = tk.Label(
            control_frame,
            text="Battery: --%",
            font=("Arial", 12),
            bg='#3b3b3b',
            fg='white'
        )
        self.battery_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Gesture label
        self.gesture_label = tk.Label(
            control_frame,
            text="Gesture: None",
            font=("Arial", 12),
            bg='#3b3b3b',
            fg='white'
        )
        self.gesture_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Performance mode toggle
        self.performance_btn = tk.Checkbutton(
            control_frame,
            text="Fast Mode",
            variable=self.performance_mode,
            font=("Arial", 10),
            bg='#3b3b3b',
            fg='white',
            selectcolor='#2b2b2b',
            activebackground='#3b3b3b',
            activeforeground='white'
        )
        self.performance_btn.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Video display frame
        video_frame = tk.Frame(main_frame, bg='#2b2b2b')
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Drone video frame
        drone_video_frame = tk.Frame(video_frame, bg='#1b1b1b', relief=tk.SUNKEN, bd=2)
        drone_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        drone_title = tk.Label(
            drone_video_frame,
            text="Drone View",
            font=("Arial", 14, "bold"),
            bg='#1b1b1b',
            fg='white'
        )
        drone_title.pack(pady=5)
        
        self.drone_canvas = tk.Canvas(
            drone_video_frame,
            bg='black',
            width=640,
            height=480
        )
        self.drone_canvas.pack(padx=10, pady=10)
        
        # Webcam video frame
        webcam_video_frame = tk.Frame(video_frame, bg='#1b1b1b', relief=tk.SUNKEN, bd=2)
        webcam_video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        webcam_title = tk.Label(
            webcam_video_frame,
            text="Gesture Detection (Mirrored)",
            font=("Arial", 14, "bold"),
            bg='#1b1b1b',
            fg='white'
        )
        webcam_title.pack(pady=5)
        
        self.webcam_canvas = tk.Canvas(
            webcam_video_frame,
            bg='black',
            width=640,
            height=480
        )
        self.webcam_canvas.pack(padx=10, pady=10)
        
        # Instructions frame
        instructions_frame = tk.Frame(main_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        instructions_frame.pack(fill=tk.X, pady=(20, 0))
        
        instructions_text = """
        Gesture Controls:
        • Open Hand: Takeoff
        • OK Sign: Land
        • Fist: Emergency Landing
        • Pointing Up/Down: Ascend/Descend
        • Pointing Left/Right: Move Left/Right
        • Thumbs Up/Down: Move Forward/Backward
        • Circular Motion: Rotate Clockwise/Counter-clockwise
        """
        
        instructions_label = tk.Label(
            instructions_frame,
            text=instructions_text,
            font=("Arial", 10),
            bg='#3b3b3b',
            fg='white',
            justify=tk.LEFT
        )
        instructions_label.pack(padx=10, pady=10)
        
    def connect_drone(self):
        """Connect to the drone"""
        if not self.is_connected:
            self.connect_btn.config(text="Connecting...", state=tk.DISABLED)
            self.status_label.config(text="Status: Connecting...", fg='orange')
            
            # Run connection in a separate thread
            threading.Thread(target=self._connect_drone_thread, daemon=True).start()
    
    def _connect_drone_thread(self):
        """Thread for connecting to drone"""
        try:
            print("Connecting to drone...")
            self.drone.connect()
            battery = self.drone.get_battery()
            print(f"Battery level: {battery}%")
            
            # Start video stream
            print("Starting video stream...")
            self.drone.streamon()
            time.sleep(2)  # Wait for stream to stabilize
            
            # Update GUI in main thread
            self.root.after(0, self._connection_success, battery)
            
        except Exception as e:
            print(f"Failed to connect to drone: {str(e)}")
            self.root.after(0, self._connection_failed, str(e))
    
    def _connection_success(self, battery):
        """Handle successful connection"""
        self.is_connected = True
        self.connect_btn.config(text="Disconnect", state=tk.NORMAL, bg='#f44336')
        self.status_label.config(text="Status: Connected & Ready", fg='#4CAF50')
        self.battery_label.config(text=f"Battery: {battery}%")
        
        messagebox.showinfo("Success", "Successfully connected to drone!")
    
    def _connection_failed(self, error):
        """Handle connection failure"""
        self.connect_btn.config(text="Connect to Drone", state=tk.NORMAL, bg='#4CAF50')
        self.status_label.config(text="Status: Connection Failed", fg='#ff6b6b')
        messagebox.showerror("Connection Error", f"Failed to connect to drone:\n{error}")
    
    def _video_loop(self):
        """Main video processing loop - optimized for faster gesture detection"""
        frame_count = 0
        last_battery_update = time.time()
        
        while self.video_running:
            try:
                frame_count += 1
                current_time = time.time()
                
                # Get webcam frame first (prioritize gesture detection)
                ret, webcam_frame = self.webcam.read()
                if ret:
                    # Process gestures immediately with adaptive frame size based on performance mode
                    if self.performance_mode.get():
                        gesture_frame = cv2.resize(webcam_frame, (320, 240))  # Smaller for faster processing
                    else:
                        gesture_frame = cv2.resize(webcam_frame, (640, 480))  # Full size for accuracy
                    
                    gesture_frame = cv2.flip(gesture_frame, 1)  # Mirror effect
                    
                    # Process gestures (this is the critical path)
                    gesture_frame, gestures = self.gesture_detector.detect_gesture(gesture_frame)
                    
                    # Process gestures immediately if detected (only if drone is connected)
                    if gestures and self.is_connected:
                        self.process_gesture(gestures)
                    
                    # Update gesture label regardless of connection status
                    if gestures:
                        if isinstance(gestures, list):
                            gesture_text = ", ".join(gestures)
                        else:
                            gesture_text = gestures
                        
                        self.root.after(0, lambda: self.gesture_label.config(text=f"Gesture: {gesture_text}"))
                    else:
                        self.root.after(0, lambda: self.gesture_label.config(text="Gesture: None"))
                    
                    # Update display frequency based on performance mode
                    display_frequency = 2 if self.performance_mode.get() else 3
                    if frame_count % display_frequency == 0:
                        # Resize for display
                        display_frame = cv2.resize(webcam_frame, (640, 480))
                        display_frame = cv2.flip(display_frame, 1)
                        
                        # Convert to PhotoImage for display
                        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        display_image = Image.fromarray(display_frame_rgb)
                        webcam_photo = ImageTk.PhotoImage(display_image)
                        
                        # Update webcam display
                        self.root.after(0, lambda: self._update_webcam_display(webcam_photo))
                
                # Get drone frame only if connected
                if self.is_connected:
                    try:
                        drone_frame = self.drone.get_frame_read().frame
                        if drone_frame is not None and frame_count % 2 == 0:  # Update drone view every 2nd frame
                            drone_frame = cv2.resize(drone_frame, (640, 480))
                            
                            # Convert drone frame to PhotoImage
                            drone_frame_rgb = cv2.cvtColor(drone_frame, cv2.COLOR_BGR2RGB)
                            drone_image = Image.fromarray(drone_frame_rgb)
                            drone_photo = ImageTk.PhotoImage(drone_image)
                            
                            # Update drone display
                            self.root.after(0, lambda: self._update_drone_display(drone_photo))
                        
                        # Update battery less frequently (every 3 seconds)
                        if current_time - last_battery_update > 3.0:
                            try:
                                battery = self.drone.get_battery()
                                self.root.after(0, lambda: self.battery_label.config(text=f"Battery: {battery}%"))
                                last_battery_update = current_time
                            except:
                                pass
                    except Exception as e:
                        print(f"Drone frame error: {e}")
                else:
                    # Show placeholder for drone view when not connected
                    if frame_count % 30 == 0:  # Update placeholder every 30 frames
                        self.root.after(0, lambda: self._update_drone_placeholder())
                
                # Adaptive frame rate based on performance mode
                if self.performance_mode.get():
                    time.sleep(0.016)  # ~60 FPS for fast mode
                else:
                    time.sleep(0.033)  # ~30 FPS for normal mode
                
            except Exception as e:
                print(f"Video loop error: {e}")
                time.sleep(0.05)  # Shorter error recovery time
    
    def _update_drone_display(self, photo):
        """Update drone video display"""
        self.drone_canvas.delete("all")
        self.drone_canvas.create_image(320, 240, image=photo, anchor=tk.CENTER)
        self.drone_frame_tk = photo  # Keep a reference
    
    def _update_webcam_display(self, photo):
        """Update webcam video display"""
        self.webcam_canvas.delete("all")
        self.webcam_canvas.create_image(320, 240, image=photo, anchor=tk.CENTER)
        self.webcam_frame_tk = photo  # Keep a reference
    
    def _update_drone_placeholder(self):
        """Update drone display with placeholder when not connected"""
        self.drone_canvas.delete("all")
        self.drone_canvas.create_text(
            320, 240, 
            text="Connect to Drone\nfor Video Feed", 
            fill="white", 
            font=("Arial", 16, "bold"),
            anchor=tk.CENTER
        )
    
    def detect_crash(self):
        """Detect if the drone has crashed"""
        current_time = time.time()
        
        try:
            battery = self.drone.get_battery()
            if battery is not None:
                self.consecutive_errors = 0
                if battery < 10:
                    return True
        except:
            self.consecutive_errors += 1
        
        if self.consecutive_errors >= 3:
            return True
        
        if current_time - self.last_successful_command > 5.0:
            if self.is_flying:
                return True
        
        return False
    
    def reset_crash_state(self):
        """Reset all state variables after a crash"""
        print("Resetting crash state...")
        self.is_flying = False
        self.is_taking_off = False
        self.is_landing = False
        self.crash_detected = False
        self.consecutive_errors = 0
        self.last_successful_command = time.time()
        self.stop_rc_control()
        self.stop_movement()
        self.root.after(0, lambda: self.status_label.config(text="Status: Crashed - Reset Ready", fg='red'))
    
    def start_rc_control(self):
        """Start RC control thread"""
        if not self.rc_running:
            self.rc_running = True
            self.rc_thread = threading.Thread(target=self._rc_control_loop, daemon=True)
            self.rc_thread.start()
    
    def stop_rc_control(self):
        """Stop RC control thread"""
        self.rc_running = False
        if self.rc_thread:
            self.rc_thread.join(timeout=1)
    
    def _rc_control_loop(self):
        """RC control loop"""
        while self.rc_running and self.is_flying:
            try:
                self.drone.send_rc_control(
                    self.current_movement['left_right'],
                    self.current_movement['forward_back'],
                    self.current_movement['up_down'],
                    self.current_movement['yaw']
                )
                self.last_successful_command = time.time()
                time.sleep(self.rc_duration)
            except Exception as e:
                print(f"RC control error: {e}")
                self.consecutive_errors += 1
                break
    
    def set_movement(self, left_right=0, forward_back=0, up_down=0, yaw=0):
        """Set movement values"""
        self.current_movement['left_right'] = left_right
        self.current_movement['forward_back'] = forward_back
        self.current_movement['up_down'] = up_down
        self.current_movement['yaw'] = yaw
    
    def stop_movement(self):
        """Stop all movement"""
        self.set_movement(0, 0, 0, 0)
    
    def takeoff_async(self):
        """Non-blocking takeoff"""
        def takeoff_thread():
            try:
                print("Starting takeoff...")
                self.drone.takeoff()
                print("Takeoff completed successfully")
                self.is_flying = True
                self.is_taking_off = False
                self.last_successful_command = time.time()
                self.start_rc_control()
                self.root.after(0, lambda: self.status_label.config(text="Status: Flying", fg='green'))
            except Exception as e:
                print(f"Takeoff failed: {str(e)}")
                self.is_taking_off = False
                self.is_flying = False
                self.consecutive_errors += 1
        
        threading.Thread(target=takeoff_thread, daemon=True).start()
    
    def land_async(self):
        """Non-blocking landing"""
        def landing_thread():
            try:
                print("Starting landing...")
                self.stop_rc_control()
                self.stop_movement()
                self.drone.land()
                print("Landing completed successfully")
                self.is_flying = False
                self.is_landing = False
                self.last_successful_command = time.time()
                self.root.after(0, lambda: self.status_label.config(text="Status: Landed", fg='orange'))
            except Exception as e:
                print(f"Landing failed: {str(e)}")
                self.is_landing = False
                self.consecutive_errors += 1
        
        threading.Thread(target=landing_thread, daemon=True).start()
    
    def emergency_land_async(self):
        """Non-blocking emergency landing"""
        def emergency_thread():
            try:
                print("Emergency landing...")
                self.stop_rc_control()
                self.stop_movement()
                self.drone.emergency()
                print("Emergency landing completed")
                self.is_flying = False
                self.is_landing = False
                self.last_successful_command = time.time()
                self.root.after(0, lambda: self.status_label.config(text="Status: Emergency Landed", fg='red'))
            except Exception as e:
                print(f"Emergency landing failed: {str(e)}")
                self.is_landing = False
                self.consecutive_errors += 1
        
        threading.Thread(target=emergency_thread, daemon=True).start()
    
    def process_gesture(self, gestures):
        """Process detected gestures"""
        # Check for crash first
        if self.detect_crash():
            if not self.crash_detected:
                print("CRASH DETECTED! Resetting drone state...")
                self.crash_detected = True
                self.reset_crash_state()
            return
        
        # Handle both single gesture and list of gestures
        if isinstance(gestures, list):
            gesture = gestures[0] if gestures else None
        else:
            gesture = gestures
        
        if not gesture:
            return
        
        print(f"Executing gesture: {gesture}")
        
        try:
            if gesture == 'Open':
                if not self.is_flying and not self.is_taking_off:
                    print("Initiating takeoff...")
                    self.is_taking_off = True
                    self.takeoff_async()
                
            elif gesture == 'OK' and self.is_flying and not self.is_landing:
                print("Initiating landing...")
                self.is_landing = True
                self.land_async()
                
            elif gesture == 'fist' and self.is_flying and not self.is_landing:
                print("Initiating emergency landing...")
                self.is_landing = True
                self.emergency_land_async()
                
            elif self.is_flying and not self.is_taking_off and not self.is_landing:
                if gesture == 'Pointing Up':
                    self.set_movement(up_down=self.rc_speed)
                elif gesture == 'Pointing Down':
                    self.set_movement(up_down=-self.rc_speed)
                elif gesture == 'Pointing Left':
                    self.set_movement(left_right=-self.rc_speed)
                elif gesture == 'Pointing Right':
                    self.set_movement(left_right=self.rc_speed)
                elif gesture == 'thumbs_up':
                    self.set_movement(forward_back=self.rc_speed)
                elif gesture == 'thumbs_down':
                    self.set_movement(forward_back=-self.rc_speed)
                elif gesture == 'rotate_clockwise':
                    self.set_movement(yaw=self.rc_speed)
                elif gesture == 'rotate_counterclockwise':
                    self.set_movement(yaw=-self.rc_speed)
                
        except Exception as e:
            print(f"Error executing drone command: {str(e)}")
            self.consecutive_errors += 1
    
    def on_closing(self):
        """Handle window closing"""
        self.video_running = False
        self.stop_rc_control()
        
        if self.is_flying:
            try:
                self.drone.land()
                time.sleep(3)
            except:
                pass
        
        try:
            self.drone.streamoff()
            self.webcam.release()
        except:
            pass
        
        self.root.destroy()

def main():
    root = tk.Tk()
    app = DroneGestureControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 