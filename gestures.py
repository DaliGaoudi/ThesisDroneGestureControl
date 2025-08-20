import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math
import tensorflow as tf
import csv
import copy
import itertools

class HandGestureDetector:
    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initializes the Hand Gesture Detector.

        Args:
            max_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence value for hand detection.
            min_tracking_confidence (float): Minimum confidence value for hand tracking.
        """
        self.previous_angle = None
        self.angle_accumulator = 0
        self.last_position = None  # Track last position for movement threshold
        self.circular_start_time = None  # Track when circular motion started

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=10)
        
        # For circular motion detection
        self.index_tip_history = deque(maxlen=20)
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.5
        
        # Cooldown for gesture acceptance (for drone control)
        self.last_accepted_gesture_time = 0  # Track last accepted gesture time
        
        # Initialize pre-trained model for static gestures
        self.keypoint_classifier = self._init_keypoint_classifier()
        
        # Load gesture labels
        self.keypoint_labels = self._load_keypoint_labels()
        
    def _init_keypoint_classifier(self):
        """Initialize the TensorFlow Lite model for static gesture classification"""
        try:
            model_path = 'hand-gesture-recognition-mediapipe/model/keypoint_classifier/keypoint_classifier.tflite'
            interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=1)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"Warning: Could not load keypoint classifier: {e}")
            return None
    
    def _load_keypoint_labels(self):
        """Load static gesture labels from CSV file"""
        try:
            labels = []
            with open('hand-gesture-recognition-mediapipe/model/keypoint_classifier/keypoint_classifier_label.csv',
                     encoding='utf-8-sig') as f:
                csv_reader = csv.reader(f)
                labels = [row[0] for row in csv_reader]
            return labels
        except Exception as e:
            print(f"Warning: Could not load keypoint labels: {e}")
            return ['Open', 'Close', 'Pointer', 'OK']

    def _get_landmark_list(self, hand_landmarks, frame_shape):
        """
        Converts normalized landmarks to pixel coordinates.

        Args:
            hand_landmarks: The landmarks for a single hand detected by MediaPipe.
            frame_shape (tuple): The shape of the camera frame (height, width).

        Returns:
            list: A list of landmark coordinates as numpy arrays.
        """
        landmark_list = []
        for lm in hand_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            px, py = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])
            landmark_list.append(np.array([px, py]))
        return landmark_list

    def _get_finger_states(self, landmarks):
        """
        Determines if each finger is extended or curled. This logic is more robust
        to hand rotation than simple y-coordinate comparisons.

        Args:
            landmarks (list): A list of landmark coordinates.

        Returns:
            dict: A dictionary with boolean values for each finger's extended state.
        """
        finger_states = {}
        
        # For the thumb, we check its horizontal extension relative to the index finger metacarpal
        # This helps differentiate a 'thumbs up' from a thumb tucked into a fist.
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # A simple distance check can determine if the thumb is extended outwards.
        # We compare the tip-to-mcp distance with ip-to-mcp distance.
        if np.linalg.norm(thumb_tip - thumb_mcp) > np.linalg.norm(thumb_ip - thumb_mcp):
             finger_states['thumb'] = True
        else:
             finger_states['thumb'] = False

        # For other fingers, we check if the tip is farther from the wrist than the pip joint.
        # This is a reliable way to check for extension regardless of orientation.
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        finger_definitions = {
            'index': [self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
            'middle': [self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            'ring': [self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP],
            'pinky': [self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP]
        }
        
        for finger, (tip_idx, pip_idx) in finger_definitions.items():
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            # If the distance from wrist to tip is greater than wrist to pip, the finger is extended.
            # Add a small tolerance for more lenient detection
            wrist_to_tip = np.linalg.norm(tip - wrist)
            wrist_to_pip = np.linalg.norm(pip - wrist)
            tolerance = wrist_to_pip * 0.1  # 10% tolerance
            
            if wrist_to_tip > (wrist_to_pip - tolerance):
                finger_states[finger] = True
            else:
                finger_states[finger] = False
                
        return finger_states

    def _is_pointing_pose(self, finger_states):
        """
        More lenient pointing pose detection that allows for slight variations.
        
        Args:
            finger_states (dict): Dictionary of finger states
            
        Returns:
            bool: True if pointing pose is detected
        """
        # Primary condition: index extended, others closed
        if finger_states['index'] and not any([finger_states['middle'], finger_states['ring'], finger_states['pinky']]):
            return True
        
        # Secondary condition: index and middle extended, others closed (more common variation)
        if finger_states['index'] and finger_states['middle'] and not any([finger_states['ring'], finger_states['pinky']]):
            return True
        
        # Tertiary condition: index extended, middle slightly extended, others closed
        if finger_states['index'] and not finger_states['ring'] and not finger_states['pinky']:
            return True
            
        return False

    def calculate_angle(self, x, y, cx, cy):
        return math.degrees(math.atan2(y - cy, x - cx))
    
    def _detect_circular_motion(self, landmarks, img):
        """
        Detects circular motion (clockwise or counterclockwise) based on index finger movement history.
        Now includes multiple safeguards to prevent false positives.

        Args:
            landmarks (list): A list of 2D landmark coordinates.
            img: Frame image.

        Returns:
            str or None: 'rotate_clockwise', 'rotate_counterclockwise', or None
        """
        
        h, w, _ = img.shape
        cx, cy = w // 2, h // 2  # Use screen center as rotation reference

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get index finger tip position (landmark 8)
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                current_position = (x, y)

                # Calculate distance from center (minimum radius requirement)
                distance_from_center = math.sqrt((x - cx)**2 + (y - cy)**2)
                min_radius = 50  # Minimum radius in pixels
                
                if distance_from_center < min_radius:
                    # Reset if too close to center
                    self.previous_angle = None
                    self.angle_accumulator = 0
                    self.last_position = None
                    self.circular_start_time = None
                    return None

                # Check minimum movement threshold
                if self.last_position is not None:
                    movement_distance = math.sqrt((x - self.last_position[0])**2 + (y - self.last_position[1])**2)
                    min_movement = 5  # Minimum movement in pixels per frame
                    
                    if movement_distance < min_movement:
                        # Too little movement, don't update angle
                        self.last_position = current_position
                        return None

                # Calculate current angle of index finger from center
                current_angle = self.calculate_angle(x, y, cx, cy)

                if self.previous_angle is not None:
                    delta = current_angle - self.previous_angle

                    # Normalize for circular overflow
                    if delta > 180:
                        delta -= 360
                    elif delta < -180:
                        delta += 360

                    # Only accumulate if the movement is significant
                    if abs(delta) > 2:  # Minimum angle change threshold
                        self.angle_accumulator += delta
                        
                        # Start timing the circular motion
                        if self.circular_start_time is None:
                            self.circular_start_time = time.time()

                        # Check for completion with time requirement
                        min_time = 0.5  # Minimum time for circular motion (seconds)
                        current_time = time.time()
                        
                        if self.circular_start_time and (current_time - self.circular_start_time) >= min_time:
                            # Threshold for detecting a full twirl (increased for more reliability)
                            if self.angle_accumulator > 270:  # Increased from 180
                                print("ccw")
                                self.angle_accumulator = 0
                                self.circular_start_time = None
                                return 'rotate_counterclockwise'
                            elif self.angle_accumulator < -270:  # Increased from -180
                                print("cw")
                                self.angle_accumulator = 0
                                self.circular_start_time = None
                                return 'rotate_clockwise'
                    else:
                        # Reset if movement is too small
                        self.angle_accumulator = 0
                        self.circular_start_time = None

                self.previous_angle = current_angle
                self.last_position = current_position

                # Draw center and finger point
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                cv2.circle(img, (x, y), 8, (255, 0, 0), -1)
                # Draw minimum radius circle
                cv2.circle(img, (cx, cy), min_radius, (0, 255, 255), 2)
        else:
            # Reset all circular motion tracking when no hand is detected
            self.previous_angle = None
            self.angle_accumulator = 0
            self.last_position = None
            self.circular_start_time = None

        return None

    def _classify_gesture(self, landmarks, hand_landmarks, frame_shape, frame):
        """
        Classifies the hand gesture using ML for static gestures and improved circular motion detection.
        Now detects both static and dynamic gestures simultaneously with priority system.

        Args:
            landmarks (list): A list of 2D landmark coordinates.
            hand_landmarks: The original MediaPipe hand landmarks.
            frame_shape (tuple): The shape of the camera frame (height, width).
            frame: The camera frame for visualization.

        Returns:
            list: A list of detected gestures with priority system applied.
        """
        detected_gestures = []
        dynamic_gestures = []
        static_gestures = []
        
        # First, try ML-based classification for static gestures
        ml_gesture = self._classify_gesture_with_ml(landmarks, hand_landmarks, frame_shape)
        if ml_gesture:
            # If it's a pointing gesture, determine the direction
            if ml_gesture == 'Pointer':
                pointing_direction = self._detect_pointing_direction(landmarks, hand_landmarks, frame_shape)
                if pointing_direction:
                    static_gestures.append(pointing_direction)
                else:
                    static_gestures.append('Pointer')  # Return generic pointer if direction not determined
            else:
                static_gestures.append(ml_gesture)
        
        # Check for circular motion regardless of finger pose
        circular_gesture = self._detect_circular_motion(landmarks, frame)
        if circular_gesture:
            dynamic_gestures.append(circular_gesture)
        
        # Check for finger states for additional gestures
        finger_states = self._get_finger_states(landmarks)
        
        # Check for pointing pose for directional gestures (if not already detected by ML)
        is_pointing_pose = self._is_pointing_pose(finger_states)
        
        # If not in a pointing pose, clear the history
        if not is_pointing_pose:
            self.index_tip_history.clear()
        
        # Check for thumbs up/down only when other fingers are closed
        if not any([finger_states['index'], finger_states['middle'], finger_states['ring'], finger_states['pinky']]):
            thumb_gesture = self._detect_thumb_direction(landmarks, hand_landmarks)
            if thumb_gesture:
                static_gestures.append(thumb_gesture)
        
        # Priority system: Dynamic gestures take precedence over static gestures
        if dynamic_gestures:
            # If dynamic gestures are detected, only return those
            detected_gestures = dynamic_gestures
            print(f"Dynamic gesture detected: {dynamic_gestures} - Overriding static gestures")
        else:
            # If no dynamic gestures, return static gestures
            detected_gestures = static_gestures
        
        return detected_gestures if detected_gestures else None

    def _detect_pointing_direction(self, landmarks, hand_landmarks, frame_shape):
        """
        Detects the direction the index finger is pointing.
        
        Args:
            landmarks (list): A list of 2D landmark coordinates.
            hand_landmarks: The original MediaPipe hand landmarks.
            frame_shape (tuple): The shape of the camera frame (height, width).
            
        Returns:
            str or None: The pointing direction ('Pointing Up', 'Pointing Down', 'Pointing Left', 'Pointing Right')
        """
        # Get index finger tip and wrist positions
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # Calculate horizontal and vertical components
        dx = index_tip[0] - wrist[0]
        dy = index_tip[1] - wrist[1]
        
        # Calculate angle (note: y-axis is flipped in image coordinates)
        angle = math.degrees(math.atan2(-dy, dx))  # Negative dy because y increases downward
        angle = (angle + 360) % 360
        
        # Determine direction based on angle ranges
        # 0° = pointing right, 90° = pointing up, 180° = pointing left, 270° = pointing down
        if 45 <= angle <= 135:  # Pointing up
            return 'Pointing Up'
        elif 225 <= angle <= 315:  # Pointing down
            return 'Pointing Down'
        elif 135 < angle < 225:  # Pointing left
            return 'Pointing Left'
        elif angle < 45 or angle > 315:  # Pointing right
            return 'Pointing Right'
        
        return None

    def _detect_thumb_direction(self, landmarks, hand_landmarks):
        """
        Detects if thumb is pointing up or down.
        
        Args:
            landmarks (list): A list of 2D landmark coordinates.
            hand_landmarks: The original MediaPipe hand landmarks.
            
        Returns:
            str or None: 'thumbs_up' or 'thumbs_down'
        """
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # Calculate thumb extension relative to wrist
        thumb_extension = thumb_tip[1] - wrist[1]
        
        # Check if thumb is significantly extended upward or downward
        if thumb_extension < -30:  # Thumb is above wrist (thumbs up)
            return 'thumbs_up'
        elif thumb_extension > 30:  # Thumb is below wrist (thumbs down)
            return 'thumbs_down'
        
        return None

    def _get_palm_orientation(self, hand_landmarks):
        """
        Determines if the palm or back of the hand is facing the camera using z-coordinates.
        Returns 'palm' or 'back'.
        """
        # Palm landmarks
        palm_indices = [
            self.mp_hands.HandLandmark.WRIST,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]
        # Finger tip landmarks
        tip_indices = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        palm_z = np.mean([hand_landmarks.landmark[i].z for i in palm_indices])
        tip_z = np.mean([hand_landmarks.landmark[i].z for i in tip_indices])
        # If palm_z < tip_z, palm is facing camera
        return 'palm' if palm_z < tip_z else 'back'

    def detect_gesture(self, frame):
        """
        Detects hand gestures in the given frame.

        Args:
            frame: The camera frame.

        Returns:
            tuple: The frame with landmarks drawn and the smoothed, detected gesture.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_gesture = None
        circular_gesture = None
        palm_center = None
        hand_detected = False  # Flag to track if hand is detected
        current_time = time.time()  # Get current time for cooldown check
        
        if results.multi_hand_landmarks:
            hand_detected = True  # Set flag when hand is detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Convert landmarks to pixel coordinates
                landmarks = self._get_landmark_list(hand_landmarks, (h, w))
                
                # Calculate screen center
                center_x = w // 2
                center_y = h // 2
                screen_center = (center_x, center_y)
                # Draw screen center
                cv2.circle(frame, screen_center, 8, (255, 0, 255), -1)
                
                # --- PALM ORIENTATION DETECTION ---
                palm_orientation = self._get_palm_orientation(hand_landmarks)
                cv2.putText(frame, f"Palm: {palm_orientation}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                # --- END PALM ORIENTATION ---
                
                # Classify gesture
                gestures = self._classify_gesture(landmarks, hand_landmarks, (h, w), frame)
                # --- COOLDOWN LOGIC START ---
                gestures_to_add = []
                if gestures:
                    # Only accept gestures if cooldown has passed
                    if current_time - self.last_accepted_gesture_time >= 2.0:
                        if isinstance(gestures, list):
                            for gesture in gestures:
                                gestures_to_add.append(gesture)
                        else:
                            gestures_to_add.append(gestures)
                        self.last_accepted_gesture_time = current_time  # Update last accepted time
                    else:
                        # Cooldown not passed, ignore gestures
                        gestures_to_add = []
                # Add gestures to history if accepted
                for gesture in gestures_to_add:
                    self.gesture_history.append(gesture)
                # --- COOLDOWN LOGIC END ---
                
                # Visual feedback for circular motion
                if gestures:
                    if isinstance(gestures, list):
                        for gesture in gestures:
                            if gesture in ['rotate_clockwise', 'rotate_counterclockwise']:
                                circular_gesture = gesture
                                color = (0, 255, 0) if gesture == 'rotate_clockwise' else (0, 0, 255)
                                text = 'Clockwise' if gesture == 'rotate_clockwise' else 'Counterclockwise'
                                cv2.putText(frame, f'Circle: {text}', (center_x + 20, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                                cv2.circle(frame, screen_center, 30, color, 4)
                                print(f'Circular gesture detected: {text}')
                    else:
                        if gestures in ['rotate_clockwise', 'rotate_counterclockwise']:
                            circular_gesture = gestures
                            color = (0, 255, 0) if gestures == 'rotate_clockwise' else (0, 0, 255)
                            text = 'Clockwise' if gestures == 'rotate_clockwise' else 'Counterclockwise'
                            cv2.putText(frame, f'Circle: {text}', (center_x + 20, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                            cv2.circle(frame, screen_center, 30, color, 4)
                            print(f'Circular gesture detected: {text}')
                
                # Only process the first hand found
                break
        
        # Clear gesture history if no hand is detected
        if not hand_detected:
            self.gesture_history.clear()
            detected_gestures = []
        else:
            # Get current gestures from history (last few frames)
            if len(self.gesture_history) > 0:
                # Get the most recent gestures (last 3 frames)
                recent_gestures = list(self.gesture_history)[-3:]
                # Remove duplicates while preserving order
                unique_gestures = []
                for gesture in recent_gestures:
                    if gesture is not None and gesture not in unique_gestures:
                        unique_gestures.append(gesture)
                detected_gestures = unique_gestures
            else:
                detected_gestures = []
        
        # Always show gesture status in top left
        overlay = frame.copy()
        # Make the box bigger to accommodate multiple gestures
        box_height = 80 + (len(detected_gestures) - 1) * 30 if detected_gestures else 80
        cv2.rectangle(overlay, (10, 10), (400, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Add a border around the status box
        cv2.rectangle(frame, (10, 10), (400, box_height), (255, 255, 255), 2)
        
        if detected_gestures and hand_detected:  # Only show gestures if hand is detected
            # Define gesture descriptions and colors
            gesture_info = {
                # ML-based static gestures (exact names)
                'Open': {'name': 'Open', 'color': (0, 255, 255), 'action': 'Takeoff'},
                'Pointer': {'name': 'Pointer', 'color': (255, 255, 0), 'action': 'Ready for motion'},
                'OK': {'name': 'OK', 'color': (255, 0, 255), 'action': 'Land'},
                
                # Pointing directions
                'Pointing Up': {'name': 'Pointing Up', 'color': (255, 255, 0), 'action': 'Ascend'},
                'Pointing Down': {'name': 'Pointing Down', 'color': (255, 255, 0), 'action': 'Descend'},
                'Pointing Left': {'name': 'Pointing Left', 'color': (255, 255, 0), 'action': 'Rotate Left'},
                'Pointing Right': {'name': 'Pointing Right', 'color': (255, 255, 0), 'action': 'Rotate Right'},
                
                # Circular motion gestures (twirl.py approach)
                'rotate_clockwise': {'name': 'Rotate Clockwise', 'color': (0, 255, 0), 'action': 'Rotation Detected!'},
                'rotate_counterclockwise': {'name': 'Rotate Counter-clockwise', 'color': (0, 255, 0), 'action': 'Rotation Detected!'},
                
                # Rule-based thumb gestures
                'thumbs_up': {'name': 'Thumbs Up', 'color': (0, 255, 255), 'action': 'Forward'},
                'thumbs_down': {'name': 'Thumbs Down', 'color': (0, 255, 255), 'action': 'Backward'},
            }
            
            # Check if any dynamic gestures are present (priority indicator)
            has_dynamic = any(gesture in ['rotate_clockwise', 'rotate_counterclockwise'] for gesture in detected_gestures)
            
            # Display each gesture on a separate line
            for i, gesture in enumerate(detected_gestures):
                info = gesture_info.get(gesture, {'name': gesture, 'color': (255, 255, 255), 'action': ''})
                y_position = 40 + i * 30
                
                # Add priority indicator for dynamic gestures
                priority_text = " [PRIORITY]" if has_dynamic and gesture in ['rotate_clockwise', 'rotate_counterclockwise'] else ""
                
                cv2.putText(
                    frame,
                    f"{info['name']} | {info['action']}{priority_text}",
                    (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    info['color'],
                    2
                )
            
            # Show priority mode indicator
            if has_dynamic:
                cv2.putText(
                    frame,
                    "DYNAMIC MODE - Static gestures overridden",
                    (20, box_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
        else:
            # Show "No hand detected" when no hand is in frame
            status_text = "No hand detected" if not hand_detected else "No gesture detected"
            cv2.putText(
                frame,
                status_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (128, 128, 128),
                2
            )
        
        return frame, detected_gestures if hand_detected else None

    def run(self):
        """
        Main loop for running the gesture detection from the webcam.
        """
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break
                
                # Flip for a selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Detect gesture
                frame, gestures = self.detect_gesture(frame)
                
                cv2.imshow('Hand Gesture Detection', frame)
                
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

    def _classify_gesture_with_ml(self, landmarks, hand_landmarks, frame_shape):
        """
        Classify gestures using the pre-trained ML model for static gestures.
        """
        if self.keypoint_classifier is None:
            return None
            
        # Convert landmarks to the format expected by the classifier
        landmark_list = []
        for lm in hand_landmarks.landmark:
            px = min(int(lm.x * frame_shape[1]), frame_shape[1] - 1)
            py = min(int(lm.y * frame_shape[0]), frame_shape[0] - 1)
            landmark_list.append([px, py])
        
        # Preprocess landmarks
        preprocessed_landmarks = self._preprocess_landmark(landmark_list)
        
        # Classify static gesture
        try:
            input_details_tensor_index = self.keypoint_classifier.get_input_details()[0]['index']
            self.keypoint_classifier.set_tensor(
                input_details_tensor_index,
                np.array([preprocessed_landmarks], dtype=np.float32))
            self.keypoint_classifier.invoke()

            output_details_tensor_index = self.keypoint_classifier.get_output_details()[0]['index']
            result = self.keypoint_classifier.get_tensor(output_details_tensor_index)
            result_array = np.squeeze(result)
            hand_sign_id = np.argmax(result_array)
            confidence = result_array[hand_sign_id]
            
            # Add confidence threshold to prevent false positives
            if confidence < 0.7:  # Only classify if confidence is above 70%
                return None
            
            # Get the gesture label
            if hand_sign_id < len(self.keypoint_labels):
                static_gesture = self.keypoint_labels[hand_sign_id]
                
                # Skip "Close" gesture to avoid conflict with thumbs up/down
                if static_gesture == 'Close':
                    return None
                
                # Return the exact gesture name from the ML model
                return static_gesture
                    
        except Exception as e:
            print(f"Error in ML classification: {e}")
            
        return None

    def _preprocess_landmark(self, landmark_list):
        """
        Preprocess landmarks for the keypoint classifier.
        Converts to relative coordinates and normalizes.
        """
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def _preprocess_point_history(self, image, point_history):
        """
        Preprocess point history for the point history classifier.
        """
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

if __name__ == "__main__":
    detector = HandGestureDetector()
    detector.run()