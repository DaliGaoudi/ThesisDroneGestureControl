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
    def __init__(self, max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initializes the Hand Gesture Detector.

        Args:
            max_hands (int): Maximum number of hands to detect. Changed to 2 for modifier logic.
            min_detection_confidence (float): Minimum confidence value for hand detection.
            min_tracking_confidence (float): Minimum confidence value for hand tracking.
        """
        self.previous_angle = None
        self.angle_accumulator = 0

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
    
    def _detect_circular_motion(self, hand_landmarks, w, h):
        """
        Detects circular motion based on index finger movement history.
        This is a simplified version that relies on external state (previous_angle, etc.).

        Args:
            hand_landmarks (list): A list of 2D landmark coordinates.
            w (int): Width of the frame.
            h (int): Height of the frame.

        Returns:
            str or None: 'rotate_clockwise', 'rotate_counterclockwise', or None
        """
        cx, cy = w // 2, h // 2
        
        # Get index finger tip position from the provided landmarks
        x = int(hand_landmarks.landmark[8].x * w)
        y = int(hand_landmarks.landmark[8].y * h)
        
        current_angle = self.calculate_angle(x, y, cx, cy)

        gesture = None
        if self.previous_angle is not None:
            delta = current_angle - self.previous_angle
            if delta > 180: delta -= 360
            elif delta < -180: delta += 360
            self.angle_accumulator += delta

            if self.angle_accumulator > 180:
                self.angle_accumulator = 0
                gesture = 'rotate_counterclockwise'
            elif self.angle_accumulator < -180:
                self.angle_accumulator = 0
                gesture = 'rotate_clockwise'
        
        self.previous_angle = current_angle
        return gesture

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
            list: A list of detected gestures with priority system.
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
        circular_gesture = self._detect_circular_motion(hand_landmarks, frame_shape[1], frame_shape[0])
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
        
        # FIX: Return only the first (most important) gesture as a string, not a list.
        if detected_gestures:
            return detected_gestures[0]
        
        return None

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

    def _classify_static_gesture_only(self, landmarks, hand_landmarks, frame_shape):
        """
        Classifies only the static pose of a hand, ignoring dynamic gestures.
        Used to identify the 'Open' modifier hand without ambiguity.

        Args:
            landmarks (list): A list of 2D landmark coordinates.
            hand_landmarks: The original MediaPipe hand landmarks.
            frame_shape (tuple): The shape of the camera frame (height, width).

        Returns:
            str or None: The static gesture ('Open', 'thumbs_up', etc.) or None
        """
        # 1. Classify the gesture using the ML model
        ml_gesture = self._classify_gesture_with_ml(landmarks, hand_landmarks, frame_shape)
        
        if ml_gesture == 'Pointer':
            return "Pointer"

        if ml_gesture:
             return ml_gesture # Return 'Open', 'OK', etc.

        # 2. Check for thumbs up/down
        thumb_gesture = self._detect_thumb_direction(landmarks, hand_landmarks)
        if thumb_gesture:
            return thumb_gesture
        
        # 3. If nothing else, check finger states for a basic open/close
        finger_states = self._get_finger_states(landmarks)
        if finger_states['index'] and finger_states['middle'] and finger_states['ring'] and finger_states['pinky'] and finger_states['thumb']:
            return 'Open'

        return None

    def detect_gesture(self, frame):
        """
        Main gesture detection function. Implements two-handed modifier logic.
        - If 1 hand is visible, all gestures are detected.
        - If 2 hands are visible and one is 'Open', the other hand is limited to dynamic gestures.

        Args:
            frame: The camera frame.

        Returns:
            tuple: The frame with landmarks drawn and the smoothed, detected gesture.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        final_gesture = None
        hand_detected = bool(results.multi_hand_landmarks)
        
        if hand_detected:
            num_hands = len(results.multi_hand_landmarks)
            
            # Draw landmarks for all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if num_hands == 2:
                # Two-handed logic: check for an 'Open' modifier hand
                landmarks1 = self._get_landmark_list(results.multi_hand_landmarks[0], (h, w))
                landmarks2 = self._get_landmark_list(results.multi_hand_landmarks[1], (h, w))
                
                static_gesture1 = self._classify_static_gesture_only(landmarks1, results.multi_hand_landmarks[0], (h, w))
                static_gesture2 = self._classify_static_gesture_only(landmarks2, results.multi_hand_landmarks[1], (h, w))

                primary_hand_landmarks = None
                modifier_active = False

                if static_gesture2 == 'Open':
                    # Hand 1 is the primary hand, Hand 2 is the modifier
                    primary_hand_landmarks = results.multi_hand_landmarks[0]
                    modifier_active = True
                    cv2.putText(frame, "Modifier Active", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif static_gesture1 == 'Open':
                    # Hand 2 is the primary hand, Hand 1 is the modifier
                    primary_hand_landmarks = results.multi_hand_landmarks[1]
                    modifier_active = True
                    cv2.putText(frame, "Modifier Active", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if modifier_active:
                    # In modifier mode, only detect circular motion
                    dynamic_gesture = self._detect_circular_motion(primary_hand_landmarks, w, h)
                    final_gesture = dynamic_gesture if dynamic_gesture else 'Fist' # Hover while waiting
                else:
                    # If two hands are present but neither is 'Open', default to single-hand logic on the first hand.
                    final_gesture = self._classify_gesture(landmarks1, results.multi_hand_landmarks[0], (h, w), frame)

            elif num_hands == 1:
                # Standard one-handed logic
                if self.previous_angle is not None: # Reset angle tracking when switching to one hand
                    self.previous_angle = None
                    self.angle_accumulator = 0
                landmarks = self._get_landmark_list(results.multi_hand_landmarks[0], (h, w))
                final_gesture = self._classify_gesture(landmarks, results.multi_hand_landmarks[0], (h, w), frame)
        else:
             # No hands detected, reset state
            self.previous_angle = None
            self.angle_accumulator = 0

        # --- Display and Smoothing Logic ---
        detected_gestures = []
        if final_gesture:
            self.gesture_history.append(final_gesture)
        
        if not hand_detected:
            self.gesture_history.clear()

        if len(self.gesture_history) > 0:
            # Use the most recent gesture for display and return
            most_recent_gesture = list(self.gesture_history)[-1]
            if most_recent_gesture:
                 detected_gestures = [most_recent_gesture]

        # Always show gesture status in top left
        # (This section is preserved from your original file for consistent UI)
        overlay = frame.copy()
        box_height = 80 + (len(detected_gestures) - 1) * 30 if detected_gestures else 80
        cv2.rectangle(overlay, (10, 10), (400, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (10, 10), (400, box_height), (255, 255, 255), 2)
        
        if detected_gestures and hand_detected:
            gesture_info = {
                'Open': {'name': 'Open', 'color': (0, 255, 255), 'action': 'Takeoff'},
                'Pointer': {'name': 'Pointer', 'color': (255, 255, 0), 'action': 'Ready for motion'},
                'OK': {'name': 'OK', 'color': (255, 0, 255), 'action': 'Land'},
                'Pointing Up': {'name': 'Pointing Up', 'color': (255, 255, 0), 'action': 'Ascend'},
                'Pointing Down': {'name': 'Pointing Down', 'color': (255, 255, 0), 'action': 'Descend'},
                'Pointing Left': {'name': 'Pointing Left', 'color': (255, 255, 0), 'action': 'Rotate Left'},
                'Pointing Right': {'name': 'Pointing Right', 'color': (255, 255, 0), 'action': 'Rotate Right'},
                'rotate_clockwise': {'name': 'Rotate Clockwise', 'color': (0, 255, 0), 'action': 'Rotation Detected!'},
                'rotate_counterclockwise': {'name': 'Rotate Counter-clockwise', 'color': (0, 255, 0), 'action': 'Rotation Detected!'},
                'thumbs_up': {'name': 'Thumbs Up', 'color': (0, 255, 255), 'action': 'Forward'},
                'thumbs_down': {'name': 'Thumbs Down', 'color': (0, 255, 255), 'action': 'Backward'},
                 'Fist': {'name': 'Fist', 'color': (128, 128, 128), 'action': 'Hover'},
            }
            
            for i, gesture in enumerate(detected_gestures):
                info = gesture_info.get(gesture, {'name': gesture, 'color': (255, 255, 255), 'action': ''})
                y_position = 40 + i * 30
                cv2.putText(frame, f"{info['name']} | {info['action']}", (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info['color'], 2)
        else:
            status_text = "No hand detected" if not hand_detected else "No gesture detected"
            cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        
        return frame, detected_gestures if detected_gestures else None

    def run(self):
        """
        Main loop for running the gesture detection from the webcam.
        """
        if not self.cap.isOpened():
            print("Error: Could not open webcam. Please ensure it is not in use by another application.")
            return

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