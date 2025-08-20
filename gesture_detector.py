import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import csv

class HandGestureDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,  # Increased for better accuracy
            min_tracking_confidence=0.5    # Increased for better accuracy
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize keypoint classifier
        self.keypoint_classifier = self._init_keypoint_classifier()
        
        # Load gesture labels
        self.gesture_labels = self._load_gesture_labels()
        
        # Initialize gesture history for smoothing
        self.gesture_history = []
        self.history_length = 5
        
        # Define our gesture mapping
        self.gesture_mapping = {
            'Open': 'open_palm',
            'Close': 'fist',
            'Pointer': self._get_pointer_direction,
            'OK': 'ok_sign',
            'Thumb Up': 'thumbs_up',
            'Thumb Down': 'thumbs_down'
        }
    
    def _init_keypoint_classifier(self):
        """Initialize the TensorFlow Lite model for gesture classification"""
        model_path = 'hand-gesture-recognition-mediapipe/model/keypoint_classifier/keypoint_classifier.tflite'
        interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=1)
        interpreter.allocate_tensors()
        return interpreter
    
    def _load_gesture_labels(self):
        """Load gesture labels from CSV file"""
        labels = []
        with open('hand-gesture-recognition-mediapipe/model/keypoint_classifier/keypoint_classifier_label.csv',
                 encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            labels = [row[0] for row in csv_reader]
        return labels
    
    def _preprocess_landmarks(self, landmarks):
        """Preprocess landmarks for the classifier"""
        landmark_list = []
        
        # Convert landmarks to relative coordinates
        base_x, base_y = landmarks.landmark[0].x, landmarks.landmark[0].y
        
        for landmark in landmarks.landmark:
            landmark_list.append(landmark.x - base_x)
            landmark_list.append(landmark.y - base_y)
        
        # Normalize
        max_value = max(list(map(abs, landmark_list)))
        landmark_list = list(map(lambda n: n / max_value, landmark_list))
        
        return landmark_list
    
    def _is_finger_up(self, landmarks, finger_tip):
        """Check if a finger is pointing up"""
        tip = landmarks.landmark[finger_tip]
        pip = landmarks.landmark[finger_tip - 2]  # PIP joint
        mcp = landmarks.landmark[finger_tip - 3]  # MCP joint
        
        # More lenient check: finger is considered up if it's extended relative to both PIP and MCP
        return tip.y < pip.y and tip.y < mcp.y
    
    def _is_thumb_up(self, landmarks):
        """Check if thumb is pointing up"""
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        
        return thumb_tip.y < thumb_ip.y and thumb_tip.y < thumb_mcp.y
    
    def _is_thumb_down(self, landmarks):
        """Check if thumb is pointing down"""
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        
        return thumb_tip.y > thumb_ip.y and thumb_tip.y > thumb_mcp.y
    
    def _is_ok_sign(self, landmarks):
        """Check if hand is making OK sign"""
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate distance between thumb and index finger tips
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        # Check if other fingers are down
        middle_up = self._is_finger_up(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
        ring_up = self._is_finger_up(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP)
        pinky_up = self._is_finger_up(landmarks, self.mp_hands.HandLandmark.PINKY_TIP)
        
        return distance < 0.15 and not (middle_up or ring_up or pinky_up)
    
    def _is_fist(self, landmarks):
        """Check if the hand is in a fist position"""
        # Get y-coordinates of finger tips and middle joints
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Get corresponding middle joints
        index_middle = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_middle = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_middle = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_middle = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        
        # Check if all fingers are bent
        fingers_bent = (
            index_tip.y > index_middle.y * 0.9 and
            middle_tip.y > middle_middle.y * 0.9 and
            ring_tip.y > ring_middle.y * 0.9 and
            pinky_tip.y > pinky_middle.y * 0.9
        )
        
        return fingers_bent
    
    def _get_pointer_direction(self, landmarks):
        """Determine the direction of pointing gesture"""
        # Get index finger tip and wrist positions
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Calculate horizontal and vertical components
        dx = index_tip.x - wrist.x
        dy = index_tip.y - wrist.y
        
        # Calculate angle
        angle = np.degrees(np.arctan2(dy, dx))
        angle = (angle + 360) % 360
        
        # Determine direction based on angle
        if abs(dy) < 0.1:  # If roughly horizontal
            if dx > 0.1:
                return 'pointing_right'
            elif dx < -0.1:
                return 'pointing_left'
        else:  # Vertical pointing
            if angle > 30 and angle < 150:
                return 'pointing_up'
            elif angle > 210 and angle < 330:
                return 'pointing_down'
        
        return None
    
    def _classify_gesture(self, landmarks):
        """Classify gesture using both ML and rule-based approaches"""
        # First, try rule-based detection for more precise control
        if self._is_ok_sign(landmarks):
            return 'ok_sign'
        
        # Check for pointing gestures
        index_up = self._is_finger_up(landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP)
        middle_up = self._is_finger_up(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
        ring_up = self._is_finger_up(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP)
        pinky_up = self._is_finger_up(landmarks, self.mp_hands.HandLandmark.PINKY_TIP)
        
        if index_up and not (middle_up or ring_up or pinky_up):
            direction = self._get_pointer_direction(landmarks)
            if direction:
                return direction
        
        # Check for thumbs up/down
        if self._is_fist(landmarks):
            if self._is_thumb_up(landmarks):
                return 'thumbs_up'
            elif self._is_thumb_down(landmarks):
                return 'thumbs_down'
        
        # Check for open palm
        if (index_up and middle_up and ring_up and pinky_up) or \
           (index_up and middle_up and (ring_up or pinky_up)):
            return 'open_palm'
        
        # Check for fist
        if self._is_fist(landmarks):
            return 'fist'
        
        # If rule-based detection fails, try ML-based detection
        landmark_list = self._preprocess_landmarks(landmarks)
        
        # Get input and output details
        input_details = self.keypoint_classifier.get_input_details()
        output_details = self.keypoint_classifier.get_output_details()
        
        # Set input tensor
        input_details_tensor_index = input_details[0]['index']
        self.keypoint_classifier.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        
        # Run inference
        self.keypoint_classifier.invoke()
        
        # Get output tensor
        output_details_tensor_index = output_details[0]['index']
        result = self.keypoint_classifier.get_tensor(output_details_tensor_index)
        
        # Get the predicted gesture
        result_index = np.argmax(np.squeeze(result))
        predicted_gesture = self.gesture_labels[result_index]
        
        # Map the predicted gesture to our gesture set
        if predicted_gesture in self.gesture_mapping:
            if callable(self.gesture_mapping[predicted_gesture]):
                gesture = self.gesture_mapping[predicted_gesture](landmarks)
            else:
                gesture = self.gesture_mapping[predicted_gesture]
            
            # Add to history for smoothing
            self.gesture_history.append(gesture)
            if len(self.gesture_history) > self.history_length:
                self.gesture_history.pop(0)
            
            # Get most common gesture from history
            if len(self.gesture_history) == self.history_length:
                most_common = max(set(self.gesture_history), key=self.gesture_history.count)
                return most_common
        
        return None
    
    def detect_gesture(self, frame):
        """
        Detect hand gestures in the given frame
        Returns the frame with landmarks drawn and the detected gesture
        """
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.hands.process(frame_rgb)
        
        # Initialize gesture as None
        gesture = None
        
        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Classify the gesture
                gesture = self._classify_gesture(hand_landmarks)
                if gesture:
                    break
        
        return frame, gesture
    
    def run(self):
        """
        Main loop for gesture detection
        """
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Detect gesture
                frame, gesture = self.detect_gesture(frame)
                
                # Display gesture on frame
                if gesture:
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                
                # Display the frame
                cv2.imshow('Hand Gesture Detection', frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    # Create and run the gesture detector
    detector = HandGestureDetector()
    detector.run() 