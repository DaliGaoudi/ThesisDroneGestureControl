import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class HandGestureDetector:
    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        # --- NEW: For circular gesture detection ---
        self.index_tip_history = deque(maxlen=20) # Store last 20 positions of the index finger tip
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.5  # 1.5 seconds cooldown
        self.circle_threshold = 2000 # Sensitivity for circle detection, may need tuning
        # --- END NEW ---
        
        self.gesture_history = deque(maxlen=10)

    def _get_landmark_list(self, hand_landmarks, frame_shape):
        landmark_list = []
        for lm in hand_landmarks.landmark:
            px, py = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])
            landmark_list.append(np.array([px, py]))
        return landmark_list

    def _get_finger_states(self, landmarks):
        finger_states = {}
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]

        # Thumb check
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        if np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_ip - wrist):
             finger_states['thumb'] = True
        else:
             finger_states['thumb'] = False

        # Other fingers check
        finger_definitions = {
            'index': [self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
            'middle': [self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            'ring': [self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP],
            'pinky': [self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP]
        }
        for finger, (tip_idx, pip_idx) in finger_definitions.items():
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            if np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist):
                finger_states[finger] = True
            else:
                finger_states[finger] = False
                
        return finger_states

    # --- NEW: Method to detect circular motion ---
    def _detect_circular_motion(self, index_tip_pos):
        """
        Analyzes the history of the index finger tip to detect a circular motion.

        Args:
            index_tip_pos: The current position of the index finger tip.

        Returns:
            str or None: 'volume_up' for clockwise, 'volume_down' for counter-clockwise, or None.
        """
        self.index_tip_history.append(index_tip_pos)
        
        # We need enough points to analyze the path
        if len(self.index_tip_history) < self.index_tip_history.maxlen:
            return None

        # Calculate the cumulative cross product to determine rotation direction
        # The cross product of two sequential vectors (p2-p1) and (p3-p2)
        # gives a z-component that is consistently positive for one direction
        # of rotation and negative for the other.
        path = np.array(self.index_tip_history)
        total_angle_change = 0
        
        for i in range(len(path) - 2):
            vec1 = path[i+1] - path[i]
            vec2 = path[i+2] - path[i+1]
            # 2D cross-product
            cross_product = np.cross(vec1, vec2)
            total_angle_change += cross_product

        # Check if a gesture was detected recently to apply cooldown
        current_time = time.time()
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return None
        
        # If the total change is significant, we have a circle
        if total_angle_change > self.circle_threshold: # Counter-clockwise
            self.last_gesture_time = current_time
            self.index_tip_history.clear() # Reset history after detection
            return 'volume_down'
        elif total_angle_change < -self.circle_threshold: # Clockwise
            self.last_gesture_time = current_time
            self.index_tip_history.clear() # Reset history after detection
            return 'volume_up'
            
        return None
    # --- END NEW ---

    def _get_pointing_direction(self, landmarks):
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        direction_vector = index_tip - index_mcp
        
        if np.linalg.norm(direction_vector) == 0: return None
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        dx, dy = direction_vector[0], -direction_vector[1]

        if abs(dx) > abs(dy):
            if dx > 0.7: return 'pointing_right'
            elif dx < -0.7: return 'pointing_left'
        else:
            if dy > 0.7: return 'pointing_up'
            elif dy < -0.7: return 'pointing_down'
            
        return None

    def _classify_gesture(self, landmarks):
        finger_states = self._get_finger_states(landmarks)
        num_fingers_extended = sum(finger_states.values())
        
        # Check for pointing pose first, as it's a prerequisite for volume control
        is_pointing_pose = finger_states['index'] and not any([finger_states['middle'], finger_states['ring'], finger_states['pinky']])
        
        if is_pointing_pose:
            index_tip_pos = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # --- MODIFIED: Check for circular motion first ---
            circular_gesture = self._detect_circular_motion(index_tip_pos)
            if circular_gesture:
                return circular_gesture
            # --- END MODIFIED ---
            
            # If not a circular gesture, check for static pointing
            return self._get_pointing_direction(landmarks)
        else:
            # If not in a pointing pose, clear the history
            self.index_tip_history.clear()

        # OK Sign
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        dist = np.linalg.norm(thumb_tip - index_tip)
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        ok_dist_threshold = np.linalg.norm(index_tip - index_mcp) * 0.6
        if dist < ok_dist_threshold and finger_states['middle'] and finger_states['ring'] and finger_states['pinky']:
            return 'ok_sign'

        # Thumbs Up / Down
        if finger_states['thumb'] and not any([finger_states['index'], finger_states['middle'], finger_states['ring'], finger_states['pinky']]):
            thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            if thumb_tip[1] < index_pip[1]:
                return 'thumbs_up'
            else:
                return 'thumbs_down'

        # Open Palm
        if all(finger_states.values()):
            return 'open_palm'
            
        # Fist
        if num_fingers_extended == 0:
            return 'fist'
        
        return None

    def detect_gesture(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_gesture = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = self._get_landmark_list(hand_landmarks, (h, w))
                gesture = self._classify_gesture(landmarks)
                self.gesture_history.append(gesture)
                break 

        if len(self.gesture_history) > 0:
            valid_gestures = [g for g in self.gesture_history if g is not None]
            if valid_gestures:
                detected_gesture = max(set(valid_gestures), key=valid_gestures.count)
        
        return frame, detected_gesture

    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                frame, gesture = self.detect_gesture(frame)
                
                if gesture:
                    # Make the text bigger and more visible
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture.upper()}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255), # White color
                        3,
                        cv2.LINE_AA
                    )
                
                cv2.imshow('Hand Gesture Detection', frame)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    detector = HandGestureDetector()
    detector.run()