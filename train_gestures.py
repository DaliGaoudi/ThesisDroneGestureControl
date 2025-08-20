import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from gesture_model import GestureModel

class GestureTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # For motion tracking
        self.index_tip_history = deque(maxlen=30)
        self.motion_guide_points = []
        self.motion_guide_center = None
        self.motion_guide_radius = 50
        
        # Initialize ML model
        self.gesture_model = GestureModel()
        self.training_mode = False
        self.current_training_label = 'none'
        self.samples_collected = 0
        self.target_samples = 50  # Number of samples to collect for each gesture
        
    def _get_landmark_list(self, hand_landmarks, frame_shape):
        """Convert normalized landmarks to pixel coordinates."""
        landmark_list = []
        for lm in hand_landmarks.landmark:
            px, py = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])
            landmark_list.append(np.array([px, py]))
        return landmark_list
        
    def _get_finger_states(self, landmarks):
        """Determine if each finger is extended or curled."""
        finger_states = {}
        
        # For the thumb
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        if np.linalg.norm(thumb_tip - thumb_mcp) > np.linalg.norm(thumb_ip - thumb_mcp):
             finger_states['thumb'] = True
        else:
             finger_states['thumb'] = False

        # For other fingers
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
            if np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist):
                finger_states[finger] = True
            else:
                finger_states[finger] = False
                
        return finger_states
        
    def _is_pointing_at_camera(self, landmarks, hand_landmarks):
        """Check if the index finger is pointing at the camera."""
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # Calculate vectors for angle
        tip_to_pip = index_tip - index_pip
        pip_to_mcp = index_pip - index_mcp
        
        # Check if finger is relatively straight
        angle = np.arccos(np.dot(tip_to_pip, pip_to_mcp) / 
                         (np.linalg.norm(tip_to_pip) * np.linalg.norm(pip_to_mcp)))
        
        # Get z-coordinates from original MediaPipe landmarks
        index_tip_z = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].z
        index_mcp_z = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].z
        
        # Check if tip is closer to camera than base
        is_closer = index_tip_z < index_mcp_z
        
        return is_closer and angle < np.pi/4
        
    def _update_motion_guide(self, index_tip):
        """Update the motion guide points."""
        if self.motion_guide_center is None:
            self.motion_guide_center = (int(index_tip[0]), int(index_tip[1]))
            # Generate points for a partial circle (about 270 degrees)
            center_x, center_y = self.motion_guide_center
            for angle in range(0, 270, 5):  # 5-degree steps
                rad = np.radians(angle)
                x = center_x + int(self.motion_guide_radius * np.cos(rad))
                y = center_y + int(self.motion_guide_radius * np.sin(rad))
                self.motion_guide_points.append((x, y))
                
    def start_training(self, label):
        """Start training mode for the specified gesture."""
        self.training_mode = True
        self.current_training_label = label
        self.samples_collected = 0
        self.gesture_model.start_training()
        
    def stop_training(self):
        """Stop training mode and save the model."""
        self.training_mode = False
        self.gesture_model.stop_training()
        self.gesture_model.save_model()
        print(f"Model saved with {self.samples_collected} samples")
        
    def run(self):
        """Main training loop."""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break
                
                # Flip for a selfie-view display
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Get landmarks
                        landmarks = self._get_landmark_list(hand_landmarks, (h, w))
                        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        
                        # Check pointing pose
                        finger_states = self._get_finger_states(landmarks)
                        is_pointing_pose = finger_states['index'] and not any([finger_states['middle'], finger_states['ring'], finger_states['pinky']])
                        
                        if is_pointing_pose:
                            # Update and draw motion guide
                            self._update_motion_guide(index_tip)
                            
                            # Draw motion guide
                            for i in range(len(self.motion_guide_points) - 1):
                                pt1 = self.motion_guide_points[i]
                                pt2 = self.motion_guide_points[i + 1]
                                alpha = i / len(self.motion_guide_points)
                                color = (0, int(255 * (1 - alpha)), int(255 * alpha))
                                cv2.line(frame, pt1, pt2, color, 2)
                            
                            # Draw current index tip position
                            cv2.circle(frame, (int(index_tip[0]), int(index_tip[1])), 8, (0, 255, 0), -1)
                            
                            # Draw the history of index finger positions
                            for i in range(len(self.index_tip_history) - 1):
                                pt1 = (int(self.index_tip_history[i][0]), int(self.index_tip_history[i][1]))
                                pt2 = (int(self.index_tip_history[i+1][0]), int(self.index_tip_history[i+1][1]))
                                alpha = i / len(self.index_tip_history)
                                color = (0, int(255 * (1 - alpha)), int(255 * alpha))
                                cv2.line(frame, pt1, pt2, color, 2)
                            
                            # Check if pointing at camera
                            is_pointing_at_camera = self._is_pointing_at_camera(landmarks, hand_landmarks)
                            status_color = (0, 255, 0) if is_pointing_at_camera else (0, 0, 255)
                            
                            # Add status text
                            cv2.putText(
                                frame,
                                "Pointing at camera" if is_pointing_at_camera else "Not pointing at camera",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                status_color,
                                2
                            )
                            
                            # Add training status
                            if self.training_mode:
                                # Add sample if pointing at camera
                                if is_pointing_at_camera:
                                    self.index_tip_history.append(index_tip)
                                    if len(self.index_tip_history) >= 10:
                                        points = list(self.index_tip_history)
                                        self.gesture_model.add_training_sample(points, self.current_training_label)
                                        self.samples_collected += 1
                                        self.index_tip_history.clear()
                                
                                # Draw training status
                                cv2.putText(
                                    frame,
                                    f"Training: {self.current_training_label} ({self.samples_collected}/{self.target_samples})",
                                    (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 255),
                                    2
                                )
                                
                                # Auto-stop when enough samples collected
                                if self.samples_collected >= self.target_samples:
                                    self.stop_training()
                        else:
                            # Reset when not in pointing pose
                            self.index_tip_history.clear()
                            self.motion_guide_center = None
                            self.motion_guide_points = []
                
                # Add instructions
                cv2.putText(
                    frame,
                    "Press 'c' to train clockwise, 'w' for counter-clockwise, 's' to save, 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                cv2.imshow('Gesture Training', frame)
                
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):  # Start training clockwise
                    self.start_training('clockwise')
                elif key == ord('w'):  # Start training counter-clockwise
                    self.start_training('counterclockwise')
                elif key == ord('s'):  # Stop training
                    self.stop_training()
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    trainer = GestureTrainer()
    trainer.run() 