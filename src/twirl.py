import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# For tracking gesture motion
previous_angle = None
angle_accumulator = 0
volume = 50  # Dummy volume level (0 to 100)

def calculate_angle(x, y, cx, cy):
    return math.degrees(math.atan2(y - cy, x - cx))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape
    cx, cy = w // 2, h // 2  # Use screen center as rotation reference

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip position (landmark 8)
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Calculate current angle of index finger from center
            current_angle = calculate_angle(x, y, cx, cy)

            if previous_angle is not None:
                delta = current_angle - previous_angle

                # Normalize for circular overflow
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360

                angle_accumulator += delta

                # Threshold for detecting a full twirl (adjustable)
                if angle_accumulator > 180:
                    volume = min(volume + 5, 100)
                    print("🔊 Volume Up:", volume)
                    angle_accumulator = 0
                elif angle_accumulator < -180:
                    volume = max(volume - 5, 0)
                    print("🔉 Volume Down:", volume)
                    angle_accumulator = 0

            previous_angle = current_angle

            # Draw center and finger point
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(img, (x, y), 8, (255, 0, 0), -1)

    else:
        previous_angle = None  # Reset if no hand is detected

    cv2.imshow("Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
