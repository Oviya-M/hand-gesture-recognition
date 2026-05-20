import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Define the standard 21-point hand skeleton connections manually
# This completely removes the reliance on mp.solutions.hands.HAND_CONNECTIONS
HAND_CONNECTIONS = [
    # Wrist to thumb and fingers base
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
    # Palm knuckles connection
    (5, 9), (9, 13), (13, 17)
]

# Initialize HandLandmarker configuration for LIVE_STREAM mode
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=lambda result, output_image, timestamp_ms: None
)

# Initialize global tracking for async frames
latest_result = None

def save_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options.result_callback = save_result
detector = vision.HandLandmarker.create_from_options(options)

# Start webcam capture
cap = cv2.VideoCapture(0)

print("Starting pipeline... Press 'q' on the window to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip horizontally for a natural mirror-view effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape  # Get frame dimensions for pixel mapping

    # Convert the frame to RGB as MediaPipe expects
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    frame_timestamp_ms = int(time.time() * 1000)
    detector.detect_async(mp_image, frame_timestamp_ms)

    # Render hand structural markers using native OpenCV primitives
    if latest_result and latest_result.hand_landmarks:
        for hand_landmarks in latest_result.hand_landmarks:
            
            # 1. Map normalized [0.0, 1.0] coordinates to absolute pixel coordinates
            pixel_landmarks = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                pixel_landmarks.append((cx, cy))
            
            # 2. Draw the skeleton bones (Connections)
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                # Verify keypoint indices exist safely in our tracked frame array
                if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                    cv2.line(frame, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)
            
            # 3. Draw the 21 key joint nodes (Landmarks)
            for idx, (cx, cy) in enumerate(pixel_landmarks):
                # Highlight fingertips (Points 4, 8, 12, 16, 20) with a distinct color
                color = (0, 0, 255) if idx in [4, 8, 12, 16, 20] else (255, 0, 0)
                cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)

    # Display the final composite image
    cv2.imshow('Hand Landmark Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
