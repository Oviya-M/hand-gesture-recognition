import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import os

# User settings
MODEL_FILENAME = "best_model.pt" # Same folder

# Model definition
class GestureNet(nn.Module):
    def __init__(self, input_dim=63, num_classes=8, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers += [
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Helper functions
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = GestureNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    class_names = checkpoint["class_names"]
    return model, class_names

def normalize_landmarks(features):
    norm = np.linalg.norm(features, ord=2, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return features / norm

def predict_gesture(model, hand_landmarks, device, class_names):
    # Extract wrist-relative coordinates (wrist becomes origin)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    coords_rel = coords - coords[0]
    features = coords_rel.flatten()
    features_norm = normalize_landmarks(features)
    input_tensor = torch.from_numpy(features_norm).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    return class_names[pred_idx], confidence

def draw_hand(frame, hand_landmarks, show_skeleton):
    h, w, _ = frame.shape
    # Convert normalized landmarks to pixel coordinates
    points = []
    for lm in hand_landmarks:
        points.append((int(lm.x * w), int(lm.y * h)))
    # Skeleton connections (21-point hand)
    connections = [
        (0,1),(0,5),(0,9),(0,13),(0,17),
        (1,2),(2,3),(3,4),
        (5,6),(6,7),(7,8),
        (9,10),(10,11),(11,12),
        (13,14),(14,15),(15,16),
        (17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    if show_skeleton:
        for a,b in connections:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], (0,255,0), 2)
        for idx, pt in enumerate(points):
            color = (0,0,255) if idx in [4,8,12,16,20] else (255,0,0)
            cv2.circle(frame, pt, 5, color, cv2.FILLED)
    return points

# Main pipeline
def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    model, class_names = load_model(model_path, device)
    print(f"Loaded gestures: {class_names}")

    # MediaPipe HandLandmarker
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=lambda result, output_image, timestamp_ms: None
    )
    latest_result = None
    def save_result(result, output_image, timestamp_ms):
        nonlocal latest_result
        latest_result = result
    options.result_callback = save_result
    detector = vision.HandLandmarker.create_from_options(options)

    # Webcam
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit, 's' to toggle skeleton drawing.")
    show_skeleton = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        # Run detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detector.detect_async(mp_img, int(time.time() * 1000))

        # Process results
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Draw skeleton/keypoints
                points = draw_hand(frame, hand_landmarks, show_skeleton)
                # Predict gesture
                gesture, conf = predict_gesture(model, hand_landmarks, device, class_names)
                # Display label above wrist
                wrist_x, wrist_y = points[0]
                cv2.putText(frame, f"{gesture} ({conf:.2f})",
                            (wrist_x - 20, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
