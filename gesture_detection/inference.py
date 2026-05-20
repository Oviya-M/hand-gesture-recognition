"""
Step 3: Live gesture recognition from a webcam.

Runs MediaPipe Hands on every frame, normalizes the landmarks,
feeds them into the trained MLP, and overlays the predicted gesture
and confidence on the video window.

Press Q or Esc to quit.

Usage:
    python 3_inference.py --model model/gesture_model.pkl \
                          --encoder model/label_encoder.pkl \
                          --camera 0
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ── Landmark constants (must match extraction script) ────────────────────────
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Identical normalization to the extraction step."""
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist
    scale = np.max(np.abs(landmarks))
    if scale > 0:
        landmarks = landmarks / scale
    return landmarks.flatten()


def load_model(model_path: str, encoder_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)
    return model, le


# ── Drawing helpers ──────────────────────────────────────────────────────────

# Color per gesture (BGR)
GESTURE_COLORS = {
    "fist":    (50,  50, 220),
    "like":    (50, 200,  50),
    "dislike": (220,  50,  50),
    "stop":    (220, 160,  50),
    "point":   (200,  50, 200),
    "heart":   (100,  50, 220),
    "ok":      (50, 200, 200),
}
DEFAULT_COLOR = (200, 200, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_prediction(
    frame: np.ndarray,
    gesture: str,
    confidence: float,
    fps: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    color = GESTURE_COLORS.get(gesture, DEFAULT_COLOR)

    # Semi-transparent banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Gesture name
    cv2.putText(
        frame,
        gesture.upper(),
        (20, 48),
        FONT,
        1.4,
        color,
        2,
        cv2.LINE_AA,
    )

    # Confidence bar
    bar_x, bar_y, bar_h = 20, 58, 6
    bar_w = int((w - 40) * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w - 40, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color, -1)

    # Confidence percentage
    conf_text = f"{confidence * 100:.0f}%"
    cv2.putText(frame, conf_text, (w - 80, 48), FONT, 0.7, color, 1, cv2.LINE_AA)

    # FPS (bottom-left, small)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, h - 12),
        FONT,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_landmarks_on_frame(frame, hand_landmarks):
    """Draw the MediaPipe hand skeleton on the frame."""
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style(),
    )


# ── Main loop ────────────────────────────────────────────────────────────────

def run(model_path: str, encoder_path: str, camera_id: int, confidence_threshold: float) -> None:
    print("Loading model ...")
    model, le = load_model(model_path, encoder_path)
    print(f"  Classes: {list(le.classes_)}")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {camera_id}.")

    # Bump resolution if camera supports it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,      # video mode — tracks hand between frames
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    print("\nCamera open. Press Q or Esc to quit.\n")

    prev_time = time.perf_counter()
    current_gesture = "—"
    current_conf = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame.")
            break

        # Mirror the frame so it feels like a mirror, not a camera
        frame = cv2.flip(frame, 1)

        # FPS
        now = time.perf_counter()
        fps = 1.0 / max(1e-6, now - prev_time)
        prev_time = now

        # Run MediaPipe (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]

            # Draw skeleton
            draw_landmarks_on_frame(frame, hand_lms)

            # Extract + normalize features
            raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            features = normalize_landmarks(raw).reshape(1, -1)  # (1, 63)

            # Predict
            probs = model.predict_proba(features)[0]
            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])
            pred_label = le.inverse_transform([pred_idx])[0]

            if pred_conf >= confidence_threshold:
                current_gesture = pred_label
                current_conf = pred_conf
            else:
                current_gesture = "uncertain"
                current_conf = pred_conf
        else:
            current_gesture = "no hand"
            current_conf = 0.0

        frame = draw_prediction(frame, current_gesture, current_conf, fps)

        cv2.imshow("Gesture Recognition  (Q / Esc to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # 27 = Esc
            break

    hands_detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live gesture recognition.")
    parser.add_argument(
        "--model",
        type=str,
        default="model/gesture_model.pkl",
        help="Path to trained MLPClassifier pickle.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="model/label_encoder.pkl",
        help="Path to LabelEncoder pickle.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Minimum confidence to display a prediction (default: 0.6).",
    )
    args = parser.parse_args()

    run(args.model, args.encoder, args.camera, args.threshold)