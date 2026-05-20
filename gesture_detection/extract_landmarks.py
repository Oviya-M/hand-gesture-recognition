"""
Step 1: Extract hand landmarks from HaGRID images.

For each gesture class, runs MediaPipe Hands on every image,
extracts the 21 3D landmarks, normalizes them, and saves to a CSV.

Expected HaGRID directory layout:
    hagrid/
        subsample/          ← or the full dataset root
            fist/
                *.jpg
            like/
            dislike/
            stop/
            point/
            heart/
            ok/

Usage:
    python 1_extract_landmarks.py --data_dir ./hagrid/subsample --out landmarks.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# The 7 gestures we care about (must match HaGRID subfolder names exactly)
GESTURES = ["fist", "like", "dislike", "stop", "point", "heart", "ok"]

# MediaPipe returns 21 landmarks × 3 coords = 63 values
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURE_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Make landmark vectors invariant to hand position and scale.

    1. Translate so wrist (landmark 0) is at the origin.
    2. Scale so the largest absolute coordinate is 1.0.
    3. Flatten to a 1-D vector of length 63.
    """
    # landmarks shape: (21, 3)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist                      # translate to origin

    scale = np.max(np.abs(landmarks))
    if scale > 0:
        landmarks = landmarks / scale                  # normalize scale

    return landmarks.flatten()                         # → (63,)


def extract_landmarks_from_image(image_path: str, hands_detector) -> np.ndarray | None:
    """
    Run MediaPipe on one image and return the normalized landmark vector,
    or None if no hand is detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # MediaPipe expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    # Use the first detected hand only
    hand_landmarks = results.multi_hand_landmarks[0]

    # Convert to numpy array of shape (21, 3)
    raw = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    return normalize_landmarks(raw)


def build_header() -> list[str]:
    """CSV column names: lm0_x, lm0_y, lm0_z, lm1_x, ..., lm20_z, label"""
    cols = []
    for i in range(NUM_LANDMARKS):
        for axis in ("x", "y", "z"):
            cols.append(f"lm{i}_{axis}")
    cols.append("label")
    return cols


def main(data_dir: str, out_path: str, max_per_class: int | None) -> None:
    data_root = Path(data_dir)

    # Validate that at least one gesture folder exists
    found = [g for g in GESTURES if (data_root / g).is_dir()]
    if not found:
        sys.exit(
            f"No gesture folders found under '{data_root}'.\n"
            f"Expected subdirectories: {GESTURES}"
        )
    missing = [g for g in GESTURES if g not in found]
    if missing:
        print(f"[warn] Missing gesture folders (will be skipped): {missing}")

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,       # we're processing individual images, not video
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    total_saved = 0
    total_skipped = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())

        for gesture in found:
            gesture_dir = data_root / gesture
            image_paths = sorted(
                [
                    p
                    for p in gesture_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                ]
            )

            if max_per_class is not None:
                image_paths = image_paths[:max_per_class]

            saved = 0
            skipped = 0

            for img_path in tqdm(image_paths, desc=f"{gesture:>10}", unit="img"):
                landmarks = extract_landmarks_from_image(str(img_path), hands_detector)
                if landmarks is None:
                    skipped += 1
                    continue

                writer.writerow([*landmarks, gesture])
                saved += 1

            print(
                f"  {gesture}: {saved} saved, {skipped} skipped "
                f"({skipped / max(1, saved + skipped):.1%} detection failure rate)"
            )
            total_saved += saved
            total_skipped += skipped

    hands_detector.close()

    print(f"\nDone. {total_saved} samples written to '{out_path}'.")
    print(f"Total skipped (no hand detected): {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from HaGRID images.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./hagrid/subsample",
        help="Root folder containing one subfolder per gesture class.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="landmarks.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Cap images per class (useful for quick tests). Default: use all.",
    )
    args = parser.parse_args()

    main(args.data_dir, args.out, args.max_per_class)