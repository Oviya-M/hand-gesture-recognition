"""
Step 2: Train a gesture classifier on extracted landmark vectors.

Reads the CSV produced by 1_extract_landmarks.py, trains a small MLP,
evaluates it, and saves the model + label encoder to disk.

Usage:
    python 2_train.py --csv landmarks.csv --out model/

Outputs:
    model/gesture_model.pkl   ← trained sklearn MLPClassifier
    model/label_encoder.pkl   ← fitted LabelEncoder (maps class names ↔ integers)
    model/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Reproducibility
RANDOM_STATE = 42


def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    print(f"Loading data from '{csv_path}' ...")
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("CSV is missing a 'label' column.")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y_str = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    print(f"  {len(X)} samples, {X.shape[1]} features, {len(le.classes_)} classes")
    print(f"  Classes: {list(le.classes_)}")

    counts = pd.Series(y_str).value_counts()
    print("\nSamples per class:")
    for cls, n in counts.items():
        print(f"  {cls:>12}: {n}")

    return X, y, le


def train(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    3-layer MLP.  Hidden layer sizes chosen to be proportional to the
    63-input feature space; dropout-equivalent regularization via alpha.
    """
    print("\nTraining MLP ...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,            # L2 regularization (helps with over-fitting)
        batch_size=128,
        learning_rate="adaptive",
        max_iter=500,
        random_state=RANDOM_STATE,
        verbose=False,
        early_stopping=True,   # hold out 10% of training data for validation
        n_iter_no_change=20,   # stop if no improvement for 20 epochs
    )
    model.fit(X_train, y_train)
    print(f"  Converged after {model.n_iter_} iterations.")
    return model


def evaluate(
    model: MLPClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    out_dir: Path,
) -> None:
    y_pred = model.predict(X_test)

    print("\n── Classification Report ──────────────────────────────────────")
    print(
        classification_report(
            y_test, y_pred, target_names=le.classes_, zero_division=0
        )
    )

    # Confusion matrix figure
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Gesture classifier — confusion matrix (test set)")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to '{cm_path}'.")


def save_artifacts(
    model: MLPClassifier, le: LabelEncoder, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "gesture_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to '{model_path}'.")

    le_path = out_dir / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to '{le_path}'.")


def main(csv_path: str, out_dir_str: str, test_size: float) -> None:
    out_dir = Path(out_dir_str)

    X, y, le = load_data(csv_path)

    # Stratified split so every class appears in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

    model = train(X_train, y_train)

    evaluate(model, X_test, y_test, le, out_dir)
    save_artifacts(model, le, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the gesture classifier.")
    parser.add_argument(
        "--csv",
        type=str,
        default="landmarks.csv",
        help="Landmark CSV produced by 1_extract_landmarks.py",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="model",
        help="Directory to save the trained model and artifacts.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Fraction of data to hold out for testing (default: 0.15).",
    )
    args = parser.parse_args()

    main(args.csv, args.out, args.test_size)