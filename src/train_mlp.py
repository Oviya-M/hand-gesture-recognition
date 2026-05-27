"""
Hand gesture recognition — full PyTorch pipeline in one file.

Dataset  : hand_landmarks.csv  (63 MediaPipe landmark features + label)
Classes  : dislike, fist, hand_heart, like, ok, peace, point, stop
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── hyperparameters ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "hand_landmarks.csv")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

INPUT_DIM    = 63
NUM_CLASSES  = 8
HIDDEN_DIMS  = [256, 128, 64]
DROPOUT      = 0.3

BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS   = 100
PATIENCE     = 15       # early stopping patience


# ── 1. data loading & preprocessing ──────────────────────────────────────────

def normalize_landmarks(X):
    """Per-sample: divide by L2 norm → unit-sphere, scale-invariant."""
    scale = np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-8)
    return (X / scale).astype(np.float32)


def load_data():
    df = pd.read_csv(DATA_PATH)
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} NaN rows → {len(df)} samples remaining")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    class_names = le.classes_

    X = normalize_landmarks(X)

    print(f"Classes : {list(class_names)}")
    print(f"Label distribution: { {c: int((y == i).sum()) for i, c in enumerate(class_names)} }")
    return X, y, class_names


# ── 2. PyTorch Dataset ────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── 3. model ──────────────────────────────────────────────────────────────────

class GestureNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = INPUT_DIM
        for hidden in HIDDEN_DIMS:
            layers += [
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── 4. train / eval helpers ───────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / total, correct / total


# ── 5. plotting helpers ───────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses,   label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(train_accs, label="Train")
    ax2.plot(val_accs,   label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves → {path}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix  → {path}")


# ── 6. main ───────────────────────────────────────────────────────────────────

def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}\n")

    # — data —
    X, y, class_names = load_data()

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    print(f"\nSplit → train: {len(y_train)}  val: {len(y_val)}  test: {len(y_test)}\n")

    train_loader = DataLoader(GestureDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(GestureDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(GestureDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE)

    # — model, loss, optimiser —
    model     = GestureNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # — training loop —
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pt")

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step(vl_loss)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"train loss {tr_loss:.4f}  acc {tr_acc:.4f} | "
              f"val loss {vl_loss:.4f}  acc {vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "class_names": list(class_names)}, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    # — test evaluation —
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nTest accuracy: {test_acc:.4f}\n")

    # gather all test predictions
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # — plots —
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(all_labels, all_preds, class_names)

    print(f"\nModel saved → {best_model_path}")


if __name__ == "__main__":
    main()
