"""
Training pipeline for the EmotionCNN model on FER-2013.

Features:
    - FER-2013 CSV data loading and augmentation
    - Train / validation split
    - Learning-rate scheduling (ReduceLROnPlateau)
    - Early stopping with best-model checkpointing
    - Training curves & confusion matrix generation
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMOTION_LABELS, AppConfig
from model.model import EmotionCNN


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class FER2013Dataset(Dataset):
    """
    PyTorch Dataset for the FER-2013 facial expression dataset.

    Expects a CSV file with columns:
        - emotion: int label (0-6)
        - pixels: space-separated pixel values (48×48 = 2304 values)
        - Usage: 'Training' | 'PublicTest' | 'PrivateTest'

    Parameters
    ----------
    csv_path : str
        Path to the FER-2013 CSV file.
    split : str
        One of 'Training', 'PublicTest', 'PrivateTest'.
    transform : callable or None
        Optional torchvision transforms to apply.
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "Training",
        transform: transforms.Compose | None = None,
    ) -> None:
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == split].reset_index(drop=True)

        self.labels = df["emotion"].values.astype(np.int64)
        self.pixels = df["pixels"].values
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Parse pixel string → 48×48 float32 image
        pixel_str = self.pixels[idx]
        image = np.fromstring(pixel_str, sep=" ", dtype=np.float32).reshape(48, 48)
        image = image / 255.0  # Normalize to [0, 1]
        image = image[..., np.newaxis]  # H×W×1 for transforms

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor (C×H×W)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return image, label


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
def get_train_transforms() -> transforms.Compose:
    """Data augmentation for training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop(48, padding=4),
        transforms.ToTensor(),
    ])


def get_val_transforms() -> transforms.Compose:
    """Minimal transforms for validation (just tensor conversion)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ──────────────────────────────────────────────
# Plotting utilities
# ──────────────────────────────────────────────
def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_dir: str,
) -> None:
    """Save training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-", label="Train Acc")
    ax2.plot(epochs, val_accs, "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to {save_dir}/training_curves.png")


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    save_dir: str,
) -> None:
    """Save a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(labels, preds, labels=range(len(EMOTION_LABELS)))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_dir}/confusion_matrix.png")


# ──────────────────────────────────────────────
# Main training entry point
# ──────────────────────────────────────────────
def main() -> None:
    """Run the full training pipeline."""
    config = AppConfig()

    csv_path = os.path.join(config.data_dir, "fer2013.csv")
    if not os.path.isfile(csv_path):
        print(f"[ERROR] FER-2013 dataset not found at: {csv_path}")
        print("        Download from https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"        Place fer2013.csv in: {config.data_dir}/")
        sys.exit(1)

    # Device
    device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Datasets & loaders
    print("[INFO] Loading datasets...")
    train_dataset = FER2013Dataset(csv_path, split="Training", transform=get_train_transforms())
    val_dataset = FER2013Dataset(csv_path, split="PublicTest", transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print(f"[INFO] Training samples : {len(train_dataset)}")
    print(f"[INFO] Validation samples: {len(val_dataset)}")

    # Model, loss, optimizer, scheduler
    model = EmotionCNN(num_classes=len(EMOTION_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters     : {total_params:,}")
    print(f"[INFO] Trainable parameters : {trainable_params:,}")

    # Training state
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    weights_dir = os.path.dirname(config.model_path)
    os.makedirs(weights_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Training EmotionCNN — {config.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)

        # Logging
        elapsed = time.time() - epoch_start
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch:3d}/{config.epochs}] "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  |  "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), config.model_path)
            print(f"  ✓ New best model saved (val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\n[INFO] Early stopping triggered after {epoch} epochs")
            break

    # ── Post-training ──────────────────────────
    print(f"\n{'='*60}")
    print(f" Training complete — Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")
    print(f" Model saved to: {config.model_path}")

    # Plot curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, weights_dir)

    # Final evaluation on test set
    print("\n[INFO] Evaluating on PrivateTest set...")
    test_dataset = FER2013Dataset(csv_path, split="PrivateTest", transform=get_val_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Load best model for final eval
    model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"[INFO] Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.2f}%")

    # Confusion matrix
    try:
        plot_confusion_matrix(test_preds, test_labels, weights_dir)
    except ImportError:
        print("[WARN] scikit-learn not installed — skipping confusion matrix")


if __name__ == "__main__":
    main()
