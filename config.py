"""
Configuration module for the Real-Time Emotion Detection application.

All tunable parameters are defined here as a dataclass.
CLI argument parsing allows runtime overrides of defaults.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────
# Emotion Labels (FER-2013 standard ordering)
# ──────────────────────────────────────────────
EMOTION_LABELS: List[str] = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# Emotion → BGR color for bounding-box rendering
EMOTION_COLORS = {
    "Angry":    (0, 0, 255),      # Red
    "Disgust":  (0, 128, 0),      # Dark Green
    "Fear":     (128, 0, 128),    # Purple
    "Happy":    (0, 255, 255),    # Yellow
    "Sad":      (255, 0, 0),      # Blue
    "Surprise": (0, 165, 255),    # Orange
    "Neutral":  (200, 200, 200),  # Light Gray
}

# ──────────────────────────────────────────────
# Base directories
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class AppConfig:
    """Central configuration for the application."""

    # ── Model ──────────────────────────────────
    model_path: str = os.path.join(PROJECT_ROOT, "model", "weights", "emotion_cnn.pth")
    confidence_threshold: float = 0.5
    use_gpu: bool = False

    # ── Camera / Inference ─────────────────────
    camera_index: int = 0
    frame_skip: int = 2          # Run prediction every N frames
    input_size: int = 48         # Model input resolution (48×48)

    # ── Smoothing ──────────────────────────────
    smoothing_window: int = 5    # Moving-average window size

    # ── Logging ────────────────────────────────
    log_dir: str = os.path.join(PROJECT_ROOT, "logs")
    db_name: str = "emotion_log.db"

    # ── Face Detection ─────────────────────────
    haar_cascade_path: str = os.path.join(
        PROJECT_ROOT, "model", "weights", "haarcascade_frontalface_default.xml"
    )
    face_scale_factor: float = 1.3
    face_min_neighbors: int = 5
    face_min_size: int = 30      # Minimum face size in pixels

    # ── Training ───────────────────────────────
    data_dir: str = os.path.join(PROJECT_ROOT, "data")
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 50
    early_stopping_patience: int = 7
    num_workers: int = 2

    # ── Screenshots ────────────────────────────
    screenshot_dir: str = os.path.join(PROJECT_ROOT, "screenshots")

    def __post_init__(self) -> None:
        """Create required directories if they don't exist."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)


def parse_args() -> AppConfig:
    """Parse command-line arguments and return an AppConfig instance."""
    parser = argparse.ArgumentParser(
        description="Real-Time Emotion Detection Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model weights (.pth)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence to display a prediction",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Run inference every N frames (higher = faster, less responsive)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera device index",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enable CUDA GPU acceleration if available",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Number of recent predictions for moving-average smoothing",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files and session reports",
    )

    args = parser.parse_args()

    # Build config, overriding only what was explicitly provided
    config = AppConfig()
    if args.model_path is not None:
        config.model_path = args.model_path
    config.confidence_threshold = args.confidence_threshold
    config.frame_skip = args.frame_skip
    config.camera_index = args.camera_index
    config.use_gpu = args.use_gpu
    config.smoothing_window = args.smoothing_window
    if args.log_dir is not None:
        config.log_dir = args.log_dir

    return config
