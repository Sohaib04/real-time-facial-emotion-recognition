"""
Inference engine for the EmotionCNN model.

Handles:
    - Model loading and device placement
    - Face-crop preprocessing (grayscale → 48×48 → normalize → tensor)
    - Prediction with disabled gradients for speed
    - Confidence thresholding
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMOTION_LABELS, AppConfig
from model.model import EmotionCNN


class EmotionPredictor:
    """
    Wraps a trained EmotionCNN for real-time inference.

    Parameters
    ----------
    config : AppConfig
        Application configuration containing model path, device settings, etc.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.labels = EMOTION_LABELS
        self.input_size = config.input_size  # 48

        # Device selection
        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load model
        self.model = EmotionCNN(num_classes=len(self.labels))
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

        print(f"[Inference] Model loaded on {self.device}")

    def _load_weights(self) -> None:
        """Load model weights from the configured path."""
        if not os.path.isfile(self.config.model_path):
            raise FileNotFoundError(
                f"Model weights not found at: {self.config.model_path}\n"
                f"Train the model first with: python model/train.py"
            )
        state_dict = torch.load(
            self.config.model_path,
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)

    def preprocess(self, face_roi: np.ndarray) -> torch.Tensor:
        """
        Preprocess a face ROI for model input.

        Steps:
            1. Convert to grayscale (if 3-channel)
            2. Resize to 48×48
            3. Normalize pixel values to [0, 1]
            4. Reshape to (1, 1, 48, 48) tensor

        Parameters
        ----------
        face_roi : np.ndarray
            Cropped face image (BGR or grayscale).

        Returns
        -------
        torch.Tensor
            Preprocessed tensor of shape (1, 1, 48, 48).
        """
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi

        # Resize to model input size
        resized = cv2.resize(
            gray, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA
        )

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Convert to tensor: (1, 1, H, W)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, face_roi: np.ndarray) -> tuple[str, float]:
        """
        Predict emotion from a face ROI.

        Parameters
        ----------
        face_roi : np.ndarray
            Cropped face image.

        Returns
        -------
        tuple[str, float]
            (emotion_label, confidence) e.g. ("Happy", 0.87).
        """
        tensor = self.preprocess(face_roi)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_idx = probabilities.max(1)
        emotion_label = self.labels[predicted_idx.item()]

        return emotion_label, confidence.item()

    @torch.no_grad()
    def predict_all(self, face_roi: np.ndarray) -> dict[str, float]:
        """
        Get probabilities for all emotion classes.

        Parameters
        ----------
        face_roi : np.ndarray
            Cropped face image.

        Returns
        -------
        dict[str, float]
            Mapping of emotion → probability.
        """
        tensor = self.preprocess(face_roi)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        return {label: float(prob) for label, prob in zip(self.labels, probabilities)}
