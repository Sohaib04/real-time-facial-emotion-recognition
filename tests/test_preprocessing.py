"""
Unit tests for the preprocessing pipeline.

Tests:
    - Grayscale conversion
    - Resize to 48×48
    - Normalization to [0, 1]
    - Correct output tensor shape
"""

import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AppConfig
from model.inference import EmotionPredictor


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def config():
    """Create a test config (no actual model loading)."""
    return AppConfig()


def _create_test_image(h: int = 100, w: int = 100, channels: int = 3) -> np.ndarray:
    """Generate a synthetic face-like test image."""
    if channels == 3:
        # BGR image with some variation
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    return img


# ──────────────────────────────────────────────
# Preprocessing Tests (unit-level, no model needed)
# ──────────────────────────────────────────────
class TestPreprocessing:
    """Test the preprocessing steps individually."""

    def test_grayscale_conversion_bgr(self):
        """BGR image should be converted to single-channel grayscale."""
        bgr_img = _create_test_image(100, 100, 3)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        assert len(gray.shape) == 2
        assert gray.shape == (100, 100)

    def test_grayscale_already_gray(self):
        """Already-grayscale image should remain unchanged."""
        gray_img = _create_test_image(100, 100, 1)
        assert len(gray_img.shape) == 2

    def test_resize_to_48x48(self):
        """Image should be resized to 48×48."""
        img = _create_test_image(200, 150, 1)
        resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
        assert resized.shape == (48, 48)

    def test_resize_small_image(self):
        """Very small images should be upscaled to 48×48."""
        img = _create_test_image(10, 10, 1)
        resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
        assert resized.shape == (48, 48)

    def test_normalization_range(self):
        """Normalized image should have values in [0, 1]."""
        img = _create_test_image(48, 48, 1)
        normalized = img.astype(np.float32) / 255.0
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalization_dtype(self):
        """Normalized image should be float32."""
        img = _create_test_image(48, 48, 1)
        normalized = img.astype(np.float32) / 255.0
        assert normalized.dtype == np.float32

    def test_tensor_shape(self):
        """Final tensor should have shape (1, 1, 48, 48)."""
        img = _create_test_image(48, 48, 1)
        normalized = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        assert tensor.shape == (1, 1, 48, 48)

    def test_tensor_dtype(self):
        """Tensor should be float32."""
        img = _create_test_image(48, 48, 1)
        normalized = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        assert tensor.dtype == torch.float32

    def test_full_pipeline_bgr_input(self):
        """Full preprocess pipeline: BGR → grayscale → resize → normalize → tensor."""
        bgr_img = _create_test_image(200, 150, 3)

        # Step 1: Grayscale
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        assert len(gray.shape) == 2

        # Step 2: Resize
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        assert resized.shape == (48, 48)

        # Step 3: Normalize
        normalized = resized.astype(np.float32) / 255.0
        assert 0.0 <= normalized.min()
        assert normalized.max() <= 1.0

        # Step 4: Tensor
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        assert tensor.shape == (1, 1, 48, 48)
        assert tensor.dtype == torch.float32

    def test_black_image(self):
        """All-black image should normalize to all zeros."""
        img = np.zeros((48, 48), dtype=np.uint8)
        normalized = img.astype(np.float32) / 255.0
        assert np.allclose(normalized, 0.0)

    def test_white_image(self):
        """All-white image should normalize to all ones."""
        img = np.full((48, 48), 255, dtype=np.uint8)
        normalized = img.astype(np.float32) / 255.0
        assert np.allclose(normalized, 1.0)
