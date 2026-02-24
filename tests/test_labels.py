"""
Unit tests for the emotion label mapping and model output shape.

Tests:
    - Correct number of labels (7)
    - Specific label ordering (FER-2013 standard)
    - Index ↔ label consistency
    - Model output shape verification
    - Color mapping completeness
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMOTION_COLORS, EMOTION_LABELS
from model.model import EmotionCNN


# ──────────────────────────────────────────────
# Label Mapping Tests
# ──────────────────────────────────────────────
class TestEmotionLabels:
    """Verify emotion label definitions and consistency."""

    EXPECTED_LABELS = [
        "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
    ]

    def test_label_count(self):
        """Should have exactly 7 emotion categories."""
        assert len(EMOTION_LABELS) == 7

    def test_label_order(self):
        """Labels should follow the FER-2013 standard order."""
        assert EMOTION_LABELS == self.EXPECTED_LABELS

    def test_index_to_label(self):
        """Index-based lookup should return the correct label."""
        assert EMOTION_LABELS[0] == "Angry"
        assert EMOTION_LABELS[3] == "Happy"
        assert EMOTION_LABELS[6] == "Neutral"

    def test_label_to_index(self):
        """Label-based lookup should return the correct index."""
        assert EMOTION_LABELS.index("Angry") == 0
        assert EMOTION_LABELS.index("Happy") == 3
        assert EMOTION_LABELS.index("Neutral") == 6

    def test_all_labels_are_strings(self):
        """All labels should be strings."""
        for label in EMOTION_LABELS:
            assert isinstance(label, str)

    def test_no_duplicate_labels(self):
        """There should be no duplicate labels."""
        assert len(EMOTION_LABELS) == len(set(EMOTION_LABELS))


# ──────────────────────────────────────────────
# Color Mapping Tests
# ──────────────────────────────────────────────
class TestEmotionColors:
    """Verify emotion color mapping."""

    def test_all_emotions_have_colors(self):
        """Every emotion label should have an assigned color."""
        for label in EMOTION_LABELS:
            assert label in EMOTION_COLORS, f"Missing color for: {label}"

    def test_colors_are_bgr_tuples(self):
        """Colors should be 3-element tuples (BGR format)."""
        for label, color in EMOTION_COLORS.items():
            assert isinstance(color, tuple), f"{label}: not a tuple"
            assert len(color) == 3, f"{label}: not 3 elements"

    def test_color_values_in_range(self):
        """Color values should be in [0, 255]."""
        for label, color in EMOTION_COLORS.items():
            for channel in color:
                assert 0 <= channel <= 255, f"{label}: value {channel} out of range"


# ──────────────────────────────────────────────
# Model Output Shape Tests
# ──────────────────────────────────────────────
class TestModelOutput:
    """Verify model architecture output dimensions."""

    @pytest.fixture
    def model(self):
        """Create a fresh model instance."""
        return EmotionCNN(num_classes=7)

    def test_output_shape_single(self, model):
        """Single input should produce shape (1, 7)."""
        x = torch.randn(1, 1, 48, 48)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 7)

    def test_output_shape_batch(self, model):
        """Batch input should produce shape (batch, 7)."""
        x = torch.randn(8, 1, 48, 48)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 7)

    def test_softmax_sums_to_one(self, model):
        """Softmax output should sum to ~1.0."""
        x = torch.randn(4, 1, 48, 48)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_output_dtype(self, model):
        """Output should be float32."""
        x = torch.randn(1, 1, 48, 48)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_model_parameter_count(self, model):
        """Model should have a reasonable number of parameters (< 5M for CNN)."""
        total = sum(p.numel() for p in model.parameters())
        assert total < 5_000_000, f"Model too large: {total:,} parameters"
        assert total > 10_000, f"Model too small: {total:,} parameters"
