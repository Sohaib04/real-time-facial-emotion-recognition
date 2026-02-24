"""
EmotionCNN — Convolutional Neural Network for facial emotion classification.

Architecture:
    3 convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool → Dropout)
    followed by a fully-connected classifier.

Input:  1×48×48 grayscale face crop
Output: 7-class probability vector (softmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Reusable convolutional block: Conv → BatchNorm → ReLU → MaxPool → Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class EmotionCNN(nn.Module):
    """
    Custom CNN for 7-class facial emotion recognition.

    Architecture
    ────────────
    Block 1: Conv2d(1  → 32,  3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25)
    Block 2: Conv2d(32 → 64,  3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25)
    Block 3: Conv2d(64 → 128, 3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25)
    Flatten  → Dense(128) → ReLU → Dropout(0.5) → Dense(7)

    Parameters
    ──────────
    num_classes : int
        Number of emotion categories (default: 7 for FER-2013).
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(1, 32),     # 48×48 → 24×24
            ConvBlock(32, 64),    # 24×24 → 12×12
            ConvBlock(64, 128),   # 12×12 → 6×6
        )

        # Classifier head
        # After 3 pools: 48 / (2^3) = 6 → feature map is 128 × 6 × 6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

        # Weight initialization (Kaiming for ReLU networks)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming initialization to conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 48, 48).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(num_classes: int = 7, pretrained_path: str | None = None) -> EmotionCNN:
    """
    Factory function to create and optionally load a pretrained EmotionCNN.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained_path : str or None
        Path to a .pth weights file. If provided, loads state dict.

    Returns
    -------
    EmotionCNN
        The model instance.
    """
    model = EmotionCNN(num_classes=num_classes)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    return model
