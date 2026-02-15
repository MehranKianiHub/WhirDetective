"""Baseline PyTorch models for bearing condition analysis."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class BaselineBearingCNN(nn.Module):
    """Simple 1D CNN baseline for bearing fault classification."""

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass for shape [batch, channels, samples]."""
        if x.ndim != 3:
            raise ValueError("Input tensor must have shape [batch, channels, samples]")
        logits = self.classifier(self.features(x))
        return cast(torch.Tensor, logits)
