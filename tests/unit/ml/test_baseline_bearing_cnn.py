"""Unit tests for baseline PyTorch CNN model."""

from __future__ import annotations

import pytest
import torch

from whirdetective.ml import BaselineBearingCNN


def test_forward_output_shape() -> None:
    model = BaselineBearingCNN(input_channels=1, num_classes=4)
    x = torch.randn(8, 1, 2048)
    y = model(x)
    assert y.shape == (8, 4)


def test_forward_supports_short_sequence_inputs() -> None:
    model = BaselineBearingCNN(input_channels=6, num_classes=3)
    x = torch.randn(4, 6, 5)
    y = model(x)
    assert y.shape == (4, 3)


def test_invalid_channels_raises() -> None:
    with pytest.raises(ValueError, match="input_channels"):
        BaselineBearingCNN(input_channels=0, num_classes=4)


def test_invalid_num_classes_raises() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        BaselineBearingCNN(input_channels=1, num_classes=1)


def test_invalid_input_shape_raises() -> None:
    model = BaselineBearingCNN(input_channels=1, num_classes=4)
    bad = torch.randn(8, 2048)
    with pytest.raises(ValueError, match="shape"):
        model(bad)
