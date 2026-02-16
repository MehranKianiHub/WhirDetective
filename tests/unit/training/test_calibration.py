"""Tests for temperature scaling calibration."""

from __future__ import annotations

import torch

from whirdetective.training import TemperatureScaler, optimize_temperature


def test_temperature_scaler_has_positive_temperature() -> None:
    scaler = TemperatureScaler(initial_temperature=1.5)
    assert float(scaler.temperature.item()) > 0.0


def test_optimize_temperature_reduces_or_preserves_nll() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    result = optimize_temperature(logits, labels, max_iter=100)

    assert result.temperature > 0.0
    assert result.nll_after <= result.nll_before + 1e-6
