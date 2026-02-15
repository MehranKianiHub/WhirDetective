"""Tests for windowing and frequency-domain transforms."""

from __future__ import annotations

import numpy as np
import pytest

from whirdetective.data.transforms import (
    frequency_bins_hz,
    magnitude_spectrum,
    sliding_windows,
    summarize_spectrum,
)


def test_sliding_windows_shape_and_values() -> None:
    signal = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float64)
    windows = sliding_windows(signal, window_size=3, step_size=2)

    assert windows.shape == (2, 3)
    assert np.allclose(windows[0], np.asarray([0.0, 1.0, 2.0]))
    assert np.allclose(windows[1], np.asarray([2.0, 3.0, 4.0]))


def test_sliding_windows_validates_input_length() -> None:
    with pytest.raises(ValueError, match=">= window_size"):
        sliding_windows(np.asarray([1.0, 2.0]), window_size=4, step_size=1)


def test_frequency_bins_and_magnitude_shape() -> None:
    window = np.asarray([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
    bins = frequency_bins_hz(window_size=window.size, sampling_rate_hz=4.0)
    mags = magnitude_spectrum(window)

    assert bins.shape == mags.shape
    assert bins.shape == (3,)


def test_summarize_spectrum_detects_dominant_frequency() -> None:
    sampling_hz = 128.0
    t = np.arange(0, 1.0, 1.0 / sampling_hz, dtype=np.float64)
    window = np.sin(2.0 * np.pi * 10.0 * t)
    summary = summarize_spectrum(window, sampling_rate_hz=sampling_hz)

    assert summary.dominant_frequency_hz == pytest.approx(10.0, abs=1.0)
    assert summary.spectral_rms > 0
    assert summary.total_energy > 0
