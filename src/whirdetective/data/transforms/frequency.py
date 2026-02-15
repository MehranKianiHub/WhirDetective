"""Frequency-domain transforms and summary features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SpectrumSummary:
    """Compact summary of one window's magnitude spectrum."""

    dominant_frequency_hz: float
    spectral_centroid_hz: float
    spectral_rms: float
    total_energy: float


def frequency_bins_hz(window_size: int, sampling_rate_hz: float) -> FloatArray:
    """Frequency bins for one-sided real FFT."""
    _validate_sampling_params(window_size=window_size, sampling_rate_hz=sampling_rate_hz)
    return cast(
        FloatArray,
        np.asarray(np.fft.rfftfreq(window_size, d=1.0 / sampling_rate_hz), dtype=np.float64),
    )


def magnitude_spectrum(window: npt.ArrayLike) -> FloatArray:
    """Compute one-sided FFT magnitude spectrum."""
    x = _as_valid_window(window)
    spectrum = np.fft.rfft(x)
    return np.asarray(np.abs(spectrum), dtype=np.float64)


def summarize_spectrum(
    window: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
) -> SpectrumSummary:
    """Extract stable frequency-domain summary features."""
    x = _as_valid_window(window)
    mags = magnitude_spectrum(x)
    freqs = frequency_bins_hz(x.size, sampling_rate_hz)

    dominant_idx = int(np.argmax(mags))
    dominant_hz = float(freqs[dominant_idx])

    mag_sum = float(np.sum(mags))
    if mag_sum <= 0:
        centroid_hz = 0.0
    else:
        centroid_hz = float(np.sum(freqs * mags) / mag_sum)

    spectral_rms = float(np.sqrt(np.mean(np.square(mags))))
    total_energy = float(np.sum(np.square(mags)))

    return SpectrumSummary(
        dominant_frequency_hz=dominant_hz,
        spectral_centroid_hz=centroid_hz,
        spectral_rms=spectral_rms,
        total_energy=total_energy,
    )


def _as_valid_window(window: npt.ArrayLike) -> FloatArray:
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("window must be 1D")
    if x.size < 2:
        raise ValueError("window must have at least 2 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("window must contain only finite values")
    return x


def _validate_sampling_params(*, window_size: int, sampling_rate_hz: float) -> None:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be > 0")
