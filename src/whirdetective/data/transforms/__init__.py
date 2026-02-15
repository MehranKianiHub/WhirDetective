"""Signal transform utilities for windowing and frequency-domain features."""

from whirdetective.data.transforms.frequency import (
    frequency_bins_hz,
    magnitude_spectrum,
    summarize_spectrum,
)
from whirdetective.data.transforms.windowing import sliding_windows

__all__ = [
    "frequency_bins_hz",
    "magnitude_spectrum",
    "sliding_windows",
    "summarize_spectrum",
]
