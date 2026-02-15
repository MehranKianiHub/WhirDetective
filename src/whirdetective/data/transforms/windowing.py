"""Windowing transforms for one-dimensional sensor signals."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


def sliding_windows(
    signal: npt.ArrayLike,
    *,
    window_size: int,
    step_size: int,
) -> FloatArray:
    """Create deterministic sliding windows with shape [num_windows, window_size]."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size < window_size:
        raise ValueError("signal length must be >= window_size")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal must contain only finite values")

    windows: list[FloatArray] = []
    for start in range(0, x.size - window_size + 1, step_size):
        windows.append(x[start : start + window_size])

    if not windows:
        raise ValueError("No windows produced; check window_size and step_size")

    return np.stack(windows, axis=0)
