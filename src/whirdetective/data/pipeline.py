"""High-level pipeline helpers to build canonical samples from raw signals."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import numpy.typing as npt

from whirdetective.data.contracts import CanonicalTrainingSample
from whirdetective.data.labeling import BearingFaultLabel
from whirdetective.data.standardize import standardize_channel_sample
from whirdetective.data.transforms.windowing import sliding_windows
from whirdetective.ml import SensorSetProjector


def build_windowed_canonical_samples(
    *,
    dataset: str,
    machine_id: str,
    run_id: str,
    label: BearingFaultLabel,
    channel_signals: Mapping[str, npt.ArrayLike],
    projector: SensorSetProjector,
    window_size: int,
    step_size: int,
) -> tuple[CanonicalTrainingSample, ...]:
    """Build canonical samples from aligned sliding windows across channels."""
    if not channel_signals:
        raise ValueError("channel_signals must not be empty")

    windows_by_channel: dict[str, npt.NDArray[np.float64]] = {}
    for channel_name, signal in channel_signals.items():
        windows_by_channel[channel_name] = sliding_windows(
            signal,
            window_size=window_size,
            step_size=step_size,
        )

    aligned_count = min(windowed.shape[0] for windowed in windows_by_channel.values())
    if aligned_count <= 0:
        raise ValueError("No aligned windows produced")

    samples: list[CanonicalTrainingSample] = []
    for window_idx in range(aligned_count):
        window_payload = {
            channel_name: windows_by_channel[channel_name][window_idx]
            for channel_name in windows_by_channel
        }
        sample = standardize_channel_sample(
            dataset=dataset,
            machine_id=machine_id,
            run_id=run_id,
            label=label,
            channel_signals=window_payload,
            projector=projector,
            metadata={"window_index": str(window_idx)},
        )
        samples.append(sample)

    return tuple(samples)
