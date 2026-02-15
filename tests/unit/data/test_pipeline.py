"""Tests for windowed canonical sample pipeline."""

from __future__ import annotations

import numpy as np

from whirdetective.data import BearingFaultLabel, build_windowed_canonical_samples
from whirdetective.ml import ProjectionPolicy, SensorSetProjector


def test_build_windowed_canonical_samples_produces_aligned_windows() -> None:
    projector = SensorSetProjector(ProjectionPolicy())
    samples = build_windowed_canonical_samples(
        dataset="cwru",
        machine_id="m-1",
        run_id="run-1",
        label=BearingFaultLabel.HEALTHY,
        channel_signals={
            "de_accel": np.arange(12, dtype=np.float64),
            "temp_bearing": np.linspace(20.0, 21.1, 12, dtype=np.float64),
        },
        projector=projector,
        window_size=4,
        step_size=2,
    )

    assert len(samples) == 5
    assert samples[0].metadata["window_index"] == "0"
    assert samples[-1].metadata["window_index"] == "4"
    assert samples[0].features.shape == (len(projector.role_order), 5)


def test_build_windowed_canonical_samples_uses_min_aligned_count() -> None:
    projector = SensorSetProjector(ProjectionPolicy())
    samples = build_windowed_canonical_samples(
        dataset="cwru",
        machine_id="m-1",
        run_id="run-2",
        label=BearingFaultLabel.UNKNOWN,
        channel_signals={
            "de_accel": np.arange(10, dtype=np.float64),  # windows: 4
            "temp_bearing": np.arange(8, dtype=np.float64),  # windows: 3
        },
        projector=projector,
        window_size=4,
        step_size=2,
    )

    assert len(samples) == 3
