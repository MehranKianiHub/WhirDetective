"""Tests for standardization into canonical training sample contracts."""

from __future__ import annotations

import numpy as np
import pytest

from whirdetective.data import BearingFaultLabel, standardize_channel_sample
from whirdetective.ml import ProjectionPolicy, SensorSetProjector


def test_standardize_channel_sample_builds_canonical_contract() -> None:
    projector = SensorSetProjector(ProjectionPolicy())
    sample = standardize_channel_sample(
        dataset="cwru",
        machine_id="m-01",
        run_id="run-0001",
        label=BearingFaultLabel.INNER_RACE,
        channel_signals={
            "de_accel": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "temp_bearing": np.asarray([30.0, 31.0, 29.5, 30.5], dtype=np.float64),
        },
        projector=projector,
        metadata={"source": "unit_test"},
    )

    assert sample.dataset == "cwru"
    assert sample.machine_id == "m-01"
    assert sample.run_id == "run-0001"
    assert sample.label == BearingFaultLabel.INNER_RACE
    assert sample.features.ndim == 2
    assert sample.presence_mask.shape[0] == sample.features.shape[0]
    assert sample.metadata["source"] == "unit_test"


def test_standardize_rejects_invalid_channel_values() -> None:
    projector = SensorSetProjector(ProjectionPolicy())
    with pytest.raises(ValueError, match="non-finite"):
        standardize_channel_sample(
            dataset="cwru",
            machine_id="m-01",
            run_id="run-0002",
            label=BearingFaultLabel.UNKNOWN,
            channel_signals={"de_accel": np.asarray([1.0, np.nan], dtype=np.float64)},
            projector=projector,
        )
