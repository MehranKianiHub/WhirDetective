"""Tests for sensor-set projection with variable channel counts."""

from __future__ import annotations

import numpy as np
import pytest

from whirdetective.ml import ProjectionPolicy, SensorRole, SensorSetProjector


def test_projection_shape_is_stable_across_channel_counts() -> None:
    projector = SensorSetProjector(ProjectionPolicy())

    dataset_a = {
        "de_accel": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "temp_bearing": np.asarray([25.0, 25.5, 26.0, 26.5], dtype=np.float64),
    }
    dataset_b = {
        "de_accel": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "fe_accel": np.asarray([0.8, 1.8, 2.8, 3.8], dtype=np.float64),
        "temp_bearing": np.asarray([25.0, 25.5, 26.0, 26.5], dtype=np.float64),
        "line_current": np.asarray([10.0, 10.2, 10.1, 10.3], dtype=np.float64),
    }

    projected_a = projector.project(dataset_a)
    projected_b = projector.project(dataset_b)

    assert projected_a.features.shape == projected_b.features.shape
    assert projected_a.features.shape == (len(projector.role_order), 5)


def test_missing_role_is_zero_and_masked_false() -> None:
    projector = SensorSetProjector(ProjectionPolicy())
    projected = projector.project({"de_accel": np.asarray([1.0, 2.0, 3.0], dtype=np.float64)})

    temperature_idx = projector.role_index(SensorRole.TEMPERATURE)
    assert not bool(projected.presence_mask[temperature_idx])
    assert np.allclose(projected.features[temperature_idx], np.zeros((5,), dtype=np.float64))


def test_multiple_channels_aggregate_per_role() -> None:
    projector = SensorSetProjector(ProjectionPolicy())
    projected = projector.project(
        {
            "de_accel": np.ones((32,), dtype=np.float64),
            "fe_accel": np.full((32,), 3.0, dtype=np.float64),
        }
    )

    accel_idx = projector.role_index(SensorRole.ACCELERATION)
    mean_idx = projected.feature_names.index("mean")
    rms_idx = projected.feature_names.index("rms")

    assert int(projected.channels_per_role[accel_idx]) == 2
    assert projected.features[accel_idx, mean_idx] == pytest.approx(2.0)
    assert projected.features[accel_idx, rms_idx] == pytest.approx(2.0)


def test_unknown_channels_require_unknown_bucket_or_are_rejected() -> None:
    with_unknown = SensorSetProjector(ProjectionPolicy(include_unknown_bucket=True))
    projected = with_unknown.project({"mystery_sensor": np.asarray([1.0, 2.0], dtype=np.float64)})
    unknown_idx = with_unknown.role_index(SensorRole.UNKNOWN)

    assert int(projected.channels_per_role[unknown_idx]) == 1
    assert bool(projected.presence_mask[unknown_idx])

    without_unknown = SensorSetProjector(ProjectionPolicy(include_unknown_bucket=False))
    with pytest.raises(ValueError, match="No channels mapped"):
        without_unknown.project({"mystery_sensor": np.asarray([1.0, 2.0], dtype=np.float64)})
