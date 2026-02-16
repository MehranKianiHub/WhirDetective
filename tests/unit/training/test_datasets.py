"""Tests for canonical sample to tensor dataset conversion."""

from __future__ import annotations

import numpy as np
import pytest

from whirdetective.data import BearingFaultLabel, CanonicalTrainingSample
from whirdetective.training import canonical_samples_to_dataset


def _sample(label: BearingFaultLabel, offset: float) -> CanonicalTrainingSample:
    features = np.full((3, 5), offset, dtype=np.float64)
    mask = np.ones((3,), dtype=np.bool_)
    return CanonicalTrainingSample(
        dataset="cwru",
        machine_id="m1",
        run_id=f"r{offset}",
        label=label,
        features=features,
        presence_mask=mask,
    )


def test_canonical_samples_to_dataset_shapes_and_mapping() -> None:
    dataset = canonical_samples_to_dataset(
        (
            _sample(BearingFaultLabel.HEALTHY, 0.0),
            _sample(BearingFaultLabel.INNER_RACE, 1.0),
            _sample(BearingFaultLabel.HEALTHY, 2.0),
        )
    )
    assert dataset.inputs.shape == (3, 3, 5)
    assert dataset.labels.shape == (3,)
    assert set(dataset.class_names) == {"healthy", "inner_race"}


def test_canonical_samples_to_dataset_requires_multiple_classes() -> None:
    with pytest.raises(ValueError, match="at least two classes"):
        canonical_samples_to_dataset((_sample(BearingFaultLabel.HEALTHY, 0.0),))
