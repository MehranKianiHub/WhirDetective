"""Standardization helpers from raw channel dictionaries to canonical samples."""

from __future__ import annotations

from typing import Mapping

import numpy.typing as npt

from whirdetective.data.contracts import CanonicalTrainingSample
from whirdetective.data.labeling import BearingFaultLabel
from whirdetective.ml import SensorSetProjector


def standardize_channel_sample(
    *,
    dataset: str,
    machine_id: str,
    run_id: str,
    label: BearingFaultLabel,
    channel_signals: Mapping[str, npt.ArrayLike],
    projector: SensorSetProjector,
    metadata: Mapping[str, str] | None = None,
) -> CanonicalTrainingSample:
    """Build canonical training sample from variable-channel signal dictionary."""
    projected = projector.project(channel_signals)
    return CanonicalTrainingSample(
        dataset=dataset,
        machine_id=machine_id,
        run_id=run_id,
        label=label,
        features=projected.features,
        presence_mask=projected.presence_mask,
        metadata={} if metadata is None else metadata,
    )
