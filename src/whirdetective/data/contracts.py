"""Canonical training sample contracts for leakage-safe dataset engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import numpy.typing as npt

from whirdetective.data.labeling import BearingFaultLabel


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CanonicalTrainingSample:
    """Fixed-shape, sensor-agnostic sample contract for training/evaluation."""

    dataset: str
    machine_id: str
    run_id: str
    label: BearingFaultLabel
    features: FloatArray
    presence_mask: BoolArray
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("dataset must not be empty")
        if not self.machine_id.strip():
            raise ValueError("machine_id must not be empty")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D array [roles, features_per_role]")
        if self.features.shape[0] <= 0 or self.features.shape[1] <= 0:
            raise ValueError("features must have non-zero dimensions")
        if self.presence_mask.ndim != 1:
            raise ValueError("presence_mask must be a 1D array [roles]")
        if self.presence_mask.shape[0] != self.features.shape[0]:
            raise ValueError("presence_mask length must match feature role dimension")
        if not np.all(np.isfinite(self.features)):
            raise ValueError("features must contain finite numeric values")

    def as_flat_vector(self) -> FloatArray:
        """Flatten role-feature matrix for simple baseline models."""
        return self.features.reshape(-1)
