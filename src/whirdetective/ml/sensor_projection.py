"""Sensor-set projection utilities for variable-channel training data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


class SensorRole(StrEnum):
    """Semantic role for a sensor channel independent of source dataset."""

    ACCELERATION = "acceleration"
    TEMPERATURE = "temperature"
    CURRENT = "current"
    SPEED = "speed"
    ACOUSTIC = "acoustic"
    UNKNOWN = "unknown"


_DEFAULT_ROLE_ORDER: tuple[SensorRole, ...] = (
    SensorRole.ACCELERATION,
    SensorRole.TEMPERATURE,
    SensorRole.CURRENT,
    SensorRole.SPEED,
    SensorRole.ACOUSTIC,
)

_DEFAULT_ROLE_ALIASES: dict[SensorRole, tuple[str, ...]] = {
    SensorRole.ACCELERATION: ("accel", "vib", "vibration"),
    SensorRole.TEMPERATURE: ("temp", "temperature", "therm"),
    SensorRole.CURRENT: ("current", "amp", "amps", "ia", "ib", "ic"),
    SensorRole.SPEED: ("rpm", "speed", "tach", "encoder"),
    SensorRole.ACOUSTIC: ("mic", "acoustic", "sound", "audio"),
}

_DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "rms",
    "peak_to_peak",
    "abs_max",
)


@dataclass(frozen=True, slots=True)
class ProjectionPolicy:
    """Configuration for converting variable channel sets into fixed features."""

    role_order: tuple[SensorRole, ...] = _DEFAULT_ROLE_ORDER
    include_unknown_bucket: bool = True
    role_aliases: Mapping[SensorRole, tuple[str, ...]] = field(
        default_factory=lambda: dict(_DEFAULT_ROLE_ALIASES)
    )
    feature_names: tuple[str, ...] = _DEFAULT_FEATURE_NAMES

    def __post_init__(self) -> None:
        if not self.role_order:
            raise ValueError("role_order must not be empty")
        if len(set(self.role_order)) != len(self.role_order):
            raise ValueError("role_order must not contain duplicates")
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")


@dataclass(frozen=True, slots=True)
class ProjectedSample:
    """Fixed-shape representation for one variable-channel input sample."""

    role_order: tuple[SensorRole, ...]
    feature_names: tuple[str, ...]
    features: FloatArray
    presence_mask: BoolArray
    channels_per_role: IntArray

    def as_flat_vector(self) -> FloatArray:
        """Return row-major flattened feature vector."""
        return self.features.reshape(-1)


class SensorSetProjector:
    """Project channel dictionaries into fixed-size role-based feature tensors."""

    def __init__(self, policy: ProjectionPolicy) -> None:
        self._policy = policy
        self._role_order = self._build_role_order(policy)
        self._role_to_index = {role: idx for idx, role in enumerate(self._role_order)}

    @property
    def role_order(self) -> tuple[SensorRole, ...]:
        """Return effective role order used in projection outputs."""
        return self._role_order

    def role_index(self, role: SensorRole) -> int:
        """Return row index for a semantic role."""
        try:
            return self._role_to_index[role]
        except KeyError as exc:
            raise ValueError(f"Role {role.value} is not active in this projector") from exc

    def infer_role(self, channel_name: str) -> SensorRole:
        """Infer semantic role from channel naming conventions."""
        normalized = channel_name.strip().lower()
        if not normalized:
            raise ValueError("channel_name must not be empty")

        for role in self._policy.role_order:
            if role == SensorRole.UNKNOWN:
                continue
            aliases = self._policy.role_aliases.get(role, ())
            if any(alias in normalized for alias in aliases):
                return role
        return SensorRole.UNKNOWN

    def project(self, channel_signals: Mapping[str, npt.ArrayLike]) -> ProjectedSample:
        """Convert arbitrary channel dictionary to fixed-shape feature representation."""
        if not channel_signals:
            raise ValueError("channel_signals must not be empty")

        num_roles = len(self._role_order)
        num_features = len(self._policy.feature_names)
        features = np.zeros((num_roles, num_features), dtype=np.float64)
        presence = np.zeros((num_roles,), dtype=np.bool_)
        channel_counts = np.zeros((num_roles,), dtype=np.int64)

        for channel_name, raw_values in channel_signals.items():
            series = _as_valid_series(raw_values, channel_name)
            role = self.infer_role(channel_name)
            if role not in self._role_to_index:
                continue

            role_idx = self._role_to_index[role]
            channel_features = _extract_channel_features(series)
            existing_count = int(channel_counts[role_idx])
            if existing_count == 0:
                features[role_idx] = channel_features
            else:
                features[role_idx] = (
                    features[role_idx] * existing_count + channel_features
                ) / (existing_count + 1)

            channel_counts[role_idx] = existing_count + 1
            presence[role_idx] = True

        if not np.any(presence):
            raise ValueError("No channels mapped to configured roles")

        return ProjectedSample(
            role_order=self._role_order,
            feature_names=self._policy.feature_names,
            features=features,
            presence_mask=presence,
            channels_per_role=channel_counts,
        )

    @staticmethod
    def _build_role_order(policy: ProjectionPolicy) -> tuple[SensorRole, ...]:
        if policy.include_unknown_bucket and SensorRole.UNKNOWN not in policy.role_order:
            return policy.role_order + (SensorRole.UNKNOWN,)
        return policy.role_order


def _as_valid_series(raw_values: npt.ArrayLike, channel_name: str) -> FloatArray:
    """Convert channel values to finite 1D float array."""
    series = np.asarray(raw_values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError(f"Channel {channel_name!r} must be 1D")
    if series.size == 0:
        raise ValueError(f"Channel {channel_name!r} must not be empty")
    if not np.all(np.isfinite(series)):
        raise ValueError(f"Channel {channel_name!r} contains non-finite values")
    return series


def _extract_channel_features(series: FloatArray) -> FloatArray:
    """Extract simple robust channel features without dependency on fixed length."""
    mean = float(np.mean(series))
    std = float(np.std(series, ddof=0))
    rms = float(np.sqrt(np.mean(np.square(series))))
    peak_to_peak = float(np.max(series) - np.min(series))
    abs_max = float(np.max(np.abs(series)))
    return np.asarray((mean, std, rms, peak_to_peak, abs_max), dtype=np.float64)
