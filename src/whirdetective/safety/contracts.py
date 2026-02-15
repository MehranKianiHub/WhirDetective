"""Safety policy contracts used by the deterministic supervisor."""

from __future__ import annotations

from dataclasses import dataclass, field

from whirdetective.domain.models import EquipmentSafetyState, SensorKind


@dataclass(frozen=True, slots=True)
class SensorThresholdPolicy:
    """Threshold and validation limits for one sensor stream."""

    sensor_id: str
    kind: SensorKind
    max_staleness_ms: int
    critical: bool = True
    warn_above: float | None = None
    degrade_above: float | None = None
    trip_above: float | None = None
    min_value: float | None = None
    max_value: float | None = None

    def __post_init__(self) -> None:
        if self.max_staleness_ms <= 0:
            raise ValueError("max_staleness_ms must be > 0")

        bounds = [self.warn_above, self.degrade_above, self.trip_above]
        filtered = [value for value in bounds if value is not None]
        if filtered != sorted(filtered):
            raise ValueError("warn/degrade/trip thresholds must be monotonic ascending")

        if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
            raise ValueError("min_value cannot be greater than max_value")


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """Global policy for deterministic machine safety assessment."""

    sensor_policies: dict[str, SensorThresholdPolicy] = field(default_factory=dict)
    required_critical_sensors: set[str] = field(default_factory=set)
    trip_debounce_samples: int = 1

    def __post_init__(self) -> None:
        if self.trip_debounce_samples <= 0:
            raise ValueError("trip_debounce_samples must be > 0")

        unknown_required = self.required_critical_sensors - set(self.sensor_policies)
        if unknown_required:
            missing = ", ".join(sorted(unknown_required))
            raise ValueError(f"required_critical_sensors missing from sensor_policies: {missing}")


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """Decision produced for one telemetry snapshot."""

    state: EquipmentSafetyState
    snapshot_timestamp_ms: int
    reasons: tuple[str, ...] = ()
    triggered_sensors: tuple[str, ...] = ()

    @property
    def should_trip(self) -> bool:
        """Whether equipment should enter trip/shutdown state."""
        return self.state == EquipmentSafetyState.TRIP
