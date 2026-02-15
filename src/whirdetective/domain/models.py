"""Core domain models for WhirDetective."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class EquipmentSafetyState(IntEnum):
    """Ordered severity states for machine safety posture."""

    NORMAL = 0
    WARNING = 1
    DEGRADED = 2
    TRIP = 3


class SensorKind(StrEnum):
    """Supported telemetry sensor categories."""

    VIBRATION_RMS = "vibration_rms"
    TEMPERATURE_C = "temperature_c"
    CURRENT_RMS = "current_rms"


@dataclass(frozen=True, slots=True)
class TelemetryPoint:
    """Single sensor measurement with source timestamp."""

    sensor_id: str
    kind: SensorKind
    value: float
    unit: str
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class MachineTelemetrySnapshot:
    """Time-aligned telemetry packet for one machine."""

    machine_id: str
    timestamp_ms: int
    points: dict[str, TelemetryPoint] = field(default_factory=dict)

    def get(self, sensor_id: str) -> TelemetryPoint | None:
        """Return sensor data by id if present."""
        return self.points.get(sensor_id)
