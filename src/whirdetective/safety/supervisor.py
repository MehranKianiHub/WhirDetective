"""Deterministic safety supervisor for industrial telemetry."""

from __future__ import annotations

from math import isfinite

from whirdetective.domain.models import EquipmentSafetyState, MachineTelemetrySnapshot
from whirdetective.safety.contracts import SafetyDecision, SafetyPolicy


class SafetySupervisor:
    """Evaluate machine telemetry and produce a fail-safe decision."""

    def __init__(self, policy: SafetyPolicy) -> None:
        self._policy = policy
        self._trip_breach_counters: dict[str, int] = {
            sensor_id: 0 for sensor_id in policy.sensor_policies
        }

    def evaluate(self, snapshot: MachineTelemetrySnapshot) -> SafetyDecision:
        """Apply deterministic rules to compute machine safety state."""
        state = EquipmentSafetyState.NORMAL
        reasons: list[str] = []
        triggered: list[str] = []

        def escalate(target: EquipmentSafetyState, reason: str, sensor_id: str) -> None:
            nonlocal state
            if target > state:
                state = target
            reasons.append(reason)
            if sensor_id not in triggered:
                triggered.append(sensor_id)

        for sensor_id in sorted(self._policy.required_critical_sensors):
            if snapshot.get(sensor_id) is None:
                escalate(
                    EquipmentSafetyState.TRIP,
                    f"critical sensor missing: {sensor_id}",
                    sensor_id,
                )

        for sensor_id in sorted(self._policy.sensor_policies):
            sensor_policy = self._policy.sensor_policies[sensor_id]
            point = snapshot.get(sensor_id)

            if point is None:
                if sensor_policy.critical:
                    escalate(
                        EquipmentSafetyState.TRIP,
                        f"required sensor missing: {sensor_id}",
                        sensor_id,
                    )
                self._trip_breach_counters[sensor_id] = 0
                continue

            if point.kind != sensor_policy.kind:
                escalate(
                    EquipmentSafetyState.TRIP,
                    f"sensor kind mismatch for {sensor_id}: expected {sensor_policy.kind.value}, got {point.kind.value}",
                    sensor_id,
                )
                self._trip_breach_counters[sensor_id] = 0
                continue

            staleness_ms = snapshot.timestamp_ms - point.timestamp_ms
            if staleness_ms < 0:
                escalate(
                    EquipmentSafetyState.TRIP,
                    f"sensor timestamp is ahead of snapshot time: {sensor_id}",
                    sensor_id,
                )
            elif staleness_ms > sensor_policy.max_staleness_ms:
                if sensor_policy.critical:
                    escalate(
                        EquipmentSafetyState.TRIP,
                        f"stale critical sensor telemetry: {sensor_id} ({staleness_ms} ms)",
                        sensor_id,
                    )
                else:
                    escalate(
                        EquipmentSafetyState.DEGRADED,
                        f"stale non-critical sensor telemetry: {sensor_id} ({staleness_ms} ms)",
                        sensor_id,
                    )

            value = point.value
            if not isfinite(value):
                escalate(
                    EquipmentSafetyState.TRIP,
                    f"non-finite sensor value: {sensor_id}",
                    sensor_id,
                )
                self._trip_breach_counters[sensor_id] = 0
                continue

            if sensor_policy.min_value is not None and value < sensor_policy.min_value:
                escalate(
                    EquipmentSafetyState.TRIP,
                    f"sensor value below minimum for {sensor_id}: {value}",
                    sensor_id,
                )

            if sensor_policy.max_value is not None and value > sensor_policy.max_value:
                escalate(
                    EquipmentSafetyState.TRIP,
                    f"sensor value above maximum for {sensor_id}: {value}",
                    sensor_id,
                )

            if sensor_policy.trip_above is not None and value >= sensor_policy.trip_above:
                self._trip_breach_counters[sensor_id] += 1
                if self._trip_breach_counters[sensor_id] >= self._policy.trip_debounce_samples:
                    escalate(
                        EquipmentSafetyState.TRIP,
                        f"trip threshold breached for {sensor_id}: {value}",
                        sensor_id,
                    )
                else:
                    escalate(
                        EquipmentSafetyState.DEGRADED,
                        f"trip threshold pending debounce for {sensor_id}: {value}",
                        sensor_id,
                    )
                continue

            self._trip_breach_counters[sensor_id] = 0

            if sensor_policy.degrade_above is not None and value >= sensor_policy.degrade_above:
                escalate(
                    EquipmentSafetyState.DEGRADED,
                    f"degraded threshold breached for {sensor_id}: {value}",
                    sensor_id,
                )
                continue

            if sensor_policy.warn_above is not None and value >= sensor_policy.warn_above:
                escalate(
                    EquipmentSafetyState.WARNING,
                    f"warning threshold breached for {sensor_id}: {value}",
                    sensor_id,
                )

        return SafetyDecision(
            state=state,
            snapshot_timestamp_ms=snapshot.timestamp_ms,
            reasons=tuple(reasons),
            triggered_sensors=tuple(triggered),
        )
