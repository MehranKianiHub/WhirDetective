"""Unit tests for deterministic safety supervision."""

from __future__ import annotations

from whirdetective.domain.models import EquipmentSafetyState, MachineTelemetrySnapshot, SensorKind, TelemetryPoint
from whirdetective.safety.contracts import SafetyPolicy, SensorThresholdPolicy
from whirdetective.safety.supervisor import SafetySupervisor


def _make_policy() -> SafetyPolicy:
    return SafetyPolicy(
        sensor_policies={
            "vib_1": SensorThresholdPolicy(
                sensor_id="vib_1",
                kind=SensorKind.VIBRATION_RMS,
                critical=True,
                max_staleness_ms=1_000,
                warn_above=3.0,
                degrade_above=5.0,
                trip_above=7.0,
                min_value=0.0,
                max_value=20.0,
            ),
            "temp_1": SensorThresholdPolicy(
                sensor_id="temp_1",
                kind=SensorKind.TEMPERATURE_C,
                critical=False,
                max_staleness_ms=2_000,
                warn_above=70.0,
                degrade_above=80.0,
                trip_above=90.0,
                min_value=-20.0,
                max_value=140.0,
            ),
        },
        required_critical_sensors={"vib_1"},
        trip_debounce_samples=2,
    )


def _snapshot(
    *,
    timestamp_ms: int = 10_000,
    vib_value: float = 2.0,
    temp_value: float = 50.0,
    vib_timestamp_ms: int | None = None,
    temp_timestamp_ms: int | None = None,
    include_vib: bool = True,
    include_temp: bool = True,
) -> MachineTelemetrySnapshot:
    vib_ts = vib_timestamp_ms if vib_timestamp_ms is not None else timestamp_ms
    temp_ts = temp_timestamp_ms if temp_timestamp_ms is not None else timestamp_ms

    points: dict[str, TelemetryPoint] = {}
    if include_vib:
        points["vib_1"] = TelemetryPoint(
            sensor_id="vib_1",
            kind=SensorKind.VIBRATION_RMS,
            value=vib_value,
            unit="g",
            timestamp_ms=vib_ts,
        )
    if include_temp:
        points["temp_1"] = TelemetryPoint(
            sensor_id="temp_1",
            kind=SensorKind.TEMPERATURE_C,
            value=temp_value,
            unit="C",
            timestamp_ms=temp_ts,
        )

    return MachineTelemetrySnapshot(machine_id="M-01", timestamp_ms=timestamp_ms, points=points)


def test_normal_telemetry_returns_normal_state() -> None:
    supervisor = SafetySupervisor(_make_policy())
    decision = supervisor.evaluate(_snapshot())

    assert decision.state == EquipmentSafetyState.NORMAL
    assert not decision.reasons


def test_warning_threshold_escalates_to_warning() -> None:
    supervisor = SafetySupervisor(_make_policy())
    decision = supervisor.evaluate(_snapshot(vib_value=3.2))

    assert decision.state == EquipmentSafetyState.WARNING
    assert any("warning threshold" in reason for reason in decision.reasons)


def test_degraded_threshold_escalates_to_degraded() -> None:
    supervisor = SafetySupervisor(_make_policy())
    decision = supervisor.evaluate(_snapshot(vib_value=5.1))

    assert decision.state == EquipmentSafetyState.DEGRADED
    assert any("degraded threshold" in reason for reason in decision.reasons)


def test_missing_critical_sensor_trips_machine() -> None:
    supervisor = SafetySupervisor(_make_policy())
    decision = supervisor.evaluate(_snapshot(include_vib=False))

    assert decision.state == EquipmentSafetyState.TRIP
    assert decision.should_trip is True
    assert any("critical sensor missing" in reason for reason in decision.reasons)


def test_stale_critical_sensor_trips_machine() -> None:
    supervisor = SafetySupervisor(_make_policy())
    decision = supervisor.evaluate(_snapshot(vib_timestamp_ms=8_500, timestamp_ms=10_000))

    assert decision.state == EquipmentSafetyState.TRIP
    assert any("stale critical" in reason for reason in decision.reasons)


def test_stale_noncritical_sensor_degrades_machine() -> None:
    supervisor = SafetySupervisor(_make_policy())
    decision = supervisor.evaluate(_snapshot(temp_timestamp_ms=7_000, timestamp_ms=10_000))

    assert decision.state == EquipmentSafetyState.DEGRADED
    assert any("stale non-critical" in reason for reason in decision.reasons)


def test_trip_threshold_uses_debounce_then_trips() -> None:
    supervisor = SafetySupervisor(_make_policy())

    first = supervisor.evaluate(_snapshot(vib_value=7.2))
    second = supervisor.evaluate(_snapshot(vib_value=7.3))

    assert first.state == EquipmentSafetyState.DEGRADED
    assert any("pending debounce" in reason for reason in first.reasons)
    assert second.state == EquipmentSafetyState.TRIP
    assert any("trip threshold breached" in reason for reason in second.reasons)


def test_trip_debounce_resets_after_safe_value() -> None:
    supervisor = SafetySupervisor(_make_policy())

    supervisor.evaluate(_snapshot(vib_value=7.2))
    supervisor.evaluate(_snapshot(vib_value=2.0))
    third = supervisor.evaluate(_snapshot(vib_value=7.4))

    assert third.state == EquipmentSafetyState.DEGRADED


def test_non_finite_value_trips_machine() -> None:
    supervisor = SafetySupervisor(_make_policy())
    decision = supervisor.evaluate(_snapshot(vib_value=float("nan")))

    assert decision.state == EquipmentSafetyState.TRIP
    assert any("non-finite" in reason for reason in decision.reasons)


def test_policy_rejects_unknown_required_critical_sensor() -> None:
    try:
        SafetyPolicy(
            sensor_policies={
                "vib_1": SensorThresholdPolicy(
                    sensor_id="vib_1",
                    kind=SensorKind.VIBRATION_RMS,
                    max_staleness_ms=1_000,
                )
            },
            required_critical_sensors={"missing_sensor"},
        )
    except ValueError as exc:
        assert "required_critical_sensors" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown required critical sensor")
