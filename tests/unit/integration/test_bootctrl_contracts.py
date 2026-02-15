"""Unit tests for BootCtrl payload normalization adapters."""

from __future__ import annotations

import pytest

from whirdetective.integration import (
    BootCtrlTelemetrySource,
    normalize_edge_variable,
    normalize_edge_variable_batch,
    normalize_mqtt_metric,
)


def test_normalize_mqtt_metric_with_epoch_seconds() -> None:
    sample = normalize_mqtt_metric(
        {
            "deviceId": "edge-001",
            "topic": "automation/edge/edge-001/bearing/vibration_rms",
            "value": 3.14,
            "ts": 1_735_801_234,
        }
    )

    assert sample.device_id == "edge-001"
    assert sample.signal_key.endswith("vibration_rms")
    assert sample.value == pytest.approx(3.14)
    assert sample.timestamp_ms == 1_735_801_234_000
    assert sample.source == BootCtrlTelemetrySource.MQTT_METRIC


def test_normalize_mqtt_metric_with_iso_timestamp() -> None:
    sample = normalize_mqtt_metric(
        {
            "deviceIdentifier": "edge-009",
            "topic": "automation/edge/edge-009/bearing/temp",
            "value": "88.2",
            "timestamp": "2026-02-15T10:12:13Z",
        }
    )

    assert sample.device_id == "edge-009"
    assert sample.value == pytest.approx(88.2)
    assert sample.timestamp_ms > 0


def test_normalize_edge_variable_uses_snapshot_timestamp() -> None:
    sample = normalize_edge_variable(
        {"name": "bearing.current_rms", "value": 12.5},
        device_id="edge-777",
        snapshot_timestamp_ms=1_735_801_999_123,
    )

    assert sample.device_id == "edge-777"
    assert sample.signal_key == "bearing.current_rms"
    assert sample.timestamp_ms == 1_735_801_999_123
    assert sample.source == BootCtrlTelemetrySource.EDGE_VARIABLE


def test_normalize_edge_variable_batch() -> None:
    samples = normalize_edge_variable_batch(
        [
            {"name": "bearing.vibration_rms", "value": 1.1, "timestamp_ms": 1_735_802_000_000},
            {"name": "bearing.temperature_c", "value": 62.0, "timestamp_ms": 1_735_802_000_001},
        ],
        device_id="edge-101",
    )

    assert len(samples) == 2
    assert samples[0].device_id == "edge-101"


def test_non_finite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_mqtt_metric(
            {
                "deviceId": "edge-x",
                "topic": "automation/edge/edge-x/bearing/vibration",
                "value": "nan",
                "ts": 1_735_801_234,
            }
        )


def test_invalid_quality_is_rejected() -> None:
    with pytest.raises(ValueError, match="quality"):
        normalize_edge_variable(
            {
                "name": "bearing.vibration_rms",
                "value": 1.0,
                "timestamp_ms": 1_735_802_010_000,
                "quality": 1.5,
            },
            device_id="edge-555",
        )
