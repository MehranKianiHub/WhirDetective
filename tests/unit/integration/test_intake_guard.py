"""Unit tests for deterministic telemetry intake guardrails."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whirdetective.integration import (
    IntakePolicy,
    RejectReason,
    TelemetryIntakeGuard,
    normalize_edge_variable,
    normalize_mqtt_metric,
)


FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "bootctrl"


def _sample(v: float = 3.0, ts_ms: int = 1_735_801_000_000):
    return normalize_mqtt_metric(
        {
            "deviceId": "edge-001",
            "topic": "automation/edge/edge-001/bearing/vibration_rms",
            "value": v,
            "timestampMs": ts_ms,
        }
    )


def test_accepts_clean_sample() -> None:
    guard = TelemetryIntakeGuard(IntakePolicy())
    sample = _sample()

    rejection = guard.evaluate(sample, ingest_time_ms=sample.timestamp_ms + 500)

    assert rejection is None
    assert guard.metrics.accepted == 1


def test_rejects_stale_sample() -> None:
    guard = TelemetryIntakeGuard(IntakePolicy(max_age_ms=1_000))
    sample = _sample(ts_ms=1_735_801_000_000)

    rejection = guard.evaluate(sample, ingest_time_ms=1_735_801_002_500)

    assert rejection is not None
    assert rejection.reason == RejectReason.STALE_TIMESTAMP


def test_rejects_future_skewed_sample() -> None:
    guard = TelemetryIntakeGuard(IntakePolicy(max_future_skew_ms=100))
    sample = _sample(ts_ms=1_735_801_000_500)

    rejection = guard.evaluate(sample, ingest_time_ms=1_735_801_000_000)

    assert rejection is not None
    assert rejection.reason == RejectReason.FUTURE_TIMESTAMP


def test_rejects_duplicate_event_id() -> None:
    guard = TelemetryIntakeGuard(IntakePolicy())
    sample = _sample()

    first = guard.evaluate(sample, ingest_time_ms=sample.timestamp_ms + 100, event_id="evt-1")
    second = guard.evaluate(sample, ingest_time_ms=sample.timestamp_ms + 200, event_id="evt-1")

    assert first is None
    assert second is not None
    assert second.reason == RejectReason.DUPLICATE_EVENT


def test_rejects_timestamp_regression_without_tolerance() -> None:
    guard = TelemetryIntakeGuard(IntakePolicy(allow_out_of_order_ms=0))

    newer = _sample(ts_ms=1_735_801_002_000)
    older = _sample(ts_ms=1_735_801_001_900)

    assert guard.evaluate(newer, ingest_time_ms=1_735_801_002_050) is None
    rejection = guard.evaluate(older, ingest_time_ms=1_735_801_002_100)

    assert rejection is not None
    assert rejection.reason == RejectReason.TIMESTAMP_REGRESSION


def test_allows_small_out_of_order_with_tolerance() -> None:
    guard = TelemetryIntakeGuard(IntakePolicy(allow_out_of_order_ms=200))

    newer = _sample(ts_ms=1_735_801_002_000)
    slightly_older = _sample(ts_ms=1_735_801_001_900)

    assert guard.evaluate(newer, ingest_time_ms=1_735_801_002_050) is None
    assert guard.evaluate(slightly_older, ingest_time_ms=1_735_801_002_100) is None


def test_process_batch_with_fixture_payloads() -> None:
    guard = TelemetryIntakeGuard(IntakePolicy(max_age_ms=100_000_000_000))

    mqtt_payloads = json.loads((FIXTURES_DIR / "mqtt_metrics.json").read_text(encoding="utf-8"))
    edge_payloads = json.loads((FIXTURES_DIR / "edge_variables.json").read_text(encoding="utf-8"))

    mqtt_samples = tuple(normalize_mqtt_metric(item) for item in mqtt_payloads)
    edge_samples = tuple(
        normalize_edge_variable(item, device_id="edge-001") for item in edge_payloads
    )

    all_samples = mqtt_samples + edge_samples
    accepted, rejected = guard.process_batch(all_samples, ingest_time_ms=1_800_000_000_000)

    assert len(accepted) == len(all_samples)
    assert not rejected


def test_batch_event_ids_must_match_length() -> None:
    guard = TelemetryIntakeGuard(IntakePolicy())
    sample = _sample()

    with pytest.raises(ValueError, match="length"):
        guard.process_batch((sample,), ingest_time_ms=sample.timestamp_ms + 1, event_ids=("a", "b"))
