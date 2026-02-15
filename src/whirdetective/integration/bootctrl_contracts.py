"""BootCtrl payload normalization contracts.

This module keeps WhirDetective transport-agnostic by normalizing upstream payloads
from BootCtrl backend/edge APIs into one strict telemetry contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite


class BootCtrlTelemetrySource(StrEnum):
    """Supported upstream source channels inside BootCtrl."""

    MQTT_METRIC = "mqtt_metric"
    EDGE_VARIABLE = "edge_variable"


@dataclass(frozen=True, slots=True)
class BootCtrlTelemetrySample:
    """Strict normalized telemetry sample used by WhirDetective pipelines."""

    device_id: str
    signal_key: str
    value: float
    timestamp_ms: int
    source: BootCtrlTelemetrySource
    quality: float = 1.0

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id is required")
        if not self.signal_key.strip():
            raise ValueError("signal_key is required")
        if not isfinite(self.value):
            raise ValueError("value must be finite")
        if self.timestamp_ms <= 0:
            raise ValueError("timestamp_ms must be > 0")
        if not (0.0 <= self.quality <= 1.0):
            raise ValueError("quality must be in range [0.0, 1.0]")


def normalize_mqtt_metric(
    payload: Mapping[str, object],
    *,
    fallback_device_id: str | None = None,
) -> BootCtrlTelemetrySample:
    """Normalize one MQTT ingestion payload from BootCtrl backend."""
    device_id = _text_or_none(
        _pick(payload, "deviceId", "device_id", "deviceIdentifier", "device")
    )
    if device_id is None:
        device_id = fallback_device_id.strip() if fallback_device_id else None
    if device_id is None or not device_id:
        raise ValueError("device_id is required for MQTT payload normalization")

    signal_key = _require_text(
        _pick(payload, "topic", "signal", "variable", "name"),
        field_name="topic/signal",
    )
    value = _require_float(_pick(payload, "value"), field_name="value")
    timestamp_ms = _require_timestamp_ms(
        _pick(payload, "timestamp_ms", "timestampMs", "timestamp", "ts", "time"),
        field_name="timestamp",
    )
    quality = _optional_quality(_pick(payload, "quality", "confidence"))

    return BootCtrlTelemetrySample(
        device_id=device_id,
        signal_key=signal_key,
        value=value,
        timestamp_ms=timestamp_ms,
        quality=quality,
        source=BootCtrlTelemetrySource.MQTT_METRIC,
    )


def normalize_edge_variable(
    payload: Mapping[str, object],
    *,
    device_id: str,
    snapshot_timestamp_ms: int | None = None,
) -> BootCtrlTelemetrySample:
    """Normalize one edge variable payload from BootCtrl edge API proxies."""
    normalized_device_id = device_id.strip()
    if not normalized_device_id:
        raise ValueError("device_id is required for edge variable normalization")

    signal_key = _require_text(
        _pick(payload, "name", "variable", "key", "id"),
        field_name="name/variable",
    )
    value = _require_float(_pick(payload, "value", "currentValue"), field_name="value")

    raw_ts = _pick(payload, "timestamp_ms", "timestampMs", "timestamp", "ts", "time")
    if raw_ts is None:
        if snapshot_timestamp_ms is None:
            raise ValueError("timestamp is required when snapshot_timestamp_ms is not provided")
        timestamp_ms = _require_timestamp_ms(snapshot_timestamp_ms, field_name="snapshot_timestamp_ms")
    else:
        timestamp_ms = _require_timestamp_ms(raw_ts, field_name="timestamp")

    quality = _optional_quality(_pick(payload, "quality", "confidence"))

    return BootCtrlTelemetrySample(
        device_id=normalized_device_id,
        signal_key=signal_key,
        value=value,
        timestamp_ms=timestamp_ms,
        quality=quality,
        source=BootCtrlTelemetrySource.EDGE_VARIABLE,
    )


def normalize_edge_variable_batch(
    payloads: Sequence[Mapping[str, object]],
    *,
    device_id: str,
    snapshot_timestamp_ms: int | None = None,
) -> tuple[BootCtrlTelemetrySample, ...]:
    """Normalize a batch of edge variable payloads."""
    samples = [
        normalize_edge_variable(
            payload,
            device_id=device_id,
            snapshot_timestamp_ms=snapshot_timestamp_ms,
        )
        for payload in payloads
    ]
    return tuple(samples)


def _pick(payload: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _text_or_none(raw: object | None) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        value = raw.strip()
        return value or None
    return str(raw).strip() or None


def _require_text(raw: object | None, *, field_name: str) -> str:
    value = _text_or_none(raw)
    if value is None:
        raise ValueError(f"{field_name} is required")
    return value


def _require_float(raw: object | None, *, field_name: str) -> float:
    if raw is None:
        raise ValueError(f"{field_name} is required")
    if isinstance(raw, bool):
        raise ValueError(f"{field_name} must be numeric")
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError(f"{field_name} is required")
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be numeric") from exc
    else:
        raise ValueError(f"{field_name} must be numeric")

    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _optional_quality(raw: object | None) -> float:
    if raw is None:
        return 1.0
    return _require_float(raw, field_name="quality")


def _require_timestamp_ms(raw: object | None, *, field_name: str) -> int:
    if raw is None:
        raise ValueError(f"{field_name} is required")

    if isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a timestamp")

    if isinstance(raw, (int, float)):
        numeric = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError(f"{field_name} is required")
        try:
            numeric = float(text)
        except ValueError:
            parsed = _parse_iso8601_timestamp_ms(text)
            if parsed <= 0:
                raise ValueError(f"{field_name} must be > 0")
            return parsed
    else:
        raise ValueError(f"{field_name} must be a timestamp")

    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    if numeric <= 0:
        raise ValueError(f"{field_name} must be > 0")

    # Heuristic: values below 1e11 are treated as epoch seconds.
    # Current epoch milliseconds are already above this threshold.
    as_int = int(numeric)
    if as_int < 100_000_000_000:
        as_int *= 1000
    return as_int


def _parse_iso8601_timestamp_ms(value: str) -> int:
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)
