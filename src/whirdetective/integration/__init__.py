"""Integration adapters for external platform payloads."""

from whirdetective.integration.bootctrl_contracts import (
    BootCtrlTelemetrySample,
    BootCtrlTelemetrySource,
    normalize_edge_variable,
    normalize_edge_variable_batch,
    normalize_mqtt_metric,
)
from whirdetective.integration.intake_guard import (
    IntakeMetrics,
    IntakePolicy,
    RejectReason,
    RejectedSample,
    TelemetryIntakeGuard,
)

__all__ = [
    "BootCtrlTelemetrySample",
    "BootCtrlTelemetrySource",
    "IntakeMetrics",
    "IntakePolicy",
    "RejectReason",
    "RejectedSample",
    "TelemetryIntakeGuard",
    "normalize_edge_variable",
    "normalize_edge_variable_batch",
    "normalize_mqtt_metric",
]
