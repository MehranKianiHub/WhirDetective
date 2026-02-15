"""Integration adapters for external platform payloads."""

from whirdetective.integration.bootctrl_contracts import (
    BootCtrlTelemetrySample,
    BootCtrlTelemetrySource,
    normalize_edge_variable,
    normalize_edge_variable_batch,
    normalize_mqtt_metric,
)

__all__ = [
    "BootCtrlTelemetrySample",
    "BootCtrlTelemetrySource",
    "normalize_edge_variable",
    "normalize_edge_variable_batch",
    "normalize_mqtt_metric",
]
