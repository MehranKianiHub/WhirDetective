"""EdgeOS runtime contract artifact helpers for model handoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whirdetective.export.manifest import sha256_file


EDGEOS_MODEL_MANIFEST_SCHEMA_VERSION = "edgeos.model_manifest.v1"


def build_edgeos_model_manifest(
    *,
    model_id: str,
    model_version: str,
    model_file: Path,
    input_channels: int,
    num_classes: int,
    class_names: tuple[str, ...],
    temperature: float,
    abstention_threshold: float,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    """Build versioned EdgeOS handoff contract payload."""
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if not model_version.strip():
        raise ValueError("model_version must not be empty")
    if input_channels <= 0:
        raise ValueError("input_channels must be > 0")
    if num_classes <= 1:
        raise ValueError("num_classes must be > 1")
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match num_classes")

    return {
        "schema_version": EDGEOS_MODEL_MANIFEST_SCHEMA_VERSION,
        "model_id": model_id,
        "version": model_version,
        "backend": "pytorch_state_dict",
        "sha256": sha256_file(model_file),
        "size_bytes": int(model_file.stat().st_size),
        "input": {
            "width": "variable",
            "dtype": "float32",
            "description": (
                "Variable-length float vector from WhirDetective inference adapter "
                "(flattened [channels, features_per_channel])."
            ),
            "channels": input_channels,
        },
        "output": {
            "width": "variable",
            "dtype": "float32",
            "description": "Class probability vector ordered by class_names.",
        },
        "classification": {
            "num_classes": num_classes,
            "class_names": list(class_names),
        },
        "calibration": {
            "temperature": temperature,
            "abstention_threshold": abstention_threshold,
        },
        "metadata": {
            "dataset_fingerprint": dataset_fingerprint,
        },
        "deployment": {
            "load_command": "ML_ModelManager COMMAND=0",
            "info_command": "ML_ModelManager COMMAND=3",
            "ota_flow": "BEGIN -> CHUNK* -> COMMIT",
            "rollback_command": "ML_OTAUpdate COMMAND=4",
            "rollback_policy": "automatic rollback on post-commit KPI/health failure",
        },
        "runtime_budget": {
            "latency_source": "release_gate",
            "metric": "p95_inference_ms",
        },
    }


def validate_edgeos_model_manifest(payload: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Validate required EdgeOS handoff fields for schema/versioning contract."""
    required_top_level = (
        "schema_version",
        "model_id",
        "version",
        "backend",
        "sha256",
        "size_bytes",
        "input",
        "output",
        "deployment",
        "runtime_budget",
    )
    failures: list[str] = []
    for field_name in required_top_level:
        if field_name not in payload:
            failures.append(f"missing_field:{field_name}")

    if payload.get("schema_version") != EDGEOS_MODEL_MANIFEST_SCHEMA_VERSION:
        failures.append(
            "invalid_schema_version:"
            f"{payload.get('schema_version')}!={EDGEOS_MODEL_MANIFEST_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("size_bytes"), int) or int(payload.get("size_bytes", 0)) <= 0:
        failures.append("invalid_size_bytes")
    if not isinstance(payload.get("sha256"), str) or len(str(payload.get("sha256"))) != 64:
        failures.append("invalid_sha256")
    if not isinstance(payload.get("deployment"), dict):
        failures.append("deployment_must_be_object")
    else:
        deployment = payload["deployment"]
        if "rollback_policy" not in deployment:
            failures.append("missing_field:deployment.rollback_policy")

    return (len(failures) == 0, tuple(failures))


def write_edgeos_model_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write EdgeOS handoff contract artifact."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
