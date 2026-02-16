"""Tests for EdgeOS runtime contract manifest helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from whirdetective.export.edgeos_contract import (
    EDGEOS_MODEL_MANIFEST_SCHEMA_VERSION,
    build_edgeos_model_manifest,
    validate_edgeos_model_manifest,
)


def test_build_edgeos_model_manifest_and_validate(tmp_path: Path) -> None:
    model_path = tmp_path / "model_state_dict.pt"
    torch.save({"fc.weight": torch.ones((2, 3))}, model_path)

    payload = build_edgeos_model_manifest(
        model_id="whirdetective.cwru.baseline",
        model_version="0.1.0+abcd1234",
        model_file=model_path,
        input_channels=3,
        num_classes=2,
        class_names=("healthy", "inner_race"),
        temperature=1.0,
        abstention_threshold=0.7,
        dataset_fingerprint="fingerprint",
    )
    assert payload["schema_version"] == EDGEOS_MODEL_MANIFEST_SCHEMA_VERSION
    ok, failures = validate_edgeos_model_manifest(payload)
    assert ok is True
    assert failures == ()


def test_validate_edgeos_model_manifest_rejects_missing_fields() -> None:
    ok, failures = validate_edgeos_model_manifest({"schema_version": "edgeos.model_manifest.v1"})
    assert ok is False
    assert "missing_field:model_id" in failures
