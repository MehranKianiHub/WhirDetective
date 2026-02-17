"""Tests for Step 5 model artifact package emission."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from whirdetective.export.edgeos_contract import validate_edgeos_model_manifest
from whirdetective.export.manifest import sha256_file
from whirdetective.export.model_package import save_edge_model_blob_package, save_edge_model_package
from whirdetective.export.verification import verify_edge_model_package


def test_save_edge_model_package_writes_artifacts_and_manifest(tmp_path: Path) -> None:
    paths = save_edge_model_package(
        output_dir=tmp_path,
        model_state_dict={
            "fc.weight": torch.ones((2, 3), dtype=torch.float32),
            "fc.bias": torch.zeros((2,), dtype=torch.float32),
        },
        model_name="BaselineBearingCNN",
        input_channels=3,
        num_classes=2,
        class_names=("healthy", "inner_race"),
        temperature=0.9,
        abstention_threshold=0.7,
        dataset_fingerprint="fp-001",
        extra_metadata={"kpi_passed": True},
    )

    assert paths.model_state_path.exists()
    assert paths.inference_config_path.exists()
    assert paths.calibration_path.exists()
    assert paths.edgeos_manifest_path.exists()
    assert paths.manifest_path.exists()
    assert paths.signature_path is None

    loaded_state = torch.load(paths.model_state_path, map_location="cpu")
    assert "fc.weight" in loaded_state
    assert tuple(loaded_state["fc.weight"].shape) == (2, 3)

    inference_config = json.loads(paths.inference_config_path.read_text(encoding="utf-8"))
    assert inference_config["num_classes"] == 2
    assert inference_config["class_names"] == ["healthy", "inner_race"]

    calibration = json.loads(paths.calibration_path.read_text(encoding="utf-8"))
    assert calibration["temperature"] == 0.9
    assert calibration["abstention_threshold"] == 0.7

    edgeos_manifest = json.loads(paths.edgeos_manifest_path.read_text(encoding="utf-8"))
    edgeos_ok, edgeos_failures = validate_edgeos_model_manifest(edgeos_manifest)
    assert edgeos_ok is True
    assert edgeos_failures == ()
    assert edgeos_manifest["model_id"] == "whirdetective.bearing.baseline"
    assert edgeos_manifest["model_file"] == "model_state_dict.pt"

    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "1.0"
    manifest_files = {entry["path"]: entry for entry in manifest["files"]}
    for filename in (
        "model_state_dict.pt",
        "inference_config.json",
        "calibration.json",
        "edgeos_model_manifest.json",
    ):
        assert filename in manifest_files

    for relative_name, entry in manifest_files.items():
        file_path = tmp_path / relative_name
        assert entry["sha256"] == sha256_file(file_path)
        assert entry["size_bytes"] == int(file_path.stat().st_size)

    verification = verify_edge_model_package(package_dir=tmp_path)
    assert verification.ok is True
    assert verification.signature_verified is None


def test_save_edge_model_package_emits_and_verifies_signature(tmp_path: Path) -> None:
    signing_key = "local-test-key"
    paths = save_edge_model_package(
        output_dir=tmp_path,
        model_state_dict={
            "fc.weight": torch.ones((2, 3), dtype=torch.float32),
            "fc.bias": torch.zeros((2,), dtype=torch.float32),
        },
        model_name="BaselineBearingCNN",
        input_channels=3,
        num_classes=2,
        class_names=("healthy", "inner_race"),
        temperature=0.9,
        abstention_threshold=0.7,
        dataset_fingerprint="fp-001",
        signing_key=signing_key,
    )
    assert paths.signature_path is not None
    assert paths.signature_path.exists()

    verification_ok = verify_edge_model_package(package_dir=tmp_path, signing_key=signing_key)
    assert verification_ok.ok is True
    assert verification_ok.signature_verified is True

    verification_bad = verify_edge_model_package(package_dir=tmp_path, signing_key="wrong-key")
    assert verification_bad.ok is False
    assert verification_bad.signature_verified is False


def test_save_edge_model_blob_package_supports_tflite_backend(tmp_path: Path) -> None:
    model_blob = tmp_path / "source_model.tflite"
    model_blob.write_bytes(b"TFL3-placeholder")

    output_dir = tmp_path / "exported"
    paths = save_edge_model_blob_package(
        output_dir=output_dir,
        model_blob_path=model_blob,
        model_artifact_name="model.tflite",
        backend="tflite",
        model_name="BaselineBearingCNN",
        input_channels=6,
        num_classes=2,
        class_names=("healthy", "fault"),
        temperature=1.2,
        abstention_threshold=0.7,
        dataset_fingerprint="fp-002",
    )

    assert paths.model_state_path.name == "model.tflite"
    assert paths.model_state_path.read_bytes() == b"TFL3-placeholder"

    edgeos_manifest = json.loads(paths.edgeos_manifest_path.read_text(encoding="utf-8"))
    edgeos_ok, edgeos_failures = validate_edgeos_model_manifest(edgeos_manifest)
    assert edgeos_ok is True
    assert edgeos_failures == ()
    assert edgeos_manifest["backend"] == "tflite"
    assert edgeos_manifest["model_file"] == "model.tflite"

    verification = verify_edge_model_package(package_dir=output_dir)
    assert verification.ok is True
