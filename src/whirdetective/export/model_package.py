"""Edge handoff package writer for trained WhirDetective models."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from whirdetective.export.edgeos_contract import build_edgeos_model_manifest, write_edgeos_model_manifest
from whirdetective.export.manifest import (
    build_manifest_payload,
    compute_hmac_sha256_signature,
    write_manifest,
    write_manifest_signature,
)


@dataclass(frozen=True, slots=True)
class EdgeModelArtifactPaths:
    """Filesystem paths for emitted model package artifacts."""

    model_state_path: Path
    inference_config_path: Path
    calibration_path: Path
    edgeos_manifest_path: Path
    manifest_path: Path
    signature_path: Path | None


def save_edge_model_blob_package(
    *,
    output_dir: Path,
    model_blob_path: Path,
    model_artifact_name: str,
    backend: str,
    model_name: str,
    input_channels: int,
    num_classes: int,
    class_names: tuple[str, ...],
    temperature: float,
    abstention_threshold: float,
    dataset_fingerprint: str,
    model_id: str = "whirdetective.bearing.baseline",
    model_version: str = "0.1.0",
    extra_metadata: dict[str, Any] | None = None,
    signing_key: str | None = None,
    signing_key_env_var: str = "WHIRDETECTIVE_MANIFEST_SIGNING_KEY",
) -> EdgeModelArtifactPaths:
    """Persist a backend-specific model blob package for EdgeOS handoff."""
    if not model_blob_path.exists():
        raise FileNotFoundError(f"model_blob_path does not exist: {model_blob_path}")
    if not model_artifact_name.strip():
        raise ValueError("model_artifact_name must not be empty")
    if "/" in model_artifact_name or "\\" in model_artifact_name:
        raise ValueError("model_artifact_name must be a file name, not a path")
    if not backend.strip():
        raise ValueError("backend must not be empty")
    if backend in {"tflite", "tflite_flatbuffer"} and not model_artifact_name.endswith(".tflite"):
        raise ValueError("tflite backends require model_artifact_name to end with .tflite")
    if input_channels <= 0:
        raise ValueError("input_channels must be > 0")
    if num_classes <= 1:
        raise ValueError("num_classes must be > 1")
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match num_classes")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if abstention_threshold <= 0.0 or abstention_threshold > 1.0:
        raise ValueError("abstention_threshold must be in (0, 1]")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_state_path = output_dir / model_artifact_name
    inference_config_path = output_dir / "inference_config.json"
    calibration_path = output_dir / "calibration.json"
    edgeos_manifest_path = output_dir / "edgeos_model_manifest.json"
    manifest_path = output_dir / "manifest.json"
    signature_path = output_dir / "manifest.sig"

    if model_blob_path.resolve() != model_state_path.resolve():
        shutil.copy2(model_blob_path, model_state_path)

    framework = "tflite" if backend in {"tflite", "tflite_flatbuffer"} else backend
    inference_config = {
        "model_name": model_name,
        "framework": framework,
        "input_channels": input_channels,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "input_shape_semantics": ["batch", "channels", "samples"],
        "dataset_fingerprint": dataset_fingerprint,
    }
    inference_config_path.write_text(
        json.dumps(inference_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    calibration = {
        "temperature": temperature,
        "abstention_threshold": abstention_threshold,
    }
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")

    edgeos_manifest = build_edgeos_model_manifest(
        model_id=model_id,
        model_version=model_version,
        model_file=model_state_path,
        backend=backend,
        input_channels=input_channels,
        num_classes=num_classes,
        class_names=class_names,
        temperature=temperature,
        abstention_threshold=abstention_threshold,
        dataset_fingerprint=dataset_fingerprint,
    )
    write_edgeos_model_manifest(edgeos_manifest_path, edgeos_manifest)

    metadata = {
        "model_name": model_name,
        "dataset_fingerprint": dataset_fingerprint,
        "class_names": list(class_names),
        "framework": framework,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    manifest_payload = build_manifest_payload(
        package_name="whirdetective-edge-package",
        package_root=output_dir,
        files=(model_state_path, inference_config_path, calibration_path, edgeos_manifest_path),
        metadata=metadata,
    )
    write_manifest(manifest_path, manifest_payload)

    resolved_signing_key = signing_key if signing_key is not None else os.environ.get(signing_key_env_var)
    emitted_signature_path: Path | None = None
    if resolved_signing_key:
        signature = compute_hmac_sha256_signature(payload=manifest_payload, key=resolved_signing_key)
        write_manifest_signature(signature_path, signature=signature)
        emitted_signature_path = signature_path

    return EdgeModelArtifactPaths(
        model_state_path=model_state_path,
        inference_config_path=inference_config_path,
        calibration_path=calibration_path,
        edgeos_manifest_path=edgeos_manifest_path,
        manifest_path=manifest_path,
        signature_path=emitted_signature_path,
    )


def save_edge_model_package(
    *,
    output_dir: Path,
    model_state_dict: Mapping[str, torch.Tensor],
    model_name: str,
    input_channels: int,
    num_classes: int,
    class_names: tuple[str, ...],
    temperature: float,
    abstention_threshold: float,
    dataset_fingerprint: str,
    model_id: str = "whirdetective.bearing.baseline",
    model_version: str = "0.1.0",
    extra_metadata: dict[str, Any] | None = None,
    signing_key: str | None = None,
    signing_key_env_var: str = "WHIRDETECTIVE_MANIFEST_SIGNING_KEY",
) -> EdgeModelArtifactPaths:
    """Persist trained model artifacts and integrity manifest for EdgeOS handoff."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_state_path = output_dir / "model_state_dict.pt"
    torch.save(dict(model_state_dict), model_state_path)
    return save_edge_model_blob_package(
        output_dir=output_dir,
        model_blob_path=model_state_path,
        model_artifact_name="model_state_dict.pt",
        backend="pytorch_state_dict",
        model_name=model_name,
        input_channels=input_channels,
        num_classes=num_classes,
        class_names=class_names,
        temperature=temperature,
        abstention_threshold=abstention_threshold,
        dataset_fingerprint=dataset_fingerprint,
        model_id=model_id,
        model_version=model_version,
        extra_metadata=extra_metadata,
        signing_key=signing_key,
        signing_key_env_var=signing_key_env_var,
    )
