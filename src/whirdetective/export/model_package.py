"""Edge handoff package writer for trained WhirDetective models."""

from __future__ import annotations

import json
import os
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
    model_state_path = output_dir / "model_state_dict.pt"
    inference_config_path = output_dir / "inference_config.json"
    calibration_path = output_dir / "calibration.json"
    edgeos_manifest_path = output_dir / "edgeos_model_manifest.json"
    manifest_path = output_dir / "manifest.json"
    signature_path = output_dir / "manifest.sig"

    torch.save(dict(model_state_dict), model_state_path)

    inference_config = {
        "model_name": model_name,
        "framework": "pytorch",
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
        "framework": "pytorch",
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
