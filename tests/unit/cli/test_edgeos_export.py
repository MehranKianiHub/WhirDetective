"""Tests for EdgeOS deployable export CLI."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from whirdetective.cli import edgeos_export
from whirdetective.export.model_package import save_edge_model_package


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_source_package(root: Path) -> Path:
    source_dir = root / "source"
    save_edge_model_package(
        output_dir=source_dir,
        model_state_dict={"fc.weight": torch.ones((2, 2), dtype=torch.float32)},
        model_name="BaselineBearingCNN",
        input_channels=6,
        num_classes=2,
        class_names=("healthy", "fault"),
        temperature=1.0,
        abstention_threshold=0.7,
        dataset_fingerprint="fp-seed",
        model_id="whirdetective.cwru.baseline",
        model_version="0.1.0+seed",
    )
    _write_json(source_dir / "kpi_report.json", {"evaluation": {"passed": True}})
    _write_json(source_dir / "release_gate.json", {"evaluation": {"passed": True}})
    _write_json(source_dir / "run_report.json", {"status": "ok"})
    _write_json(source_dir / "model_card.json", {"model": "baseline"})
    return source_dir


def test_edgeos_export_main_builds_tflite_package(tmp_path: Path) -> None:
    source_dir = _seed_source_package(tmp_path)
    model_blob = tmp_path / "candidate.tflite"
    model_blob.write_bytes(b"TFL3-FAKE")

    output_dir = tmp_path / "exported"
    exit_code = edgeos_export.main(
        [
            "--workspace-root",
            str(tmp_path),
            "--source-package-dir",
            str(source_dir),
            "--model-blob",
            str(model_blob),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    report = json.loads((output_dir / "edgeos_export_report.json").read_text(encoding="utf-8"))
    assert report["evaluation"]["passed"] is True
    assert report["backend"] == "tflite"
    manifest = json.loads((output_dir / "edgeos_model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "tflite"
    assert manifest["model_file"] == "model.tflite"


def test_edgeos_export_fails_when_source_kpi_fails(tmp_path: Path) -> None:
    source_dir = _seed_source_package(tmp_path)
    _write_json(source_dir / "kpi_report.json", {"evaluation": {"passed": False}})
    model_blob = tmp_path / "candidate.tflite"
    model_blob.write_bytes(b"TFL3-FAKE")

    output_dir = tmp_path / "exported-fail"
    exit_code = edgeos_export.main(
        [
            "--workspace-root",
            str(tmp_path),
            "--source-package-dir",
            str(source_dir),
            "--model-blob",
            str(model_blob),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 1
    report = json.loads((output_dir / "edgeos_export_report.json").read_text(encoding="utf-8"))
    assert report["evaluation"]["passed"] is False
    assert "source_kpi_gate_failed" in report["evaluation"]["failed_checks"]
