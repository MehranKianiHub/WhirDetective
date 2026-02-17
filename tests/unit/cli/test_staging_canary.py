"""Tests for staging canary preparation CLI."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import torch

from whirdetective.cli import staging_canary
from whirdetective.export.manifest import sha256_file
from whirdetective.export.model_package import save_edge_model_blob_package, save_edge_model_package


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_freeze_bundle(root: Path) -> Path:
    freeze_dir = root / "artifacts" / "pilot" / "freeze" / "unit-freeze"
    package = freeze_dir / "diagnosis" / "cwru"
    save_edge_model_package(
        output_dir=package,
        model_state_dict={"linear.weight": torch.ones((2, 2), dtype=torch.float32)},
        model_name="BaselineBearingCNN",
        input_channels=6,
        num_classes=2,
        class_names=("healthy", "fault"),
        temperature=1.0,
        abstention_threshold=0.7,
        dataset_fingerprint="fp-unit",
        model_id="whirdetective.cwru.baseline",
        model_version="0.1.0+unit",
    )
    for file_name in ("kpi_report.json", "release_gate.json", "run_report.json", "model_card.json"):
        _write_json(package / file_name, {"evaluation": {"passed": True}})

    _write_json(freeze_dir / "pilot_release_signoff.json", {"evaluation": {"passed": True}})

    files: list[dict[str, object]] = []
    for file_path in sorted(freeze_dir.rglob("*")):
        if not file_path.is_file() or file_path.name == "freeze_manifest.json":
            continue
        files.append(
            {
                "path": str(file_path.relative_to(freeze_dir)),
                "sha256": sha256_file(file_path),
                "size_bytes": int(file_path.stat().st_size),
            }
        )
    _write_json(
        freeze_dir / "freeze_manifest.json",
        {
            "manifest_type": "pilot_runtime_freeze",
            "manifest_version": "1.0",
            "files": files,
        },
    )
    return freeze_dir


def test_main_pass_creates_canary_outputs(tmp_path: Path, monkeypatch: object) -> None:
    freeze_dir = _seed_freeze_bundle(tmp_path)
    forte_test = tmp_path / "edgeos" / "build" / "posix-debug" / "4diacFORTE" / "tests" / "forte_test"
    forte_test.parent.mkdir(parents=True, exist_ok=True)
    forte_test.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def _fake_run(command, capture_output, text, check):
        del command, capture_output, text, check
        return CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(staging_canary.subprocess, "run", _fake_run)

    exit_code = staging_canary.main(
        [
            "--workspace-root",
            str(tmp_path),
            "--freeze-dir",
            str(freeze_dir),
            "--edgeos-root",
            str(tmp_path / "edgeos"),
            "--track",
            "cwru",
            "--allow-unsupported-backend",
        ]
    )
    assert exit_code == 0

    output_dir = freeze_dir / "canary" / "cwru"
    report = json.loads((output_dir / "staging_canary_report.json").read_text(encoding="utf-8"))
    assert report["evaluation"]["passed"] is True
    backend_gate = next(g for g in report["gates"] if g["name"] == "edgeos_backend_compatibility_gate_passed")
    assert backend_gate["details"]["override_used"] is True
    assert (output_dir / "EDGEOS_STAGING_CANARY_COMMANDS.md").exists()


def test_main_fails_when_freeze_signoff_failed(tmp_path: Path) -> None:
    freeze_dir = _seed_freeze_bundle(tmp_path)
    _write_json(freeze_dir / "pilot_release_signoff.json", {"evaluation": {"passed": False}})

    exit_code = staging_canary.main(
        [
            "--workspace-root",
            str(tmp_path),
            "--freeze-dir",
            str(freeze_dir),
            "--track",
            "cwru",
            "--skip-runtime-tests",
        ]
    )
    assert exit_code == 1


def test_main_strict_passes_with_tflite_package_and_no_override(
    tmp_path: Path, monkeypatch: object
) -> None:
    freeze_dir = _seed_freeze_bundle(tmp_path)
    source_package = freeze_dir / "diagnosis" / "cwru"
    tflite_source = tmp_path / "candidate.tflite"
    tflite_source.write_bytes(b"TFL3-PLACEHOLDER")
    deployable_package = tmp_path / "deployable-cwru"
    save_edge_model_blob_package(
        output_dir=deployable_package,
        model_blob_path=tflite_source,
        model_artifact_name="model.tflite",
        backend="tflite",
        model_name="whirdetective.cwru.baseline",
        input_channels=6,
        num_classes=2,
        class_names=("healthy", "fault"),
        temperature=1.0,
        abstention_threshold=0.7,
        dataset_fingerprint="fp-unit",
        model_id="whirdetective.cwru.baseline",
        model_version="0.1.0+tflite",
    )
    for evidence_name in ("kpi_report.json", "release_gate.json", "run_report.json", "model_card.json"):
        _write_json(deployable_package / evidence_name, json.loads((source_package / evidence_name).read_text(encoding="utf-8")))

    forte_test = tmp_path / "edgeos" / "build" / "posix-debug" / "4diacFORTE" / "tests" / "forte_test"
    forte_test.parent.mkdir(parents=True, exist_ok=True)
    forte_test.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def _fake_run(command, capture_output, text, check):
        del command, capture_output, text, check
        return CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(staging_canary.subprocess, "run", _fake_run)
    exit_code = staging_canary.main(
        [
            "--workspace-root",
            str(tmp_path),
            "--freeze-dir",
            str(freeze_dir),
            "--edgeos-root",
            str(tmp_path / "edgeos"),
            "--track",
            "cwru",
            "--package-dir",
            str(deployable_package),
            "--output-dir",
            str(freeze_dir / "canary" / "cwru-tflite"),
        ]
    )
    assert exit_code == 0
    report = json.loads((freeze_dir / "canary" / "cwru-tflite" / "staging_canary_report.json").read_text(encoding="utf-8"))
    backend_gate = next(g for g in report["gates"] if g["name"] == "edgeos_backend_compatibility_gate_passed")
    assert backend_gate["passed"] is True
    assert backend_gate["details"]["override_used"] is False
