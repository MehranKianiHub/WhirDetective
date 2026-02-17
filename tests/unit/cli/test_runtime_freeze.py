"""Tests for controlled runtime qualification + freeze CLI."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from whirdetective.cli import runtime_freeze


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_pilot_artifacts(workspace_root: Path, *, preflight_cuda: bool = True) -> None:
    pilot_root = workspace_root / "artifacts" / "pilot"

    cwru = pilot_root / "cwru-full-gpu-e12-v3"
    paderborn = pilot_root / "paderborn-full-gpu-e8-v2"
    for package_dir in (cwru, paderborn):
        package_dir.mkdir(parents=True, exist_ok=True)
        _write_json(package_dir / "kpi_report.json", {"evaluation": {"passed": True}})
        _write_json(package_dir / "release_gate.json", {"evaluation": {"passed": True}})
        (package_dir / "model_state_dict.pt").write_bytes(b"model")
        _write_json(package_dir / "manifest.json", {"files": []})

    _write_json(
        pilot_root / "xjtu-full" / "xjtu_prognostics_report.json",
        {"evaluation": {"passed": True}},
    )
    _write_json(
        pilot_root / "preflight" / "preflight_report.json",
        {
            "datasets": {
                "cwru": {"exists": True},
                "paderborn": {"exists": True},
                "xjtu_sy": {"exists": True},
            },
            "torch": {"cuda_available": preflight_cuda},
        },
    )
    _write_json(
        pilot_root / "pilot_progress.json",
        {
            "status": {
                "preflight": True,
                "cwru": True,
                "paderborn": True,
                "xjtu": True,
            }
        },
    )


def test_main_creates_freeze_bundle_when_all_gates_pass(tmp_path: Path, monkeypatch: object) -> None:
    workspace_root = tmp_path
    _seed_pilot_artifacts(workspace_root)

    forte_test = workspace_root / "edgeos" / "build" / "posix-debug" / "4diacFORTE" / "tests" / "forte_test"
    forte_test.parent.mkdir(parents=True, exist_ok=True)
    forte_test.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    def _fake_subprocess_run(command, capture_output, text, check):
        del command, capture_output, text, check
        return CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(runtime_freeze.subprocess, "run", _fake_subprocess_run)

    exit_code = runtime_freeze.main(
        [
            "--workspace-root",
            str(workspace_root),
            "--edgeos-root",
            str(workspace_root / "edgeos"),
            "--freeze-label",
            "unit-freeze-pass",
        ]
    )
    assert exit_code == 0

    freeze_dir = workspace_root / "artifacts" / "pilot" / "freeze" / "unit-freeze-pass"
    payload = json.loads((freeze_dir / "pilot_release_signoff.json").read_text(encoding="utf-8"))
    assert payload["evaluation"]["passed"] is True
    assert (freeze_dir / "PILOT_RELEASE_SIGNOFF.md").exists()
    assert (freeze_dir / "freeze_manifest.json").exists()
    assert (freeze_dir / "runtime_tests" / "ML_InferenceTests.log").exists()


def test_main_returns_one_when_preflight_gate_fails(tmp_path: Path) -> None:
    workspace_root = tmp_path
    _seed_pilot_artifacts(workspace_root, preflight_cuda=False)

    exit_code = runtime_freeze.main(
        [
            "--workspace-root",
            str(workspace_root),
            "--freeze-label",
            "unit-freeze-fail",
            "--skip-runtime-tests",
        ]
    )
    assert exit_code == 1

    freeze_dir = workspace_root / "artifacts" / "pilot" / "freeze" / "unit-freeze-fail"
    payload = json.loads((freeze_dir / "pilot_release_signoff.json").read_text(encoding="utf-8"))
    assert payload["evaluation"]["passed"] is False
    assert "preflight_gate_passed" in payload["evaluation"]["failed_checks"]
