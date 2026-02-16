"""Tests for required Step 4 sign-off CLI runner."""

from __future__ import annotations

import json
from pathlib import Path

from whirdetective.cli import signoff
from whirdetective.cli.runner import Step4CliArtifacts
from whirdetective.prognostics import XjtuPrognosticsEvaluation, XjtuPrognosticsResult


def test_signoff_main_writes_consolidated_report(tmp_path: Path, monkeypatch: object) -> None:
    workspace_root = tmp_path
    cwru_dir = workspace_root / "data" / "raw" / "cwru"
    paderborn_dir = workspace_root / "data" / "raw" / "paderborn"
    xjtu_dir = workspace_root / "data" / "raw" / "xjtu_sy" / "Data"
    cwru_dir.mkdir(parents=True)
    paderborn_dir.mkdir(parents=True)
    xjtu_dir.mkdir(parents=True)

    def _fake_run_step4(args):
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model_card.json").write_text("{}", encoding="utf-8")
        (out / "kpi_report.json").write_text("{}", encoding="utf-8")
        (out / "run_report.json").write_text("{}", encoding="utf-8")
        (out / "release_gate.json").write_text("{}", encoding="utf-8")
        return Step4CliArtifacts(
            model_card_path=out / "model_card.json",
            kpi_report_path=out / "kpi_report.json",
            run_report_path=out / "run_report.json",
            release_gate_path=out / "release_gate.json",
            kpi_passed=True,
            release_gate_passed=True,
        )

    monkeypatch.setattr(signoff.runner, "run_step4_from_args", _fake_run_step4)
    monkeypatch.setattr(
        signoff,
        "run_xjtu_prognostics_baseline",
        lambda config, targets: XjtuPrognosticsResult(
            evaluation=XjtuPrognosticsEvaluation(
                passed=True,
                failed_checks=(),
                num_bearings=5,
                num_samples=200,
                test_mae=0.2,
                test_spearman=0.95,
                mean_group_spearman=0.9,
            ),
            selected_lambda=1.0,
            val_mae=0.2,
            part_files=(),
            split_sizes={"train": 100, "val": 50, "test": 50},
        ),
    )

    exit_code = signoff.main(
        [
            "--workspace-root",
            str(workspace_root),
            "--output-dir",
            "artifacts/step4-required-signoff-test",
        ]
    )
    assert exit_code == 0

    report_path = workspace_root / "artifacts" / "step4-required-signoff-test" / "step4_required_signoff.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
