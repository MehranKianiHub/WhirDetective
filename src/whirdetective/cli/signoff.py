"""Required benchmark sign-off runner for Step 4."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from whirdetective.cli import runner
from whirdetective.prognostics import (
    XjtuPrognosticsTargets,
    XjtuRunConfig,
    run_xjtu_prognostics_baseline,
)


def build_parser() -> argparse.ArgumentParser:
    """Create parser for full Step 4 required-scope sign-off execution."""
    parser = argparse.ArgumentParser(
        prog="whirdetective-signoff",
        description=(
            "Run required Step 4 benchmark sign-off: diagnosis (CWRU + Paderborn) "
            "and prognostics (XJTU-SY baseline)."
        ),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/step4-required-signoff"),
    )

    parser.add_argument("--cwru-root", type=Path, default=Path("data/raw/cwru"))
    parser.add_argument("--cwru-max-files", type=int, default=40)
    parser.add_argument("--cwru-epochs", type=int, default=3)

    parser.add_argument("--paderborn-root", type=Path, default=Path("data/raw/paderborn"))
    parser.add_argument("--paderborn-max-archives", type=int, default=20)
    parser.add_argument("--paderborn-max-entries-per-archive", type=int, default=6)
    parser.add_argument("--paderborn-epochs", type=int, default=2)

    parser.add_argument("--xjtu-root", type=Path, default=Path("data/raw/xjtu_sy/Data"))
    parser.add_argument("--xjtu-max-parts", type=int, default=6)
    parser.add_argument("--xjtu-sample-stride", type=int, default=25)
    parser.add_argument("--xjtu-max-entries-per-bearing", type=int, default=120)
    parser.add_argument("--xjtu-csv-max-rows", type=int, default=4096)

    parser.add_argument(
        "--fail-on-signoff",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for required benchmark sign-off workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    output_dir = _resolve_path(workspace_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cwru_artifacts = _run_diagnosis_track(
            workspace_root=workspace_root,
            output_dir=output_dir / "cwru_diagnosis",
            dataset_name="cwru",
            dataset_root=args.cwru_root,
            epochs=args.cwru_epochs,
            max_files=args.cwru_max_files,
            max_archives=None,
            max_entries_per_archive=8,
            min_distinct_classes=4,
            min_train_classes=4,
            min_eval_classes=4,
            kpi_min_accuracy=0.80,
            kpi_min_macro_recall=0.75,
            kpi_min_coverage=0.70,
            kpi_min_selective_accuracy=0.80,
        )
        paderborn_artifacts = _run_diagnosis_track(
            workspace_root=workspace_root,
            output_dir=output_dir / "paderborn_diagnosis",
            dataset_name="paderborn",
            dataset_root=args.paderborn_root,
            epochs=args.paderborn_epochs,
            max_files=None,
            max_archives=args.paderborn_max_archives,
            max_entries_per_archive=args.paderborn_max_entries_per_archive,
            min_distinct_classes=2,
            min_train_classes=2,
            min_eval_classes=2,
            kpi_min_accuracy=0.70,
            kpi_min_macro_recall=0.65,
            kpi_min_coverage=0.60,
            kpi_min_selective_accuracy=0.78,
        )
        xjtu_result = run_xjtu_prognostics_baseline(
            config=XjtuRunConfig(
                root_dir=_resolve_path(workspace_root, args.xjtu_root),
                max_parts=args.xjtu_max_parts,
                sample_stride=args.xjtu_sample_stride,
                max_entries_per_bearing=args.xjtu_max_entries_per_bearing,
                csv_max_rows=args.xjtu_csv_max_rows,
            ),
            targets=XjtuPrognosticsTargets(),
        )
    except Exception as exc:
        print(f"[ERROR] Step 4 required sign-off failed: {exc}", file=sys.stderr)
        return 2

    xjtu_report_path = output_dir / "xjtu_prognostics_report.json"
    _write_json(
        xjtu_report_path,
        {
            "targets": asdict(XjtuPrognosticsTargets()),
            "evaluation": asdict(xjtu_result.evaluation),
            "selected_lambda": xjtu_result.selected_lambda,
            "val_mae": xjtu_result.val_mae,
            "part_files": list(xjtu_result.part_files),
            "split_sizes": dict(xjtu_result.split_sizes),
        },
    )

    passed = (
        cwru_artifacts.kpi_passed
        and paderborn_artifacts.kpi_passed
        and xjtu_result.evaluation.passed
    )
    signoff_report_path = output_dir / "step4_required_signoff.json"
    _write_json(
        signoff_report_path,
        {
            "passed": passed,
            "tracks": {
                "cwru_diagnosis": {
                    "kpi_passed": cwru_artifacts.kpi_passed,
                    "model_card_path": str(cwru_artifacts.model_card_path),
                    "kpi_report_path": str(cwru_artifacts.kpi_report_path),
                    "run_report_path": str(cwru_artifacts.run_report_path),
                    "release_gate_path": str(cwru_artifacts.release_gate_path),
                },
                "paderborn_diagnosis": {
                    "kpi_passed": paderborn_artifacts.kpi_passed,
                    "model_card_path": str(paderborn_artifacts.model_card_path),
                    "kpi_report_path": str(paderborn_artifacts.kpi_report_path),
                    "run_report_path": str(paderborn_artifacts.run_report_path),
                    "release_gate_path": str(paderborn_artifacts.release_gate_path),
                },
                "xjtu_prognostics": {
                    "kpi_passed": xjtu_result.evaluation.passed,
                    "report_path": str(xjtu_report_path),
                },
            },
        },
    )

    print(f"step4_required_signoff: {signoff_report_path}")
    print(f"passed: {passed}")
    if args.fail_on_signoff and not passed:
        return 1
    return 0


def _run_diagnosis_track(
    *,
    workspace_root: Path,
    output_dir: Path,
    dataset_name: str,
    dataset_root: Path,
    epochs: int,
    max_files: int | None,
    max_archives: int | None,
    max_entries_per_archive: int,
    min_distinct_classes: int,
    min_train_classes: int,
    min_eval_classes: int,
    kpi_min_accuracy: float,
    kpi_min_macro_recall: float,
    kpi_min_coverage: float,
    kpi_min_selective_accuracy: float,
) -> runner.Step4CliArtifacts:
    parsed_args = runner.build_parser().parse_args(
        [
            "--workspace-root",
            str(workspace_root),
            "--dataset-name",
            dataset_name,
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(epochs),
            "--window-size",
            "512",
            "--step-size",
            "512",
            "--split-min-distinct-classes",
            str(min_distinct_classes),
            "--min-train-classes",
            str(min_train_classes),
            "--min-eval-classes",
            str(min_eval_classes),
            "--split-search-attempts",
            "4096",
            "--max-entries-per-archive",
            str(max_entries_per_archive),
            "--kpi-min-accuracy",
            str(kpi_min_accuracy),
            "--kpi-min-macro-recall",
            str(kpi_min_macro_recall),
            "--kpi-min-coverage",
            str(kpi_min_coverage),
            "--kpi-min-selective-accuracy",
            str(kpi_min_selective_accuracy),
        ]
        + ([] if max_files is None else ["--max-files", str(max_files)])
        + ([] if max_archives is None else ["--max-archives", str(max_archives)])
    )
    return runner.run_step4_from_args(parsed_args)


def _resolve_path(base_dir: Path, path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    return (base_dir / path_value).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
