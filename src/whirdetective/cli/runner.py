"""CLI runner for Step 4 baseline training/evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from whirdetective.data import BuiltCanonicalDataset, CwruBuildConfig, build_cwru_canonical_dataset
from whirdetective.evaluation import Step4KpiTargets, evaluate_step4_kpis, model_card_to_jsonable
from whirdetective.ml import ProjectionPolicy, SensorSetProjector
from whirdetective.training import TrainerConfig, run_baseline_workflow


@dataclass(frozen=True, slots=True)
class Step4CliArtifacts:
    """Paths and pass/fail status produced by one Step 4 CLI execution."""

    model_card_path: Path
    kpi_report_path: Path
    run_report_path: Path
    kpi_passed: bool


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for Step 4 workflow execution."""
    parser = argparse.ArgumentParser(
        prog="whirdetective-step4",
        description=(
            "Run Step 4 baseline workflow on local CWRU data and emit model-card + KPI report."
        ),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root path used to resolve relative inputs/outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/cwru"),
        help="Path to CWRU raw dataset folder (expects .mat files).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/step4"),
        help="Directory for model_card.json, kpi_report.json, and run_report.json.",
    )
    parser.add_argument("--window-size", type=int, default=256, help="Sliding window size.")
    parser.add_argument("--step-size", type=int, default=128, help="Sliding window stride.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional max number of files.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--split-seed", type=int, default=42, help="Split RNG seed.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device identifier.")
    parser.add_argument("--trainer-seed", type=int, default=7, help="Model/training RNG seed.")
    parser.add_argument(
        "--abstention-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for abstention metric evaluation.",
    )
    parser.add_argument("--kpi-min-accuracy", type=float, default=0.80)
    parser.add_argument("--kpi-min-macro-recall", type=float, default=0.75)
    parser.add_argument("--kpi-max-ece", type=float, default=0.20)
    parser.add_argument("--kpi-min-coverage", type=float, default=0.70)
    parser.add_argument("--kpi-min-selective-accuracy", type=float, default=0.80)
    parser.add_argument(
        "--fail-on-kpi",
        action="store_true",
        help="Return exit code 1 if KPI gate fails.",
    )
    return parser


def run_step4_from_args(args: argparse.Namespace) -> Step4CliArtifacts:
    """Execute Step 4 workflow and persist output artifacts."""
    workspace_root = args.workspace_root.resolve()
    dataset_root = _resolve_path(workspace_root, args.dataset_root)
    output_dir = _resolve_path(workspace_root, args.output_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    build_config = CwruBuildConfig(
        root_dir=dataset_root,
        window_size=args.window_size,
        step_size=args.step_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        max_files=args.max_files,
    )
    projector = SensorSetProjector(policy=ProjectionPolicy())
    built_dataset = build_cwru_canonical_dataset(config=build_config, projector=projector)
    _assert_split_class_coverage(built_dataset=built_dataset, min_classes=2)

    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.trainer_seed,
    )
    workflow_result = run_baseline_workflow(
        built_dataset=built_dataset,
        trainer_config=trainer_config,
        abstention_threshold=args.abstention_threshold,
    )

    kpi_targets = Step4KpiTargets(
        min_accuracy=args.kpi_min_accuracy,
        min_macro_recall=args.kpi_min_macro_recall,
        max_expected_calibration_error=args.kpi_max_ece,
        min_coverage=args.kpi_min_coverage,
        min_selective_accuracy=args.kpi_min_selective_accuracy,
    )
    kpi_evaluation = evaluate_step4_kpis(workflow_result.model_card, targets=kpi_targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_card_path = output_dir / "model_card.json"
    kpi_report_path = output_dir / "kpi_report.json"
    run_report_path = output_dir / "run_report.json"

    _write_json(model_card_path, model_card_to_jsonable(workflow_result.model_card))
    _write_json(
        kpi_report_path,
        {
            "targets": asdict(kpi_targets),
            "evaluation": asdict(kpi_evaluation),
        },
    )
    _write_json(
        run_report_path,
        {
            "workspace_root": str(workspace_root),
            "dataset_root": str(dataset_root),
            "dataset_fingerprint": built_dataset.fingerprint,
            "source_file_count": len(built_dataset.source_files),
            "num_samples": len(built_dataset.samples),
            "split_sizes": {
                "train": len(built_dataset.split.train_indices),
                "val": len(built_dataset.split.val_indices),
                "test": len(built_dataset.split.test_indices),
            },
            "trainer_config": asdict(trainer_config),
            "training_history": {
                "train_losses": list(workflow_result.history.train_losses),
                "val_losses": list(workflow_result.history.val_losses),
            },
            "temperature": asdict(workflow_result.temperature),
            "abstention_threshold": float(args.abstention_threshold),
            "kpi_passed": kpi_evaluation.passed,
        },
    )

    return Step4CliArtifacts(
        model_card_path=model_card_path,
        kpi_report_path=kpi_report_path,
        run_report_path=run_report_path,
        kpi_passed=kpi_evaluation.passed,
    )


def _assert_split_class_coverage(
    *,
    built_dataset: BuiltCanonicalDataset,
    min_classes: int,
) -> None:
    train_labels = {built_dataset.samples[idx].label.value for idx in built_dataset.split.train_indices}
    if len(train_labels) < min_classes:
        raise ValueError(
            f"train split has only {len(train_labels)} class(es): {sorted(train_labels)}. "
            "Increase data diversity (or max-files) so the training split covers at least 2 classes."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifacts = run_step4_from_args(args)
    except Exception as exc:
        print(f"[ERROR] Step 4 runner failed: {exc}", file=sys.stderr)
        return 2

    print(f"model_card: {artifacts.model_card_path}")
    print(f"kpi_report: {artifacts.kpi_report_path}")
    print(f"run_report: {artifacts.run_report_path}")
    print(f"kpi_passed: {artifacts.kpi_passed}")
    if args.fail_on_kpi and not artifacts.kpi_passed:
        print("[ERROR] KPI gate failed and --fail-on-kpi is set.", file=sys.stderr)
        return 1
    return 0


def _resolve_path(base_dir: Path, path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    return (base_dir / path_value).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
