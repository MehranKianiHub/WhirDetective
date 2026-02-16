"""CLI runner for Step 4 baseline training/evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from whirdetective.data import (
    BuiltCanonicalDataset,
    CwruBuildConfig,
    PaderbornBuildConfig,
    build_cwru_canonical_dataset,
    build_paderborn_canonical_dataset,
)
from whirdetective.evaluation import Step4KpiTargets, evaluate_step4_kpis, model_card_to_jsonable
from whirdetective.export import (
    ReleaseGateTargets,
    evaluate_release_gate,
    save_edge_model_package,
    validate_edgeos_model_manifest,
    verify_edge_model_package,
)
from whirdetective.ml import BaselineBearingCNN
from whirdetective.ml import ProjectionPolicy, SensorSetProjector
from whirdetective.training import TrainerConfig, run_baseline_workflow


@dataclass(frozen=True, slots=True)
class Step4CliArtifacts:
    """Paths and pass/fail status produced by one Step 4 CLI execution."""

    model_card_path: Path
    kpi_report_path: Path
    run_report_path: Path
    release_gate_path: Path
    kpi_passed: bool
    release_gate_passed: bool


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for Step 4 workflow execution."""
    parser = argparse.ArgumentParser(
        prog="whirdetective-runner",
        description=(
            "Run Step 4 baseline workflow on local diagnosis data and emit model-card + KPI report."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        choices=("cwru", "paderborn"),
        default="cwru",
        help="Diagnosis dataset adapter to use.",
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
        help="Path to selected raw dataset folder.",
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
    parser.add_argument(
        "--max-archives",
        type=int,
        default=None,
        help="Optional max number of Paderborn archives.",
    )
    parser.add_argument(
        "--max-entries-per-archive",
        type=int,
        default=8,
        help="Optional max number of MAT entries consumed per Paderborn archive.",
    )
    parser.add_argument(
        "--paderborn-collapse-fault-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse non-healthy Paderborn labels into one combined fault class.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--split-seed", type=int, default=42, help="Split RNG seed.")
    parser.add_argument(
        "--split-min-distinct-classes",
        type=int,
        default=2,
        help="Minimum distinct classes required in each split during grouped split search.",
    )
    parser.add_argument(
        "--split-search-attempts",
        type=int,
        default=1024,
        help="Number of candidate grouped split assignments to evaluate.",
    )
    parser.add_argument(
        "--split-require-all-classes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require every observed class label to appear in train/val/test. "
            "Defaults to enabled for CWRU and disabled for Paderborn."
        ),
    )
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use weighted class-balanced sampling in training mini-batches.",
    )
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
        "--min-train-classes",
        type=int,
        default=2,
        help="Minimum distinct classes required in the train split.",
    )
    parser.add_argument(
        "--min-eval-classes",
        type=int,
        default=2,
        help="Minimum distinct classes required in both val and test splits.",
    )
    parser.add_argument(
        "--fail-on-kpi",
        action="store_true",
        help="Return exit code 1 if KPI gate fails.",
    )
    parser.add_argument(
        "--fail-on-release-gate",
        action="store_true",
        help="Return exit code 1 if release gate fails.",
    )
    parser.add_argument(
        "--verify-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify emitted package integrity against manifest.",
    )
    parser.add_argument(
        "--require-signature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail run when manifest signature is missing or invalid.",
    )
    parser.add_argument(
        "--manifest-signing-key-env",
        type=str,
        default="WHIRDETECTIVE_MANIFEST_SIGNING_KEY",
        help="Environment variable name used to read optional manifest signing key.",
    )
    parser.add_argument(
        "--release-max-model-size-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="Maximum allowed model_state_dict.pt size for rollout gate.",
    )
    parser.add_argument(
        "--release-max-parameter-count",
        type=int,
        default=1_000_000,
        help="Maximum allowed model parameter count for rollout gate.",
    )
    parser.add_argument(
        "--release-max-p95-inference-ms",
        type=float,
        default=20.0,
        help="Maximum allowed p95 inference latency in milliseconds for rollout gate.",
    )
    parser.add_argument(
        "--release-require-artifact-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require artifact verification success for rollout gate pass.",
    )
    return parser


def run_step4_from_args(args: argparse.Namespace) -> Step4CliArtifacts:
    """Execute Step 4 workflow and persist output artifacts."""
    workspace_root = args.workspace_root.resolve()
    dataset_root = _resolve_path(workspace_root, args.dataset_root)
    output_dir = _resolve_path(workspace_root, args.output_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    projector = SensorSetProjector(policy=ProjectionPolicy())
    built_dataset = _build_dataset_from_args(
        args=args,
        dataset_root=dataset_root,
        projector=projector,
    )
    class_coverage = _assert_split_class_coverage(
        built_dataset=built_dataset,
        min_train_classes=args.min_train_classes,
        min_eval_classes=args.min_eval_classes,
        require_all_classes=bool(_resolve_split_require_all_classes(args)),
    )

    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_balanced_sampling=bool(args.balanced_sampling),
        device=args.device,
        seed=args.trainer_seed,
    )
    workflow_result = run_baseline_workflow(
        built_dataset=built_dataset,
        trainer_config=trainer_config,
        abstention_threshold=args.abstention_threshold,
        abstention_min_coverage_target=args.kpi_min_coverage,
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
    release_gate_path = output_dir / "release_gate.json"
    run_report_path = output_dir / "run_report.json"

    _write_json(model_card_path, model_card_to_jsonable(workflow_result.model_card))
    _write_json(
        kpi_report_path,
        {
            "targets": asdict(kpi_targets),
            "evaluation": asdict(kpi_evaluation),
        },
    )
    effective_abstention_threshold = float(workflow_result.model_card.abstention.threshold)
    requested_abstention_threshold = float(args.abstention_threshold)
    threshold_tuned = abs(effective_abstention_threshold - requested_abstention_threshold) > 1e-12
    model_version = f"0.1.0+{built_dataset.fingerprint[:8]}"
    latency_metrics = _benchmark_inference_latency_ms(
        model_state_dict=workflow_result.model_state_dict,
        input_channels=workflow_result.model_input_channels,
        num_classes=workflow_result.model_num_classes,
        sample_batch=workflow_result.test_dataset.inputs[:1],
        device=args.device,
    )
    model_artifacts = save_edge_model_package(
        output_dir=output_dir,
        model_state_dict=workflow_result.model_state_dict,
        model_name=workflow_result.model_card.model_name,
        input_channels=workflow_result.model_input_channels,
        num_classes=workflow_result.model_num_classes,
        class_names=workflow_result.train_dataset.class_names,
        temperature=workflow_result.temperature.temperature,
        abstention_threshold=effective_abstention_threshold,
        dataset_fingerprint=built_dataset.fingerprint,
        model_id=f"whirdetective.{args.dataset_name}.baseline",
        model_version=model_version,
        extra_metadata={
            "kpi_passed": kpi_evaluation.passed,
            "trainer_seed": trainer_config.seed,
        },
        signing_key_env_var=args.manifest_signing_key_env,
    )
    edgeos_manifest_payload = json.loads(model_artifacts.edgeos_manifest_path.read_text(encoding="utf-8"))
    edgeos_contract_valid, edgeos_contract_failures = validate_edgeos_model_manifest(edgeos_manifest_payload)
    verification = None
    if args.verify_artifacts:
        verification_key = os.environ.get(args.manifest_signing_key_env)
        verification = verify_edge_model_package(
            package_dir=output_dir,
            signing_key=verification_key,
        )
        if args.require_signature and verification.signature_verified is None:
            raise ValueError("manifest signature is required but no signature file was emitted")
        if not verification.ok:
            raise ValueError(
                "package verification failed: "
                f"missing={list(verification.missing_files)}, "
                f"checksum_mismatches={list(verification.checksum_mismatches)}, "
                f"size_mismatches={list(verification.size_mismatches)}, "
                f"signature_verified={verification.signature_verified}, "
                f"signature_reason={verification.signature_reason}"
            )

    package_verified = verification.ok if verification is not None else None
    signature_verified = verification.signature_verified if verification is not None else None
    model_size_bytes = int(model_artifacts.model_state_path.stat().st_size)
    model_parameter_count = _count_model_parameters(workflow_result.model_state_dict)
    release_gate_targets = ReleaseGateTargets(
        require_kpi_passed=True,
        require_artifact_verification=bool(args.release_require_artifact_verification),
        require_signature=bool(args.require_signature),
        max_model_size_bytes=args.release_max_model_size_bytes,
        max_model_parameters=args.release_max_parameter_count,
        max_p95_inference_ms=args.release_max_p95_inference_ms,
    )
    release_gate = evaluate_release_gate(
        kpi_passed=kpi_evaluation.passed,
        package_verified=package_verified,
        signature_verified=signature_verified,
        model_size_bytes=model_size_bytes,
        model_parameter_count=model_parameter_count,
        p95_inference_ms=float(latency_metrics["p95_ms"]),
        targets=release_gate_targets,
    )
    if not edgeos_contract_valid:
        release_gate = type(release_gate)(
            passed=False,
            failed_checks=tuple((*release_gate.failed_checks, "edgeos_contract_invalid")),
            kpi_passed=release_gate.kpi_passed,
            package_verified=release_gate.package_verified,
            signature_verified=release_gate.signature_verified,
            model_size_bytes=release_gate.model_size_bytes,
            model_parameter_count=release_gate.model_parameter_count,
            p95_inference_ms=release_gate.p95_inference_ms,
        )
    _write_json(
        release_gate_path,
        {
            "targets": asdict(release_gate_targets),
            "evaluation": asdict(release_gate),
            "edgeos_contract": {
                "valid": edgeos_contract_valid,
                "failures": list(edgeos_contract_failures),
            },
        },
    )

    _write_json(
        run_report_path,
        {
            "workspace_root": str(workspace_root),
            "dataset_root": str(dataset_root),
            "dataset_name": str(args.dataset_name),
            "dataset_fingerprint": built_dataset.fingerprint,
            "source_file_count": len(built_dataset.source_files),
            "num_samples": len(built_dataset.samples),
            "split_sizes": {
                "train": len(built_dataset.split.train_indices),
                "val": len(built_dataset.split.val_indices),
                "test": len(built_dataset.split.test_indices),
            },
            "split_class_coverage": class_coverage,
            "trainer_config": asdict(trainer_config),
            "training_history": {
                "train_losses": list(workflow_result.history.train_losses),
                "val_losses": list(workflow_result.history.val_losses),
            },
            "temperature": asdict(workflow_result.temperature),
            "abstention_threshold": effective_abstention_threshold,
            "requested_abstention_threshold": requested_abstention_threshold,
            "abstention_threshold_tuned": threshold_tuned,
            "model_artifacts": {
                "model_state_path": str(model_artifacts.model_state_path),
                "inference_config_path": str(model_artifacts.inference_config_path),
                "calibration_path": str(model_artifacts.calibration_path),
                "edgeos_manifest_path": str(model_artifacts.edgeos_manifest_path),
                "manifest_path": str(model_artifacts.manifest_path),
                "signature_path": (
                    str(model_artifacts.signature_path) if model_artifacts.signature_path is not None else None
                ),
            },
            "artifact_verification": {
                "enabled": bool(args.verify_artifacts),
                "require_signature": bool(args.require_signature),
                "signature_key_env": args.manifest_signing_key_env,
                "result": (
                    {
                        "ok": verification.ok,
                        "missing_files": list(verification.missing_files),
                        "checksum_mismatches": list(verification.checksum_mismatches),
                        "size_mismatches": list(verification.size_mismatches),
                        "signature_verified": verification.signature_verified,
                        "signature_reason": verification.signature_reason,
                    }
                    if verification is not None
                    else None
                ),
            },
            "release_gate": {
                "path": str(release_gate_path),
                "passed": release_gate.passed,
                "failed_checks": list(release_gate.failed_checks),
                "model_size_bytes": release_gate.model_size_bytes,
                "model_parameter_count": release_gate.model_parameter_count,
                "p95_inference_ms": release_gate.p95_inference_ms,
                "edgeos_contract_valid": edgeos_contract_valid,
                "edgeos_contract_failures": list(edgeos_contract_failures),
            },
            "latency": latency_metrics,
            "kpi_passed": kpi_evaluation.passed,
        },
    )

    return Step4CliArtifacts(
        model_card_path=model_card_path,
        kpi_report_path=kpi_report_path,
        run_report_path=run_report_path,
        release_gate_path=release_gate_path,
        kpi_passed=kpi_evaluation.passed,
        release_gate_passed=release_gate.passed,
    )


def _assert_split_class_coverage(
    *,
    built_dataset: BuiltCanonicalDataset,
    min_train_classes: int,
    min_eval_classes: int,
    require_all_classes: bool,
) -> dict[str, list[str]]:
    if min_train_classes <= 0:
        raise ValueError("min_train_classes must be > 0")
    if min_eval_classes <= 0:
        raise ValueError("min_eval_classes must be > 0")

    train_labels = {built_dataset.samples[idx].label.value for idx in built_dataset.split.train_indices}
    val_labels = {built_dataset.samples[idx].label.value for idx in built_dataset.split.val_indices}
    test_labels = {built_dataset.samples[idx].label.value for idx in built_dataset.split.test_indices}

    if len(train_labels) < min_train_classes:
        raise ValueError(
            f"train split has only {len(train_labels)} class(es): {sorted(train_labels)}. "
            "Increase data diversity so training coverage meets the required minimum."
        )
    if len(val_labels) < min_eval_classes:
        raise ValueError(
            f"val split has only {len(val_labels)} class(es): {sorted(val_labels)}. "
            "Increase data diversity or split-search-attempts."
        )
    if len(test_labels) < min_eval_classes:
        raise ValueError(
            f"test split has only {len(test_labels)} class(es): {sorted(test_labels)}. "
            "Increase data diversity or split-search-attempts."
        )

    if not val_labels.issubset(train_labels):
        raise ValueError(
            "val split contains classes not present in train split: "
            f"{sorted(val_labels.difference(train_labels))}"
        )
    if not test_labels.issubset(train_labels):
        raise ValueError(
            "test split contains classes not present in train split: "
            f"{sorted(test_labels.difference(train_labels))}"
        )
    if require_all_classes:
        if val_labels != train_labels:
            raise ValueError(
                "split-require-all-classes is enabled but val split class coverage differs "
                f"from train. train={sorted(train_labels)} val={sorted(val_labels)}"
            )
        if test_labels != train_labels:
            raise ValueError(
                "split-require-all-classes is enabled but test split class coverage differs "
                f"from train. train={sorted(train_labels)} test={sorted(test_labels)}"
            )
    return {
        "train": sorted(train_labels),
        "val": sorted(val_labels),
        "test": sorted(test_labels),
    }


def _build_dataset_from_args(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    projector: SensorSetProjector,
) -> BuiltCanonicalDataset:
    require_all_classes = _resolve_split_require_all_classes(args)
    if args.dataset_name == "cwru":
        cwru_config = CwruBuildConfig(
            root_dir=dataset_root,
            window_size=args.window_size,
            step_size=args.step_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            max_files=args.max_files,
            min_distinct_labels_per_split=args.split_min_distinct_classes,
            require_all_labels_per_split=bool(require_all_classes),
            split_search_attempts=args.split_search_attempts,
        )
        return build_cwru_canonical_dataset(config=cwru_config, projector=projector)

    if args.dataset_name == "paderborn":
        paderborn_config = PaderbornBuildConfig(
            root_dir=dataset_root,
            window_size=args.window_size,
            step_size=args.step_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            max_archives=args.max_archives,
            max_entries_per_archive=args.max_entries_per_archive,
            min_distinct_labels_per_split=args.split_min_distinct_classes,
            require_all_labels_per_split=bool(require_all_classes),
            split_search_attempts=args.split_search_attempts,
            collapse_fault_classes=bool(args.paderborn_collapse_fault_classes),
        )
        return build_paderborn_canonical_dataset(config=paderborn_config, projector=projector)

    raise ValueError(f"unsupported dataset_name: {args.dataset_name}")


def _benchmark_inference_latency_ms(
    *,
    model_state_dict: Mapping[str, torch.Tensor],
    input_channels: int,
    num_classes: int,
    sample_batch: torch.Tensor,
    device: str,
    warmup_runs: int = 2,
    timed_runs: int = 10,
) -> dict[str, float]:
    if warmup_runs <= 0:
        raise ValueError("warmup_runs must be > 0")
    if timed_runs <= 0:
        raise ValueError("timed_runs must be > 0")
    if sample_batch.ndim != 3:
        raise ValueError("sample_batch must have shape [batch, channels, samples]")
    if sample_batch.shape[0] <= 0:
        raise ValueError("sample_batch must contain at least one example")

    if hasattr(torch.backends, "nnpack"):
        try:
            setattr(torch.backends.nnpack, "enabled", False)
        except Exception:
            pass

    resolved_device = "cpu" if device != "cpu" and not torch.cuda.is_available() else device
    model = BaselineBearingCNN(input_channels=input_channels, num_classes=num_classes)
    model.load_state_dict(dict(model_state_dict), strict=False)
    model.eval()
    model.to(resolved_device)
    input_tensor = sample_batch[:1].to(resolved_device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
        timings: list[float] = []
        for _ in range(timed_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            if resolved_device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(float(elapsed_ms))

    samples = np.asarray(timings, dtype=np.float64)
    return {
        "p50_ms": float(np.quantile(samples, 0.50)),
        "p95_ms": float(np.quantile(samples, 0.95)),
        "p99_ms": float(np.quantile(samples, 0.99)),
        "mean_ms": float(np.mean(samples)),
    }


def _resolve_split_require_all_classes(args: argparse.Namespace) -> bool:
    requested = getattr(args, "split_require_all_classes", None)
    if requested is not None:
        return bool(requested)
    return str(getattr(args, "dataset_name", "")) == "cwru"


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
    print(f"release_gate: {artifacts.release_gate_path}")
    print(f"kpi_passed: {artifacts.kpi_passed}")
    print(f"release_gate_passed: {artifacts.release_gate_passed}")
    if args.fail_on_kpi and not artifacts.kpi_passed:
        print("[ERROR] KPI gate failed and --fail-on-kpi is set.", file=sys.stderr)
        return 1
    if args.fail_on_release_gate and not artifacts.release_gate_passed:
        print("[ERROR] Release gate failed and --fail-on-release-gate is set.", file=sys.stderr)
        return 1
    return 0


def _resolve_path(base_dir: Path, path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    return (base_dir / path_value).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _count_model_parameters(state_dict: Mapping[str, Any]) -> int:
    total = 0
    for tensor in state_dict.values():
        numel = getattr(tensor, "numel", None)
        if numel is None:
            raise TypeError("model_state_dict contains a non-tensor value")
        total += int(numel())
    if total <= 0:
        raise ValueError("model_state_dict produced zero parameters")
    return total


if __name__ == "__main__":
    raise SystemExit(main())
