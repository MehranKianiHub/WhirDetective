"""Controlled EdgeOS-deployable export track for strict canary qualification."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from whirdetective.export import (
    save_edge_model_blob_package,
    validate_edgeos_model_manifest,
    verify_edge_model_package,
)


@dataclass(frozen=True, slots=True)
class EdgeosExportArtifacts:
    """Artifacts emitted by one controlled EdgeOS export execution."""

    output_dir: Path
    report_path: Path
    passed: bool


def build_parser() -> argparse.ArgumentParser:
    """Create parser for EdgeOS deployable export."""
    parser = argparse.ArgumentParser(
        prog="whirdetective-edgeos-export",
        description=(
            "Build a controlled EdgeOS deployable package (tflite backend) from a frozen diagnosis "
            "package and an explicit model blob."
        ),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-package-dir",
        type=Path,
        required=True,
        help="Path to source diagnosis package (contains KPI/release evidence and metadata).",
    )
    parser.add_argument(
        "--model-blob",
        type=Path,
        required=True,
        help="EdgeOS deployable model blob path (for tflite backend, this should be .tflite).",
    )
    parser.add_argument(
        "--backend",
        choices=("tflite", "tflite_flatbuffer"),
        default="tflite",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--model-version", type=str, default=None)
    parser.add_argument(
        "--model-artifact-name",
        type=str,
        default="model.tflite",
        help="File name used inside output package.",
    )
    parser.add_argument(
        "--validate-tflite-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Attempt interpreter-level model load validation for tflite backends "
            "(requires tflite_runtime or tensorflow)."
        ),
    )
    parser.add_argument(
        "--require-kpi-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-release-gate-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--manifest-signing-key-env",
        type=str,
        default="WHIRDETECTIVE_MANIFEST_SIGNING_KEY",
    )
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def run_edgeos_export_from_args(args: argparse.Namespace) -> EdgeosExportArtifacts:
    """Run controlled deployable export flow."""
    workspace_root = args.workspace_root.resolve()
    source_dir = _resolve_path(workspace_root, args.source_package_dir)
    model_blob = _resolve_path(workspace_root, args.model_blob)
    output_dir = _resolve_path(workspace_root, args.output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"source package dir does not exist: {source_dir}")
    if not model_blob.exists():
        raise FileNotFoundError(f"model blob does not exist: {model_blob}")

    source_manifest = _load_json(source_dir / "edgeos_model_manifest.json")
    source_calibration = _load_json(source_dir / "calibration.json")

    source_classification = source_manifest.get("classification")
    if not isinstance(source_classification, dict):
        raise ValueError("source edgeos manifest missing classification section")
    class_names_raw = source_classification.get("class_names")
    if not isinstance(class_names_raw, list) or len(class_names_raw) <= 1:
        raise ValueError("source edgeos manifest classification.class_names must contain at least 2 classes")
    class_names = tuple(str(name) for name in class_names_raw)

    input_section = source_manifest.get("input")
    if not isinstance(input_section, dict):
        raise ValueError("source edgeos manifest missing input section")
    input_channels = int(input_section.get("channels", 0))
    if input_channels <= 0:
        raise ValueError("source edgeos manifest input.channels must be > 0")

    temperature = float(source_calibration.get("temperature", 0.0))
    abstention_threshold = float(source_calibration.get("abstention_threshold", 0.0))
    dataset_fingerprint = str(source_manifest.get("metadata", {}).get("dataset_fingerprint", ""))
    if not dataset_fingerprint:
        raise ValueError("source edgeos manifest missing metadata.dataset_fingerprint")

    source_kpi_report = source_dir / "kpi_report.json"
    source_release_gate = source_dir / "release_gate.json"
    kpi_passed = _extract_bool(source_kpi_report, ("evaluation", "passed"))
    release_gate_passed = _extract_bool(source_release_gate, ("evaluation", "passed"))

    failed_checks: list[str] = []
    if args.require_kpi_pass and not kpi_passed:
        failed_checks.append("source_kpi_gate_failed")
    if args.require_release_gate_pass and not release_gate_passed:
        failed_checks.append("source_release_gate_failed")

    model_id = str(args.model_id) if args.model_id else str(source_manifest.get("model_id", ""))
    model_version = (
        str(args.model_version)
        if args.model_version
        else str(source_manifest.get("version", ""))
    )
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if not model_version.strip():
        raise ValueError("model_version must not be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    package = save_edge_model_blob_package(
        output_dir=output_dir,
        model_blob_path=model_blob,
        model_artifact_name=args.model_artifact_name,
        backend=args.backend,
        model_name=str(source_manifest.get("model_id", model_id)),
        input_channels=input_channels,
        num_classes=len(class_names),
        class_names=class_names,
        temperature=temperature,
        abstention_threshold=abstention_threshold,
        dataset_fingerprint=dataset_fingerprint,
        model_id=model_id,
        model_version=model_version,
        extra_metadata={
            "source_package_dir": str(source_dir),
            "source_kpi_passed": kpi_passed,
            "source_release_gate_passed": release_gate_passed,
        },
        signing_key_env_var=args.manifest_signing_key_env,
    )

    for evidence_name in (
        "model_card.json",
        "kpi_report.json",
        "run_report.json",
        "release_gate.json",
    ):
        source_path = source_dir / evidence_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / evidence_name)

    edgeos_manifest_payload = _load_json(package.edgeos_manifest_path)
    edgeos_ok, edgeos_failures = validate_edgeos_model_manifest(edgeos_manifest_payload)
    verification = verify_edge_model_package(package_dir=output_dir)
    model_blob_validation = _validate_model_blob_load(
        model_path=package.model_state_path,
        backend=args.backend,
        requested=bool(args.validate_tflite_load),
    )

    if not edgeos_ok:
        failed_checks.extend(f"edgeos_manifest:{failure}" for failure in edgeos_failures)
    if not verification.ok:
        failed_checks.append("package_verification_failed")
    if model_blob_validation["requested"] and not bool(model_blob_validation["passed"]):
        failed_checks.append("model_blob_load_validation_failed")

    report_payload = {
        "source_package_dir": str(source_dir),
        "output_dir": str(output_dir),
        "model_blob": str(model_blob),
        "backend": args.backend,
        "model_id": model_id,
        "model_version": model_version,
        "source_gates": {
            "kpi_passed": kpi_passed,
            "release_gate_passed": release_gate_passed,
        },
        "edgeos_contract": {
            "valid": edgeos_ok,
            "failures": list(edgeos_failures),
        },
        "package_verification": asdict(verification),
        "model_blob_validation": model_blob_validation,
        "evaluation": {
            "passed": len(failed_checks) == 0,
            "failed_checks": sorted(failed_checks),
        },
        "artifacts": {
            "model_blob_path": str(package.model_state_path),
            "edgeos_manifest_path": str(package.edgeos_manifest_path),
            "manifest_path": str(package.manifest_path),
        },
    }

    report_path = output_dir / "edgeos_export_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    return EdgeosExportArtifacts(
        output_dir=output_dir,
        report_path=report_path,
        passed=(len(failed_checks) == 0),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for deployable export track."""
    args = build_parser().parse_args(argv)
    try:
        artifacts = run_edgeos_export_from_args(args)
    except Exception as exc:
        print(f"[ERROR] edgeos export failed: {exc}", file=sys.stderr)
        return 2

    print(f"output_dir: {artifacts.output_dir}")
    print(f"report: {artifacts.report_path}")
    print(f"passed: {artifacts.passed}")
    if args.fail_on_gate and not artifacts.passed:
        return 1
    return 0


def _resolve_path(base_dir: Path, path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    return (base_dir / path_value).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _extract_bool(path: Path, key_path: tuple[str, ...]) -> bool:
    payload: Any = _load_json(path)
    for key in key_path:
        if not isinstance(payload, dict) or key not in payload:
            return False
        payload = payload[key]
    return bool(payload)


def _validate_model_blob_load(*, model_path: Path, backend: str, requested: bool) -> dict[str, Any]:
    if backend not in {"tflite", "tflite_flatbuffer"}:
        return {
            "requested": requested,
            "attempted": False,
            "passed": None,
            "reason": "backend_does_not_require_tflite_interpreter_validation",
        }
    if not requested:
        return {
            "requested": False,
            "attempted": False,
            "passed": None,
            "reason": "not_requested",
        }

    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]

        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return {
            "requested": True,
            "attempted": True,
            "passed": True,
            "reason": "loaded_with_tflite_runtime",
        }
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        return {
            "requested": True,
            "attempted": True,
            "passed": False,
            "reason": f"tflite_runtime_load_failed:{exc}",
        }

    try:
        import tensorflow as tf  # type: ignore[import-untyped]

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return {
            "requested": True,
            "attempted": True,
            "passed": True,
            "reason": "loaded_with_tensorflow_lite",
        }
    except ModuleNotFoundError:
        return {
            "requested": True,
            "attempted": False,
            "passed": False,
            "reason": "no_tflite_runtime_or_tensorflow_installed",
        }
    except Exception as exc:
        return {
            "requested": True,
            "attempted": True,
            "passed": False,
            "reason": f"tensorflow_lite_load_failed:{exc}",
        }


if __name__ == "__main__":
    raise SystemExit(main())
