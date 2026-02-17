"""Controlled Runtime Qualification + Freeze workflow for pilot release candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence, cast

from whirdetective.export.manifest import sha256_file

_PACKAGE_FILES: tuple[str, ...] = (
    "model_state_dict.pt",
    "inference_config.json",
    "calibration.json",
    "edgeos_model_manifest.json",
    "manifest.json",
    "manifest.sig",
    "model_card.json",
    "kpi_report.json",
    "run_report.json",
    "release_gate.json",
)

_DEFAULT_RUNTIME_SUITES: tuple[str, ...] = (
    "ML_InferenceTests",
    "ML_ModelManagerTests",
    "ML_OTAUpdateTests",
    "ML_MonitoringTests",
    "ModelRegistryTests",
)


@dataclass(frozen=True, slots=True)
class RuntimeSuiteResult:
    """Outcome of one EdgeOS runtime qualification test suite."""

    suite: str
    passed: bool
    return_code: int
    elapsed_seconds: float
    log_path: Path


@dataclass(frozen=True, slots=True)
class RuntimeFreezeArtifacts:
    """Primary outputs of one runtime qualification + freeze execution."""

    freeze_dir: Path
    signoff_json_path: Path
    signoff_markdown_path: Path
    freeze_manifest_path: Path
    passed: bool


def build_parser() -> argparse.ArgumentParser:
    """Create parser for controlled runtime qualification + freeze execution."""
    parser = argparse.ArgumentParser(
        prog="whirdetective-runtime-freeze",
        description=(
            "Validate pilot benchmark gates, execute EdgeOS runtime qualification tests, "
            "and freeze a release-candidate bundle."
        ),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--pilot-root", type=Path, default=Path("artifacts/pilot"))
    parser.add_argument(
        "--edgeos-root",
        type=Path,
        default=Path("/home/bootctrl/Projects/BootCtrl-EdgeOS"),
    )
    parser.add_argument(
        "--forte-test-path",
        type=Path,
        default=Path("build/posix-debug/4diacFORTE/tests/forte_test"),
        help="Path to EdgeOS forte_test binary (relative to edgeos-root by default).",
    )
    parser.add_argument(
        "--freeze-root",
        type=Path,
        default=Path("artifacts/pilot/freeze"),
    )
    parser.add_argument(
        "--freeze-label",
        type=str,
        default=None,
        help="Stable release candidate label (default: UTC timestamp).",
    )
    parser.add_argument(
        "--cwru-dir-name",
        type=str,
        default="cwru-full-gpu-e12-v3",
    )
    parser.add_argument(
        "--paderborn-dir-name",
        type=str,
        default="paderborn-full-gpu-e8-v2",
    )
    parser.add_argument(
        "--xjtu-report",
        type=Path,
        default=Path("xjtu-full/xjtu_prognostics_report.json"),
        help="Path to XJTU prognostics report relative to pilot-root.",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=Path("preflight/preflight_report.json"),
        help="Path to preflight report relative to pilot-root.",
    )
    parser.add_argument(
        "--pilot-progress",
        type=Path,
        default=Path("pilot_progress.json"),
        help="Path to pilot progress JSON relative to pilot-root.",
    )
    parser.add_argument(
        "--runtime-suites",
        type=str,
        default=",".join(_DEFAULT_RUNTIME_SUITES),
        help="Comma-separated Boost.Test suite names to run in forte_test.",
    )
    parser.add_argument(
        "--skip-runtime-tests",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip EdgeOS runtime suite execution (not recommended for controlled freeze).",
    )
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit with code 1 when any qualification gate fails.",
    )
    return parser


def run_runtime_freeze_from_args(args: argparse.Namespace) -> RuntimeFreezeArtifacts:
    """Execute pilot runtime qualification and freeze candidate artifacts."""
    workspace_root = args.workspace_root.resolve()
    pilot_root = _resolve_path(workspace_root, args.pilot_root)
    edgeos_root = _resolve_path(workspace_root, args.edgeos_root)
    freeze_root = _resolve_path(workspace_root, args.freeze_root)

    label = args.freeze_label or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    freeze_dir = freeze_root / label
    if freeze_dir.exists():
        raise FileExistsError(f"freeze directory already exists: {freeze_dir}")
    freeze_dir.mkdir(parents=True, exist_ok=False)

    cwru_dir = pilot_root / args.cwru_dir_name
    paderborn_dir = pilot_root / args.paderborn_dir_name
    xjtu_report_path = pilot_root / args.xjtu_report
    preflight_report_path = pilot_root / args.preflight_report
    pilot_progress_path = pilot_root / args.pilot_progress

    gate_checks: list[dict[str, Any]] = []
    failed_checks: list[str] = []

    _append_gate(
        gate_checks,
        failed_checks,
        _check_bool_json_gate(
            name="cwru_kpi_passed",
            path=cwru_dir / "kpi_report.json",
            key_path=("evaluation", "passed"),
            expected=True,
        ),
    )
    _append_gate(
        gate_checks,
        failed_checks,
        _check_bool_json_gate(
            name="cwru_release_gate_passed",
            path=cwru_dir / "release_gate.json",
            key_path=("evaluation", "passed"),
            expected=True,
        ),
    )
    _append_gate(
        gate_checks,
        failed_checks,
        _check_bool_json_gate(
            name="paderborn_kpi_passed",
            path=paderborn_dir / "kpi_report.json",
            key_path=("evaluation", "passed"),
            expected=True,
        ),
    )
    _append_gate(
        gate_checks,
        failed_checks,
        _check_bool_json_gate(
            name="paderborn_release_gate_passed",
            path=paderborn_dir / "release_gate.json",
            key_path=("evaluation", "passed"),
            expected=True,
        ),
    )
    _append_gate(
        gate_checks,
        failed_checks,
        _check_bool_json_gate(
            name="xjtu_prognostics_passed",
            path=xjtu_report_path,
            key_path=("evaluation", "passed"),
            expected=True,
        ),
    )
    _append_gate(
        gate_checks,
        failed_checks,
        _check_preflight_report(preflight_report_path),
    )
    _append_gate(
        gate_checks,
        failed_checks,
        _check_pilot_progress(pilot_progress_path),
    )

    runtime_results: list[RuntimeSuiteResult] = []
    runtime_qualification_passed = True
    runtime_failed_suites: list[str] = []

    runtime_logs_dir = freeze_dir / "runtime_tests"
    runtime_logs_dir.mkdir(parents=True, exist_ok=True)
    suites = _parse_runtime_suites(args.runtime_suites)
    forte_test_path = _resolve_path(edgeos_root, args.forte_test_path)

    if args.skip_runtime_tests:
        runtime_qualification_passed = False
        runtime_failed_suites.append("runtime_tests_skipped")
    elif not forte_test_path.exists():
        runtime_qualification_passed = False
        runtime_failed_suites.append(f"forte_test_missing:{forte_test_path}")
    else:
        for suite in suites:
            result = _run_forte_suite(
                forte_test_path=forte_test_path,
                suite=suite,
                output_dir=runtime_logs_dir,
            )
            runtime_results.append(result)
            if not result.passed:
                runtime_qualification_passed = False
                runtime_failed_suites.append(suite)

    if not runtime_qualification_passed:
        failed_checks.append("edgeos_runtime_qualification_failed")

    freeze_sources = {
        "diagnosis/cwru": cwru_dir,
        "diagnosis/paderborn": paderborn_dir,
    }
    for destination_rel, source_dir in freeze_sources.items():
        destination = freeze_dir / destination_rel
        destination.mkdir(parents=True, exist_ok=True)
        _copy_package_files(source_dir=source_dir, destination_dir=destination)

    preflight_dest = freeze_dir / "preflight"
    preflight_dest.mkdir(parents=True, exist_ok=True)
    if preflight_report_path.exists():
        shutil.copy2(preflight_report_path, preflight_dest / "preflight_report.json")

    xjtu_dest = freeze_dir / "prognostics"
    xjtu_dest.mkdir(parents=True, exist_ok=True)
    if xjtu_report_path.exists():
        shutil.copy2(xjtu_report_path, xjtu_dest / "xjtu_prognostics_report.json")

    passed = len(failed_checks) == 0
    signoff_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "workspace_root": str(workspace_root),
        "pilot_root": str(pilot_root),
        "edgeos_root": str(edgeos_root),
        "freeze_dir": str(freeze_dir),
        "freeze_label": label,
        "evaluation": {
            "passed": passed,
            "failed_checks": sorted(failed_checks),
        },
        "artifact_gates": gate_checks,
        "runtime_qualification": {
            "passed": runtime_qualification_passed,
            "forte_test_path": str(forte_test_path),
            "suites": [
                {
                    **asdict(result),
                    "log_path": str(result.log_path),
                }
                for result in runtime_results
            ],
            "failed_suites": sorted(runtime_failed_suites),
        },
    }

    signoff_json_path = freeze_dir / "pilot_release_signoff.json"
    signoff_json_path.write_text(json.dumps(signoff_payload, indent=2, sort_keys=True), encoding="utf-8")

    signoff_markdown_path = freeze_dir / "PILOT_RELEASE_SIGNOFF.md"
    signoff_markdown_path.write_text(_build_signoff_markdown(signoff_payload), encoding="utf-8")

    freeze_manifest_path = freeze_dir / "freeze_manifest.json"
    freeze_manifest_path.write_text(
        json.dumps(_build_freeze_manifest_payload(freeze_dir), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return RuntimeFreezeArtifacts(
        freeze_dir=freeze_dir,
        signoff_json_path=signoff_json_path,
        signoff_markdown_path=signoff_markdown_path,
        freeze_manifest_path=freeze_manifest_path,
        passed=passed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for runtime qualification + freeze."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        artifacts = run_runtime_freeze_from_args(args)
    except Exception as exc:
        print(f"[ERROR] Runtime qualification + freeze failed: {exc}", file=sys.stderr)
        return 2

    print(f"freeze_dir: {artifacts.freeze_dir}")
    print(f"signoff_json: {artifacts.signoff_json_path}")
    print(f"signoff_markdown: {artifacts.signoff_markdown_path}")
    print(f"freeze_manifest: {artifacts.freeze_manifest_path}")
    print(f"passed: {artifacts.passed}")
    if args.fail_on_gate and not artifacts.passed:
        return 1
    return 0


def _resolve_path(base_dir: Path, value: Path) -> Path:
    if value.is_absolute():
        return value.resolve()
    return (base_dir / value).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return cast(dict[str, Any], payload)


def _check_bool_json_gate(
    *,
    name: str,
    path: Path,
    key_path: tuple[str, ...],
    expected: bool,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "name": name,
            "passed": False,
            "details": f"missing_file:{path}",
        }
    payload = _load_json(path)
    current: Any = payload
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return {
                "name": name,
                "passed": False,
                "details": f"missing_key:{'.'.join(key_path)}",
            }
        current = current[key]

    passed = bool(current) is expected
    return {
        "name": name,
        "passed": passed,
        "details": {
            "path": str(path),
            "value": bool(current),
            "expected": expected,
        },
    }


def _check_preflight_report(path: Path) -> dict[str, Any]:
    name = "preflight_gate_passed"
    if not path.exists():
        return {
            "name": name,
            "passed": False,
            "details": f"missing_file:{path}",
        }
    payload = _load_json(path)
    datasets = payload.get("datasets")
    torch_info = payload.get("torch")
    if not isinstance(datasets, dict) or not isinstance(torch_info, dict):
        return {
            "name": name,
            "passed": False,
            "details": "invalid_preflight_schema",
        }

    dataset_flags = []
    for entry in datasets.values():
        if isinstance(entry, dict):
            dataset_flags.append(bool(entry.get("exists", False)))
        else:
            dataset_flags.append(False)

    cuda_available = bool(torch_info.get("cuda_available", False))
    passed = all(dataset_flags) and cuda_available
    return {
        "name": name,
        "passed": passed,
        "details": {
            "path": str(path),
            "dataset_exists": all(dataset_flags),
            "cuda_available": cuda_available,
        },
    }


def _check_pilot_progress(path: Path) -> dict[str, Any]:
    name = "pilot_progress_gate_passed"
    if not path.exists():
        return {
            "name": name,
            "passed": False,
            "details": f"missing_file:{path}",
        }
    payload = _load_json(path)
    status_payload = payload.get("status")
    if not isinstance(status_payload, dict):
        return {
            "name": name,
            "passed": False,
            "details": "missing_or_invalid_status",
        }
    required = ("preflight", "cwru", "paderborn", "xjtu")
    values = {name: bool(status_payload.get(name, False)) for name in required}
    passed = all(values.values())
    return {
        "name": name,
        "passed": passed,
        "details": {
            "path": str(path),
            "status": values,
        },
    }


def _append_gate(
    gate_checks: list[dict[str, Any]],
    failed_checks: list[str],
    gate: dict[str, Any],
) -> None:
    gate_checks.append(gate)
    if not bool(gate.get("passed", False)):
        failed_checks.append(str(gate.get("name", "unknown_gate_failed")))


def _parse_runtime_suites(raw: str) -> tuple[str, ...]:
    suites = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(suites) == 0:
        raise ValueError("at least one runtime suite must be provided")
    return suites


def _run_forte_suite(*, forte_test_path: Path, suite: str, output_dir: Path) -> RuntimeSuiteResult:
    command = [
        str(forte_test_path),
        f"--run_test={suite}",
        "--report_level=short",
        "--log_level=warning",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started

    log_path = output_dir / f"{suite}.log"
    combined_output = "\n".join(
        [
            f"command: {' '.join(command)}",
            f"return_code: {completed.returncode}",
            "stdout:",
            completed.stdout,
            "stderr:",
            completed.stderr,
        ]
    )
    log_path.write_text(combined_output, encoding="utf-8")

    return RuntimeSuiteResult(
        suite=suite,
        passed=(completed.returncode == 0),
        return_code=completed.returncode,
        elapsed_seconds=elapsed,
        log_path=log_path,
    )


def _copy_package_files(*, source_dir: Path, destination_dir: Path) -> None:
    for file_name in _PACKAGE_FILES:
        source_path = source_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, destination_dir / file_name)


def _build_signoff_markdown(payload: dict[str, Any]) -> str:
    evaluation = payload["evaluation"]
    runtime = payload["runtime_qualification"]
    lines = [
        "# Pilot Runtime Qualification + Freeze Signoff",
        "",
        f"- Generated at: {payload['generated_at_utc']}",
        f"- Freeze label: {payload['freeze_label']}",
        f"- Freeze dir: {payload['freeze_dir']}",
        f"- Overall passed: {evaluation['passed']}",
        "",
        "## Artifact Gates",
    ]
    for gate in payload["artifact_gates"]:
        lines.append(f"- {gate['name']}: {gate['passed']}")

    lines.extend(
        [
            "",
            "## Runtime Qualification",
            f"- forte_test: {runtime['forte_test_path']}",
            f"- Runtime passed: {runtime['passed']}",
        ]
    )
    for suite in runtime["suites"]:
        lines.append(
            f"- {suite['suite']}: passed={suite['passed']} "
            f"return_code={suite['return_code']} elapsed_seconds={suite['elapsed_seconds']:.3f}"
        )

    failed_checks = evaluation.get("failed_checks", [])
    lines.extend(["", "## Failures"])
    if failed_checks:
        for failed in failed_checks:
            lines.append(f"- {failed}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def _build_freeze_manifest_payload(freeze_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(freeze_dir.rglob("*")):
        if not path.is_file() or path.name == "freeze_manifest.json":
            continue
        relative = path.relative_to(freeze_dir)
        entries.append(
            {
                "path": str(relative),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return {
        "manifest_type": "pilot_runtime_freeze",
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": entries,
    }


if __name__ == "__main__":
    raise SystemExit(main())
