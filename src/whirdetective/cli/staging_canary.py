"""Controlled staging canary qualification and rollout input generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from whirdetective.export import verify_edge_model_package
from whirdetective.export.manifest import sha256_file

_DEFAULT_RUNTIME_SUITES: tuple[str, ...] = (
    "ML_ModelManagerTests",
    "ML_OTAUpdateTests",
    "ML_InferenceTests",
)


@dataclass(frozen=True, slots=True)
class RuntimeSuiteResult:
    """Result from one selected EdgeOS runtime suite execution."""

    suite: str
    passed: bool
    return_code: int
    elapsed_seconds: float
    log_path: Path


@dataclass(frozen=True, slots=True)
class CanaryArtifacts:
    """Primary artifacts produced by canary preparation run."""

    output_dir: Path
    report_path: Path
    commands_path: Path
    passed: bool


def build_parser() -> argparse.ArgumentParser:
    """Create parser for controlled staging canary preparation."""
    parser = argparse.ArgumentParser(
        prog="whirdetective-staging-canary",
        description=(
            "Validate frozen release-candidate integrity and generate staged canary rollout inputs "
            "with rollback drill commands."
        ),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("artifacts/pilot/freeze/pilot-runtime-freeze-20260217"),
    )
    parser.add_argument("--track", choices=("cwru", "paderborn"), default="cwru")
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help="Optional package directory override (default: <freeze-dir>/diagnosis/<track>).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Canary preparation output directory (default: <freeze-dir>/canary/<track>).",
    )
    parser.add_argument(
        "--edgeos-root",
        type=Path,
        default=Path("/home/bootctrl/Projects/BootCtrl-EdgeOS"),
    )
    parser.add_argument(
        "--forte-test-path",
        type=Path,
        default=Path("build/posix-debug/4diacFORTE/tests/forte_test"),
    )
    parser.add_argument(
        "--runtime-suites",
        type=str,
        default=",".join(_DEFAULT_RUNTIME_SUITES),
    )
    parser.add_argument(
        "--skip-runtime-tests",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip runtime suite checks (not recommended).",
    )
    parser.add_argument(
        "--source-uri",
        type=str,
        default="https://updates.bootctrl.local/models/whirdetective/model_state_dict.pt",
        help="URI consumed by ML_OTAUpdate BEGIN command.",
    )
    parser.add_argument(
        "--nonce",
        type=str,
        default=None,
        help="Rollout nonce. Default uses UTC timestamp.",
    )
    parser.add_argument(
        "--chunk-size-bytes",
        type=int,
        default=65536,
    )
    parser.add_argument(
        "--allow-unsupported-backend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow canary prep to continue when model backend is not EdgeOS runtime-compatible. "
            "Use only for dry-run command generation."
        ),
    )
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def run_canary_from_args(args: argparse.Namespace) -> CanaryArtifacts:
    """Execute canary qualification checks and write rollout helper files."""
    workspace_root = args.workspace_root.resolve()
    freeze_dir = _resolve_path(workspace_root, args.freeze_dir)
    edgeos_root = _resolve_path(workspace_root, args.edgeos_root)
    output_dir = (
        _resolve_path(workspace_root, args.output_dir)
        if args.output_dir is not None
        else (freeze_dir / "canary" / args.track)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gates: list[dict[str, Any]] = []
    failed_checks: list[str] = []

    freeze_manifest = freeze_dir / "freeze_manifest.json"
    signoff = freeze_dir / "pilot_release_signoff.json"

    _append_gate(gates, failed_checks, _check_signoff_gate(signoff))
    _append_gate(gates, failed_checks, _check_freeze_manifest(freeze_dir, freeze_manifest))

    package_dir = (
        _resolve_path(workspace_root, args.package_dir)
        if args.package_dir is not None
        else (freeze_dir / "diagnosis" / args.track)
    )
    _append_gate(gates, failed_checks, _check_package_dir(package_dir))

    package_verification = verify_edge_model_package(package_dir=package_dir)
    package_gate = {
        "name": "package_integrity_gate_passed",
        "passed": package_verification.ok,
        "details": {
            "missing_files": list(package_verification.missing_files),
            "checksum_mismatches": list(package_verification.checksum_mismatches),
            "size_mismatches": list(package_verification.size_mismatches),
            "signature_verified": package_verification.signature_verified,
            "signature_reason": package_verification.signature_reason,
        },
    }
    _append_gate(gates, failed_checks, package_gate)

    model_manifest_path = package_dir / "edgeos_model_manifest.json"
    model_manifest = _load_json(model_manifest_path)
    model_path, model_path_gate = _resolve_model_artifact_path(
        package_dir=package_dir,
        model_manifest=model_manifest,
    )
    _append_gate(gates, failed_checks, model_path_gate)
    if model_path is None:
        raise FileNotFoundError("model artifact path could not be resolved from package")
    model_sha256 = sha256_file(model_path)
    model_size_bytes = int(model_path.stat().st_size)

    manifest_model_id = str(model_manifest["model_id"])
    manifest_version = str(model_manifest["version"])
    manifest_backend = str(model_manifest.get("backend", ""))

    _append_gate(
        gates,
        failed_checks,
        {
            "name": "manifest_model_hash_matches",
            "passed": (
                str(model_manifest.get("sha256")) == model_sha256
                and _safe_int(model_manifest.get("size_bytes"), default=-1) == model_size_bytes
            ),
            "details": {
                "expected_sha256": str(model_manifest.get("sha256")),
                "actual_sha256": model_sha256,
                "expected_size_bytes": _safe_int(model_manifest.get("size_bytes"), default=-1),
                "actual_size_bytes": model_size_bytes,
            },
        },
    )
    _append_gate(
        gates,
        failed_checks,
        _check_edgeos_backend_compatibility(
            model_id=manifest_model_id,
            backend=manifest_backend,
            allow_unsupported_backend=bool(args.allow_unsupported_backend),
        ),
    )

    runtime_results: list[RuntimeSuiteResult] = []
    runtime_suites = _parse_suites(args.runtime_suites)
    forte_test_path = _resolve_path(edgeos_root, args.forte_test_path)
    runtime_logs_dir = output_dir / "runtime_tests"
    runtime_logs_dir.mkdir(parents=True, exist_ok=True)

    runtime_passed = True
    runtime_failed_suites: list[str] = []
    if args.skip_runtime_tests:
        runtime_passed = False
        runtime_failed_suites.append("runtime_tests_skipped")
    elif not forte_test_path.exists():
        runtime_passed = False
        runtime_failed_suites.append(f"forte_test_missing:{forte_test_path}")
    else:
        for suite in runtime_suites:
            result = _run_suite(
                forte_test_path=forte_test_path,
                suite=suite,
                output_dir=runtime_logs_dir,
            )
            runtime_results.append(result)
            if not result.passed:
                runtime_passed = False
                runtime_failed_suites.append(suite)

    _append_gate(
        gates,
        failed_checks,
        {
            "name": "edgeos_runtime_smoke_gate_passed",
            "passed": runtime_passed,
            "details": {
                "forte_test_path": str(forte_test_path),
                "failed_suites": sorted(runtime_failed_suites),
            },
        },
    )

    nonce = args.nonce or datetime.now(UTC).strftime("canary-%Y%m%dT%H%M%SZ")
    chunk_manifest = _write_chunks(
        source_file=model_path,
        chunk_size_bytes=args.chunk_size_bytes,
        output_dir=output_dir / "chunks",
    )

    command_payloads_dir = output_dir / "command_payloads"
    command_payloads_dir.mkdir(parents=True, exist_ok=True)

    begin_payload = {
        "COMMAND": 0,
        "MODEL_ID": manifest_model_id,
        "VERSION": manifest_version,
        "EXPECTED_SIZE": model_size_bytes,
        "EXPECTED_SHA256": f"sha256:{model_sha256}",
        "SOURCE_URI": args.source_uri,
        "TRANSPORT_SECURE": True,
        "NONCE": nonce,
    }
    commit_payload = {
        "COMMAND": 2,
        "MODEL_ID": manifest_model_id,
        "VERSION": manifest_version,
        "NONCE": nonce,
    }
    rollback_payload = {
        "COMMAND": 4,
        "MODEL_ID": manifest_model_id,
        "VERSION": manifest_version,
        "NONCE": nonce,
    }

    _write_json(command_payloads_dir / "ota_begin.json", begin_payload)
    _write_json(command_payloads_dir / "ota_commit.json", commit_payload)
    _write_json(command_payloads_dir / "ota_rollback.json", rollback_payload)
    _write_json(command_payloads_dir / "chunk_manifest.json", chunk_manifest)

    commands_path = output_dir / "EDGEOS_STAGING_CANARY_COMMANDS.md"
    commands_path.write_text(
        _build_commands_markdown(
            track=args.track,
            output_dir=output_dir,
            model_id=manifest_model_id,
            version=manifest_version,
            nonce=nonce,
            source_uri=args.source_uri,
            expected_size=model_size_bytes,
            expected_sha256=model_sha256,
            backend=manifest_backend,
            runtime_suites=runtime_suites,
            forte_test_path=forte_test_path,
        ),
        encoding="utf-8",
    )

    passed = len(failed_checks) == 0
    report_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "freeze_dir": str(freeze_dir),
        "output_dir": str(output_dir),
        "track": args.track,
        "evaluation": {
            "passed": passed,
            "failed_checks": sorted(failed_checks),
        },
        "gates": gates,
        "selected_model": {
            "model_id": manifest_model_id,
            "version": manifest_version,
            "backend": manifest_backend,
            "sha256": model_sha256,
            "size_bytes": model_size_bytes,
            "model_artifact_path": str(model_path),
            "package_dir": str(package_dir),
        },
        "runtime_suites": [
            {
                **asdict(result),
                "log_path": str(result.log_path),
            }
            for result in runtime_results
        ],
        "ota_inputs": {
            "nonce": nonce,
            "source_uri": args.source_uri,
            "command_payload_dir": str(command_payloads_dir),
            "chunk_manifest_path": str(command_payloads_dir / "chunk_manifest.json"),
        },
        "operator_commands_path": str(commands_path),
    }
    report_path = output_dir / "staging_canary_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    return CanaryArtifacts(
        output_dir=output_dir,
        report_path=report_path,
        commands_path=commands_path,
        passed=passed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for staging canary preparation."""
    args = build_parser().parse_args(argv)
    try:
        artifacts = run_canary_from_args(args)
    except Exception as exc:
        print(f"[ERROR] staging canary preparation failed: {exc}", file=sys.stderr)
        return 2

    print(f"output_dir: {artifacts.output_dir}")
    print(f"report: {artifacts.report_path}")
    print(f"commands: {artifacts.commands_path}")
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
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _safe_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default
    return default


def _append_gate(gates: list[dict[str, Any]], failed_checks: list[str], gate: dict[str, Any]) -> None:
    gates.append(gate)
    if not bool(gate.get("passed", False)):
        failed_checks.append(str(gate.get("name", "gate_failed")))


def _check_signoff_gate(path: Path) -> dict[str, Any]:
    name = "freeze_signoff_gate_passed"
    if not path.exists():
        return {"name": name, "passed": False, "details": f"missing_file:{path}"}
    payload = _load_json(path)
    evaluation = payload.get("evaluation")
    passed = bool(evaluation.get("passed", False)) if isinstance(evaluation, dict) else False
    return {
        "name": name,
        "passed": passed,
        "details": {
            "path": str(path),
            "evaluation_passed": passed,
        },
    }


def _check_freeze_manifest(freeze_dir: Path, manifest_path: Path) -> dict[str, Any]:
    name = "freeze_manifest_integrity_passed"
    if not manifest_path.exists():
        return {"name": name, "passed": False, "details": f"missing_file:{manifest_path}"}
    payload = _load_json(manifest_path)
    files = payload.get("files")
    if not isinstance(files, list):
        return {"name": name, "passed": False, "details": "invalid_manifest_files"}

    missing: list[str] = []
    sha_mismatch: list[str] = []
    size_mismatch: list[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        rel = str(entry.get("path", ""))
        expected_sha = str(entry.get("sha256", ""))
        expected_size = int(entry.get("size_bytes", -1))
        abs_path = freeze_dir / rel
        if not abs_path.exists():
            missing.append(rel)
            continue
        if int(abs_path.stat().st_size) != expected_size:
            size_mismatch.append(rel)
        if sha256_file(abs_path) != expected_sha:
            sha_mismatch.append(rel)

    passed = len(missing) == 0 and len(sha_mismatch) == 0 and len(size_mismatch) == 0
    return {
        "name": name,
        "passed": passed,
        "details": {
            "path": str(manifest_path),
            "missing_files": sorted(missing),
            "sha_mismatch": sorted(sha_mismatch),
            "size_mismatch": sorted(size_mismatch),
        },
    }


def _check_package_dir(path: Path) -> dict[str, Any]:
    name = "package_dir_exists"
    required = (
        "inference_config.json",
        "calibration.json",
        "edgeos_model_manifest.json",
        "manifest.json",
        "kpi_report.json",
        "release_gate.json",
    )
    missing = [name for name in required if not (path / name).exists()]
    return {
        "name": name,
        "passed": len(missing) == 0,
        "details": {
            "path": str(path),
            "missing": sorted(missing),
        },
    }


def _resolve_model_artifact_path(
    *,
    package_dir: Path,
    model_manifest: dict[str, Any],
) -> tuple[Path | None, dict[str, Any]]:
    model_file_value = model_manifest.get("model_file")
    if isinstance(model_file_value, str) and model_file_value.strip():
        candidate = package_dir / model_file_value
        passed = candidate.exists()
        return candidate if passed else None, {
            "name": "model_artifact_path_resolved",
            "passed": passed,
            "details": {
                "mode": "manifest.model_file",
                "path": str(candidate),
            },
        }

    fallback_names = ("model_state_dict.pt", "model.tflite")
    for fallback in fallback_names:
        candidate = package_dir / fallback
        if candidate.exists():
            return candidate, {
                "name": "model_artifact_path_resolved",
                "passed": True,
                "details": {
                    "mode": "fallback",
                    "path": str(candidate),
                },
            }

    return None, {
        "name": "model_artifact_path_resolved",
        "passed": False,
        "details": {
            "mode": "missing",
            "path": str(package_dir),
            "expected": ["model_file in edgeos_model_manifest.json", *fallback_names],
        },
    }


def _check_edgeos_backend_compatibility(
    *,
    model_id: str,
    backend: str,
    allow_unsupported_backend: bool,
) -> dict[str, Any]:
    name = "edgeos_backend_compatibility_gate_passed"
    if model_id.startswith("mock."):
        compatible = True
        reason = "mock_model_id_uses_null_backend"
    else:
        compatible = backend in {"tflite", "tflite_flatbuffer"}
        reason = "non_mock_ids_require_tflite_backend"
    passed = compatible or allow_unsupported_backend
    return {
        "name": name,
        "passed": passed,
        "details": {
            "model_id": model_id,
            "backend": backend,
            "compatible": compatible,
            "override_used": allow_unsupported_backend and (not compatible),
            "reason": reason,
        },
    }


def _parse_suites(raw: str) -> tuple[str, ...]:
    suites = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(suites) == 0:
        raise ValueError("runtime_suites cannot be empty")
    return suites


def _run_suite(*, forte_test_path: Path, suite: str, output_dir: Path) -> RuntimeSuiteResult:
    cmd = [
        str(forte_test_path),
        f"--run_test={suite}",
        "--report_level=short",
        "--log_level=warning",
    ]
    started = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started

    log_path = output_dir / f"{suite}.log"
    log_path.write_text(
        "\n".join(
            [
                f"command: {' '.join(cmd)}",
                f"return_code: {completed.returncode}",
                "stdout:",
                completed.stdout,
                "stderr:",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return RuntimeSuiteResult(
        suite=suite,
        passed=completed.returncode == 0,
        return_code=completed.returncode,
        elapsed_seconds=elapsed,
        log_path=log_path,
    )


def _write_chunks(*, source_file: Path, chunk_size_bytes: int, output_dir: Path) -> dict[str, Any]:
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be > 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = source_file.read_bytes()

    chunks: list[dict[str, Any]] = []
    for idx, offset in enumerate(range(0, len(data), chunk_size_bytes)):
        chunk = data[offset : offset + chunk_size_bytes]
        file_name = f"chunk_{idx:04d}.bin"
        chunk_path = output_dir / file_name
        chunk_path.write_bytes(chunk)
        chunks.append(
            {
                "index": idx,
                "path": str(chunk_path),
                "size_bytes": len(chunk),
                "sha256": sha256_file(chunk_path),
            }
        )

    return {
        "source_file": str(source_file),
        "source_size_bytes": len(data),
        "chunk_size_bytes": chunk_size_bytes,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_commands_markdown(
    *,
    track: str,
    output_dir: Path,
    model_id: str,
    version: str,
    nonce: str,
    source_uri: str,
    expected_size: int,
    expected_sha256: str,
    backend: str,
    runtime_suites: tuple[str, ...],
    forte_test_path: Path,
) -> str:
    suites = ",".join(runtime_suites)
    backend_note = (
        "WARNING: backend is not EdgeOS production-compatible for non-mock IDs. "
        "Expected `tflite` or `tflite_flatbuffer` for real deployment."
        if (not model_id.startswith("mock.") and backend not in {"tflite", "tflite_flatbuffer"})
        else "Backend compatibility check passed for deployment path."
    )
    return (
        "# EdgeOS Staging Canary Command Guide\n\n"
        f"Selected track: `{track}`\n"
        f"Model ID: `{model_id}`\n"
        f"Version: `{version}`\n"
        f"Backend: `{backend}`\n"
        f"{backend_note}\n"
        f"Nonce: `{nonce}`\n\n"
        "## 1) Runtime Smoke (already executed by prep CLI, rerun if needed)\n\n"
        "```bash\n"
        f"{forte_test_path} --run_test={suites} --report_level=short --log_level=warning\n"
        "```\n\n"
        "## 2) OTA BEGIN payload\n\n"
        "Use this payload through your BootCtrl MQTT/command path to `ML_OTAUpdate`:\n\n"
        "```json\n"
        + json.dumps(
            {
                "COMMAND": 0,
                "MODEL_ID": model_id,
                "VERSION": version,
                "EXPECTED_SIZE": expected_size,
                "EXPECTED_SHA256": f"sha256:{expected_sha256}",
                "SOURCE_URI": source_uri,
                "TRANSPORT_SECURE": True,
                "NONCE": nonce,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```\n\n"
        "Generated payload files:\n"
        f"- `{output_dir / 'command_payloads' / 'ota_begin.json'}`\n"
        f"- `{output_dir / 'command_payloads' / 'chunk_manifest.json'}`\n"
        f"- `{output_dir / 'command_payloads' / 'ota_commit.json'}`\n"
        f"- `{output_dir / 'command_payloads' / 'ota_rollback.json'}`\n\n"
        "## 3) Canary Rollout Sequence\n\n"
        "1. Send `ota_begin.json` as `ML_OTAUpdate COMMAND=0`.\n"
        "2. Stream chunk files listed in `chunk_manifest.json` as `ML_OTAUpdate COMMAND=1` events.\n"
        "3. Send `ota_commit.json` as `ML_OTAUpdate COMMAND=2`.\n"
        "4. Monitor runtime metrics/alarms for canary window.\n\n"
        "## 4) Rollback Drill (mandatory)\n\n"
        "If any safety/runtime regression occurs, send:\n\n"
        "```json\n"
        + json.dumps({"COMMAND": 4, "MODEL_ID": model_id, "VERSION": version, "NONCE": nonce}, indent=2, sort_keys=True)
        + "\n```\n\n"
        "Expected result: `STATE=ROLLED_BACK` and previous active model restored.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
