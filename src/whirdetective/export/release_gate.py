"""Deterministic release-gate evaluation for EdgeOS rollout controls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReleaseGateTargets:
    """Hard constraints that must pass before edge rollout is allowed."""

    require_kpi_passed: bool = True
    require_artifact_verification: bool = True
    require_signature: bool = False
    max_model_size_bytes: int | None = 16 * 1024 * 1024
    max_model_parameters: int | None = 1_000_000
    max_p95_inference_ms: float | None = 20.0

    def __post_init__(self) -> None:
        if self.max_model_size_bytes is not None and self.max_model_size_bytes <= 0:
            raise ValueError("max_model_size_bytes must be > 0 when set")
        if self.max_model_parameters is not None and self.max_model_parameters <= 0:
            raise ValueError("max_model_parameters must be > 0 when set")
        if self.max_p95_inference_ms is not None and self.max_p95_inference_ms <= 0.0:
            raise ValueError("max_p95_inference_ms must be > 0 when set")


@dataclass(frozen=True, slots=True)
class ReleaseGateEvaluation:
    """Result of applying release-gate targets to one trained package."""

    passed: bool
    failed_checks: tuple[str, ...]
    kpi_passed: bool
    package_verified: bool | None
    signature_verified: bool | None
    model_size_bytes: int
    model_parameter_count: int
    p95_inference_ms: float | None


def evaluate_release_gate(
    *,
    kpi_passed: bool,
    package_verified: bool | None,
    signature_verified: bool | None,
    model_size_bytes: int,
    model_parameter_count: int,
    p95_inference_ms: float | None,
    targets: ReleaseGateTargets,
) -> ReleaseGateEvaluation:
    """Evaluate hard rollout constraints for a candidate package."""
    if model_size_bytes <= 0:
        raise ValueError("model_size_bytes must be > 0")
    if model_parameter_count <= 0:
        raise ValueError("model_parameter_count must be > 0")

    failed_checks: list[str] = []
    if targets.require_kpi_passed and not kpi_passed:
        failed_checks.append("kpi_gate_failed")
    if targets.require_artifact_verification:
        if package_verified is None:
            failed_checks.append("artifact_verification_not_executed")
        elif not package_verified:
            failed_checks.append("artifact_verification_failed")
    if targets.require_signature and signature_verified is not True:
        failed_checks.append("signature_required_but_missing_or_invalid")
    if targets.max_model_size_bytes is not None and model_size_bytes > targets.max_model_size_bytes:
        failed_checks.append(
            "model_size_exceeded:"
            f"{model_size_bytes}>{targets.max_model_size_bytes}"
        )
    if targets.max_model_parameters is not None and model_parameter_count > targets.max_model_parameters:
        failed_checks.append(
            "model_parameter_count_exceeded:"
            f"{model_parameter_count}>{targets.max_model_parameters}"
        )
    if targets.max_p95_inference_ms is not None:
        if p95_inference_ms is None:
            failed_checks.append("latency_measurement_missing")
        elif p95_inference_ms > targets.max_p95_inference_ms:
            failed_checks.append(
                "p95_inference_ms_exceeded:"
                f"{p95_inference_ms:.4f}>{targets.max_p95_inference_ms:.4f}"
            )

    return ReleaseGateEvaluation(
        passed=(len(failed_checks) == 0),
        failed_checks=tuple(failed_checks),
        kpi_passed=kpi_passed,
        package_verified=package_verified,
        signature_verified=signature_verified,
        model_size_bytes=model_size_bytes,
        model_parameter_count=model_parameter_count,
        p95_inference_ms=p95_inference_ms,
    )
