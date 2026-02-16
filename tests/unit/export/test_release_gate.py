"""Tests for deterministic release-gate rollout policy checks."""

from __future__ import annotations

from whirdetective.export import ReleaseGateTargets, evaluate_release_gate


def test_evaluate_release_gate_passes_when_all_constraints_hold() -> None:
    evaluation = evaluate_release_gate(
        kpi_passed=True,
        package_verified=True,
        signature_verified=True,
        model_size_bytes=1024,
        model_parameter_count=20_000,
        p95_inference_ms=5.0,
        targets=ReleaseGateTargets(
            require_kpi_passed=True,
            require_artifact_verification=True,
            require_signature=True,
            max_model_size_bytes=2048,
            max_model_parameters=30_000,
            max_p95_inference_ms=10.0,
        ),
    )
    assert evaluation.passed is True
    assert evaluation.failed_checks == ()


def test_evaluate_release_gate_fails_for_missing_verification_and_budget_exceedance() -> None:
    evaluation = evaluate_release_gate(
        kpi_passed=False,
        package_verified=None,
        signature_verified=None,
        model_size_bytes=4096,
        model_parameter_count=50_000,
        p95_inference_ms=25.0,
        targets=ReleaseGateTargets(
            require_kpi_passed=True,
            require_artifact_verification=True,
            require_signature=True,
            max_model_size_bytes=2048,
            max_model_parameters=30_000,
            max_p95_inference_ms=20.0,
        ),
    )
    assert evaluation.passed is False
    assert "kpi_gate_failed" in evaluation.failed_checks
    assert "artifact_verification_not_executed" in evaluation.failed_checks
    assert "signature_required_but_missing_or_invalid" in evaluation.failed_checks
    assert "model_size_exceeded:4096>2048" in evaluation.failed_checks
    assert "model_parameter_count_exceeded:50000>30000" in evaluation.failed_checks
    assert "p95_inference_ms_exceeded:25.0000>20.0000" in evaluation.failed_checks
