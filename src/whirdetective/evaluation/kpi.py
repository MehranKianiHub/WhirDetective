"""KPI gate evaluation for Step 4 baseline readiness decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from whirdetective.evaluation.model_card import ModelCard


@dataclass(frozen=True, slots=True)
class Step4KpiTargets:
    """Thresholds used to decide if Step 4 benchmark quality is acceptable."""

    min_accuracy: float = 0.80
    min_macro_recall: float = 0.75
    max_expected_calibration_error: float = 0.20
    min_coverage: float = 0.70
    min_selective_accuracy: float = 0.80

    def __post_init__(self) -> None:
        _assert_unit_interval(self.min_accuracy, field_name="min_accuracy")
        _assert_unit_interval(self.min_macro_recall, field_name="min_macro_recall")
        _assert_unit_interval(
            self.max_expected_calibration_error,
            field_name="max_expected_calibration_error",
        )
        _assert_unit_interval(self.min_coverage, field_name="min_coverage")
        _assert_unit_interval(self.min_selective_accuracy, field_name="min_selective_accuracy")


@dataclass(frozen=True, slots=True)
class Step4KpiEvaluation:
    """Result of applying KPI targets to model-card metrics."""

    passed: bool
    failed_checks: tuple[str, ...]
    accuracy: float
    macro_recall: float
    expected_calibration_error: float
    coverage: float
    selective_accuracy: float


def evaluate_step4_kpis(card: ModelCard, *, targets: Step4KpiTargets) -> Step4KpiEvaluation:
    """Evaluate model-card metrics against Step 4 KPI thresholds."""
    if len(card.classification.per_class_recall) == 0:
        raise ValueError("per_class_recall must not be empty")

    accuracy = float(card.classification.accuracy)
    macro_recall = float(np.mean(np.asarray(card.classification.per_class_recall, dtype=np.float64)))
    expected_calibration_error = float(card.calibration.expected_calibration_error)
    coverage = float(card.abstention.coverage)
    selective_accuracy = float(card.abstention.selective_accuracy)

    failed_checks: list[str] = []
    if accuracy < targets.min_accuracy:
        failed_checks.append(
            f"accuracy {accuracy:.4f} < min_accuracy {targets.min_accuracy:.4f}"
        )
    if macro_recall < targets.min_macro_recall:
        failed_checks.append(
            f"macro_recall {macro_recall:.4f} < min_macro_recall {targets.min_macro_recall:.4f}"
        )
    if expected_calibration_error > targets.max_expected_calibration_error:
        failed_checks.append(
            "expected_calibration_error "
            f"{expected_calibration_error:.4f} > max_expected_calibration_error "
            f"{targets.max_expected_calibration_error:.4f}"
        )
    if coverage < targets.min_coverage:
        failed_checks.append(
            f"coverage {coverage:.4f} < min_coverage {targets.min_coverage:.4f}"
        )
    if selective_accuracy < targets.min_selective_accuracy:
        failed_checks.append(
            "selective_accuracy "
            f"{selective_accuracy:.4f} < min_selective_accuracy "
            f"{targets.min_selective_accuracy:.4f}"
        )

    return Step4KpiEvaluation(
        passed=(len(failed_checks) == 0),
        failed_checks=tuple(failed_checks),
        accuracy=accuracy,
        macro_recall=macro_recall,
        expected_calibration_error=expected_calibration_error,
        coverage=coverage,
        selective_accuracy=selective_accuracy,
    )


def _assert_unit_interval(value: float, *, field_name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
