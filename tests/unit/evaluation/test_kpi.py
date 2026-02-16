"""Tests for Step 4 KPI gate evaluation."""

from __future__ import annotations

import numpy as np

from whirdetective.evaluation import (
    AbstentionMetrics,
    CalibrationMetrics,
    ClassificationMetrics,
    ModelCard,
    Step4KpiTargets,
    evaluate_step4_kpis,
)


def _model_card(
    *,
    accuracy: float,
    recalls: tuple[float, ...],
    ece: float,
    coverage: float,
    selective_accuracy: float,
) -> ModelCard:
    return ModelCard(
        model_name="BaselineBearingCNN",
        created_at_utc=ModelCard.now_timestamp(),
        training_data_fingerprint="fp",
        class_names=("healthy", "inner_race"),
        classification=ClassificationMetrics(
            accuracy=accuracy,
            confusion_matrix=np.asarray([[8, 2], [2, 8]], dtype=np.int64),
            per_class_recall=recalls,
        ),
        calibration=CalibrationMetrics(
            expected_calibration_error=ece,
            avg_confidence=0.85,
        ),
        abstention=AbstentionMetrics(
            threshold=0.7,
            coverage=coverage,
            selective_accuracy=selective_accuracy,
            abstained_fraction=1.0 - coverage,
        ),
        known_limitations=("Domain shift risk",),
        safety_boundary_statement="Safety supervisor retains trip authority.",
    )


def test_evaluate_step4_kpis_passes_when_all_targets_met() -> None:
    card = _model_card(
        accuracy=0.90,
        recalls=(0.85, 0.80),
        ece=0.06,
        coverage=0.88,
        selective_accuracy=0.91,
    )
    result = evaluate_step4_kpis(card, targets=Step4KpiTargets())

    assert result.passed is True
    assert result.failed_checks == ()


def test_evaluate_step4_kpis_reports_failed_checks() -> None:
    card = _model_card(
        accuracy=0.60,
        recalls=(0.65, 0.40),
        ece=0.35,
        coverage=0.50,
        selective_accuracy=0.55,
    )
    result = evaluate_step4_kpis(card, targets=Step4KpiTargets())

    assert result.passed is False
    assert len(result.failed_checks) == 5
