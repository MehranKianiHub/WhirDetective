"""Tests for model card serialization."""

from __future__ import annotations

import numpy as np

from whirdetective.evaluation import (
    AbstentionMetrics,
    CalibrationMetrics,
    ClassificationMetrics,
    ModelCard,
    model_card_to_jsonable,
)


def test_model_card_to_jsonable_converts_confusion_matrix() -> None:
    card = ModelCard(
        model_name="BaselineBearingCNN",
        created_at_utc=ModelCard.now_timestamp(),
        training_data_fingerprint="abc123",
        class_names=("healthy", "inner_race"),
        classification=ClassificationMetrics(
            accuracy=0.8,
            confusion_matrix=np.asarray([[8, 2], [2, 8]], dtype=np.int64),
            per_class_recall=(0.8, 0.8),
        ),
        calibration=CalibrationMetrics(expected_calibration_error=0.03, avg_confidence=0.85),
        abstention=AbstentionMetrics(
            threshold=0.7,
            coverage=0.9,
            selective_accuracy=0.88,
            abstained_fraction=0.1,
        ),
        known_limitations=("Domain shift risk",),
        safety_boundary_statement="Deterministic supervisor retains trip authority.",
    )

    payload = model_card_to_jsonable(card)
    assert payload["model_name"] == "BaselineBearingCNN"
    assert payload["classification"]["confusion_matrix"] == [[8, 2], [2, 8]]
