"""Tests for classification/calibration/abstention metrics."""

from __future__ import annotations

import numpy as np
import pytest

from whirdetective.evaluation import (
    compute_abstention_metrics,
    compute_calibration_metrics,
    compute_classification_metrics,
)


def test_compute_classification_metrics_outputs_confusion_and_recall() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    preds = np.asarray([0, 1, 0, 0], dtype=np.int64)
    metrics = compute_classification_metrics(labels, preds, num_classes=2)

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.confusion_matrix.shape == (2, 2)
    assert metrics.per_class_recall[0] == pytest.approx(1.0)
    assert metrics.per_class_recall[1] == pytest.approx(0.5)


def test_compute_calibration_metrics_for_perfect_probs() -> None:
    probs = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    metrics = compute_calibration_metrics(probs, labels, num_bins=5)

    assert metrics.expected_calibration_error == pytest.approx(0.0)
    assert metrics.avg_confidence == pytest.approx(1.0)


def test_compute_abstention_metrics_coverage_and_accuracy() -> None:
    probs = np.asarray(
        [
            [0.95, 0.05],
            [0.55, 0.45],
            [0.10, 0.90],
            [0.51, 0.49],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    metrics = compute_abstention_metrics(probs, labels, threshold=0.8)

    assert metrics.coverage == pytest.approx(0.5)
    assert metrics.abstained_fraction == pytest.approx(0.5)
    assert metrics.selective_accuracy == pytest.approx(1.0)
