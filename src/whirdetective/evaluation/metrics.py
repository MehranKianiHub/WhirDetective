"""Classification, calibration, and abstention evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Core classification metrics and confusion matrix."""

    accuracy: float
    confusion_matrix: IntArray
    per_class_recall: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Calibration summary metrics."""

    expected_calibration_error: float
    avg_confidence: float


@dataclass(frozen=True, slots=True)
class AbstentionMetrics:
    """Selective prediction metrics under confidence thresholding."""

    threshold: float
    coverage: float
    selective_accuracy: float
    abstained_fraction: float


def compute_classification_metrics(
    labels: npt.ArrayLike,
    predictions: npt.ArrayLike,
    *,
    num_classes: int,
) -> ClassificationMetrics:
    """Compute confusion matrix, per-class recall, and accuracy."""
    y_true = _to_int_vector(labels, name="labels")
    y_pred = _to_int_vector(predictions, name="predictions")
    if y_true.shape != y_pred.shape:
        raise ValueError("labels and predictions must have same shape")
    if num_classes <= 1:
        raise ValueError("num_classes must be > 1")

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if truth < 0 or truth >= num_classes:
            raise ValueError("label index out of range")
        if pred < 0 or pred >= num_classes:
            raise ValueError("prediction index out of range")
        matrix[truth, pred] += 1

    recalls: list[float] = []
    for class_idx in range(num_classes):
        row_sum = int(np.sum(matrix[class_idx, :]))
        recalls.append(0.0 if row_sum == 0 else float(matrix[class_idx, class_idx] / row_sum))

    accuracy = float(np.mean(y_true == y_pred))
    return ClassificationMetrics(
        accuracy=accuracy,
        confusion_matrix=matrix,
        per_class_recall=tuple(recalls),
    )


def compute_calibration_metrics(
    probabilities: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    num_bins: int = 15,
) -> CalibrationMetrics:
    """Compute ECE and average confidence for probabilistic predictions."""
    probs = _to_prob_matrix(probabilities)
    y_true = _to_int_vector(labels, name="labels")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities and labels batch size must match")
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for bin_idx in range(num_bins):
        low, high = edges[bin_idx], edges[bin_idx + 1]
        if bin_idx == num_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)
        if not np.any(in_bin):
            continue
        bin_acc = float(np.mean(correctness[in_bin]))
        bin_conf = float(np.mean(confidences[in_bin]))
        bin_frac = float(np.mean(in_bin.astype(np.float64)))
        ece += abs(bin_acc - bin_conf) * bin_frac

    return CalibrationMetrics(
        expected_calibration_error=float(ece),
        avg_confidence=float(np.mean(confidences)),
    )


def compute_abstention_metrics(
    probabilities: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    threshold: float,
) -> AbstentionMetrics:
    """Compute selective accuracy and coverage for confidence-threshold abstention."""
    probs = _to_prob_matrix(probabilities)
    y_true = _to_int_vector(labels, name="labels")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities and labels batch size must match")
    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in (0, 1]")

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accepted = confidences >= threshold
    accepted_count = int(np.sum(accepted))
    total = int(y_true.shape[0])

    if accepted_count == 0:
        selective_accuracy = 0.0
    else:
        selective_accuracy = float(np.mean(predictions[accepted] == y_true[accepted]))
    coverage = float(accepted_count / total)
    abstained_fraction = 1.0 - coverage

    return AbstentionMetrics(
        threshold=threshold,
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        abstained_fraction=abstained_fraction,
    )


def _to_int_vector(values: npt.ArrayLike, *, name: str) -> IntArray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return array


def _to_prob_matrix(values: npt.ArrayLike) -> FloatArray:
    probs = np.asarray(values, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be 2D [batch, classes]")
    if probs.shape[1] <= 1:
        raise ValueError("probabilities must have >1 class column")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must be in [0, 1]")

    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, np.ones_like(row_sums), atol=1e-6):
        raise ValueError("each probability row must sum to 1")
    return probs
