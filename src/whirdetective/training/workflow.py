"""Step 4 baseline diagnosis workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from whirdetective.data import BuiltCanonicalDataset
from whirdetective.evaluation import (
    ModelCard,
    compute_abstention_metrics,
    compute_calibration_metrics,
    compute_classification_metrics,
)
from whirdetective.ml import BaselineBearingCNN
from whirdetective.training.calibration import TemperatureOptimizationResult, optimize_temperature
from whirdetective.training.datasets import CanonicalTensorDataset, canonical_samples_to_dataset
from whirdetective.training.trainer import BaselineTrainer, TrainerConfig, TrainingHistory


@dataclass(frozen=True, slots=True)
class BaselineWorkflowResult:
    """Output artifacts from Step 4 baseline workflow."""

    history: TrainingHistory
    temperature: TemperatureOptimizationResult
    model_card: ModelCard
    model_state_dict: dict[str, torch.Tensor]
    model_input_channels: int
    model_num_classes: int
    train_dataset: CanonicalTensorDataset
    val_dataset: CanonicalTensorDataset
    test_dataset: CanonicalTensorDataset


def run_baseline_workflow(
    *,
    built_dataset: BuiltCanonicalDataset,
    trainer_config: TrainerConfig,
    abstention_threshold: float = 0.7,
    abstention_min_coverage_target: float | None = None,
) -> BaselineWorkflowResult:
    """Train baseline model, calibrate confidence, and produce model card."""
    if abstention_min_coverage_target is not None:
        if abstention_min_coverage_target <= 0.0 or abstention_min_coverage_target > 1.0:
            raise ValueError("abstention_min_coverage_target must be in (0, 1]")

    train_samples = tuple(built_dataset.samples[idx] for idx in built_dataset.split.train_indices)
    val_samples = tuple(built_dataset.samples[idx] for idx in built_dataset.split.val_indices)
    test_samples = tuple(built_dataset.samples[idx] for idx in built_dataset.split.test_indices)
    if not train_samples or not val_samples or not test_samples:
        raise ValueError("split must provide non-empty train/val/test subsets")

    train_dataset = canonical_samples_to_dataset(train_samples)
    val_dataset = canonical_samples_to_dataset(
        val_samples,
        class_to_index=train_dataset.class_to_index,
    )
    test_dataset = canonical_samples_to_dataset(
        test_samples,
        class_to_index=train_dataset.class_to_index,
    )
    num_classes = len(train_dataset.class_names)
    input_channels = int(train_dataset.inputs.shape[1])

    torch.manual_seed(trainer_config.seed)
    model = BaselineBearingCNN(input_channels=input_channels, num_classes=num_classes)
    trainer = BaselineTrainer(model=model, config=trainer_config)
    class_weights = _compute_balanced_class_weights(train_dataset.labels, num_classes=num_classes)
    history = trainer.fit(
        train_inputs=train_dataset.inputs,
        train_labels=train_dataset.labels,
        val_inputs=val_dataset.inputs,
        val_labels=val_dataset.labels,
        class_weights=class_weights,
    )

    val_logits = trainer.predict_logits(val_dataset.inputs)
    temperature = optimize_temperature(val_logits, val_dataset.labels)
    calibrated_val_logits = val_logits / temperature.temperature
    val_probs = torch.softmax(calibrated_val_logits, dim=1).cpu().numpy().astype(np.float64)
    val_labels = val_dataset.labels.cpu().numpy()

    effective_abstention_threshold = float(abstention_threshold)
    if abstention_min_coverage_target is not None:
        effective_abstention_threshold = _select_abstention_threshold_from_validation(
            probabilities=val_probs,
            labels=val_labels,
            min_coverage=abstention_min_coverage_target,
            default_threshold=abstention_threshold,
        )

    test_logits = trainer.predict_logits(test_dataset.inputs)
    calibrated_test_logits = test_logits / temperature.temperature
    test_probs = torch.softmax(calibrated_test_logits, dim=1).cpu().numpy().astype(np.float64)
    test_preds = np.argmax(test_probs, axis=1)
    test_labels = test_dataset.labels.cpu().numpy()

    classification = compute_classification_metrics(
        test_labels,
        test_preds,
        num_classes=num_classes,
    )
    calibration = compute_calibration_metrics(test_probs, test_labels)
    abstention = compute_abstention_metrics(
        test_probs,
        test_labels,
        threshold=effective_abstention_threshold,
    )

    model_card = ModelCard(
        model_name="BaselineBearingCNN",
        created_at_utc=ModelCard.now_timestamp(),
        training_data_fingerprint=built_dataset.fingerprint,
        class_names=train_dataset.class_names,
        classification=classification,
        calibration=calibration,
        abstention=abstention,
        known_limitations=(
            "Trained on public benchmark data; domain shift risk remains high.",
            "Model outputs are advisory; deterministic safety rules retain trip authority.",
        ),
        safety_boundary_statement=(
            "This model must not directly actuate safety trips. Safety supervisor remains authoritative."
        ),
    )

    return BaselineWorkflowResult(
        history=history,
        temperature=temperature,
        model_card=model_card,
        model_state_dict=_clone_state_dict_cpu(trainer.model.state_dict()),
        model_input_channels=input_channels,
        model_num_classes=num_classes,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


def _select_abstention_threshold_from_validation(
    *,
    probabilities: np.ndarray,
    labels: np.ndarray,
    min_coverage: float,
    default_threshold: float,
) -> float:
    if min_coverage <= 0.0 or min_coverage > 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if default_threshold <= 0.0 or default_threshold > 1.0:
        raise ValueError("default_threshold must be in (0, 1]")

    confidences = np.max(probabilities, axis=1)
    candidate_thresholds = sorted(set(float(value) for value in confidences))
    candidate_thresholds.append(float(default_threshold))
    candidate_thresholds = sorted(
        threshold for threshold in set(candidate_thresholds) if threshold <= default_threshold + 1e-12
    )

    if not candidate_thresholds:
        return float(default_threshold)

    best_threshold = float(default_threshold)
    best_selective_accuracy = -1.0
    best_coverage = -1.0
    for threshold in candidate_thresholds:
        metrics = compute_abstention_metrics(probabilities, labels, threshold=threshold)
        if metrics.coverage + 1e-12 < min_coverage:
            continue
        if metrics.selective_accuracy > best_selective_accuracy + 1e-12:
            best_selective_accuracy = metrics.selective_accuracy
            best_coverage = metrics.coverage
            best_threshold = threshold
            continue
        if abs(metrics.selective_accuracy - best_selective_accuracy) <= 1e-12:
            if metrics.coverage > best_coverage + 1e-12:
                best_coverage = metrics.coverage
                best_threshold = threshold
                continue
            if abs(metrics.coverage - best_coverage) <= 1e-12 and threshold > best_threshold:
                best_threshold = threshold

    if best_selective_accuracy < 0.0:
        # No threshold met required coverage; choose highest-coverage relaxed threshold.
        fallback_threshold = float(default_threshold)
        fallback_coverage = -1.0
        for threshold in candidate_thresholds:
            metrics = compute_abstention_metrics(probabilities, labels, threshold=threshold)
            if metrics.coverage > fallback_coverage + 1e-12:
                fallback_coverage = metrics.coverage
                fallback_threshold = threshold
                continue
            if abs(metrics.coverage - fallback_coverage) <= 1e-12 and threshold > fallback_threshold:
                fallback_threshold = threshold
        return float(fallback_threshold)

    return float(best_threshold)


def _clone_state_dict_cpu(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _compute_balanced_class_weights(labels: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.cpu(), minlength=num_classes).to(torch.float32)
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * float(num_classes))
    weights = weights / weights.mean()
    return weights
