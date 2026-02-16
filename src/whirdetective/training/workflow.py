"""Step 4 baseline diagnosis workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass

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
    train_dataset: CanonicalTensorDataset
    val_dataset: CanonicalTensorDataset
    test_dataset: CanonicalTensorDataset


def run_baseline_workflow(
    *,
    built_dataset: BuiltCanonicalDataset,
    trainer_config: TrainerConfig,
    abstention_threshold: float = 0.7,
) -> BaselineWorkflowResult:
    """Train baseline model, calibrate confidence, and produce model card."""
    train_samples = tuple(built_dataset.samples[idx] for idx in built_dataset.split.train_indices)
    val_samples = tuple(built_dataset.samples[idx] for idx in built_dataset.split.val_indices)
    test_samples = tuple(built_dataset.samples[idx] for idx in built_dataset.split.test_indices)
    if not train_samples or not val_samples or not test_samples:
        raise ValueError("split must provide non-empty train/val/test subsets")

    train_dataset = canonical_samples_to_dataset(train_samples)
    val_dataset = canonical_samples_to_dataset(val_samples)
    test_dataset = canonical_samples_to_dataset(test_samples)

    _assert_class_mapping_consistency(train_dataset, val_dataset, test_dataset)
    num_classes = len(train_dataset.class_names)
    input_channels = int(train_dataset.inputs.shape[1])

    torch.manual_seed(trainer_config.seed)
    model = BaselineBearingCNN(input_channels=input_channels, num_classes=num_classes)
    trainer = BaselineTrainer(model=model, config=trainer_config)
    history = trainer.fit(
        train_inputs=train_dataset.inputs,
        train_labels=train_dataset.labels,
        val_inputs=val_dataset.inputs,
        val_labels=val_dataset.labels,
    )

    val_logits = trainer.predict_logits(val_dataset.inputs)
    temperature = optimize_temperature(val_logits, val_dataset.labels)

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
        threshold=abstention_threshold,
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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


def _assert_class_mapping_consistency(
    train_dataset: CanonicalTensorDataset,
    val_dataset: CanonicalTensorDataset,
    test_dataset: CanonicalTensorDataset,
) -> None:
    expected = train_dataset.class_names
    if val_dataset.class_names != expected or test_dataset.class_names != expected:
        raise ValueError("train/val/test class mappings must match")
