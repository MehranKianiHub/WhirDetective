"""Tests for baseline Step 4 workflow orchestration."""

from __future__ import annotations

import numpy as np

from whirdetective.data import BearingFaultLabel, BuiltCanonicalDataset, CanonicalTrainingSample, GroupedSplit
from whirdetective.training import TrainerConfig, run_baseline_workflow


def _sample(label: BearingFaultLabel, seed: int) -> CanonicalTrainingSample:
    rng = np.random.default_rng(seed)
    features = rng.normal(loc=(1.0 if label == BearingFaultLabel.INNER_RACE else -1.0), scale=0.1, size=(3, 64))
    mask = np.ones((3,), dtype=np.bool_)
    return CanonicalTrainingSample(
        dataset="cwru",
        machine_id="m1",
        run_id=f"r{seed}",
        label=label,
        features=features.astype(np.float64),
        presence_mask=mask,
    )


def test_run_baseline_workflow_returns_model_card_and_metrics() -> None:
    samples = (
        _sample(BearingFaultLabel.HEALTHY, 1),
        _sample(BearingFaultLabel.INNER_RACE, 2),
        _sample(BearingFaultLabel.HEALTHY, 3),
        _sample(BearingFaultLabel.INNER_RACE, 4),
        _sample(BearingFaultLabel.HEALTHY, 5),
        _sample(BearingFaultLabel.INNER_RACE, 6),
        _sample(BearingFaultLabel.HEALTHY, 7),
        _sample(BearingFaultLabel.INNER_RACE, 8),
    )
    split = GroupedSplit(
        train_indices=(0, 1, 2, 3),
        val_indices=(4, 5),
        test_indices=(6, 7),
        train_groups=("g1", "g2"),
        val_groups=("g3",),
        test_groups=("g4",),
    )
    dataset = BuiltCanonicalDataset(
        samples=samples,
        group_ids=("g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"),
        split=split,
        source_files=tuple(),
        fingerprint="fp-test-001",
    )

    result = run_baseline_workflow(
        built_dataset=dataset,
        trainer_config=TrainerConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=9),
        abstention_threshold=0.6,
    )

    assert result.temperature.temperature > 0.0
    assert result.model_card.training_data_fingerprint == "fp-test-001"
    assert result.model_card.classification.confusion_matrix.shape == (2, 2)


def test_run_baseline_workflow_tunes_abstention_threshold_from_validation() -> None:
    samples = (
        _sample(BearingFaultLabel.HEALTHY, 11),
        _sample(BearingFaultLabel.INNER_RACE, 12),
        _sample(BearingFaultLabel.HEALTHY, 13),
        _sample(BearingFaultLabel.INNER_RACE, 14),
        _sample(BearingFaultLabel.HEALTHY, 15),
        _sample(BearingFaultLabel.INNER_RACE, 16),
        _sample(BearingFaultLabel.HEALTHY, 17),
        _sample(BearingFaultLabel.INNER_RACE, 18),
    )
    split = GroupedSplit(
        train_indices=(0, 1, 2, 3),
        val_indices=(4, 5),
        test_indices=(6, 7),
        train_groups=("g1", "g2"),
        val_groups=("g3",),
        test_groups=("g4",),
    )
    dataset = BuiltCanonicalDataset(
        samples=samples,
        group_ids=("g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"),
        split=split,
        source_files=tuple(),
        fingerprint="fp-test-002",
    )

    result = run_baseline_workflow(
        built_dataset=dataset,
        trainer_config=TrainerConfig(epochs=2, batch_size=2, learning_rate=1e-3, seed=9),
        abstention_threshold=0.95,
        abstention_min_coverage_target=0.9,
    )

    assert result.model_card.abstention.coverage >= 0.9
    assert result.model_card.abstention.threshold <= 0.95
