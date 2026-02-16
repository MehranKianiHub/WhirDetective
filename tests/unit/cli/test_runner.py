"""Tests for Step 4 CLI runner artifact emission and exit behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from whirdetective.cli import runner
from whirdetective.data import (
    BearingFaultLabel,
    BuiltCanonicalDataset,
    CanonicalTrainingSample,
    GroupedSplit,
)
from whirdetective.evaluation import AbstentionMetrics, CalibrationMetrics, ClassificationMetrics, ModelCard
from whirdetective.training.calibration import TemperatureOptimizationResult
from whirdetective.training.datasets import CanonicalTensorDataset
from whirdetective.training.trainer import TrainingHistory
from whirdetective.training.workflow import BaselineWorkflowResult


def _dummy_built_dataset() -> BuiltCanonicalDataset:
    split = GroupedSplit(
        train_indices=(0, 1),
        val_indices=(2, 3),
        test_indices=(4, 5),
        train_groups=("g1",),
        val_groups=("g2",),
        test_groups=("g3",),
    )
    features = np.ones((3, 5), dtype=np.float64)
    mask = np.ones((3,), dtype=np.bool_)
    samples = (
        CanonicalTrainingSample(
            dataset="cwru",
            machine_id="m1",
            run_id="r1",
            label=BearingFaultLabel.HEALTHY,
            features=features,
            presence_mask=mask,
        ),
        CanonicalTrainingSample(
            dataset="cwru",
            machine_id="m1",
            run_id="r2",
            label=BearingFaultLabel.INNER_RACE,
            features=features,
            presence_mask=mask,
        ),
        CanonicalTrainingSample(
            dataset="cwru",
            machine_id="m2",
            run_id="r3",
            label=BearingFaultLabel.HEALTHY,
            features=features,
            presence_mask=mask,
        ),
        CanonicalTrainingSample(
            dataset="cwru",
            machine_id="m2",
            run_id="r4",
            label=BearingFaultLabel.INNER_RACE,
            features=features,
            presence_mask=mask,
        ),
        CanonicalTrainingSample(
            dataset="cwru",
            machine_id="m3",
            run_id="r5",
            label=BearingFaultLabel.HEALTHY,
            features=features,
            presence_mask=mask,
        ),
        CanonicalTrainingSample(
            dataset="cwru",
            machine_id="m3",
            run_id="r6",
            label=BearingFaultLabel.INNER_RACE,
            features=features,
            presence_mask=mask,
        ),
    )
    return BuiltCanonicalDataset(
        samples=samples,
        group_ids=("g1", "g1", "g2", "g2", "g3", "g3"),
        split=split,
        source_files=(Path("a.mat"), Path("b.mat")),
        fingerprint="fp-123",
    )


def _dummy_workflow_result(*, pass_kpi: bool) -> BaselineWorkflowResult:
    tensor_dataset = CanonicalTensorDataset(
        inputs=torch.zeros((2, 3, 5), dtype=torch.float32),
        labels=torch.tensor([0, 1], dtype=torch.int64),
        class_names=("healthy", "inner_race"),
        class_to_index={"healthy": 0, "inner_race": 1},
    )
    if pass_kpi:
        accuracy = 0.90
        recalls = (0.85, 0.80)
        ece = 0.07
        coverage = 0.88
        selective_accuracy = 0.91
    else:
        accuracy = 0.60
        recalls = (0.60, 0.50)
        ece = 0.40
        coverage = 0.50
        selective_accuracy = 0.60

    card = ModelCard(
        model_name="BaselineBearingCNN",
        created_at_utc=ModelCard.now_timestamp(),
        training_data_fingerprint="fp-123",
        class_names=("healthy", "inner_race"),
        classification=ClassificationMetrics(
            accuracy=accuracy,
            confusion_matrix=np.asarray([[9, 1], [1, 9]], dtype=np.int64),
            per_class_recall=recalls,
        ),
        calibration=CalibrationMetrics(
            expected_calibration_error=ece,
            avg_confidence=0.86,
        ),
        abstention=AbstentionMetrics(
            threshold=0.7,
            coverage=coverage,
            selective_accuracy=selective_accuracy,
            abstained_fraction=1.0 - coverage,
        ),
        known_limitations=("Domain shift risk",),
        safety_boundary_statement="Deterministic safety supervisor remains authoritative.",
    )

    return BaselineWorkflowResult(
        history=TrainingHistory(
            train_losses=(1.2, 0.7),
            val_losses=(1.3, 0.8),
        ),
        temperature=TemperatureOptimizationResult(
            temperature=1.1,
            nll_before=0.6,
            nll_after=0.55,
        ),
        model_card=card,
        model_state_dict={
            "layer.weight": torch.ones((2, 3), dtype=torch.float32),
            "layer.bias": torch.zeros((2,), dtype=torch.float32),
        },
        model_input_channels=3,
        model_num_classes=2,
        train_dataset=tensor_dataset,
        val_dataset=tensor_dataset,
        test_dataset=tensor_dataset,
    )


def test_main_writes_reports_and_returns_zero(tmp_path: Path, monkeypatch: object) -> None:
    workspace_root = tmp_path
    dataset_root = workspace_root / "data" / "raw" / "cwru"
    dataset_root.mkdir(parents=True)

    monkeypatch.setattr(
        runner,
        "build_cwru_canonical_dataset",
        lambda *, config, projector: _dummy_built_dataset(),
    )
    monkeypatch.setattr(
        runner,
        "run_baseline_workflow",
        lambda *, built_dataset, trainer_config, abstention_threshold, abstention_min_coverage_target: (
            _dummy_workflow_result(pass_kpi=True)
        ),
    )

    exit_code = runner.main(
        [
            "--workspace-root",
            str(workspace_root),
            "--dataset-root",
            "data/raw/cwru",
            "--output-dir",
            "artifacts/step4-cli",
        ]
    )
    assert exit_code == 0

    output_dir = workspace_root / "artifacts" / "step4-cli"
    assert (output_dir / "model_card.json").exists()
    assert (output_dir / "kpi_report.json").exists()
    assert (output_dir / "run_report.json").exists()
    assert (output_dir / "release_gate.json").exists()
    assert (output_dir / "model_state_dict.pt").exists()
    assert (output_dir / "inference_config.json").exists()
    assert (output_dir / "calibration.json").exists()
    assert (output_dir / "edgeos_model_manifest.json").exists()
    assert (output_dir / "manifest.json").exists()

    kpi_payload = json.loads((output_dir / "kpi_report.json").read_text(encoding="utf-8"))
    assert kpi_payload["evaluation"]["passed"] is True
    release_payload = json.loads((output_dir / "release_gate.json").read_text(encoding="utf-8"))
    assert release_payload["evaluation"]["passed"] is True


def test_main_returns_one_when_fail_on_kpi_enabled(tmp_path: Path, monkeypatch: object) -> None:
    workspace_root = tmp_path
    dataset_root = workspace_root / "data" / "raw" / "cwru"
    dataset_root.mkdir(parents=True)

    monkeypatch.setattr(
        runner,
        "build_cwru_canonical_dataset",
        lambda *, config, projector: _dummy_built_dataset(),
    )
    monkeypatch.setattr(
        runner,
        "run_baseline_workflow",
        lambda *, built_dataset, trainer_config, abstention_threshold, abstention_min_coverage_target: (
            _dummy_workflow_result(pass_kpi=False)
        ),
    )

    exit_code = runner.main(
        [
            "--workspace-root",
            str(workspace_root),
            "--dataset-root",
            "data/raw/cwru",
            "--output-dir",
            "artifacts/step4-cli-fail",
            "--fail-on-kpi",
        ]
    )
    assert exit_code == 1


def test_main_require_signature_fails_without_signing_key(
    tmp_path: Path, monkeypatch: object
) -> None:
    workspace_root = tmp_path
    dataset_root = workspace_root / "data" / "raw" / "cwru"
    dataset_root.mkdir(parents=True)

    monkeypatch.setattr(
        runner,
        "build_cwru_canonical_dataset",
        lambda *, config, projector: _dummy_built_dataset(),
    )
    monkeypatch.setattr(
        runner,
        "run_baseline_workflow",
        lambda *, built_dataset, trainer_config, abstention_threshold, abstention_min_coverage_target: (
            _dummy_workflow_result(pass_kpi=True)
        ),
    )

    exit_code = runner.main(
        [
            "--workspace-root",
            str(workspace_root),
            "--dataset-root",
            "data/raw/cwru",
            "--output-dir",
            "artifacts/step4-cli-signature-fail",
            "--require-signature",
        ]
    )
    assert exit_code == 2


def test_main_require_signature_succeeds_with_signing_key(
    tmp_path: Path, monkeypatch: object
) -> None:
    workspace_root = tmp_path
    dataset_root = workspace_root / "data" / "raw" / "cwru"
    dataset_root.mkdir(parents=True)

    monkeypatch.setenv("WHIRDETECTIVE_MANIFEST_SIGNING_KEY", "unit-test-key")
    monkeypatch.setattr(
        runner,
        "build_cwru_canonical_dataset",
        lambda *, config, projector: _dummy_built_dataset(),
    )
    monkeypatch.setattr(
        runner,
        "run_baseline_workflow",
        lambda *, built_dataset, trainer_config, abstention_threshold, abstention_min_coverage_target: (
            _dummy_workflow_result(pass_kpi=True)
        ),
    )

    exit_code = runner.main(
        [
            "--workspace-root",
            str(workspace_root),
            "--dataset-root",
            "data/raw/cwru",
            "--output-dir",
            "artifacts/step4-cli-signature-ok",
            "--require-signature",
        ]
    )
    assert exit_code == 0
    output_dir = workspace_root / "artifacts" / "step4-cli-signature-ok"
    assert (output_dir / "manifest.sig").exists()


def test_main_returns_one_when_fail_on_release_gate_enabled(
    tmp_path: Path, monkeypatch: object
) -> None:
    workspace_root = tmp_path
    dataset_root = workspace_root / "data" / "raw" / "cwru"
    dataset_root.mkdir(parents=True)

    monkeypatch.setattr(
        runner,
        "build_cwru_canonical_dataset",
        lambda *, config, projector: _dummy_built_dataset(),
    )
    monkeypatch.setattr(
        runner,
        "run_baseline_workflow",
        lambda *, built_dataset, trainer_config, abstention_threshold, abstention_min_coverage_target: (
            _dummy_workflow_result(pass_kpi=True)
        ),
    )

    exit_code = runner.main(
        [
            "--workspace-root",
            str(workspace_root),
            "--dataset-root",
            "data/raw/cwru",
            "--output-dir",
            "artifacts/step4-cli-release-fail",
            "--release-max-model-size-bytes",
            "1",
            "--fail-on-release-gate",
        ]
    )
    assert exit_code == 1
