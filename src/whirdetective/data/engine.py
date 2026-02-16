"""Deterministic dataset build engine for canonical training artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whirdetective.data.adapters import (
    infer_cwru_label_from_path,
    list_cwru_mat_files,
    load_cwru_channels,
)
from whirdetective.data.contracts import CanonicalTrainingSample
from whirdetective.data.pipeline import build_windowed_canonical_samples
from whirdetective.data.splitting import GroupedSplit, assert_group_isolation, split_by_group
from whirdetective.data.versioning import dataset_fingerprint
from whirdetective.ml import SensorSetProjector


@dataclass(frozen=True, slots=True)
class CwruBuildConfig:
    """Configuration for deterministic CWRU canonical dataset build."""

    root_dir: Path
    window_size: int
    step_size: int
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    max_files: int | None = None
    min_distinct_labels_per_split: int | None = None
    split_search_attempts: int = 256

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0")
        if self.max_files is not None and self.max_files <= 0:
            raise ValueError("max_files must be > 0 when set")
        if self.min_distinct_labels_per_split is not None and self.min_distinct_labels_per_split <= 0:
            raise ValueError("min_distinct_labels_per_split must be > 0 when set")
        if self.split_search_attempts <= 0:
            raise ValueError("split_search_attempts must be > 0")


@dataclass(frozen=True, slots=True)
class BuiltCanonicalDataset:
    """Deterministic built dataset with split metadata and fingerprint."""

    samples: tuple[CanonicalTrainingSample, ...]
    group_ids: tuple[str, ...]
    split: GroupedSplit
    source_files: tuple[Path, ...]
    fingerprint: str


def build_cwru_canonical_dataset(
    *,
    config: CwruBuildConfig,
    projector: SensorSetProjector,
) -> BuiltCanonicalDataset:
    """Build canonical CWRU dataset windows with deterministic leakage-safe split."""
    mat_files = list_cwru_mat_files(config.root_dir)
    if len(mat_files) < 3:
        raise ValueError("CWRU build requires at least 3 .mat files for train/val/test split")

    selected_files = _select_files_for_build(mat_files, max_files=config.max_files)
    if len(selected_files) < 3:
        raise ValueError("Selected file set must contain at least 3 .mat files")

    all_samples: list[CanonicalTrainingSample] = []
    all_group_ids: list[str] = []
    source_files: list[Path] = []

    for file_path in selected_files:
        channels = load_cwru_channels(file_path)
        label = infer_cwru_label_from_path(file_path)
        machine_id = f"cwru_{file_path.parent.name}"
        run_id = file_path.stem

        run_samples = build_windowed_canonical_samples(
            dataset="cwru",
            machine_id=machine_id,
            run_id=run_id,
            label=label,
            channel_signals=channels,
            projector=projector,
            window_size=config.window_size,
            step_size=config.step_size,
        )
        group_key = f"{machine_id}:{run_id}"
        all_samples.extend(run_samples)
        all_group_ids.extend([group_key] * len(run_samples))
        source_files.append(file_path)

    if not all_samples:
        raise ValueError("No samples were produced from selected CWRU files")

    split = split_by_group(
        tuple(all_group_ids),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
        labels=tuple(sample.label.value for sample in all_samples),
        min_distinct_labels_per_split=config.min_distinct_labels_per_split,
        search_attempts=config.split_search_attempts,
    )
    assert_group_isolation(tuple(all_group_ids), split)

    return BuiltCanonicalDataset(
        samples=tuple(all_samples),
        group_ids=tuple(all_group_ids),
        split=split,
        source_files=tuple(source_files),
        fingerprint=dataset_fingerprint(source_files),
    )


def _select_files_for_build(mat_files: tuple[Path, ...], *, max_files: int | None) -> tuple[Path, ...]:
    if max_files is None or max_files >= len(mat_files):
        return mat_files

    selected = list(mat_files[:max_files])
    if max_files <= 1:
        return tuple(selected)

    selected_labels = {infer_cwru_label_from_path(path) for path in selected}
    if len(selected_labels) >= 2:
        return tuple(selected)

    first_label = infer_cwru_label_from_path(selected[0])
    replacement = next(
        (
            path
            for path in mat_files[max_files:]
            if infer_cwru_label_from_path(path) != first_label
        ),
        None,
    )
    if replacement is not None:
        selected[-1] = replacement
    return tuple(selected)
