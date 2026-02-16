"""Deterministic dataset build engine for canonical training artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from whirdetective.data.adapters import (
    infer_cwru_label_from_path,
    infer_paderborn_label_from_archive,
    iter_paderborn_mat_payloads,
    list_cwru_mat_files,
    list_paderborn_archives,
    list_paderborn_mat_entries,
    load_cwru_channels,
    load_paderborn_channels_from_mat_payload,
)
from whirdetective.data.contracts import CanonicalTrainingSample
from whirdetective.data.labeling import BearingFaultLabel
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
    exclude_unknown_labels: bool = True

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
class PaderbornBuildConfig:
    """Configuration for deterministic Paderborn canonical dataset build."""

    root_dir: Path
    window_size: int
    step_size: int
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    max_archives: int | None = None
    max_entries_per_archive: int | None = None
    min_distinct_labels_per_split: int | None = None
    split_search_attempts: int = 256
    exclude_unknown_labels: bool = True
    collapse_fault_classes: bool = True
    min_signal_length: int = 1024
    min_length_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0")
        if self.max_archives is not None and self.max_archives <= 0:
            raise ValueError("max_archives must be > 0 when set")
        if self.max_entries_per_archive is not None and self.max_entries_per_archive <= 0:
            raise ValueError("max_entries_per_archive must be > 0 when set")
        if self.min_distinct_labels_per_split is not None and self.min_distinct_labels_per_split <= 0:
            raise ValueError("min_distinct_labels_per_split must be > 0 when set")
        if self.split_search_attempts <= 0:
            raise ValueError("split_search_attempts must be > 0")
        if self.min_signal_length <= 0:
            raise ValueError("min_signal_length must be > 0")
        if self.min_length_ratio <= 0.0 or self.min_length_ratio > 1.0:
            raise ValueError("min_length_ratio must be in (0, 1]")


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
        if config.exclude_unknown_labels and label == BearingFaultLabel.UNKNOWN:
            continue
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
        raise ValueError("No samples were produced from selected CWRU files after filtering")

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


def build_paderborn_canonical_dataset(
    *,
    config: PaderbornBuildConfig,
    projector: SensorSetProjector,
) -> BuiltCanonicalDataset:
    """Build canonical Paderborn dataset windows with deterministic leakage-safe split."""
    archives = list_paderborn_archives(config.root_dir)
    if len(archives) < 3:
        raise ValueError("Paderborn build requires at least 3 .rar archives for train/val/test split")

    selected_archives = _select_archives_for_build(archives, max_archives=config.max_archives)
    if len(selected_archives) < 3:
        raise ValueError("Selected archive set must contain at least 3 .rar files")

    all_samples: list[CanonicalTrainingSample] = []
    all_group_ids: list[str] = []
    source_archives: list[Path] = []
    fingerprint_entries: list[tuple[Path, str]] = []

    for archive_path in selected_archives:
        label = infer_paderborn_label_from_archive(archive_path)
        if config.exclude_unknown_labels and label == BearingFaultLabel.UNKNOWN:
            continue
        if config.collapse_fault_classes and label != BearingFaultLabel.HEALTHY:
            label = BearingFaultLabel.COMBINED

        mat_entries = list_paderborn_mat_entries(archive_path)
        selected_entries = _select_entries_for_build(
            mat_entries,
            max_entries=config.max_entries_per_archive,
        )
        if not selected_entries:
            continue

        payloads = iter_paderborn_mat_payloads(
            archive_path,
            entry_whitelist=set(selected_entries),
        )
        payload_map = {entry_name: payload for entry_name, payload in payloads}
        source_archives.append(archive_path)
        archive_machine_id = f"paderborn_{archive_path.stem.lower()}"

        for entry_name in selected_entries:
            payload = payload_map.get(entry_name)
            if payload is None:
                continue

            channels = load_paderborn_channels_from_mat_payload(
                payload,
                min_signal_length=config.min_signal_length,
                min_length_ratio=config.min_length_ratio,
            )
            run_id = Path(entry_name).stem
            run_samples = build_windowed_canonical_samples(
                dataset="paderborn",
                machine_id=archive_machine_id,
                run_id=run_id,
                label=label,
                channel_signals=channels,
                projector=projector,
                window_size=config.window_size,
                step_size=config.step_size,
            )
            group_key = f"{archive_machine_id}:{run_id}"
            all_samples.extend(run_samples)
            all_group_ids.extend([group_key] * len(run_samples))
            fingerprint_entries.append((archive_path, entry_name))

    if not all_samples:
        raise ValueError("No samples were produced from selected Paderborn archives after filtering")

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
        source_files=tuple(sorted(set(source_archives))),
        fingerprint=_fingerprint_archive_entries(fingerprint_entries),
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


def _select_archives_for_build(
    archives: tuple[Path, ...],
    *,
    max_archives: int | None,
) -> tuple[Path, ...]:
    if max_archives is None or max_archives >= len(archives):
        return archives

    grouped: dict[BearingFaultLabel, list[Path]] = {}
    for archive in archives:
        label = infer_paderborn_label_from_archive(archive)
        grouped.setdefault(label, []).append(archive)

    selected: list[Path] = []
    seen: set[Path] = set()
    label_order = sorted(grouped.keys(), key=lambda label: label.value)
    while len(selected) < max_archives:
        added = False
        for label in label_order:
            candidates = grouped[label]
            if not candidates:
                continue
            candidate = candidates.pop(0)
            if candidate in seen:
                continue
            selected.append(candidate)
            seen.add(candidate)
            added = True
            if len(selected) >= max_archives:
                break
        if not added:
            break
    return tuple(selected)


def _select_entries_for_build(entries: tuple[str, ...], *, max_entries: int | None) -> tuple[str, ...]:
    if max_entries is None or max_entries >= len(entries):
        return entries
    return entries[:max_entries]


def _fingerprint_archive_entries(entries: list[tuple[Path, str]]) -> str:
    if not entries:
        raise ValueError("entries must not be empty for fingerprinting")
    normalized = sorted(
        (
            str(archive_path.resolve()),
            int(archive_path.stat().st_size),
            entry_name,
        )
        for archive_path, entry_name in entries
    )
    digest = hashlib.sha256()
    for archive_path, archive_size, entry_name in normalized:
        digest.update(archive_path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(archive_size).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(entry_name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
