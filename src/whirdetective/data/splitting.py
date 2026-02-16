"""Leakage-safe grouped splitting utilities for dataset engine."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from math import floor
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class GroupedSplit:
    """Grouped split indices and group memberships for auditability."""

    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


def split_by_group(
    group_ids: Sequence[str],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    labels: Sequence[str] | None = None,
    min_distinct_labels_per_split: int | None = None,
    required_labels_per_split: Sequence[str] | None = None,
    search_attempts: int = 256,
) -> GroupedSplit:
    """Split sample indices by group id so groups never span multiple splits.

    If ``labels`` are provided, the splitter can enforce class-coverage constraints
    and search for a split with lower label-distribution drift.
    """
    if not group_ids:
        raise ValueError("group_ids must not be empty")
    _validate_ratios(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    if labels is not None and len(labels) != len(group_ids):
        raise ValueError("labels length must match group_ids length")
    if min_distinct_labels_per_split is not None and labels is None:
        raise ValueError("labels are required when min_distinct_labels_per_split is set")
    if required_labels_per_split is not None and labels is None:
        raise ValueError("labels are required when required_labels_per_split is set")
    if min_distinct_labels_per_split is not None and min_distinct_labels_per_split <= 0:
        raise ValueError("min_distinct_labels_per_split must be > 0")
    if search_attempts <= 0:
        raise ValueError("search_attempts must be > 0")

    unique_groups = sorted(set(group_ids))
    split_group_counts = _allocate_group_counts(
        total_groups=len(unique_groups),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    if labels is None:
        rng = np.random.default_rng(seed)
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        train_groups, val_groups, test_groups = _slice_group_partitions(shuffled_groups, split_group_counts)
        return _build_split_from_group_partitions(
            group_ids=group_ids,
            train_groups=train_groups,
            val_groups=val_groups,
            test_groups=test_groups,
        )

    return _split_by_group_stratified_search(
        group_ids=group_ids,
        labels=labels,
        split_group_counts=split_group_counts,
        seed=seed,
        min_distinct_labels_per_split=min_distinct_labels_per_split,
        required_labels_per_split=required_labels_per_split,
        search_attempts=search_attempts,
    )


def assert_group_isolation(group_ids: Sequence[str], split: GroupedSplit) -> None:
    """Raise if any group appears in more than one split."""
    if len(group_ids) == 0:
        raise ValueError("group_ids must not be empty")

    train_groups = {group_ids[idx] for idx in split.train_indices}
    val_groups = {group_ids[idx] for idx in split.val_indices}
    test_groups = {group_ids[idx] for idx in split.test_indices}

    if train_groups.intersection(val_groups):
        raise ValueError("Leakage detected between train and val groups")
    if train_groups.intersection(test_groups):
        raise ValueError("Leakage detected between train and test groups")
    if val_groups.intersection(test_groups):
        raise ValueError("Leakage detected between val and test groups")


def _split_by_group_stratified_search(
    *,
    group_ids: Sequence[str],
    labels: Sequence[str],
    split_group_counts: tuple[int, int, int],
    seed: int,
    min_distinct_labels_per_split: int | None,
    required_labels_per_split: Sequence[str] | None,
    search_attempts: int,
) -> GroupedSplit:
    unique_groups = sorted(set(group_ids))
    label_names = sorted(set(labels))
    if min_distinct_labels_per_split is not None and min_distinct_labels_per_split > len(label_names):
        raise ValueError(
            "min_distinct_labels_per_split exceeds number of available labels "
            f"({len(label_names)})"
        )
    required_labels = _normalize_required_labels(
        required_labels_per_split=required_labels_per_split,
        available_labels=tuple(label_names),
    )

    group_label_counts = _group_label_counts(group_ids=group_ids, labels=labels)
    global_label_distribution = _normalize_label_counts(
        _aggregate_label_counts(tuple(group_label_counts.values()))
    )
    min_required = 1 if min_distinct_labels_per_split is None else min_distinct_labels_per_split

    best_split: GroupedSplit | None = None
    best_score: float | None = None
    best_min_distinct = -1
    best_required_present = 0

    for offset in count(0):
        if offset >= search_attempts:
            break
        rng = np.random.default_rng(seed + offset)
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        train_groups, val_groups, test_groups = _slice_group_partitions(shuffled_groups, split_group_counts)
        split_groups = (train_groups, val_groups, test_groups)
        split_label_counts = _split_label_counts(split_groups, group_label_counts=group_label_counts)
        distinct_per_split = tuple(sum(1 for count_value in counts.values() if count_value > 0) for counts in split_label_counts)
        current_min_distinct = min(distinct_per_split)
        if current_min_distinct < min_required:
            best_min_distinct = max(best_min_distinct, current_min_distinct)
            continue
        required_present = _required_labels_presence_count(
            split_label_counts=split_label_counts,
            required_labels=required_labels,
        )
        if required_present < len(required_labels) * 3:
            best_required_present = max(best_required_present, required_present)
            continue

        score = _distribution_drift_score(
            split_label_counts=split_label_counts,
            global_distribution=global_label_distribution,
        )
        if best_score is None or score < best_score:
            best_split = _build_split_from_group_partitions(
                group_ids=group_ids,
                train_groups=train_groups,
                val_groups=val_groups,
                test_groups=test_groups,
            )
            best_score = score

    if best_split is None:
        required_text = ""
        if required_labels:
            required_text = (
                f", required labels in each split: {list(required_labels)} "
                f"(best satisfied {best_required_present} of {len(required_labels) * 3})"
            )
        raise ValueError(
            "Could not find grouped split satisfying class coverage constraints "
            f"(required min distinct labels per split: {min_required}, "
            f"best achieved: {best_min_distinct}{required_text})"
        )
    return best_split


def _validate_ratios(*, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    for name, value in (
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


def _allocate_group_counts(
    *,
    total_groups: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if total_groups < 3:
        raise ValueError("At least 3 unique groups are required for train/val/test split")

    raw = (
        total_groups * train_ratio,
        total_groups * val_ratio,
        total_groups * test_ratio,
    )
    counts = [floor(value) for value in raw]
    remainder = total_groups - sum(counts)

    fractional_order = sorted(
        range(3),
        key=lambda idx: raw[idx] - counts[idx],
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        counts[idx] += 1

    # Guarantee at least one group in each split.
    for idx in range(3):
        if counts[idx] == 0:
            largest_idx = int(np.argmax(np.asarray(counts, dtype=np.int64)))
            counts[largest_idx] -= 1
            counts[idx] += 1

    return counts[0], counts[1], counts[2]


def _slice_group_partitions(
    shuffled_groups: Sequence[str],
    split_group_counts: tuple[int, int, int],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    train_count, val_count, test_count = split_group_counts
    train_groups = tuple(shuffled_groups[:train_count])
    val_groups = tuple(shuffled_groups[train_count : train_count + val_count])
    test_groups = tuple(shuffled_groups[train_count + val_count : train_count + val_count + test_count])
    return train_groups, val_groups, test_groups


def _build_split_from_group_partitions(
    *,
    group_ids: Sequence[str],
    train_groups: tuple[str, ...],
    val_groups: tuple[str, ...],
    test_groups: tuple[str, ...],
) -> GroupedSplit:
    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for idx, group in enumerate(group_ids):
        if group in train_set:
            train_indices.append(idx)
        elif group in val_set:
            val_indices.append(idx)
        elif group in test_set:
            test_indices.append(idx)
        else:
            raise RuntimeError(f"Group {group!r} was not assigned to any split")

    return GroupedSplit(
        train_indices=tuple(train_indices),
        val_indices=tuple(val_indices),
        test_indices=tuple(test_indices),
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
    )


def _group_label_counts(
    *,
    group_ids: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    group_label_counts: dict[str, dict[str, int]] = {}
    for group, label in zip(group_ids, labels):
        label_counts = group_label_counts.setdefault(group, {})
        label_counts[label] = label_counts.get(label, 0) + 1
    return group_label_counts


def _aggregate_label_counts(label_counts_list: Sequence[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for label_counts in label_counts_list:
        for label_name, count_value in label_counts.items():
            merged[label_name] = merged.get(label_name, 0) + count_value
    return merged


def _normalize_label_counts(label_counts: dict[str, int]) -> dict[str, float]:
    total = sum(label_counts.values())
    if total <= 0:
        raise ValueError("label counts must contain at least one sample")
    return {label_name: count_value / total for label_name, count_value in label_counts.items()}


def _split_label_counts(
    split_groups: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    *,
    group_label_counts: dict[str, dict[str, int]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    merged: list[dict[str, int]] = []
    for groups in split_groups:
        merged.append(
            _aggregate_label_counts([group_label_counts[group] for group in groups])
        )
    return merged[0], merged[1], merged[2]


def _distribution_drift_score(
    *,
    split_label_counts: tuple[dict[str, int], dict[str, int], dict[str, int]],
    global_distribution: dict[str, float],
) -> float:
    score = 0.0
    for label_counts in split_label_counts:
        split_total = sum(label_counts.values())
        if split_total == 0:
            score += float(len(global_distribution))
            continue
        for label_name, global_ratio in global_distribution.items():
            split_ratio = label_counts.get(label_name, 0) / split_total
            score += abs(split_ratio - global_ratio)
    return score


def _normalize_required_labels(
    *,
    required_labels_per_split: Sequence[str] | None,
    available_labels: tuple[str, ...],
) -> tuple[str, ...]:
    if required_labels_per_split is None:
        return tuple()
    normalized = tuple(sorted(set(required_labels_per_split)))
    unknown = sorted(set(normalized).difference(available_labels))
    if unknown:
        raise ValueError(
            "required_labels_per_split includes labels not present in data: "
            f"{unknown}"
        )
    return normalized


def _required_labels_presence_count(
    *,
    split_label_counts: tuple[dict[str, int], dict[str, int], dict[str, int]],
    required_labels: tuple[str, ...],
) -> int:
    if not required_labels:
        return 0
    present = 0
    for label_counts in split_label_counts:
        for label_name in required_labels:
            if label_counts.get(label_name, 0) > 0:
                present += 1
    return present
