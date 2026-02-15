"""Leakage-safe grouped splitting utilities for dataset engine."""

from __future__ import annotations

from dataclasses import dataclass
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
) -> GroupedSplit:
    """Split sample indices by group id so groups never span multiple splits."""
    if not group_ids:
        raise ValueError("group_ids must not be empty")
    _validate_ratios(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

    unique_groups = sorted(set(group_ids))
    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    train_count, val_count, test_count = _allocate_group_counts(
        total_groups=len(shuffled_groups),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_groups = tuple(shuffled_groups[:train_count])
    val_groups = tuple(shuffled_groups[train_count : train_count + val_count])
    test_groups = tuple(shuffled_groups[train_count + val_count : train_count + val_count + test_count])

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
