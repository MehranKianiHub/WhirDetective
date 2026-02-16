"""Tests for leakage-safe grouped splitting."""

from __future__ import annotations

import pytest

from whirdetective.data import GroupedSplit, assert_group_isolation, split_by_group


def test_grouped_split_has_no_group_leakage() -> None:
    group_ids = (
        "machine_a_run_1",
        "machine_a_run_1",
        "machine_a_run_2",
        "machine_b_run_1",
        "machine_b_run_2",
        "machine_c_run_1",
        "machine_d_run_1",
        "machine_e_run_1",
    )
    split = split_by_group(group_ids, seed=7)

    train_groups = set(split.train_groups)
    val_groups = set(split.val_groups)
    test_groups = set(split.test_groups)

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    for idx in split.train_indices:
        assert group_ids[idx] in train_groups
    for idx in split.val_indices:
        assert group_ids[idx] in val_groups
    for idx in split.test_indices:
        assert group_ids[idx] in test_groups


def test_grouped_split_is_deterministic_with_seed() -> None:
    groups = tuple(f"group_{idx // 2}" for idx in range(20))
    first = split_by_group(groups, seed=123)
    second = split_by_group(groups, seed=123)

    assert first == second


def test_grouped_split_validates_ratio_sum() -> None:
    groups = ("a", "b", "c")
    with pytest.raises(ValueError, match="must equal 1.0"):
        split_by_group(groups, train_ratio=0.6, val_ratio=0.2, test_ratio=0.3)


def test_assert_group_isolation_detects_leakage() -> None:
    group_ids = ("g1", "g1", "g2", "g3")
    leaking_split = GroupedSplit(
        train_indices=(0,),
        val_indices=(1,),
        test_indices=(2, 3),
        train_groups=("g1",),
        val_groups=("g1",),
        test_groups=("g2", "g3"),
    )
    with pytest.raises(ValueError, match="Leakage detected"):
        assert_group_isolation(group_ids, leaking_split)


def test_grouped_split_with_labels_enforces_min_distinct_class_coverage() -> None:
    group_ids = tuple(
        f"group_{label}_{group_idx}"
        for label in ("healthy", "inner_race", "ball")
        for group_idx in range(4)
    )
    labels = tuple(
        label
        for label in ("healthy", "inner_race", "ball")
        for _ in range(4)
    )
    split = split_by_group(
        group_ids,
        labels=labels,
        min_distinct_labels_per_split=2,
        search_attempts=256,
        seed=11,
    )
    assert_group_isolation(group_ids, split)

    split_to_indices = {
        "train": split.train_indices,
        "val": split.val_indices,
        "test": split.test_indices,
    }
    for indices in split_to_indices.values():
        distinct_labels = {labels[idx] for idx in indices}
        assert len(distinct_labels) >= 2


def test_grouped_split_with_unreachable_class_coverage_raises() -> None:
    group_ids = ("g_h", "g_h", "g_i", "g_i", "g_b", "g_b")
    labels = ("healthy", "healthy", "inner_race", "inner_race", "ball", "ball")
    with pytest.raises(ValueError, match="Could not find grouped split"):
        split_by_group(
            group_ids,
            labels=labels,
            min_distinct_labels_per_split=2,
            search_attempts=64,
            seed=3,
        )


def test_grouped_split_can_require_specific_labels_in_each_split() -> None:
    group_ids = tuple(
        f"group_{label}_{group_idx}"
        for label in ("healthy", "inner_race", "ball")
        for group_idx in range(4)
    )
    labels = tuple(
        label
        for label in ("healthy", "inner_race", "ball")
        for _ in range(4)
    )
    split = split_by_group(
        group_ids,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        labels=labels,
        min_distinct_labels_per_split=2,
        required_labels_per_split=("healthy", "inner_race", "ball"),
        search_attempts=512,
        seed=13,
    )

    split_to_indices = {
        "train": split.train_indices,
        "val": split.val_indices,
        "test": split.test_indices,
    }
    for indices in split_to_indices.values():
        present = {labels[idx] for idx in indices}
        assert {"healthy", "inner_race", "ball"}.issubset(present)


def test_grouped_split_required_labels_validates_unknown_label() -> None:
    group_ids = ("g_h_1", "g_h_2", "g_i_1", "g_i_2", "g_b_1", "g_b_2")
    labels = ("healthy", "healthy", "inner_race", "inner_race", "ball", "ball")
    with pytest.raises(ValueError, match="not present in data"):
        split_by_group(
            group_ids,
            labels=labels,
            required_labels_per_split=("healthy", "outer_race"),
            search_attempts=64,
            seed=3,
        )
