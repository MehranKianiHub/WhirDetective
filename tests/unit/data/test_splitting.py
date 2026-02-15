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
