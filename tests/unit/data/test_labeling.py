"""Tests for canonical bearing fault label normalization."""

from __future__ import annotations

from whirdetective.data import BearingFaultLabel, normalize_fault_label


def test_generic_normalization_tokens() -> None:
    assert normalize_fault_label("normal") == BearingFaultLabel.HEALTHY
    assert normalize_fault_label("inner-race") == BearingFaultLabel.INNER_RACE
    assert normalize_fault_label("OUTER RACE") == BearingFaultLabel.OUTER_RACE
    assert normalize_fault_label("ball") == BearingFaultLabel.BALL


def test_dataset_override_mapping() -> None:
    assert normalize_fault_label("ir", dataset="cwru") == BearingFaultLabel.INNER_RACE
    assert normalize_fault_label("real", dataset="paderborn") == BearingFaultLabel.COMBINED


def test_unknown_labels_fall_back_to_unknown() -> None:
    assert normalize_fault_label("mystery-class") == BearingFaultLabel.UNKNOWN
