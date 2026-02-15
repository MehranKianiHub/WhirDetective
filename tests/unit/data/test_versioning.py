"""Tests for reproducible dataset fingerprinting."""

from __future__ import annotations

from pathlib import Path

import pytest

from whirdetective.data.versioning import dataset_fingerprint


def test_dataset_fingerprint_is_deterministic_and_order_invariant(tmp_path: Path) -> None:
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")

    first = dataset_fingerprint((file_a, file_b))
    second = dataset_fingerprint((file_b, file_a))
    assert first == second


def test_dataset_fingerprint_changes_when_file_size_changes(tmp_path: Path) -> None:
    file_a = tmp_path / "a.txt"
    file_a.write_text("alpha", encoding="utf-8")
    before = dataset_fingerprint((file_a,))

    file_a.write_text("alpha-extended", encoding="utf-8")
    after = dataset_fingerprint((file_a,))

    assert before != after


def test_dataset_fingerprint_validates_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        dataset_fingerprint((tmp_path / "missing.txt",))
