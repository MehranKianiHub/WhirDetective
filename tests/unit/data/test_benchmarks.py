"""Tests for benchmark dataset contract validators."""

from __future__ import annotations

from pathlib import Path

from whirdetective.data.benchmarks import (
    DatasetContract,
    validate_dataset_contract,
    validate_default_benchmarks,
)


def test_validate_dataset_contract_reports_ok(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "raw" / "cwru"
    data_root.mkdir(parents=True)
    (data_root / "100.mat").write_text("x", encoding="utf-8")
    (data_root / "101.mat").write_text("x", encoding="utf-8")

    contract = DatasetContract(
        name="cwru",
        relative_path="data/raw/cwru",
        file_glob="*.mat",
        min_file_count=2,
    )
    result = validate_dataset_contract(tmp_path, contract)
    assert result.ok
    assert result.discovered_files == 2


def test_validate_dataset_contract_reports_missing_root(tmp_path: Path) -> None:
    contract = DatasetContract(
        name="cwru",
        relative_path="data/raw/cwru",
        file_glob="*.mat",
        min_file_count=1,
    )
    result = validate_dataset_contract(tmp_path, contract)
    assert not result.ok
    assert result.discovered_files == 0


def test_validate_default_benchmarks_returns_all_results(tmp_path: Path) -> None:
    results = validate_default_benchmarks(tmp_path)
    assert len(results) == 5
    assert all(result.ok is False for result in results)
