"""Tests for XJTU-SY prognostics baseline evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from whirdetective.data.adapters.xjtu_sy import XjtuCsvEntry
from whirdetective.prognostics import (
    XjtuPrognosticsTargets,
    XjtuRunConfig,
    run_xjtu_prognostics_baseline,
)


def test_run_xjtu_prognostics_baseline_passes_on_synthetic_signal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    part_file = tmp_path / "XJTU-SY_Bearing_Datasets.part01.rar"
    part_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "whirdetective.prognostics.xjtu_baseline.list_xjtu_parts",
        lambda root: (part_file,),
    )

    def _fake_payloads(_part_file: Path):
        payloads = []
        for bearing in ("Bearing1_1", "Bearing1_2", "Bearing1_3"):
            for idx in range(25, 500, 25):
                amplitude = 2.0 - (idx / 400.0)
                csv = "Horizontal_vibration_signals,Vertical_vibration_signals\n"
                rows = [
                    f"{amplitude * 0.5},{amplitude * 0.2}",
                    f"{-amplitude * 0.5},{-amplitude * 0.2}",
                    f"{amplitude * 0.3},{amplitude * 0.1}",
                ]
                payloads.append(
                    (
                        XjtuCsvEntry(
                            condition="35Hz12kN",
                            bearing_id=bearing,
                            snapshot_index=idx,
                            archive_entry_path=f"XJTU-SY_Bearing_Datasets/35Hz12kN/{bearing}/{idx}.csv",
                        ),
                        (csv + "\n".join(rows) + "\n").encode("utf-8"),
                    )
                )
        return tuple(payloads)

    monkeypatch.setattr(
        "whirdetective.prognostics.xjtu_baseline.iter_xjtu_csv_payloads",
        _fake_payloads,
    )

    result = run_xjtu_prognostics_baseline(
        config=XjtuRunConfig(root_dir=tmp_path, sample_stride=25, csv_max_rows=128),
        targets=XjtuPrognosticsTargets(
            min_bearings=3,
            min_samples=40,
            max_test_mae=0.60,
            min_test_spearman=0.80,
            min_mean_group_spearman=0.70,
        ),
    )
    assert result.evaluation.passed is True
    assert result.evaluation.num_bearings == 3
    assert result.evaluation.num_samples >= 40


def test_run_xjtu_prognostics_baseline_fails_with_no_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    part_file = tmp_path / "XJTU-SY_Bearing_Datasets.part01.rar"
    part_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "whirdetective.prognostics.xjtu_baseline.list_xjtu_parts",
        lambda root: (part_file,),
    )
    monkeypatch.setattr(
        "whirdetective.prognostics.xjtu_baseline.iter_xjtu_csv_payloads",
        lambda _part_file: (),
    )

    result = run_xjtu_prognostics_baseline(
        config=XjtuRunConfig(root_dir=tmp_path),
    )
    assert result.evaluation.passed is False
    assert "no_xjtu_records_collected" in result.evaluation.failed_checks


def test_spearman_tie_ranks_are_stable_under_within_tie_swaps() -> None:
    from whirdetective.prognostics.xjtu_baseline import _spearman

    y_pred = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.float64)
    y_true_order_a = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64)
    y_true_order_b = np.asarray([2, 1, 4, 3, 5, 6], dtype=np.float64)

    score_a = _spearman(y_true_order_a, y_pred)
    score_b = _spearman(y_true_order_b, y_pred)

    assert score_a == pytest.approx(score_b, abs=1e-12)
    assert score_a < 1.0
