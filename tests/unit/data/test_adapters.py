"""Tests for dataset adapter indexing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from whirdetective.data.adapters.cwru import (
    infer_cwru_label_from_path,
    list_cwru_mat_files,
    load_cwru_channels,
)
from whirdetective.data.adapters.paderborn import list_paderborn_mat_entries
from whirdetective.data.labeling import BearingFaultLabel


def test_list_cwru_mat_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "100.mat").write_text("x", encoding="utf-8")
    (tmp_path / "b.mat").write_text("x", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    files = list_cwru_mat_files(tmp_path)
    assert [item.name for item in files] == ["100.mat", "b.mat"]


def test_infer_cwru_label_from_path_tokens() -> None:
    assert infer_cwru_label_from_path("cwru/normal/100.mat") == BearingFaultLabel.HEALTHY
    assert infer_cwru_label_from_path("cwru/drive_ir/118.mat") == BearingFaultLabel.INNER_RACE
    assert infer_cwru_label_from_path("cwru/fe_or/197.mat") == BearingFaultLabel.OUTER_RACE
    assert infer_cwru_label_from_path("cwru/ball_fault/222.mat") == BearingFaultLabel.BALL


def test_infer_cwru_label_from_path_avoids_bootctrl_false_positive() -> None:
    label = infer_cwru_label_from_path("/home/bootctrl/Projects/WhirDetective/data/raw/cwru/302.mat")
    assert label == BearingFaultLabel.UNKNOWN


def test_infer_cwru_label_from_numeric_ids() -> None:
    assert infer_cwru_label_from_path("cwru/100.mat") == BearingFaultLabel.HEALTHY
    assert infer_cwru_label_from_path("cwru/107.mat") == BearingFaultLabel.INNER_RACE
    assert infer_cwru_label_from_path("cwru/118.mat") == BearingFaultLabel.BALL
    assert infer_cwru_label_from_path("cwru/130.mat") == BearingFaultLabel.OUTER_RACE


def test_load_cwru_channels_extracts_known_signals(tmp_path: Path) -> None:
    mat_path = tmp_path / "100.mat"
    savemat(
        mat_path,
        {
            "X100_DE_time": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64),
            "X100_FE_time": np.asarray([[4.0], [5.0], [6.0]], dtype=np.float64),
            "noise": np.asarray([[1.0]], dtype=np.float64),
        },
    )

    channels = load_cwru_channels(mat_path)
    assert set(channels.keys()) == {"de_accel", "fe_accel"}
    assert channels["de_accel"].shape == (3,)
    assert channels["fe_accel"].shape == (3,)


def test_list_paderborn_mat_entries_with_fake_reader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rar_path = tmp_path / "K001.rar"
    rar_path.write_text("placeholder", encoding="utf-8")

    class _Entry:
        def __init__(self, pathname: str) -> None:
            self.pathname = pathname

    class _FakeReader:
        def __enter__(self) -> "_FakeReader":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def __iter__(self):
            return iter(
                (
                    _Entry("K001/N15_M07_F04_K001_4.mat"),
                    _Entry("K001/readme.txt"),
                    _Entry("K001/N15_M07_F10_K001_11.mat"),
                )
            )

    monkeypatch.setattr(
        "whirdetective.data.adapters.paderborn.libarchive.file_reader",
        lambda _: _FakeReader(),
    )

    entries = list_paderborn_mat_entries(rar_path)
    assert entries == (
        "K001/N15_M07_F04_K001_4.mat",
        "K001/N15_M07_F10_K001_11.mat",
    )


def test_list_paderborn_mat_entries_validates_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_paderborn_mat_entries(tmp_path / "missing.rar")

    wrong_suffix = tmp_path / "K001.zip"
    wrong_suffix.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match=".rar"):
        list_paderborn_mat_entries(wrong_suffix)


def test_load_cwru_channels_validates_suffix(tmp_path: Path) -> None:
    text_file = tmp_path / "sample.txt"
    text_file.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match=".mat"):
        load_cwru_channels(text_file)
