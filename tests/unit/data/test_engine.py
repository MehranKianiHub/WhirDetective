"""Tests for deterministic CWRU canonical dataset build engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from whirdetective.data import (
    CwruBuildConfig,
    PaderbornBuildConfig,
    build_cwru_canonical_dataset,
    build_paderborn_canonical_dataset,
)
from whirdetective.data.adapters import infer_cwru_label_from_path
from whirdetective.ml import ProjectionPolicy, SensorSetProjector


def _write_cwru_mat(path: Path, *, scale: float) -> None:
    savemat(
        path,
        {
            "X100_DE_time": (np.arange(16, dtype=np.float64) * scale).reshape(-1, 1),
            "X100_FE_time": (np.arange(16, dtype=np.float64) * (scale + 0.5)).reshape(-1, 1),
        },
    )


def test_build_cwru_canonical_dataset_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "normal").mkdir()
    (tmp_path / "inner").mkdir()
    (tmp_path / "outer").mkdir()
    _write_cwru_mat(tmp_path / "normal" / "100.mat", scale=1.0)
    _write_cwru_mat(tmp_path / "inner" / "101.mat", scale=2.0)
    _write_cwru_mat(tmp_path / "outer" / "102.mat", scale=3.0)

    projector = SensorSetProjector(ProjectionPolicy())
    config = CwruBuildConfig(
        root_dir=tmp_path,
        window_size=8,
        step_size=4,
        split_seed=17,
    )

    first = build_cwru_canonical_dataset(config=config, projector=projector)
    second = build_cwru_canonical_dataset(config=config, projector=projector)

    assert len(first.samples) == 9  # 3 files * 3 windows each
    assert first.split == second.split
    assert first.fingerprint == second.fingerprint
    assert len(first.source_files) == 3


def test_build_cwru_canonical_dataset_requires_enough_files(tmp_path: Path) -> None:
    _write_cwru_mat(tmp_path / "only_one.mat", scale=1.0)
    projector = SensorSetProjector(ProjectionPolicy())
    config = CwruBuildConfig(root_dir=tmp_path, window_size=8, step_size=4)

    with pytest.raises(ValueError, match="at least 3 .mat files"):
        build_cwru_canonical_dataset(config=config, projector=projector)


def test_build_cwru_canonical_dataset_excludes_unknown_labels_by_default(tmp_path: Path) -> None:
    (tmp_path / "normal").mkdir()
    (tmp_path / "inner").mkdir()
    (tmp_path / "outer").mkdir()
    (tmp_path / "misc").mkdir()
    _write_cwru_mat(tmp_path / "normal" / "100.mat", scale=1.0)
    _write_cwru_mat(tmp_path / "inner" / "101.mat", scale=2.0)
    _write_cwru_mat(tmp_path / "outer" / "102.mat", scale=3.0)
    _write_cwru_mat(tmp_path / "misc" / "302.mat", scale=4.0)  # numeric ID outside known mapping => unknown

    projector = SensorSetProjector(ProjectionPolicy())
    config = CwruBuildConfig(
        root_dir=tmp_path,
        window_size=8,
        step_size=4,
        split_seed=17,
    )
    built = build_cwru_canonical_dataset(config=config, projector=projector)

    assert len(built.source_files) == 3
    assert all(path.stem != "302" for path in built.source_files)


def test_build_cwru_canonical_dataset_max_files_preserves_class_diversity(tmp_path: Path) -> None:
    (tmp_path / "a_normal").mkdir()
    (tmp_path / "b_normal").mkdir()
    (tmp_path / "c_normal").mkdir()
    (tmp_path / "z_inner").mkdir()
    _write_cwru_mat(tmp_path / "a_normal" / "100.mat", scale=1.0)
    _write_cwru_mat(tmp_path / "b_normal" / "101.mat", scale=1.1)
    _write_cwru_mat(tmp_path / "c_normal" / "102.mat", scale=1.2)
    _write_cwru_mat(tmp_path / "z_inner" / "201.mat", scale=2.0)

    projector = SensorSetProjector(ProjectionPolicy())
    config = CwruBuildConfig(
        root_dir=tmp_path,
        window_size=8,
        step_size=4,
        split_seed=7,
        max_files=3,
    )
    built = build_cwru_canonical_dataset(config=config, projector=projector)
    labels = {infer_cwru_label_from_path(path).value for path in built.source_files}

    assert len(built.source_files) == 3
    assert len(labels) >= 2


def test_build_paderborn_canonical_dataset_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a1 = tmp_path / "K001.rar"
    a2 = tmp_path / "KI01.rar"
    a3 = tmp_path / "KA01.rar"
    for archive in (a1, a2, a3):
        archive.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "whirdetective.data.engine.list_paderborn_archives",
        lambda _root: (a1, a2, a3),
    )
    monkeypatch.setattr(
        "whirdetective.data.engine.list_paderborn_mat_entries",
        lambda _archive: ("entry_1.mat",),
    )
    monkeypatch.setattr(
        "whirdetective.data.engine.iter_paderborn_mat_payloads",
        lambda _archive, entry_whitelist=None: (("entry_1.mat", b"dummy"),),
    )
    monkeypatch.setattr(
        "whirdetective.data.engine.load_paderborn_channels_from_mat_payload",
        lambda payload, min_signal_length, min_length_ratio: {
            "vibration_1": np.arange(16, dtype=np.float64),
            "phase_current_1": np.arange(16, dtype=np.float64) + 1.0,
        },
    )

    projector = SensorSetProjector(ProjectionPolicy())
    config = PaderbornBuildConfig(
        root_dir=tmp_path,
        window_size=8,
        step_size=4,
        split_seed=11,
    )

    first = build_paderborn_canonical_dataset(config=config, projector=projector)
    second = build_paderborn_canonical_dataset(config=config, projector=projector)
    assert len(first.samples) == 9
    assert first.split == second.split
    assert first.fingerprint == second.fingerprint
    assert len(first.source_files) == 3


def test_build_paderborn_canonical_dataset_skips_malformed_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a1 = tmp_path / "K001.rar"
    a2 = tmp_path / "KI01.rar"
    a3 = tmp_path / "KA01.rar"
    for archive in (a1, a2, a3):
        archive.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "whirdetective.data.engine.list_paderborn_archives",
        lambda _root: (a1, a2, a3),
    )
    monkeypatch.setattr(
        "whirdetective.data.engine.list_paderborn_mat_entries",
        lambda _archive: ("entry_bad.mat", "entry_good.mat"),
    )
    monkeypatch.setattr(
        "whirdetective.data.engine.iter_paderborn_mat_payloads",
        lambda _archive, entry_whitelist=None: (
            ("entry_bad.mat", b"bad"),
            ("entry_good.mat", b"good"),
        ),
    )

    def _load_payload(
        payload: bytes,
        min_signal_length: int,
        min_length_ratio: float,
    ) -> dict[str, np.ndarray]:
        del min_signal_length, min_length_ratio
        if payload == b"bad":
            raise ValueError("malformed entry")
        return {
            "vibration_1": np.arange(16, dtype=np.float64),
            "phase_current_1": np.arange(16, dtype=np.float64) + 1.0,
        }

    monkeypatch.setattr(
        "whirdetective.data.engine.load_paderborn_channels_from_mat_payload",
        _load_payload,
    )

    projector = SensorSetProjector(ProjectionPolicy())
    config = PaderbornBuildConfig(
        root_dir=tmp_path,
        window_size=8,
        step_size=4,
        split_seed=11,
    )

    built = build_paderborn_canonical_dataset(config=config, projector=projector)
    assert len(built.samples) > 0
    assert len(built.source_files) == 3
