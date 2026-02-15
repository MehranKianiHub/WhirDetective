"""Tests for deterministic CWRU canonical dataset build engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from whirdetective.data import CwruBuildConfig, build_cwru_canonical_dataset
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
