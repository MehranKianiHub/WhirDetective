"""CWRU dataset indexing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from whirdetective.data.labeling import BearingFaultLabel


def list_cwru_mat_files(root_dir: str | Path) -> tuple[Path, ...]:
    """List CWRU `.mat` files under a root directory in stable order."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"CWRU root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"CWRU root is not a directory: {root}")

    return tuple(sorted(path for path in root.rglob("*.mat") if path.is_file()))


def infer_cwru_label_from_path(file_path: str | Path) -> BearingFaultLabel:
    """Infer coarse CWRU label from path tokens when available."""
    path = Path(file_path)
    haystack = "_".join(part.lower() for part in path.parts)
    if "normal" in haystack or "healthy" in haystack:
        return BearingFaultLabel.HEALTHY
    if "inner" in haystack or "_ir" in haystack:
        return BearingFaultLabel.INNER_RACE
    if "outer" in haystack or "_or" in haystack:
        return BearingFaultLabel.OUTER_RACE
    if "ball" in haystack or "_b" in haystack:
        return BearingFaultLabel.BALL
    return BearingFaultLabel.UNKNOWN


def load_cwru_channels(mat_file: str | Path) -> dict[str, npt.NDArray[np.float64]]:
    """Load CWRU `.mat` file and extract known acceleration channels."""
    path = Path(mat_file)
    if not path.exists():
        raise FileNotFoundError(f"CWRU file does not exist: {path}")
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Expected .mat file, got: {path}")

    raw = _load_mat_dict(path)
    extracted: dict[str, npt.NDArray[np.float64]] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        if not isinstance(value, np.ndarray):
            continue

        flattened = np.asarray(value, dtype=np.float64).reshape(-1)
        if flattened.size == 0 or not np.all(np.isfinite(flattened)):
            continue

        normalized_key = key.lower()
        if normalized_key.endswith("_de_time"):
            extracted["de_accel"] = flattened
        elif normalized_key.endswith("_fe_time"):
            extracted["fe_accel"] = flattened
        elif normalized_key.endswith("_ba_time"):
            extracted["ba_accel"] = flattened

    if not extracted:
        raise ValueError(f"No supported CWRU channels found in {path}")
    return extracted


def _load_mat_dict(path: Path) -> dict[str, object]:
    try:
        from scipy.io import loadmat  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required to read .mat files. Install with `pip install -e '.[data]'`."
        ) from exc
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    return cast(dict[str, object], data)
