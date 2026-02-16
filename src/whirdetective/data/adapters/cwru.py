"""CWRU dataset indexing helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import cast

import numpy as np
import numpy.typing as npt

from whirdetective.data.labeling import BearingFaultLabel

_HEALTHY_IDS: frozenset[int] = frozenset({97, 98, 99, 100})
_INNER_RACE_IDS: frozenset[int] = frozenset(
    set(range(105, 109))
    | set(range(169, 173))
    | set(range(209, 213))
)
_BALL_IDS: frozenset[int] = frozenset(
    set(range(118, 122))
    | set(range(185, 189))
    | set(range(222, 226))
)
_OUTER_RACE_IDS: frozenset[int] = frozenset(
    set(range(130, 134))
    | set(range(197, 201))
    | set(range(234, 238))
)


def list_cwru_mat_files(root_dir: str | Path) -> tuple[Path, ...]:
    """List CWRU `.mat` files under a root directory in stable order."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"CWRU root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"CWRU root is not a directory: {root}")

    mat_files = tuple(path for path in root.rglob("*.mat") if path.is_file())
    return tuple(sorted(mat_files, key=_cwru_sort_key))


def infer_cwru_label_from_path(file_path: str | Path) -> BearingFaultLabel:
    """Infer coarse CWRU label from path tokens when available."""
    path = Path(file_path)
    tokens = _path_label_tokens(path)
    if any(token in {"normal", "healthy"} for token in tokens):
        return BearingFaultLabel.HEALTHY
    if any(token == "inner" or token.startswith("ir") for token in tokens):
        return BearingFaultLabel.INNER_RACE
    if any(token == "outer" or token.startswith("or") for token in tokens):
        return BearingFaultLabel.OUTER_RACE
    if any(token == "ball" or (token.startswith("b") and token[1:].isdigit()) for token in tokens):
        return BearingFaultLabel.BALL

    numeric_id = _extract_numeric_file_id(path)
    if numeric_id is not None:
        if numeric_id in _HEALTHY_IDS:
            return BearingFaultLabel.HEALTHY
        if numeric_id in _INNER_RACE_IDS:
            return BearingFaultLabel.INNER_RACE
        if numeric_id in _BALL_IDS:
            return BearingFaultLabel.BALL
        if numeric_id in _OUTER_RACE_IDS:
            return BearingFaultLabel.OUTER_RACE

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


def _path_label_tokens(path: Path) -> tuple[str, ...]:
    parts = [path.stem.lower(), *(part.lower() for part in path.parent.parts)]
    tokens: list[str] = []
    for part in parts:
        tokens.extend(token for token in re.split(r"[^a-z0-9]+", part) if token)
    return tuple(tokens)


def _extract_numeric_file_id(path: Path) -> int | None:
    stem = path.stem.strip().lower()
    if stem.isdigit():
        return int(stem)
    return None


def _cwru_sort_key(path: Path) -> tuple[int, int, str]:
    numeric_id = _extract_numeric_file_id(path)
    if numeric_id is not None:
        return (0, numeric_id, str(path))
    return (1, 0, str(path))
