"""Paderborn dataset archive indexing and loading helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import cast

import libarchive  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from whirdetective.data.labeling import BearingFaultLabel


def list_paderborn_archives(root_dir: str | Path) -> tuple[Path, ...]:
    """List Paderborn `.rar` archives from root directory in stable order."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Paderborn root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Paderborn root is not a directory: {root}")
    archives = tuple(path for path in root.glob("*.rar") if path.is_file())
    return tuple(sorted(archives))


def infer_paderborn_label_from_archive(archive_path: str | Path) -> BearingFaultLabel:
    """Infer canonical coarse fault label from Paderborn archive stem."""
    stem = Path(archive_path).stem.upper()
    if stem.startswith("K") and stem[1:].isdigit():
        return BearingFaultLabel.HEALTHY
    if stem.startswith("KI"):
        return BearingFaultLabel.INNER_RACE
    if stem.startswith("KA"):
        return BearingFaultLabel.OUTER_RACE
    if stem.startswith("KB"):
        return BearingFaultLabel.BALL
    return BearingFaultLabel.UNKNOWN


def list_paderborn_mat_entries(rar_file: str | Path) -> tuple[str, ...]:
    """List `.mat` entries inside a Paderborn `.rar` archive."""
    archive_path = Path(rar_file)
    if not archive_path.exists():
        raise FileNotFoundError(f"Paderborn archive does not exist: {archive_path}")
    if archive_path.suffix.lower() != ".rar":
        raise ValueError(f"Expected .rar archive, got: {archive_path}")

    entries: list[str] = []
    with libarchive.file_reader(str(archive_path)) as archive:
        for entry in archive:
            pathname = str(entry.pathname)
            if pathname.lower().endswith(".mat"):
                entries.append(pathname)

    return tuple(sorted(entries))


def iter_paderborn_mat_payloads(
    rar_file: str | Path,
    *,
    entry_whitelist: set[str] | None = None,
) -> tuple[tuple[str, bytes], ...]:
    """Read `.mat` entry payloads from one archive, optionally filtering by path."""
    archive_path = Path(rar_file)
    if not archive_path.exists():
        raise FileNotFoundError(f"Paderborn archive does not exist: {archive_path}")
    if archive_path.suffix.lower() != ".rar":
        raise ValueError(f"Expected .rar archive, got: {archive_path}")

    payloads: list[tuple[str, bytes]] = []
    with libarchive.file_reader(str(archive_path)) as archive:
        for entry in archive:
            pathname = str(entry.pathname)
            if not pathname.lower().endswith(".mat"):
                continue
            if entry_whitelist is not None and pathname not in entry_whitelist:
                continue
            payloads.append((pathname, b"".join(entry.get_blocks())))
    return tuple(payloads)


def load_paderborn_channels_from_mat_payload(
    payload: bytes,
    *,
    min_signal_length: int = 1024,
    min_length_ratio: float = 0.5,
) -> dict[str, npt.NDArray[np.float64]]:
    """Extract signal channels from one Paderborn MAT payload."""
    if min_signal_length <= 0:
        raise ValueError("min_signal_length must be > 0")
    if min_length_ratio <= 0.0 or min_length_ratio > 1.0:
        raise ValueError("min_length_ratio must be in (0, 1]")
    if len(payload) == 0:
        raise ValueError("payload must not be empty")

    mat_data = _load_mat_payload(payload)
    candidates: list[tuple[str, npt.NDArray[np.float64]]] = []
    for key, value in mat_data.items():
        if key.startswith("__"):
            continue
        if not isinstance(value, np.ndarray) or value.dtype != object:
            continue
        if value.shape != (1, 1):
            continue
        root = value[0, 0]
        candidates.extend(_extract_struct_group_channels(root, "X"))
        candidates.extend(_extract_struct_group_channels(root, "Y"))

    if not candidates:
        raise ValueError("No usable channels found in Paderborn MAT payload")

    longest_signal = max(array.size for _, array in candidates)
    required_length = max(min_signal_length, int(longest_signal * min_length_ratio))
    selected = {
        name: array
        for name, array in candidates
        if array.size >= required_length
    }
    if not selected:
        raise ValueError("No channels met minimum length threshold in Paderborn payload")
    return selected


def _extract_struct_group_channels(
    root: object,
    group_name: str,
) -> list[tuple[str, npt.NDArray[np.float64]]]:
    group = getattr(root, group_name, None)
    if not isinstance(group, np.ndarray):
        return []
    if group.ndim != 2 or group.shape[0] != 1:
        return []

    channels: list[tuple[str, npt.NDArray[np.float64]]] = []
    for idx in range(group.shape[1]):
        entry = group[0, idx]
        data = getattr(entry, "Data", None)
        if data is None:
            continue
        series = np.asarray(data, dtype=np.float64).reshape(-1)
        if series.size == 0 or not np.all(np.isfinite(series)):
            continue
        raw_name = _extract_name_token(getattr(entry, "Name", None))
        channel_name = _sanitize_channel_name(raw_name, fallback=f"{group_name.lower()}_{idx}")
        channels.append((channel_name, series))
    return channels


def _extract_name_token(raw_name: object) -> str:
    if raw_name is None:
        return ""
    arr = np.asarray(raw_name).reshape(-1)
    if arr.size == 0:
        return ""
    return str(arr[0])


def _sanitize_channel_name(name: str, *, fallback: str) -> str:
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    if not normalized:
        return fallback
    return normalized


def _load_mat_payload(payload: bytes) -> dict[str, object]:
    try:
        from scipy.io import loadmat  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required to read Paderborn .mat payloads. Install with `pip install -e '.[data]'`."
        ) from exc

    loaded = loadmat(BytesIO(payload), squeeze_me=False, struct_as_record=False)
    return cast(dict[str, object], loaded)
