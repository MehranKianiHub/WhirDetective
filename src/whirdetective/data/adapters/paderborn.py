"""Paderborn dataset archive indexing helpers."""

from __future__ import annotations

from pathlib import Path

import libarchive  # type: ignore[import-untyped]


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
