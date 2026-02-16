"""XJTU-SY archive readers for prognostics baseline workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import libarchive  # type: ignore[import-untyped]


_XJTU_ENTRY_PATTERN = re.compile(
    r"^XJTU-SY_Bearing_Datasets/(?P<condition>[^/]+)/(?P<bearing>[^/]+)/(?P<snapshot>\d+)\.csv$"
)


@dataclass(frozen=True, slots=True)
class XjtuCsvEntry:
    """Structured metadata for one XJTU-SY CSV snapshot entry."""

    condition: str
    bearing_id: str
    snapshot_index: int
    archive_entry_path: str


def list_xjtu_parts(root_dir: str | Path) -> tuple[Path, ...]:
    """List XJTU-SY multi-part `.rar` archives from a data root."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"XJTU-SY root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"XJTU-SY root is not a directory: {root}")
    parts = tuple(path for path in root.glob("*.part*.rar") if path.is_file())
    if not parts:
        raise FileNotFoundError(f"No XJTU-SY part archives found under: {root}")
    return tuple(sorted(parts))


def iter_xjtu_csv_payloads(
    part_file: str | Path,
    *,
    tolerate_trailing_errors: bool = True,
) -> tuple[tuple[XjtuCsvEntry, bytes], ...]:
    """Iterate XJTU-SY CSV payloads from one part archive."""
    archive_path = Path(part_file)
    if not archive_path.exists():
        raise FileNotFoundError(f"XJTU-SY archive does not exist: {archive_path}")
    if archive_path.suffix.lower() != ".rar":
        raise ValueError(f"Expected .rar archive, got: {archive_path}")

    payloads: list[tuple[XjtuCsvEntry, bytes]] = []
    try:
        with libarchive.file_reader(str(archive_path)) as archive:
            for entry in archive:
                pathname = str(entry.pathname)
                parsed = _parse_xjtu_csv_entry(pathname)
                if parsed is None:
                    continue
                payloads.append((parsed, b"".join(entry.get_blocks())))
    except Exception as exc:  # pragma: no cover - libarchive error class typing is not stable
        if not tolerate_trailing_errors:
            raise
        error_text = str(exc).lower()
        if "too small block encountered" not in error_text:
            raise

    return tuple(payloads)


def _parse_xjtu_csv_entry(pathname: str) -> XjtuCsvEntry | None:
    match = _XJTU_ENTRY_PATTERN.match(pathname)
    if match is None:
        return None
    return XjtuCsvEntry(
        condition=match.group("condition"),
        bearing_id=match.group("bearing"),
        snapshot_index=int(match.group("snapshot")),
        archive_entry_path=pathname,
    )
