"""Reproducible dataset fingerprint helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence


def dataset_fingerprint(paths: Sequence[str | Path]) -> str:
    """Create deterministic SHA256 fingerprint from path+size metadata."""
    if not paths:
        raise ValueError("paths must not be empty")

    normalized: list[tuple[str, int]] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Expected file path, got: {path}")
        normalized.append((str(path.resolve()), path.stat().st_size))

    normalized.sort()
    digest = hashlib.sha256()
    for abs_path, file_size in normalized:
        digest.update(abs_path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(file_size).encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()
