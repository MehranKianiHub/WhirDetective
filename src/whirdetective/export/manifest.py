"""Manifest helpers for model artifact integrity and traceability."""

from __future__ import annotations

import hmac
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class ManifestFileEntry:
    """File-level integrity metadata."""

    path: str
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_payload(
    *,
    package_name: str,
    package_root: Path,
    files: tuple[Path, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build JSON-safe manifest payload from emitted package files."""
    entries: list[ManifestFileEntry] = []
    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(f"cannot add missing file to manifest: {file_path}")
        relative = file_path.relative_to(package_root)
        entries.append(
            ManifestFileEntry(
                path=str(relative),
                sha256=sha256_file(file_path),
                size_bytes=int(file_path.stat().st_size),
            )
        )

    return {
        "manifest_version": "1.0",
        "package_name": package_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": [
            {
                "path": entry.path,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in entries
        ],
        "metadata": metadata,
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON manifest."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    """Read manifest JSON payload."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def canonical_manifest_bytes(payload: dict[str, Any]) -> bytes:
    """Stable bytes representation used for signature creation/verification."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_hmac_sha256_signature(*, payload: dict[str, Any], key: str) -> str:
    """Compute HMAC-SHA256 signature for manifest payload."""
    if not key:
        raise ValueError("signing key must not be empty")
    return hmac.new(
        key=key.encode("utf-8"),
        msg=canonical_manifest_bytes(payload),
        digestmod=hashlib.sha256,
    ).hexdigest()


def write_manifest_signature(path: Path, *, signature: str, algorithm: str = "hmac-sha256") -> None:
    """Write detached signature metadata for manifest."""
    payload = {
        "algorithm": algorithm,
        "signature": signature,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_manifest_signature(path: Path) -> dict[str, str]:
    """Read detached manifest signature metadata."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "algorithm" not in payload or "signature" not in payload:
        raise ValueError("invalid manifest signature payload")
    return {
        "algorithm": str(payload["algorithm"]),
        "signature": str(payload["signature"]),
    }


def verify_hmac_sha256_signature(
    *,
    payload: dict[str, Any],
    key: str,
    expected_signature: str,
) -> bool:
    """Verify HMAC-SHA256 signature using constant-time comparison."""
    actual = compute_hmac_sha256_signature(payload=payload, key=key)
    return hmac.compare_digest(actual, expected_signature)
