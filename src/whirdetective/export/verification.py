"""Verification helpers for emitted Edge model packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from whirdetective.export.manifest import (
    read_manifest,
    read_manifest_signature,
    sha256_file,
    verify_hmac_sha256_signature,
)


@dataclass(frozen=True, slots=True)
class PackageVerificationResult:
    """Outcome of verifying package integrity metadata and files."""

    ok: bool
    missing_files: tuple[str, ...]
    checksum_mismatches: tuple[str, ...]
    size_mismatches: tuple[str, ...]
    signature_verified: bool | None
    signature_reason: str | None


def verify_edge_model_package(
    *,
    package_dir: Path,
    signing_key: str | None = None,
) -> PackageVerificationResult:
    """Verify manifest-described files and optional manifest signature."""
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        return PackageVerificationResult(
            ok=False,
            missing_files=("manifest.json",),
            checksum_mismatches=(),
            size_mismatches=(),
            signature_verified=None,
            signature_reason="manifest_missing",
        )

    manifest = read_manifest(manifest_path)
    entries = _manifest_entries(manifest)

    missing_files: list[str] = []
    checksum_mismatches: list[str] = []
    size_mismatches: list[str] = []
    for entry in entries:
        relative_path = str(entry["path"])
        expected_sha256 = str(entry["sha256"])
        expected_size = int(entry["size_bytes"])
        file_path = package_dir / relative_path
        if not file_path.exists():
            missing_files.append(relative_path)
            continue
        if int(file_path.stat().st_size) != expected_size:
            size_mismatches.append(relative_path)
        actual_sha256 = sha256_file(file_path)
        if actual_sha256 != expected_sha256:
            checksum_mismatches.append(relative_path)

    signature_verified, signature_reason = _verify_manifest_signature(
        package_dir=package_dir,
        manifest=manifest,
        signing_key=signing_key,
    )
    ok = (
        len(missing_files) == 0
        and len(checksum_mismatches) == 0
        and len(size_mismatches) == 0
        and (signature_verified is True or signature_verified is None)
    )
    return PackageVerificationResult(
        ok=ok,
        missing_files=tuple(sorted(missing_files)),
        checksum_mismatches=tuple(sorted(checksum_mismatches)),
        size_mismatches=tuple(sorted(size_mismatches)),
        signature_verified=signature_verified,
        signature_reason=signature_reason,
    )


def _manifest_entries(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("manifest files must be a list")
    entries: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("manifest file entries must be objects")
        for required in ("path", "sha256", "size_bytes"):
            if required not in entry:
                raise ValueError(f"manifest file entry missing required field: {required}")
        entries.append(entry)
    return tuple(entries)


def _verify_manifest_signature(
    *,
    package_dir: Path,
    manifest: dict[str, Any],
    signing_key: str | None,
) -> tuple[bool | None, str | None]:
    signature_path = package_dir / "manifest.sig"
    if not signature_path.exists():
        return None, None
    signature_payload = read_manifest_signature(signature_path)
    algorithm = signature_payload["algorithm"]
    signature = signature_payload["signature"]
    if algorithm != "hmac-sha256":
        return False, f"unsupported_signature_algorithm:{algorithm}"
    if signing_key is None:
        return False, "signing_key_missing"
    ok = verify_hmac_sha256_signature(
        payload=manifest,
        key=signing_key,
        expected_signature=signature,
    )
    return (True, None) if ok else (False, "signature_mismatch")

