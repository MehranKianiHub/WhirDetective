"""Benchmark dataset contract validators for Step 3 readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetContract:
    """Minimal on-disk contract for one benchmark dataset."""

    name: str
    relative_path: str
    file_glob: str
    min_file_count: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.relative_path.strip():
            raise ValueError("relative_path must not be empty")
        if not self.file_glob.strip():
            raise ValueError("file_glob must not be empty")
        if self.min_file_count <= 0:
            raise ValueError("min_file_count must be > 0")


@dataclass(frozen=True, slots=True)
class DatasetContractResult:
    """Outcome of validating one dataset contract."""

    name: str
    root_path: Path
    discovered_files: int
    expected_min_files: int
    ok: bool


DEFAULT_BENCHMARK_CONTRACTS: tuple[DatasetContract, ...] = (
    DatasetContract("cwru", "data/raw/cwru", "*.mat", 100),
    DatasetContract("paderborn", "data/raw/paderborn", "*.rar", 20),
    DatasetContract("xjtu_sy", "data/raw/xjtu_sy/Data", "*.rar", 6),
    DatasetContract(
        "femto_st",
        "data/raw/femto_st/ieee-phm-2012-data-challenge-dataset/Learning_set",
        "*",
        1,
    ),
    DatasetContract("ims", "data/raw/ims", "IMS.zip", 1),
)


def validate_dataset_contract(
    workspace_root: str | Path,
    contract: DatasetContract,
) -> DatasetContractResult:
    """Validate one dataset contract against local storage."""
    root = Path(workspace_root) / contract.relative_path
    if not root.exists():
        return DatasetContractResult(
            name=contract.name,
            root_path=root,
            discovered_files=0,
            expected_min_files=contract.min_file_count,
            ok=False,
        )

    discovered = len(tuple(root.glob(contract.file_glob)))
    return DatasetContractResult(
        name=contract.name,
        root_path=root,
        discovered_files=discovered,
        expected_min_files=contract.min_file_count,
        ok=discovered >= contract.min_file_count,
    )


def validate_default_benchmarks(workspace_root: str | Path) -> tuple[DatasetContractResult, ...]:
    """Validate all default benchmark contracts."""
    return tuple(
        validate_dataset_contract(workspace_root, contract) for contract in DEFAULT_BENCHMARK_CONTRACTS
    )
