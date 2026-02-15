"""Dataset engine primitives for Step 3 signal/data workflows."""

from whirdetective.data.benchmarks import (
    DEFAULT_BENCHMARK_CONTRACTS,
    DatasetContract,
    DatasetContractResult,
    validate_dataset_contract,
    validate_default_benchmarks,
)
from whirdetective.data.contracts import CanonicalTrainingSample
from whirdetective.data.engine import BuiltCanonicalDataset, CwruBuildConfig, build_cwru_canonical_dataset
from whirdetective.data.labeling import BearingFaultLabel, normalize_fault_label
from whirdetective.data.pipeline import build_windowed_canonical_samples
from whirdetective.data.splitting import GroupedSplit, assert_group_isolation, split_by_group
from whirdetective.data.standardize import standardize_channel_sample
from whirdetective.data.versioning import dataset_fingerprint

__all__ = [
    "DEFAULT_BENCHMARK_CONTRACTS",
    "BearingFaultLabel",
    "BuiltCanonicalDataset",
    "CanonicalTrainingSample",
    "CwruBuildConfig",
    "DatasetContract",
    "DatasetContractResult",
    "GroupedSplit",
    "assert_group_isolation",
    "build_windowed_canonical_samples",
    "build_cwru_canonical_dataset",
    "dataset_fingerprint",
    "normalize_fault_label",
    "split_by_group",
    "standardize_channel_sample",
    "validate_dataset_contract",
    "validate_default_benchmarks",
]
