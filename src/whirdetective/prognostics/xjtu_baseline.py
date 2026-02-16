"""XJTU-SY prognostics baseline and KPI evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re

import numpy as np
import numpy.typing as npt

from whirdetective.data.adapters import iter_xjtu_csv_payloads, list_xjtu_parts
from whirdetective.data.splitting import assert_group_isolation, split_by_group


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class XjtuRunConfig:
    """Runtime configuration for XJTU-SY baseline evaluation."""

    root_dir: Path
    max_parts: int | None = None
    sample_stride: int = 25
    max_entries_per_bearing: int = 120
    csv_max_rows: int = 4096
    split_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        if self.max_parts is not None and self.max_parts <= 0:
            raise ValueError("max_parts must be > 0 when set")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be > 0")
        if self.max_entries_per_bearing <= 0:
            raise ValueError("max_entries_per_bearing must be > 0")
        if self.csv_max_rows <= 0:
            raise ValueError("csv_max_rows must be > 0")
        for value in (self.train_ratio, self.val_ratio, self.test_ratio):
            if value <= 0.0:
                raise ValueError("split ratios must be > 0")
        if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) > 1e-9:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


@dataclass(frozen=True, slots=True)
class XjtuPrognosticsTargets:
    """Step 4 prognostics KPI targets for XJTU-SY baseline."""

    min_bearings: int = 3
    min_samples: int = 120
    max_test_mae: float = 0.45
    min_test_spearman: float = 0.90
    min_mean_group_spearman: float = 0.80

    def __post_init__(self) -> None:
        if self.min_bearings <= 0:
            raise ValueError("min_bearings must be > 0")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be > 0")
        if self.max_test_mae <= 0.0:
            raise ValueError("max_test_mae must be > 0")
        if self.min_test_spearman < -1.0 or self.min_test_spearman > 1.0:
            raise ValueError("min_test_spearman must be in [-1, 1]")
        if self.min_mean_group_spearman < -1.0 or self.min_mean_group_spearman > 1.0:
            raise ValueError("min_mean_group_spearman must be in [-1, 1]")


@dataclass(frozen=True, slots=True)
class XjtuPrognosticsEvaluation:
    """KPI evaluation outcome for the XJTU-SY prognostics baseline."""

    passed: bool
    failed_checks: tuple[str, ...]
    num_bearings: int
    num_samples: int
    test_mae: float
    test_spearman: float
    mean_group_spearman: float


@dataclass(frozen=True, slots=True)
class XjtuPrognosticsResult:
    """End-to-end result payload from one XJTU-SY baseline run."""

    evaluation: XjtuPrognosticsEvaluation
    selected_lambda: float
    val_mae: float
    part_files: tuple[str, ...]
    split_sizes: dict[str, int]


@dataclass(frozen=True, slots=True)
class _Record:
    condition: str
    bearing_id: str
    snapshot_index: int
    features: FloatArray
    target: float

    @property
    def group_id(self) -> str:
        return f"{self.condition}:{self.bearing_id}"


def run_xjtu_prognostics_baseline(
    *,
    config: XjtuRunConfig,
    targets: XjtuPrognosticsTargets | None = None,
) -> XjtuPrognosticsResult:
    """Run leakage-safe XJTU-SY baseline regression and evaluate KPI targets."""
    resolved_targets = XjtuPrognosticsTargets() if targets is None else targets
    records, part_files = _collect_records(config)
    split_sizes = {"train": 0, "val": 0, "test": 0}
    if not records:
        evaluation = XjtuPrognosticsEvaluation(
            passed=False,
            failed_checks=("no_xjtu_records_collected",),
            num_bearings=0,
            num_samples=0,
            test_mae=float("nan"),
            test_spearman=float("nan"),
            mean_group_spearman=float("nan"),
        )
        return XjtuPrognosticsResult(
            evaluation=evaluation,
            selected_lambda=float("nan"),
            val_mae=float("nan"),
            part_files=part_files,
            split_sizes=split_sizes,
        )

    group_ids = tuple(record.group_id for record in records)
    split = split_by_group(
        group_ids,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
    )
    assert_group_isolation(group_ids, split)
    split_sizes = {
        "train": len(split.train_indices),
        "val": len(split.val_indices),
        "test": len(split.test_indices),
    }

    train_records = tuple(records[idx] for idx in split.train_indices)
    val_records = tuple(records[idx] for idx in split.val_indices)
    test_records = tuple(records[idx] for idx in split.test_indices)
    x_train, y_train = _records_to_arrays(train_records)
    x_val, y_val = _records_to_arrays(val_records)
    x_test, y_test = _records_to_arrays(test_records)

    selected_lambda, val_mae, weights, feature_mean, feature_std = _fit_ridge_with_validation(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )
    x_test_normalized = (x_test - feature_mean) / feature_std
    y_pred = np.clip(x_test_normalized @ weights, 0.0, 1.0)

    test_mae = float(np.mean(np.abs(y_pred - y_test)))
    test_spearman = _spearman(y_test, y_pred)
    mean_group_spearman = _mean_group_spearman(test_records, y_pred)
    num_bearings = len({record.group_id for record in records})
    num_samples = len(records)
    evaluation = _evaluate_targets(
        targets=resolved_targets,
        num_bearings=num_bearings,
        num_samples=num_samples,
        test_mae=test_mae,
        test_spearman=test_spearman,
        mean_group_spearman=mean_group_spearman,
    )
    return XjtuPrognosticsResult(
        evaluation=evaluation,
        selected_lambda=selected_lambda,
        val_mae=val_mae,
        part_files=part_files,
        split_sizes=split_sizes,
    )


def _collect_records(config: XjtuRunConfig) -> tuple[tuple[_Record, ...], tuple[str, ...]]:
    parts = list_xjtu_parts(config.root_dir)
    if config.max_parts is not None:
        parts = parts[: config.max_parts]

    by_bearing: dict[tuple[str, str], dict[int, FloatArray]] = {}
    for part_file in parts:
        payloads = iter_xjtu_csv_payloads(part_file)
        for entry, payload in payloads:
            if entry.snapshot_index % config.sample_stride != 0:
                continue
            features = _extract_snapshot_features(
                payload=payload,
                snapshot_index=entry.snapshot_index,
                condition=entry.condition,
                csv_max_rows=config.csv_max_rows,
            )
            key = (entry.condition, entry.bearing_id)
            per_index = by_bearing.setdefault(key, {})
            per_index[entry.snapshot_index] = features

    records: list[_Record] = []
    for (condition, bearing_id), by_index in by_bearing.items():
        if len(by_index) < 10:
            continue
        ordered_indices = sorted(by_index.keys())
        selected_indices = _downsample_indices(
            ordered_indices,
            max_count=config.max_entries_per_bearing,
        )
        max_index = float(max(selected_indices))
        if max_index <= 0.0:
            continue
        for snapshot_index in selected_indices:
            features = by_index[snapshot_index]
            target = (max_index - float(snapshot_index)) / max_index
            records.append(
                _Record(
                    condition=condition,
                    bearing_id=bearing_id,
                    snapshot_index=snapshot_index,
                    features=features,
                    target=float(target),
                )
            )
    records.sort(key=lambda record: (record.condition, record.bearing_id, record.snapshot_index))
    return tuple(records), tuple(str(path) for path in parts)


def _extract_snapshot_features(
    *,
    payload: bytes,
    snapshot_index: int,
    condition: str,
    csv_max_rows: int,
) -> FloatArray:
    values = np.loadtxt(
        BytesIO(payload),
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
        max_rows=csv_max_rows,
    )
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("XJTU CSV payload must contain at least two columns")

    horizontal = values[:, 0]
    vertical = values[:, 1]
    if horizontal.size == 0 or vertical.size == 0:
        raise ValueError("XJTU CSV payload contains empty channels")

    hz, load_kn = _parse_condition(condition)
    correlation = float(np.corrcoef(horizontal, vertical)[0, 1]) if horizontal.size > 1 else 0.0
    return np.asarray(
        (
            float(np.mean(horizontal)),
            float(np.std(horizontal, ddof=0)),
            float(np.sqrt(np.mean(np.square(horizontal)))),
            float(np.max(np.abs(horizontal))),
            float(np.mean(vertical)),
            float(np.std(vertical, ddof=0)),
            float(np.sqrt(np.mean(np.square(vertical)))),
            float(np.max(np.abs(vertical))),
            correlation,
            float(hz) / 100.0,
            float(load_kn) / 100.0,
            float(np.log1p(snapshot_index)),
            float(np.sqrt(snapshot_index)),
        ),
        dtype=np.float64,
    )


def _parse_condition(condition: str) -> tuple[int, int]:
    match = re.match(r"^(?P<hz>\d+(?:\.\d+)?)Hz(?P<kn>\d+(?:\.\d+)?)kN$", condition)
    if match is None:
        return 0, 0
    hz = int(float(match.group("hz")))
    load_kn = int(float(match.group("kn")))
    return hz, load_kn


def _downsample_indices(indices: list[int], *, max_count: int) -> tuple[int, ...]:
    if len(indices) <= max_count:
        return tuple(indices)
    positions = np.linspace(0, len(indices) - 1, max_count, dtype=np.int64)
    selected = [indices[int(position)] for position in positions]
    return tuple(sorted(set(selected)))


def _records_to_arrays(records: tuple[_Record, ...]) -> tuple[FloatArray, FloatArray]:
    if not records:
        raise ValueError("records must not be empty")
    x = np.stack([record.features for record in records], axis=0).astype(np.float64)
    y = np.asarray([record.target for record in records], dtype=np.float64)
    return x, y


def _fit_ridge_with_validation(
    *,
    x_train: FloatArray,
    y_train: FloatArray,
    x_val: FloatArray,
    y_val: FloatArray,
) -> tuple[float, float, FloatArray, FloatArray, FloatArray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    x_train_normalized = (x_train - mean) / std
    x_val_normalized = (x_val - mean) / std

    lambda_grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    best_lambda = float(lambda_grid[0])
    best_val_mae = float("inf")
    best_weights: FloatArray | None = None
    for lambda_value in lambda_grid:
        identity = np.eye(x_train_normalized.shape[1], dtype=np.float64)
        weights = np.linalg.solve(
            x_train_normalized.T @ x_train_normalized + lambda_value * identity,
            x_train_normalized.T @ y_train,
        )
        val_pred = np.clip(x_val_normalized @ weights, 0.0, 1.0)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_lambda = float(lambda_value)
            best_weights = weights

    if best_weights is None:
        raise RuntimeError("failed to fit ridge baseline")
    return best_lambda, best_val_mae, best_weights, mean, std


def _evaluate_targets(
    *,
    targets: XjtuPrognosticsTargets,
    num_bearings: int,
    num_samples: int,
    test_mae: float,
    test_spearman: float,
    mean_group_spearman: float,
) -> XjtuPrognosticsEvaluation:
    failed_checks: list[str] = []
    if num_bearings < targets.min_bearings:
        failed_checks.append(
            f"num_bearings {num_bearings} < min_bearings {targets.min_bearings}"
        )
    if num_samples < targets.min_samples:
        failed_checks.append(
            f"num_samples {num_samples} < min_samples {targets.min_samples}"
        )
    if test_mae > targets.max_test_mae:
        failed_checks.append(f"test_mae {test_mae:.4f} > max_test_mae {targets.max_test_mae:.4f}")
    if test_spearman < targets.min_test_spearman:
        failed_checks.append(
            f"test_spearman {test_spearman:.4f} < min_test_spearman {targets.min_test_spearman:.4f}"
        )
    if mean_group_spearman < targets.min_mean_group_spearman:
        failed_checks.append(
            "mean_group_spearman "
            f"{mean_group_spearman:.4f} < min_mean_group_spearman {targets.min_mean_group_spearman:.4f}"
        )
    return XjtuPrognosticsEvaluation(
        passed=(len(failed_checks) == 0),
        failed_checks=tuple(failed_checks),
        num_bearings=num_bearings,
        num_samples=num_samples,
        test_mae=test_mae,
        test_spearman=test_spearman,
        mean_group_spearman=mean_group_spearman,
    )


def _spearman(y_true: FloatArray, y_pred: FloatArray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("spearman inputs must be non-empty")
    rank_true = np.argsort(np.argsort(y_true)).astype(np.float64)
    rank_pred = np.argsort(np.argsort(y_pred)).astype(np.float64)
    std_true = float(np.std(rank_true))
    std_pred = float(np.std(rank_pred))
    if std_true < 1e-12 or std_pred < 1e-12:
        return 0.0
    rank_true = (rank_true - np.mean(rank_true)) / std_true
    rank_pred = (rank_pred - np.mean(rank_pred)) / std_pred
    return float(np.mean(rank_true * rank_pred))


def _mean_group_spearman(records: tuple[_Record, ...], predictions: FloatArray) -> float:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for record, prediction in zip(records, predictions):
        grouped.setdefault(record.group_id, []).append((record.target, float(prediction)))

    scores: list[float] = []
    for group_rows in grouped.values():
        if len(group_rows) < 3:
            continue
        y_true = np.asarray([item[0] for item in group_rows], dtype=np.float64)
        y_pred = np.asarray([item[1] for item in group_rows], dtype=np.float64)
        scores.append(_spearman(y_true, y_pred))

    if not scores:
        return 0.0
    return float(np.mean(np.asarray(scores, dtype=np.float64)))
