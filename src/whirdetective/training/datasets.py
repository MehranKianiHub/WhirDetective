"""Tensor dataset conversion from canonical training samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from whirdetective.data import CanonicalTrainingSample


@dataclass(frozen=True, slots=True)
class CanonicalTensorDataset:
    """Torch-ready dataset tensors and label mapping metadata."""

    inputs: torch.Tensor
    labels: torch.Tensor
    class_names: tuple[str, ...]
    class_to_index: dict[str, int]


def canonical_samples_to_dataset(
    samples: tuple[CanonicalTrainingSample, ...],
    *,
    class_to_index: dict[str, int] | None = None,
) -> CanonicalTensorDataset:
    """Convert canonical samples into tensors with deterministic class indices."""
    if not samples:
        raise ValueError("samples must not be empty")

    first_shape = samples[0].features.shape
    if first_shape[0] <= 0 or first_shape[1] <= 0:
        raise ValueError("invalid sample feature shape")

    for sample in samples[1:]:
        if sample.features.shape != first_shape:
            raise ValueError("all samples must share identical feature shape")

    if class_to_index is None:
        class_names = tuple(sorted({sample.label.value for sample in samples}))
        if len(class_names) <= 1:
            raise ValueError("at least two classes are required for classification")
        resolved_class_to_index = {
            class_name: idx for idx, class_name in enumerate(class_names)
        }
    else:
        resolved_class_to_index = _validate_class_mapping(class_to_index)
        class_names = tuple(
            class_name
            for class_name, _ in sorted(
                resolved_class_to_index.items(),
                key=lambda item: item[1],
            )
        )

    inputs_np = np.stack([sample.features for sample in samples], axis=0).astype(np.float32)
    label_indices: list[int] = []
    for sample in samples:
        label_name = sample.label.value
        if label_name not in resolved_class_to_index:
            raise ValueError(
                f"Sample label {label_name!r} is not present in provided class mapping"
            )
        label_indices.append(resolved_class_to_index[label_name])
    labels_np = np.asarray(label_indices, dtype=np.int64)

    inputs = torch.from_numpy(inputs_np)
    labels = torch.from_numpy(labels_np)

    return CanonicalTensorDataset(
        inputs=inputs,
        labels=labels,
        class_names=class_names,
        class_to_index=resolved_class_to_index,
    )


def _validate_class_mapping(class_to_index: dict[str, int]) -> dict[str, int]:
    if not class_to_index:
        raise ValueError("class_to_index must not be empty")
    indices = sorted(class_to_index.values())
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError("class_to_index values must be contiguous indices starting at 0")
    if len(set(class_to_index.keys())) != len(class_to_index):
        raise ValueError("class_to_index keys must be unique")
    return dict(class_to_index)
