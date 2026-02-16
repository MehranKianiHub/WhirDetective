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

    class_names = tuple(sorted({sample.label.value for sample in samples}))
    if len(class_names) <= 1:
        raise ValueError("at least two classes are required for classification")

    class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    inputs_np = np.stack([sample.features for sample in samples], axis=0).astype(np.float32)
    labels_np = np.asarray(
        [class_to_index[sample.label.value] for sample in samples],
        dtype=np.int64,
    )

    inputs = torch.from_numpy(inputs_np)
    labels = torch.from_numpy(labels_np)

    return CanonicalTensorDataset(
        inputs=inputs,
        labels=labels,
        class_names=class_names,
        class_to_index=class_to_index,
    )
