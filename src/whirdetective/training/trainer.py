"""Deterministic PyTorch trainer for baseline fault diagnosis model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from whirdetective.ml import BaselineBearingCNN


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Training hyperparameters for baseline model."""

    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"
    seed: int = 7

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Epoch-wise training/validation losses for diagnostics."""

    train_losses: tuple[float, ...]
    val_losses: tuple[float, ...]


class BaselineTrainer:
    """Train/evaluate BaselineBearingCNN using canonical feature tensors."""

    def __init__(self, *, model: BaselineBearingCNN, config: TrainerConfig) -> None:
        torch.manual_seed(config.seed)
        self._model = model
        self._config = config
        self._device = torch.device(config.device)
        self._model.to(self._device)
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    @property
    def model(self) -> BaselineBearingCNN:
        """Expose trained model instance."""
        return self._model

    def fit(
        self,
        *,
        train_inputs: torch.Tensor,
        train_labels: torch.Tensor,
        val_inputs: torch.Tensor,
        val_labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> TrainingHistory:
        """Run training and return epoch losses."""
        if class_weights is not None:
            if class_weights.ndim != 1:
                raise ValueError("class_weights must be 1D when provided")
            self._criterion = nn.CrossEntropyLoss(weight=class_weights.to(self._device))
        else:
            self._criterion = nn.CrossEntropyLoss()

        train_loader = _make_loader(
            train_inputs,
            train_labels,
            batch_size=self._config.batch_size,
            shuffle=True,
            seed=self._config.seed,
        )
        val_loader = _make_loader(
            val_inputs,
            val_labels,
            batch_size=self._config.batch_size,
            shuffle=False,
            seed=None,
        )

        train_losses: list[float] = []
        val_losses: list[float] = []
        for _ in range(self._config.epochs):
            train_losses.append(self._run_train_epoch(train_loader))
            val_losses.append(self.evaluate_loss(val_loader))

        return TrainingHistory(
            train_losses=tuple(train_losses),
            val_losses=tuple(val_losses),
        )

    def predict_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return model logits for a batch of inputs."""
        self._model.eval()
        with torch.no_grad():
            logits = self._model(inputs.to(self._device))
        return cast(torch.Tensor, logits.cpu())

    def predict_classes(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return argmax class indices for inputs."""
        logits = self.predict_logits(inputs)
        return torch.argmax(logits, dim=1)

    def evaluate_accuracy(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute accuracy over provided tensors."""
        preds = self.predict_classes(inputs)
        labels_cpu = labels.cpu()
        if labels_cpu.numel() == 0:
            raise ValueError("labels must not be empty")
        return float((preds == labels_cpu).float().mean().item())

    def evaluate_loss(self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Evaluate average cross-entropy loss."""
        self._model.eval()
        loss_sum = 0.0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                logits = self._model(x_batch.to(self._device))
                loss = self._criterion(logits, y_batch.to(self._device))
                batch_size = int(y_batch.shape[0])
                loss_sum += float(loss.item()) * batch_size
                total += batch_size
        if total == 0:
            raise ValueError("empty loader")
        return loss_sum / total

    def _run_train_epoch(self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> float:
        self._model.train()
        loss_sum = 0.0
        total = 0
        for x_batch, y_batch in loader:
            self._optimizer.zero_grad(set_to_none=True)
            logits = self._model(x_batch.to(self._device))
            loss = self._criterion(logits, y_batch.to(self._device))
            loss.backward()
            self._optimizer.step()

            batch_size = int(y_batch.shape[0])
            loss_sum += float(loss.item()) * batch_size
            total += batch_size

        if total == 0:
            raise ValueError("empty training loader")
        return loss_sum / total


def _make_loader(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [batch, channels, samples]")
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")
    if int(inputs.shape[0]) != int(labels.shape[0]):
        raise ValueError("inputs and labels batch sizes must match")

    dataset = cast(Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(inputs, labels))
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
