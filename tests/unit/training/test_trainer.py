"""Tests for baseline trainer behavior."""

from __future__ import annotations

import torch

from whirdetective.ml import BaselineBearingCNN
from whirdetective.training import BaselineTrainer, TrainerConfig


def test_baseline_trainer_improves_on_easy_dataset() -> None:
    torch.manual_seed(0)
    channels, samples = 3, 64
    n_train, n_val = 64, 32

    train_class0 = torch.randn(n_train // 2, channels, samples) * 0.1 - 1.0
    train_class1 = torch.randn(n_train // 2, channels, samples) * 0.1 + 1.0
    val_class0 = torch.randn(n_val // 2, channels, samples) * 0.1 - 1.0
    val_class1 = torch.randn(n_val // 2, channels, samples) * 0.1 + 1.0

    train_x = torch.cat((train_class0, train_class1), dim=0)
    train_y = torch.cat(
        (
            torch.zeros(n_train // 2, dtype=torch.int64),
            torch.ones(n_train // 2, dtype=torch.int64),
        ),
        dim=0,
    )
    val_x = torch.cat((val_class0, val_class1), dim=0)
    val_y = torch.cat(
        (
            torch.zeros(n_val // 2, dtype=torch.int64),
            torch.ones(n_val // 2, dtype=torch.int64),
        ),
        dim=0,
    )

    model = BaselineBearingCNN(input_channels=channels, num_classes=2)
    trainer = BaselineTrainer(
        model=model,
        config=TrainerConfig(epochs=6, batch_size=16, learning_rate=1e-3, seed=3),
    )
    history = trainer.fit(
        train_inputs=train_x,
        train_labels=train_y,
        val_inputs=val_x,
        val_labels=val_y,
    )
    accuracy = trainer.evaluate_accuracy(val_x, val_y)

    assert history.train_losses[-1] < history.train_losses[0]
    assert accuracy >= 0.9
