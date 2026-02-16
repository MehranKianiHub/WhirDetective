"""Post-hoc confidence calibration for classification logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    """Learnable temperature scaling module for logits calibration."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be > 0")
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(initial_temperature), dtype=torch.float32))
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Positive scalar temperature."""
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature


@dataclass(frozen=True, slots=True)
class TemperatureOptimizationResult:
    """Calibration optimization result."""

    temperature: float
    nll_before: float
    nll_after: float


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Cross-entropy NLL helper."""
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [batch, classes]")
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        return float(criterion(logits, labels).item())


def optimize_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int = 50,
) -> TemperatureOptimizationResult:
    """Fit temperature on validation logits/labels via NLL minimization."""
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [batch, classes]")
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")
    if int(logits.shape[0]) != int(labels.shape[0]):
        raise ValueError("logits and labels batch size must match")

    scaler = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=0.1, max_iter=max_iter)

    nll_before = negative_log_likelihood(logits, labels)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return cast(torch.Tensor, loss)

    optimizer.step(closure)  # type: ignore[no-untyped-call]

    with torch.no_grad():
        calibrated_logits = scaler(logits)
    nll_after = negative_log_likelihood(calibrated_logits, labels)

    return TemperatureOptimizationResult(
        temperature=float(scaler.temperature.item()),
        nll_before=nll_before,
        nll_after=nll_after,
    )
