"""PyTorch models for WhirDetective ML workloads."""

from whirdetective.ml.models import BaselineBearingCNN
from whirdetective.ml.sensor_projection import (
    ProjectedSample,
    ProjectionPolicy,
    SensorRole,
    SensorSetProjector,
)

__all__ = [
    "BaselineBearingCNN",
    "ProjectedSample",
    "ProjectionPolicy",
    "SensorRole",
    "SensorSetProjector",
]
