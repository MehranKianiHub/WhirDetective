"""Model card metadata structure for transparent model reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from whirdetective.evaluation.metrics import AbstentionMetrics, CalibrationMetrics, ClassificationMetrics


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Concise metadata card for model limitations and evaluation context."""

    model_name: str
    created_at_utc: str
    training_data_fingerprint: str
    class_names: tuple[str, ...]
    classification: ClassificationMetrics
    calibration: CalibrationMetrics
    abstention: AbstentionMetrics
    known_limitations: tuple[str, ...]
    safety_boundary_statement: str

    @staticmethod
    def now_timestamp() -> str:
        """ISO-8601 UTC timestamp helper."""
        return datetime.now(UTC).isoformat()


def model_card_to_jsonable(card: ModelCard) -> dict[str, Any]:
    """Serialize model card into JSON-safe structure."""
    payload = asdict(card)
    payload["classification"]["confusion_matrix"] = payload["classification"]["confusion_matrix"].tolist()
    return payload
