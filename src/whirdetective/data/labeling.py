"""Canonical bearing fault ontology and dataset label normalization."""

from __future__ import annotations

from enum import StrEnum


class BearingFaultLabel(StrEnum):
    """Canonical labels used across heterogeneous bearing datasets."""

    HEALTHY = "healthy"
    INNER_RACE = "inner_race"
    OUTER_RACE = "outer_race"
    BALL = "ball"
    CAGE = "cage"
    COMBINED = "combined"
    UNKNOWN = "unknown"


_GENERIC_LABEL_MAP: dict[str, BearingFaultLabel] = {
    "normal": BearingFaultLabel.HEALTHY,
    "healthy": BearingFaultLabel.HEALTHY,
    "ok": BearingFaultLabel.HEALTHY,
    "ir": BearingFaultLabel.INNER_RACE,
    "inner": BearingFaultLabel.INNER_RACE,
    "innerrace": BearingFaultLabel.INNER_RACE,
    "inner_race": BearingFaultLabel.INNER_RACE,
    "or": BearingFaultLabel.OUTER_RACE,
    "outer": BearingFaultLabel.OUTER_RACE,
    "outerrace": BearingFaultLabel.OUTER_RACE,
    "outer_race": BearingFaultLabel.OUTER_RACE,
    "ball": BearingFaultLabel.BALL,
    "rolling_element": BearingFaultLabel.BALL,
    "cage": BearingFaultLabel.CAGE,
    "combined": BearingFaultLabel.COMBINED,
    "compound": BearingFaultLabel.COMBINED,
    "mixed": BearingFaultLabel.COMBINED,
    "unknown": BearingFaultLabel.UNKNOWN,
}

_DATASET_OVERRIDES: dict[str, dict[str, BearingFaultLabel]] = {
    # CWRU naming conventions commonly used in derived CSV/metadata exports.
    "cwru": {
        "b": BearingFaultLabel.BALL,
        "ir": BearingFaultLabel.INNER_RACE,
        "or": BearingFaultLabel.OUTER_RACE,
        "normal": BearingFaultLabel.HEALTHY,
    },
    # Paderborn coarse buckets often used in benchmark scripts.
    "paderborn": {
        "healthy": BearingFaultLabel.HEALTHY,
        "artificial": BearingFaultLabel.COMBINED,
        "real": BearingFaultLabel.COMBINED,
    },
}


def normalize_fault_label(raw_label: str, *, dataset: str | None = None) -> BearingFaultLabel:
    """Normalize free-form dataset label strings into canonical ontology."""
    normalized = _normalize_token(raw_label)
    if dataset is not None:
        dataset_key = dataset.strip().lower()
        dataset_map = _DATASET_OVERRIDES.get(dataset_key)
        if dataset_map is not None and normalized in dataset_map:
            return dataset_map[normalized]

    return _GENERIC_LABEL_MAP.get(normalized, BearingFaultLabel.UNKNOWN)


def _normalize_token(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")
