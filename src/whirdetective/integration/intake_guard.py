"""Deterministic telemetry intake guardrails for BootCtrl-integrated payloads."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from whirdetective.integration.bootctrl_contracts import BootCtrlTelemetrySample, BootCtrlTelemetrySource


class RejectReason(StrEnum):
    """Reasons why a telemetry sample is rejected by intake policies."""

    STALE_TIMESTAMP = "stale_timestamp"
    FUTURE_TIMESTAMP = "future_timestamp"
    TIMESTAMP_REGRESSION = "timestamp_regression"
    DUPLICATE_EVENT = "duplicate_event"


@dataclass(frozen=True, slots=True)
class IntakePolicy:
    """Runtime policy for deterministic intake validation."""

    max_age_ms: int = 300_000
    max_future_skew_ms: int = 2_000
    allow_out_of_order_ms: int = 0
    dedupe_history_size: int = 100_000

    def __post_init__(self) -> None:
        if self.max_age_ms <= 0:
            raise ValueError("max_age_ms must be > 0")
        if self.max_future_skew_ms < 0:
            raise ValueError("max_future_skew_ms must be >= 0")
        if self.allow_out_of_order_ms < 0:
            raise ValueError("allow_out_of_order_ms must be >= 0")
        if self.dedupe_history_size <= 0:
            raise ValueError("dedupe_history_size must be > 0")


@dataclass(frozen=True, slots=True)
class RejectedSample:
    """Rejected sample with deterministic reason and detail."""

    sample: BootCtrlTelemetrySample
    reason: RejectReason
    detail: str


@dataclass(frozen=True, slots=True)
class IntakeMetrics:
    """Intake counters for observability and quality tracking."""

    total_seen: int
    accepted: int
    rejected: int
    rejected_by_reason: tuple[tuple[str, int], ...]


class TelemetryIntakeGuard:
    """In-memory guard for idempotent, deterministic telemetry intake."""

    def __init__(self, policy: IntakePolicy) -> None:
        self._policy = policy
        self._seen_event_ids: set[str] = set()
        self._event_order: deque[str] = deque(maxlen=policy.dedupe_history_size)
        self._last_timestamp_by_stream: dict[tuple[str, str, BootCtrlTelemetrySource], int] = {}

        self._total_seen = 0
        self._accepted = 0
        self._rejected = 0
        self._rejected_by_reason: dict[RejectReason, int] = {reason: 0 for reason in RejectReason}

    def evaluate(
        self,
        sample: BootCtrlTelemetrySample,
        *,
        ingest_time_ms: int,
        event_id: str | None = None,
    ) -> RejectedSample | None:
        """Return `None` if accepted, otherwise rejection details."""
        if ingest_time_ms <= 0:
            raise ValueError("ingest_time_ms must be > 0")

        self._total_seen += 1
        dedupe_key = event_id.strip() if event_id is not None else self._fingerprint(sample)

        if self._is_duplicate(dedupe_key):
            return self._reject(
                sample,
                RejectReason.DUPLICATE_EVENT,
                f"duplicate event id: {dedupe_key}",
            )

        age_ms = ingest_time_ms - sample.timestamp_ms
        if age_ms > self._policy.max_age_ms:
            return self._reject(
                sample,
                RejectReason.STALE_TIMESTAMP,
                f"sample age {age_ms} ms exceeds max_age_ms={self._policy.max_age_ms}",
            )

        if age_ms < -self._policy.max_future_skew_ms:
            return self._reject(
                sample,
                RejectReason.FUTURE_TIMESTAMP,
                f"sample timestamp is {-age_ms} ms ahead of ingest clock",
            )

        stream_key = (sample.device_id, sample.signal_key, sample.source)
        previous_ts = self._last_timestamp_by_stream.get(stream_key)
        if previous_ts is not None and sample.timestamp_ms + self._policy.allow_out_of_order_ms < previous_ts:
            return self._reject(
                sample,
                RejectReason.TIMESTAMP_REGRESSION,
                (
                    f"timestamp regression: sample={sample.timestamp_ms}, "
                    f"last={previous_ts}, allow_out_of_order_ms={self._policy.allow_out_of_order_ms}"
                ),
            )

        self._remember_event(dedupe_key)
        self._last_timestamp_by_stream[stream_key] = (
            max(previous_ts, sample.timestamp_ms) if previous_ts is not None else sample.timestamp_ms
        )
        self._accepted += 1
        return None

    def process_batch(
        self,
        samples: tuple[BootCtrlTelemetrySample, ...],
        *,
        ingest_time_ms: int,
        event_ids: tuple[str | None, ...] | None = None,
    ) -> tuple[tuple[BootCtrlTelemetrySample, ...], tuple[RejectedSample, ...]]:
        """Process a batch and return accepted and rejected samples."""
        if event_ids is not None and len(event_ids) != len(samples):
            raise ValueError("event_ids length must match samples length")

        accepted: list[BootCtrlTelemetrySample] = []
        rejected: list[RejectedSample] = []

        for idx, sample in enumerate(samples):
            event_id = None if event_ids is None else event_ids[idx]
            decision = self.evaluate(sample, ingest_time_ms=ingest_time_ms, event_id=event_id)
            if decision is None:
                accepted.append(sample)
            else:
                rejected.append(decision)

        return tuple(accepted), tuple(rejected)

    @property
    def metrics(self) -> IntakeMetrics:
        """Return intake counters in stable order for tests/reporting."""
        return IntakeMetrics(
            total_seen=self._total_seen,
            accepted=self._accepted,
            rejected=self._rejected,
            rejected_by_reason=tuple(
                (reason.value, self._rejected_by_reason[reason]) for reason in RejectReason
            ),
        )

    def _reject(
        self,
        sample: BootCtrlTelemetrySample,
        reason: RejectReason,
        detail: str,
    ) -> RejectedSample:
        self._rejected += 1
        self._rejected_by_reason[reason] += 1
        return RejectedSample(sample=sample, reason=reason, detail=detail)

    def _is_duplicate(self, event_id: str) -> bool:
        return event_id in self._seen_event_ids

    def _remember_event(self, event_id: str) -> None:
        if len(self._event_order) == self._event_order.maxlen:
            oldest = self._event_order.popleft()
            self._seen_event_ids.remove(oldest)

        self._event_order.append(event_id)
        self._seen_event_ids.add(event_id)

    @staticmethod
    def _fingerprint(sample: BootCtrlTelemetrySample) -> str:
        return (
            f"{sample.source.value}|{sample.device_id}|{sample.signal_key}|"
            f"{sample.timestamp_ms}|{sample.value:.12g}"
        )
