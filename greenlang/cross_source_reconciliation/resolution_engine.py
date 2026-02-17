# -*- coding: utf-8 -*-
"""
Resolution Engine - AGENT-DATA-015 (Engine 5 of 7)

Pure-Python resolution engine that applies configurable strategies to
resolve discrepancies detected during cross-source reconciliation and
assembles golden records from multiple source records.

Zero-Hallucination: All calculations use deterministic Python
arithmetic (math, statistics, collections). No LLM calls for numeric
computations. No external libraries beyond the standard library.

Supported resolution strategies:
    - Priority Wins:      Highest-credibility source wins
    - Most Recent:        Most recently timestamped source wins
    - Weighted Average:   Credibility-weighted average (numeric only)
    - Most Complete:      Source with fewest nulls wins
    - Consensus:          Majority vote across 3+ sources
    - Manual Review:      Flag for human decision or apply manual value
    - Custom:             User-supplied resolver function

Example:
    >>> from greenlang.cross_source_reconciliation.resolution_engine import (
    ...     ResolutionEngine,
    ... )
    >>> engine = ResolutionEngine()
    >>> discrepancy = Discrepancy(
    ...     discrepancy_id="d-001",
    ...     entity_id="entity-1",
    ...     field_name="emissions_total",
    ...     field_type=FieldType.NUMERIC,
    ...     severity=DiscrepancySeverity.MEDIUM,
    ...     source_values={"src-a": 100.0, "src-b": 110.0},
    ... )
    >>> creds = {
    ...     "src-a": SourceCredibility(source_id="src-a", source_name="ERP",
    ...         credibility_score=0.9, priority=1),
    ...     "src-b": SourceCredibility(source_id="src-b", source_name="Invoice",
    ...         credibility_score=0.7, priority=2),
    ... }
    >>> decision = engine.resolve_discrepancy(
    ...     discrepancy=discrepancy,
    ...     strategy=ResolutionStrategy.PRIORITY_WINS,
    ...     source_credibilities=creds,
    ...     source_records={"src-a": {"emissions_total": 100.0},
    ...                     "src-b": {"emissions_total": 110.0}},
    ... )
    >>> assert decision.status == ResolutionStatus.RESOLVED

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from greenlang.cross_source_reconciliation.metrics import (
    inc_errors,
    inc_golden_records,
    inc_resolutions,
    observe_confidence,
    observe_duration,
)
from greenlang.cross_source_reconciliation.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _normalize_value(value: Any) -> Any:
    """Normalize a value for deterministic serialization.

    Handles float precision, NaN/Inf edge cases, and recursive
    normalization of nested structures.

    Args:
        value: Any Python value to normalize.

    Returns:
        Normalized value safe for deterministic JSON serialization.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "__NaN__"
        if math.isinf(value):
            return "__Inf__" if value > 0 else "__-Inf__"
        return round(value, 10)
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return value


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, number, or other).

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    normalized = _normalize_value(data)
    serialized = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_numeric(value: Any) -> bool:
    """Check whether a value is numeric (int or float, not NaN/Inf).

    Args:
        value: Value to check.

    Returns:
        True if value is a finite int or float.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return False
        return True
    return False


def _try_parse_numeric(value: Any) -> Optional[float]:
    """Attempt to parse a value as a float.

    Args:
        value: Value to parse (str, int, float, or other).

    Returns:
        Parsed float or None if not parseable.
    """
    if _is_numeric(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
            if not math.isnan(parsed) and not math.isinf(parsed):
                return parsed
        except (ValueError, TypeError):
            pass
    return None


def _is_empty(value: Any) -> bool:
    """Check whether a value is null, empty string, or NaN.

    Args:
        value: Value to check.

    Returns:
        True if the value is considered empty.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _compute_completeness(record: Dict[str, Any]) -> float:
    """Compute completeness ratio for a record (non-empty fields / total).

    Args:
        record: Dictionary of field name to value.

    Returns:
        Completeness ratio between 0.0 and 1.0.
    """
    if not record:
        return 0.0
    total = len(record)
    non_empty = sum(1 for v in record.values() if not _is_empty(v))
    return non_empty / total


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ResolutionStrategy(str, Enum):
    """Strategy for resolving discrepancies between sources.

    PRIORITY_WINS: Pick value from the highest-credibility/priority source.
    MOST_RECENT: Pick value from the source with the most recent data.
    WEIGHTED_AVERAGE: Compute credibility-weighted average (numeric only).
    MOST_COMPLETE: Pick value from source with fewest null fields.
    CONSENSUS: Majority vote across sources (3+ sources required).
    MANUAL_REVIEW: Flag for human review or accept manual override.
    CUSTOM: Apply user-provided resolver function.
    AUTO: Automatically select strategy based on severity and field type.
    """

    PRIORITY_WINS = "priority_wins"
    MOST_RECENT = "most_recent"
    WEIGHTED_AVERAGE = "weighted_average"
    MOST_COMPLETE = "most_complete"
    CONSENSUS = "consensus"
    MANUAL_REVIEW = "manual_review"
    CUSTOM = "custom"
    AUTO = "auto"


class ResolutionStatus(str, Enum):
    """Status of a resolution decision.

    RESOLVED: Discrepancy was resolved automatically.
    PENDING_REVIEW: Discrepancy is flagged for human review.
    MANUAL_OVERRIDE: Discrepancy was resolved via manual override.
    FAILED: Resolution attempt failed.
    SKIPPED: Resolution was skipped (e.g. no discrepancy).
    """

    RESOLVED = "resolved"
    PENDING_REVIEW = "pending_review"
    MANUAL_OVERRIDE = "manual_override"
    FAILED = "failed"
    SKIPPED = "skipped"


class DiscrepancySeverity(str, Enum):
    """Severity classification for a detected discrepancy.

    CRITICAL: Requires immediate manual intervention.
    HIGH: Significant deviation that may impact reporting accuracy.
    MEDIUM: Moderate deviation, auto-resolution acceptable.
    LOW: Minor deviation, auto-resolution preferred.
    INFO: Informational only, no action needed.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FieldType(str, Enum):
    """Data type classification for a field involved in a discrepancy.

    NUMERIC: Numeric value (int, float, decimal).
    STRING: Text/string value.
    DATE: Date or datetime value.
    BOOLEAN: Boolean value.
    CATEGORICAL: Value from a fixed set of categories.
    IDENTIFIER: Unique identifier (ID, code, SKU).
    COMPOSITE: Nested or structured value.
    """

    NUMERIC = "numeric"
    STRING = "string"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    COMPOSITE = "composite"


class CredibilityFactor(str, Enum):
    """Factors that contribute to a source's credibility score.

    DATA_QUALITY: Overall quality of data from the source.
    TIMELINESS: How recently the source was updated.
    AUTHORITY: Authoritative weight of the source.
    COMPLETENESS: Completeness of records from the source.
    ACCURACY: Historical accuracy of the source.
    CONSISTENCY: Internal consistency of the source's data.
    """

    DATA_QUALITY = "data_quality"
    TIMELINESS = "timeliness"
    AUTHORITY = "authority"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"


# ---------------------------------------------------------------------------
# Lightweight data models (self-contained until models.py adds these)
# ---------------------------------------------------------------------------


@dataclass
class SourceCredibility:
    """Credibility profile for a data source.

    Attributes:
        source_id: Unique identifier for the source.
        source_name: Human-readable source name.
        credibility_score: Overall credibility score (0.0-1.0).
        priority: Priority rank (1 = highest priority).
        factor_scores: Per-factor credibility breakdown.
        last_updated: ISO-formatted timestamp of the last update.
        metadata: Additional source metadata.
    """

    source_id: str = ""
    source_name: str = ""
    credibility_score: float = 0.5
    priority: int = 1
    factor_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Clamp credibility_score to [0.0, 1.0]."""
        self.credibility_score = max(0.0, min(1.0, self.credibility_score))
        if not self.last_updated:
            self.last_updated = _utcnow().isoformat()


@dataclass
class Discrepancy:
    """A detected discrepancy between source values for a field.

    Attributes:
        discrepancy_id: Unique identifier for this discrepancy.
        entity_id: Entity that the discrepancy belongs to.
        field_name: Name of the field with the discrepancy.
        field_type: Data type of the field.
        severity: Severity classification.
        source_values: Mapping of source_id to that source's value.
        expected_value: Optional expected/reference value.
        magnitude: Percentage deviation (for numeric fields).
        detected_at: ISO-formatted detection timestamp.
        metadata: Additional discrepancy context.
    """

    discrepancy_id: str = ""
    entity_id: str = ""
    field_name: str = ""
    field_type: FieldType = FieldType.STRING
    severity: DiscrepancySeverity = DiscrepancySeverity.MEDIUM
    source_values: Dict[str, Any] = field(default_factory=dict)
    expected_value: Any = None
    magnitude: float = 0.0
    detected_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults for missing fields."""
        if not self.discrepancy_id:
            self.discrepancy_id = f"disc-{uuid4().hex[:12]}"
        if not self.detected_at:
            self.detected_at = _utcnow().isoformat()


@dataclass
class ResolutionDecision:
    """The outcome of a resolution strategy applied to a discrepancy.

    Attributes:
        decision_id: Unique identifier for this decision.
        discrepancy_id: ID of the discrepancy that was resolved.
        entity_id: Entity that the discrepancy belongs to.
        field_name: Name of the resolved field.
        strategy: Resolution strategy that was applied.
        status: Status of the resolution.
        winning_source_id: Source whose value was selected.
        winning_source_name: Human-readable name of the winning source.
        resolved_value: The final resolved value.
        original_values: All source values that were considered.
        confidence: Confidence in the resolution (0.0-1.0).
        justification: Human-readable explanation of the decision.
        is_auto: Whether the resolution was automatic (True) or manual.
        resolved_at: ISO-formatted resolution timestamp.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance chain hash.
        metadata: Additional decision context.
    """

    decision_id: str = ""
    discrepancy_id: str = ""
    entity_id: str = ""
    field_name: str = ""
    strategy: str = ""
    status: str = ResolutionStatus.RESOLVED.value
    winning_source_id: str = ""
    winning_source_name: str = ""
    resolved_value: Any = None
    original_values: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    justification: str = ""
    is_auto: bool = True
    resolved_at: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults for missing fields."""
        if not self.decision_id:
            self.decision_id = f"res-{uuid4().hex[:12]}"
        if not self.resolved_at:
            self.resolved_at = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision to a dictionary.

        Returns:
            Dictionary representation of this decision.
        """
        return asdict(self)


@dataclass
class FieldLineage:
    """Provenance lineage for a single field in a golden record.

    Attributes:
        field_name: Name of the field.
        source_id: ID of the source that contributed the value.
        source_name: Human-readable source name.
        original_value: The original value from the source.
        resolved_value: The final value in the golden record.
        strategy: Resolution strategy applied (if any).
        confidence: Confidence in this field's value (0.0-1.0).
        was_discrepant: Whether the field had a discrepancy.
        metadata: Additional lineage context.
    """

    field_name: str = ""
    source_id: str = ""
    source_name: str = ""
    original_value: Any = None
    resolved_value: Any = None
    strategy: str = ""
    confidence: float = 1.0
    was_discrepant: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lineage to a dictionary.

        Returns:
            Dictionary representation of this lineage record.
        """
        return asdict(self)


@dataclass
class GoldenRecord:
    """A reconciled master record assembled from multiple sources.

    Attributes:
        record_id: Unique identifier for this golden record.
        entity_id: Entity this golden record represents.
        period: Reporting period (e.g. "2025-Q4").
        fields: Resolved field values (field_name -> value).
        field_sources: Mapping of field_name -> source_id that contributed.
        field_confidences: Mapping of field_name -> confidence score.
        total_confidence: Mean confidence across all fields.
        source_count: Number of sources that contributed.
        discrepancy_count: Number of discrepancies resolved.
        resolution_count: Number of resolution decisions applied.
        created_at: ISO-formatted creation timestamp.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance chain hash.
        metadata: Additional golden record context.
    """

    record_id: str = ""
    entity_id: str = ""
    period: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)
    field_sources: Dict[str, str] = field(default_factory=dict)
    field_confidences: Dict[str, float] = field(default_factory=dict)
    total_confidence: float = 0.0
    source_count: int = 0
    discrepancy_count: int = 0
    resolution_count: int = 0
    created_at: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults for missing fields."""
        if not self.record_id:
            self.record_id = f"gr-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize golden record to a dictionary.

        Returns:
            Dictionary representation of this golden record.
        """
        return asdict(self)


@dataclass
class ResolutionSummary:
    """Aggregate statistics for a batch of resolution decisions.

    Attributes:
        total_decisions: Total number of resolution decisions.
        resolved_count: Decisions with status RESOLVED.
        pending_count: Decisions with status PENDING_REVIEW.
        manual_count: Decisions with status MANUAL_OVERRIDE.
        failed_count: Decisions with status FAILED.
        skipped_count: Decisions with status SKIPPED.
        auto_count: Decisions made automatically.
        manual_review_count: Decisions requiring manual review.
        strategy_counts: Breakdown by strategy name.
        average_confidence: Mean confidence across all decisions.
        min_confidence: Minimum confidence observed.
        max_confidence: Maximum confidence observed.
        total_processing_time_ms: Cumulative processing time.
        summary_generated_at: ISO-formatted generation timestamp.
        provenance_hash: SHA-256 hash of the summary data.
    """

    total_decisions: int = 0
    resolved_count: int = 0
    pending_count: int = 0
    manual_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    auto_count: int = 0
    manual_review_count: int = 0
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    total_processing_time_ms: float = 0.0
    summary_generated_at: str = ""
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Set defaults for missing fields."""
        if not self.summary_generated_at:
            self.summary_generated_at = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize summary to a dictionary.

        Returns:
            Dictionary representation of this summary.
        """
        return asdict(self)


# ===========================================================================
# ResolutionEngine
# ===========================================================================


class ResolutionEngine:
    """Engine for resolving cross-source discrepancies and assembling golden records.

    Applies configurable resolution strategies to discrepancies detected
    during cross-source reconciliation. Supports automatic strategy
    selection based on discrepancy severity and field type. Assembles
    golden records by merging resolved field values from multiple sources
    with full provenance tracking.

    Zero-Hallucination Guarantees:
        - All numeric calculations use deterministic Python arithmetic
        - Weighted averages use explicit credibility weights
        - Majority vote uses collections.Counter for exact counting
        - No LLM calls in any calculation path
        - SHA-256 provenance hash on every decision and golden record

    Attributes:
        _provenance: ProvenanceTracker for SHA-256 audit trails.
        _total_resolutions: Running count of resolutions performed.
        _total_golden_records: Running count of golden records assembled.
        _timestamp_field: Default field name for temporal ordering.

    Example:
        >>> engine = ResolutionEngine()
        >>> discrepancy = Discrepancy(
        ...     field_name="co2_tonnes",
        ...     field_type=FieldType.NUMERIC,
        ...     severity=DiscrepancySeverity.MEDIUM,
        ...     source_values={"erp": 100.0, "invoice": 110.0},
        ... )
        >>> creds = {
        ...     "erp": SourceCredibility(source_id="erp",
        ...         credibility_score=0.9, priority=1),
        ...     "invoice": SourceCredibility(source_id="invoice",
        ...         credibility_score=0.7, priority=2),
        ... }
        >>> decision = engine.resolve_discrepancy(
        ...     discrepancy, ResolutionStrategy.PRIORITY_WINS, creds, {}
        ... )
        >>> assert decision.status == ResolutionStatus.RESOLVED.value
    """

    # Default timestamp field name for most-recent strategy
    DEFAULT_TIMESTAMP_FIELD = "data_timestamp"

    def __init__(
        self,
        provenance_tracker: Optional[ProvenanceTracker] = None,
        timestamp_field: str = "",
    ) -> None:
        """Initialize ResolutionEngine.

        Args:
            provenance_tracker: Optional ProvenanceTracker instance.
                If None, the global singleton is used.
            timestamp_field: Default field name used for temporal
                ordering in the most-recent strategy. Defaults to
                ``data_timestamp``.
        """
        self._provenance = provenance_tracker or get_provenance_tracker()
        self._timestamp_field = (
            timestamp_field if timestamp_field else self.DEFAULT_TIMESTAMP_FIELD
        )
        self._total_resolutions: int = 0
        self._total_golden_records: int = 0
        logger.info(
            "ResolutionEngine initialized (timestamp_field=%s)",
            self._timestamp_field,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_resolutions(self) -> int:
        """Return the total number of resolutions performed."""
        return self._total_resolutions

    @property
    def total_golden_records(self) -> int:
        """Return the total number of golden records assembled."""
        return self._total_golden_records

    # ------------------------------------------------------------------
    # Primary dispatch: resolve_discrepancy
    # ------------------------------------------------------------------

    def resolve_discrepancy(
        self,
        discrepancy: Discrepancy,
        strategy: ResolutionStrategy,
        source_credibilities: Dict[str, SourceCredibility],
        source_records: Dict[str, Dict[str, Any]],
        manual_value: Any = None,
        custom_resolver_fn: Optional[Callable[..., Any]] = None,
        timestamp_field: Optional[str] = None,
    ) -> ResolutionDecision:
        """Resolve a single discrepancy using the specified strategy.

        Dispatches to the appropriate strategy-specific resolution method.
        If strategy is AUTO, auto-selects the best strategy based on
        discrepancy severity and field type.

        Args:
            discrepancy: The discrepancy to resolve.
            strategy: Resolution strategy to apply.
            source_credibilities: Credibility profiles keyed by source_id.
            source_records: Full records keyed by source_id.
            manual_value: Value to use for MANUAL_REVIEW strategy.
            custom_resolver_fn: Callable for CUSTOM strategy.
            timestamp_field: Override for the timestamp field name used
                in MOST_RECENT strategy.

        Returns:
            ResolutionDecision with the resolved value and provenance.

        Raises:
            ValueError: If strategy is CUSTOM but no resolver function
                is provided.
        """
        start_ns = time.perf_counter_ns()
        ts_field = timestamp_field or self._timestamp_field

        logger.info(
            "Resolving discrepancy %s (field=%s, strategy=%s, severity=%s)",
            discrepancy.discrepancy_id,
            discrepancy.field_name,
            strategy.value,
            discrepancy.severity.value
            if isinstance(discrepancy.severity, DiscrepancySeverity)
            else discrepancy.severity,
        )

        # Auto-select strategy if requested
        effective_strategy = strategy
        if strategy == ResolutionStrategy.AUTO:
            effective_strategy = self.auto_select_strategy(
                discrepancy, source_credibilities
            )
            logger.info(
                "Auto-selected strategy %s for discrepancy %s",
                effective_strategy.value,
                discrepancy.discrepancy_id,
            )

        # Dispatch to strategy-specific method
        try:
            decision = self._dispatch_strategy(
                discrepancy=discrepancy,
                strategy=effective_strategy,
                source_credibilities=source_credibilities,
                source_records=source_records,
                manual_value=manual_value,
                custom_resolver_fn=custom_resolver_fn,
                timestamp_field=ts_field,
            )
        except Exception as exc:
            logger.error(
                "Resolution failed for discrepancy %s: %s",
                discrepancy.discrepancy_id,
                str(exc),
                exc_info=True,
            )
            inc_errors("resolution")
            decision = self._build_failed_decision(
                discrepancy, effective_strategy, str(exc)
            )

        # Finalize timing and provenance
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        decision.processing_time_ms = round(elapsed_ms, 3)
        decision.provenance_hash = self._compute_provenance(
            operation="resolve_discrepancy",
            input_data={
                "discrepancy_id": discrepancy.discrepancy_id,
                "strategy": effective_strategy.value,
                "source_values": discrepancy.source_values,
            },
            output_data={
                "decision_id": decision.decision_id,
                "resolved_value": decision.resolved_value,
                "confidence": decision.confidence,
                "status": decision.status,
            },
        )

        # Metrics
        self._total_resolutions += 1
        inc_resolutions(effective_strategy.value)
        observe_confidence(decision.confidence)
        observe_duration(elapsed_ms / 1000.0)

        logger.info(
            "Resolution complete: id=%s status=%s confidence=%.3f "
            "strategy=%s time_ms=%.1f",
            decision.decision_id,
            decision.status,
            decision.confidence,
            effective_strategy.value,
            elapsed_ms,
        )
        return decision

    # ------------------------------------------------------------------
    # Strategy dispatch
    # ------------------------------------------------------------------

    def _dispatch_strategy(
        self,
        discrepancy: Discrepancy,
        strategy: ResolutionStrategy,
        source_credibilities: Dict[str, SourceCredibility],
        source_records: Dict[str, Dict[str, Any]],
        manual_value: Any,
        custom_resolver_fn: Optional[Callable[..., Any]],
        timestamp_field: str,
    ) -> ResolutionDecision:
        """Dispatch to the correct strategy-specific resolver.

        Args:
            discrepancy: The discrepancy to resolve.
            strategy: Effective strategy to apply.
            source_credibilities: Credibility profiles by source_id.
            source_records: Full records by source_id.
            manual_value: Manual override value (for MANUAL_REVIEW).
            custom_resolver_fn: User function (for CUSTOM).
            timestamp_field: Name of the timestamp field.

        Returns:
            ResolutionDecision from the selected strategy.

        Raises:
            ValueError: If CUSTOM strategy lacks a resolver function.
        """
        if strategy == ResolutionStrategy.PRIORITY_WINS:
            return self.resolve_priority_wins(discrepancy, source_credibilities)

        if strategy == ResolutionStrategy.MOST_RECENT:
            return self.resolve_most_recent(
                discrepancy, source_records, timestamp_field
            )

        if strategy == ResolutionStrategy.WEIGHTED_AVERAGE:
            return self.resolve_weighted_average(
                discrepancy, source_credibilities, source_records
            )

        if strategy == ResolutionStrategy.MOST_COMPLETE:
            return self.resolve_most_complete(discrepancy, source_records)

        if strategy == ResolutionStrategy.CONSENSUS:
            return self.resolve_consensus(discrepancy, source_records)

        if strategy == ResolutionStrategy.MANUAL_REVIEW:
            return self.resolve_manual_review(discrepancy, manual_value)

        if strategy == ResolutionStrategy.CUSTOM:
            if custom_resolver_fn is None:
                raise ValueError(
                    "CUSTOM strategy requires a custom_resolver_fn argument"
                )
            return self.resolve_custom(discrepancy, custom_resolver_fn)

        # Fallback: unknown strategy
        logger.warning("Unknown strategy %s, falling back to priority_wins", strategy)
        return self.resolve_priority_wins(discrepancy, source_credibilities)

    # ------------------------------------------------------------------
    # Strategy: priority_wins
    # ------------------------------------------------------------------

    def resolve_priority_wins(
        self,
        discrepancy: Discrepancy,
        source_credibilities: Dict[str, SourceCredibility],
    ) -> ResolutionDecision:
        """Resolve by selecting the value from the highest-priority source.

        Ranks sources by ``credibility_score * (1 / priority)`` so that
        higher credibility and lower priority number both increase the
        effective score. Picks the source with the highest composite score.

        Args:
            discrepancy: The discrepancy to resolve.
            source_credibilities: Credibility profiles keyed by source_id.

        Returns:
            ResolutionDecision with the winning source's value.
        """
        logger.debug(
            "resolve_priority_wins: field=%s sources=%s",
            discrepancy.field_name,
            list(discrepancy.source_values.keys()),
        )

        # Score each source: credibility * inverse priority
        scored_sources = self._score_sources_by_priority(
            discrepancy.source_values, source_credibilities
        )

        if not scored_sources:
            return self._build_no_sources_decision(
                discrepancy, ResolutionStrategy.PRIORITY_WINS
            )

        # Pick the top-scoring source
        winner_id, winner_score = scored_sources[0]
        winner_cred = source_credibilities.get(winner_id)
        winner_name = winner_cred.source_name if winner_cred else winner_id
        resolved_value = discrepancy.source_values[winner_id]

        confidence = winner_cred.credibility_score if winner_cred else 0.5
        justification = (
            f"Resolved using highest-priority source: {winner_name} "
            f"(credibility={confidence:.3f}, composite_score={winner_score:.4f})"
        )

        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=ResolutionStrategy.PRIORITY_WINS.value,
            status=ResolutionStatus.RESOLVED.value,
            winning_source_id=winner_id,
            winning_source_name=winner_name,
            resolved_value=resolved_value,
            original_values=dict(discrepancy.source_values),
            confidence=confidence,
            justification=justification,
            is_auto=True,
        )

    def _score_sources_by_priority(
        self,
        source_values: Dict[str, Any],
        source_credibilities: Dict[str, SourceCredibility],
    ) -> List[Tuple[str, float]]:
        """Score sources by credibility * inverse priority.

        Sources not present in credibilities receive a default score of
        0.5 with priority 100 (lowest).

        Args:
            source_values: Mapping of source_id to value.
            source_credibilities: Credibility profiles by source_id.

        Returns:
            List of (source_id, composite_score) sorted descending.
        """
        scored: List[Tuple[str, float]] = []
        for src_id in source_values:
            cred = source_credibilities.get(src_id)
            if cred:
                # Inverse priority: priority 1 -> 1.0, priority 2 -> 0.5, etc.
                inv_priority = 1.0 / max(cred.priority, 1)
                composite = cred.credibility_score * inv_priority
            else:
                composite = 0.5 * (1.0 / 100.0)
            scored.append((src_id, composite))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Strategy: most_recent
    # ------------------------------------------------------------------

    def resolve_most_recent(
        self,
        discrepancy: Discrepancy,
        source_records: Dict[str, Dict[str, Any]],
        timestamp_field: Optional[str] = None,
    ) -> ResolutionDecision:
        """Resolve by selecting the value from the most recently updated source.

        Looks for a timestamp field in each source record and picks the
        source with the most recent (latest) timestamp. Timestamps are
        parsed as ISO-format strings.

        Args:
            discrepancy: The discrepancy to resolve.
            source_records: Full records keyed by source_id.
            timestamp_field: Name of the field containing timestamps.
                Defaults to the engine's configured timestamp field.

        Returns:
            ResolutionDecision. Confidence is 0.7 because temporal
            recency does not guarantee accuracy.
        """
        ts_field = timestamp_field or self._timestamp_field
        logger.debug(
            "resolve_most_recent: field=%s ts_field=%s",
            discrepancy.field_name,
            ts_field,
        )

        # Build list of (source_id, timestamp) where timestamp is valid
        timed_sources = self._extract_timestamps(
            discrepancy.source_values, source_records, ts_field
        )

        if not timed_sources:
            logger.warning(
                "No valid timestamps found for field %s; falling back to "
                "first available source",
                discrepancy.field_name,
            )
            return self._build_fallback_decision(
                discrepancy, ResolutionStrategy.MOST_RECENT
            )

        # Sort descending by timestamp (most recent first)
        timed_sources.sort(key=lambda x: x[1], reverse=True)
        winner_id = timed_sources[0][0]
        winner_ts = timed_sources[0][1]
        resolved_value = discrepancy.source_values.get(winner_id)

        # Confidence is fixed at 0.7: recency does not guarantee accuracy
        confidence = 0.7
        justification = (
            f"Resolved using most recent source: {winner_id} "
            f"(timestamp={winner_ts.isoformat()}, "
            f"confidence={confidence:.1f} - temporal recency does not "
            f"guarantee accuracy)"
        )

        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=ResolutionStrategy.MOST_RECENT.value,
            status=ResolutionStatus.RESOLVED.value,
            winning_source_id=winner_id,
            winning_source_name=winner_id,
            resolved_value=resolved_value,
            original_values=dict(discrepancy.source_values),
            confidence=confidence,
            justification=justification,
            is_auto=True,
        )

    def _extract_timestamps(
        self,
        source_values: Dict[str, Any],
        source_records: Dict[str, Dict[str, Any]],
        ts_field: str,
    ) -> List[Tuple[str, datetime]]:
        """Extract and parse timestamps from source records.

        Supports ISO-format strings and datetime objects. Skips sources
        where the timestamp is missing or unparseable.

        Args:
            source_values: Mapping of source_id to value (used for
                filtering to only relevant sources).
            source_records: Full records keyed by source_id.
            ts_field: Name of the timestamp field.

        Returns:
            List of (source_id, datetime) tuples with valid timestamps.
        """
        results: List[Tuple[str, datetime]] = []
        for src_id in source_values:
            record = source_records.get(src_id, {})
            ts_raw = record.get(ts_field)
            if ts_raw is None:
                continue
            ts_parsed = self._parse_timestamp(ts_raw)
            if ts_parsed is not None:
                results.append((src_id, ts_parsed))
        return results

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        """Parse a value into a datetime object.

        Supports ISO-format strings and datetime objects.

        Args:
            value: Value to parse.

        Returns:
            Parsed datetime or None if unparseable.
        """
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            # Try fromisoformat as a last resort
            try:
                return datetime.fromisoformat(value)
            except (ValueError, TypeError):
                pass
        return None

    # ------------------------------------------------------------------
    # Strategy: weighted_average
    # ------------------------------------------------------------------

    def resolve_weighted_average(
        self,
        discrepancy: Discrepancy,
        source_credibilities: Dict[str, SourceCredibility],
        source_records: Dict[str, Dict[str, Any]],
    ) -> ResolutionDecision:
        """Resolve numeric fields using credibility-weighted average.

        For numeric fields: ``resolved = sum(value_i * cred_i) / sum(cred_i)``.
        Confidence is the maximum credibility score among the sources.
        For non-numeric fields, falls back to priority_wins strategy.

        Args:
            discrepancy: The discrepancy to resolve.
            source_credibilities: Credibility profiles keyed by source_id.
            source_records: Full records keyed by source_id.

        Returns:
            ResolutionDecision with the weighted average value or
            a priority_wins fallback for non-numeric fields.
        """
        logger.debug(
            "resolve_weighted_average: field=%s type=%s",
            discrepancy.field_name,
            discrepancy.field_type,
        )

        # Attempt to parse all source values as numeric
        numeric_pairs = self._extract_numeric_pairs(
            discrepancy.source_values, source_credibilities
        )

        if not numeric_pairs:
            logger.info(
                "No numeric values for weighted_average on field %s; "
                "falling back to priority_wins",
                discrepancy.field_name,
            )
            decision = self.resolve_priority_wins(discrepancy, source_credibilities)
            decision.justification = (
                f"Weighted average not applicable for non-numeric field "
                f"'{discrepancy.field_name}'; fell back to priority_wins. "
                f"{decision.justification}"
            )
            return decision

        # Compute weighted average
        resolved_value, max_cred, contributing = self._compute_weighted_avg(
            numeric_pairs
        )

        confidence = max_cred
        justification = (
            f"Resolved using credibility-weighted average of "
            f"{len(contributing)} sources: "
            f"resolved_value={resolved_value:.6f}, "
            f"confidence={confidence:.3f} (max credibility)"
        )

        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=ResolutionStrategy.WEIGHTED_AVERAGE.value,
            status=ResolutionStatus.RESOLVED.value,
            winning_source_id="weighted_average",
            winning_source_name="weighted_average",
            resolved_value=resolved_value,
            original_values=dict(discrepancy.source_values),
            confidence=confidence,
            justification=justification,
            is_auto=True,
            metadata={"contributing_sources": contributing},
        )

    def _extract_numeric_pairs(
        self,
        source_values: Dict[str, Any],
        source_credibilities: Dict[str, SourceCredibility],
    ) -> List[Tuple[str, float, float]]:
        """Extract (source_id, numeric_value, credibility) triples.

        Skips sources whose values cannot be parsed as finite numbers.

        Args:
            source_values: Mapping of source_id to value.
            source_credibilities: Credibility profiles by source_id.

        Returns:
            List of (source_id, numeric_value, credibility_score) tuples.
        """
        pairs: List[Tuple[str, float, float]] = []
        for src_id, raw_val in source_values.items():
            numeric_val = _try_parse_numeric(raw_val)
            if numeric_val is None:
                continue
            cred = source_credibilities.get(src_id)
            cred_score = cred.credibility_score if cred else 0.5
            pairs.append((src_id, numeric_val, cred_score))
        return pairs

    @staticmethod
    def _compute_weighted_avg(
        pairs: List[Tuple[str, float, float]],
    ) -> Tuple[float, float, List[str]]:
        """Compute credibility-weighted average from numeric pairs.

        Formula: ``sum(value_i * cred_i) / sum(cred_i)``

        Args:
            pairs: List of (source_id, value, credibility).

        Returns:
            Tuple of (weighted_average, max_credibility, source_ids).
        """
        total_weighted = 0.0
        total_cred = 0.0
        max_cred = 0.0
        contributing: List[str] = []

        for src_id, val, cred in pairs:
            # Ensure non-zero weight
            weight = max(cred, 0.01)
            total_weighted += val * weight
            total_cred += weight
            max_cred = max(max_cred, cred)
            contributing.append(src_id)

        if total_cred == 0.0:
            return 0.0, 0.0, contributing

        weighted_avg = total_weighted / total_cred
        return round(weighted_avg, 10), max_cred, contributing

    # ------------------------------------------------------------------
    # Strategy: most_complete
    # ------------------------------------------------------------------

    def resolve_most_complete(
        self,
        discrepancy: Discrepancy,
        source_records: Dict[str, Dict[str, Any]],
    ) -> ResolutionDecision:
        """Resolve by selecting the value from the source with fewest nulls.

        Computes a completeness ratio for each source record (non-empty
        fields / total fields) and picks the source with the highest
        completeness. Confidence equals the completeness ratio of the
        winning source.

        Args:
            discrepancy: The discrepancy to resolve.
            source_records: Full records keyed by source_id.

        Returns:
            ResolutionDecision with the most complete source's value.
        """
        logger.debug(
            "resolve_most_complete: field=%s sources=%d",
            discrepancy.field_name,
            len(discrepancy.source_values),
        )

        # Compute completeness for each source
        completeness_scores = self._compute_completeness_scores(
            discrepancy.source_values, source_records
        )

        if not completeness_scores:
            return self._build_no_sources_decision(
                discrepancy, ResolutionStrategy.MOST_COMPLETE
            )

        # Sort descending by completeness
        completeness_scores.sort(key=lambda x: x[1], reverse=True)
        winner_id = completeness_scores[0][0]
        winner_completeness = completeness_scores[0][1]
        resolved_value = discrepancy.source_values.get(winner_id)

        confidence = round(winner_completeness, 4)
        justification = (
            f"Resolved using most complete source: {winner_id} "
            f"(completeness={winner_completeness:.2%}, "
            f"confidence={confidence:.3f})"
        )

        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=ResolutionStrategy.MOST_COMPLETE.value,
            status=ResolutionStatus.RESOLVED.value,
            winning_source_id=winner_id,
            winning_source_name=winner_id,
            resolved_value=resolved_value,
            original_values=dict(discrepancy.source_values),
            confidence=confidence,
            justification=justification,
            is_auto=True,
            metadata={
                "completeness_scores": {
                    s: round(c, 4) for s, c in completeness_scores
                }
            },
        )

    def _compute_completeness_scores(
        self,
        source_values: Dict[str, Any],
        source_records: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        """Compute completeness ratio for each source with a value.

        Args:
            source_values: Mapping of source_id to discrepant value.
            source_records: Full records keyed by source_id.

        Returns:
            List of (source_id, completeness_ratio) tuples.
        """
        scores: List[Tuple[str, float]] = []
        for src_id in source_values:
            record = source_records.get(src_id, {})
            completeness = _compute_completeness(record)
            scores.append((src_id, completeness))
        return scores

    # ------------------------------------------------------------------
    # Strategy: consensus
    # ------------------------------------------------------------------

    def resolve_consensus(
        self,
        discrepancy: Discrepancy,
        source_records: Dict[str, Dict[str, Any]],
    ) -> ResolutionDecision:
        """Resolve using majority vote across sources.

        For 3+ sources: the most common value wins. Confidence is the
        agreement ratio (n_agreeing / n_total).

        For 2 sources: compares values; if identical returns RESOLVED,
        otherwise flags for manual review since no majority is possible.

        For 1 source: uses the single value with confidence 0.5.

        Args:
            discrepancy: The discrepancy to resolve.
            source_records: Full records keyed by source_id.

        Returns:
            ResolutionDecision with the consensus value or a
            pending-review decision.
        """
        source_values = discrepancy.source_values
        n_sources = len(source_values)
        logger.debug(
            "resolve_consensus: field=%s n_sources=%d",
            discrepancy.field_name,
            n_sources,
        )

        if n_sources == 0:
            return self._build_no_sources_decision(
                discrepancy, ResolutionStrategy.CONSENSUS
            )

        if n_sources == 1:
            return self._resolve_single_source(discrepancy)

        if n_sources == 2:
            return self._resolve_two_source_consensus(discrepancy)

        return self._resolve_majority_vote(discrepancy)

    def _resolve_single_source(
        self, discrepancy: Discrepancy
    ) -> ResolutionDecision:
        """Handle consensus resolution with only one source.

        Args:
            discrepancy: The discrepancy with a single source.

        Returns:
            ResolutionDecision with the single value and low confidence.
        """
        src_id = next(iter(discrepancy.source_values))
        value = discrepancy.source_values[src_id]
        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=ResolutionStrategy.CONSENSUS.value,
            status=ResolutionStatus.RESOLVED.value,
            winning_source_id=src_id,
            winning_source_name=src_id,
            resolved_value=value,
            original_values=dict(discrepancy.source_values),
            confidence=0.5,
            justification=(
                f"Single source consensus: {src_id} (confidence=0.5 - "
                f"no other sources to corroborate)"
            ),
            is_auto=True,
        )

    def _resolve_two_source_consensus(
        self, discrepancy: Discrepancy
    ) -> ResolutionDecision:
        """Handle consensus resolution with exactly two sources.

        If both values agree, returns RESOLVED. Otherwise flags for
        manual review since no majority is possible with two sources.

        Args:
            discrepancy: The discrepancy with two source values.

        Returns:
            ResolutionDecision (resolved or pending_review).
        """
        sources = list(discrepancy.source_values.items())
        src_a_id, val_a = sources[0]
        src_b_id, val_b = sources[1]

        if self._values_agree(val_a, val_b):
            return ResolutionDecision(
                discrepancy_id=discrepancy.discrepancy_id,
                entity_id=discrepancy.entity_id,
                field_name=discrepancy.field_name,
                strategy=ResolutionStrategy.CONSENSUS.value,
                status=ResolutionStatus.RESOLVED.value,
                winning_source_id=src_a_id,
                winning_source_name=src_a_id,
                resolved_value=val_a,
                original_values=dict(discrepancy.source_values),
                confidence=1.0,
                justification=(
                    f"Both sources agree: {src_a_id} and {src_b_id} "
                    f"have matching values (confidence=1.0)"
                ),
                is_auto=True,
            )

        # No consensus possible with two disagreeing sources
        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=ResolutionStrategy.CONSENSUS.value,
            status=ResolutionStatus.PENDING_REVIEW.value,
            winning_source_id="",
            winning_source_name="",
            resolved_value=None,
            original_values=dict(discrepancy.source_values),
            confidence=0.0,
            justification=(
                f"No consensus: two sources ({src_a_id}, {src_b_id}) "
                f"disagree and no majority is possible. Flagged for "
                f"manual review."
            ),
            is_auto=False,
        )

    def _resolve_majority_vote(
        self, discrepancy: Discrepancy
    ) -> ResolutionDecision:
        """Perform majority vote across 3+ sources.

        Uses collections.Counter to find the most common value.
        Confidence = n_agreeing / n_total.

        Args:
            discrepancy: The discrepancy with 3+ source values.

        Returns:
            ResolutionDecision with the majority value.
        """
        # Normalize values for comparison
        normalized: Dict[str, str] = {}
        for src_id, val in discrepancy.source_values.items():
            normalized[src_id] = self._normalize_for_comparison(val)

        # Count occurrences of each normalized value
        value_counter: Counter = Counter(normalized.values())
        most_common_normalized, n_agreeing = value_counter.most_common(1)[0]
        n_total = len(discrepancy.source_values)
        agreement_ratio = n_agreeing / n_total

        # Find the original value and a winning source
        winning_src_id = ""
        resolved_value = None
        for src_id, norm_val in normalized.items():
            if norm_val == most_common_normalized:
                winning_src_id = src_id
                resolved_value = discrepancy.source_values[src_id]
                break

        # Collect agreeing sources
        agreeing_sources = [
            s for s, v in normalized.items() if v == most_common_normalized
        ]

        confidence = round(agreement_ratio, 4)
        justification = (
            f"Consensus by majority vote: {n_agreeing}/{n_total} sources "
            f"agree (sources: {', '.join(agreeing_sources)}). "
            f"Agreement ratio={agreement_ratio:.2%}, confidence={confidence:.3f}"
        )

        # If less than half agree, mark as pending review
        status = ResolutionStatus.RESOLVED.value
        is_auto = True
        if agreement_ratio < 0.5:
            status = ResolutionStatus.PENDING_REVIEW.value
            is_auto = False
            justification += (
                ". WARNING: No true majority (<50% agreement); "
                "flagged for manual review."
            )

        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=ResolutionStrategy.CONSENSUS.value,
            status=status,
            winning_source_id=winning_src_id,
            winning_source_name=winning_src_id,
            resolved_value=resolved_value,
            original_values=dict(discrepancy.source_values),
            confidence=confidence,
            justification=justification,
            is_auto=is_auto,
            metadata={
                "agreeing_sources": agreeing_sources,
                "n_agreeing": n_agreeing,
                "n_total": n_total,
                "agreement_ratio": agreement_ratio,
            },
        )

    @staticmethod
    def _values_agree(val_a: Any, val_b: Any) -> bool:
        """Compare two values for agreement.

        Numeric values are compared with a tolerance of 1e-9.
        String values are compared case-insensitively after stripping.

        Args:
            val_a: First value.
            val_b: Second value.

        Returns:
            True if the values are considered equal.
        """
        # Both None
        if val_a is None and val_b is None:
            return True
        if val_a is None or val_b is None:
            return False

        # Numeric comparison with tolerance
        num_a = _try_parse_numeric(val_a)
        num_b = _try_parse_numeric(val_b)
        if num_a is not None and num_b is not None:
            return abs(num_a - num_b) < 1e-9

        # String comparison (case-insensitive, stripped)
        str_a = str(val_a).strip().lower()
        str_b = str(val_b).strip().lower()
        return str_a == str_b

    @staticmethod
    def _normalize_for_comparison(value: Any) -> str:
        """Normalize a value to a canonical string for comparison.

        Numbers are rounded to 6 decimal places. Strings are lowercased
        and stripped.

        Args:
            value: Value to normalize.

        Returns:
            Canonical string representation.
        """
        if value is None:
            return "__none__"
        num_val = _try_parse_numeric(value)
        if num_val is not None:
            return f"__num_{round(num_val, 6)}"
        return str(value).strip().lower()

    # ------------------------------------------------------------------
    # Strategy: manual_review
    # ------------------------------------------------------------------

    def resolve_manual_review(
        self,
        discrepancy: Discrepancy,
        manual_value: Any = None,
    ) -> ResolutionDecision:
        """Resolve via manual review or apply a manual override value.

        If ``manual_value`` is provided, the discrepancy is resolved with
        confidence 1.0 as a MANUAL_OVERRIDE. If not provided, the
        discrepancy is marked as PENDING_REVIEW with confidence 0.0.

        Args:
            discrepancy: The discrepancy to resolve.
            manual_value: Optional value provided by a human reviewer.

        Returns:
            ResolutionDecision (manual override or pending review).
        """
        logger.debug(
            "resolve_manual_review: field=%s manual_value=%s",
            discrepancy.field_name,
            "provided" if manual_value is not None else "not provided",
        )

        if manual_value is not None:
            return ResolutionDecision(
                discrepancy_id=discrepancy.discrepancy_id,
                entity_id=discrepancy.entity_id,
                field_name=discrepancy.field_name,
                strategy=ResolutionStrategy.MANUAL_REVIEW.value,
                status=ResolutionStatus.MANUAL_OVERRIDE.value,
                winning_source_id="manual",
                winning_source_name="Manual Override",
                resolved_value=manual_value,
                original_values=dict(discrepancy.source_values),
                confidence=1.0,
                justification=(
                    f"Manually resolved by human reviewer with value: "
                    f"{manual_value}"
                ),
                is_auto=False,
            )

        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=ResolutionStrategy.MANUAL_REVIEW.value,
            status=ResolutionStatus.PENDING_REVIEW.value,
            winning_source_id="",
            winning_source_name="",
            resolved_value=None,
            original_values=dict(discrepancy.source_values),
            confidence=0.0,
            justification=(
                f"Flagged for manual review. No manual value provided. "
                f"Source values: {discrepancy.source_values}"
            ),
            is_auto=False,
        )

    # ------------------------------------------------------------------
    # Strategy: custom
    # ------------------------------------------------------------------

    def resolve_custom(
        self,
        discrepancy: Discrepancy,
        custom_resolver_fn: Callable[..., Any],
    ) -> ResolutionDecision:
        """Resolve using a user-provided custom resolver function.

        The custom function receives the discrepancy and should return a
        dictionary with at least ``resolved_value``. Optional keys:
        ``confidence``, ``winning_source_id``, ``justification``.

        Args:
            discrepancy: The discrepancy to resolve.
            custom_resolver_fn: Callable that takes a Discrepancy and
                returns a dict with resolution details.

        Returns:
            ResolutionDecision wrapping the custom function's result.

        Raises:
            TypeError: If the function does not return a dict.
            RuntimeError: If the custom function raises an exception.
        """
        logger.debug(
            "resolve_custom: field=%s fn=%s",
            discrepancy.field_name,
            getattr(custom_resolver_fn, "__name__", "anonymous"),
        )

        try:
            result = custom_resolver_fn(discrepancy)
        except Exception as exc:
            logger.error(
                "Custom resolver function failed: %s", str(exc), exc_info=True
            )
            raise RuntimeError(
                f"Custom resolver function raised: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise TypeError(
                f"Custom resolver must return a dict, got {type(result).__name__}"
            )

        resolved_value = result.get("resolved_value")
        confidence = float(result.get("confidence", 0.5))
        winning_source_id = str(result.get("winning_source_id", "custom"))
        justification = str(
            result.get(
                "justification",
                f"Resolved via custom function: "
                f"{getattr(custom_resolver_fn, '__name__', 'anonymous')}",
            )
        )

        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=ResolutionStrategy.CUSTOM.value,
            status=ResolutionStatus.RESOLVED.value,
            winning_source_id=winning_source_id,
            winning_source_name=winning_source_id,
            resolved_value=resolved_value,
            original_values=dict(discrepancy.source_values),
            confidence=confidence,
            justification=justification,
            is_auto=True,
            metadata={"custom_result": result},
        )

    # ------------------------------------------------------------------
    # Batch resolution
    # ------------------------------------------------------------------

    def resolve_batch(
        self,
        discrepancies: List[Discrepancy],
        strategy: ResolutionStrategy,
        source_credibilities: Dict[str, SourceCredibility],
        source_records: Dict[str, Dict[str, Any]],
        manual_value: Any = None,
        custom_resolver_fn: Optional[Callable[..., Any]] = None,
        timestamp_field: Optional[str] = None,
    ) -> List[ResolutionDecision]:
        """Resolve a batch of discrepancies using the same strategy.

        Iterates over all discrepancies and applies the specified
        strategy to each one. Tracks aggregate statistics and logs
        a batch summary.

        Args:
            discrepancies: List of discrepancies to resolve.
            strategy: Resolution strategy to apply to all.
            source_credibilities: Credibility profiles keyed by source_id.
            source_records: Full records keyed by source_id.
            manual_value: Value for MANUAL_REVIEW strategy.
            custom_resolver_fn: Callable for CUSTOM strategy.
            timestamp_field: Override for the timestamp field name.

        Returns:
            List of ResolutionDecision objects, one per discrepancy.
        """
        start_ns = time.perf_counter_ns()
        n_total = len(discrepancies)
        logger.info(
            "resolve_batch: resolving %d discrepancies with strategy=%s",
            n_total,
            strategy.value,
        )

        decisions: List[ResolutionDecision] = []
        n_resolved = 0
        n_pending = 0
        n_failed = 0

        for idx, disc in enumerate(discrepancies):
            decision = self.resolve_discrepancy(
                discrepancy=disc,
                strategy=strategy,
                source_credibilities=source_credibilities,
                source_records=source_records,
                manual_value=manual_value,
                custom_resolver_fn=custom_resolver_fn,
                timestamp_field=timestamp_field,
            )
            decisions.append(decision)

            if decision.status == ResolutionStatus.RESOLVED.value:
                n_resolved += 1
            elif decision.status == ResolutionStatus.PENDING_REVIEW.value:
                n_pending += 1
            elif decision.status == ResolutionStatus.FAILED.value:
                n_failed += 1

            if (idx + 1) % 100 == 0:
                logger.info(
                    "Batch progress: %d/%d resolved", idx + 1, n_total
                )

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        logger.info(
            "resolve_batch complete: total=%d resolved=%d pending=%d "
            "failed=%d time_ms=%.1f",
            n_total,
            n_resolved,
            n_pending,
            n_failed,
            elapsed_ms,
        )
        return decisions

    # ------------------------------------------------------------------
    # Golden record assembly
    # ------------------------------------------------------------------

    def assemble_golden_record(
        self,
        entity_id: str,
        period: str,
        source_records: Dict[str, Dict[str, Any]],
        resolutions: List[ResolutionDecision],
        source_credibilities: Dict[str, SourceCredibility],
    ) -> GoldenRecord:
        """Assemble a golden record from source records and resolution decisions.

        For each field:
          1. If a resolution decision exists, use the resolved value.
          2. Otherwise, pick the value from the highest-credibility source.

        Computes per-field confidence and total_confidence (mean of all
        field confidences).

        Args:
            entity_id: Entity identifier.
            period: Reporting period (e.g. "2025-Q4").
            source_records: Full records keyed by source_id.
            resolutions: List of resolution decisions for this entity.
            source_credibilities: Credibility profiles keyed by source_id.

        Returns:
            GoldenRecord with merged fields, field sources, confidences,
            and provenance hash.
        """
        start_ns = time.perf_counter_ns()
        logger.info(
            "Assembling golden record for entity=%s period=%s "
            "sources=%d resolutions=%d",
            entity_id,
            period,
            len(source_records),
            len(resolutions),
        )

        # Build resolution index: field_name -> ResolutionDecision
        resolution_index = self._build_resolution_index(resolutions)

        # Collect all unique field names across all sources
        all_fields = self._collect_all_fields(source_records)

        # Rank sources by credibility for fallback ordering
        ranked_sources = self._rank_sources_by_credibility(
            source_credibilities, list(source_records.keys())
        )

        # Build the golden record fields
        fields: Dict[str, Any] = {}
        field_sources: Dict[str, str] = {}
        field_confidences: Dict[str, float] = {}
        discrepancy_count = 0

        for field_name in sorted(all_fields):
            value, source_id, confidence, was_resolved = self._resolve_field(
                field_name=field_name,
                resolution_index=resolution_index,
                source_records=source_records,
                ranked_sources=ranked_sources,
                source_credibilities=source_credibilities,
            )
            fields[field_name] = value
            field_sources[field_name] = source_id
            field_confidences[field_name] = confidence
            if was_resolved:
                discrepancy_count += 1

        # Compute total confidence
        total_confidence = self._compute_mean_confidence(field_confidences)

        # Build golden record
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        golden = GoldenRecord(
            entity_id=entity_id,
            period=period,
            fields=fields,
            field_sources=field_sources,
            field_confidences=field_confidences,
            total_confidence=total_confidence,
            source_count=len(source_records),
            discrepancy_count=discrepancy_count,
            resolution_count=len(resolutions),
            processing_time_ms=round(elapsed_ms, 3),
        )

        # Provenance
        golden.provenance_hash = self._compute_provenance(
            operation="assemble_golden_record",
            input_data={
                "entity_id": entity_id,
                "period": period,
                "source_ids": sorted(source_records.keys()),
                "resolution_ids": [r.decision_id for r in resolutions],
            },
            output_data={
                "record_id": golden.record_id,
                "total_confidence": total_confidence,
                "field_count": len(fields),
            },
        )

        # Metrics
        self._total_golden_records += 1
        inc_golden_records("created")
        observe_duration(elapsed_ms / 1000.0)

        logger.info(
            "Golden record assembled: id=%s entity=%s fields=%d "
            "confidence=%.3f time_ms=%.1f",
            golden.record_id,
            entity_id,
            len(fields),
            total_confidence,
            elapsed_ms,
        )
        return golden

    def _build_resolution_index(
        self, resolutions: List[ResolutionDecision]
    ) -> Dict[str, ResolutionDecision]:
        """Build a lookup of field_name to the latest resolution decision.

        If multiple decisions exist for the same field, the last one wins.

        Args:
            resolutions: List of resolution decisions.

        Returns:
            Dictionary mapping field_name to ResolutionDecision.
        """
        index: Dict[str, ResolutionDecision] = {}
        for decision in resolutions:
            if decision.field_name:
                index[decision.field_name] = decision
        return index

    @staticmethod
    def _collect_all_fields(
        source_records: Dict[str, Dict[str, Any]]
    ) -> set:
        """Collect all unique field names across source records.

        Args:
            source_records: Full records keyed by source_id.

        Returns:
            Set of all field names.
        """
        all_fields: set = set()
        for record in source_records.values():
            all_fields.update(record.keys())
        return all_fields

    def _rank_sources_by_credibility(
        self,
        source_credibilities: Dict[str, SourceCredibility],
        source_ids: List[str],
    ) -> List[str]:
        """Rank source IDs by descending credibility score.

        Sources not in credibilities are placed last with score 0.0.

        Args:
            source_credibilities: Credibility profiles by source_id.
            source_ids: List of source IDs to rank.

        Returns:
            List of source IDs sorted by descending credibility.
        """
        scored = []
        for src_id in source_ids:
            cred = source_credibilities.get(src_id)
            score = cred.credibility_score if cred else 0.0
            scored.append((src_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored]

    def _resolve_field(
        self,
        field_name: str,
        resolution_index: Dict[str, ResolutionDecision],
        source_records: Dict[str, Dict[str, Any]],
        ranked_sources: List[str],
        source_credibilities: Dict[str, SourceCredibility],
    ) -> Tuple[Any, str, float, bool]:
        """Resolve a single field value for the golden record.

        Uses the resolution decision if available, otherwise picks the
        value from the highest-credibility source.

        Args:
            field_name: Name of the field to resolve.
            resolution_index: Lookup of field_name -> ResolutionDecision.
            source_records: Full records keyed by source_id.
            ranked_sources: Source IDs ranked by credibility.
            source_credibilities: Credibility profiles by source_id.

        Returns:
            Tuple of (value, source_id, confidence, was_resolved).
        """
        # Check if there is a resolution decision for this field
        decision = resolution_index.get(field_name)
        if decision and decision.status in (
            ResolutionStatus.RESOLVED.value,
            ResolutionStatus.MANUAL_OVERRIDE.value,
        ):
            source_id = decision.winning_source_id or "resolution"
            return decision.resolved_value, source_id, decision.confidence, True

        # Fallback: pick from highest-credibility source that has the field
        for src_id in ranked_sources:
            record = source_records.get(src_id, {})
            if field_name in record and not _is_empty(record[field_name]):
                cred = source_credibilities.get(src_id)
                confidence = cred.credibility_score if cred else 0.5
                return record[field_name], src_id, confidence, False

        # No source has the field with a non-empty value
        return None, "", 0.0, False

    @staticmethod
    def _compute_mean_confidence(
        field_confidences: Dict[str, float],
    ) -> float:
        """Compute the mean confidence across all fields.

        Args:
            field_confidences: Mapping of field_name to confidence.

        Returns:
            Mean confidence (0.0 if no fields).
        """
        if not field_confidences:
            return 0.0
        values = list(field_confidences.values())
        return round(statistics.mean(values), 4)

    # ------------------------------------------------------------------
    # Batch golden record assembly
    # ------------------------------------------------------------------

    def assemble_golden_records_batch(
        self,
        entities: List[Tuple[str, str]],
        source_records_map: Dict[str, Dict[str, Dict[str, Any]]],
        resolutions_map: Dict[str, List[ResolutionDecision]],
        source_credibilities: Dict[str, SourceCredibility],
    ) -> List[GoldenRecord]:
        """Assemble golden records for multiple entities in batch.

        Args:
            entities: List of (entity_id, period) tuples.
            source_records_map: Mapping of entity_id to source_records
                dict (source_id -> record dict).
            resolutions_map: Mapping of entity_id to list of
                ResolutionDecision objects.
            source_credibilities: Credibility profiles keyed by source_id.

        Returns:
            List of GoldenRecord objects, one per entity.
        """
        start_ns = time.perf_counter_ns()
        n_total = len(entities)
        logger.info(
            "assemble_golden_records_batch: assembling %d golden records",
            n_total,
        )

        golden_records: List[GoldenRecord] = []
        for idx, (entity_id, period) in enumerate(entities):
            src_records = source_records_map.get(entity_id, {})
            resolutions = resolutions_map.get(entity_id, [])

            golden = self.assemble_golden_record(
                entity_id=entity_id,
                period=period,
                source_records=src_records,
                resolutions=resolutions,
                source_credibilities=source_credibilities,
            )
            golden_records.append(golden)

            if (idx + 1) % 50 == 0:
                logger.info(
                    "Batch golden records progress: %d/%d", idx + 1, n_total
                )

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        logger.info(
            "assemble_golden_records_batch complete: total=%d time_ms=%.1f",
            n_total,
            elapsed_ms,
        )
        return golden_records

    # ------------------------------------------------------------------
    # Field lineage
    # ------------------------------------------------------------------

    def get_field_lineage(
        self,
        golden_record: GoldenRecord,
        source_records: Dict[str, Dict[str, Any]],
        resolutions: Optional[List[ResolutionDecision]] = None,
        source_credibilities: Optional[Dict[str, SourceCredibility]] = None,
    ) -> List[FieldLineage]:
        """Retrieve the full lineage for every field in a golden record.

        For each field, returns the source that contributed the value,
        the original value from that source, the resolved value, the
        strategy used (if applicable), and the confidence.

        Args:
            golden_record: The assembled golden record.
            source_records: Full records keyed by source_id.
            resolutions: Optional list of resolution decisions for
                additional context.
            source_credibilities: Optional credibility profiles for
                source name lookup.

        Returns:
            List of FieldLineage objects, one per field.
        """
        logger.debug(
            "get_field_lineage: record=%s fields=%d",
            golden_record.record_id,
            len(golden_record.fields),
        )

        resolution_index: Dict[str, ResolutionDecision] = {}
        if resolutions:
            resolution_index = self._build_resolution_index(resolutions)

        lineage_records: List[FieldLineage] = []

        for field_name in sorted(golden_record.fields.keys()):
            source_id = golden_record.field_sources.get(field_name, "")
            confidence = golden_record.field_confidences.get(field_name, 0.0)
            resolved_value = golden_record.fields.get(field_name)

            # Look up original value from the contributing source
            original_value = None
            if source_id and source_id not in ("resolution", "manual",
                                                "weighted_average", ""):
                src_record = source_records.get(source_id, {})
                original_value = src_record.get(field_name)

            # Look up source name
            source_name = source_id
            if source_credibilities and source_id in source_credibilities:
                source_name = source_credibilities[source_id].source_name

            # Determine strategy and whether it was discrepant
            strategy = ""
            was_discrepant = False
            decision = resolution_index.get(field_name)
            if decision:
                strategy = decision.strategy
                was_discrepant = True
                original_value = original_value or decision.original_values.get(
                    source_id
                )

            lineage_records.append(
                FieldLineage(
                    field_name=field_name,
                    source_id=source_id,
                    source_name=source_name,
                    original_value=original_value,
                    resolved_value=resolved_value,
                    strategy=strategy,
                    confidence=confidence,
                    was_discrepant=was_discrepant,
                )
            )

        logger.info(
            "Field lineage generated: record=%s lineage_count=%d",
            golden_record.record_id,
            len(lineage_records),
        )
        return lineage_records

    # ------------------------------------------------------------------
    # Resolution summary
    # ------------------------------------------------------------------

    def summarize_resolutions(
        self,
        decisions: List[ResolutionDecision],
    ) -> ResolutionSummary:
        """Generate aggregate statistics for a batch of resolution decisions.

        Counts decisions by status, strategy, auto vs manual. Computes
        average, minimum, and maximum confidence. Tracks cumulative
        processing time.

        Args:
            decisions: List of resolution decisions to summarize.

        Returns:
            ResolutionSummary with complete statistics.
        """
        logger.debug(
            "summarize_resolutions: summarizing %d decisions", len(decisions)
        )

        total = len(decisions)
        resolved = 0
        pending = 0
        manual = 0
        failed = 0
        skipped = 0
        auto_count = 0
        manual_review_count = 0
        strategy_counts: Dict[str, int] = {}
        confidences: List[float] = []
        total_time_ms = 0.0

        for d in decisions:
            # Status counts
            if d.status == ResolutionStatus.RESOLVED.value:
                resolved += 1
            elif d.status == ResolutionStatus.PENDING_REVIEW.value:
                pending += 1
            elif d.status == ResolutionStatus.MANUAL_OVERRIDE.value:
                manual += 1
            elif d.status == ResolutionStatus.FAILED.value:
                failed += 1
            elif d.status == ResolutionStatus.SKIPPED.value:
                skipped += 1

            # Auto vs manual
            if d.is_auto:
                auto_count += 1
            else:
                manual_review_count += 1

            # Strategy counts
            strat = d.strategy or "unknown"
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

            # Confidence
            confidences.append(d.confidence)

            # Processing time
            total_time_ms += d.processing_time_ms

        # Compute confidence statistics
        avg_conf = 0.0
        min_conf = 0.0
        max_conf = 0.0
        if confidences:
            avg_conf = round(statistics.mean(confidences), 4)
            min_conf = round(min(confidences), 4)
            max_conf = round(max(confidences), 4)

        summary = ResolutionSummary(
            total_decisions=total,
            resolved_count=resolved,
            pending_count=pending,
            manual_count=manual,
            failed_count=failed,
            skipped_count=skipped,
            auto_count=auto_count,
            manual_review_count=manual_review_count,
            strategy_counts=strategy_counts,
            average_confidence=avg_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            total_processing_time_ms=round(total_time_ms, 3),
        )

        # Provenance
        summary.provenance_hash = self._compute_provenance(
            operation="summarize_resolutions",
            input_data={
                "total_decisions": total,
                "decision_ids": [d.decision_id for d in decisions],
            },
            output_data={
                "resolved": resolved,
                "pending": pending,
                "manual": manual,
                "failed": failed,
                "average_confidence": avg_conf,
            },
        )

        logger.info(
            "Resolution summary: total=%d resolved=%d pending=%d "
            "manual=%d failed=%d avg_confidence=%.3f",
            total,
            resolved,
            pending,
            manual,
            failed,
            avg_conf,
        )
        return summary

    # ------------------------------------------------------------------
    # Auto strategy selection
    # ------------------------------------------------------------------

    def auto_select_strategy(
        self,
        discrepancy: Discrepancy,
        source_credibilities: Dict[str, SourceCredibility],
    ) -> ResolutionStrategy:
        """Automatically select the best resolution strategy.

        Selection rules:
        - CRITICAL severity -> MANUAL_REVIEW
        - HIGH severity + credibility spread > 0.3 -> PRIORITY_WINS
        - HIGH severity + credibility spread <= 0.3 -> MANUAL_REVIEW
        - MEDIUM severity -> WEIGHTED_AVERAGE (numeric), PRIORITY_WINS (other)
        - LOW severity -> PRIORITY_WINS
        - INFO severity -> PRIORITY_WINS

        Args:
            discrepancy: The discrepancy to evaluate.
            source_credibilities: Credibility profiles by source_id.

        Returns:
            The recommended ResolutionStrategy.
        """
        severity = discrepancy.severity
        if isinstance(severity, str):
            try:
                severity = DiscrepancySeverity(severity)
            except ValueError:
                severity = DiscrepancySeverity.MEDIUM

        field_type = discrepancy.field_type
        if isinstance(field_type, str):
            try:
                field_type = FieldType(field_type)
            except ValueError:
                field_type = FieldType.STRING

        # Compute credibility spread
        cred_spread = self._compute_credibility_spread(
            discrepancy.source_values, source_credibilities
        )

        logger.debug(
            "auto_select_strategy: severity=%s field_type=%s "
            "cred_spread=%.3f",
            severity.value,
            field_type.value,
            cred_spread,
        )

        # CRITICAL -> always manual review
        if severity == DiscrepancySeverity.CRITICAL:
            return ResolutionStrategy.MANUAL_REVIEW

        # HIGH severity
        if severity == DiscrepancySeverity.HIGH:
            if cred_spread > 0.3:
                return ResolutionStrategy.PRIORITY_WINS
            return ResolutionStrategy.MANUAL_REVIEW

        # MEDIUM severity
        if severity == DiscrepancySeverity.MEDIUM:
            if field_type == FieldType.NUMERIC:
                return ResolutionStrategy.WEIGHTED_AVERAGE
            return ResolutionStrategy.PRIORITY_WINS

        # LOW and INFO -> priority_wins
        return ResolutionStrategy.PRIORITY_WINS

    def _compute_credibility_spread(
        self,
        source_values: Dict[str, Any],
        source_credibilities: Dict[str, SourceCredibility],
    ) -> float:
        """Compute the spread (range) of credibility scores.

        Args:
            source_values: Mapping of source_id to value.
            source_credibilities: Credibility profiles by source_id.

        Returns:
            Difference between max and min credibility scores.
            Returns 0.0 if fewer than 2 sources have credibility.
        """
        scores = []
        for src_id in source_values:
            cred = source_credibilities.get(src_id)
            if cred:
                scores.append(cred.credibility_score)

        if len(scores) < 2:
            return 0.0
        return max(scores) - min(scores)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _compute_provenance(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        """Compute SHA-256 provenance hash and record in provenance chain.

        Args:
            operation: Name of the operation.
            input_data: Input data to hash.
            output_data: Output data to hash.

        Returns:
            Chain hash from the provenance tracker.
        """
        input_hash = _compute_hash(input_data)
        output_hash = _compute_hash(output_data)

        chain_hash = self._provenance.add_to_chain(
            operation=operation,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "engine": "ResolutionEngine",
                "agent": "AGENT-DATA-015",
            },
        )
        return chain_hash

    # ------------------------------------------------------------------
    # Internal helpers for building decisions
    # ------------------------------------------------------------------

    def _build_no_sources_decision(
        self,
        discrepancy: Discrepancy,
        strategy: ResolutionStrategy,
    ) -> ResolutionDecision:
        """Build a FAILED decision when no valid sources are available.

        Args:
            discrepancy: The discrepancy that could not be resolved.
            strategy: The strategy that was attempted.

        Returns:
            ResolutionDecision with FAILED status.
        """
        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=strategy.value,
            status=ResolutionStatus.FAILED.value,
            resolved_value=None,
            original_values=dict(discrepancy.source_values),
            confidence=0.0,
            justification=(
                f"Resolution failed: no valid sources found for "
                f"strategy={strategy.value} on field "
                f"'{discrepancy.field_name}'"
            ),
            is_auto=True,
        )

    def _build_fallback_decision(
        self,
        discrepancy: Discrepancy,
        strategy: ResolutionStrategy,
    ) -> ResolutionDecision:
        """Build a decision using the first available source value.

        Used when the primary strategy cannot identify a winner (e.g.
        no valid timestamps for most_recent).

        Args:
            discrepancy: The discrepancy to resolve.
            strategy: The original strategy that was attempted.

        Returns:
            ResolutionDecision with the first source's value.
        """
        if not discrepancy.source_values:
            return self._build_no_sources_decision(discrepancy, strategy)

        first_src = next(iter(discrepancy.source_values))
        first_val = discrepancy.source_values[first_src]

        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=strategy.value,
            status=ResolutionStatus.RESOLVED.value,
            winning_source_id=first_src,
            winning_source_name=first_src,
            resolved_value=first_val,
            original_values=dict(discrepancy.source_values),
            confidence=0.3,
            justification=(
                f"Fallback resolution: strategy={strategy.value} could not "
                f"identify a clear winner; used first available source "
                f"'{first_src}' (confidence=0.3)"
            ),
            is_auto=True,
        )

    @staticmethod
    def _build_failed_decision(
        discrepancy: Discrepancy,
        strategy: ResolutionStrategy,
        error_message: str,
    ) -> ResolutionDecision:
        """Build a FAILED decision due to an exception.

        Args:
            discrepancy: The discrepancy that failed.
            strategy: The strategy that was attempted.
            error_message: Error description.

        Returns:
            ResolutionDecision with FAILED status and error details.
        """
        return ResolutionDecision(
            discrepancy_id=discrepancy.discrepancy_id,
            entity_id=discrepancy.entity_id,
            field_name=discrepancy.field_name,
            strategy=strategy.value,
            status=ResolutionStatus.FAILED.value,
            resolved_value=None,
            original_values=dict(discrepancy.source_values),
            confidence=0.0,
            justification=f"Resolution failed with error: {error_message}",
            is_auto=True,
            metadata={"error": error_message},
        )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = ["ResolutionEngine"]
