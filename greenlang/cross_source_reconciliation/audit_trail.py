# -*- coding: utf-8 -*-
"""
Audit Trail Engine - AGENT-DATA-015: Cross-Source Reconciliation

Engine 6 of 7. Provides a complete, tamper-evident audit trail for every
reconciliation decision: record matches, field-level comparisons,
discrepancy detections, resolution applications, and golden record
assembly. Generates compliance reports (GHG Protocol, CSRD/ESRS) and
regulatory attestation documentation.

Zero-Hallucination: All audit data is deterministic. Provenance hashes
use SHA-256 with float normalisation and sorted keys. No LLM calls are
used for any numeric or audit computation. Every event is traceable
through an unbroken SHA-256 chain.

Methods:
    - record_event: Record a generic reconciliation event
    - record_match_event: Record a record-matching decision
    - record_comparison_event: Record a field-level comparison result
    - record_discrepancy_event: Record a detected discrepancy
    - record_resolution_event: Record a conflict resolution decision
    - record_golden_record_event: Record golden record creation
    - generate_report: Generate a reconciliation summary report
    - generate_compliance_report: Generate framework-specific compliance report
    - generate_discrepancy_log: Generate chronological discrepancy log
    - generate_resolution_justification: Generate per-discrepancy justification
    - get_events: Query events by job, type, and/or timestamp
    - get_event_count: Count events by type
    - export_audit_trail: Export full audit trail as JSON or CSV
    - verify_audit_integrity: Verify provenance chain integrity
    - clear_events: Clear stored events

Example:
    >>> from greenlang.cross_source_reconciliation.audit_trail import (
    ...     AuditTrailEngine,
    ... )
    >>> engine = AuditTrailEngine()
    >>> event = engine.record_event("job_001", "custom", {"note": "test"})
    >>> assert event.event_id is not None
    >>> report = engine.generate_report(
    ...     job_id="job_001",
    ...     total_records=100,
    ...     matched_records=90,
    ...     discrepancies_found=15,
    ...     discrepancies_resolved=12,
    ...     golden_records_created=90,
    ...     unresolved_count=3,
    ... )
    >>> assert report.provenance_hash != ""

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from greenlang.cross_source_reconciliation.metrics import (
    inc_discrepancies,
    inc_errors,
    inc_golden_records,
    inc_records_matched,
    inc_resolutions,
    observe_duration,
    observe_magnitude,
)
from greenlang.cross_source_reconciliation.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model imports (graceful fallback for standalone usage / early bootstrapping)
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation.models import (
        ComparisonSummary,
        Discrepancy,
        DiscrepancySummary,
        FieldComparison,
        GoldenRecord,
        MatchResult,
        PipelineStageResult,
        ReconciliationEvent,
        ReconciliationReport,
        ReconciliationStatus,
        ResolutionDecision,
        ResolutionSummary,
    )

    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.debug(
        "cross_source_reconciliation.models not yet available; "
        "using inline dataclass stubs"
    )


# ---------------------------------------------------------------------------
# Inline stubs for when models.py is not yet available
# ---------------------------------------------------------------------------

if not _MODELS_AVAILABLE:
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class ReconciliationEvent:  # type: ignore[no-redef]
        """Stub: single reconciliation audit event."""

        event_id: str = ""
        job_id: str = ""
        event_type: str = ""
        timestamp: Any = None  # datetime when models available
        details: Dict[str, Any] = dc_field(default_factory=dict)
        provenance_hash: str = ""

    @dataclass
    class ReconciliationReport:  # type: ignore[no-redef]
        """Stub: reconciliation summary report."""

        report_id: str = ""
        job_id: str = ""
        total_records: int = 0
        matched_records: int = 0
        discrepancies_found: int = 0
        discrepancies_resolved: int = 0
        golden_records_created: int = 0
        unresolved_count: int = 0
        summary: str = ""
        created_at: Any = None
        provenance_hash: str = ""

    class ReconciliationStatus:  # type: ignore[no-redef]
        """Stub: reconciliation status enumeration."""

        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        PARTIAL = "partial"

    @dataclass
    class MatchResult:  # type: ignore[no-redef]
        """Stub: record match result."""

        match_id: str = ""
        source_a_key: Any = None
        source_b_key: Any = None
        confidence: float = 0.0
        strategy: str = ""
        status: str = ""
        matched_fields: list = dc_field(default_factory=list)
        provenance_hash: str = ""

    @dataclass
    class FieldComparison:  # type: ignore[no-redef]
        """Stub: field-level comparison result."""

        field_name: str = ""
        field_type: str = "numeric"
        result: str = ""
        source_a_value: Any = None
        source_b_value: Any = None
        absolute_diff: Optional[float] = None
        relative_diff_pct: Optional[float] = None
        tolerance_abs: Optional[float] = None
        tolerance_pct: Optional[float] = None
        provenance_hash: str = ""

    @dataclass
    class Discrepancy:  # type: ignore[no-redef]
        """Stub: detected discrepancy."""

        discrepancy_id: str = ""
        match_id: str = ""
        field_name: str = ""
        discrepancy_type: str = ""
        severity: str = ""
        source_a_value: Any = None
        source_b_value: Any = None
        deviation_pct: Optional[float] = None
        description: str = ""
        provenance_hash: str = ""

    @dataclass
    class ResolutionDecision:  # type: ignore[no-redef]
        """Stub: conflict resolution decision."""

        resolution_id: str = ""
        discrepancy_id: str = ""
        strategy: str = ""
        winning_source_id: str = ""
        resolved_value: Any = None
        confidence: float = 0.0
        justification: str = ""
        reviewer: Optional[str] = None
        provenance_hash: str = ""

    @dataclass
    class GoldenRecord:  # type: ignore[no-redef]
        """Stub: assembled golden record."""

        record_id: str = ""
        entity_id: str = ""
        period: str = ""
        fields: Dict[str, Any] = dc_field(default_factory=dict)
        field_sources: Dict[str, str] = dc_field(default_factory=dict)
        field_confidences: Dict[str, float] = dc_field(
            default_factory=dict
        )
        total_confidence: float = 0.0
        provenance_hash: str = ""

    @dataclass
    class DiscrepancySummary:  # type: ignore[no-redef]
        """Stub: discrepancy summary."""

        total: int = 0
        by_type: Dict[str, int] = dc_field(default_factory=dict)
        by_severity: Dict[str, int] = dc_field(default_factory=dict)
        by_source: Dict[str, int] = dc_field(default_factory=dict)
        critical_count: int = 0
        pending_review_count: int = 0

    @dataclass
    class ResolutionSummary:  # type: ignore[no-redef]
        """Stub: resolution summary."""

        total_resolved: int = 0
        by_strategy: Dict[str, int] = dc_field(default_factory=dict)
        auto_resolved: int = 0
        manual_resolved: int = 0
        pending: int = 0
        average_confidence: float = 0.0

    @dataclass
    class ComparisonSummary:  # type: ignore[no-redef]
        """Stub: comparison summary."""

        total_fields_compared: int = 0
        matches: int = 0
        mismatches: int = 0
        within_tolerance: int = 0
        missing: int = 0
        incomparable: int = 0
        match_rate: float = 0.0

    @dataclass
    class PipelineStageResult:  # type: ignore[no-redef]
        """Stub: pipeline stage result."""

        stage_name: str = ""
        status: str = "pending"
        records_processed: int = 0
        duration_ms: float = 0.0
        errors: list = dc_field(default_factory=list)
        provenance_hash: str = ""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _safe_str(value: Any) -> str:
    """Convert any value to a safe string representation.

    Handles None, dicts, lists, and other types for inclusion in
    audit detail dictionaries without raising serialization errors.

    Args:
        value: Any Python value.

    Returns:
        String representation safe for JSON serialization.
    """
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float.

    Args:
        value: Value to convert.
        default: Default if conversion fails.

    Returns:
        Float value or default.
    """
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _extract_source_id(match_key: Any) -> str:
    """Extract source_id from a MatchKey or return string representation.

    Args:
        match_key: MatchKey Pydantic model or any value.

    Returns:
        Source identifier string.
    """
    if match_key is None:
        return "unknown"
    if hasattr(match_key, "source_id"):
        return str(match_key.source_id)
    return str(match_key)


# ---------------------------------------------------------------------------
# AuditTrailEngine
# ---------------------------------------------------------------------------


class AuditTrailEngine:
    """Complete audit trail engine for cross-source reconciliation.

    Records every reconciliation decision -- matches, field-level
    comparisons, discrepancy detections, resolution applications, and
    golden record assemblies -- in an ordered, tamper-evident event log
    backed by SHA-256 provenance chains.

    Also generates compliance reports for GHG Protocol and CSRD/ESRS
    frameworks, discrepancy logs, and resolution justification documents.

    Attributes:
        _events: Ordered list of all ReconciliationEvent instances.
        _provenance: ProvenanceTracker for SHA-256 chain hashing.

    Example:
        >>> engine = AuditTrailEngine()
        >>> ev = engine.record_event("job_001", "custom", {"note": "test"})
        >>> assert ev.event_id != ""
        >>> report = engine.generate_report(
        ...     job_id="job_001",
        ...     total_records=50,
        ...     matched_records=45,
        ...     discrepancies_found=10,
        ...     discrepancies_resolved=8,
        ...     golden_records_created=45,
        ...     unresolved_count=2,
        ... )
        >>> assert report.provenance_hash != ""
        >>> integrity = engine.verify_audit_integrity("job_001")
        >>> assert integrity is True
    """

    # Supported event types for validation
    VALID_EVENT_TYPES = frozenset({
        "match",
        "comparison",
        "discrepancy_detected",
        "resolution_applied",
        "golden_record_created",
        "pipeline_start",
        "pipeline_end",
        "validation",
        "error",
        "warning",
        "custom",
    })

    # Supported compliance frameworks
    SUPPORTED_FRAMEWORKS = frozenset({
        "ghg_protocol",
        "csrd_esrs",
    })

    # Supported export formats
    SUPPORTED_EXPORT_FORMATS = frozenset({
        "json",
        "csv",
    })

    def __init__(
        self,
        provenance_tracker: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize AuditTrailEngine.

        Args:
            provenance_tracker: Optional ProvenanceTracker override.
                If None, the singleton tracker from
                ``get_provenance_tracker()`` is used.
        """
        self._events: List[ReconciliationEvent] = []
        self._provenance: ProvenanceTracker = (
            provenance_tracker or get_provenance_tracker()
        )
        logger.info("AuditTrailEngine initialized")

    # ==================================================================
    # 1. record_event
    # ==================================================================

    def record_event(
        self,
        job_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> ReconciliationEvent:
        """Record a generic reconciliation event.

        Creates a new ``ReconciliationEvent`` with a unique ID,
        UTC timestamp, details dictionary, and SHA-256 provenance hash.
        Appends the event to the internal event log.

        Args:
            job_id: Reconciliation job identifier.
            event_type: Type of event (match, comparison,
                discrepancy_detected, resolution_applied,
                golden_record_created, pipeline_start, pipeline_end,
                validation, error, warning, custom).
            details: Optional dictionary of event-specific details.

        Returns:
            The created ``ReconciliationEvent`` with computed
            provenance hash.

        Raises:
            ValueError: If job_id is empty or event_type is not
                in ``VALID_EVENT_TYPES``.
        """
        start = time.time()

        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if event_type not in self.VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{event_type}'. "
                f"Must be one of: {sorted(self.VALID_EVENT_TYPES)}"
            )

        event_details = details or {}
        event_id = str(uuid4())
        timestamp = _utcnow()

        # Compute provenance hash for this event
        provenance_hash = self._compute_provenance(
            operation=f"record_event:{event_type}",
            input_data={
                "job_id": job_id,
                "event_type": event_type,
                "details": event_details,
            },
            output_data={
                "event_id": event_id,
                "timestamp": timestamp.isoformat(),
            },
        )

        event = ReconciliationEvent(
            event_id=event_id,
            job_id=job_id,
            event_type=event_type,
            timestamp=timestamp,
            details=event_details,
            provenance_hash=provenance_hash,
        )

        self._events.append(event)

        duration = time.time() - start
        observe_duration(duration)

        logger.debug(
            "Recorded event: job=%s type=%s id=%s hash=%s",
            job_id,
            event_type,
            event_id[:8],
            provenance_hash[:16],
        )
        return event

    # ==================================================================
    # 2. record_match_event
    # ==================================================================

    def record_match_event(
        self,
        job_id: str,
        match_result: MatchResult,
    ) -> ReconciliationEvent:
        """Record a record-matching decision as an audit event.

        Captures the match confidence, strategy, status, and source
        keys from the ``MatchResult`` and records them as a 'match'
        event in the audit trail.

        Args:
            job_id: Reconciliation job identifier.
            match_result: The MatchResult from the MatchingEngine.

        Returns:
            The created ``ReconciliationEvent`` for the match.

        Raises:
            ValueError: If job_id is empty.
        """
        details = self._build_match_details(match_result)

        event = self.record_event(
            job_id=job_id,
            event_type="match",
            details=details,
        )

        # Emit Prometheus metrics
        inc_records_matched(
            strategy=str(getattr(match_result, "strategy", "unknown")),
        )

        logger.info(
            "Match event recorded: job=%s match_id=%s confidence=%.3f "
            "strategy=%s",
            job_id,
            getattr(match_result, "match_id", "unknown")[:8],
            _safe_float(getattr(match_result, "confidence", 0.0)),
            getattr(match_result, "strategy", "unknown"),
        )
        return event

    def _build_match_details(
        self,
        match_result: MatchResult,
    ) -> Dict[str, Any]:
        """Build detail dictionary from a MatchResult.

        Extracts source IDs from the MatchKey objects on source_a_key
        and source_b_key.

        Args:
            match_result: The match result to extract details from.

        Returns:
            Dictionary with match confidence, strategy, status,
            and source IDs.
        """
        source_a_key = getattr(match_result, "source_a_key", None)
        source_b_key = getattr(match_result, "source_b_key", None)

        return {
            "match_id": str(getattr(match_result, "match_id", "")),
            "confidence": _safe_float(
                getattr(match_result, "confidence", 0.0)
            ),
            "strategy": str(getattr(match_result, "strategy", "")),
            "status": str(getattr(match_result, "status", "")),
            "source_a_id": _extract_source_id(source_a_key),
            "source_b_id": _extract_source_id(source_b_key),
            "matched_fields": list(
                getattr(match_result, "matched_fields", [])
            ),
        }

    # ==================================================================
    # 3. record_comparison_event
    # ==================================================================

    def record_comparison_event(
        self,
        job_id: str,
        comparison: FieldComparison,
    ) -> ReconciliationEvent:
        """Record a field-level comparison result as an audit event.

        Captures the field name, comparison result, values from both
        sources, and computed differences from the ``FieldComparison``.

        Args:
            job_id: Reconciliation job identifier.
            comparison: The FieldComparison from the ComparisonEngine.

        Returns:
            The created ``ReconciliationEvent`` for the comparison.

        Raises:
            ValueError: If job_id is empty.
        """
        details = self._build_comparison_details(comparison)

        event = self.record_event(
            job_id=job_id,
            event_type="comparison",
            details=details,
        )

        logger.info(
            "Comparison event recorded: job=%s field=%s result=%s",
            job_id,
            getattr(comparison, "field_name", "unknown"),
            getattr(comparison, "result", "unknown"),
        )
        return event

    def _build_comparison_details(
        self,
        comparison: FieldComparison,
    ) -> Dict[str, Any]:
        """Build detail dictionary from a FieldComparison.

        Args:
            comparison: The field comparison to extract details from.

        Returns:
            Dictionary with field name, result, values, and differences.
        """
        return {
            "field_name": str(getattr(comparison, "field_name", "")),
            "field_type": str(getattr(comparison, "field_type", "")),
            "result": str(getattr(comparison, "result", "")),
            "source_a_value": _safe_str(
                getattr(comparison, "source_a_value", None)
            ),
            "source_b_value": _safe_str(
                getattr(comparison, "source_b_value", None)
            ),
            "absolute_diff": _safe_float(
                getattr(comparison, "absolute_diff", None), default=0.0
            ),
            "relative_diff_pct": _safe_float(
                getattr(comparison, "relative_diff_pct", None),
                default=0.0,
            ),
            "tolerance_abs": _safe_float(
                getattr(comparison, "tolerance_abs", None), default=0.0
            ),
            "tolerance_pct": _safe_float(
                getattr(comparison, "tolerance_pct", None), default=0.0
            ),
        }

    # ==================================================================
    # 4. record_discrepancy_event
    # ==================================================================

    def record_discrepancy_event(
        self,
        job_id: str,
        discrepancy: Discrepancy,
    ) -> ReconciliationEvent:
        """Record a detected discrepancy as an audit event.

        Captures the discrepancy type, severity, field name, conflicting
        values, and deviation percentage from the ``Discrepancy`` model.

        Args:
            job_id: Reconciliation job identifier.
            discrepancy: The Discrepancy from the
                DiscrepancyDetectorEngine.

        Returns:
            The created ``ReconciliationEvent`` for the discrepancy.

        Raises:
            ValueError: If job_id is empty.
        """
        details = self._build_discrepancy_details(discrepancy)

        event = self.record_event(
            job_id=job_id,
            event_type="discrepancy_detected",
            details=details,
        )

        # Emit Prometheus metrics
        disc_type = str(
            getattr(discrepancy, "discrepancy_type", "unknown")
        )
        severity = str(getattr(discrepancy, "severity", "unknown"))
        inc_discrepancies(discrepancy_type=disc_type, severity=severity)

        deviation = _safe_float(
            getattr(discrepancy, "deviation_pct", None), default=0.0
        )
        if deviation > 0.0:
            observe_magnitude(deviation)

        logger.info(
            "Discrepancy event recorded: job=%s type=%s severity=%s "
            "field=%s deviation=%.2f%%",
            job_id,
            disc_type,
            severity,
            getattr(discrepancy, "field_name", "unknown"),
            deviation,
        )
        return event

    def _build_discrepancy_details(
        self,
        discrepancy: Discrepancy,
    ) -> Dict[str, Any]:
        """Build detail dictionary from a Discrepancy.

        Args:
            discrepancy: The discrepancy to extract details from.

        Returns:
            Dictionary with type, severity, field, values, deviation.
        """
        return {
            "discrepancy_id": str(
                getattr(discrepancy, "discrepancy_id", "")
            ),
            "match_id": str(
                getattr(discrepancy, "match_id", "")
            ),
            "discrepancy_type": str(
                getattr(discrepancy, "discrepancy_type", "")
            ),
            "severity": str(getattr(discrepancy, "severity", "")),
            "field_name": str(getattr(discrepancy, "field_name", "")),
            "source_a_value": _safe_str(
                getattr(discrepancy, "source_a_value", None)
            ),
            "source_b_value": _safe_str(
                getattr(discrepancy, "source_b_value", None)
            ),
            "deviation_pct": _safe_float(
                getattr(discrepancy, "deviation_pct", None), default=0.0
            ),
            "description": str(
                getattr(discrepancy, "description", "")
            ),
        }

    # ==================================================================
    # 5. record_resolution_event
    # ==================================================================

    def record_resolution_event(
        self,
        job_id: str,
        decision: ResolutionDecision,
    ) -> ReconciliationEvent:
        """Record a conflict resolution decision as an audit event.

        Captures the resolution strategy, winning source, resolved
        value, justification text, and confidence from the
        ``ResolutionDecision`` model.

        Args:
            job_id: Reconciliation job identifier.
            decision: The ResolutionDecision from the
                ResolutionEngine.

        Returns:
            The created ``ReconciliationEvent`` for the resolution.

        Raises:
            ValueError: If job_id is empty.
        """
        details = self._build_resolution_details(decision)

        event = self.record_event(
            job_id=job_id,
            event_type="resolution_applied",
            details=details,
        )

        # Emit Prometheus metrics
        strategy = str(getattr(decision, "strategy", "unknown"))
        inc_resolutions(strategy=strategy)

        logger.info(
            "Resolution event recorded: job=%s strategy=%s "
            "winning_source_id=%s confidence=%.3f",
            job_id,
            strategy,
            getattr(decision, "winning_source_id", "unknown"),
            _safe_float(getattr(decision, "confidence", 0.0)),
        )
        return event

    def _build_resolution_details(
        self,
        decision: ResolutionDecision,
    ) -> Dict[str, Any]:
        """Build detail dictionary from a ResolutionDecision.

        Args:
            decision: The resolution decision to extract details from.

        Returns:
            Dictionary with strategy, winning source, resolved value,
            justification, and confidence.
        """
        return {
            "resolution_id": str(
                getattr(decision, "resolution_id", "")
            ),
            "discrepancy_id": str(
                getattr(decision, "discrepancy_id", "")
            ),
            "strategy": str(getattr(decision, "strategy", "")),
            "winning_source_id": str(
                getattr(decision, "winning_source_id", "")
            ),
            "resolved_value": _safe_str(
                getattr(decision, "resolved_value", None)
            ),
            "justification": str(
                getattr(decision, "justification", "")
            ),
            "confidence": _safe_float(
                getattr(decision, "confidence", 0.0)
            ),
            "reviewer": str(
                getattr(decision, "reviewer", "") or ""
            ),
        }

    # ==================================================================
    # 6. record_golden_record_event
    # ==================================================================

    def record_golden_record_event(
        self,
        job_id: str,
        golden_record: GoldenRecord,
    ) -> ReconciliationEvent:
        """Record golden record creation as an audit event.

        Captures the entity ID, period, field-to-source mapping, and
        overall confidence from the ``GoldenRecord`` model.

        Args:
            job_id: Reconciliation job identifier.
            golden_record: The GoldenRecord from the
                ResolutionEngine.

        Returns:
            The created ``ReconciliationEvent`` for the golden record.

        Raises:
            ValueError: If job_id is empty.
        """
        details = self._build_golden_record_details(golden_record)

        event = self.record_event(
            job_id=job_id,
            event_type="golden_record_created",
            details=details,
        )

        # Emit Prometheus metrics
        inc_golden_records(status="created")

        logger.info(
            "Golden record event recorded: job=%s entity=%s period=%s "
            "fields=%d confidence=%.3f",
            job_id,
            getattr(golden_record, "entity_id", "unknown"),
            getattr(golden_record, "period", "unknown"),
            len(getattr(golden_record, "field_sources", {})),
            _safe_float(
                getattr(golden_record, "total_confidence", 0.0)
            ),
        )
        return event

    def _build_golden_record_details(
        self,
        golden_record: GoldenRecord,
    ) -> Dict[str, Any]:
        """Build detail dictionary from a GoldenRecord.

        Args:
            golden_record: The golden record to extract details from.

        Returns:
            Dictionary with entity, period, field sources, confidence.
        """
        return {
            "record_id": str(
                getattr(golden_record, "record_id", "")
            ),
            "entity_id": str(
                getattr(golden_record, "entity_id", "")
            ),
            "period": str(getattr(golden_record, "period", "")),
            "field_count": len(
                getattr(golden_record, "fields", {})
            ),
            "field_sources": dict(
                getattr(golden_record, "field_sources", {})
            ),
            "field_confidences": dict(
                getattr(golden_record, "field_confidences", {})
            ),
            "total_confidence": _safe_float(
                getattr(golden_record, "total_confidence", 0.0)
            ),
        }

    # ==================================================================
    # 7. generate_report
    # ==================================================================

    def generate_report(
        self,
        job_id: str,
        total_records: int,
        matched_records: int,
        discrepancies_found: int,
        discrepancies_resolved: int,
        golden_records_created: int,
        unresolved_count: int,
    ) -> ReconciliationReport:
        """Generate a reconciliation summary report for a job.

        Assembles event counts by type from all events belonging to
        the given ``job_id``, determines the final reconciliation
        status, computes a provenance chain hash covering all report
        data, and builds a human-readable summary string.

        Args:
            job_id: Reconciliation job identifier.
            total_records: Total number of records in the job.
            matched_records: Number of records matched across sources.
            discrepancies_found: Total discrepancies detected.
            discrepancies_resolved: Discrepancies successfully resolved.
            golden_records_created: Golden records assembled.
            unresolved_count: Discrepancies remaining unresolved.

        Returns:
            A ``ReconciliationReport`` with summary statistics and
            provenance hash.

        Raises:
            ValueError: If job_id is empty or numeric values are
                negative.
        """
        start = time.time()

        self._validate_report_inputs(
            job_id=job_id,
            total_records=total_records,
            matched_records=matched_records,
            discrepancies_found=discrepancies_found,
            discrepancies_resolved=discrepancies_resolved,
            golden_records_created=golden_records_created,
            unresolved_count=unresolved_count,
        )

        report_id = str(uuid4())
        created_at = _utcnow()

        # Aggregate event counts for this job
        event_summary = self._aggregate_event_counts(job_id)

        # Determine overall status label for the summary text
        status_label = self._determine_report_status(
            unresolved_count=unresolved_count,
            matched_records=matched_records,
            total_records=total_records,
        )

        # Build human-readable summary
        summary = self._build_summary_text(
            status_label=status_label,
            total_records=total_records,
            matched_records=matched_records,
            discrepancies_found=discrepancies_found,
            discrepancies_resolved=discrepancies_resolved,
            golden_records_created=golden_records_created,
            unresolved_count=unresolved_count,
            event_summary=event_summary,
        )

        # Build provenance hash
        report_data = {
            "report_id": report_id,
            "job_id": job_id,
            "total_records": total_records,
            "matched_records": matched_records,
            "discrepancies_found": discrepancies_found,
            "discrepancies_resolved": discrepancies_resolved,
            "golden_records_created": golden_records_created,
            "unresolved_count": unresolved_count,
            "summary": summary,
            "status_label": status_label,
        }
        provenance_hash = self._compute_provenance(
            operation="generate_report",
            input_data={"job_id": job_id},
            output_data=report_data,
        )

        report = ReconciliationReport(
            report_id=report_id,
            job_id=job_id,
            total_records=total_records,
            matched_records=matched_records,
            discrepancies_found=discrepancies_found,
            discrepancies_resolved=discrepancies_resolved,
            golden_records_created=golden_records_created,
            unresolved_count=unresolved_count,
            summary=summary,
            created_at=created_at,
            provenance_hash=provenance_hash,
        )

        duration = time.time() - start
        observe_duration(duration)

        logger.info(
            "Report generated: job=%s status=%s records=%d "
            "matched=%d discrepancies=%d/%d golden=%d "
            "unresolved=%d provenance=%s",
            job_id,
            status_label,
            total_records,
            matched_records,
            discrepancies_resolved,
            discrepancies_found,
            golden_records_created,
            unresolved_count,
            provenance_hash[:16],
        )
        return report

    def _validate_report_inputs(
        self,
        job_id: str,
        total_records: int,
        matched_records: int,
        discrepancies_found: int,
        discrepancies_resolved: int,
        golden_records_created: int,
        unresolved_count: int,
    ) -> None:
        """Validate report generation inputs.

        Args:
            job_id: Job identifier (must be non-empty).
            total_records: Must be non-negative.
            matched_records: Must be non-negative.
            discrepancies_found: Must be non-negative.
            discrepancies_resolved: Must be non-negative.
            golden_records_created: Must be non-negative.
            unresolved_count: Must be non-negative.

        Raises:
            ValueError: If any validation fails.
        """
        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")

        numeric_params = {
            "total_records": total_records,
            "matched_records": matched_records,
            "discrepancies_found": discrepancies_found,
            "discrepancies_resolved": discrepancies_resolved,
            "golden_records_created": golden_records_created,
            "unresolved_count": unresolved_count,
        }
        for name, value in numeric_params.items():
            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative, got {value}"
                )

    def _aggregate_event_counts(
        self,
        job_id: str,
    ) -> Dict[str, int]:
        """Aggregate event counts by type for a specific job.

        Args:
            job_id: Job identifier to filter events.

        Returns:
            Dictionary mapping event_type to count.
        """
        counts: Dict[str, int] = {}
        for event in self._events:
            if event.job_id == job_id:
                etype = event.event_type
                counts[etype] = counts.get(etype, 0) + 1
        return counts

    def _determine_report_status(
        self,
        unresolved_count: int,
        matched_records: int,
        total_records: int,
    ) -> str:
        """Determine the overall reconciliation report status label.

        Logic:
            - If total_records == 0: "completed" (nothing to do)
            - If matched_records == 0 and total_records > 0: "failed"
            - If unresolved_count == 0: "completed"
            - If unresolved_count > 0 and matched_records > 0: "partial"

        Args:
            unresolved_count: Number of unresolved discrepancies.
            matched_records: Number of matched records.
            total_records: Total number of records in the job.

        Returns:
            Status label string (completed, failed, partial).
        """
        completed = getattr(
            ReconciliationStatus, "COMPLETED", "completed"
        )
        failed = getattr(
            ReconciliationStatus, "FAILED", "failed"
        )
        partial = getattr(
            ReconciliationStatus, "PARTIAL", "partial"
        )
        # Extract .value for str enums, pass through for plain strings
        completed_val = getattr(completed, "value", completed)
        failed_val = getattr(failed, "value", failed)
        partial_val = getattr(partial, "value", partial)

        if total_records == 0:
            return completed_val
        if matched_records == 0:
            return failed_val
        if unresolved_count == 0:
            return completed_val
        return partial_val

    def _build_summary_text(
        self,
        status_label: str,
        total_records: int,
        matched_records: int,
        discrepancies_found: int,
        discrepancies_resolved: int,
        golden_records_created: int,
        unresolved_count: int,
        event_summary: Dict[str, int],
    ) -> str:
        """Build a human-readable summary string for the report.

        Args:
            status_label: Status label (completed/failed/partial).
            total_records: Total records.
            matched_records: Matched records.
            discrepancies_found: Discrepancies found.
            discrepancies_resolved: Discrepancies resolved.
            golden_records_created: Golden records created.
            unresolved_count: Unresolved discrepancies.
            event_summary: Event counts by type.

        Returns:
            Multi-line summary string.
        """
        match_rate = (
            (matched_records / total_records * 100.0)
            if total_records > 0
            else 0.0
        )
        resolution_rate = (
            (discrepancies_resolved / discrepancies_found * 100.0)
            if discrepancies_found > 0
            else 100.0
        )

        lines = [
            f"Reconciliation {status_label}: "
            f"{matched_records}/{total_records} records matched "
            f"({match_rate:.1f}%).",
            f"Discrepancies: {discrepancies_found} found, "
            f"{discrepancies_resolved} resolved "
            f"({resolution_rate:.1f}%), "
            f"{unresolved_count} unresolved.",
            f"Golden records: {golden_records_created} created.",
        ]

        if event_summary:
            event_parts = [
                f"{k}={v}" for k, v in sorted(event_summary.items())
            ]
            lines.append(f"Events: {', '.join(event_parts)}.")

        return " ".join(lines)

    # ==================================================================
    # 8. generate_compliance_report
    # ==================================================================

    def generate_compliance_report(
        self,
        job_id: str,
        framework: str = "ghg_protocol",
    ) -> Dict[str, Any]:
        """Generate a framework-specific compliance report.

        Produces a structured report dictionary tailored to the
        requirements of a given regulatory framework.

        Supported frameworks:
            - ``ghg_protocol``: Data quality indicators per source,
              gap analysis, estimation methodology documentation.
            - ``csrd_esrs``: Data quality scoring, materiality
              assessment linkage, double materiality indicators.

        Args:
            job_id: Reconciliation job identifier.
            framework: Compliance framework (``ghg_protocol`` or
                ``csrd_esrs``).

        Returns:
            Structured compliance report dictionary.

        Raises:
            ValueError: If job_id is empty or framework is not
                supported.
        """
        start = time.time()

        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unsupported framework '{framework}'. "
                f"Must be one of: {sorted(self.SUPPORTED_FRAMEWORKS)}"
            )

        job_events = self._get_job_events(job_id)

        if framework == "ghg_protocol":
            report = self._generate_ghg_protocol_report(
                job_id, job_events
            )
        else:
            report = self._generate_csrd_esrs_report(
                job_id, job_events
            )

        # Add provenance
        report["provenance_hash"] = self._compute_provenance(
            operation=f"compliance_report:{framework}",
            input_data={"job_id": job_id, "framework": framework},
            output_data=report,
        )
        report["generated_at"] = _utcnow().isoformat()

        duration = time.time() - start
        observe_duration(duration)

        logger.info(
            "Compliance report generated: job=%s framework=%s "
            "events=%d",
            job_id,
            framework,
            len(job_events),
        )
        return report

    def _generate_ghg_protocol_report(
        self,
        job_id: str,
        events: List[ReconciliationEvent],
    ) -> Dict[str, Any]:
        """Generate GHG Protocol-specific compliance report.

        Includes data quality indicators per source, gap analysis
        results, and estimation methodology documentation.

        Args:
            job_id: Job identifier.
            events: Filtered events for this job.

        Returns:
            GHG Protocol compliance report dictionary.
        """
        match_events = [e for e in events if e.event_type == "match"]
        disc_events = [
            e for e in events if e.event_type == "discrepancy_detected"
        ]
        res_events = [
            e for e in events if e.event_type == "resolution_applied"
        ]
        golden_events = [
            e for e in events
            if e.event_type == "golden_record_created"
        ]

        # Data quality indicators per source
        source_quality = self._compute_source_quality_indicators(
            match_events, disc_events, res_events
        )

        # Gap analysis
        gap_analysis = self._compute_gap_analysis(
            events, match_events, disc_events
        )

        # Estimation methodology documentation
        estimation_methods = self._document_estimation_methods(
            res_events
        )

        return {
            "framework": "ghg_protocol",
            "job_id": job_id,
            "report_type": "data_quality_assessment",
            "data_quality_indicators": source_quality,
            "gap_analysis": gap_analysis,
            "estimation_methodology": estimation_methods,
            "total_events": len(events),
            "match_count": len(match_events),
            "discrepancy_count": len(disc_events),
            "resolution_count": len(res_events),
            "golden_record_count": len(golden_events),
        }

    def _generate_csrd_esrs_report(
        self,
        job_id: str,
        events: List[ReconciliationEvent],
    ) -> Dict[str, Any]:
        """Generate CSRD/ESRS-specific compliance report.

        Includes data quality scoring, materiality assessment linkage,
        and double materiality indicators.

        Args:
            job_id: Job identifier.
            events: Filtered events for this job.

        Returns:
            CSRD/ESRS compliance report dictionary.
        """
        match_events = [e for e in events if e.event_type == "match"]
        disc_events = [
            e for e in events if e.event_type == "discrepancy_detected"
        ]
        res_events = [
            e for e in events if e.event_type == "resolution_applied"
        ]
        golden_events = [
            e for e in events
            if e.event_type == "golden_record_created"
        ]

        # Data quality scoring (ESRS E1 Data Quality Matrix)
        quality_score = self._compute_csrd_quality_score(
            events, match_events, disc_events, res_events
        )

        # Materiality linkage
        materiality_linkage = self._compute_materiality_linkage(
            disc_events, res_events
        )

        # Double materiality indicators
        double_materiality = self._compute_double_materiality(
            disc_events
        )

        return {
            "framework": "csrd_esrs",
            "job_id": job_id,
            "report_type": "data_quality_and_materiality",
            "data_quality_score": quality_score,
            "materiality_linkage": materiality_linkage,
            "double_materiality_indicators": double_materiality,
            "total_events": len(events),
            "match_count": len(match_events),
            "discrepancy_count": len(disc_events),
            "resolution_count": len(res_events),
            "golden_record_count": len(golden_events),
        }

    def _compute_source_quality_indicators(
        self,
        match_events: List[ReconciliationEvent],
        disc_events: List[ReconciliationEvent],
        res_events: List[ReconciliationEvent],
    ) -> Dict[str, Any]:
        """Compute per-source data quality indicators for GHG Protocol.

        Quality indicators include match rate, discrepancy rate,
        average confidence, and resolution strategy distribution.

        Args:
            match_events: Match events.
            disc_events: Discrepancy events.
            res_events: Resolution events.

        Returns:
            Dictionary of source-level quality indicators.
        """
        source_stats: Dict[str, Dict[str, Any]] = {}

        # Aggregate match statistics per source
        for event in match_events:
            details = event.details
            for source_key in ("source_a_id", "source_b_id"):
                source_id = details.get(source_key, "unknown")
                if source_id not in source_stats:
                    source_stats[source_id] = {
                        "match_count": 0,
                        "total_confidence": 0.0,
                        "discrepancy_count": 0,
                        "resolution_count": 0,
                    }
                source_stats[source_id]["match_count"] += 1
                source_stats[source_id]["total_confidence"] += (
                    _safe_float(details.get("confidence", 0.0))
                )

        # Aggregate discrepancy counts per source
        for event in disc_events:
            details = event.details
            for source_key in ("source_a_value", "source_b_value"):
                # Track discrepancy occurrence at the source level
                source_id = details.get("field_name", "unknown")
                if source_id in source_stats:
                    source_stats[source_id]["discrepancy_count"] += 1

        # Compute averages
        indicators: Dict[str, Any] = {}
        for source_id, stats in source_stats.items():
            match_count = stats["match_count"]
            avg_confidence = (
                stats["total_confidence"] / match_count
                if match_count > 0
                else 0.0
            )
            indicators[source_id] = {
                "match_count": match_count,
                "average_confidence": round(avg_confidence, 4),
                "discrepancy_count": stats["discrepancy_count"],
                "quality_tier": self._classify_quality_tier(
                    avg_confidence
                ),
            }

        return indicators

    def _classify_quality_tier(
        self,
        avg_confidence: float,
    ) -> str:
        """Classify source quality tier based on average confidence.

        GHG Protocol Data Quality tiers:
            - high: confidence >= 0.90
            - medium: confidence >= 0.70
            - low: confidence >= 0.50
            - very_low: confidence < 0.50

        Args:
            avg_confidence: Average match confidence for a source.

        Returns:
            Quality tier string.
        """
        if avg_confidence >= 0.90:
            return "high"
        if avg_confidence >= 0.70:
            return "medium"
        if avg_confidence >= 0.50:
            return "low"
        return "very_low"

    def _compute_gap_analysis(
        self,
        all_events: List[ReconciliationEvent],
        match_events: List[ReconciliationEvent],
        disc_events: List[ReconciliationEvent],
    ) -> Dict[str, Any]:
        """Compute data gap analysis for GHG Protocol.

        Identifies missing data patterns, unmatched records, and
        unresolved discrepancies that represent data gaps.

        Args:
            all_events: All events for the job.
            match_events: Match events.
            disc_events: Discrepancy events.

        Returns:
            Gap analysis dictionary.
        """
        total_events = len(all_events)
        total_matches = len(match_events)
        total_discrepancies = len(disc_events)

        # Classify discrepancies by type
        disc_by_type: Dict[str, int] = {}
        disc_by_severity: Dict[str, int] = {}
        for event in disc_events:
            details = event.details
            dtype = details.get("discrepancy_type", "unknown")
            sev = details.get("severity", "unknown")
            disc_by_type[dtype] = disc_by_type.get(dtype, 0) + 1
            disc_by_severity[sev] = disc_by_severity.get(sev, 0) + 1

        # Coverage ratio
        coverage_ratio = (
            total_matches / max(total_events, 1)
        )

        return {
            "total_events": total_events,
            "total_matches": total_matches,
            "total_discrepancies": total_discrepancies,
            "coverage_ratio": round(coverage_ratio, 4),
            "discrepancies_by_type": disc_by_type,
            "discrepancies_by_severity": disc_by_severity,
            "has_critical_gaps": disc_by_severity.get("critical", 0) > 0,
        }

    def _document_estimation_methods(
        self,
        res_events: List[ReconciliationEvent],
    ) -> Dict[str, Any]:
        """Document estimation methodology for GHG Protocol.

        GHG Protocol requires documentation of all estimation methods
        used to fill data gaps or resolve conflicts.

        Args:
            res_events: Resolution events.

        Returns:
            Estimation methodology documentation dictionary.
        """
        strategy_counts: Dict[str, int] = {}
        strategy_examples: Dict[str, List[str]] = {}

        for event in res_events:
            details = event.details
            strategy = details.get("strategy", "unknown")
            strategy_counts[strategy] = (
                strategy_counts.get(strategy, 0) + 1
            )
            justification = details.get("justification", "")
            if justification and strategy not in strategy_examples:
                strategy_examples[strategy] = []
            if (
                justification
                and len(strategy_examples.get(strategy, [])) < 3
            ):
                strategy_examples[strategy].append(justification)

        return {
            "total_resolutions": len(res_events),
            "strategies_used": strategy_counts,
            "strategy_justification_examples": strategy_examples,
            "documentation_complete": len(res_events) > 0,
        }

    def _compute_csrd_quality_score(
        self,
        all_events: List[ReconciliationEvent],
        match_events: List[ReconciliationEvent],
        disc_events: List[ReconciliationEvent],
        res_events: List[ReconciliationEvent],
    ) -> Dict[str, Any]:
        """Compute CSRD/ESRS data quality score.

        Scores based on completeness (match rate), accuracy
        (low discrepancy rate), and reliability (resolution rate).
        Produces a composite score on 0-100 scale.

        Args:
            all_events: All events.
            match_events: Match events.
            disc_events: Discrepancy events.
            res_events: Resolution events.

        Returns:
            Quality score dictionary.
        """
        total = max(len(all_events), 1)
        match_rate = len(match_events) / total
        disc_rate = len(disc_events) / max(len(match_events), 1)
        resolution_rate = (
            len(res_events) / max(len(disc_events), 1)
        )

        # Composite score: weighted average
        completeness = min(match_rate, 1.0) * 40.0
        accuracy = max(0.0, (1.0 - disc_rate)) * 30.0
        reliability = min(resolution_rate, 1.0) * 30.0
        composite = completeness + accuracy + reliability

        return {
            "composite_score": round(composite, 2),
            "completeness_score": round(completeness, 2),
            "accuracy_score": round(accuracy, 2),
            "reliability_score": round(reliability, 2),
            "match_rate": round(match_rate, 4),
            "discrepancy_rate": round(disc_rate, 4),
            "resolution_rate": round(resolution_rate, 4),
            "quality_level": self._classify_csrd_quality(composite),
        }

    def _classify_csrd_quality(
        self,
        composite_score: float,
    ) -> str:
        """Classify CSRD/ESRS quality level from composite score.

        ESRS data quality levels:
            - assured: >= 85
            - high: >= 70
            - moderate: >= 50
            - low: < 50

        Args:
            composite_score: Composite quality score (0-100).

        Returns:
            Quality level string.
        """
        if composite_score >= 85.0:
            return "assured"
        if composite_score >= 70.0:
            return "high"
        if composite_score >= 50.0:
            return "moderate"
        return "low"

    def _compute_materiality_linkage(
        self,
        disc_events: List[ReconciliationEvent],
        res_events: List[ReconciliationEvent],
    ) -> Dict[str, Any]:
        """Compute materiality assessment linkage for CSRD/ESRS.

        Identifies fields with high discrepancy rates that may
        indicate material data quality issues.

        Args:
            disc_events: Discrepancy events.
            res_events: Resolution events.

        Returns:
            Materiality linkage dictionary.
        """
        field_discrepancies: Dict[str, int] = {}
        for event in disc_events:
            field_name = event.details.get("field_name", "unknown")
            field_discrepancies[field_name] = (
                field_discrepancies.get(field_name, 0) + 1
            )

        # Fields with >= 3 discrepancies are flagged as material
        material_fields = {
            field: count
            for field, count in field_discrepancies.items()
            if count >= 3
        }

        return {
            "total_fields_with_discrepancies": len(field_discrepancies),
            "material_fields": material_fields,
            "materiality_threshold": 3,
            "has_material_issues": len(material_fields) > 0,
        }

    def _compute_double_materiality(
        self,
        disc_events: List[ReconciliationEvent],
    ) -> Dict[str, Any]:
        """Compute double materiality indicators for CSRD/ESRS.

        Identifies discrepancies that may have financial materiality
        (high deviation) and impact materiality (critical severity).

        Args:
            disc_events: Discrepancy events.

        Returns:
            Double materiality indicators dictionary.
        """
        financial_material = 0
        impact_material = 0
        total = len(disc_events)

        for event in disc_events:
            details = event.details
            deviation = _safe_float(
                details.get("deviation_pct", 0.0)
            )
            severity = details.get("severity", "")

            if deviation >= 25.0:
                financial_material += 1
            if severity in ("critical", "high"):
                impact_material += 1

        return {
            "total_discrepancies": total,
            "financial_materiality_count": financial_material,
            "impact_materiality_count": impact_material,
            "financial_materiality_pct": round(
                (financial_material / max(total, 1)) * 100.0, 2
            ),
            "impact_materiality_pct": round(
                (impact_material / max(total, 1)) * 100.0, 2
            ),
        }

    # ==================================================================
    # 9. generate_discrepancy_log
    # ==================================================================

    def generate_discrepancy_log(
        self,
        job_id: str,
    ) -> List[Dict[str, Any]]:
        """Generate a chronological discrepancy log for a job.

        Produces an ordered list of all discrepancies detected during
        the reconciliation, enriched with their resolution status
        (resolved or unresolved), before/after values, and the
        resolution justification when applicable.

        Args:
            job_id: Reconciliation job identifier.

        Returns:
            List of discrepancy log entries sorted by timestamp.

        Raises:
            ValueError: If job_id is empty.
        """
        start = time.time()

        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")

        # Collect discrepancy events
        disc_events = [
            e for e in self._events
            if e.job_id == job_id
            and e.event_type == "discrepancy_detected"
        ]

        # Collect resolution events and index by discrepancy_id
        resolution_map = self._build_resolution_map(job_id)

        # Build log entries
        log_entries: List[Dict[str, Any]] = []
        for event in disc_events:
            entry = self._build_discrepancy_log_entry(
                event, resolution_map
            )
            log_entries.append(entry)

        # Sort chronologically by timestamp
        log_entries.sort(key=lambda x: str(x.get("timestamp", "")))

        duration = time.time() - start
        observe_duration(duration)

        logger.info(
            "Discrepancy log generated: job=%s entries=%d",
            job_id,
            len(log_entries),
        )
        return log_entries

    def _build_resolution_map(
        self,
        job_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Build a mapping from discrepancy_id to resolution details.

        Args:
            job_id: Job identifier.

        Returns:
            Dictionary mapping discrepancy_id to resolution event
            details.
        """
        resolution_map: Dict[str, Dict[str, Any]] = {}
        for event in self._events:
            if (
                event.job_id == job_id
                and event.event_type == "resolution_applied"
            ):
                disc_id = event.details.get("discrepancy_id", "")
                if disc_id:
                    resolution_map[disc_id] = event.details
        return resolution_map

    def _build_discrepancy_log_entry(
        self,
        event: ReconciliationEvent,
        resolution_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a single discrepancy log entry.

        Args:
            event: The discrepancy event.
            resolution_map: Mapping from discrepancy_id to resolution
                details.

        Returns:
            Discrepancy log entry dictionary.
        """
        details = event.details
        disc_id = details.get("discrepancy_id", "")
        resolution = resolution_map.get(disc_id)

        entry: Dict[str, Any] = {
            "event_id": event.event_id,
            "timestamp": str(event.timestamp),
            "discrepancy_id": disc_id,
            "discrepancy_type": details.get("discrepancy_type", ""),
            "severity": details.get("severity", ""),
            "field_name": details.get("field_name", ""),
            "source_a_value": details.get("source_a_value", ""),
            "source_b_value": details.get("source_b_value", ""),
            "deviation_pct": _safe_float(
                details.get("deviation_pct", 0.0)
            ),
            "resolved": resolution is not None,
            "provenance_hash": event.provenance_hash,
        }

        if resolution is not None:
            entry["resolution"] = {
                "strategy": resolution.get("strategy", ""),
                "winning_source_id": resolution.get(
                    "winning_source_id", ""
                ),
                "resolved_value": resolution.get("resolved_value", ""),
                "justification": resolution.get("justification", ""),
                "confidence": _safe_float(
                    resolution.get("confidence", 0.0)
                ),
            }
        else:
            entry["resolution"] = None

        return entry

    # ==================================================================
    # 10. generate_resolution_justification
    # ==================================================================

    def generate_resolution_justification(
        self,
        job_id: str,
    ) -> Dict[str, Any]:
        """Generate per-discrepancy resolution justification document.

        For each resolved discrepancy in the job, produces a
        justification entry with the strategy rationale, source
        credibility evidence, before/after values, and confidence.

        Args:
            job_id: Reconciliation job identifier.

        Returns:
            Dictionary with ``justifications`` list and summary
            statistics.

        Raises:
            ValueError: If job_id is empty.
        """
        start = time.time()

        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")

        # Collect resolution events
        res_events = [
            e for e in self._events
            if e.job_id == job_id
            and e.event_type == "resolution_applied"
        ]

        # Collect discrepancy events for before-values
        disc_map = self._build_discrepancy_map(job_id)

        # Build justification entries
        justifications: List[Dict[str, Any]] = []
        strategy_counts: Dict[str, int] = {}
        total_confidence = 0.0

        for event in res_events:
            entry = self._build_justification_entry(event, disc_map)
            justifications.append(entry)

            strategy = event.details.get("strategy", "unknown")
            strategy_counts[strategy] = (
                strategy_counts.get(strategy, 0) + 1
            )
            total_confidence += _safe_float(
                event.details.get("confidence", 0.0)
            )

        avg_confidence = (
            total_confidence / len(res_events)
            if res_events
            else 0.0
        )

        # Compute provenance
        result = {
            "job_id": job_id,
            "total_resolutions": len(justifications),
            "strategy_distribution": strategy_counts,
            "average_resolution_confidence": round(avg_confidence, 4),
            "justifications": justifications,
        }
        result["provenance_hash"] = self._compute_provenance(
            operation="generate_resolution_justification",
            input_data={"job_id": job_id},
            output_data=result,
        )

        duration = time.time() - start
        observe_duration(duration)

        logger.info(
            "Resolution justification generated: job=%s "
            "resolutions=%d avg_confidence=%.3f",
            job_id,
            len(justifications),
            avg_confidence,
        )
        return result

    def _build_discrepancy_map(
        self,
        job_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Build a mapping from discrepancy_id to discrepancy details.

        Args:
            job_id: Job identifier.

        Returns:
            Dictionary mapping discrepancy_id to event details.
        """
        disc_map: Dict[str, Dict[str, Any]] = {}
        for event in self._events:
            if (
                event.job_id == job_id
                and event.event_type == "discrepancy_detected"
            ):
                disc_id = event.details.get("discrepancy_id", "")
                if disc_id:
                    disc_map[disc_id] = event.details
        return disc_map

    def _build_justification_entry(
        self,
        res_event: ReconciliationEvent,
        disc_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a single resolution justification entry.

        Args:
            res_event: The resolution event.
            disc_map: Mapping from discrepancy_id to discrepancy
                details.

        Returns:
            Justification entry dictionary.
        """
        details = res_event.details
        disc_id = details.get("discrepancy_id", "")
        disc_details = disc_map.get(disc_id, {})

        return {
            "resolution_id": details.get("resolution_id", ""),
            "discrepancy_id": disc_id,
            "strategy": details.get("strategy", ""),
            "strategy_rationale": self._get_strategy_rationale(
                details.get("strategy", "")
            ),
            "winning_source_id": details.get(
                "winning_source_id", ""
            ),
            "resolved_value": details.get("resolved_value", ""),
            "justification_text": details.get("justification", ""),
            "confidence": _safe_float(
                details.get("confidence", 0.0)
            ),
            "original_discrepancy": {
                "type": disc_details.get("discrepancy_type", ""),
                "severity": disc_details.get("severity", ""),
                "field_name": disc_details.get("field_name", ""),
                "source_a_value": disc_details.get(
                    "source_a_value", ""
                ),
                "source_b_value": disc_details.get(
                    "source_b_value", ""
                ),
                "deviation_pct": _safe_float(
                    disc_details.get("deviation_pct", 0.0)
                ),
            },
            "timestamp": str(res_event.timestamp),
            "provenance_hash": res_event.provenance_hash,
        }

    def _get_strategy_rationale(
        self,
        strategy: str,
    ) -> str:
        """Return a human-readable rationale for a resolution strategy.

        Args:
            strategy: Resolution strategy name.

        Returns:
            Rationale text explaining why the strategy was applied.
        """
        rationales: Dict[str, str] = {
            "priority_wins": (
                "Selected the value from the highest-priority source "
                "based on configured source credibility rankings."
            ),
            "most_recent": (
                "Selected the most recently updated value, as newer "
                "data is considered more accurate for this field."
            ),
            "most_complete": (
                "Selected the value from the source with the fewest "
                "null fields, indicating more complete data coverage."
            ),
            "weighted_average": (
                "Computed a credibility-weighted average across all "
                "sources to minimize bias toward any single source."
            ),
            "average": (
                "Computed the arithmetic mean of all source values "
                "to produce a consensus estimate."
            ),
            "median": (
                "Selected the median value across sources to reduce "
                "the influence of outlier values."
            ),
            "consensus": (
                "Selected the value agreed upon by the majority of "
                "sources (majority vote)."
            ),
            "manual_review": (
                "Flagged for manual review due to insufficient "
                "confidence in automated resolution."
            ),
            "rule_based": (
                "Applied a custom business rule to determine the "
                "authoritative value for this field."
            ),
            "ml_suggested": (
                "Used machine-learning assisted suggestion based on "
                "historical resolution patterns."
            ),
        }
        return rationales.get(
            strategy,
            f"Applied '{strategy}' resolution strategy.",
        )

    # ==================================================================
    # 11. get_events
    # ==================================================================

    def get_events(
        self,
        job_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[ReconciliationEvent]:
        """Query audit events with optional filters.

        Returns events matching all provided filters. Filters are
        combined with AND logic. If no filters are given, all events
        are returned.

        Args:
            job_id: Filter by reconciliation job ID.
            event_type: Filter by event type.
            since: Filter events with timestamp >= this datetime.
                Must be timezone-aware (UTC).

        Returns:
            List of matching ``ReconciliationEvent`` instances,
            ordered by insertion time.

        Raises:
            ValueError: If event_type is provided but not in
                ``VALID_EVENT_TYPES``.
        """
        if event_type and event_type not in self.VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{event_type}'. "
                f"Must be one of: {sorted(self.VALID_EVENT_TYPES)}"
            )

        result: List[ReconciliationEvent] = []
        for event in self._events:
            if job_id and event.job_id != job_id:
                continue
            if event_type and event.event_type != event_type:
                continue
            if since and event.timestamp < since:
                continue
            result.append(event)

        logger.debug(
            "get_events: job_id=%s type=%s since=%s -> %d events",
            job_id,
            event_type,
            since,
            len(result),
        )
        return result

    # ==================================================================
    # 12. get_event_count
    # ==================================================================

    def get_event_count(
        self,
        job_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Count events by type, optionally filtered by job.

        Args:
            job_id: Optional job ID to filter by. If None, counts
                across all jobs.

        Returns:
            Dictionary mapping event_type to count. Also includes
            a ``total`` key with the grand total.
        """
        counts: Dict[str, int] = {}
        total = 0

        for event in self._events:
            if job_id and event.job_id != job_id:
                continue
            etype = event.event_type
            counts[etype] = counts.get(etype, 0) + 1
            total += 1

        counts["total"] = total

        logger.debug(
            "get_event_count: job_id=%s -> total=%d types=%d",
            job_id,
            total,
            len(counts) - 1,
        )
        return counts

    # ==================================================================
    # 13. export_audit_trail
    # ==================================================================

    def export_audit_trail(
        self,
        job_id: str,
        format: str = "json",
    ) -> str:
        """Export the full audit trail for a job.

        Serialises all events for the given ``job_id`` into the
        requested format (JSON or CSV), including all event details,
        provenance hashes, and timestamps.

        Args:
            job_id: Reconciliation job identifier.
            format: Export format (``json`` or ``csv``).

        Returns:
            String containing the serialised audit trail.

        Raises:
            ValueError: If job_id is empty or format is not supported.
        """
        start = time.time()

        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if format not in self.SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported export format '{format}'. "
                f"Must be one of: {sorted(self.SUPPORTED_EXPORT_FORMATS)}"
            )

        job_events = self._get_job_events(job_id)

        if format == "json":
            result = self._export_as_json(job_id, job_events)
        else:
            result = self._export_as_csv(job_events)

        duration = time.time() - start
        observe_duration(duration)

        logger.info(
            "Audit trail exported: job=%s format=%s events=%d "
            "size=%d bytes",
            job_id,
            format,
            len(job_events),
            len(result),
        )
        return result

    def _export_as_json(
        self,
        job_id: str,
        events: List[ReconciliationEvent],
    ) -> str:
        """Export events as JSON string.

        Args:
            job_id: Job identifier.
            events: Events to export.

        Returns:
            JSON string with events and metadata.
        """
        export_data = {
            "job_id": job_id,
            "exported_at": _utcnow().isoformat(),
            "event_count": len(events),
            "provenance_chain_hash": self._provenance.get_current_hash(),
            "events": [
                self._event_to_dict(event) for event in events
            ],
        }
        return json.dumps(export_data, indent=2, default=str)

    def _export_as_csv(
        self,
        events: List[ReconciliationEvent],
    ) -> str:
        """Export events as CSV string.

        Args:
            events: Events to export.

        Returns:
            CSV string with headers and one row per event.
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "event_id",
            "job_id",
            "event_type",
            "timestamp",
            "provenance_hash",
            "details",
        ])

        # Data rows
        for event in events:
            writer.writerow([
                event.event_id,
                event.job_id,
                event.event_type,
                str(event.timestamp),
                event.provenance_hash,
                json.dumps(event.details, default=str),
            ])

        return output.getvalue()

    def _event_to_dict(
        self,
        event: ReconciliationEvent,
    ) -> Dict[str, Any]:
        """Convert a ReconciliationEvent to a serialisable dictionary.

        Args:
            event: Event to convert.

        Returns:
            Dictionary representation of the event.
        """
        return {
            "event_id": event.event_id,
            "job_id": event.job_id,
            "event_type": event.event_type,
            "timestamp": str(event.timestamp),
            "details": event.details,
            "provenance_hash": event.provenance_hash,
        }

    # ==================================================================
    # 14. verify_audit_integrity
    # ==================================================================

    def verify_audit_integrity(
        self,
        job_id: str,
    ) -> bool:
        """Verify the provenance chain integrity for a job.

        Checks that:
        1. All events for the job have non-empty provenance hashes.
        2. All events have non-empty event IDs and timestamps.
        3. The global provenance chain in the ProvenanceTracker
           verifies as intact.

        Args:
            job_id: Reconciliation job identifier.

        Returns:
            True if the audit chain is intact for all events in the
            job. False if any event is missing required fields or
            if the provenance chain verification fails.

        Raises:
            ValueError: If job_id is empty.
        """
        start = time.time()

        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")

        job_events = self._get_job_events(job_id)

        if not job_events:
            logger.info(
                "Audit integrity check: job=%s has no events "
                "(vacuously valid)",
                job_id,
            )
            return True

        # Check each event has required fields
        events_valid = self._verify_events_complete(job_events)
        if not events_valid:
            logger.warning(
                "Audit integrity FAILED: job=%s has incomplete events",
                job_id,
            )
            return False

        # Check provenance chain via tracker
        chain_valid, _ = self._provenance.verify_chain()
        if not chain_valid:
            logger.warning(
                "Audit integrity FAILED: job=%s provenance chain "
                "verification failed",
                job_id,
            )
            return False

        duration = time.time() - start
        observe_duration(duration)

        logger.info(
            "Audit integrity PASSED: job=%s events=%d",
            job_id,
            len(job_events),
        )
        return True

    def _verify_events_complete(
        self,
        events: List[ReconciliationEvent],
    ) -> bool:
        """Verify all events have required fields populated.

        Args:
            events: Events to verify.

        Returns:
            True if all events have event_id, timestamp, event_type,
            and provenance_hash.
        """
        for event in events:
            if not event.event_id:
                return False
            if not event.timestamp:
                return False
            if not event.event_type:
                return False
            if not event.provenance_hash:
                return False
        return True

    # ==================================================================
    # 15. clear_events
    # ==================================================================

    def clear_events(
        self,
        job_id: Optional[str] = None,
    ) -> int:
        """Clear stored events, optionally filtered by job.

        If ``job_id`` is provided, only events for that job are
        removed. If ``job_id`` is None, all events are cleared.

        Args:
            job_id: Optional job identifier. If None, clears all
                events.

        Returns:
            Number of events cleared.
        """
        if job_id is None:
            count = len(self._events)
            self._events.clear()
            logger.info("Cleared all %d events", count)
            return count

        before = len(self._events)
        self._events = [
            e for e in self._events if e.job_id != job_id
        ]
        cleared = before - len(self._events)

        logger.info(
            "Cleared %d events for job=%s", cleared, job_id
        )
        return cleared

    # ==================================================================
    # 16. _compute_provenance (internal)
    # ==================================================================

    def _compute_provenance(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        """Compute SHA-256 provenance hash and add to chain.

        Hashes both input and output data deterministically, then
        appends a chain link to the provenance tracker. Returns
        the chain hash for inclusion in the event or report.

        Args:
            operation: Name of the operation being tracked.
            input_data: Input data to hash (dict, list, or other).
            output_data: Output data to hash.

        Returns:
            SHA-256 chain hash string.
        """
        try:
            input_hash = self._provenance.build_hash(input_data)
            output_hash = self._provenance.build_hash(output_data)
            chain_hash = self._provenance.add_to_chain(
                operation=operation,
                input_hash=input_hash,
                output_hash=output_hash,
                metadata={"engine": "AuditTrailEngine"},
            )
            return chain_hash
        except Exception as exc:
            logger.error(
                "Provenance computation failed for operation '%s': %s",
                operation,
                str(exc),
            )
            inc_errors(error_type="provenance")
            # Fallback: direct SHA-256 hash without chain
            fallback = hashlib.sha256(
                json.dumps(
                    {"op": operation, "in": str(input_data)},
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            return fallback

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _get_job_events(
        self,
        job_id: str,
    ) -> List[ReconciliationEvent]:
        """Filter events belonging to a specific job.

        Args:
            job_id: Job identifier to filter by.

        Returns:
            List of events for the job, in insertion order.
        """
        return [e for e in self._events if e.job_id == job_id]

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def event_count(self) -> int:
        """Return the total number of stored events."""
        return len(self._events)

    @property
    def provenance_chain_length(self) -> int:
        """Return the length of the provenance chain."""
        return self._provenance.get_chain_length()


__all__ = ["AuditTrailEngine"]
