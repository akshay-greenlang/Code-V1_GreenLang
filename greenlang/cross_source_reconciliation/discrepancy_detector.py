# -*- coding: utf-8 -*-
"""
Discrepancy Detector Engine - AGENT-DATA-015 Cross-Source Reconciliation

Pure-Python engine for detecting, classifying, scoring, and prioritising
discrepancies that arise when field-level comparison results indicate
mismatches between two data sources.  The engine converts raw comparison
outcomes into actionable ``Discrepancy`` records, each enriched with a
type classification, severity grade, impact score, and human-readable
description.

Beyond point detection the engine surfaces higher-order patterns
(systematic bias, field hotspots, source reliability issues, temporal
clustering) and produces a ``DiscrepancySummary`` for dashboard
consumption.

Zero-Hallucination: All calculations use deterministic Python arithmetic.
No LLM calls for numeric computations.

Engine 4 of 7 in the Cross-Source Reconciliation pipeline.

Example:
    >>> from greenlang.cross_source_reconciliation.discrepancy_detector import (
    ...     DiscrepancyDetectorEngine,
    ... )
    >>> engine = DiscrepancyDetectorEngine()
    >>> discrepancies = engine.detect_discrepancies(
    ...     comparisons=comparisons,
    ...     match_id="m-001",
    ...     source_a_id="erp",
    ...     source_b_id="invoice",
    ... )
    >>> summary = engine.summarize(discrepancies)
    >>> print(summary.total, summary.critical_count)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from greenlang.cross_source_reconciliation.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation.metrics import (
        inc_discrepancies_detected,
        observe_duration,
    )

    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

    def inc_discrepancies_detected(count: int = 1) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    def observe_duration(operation: str, duration: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    logger.info(
        "cross_source_reconciliation.metrics not available; "
        "discrepancy detector metrics disabled"
    )


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ComparisonResult(str, Enum):
    """Outcome of a single field-level comparison.

    MATCH: Values are identical after normalisation.
    WITHIN_TOLERANCE: Values differ but within acceptable tolerance.
    MISMATCH: Values differ beyond tolerance.
    MISSING_LEFT: Value missing in source A.
    MISSING_RIGHT: Value missing in source B.
    NOT_COMPARABLE: Fields cannot be meaningfully compared.
    """

    MATCH = "match"
    WITHIN_TOLERANCE = "within_tolerance"
    MISMATCH = "mismatch"
    MISSING_LEFT = "missing_left"
    MISSING_RIGHT = "missing_right"
    NOT_COMPARABLE = "not_comparable"


class FieldType(str, Enum):
    """Data type classification for a comparison field.

    Determines which discrepancy classification rules apply.
    """

    STRING = "string"
    NUMERIC = "numeric"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    CURRENCY = "currency"
    UNIT = "unit"
    IDENTIFIER = "identifier"


class DiscrepancyType(str, Enum):
    """Classification of a discrepancy by its root-cause category.

    VALUE_MISMATCH: Core numeric or string value differs.
    MISSING_IN_SOURCE: Value present in one source, absent in other.
    TIMING_DIFFERENCE: Date/timestamp fields disagree.
    UNIT_DIFFERENCE: Values match numerically but units differ.
    AGGREGATION_MISMATCH: Granularity mismatch (e.g. monthly vs daily).
    FORMAT_DIFFERENCE: Same logical value in different representations.
    ROUNDING_DIFFERENCE: Numeric values differ by rounding only.
    CLASSIFICATION_MISMATCH: Categorical codes differ.
    """

    VALUE_MISMATCH = "value_mismatch"
    MISSING_IN_SOURCE = "missing_in_source"
    TIMING_DIFFERENCE = "timing_difference"
    UNIT_DIFFERENCE = "unit_difference"
    AGGREGATION_MISMATCH = "aggregation_mismatch"
    FORMAT_DIFFERENCE = "format_difference"
    ROUNDING_DIFFERENCE = "rounding_difference"
    CLASSIFICATION_MISMATCH = "classification_mismatch"


class DiscrepancySeverity(str, Enum):
    """Severity grade for a detected discrepancy.

    CRITICAL: Deviation >= critical threshold; immediate action required.
    HIGH: Deviation >= high threshold; review urgently.
    MEDIUM: Deviation >= medium threshold; investigate.
    LOW: Deviation > 0 but below medium threshold; informational.
    INFO: Zero deviation or cosmetic difference.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ---------------------------------------------------------------------------
# Severity ordering for comparison and sorting
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: Dict[DiscrepancySeverity, int] = {
    DiscrepancySeverity.CRITICAL: 0,
    DiscrepancySeverity.HIGH: 1,
    DiscrepancySeverity.MEDIUM: 2,
    DiscrepancySeverity.LOW: 3,
    DiscrepancySeverity.INFO: 4,
}

_SEVERITY_WEIGHT: Dict[DiscrepancySeverity, float] = {
    DiscrepancySeverity.CRITICAL: 1.0,
    DiscrepancySeverity.HIGH: 0.75,
    DiscrepancySeverity.MEDIUM: 0.5,
    DiscrepancySeverity.LOW: 0.25,
    DiscrepancySeverity.INFO: 0.1,
}


# ---------------------------------------------------------------------------
# Data models (dataclasses -- pure Python, no Pydantic dependency)
# ---------------------------------------------------------------------------


@dataclass
class FieldComparison:
    """Result of comparing a single field between two source records.

    Attributes:
        field_name: Name of the compared field.
        field_type: Data type of the field.
        result: Comparison outcome (match, mismatch, missing_left, ...).
        value_a: Value from source A (may be None for missing_left).
        value_b: Value from source B (may be None for missing_right).
        deviation_pct: Percentage deviation for numeric fields (0.0 if
            non-numeric or identical).
        tolerance_used: Tolerance threshold applied during comparison.
        unit_a: Optional unit string from source A.
        unit_b: Optional unit string from source B.
        confidence: Confidence score of the comparison (0.0-1.0).
        metadata: Additional comparison metadata.
    """

    field_name: str
    field_type: FieldType = FieldType.STRING
    result: ComparisonResult = ComparisonResult.MATCH
    value_a: Any = None
    value_b: Any = None
    deviation_pct: float = 0.0
    tolerance_used: float = 0.0
    unit_a: Optional[str] = None
    unit_b: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Discrepancy:
    """A detected discrepancy between two source records.

    Attributes:
        discrepancy_id: Unique UUID for the discrepancy.
        match_id: Identifier of the record-match this belongs to.
        source_a_id: Identifier of the first data source.
        source_b_id: Identifier of the second data source.
        field_name: Field in which the discrepancy was detected.
        field_type: Data type of the field.
        discrepancy_type: Classification of the discrepancy.
        severity: Severity grade.
        value_a: Value from source A.
        value_b: Value from source B.
        deviation_pct: Percentage deviation (absolute).
        signed_deviation_pct: Signed percentage deviation (A - B) / B.
        impact_score: Combined severity + deviation impact (0.0-1.0).
        description: Human-readable description of the discrepancy.
        confidence: Confidence that this is a genuine discrepancy.
        requires_manual_review: Whether manual review is recommended.
        resolution_status: Current resolution status.
        entity_id: Optional higher-level entity grouping.
        period: Optional time period the discrepancy relates to.
        provenance_hash: SHA-256 provenance hash.
        detected_at: ISO timestamp of detection.
        metadata: Additional context and details.
    """

    discrepancy_id: str = ""
    match_id: str = ""
    source_a_id: str = ""
    source_b_id: str = ""
    field_name: str = ""
    field_type: FieldType = FieldType.STRING
    discrepancy_type: DiscrepancyType = DiscrepancyType.VALUE_MISMATCH
    severity: DiscrepancySeverity = DiscrepancySeverity.INFO
    value_a: Any = None
    value_b: Any = None
    deviation_pct: float = 0.0
    signed_deviation_pct: float = 0.0
    impact_score: float = 0.0
    description: str = ""
    confidence: float = 1.0
    requires_manual_review: bool = False
    resolution_status: str = "pending"
    entity_id: str = ""
    period: str = ""
    provenance_hash: str = ""
    detected_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscrepancySummary:
    """Aggregated summary of discrepancies for a reconciliation run.

    Attributes:
        total: Total number of discrepancies detected.
        by_type: Count per discrepancy type.
        by_severity: Count per severity grade.
        by_source: Count per source identifier.
        by_field: Count per field name.
        critical_count: Number of CRITICAL severity discrepancies.
        high_count: Number of HIGH severity discrepancies.
        pending_review_count: Number requiring manual review.
        mean_deviation_pct: Average absolute deviation across all.
        max_deviation_pct: Maximum absolute deviation observed.
        mean_impact_score: Average impact score.
        top_fields: Top 10 fields by discrepancy count.
        provenance_hash: SHA-256 hash of the summary.
        generated_at: ISO timestamp of summary generation.
    """

    total: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    by_field: Dict[str, int] = field(default_factory=dict)
    critical_count: int = 0
    high_count: int = 0
    pending_review_count: int = 0
    mean_deviation_pct: float = 0.0
    max_deviation_pct: float = 0.0
    mean_impact_score: float = 0.0
    top_fields: List[Tuple[str, int]] = field(default_factory=list)
    provenance_hash: str = ""
    generated_at: str = ""


# ---------------------------------------------------------------------------
# Helper: safe UTC now
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _safe_abs(value: float) -> float:
    """Return absolute value, handling NaN/Inf gracefully.

    Args:
        value: Numeric value.

    Returns:
        Absolute value, or 0.0 if NaN/Inf.
    """
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return abs(value)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide, returning 0.0 on zero-denominator or NaN.

    Args:
        numerator: Dividend.
        denominator: Divisor.

    Returns:
        Result of division, or 0.0 if division is undefined.
    """
    if denominator == 0.0 or math.isnan(denominator) or math.isinf(denominator):
        return 0.0
    result = numerator / denominator
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


# ---------------------------------------------------------------------------
# DiscrepancyDetectorEngine
# ---------------------------------------------------------------------------


class DiscrepancyDetectorEngine:
    """Detects, classifies, scores, and prioritises discrepancies.

    Processes field-level comparison results from Engine 3
    (FieldComparatorEngine) and produces enriched ``Discrepancy``
    records.  Each discrepancy is classified by type (value mismatch,
    missing, timing, unit, aggregation, format, rounding,
    classification) and graded by severity (CRITICAL through INFO).

    Additionally exposes pattern detection (systematic bias, field
    hotspots, source reliability) and aggregation helpers for
    dashboard integration.

    Attributes:
        _provenance: SHA-256 provenance tracker for audit trails.
        _manual_review_confidence_threshold: Confidence below this
            triggers manual review flagging.

    Example:
        >>> engine = DiscrepancyDetectorEngine()
        >>> comps = [
        ...     FieldComparison(
        ...         field_name="emissions_total",
        ...         field_type=FieldType.NUMERIC,
        ...         result=ComparisonResult.MISMATCH,
        ...         value_a=1234.56,
        ...         value_b=1289.12,
        ...         deviation_pct=4.4,
        ...     ),
        ... ]
        >>> discs = engine.detect_discrepancies(
        ...     comps, "m-001", "erp", "invoice",
        ... )
        >>> assert len(discs) == 1
        >>> assert discs[0].severity == DiscrepancySeverity.LOW
    """

    # Default severity thresholds (percentage deviation)
    DEFAULT_CRITICAL_PCT: float = 50.0
    DEFAULT_HIGH_PCT: float = 25.0
    DEFAULT_MEDIUM_PCT: float = 10.0

    # Default confidence threshold for manual review flagging
    DEFAULT_MANUAL_REVIEW_CONFIDENCE: float = 0.7

    def __init__(
        self,
        critical_pct: float = DEFAULT_CRITICAL_PCT,
        high_pct: float = DEFAULT_HIGH_PCT,
        medium_pct: float = DEFAULT_MEDIUM_PCT,
        manual_review_confidence: float = DEFAULT_MANUAL_REVIEW_CONFIDENCE,
    ) -> None:
        """Initialize DiscrepancyDetectorEngine.

        Args:
            critical_pct: Deviation percentage threshold for CRITICAL.
            high_pct: Deviation percentage threshold for HIGH.
            medium_pct: Deviation percentage threshold for MEDIUM.
            manual_review_confidence: Confidence threshold below which
                CRITICAL/HIGH discrepancies are flagged for manual review.
        """
        self._provenance = ProvenanceTracker()
        self._critical_pct = critical_pct
        self._high_pct = high_pct
        self._medium_pct = medium_pct
        self._manual_review_confidence_threshold = manual_review_confidence
        logger.info(
            "DiscrepancyDetectorEngine initialized "
            "(critical=%.1f%%, high=%.1f%%, medium=%.1f%%)",
            self._critical_pct,
            self._high_pct,
            self._medium_pct,
        )

    # ------------------------------------------------------------------
    # 1. Primary detection
    # ------------------------------------------------------------------

    def detect_discrepancies(
        self,
        comparisons: List[FieldComparison],
        match_id: str,
        source_a_id: str,
        source_b_id: str,
        entity_id: str = "",
        period: str = "",
    ) -> List[Discrepancy]:
        """Detect discrepancies from field-level comparison results.

        Iterates through comparisons and generates a ``Discrepancy``
        for every result that is neither ``MATCH`` nor
        ``WITHIN_TOLERANCE``.  Each discrepancy is classified by type
        and severity, scored for impact, and assigned a unique UUID.

        Args:
            comparisons: Field comparison results from Engine 3.
            match_id: Identifier of the record-match being analysed.
            source_a_id: Identifier of source A.
            source_b_id: Identifier of source B.
            entity_id: Optional entity-level grouping identifier.
            period: Optional time period string.

        Returns:
            List of detected ``Discrepancy`` records, sorted by
            severity (CRITICAL first).

        Example:
            >>> engine = DiscrepancyDetectorEngine()
            >>> comps = [
            ...     FieldComparison("co2", FieldType.NUMERIC,
            ...         ComparisonResult.MISMATCH, 100.0, 200.0, 66.7),
            ... ]
            >>> discs = engine.detect_discrepancies(
            ...     comps, "m-001", "erp", "inv",
            ... )
            >>> assert discs[0].severity == DiscrepancySeverity.CRITICAL
        """
        start = time.time()
        detected_at = _utcnow().isoformat()
        discrepancies: List[Discrepancy] = []

        for comp in comparisons:
            if comp.result in (
                ComparisonResult.MATCH,
                ComparisonResult.WITHIN_TOLERANCE,
            ):
                continue

            # Skip non-comparable fields
            if comp.result == ComparisonResult.NOT_COMPARABLE:
                logger.debug(
                    "Skipping non-comparable field: %s", comp.field_name,
                )
                continue

            disc_type = self.classify_type(comp)
            deviation = _safe_abs(comp.deviation_pct)
            severity = self.classify_severity(deviation)
            signed_dev = self._compute_signed_deviation(comp)
            impact = self._compute_impact_from_severity_and_deviation(
                severity, deviation,
            )
            description = self._generate_description(comp, disc_type)
            requires_review = self._should_flag_manual_review(
                severity, comp.confidence,
            )

            disc = Discrepancy(
                discrepancy_id=str(uuid4()),
                match_id=match_id,
                source_a_id=source_a_id,
                source_b_id=source_b_id,
                field_name=comp.field_name,
                field_type=comp.field_type,
                discrepancy_type=disc_type,
                severity=severity,
                value_a=comp.value_a,
                value_b=comp.value_b,
                deviation_pct=deviation,
                signed_deviation_pct=signed_dev,
                impact_score=impact,
                description=description,
                confidence=comp.confidence,
                requires_manual_review=requires_review,
                resolution_status="pending",
                entity_id=entity_id,
                period=period,
                detected_at=detected_at,
                metadata={
                    "tolerance_used": comp.tolerance_used,
                    "unit_a": comp.unit_a,
                    "unit_b": comp.unit_b,
                    "comparison_result": comp.result.value,
                },
            )

            # Provenance for individual discrepancy
            disc.provenance_hash = self._compute_provenance(
                "detect_discrepancy",
                {
                    "match_id": match_id,
                    "field_name": comp.field_name,
                    "result": comp.result.value,
                    "deviation_pct": deviation,
                },
                {
                    "discrepancy_id": disc.discrepancy_id,
                    "type": disc_type.value,
                    "severity": severity.value,
                    "impact_score": impact,
                },
            )

            discrepancies.append(disc)

        # Sort by severity (CRITICAL first), then deviation descending
        discrepancies.sort(
            key=lambda d: (
                _SEVERITY_ORDER.get(d.severity, 99),
                -d.deviation_pct,
            ),
        )

        elapsed = time.time() - start
        processing_time_ms = elapsed * 1000.0

        # Metrics
        inc_discrepancies_detected(len(discrepancies))
        observe_duration("detect_discrepancies", elapsed)

        logger.info(
            "Discrepancy detection complete: match_id=%s, "
            "comparisons=%d, discrepancies=%d, %.3fms",
            match_id,
            len(comparisons),
            len(discrepancies),
            processing_time_ms,
        )
        return discrepancies

    # ------------------------------------------------------------------
    # 2. Type classification
    # ------------------------------------------------------------------

    def classify_type(
        self,
        comparison: FieldComparison,
    ) -> DiscrepancyType:
        """Classify a field comparison into a discrepancy type.

        Uses the comparison result, field type, and metadata to
        determine the most appropriate discrepancy classification.

        Args:
            comparison: The field comparison to classify.

        Returns:
            Classified ``DiscrepancyType``.

        Example:
            >>> engine = DiscrepancyDetectorEngine()
            >>> comp = FieldComparison(
            ...     "date_col", FieldType.DATE,
            ...     ComparisonResult.MISMATCH,
            ... )
            >>> engine.classify_type(comp)
            <DiscrepancyType.TIMING_DIFFERENCE: 'timing_difference'>
        """
        result = comparison.result

        # Missing value cases
        if result == ComparisonResult.MISSING_LEFT:
            return DiscrepancyType.MISSING_IN_SOURCE
        if result == ComparisonResult.MISSING_RIGHT:
            return DiscrepancyType.MISSING_IN_SOURCE

        # Mismatch sub-classification by field type and context
        if result == ComparisonResult.MISMATCH:
            return self._classify_mismatch(comparison)

        # Fallback
        return DiscrepancyType.VALUE_MISMATCH

    def _classify_mismatch(
        self,
        comparison: FieldComparison,
    ) -> DiscrepancyType:
        """Sub-classify a MISMATCH result using field type and metadata.

        Checks for unit differences, aggregation mismatches, format
        differences, rounding differences, and date/classification
        mismatches.

        Args:
            comparison: The mismatched field comparison.

        Returns:
            Specific ``DiscrepancyType`` for the mismatch.
        """
        ft = comparison.field_type
        meta = comparison.metadata

        # Date fields → timing difference
        if ft == FieldType.DATE:
            return DiscrepancyType.TIMING_DIFFERENCE

        # Check for unit difference (units provided and different)
        if comparison.unit_a and comparison.unit_b:
            if comparison.unit_a != comparison.unit_b:
                return DiscrepancyType.UNIT_DIFFERENCE

        # Check for aggregation mismatch (metadata hint)
        if meta.get("aggregation_mismatch", False):
            return DiscrepancyType.AGGREGATION_MISMATCH

        # Check for format difference (metadata hint)
        if meta.get("format_difference", False):
            return DiscrepancyType.FORMAT_DIFFERENCE

        # Check for rounding difference on numeric fields
        if ft in (FieldType.NUMERIC, FieldType.CURRENCY):
            if self._is_rounding_difference(comparison):
                return DiscrepancyType.ROUNDING_DIFFERENCE
            return DiscrepancyType.VALUE_MISMATCH

        # Categorical / boolean → classification mismatch
        if ft in (FieldType.CATEGORICAL, FieldType.BOOLEAN):
            return DiscrepancyType.CLASSIFICATION_MISMATCH

        # Default for string, identifier, unit fields
        return DiscrepancyType.VALUE_MISMATCH

    def _is_rounding_difference(
        self,
        comparison: FieldComparison,
    ) -> bool:
        """Determine whether a numeric mismatch is due to rounding only.

        Compares values rounded to various decimal places (0-4) and
        returns True if they become equal at any rounding precision.

        Args:
            comparison: The numeric field comparison.

        Returns:
            True if the mismatch is attributable to rounding.
        """
        val_a = comparison.value_a
        val_b = comparison.value_b

        if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
            return False

        if math.isnan(val_a) or math.isnan(val_b):
            return False
        if math.isinf(val_a) or math.isinf(val_b):
            return False

        for decimals in range(5):
            if round(float(val_a), decimals) == round(float(val_b), decimals):
                return True

        return False

    # ------------------------------------------------------------------
    # 3. Severity classification
    # ------------------------------------------------------------------

    def classify_severity(
        self,
        deviation_pct: float,
        critical_pct: Optional[float] = None,
        high_pct: Optional[float] = None,
        medium_pct: Optional[float] = None,
    ) -> DiscrepancySeverity:
        """Classify deviation percentage into a severity grade.

        Uses threshold ranges to map absolute deviation percentage to
        one of five severity grades.

        Args:
            deviation_pct: Absolute percentage deviation (>= 0.0).
            critical_pct: Override critical threshold (default from init).
            high_pct: Override high threshold (default from init).
            medium_pct: Override medium threshold (default from init).

        Returns:
            Classified ``DiscrepancySeverity``.

        Example:
            >>> engine = DiscrepancyDetectorEngine()
            >>> engine.classify_severity(55.0)
            <DiscrepancySeverity.CRITICAL: 'critical'>
            >>> engine.classify_severity(12.0)
            <DiscrepancySeverity.MEDIUM: 'medium'>
            >>> engine.classify_severity(0.0)
            <DiscrepancySeverity.INFO: 'info'>
        """
        c_pct = critical_pct if critical_pct is not None else self._critical_pct
        h_pct = high_pct if high_pct is not None else self._high_pct
        m_pct = medium_pct if medium_pct is not None else self._medium_pct

        dev = _safe_abs(deviation_pct)

        if dev >= c_pct:
            return DiscrepancySeverity.CRITICAL
        if dev >= h_pct:
            return DiscrepancySeverity.HIGH
        if dev >= m_pct:
            return DiscrepancySeverity.MEDIUM
        if dev > 0.0:
            return DiscrepancySeverity.LOW
        return DiscrepancySeverity.INFO

    # ------------------------------------------------------------------
    # 4. Pattern detection (higher-order)
    # ------------------------------------------------------------------

    def detect_patterns(
        self,
        discrepancies: List[Discrepancy],
    ) -> Dict[str, Any]:
        """Detect higher-order patterns across a set of discrepancies.

        Analyses the discrepancy population for systematic bias, type
        clustering, field concentration, source correlation, and
        temporal clustering.

        Args:
            discrepancies: List of detected discrepancies.

        Returns:
            Dictionary with pattern analysis results::

                {
                    "systematic_bias": {...},
                    "type_distribution": {...},
                    "field_hotspots": [...],
                    "source_correlation": {...},
                    "temporal_patterns": {...},
                    "total_analysed": int,
                }

        Example:
            >>> patterns = engine.detect_patterns(discrepancies)
            >>> print(patterns["field_hotspots"][0])
        """
        start = time.time()

        if not discrepancies:
            return {
                "systematic_bias": {},
                "type_distribution": {},
                "field_hotspots": [],
                "source_correlation": {},
                "temporal_patterns": {},
                "total_analysed": 0,
            }

        systematic_bias = self.detect_systematic_bias(discrepancies)
        type_distribution = self._compute_type_distribution(discrepancies)
        field_hotspots = self.detect_field_hotspots(discrepancies)
        source_correlation = self._compute_source_correlation(discrepancies)
        temporal_patterns = self._detect_temporal_patterns(discrepancies)

        elapsed = time.time() - start
        observe_duration("detect_patterns", elapsed)

        result: Dict[str, Any] = {
            "systematic_bias": systematic_bias,
            "type_distribution": type_distribution,
            "field_hotspots": field_hotspots,
            "source_correlation": source_correlation,
            "temporal_patterns": temporal_patterns,
            "total_analysed": len(discrepancies),
        }

        # Provenance
        self._compute_provenance(
            "detect_patterns",
            {"total_discrepancies": len(discrepancies)},
            {
                "bias_keys": list(systematic_bias.keys()),
                "hotspot_count": len(field_hotspots),
            },
        )

        logger.info(
            "Pattern detection complete: discrepancies=%d, "
            "bias_pairs=%d, hotspots=%d, %.3fms",
            len(discrepancies),
            len(systematic_bias),
            len(field_hotspots),
            elapsed * 1000.0,
        )
        return result

    # ------------------------------------------------------------------
    # 5. Systematic bias detection
    # ------------------------------------------------------------------

    def detect_systematic_bias(
        self,
        discrepancies: List[Discrepancy],
    ) -> Dict[str, float]:
        """Detect systematic bias per source pair.

        For numeric discrepancies, computes the mean signed deviation
        per source pair key.  A positive value indicates source A is
        consistently higher; negative indicates source B is higher.

        Args:
            discrepancies: List of discrepancies to analyse.

        Returns:
            Dictionary mapping source-pair key (``"src_a|src_b"``) to
            mean signed deviation percentage.

        Example:
            >>> bias = engine.detect_systematic_bias(discrepancies)
            >>> # positive = source A consistently higher
            >>> print(bias.get("erp|invoice", 0.0))
        """
        pair_deviations: Dict[str, List[float]] = defaultdict(list)

        for disc in discrepancies:
            if disc.field_type not in (
                FieldType.NUMERIC,
                FieldType.CURRENCY,
            ):
                continue
            if disc.signed_deviation_pct == 0.0:
                continue

            pair_key = f"{disc.source_a_id}|{disc.source_b_id}"
            pair_deviations[pair_key].append(disc.signed_deviation_pct)

        bias: Dict[str, float] = {}
        for pair_key, deviations in pair_deviations.items():
            if deviations:
                mean_dev = sum(deviations) / len(deviations)
                bias[pair_key] = round(mean_dev, 4)

        logger.debug(
            "Systematic bias analysis: %d source pairs evaluated",
            len(bias),
        )
        return bias

    # ------------------------------------------------------------------
    # 6. Field hotspot detection
    # ------------------------------------------------------------------

    def detect_field_hotspots(
        self,
        discrepancies: List[Discrepancy],
    ) -> List[Tuple[str, int, float]]:
        """Identify fields with the highest discrepancy concentration.

        Counts discrepancies per field and computes the average
        severity weight for each field.

        Args:
            discrepancies: List of discrepancies to analyse.

        Returns:
            List of ``(field_name, count, avg_severity_weight)``
            tuples sorted by count descending.

        Example:
            >>> hotspots = engine.detect_field_hotspots(discrepancies)
            >>> top_field, count, avg_sev = hotspots[0]
        """
        field_counts: Dict[str, int] = Counter()
        field_severity_sum: Dict[str, float] = defaultdict(float)

        for disc in discrepancies:
            field_counts[disc.field_name] += 1
            weight = _SEVERITY_WEIGHT.get(disc.severity, 0.1)
            field_severity_sum[disc.field_name] += weight

        hotspots: List[Tuple[str, int, float]] = []
        for fname, count in field_counts.items():
            avg_weight = round(
                _safe_divide(field_severity_sum[fname], count), 4,
            )
            hotspots.append((fname, count, avg_weight))

        hotspots.sort(key=lambda x: (-x[1], -x[2]))

        logger.debug(
            "Field hotspot analysis: %d fields with discrepancies",
            len(hotspots),
        )
        return hotspots

    # ------------------------------------------------------------------
    # 7. Source reliability detection
    # ------------------------------------------------------------------

    def detect_source_reliability_issues(
        self,
        discrepancies: List[Discrepancy],
        source_ids: List[str],
        total_comparisons: int = 0,
    ) -> Dict[str, float]:
        """Compute per-source discrepancy rates as reliability proxies.

        For each source, counts the number of discrepancies involving
        that source and divides by total comparisons to produce a
        discrepancy rate.  Higher rate implies lower reliability.

        Args:
            discrepancies: List of detected discrepancies.
            source_ids: List of all source identifiers.
            total_comparisons: Total number of field comparisons
                performed.  If zero, uses the discrepancy count as a
                denominator lower-bound.

        Returns:
            Dictionary mapping source ID to discrepancy rate (0.0-1.0).

        Example:
            >>> rates = engine.detect_source_reliability_issues(
            ...     discrepancies, ["erp", "invoice"], 200,
            ... )
            >>> print(rates)
            {'erp': 0.15, 'invoice': 0.25}
        """
        source_disc_count: Dict[str, int] = {sid: 0 for sid in source_ids}

        for disc in discrepancies:
            if disc.source_a_id in source_disc_count:
                source_disc_count[disc.source_a_id] += 1
            if disc.source_b_id in source_disc_count:
                source_disc_count[disc.source_b_id] += 1

        denominator = max(total_comparisons, len(discrepancies), 1)

        rates: Dict[str, float] = {}
        for sid in source_ids:
            count = source_disc_count.get(sid, 0)
            rates[sid] = round(_safe_divide(count, denominator), 4)

        logger.debug(
            "Source reliability analysis: %d sources, denominator=%d",
            len(source_ids),
            denominator,
        )
        return rates

    # ------------------------------------------------------------------
    # 8. Filtering
    # ------------------------------------------------------------------

    def filter_discrepancies(
        self,
        discrepancies: List[Discrepancy],
        min_severity: Optional[DiscrepancySeverity] = None,
        discrepancy_type: Optional[DiscrepancyType] = None,
        field_name: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[Discrepancy]:
        """Filter discrepancies by severity, type, field, or source.

        All filter parameters are optional; when omitted that
        dimension is not filtered.

        Args:
            discrepancies: Discrepancies to filter.
            min_severity: Minimum severity (CRITICAL is highest).
            discrepancy_type: Keep only this discrepancy type.
            field_name: Keep only discrepancies for this field.
            source_id: Keep only discrepancies involving this source.

        Returns:
            Filtered list of discrepancies preserving original order.

        Example:
            >>> critical = engine.filter_discrepancies(
            ...     discrepancies,
            ...     min_severity=DiscrepancySeverity.HIGH,
            ... )
        """
        result: List[Discrepancy] = []
        min_sev_order = (
            _SEVERITY_ORDER.get(min_severity, 99)
            if min_severity is not None
            else 99
        )

        for disc in discrepancies:
            # Severity filter
            if min_severity is not None:
                disc_order = _SEVERITY_ORDER.get(disc.severity, 99)
                if disc_order > min_sev_order:
                    continue

            # Type filter
            if discrepancy_type is not None:
                if disc.discrepancy_type != discrepancy_type:
                    continue

            # Field filter
            if field_name is not None:
                if disc.field_name != field_name:
                    continue

            # Source filter
            if source_id is not None:
                if disc.source_a_id != source_id and disc.source_b_id != source_id:
                    continue

            result.append(disc)

        logger.debug(
            "Filtered %d -> %d discrepancies (severity=%s, type=%s, "
            "field=%s, source=%s)",
            len(discrepancies),
            len(result),
            min_severity,
            discrepancy_type,
            field_name,
            source_id,
        )
        return result

    # ------------------------------------------------------------------
    # 9. Summarization
    # ------------------------------------------------------------------

    def summarize(
        self,
        discrepancies: List[Discrepancy],
    ) -> DiscrepancySummary:
        """Produce an aggregated summary of discrepancies.

        Counts by type, severity, source, and field.  Computes mean
        and max deviation as well as mean impact score.

        Args:
            discrepancies: List of discrepancies to summarise.

        Returns:
            ``DiscrepancySummary`` with aggregated statistics.

        Example:
            >>> summary = engine.summarize(discrepancies)
            >>> print(summary.total, summary.critical_count)
        """
        start = time.time()

        if not discrepancies:
            return DiscrepancySummary(
                generated_at=_utcnow().isoformat(),
                provenance_hash=self._compute_provenance(
                    "summarize", {"count": 0}, {"total": 0},
                ),
            )

        by_type: Dict[str, int] = Counter()
        by_severity: Dict[str, int] = Counter()
        by_source: Dict[str, int] = Counter()
        by_field: Dict[str, int] = Counter()
        deviations: List[float] = []
        impacts: List[float] = []
        critical_count = 0
        high_count = 0
        pending_count = 0

        for disc in discrepancies:
            by_type[disc.discrepancy_type.value] += 1
            by_severity[disc.severity.value] += 1
            by_source[disc.source_a_id] += 1
            by_source[disc.source_b_id] += 1
            by_field[disc.field_name] += 1
            deviations.append(disc.deviation_pct)
            impacts.append(disc.impact_score)

            if disc.severity == DiscrepancySeverity.CRITICAL:
                critical_count += 1
            if disc.severity == DiscrepancySeverity.HIGH:
                high_count += 1
            if disc.requires_manual_review:
                pending_count += 1

        total = len(discrepancies)
        mean_dev = round(sum(deviations) / total, 4) if total else 0.0
        max_dev = round(max(deviations), 4) if deviations else 0.0
        mean_impact = round(sum(impacts) / total, 4) if total else 0.0

        # Top fields by count (up to 10)
        top_fields: List[Tuple[str, int]] = sorted(
            by_field.items(), key=lambda x: -x[1],
        )[:10]

        # Provenance
        prov_hash = self._compute_provenance(
            "summarize",
            {"total": total},
            {
                "critical": critical_count,
                "high": high_count,
                "mean_deviation": mean_dev,
            },
        )

        elapsed = time.time() - start
        observe_duration("summarize", elapsed)

        summary = DiscrepancySummary(
            total=total,
            by_type=dict(by_type),
            by_severity=dict(by_severity),
            by_source=dict(by_source),
            by_field=dict(by_field),
            critical_count=critical_count,
            high_count=high_count,
            pending_review_count=pending_count,
            mean_deviation_pct=mean_dev,
            max_deviation_pct=max_dev,
            mean_impact_score=mean_impact,
            top_fields=top_fields,
            provenance_hash=prov_hash,
            generated_at=_utcnow().isoformat(),
        )

        logger.info(
            "Discrepancy summary: total=%d, critical=%d, high=%d, "
            "mean_dev=%.2f%%, max_dev=%.2f%%",
            total,
            critical_count,
            high_count,
            mean_dev,
            max_dev,
        )
        return summary

    # ------------------------------------------------------------------
    # 10. Prioritization
    # ------------------------------------------------------------------

    def prioritize(
        self,
        discrepancies: List[Discrepancy],
    ) -> List[Discrepancy]:
        """Sort discrepancies by priority for review.

        Sorts by severity (CRITICAL first), then by deviation
        percentage descending.  Flags CRITICAL and HIGH discrepancies
        as requiring manual review when confidence is below the
        configured threshold.

        Args:
            discrepancies: Discrepancies to prioritise.

        Returns:
            Priority-sorted list with manual review flags updated.

        Example:
            >>> prioritized = engine.prioritize(discrepancies)
            >>> assert prioritized[0].severity == DiscrepancySeverity.CRITICAL
        """
        for disc in discrepancies:
            disc.requires_manual_review = self._should_flag_manual_review(
                disc.severity, disc.confidence,
            )

        prioritized = sorted(
            discrepancies,
            key=lambda d: (
                _SEVERITY_ORDER.get(d.severity, 99),
                -d.deviation_pct,
            ),
        )

        logger.debug(
            "Prioritized %d discrepancies (top severity: %s)",
            len(prioritized),
            prioritized[0].severity.value if prioritized else "none",
        )
        return prioritized

    # ------------------------------------------------------------------
    # 11. Group by entity
    # ------------------------------------------------------------------

    def group_by_entity(
        self,
        discrepancies: List[Discrepancy],
        matches: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Discrepancy]]:
        """Group discrepancies by entity identifier for entity-level review.

        Uses the ``entity_id`` field on each discrepancy.  If
        ``entity_id`` is empty, falls back to the ``match_id``.
        Optionally enriches grouping using the provided ``matches``
        dictionary.

        Args:
            discrepancies: Discrepancies to group.
            matches: Optional match context dictionary mapping
                match_id to match metadata (may contain entity_id).

        Returns:
            Dictionary mapping entity ID to its list of discrepancies.

        Example:
            >>> groups = engine.group_by_entity(discrepancies)
            >>> for entity_id, disc_list in groups.items():
            ...     print(entity_id, len(disc_list))
        """
        groups: Dict[str, List[Discrepancy]] = defaultdict(list)
        matches = matches or {}

        for disc in discrepancies:
            # Determine entity key
            entity_key = disc.entity_id
            if not entity_key:
                # Try to get from matches metadata
                match_meta = matches.get(disc.match_id, {})
                if isinstance(match_meta, dict):
                    entity_key = match_meta.get("entity_id", "")
            if not entity_key:
                entity_key = disc.match_id

            groups[entity_key].append(disc)

        logger.debug(
            "Grouped %d discrepancies into %d entities",
            len(discrepancies),
            len(groups),
        )
        return dict(groups)

    # ------------------------------------------------------------------
    # 12. Impact score computation
    # ------------------------------------------------------------------

    def compute_impact_score(
        self,
        discrepancy: Discrepancy,
    ) -> float:
        """Compute a normalised impact score for a discrepancy.

        Combines the severity weight (CRITICAL=1.0, HIGH=0.75,
        MEDIUM=0.5, LOW=0.25, INFO=0.1) with a normalised deviation
        magnitude to produce a score in [0.0, 1.0].

        The formula is::

            impact = severity_weight * 0.6 + deviation_norm * 0.4

        where ``deviation_norm = min(deviation_pct / 100.0, 1.0)``.

        Args:
            discrepancy: The discrepancy to score.

        Returns:
            Impact score in [0.0, 1.0].

        Example:
            >>> disc = Discrepancy(
            ...     severity=DiscrepancySeverity.HIGH,
            ...     deviation_pct=30.0,
            ... )
            >>> score = engine.compute_impact_score(disc)
            >>> assert 0.0 <= score <= 1.0
        """
        return self._compute_impact_from_severity_and_deviation(
            discrepancy.severity,
            discrepancy.deviation_pct,
        )

    # ------------------------------------------------------------------
    # 13. Description generation
    # ------------------------------------------------------------------

    def _generate_description(
        self,
        comparison: FieldComparison,
        disc_type: DiscrepancyType,
    ) -> str:
        """Generate a human-readable description of a discrepancy.

        Format varies by discrepancy type to convey the most relevant
        context for each kind of mismatch.

        Args:
            comparison: The field comparison that triggered the
                discrepancy.
            disc_type: The classified discrepancy type.

        Returns:
            Human-readable description string.

        Example:
            >>> desc = engine._generate_description(comp, disc_type)
            >>> # "Field 'emissions_total' value mismatch: source A=1234.56,
            >>> #  source B=1289.12, deviation=4.4%"
        """
        fname = comparison.field_name
        val_a = comparison.value_a
        val_b = comparison.value_b
        dev = comparison.deviation_pct

        if disc_type == DiscrepancyType.MISSING_IN_SOURCE:
            return self._describe_missing(comparison)

        if disc_type == DiscrepancyType.TIMING_DIFFERENCE:
            return (
                f"Field '{fname}' timing difference: "
                f"source A={val_a}, source B={val_b}"
            )

        if disc_type == DiscrepancyType.UNIT_DIFFERENCE:
            return (
                f"Field '{fname}' unit difference: "
                f"source A={val_a} ({comparison.unit_a}), "
                f"source B={val_b} ({comparison.unit_b})"
            )

        if disc_type == DiscrepancyType.AGGREGATION_MISMATCH:
            return (
                f"Field '{fname}' aggregation mismatch: "
                f"source A={val_a}, source B={val_b}, "
                f"deviation={dev:.1f}%"
            )

        if disc_type == DiscrepancyType.FORMAT_DIFFERENCE:
            return (
                f"Field '{fname}' format difference: "
                f"source A='{val_a}', source B='{val_b}'"
            )

        if disc_type == DiscrepancyType.ROUNDING_DIFFERENCE:
            return (
                f"Field '{fname}' rounding difference: "
                f"source A={val_a}, source B={val_b}, "
                f"deviation={dev:.1f}%"
            )

        if disc_type == DiscrepancyType.CLASSIFICATION_MISMATCH:
            return (
                f"Field '{fname}' classification mismatch: "
                f"source A='{val_a}', source B='{val_b}'"
            )

        # Default: value mismatch
        return (
            f"Field '{fname}' value mismatch: "
            f"source A={val_a}, source B={val_b}, "
            f"deviation={dev:.1f}%"
        )

    def _describe_missing(
        self,
        comparison: FieldComparison,
    ) -> str:
        """Generate description for a missing-value discrepancy.

        Args:
            comparison: The field comparison with a missing value.

        Returns:
            Human-readable missing value description.
        """
        fname = comparison.field_name

        if comparison.result == ComparisonResult.MISSING_LEFT:
            return (
                f"Field '{fname}' missing in source A: "
                f"source B has value '{comparison.value_b}'"
            )
        if comparison.result == ComparisonResult.MISSING_RIGHT:
            return (
                f"Field '{fname}' missing in source B: "
                f"source A has value '{comparison.value_a}'"
            )
        return f"Field '{fname}' missing in one of the sources"

    # ------------------------------------------------------------------
    # 14. Provenance computation
    # ------------------------------------------------------------------

    def _compute_provenance(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        """Compute SHA-256 provenance hash and record chain entry.

        Hashes the input and output data, then adds a chain entry
        linking to the previous provenance hash for tamper-evident
        audit trail.

        Args:
            operation: Name of the operation.
            input_data: Input data to hash.
            output_data: Output data to hash.

        Returns:
            SHA-256 chain hash string.
        """
        input_hash = self._provenance.build_hash(input_data)
        output_hash = self._provenance.build_hash(output_data)

        chain_hash = self._provenance.add_to_chain(
            operation=operation,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={"engine": "DiscrepancyDetectorEngine"},
        )
        return chain_hash

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_signed_deviation(
        self,
        comparison: FieldComparison,
    ) -> float:
        """Compute signed deviation percentage (A - B) / |B| * 100.

        For non-numeric fields or zero-denominator cases, returns 0.0.

        Args:
            comparison: The field comparison.

        Returns:
            Signed deviation as a percentage.
        """
        val_a = comparison.value_a
        val_b = comparison.value_b

        if not isinstance(val_a, (int, float)):
            return 0.0
        if not isinstance(val_b, (int, float)):
            return 0.0

        a = float(val_a)
        b = float(val_b)

        if math.isnan(a) or math.isnan(b):
            return 0.0
        if math.isinf(a) or math.isinf(b):
            return 0.0

        if b == 0.0:
            if a == 0.0:
                return 0.0
            return 100.0 if a > 0 else -100.0

        return round((a - b) / abs(b) * 100.0, 4)

    def _compute_impact_from_severity_and_deviation(
        self,
        severity: DiscrepancySeverity,
        deviation_pct: float,
    ) -> float:
        """Compute normalised impact score from severity and deviation.

        Args:
            severity: Discrepancy severity grade.
            deviation_pct: Absolute deviation percentage.

        Returns:
            Impact score in [0.0, 1.0].
        """
        sev_weight = _SEVERITY_WEIGHT.get(severity, 0.1)
        dev_norm = min(_safe_abs(deviation_pct) / 100.0, 1.0)
        impact = sev_weight * 0.6 + dev_norm * 0.4
        return round(min(max(impact, 0.0), 1.0), 4)

    def _should_flag_manual_review(
        self,
        severity: DiscrepancySeverity,
        confidence: float,
    ) -> bool:
        """Determine whether a discrepancy should be flagged for review.

        CRITICAL and HIGH severity discrepancies are flagged when the
        comparison confidence falls below the manual review threshold.
        All CRITICAL discrepancies are always flagged regardless of
        confidence.

        Args:
            severity: Discrepancy severity.
            confidence: Comparison confidence (0.0-1.0).

        Returns:
            True if manual review is recommended.
        """
        if severity == DiscrepancySeverity.CRITICAL:
            return True
        if severity == DiscrepancySeverity.HIGH:
            return confidence < self._manual_review_confidence_threshold
        return False

    def _compute_type_distribution(
        self,
        discrepancies: List[Discrepancy],
    ) -> Dict[str, int]:
        """Compute discrepancy type distribution.

        Args:
            discrepancies: List of discrepancies.

        Returns:
            Dictionary mapping type value to count.
        """
        distribution: Dict[str, int] = Counter()
        for disc in discrepancies:
            distribution[disc.discrepancy_type.value] += 1
        return dict(distribution)

    def _compute_source_correlation(
        self,
        discrepancies: List[Discrepancy],
    ) -> Dict[str, int]:
        """Compute source-pair discrepancy counts.

        Args:
            discrepancies: List of discrepancies.

        Returns:
            Dictionary mapping source pair key to discrepancy count.
        """
        pair_counts: Dict[str, int] = Counter()
        for disc in discrepancies:
            pair_key = f"{disc.source_a_id}|{disc.source_b_id}"
            pair_counts[pair_key] += 1
        return dict(pair_counts)

    def _detect_temporal_patterns(
        self,
        discrepancies: List[Discrepancy],
    ) -> Dict[str, Any]:
        """Detect temporal clustering of discrepancies.

        Groups discrepancies by their ``period`` field and computes
        distribution and concentration metrics.

        Args:
            discrepancies: List of discrepancies with period fields.

        Returns:
            Dictionary with temporal pattern analysis::

                {
                    "by_period": {"2024-Q1": 5, ...},
                    "concentration": float,
                    "most_affected_period": str,
                }
        """
        period_counts: Dict[str, int] = Counter()
        for disc in discrepancies:
            period_key = disc.period if disc.period else "unspecified"
            period_counts[period_key] += 1

        if not period_counts:
            return {
                "by_period": {},
                "concentration": 0.0,
                "most_affected_period": "",
            }

        total = sum(period_counts.values())
        most_affected = max(period_counts, key=period_counts.get)  # type: ignore[arg-type]
        max_count = period_counts[most_affected]

        # Concentration: Herfindahl-like index (sum of squared shares)
        concentration = 0.0
        if total > 0:
            for count in period_counts.values():
                share = count / total
                concentration += share * share
        concentration = round(concentration, 4)

        return {
            "by_period": dict(period_counts),
            "concentration": concentration,
            "most_affected_period": most_affected,
            "most_affected_count": max_count,
        }

    # ------------------------------------------------------------------
    # Batch processing helpers
    # ------------------------------------------------------------------

    def detect_discrepancies_batch(
        self,
        comparison_sets: List[Dict[str, Any]],
    ) -> List[List[Discrepancy]]:
        """Detect discrepancies for multiple match comparison sets.

        Each entry in ``comparison_sets`` should contain keys:
        ``comparisons``, ``match_id``, ``source_a_id``,
        ``source_b_id``, and optionally ``entity_id`` and ``period``.

        Args:
            comparison_sets: List of comparison set dictionaries.

        Returns:
            List of discrepancy lists, one per comparison set.

        Example:
            >>> sets = [
            ...     {"comparisons": comps1, "match_id": "m1",
            ...      "source_a_id": "a", "source_b_id": "b"},
            ...     {"comparisons": comps2, "match_id": "m2",
            ...      "source_a_id": "a", "source_b_id": "b"},
            ... ]
            >>> results = engine.detect_discrepancies_batch(sets)
        """
        start = time.time()
        all_results: List[List[Discrepancy]] = []

        for i, cset in enumerate(comparison_sets):
            comps = cset.get("comparisons", [])
            match_id = cset.get("match_id", f"batch_{i}")
            src_a = cset.get("source_a_id", "")
            src_b = cset.get("source_b_id", "")
            entity_id = cset.get("entity_id", "")
            period = cset.get("period", "")

            discs = self.detect_discrepancies(
                comparisons=comps,
                match_id=match_id,
                source_a_id=src_a,
                source_b_id=src_b,
                entity_id=entity_id,
                period=period,
            )
            all_results.append(discs)

        elapsed = time.time() - start
        total_discs = sum(len(r) for r in all_results)
        observe_duration("detect_discrepancies_batch", elapsed)

        logger.info(
            "Batch discrepancy detection: %d sets, %d total "
            "discrepancies, %.3fms",
            len(comparison_sets),
            total_discs,
            elapsed * 1000.0,
        )
        return all_results

    def summarize_by_source_pair(
        self,
        discrepancies: List[Discrepancy],
    ) -> Dict[str, DiscrepancySummary]:
        """Produce per-source-pair summaries.

        Args:
            discrepancies: List of discrepancies.

        Returns:
            Dictionary mapping source pair key to summary.
        """
        pair_groups: Dict[str, List[Discrepancy]] = defaultdict(list)
        for disc in discrepancies:
            pair_key = f"{disc.source_a_id}|{disc.source_b_id}"
            pair_groups[pair_key].append(disc)

        summaries: Dict[str, DiscrepancySummary] = {}
        for pair_key, group in pair_groups.items():
            summaries[pair_key] = self.summarize(group)

        return summaries

    def summarize_by_entity(
        self,
        discrepancies: List[Discrepancy],
        matches: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DiscrepancySummary]:
        """Produce per-entity summaries.

        Args:
            discrepancies: List of discrepancies.
            matches: Optional match metadata dictionary.

        Returns:
            Dictionary mapping entity ID to summary.
        """
        entity_groups = self.group_by_entity(discrepancies, matches)

        summaries: Dict[str, DiscrepancySummary] = {}
        for entity_id, group in entity_groups.items():
            summaries[entity_id] = self.summarize(group)

        return summaries

    def get_critical_discrepancies(
        self,
        discrepancies: List[Discrepancy],
    ) -> List[Discrepancy]:
        """Return only CRITICAL severity discrepancies.

        Args:
            discrepancies: All discrepancies.

        Returns:
            Filtered list of CRITICAL discrepancies.
        """
        return self.filter_discrepancies(
            discrepancies,
            min_severity=DiscrepancySeverity.CRITICAL,
        )

    def get_review_required(
        self,
        discrepancies: List[Discrepancy],
    ) -> List[Discrepancy]:
        """Return discrepancies that require manual review.

        Args:
            discrepancies: All discrepancies.

        Returns:
            Discrepancies with ``requires_manual_review=True``.
        """
        return [d for d in discrepancies if d.requires_manual_review]

    def compute_reconciliation_score(
        self,
        total_comparisons: int,
        discrepancies: List[Discrepancy],
    ) -> float:
        """Compute an overall reconciliation quality score.

        The score is 1.0 minus the weighted discrepancy ratio.
        Weights are based on severity.

        Args:
            total_comparisons: Total field comparisons performed.
            discrepancies: Detected discrepancies.

        Returns:
            Score in [0.0, 1.0] where 1.0 is perfect reconciliation.
        """
        if total_comparisons <= 0:
            return 1.0 if not discrepancies else 0.0

        weighted_disc_sum = 0.0
        for disc in discrepancies:
            weight = _SEVERITY_WEIGHT.get(disc.severity, 0.1)
            weighted_disc_sum += weight

        ratio = _safe_divide(weighted_disc_sum, total_comparisons)
        score = max(0.0, 1.0 - ratio)
        return round(score, 4)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def discrepancies_to_dicts(
        self,
        discrepancies: List[Discrepancy],
    ) -> List[Dict[str, Any]]:
        """Serialize discrepancies to plain dictionaries.

        Args:
            discrepancies: List of discrepancies.

        Returns:
            List of serializable dictionaries.
        """
        result: List[Dict[str, Any]] = []
        for disc in discrepancies:
            d: Dict[str, Any] = {
                "discrepancy_id": disc.discrepancy_id,
                "match_id": disc.match_id,
                "source_a_id": disc.source_a_id,
                "source_b_id": disc.source_b_id,
                "field_name": disc.field_name,
                "field_type": disc.field_type.value,
                "discrepancy_type": disc.discrepancy_type.value,
                "severity": disc.severity.value,
                "value_a": disc.value_a,
                "value_b": disc.value_b,
                "deviation_pct": disc.deviation_pct,
                "signed_deviation_pct": disc.signed_deviation_pct,
                "impact_score": disc.impact_score,
                "description": disc.description,
                "confidence": disc.confidence,
                "requires_manual_review": disc.requires_manual_review,
                "resolution_status": disc.resolution_status,
                "entity_id": disc.entity_id,
                "period": disc.period,
                "provenance_hash": disc.provenance_hash,
                "detected_at": disc.detected_at,
                "metadata": disc.metadata,
            }
            result.append(d)
        return result

    def summary_to_dict(
        self,
        summary: DiscrepancySummary,
    ) -> Dict[str, Any]:
        """Serialize a DiscrepancySummary to a plain dictionary.

        Args:
            summary: The summary to serialize.

        Returns:
            Serializable dictionary.
        """
        return {
            "total": summary.total,
            "by_type": summary.by_type,
            "by_severity": summary.by_severity,
            "by_source": summary.by_source,
            "by_field": summary.by_field,
            "critical_count": summary.critical_count,
            "high_count": summary.high_count,
            "pending_review_count": summary.pending_review_count,
            "mean_deviation_pct": summary.mean_deviation_pct,
            "max_deviation_pct": summary.max_deviation_pct,
            "mean_impact_score": summary.mean_impact_score,
            "top_fields": summary.top_fields,
            "provenance_hash": summary.provenance_hash,
            "generated_at": summary.generated_at,
        }

    # ------------------------------------------------------------------
    # Provenance accessors
    # ------------------------------------------------------------------

    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Return the full provenance chain for audit inspection.

        Returns:
            List of provenance entries, oldest first.
        """
        return self._provenance.get_chain()

    def verify_provenance(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Verify the integrity of the provenance chain.

        Returns:
            Tuple of (is_valid, chain_entries).
        """
        return self._provenance.verify_chain()

    def get_provenance_hash(self) -> str:
        """Return the current head of the provenance chain.

        Returns:
            SHA-256 chain hash string.
        """
        return self._provenance.get_current_hash()

    def reset_provenance(self) -> None:
        """Reset the provenance tracker to genesis state."""
        self._provenance.reset()
        logger.info("DiscrepancyDetectorEngine provenance reset")


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = ["DiscrepancyDetectorEngine"]
