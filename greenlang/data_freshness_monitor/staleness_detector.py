# -*- coding: utf-8 -*-
"""
Staleness Detector Engine - AGENT-DATA-016 (Engine 4 of 7)

Detects staleness patterns across datasets by analysing refresh history.
Classifies patterns into six categories (recurring, seasonal, source
failure, drift, random gaps, systematic delay) and tracks per-source
reliability over time.

Zero-Hallucination: All pattern detection uses deterministic Python
arithmetic and statistics. No LLM calls for numeric computations.

Pattern Detection Algorithms:
    1. Recurring Staleness -- interval day-of-week clustering
    2. Seasonal Degradation -- monthly missed-cadence variance
    3. Source Failure -- trailing window of consecutive late refreshes
    4. Refresh Drift -- linear regression slope on intervals
    5. Random Gaps -- coefficient of variation of gap positions
    6. Systematic Delay -- consistent mean > cadence with low stddev

Example:
    >>> from greenlang.data_freshness_monitor.staleness_detector import (
    ...     StalenessDetectorEngine,
    ... )
    >>> engine = StalenessDetectorEngine()
    >>> from datetime import datetime, timedelta
    >>> now = datetime.utcnow()
    >>> history = [now - timedelta(hours=i * 24) for i in range(30, 0, -1)]
    >>> patterns = engine.detect_patterns("ds_001", history, cadence_hours=24.0)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Graceful imports -- models.py
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.models import (
        BreachSeverity,
        PatternType,
        SourceReliability,
        StalenessPattern,
    )
except ImportError:  # pragma: no cover -- fallback when models not yet built

    class PatternType(str, Enum):  # type: ignore[no-redef]
        """Classification of staleness pattern detected."""

        RECURRING = "recurring"
        SEASONAL = "seasonal"
        SOURCE_FAILURE = "source_failure"
        DRIFT = "drift"
        RANDOM_GAPS = "random_gaps"
        SYSTEMATIC_DELAY = "systematic_delay"

    class BreachSeverity(str, Enum):  # type: ignore[no-redef]
        """Severity level for SLA breach events."""

        INFO = "info"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    class StalenessPattern(BaseModel):  # type: ignore[no-redef]
        """Describes a detected staleness pattern for a dataset."""

        pattern_id: str = Field(
            default_factory=lambda: str(uuid.uuid4()),
            description="Unique pattern identifier",
        )
        dataset_id: str = Field(
            default="", description="Dataset this pattern applies to",
        )
        pattern_type: PatternType = Field(
            ..., description="Classification of the staleness pattern",
        )
        severity: BreachSeverity = Field(
            default=BreachSeverity.MEDIUM,
            description="Severity of the staleness issue",
        )
        description: str = Field(
            default="", description="Human-readable pattern description",
        )
        confidence: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Detection confidence (0.0-1.0)",
        )
        details: Dict[str, Any] = Field(
            default_factory=dict,
            description="Algorithm-specific details",
        )
        detected_at: datetime = Field(
            default_factory=lambda: datetime.now(timezone.utc),
            description="Timestamp of detection",
        )
        provenance_hash: str = Field(
            default="", description="SHA-256 provenance hash",
        )

    class SourceReliability(BaseModel):  # type: ignore[no-redef]
        """Reliability metrics for a data source."""

        source_name: str = Field(
            ..., description="Data source identifier",
        )
        total_refreshes: int = Field(
            default=0, ge=0, description="Total refresh events observed",
        )
        on_time_refreshes: int = Field(
            default=0, ge=0, description="Refreshes within cadence * 1.5",
        )
        reliability_pct: float = Field(
            default=0.0,
            ge=0.0,
            le=100.0,
            description="Percentage of on-time refreshes",
        )
        avg_delay_hours: float = Field(
            default=0.0,
            ge=0.0,
            description="Mean delay hours for late refreshes",
        )
        trend: str = Field(
            default="stable",
            description="Trend: improving, degrading, or stable",
        )
        provenance_hash: str = Field(
            default="", description="SHA-256 provenance hash",
        )


# ---------------------------------------------------------------------------
# Graceful imports -- provenance.py
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.provenance import ProvenanceTracker
except ImportError:  # pragma: no cover

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal fallback ProvenanceTracker when module unavailable."""

        GENESIS_HASH = hashlib.sha256(
            b"greenlang-data-freshness-monitor-genesis"
        ).hexdigest()

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._chain: List[Dict[str, Any]] = []
            self._last: str = self.GENESIS_HASH

        def hash_record(self, data: Dict[str, Any]) -> str:
            s = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        def build_hash(self, data: Any) -> str:
            s = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        def record(
            self,
            entity_type: str,
            entity_id: str,
            action: str,
            data_hash: str,
            user_id: str = "system",
        ) -> str:
            ts = datetime.now(timezone.utc).isoformat()
            combined = json.dumps({
                "previous": self._last,
                "input": data_hash,
                "output": data_hash,
                "operation": action,
                "timestamp": ts,
            }, sort_keys=True)
            chain_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
            with self._lock:
                self._chain.append({
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "action": action,
                    "data_hash": data_hash,
                    "timestamp": ts,
                    "chain_hash": chain_hash,
                })
                self._last = chain_hash
            return chain_hash

        def reset(self) -> None:
            with self._lock:
                self._chain.clear()
                self._last = self.GENESIS_HASH

        @property
        def entry_count(self) -> int:
            with self._lock:
                return len(self._chain)


# ---------------------------------------------------------------------------
# Graceful imports -- metrics.py
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.metrics import (
        inc_errors as _inc_errors,
        inc_patterns_detected as _inc_patterns_detected,
        observe_processing_duration as _observe_duration,
    )
except ImportError:  # pragma: no cover

    def _inc_errors(error_type: str) -> None:  # type: ignore[misc]
        """No-op metric stub."""

    def _inc_patterns_detected(pattern_type: str, count: int = 1) -> None:  # type: ignore[misc]
        """No-op metric stub."""

    def _observe_duration(operation: str, seconds: float) -> None:  # type: ignore[misc]
        """No-op metric stub."""


# ---------------------------------------------------------------------------
# Graceful imports -- config.py
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.config import get_config
except ImportError:  # pragma: no cover

    def get_config() -> Any:  # type: ignore[misc]
        return None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python statistics helpers (no numpy/scipy)
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean or 0.0.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: List[float], mean: Optional[float] = None) -> float:
    """Compute population standard deviation.

    Args:
        values: List of numeric values.
        mean: Pre-computed mean (computed if not provided).

    Returns:
        Population standard deviation or 0.0.
    """
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _safe_mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _safe_median(values: List[float]) -> float:
    """Compute median of values.

    Args:
        values: List of numeric values.

    Returns:
        Median value or 0.0.
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _coefficient_of_variation(values: List[float]) -> float:
    """Compute coefficient of variation (stddev / mean).

    Args:
        values: List of numeric values.

    Returns:
        CV value, or 0.0 if mean is zero.
    """
    mean = _safe_mean(values)
    if abs(mean) < 1e-12:
        return 0.0
    std = _safe_std(values, mean)
    return std / abs(mean)


def _linear_regression_slope(values: List[float]) -> float:
    """Compute the slope of a simple linear regression y = a + bx.

    Uses the least-squares formula:
        slope = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)

    Args:
        values: y-values indexed by 0, 1, 2, ...

    Returns:
        Regression slope, or 0.0 if fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    for i, y in enumerate(values):
        x = float(i)
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def _intervals_from_history(
    refresh_history: List[datetime],
) -> List[float]:
    """Compute intervals in hours between consecutive refreshes.

    Args:
        refresh_history: Sorted list of refresh timestamps.

    Returns:
        List of interval durations in hours.
    """
    if len(refresh_history) < 2:
        return []
    sorted_ts = sorted(refresh_history)
    intervals: List[float] = []
    for i in range(1, len(sorted_ts)):
        delta = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds() / 3600.0
        intervals.append(delta)
    return intervals


def _severity_from_ratio(ratio: float) -> BreachSeverity:
    """Map a staleness ratio to a BreachSeverity level.

    A ratio represents how bad the staleness is relative to cadence.
    Higher ratios are more severe.

    Args:
        ratio: Staleness ratio (e.g. mean_interval / cadence).

    Returns:
        BreachSeverity classification.
    """
    if ratio >= 5.0:
        return BreachSeverity.CRITICAL
    if ratio >= 3.0:
        return BreachSeverity.HIGH
    if ratio >= 2.0:
        return BreachSeverity.MEDIUM
    if ratio >= 1.5:
        return BreachSeverity.LOW
    return BreachSeverity.INFO


# ---------------------------------------------------------------------------
# StalenessDetectorEngine
# ---------------------------------------------------------------------------


class StalenessDetectorEngine:
    """Detects staleness patterns in dataset refresh histories.

    Analyses refresh timestamp sequences against expected cadence to
    identify six categories of staleness patterns. Also computes
    per-source reliability metrics and maintains provenance records
    for all detections.

    Pattern Types:
        - RECURRING: Intervals frequently exceed 2x cadence at repeating
          day-of-week / time-of-day positions.
        - SEASONAL: Monthly refresh miss rates vary >50% between best
          and worst months.
        - SOURCE_FAILURE: The last N (>=3) consecutive refreshes are
          all late by >2x cadence.
        - DRIFT: Linear regression slope on intervals exceeds
          0.1 * cadence per step (growing delay).
        - RANDOM_GAPS: Gaps exist without temporal clustering
          (CV of gap positions > 1.0).
        - SYSTEMATIC_DELAY: Mean interval consistently exceeds
          cadence * 1.2 with low variance (stddev < 0.3 * mean).

    Attributes:
        _config: Data freshness monitor configuration.
        _provenance: SHA-256 provenance tracker.
        _patterns: Per-dataset detected patterns.
        _source_reliability: Per-source reliability records.
        _lock: Thread-safety lock for all mutable state.
        _detection_count: Running count of patterns detected.
        _error_count: Running count of detection errors.

    Example:
        >>> engine = StalenessDetectorEngine()
        >>> from datetime import datetime, timedelta
        >>> now = datetime.utcnow()
        >>> history = [now - timedelta(hours=i * 6) for i in range(50, 0, -1)]
        >>> patterns = engine.detect_patterns("ds_x", history, 6.0)
        >>> print(len(patterns))
    """

    # Minimum history length for meaningful analysis
    _MIN_HISTORY_LEN: int = 4

    # Multiplier to classify an interval as "late"
    _LATE_MULTIPLIER: float = 2.0

    # Fraction of intervals that must be late for recurring classification
    _RECURRING_LATE_FRAC: float = 0.30

    # Monthly miss rate variance threshold for seasonal classification
    _SEASONAL_MISS_THRESHOLD: float = 0.50

    # Trailing window size for source failure detection
    _SOURCE_FAILURE_WINDOW: int = 3

    # Slope threshold factor for drift detection (relative to cadence)
    _DRIFT_SLOPE_FACTOR: float = 0.1

    # CV threshold for random gap classification
    _RANDOM_GAP_CV_THRESHOLD: float = 1.0

    # Systematic delay: mean must exceed cadence by this factor
    _SYSTEMATIC_DELAY_FACTOR: float = 1.2

    # Systematic delay: stddev must be below this fraction of mean
    _SYSTEMATIC_DELAY_STDDEV_FRAC: float = 0.3

    # On-time threshold multiplier for reliability scoring
    _ON_TIME_MULTIPLIER: float = 1.5

    # Trend comparison window size
    _TREND_WINDOW: int = 10

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize StalenessDetectorEngine.

        Args:
            config: Optional DataFreshnessMonitorConfig override.
                Falls back to ``get_config()`` when not provided.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._patterns: Dict[str, List[StalenessPattern]] = {}
        self._source_reliability: Dict[str, SourceReliability] = {}
        self._lock = threading.Lock()
        self._detection_count: int = 0
        self._error_count: int = 0
        logger.info("StalenessDetectorEngine initialized")

    # ------------------------------------------------------------------
    # Public API -- pattern detection
    # ------------------------------------------------------------------

    def detect_patterns(
        self,
        dataset_id: str,
        refresh_history: List[datetime],
        cadence_hours: float,
    ) -> List[StalenessPattern]:
        """Run all pattern detectors on a dataset's refresh history.

        Executes six detection algorithms sequentially and returns every
        pattern that meets its classification threshold. Results are
        stored internally and accessible via ``get_patterns()``.

        Args:
            dataset_id: Unique dataset identifier.
            refresh_history: Chronological list of refresh timestamps.
            cadence_hours: Expected refresh cadence in hours (> 0).

        Returns:
            List of detected StalenessPattern objects (may be empty).

        Raises:
            ValueError: If cadence_hours <= 0 or dataset_id is empty.
        """
        start = time.time()

        # Input validation
        if not dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if cadence_hours <= 0.0:
            raise ValueError("cadence_hours must be positive")

        if len(refresh_history) < self._MIN_HISTORY_LEN:
            logger.info(
                "Insufficient history for %s (%d < %d), skipping detection",
                dataset_id, len(refresh_history), self._MIN_HISTORY_LEN,
            )
            return []

        # Sort history chronologically
        sorted_history = sorted(refresh_history)

        patterns: List[StalenessPattern] = []

        # Run each detector; collect non-None results
        detectors = [
            ("recurring", self.detect_recurring_staleness),
            ("seasonal", self.detect_seasonal_degradation),
            ("source_failure", self.detect_source_failure),
            ("drift", self.detect_refresh_drift),
            ("random_gaps", self.detect_random_gaps),
            ("systematic_delay", self.detect_systematic_delay),
        ]

        for name, detector_fn in detectors:
            try:
                result = detector_fn(sorted_history, cadence_hours)
                if result is not None:
                    # Stamp dataset_id onto the pattern
                    result.dataset_id = dataset_id
                    patterns.append(result)
                    _inc_patterns_detected(name)
            except Exception as exc:
                self._error_count += 1
                _inc_errors(f"detection_{name}")
                logger.warning(
                    "Detector '%s' failed for %s: %s",
                    name, dataset_id, exc,
                )

        # Store patterns
        with self._lock:
            self._patterns[dataset_id] = patterns
            self._detection_count += len(patterns)

        # Provenance
        provenance_data = {
            "dataset_id": dataset_id,
            "cadence_hours": cadence_hours,
            "history_len": len(refresh_history),
            "patterns_detected": len(patterns),
            "pattern_types": [p.pattern_type.value for p in patterns],
        }
        data_hash = self._provenance.build_hash(provenance_data)
        self._provenance.record(
            "staleness_pattern", dataset_id, "detect_patterns", data_hash,
        )

        elapsed = time.time() - start
        _observe_duration("detect_patterns", elapsed)
        logger.info(
            "Staleness detection for %s: %d patterns in %.3fs",
            dataset_id, len(patterns), elapsed,
        )
        return patterns

    # ------------------------------------------------------------------
    # Detector 1: Recurring Staleness
    # ------------------------------------------------------------------

    def detect_recurring_staleness(
        self,
        refresh_history: List[datetime],
        cadence_hours: float,
    ) -> Optional[StalenessPattern]:
        """Detect recurring staleness based on day-of-week clustering.

        Computes intervals between consecutive refreshes. If more than
        30% of intervals exceed 2x the expected cadence AND those late
        intervals cluster by day-of-week (majority share same weekday),
        the pattern is classified as recurring.

        Args:
            refresh_history: Sorted list of refresh timestamps.
            cadence_hours: Expected refresh cadence in hours.

        Returns:
            StalenessPattern with type RECURRING, or None.
        """
        intervals = _intervals_from_history(refresh_history)
        if not intervals:
            return None

        threshold = cadence_hours * self._LATE_MULTIPLIER
        late_indices = [
            i for i, iv in enumerate(intervals) if iv > threshold
        ]

        # Must exceed fraction threshold
        late_frac = len(late_indices) / len(intervals)
        if late_frac < self._RECURRING_LATE_FRAC:
            return None

        # Check day-of-week clustering among late intervals
        sorted_history = sorted(refresh_history)
        weekday_counts: Dict[int, int] = defaultdict(int)
        for idx in late_indices:
            # The late interval starts at sorted_history[idx]
            if idx < len(sorted_history):
                weekday_counts[sorted_history[idx].weekday()] += 1

        if not weekday_counts:
            return None

        # If the most-common weekday accounts for >= 40% of late events,
        # classify as recurring (day-of-week pattern)
        max_weekday_count = max(weekday_counts.values())
        weekday_concentration = max_weekday_count / len(late_indices)
        is_recurring = weekday_concentration >= 0.40

        if not is_recurring:
            return None

        dominant_weekday = max(weekday_counts, key=weekday_counts.get)  # type: ignore[arg-type]
        weekday_names = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        confidence = min(1.0, late_frac * weekday_concentration * 2.0)

        details = {
            "late_fraction": round(late_frac, 4),
            "late_count": len(late_indices),
            "total_intervals": len(intervals),
            "dominant_weekday": weekday_names[dominant_weekday],
            "dominant_weekday_pct": round(weekday_concentration * 100, 1),
            "weekday_distribution": {
                weekday_names[k]: v for k, v in weekday_counts.items()
            },
        }

        provenance_hash = self._provenance.build_hash(details)

        return StalenessPattern(
            dataset_id="",
            pattern_type=PatternType.RECURRING,
            severity=_severity_from_ratio(late_frac * 5),
            description=(
                f"Recurring staleness detected: {late_frac:.0%} of intervals "
                f"exceed {threshold:.1f}h, concentrated on "
                f"{weekday_names[dominant_weekday]}s "
                f"({weekday_concentration:.0%})"
            ),
            confidence=round(confidence, 4),
            details=details,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Detector 2: Seasonal Degradation
    # ------------------------------------------------------------------

    def detect_seasonal_degradation(
        self,
        refresh_history: List[datetime],
        cadence_hours: float,
    ) -> Optional[StalenessPattern]:
        """Detect seasonal degradation in refresh patterns.

        Buckets refresh history into months. For each month, computes
        the expected number of refreshes and the actual count. If any
        month has >50% more missed cadences than the best month, the
        pattern is classified as seasonal degradation.

        Args:
            refresh_history: Sorted list of refresh timestamps.
            cadence_hours: Expected refresh cadence in hours.

        Returns:
            StalenessPattern with type SEASONAL, or None.
        """
        if len(refresh_history) < 2:
            return None

        sorted_history = sorted(refresh_history)

        # Bucket refreshes by (year, month)
        monthly_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for ts in sorted_history:
            monthly_counts[(ts.year, ts.month)] += 1

        if len(monthly_counts) < 2:
            return None

        # Compute expected refreshes per month (~730 hours / cadence)
        hours_per_month = 730.0
        expected_per_month = hours_per_month / max(cadence_hours, 0.01)

        # Compute miss rate per month
        miss_rates: Dict[str, float] = {}
        for (year, month), count in monthly_counts.items():
            missed = max(0.0, expected_per_month - count)
            miss_rate = missed / max(expected_per_month, 1.0)
            miss_rates[f"{year}-{month:02d}"] = round(miss_rate, 4)

        rates = list(miss_rates.values())
        best_rate = min(rates)
        worst_rate = max(rates)

        # Check if worst month exceeds best by >50 percentage points
        degradation = worst_rate - best_rate
        if degradation < self._SEASONAL_MISS_THRESHOLD:
            return None

        worst_month = max(miss_rates, key=miss_rates.get)  # type: ignore[arg-type]
        best_month = min(miss_rates, key=miss_rates.get)  # type: ignore[arg-type]
        confidence = min(1.0, degradation * 1.5)

        details = {
            "monthly_miss_rates": miss_rates,
            "best_month": best_month,
            "best_miss_rate": best_rate,
            "worst_month": worst_month,
            "worst_miss_rate": worst_rate,
            "degradation": round(degradation, 4),
            "expected_per_month": round(expected_per_month, 2),
        }

        provenance_hash = self._provenance.build_hash(details)

        return StalenessPattern(
            dataset_id="",
            pattern_type=PatternType.SEASONAL,
            severity=_severity_from_ratio(1.0 + degradation * 4),
            description=(
                f"Seasonal degradation: worst month ({worst_month}) "
                f"miss rate {worst_rate:.0%} vs best month ({best_month}) "
                f"miss rate {best_rate:.0%}, delta {degradation:.0%}"
            ),
            confidence=round(confidence, 4),
            details=details,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Detector 3: Source Failure
    # ------------------------------------------------------------------

    def detect_source_failure(
        self,
        refresh_history: List[datetime],
        cadence_hours: float,
    ) -> Optional[StalenessPattern]:
        """Detect source failure from trailing late refreshes.

        If the last N (N >= 3) intervals are all late by >2x cadence,
        classify as source failure. This indicates the data source is
        consistently failing to deliver updates.

        Args:
            refresh_history: Sorted list of refresh timestamps.
            cadence_hours: Expected refresh cadence in hours.

        Returns:
            StalenessPattern with type SOURCE_FAILURE, or None.
        """
        intervals = _intervals_from_history(refresh_history)
        if len(intervals) < self._SOURCE_FAILURE_WINDOW:
            return None

        threshold = cadence_hours * self._LATE_MULTIPLIER
        trailing = intervals[-self._SOURCE_FAILURE_WINDOW:]

        # All trailing intervals must exceed the threshold
        if not all(iv > threshold for iv in trailing):
            return None

        avg_trailing = _safe_mean(trailing)
        max_trailing = max(trailing)
        ratio = avg_trailing / max(cadence_hours, 0.01)
        confidence = min(1.0, ratio / 5.0)

        details = {
            "trailing_window": self._SOURCE_FAILURE_WINDOW,
            "trailing_intervals_hours": [round(iv, 2) for iv in trailing],
            "avg_trailing_hours": round(avg_trailing, 2),
            "max_trailing_hours": round(max_trailing, 2),
            "threshold_hours": round(threshold, 2),
            "avg_to_cadence_ratio": round(ratio, 4),
        }

        provenance_hash = self._provenance.build_hash(details)

        return StalenessPattern(
            dataset_id="",
            pattern_type=PatternType.SOURCE_FAILURE,
            severity=_severity_from_ratio(ratio),
            description=(
                f"Source failure: last {self._SOURCE_FAILURE_WINDOW} "
                f"intervals all exceed {threshold:.1f}h threshold "
                f"(avg {avg_trailing:.1f}h, max {max_trailing:.1f}h)"
            ),
            confidence=round(confidence, 4),
            details=details,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Detector 4: Refresh Drift
    # ------------------------------------------------------------------

    def detect_refresh_drift(
        self,
        refresh_history: List[datetime],
        cadence_hours: float,
    ) -> Optional[StalenessPattern]:
        """Detect refresh drift via linear regression on intervals.

        Computes a linear regression of interval durations over time.
        If the slope exceeds 0.1 * cadence per interval step, the
        intervals are growing, indicating drift.

        Args:
            refresh_history: Sorted list of refresh timestamps.
            cadence_hours: Expected refresh cadence in hours.

        Returns:
            StalenessPattern with type DRIFT, or None.
        """
        intervals = _intervals_from_history(refresh_history)
        if len(intervals) < 3:
            return None

        slope = _linear_regression_slope(intervals)
        slope_threshold = self._DRIFT_SLOPE_FACTOR * cadence_hours

        if slope <= slope_threshold:
            return None

        # Confidence scales with how much slope exceeds threshold
        drift_ratio = slope / max(slope_threshold, 0.01)
        confidence = min(1.0, drift_ratio / 3.0)

        first_half_mean = _safe_mean(intervals[: len(intervals) // 2])
        second_half_mean = _safe_mean(intervals[len(intervals) // 2:])

        details = {
            "slope_hours_per_step": round(slope, 6),
            "slope_threshold": round(slope_threshold, 6),
            "drift_ratio": round(drift_ratio, 4),
            "first_half_mean_hours": round(first_half_mean, 2),
            "second_half_mean_hours": round(second_half_mean, 2),
            "interval_count": len(intervals),
        }

        provenance_hash = self._provenance.build_hash(details)

        return StalenessPattern(
            dataset_id="",
            pattern_type=PatternType.DRIFT,
            severity=_severity_from_ratio(drift_ratio),
            description=(
                f"Refresh drift: interval slope {slope:.4f}h/step "
                f"exceeds threshold {slope_threshold:.4f}h/step "
                f"(first-half avg {first_half_mean:.1f}h, "
                f"second-half avg {second_half_mean:.1f}h)"
            ),
            confidence=round(confidence, 4),
            details=details,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Detector 5: Random Gaps
    # ------------------------------------------------------------------

    def detect_random_gaps(
        self,
        refresh_history: List[datetime],
        cadence_hours: float,
    ) -> Optional[StalenessPattern]:
        """Detect random (non-patterned) gaps in refresh history.

        Identifies gaps (intervals > 2x cadence). If gaps exist but
        their positions have a coefficient of variation > 1.0 (i.e. they
        are spread randomly through the timeline rather than clustered),
        classify as random gaps.

        Args:
            refresh_history: Sorted list of refresh timestamps.
            cadence_hours: Expected refresh cadence in hours.

        Returns:
            StalenessPattern with type RANDOM_GAPS, or None.
        """
        intervals = _intervals_from_history(refresh_history)
        if not intervals:
            return None

        threshold = cadence_hours * self._LATE_MULTIPLIER
        gap_positions = [
            float(i) for i, iv in enumerate(intervals) if iv > threshold
        ]

        # Need at least 2 gaps to assess randomness
        if len(gap_positions) < 2:
            return None

        cv = _coefficient_of_variation(gap_positions)

        if cv <= self._RANDOM_GAP_CV_THRESHOLD:
            return None

        gap_durations = [
            intervals[int(pos)] for pos in gap_positions
        ]
        avg_gap_hours = _safe_mean(gap_durations)
        gap_frac = len(gap_positions) / len(intervals)
        confidence = min(1.0, cv / 3.0)

        details = {
            "gap_count": len(gap_positions),
            "total_intervals": len(intervals),
            "gap_fraction": round(gap_frac, 4),
            "position_cv": round(cv, 4),
            "cv_threshold": self._RANDOM_GAP_CV_THRESHOLD,
            "avg_gap_duration_hours": round(avg_gap_hours, 2),
            "gap_positions": [int(p) for p in gap_positions],
        }

        provenance_hash = self._provenance.build_hash(details)

        return StalenessPattern(
            dataset_id="",
            pattern_type=PatternType.RANDOM_GAPS,
            severity=_severity_from_ratio(1.0 + gap_frac * 3),
            description=(
                f"Random gaps: {len(gap_positions)} gaps across "
                f"{len(intervals)} intervals (CV={cv:.2f}), "
                f"avg gap {avg_gap_hours:.1f}h"
            ),
            confidence=round(confidence, 4),
            details=details,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Detector 6: Systematic Delay
    # ------------------------------------------------------------------

    def detect_systematic_delay(
        self,
        refresh_history: List[datetime],
        cadence_hours: float,
    ) -> Optional[StalenessPattern]:
        """Detect systematic delay in refresh intervals.

        If the mean interval exceeds cadence * 1.2 consistently (i.e.
        standard deviation < 0.3 * mean), classify as systematic delay.
        This means every refresh is slightly late, not just occasional
        gaps.

        Args:
            refresh_history: Sorted list of refresh timestamps.
            cadence_hours: Expected refresh cadence in hours.

        Returns:
            StalenessPattern with type SYSTEMATIC_DELAY, or None.
        """
        intervals = _intervals_from_history(refresh_history)
        if len(intervals) < 3:
            return None

        mean_iv = _safe_mean(intervals)
        std_iv = _safe_std(intervals, mean_iv)
        delay_threshold = cadence_hours * self._SYSTEMATIC_DELAY_FACTOR

        # Mean must exceed threshold
        if mean_iv <= delay_threshold:
            return None

        # Variance must be low (consistent delay, not random spikes)
        if mean_iv > 0 and std_iv >= self._SYSTEMATIC_DELAY_STDDEV_FRAC * mean_iv:
            return None

        delay_ratio = mean_iv / max(cadence_hours, 0.01)
        confidence = min(1.0, delay_ratio / 2.0)

        details = {
            "mean_interval_hours": round(mean_iv, 2),
            "std_interval_hours": round(std_iv, 2),
            "cadence_hours": cadence_hours,
            "delay_threshold_hours": round(delay_threshold, 2),
            "delay_ratio": round(delay_ratio, 4),
            "cv": round(std_iv / mean_iv if mean_iv > 0 else 0.0, 4),
            "interval_count": len(intervals),
        }

        provenance_hash = self._provenance.build_hash(details)

        return StalenessPattern(
            dataset_id="",
            pattern_type=PatternType.SYSTEMATIC_DELAY,
            severity=_severity_from_ratio(delay_ratio),
            description=(
                f"Systematic delay: mean interval {mean_iv:.1f}h "
                f"exceeds cadence {cadence_hours:.1f}h by "
                f"{delay_ratio:.2f}x with low variance "
                f"(stddev {std_iv:.1f}h)"
            ),
            confidence=round(confidence, 4),
            details=details,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Source Reliability
    # ------------------------------------------------------------------

    def compute_source_reliability(
        self,
        source_name: str,
        refresh_events: List[Dict[str, Any]],
        expected_cadence_hours: float,
    ) -> SourceReliability:
        """Compute reliability metrics for a data source.

        Each refresh event should contain at least a ``timestamp``
        (datetime) key. Events are sorted chronologically, intervals
        computed, and on-time / late classifications applied.

        Trend is determined by comparing the mean interval of the first
        10 events to the last 10.

        Args:
            source_name: Identifier of the data source.
            refresh_events: List of dicts, each with a ``timestamp`` key.
            expected_cadence_hours: Expected refresh cadence in hours.

        Returns:
            SourceReliability with computed metrics.

        Raises:
            ValueError: If source_name is empty or cadence <= 0.
        """
        start = time.time()

        if not source_name:
            raise ValueError("source_name must be non-empty")
        if expected_cadence_hours <= 0.0:
            raise ValueError("expected_cadence_hours must be positive")

        # Extract and sort timestamps
        timestamps: List[datetime] = []
        for event in refresh_events:
            ts = event.get("timestamp")
            if isinstance(ts, datetime):
                timestamps.append(ts)
        timestamps.sort()

        total = len(timestamps)
        if total < 2:
            result = SourceReliability(
                source_name=source_name,
                total_refreshes=total,
                on_time_refreshes=total,
                reliability_pct=100.0 if total > 0 else 0.0,
                avg_delay_hours=0.0,
                trend="stable",
                provenance_hash="",
            )
            result.provenance_hash = self._provenance.build_hash(
                result.model_dump()
            )
            with self._lock:
                self._source_reliability[source_name] = result
            return result

        # Compute intervals
        intervals = _intervals_from_history(timestamps)
        on_time_threshold = expected_cadence_hours * self._ON_TIME_MULTIPLIER

        on_time = 0
        late_delays: List[float] = []
        for iv in intervals:
            if iv <= on_time_threshold:
                on_time += 1
            else:
                late_delays.append(iv - expected_cadence_hours)

        reliability_pct = (on_time / len(intervals)) * 100.0 if intervals else 0.0
        avg_delay = _safe_mean(late_delays) if late_delays else 0.0

        # Trend analysis: compare first window to last window
        trend = self._compute_trend(intervals)

        result = SourceReliability(
            source_name=source_name,
            total_refreshes=total,
            on_time_refreshes=on_time,
            reliability_pct=round(reliability_pct, 2),
            avg_delay_hours=round(avg_delay, 2),
            trend=trend,
            provenance_hash="",
        )
        result.provenance_hash = self._provenance.build_hash(
            result.model_dump()
        )

        # Store
        with self._lock:
            self._source_reliability[source_name] = result

        # Provenance record
        self._provenance.record(
            "source_reliability", source_name, "compute",
            self._provenance.build_hash({"source": source_name, "total": total}),
        )

        elapsed = time.time() - start
        _observe_duration("compute_source_reliability", elapsed)
        logger.info(
            "Source reliability for '%s': %.1f%% (%d/%d on-time), trend=%s",
            source_name, reliability_pct, on_time, len(intervals), trend,
        )
        return result

    def compute_refresh_regularity(
        self,
        intervals_hours: List[float],
    ) -> float:
        """Compute refresh regularity score from interval durations.

        Regularity is defined as 1 - CV (coefficient of variation) of
        the intervals, clamped to [0.0, 1.0]. A score of 1.0 means
        perfectly regular intervals; 0.0 means highly irregular.

        Args:
            intervals_hours: List of interval durations in hours.

        Returns:
            Regularity score between 0.0 and 1.0.
        """
        if len(intervals_hours) < 2:
            return 1.0  # Single or no interval is "regular" by default

        cv = _coefficient_of_variation(intervals_hours)
        regularity = max(0.0, min(1.0, 1.0 - cv))
        logger.debug(
            "Refresh regularity: CV=%.4f, regularity=%.4f (%d intervals)",
            cv, regularity, len(intervals_hours),
        )
        return round(regularity, 4)

    # ------------------------------------------------------------------
    # Gap identification
    # ------------------------------------------------------------------

    def identify_gap_periods(
        self,
        refresh_history: List[datetime],
        cadence_hours: float,
        threshold_multiplier: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Identify periods where refreshes were missing.

        A gap is defined as an interval exceeding
        ``cadence_hours * threshold_multiplier``.

        Args:
            refresh_history: Sorted list of refresh timestamps.
            cadence_hours: Expected refresh cadence in hours.
            threshold_multiplier: Multiplier for gap detection
                (default 2.0).

        Returns:
            List of dicts with keys: ``start``, ``end``,
            ``duration_hours``.
        """
        if len(refresh_history) < 2:
            return []

        sorted_history = sorted(refresh_history)
        threshold = cadence_hours * threshold_multiplier
        gaps: List[Dict[str, Any]] = []

        for i in range(1, len(sorted_history)):
            delta_hours = (
                (sorted_history[i] - sorted_history[i - 1]).total_seconds()
                / 3600.0
            )
            if delta_hours > threshold:
                gaps.append({
                    "start": sorted_history[i - 1].isoformat(),
                    "end": sorted_history[i].isoformat(),
                    "duration_hours": round(delta_hours, 2),
                })

        logger.debug(
            "Gap identification: %d gaps found (threshold %.1fh)",
            len(gaps), threshold,
        )
        return gaps

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------

    def get_patterns(self, dataset_id: str) -> List[StalenessPattern]:
        """Retrieve detected patterns for a specific dataset.

        Args:
            dataset_id: Dataset identifier to look up.

        Returns:
            List of StalenessPattern objects (empty if none detected).
        """
        with self._lock:
            return list(self._patterns.get(dataset_id, []))

    def get_all_patterns(self) -> List[StalenessPattern]:
        """Retrieve all detected patterns across all datasets.

        Returns:
            Flat list of all StalenessPattern objects.
        """
        with self._lock:
            result: List[StalenessPattern] = []
            for patterns in self._patterns.values():
                result.extend(patterns)
            return result

    def get_source_reliability_rankings(self) -> List[SourceReliability]:
        """Return source reliability records ranked by reliability_pct descending.

        Returns:
            List of SourceReliability objects sorted by reliability
            percentage from highest to lowest.
        """
        with self._lock:
            rankings = list(self._source_reliability.values())
        rankings.sort(key=lambda r: r.reliability_pct, reverse=True)
        return rankings

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for the staleness detector.

        Returns:
            Dictionary with keys: datasets_analysed, total_patterns,
            patterns_by_type, sources_tracked, detection_count,
            error_count, provenance_entries.
        """
        with self._lock:
            datasets_analysed = len(self._patterns)
            total_patterns = sum(
                len(pats) for pats in self._patterns.values()
            )
            type_counts: Dict[str, int] = defaultdict(int)
            for pats in self._patterns.values():
                for p in pats:
                    type_counts[p.pattern_type.value] += 1
            sources_tracked = len(self._source_reliability)
            detection_count = self._detection_count
            error_count = self._error_count

        return {
            "datasets_analysed": datasets_analysed,
            "total_patterns": total_patterns,
            "patterns_by_type": dict(type_counts),
            "sources_tracked": sources_tracked,
            "detection_count": detection_count,
            "error_count": error_count,
            "provenance_entries": self._provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state to initial empty condition.

        Clears detected patterns, source reliability records,
        counters, and provenance history.
        """
        with self._lock:
            self._patterns.clear()
            self._source_reliability.clear()
            self._detection_count = 0
            self._error_count = 0
        self._provenance.reset()
        logger.info("StalenessDetectorEngine reset")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_trend(self, intervals: List[float]) -> str:
        """Determine trend direction from interval time series.

        Compares the mean of the first ``_TREND_WINDOW`` intervals to
        the mean of the last ``_TREND_WINDOW`` intervals. If the
        difference exceeds 10% of the overall mean, a trend is declared.

        Args:
            intervals: List of interval durations in hours.

        Returns:
            One of ``"improving"``, ``"degrading"``, or ``"stable"``.
        """
        window = self._TREND_WINDOW
        if len(intervals) < window * 2:
            return "stable"

        first_mean = _safe_mean(intervals[:window])
        last_mean = _safe_mean(intervals[-window:])
        overall_mean = _safe_mean(intervals)

        if overall_mean < 1e-12:
            return "stable"

        # "Improving" means intervals are getting shorter (more frequent)
        change_pct = (last_mean - first_mean) / overall_mean

        if change_pct < -0.10:
            return "improving"
        if change_pct > 0.10:
            return "degrading"
        return "stable"


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "StalenessDetectorEngine",
    # Re-exported for convenience
    "PatternType",
    "BreachSeverity",
    "StalenessPattern",
    "SourceReliability",
]
