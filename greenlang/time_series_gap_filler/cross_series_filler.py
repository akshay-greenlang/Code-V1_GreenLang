# -*- coding: utf-8 -*-
"""
Cross-Series Gap Filler Engine - AGENT-DATA-014 (Engine 6 of 7)

Pure-Python cross-series correlation-based gap filling engine. Uses
Pearson correlation to identify donor series that are statistically
related to a target series, then fills gaps in the target using
regression, ratio, or multi-donor consensus methods.

Zero-Hallucination: All calculations use deterministic Python
arithmetic (math, statistics). No LLM calls for numeric computations.
No external libraries beyond the standard library.

Supported fill methods:
    - Regression:      OLS linear regression from a single best donor
    - Ratio:           Proportional scaling from a single best donor
    - Donor matching:  Weighted average from multiple correlated donors

Example:
    >>> from greenlang.time_series_gap_filler.cross_series_filler import (
    ...     CrossSeriesFillerEngine,
    ... )
    >>> engine = CrossSeriesFillerEngine()
    >>> ref = engine.register_reference_series("temp", [20, 22, 21, 24, 25])
    >>> result = engine.fill_regression(
    ...     target=[100, None, None, 130, 140],
    ...     donor=ref,
    ... )
    >>> assert result.gaps_filled > 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from greenlang.time_series_gap_filler.config import get_config
from greenlang.time_series_gap_filler.metrics import (
    inc_gaps_filled,
    observe_confidence,
    observe_duration,
    inc_errors,
)
from greenlang.time_series_gap_filler.provenance import (
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


def _is_missing(value: Any) -> bool:
    """Determine whether a value represents a missing data point.

    Treats None and float('nan') as missing.

    Args:
        value: Value to check.

    Returns:
        True if the value is considered missing.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


# ---------------------------------------------------------------------------
# Lightweight data models (self-contained until models.py adds these)
# ---------------------------------------------------------------------------


@dataclass
class ReferenceSeries:
    """A registered reference (donor) series available for cross-series filling.

    Attributes:
        series_id: Unique identifier for this reference series.
        values: Ordered numeric values (may contain None for gaps).
        timestamps: Optional ordered timestamps aligned with values.
        name: Optional human-readable name.
        registered_at: ISO-formatted UTC timestamp of registration.
    """

    series_id: str
    values: List[Optional[float]]
    timestamps: Optional[List[Any]] = None
    name: Optional[str] = None
    registered_at: str = ""

    def __post_init__(self) -> None:
        """Set default registration timestamp if not provided."""
        if not self.registered_at:
            self.registered_at = _utcnow().isoformat()


@dataclass
class FilledPoint:
    """A single filled data point produced by cross-series filling.

    Attributes:
        index: Position in the target series.
        original_value: Original value (None when it was a gap).
        filled_value: Value assigned by the fill operation.
        confidence: Confidence in this specific fill (0.0-1.0).
        donor_id: Identifier of the donor series used.
        method: Fill method that produced this value.
    """

    index: int = 0
    original_value: Optional[float] = None
    filled_value: float = 0.0
    confidence: float = 0.0
    donor_id: str = ""
    method: str = "cross_series"


@dataclass
class FillResult:
    """Result of a single-donor cross-series fill operation.

    Attributes:
        values: Complete series with gaps filled.
        original: The original input series.
        filled_indices: Indices where gaps were filled.
        fill_values: Mapping of index to the value used for filling.
        method: Fill method used (regression, ratio).
        confidence: Overall confidence in the fill quality.
        per_point_confidence: Per-index confidence scores.
        gaps_filled: Number of gaps filled.
        gaps_remaining: Number of gaps that could not be filled.
        correlation: Pearson correlation between target and donor.
        r_squared: R-squared of the regression fit (regression only).
        ratio: Average ratio used (ratio method only).
        slope: Regression slope (regression only).
        intercept: Regression intercept (regression only).
        donor_id: Identifier of the donor series used.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance chain hash.
        details: Additional method-specific metadata.
    """

    values: List[Optional[float]] = field(default_factory=list)
    original: List[Optional[float]] = field(default_factory=list)
    filled_indices: List[int] = field(default_factory=list)
    fill_values: Dict[int, float] = field(default_factory=dict)
    method: str = "regression"
    confidence: float = 0.0
    per_point_confidence: Dict[int, float] = field(default_factory=dict)
    gaps_filled: int = 0
    gaps_remaining: int = 0
    correlation: float = 0.0
    r_squared: float = 0.0
    ratio: float = 0.0
    slope: float = 0.0
    intercept: float = 0.0
    donor_id: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DonorContribution:
    """Contribution from a single donor to a multi-donor fill.

    Attributes:
        donor_id: Identifier of the contributing donor series.
        correlation: Pearson correlation with the target.
        predicted_value: Value predicted by this donor.
        weight: Weight assigned (based on correlation strength).
    """

    donor_id: str = ""
    correlation: float = 0.0
    predicted_value: float = 0.0
    weight: float = 0.0


@dataclass
class CrossSeriesResult:
    """Result of multi-donor consensus cross-series gap filling.

    Attributes:
        result_id: Unique identifier for this result.
        values: Complete series with gaps filled.
        original: The original input series.
        filled_indices: Indices where gaps were filled.
        fill_values: Mapping of index to the value used for filling.
        per_point_confidence: Per-index confidence scores.
        gaps_filled: Number of gaps filled.
        gaps_remaining: Number of gaps that could not be filled.
        donors_used: Number of donor series that contributed.
        donor_ids: Identifiers of all contributing donors.
        contributions: Per-position donor contributions.
        avg_confidence: Average confidence across all fills.
        avg_correlation: Average correlation of contributing donors.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance chain hash.
        details: Additional method-specific metadata.
    """

    result_id: str = ""
    values: List[Optional[float]] = field(default_factory=list)
    original: List[Optional[float]] = field(default_factory=list)
    filled_indices: List[int] = field(default_factory=list)
    fill_values: Dict[int, float] = field(default_factory=dict)
    per_point_confidence: Dict[int, float] = field(default_factory=dict)
    gaps_filled: int = 0
    gaps_remaining: int = 0
    donors_used: int = 0
    donor_ids: List[str] = field(default_factory=list)
    contributions: Dict[int, List[DonorContribution]] = field(
        default_factory=dict,
    )
    avg_confidence: float = 0.0
    avg_correlation: float = 0.0
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default result_id if not provided."""
        if not self.result_id:
            self.result_id = str(uuid4())


# ---------------------------------------------------------------------------
# Pure-Python statistical helpers
# ---------------------------------------------------------------------------


def _overlap_indices(
    a: List[Optional[float]],
    b: List[Optional[float]],
) -> List[int]:
    """Return indices where both series have non-missing values.

    Args:
        a: First series (may contain None/NaN).
        b: Second series (may contain None/NaN).

    Returns:
        Sorted list of integer indices where neither series is missing.
    """
    length = min(len(a), len(b))
    return [
        i for i in range(length)
        if not _is_missing(a[i]) and not _is_missing(b[i])
    ]


def _pearson_r(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two lists.

    Uses the standard formula:
        r = sum((xi - mx)(yi - my))
            / sqrt(sum((xi - mx)^2) * sum((yi - my)^2))

    Args:
        x: First numeric list (no missing values).
        y: Second numeric list (no missing values, same length as x).

    Returns:
        Pearson r in [-1.0, 1.0], or 0.0 when computation is
        impossible (e.g. zero variance or insufficient points).

    Raises:
        ValueError: If x and y have different lengths.
    """
    n = len(x)
    if n != len(y):
        raise ValueError(
            f"x and y must have the same length, got {n} and {len(y)}"
        )
    if n < 3:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = 0.0
    denom_x = 0.0
    denom_y = 0.0

    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        numerator += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy

    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0

    denominator = math.sqrt(denom_x * denom_y)
    if denominator == 0.0:
        return 0.0

    r = numerator / denominator
    # Clamp to [-1, 1] to guard against floating-point drift
    return max(-1.0, min(1.0, r))


def _ols_fit(
    x: List[float],
    y: List[float],
) -> Tuple[float, float, float]:
    """Fit ordinary least squares regression: y = slope * x + intercept.

    Also computes R-squared (coefficient of determination).

    Args:
        x: Independent variable values (no missing).
        y: Dependent variable values (no missing, same length).

    Returns:
        Tuple of (slope, intercept, r_squared).  All 0.0 when the
        fit is degenerate (e.g. zero variance in x or < 2 points).

    Raises:
        ValueError: If x and y have different lengths.
    """
    n = len(x)
    if n != len(y):
        raise ValueError(
            f"x and y must have the same length, got {n} and {len(y)}"
        )
    if n < 2:
        return 0.0, 0.0, 0.0

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_xx = sum(xi * xi for xi in x)

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        # Zero variance in x -- degenerate
        return 0.0, 0.0, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    mean_y = sum_y / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    if ss_tot == 0.0:
        # All y values identical -- perfect fit trivially
        r_squared = 1.0
    else:
        ss_res = sum(
            (yi - (slope * xi + intercept)) ** 2
            for xi, yi in zip(x, y)
        )
        r_squared = max(0.0, 1.0 - ss_res / ss_tot)

    return slope, intercept, r_squared


def _compute_confidence(
    r_squared: float,
    n_donors: int,
    method: str,
) -> float:
    """Compute a confidence score for a cross-series fill.

    Combines the regression fit quality (R-squared) with the number
    of donors contributing, weighted by method.

    Args:
        r_squared: R-squared or correlation-squared for the fit.
        n_donors: Number of donor series contributing.
        method: Fill method (regression, ratio, donor_matching).

    Returns:
        Confidence score in [0.0, 1.0].
    """
    # Base confidence from fit quality
    base = max(0.0, min(1.0, r_squared))

    # Bonus for multi-donor consensus (capped at 0.15)
    donor_bonus = 0.0
    if n_donors > 1:
        donor_bonus = min(0.15, 0.05 * (n_donors - 1))

    # Method-specific scaling factor
    method_factor: float = {
        "regression": 1.0,
        "ratio": 0.9,
        "donor_matching": 0.95,
    }.get(method, 0.85)

    confidence = base * method_factor + donor_bonus
    return max(0.0, min(1.0, confidence))


def _count_missing(series: List[Optional[float]]) -> int:
    """Count missing values in a series.

    Args:
        series: List that may contain None/NaN.

    Returns:
        Number of missing entries.
    """
    return sum(1 for v in series if _is_missing(v))


# ---------------------------------------------------------------------------
# CrossSeriesFillerEngine
# ---------------------------------------------------------------------------


class CrossSeriesFillerEngine:
    """Pure-Python cross-series correlation-based gap filling engine.

    Maintains an internal registry of reference (donor) series and
    provides methods to fill gaps in a target series using the
    correlation structure between target and donor series.

    Supported strategies:
        - **Regression**: OLS linear regression from the best donor.
        - **Ratio**: Proportional scaling from the best donor.
        - **Donor matching**: Weighted consensus from multiple donors.

    All calculations are deterministic pure Python.  No external
    numeric libraries are required.

    Attributes:
        _config: TimeSeriesGapFillerConfig instance.
        _provenance: SHA-256 provenance tracker.
        _registry: Mapping of series_id to ReferenceSeries.

    Example:
        >>> engine = CrossSeriesFillerEngine()
        >>> engine.register_reference_series("s1", [10, 20, 30, 40, 50])
        >>> engine.register_reference_series("s2", [11, 21, 31, 41, 51])
        >>> result = engine.fill_donor_matching(
        ...     target=[100, None, 300, None, 500],
        ...     donors=[engine._registry["s1"], engine._registry["s2"]],
        ... )
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CrossSeriesFillerEngine.

        Args:
            config: Optional TimeSeriesGapFillerConfig override.
                Falls back to the singleton from get_config().
        """
        self._config = config or get_config()
        self._provenance: ProvenanceTracker = get_provenance_tracker()
        self._registry: Dict[str, ReferenceSeries] = {}
        logger.info("CrossSeriesFillerEngine initialized")

    # ------------------------------------------------------------------
    # Reference series registry
    # ------------------------------------------------------------------

    def register_reference_series(
        self,
        series_id: str,
        values: List[Optional[float]],
        timestamps: Optional[List[Any]] = None,
        name: Optional[str] = None,
    ) -> ReferenceSeries:
        """Register a reference (donor) series for cross-series filling.

        Stores the series in the internal registry keyed by series_id.
        If a series with the same ID already exists it is overwritten.

        Args:
            series_id: Unique identifier for the reference series.
            values: Ordered numeric values (None for missing).
            timestamps: Optional aligned timestamps.
            name: Optional human-readable name.

        Returns:
            The newly registered ReferenceSeries.

        Raises:
            ValueError: If series_id is empty or values is empty.
        """
        if not series_id or not series_id.strip():
            raise ValueError("series_id must be non-empty")
        if not values:
            raise ValueError("values must be non-empty")

        ref = ReferenceSeries(
            series_id=series_id,
            values=list(values),
            timestamps=list(timestamps) if timestamps else None,
            name=name or series_id,
        )
        self._registry[series_id] = ref

        logger.info(
            "Registered reference series '%s' with %d values (%d missing)",
            series_id, len(values), _count_missing(values),
        )

        # Record provenance
        if self._config.enable_provenance:
            data_hash = self._provenance.build_hash({
                "series_id": series_id,
                "length": len(values),
                "missing": _count_missing(values),
            })
            self._provenance.record(
                entity_type="cross_series",
                entity_id=series_id,
                action="register",
                data_hash=data_hash,
            )

        return ref

    def get_registry(self) -> Dict[str, ReferenceSeries]:
        """Return the current reference series registry.

        Returns:
            Dictionary mapping series_id to ReferenceSeries.
        """
        return dict(self._registry)

    # ------------------------------------------------------------------
    # Correlation computation
    # ------------------------------------------------------------------

    def compute_correlation(
        self,
        series_a: List[Optional[float]],
        series_b: List[Optional[float]],
    ) -> float:
        """Compute Pearson correlation between two series.

        Only positions where both series have non-missing values are
        used.  Returns 0.0 when there are fewer than 3 overlapping
        points.

        Args:
            series_a: First series (may contain None/NaN).
            series_b: Second series (may contain None/NaN).

        Returns:
            Pearson r in [-1.0, 1.0], or 0.0 if insufficient overlap.
        """
        indices = _overlap_indices(series_a, series_b)
        if len(indices) < 3:
            logger.debug(
                "Insufficient overlap (%d points) for correlation",
                len(indices),
            )
            return 0.0

        x = [float(series_a[i]) for i in indices]  # type: ignore[arg-type]
        y = [float(series_b[i]) for i in indices]  # type: ignore[arg-type]

        return _pearson_r(x, y)

    # ------------------------------------------------------------------
    # Find best donor
    # ------------------------------------------------------------------

    def find_best_donor(
        self,
        target: List[Optional[float]],
        reference_series: List[ReferenceSeries],
    ) -> Optional[ReferenceSeries]:
        """Find the reference series with highest correlation to target.

        Iterates through the candidate list, computes Pearson r for
        each, and returns the one with the highest absolute
        correlation that exceeds the configured threshold.

        Args:
            target: Target series with gaps (may contain None/NaN).
            reference_series: Candidate donor series.

        Returns:
            The best donor ReferenceSeries, or None if no candidate
            meets the correlation threshold.
        """
        if not reference_series:
            logger.debug("No reference series provided for donor search")
            return None

        threshold = self._config.correlation_threshold
        best_donor: Optional[ReferenceSeries] = None
        best_corr = 0.0

        for ref in reference_series:
            r = self.compute_correlation(target, ref.values)
            abs_r = abs(r)

            logger.debug(
                "Donor candidate '%s': r=%.4f (|r|=%.4f, threshold=%.2f)",
                ref.series_id, r, abs_r, threshold,
            )

            if abs_r >= threshold and abs_r > best_corr:
                best_corr = abs_r
                best_donor = ref

        if best_donor is not None:
            logger.info(
                "Best donor found: '%s' with |r|=%.4f",
                best_donor.series_id, best_corr,
            )
        else:
            logger.info(
                "No donor exceeds correlation threshold %.2f", threshold,
            )

        return best_donor

    # ------------------------------------------------------------------
    # Fill: regression
    # ------------------------------------------------------------------

    def fill_regression(
        self,
        target: List[Optional[float]],
        donor: ReferenceSeries,
    ) -> FillResult:
        """Fill gaps in target using OLS regression on the donor series.

        Fits y = slope * x + intercept over overlapping non-missing
        positions, then predicts target values at gap positions using
        the donor's values at those positions.

        Args:
            target: Target series with gaps (None for missing).
            donor: Donor ReferenceSeries to regress against.

        Returns:
            FillResult with filled values and regression diagnostics.
        """
        start = time.time()
        total_missing = _count_missing(target)
        result = FillResult(
            method="regression",
            donor_id=donor.series_id,
            original=list(target),
        )

        # Identify overlapping non-missing positions
        indices = _overlap_indices(target, donor.values)
        if len(indices) < 3:
            logger.warning(
                "Insufficient overlap (%d) for regression fill "
                "from donor '%s'",
                len(indices), donor.series_id,
            )
            result.values = list(target)
            result.gaps_remaining = total_missing
            result.processing_time_ms = (time.time() - start) * 1000.0
            inc_errors("cross_series")
            return result

        # Extract overlapping values for fitting
        x_vals = [float(donor.values[i]) for i in indices]  # type: ignore[arg-type]
        y_vals = [float(target[i]) for i in indices]  # type: ignore[arg-type]

        # Compute correlation
        result.correlation = _pearson_r(x_vals, y_vals)

        # Fit OLS
        slope, intercept, r_squared = _ols_fit(x_vals, y_vals)
        result.slope = slope
        result.intercept = intercept
        result.r_squared = r_squared

        # Build filled series
        filled = list(target)
        filled_indices: List[int] = []
        fill_values: Dict[int, float] = {}
        per_point_conf: Dict[int, float] = {}
        length = min(len(target), len(donor.values))

        for i in range(length):
            if not _is_missing(target[i]):
                continue
            if _is_missing(donor.values[i]):
                continue

            predicted = slope * float(donor.values[i]) + intercept  # type: ignore[arg-type]
            point_conf = _compute_confidence(r_squared, 1, "regression")

            filled[i] = predicted
            filled_indices.append(i)
            fill_values[i] = predicted
            per_point_conf[i] = point_conf

        result.values = filled
        result.filled_indices = filled_indices
        result.fill_values = fill_values
        result.per_point_confidence = per_point_conf
        result.gaps_filled = len(filled_indices)
        result.gaps_remaining = total_missing - result.gaps_filled
        result.confidence = _compute_confidence(r_squared, 1, "regression")

        result.details = {
            "overlap_points": len(indices),
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "correlation": result.correlation,
        }

        # Metrics
        elapsed = time.time() - start
        result.processing_time_ms = elapsed * 1000.0

        if result.gaps_filled > 0:
            inc_gaps_filled("cross_series", result.gaps_filled)
            observe_confidence(result.confidence)
        observe_duration("cross_series_regression", elapsed)

        # Provenance
        result.provenance_hash = self._record_provenance(
            operation="cross_series_regression",
            donor_id=donor.series_id,
            target_length=len(target),
            overlap_points=len(indices),
            gaps_filled=result.gaps_filled,
            extras={
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "correlation": result.correlation,
            },
        )

        logger.info(
            "Regression fill: donor='%s', r=%.4f, R2=%.4f, "
            "filled=%d/%d, confidence=%.4f, %.1fms",
            donor.series_id, result.correlation, r_squared,
            result.gaps_filled, total_missing, result.confidence,
            result.processing_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Fill: ratio
    # ------------------------------------------------------------------

    def fill_ratio(
        self,
        target: List[Optional[float]],
        donor: ReferenceSeries,
    ) -> FillResult:
        """Fill gaps in target using average ratio to donor series.

        Computes ratio = mean(target / donor) over overlapping
        non-missing positions where donor is non-zero, then fills
        gaps as: target_missing = ratio * donor_value.

        Args:
            target: Target series with gaps (None for missing).
            donor: Donor ReferenceSeries to compute ratio against.

        Returns:
            FillResult with filled values and ratio diagnostics.
        """
        start = time.time()
        total_missing = _count_missing(target)
        result = FillResult(
            method="ratio",
            donor_id=donor.series_id,
            original=list(target),
        )

        # Identify overlapping non-missing positions
        indices = _overlap_indices(target, donor.values)
        if len(indices) < 3:
            logger.warning(
                "Insufficient overlap (%d) for ratio fill "
                "from donor '%s'",
                len(indices), donor.series_id,
            )
            result.values = list(target)
            result.gaps_remaining = total_missing
            result.processing_time_ms = (time.time() - start) * 1000.0
            inc_errors("cross_series")
            return result

        # Extract overlapping values
        x_vals = [float(donor.values[i]) for i in indices]  # type: ignore[arg-type]
        y_vals = [float(target[i]) for i in indices]  # type: ignore[arg-type]

        # Compute correlation for confidence
        result.correlation = _pearson_r(x_vals, y_vals)

        # Compute average ratio (skip positions where donor is zero)
        ratios: List[float] = []
        for xi, yi in zip(x_vals, y_vals):
            if abs(xi) > 1e-15:
                ratios.append(yi / xi)

        if not ratios:
            logger.warning(
                "Cannot compute ratio: all donor values near zero "
                "for donor '%s'",
                donor.series_id,
            )
            result.values = list(target)
            result.gaps_remaining = total_missing
            result.processing_time_ms = (time.time() - start) * 1000.0
            inc_errors("cross_series")
            return result

        avg_ratio = sum(ratios) / len(ratios)
        result.ratio = avg_ratio

        # Use correlation-squared as R-squared proxy
        result.r_squared = result.correlation ** 2

        # Build filled series
        filled = list(target)
        filled_indices: List[int] = []
        fill_values: Dict[int, float] = {}
        per_point_conf: Dict[int, float] = {}
        length = min(len(target), len(donor.values))

        for i in range(length):
            if not _is_missing(target[i]):
                continue
            if _is_missing(donor.values[i]):
                continue

            donor_val = float(donor.values[i])  # type: ignore[arg-type]
            predicted = avg_ratio * donor_val
            point_conf = _compute_confidence(
                result.r_squared, 1, "ratio",
            )

            filled[i] = predicted
            filled_indices.append(i)
            fill_values[i] = predicted
            per_point_conf[i] = point_conf

        result.values = filled
        result.filled_indices = filled_indices
        result.fill_values = fill_values
        result.per_point_confidence = per_point_conf
        result.gaps_filled = len(filled_indices)
        result.gaps_remaining = total_missing - result.gaps_filled
        result.confidence = _compute_confidence(
            result.r_squared, 1, "ratio",
        )

        result.details = {
            "overlap_points": len(indices),
            "avg_ratio": avg_ratio,
            "ratio_count": len(ratios),
            "correlation": result.correlation,
            "r_squared": result.r_squared,
        }

        # Metrics
        elapsed = time.time() - start
        result.processing_time_ms = elapsed * 1000.0

        if result.gaps_filled > 0:
            inc_gaps_filled("cross_series", result.gaps_filled)
            observe_confidence(result.confidence)
        observe_duration("cross_series_ratio", elapsed)

        # Provenance
        result.provenance_hash = self._record_provenance(
            operation="cross_series_ratio",
            donor_id=donor.series_id,
            target_length=len(target),
            overlap_points=len(indices),
            gaps_filled=result.gaps_filled,
            extras={
                "avg_ratio": avg_ratio,
                "ratio_count": len(ratios),
                "correlation": result.correlation,
            },
        )

        logger.info(
            "Ratio fill: donor='%s', r=%.4f, ratio=%.6f, "
            "filled=%d/%d, confidence=%.4f, %.1fms",
            donor.series_id, result.correlation, avg_ratio,
            result.gaps_filled, total_missing, result.confidence,
            result.processing_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Fill: donor matching (multi-donor consensus)
    # ------------------------------------------------------------------

    def fill_donor_matching(
        self,
        target: List[Optional[float]],
        donors: List[ReferenceSeries],
    ) -> CrossSeriesResult:
        """Fill gaps using weighted consensus from multiple donors.

        For each gap position, collects fill estimates from all
        donors whose correlation exceeds the configured threshold,
        then produces a final fill as the correlation-weighted
        average of predictions.

        Args:
            target: Target series with gaps (None for missing).
            donors: List of candidate donor series.

        Returns:
            CrossSeriesResult with multi-donor consensus fills.
        """
        start = time.time()
        total_missing = _count_missing(target)
        threshold = self._config.correlation_threshold
        result = CrossSeriesResult(original=list(target))

        # Pre-compute correlations and filter viable donors
        viable: List[Tuple[ReferenceSeries, float]] = []
        for ref in donors:
            r = self.compute_correlation(target, ref.values)
            abs_r = abs(r)
            if abs_r >= threshold:
                viable.append((ref, r))
                logger.debug(
                    "Viable donor '%s': r=%.4f", ref.series_id, r,
                )

        if not viable:
            logger.info(
                "No viable donors found (threshold=%.2f)", threshold,
            )
            result.values = list(target)
            result.gaps_remaining = total_missing
            result.processing_time_ms = (time.time() - start) * 1000.0
            return result

        # Pre-fit regression for each viable donor
        donor_fits = self._fit_all_donors(target, viable)

        if not donor_fits:
            logger.info("No donors with sufficient overlap for fitting")
            result.values = list(target)
            result.gaps_remaining = total_missing
            result.processing_time_ms = (time.time() - start) * 1000.0
            return result

        # Fill each gap position via weighted consensus
        filled = list(target)
        filled_indices: List[int] = []
        fill_values: Dict[int, float] = {}
        per_point_conf: Dict[int, float] = {}
        contributions: Dict[int, List[DonorContribution]] = {}
        donor_id_set: set = set()

        for i in range(len(target)):
            if not _is_missing(target[i]):
                continue

            contribs = self._collect_contributions(i, donor_fits)
            if not contribs:
                continue

            consensus, point_conf = self._weighted_consensus(contribs)

            filled[i] = consensus
            filled_indices.append(i)
            fill_values[i] = consensus
            per_point_conf[i] = point_conf
            contributions[i] = contribs
            for c in contribs:
                donor_id_set.add(c.donor_id)

        # Populate result
        result.values = filled
        result.filled_indices = filled_indices
        result.fill_values = fill_values
        result.per_point_confidence = per_point_conf
        result.gaps_filled = len(filled_indices)
        result.gaps_remaining = total_missing - result.gaps_filled
        result.donors_used = len(donor_id_set)
        result.donor_ids = sorted(donor_id_set)
        result.contributions = contributions

        if filled_indices:
            result.avg_confidence = (
                sum(per_point_conf.values()) / len(per_point_conf)
            )
        result.avg_correlation = (
            sum(abs(r) for _, r, _, _, _ in donor_fits) / len(donor_fits)
            if donor_fits else 0.0
        )

        result.details = {
            "threshold": threshold,
            "candidates_evaluated": len(donors),
            "viable_donors": len(viable),
            "donors_used": result.donors_used,
        }

        # Metrics
        elapsed = time.time() - start
        result.processing_time_ms = elapsed * 1000.0

        if result.gaps_filled > 0:
            inc_gaps_filled("cross_series", result.gaps_filled)
            observe_confidence(result.avg_confidence)
        observe_duration("cross_series_donor_matching", elapsed)

        # Provenance
        if self._config.enable_provenance:
            input_hash = self._provenance.build_hash({
                "method": "donor_matching",
                "target_length": len(target),
                "donor_count": len(donors),
                "viable_count": len(viable),
            })
            output_hash = self._provenance.build_hash({
                "gaps_filled": result.gaps_filled,
                "donors_used": result.donors_used,
                "avg_confidence": result.avg_confidence,
                "avg_correlation": result.avg_correlation,
            })
            result.provenance_hash = self._provenance.add_to_chain(
                operation="cross_series_donor_matching",
                input_hash=input_hash,
                output_hash=output_hash,
                metadata={
                    "donor_ids": result.donor_ids,
                    "donors_used": result.donors_used,
                    "gaps_filled": result.gaps_filled,
                    "avg_confidence": result.avg_confidence,
                },
            )

        logger.info(
            "Donor matching fill: %d donors, filled=%d/%d, "
            "avg_conf=%.4f, avg_corr=%.4f, %.1fms",
            result.donors_used, result.gaps_filled, total_missing,
            result.avg_confidence, result.avg_correlation,
            result.processing_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Similarity matrix
    # ------------------------------------------------------------------

    def compute_similarity_matrix(
        self,
        series_list: List[ReferenceSeries],
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise Pearson correlation matrix for all series.

        Returns a nested dictionary where
        ``matrix[series_a_id][series_b_id] = correlation``.
        The diagonal is always 1.0.

        Args:
            series_list: List of ReferenceSeries to compare.

        Returns:
            Nested dict ``{series_a_id: {series_b_id: correlation}}``.
        """
        start = time.time()
        n = len(series_list)
        matrix: Dict[str, Dict[str, float]] = {}

        # Initialise all entries to 0.0
        for ref in series_list:
            matrix[ref.series_id] = {}
            for other in series_list:
                matrix[ref.series_id][other.series_id] = 0.0

        # Compute upper triangle and mirror
        for i in range(n):
            sid_a = series_list[i].series_id
            matrix[sid_a][sid_a] = 1.0  # Diagonal

            for j in range(i + 1, n):
                sid_b = series_list[j].series_id
                r = self.compute_correlation(
                    series_list[i].values,
                    series_list[j].values,
                )
                matrix[sid_a][sid_b] = r
                matrix[sid_b][sid_a] = r

        elapsed = time.time() - start
        observe_duration("cross_series_similarity", elapsed)

        pairs_computed = n * (n - 1) // 2
        logger.info(
            "Similarity matrix: %d series, %d pairs, %.1fms",
            n, pairs_computed, elapsed * 1000.0,
        )

        # Provenance
        if self._config.enable_provenance:
            input_hash = self._provenance.build_hash({
                "operation": "similarity_matrix",
                "series_count": n,
                "series_ids": sorted(s.series_id for s in series_list),
            })
            output_hash = self._provenance.build_hash({
                "pairs_computed": pairs_computed,
            })
            self._provenance.add_to_chain(
                operation="cross_series_similarity_matrix",
                input_hash=input_hash,
                output_hash=output_hash,
                metadata={
                    "series_count": n,
                    "pairs_computed": pairs_computed,
                },
            )

        return matrix

    # ------------------------------------------------------------------
    # Convenience: auto-fill from registry
    # ------------------------------------------------------------------

    def auto_fill(
        self,
        target: List[Optional[float]],
        method: str = "regression",
    ) -> FillResult:
        """Fill gaps in target using the best registered donor.

        Searches the internal registry for the most highly correlated
        donor and applies the specified fill method.

        Args:
            target: Target series with gaps (None for missing).
            method: Fill method -- ``'regression'`` or ``'ratio'``.

        Returns:
            FillResult from the chosen method.

        Raises:
            ValueError: If method is not ``'regression'`` or ``'ratio'``.
        """
        if method not in ("regression", "ratio"):
            raise ValueError(
                f"method must be 'regression' or 'ratio', got '{method}'"
            )

        all_refs = list(self._registry.values())
        best = self.find_best_donor(target, all_refs)

        if best is None:
            logger.warning("auto_fill: no suitable donor in registry")
            return FillResult(
                method=method,
                values=list(target),
                original=list(target),
                gaps_remaining=_count_missing(target),
            )

        if method == "regression":
            return self.fill_regression(target, best)
        return self.fill_ratio(target, best)

    def auto_fill_consensus(
        self,
        target: List[Optional[float]],
    ) -> CrossSeriesResult:
        """Fill gaps using all suitable registered donors via consensus.

        Convenience wrapper around ``fill_donor_matching`` that uses
        the entire internal registry.

        Args:
            target: Target series with gaps (None for missing).

        Returns:
            CrossSeriesResult from multi-donor consensus.
        """
        all_refs = list(self._registry.values())
        return self.fill_donor_matching(target, all_refs)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnose_donor(
        self,
        target: List[Optional[float]],
        donor: ReferenceSeries,
    ) -> Dict[str, Any]:
        """Return diagnostic statistics for a target-donor pair.

        Useful for evaluating whether a donor series is suitable
        before committing to a fill.

        Args:
            target: Target series (may contain gaps).
            donor: Candidate donor series.

        Returns:
            Dictionary with diagnostic statistics including
            correlation, overlap_count, fillable_count, OLS fit
            parameters, and whether the threshold is exceeded.
        """
        indices = _overlap_indices(target, donor.values)
        correlation = 0.0
        slope = 0.0
        intercept = 0.0
        r_squared = 0.0

        if len(indices) >= 3:
            x_vals = [float(donor.values[i]) for i in indices]  # type: ignore[arg-type]
            y_vals = [float(target[i]) for i in indices]  # type: ignore[arg-type]
            correlation = _pearson_r(x_vals, y_vals)
            slope, intercept, r_squared = _ols_fit(x_vals, y_vals)

        target_gaps = _count_missing(target)
        donor_gaps = _count_missing(donor.values)

        # Fillable: target missing AND donor has a value
        length = min(len(target), len(donor.values))
        fillable = sum(
            1 for i in range(length)
            if _is_missing(target[i]) and not _is_missing(donor.values[i])
        )

        return {
            "donor_id": donor.series_id,
            "donor_name": donor.name,
            "correlation": correlation,
            "abs_correlation": abs(correlation),
            "overlap_count": len(indices),
            "target_length": len(target),
            "donor_length": len(donor.values),
            "target_gap_count": target_gaps,
            "donor_gap_count": donor_gaps,
            "fillable_count": fillable,
            "exceeds_threshold": (
                abs(correlation) >= self._config.correlation_threshold
            ),
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        }

    def rank_donors(
        self,
        target: List[Optional[float]],
        reference_series: List[ReferenceSeries],
    ) -> List[Dict[str, Any]]:
        """Rank candidate donors by absolute correlation with target.

        Args:
            target: Target series (may contain gaps).
            reference_series: Candidate donor series.

        Returns:
            List of diagnostic dicts sorted by descending ``|r|``.
        """
        diagnostics = [
            self.diagnose_donor(target, ref) for ref in reference_series
        ]
        diagnostics.sort(key=lambda d: d["abs_correlation"], reverse=True)
        return diagnostics

    # ------------------------------------------------------------------
    # Summary / health
    # ------------------------------------------------------------------

    def get_engine_summary(self) -> Dict[str, Any]:
        """Return a summary of the engine's current state.

        Returns:
            Dictionary with registry size, configuration thresholds,
            and provenance chain length.
        """
        return {
            "engine": "CrossSeriesFillerEngine",
            "registered_series": len(self._registry),
            "series_ids": sorted(self._registry.keys()),
            "correlation_threshold": self._config.correlation_threshold,
            "confidence_threshold": self._config.confidence_threshold,
            "cross_series_enabled": self._config.enable_cross_series,
            "provenance_enabled": self._config.enable_provenance,
            "provenance_chain_length": self._provenance.get_chain_length(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_all_donors(
        self,
        target: List[Optional[float]],
        viable: List[Tuple[ReferenceSeries, float]],
    ) -> List[Tuple[ReferenceSeries, float, float, float, float]]:
        """Fit OLS regression for each viable donor.

        Args:
            target: Target series.
            viable: List of (ReferenceSeries, correlation) tuples.

        Returns:
            List of (ref, r, slope, intercept, r_squared) tuples for
            donors with sufficient overlap (>= 3 points).
        """
        donor_fits: List[
            Tuple[ReferenceSeries, float, float, float, float]
        ] = []

        for ref, r in viable:
            indices = _overlap_indices(target, ref.values)
            if len(indices) < 3:
                logger.debug(
                    "Skipping donor '%s': only %d overlap points",
                    ref.series_id, len(indices),
                )
                continue

            x_vals = [float(ref.values[i]) for i in indices]  # type: ignore[arg-type]
            y_vals = [float(target[i]) for i in indices]  # type: ignore[arg-type]
            slope, intercept, r_sq = _ols_fit(x_vals, y_vals)
            donor_fits.append((ref, r, slope, intercept, r_sq))

        return donor_fits

    def _collect_contributions(
        self,
        index: int,
        donor_fits: List[Tuple[ReferenceSeries, float, float, float, float]],
    ) -> List[DonorContribution]:
        """Collect predictions from all donors at a specific index.

        Args:
            index: Position in the target series.
            donor_fits: Pre-fitted donor tuples from _fit_all_donors.

        Returns:
            List of DonorContribution for donors that have a value
            at the given index.
        """
        contribs: List[DonorContribution] = []

        for ref, r, slope, intercept, r_sq in donor_fits:
            if index >= len(ref.values):
                continue
            if _is_missing(ref.values[index]):
                continue

            donor_val = float(ref.values[index])  # type: ignore[arg-type]
            predicted = slope * donor_val + intercept

            contribs.append(DonorContribution(
                donor_id=ref.series_id,
                correlation=r,
                predicted_value=predicted,
                weight=abs(r),
            ))

        return contribs

    @staticmethod
    def _weighted_consensus(
        contribs: List[DonorContribution],
    ) -> Tuple[float, float]:
        """Compute weighted average and confidence from contributions.

        Args:
            contribs: Non-empty list of donor contributions.

        Returns:
            Tuple of (consensus_value, confidence).
        """
        total_weight = sum(c.weight for c in contribs)
        if total_weight == 0.0:
            # Fallback to simple average
            avg = sum(c.predicted_value for c in contribs) / len(contribs)
            return avg, 0.0

        weighted_sum = sum(
            c.predicted_value * c.weight for c in contribs
        )
        consensus = weighted_sum / total_weight

        # Confidence: average of weight-squared as R-squared proxy,
        # combined with donor count bonus
        avg_r_sq = sum(c.weight ** 2 for c in contribs) / len(contribs)
        point_conf = _compute_confidence(
            avg_r_sq, len(contribs), "donor_matching",
        )

        return consensus, point_conf

    def _record_provenance(
        self,
        operation: str,
        donor_id: str,
        target_length: int,
        overlap_points: int,
        gaps_filled: int,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a provenance entry for a fill operation.

        Args:
            operation: Operation name for the provenance chain.
            donor_id: Donor series identifier.
            target_length: Length of the target series.
            overlap_points: Number of overlapping non-missing points.
            gaps_filled: Number of gaps filled.
            extras: Additional metadata to include.

        Returns:
            Chain hash string, or empty string if provenance disabled.
        """
        if not self._config.enable_provenance:
            return ""

        input_hash = self._provenance.build_hash({
            "operation": operation,
            "donor_id": donor_id,
            "target_length": target_length,
            "overlap_points": overlap_points,
        })
        output_hash = self._provenance.build_hash({
            "gaps_filled": gaps_filled,
            **(extras or {}),
        })

        metadata = {
            "donor_id": donor_id,
            "gaps_filled": gaps_filled,
        }
        if extras:
            metadata.update(extras)

        return self._provenance.add_to_chain(
            operation=operation,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = [
    # Data models
    "ReferenceSeries",
    "FilledPoint",
    "FillResult",
    "DonorContribution",
    "CrossSeriesResult",
    # Helper functions
    "_overlap_indices",
    "_pearson_r",
    "_ols_fit",
    "_is_missing",
    "_utcnow",
    "_compute_confidence",
    "_count_missing",
    # Engine
    "CrossSeriesFillerEngine",
]
