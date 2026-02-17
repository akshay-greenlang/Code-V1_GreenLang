# -*- coding: utf-8 -*-
"""
Seasonal Gap Filling Engine - AGENT-DATA-014

Pure-Python seasonal decomposition and pattern-based gap filling for
time series data. Implements STL-style additive decomposition using
centred moving averages, autocorrelation-based seasonality detection,
calendar-aware filling, and day-of-week / month-of-year pattern fills.

Engine 4 of 7 in the Time Series Gap Filler Agent SDK.

Zero-Hallucination: All calculations use deterministic Python arithmetic
(math, statistics, datetime). No LLM calls for numeric computations.
No external numerical libraries required.

Example:
    >>> from greenlang.time_series_gap_filler.seasonal_filler import SeasonalFillerEngine
    >>> engine = SeasonalFillerEngine()
    >>> result = engine.fill_seasonal([1.0, None, 3.0, None, 5.0, 6.0, 7.0, None, 9.0, 10.0, 11.0, 12.0])
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

from greenlang.time_series_gap_filler.config import get_config
from greenlang.time_series_gap_filler.metrics import (
    inc_gaps_filled,
    observe_confidence,
    observe_duration,
)
from greenlang.time_series_gap_filler.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _is_missing(value: Any) -> bool:
    """Determine whether a value represents a missing data point.

    Treats None, float('nan'), and float('inf') as missing.

    Args:
        value: Value to check.

    Returns:
        True if the value is considered missing.
    """
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return False


# ---------------------------------------------------------------------------
# Lightweight data models (self-contained until models.py is built)
# ---------------------------------------------------------------------------


@dataclass
class SeasonalDecomposition:
    """Result of additive seasonal decomposition.

    Attributes:
        trend: Trend component (centred moving average). None entries
            appear at edges where the window cannot be fully centred.
        seasonal: Seasonal component repeating with the detected period.
        residual: Residual = original - trend - seasonal.
        period: The period length used for decomposition.
        original: The original time series values.
        provenance_hash: SHA-256 audit trail hash.
    """

    trend: List[Optional[float]] = field(default_factory=list)
    seasonal: List[float] = field(default_factory=list)
    residual: List[Optional[float]] = field(default_factory=list)
    period: int = 0
    original: List[Optional[float]] = field(default_factory=list)
    provenance_hash: str = ""


@dataclass
class FillResult:
    """Result of a gap filling operation.

    Attributes:
        values: The series with gaps filled.
        original: The original input series.
        filled_indices: Indices where gaps were filled.
        fill_values: Mapping of index to the value used for filling.
        method: Name of the fill method used.
        confidence: Overall confidence score (0.0-1.0).
        per_point_confidence: Per-index confidence scores.
        gaps_filled: Total number of gaps filled.
        gaps_remaining: Number of gaps that could not be filled.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 audit trail hash.
        details: Additional method-specific metadata.
    """

    values: List[Optional[float]] = field(default_factory=list)
    original: List[Optional[float]] = field(default_factory=list)
    filled_indices: List[int] = field(default_factory=list)
    fill_values: Dict[int, float] = field(default_factory=dict)
    method: str = ""
    confidence: float = 0.0
    per_point_confidence: Dict[int, float] = field(default_factory=dict)
    gaps_filled: int = 0
    gaps_remaining: int = 0
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalendarDefinition:
    """Calendar specification for calendar-aware gap filling.

    Attributes:
        business_days: Weekday numbers considered business days
            (0=Monday through 6=Sunday). Default Mon-Fri.
        holidays: Set of dates (YYYY-MM-DD strings) that are holidays.
        fiscal_periods: Mapping of period name to (start_month, end_month).
    """

    business_days: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4],
    )
    holidays: List[str] = field(default_factory=list)
    fiscal_periods: Dict[str, Tuple[int, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Private computational helpers
# ---------------------------------------------------------------------------


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


def _centered_moving_average(
    values: List[Optional[float]],
    window: int,
) -> List[Optional[float]]:
    """Compute a centred moving average for a time series.

    For even window sizes the result is centred by averaging two
    consecutive simple moving averages (2xMA approach for STL).

    Args:
        values: Time series values (may contain None for missing).
        window: Moving average window length.

    Returns:
        List of same length as *values* with None at edges where
        the centred window cannot be fully computed.
    """
    n = len(values)
    if n == 0 or window <= 0:
        return [None] * n

    result: List[Optional[float]] = [None] * n

    # Simple moving average first pass
    sma: List[Optional[float]] = [None] * n
    half = window // 2

    for i in range(n):
        lo = i - half
        hi = i + half
        # For even windows, shift to make centring work
        if window % 2 == 0:
            hi -= 1

        if lo < 0 or hi >= n:
            sma[i] = None
            continue

        segment = values[lo:hi + 1]
        valid = [v for v in segment if not _is_missing(v)]
        if len(valid) < len(segment) * 0.5:
            sma[i] = None
        else:
            sma[i] = sum(valid) / len(valid)

    if window % 2 == 1:
        # Odd window: SMA is already centred
        return sma

    # Even window: average adjacent SMAs (2xMA centring)
    for i in range(n):
        if i + 1 >= n:
            result[i] = None
            continue
        a = sma[i]
        b = sma[i + 1]
        if a is not None and b is not None:
            result[i] = (a + b) / 2.0
        else:
            result[i] = None

    return result


def _autocorrelation(values: List[float], lag: int) -> float:
    """Compute the autocorrelation of *values* at the given *lag*.

    Uses the standard Pearson correlation between the series and
    its lagged version. Returns 0.0 if the series is too short or
    has zero variance.

    Args:
        values: List of numeric values (no missing values).
        lag: Lag in number of observations (must be >= 1).

    Returns:
        Autocorrelation coefficient in the range [-1.0, 1.0].
    """
    n = len(values)
    if lag < 1 or lag >= n:
        return 0.0

    mean_val = _safe_mean(values)
    var_sum = sum((v - mean_val) ** 2 for v in values)
    if var_sum == 0.0:
        return 0.0

    cov_sum = 0.0
    for i in range(n - lag):
        cov_sum += (values[i] - mean_val) * (values[i + lag] - mean_val)

    return cov_sum / var_sum


def _detrend(
    values: List[Optional[float]],
    trend: List[Optional[float]],
) -> List[Optional[float]]:
    """Subtract trend from original values (additive decomposition).

    Args:
        values: Original time series.
        trend: Trend component.

    Returns:
        Detrended series (values - trend), with None where either
        input is None.
    """
    result: List[Optional[float]] = []
    for v, t in zip(values, trend):
        if _is_missing(v) or _is_missing(t):
            result.append(None)
        else:
            result.append(v - t)  # type: ignore[operator]
    return result


def _compute_confidence(num_cycles: int, method: str) -> float:
    """Compute a fill confidence score based on available seasonal cycles.

    More complete cycles yield higher confidence. The *method* name
    adjusts the baseline since some methods are inherently more
    reliable.

    Args:
        num_cycles: Number of complete seasonal cycles of non-missing data.
        method: Fill method name (seasonal, calendar, day_of_week,
            month_pattern).

    Returns:
        Confidence score in the range [0.0, 1.0].
    """
    # Base confidence from cycle count (diminishing returns)
    if num_cycles <= 0:
        base = 0.2
    elif num_cycles == 1:
        base = 0.45
    elif num_cycles == 2:
        base = 0.60
    elif num_cycles == 3:
        base = 0.72
    elif num_cycles <= 5:
        base = 0.80
    else:
        base = min(0.95, 0.80 + 0.03 * (num_cycles - 5))

    # Method reliability adjustment
    method_bonus: Dict[str, float] = {
        "seasonal": 0.0,
        "calendar": 0.02,
        "day_of_week": -0.05,
        "month_pattern": 0.03,
    }
    adjustment = method_bonus.get(method, 0.0)
    return max(0.0, min(1.0, base + adjustment))


# ===========================================================================
# SeasonalFillerEngine
# ===========================================================================


class SeasonalFillerEngine:
    """Pure-Python seasonal decomposition and pattern-based gap filler.

    Implements STL-style additive decomposition, autocorrelation-based
    seasonality detection, calendar-aware filling, and day-of-week /
    month-of-year pattern fills. All arithmetic is deterministic Python
    (zero-hallucination).

    Attributes:
        _config: Time Series Gap Filler configuration singleton.
        _provenance: SHA-256 provenance tracker for audit trails.

    Example:
        >>> engine = SeasonalFillerEngine()
        >>> result = engine.fill_seasonal([10, None, 30, 40, 50, None, 70, 80, 90, None, 110, 120])
        >>> assert result.gaps_filled > 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize SeasonalFillerEngine.

        Args:
            config: Optional TimeSeriesGapFillerConfig override.
                Falls back to the singleton from ``get_config()``.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        logger.info("SeasonalFillerEngine initialized")

    # ------------------------------------------------------------------
    # 1. decompose_seasonal
    # ------------------------------------------------------------------

    def decompose_seasonal(
        self,
        values: List[Optional[float]],
        period: Optional[int] = None,
    ) -> SeasonalDecomposition:
        """Perform STL-style additive seasonal decomposition.

        Decomposes *values* into trend, seasonal, and residual
        components using centred moving averages.

        If *period* is None, it is auto-detected via autocorrelation.
        The trend is computed as a centred moving average of length
        equal to the period. The seasonal component is the average
        of detrended values at each seasonal position. The residual
        is original - trend - seasonal.

        Args:
            values: Time series values (None or NaN for missing).
            period: Seasonal period length. Auto-detected when None.

        Returns:
            SeasonalDecomposition with trend, seasonal, residual.

        Raises:
            ValueError: If the series is too short for decomposition.
        """
        start = time.time()
        n = len(values)

        if n < 4:
            raise ValueError(
                f"Series too short for decomposition ({n} points, need >= 4)"
            )

        # Resolve period
        effective_period = period or self._auto_detect_period(values)
        effective_period = max(2, min(effective_period, n // 2))

        logger.debug(
            "decompose_seasonal: n=%d, period=%d", n, effective_period,
        )

        # Step 1: Trend via centred moving average
        trend = _centered_moving_average(values, effective_period)

        # Step 2: Detrend
        detrended = _detrend(values, trend)

        # Step 3: Seasonal component - average at each position
        seasonal_pattern = self._compute_seasonal_pattern(
            detrended, effective_period,
        )

        # Tile the pattern across the full series
        seasonal = [
            seasonal_pattern[i % effective_period] for i in range(n)
        ]

        # Step 4: Residual = original - trend - seasonal
        residual: List[Optional[float]] = []
        for i in range(n):
            if _is_missing(values[i]) or _is_missing(trend[i]):
                residual.append(None)
            else:
                residual.append(
                    values[i] - trend[i] - seasonal[i]  # type: ignore[operator]
                )

        # Provenance
        provenance_hash = self._provenance.build_hash({
            "operation": "decompose_seasonal",
            "n": n,
            "period": effective_period,
            "pattern": seasonal_pattern,
        })
        self._provenance.record(
            "seasonal_decomposition", f"p{effective_period}_n{n}",
            "decompose", provenance_hash,
        )

        elapsed = time.time() - start
        observe_duration("decompose_seasonal", elapsed)
        logger.info(
            "Seasonal decomposition complete: n=%d, period=%d, %.3fs",
            n, effective_period, elapsed,
        )

        return SeasonalDecomposition(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            period=effective_period,
            original=list(values),
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # 2. fill_seasonal
    # ------------------------------------------------------------------

    def fill_seasonal(
        self,
        values: List[Optional[float]],
        period: Optional[int] = None,
        timestamps: Optional[List[datetime]] = None,
    ) -> FillResult:
        """Fill gaps using the seasonal pattern from non-missing data.

        Decomposes the series to extract seasonal and trend components.
        For each gap, the fill value is the seasonal component at that
        position plus a linearly interpolated trend estimate.

        Confidence is computed from the number of complete seasonal
        cycles available in the non-missing data.

        Args:
            values: Time series values (None or NaN for missing).
            period: Seasonal period. Auto-detected when None.
            timestamps: Optional timestamps for provenance only.

        Returns:
            FillResult with filled values, confidence, and provenance.
        """
        start = time.time()
        n = len(values)
        method_name = "seasonal"

        # Identify gaps
        gap_indices = [i for i in range(n) if _is_missing(values[i])]
        if not gap_indices:
            return self._empty_fill_result(values, method_name, start)

        # Need sufficient non-missing data
        valid_count = n - len(gap_indices)
        if valid_count < self._config.min_data_points:
            logger.warning(
                "fill_seasonal: insufficient data (%d valid, need %d)",
                valid_count, self._config.min_data_points,
            )
            return self._insufficient_fill_result(
                values, method_name, gap_indices, start,
            )

        # Resolve period
        effective_period = period or self._auto_detect_period(values)
        effective_period = max(2, min(effective_period, n // 2))

        # Extract seasonal pattern from valid data only
        seasonal_pattern = self.get_seasonal_pattern(values, effective_period)

        # Compute trend anchors from non-missing data for interpolation
        trend_anchors = self._build_trend_anchors(values, effective_period)

        # Fill gaps
        filled = list(values)
        fill_values: Dict[int, float] = {}
        per_point_confidence: Dict[int, float] = {}

        num_cycles = valid_count // effective_period

        for idx in gap_indices:
            pos = idx % effective_period
            seasonal_val = seasonal_pattern[pos]

            # Interpolate trend at this position
            trend_val = self._interpolate_trend_at(idx, trend_anchors, n)

            fill_val = seasonal_val + trend_val
            filled[idx] = fill_val
            fill_values[idx] = fill_val

            # Per-point confidence: weaker at edges, stronger in middle
            point_conf = _compute_confidence(num_cycles, method_name)
            edge_penalty = self._edge_penalty(idx, n)
            per_point_confidence[idx] = max(
                0.0, min(1.0, point_conf * edge_penalty),
            )

        gaps_filled = len(fill_values)
        overall_confidence = _compute_confidence(num_cycles, method_name)

        # Record metrics
        inc_gaps_filled(method_name, gaps_filled)
        observe_confidence(overall_confidence)

        # Provenance
        provenance_hash = self._provenance.build_hash({
            "operation": "fill_seasonal",
            "n": n,
            "period": effective_period,
            "gaps_filled": gaps_filled,
            "confidence": overall_confidence,
        })
        self._provenance.record(
            "seasonal_fill", f"p{effective_period}_n{n}",
            "fill", provenance_hash,
        )

        elapsed = time.time() - start
        observe_duration("fill_seasonal", elapsed)

        logger.info(
            "fill_seasonal: filled %d/%d gaps, period=%d, conf=%.3f, %.3fs",
            gaps_filled, len(gap_indices), effective_period,
            overall_confidence, elapsed,
        )

        return FillResult(
            values=filled,
            original=list(values),
            filled_indices=list(fill_values.keys()),
            fill_values=fill_values,
            method=method_name,
            confidence=overall_confidence,
            per_point_confidence=per_point_confidence,
            gaps_filled=gaps_filled,
            gaps_remaining=len(gap_indices) - gaps_filled,
            processing_time_ms=elapsed * 1000.0,
            provenance_hash=provenance_hash,
            details={
                "period": effective_period,
                "num_cycles": num_cycles,
                "seasonal_pattern": seasonal_pattern,
            },
        )

    # ------------------------------------------------------------------
    # 3. fill_calendar_aware
    # ------------------------------------------------------------------

    def fill_calendar_aware(
        self,
        values: List[Optional[float]],
        timestamps: List[datetime],
        calendar: CalendarDefinition,
    ) -> FillResult:
        """Fill gaps using calendar-aware patterns.

        Uses the CalendarDefinition to recognise business days, holidays,
        and fiscal periods. Builds day-of-week and month-of-year
        averages from non-missing business-day data, then fills gaps
        on business days using those averages. Non-business days and
        holidays are filled with nearby business-day averages scaled
        by a holiday factor.

        Args:
            values: Time series values (None or NaN for missing).
            timestamps: Corresponding timestamps (same length as values).
            calendar: CalendarDefinition with business_days, holidays,
                and fiscal_periods.

        Returns:
            FillResult with calendar-aware filled values.

        Raises:
            ValueError: If lengths of values and timestamps differ.
        """
        start = time.time()
        method_name = "calendar"
        n = len(values)

        if n != len(timestamps):
            raise ValueError(
                f"values length ({n}) != timestamps length ({len(timestamps)})"
            )

        gap_indices = [i for i in range(n) if _is_missing(values[i])]
        if not gap_indices:
            return self._empty_fill_result(values, method_name, start)

        holiday_set = set(calendar.holidays)
        biz_days = set(calendar.business_days)

        # Build day-of-week averages from non-missing business days
        dow_sums: Dict[int, List[float]] = {d: [] for d in range(7)}
        month_sums: Dict[int, List[float]] = {m: [] for m in range(1, 13)}

        for i in range(n):
            if _is_missing(values[i]):
                continue
            ts = timestamps[i]
            weekday = ts.weekday()
            ts_str = ts.strftime("%Y-%m-%d")

            if weekday in biz_days and ts_str not in holiday_set:
                dow_sums[weekday].append(float(values[i]))  # type: ignore[arg-type]
                month_sums[ts.month].append(float(values[i]))  # type: ignore[arg-type]

        dow_avg: Dict[int, float] = {}
        for d, vals in dow_sums.items():
            dow_avg[d] = _safe_mean(vals) if vals else 0.0

        month_avg: Dict[int, float] = {}
        for m, vals in month_sums.items():
            month_avg[m] = _safe_mean(vals) if vals else 0.0

        # Global mean for fallback
        all_valid = [
            float(v) for v in values if not _is_missing(v)  # type: ignore[arg-type]
        ]
        global_mean = _safe_mean(all_valid)

        # Fill gaps
        filled = list(values)
        fill_values: Dict[int, float] = {}
        per_point_confidence: Dict[int, float] = {}

        total_biz_samples = sum(len(v) for v in dow_sums.values())
        num_cycles = total_biz_samples // max(len(biz_days), 1) // 4

        for idx in gap_indices:
            ts = timestamps[idx]
            weekday = ts.weekday()
            month = ts.month
            ts_str = ts.strftime("%Y-%m-%d")
            is_holiday = ts_str in holiday_set
            is_business = weekday in biz_days and not is_holiday

            if is_business:
                # Use day-of-week average weighted with month average
                dw = dow_avg.get(weekday, global_mean)
                ma = month_avg.get(month, global_mean)
                fill_val = 0.6 * dw + 0.4 * ma if dw != 0.0 else ma
                conf = _compute_confidence(num_cycles, method_name)
            elif is_holiday:
                # Holiday: use reduced business-day average
                ma = month_avg.get(month, global_mean)
                fill_val = ma * 0.3  # holidays typically lower activity
                conf = _compute_confidence(num_cycles, method_name) * 0.7
            else:
                # Non-business day
                ma = month_avg.get(month, global_mean)
                fill_val = ma * 0.2
                conf = _compute_confidence(num_cycles, method_name) * 0.5

            filled[idx] = fill_val
            fill_values[idx] = fill_val
            per_point_confidence[idx] = max(0.0, min(1.0, conf))

        gaps_filled = len(fill_values)
        overall_confidence = _compute_confidence(num_cycles, method_name)

        inc_gaps_filled(method_name, gaps_filled)
        observe_confidence(overall_confidence)

        provenance_hash = self._provenance.build_hash({
            "operation": "fill_calendar_aware",
            "n": n,
            "gaps_filled": gaps_filled,
            "business_days": sorted(biz_days),
            "num_holidays": len(holiday_set),
        })
        self._provenance.record(
            "calendar_fill", f"cal_n{n}",
            "fill", provenance_hash,
        )

        elapsed = time.time() - start
        observe_duration("fill_calendar_aware", elapsed)

        logger.info(
            "fill_calendar_aware: filled %d/%d gaps, conf=%.3f, %.3fs",
            gaps_filled, len(gap_indices), overall_confidence, elapsed,
        )

        return FillResult(
            values=filled,
            original=list(values),
            filled_indices=list(fill_values.keys()),
            fill_values=fill_values,
            method=method_name,
            confidence=overall_confidence,
            per_point_confidence=per_point_confidence,
            gaps_filled=gaps_filled,
            gaps_remaining=len(gap_indices) - gaps_filled,
            processing_time_ms=elapsed * 1000.0,
            provenance_hash=provenance_hash,
            details={
                "dow_averages": dow_avg,
                "month_averages": month_avg,
                "business_days": sorted(biz_days),
                "num_holidays": len(holiday_set),
            },
        )

    # ------------------------------------------------------------------
    # 4. detect_seasonality
    # ------------------------------------------------------------------

    def detect_seasonality(
        self,
        values: List[Optional[float]],
        max_period: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Detect seasonal patterns via autocorrelation analysis.

        Computes the autocorrelation function for lags from 2 up to
        *max_period* (default half the series length) and identifies
        peaks that exceed a significance threshold.

        Supports detection of multiple overlapping seasonal patterns
        (e.g. weekly + annual).

        Args:
            values: Time series values (None or NaN for missing).
            max_period: Maximum lag to check. Defaults to n // 2.

        Returns:
            Dict with keys:
                - detected_periods: List of detected period lengths.
                - strengths: Dict mapping period to autocorrelation strength.
                - confidence: Overall confidence in detection (0.0-1.0).
                - acf_values: Dict mapping lag to autocorrelation value.
                - dominant_period: Period with the highest strength.
                - is_seasonal: Whether any significant seasonality was found.
                - provenance_hash: SHA-256 hash.
        """
        start = time.time()

        # Strip missing values for autocorrelation
        clean = [float(v) for v in values if not _is_missing(v)]  # type: ignore[arg-type]
        n = len(clean)

        if n < 6:
            return self._no_seasonality_result(values, start)

        max_lag = max_period if max_period else n // 2
        max_lag = min(max_lag, n // 2)
        max_lag = max(max_lag, 2)

        # Compute ACF
        acf_values: Dict[int, float] = {}
        for lag in range(1, max_lag + 1):
            acf_values[lag] = _autocorrelation(clean, lag)

        # Significance threshold: 2 / sqrt(n)
        significance = 2.0 / math.sqrt(n) if n > 0 else 0.5

        # Find peaks in ACF (local maxima above significance)
        peaks = self._find_acf_peaks(acf_values, significance)

        # Rank by strength
        strengths: Dict[int, float] = {}
        for period_val in peaks:
            strengths[period_val] = acf_values.get(period_val, 0.0)

        # Sort periods by strength descending
        detected_periods = sorted(
            peaks, key=lambda p: strengths.get(p, 0.0), reverse=True,
        )

        dominant_period = detected_periods[0] if detected_periods else 0
        is_seasonal = len(detected_periods) > 0

        # Confidence based on strongest ACF value
        max_strength = max(strengths.values()) if strengths else 0.0
        confidence = min(1.0, max_strength * 1.5) if is_seasonal else 0.0

        provenance_hash = self._provenance.build_hash({
            "operation": "detect_seasonality",
            "n_original": len(values),
            "n_clean": n,
            "max_lag": max_lag,
            "detected_periods": detected_periods,
        })
        self._provenance.record(
            "seasonality_detection", f"n{n}_ml{max_lag}",
            "detect", provenance_hash,
        )

        elapsed = time.time() - start
        observe_duration("detect_seasonality", elapsed)

        logger.info(
            "detect_seasonality: n=%d, periods=%s, dominant=%d, conf=%.3f, %.3fs",
            n, detected_periods, dominant_period, confidence, elapsed,
        )

        return {
            "detected_periods": detected_periods,
            "strengths": strengths,
            "confidence": confidence,
            "acf_values": acf_values,
            "dominant_period": dominant_period,
            "is_seasonal": is_seasonal,
            "significance_threshold": significance,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # 5. get_seasonal_pattern
    # ------------------------------------------------------------------

    def get_seasonal_pattern(
        self,
        values: List[Optional[float]],
        period: int,
    ) -> List[float]:
        """Extract the repeating seasonal pattern of given period length.

        Averages values at each position across all complete cycles.
        Missing values are excluded from the average at that position.
        If no valid observations exist for a position, it receives 0.0.

        Args:
            values: Time series values (None or NaN for missing).
            period: Period length (must be >= 2).

        Returns:
            List of *period* floats representing one seasonal cycle.
        """
        period = max(2, period)
        buckets: Dict[int, List[float]] = {p: [] for p in range(period)}

        for i, v in enumerate(values):
            if not _is_missing(v):
                pos = i % period
                buckets[pos].append(float(v))  # type: ignore[arg-type]

        pattern: List[float] = []
        for p in range(period):
            vals = buckets[p]
            pattern.append(_safe_mean(vals))

        # Centre the pattern (subtract mean so seasonal sums to ~0)
        pattern_mean = _safe_mean(pattern)
        centred = [v - pattern_mean for v in pattern]

        logger.debug(
            "get_seasonal_pattern: period=%d, raw_mean=%.4f", period, pattern_mean,
        )
        return centred

    # ------------------------------------------------------------------
    # 6. fill_day_of_week_pattern
    # ------------------------------------------------------------------

    def fill_day_of_week_pattern(
        self,
        values: List[Optional[float]],
        timestamps: List[datetime],
    ) -> FillResult:
        """Fill gaps using average value for each day of week (Mon-Sun).

        Computes the mean value observed on each weekday (0=Monday
        through 6=Sunday) from non-missing data, then fills each gap
        with the corresponding weekday average.

        Best suited for daily data with weekly patterns (e.g. energy
        consumption, traffic).

        Args:
            values: Time series values (None or NaN for missing).
            timestamps: Corresponding timestamps (same length).

        Returns:
            FillResult with weekday-pattern-filled values.

        Raises:
            ValueError: If lengths of values and timestamps differ.
        """
        start = time.time()
        method_name = "day_of_week"
        n = len(values)

        if n != len(timestamps):
            raise ValueError(
                f"values length ({n}) != timestamps length ({len(timestamps)})"
            )

        gap_indices = [i for i in range(n) if _is_missing(values[i])]
        if not gap_indices:
            return self._empty_fill_result(values, method_name, start)

        # Build day-of-week averages
        dow_buckets: Dict[int, List[float]] = {d: [] for d in range(7)}
        for i in range(n):
            if not _is_missing(values[i]):
                dow = timestamps[i].weekday()
                dow_buckets[dow].append(float(values[i]))  # type: ignore[arg-type]

        dow_avg: Dict[int, float] = {}
        for d in range(7):
            dow_avg[d] = _safe_mean(dow_buckets[d])

        # Count weeks for confidence
        total_obs = sum(len(v) for v in dow_buckets.values())
        num_weeks = total_obs // 7

        # Fill gaps
        filled = list(values)
        fill_values: Dict[int, float] = {}
        per_point_confidence: Dict[int, float] = {}

        for idx in gap_indices:
            dow = timestamps[idx].weekday()
            fill_val = dow_avg.get(dow, 0.0)
            filled[idx] = fill_val
            fill_values[idx] = fill_val

            dow_count = len(dow_buckets.get(dow, []))
            per_point_confidence[idx] = _compute_confidence(
                dow_count, method_name,
            )

        gaps_filled = len(fill_values)
        overall_confidence = _compute_confidence(num_weeks, method_name)

        inc_gaps_filled(method_name, gaps_filled)
        observe_confidence(overall_confidence)

        provenance_hash = self._provenance.build_hash({
            "operation": "fill_day_of_week_pattern",
            "n": n,
            "gaps_filled": gaps_filled,
            "dow_averages": dow_avg,
        })
        self._provenance.record(
            "dow_fill", f"dow_n{n}",
            "fill", provenance_hash,
        )

        elapsed = time.time() - start
        observe_duration("fill_day_of_week_pattern", elapsed)

        logger.info(
            "fill_day_of_week_pattern: filled %d/%d gaps, conf=%.3f, %.3fs",
            gaps_filled, len(gap_indices), overall_confidence, elapsed,
        )

        return FillResult(
            values=filled,
            original=list(values),
            filled_indices=list(fill_values.keys()),
            fill_values=fill_values,
            method=method_name,
            confidence=overall_confidence,
            per_point_confidence=per_point_confidence,
            gaps_filled=gaps_filled,
            gaps_remaining=len(gap_indices) - gaps_filled,
            processing_time_ms=elapsed * 1000.0,
            provenance_hash=provenance_hash,
            details={
                "dow_averages": dow_avg,
                "num_weeks": num_weeks,
                "dow_sample_counts": {
                    d: len(v) for d, v in dow_buckets.items()
                },
            },
        )

    # ------------------------------------------------------------------
    # 7. fill_month_pattern
    # ------------------------------------------------------------------

    def fill_month_pattern(
        self,
        values: List[Optional[float]],
        timestamps: List[datetime],
    ) -> FillResult:
        """Fill gaps using average value for each month (Jan-Dec).

        Computes the mean value observed in each calendar month from
        non-missing data, then fills each gap with the corresponding
        month average.

        Best suited for monthly or sub-monthly data with annual
        patterns (e.g. heating fuel, agricultural cycles).

        Args:
            values: Time series values (None or NaN for missing).
            timestamps: Corresponding timestamps (same length).

        Returns:
            FillResult with month-pattern-filled values.

        Raises:
            ValueError: If lengths of values and timestamps differ.
        """
        start = time.time()
        method_name = "month_pattern"
        n = len(values)

        if n != len(timestamps):
            raise ValueError(
                f"values length ({n}) != timestamps length ({len(timestamps)})"
            )

        gap_indices = [i for i in range(n) if _is_missing(values[i])]
        if not gap_indices:
            return self._empty_fill_result(values, method_name, start)

        # Build month averages
        month_buckets: Dict[int, List[float]] = {m: [] for m in range(1, 13)}
        for i in range(n):
            if not _is_missing(values[i]):
                month = timestamps[i].month
                month_buckets[month].append(float(values[i]))  # type: ignore[arg-type]

        month_avg: Dict[int, float] = {}
        for m in range(1, 13):
            month_avg[m] = _safe_mean(month_buckets[m])

        # Count years for confidence
        years_seen: set = set()
        for i in range(n):
            if not _is_missing(values[i]):
                years_seen.add(timestamps[i].year)
        num_years = len(years_seen)

        # Fill gaps
        filled = list(values)
        fill_values: Dict[int, float] = {}
        per_point_confidence: Dict[int, float] = {}

        for idx in gap_indices:
            month = timestamps[idx].month
            fill_val = month_avg.get(month, 0.0)
            filled[idx] = fill_val
            fill_values[idx] = fill_val

            month_count = len(month_buckets.get(month, []))
            per_point_confidence[idx] = _compute_confidence(
                month_count, method_name,
            )

        gaps_filled = len(fill_values)
        overall_confidence = _compute_confidence(num_years, method_name)

        inc_gaps_filled(method_name, gaps_filled)
        observe_confidence(overall_confidence)

        provenance_hash = self._provenance.build_hash({
            "operation": "fill_month_pattern",
            "n": n,
            "gaps_filled": gaps_filled,
            "month_averages": month_avg,
            "years_seen": sorted(years_seen),
        })
        self._provenance.record(
            "month_fill", f"month_n{n}",
            "fill", provenance_hash,
        )

        elapsed = time.time() - start
        observe_duration("fill_month_pattern", elapsed)

        logger.info(
            "fill_month_pattern: filled %d/%d gaps, years=%d, conf=%.3f, %.3fs",
            gaps_filled, len(gap_indices), num_years,
            overall_confidence, elapsed,
        )

        return FillResult(
            values=filled,
            original=list(values),
            filled_indices=list(fill_values.keys()),
            fill_values=fill_values,
            method=method_name,
            confidence=overall_confidence,
            per_point_confidence=per_point_confidence,
            gaps_filled=gaps_filled,
            gaps_remaining=len(gap_indices) - gaps_filled,
            processing_time_ms=elapsed * 1000.0,
            provenance_hash=provenance_hash,
            details={
                "month_averages": month_avg,
                "num_years": num_years,
                "month_sample_counts": {
                    m: len(v) for m, v in month_buckets.items()
                },
            },
        )

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _auto_detect_period(
        self,
        values: List[Optional[float]],
    ) -> int:
        """Auto-detect the seasonal period via autocorrelation peaks.

        Falls back to ``config.seasonal_periods`` if detection fails
        or returns an implausible result.

        Args:
            values: Time series with possible missing values.

        Returns:
            Detected period length (>= 2).
        """
        clean = [float(v) for v in values if not _is_missing(v)]  # type: ignore[arg-type]
        n = len(clean)
        default_period = self._config.seasonal_periods

        if n < 6:
            logger.debug(
                "_auto_detect_period: too few points (%d), using default %d",
                n, default_period,
            )
            return default_period

        max_lag = n // 2
        best_lag = 0
        best_acf = -1.0

        for lag in range(2, max_lag + 1):
            acf = _autocorrelation(clean, lag)
            if acf > best_acf:
                best_acf = acf
                best_lag = lag

        # Require the best lag to be meaningfully positive
        significance = 2.0 / math.sqrt(n) if n > 0 else 0.5
        if best_acf < significance or best_lag < 2:
            logger.debug(
                "_auto_detect_period: no significant peak (best=%.3f at lag %d), "
                "using default %d",
                best_acf, best_lag, default_period,
            )
            return default_period

        logger.debug(
            "_auto_detect_period: detected period=%d (acf=%.3f)",
            best_lag, best_acf,
        )
        return best_lag

    def _compute_seasonal_pattern(
        self,
        detrended: List[Optional[float]],
        period: int,
    ) -> List[float]:
        """Compute the seasonal pattern from detrended values.

        Averages valid detrended values at each seasonal position
        to produce a single-cycle pattern.

        Args:
            detrended: Detrended time series (may contain None).
            period: Seasonal period length.

        Returns:
            List of *period* floats representing one centred cycle.
        """
        buckets: Dict[int, List[float]] = {p: [] for p in range(period)}

        for i, v in enumerate(detrended):
            if not _is_missing(v):
                pos = i % period
                buckets[pos].append(float(v))  # type: ignore[arg-type]

        pattern: List[float] = []
        for p in range(period):
            vals = buckets[p]
            pattern.append(_safe_mean(vals))

        # Centre the pattern
        pattern_mean = _safe_mean(pattern)
        return [v - pattern_mean for v in pattern]

    def _build_trend_anchors(
        self,
        values: List[Optional[float]],
        period: int,
    ) -> List[Tuple[int, float]]:
        """Build trend anchor points from non-missing data.

        Computes a local average around each non-missing point to
        produce smoothed trend anchors for linear interpolation
        at gap positions.

        Args:
            values: Original time series.
            period: Seasonal period (used as smoothing window).

        Returns:
            Sorted list of (index, trend_value) tuples.
        """
        n = len(values)
        half = max(1, period // 2)
        anchors: List[Tuple[int, float]] = []

        for i in range(n):
            if _is_missing(values[i]):
                continue
            # Local window average
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            segment = [
                float(values[j])  # type: ignore[arg-type]
                for j in range(lo, hi)
                if not _is_missing(values[j])
            ]
            if segment:
                anchors.append((i, _safe_mean(segment)))

        return anchors

    def _interpolate_trend_at(
        self,
        idx: int,
        anchors: List[Tuple[int, float]],
        n: int,
    ) -> float:
        """Linearly interpolate a trend value at a given index.

        Finds the two nearest anchor points bracketing *idx* and
        performs linear interpolation. Extrapolates from the nearest
        single anchor if *idx* is outside all anchors.

        Args:
            idx: Index to interpolate at.
            anchors: Sorted (index, value) anchor points.
            n: Total series length.

        Returns:
            Interpolated trend value.
        """
        if not anchors:
            return 0.0

        # Find bracketing anchors
        left: Optional[Tuple[int, float]] = None
        right: Optional[Tuple[int, float]] = None

        for anchor_idx, anchor_val in anchors:
            if anchor_idx <= idx:
                left = (anchor_idx, anchor_val)
            if anchor_idx >= idx and right is None:
                right = (anchor_idx, anchor_val)

        if left is None and right is None:
            return 0.0
        if left is None:
            return right[1]  # type: ignore[index]
        if right is None:
            return left[1]
        if left[0] == right[0]:
            return left[1]

        # Linear interpolation
        span = right[0] - left[0]
        frac = (idx - left[0]) / span
        return left[1] + frac * (right[1] - left[1])

    def _find_acf_peaks(
        self,
        acf_values: Dict[int, float],
        significance: float,
    ) -> List[int]:
        """Find significant peaks in the autocorrelation function.

        A lag is a peak if its ACF value exceeds the significance
        threshold and is greater than its immediate neighbours.

        Args:
            acf_values: Mapping of lag to ACF value.
            significance: Minimum ACF value for significance.

        Returns:
            List of lag values that are significant peaks.
        """
        lags = sorted(acf_values.keys())
        peaks: List[int] = []

        for i, lag in enumerate(lags):
            # Period-1 is meaningless for seasonality detection; skip it.
            if lag < 2:
                continue

            val = acf_values[lag]
            if val < significance:
                continue

            # Check left neighbour
            if i > 0:
                left_val = acf_values[lags[i - 1]]
                if val < left_val:
                    continue

            # Check right neighbour
            if i < len(lags) - 1:
                right_val = acf_values[lags[i + 1]]
                if val < right_val:
                    continue

            peaks.append(lag)

        return peaks

    @staticmethod
    def _edge_penalty(idx: int, n: int) -> float:
        """Compute an edge penalty factor for confidence at position *idx*.

        Points near the edges of the series have less context and
        receive a reduced confidence multiplier.

        Args:
            idx: Position in the series.
            n: Total series length.

        Returns:
            Multiplier in the range [0.5, 1.0].
        """
        if n <= 1:
            return 0.5
        dist_from_edge = min(idx, n - 1 - idx)
        ratio = dist_from_edge / (n * 0.2)
        return min(1.0, 0.5 + 0.5 * ratio)

    def _empty_fill_result(
        self,
        values: List[Optional[float]],
        method: str,
        start_time: float,
    ) -> FillResult:
        """Return a FillResult with no gaps filled (all data present).

        Args:
            values: The original series (no gaps).
            method: Method name for labelling.
            start_time: Start timestamp for duration computation.

        Returns:
            FillResult with zero gaps filled.
        """
        elapsed = time.time() - start_time
        provenance_hash = self._provenance.build_hash({
            "operation": f"fill_{method}_no_gaps",
            "n": len(values),
        })
        return FillResult(
            values=list(values),
            original=list(values),
            filled_indices=[],
            fill_values={},
            method=method,
            confidence=1.0,
            per_point_confidence={},
            gaps_filled=0,
            gaps_remaining=0,
            processing_time_ms=elapsed * 1000.0,
            provenance_hash=provenance_hash,
            details={"note": "no gaps detected"},
        )

    def _insufficient_fill_result(
        self,
        values: List[Optional[float]],
        method: str,
        gap_indices: List[int],
        start_time: float,
    ) -> FillResult:
        """Return a FillResult when insufficient data prevents filling.

        Args:
            values: The original series.
            method: Method name.
            gap_indices: Indices of detected gaps.
            start_time: Start timestamp for duration computation.

        Returns:
            FillResult with zero gaps filled and low confidence.
        """
        elapsed = time.time() - start_time
        provenance_hash = self._provenance.build_hash({
            "operation": f"fill_{method}_insufficient",
            "n": len(values),
            "gaps": len(gap_indices),
        })
        return FillResult(
            values=list(values),
            original=list(values),
            filled_indices=[],
            fill_values={},
            method=method,
            confidence=0.0,
            per_point_confidence={},
            gaps_filled=0,
            gaps_remaining=len(gap_indices),
            processing_time_ms=elapsed * 1000.0,
            provenance_hash=provenance_hash,
            details={"note": "insufficient data for gap filling"},
        )

    def _no_seasonality_result(
        self,
        values: List[Optional[float]],
        start_time: float,
    ) -> Dict[str, Any]:
        """Return a seasonality detection result when data is insufficient.

        Args:
            values: The original series.
            start_time: Start timestamp.

        Returns:
            Dict with is_seasonal=False and zero confidence.
        """
        elapsed = time.time() - start_time
        observe_duration("detect_seasonality", elapsed)

        provenance_hash = self._provenance.build_hash({
            "operation": "detect_seasonality_insufficient",
            "n": len(values),
        })
        return {
            "detected_periods": [],
            "strengths": {},
            "confidence": 0.0,
            "acf_values": {},
            "dominant_period": 0,
            "is_seasonal": False,
            "significance_threshold": 0.0,
            "provenance_hash": provenance_hash,
        }


__all__ = [
    "SeasonalFillerEngine",
    "SeasonalDecomposition",
    "FillResult",
    "CalendarDefinition",
]
