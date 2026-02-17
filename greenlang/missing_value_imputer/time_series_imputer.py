# -*- coding: utf-8 -*-
"""
Time Series Imputer Engine - AGENT-DATA-012: Missing Value Imputer (GL-DATA-X-015)

Provides time-series-aware imputation methods: linear interpolation, cubic
spline interpolation, seasonal decomposition (STL-style), moving average,
exponential smoothing, and trend extrapolation. Includes seasonality and
trend detection utilities.

All algorithms are implemented in pure Python with no external library
dependencies. Interpolation uses piecewise polynomial construction,
seasonal decomposition uses moving-average-based STL approximation,
and trend detection uses OLS linear regression.

Zero-Hallucination Guarantees:
    - All interpolation values are deterministic polynomial evaluation
    - Seasonal decomposition uses moving-average arithmetic only
    - Trend detection uses closed-form OLS
    - No ML/LLM calls in any code path
    - SHA-256 provenance on every imputed value

Example:
    >>> from greenlang.missing_value_imputer.time_series_imputer import TimeSeriesImputerEngine
    >>> from greenlang.missing_value_imputer.config import MissingValueImputerConfig
    >>> engine = TimeSeriesImputerEngine(MissingValueImputerConfig())
    >>> series = [1.0, None, 3.0, None, 5.0]
    >>> result = engine.impute_linear_interpolation(series)
    >>> print(result[0].imputed_value)  # 2.0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.models import (
    ConfidenceLevel,
    ImputationStrategy,
    ImputedValue,
)
from greenlang.missing_value_imputer.metrics import (
    inc_values_imputed,
    observe_confidence,
    observe_duration,
)
from greenlang.missing_value_imputer.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

__all__ = [
    "TimeSeriesImputerEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _is_missing(value: Any) -> bool:
    """Determine whether a value is considered missing."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _classify_confidence(score: float) -> ConfidenceLevel:
    """Classify a numeric confidence score into a level."""
    if score >= 0.85:
        return ConfidenceLevel.HIGH
    if score >= 0.70:
        return ConfidenceLevel.MEDIUM
    if score >= 0.50:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


def _safe_stdev(values: List[float]) -> float:
    """Compute sample standard deviation, returning 0.0 for < 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) if var > 0 else 0.0


# ===========================================================================
# TimeSeriesImputerEngine
# ===========================================================================


class TimeSeriesImputerEngine:
    """Time-series-aware imputation engine.

    Provides interpolation, decomposition, smoothing, and extrapolation
    methods for imputing missing values in ordered series data. Includes
    seasonality and trend detection utilities.

    Attributes:
        config: Service configuration.
        provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = TimeSeriesImputerEngine(MissingValueImputerConfig())
        >>> result = engine.impute_linear_interpolation([1.0, None, 3.0])
        >>> assert result[0].imputed_value == 2.0
    """

    def __init__(self, config: MissingValueImputerConfig) -> None:
        """Initialize the TimeSeriesImputerEngine.

        Args:
            config: Service configuration instance.
        """
        self.config = config
        self.provenance = ProvenanceTracker()
        logger.info("TimeSeriesImputerEngine initialized")

    # ------------------------------------------------------------------
    # Linear interpolation
    # ------------------------------------------------------------------

    def impute_linear_interpolation(
        self,
        series: List[Optional[float]],
        timestamps: Optional[List[Any]] = None,
    ) -> List[ImputedValue]:
        """Impute missing values using linear interpolation.

        Fills each gap by drawing a straight line between the nearest
        non-missing points on either side. Edge missing values (leading/
        trailing) are not interpolated unless bounded on both sides.

        Args:
            series: Ordered list of values (None = missing).
            timestamps: Optional timestamp list for time-weighted interpolation.

        Returns:
            List of ImputedValue for each filled position.
        """
        start = time.monotonic()
        if not self._validate_time_series(series):
            return []

        n = len(series)
        imputed: List[ImputedValue] = []

        for i in range(n):
            if not _is_missing(series[i]):
                continue

            # Find left and right non-missing neighbors
            left_idx, left_val = self._find_left(series, i)
            right_idx, right_val = self._find_right(series, i)

            if left_val is None or right_val is None:
                continue

            # Linear interpolation
            if timestamps is not None and len(timestamps) == n:
                t_left = self._to_numeric_time(timestamps[left_idx])
                t_right = self._to_numeric_time(timestamps[right_idx])
                t_i = self._to_numeric_time(timestamps[i])
                if t_right - t_left > 0:
                    fraction = (t_i - t_left) / (t_right - t_left)
                else:
                    fraction = 0.5
            else:
                span = right_idx - left_idx
                fraction = (i - left_idx) / span if span > 0 else 0.5

            imp_val = left_val + fraction * (right_val - left_val)

            # Confidence decays with gap size
            gap_size = right_idx - left_idx
            confidence = max(0.50, 0.92 - 0.03 * gap_size)

            prov = _compute_provenance(
                "linear_interpolation", f"{i}:{imp_val}"
            )
            iv = ImputedValue(
                record_index=i,
                column_name="series",
                imputed_value=round(imp_val, 8),
                original_value=None,
                strategy=ImputationStrategy.LINEAR_INTERPOLATION,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=2,
                provenance_hash=prov,
            )
            imputed.append(iv)

        self._record_metrics("linear_interpolation", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Spline interpolation
    # ------------------------------------------------------------------

    def impute_spline_interpolation(
        self,
        series: List[Optional[float]],
        timestamps: Optional[List[Any]] = None,
        order: int = 3,
    ) -> List[ImputedValue]:
        """Impute missing values using cubic spline interpolation.

        Fits piecewise cubic polynomials through the observed data points,
        ensuring smooth first and second derivatives at knot points.

        Uses natural cubic spline boundary conditions (second derivative = 0
        at endpoints).

        Args:
            series: Ordered list of values (None = missing).
            timestamps: Optional timestamp list.
            order: Spline order (default 3 for cubic).

        Returns:
            List of ImputedValue for each filled position.
        """
        start = time.monotonic()
        if not self._validate_time_series(series):
            return []

        # Collect observed points
        observed_x: List[float] = []
        observed_y: List[float] = []
        for i, val in enumerate(series):
            if not _is_missing(val):
                if timestamps and len(timestamps) == len(series):
                    observed_x.append(self._to_numeric_time(timestamps[i]))
                else:
                    observed_x.append(float(i))
                observed_y.append(float(val))

        if len(observed_x) < 4:
            # Fall back to linear for insufficient points
            return self.impute_linear_interpolation(series, timestamps)

        # Compute natural cubic spline coefficients
        coeffs = self._compute_cubic_spline(observed_x, observed_y)

        n = len(series)
        imputed: List[ImputedValue] = []

        for i in range(n):
            if not _is_missing(series[i]):
                continue

            if timestamps and len(timestamps) == n:
                x_i = self._to_numeric_time(timestamps[i])
            else:
                x_i = float(i)

            # Evaluate spline at x_i
            imp_val = self._evaluate_spline(
                x_i, observed_x, observed_y, coeffs
            )

            if imp_val is None:
                continue

            # Confidence based on distance to nearest knot
            dist_to_nearest = self._distance_to_nearest_knot(x_i, observed_x)
            max_span = observed_x[-1] - observed_x[0] if len(observed_x) > 1 else 1.0
            rel_dist = dist_to_nearest / max_span if max_span > 0 else 0.5
            confidence = max(0.50, 0.88 - 0.30 * rel_dist)

            prov = _compute_provenance(
                "spline_interpolation", f"{i}:{imp_val}"
            )
            iv = ImputedValue(
                record_index=i,
                column_name="series",
                imputed_value=round(imp_val, 8),
                original_value=None,
                strategy=ImputationStrategy.SPLINE_INTERPOLATION,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=len(observed_x),
                provenance_hash=prov,
            )
            imputed.append(iv)

        self._record_metrics("spline_interpolation", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Seasonal decomposition
    # ------------------------------------------------------------------

    def impute_seasonal_decomposition(
        self,
        series: List[Optional[float]],
        period: Optional[int] = None,
    ) -> List[ImputedValue]:
        """Impute missing values using STL-style seasonal decomposition.

        Decomposes the series into trend + seasonal + residual components
        using moving average smoothing. Missing values are filled from
        the reconstructed trend + seasonal components.

        Args:
            series: Ordered list of values (None = missing).
            period: Seasonal period. Defaults to config.seasonal_period.

        Returns:
            List of ImputedValue for each filled position.
        """
        start = time.monotonic()
        period = period or self.config.seasonal_period

        if not self._validate_time_series(series):
            return []

        n = len(series)
        if n < 2 * period:
            logger.warning(
                "Series length %d < 2 * period %d, falling back to linear",
                n, period,
            )
            return self.impute_linear_interpolation(series)

        # Step 1: Initial fill with linear interpolation for decomposition
        filled = self._initial_fill(series)

        # Step 2: Extract trend via centered moving average
        trend = self._moving_average_smooth(filled, period)

        # Step 3: Detrend and compute seasonal component
        detrended = [
            filled[i] - trend[i] if trend[i] is not None else 0.0
            for i in range(n)
        ]

        seasonal = self._compute_seasonal_component(detrended, period)

        # Step 4: Reconstruct missing values from trend + seasonal
        imputed: List[ImputedValue] = []
        for i in range(n):
            if not _is_missing(series[i]):
                continue

            trend_val = trend[i] if trend[i] is not None else filled[i]
            seasonal_val = seasonal[i % period] if i % period < len(seasonal) else 0.0
            imp_val = trend_val + seasonal_val

            # Confidence based on seasonal strength and trend stability
            seasonality_info = self.detect_seasonality(
                [v for v in filled if v is not None]
            )
            confidence = 0.72
            if seasonality_info.get("significant", False):
                confidence += 0.10
            if trend[i] is not None:
                confidence += 0.05
            confidence = min(confidence, 0.92)

            prov = _compute_provenance(
                "seasonal_decomposition", f"{i}:{imp_val}"
            )
            iv = ImputedValue(
                record_index=i,
                column_name="series",
                imputed_value=round(imp_val, 8),
                original_value=None,
                strategy=ImputationStrategy.SEASONAL_DECOMPOSITION,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=n,
                provenance_hash=prov,
            )
            imputed.append(iv)

        self._record_metrics("seasonal_decomposition", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Moving average
    # ------------------------------------------------------------------

    def impute_moving_average(
        self,
        series: List[Optional[float]],
        window: Optional[int] = None,
    ) -> List[ImputedValue]:
        """Impute missing values using a centered moving average.

        For each missing position, computes the mean of the nearest
        non-missing values within the window.

        Args:
            series: Ordered list of values (None = missing).
            window: Window size. Defaults to config.trend_window.

        Returns:
            List of ImputedValue for each filled position.
        """
        start = time.monotonic()
        window = window or self.config.trend_window

        if not self._validate_time_series(series):
            return []

        n = len(series)
        half_w = window // 2
        imputed: List[ImputedValue] = []

        for i in range(n):
            if not _is_missing(series[i]):
                continue

            # Gather values within window
            lo = max(0, i - half_w)
            hi = min(n, i + half_w + 1)
            window_vals = [
                float(series[j]) for j in range(lo, hi)
                if not _is_missing(series[j])
            ]

            if not window_vals:
                continue

            imp_val = sum(window_vals) / len(window_vals)
            contributing = len(window_vals)

            # Confidence based on window coverage
            coverage = contributing / window if window > 0 else 0.5
            confidence = max(0.45, 0.50 + coverage * 0.40)

            prov = _compute_provenance(
                "moving_average", f"{i}:{imp_val}:w={window}"
            )
            iv = ImputedValue(
                record_index=i,
                column_name="series",
                imputed_value=round(imp_val, 8),
                original_value=None,
                strategy=ImputationStrategy.LINEAR_INTERPOLATION,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=contributing,
                provenance_hash=prov,
            )
            imputed.append(iv)

        self._record_metrics("moving_average", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Exponential smoothing
    # ------------------------------------------------------------------

    def impute_exponential_smoothing(
        self,
        series: List[Optional[float]],
        alpha: float = 0.3,
    ) -> List[ImputedValue]:
        """Impute missing values using simple exponential smoothing.

        Applies the recurrence S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
        on observed values. Missing positions use the current smoothed value.

        Args:
            series: Ordered list of values (None = missing).
            alpha: Smoothing factor in (0, 1). Higher = more responsive.

        Returns:
            List of ImputedValue for each filled position.
        """
        start = time.monotonic()
        alpha = max(0.01, min(0.99, alpha))

        if not self._validate_time_series(series):
            return []

        n = len(series)

        # Find first observed value to initialize
        smoothed: Optional[float] = None
        for val in series:
            if not _is_missing(val):
                smoothed = float(val)
                break

        if smoothed is None:
            return []

        imputed: List[ImputedValue] = []
        steps_since_obs = 0

        for i in range(n):
            val = series[i]
            if not _is_missing(val):
                smoothed = alpha * float(val) + (1.0 - alpha) * smoothed
                steps_since_obs = 0
            else:
                steps_since_obs += 1
                imp_val = smoothed

                # Confidence decays with steps since last observation
                confidence = max(0.40, 0.82 - 0.04 * steps_since_obs)

                prov = _compute_provenance(
                    "exponential_smoothing", f"{i}:{imp_val}:a={alpha}"
                )
                iv = ImputedValue(
                    record_index=i,
                    column_name="series",
                    imputed_value=round(imp_val, 8),
                    original_value=None,
                    strategy=ImputationStrategy.LINEAR_INTERPOLATION,
                    confidence=round(confidence, 4),
                    confidence_level=_classify_confidence(confidence),
                    contributing_records=1,
                    provenance_hash=prov,
                )
                imputed.append(iv)

        self._record_metrics("exponential_smoothing", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Trend extrapolation
    # ------------------------------------------------------------------

    def impute_trend_extrapolation(
        self,
        series: List[Optional[float]],
        timestamps: Optional[List[Any]] = None,
    ) -> List[ImputedValue]:
        """Impute missing values using linear trend extrapolation.

        Fits a linear trend (y = a + b*t) to observed values and uses
        it to fill missing positions. Suitable for series with strong
        linear trends.

        Args:
            series: Ordered list of values (None = missing).
            timestamps: Optional timestamp list.

        Returns:
            List of ImputedValue for each filled position.
        """
        start = time.monotonic()
        if not self._validate_time_series(series):
            return []

        n = len(series)

        # Collect observed (x, y) pairs
        observed_x: List[float] = []
        observed_y: List[float] = []
        for i, val in enumerate(series):
            if not _is_missing(val):
                if timestamps and len(timestamps) == n:
                    observed_x.append(self._to_numeric_time(timestamps[i]))
                else:
                    observed_x.append(float(i))
                observed_y.append(float(val))

        if len(observed_x) < 2:
            return []

        # Fit linear trend: y = a + b * x
        slope, intercept, r_squared = self._fit_linear_trend(observed_x, observed_y)

        imputed: List[ImputedValue] = []
        for i in range(n):
            if not _is_missing(series[i]):
                continue

            if timestamps and len(timestamps) == n:
                x_i = self._to_numeric_time(timestamps[i])
            else:
                x_i = float(i)

            imp_val = intercept + slope * x_i

            # Confidence based on R-squared and extrapolation distance
            x_range = max(observed_x) - min(observed_x) if observed_x else 1.0
            extrap_dist = 0.0
            if x_i < min(observed_x):
                extrap_dist = (min(observed_x) - x_i) / x_range if x_range > 0 else 1.0
            elif x_i > max(observed_x):
                extrap_dist = (x_i - max(observed_x)) / x_range if x_range > 0 else 1.0

            confidence = max(
                0.35,
                0.55 + r_squared * 0.35 - extrap_dist * 0.20,
            )

            prov = _compute_provenance(
                "trend_extrapolation", f"{i}:{imp_val}"
            )
            iv = ImputedValue(
                record_index=i,
                column_name="series",
                imputed_value=round(imp_val, 8),
                original_value=None,
                strategy=ImputationStrategy.LINEAR_INTERPOLATION,
                confidence=round(confidence, 4),
                confidence_level=_classify_confidence(confidence),
                contributing_records=len(observed_x),
                provenance_hash=prov,
            )
            imputed.append(iv)

        self._record_metrics("trend_extrapolation", imputed, start)
        return imputed

    # ------------------------------------------------------------------
    # Detection utilities
    # ------------------------------------------------------------------

    def detect_seasonality(self, series: List[float]) -> Dict[str, Any]:
        """Detect seasonality in a time series.

        Uses autocorrelation analysis to find the dominant seasonal period
        and its strength.

        Args:
            series: List of numeric values (no missing).

        Returns:
            Dict with keys: period, amplitude, significance, significant,
                acf_values, provenance_hash.
        """
        if len(series) < 6:
            return {
                "period": 0,
                "amplitude": 0.0,
                "significance": 0.0,
                "significant": False,
                "acf_values": [],
                "provenance_hash": _compute_provenance("detect_seasonality", "short"),
            }

        n = len(series)
        max_lag = min(n // 2, self.config.seasonal_period * 3)
        mean_s = sum(series) / n
        var_s = sum((x - mean_s) ** 2 for x in series)

        if var_s < 1e-12:
            return {
                "period": 0,
                "amplitude": 0.0,
                "significance": 0.0,
                "significant": False,
                "acf_values": [],
                "provenance_hash": _compute_provenance("detect_seasonality", "zero_var"),
            }

        # Compute autocorrelation
        acf_values: List[float] = []
        for lag in range(1, max_lag + 1):
            acf_val = sum(
                (series[t] - mean_s) * (series[t - lag] - mean_s)
                for t in range(lag, n)
            ) / var_s
            acf_values.append(round(acf_val, 6))

        # Find the dominant period (first peak in ACF after initial decay)
        best_period = 0
        best_acf = 0.0
        for lag_idx in range(1, len(acf_values) - 1):
            acf_val = acf_values[lag_idx]
            if (acf_val > acf_values[lag_idx - 1] and
                    acf_val > acf_values[lag_idx + 1] if lag_idx + 1 < len(acf_values) else True):
                if acf_val > best_acf:
                    best_acf = acf_val
                    best_period = lag_idx + 1

        # Significance threshold: 2/sqrt(n)
        sig_threshold = 2.0 / math.sqrt(n)
        significant = best_acf > sig_threshold

        # Amplitude estimation
        amplitude = 0.0
        if best_period > 0:
            period_means: List[float] = []
            for offset in range(best_period):
                vals = [series[i] for i in range(offset, n, best_period)]
                if vals:
                    period_means.append(sum(vals) / len(vals))
            if period_means:
                amplitude = max(period_means) - min(period_means)

        return {
            "period": best_period,
            "amplitude": round(amplitude, 6),
            "significance": round(best_acf, 6),
            "significant": significant,
            "acf_values": acf_values[:20],
            "provenance_hash": _compute_provenance(
                "detect_seasonality", f"p={best_period}:s={best_acf:.4f}"
            ),
        }

    def detect_trend(self, series: List[float]) -> Dict[str, Any]:
        """Detect trend in a time series.

        Fits a linear trend and reports direction, slope, R-squared.

        Args:
            series: List of numeric values (no missing).

        Returns:
            Dict with keys: direction, slope, intercept, r_squared,
                significant, provenance_hash.
        """
        if len(series) < 3:
            return {
                "direction": "none",
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "significant": False,
                "provenance_hash": _compute_provenance("detect_trend", "short"),
            }

        x_vals = [float(i) for i in range(len(series))]
        slope, intercept, r_squared = self._fit_linear_trend(x_vals, series)

        # Direction
        if abs(slope) < 1e-8:
            direction = "none"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Significance: R-squared > 0.3 and slope meaningfully different from 0
        significant = r_squared > 0.3

        return {
            "direction": direction,
            "slope": round(slope, 8),
            "intercept": round(intercept, 8),
            "r_squared": round(r_squared, 6),
            "significant": significant,
            "provenance_hash": _compute_provenance(
                "detect_trend", f"s={slope:.6f}:r2={r_squared:.4f}"
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_time_series(self, series: List[Optional[float]]) -> bool:
        """Check if a series has sufficient data for imputation.

        Args:
            series: The time series to validate.

        Returns:
            True if the series has at least 2 non-missing values.
        """
        if not series:
            return False
        observed = sum(1 for v in series if not _is_missing(v))
        if observed < 2:
            logger.warning("Time series has < 2 observed values, cannot impute")
            return False
        return True

    def _find_left(
        self, series: List[Optional[float]], idx: int
    ) -> Tuple[int, Optional[float]]:
        """Find nearest non-missing value to the left.

        Args:
            series: The series.
            idx: Current index.

        Returns:
            Tuple of (index, value) or (-1, None) if not found.
        """
        for j in range(idx - 1, -1, -1):
            if not _is_missing(series[j]):
                return j, float(series[j])
        return -1, None

    def _find_right(
        self, series: List[Optional[float]], idx: int
    ) -> Tuple[int, Optional[float]]:
        """Find nearest non-missing value to the right.

        Args:
            series: The series.
            idx: Current index.

        Returns:
            Tuple of (index, value) or (len, None) if not found.
        """
        for j in range(idx + 1, len(series)):
            if not _is_missing(series[j]):
                return j, float(series[j])
        return len(series), None

    def _initial_fill(self, series: List[Optional[float]]) -> List[float]:
        """Create initial fill of missing values using linear interpolation.

        Used as preprocessing for decomposition methods.

        Args:
            series: Series with missing values.

        Returns:
            Fully filled series (no Nones).
        """
        n = len(series)
        filled = [0.0] * n
        observed = [not _is_missing(v) for v in series]

        # Copy observed values
        for i in range(n):
            if observed[i]:
                filled[i] = float(series[i])

        # Forward fill leading
        first_obs = None
        for i in range(n):
            if observed[i]:
                first_obs = filled[i]
                break
        if first_obs is not None:
            for i in range(n):
                if observed[i]:
                    break
                filled[i] = first_obs

        # Backward fill trailing
        last_obs = None
        for i in range(n - 1, -1, -1):
            if observed[i]:
                last_obs = filled[i]
                break
        if last_obs is not None:
            for i in range(n - 1, -1, -1):
                if observed[i]:
                    break
                filled[i] = last_obs

        # Linear interpolation for gaps
        i = 0
        while i < n:
            if not observed[i]:
                left_idx = i - 1
                right_idx = i
                while right_idx < n and not observed[right_idx]:
                    right_idx += 1
                if left_idx >= 0 and right_idx < n:
                    span = right_idx - left_idx
                    for j in range(left_idx + 1, right_idx):
                        frac = (j - left_idx) / span
                        filled[j] = (
                            filled[left_idx] + frac * (filled[right_idx] - filled[left_idx])
                        )
                i = right_idx
            else:
                i += 1

        return filled

    def _moving_average_smooth(
        self,
        values: List[float],
        window: int,
    ) -> List[Optional[float]]:
        """Apply centered moving average smoothing.

        Args:
            values: Input values.
            window: Window size.

        Returns:
            Smoothed values (None at edges where window is incomplete).
        """
        n = len(values)
        half = window // 2
        result: List[Optional[float]] = [None] * n

        for i in range(half, n - half):
            total = sum(values[i - half:i + half + 1])
            result[i] = total / window

        # Fill edges with nearest computed value
        for i in range(half):
            result[i] = result[half] if result[half] is not None else values[i]
        for i in range(n - half, n):
            val = result[n - half - 1]
            result[i] = val if val is not None else values[i]

        return result

    def _compute_seasonal_component(
        self,
        detrended: List[float],
        period: int,
    ) -> List[float]:
        """Compute seasonal component by averaging across periods.

        Args:
            detrended: Detrended series values.
            period: Seasonal period.

        Returns:
            List of seasonal factors (length = period).
        """
        n = len(detrended)
        seasonal: List[float] = []

        for offset in range(period):
            vals = [detrended[i] for i in range(offset, n, period)]
            seasonal.append(sum(vals) / len(vals) if vals else 0.0)

        # Center seasonal component (subtract mean)
        s_mean = sum(seasonal) / len(seasonal) if seasonal else 0.0
        seasonal = [s - s_mean for s in seasonal]

        return seasonal

    def _compute_cubic_spline(
        self,
        x: List[float],
        y: List[float],
    ) -> List[List[float]]:
        """Compute natural cubic spline coefficients.

        For n+1 data points, computes n piecewise cubic polynomials.
        Each piece: S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3

        Args:
            x: Sorted x-coordinates.
            y: Corresponding y-values.

        Returns:
            List of [a, b, c, d] coefficients for each interval.
        """
        n = len(x) - 1
        if n < 1:
            return []

        h = [x[i + 1] - x[i] for i in range(n)]

        # Natural spline: solve for c coefficients
        # Tridiagonal system
        alpha = [0.0] * (n + 1)
        for i in range(1, n):
            if h[i - 1] > 0 and h[i] > 0:
                alpha[i] = (
                    3.0 / h[i] * (y[i + 1] - y[i]) -
                    3.0 / h[i - 1] * (y[i] - y[i - 1])
                )

        # Thomas algorithm
        l = [1.0] + [0.0] * n
        mu = [0.0] * (n + 1)
        z = [0.0] * (n + 1)

        for i in range(1, n):
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            if abs(l[i]) < 1e-12:
                l[i] = 1e-12
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        l[n] = 1.0
        z[n] = 0.0

        c = [0.0] * (n + 1)
        b = [0.0] * n
        d = [0.0] * n

        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            if abs(h[j]) > 1e-12:
                b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
                d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])
            else:
                b[j] = 0.0
                d[j] = 0.0

        coeffs = []
        for i in range(n):
            coeffs.append([y[i], b[i], c[i], d[i]])

        return coeffs

    def _evaluate_spline(
        self,
        x_val: float,
        knots_x: List[float],
        knots_y: List[float],
        coeffs: List[List[float]],
    ) -> Optional[float]:
        """Evaluate cubic spline at a point.

        Args:
            x_val: Point to evaluate.
            knots_x: Knot x-coordinates.
            knots_y: Knot y-values.
            coeffs: Spline coefficients.

        Returns:
            Interpolated value or None if out of range.
        """
        n = len(knots_x) - 1
        if n < 1 or not coeffs:
            return None

        # Clamp to range
        if x_val <= knots_x[0]:
            return knots_y[0]
        if x_val >= knots_x[-1]:
            return knots_y[-1]

        # Find interval
        seg = 0
        for i in range(n):
            if knots_x[i] <= x_val <= knots_x[i + 1]:
                seg = i
                break

        if seg >= len(coeffs):
            return None

        a, b, c, d = coeffs[seg]
        dx = x_val - knots_x[seg]
        return a + b * dx + c * dx ** 2 + d * dx ** 3

    def _distance_to_nearest_knot(
        self, x_val: float, knots: List[float]
    ) -> float:
        """Compute distance from x_val to the nearest knot.

        Args:
            x_val: Query point.
            knots: List of knot positions.

        Returns:
            Minimum distance.
        """
        if not knots:
            return float("inf")
        return min(abs(x_val - k) for k in knots)

    def _to_numeric_time(self, t: Any) -> float:
        """Convert a timestamp to numeric value.

        Args:
            t: Timestamp (datetime, numeric, or string).

        Returns:
            Numeric representation.
        """
        if isinstance(t, datetime):
            return t.timestamp()
        if isinstance(t, (int, float)):
            return float(t)
        try:
            return float(t)
        except (ValueError, TypeError):
            return 0.0

    def _fit_linear_trend(
        self,
        x: List[float],
        y: List[float],
    ) -> Tuple[float, float, float]:
        """Fit y = a + b*x via OLS.

        Args:
            x: Independent variable values.
            y: Dependent variable values.

        Returns:
            Tuple of (slope, intercept, r_squared).
        """
        n = len(x)
        if n < 2:
            return 0.0, 0.0, 0.0

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        ss_xx = sum((xi - x_mean) ** 2 for xi in x)

        if abs(ss_xx) < 1e-12:
            return 0.0, y_mean, 0.0

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

        # R-squared
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        if ss_tot < 1e-12:
            r_squared = 1.0
        else:
            ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
            r_squared = max(0.0, 1.0 - ss_res / ss_tot)

        return slope, intercept, r_squared

    def _record_metrics(
        self,
        method: str,
        imputed: List[ImputedValue],
        start: float,
    ) -> None:
        """Record Prometheus metrics for an imputation."""
        elapsed = time.monotonic() - start
        observe_duration("impute", elapsed)
        if imputed:
            inc_values_imputed(method, len(imputed))
            for iv in imputed:
                observe_confidence(method, iv.confidence)
        logger.info(
            "%s imputation: %d values, elapsed=%.3fs",
            method, len(imputed), elapsed,
        )
