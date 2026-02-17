# -*- coding: utf-8 -*-
"""
Trend Extrapolation Engine - AGENT-DATA-014: Time Series Gap Filler

Engine 5 of 7. Implements trend-based gap filling methods including
OLS linear regression, single/double/triple exponential smoothing
(SES, Holt's, Holt-Winters additive), and moving average extrapolation.
Also provides trend detection and classification.

Zero-Hallucination: All calculations use deterministic Python arithmetic
(``math`` module only). No LLM calls for numeric computations. Every
filled value is traceable through SHA-256 provenance chains.

Methods:
    - fit_linear_trend: OLS linear regression with R-squared
    - fill_linear_trend: Gap fill using fitted linear trend
    - fill_exponential_smoothing: Single Exponential Smoothing (SES)
    - fill_double_exponential: Holt's double exponential (level + trend)
    - fill_holt_winters: Triple exponential smoothing (additive)
    - fill_moving_average: Simple moving average extrapolation
    - detect_trend: Classify trend type using R-squared thresholds

Example:
    >>> from greenlang.time_series_gap_filler.trend_extrapolator import (
    ...     TrendExtrapolatorEngine,
    ... )
    >>> engine = TrendExtrapolatorEngine()
    >>> result = engine.fill_linear_trend([1.0, None, 3.0, None, 5.0])
    >>> assert result.points_filled == 2

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.time_series_gap_filler.config import get_config
from greenlang.time_series_gap_filler.metrics import (
    inc_errors,
    inc_gaps_filled,
    inc_strategies,
    observe_confidence,
    observe_duration,
)
from greenlang.time_series_gap_filler.models import (
    FillMethod,
    FillPoint,
    TrendAnalysis,
    TrendType,
)
from greenlang.time_series_gap_filler.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _is_missing(value: Any) -> bool:
    """Check whether a value represents a missing data point.

    Treats None, float NaN, and the string ``"nan"`` (case-insensitive)
    as missing.

    Args:
        value: Value to check.

    Returns:
        True if the value is missing.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() == "nan":
        return True
    return False


def _ols_fit(
    x: List[float],
    y: List[float],
) -> Tuple[float, float, float]:
    """Compute Ordinary Least Squares linear regression.

    Fits y = slope * x + intercept and computes R-squared.

    Args:
        x: Independent variable values.
        y: Dependent variable values (same length as x).

    Returns:
        Tuple of (slope, intercept, r_squared).

    Raises:
        ValueError: If x and y have different lengths or fewer than
            2 data points.
    """
    n = len(x)
    if n != len(y):
        raise ValueError(
            f"x and y must have the same length, got {n} and {len(y)}"
        )
    if n < 2:
        raise ValueError(
            f"OLS requires at least 2 data points, got {n}"
        )

    # Means
    x_mean = sum(x) / n
    y_mean = sum(y) / n

    # Sums for slope computation
    ss_xy = 0.0
    ss_xx = 0.0
    for xi, yi in zip(x, y):
        dx = xi - x_mean
        ss_xy += dx * (yi - y_mean)
        ss_xx += dx * dx

    # Slope and intercept
    if ss_xx == 0.0:
        slope = 0.0
        intercept = y_mean
    else:
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

    # R-squared
    ss_res = 0.0
    ss_tot = 0.0
    for xi, yi in zip(x, y):
        y_pred = slope * xi + intercept
        ss_res += (yi - y_pred) ** 2
        ss_tot += (yi - y_mean) ** 2

    if ss_tot == 0.0:
        r_squared = 1.0  # Perfect fit (all values identical)
    else:
        r_squared = max(0.0, 1.0 - ss_res / ss_tot)

    return slope, intercept, r_squared


def _compute_confidence(
    r_squared: float,
    method: str,
    gap_length: int,
) -> float:
    """Compute fill confidence based on fit quality and gap properties.

    Confidence is primarily driven by R-squared (or equivalent quality
    metric) and penalised for longer gaps where extrapolation uncertainty
    increases.

    Args:
        r_squared: Goodness of fit measure (0.0-1.0).
        method: Fill method name (for method-specific adjustments).
        gap_length: Number of consecutive missing points in the gap.

    Returns:
        Confidence score clamped to [0.0, 1.0].
    """
    # Base confidence from R-squared (or equivalent quality metric)
    base = r_squared

    # Method-specific adjustment
    method_bonus: Dict[str, float] = {
        "linear_trend": 0.0,
        "exponential_smoothing": -0.05,
        "double_exponential": 0.0,
        "holt_winters": 0.05,
        "moving_average": -0.10,
    }
    adjustment = method_bonus.get(method, 0.0)

    # Gap length penalty: longer gaps reduce confidence
    if gap_length <= 1:
        length_penalty = 0.0
    elif gap_length <= 3:
        length_penalty = 0.05
    elif gap_length <= 6:
        length_penalty = 0.10
    elif gap_length <= 12:
        length_penalty = 0.20
    else:
        length_penalty = 0.30

    confidence = base + adjustment - length_penalty
    return max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# TrendExtrapolatorEngine
# ---------------------------------------------------------------------------


class TrendExtrapolatorEngine:
    """Trend-based gap filling engine for time series data.

    Implements OLS linear regression, single/double/triple exponential
    smoothing, and moving average extrapolation for filling gaps in
    time series. All computations use deterministic Python arithmetic
    (zero-hallucination).

    Attributes:
        _config: TimeSeriesGapFillerConfig instance.
        _provenance: SHA-256 provenance tracker for audit trails.

    Example:
        >>> engine = TrendExtrapolatorEngine()
        >>> trend = engine.fit_linear_trend([1.0, 2.0, 3.0, 4.0])
        >>> assert trend["r_squared"] > 0.99
        >>> result = engine.fill_linear_trend([1.0, None, 3.0, None, 5.0])
        >>> assert result.points_filled == 2
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize TrendExtrapolatorEngine.

        Args:
            config: Optional TimeSeriesGapFillerConfig override.
                If None, the singleton config from ``get_config()``
                is used.
        """
        self._config = config or get_config()
        self._provenance: ProvenanceTracker = get_provenance_tracker()
        logger.info("TrendExtrapolatorEngine initialized")

    # ==================================================================
    # 1. fit_linear_trend
    # ==================================================================

    def fit_linear_trend(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Fit an OLS linear regression to non-missing values.

        Computes y = slope * x + intercept using Ordinary Least Squares
        on the observed (non-missing) data points. Returns the fitted
        parameters and R-squared goodness of fit.

        Args:
            values: Series values (None or NaN for missing).
            timestamps: Optional numeric timestamps for the x-axis.
                If None, integer indices 0..n-1 are used.

        Returns:
            Dictionary with keys:
                - slope (float): Fitted slope.
                - intercept (float): Fitted intercept.
                - r_squared (float): Coefficient of determination.
                - n_points (int): Number of non-missing points used.
                - provenance_hash (str): SHA-256 audit hash.

        Raises:
            ValueError: If fewer than 2 non-missing points exist.
        """
        start = time.time()

        # Extract non-missing (x, y) pairs
        x_vals, y_vals = self._extract_known_points(values, timestamps)

        if len(x_vals) < 2:
            raise ValueError(
                f"fit_linear_trend requires at least 2 non-missing "
                f"points, got {len(x_vals)}"
            )

        slope, intercept, r_squared = _ols_fit(x_vals, y_vals)

        # Provenance
        provenance_hash = self._provenance.build_hash({
            "operation": "fit_linear_trend",
            "n_points": len(x_vals),
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        })
        self._provenance.record(
            entity_type="trend_extrapolation",
            entity_id="fit_linear_trend",
            action="fit",
            data_hash=provenance_hash,
        )

        elapsed = time.time() - start
        observe_duration("fit_linear_trend", elapsed)

        logger.info(
            "fit_linear_trend: slope=%.6f intercept=%.6f "
            "r_squared=%.4f n_points=%d elapsed=%.3fms",
            slope, intercept, r_squared, len(x_vals), elapsed * 1000,
        )

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "n_points": len(x_vals),
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # 2. fill_linear_trend
    # ==================================================================

    def fill_linear_trend(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Fill gaps using fitted linear trend extrapolation.

        Fits OLS linear regression to the non-missing data and uses
        the resulting line y = slope * x + intercept to fill missing
        values. Confidence is proportional to R-squared.

        Args:
            values: Series values (None or NaN for missing).
            timestamps: Optional numeric timestamps for the x-axis.
                If None, integer indices 0..n-1 are used.

        Returns:
            Dict with filled values and per-point details.
        """
        start = time.time()
        n = len(values)
        method = FillMethod.LINEAR_TREND

        # Extract known points
        x_known, y_known = self._extract_known_points(values, timestamps)

        # Insufficient data: return original series
        if len(x_known) < 2:
            logger.warning(
                "fill_linear_trend: insufficient data (%d known points), "
                "returning original series",
                len(x_known),
            )
            inc_errors("insufficient_data")
            return self._build_unfilled_result(values, method, start)

        slope, intercept, r_squared = _ols_fit(x_known, y_known)

        # Build filled series
        filled_values: List[Optional[float]] = []
        fill_points: List[FillPoint] = []
        points_filled = 0
        gap_count = 0
        in_gap = False

        for i in range(n):
            x_i = timestamps[i] if timestamps else float(i)
            original = values[i]
            is_miss = _is_missing(original)

            if is_miss:
                predicted = slope * x_i + intercept
                filled_values.append(predicted)
                points_filled += 1

                if not in_gap:
                    gap_count += 1
                    in_gap = True

                # Compute gap-local length for confidence
                gap_len = self._current_gap_length(values, i)
                conf = _compute_confidence(r_squared, "linear_trend", gap_len)

                provenance_hash = self._provenance.build_hash({
                    "operation": "fill_linear_trend",
                    "index": i,
                    "x": x_i,
                    "predicted": predicted,
                    "slope": slope,
                    "intercept": intercept,
                })

                fill_points.append(FillPoint(
                    index=i,
                    original_value=None,
                    filled_value=predicted,
                    was_missing=True,
                    method=method,
                    confidence=conf,
                    provenance_hash=provenance_hash,
                ))
            else:
                in_gap = False
                val = float(original)  # type: ignore[arg-type]
                filled_values.append(val)
                fill_points.append(FillPoint(
                    index=i,
                    original_value=val,
                    filled_value=val,
                    was_missing=False,
                    method=method,
                    confidence=1.0,
                    provenance_hash=self._provenance.build_hash({
                        "operation": "fill_linear_trend",
                        "index": i,
                        "value": val,
                        "kept": True,
                    }),
                ))

        # Aggregate confidence
        filled_confs = [
            fp.confidence for fp in fill_points if fp.was_missing
        ]
        mean_conf = (
            sum(filled_confs) / len(filled_confs) if filled_confs else 1.0
        )
        min_conf = min(filled_confs) if filled_confs else 1.0

        # Provenance chain
        result_hash = self._provenance.build_hash({
            "operation": "fill_linear_trend",
            "n": n,
            "points_filled": points_filled,
            "r_squared": r_squared,
            "slope": slope,
            "intercept": intercept,
        })
        self._provenance.record(
            entity_type="trend_extrapolation",
            entity_id="fill_linear_trend",
            action="fill",
            data_hash=result_hash,
        )

        elapsed = time.time() - start
        observe_duration("fill_linear_trend", elapsed)
        inc_gaps_filled("linear_trend", gap_count)
        inc_strategies("linear_trend")
        observe_confidence(mean_conf)

        logger.info(
            "fill_linear_trend: n=%d filled=%d gaps=%d "
            "r2=%.4f mean_conf=%.4f elapsed=%.3fms",
            n, points_filled, gap_count, r_squared,
            mean_conf, elapsed * 1000,
        )

        return {
            "method": method.value,
            "filled_values": filled_values,
            "fill_points": fill_points,
            "gaps_filled": gap_count,
            "points_filled": points_filled,
            "total_points": n,
            "fill_count": points_filled,
            "mean_confidence": mean_conf,
            "min_confidence": min_conf,
            "confidence": mean_conf,
            "confidence_scores": [fp.confidence for fp in fill_points],
            "r_squared": r_squared,
            "processing_time_ms": elapsed * 1000,
            "metadata": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "n_known": len(x_known),
            },
            "provenance_hash": result_hash,
        }

    # ==================================================================
    # 3. fill_exponential_smoothing
    # ==================================================================

    def fill_exponential_smoothing(
        self,
        values: List[Optional[float]],
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fill gaps using Single Exponential Smoothing (SES).

        Applies the recurrence s_t = alpha * y_t + (1 - alpha) * s_{t-1}
        over observed values. Missing values are filled with the last
        computed smoothed value.

        Args:
            values: Series values (None or NaN for missing).
            alpha: Smoothing coefficient (0 < alpha <= 1). If None,
                uses ``config.smoothing_alpha``.

        Returns:
            Dict with filled values and per-point details.
        """
        start = time.time()
        n = len(values)
        method = FillMethod.EXPONENTIAL_SMOOTHING
        a = alpha if alpha is not None else self._config.smoothing_alpha
        a = max(0.01, min(1.0, a))

        # Initialize smoothed value from first non-missing
        first_known = self._find_first_known(values)
        if first_known is None:
            logger.warning(
                "fill_exponential_smoothing: no non-missing values"
            )
            inc_errors("no_data")
            return self._build_unfilled_result(values, method, start)

        smoothed = first_known
        filled_values: List[Optional[float]] = []
        fill_points: List[FillPoint] = []
        points_filled = 0
        gap_count = 0
        in_gap = False

        for i in range(n):
            original = values[i]
            is_miss = _is_missing(original)

            if is_miss:
                # Fill with current smoothed value
                filled_values.append(smoothed)
                points_filled += 1

                if not in_gap:
                    gap_count += 1
                    in_gap = True

                gap_len = self._current_gap_length(values, i)
                conf = _compute_confidence(
                    self._ses_quality(values, a),
                    "exponential_smoothing",
                    gap_len,
                )

                provenance_hash = self._provenance.build_hash({
                    "operation": "fill_exponential_smoothing",
                    "index": i,
                    "smoothed": smoothed,
                    "alpha": a,
                })

                fill_points.append(FillPoint(
                    index=i,
                    original_value=None,
                    filled_value=smoothed,
                    was_missing=True,
                    method=method,
                    confidence=conf,
                    provenance_hash=provenance_hash,
                ))
            else:
                in_gap = False
                val = float(original)  # type: ignore[arg-type]
                # Update smoothed value: s_t = alpha * y_t + (1-alpha) * s_{t-1}
                smoothed = a * val + (1.0 - a) * smoothed
                filled_values.append(val)

                fill_points.append(FillPoint(
                    index=i,
                    original_value=val,
                    filled_value=val,
                    was_missing=False,
                    method=method,
                    confidence=1.0,
                    provenance_hash=self._provenance.build_hash({
                        "operation": "fill_exponential_smoothing",
                        "index": i,
                        "value": val,
                        "smoothed": smoothed,
                    }),
                ))

        # Aggregate confidence
        filled_confs = [
            fp.confidence for fp in fill_points if fp.was_missing
        ]
        mean_conf = (
            sum(filled_confs) / len(filled_confs) if filled_confs else 1.0
        )
        min_conf = min(filled_confs) if filled_confs else 1.0

        result_hash = self._provenance.build_hash({
            "operation": "fill_exponential_smoothing",
            "n": n,
            "points_filled": points_filled,
            "alpha": a,
        })
        self._provenance.record(
            entity_type="trend_extrapolation",
            entity_id="fill_exponential_smoothing",
            action="fill",
            data_hash=result_hash,
        )

        elapsed = time.time() - start
        observe_duration("fill_exponential_smoothing", elapsed)
        inc_gaps_filled("exponential_smoothing", gap_count)
        inc_strategies("exponential_smoothing")
        observe_confidence(mean_conf)

        logger.info(
            "fill_exponential_smoothing: n=%d filled=%d gaps=%d "
            "alpha=%.3f mean_conf=%.4f elapsed=%.3fms",
            n, points_filled, gap_count, a, mean_conf, elapsed * 1000,
        )

        return {
            "method": method.value,
            "filled_values": filled_values,
            "fill_points": fill_points,
            "gaps_filled": gap_count,
            "points_filled": points_filled,
            "total_points": n,
            "fill_count": points_filled,
            "mean_confidence": mean_conf,
            "min_confidence": min_conf,
            "confidence": mean_conf,
            "confidence_scores": [fp.confidence for fp in fill_points],
            "r_squared": None,
            "processing_time_ms": elapsed * 1000,
            "metadata": {"alpha": a},
            "provenance_hash": result_hash,
        }

    # ==================================================================
    # 4. fill_double_exponential
    # ==================================================================

    def fill_double_exponential(
        self,
        values: List[Optional[float]],
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fill gaps using Holt's double exponential smoothing.

        Maintains level (l_t) and trend (b_t) components:
            l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
            b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        Forecast k steps ahead: l_t + k * b_t

        Args:
            values: Series values (None or NaN for missing).
            alpha: Level smoothing coefficient. If None, uses
                ``config.smoothing_alpha``.
            beta: Trend smoothing coefficient. If None, uses
                ``config.smoothing_beta``.

        Returns:
            Dict with filled values and per-point details.
        """
        start = time.time()
        n = len(values)
        method = FillMethod.DOUBLE_EXPONENTIAL
        a = alpha if alpha is not None else self._config.smoothing_alpha
        b = beta if beta is not None else self._config.smoothing_beta
        a = max(0.01, min(1.0, a))
        b = max(0.01, min(1.0, b))

        # Initialize level and trend from first two known values
        known_vals = self._collect_known_values(values)
        if len(known_vals) < 2:
            logger.warning(
                "fill_double_exponential: insufficient known values (%d)",
                len(known_vals),
            )
            inc_errors("insufficient_data")
            return self._build_unfilled_result(values, method, start)

        level = known_vals[0]
        trend = known_vals[1] - known_vals[0]

        filled_values: List[Optional[float]] = []
        fill_points: List[FillPoint] = []
        points_filled = 0
        gap_count = 0
        in_gap = False
        steps_ahead = 0

        for i in range(n):
            original = values[i]
            is_miss = _is_missing(original)

            if is_miss:
                steps_ahead += 1
                predicted = level + steps_ahead * trend
                filled_values.append(predicted)
                points_filled += 1

                if not in_gap:
                    gap_count += 1
                    in_gap = True

                gap_len = self._current_gap_length(values, i)
                conf = _compute_confidence(
                    self._double_exp_quality(values, a, b),
                    "double_exponential",
                    gap_len,
                )

                provenance_hash = self._provenance.build_hash({
                    "operation": "fill_double_exponential",
                    "index": i,
                    "predicted": predicted,
                    "level": level,
                    "trend": trend,
                    "steps_ahead": steps_ahead,
                })

                fill_points.append(FillPoint(
                    index=i,
                    original_value=None,
                    filled_value=predicted,
                    was_missing=True,
                    method=method,
                    confidence=conf,
                    provenance_hash=provenance_hash,
                ))
            else:
                in_gap = False
                steps_ahead = 0
                val = float(original)  # type: ignore[arg-type]

                # Update Holt equations
                prev_level = level
                level = a * val + (1.0 - a) * (level + trend)
                trend = b * (level - prev_level) + (1.0 - b) * trend

                filled_values.append(val)
                fill_points.append(FillPoint(
                    index=i,
                    original_value=val,
                    filled_value=val,
                    was_missing=False,
                    method=method,
                    confidence=1.0,
                    provenance_hash=self._provenance.build_hash({
                        "operation": "fill_double_exponential",
                        "index": i,
                        "value": val,
                        "level": level,
                        "trend": trend,
                    }),
                ))

        # Aggregate confidence
        filled_confs = [
            fp.confidence for fp in fill_points if fp.was_missing
        ]
        mean_conf = (
            sum(filled_confs) / len(filled_confs) if filled_confs else 1.0
        )
        min_conf = min(filled_confs) if filled_confs else 1.0

        result_hash = self._provenance.build_hash({
            "operation": "fill_double_exponential",
            "n": n,
            "points_filled": points_filled,
            "alpha": a,
            "beta": b,
        })
        self._provenance.record(
            entity_type="trend_extrapolation",
            entity_id="fill_double_exponential",
            action="fill",
            data_hash=result_hash,
        )

        elapsed = time.time() - start
        observe_duration("fill_double_exponential", elapsed)
        inc_gaps_filled("double_exponential", gap_count)
        inc_strategies("double_exponential")
        observe_confidence(mean_conf)

        logger.info(
            "fill_double_exponential: n=%d filled=%d gaps=%d "
            "alpha=%.3f beta=%.3f mean_conf=%.4f elapsed=%.3fms",
            n, points_filled, gap_count, a, b, mean_conf, elapsed * 1000,
        )

        return {
            "method": method.value,
            "filled_values": filled_values,
            "fill_points": fill_points,
            "gaps_filled": gap_count,
            "points_filled": points_filled,
            "total_points": n,
            "fill_count": points_filled,
            "mean_confidence": mean_conf,
            "min_confidence": min_conf,
            "confidence": mean_conf,
            "confidence_scores": [fp.confidence for fp in fill_points],
            "r_squared": None,
            "processing_time_ms": elapsed * 1000,
            "metadata": {"alpha": a, "beta": b},
            "provenance_hash": result_hash,
        }

    # ==================================================================
    # 5. fill_holt_winters
    # ==================================================================

    def fill_holt_winters(
        self,
        values: List[Optional[float]],
        period: Optional[int] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fill gaps using additive Holt-Winters triple exponential smoothing.

        Maintains level (l_t), trend (b_t), and seasonal (s_t) components:
            l_t = alpha * (y_t - s_{t-period}) + (1-alpha) * (l_{t-1} + b_{t-1})
            b_t = beta * (l_t - l_{t-1}) + (1-beta) * b_{t-1}
            s_t = gamma * (y_t - l_t) + (1-gamma) * s_{t-period}
        Forecast k steps ahead: l_t + k*b_t + s_{t-period+k%%period}

        Args:
            values: Series values (None or NaN for missing).
            period: Seasonal period. If None, uses
                ``config.seasonal_periods``.
            alpha: Level smoothing coefficient. If None, uses
                ``config.smoothing_alpha``.
            beta: Trend smoothing coefficient. If None, uses
                ``config.smoothing_beta``.
            gamma: Seasonal smoothing coefficient. If None, uses
                ``config.smoothing_gamma``.

        Returns:
            Dict with filled values and per-point details.
        """
        start = time.time()
        n = len(values)
        method = FillMethod.HOLT_WINTERS
        p = period if period is not None else self._config.seasonal_periods
        a = alpha if alpha is not None else self._config.smoothing_alpha
        b = beta if beta is not None else self._config.smoothing_beta
        g = gamma if gamma is not None else self._config.smoothing_gamma
        a = max(0.01, min(1.0, a))
        b = max(0.01, min(1.0, b))
        g = max(0.01, min(1.0, g))

        # Validate minimum data for Holt-Winters
        known_vals = self._collect_known_values(values)
        if len(known_vals) < p * 2:
            logger.warning(
                "fill_holt_winters: insufficient data (%d known, need %d "
                "for period=%d)",
                len(known_vals), p * 2, p,
            )
            inc_errors("insufficient_data")
            return self._build_unfilled_result(values, method, start)

        if p < 2:
            logger.warning(
                "fill_holt_winters: period=%d too small, falling back "
                "to double exponential",
                p,
            )
            return self.fill_double_exponential(values, alpha=a, beta=b)

        # Initialize components from first complete cycle
        level, trend_val, seasonal = self._init_holt_winters(
            values, p,
        )

        filled_values: List[Optional[float]] = []
        fill_points: List[FillPoint] = []
        points_filled = 0
        gap_count = 0
        in_gap = False
        steps_ahead = 0

        for i in range(n):
            original = values[i]
            is_miss = _is_missing(original)

            # Seasonal index for current position
            s_idx = i % p

            if is_miss:
                steps_ahead += 1
                # Forecast: level + steps_ahead * trend + seasonal
                predicted = level + steps_ahead * trend_val + seasonal[s_idx]
                filled_values.append(predicted)
                points_filled += 1

                if not in_gap:
                    gap_count += 1
                    in_gap = True

                gap_len = self._current_gap_length(values, i)
                conf = _compute_confidence(
                    self._holt_winters_quality(values, p, a, b, g),
                    "holt_winters",
                    gap_len,
                )

                provenance_hash = self._provenance.build_hash({
                    "operation": "fill_holt_winters",
                    "index": i,
                    "predicted": predicted,
                    "level": level,
                    "trend": trend_val,
                    "seasonal": seasonal[s_idx],
                    "steps_ahead": steps_ahead,
                })

                fill_points.append(FillPoint(
                    index=i,
                    original_value=None,
                    filled_value=predicted,
                    was_missing=True,
                    method=method,
                    confidence=conf,
                    provenance_hash=provenance_hash,
                ))
            else:
                in_gap = False
                steps_ahead = 0
                val = float(original)  # type: ignore[arg-type]

                # Update Holt-Winters equations
                prev_level = level
                level = (
                    a * (val - seasonal[s_idx])
                    + (1.0 - a) * (level + trend_val)
                )
                trend_val = (
                    b * (level - prev_level)
                    + (1.0 - b) * trend_val
                )
                seasonal[s_idx] = (
                    g * (val - level)
                    + (1.0 - g) * seasonal[s_idx]
                )

                filled_values.append(val)
                fill_points.append(FillPoint(
                    index=i,
                    original_value=val,
                    filled_value=val,
                    was_missing=False,
                    method=method,
                    confidence=1.0,
                    provenance_hash=self._provenance.build_hash({
                        "operation": "fill_holt_winters",
                        "index": i,
                        "value": val,
                        "level": level,
                        "trend": trend_val,
                        "seasonal": seasonal[s_idx],
                    }),
                ))

        # Aggregate confidence
        filled_confs = [
            fp.confidence for fp in fill_points if fp.was_missing
        ]
        mean_conf = (
            sum(filled_confs) / len(filled_confs) if filled_confs else 1.0
        )
        min_conf = min(filled_confs) if filled_confs else 1.0

        result_hash = self._provenance.build_hash({
            "operation": "fill_holt_winters",
            "n": n,
            "points_filled": points_filled,
            "period": p,
            "alpha": a,
            "beta": b,
            "gamma": g,
        })
        self._provenance.record(
            entity_type="trend_extrapolation",
            entity_id="fill_holt_winters",
            action="fill",
            data_hash=result_hash,
        )

        elapsed = time.time() - start
        observe_duration("fill_holt_winters", elapsed)
        inc_gaps_filled("holt_winters", gap_count)
        inc_strategies("holt_winters")
        observe_confidence(mean_conf)

        logger.info(
            "fill_holt_winters: n=%d filled=%d gaps=%d period=%d "
            "alpha=%.3f beta=%.3f gamma=%.3f mean_conf=%.4f "
            "elapsed=%.3fms",
            n, points_filled, gap_count, p, a, b, g,
            mean_conf, elapsed * 1000,
        )

        return {
            "method": method.value,
            "filled_values": filled_values,
            "fill_points": fill_points,
            "gaps_filled": gap_count,
            "points_filled": points_filled,
            "total_points": n,
            "fill_count": points_filled,
            "mean_confidence": mean_conf,
            "min_confidence": min_conf,
            "confidence": mean_conf,
            "confidence_scores": [fp.confidence for fp in fill_points],
            "r_squared": None,
            "processing_time_ms": elapsed * 1000,
            "metadata": {
                "period": p,
                "alpha": a,
                "beta": b,
                "gamma": g,
            },
            "provenance_hash": result_hash,
        }

    # ==================================================================
    # 6. fill_moving_average
    # ==================================================================

    def fill_moving_average(
        self,
        values: List[Optional[float]],
        window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fill gaps using simple moving average extrapolation.

        For each missing value, computes the mean of the last ``window``
        known (non-missing) values. If fewer than ``window`` known values
        are available, uses all available known values.

        Args:
            values: Series values (None or NaN for missing).
            window: Number of preceding known values to average.
                Defaults to 5 if None.

        Returns:
            Dict with filled values and per-point details.
        """
        start = time.time()
        n = len(values)
        method = FillMethod.MOVING_AVERAGE
        w = window if window is not None else 5
        w = max(1, w)

        # Need at least 1 known value
        known_vals = self._collect_known_values(values)
        if not known_vals:
            logger.warning("fill_moving_average: no non-missing values")
            inc_errors("no_data")
            return self._build_unfilled_result(values, method, start)

        filled_values: List[Optional[float]] = []
        fill_points: List[FillPoint] = []
        points_filled = 0
        gap_count = 0
        in_gap = False

        # Maintain a running buffer of recent known values
        recent_known: List[float] = []

        for i in range(n):
            original = values[i]
            is_miss = _is_missing(original)

            if is_miss:
                if not recent_known:
                    # No known values yet, look ahead for first known
                    first_val = self._find_first_known(values)
                    if first_val is not None:
                        fill_val = first_val
                    else:
                        fill_val = 0.0
                else:
                    # Average of last w known values
                    window_vals = recent_known[-w:]
                    fill_val = sum(window_vals) / len(window_vals)

                filled_values.append(fill_val)
                points_filled += 1

                if not in_gap:
                    gap_count += 1
                    in_gap = True

                gap_len = self._current_gap_length(values, i)
                # Quality based on window coverage
                coverage = (
                    min(len(recent_known), w) / w if w > 0 else 0.0
                )
                conf = _compute_confidence(
                    coverage * 0.8,
                    "moving_average",
                    gap_len,
                )

                provenance_hash = self._provenance.build_hash({
                    "operation": "fill_moving_average",
                    "index": i,
                    "fill_val": fill_val,
                    "window": w,
                    "n_recent": len(recent_known[-w:]),
                })

                fill_points.append(FillPoint(
                    index=i,
                    original_value=None,
                    filled_value=fill_val,
                    was_missing=True,
                    method=method,
                    confidence=conf,
                    provenance_hash=provenance_hash,
                ))
            else:
                in_gap = False
                val = float(original)  # type: ignore[arg-type]
                recent_known.append(val)
                filled_values.append(val)

                fill_points.append(FillPoint(
                    index=i,
                    original_value=val,
                    filled_value=val,
                    was_missing=False,
                    method=method,
                    confidence=1.0,
                    provenance_hash=self._provenance.build_hash({
                        "operation": "fill_moving_average",
                        "index": i,
                        "value": val,
                    }),
                ))

        # Aggregate confidence
        filled_confs = [
            fp.confidence for fp in fill_points if fp.was_missing
        ]
        mean_conf = (
            sum(filled_confs) / len(filled_confs) if filled_confs else 1.0
        )
        min_conf = min(filled_confs) if filled_confs else 1.0

        result_hash = self._provenance.build_hash({
            "operation": "fill_moving_average",
            "n": n,
            "points_filled": points_filled,
            "window": w,
        })
        self._provenance.record(
            entity_type="trend_extrapolation",
            entity_id="fill_moving_average",
            action="fill",
            data_hash=result_hash,
        )

        elapsed = time.time() - start
        observe_duration("fill_moving_average", elapsed)
        inc_gaps_filled("moving_average", gap_count)
        inc_strategies("moving_average")
        observe_confidence(mean_conf)

        logger.info(
            "fill_moving_average: n=%d filled=%d gaps=%d window=%d "
            "mean_conf=%.4f elapsed=%.3fms",
            n, points_filled, gap_count, w, mean_conf, elapsed * 1000,
        )

        return {
            "method": method.value,
            "filled_values": filled_values,
            "fill_points": fill_points,
            "gaps_filled": gap_count,
            "points_filled": points_filled,
            "total_points": n,
            "fill_count": points_filled,
            "mean_confidence": mean_conf,
            "min_confidence": min_conf,
            "confidence": mean_conf,
            "confidence_scores": [fp.confidence for fp in fill_points],
            "r_squared": None,
            "processing_time_ms": elapsed * 1000,
            "metadata": {"window": w},
            "provenance_hash": result_hash,
        }

    # ==================================================================
    # 7. detect_trend
    # ==================================================================

    def detect_trend(
        self,
        values: List[Optional[float]],
    ) -> TrendType:
        """Detect and classify the trend in a time series.

        Fits an OLS linear regression to the non-missing values and
        classifies the trend based on R-squared thresholds:
            - R-squared > 0.8: LINEAR (clear trend)
            - 0.5 < R-squared <= 0.8: MODERATE_LINEAR
            - R-squared <= 0.5: STATIONARY (weak/no trend)

        Also checks for exponential patterns by fitting a log-linear
        model and for polynomial trends via quadratic fitting.

        Args:
            values: Series values (None or NaN for missing).

        Returns:
            TrendType classification.
        """
        start = time.time()

        known_vals = self._collect_known_values(values)

        if len(known_vals) < 3:
            logger.info(
                "detect_trend: insufficient data (%d points), "
                "returning UNKNOWN",
                len(known_vals),
            )
            return TrendType.UNKNOWN

        # Fit linear
        x_lin = [float(i) for i in range(len(known_vals))]
        try:
            slope_lin, _, r2_lin = _ols_fit(x_lin, known_vals)
        except ValueError:
            return TrendType.UNKNOWN

        # Fit log-linear (exponential check)
        r2_exp = 0.0
        if all(v > 0 for v in known_vals):
            log_vals = [math.log(v) for v in known_vals]
            try:
                _, _, r2_exp = _ols_fit(x_lin, log_vals)
            except ValueError:
                r2_exp = 0.0

        # Fit quadratic (polynomial check via R-squared improvement)
        r2_quad = self._fit_quadratic_r2(x_lin, known_vals)

        # Classify
        trend_type = self._classify_trend(r2_lin, r2_exp, r2_quad)

        # Provenance
        provenance_hash = self._provenance.build_hash({
            "operation": "detect_trend",
            "n_known": len(known_vals),
            "r2_linear": r2_lin,
            "r2_exponential": r2_exp,
            "r2_quadratic": r2_quad,
            "trend_type": trend_type.value,
        })
        self._provenance.record(
            entity_type="trend_extrapolation",
            entity_id="detect_trend",
            action="detect",
            data_hash=provenance_hash,
        )

        elapsed = time.time() - start
        observe_duration("detect_trend", elapsed)

        logger.info(
            "detect_trend: n=%d r2_lin=%.4f r2_exp=%.4f r2_quad=%.4f "
            "trend=%s elapsed=%.3fms",
            len(known_vals), r2_lin, r2_exp, r2_quad,
            trend_type.value, elapsed * 1000,
        )

        return trend_type

    # ==================================================================
    # Trend analysis (full result)
    # ==================================================================

    def analyze_trend(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> TrendAnalysis:
        """Perform full trend analysis returning a TrendAnalysis model.

        Fits OLS linear regression and classifies the trend, returning
        all parameters in a structured TrendAnalysis result.

        Args:
            values: Series values (None or NaN for missing).
            timestamps: Optional numeric timestamps for the x-axis.

        Returns:
            TrendAnalysis with slope, intercept, R-squared, and type.
        """
        start = time.time()

        x_known, y_known = self._extract_known_points(values, timestamps)

        if len(x_known) < 2:
            provenance_hash = self._provenance.build_hash({
                "operation": "analyze_trend",
                "n_known": len(x_known),
                "reason": "insufficient_data",
            })
            return TrendAnalysis(
                trend_type=TrendType.UNKNOWN,
                slope=0.0,
                intercept=0.0,
                r_squared=0.0,
                confidence=0.0,
                series_length=len(values),
                provenance_hash=provenance_hash,
            )

        slope, intercept, r_squared = _ols_fit(x_known, y_known)
        trend_type = self.detect_trend(values)

        # Confidence based on R-squared and data coverage
        n_total = len(values)
        n_known = len(x_known)
        coverage = n_known / n_total if n_total > 0 else 0.0
        confidence = min(1.0, r_squared * 0.7 + coverage * 0.3)

        provenance_hash = self._provenance.build_hash({
            "operation": "analyze_trend",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "trend_type": trend_type.value,
        })
        self._provenance.record(
            entity_type="trend_extrapolation",
            entity_id="analyze_trend",
            action="analyze",
            data_hash=provenance_hash,
        )

        elapsed = time.time() - start
        observe_duration("analyze_trend", elapsed)

        logger.info(
            "analyze_trend: trend=%s slope=%.6f r2=%.4f "
            "confidence=%.4f elapsed=%.3fms",
            trend_type.value, slope, r_squared,
            confidence, elapsed * 1000,
        )

        return TrendAnalysis(
            trend_type=trend_type,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            confidence=confidence,
            series_length=n_total,
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # Private helpers: data extraction
    # ==================================================================

    @staticmethod
    def _extract_known_points(
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float]]:
        """Extract non-missing (x, y) pairs from the series.

        Args:
            values: Series values (may contain None/NaN).
            timestamps: Optional x-axis values. If None, uses
                integer indices.

        Returns:
            Tuple of (x_values, y_values) for non-missing points.
        """
        x_vals: List[float] = []
        y_vals: List[float] = []
        for i, v in enumerate(values):
            if not _is_missing(v):
                x_i = timestamps[i] if timestamps else float(i)
                x_vals.append(x_i)
                y_vals.append(float(v))  # type: ignore[arg-type]
        return x_vals, y_vals

    @staticmethod
    def _collect_known_values(
        values: List[Optional[float]],
    ) -> List[float]:
        """Collect all non-missing values preserving order.

        Args:
            values: Series values.

        Returns:
            List of non-missing float values.
        """
        return [
            float(v)  # type: ignore[arg-type]
            for v in values
            if not _is_missing(v)
        ]

    @staticmethod
    def _find_first_known(
        values: List[Optional[float]],
    ) -> Optional[float]:
        """Find the first non-missing value in the series.

        Args:
            values: Series values.

        Returns:
            First non-missing float or None.
        """
        for v in values:
            if not _is_missing(v):
                return float(v)  # type: ignore[arg-type]
        return None

    @staticmethod
    def _current_gap_length(
        values: List[Optional[float]],
        index: int,
    ) -> int:
        """Compute the length of the gap containing the given index.

        Scans backward and forward from ``index`` to find the full
        extent of consecutive missing values.

        Args:
            values: Series values.
            index: Position within a gap.

        Returns:
            Total length of the contiguous gap.
        """
        n = len(values)
        # Scan backward
        start_idx = index
        while start_idx > 0 and _is_missing(values[start_idx - 1]):
            start_idx -= 1
        # Scan forward
        end_idx = index
        while end_idx < n - 1 and _is_missing(values[end_idx + 1]):
            end_idx += 1
        return end_idx - start_idx + 1

    # ==================================================================
    # Private helpers: Holt-Winters initialization
    # ==================================================================

    def _init_holt_winters(
        self,
        values: List[Optional[float]],
        period: int,
    ) -> Tuple[float, float, List[float]]:
        """Initialize Holt-Winters level, trend, and seasonal components.

        Uses the first complete cycle of known values to estimate
        initial seasonal indices, level, and trend.

        Args:
            values: Series values.
            period: Seasonal period.

        Returns:
            Tuple of (level, trend, seasonal_list).
        """
        known = self._collect_known_values(values)

        # Initial level: mean of first cycle
        first_cycle = known[:period]
        level = sum(first_cycle) / len(first_cycle)

        # Initial trend: average slope between first two cycles
        if len(known) >= period * 2:
            second_cycle = known[period:period * 2]
            cycle1_mean = sum(first_cycle) / len(first_cycle)
            cycle2_mean = sum(second_cycle) / len(second_cycle)
            trend = (cycle2_mean - cycle1_mean) / period
        else:
            trend = 0.0

        # Initial seasonal indices: deviation from cycle mean
        seasonal = [0.0] * period
        n_cycles = max(1, len(known) // period)
        for pos in range(period):
            total = 0.0
            count = 0
            for c in range(n_cycles):
                idx = c * period + pos
                if idx < len(known):
                    total += known[idx]
                    count += 1
            if count > 0:
                seasonal[pos] = (total / count) - level

        return level, trend, seasonal

    # ==================================================================
    # Private helpers: quality estimation
    # ==================================================================

    def _ses_quality(
        self,
        values: List[Optional[float]],
        alpha: float,
    ) -> float:
        """Estimate SES quality by computing one-step-ahead error.

        Runs SES on known values and computes normalised RMSE as a
        proxy for R-squared.

        Args:
            values: Series values.
            alpha: Smoothing coefficient.

        Returns:
            Quality score (0.0-1.0).
        """
        known = self._collect_known_values(values)
        if len(known) < 3:
            return 0.3

        smoothed = known[0]
        errors: List[float] = []
        for i in range(1, len(known)):
            error = known[i] - smoothed
            errors.append(error * error)
            smoothed = alpha * known[i] + (1.0 - alpha) * smoothed

        mse = sum(errors) / len(errors)
        rmse = math.sqrt(mse)

        # Normalise against data range
        data_range = max(known) - min(known)
        if data_range == 0:
            return 0.8

        nrmse = rmse / data_range
        quality = max(0.0, min(1.0, 1.0 - nrmse))
        return quality

    def _double_exp_quality(
        self,
        values: List[Optional[float]],
        alpha: float,
        beta: float,
    ) -> float:
        """Estimate Holt's double exponential quality.

        Runs the Holt model on known values and computes normalised
        RMSE as a quality proxy.

        Args:
            values: Series values.
            alpha: Level coefficient.
            beta: Trend coefficient.

        Returns:
            Quality score (0.0-1.0).
        """
        known = self._collect_known_values(values)
        if len(known) < 4:
            return 0.3

        level = known[0]
        trend = known[1] - known[0]
        errors: List[float] = []

        for i in range(2, len(known)):
            predicted = level + trend
            error = known[i] - predicted
            errors.append(error * error)

            prev_level = level
            level = alpha * known[i] + (1.0 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1.0 - beta) * trend

        mse = sum(errors) / len(errors)
        rmse = math.sqrt(mse)

        data_range = max(known) - min(known)
        if data_range == 0:
            return 0.8

        nrmse = rmse / data_range
        quality = max(0.0, min(1.0, 1.0 - nrmse))
        return quality

    def _holt_winters_quality(
        self,
        values: List[Optional[float]],
        period: int,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> float:
        """Estimate Holt-Winters quality via one-step-ahead NRMSE.

        Args:
            values: Series values.
            period: Seasonal period.
            alpha: Level coefficient.
            beta: Trend coefficient.
            gamma: Seasonal coefficient.

        Returns:
            Quality score (0.0-1.0).
        """
        known = self._collect_known_values(values)
        if len(known) < period * 2 + 2:
            return 0.4

        # Initialize
        level, trend, seasonal = self._init_holt_winters(values, period)

        errors: List[float] = []
        for i in range(period, len(known)):
            s_idx = i % period
            predicted = level + trend + seasonal[s_idx]
            error = known[i] - predicted
            errors.append(error * error)

            prev_level = level
            level = (
                alpha * (known[i] - seasonal[s_idx])
                + (1.0 - alpha) * (level + trend)
            )
            trend = (
                beta * (level - prev_level)
                + (1.0 - beta) * trend
            )
            seasonal[s_idx] = (
                gamma * (known[i] - level)
                + (1.0 - gamma) * seasonal[s_idx]
            )

        if not errors:
            return 0.4

        mse = sum(errors) / len(errors)
        rmse = math.sqrt(mse)

        data_range = max(known) - min(known)
        if data_range == 0:
            return 0.8

        nrmse = rmse / data_range
        quality = max(0.0, min(1.0, 1.0 - nrmse))
        return quality

    # ==================================================================
    # Private helpers: trend classification
    # ==================================================================

    @staticmethod
    def _fit_quadratic_r2(
        x: List[float],
        y: List[float],
    ) -> float:
        """Fit a quadratic y = a*x^2 + b*x + c via normal equations.

        Uses a simplified 3-parameter OLS to compute R-squared for
        a quadratic fit, used to detect polynomial trends.

        Args:
            x: Independent variable values.
            y: Dependent variable values.

        Returns:
            R-squared for the quadratic fit.
        """
        n = len(x)
        if n < 3:
            return 0.0

        # Build sums for the 3x3 normal equation system
        # y = a*x^2 + b*x + c
        sx = sum(x)
        sy = sum(y)
        sx2 = sum(xi * xi for xi in x)
        sx3 = sum(xi * xi * xi for xi in x)
        sx4 = sum(xi ** 4 for xi in x)
        sxy = sum(xi * yi for xi, yi in zip(x, y))
        sx2y = sum(xi * xi * yi for xi, yi in zip(x, y))

        # Solve via Cramer's rule for the 3x3 system
        det_m = (
            sx4 * (sx2 * n - sx * sx)
            - sx3 * (sx3 * n - sx * sx2)
            + sx2 * (sx3 * sx - sx2 * sx2)
        )

        if abs(det_m) < 1e-12:
            return 0.0

        det_a = (
            sx2y * (sx2 * n - sx * sx)
            - sx3 * (sxy * n - sx * sy)
            + sx2 * (sxy * sx - sx2 * sy)
        )
        det_b = (
            sx4 * (sxy * n - sx * sy)
            - sx2y * (sx3 * n - sx * sx2)
            + sx2 * (sx3 * sy - sxy * sx2)
        )
        det_c = (
            sx4 * (sx2 * sy - sx * sxy)
            - sx3 * (sx3 * sy - sx * sx2y)
            + sx2y * (sx3 * sx - sx2 * sx2)
        )

        a_coef = det_a / det_m
        b_coef = det_b / det_m
        c_coef = det_c / det_m

        # Compute R-squared
        y_mean = sy / n
        ss_res = 0.0
        ss_tot = 0.0
        for xi, yi in zip(x, y):
            y_pred = a_coef * xi * xi + b_coef * xi + c_coef
            ss_res += (yi - y_pred) ** 2
            ss_tot += (yi - y_mean) ** 2

        if ss_tot == 0.0:
            return 1.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    @staticmethod
    def _classify_trend(
        r2_linear: float,
        r2_exponential: float,
        r2_quadratic: float,
    ) -> TrendType:
        """Classify trend type based on R-squared values.

        Decision logic:
            1. If exponential R2 > linear + 0.05 and > 0.8: EXPONENTIAL
            2. If quadratic R2 > linear + 0.1 and > 0.8: POLYNOMIAL
            3. If linear R2 > 0.8: LINEAR
            4. If linear R2 > 0.5: MODERATE_LINEAR
            5. Otherwise: STATIONARY

        Args:
            r2_linear: R-squared from linear fit.
            r2_exponential: R-squared from log-linear fit.
            r2_quadratic: R-squared from quadratic fit.

        Returns:
            TrendType classification.
        """
        # Check for exponential (log-linear fit is better)
        if (
            r2_exponential > r2_linear + 0.05
            and r2_exponential > 0.8
        ):
            return TrendType.EXPONENTIAL

        # Check for polynomial (quadratic is significantly better)
        if (
            r2_quadratic > r2_linear + 0.1
            and r2_quadratic > 0.8
        ):
            return TrendType.POLYNOMIAL

        # Linear classification by R-squared threshold
        if r2_linear > 0.8:
            return TrendType.LINEAR
        if r2_linear > 0.5:
            return TrendType.MODERATE_LINEAR

        return TrendType.STATIONARY

    # ==================================================================
    # Private helpers: result builders
    # ==================================================================

    def _build_unfilled_result(
        self,
        values: List[Optional[float]],
        method: FillMethod,
        start_time: float,
    ) -> Dict[str, Any]:
        """Build an unfilled result dict when filling cannot be performed.

        Returns the original series unchanged with zero confidence
        for missing points.

        Args:
            values: Original series values.
            method: Fill method that was attempted.
            start_time: ``time.time()`` when processing started.

        Returns:
            Dict with original values and zero-confidence fills.
        """
        n = len(values)
        filled_values: List[Optional[float]] = []
        fill_points: List[FillPoint] = []
        missing_count = 0

        for i, v in enumerate(values):
            is_miss = _is_missing(v)
            if is_miss:
                filled_values.append(None)
                missing_count += 1
                fill_points.append(FillPoint(
                    index=i,
                    original_value=None,
                    filled_value=0.0,
                    was_missing=True,
                    method=method,
                    confidence=0.0,
                    provenance_hash=self._provenance.build_hash({
                        "operation": "unfilled",
                        "index": i,
                        "reason": "insufficient_data",
                    }),
                ))
            else:
                val = float(v)  # type: ignore[arg-type]
                filled_values.append(val)
                fill_points.append(FillPoint(
                    index=i,
                    original_value=val,
                    filled_value=val,
                    was_missing=False,
                    method=method,
                    confidence=1.0,
                    provenance_hash=self._provenance.build_hash({
                        "operation": "unfilled",
                        "index": i,
                        "value": val,
                    }),
                ))

        elapsed = time.time() - start_time
        result_hash = self._provenance.build_hash({
            "operation": "unfilled",
            "method": method.value,
            "n": n,
            "missing": missing_count,
        })

        return {
            "method": method.value,
            "filled_values": filled_values,
            "fill_points": fill_points,
            "gaps_filled": 0,
            "points_filled": 0,
            "total_points": n,
            "fill_count": 0,
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "confidence": 0.0,
            "confidence_scores": [fp.confidence for fp in fill_points],
            "r_squared": None,
            "processing_time_ms": elapsed * 1000,
            "metadata": {"reason": "insufficient_data"},
            "provenance_hash": result_hash,
        }


__all__ = [
    "TrendExtrapolatorEngine",
]
