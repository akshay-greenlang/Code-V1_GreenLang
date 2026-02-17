# -*- coding: utf-8 -*-
"""
Temporal Outlier Detection Engine - AGENT-DATA-013

Time-series anomaly detection methods including CUSUM change-point
detection, trend break detection, seasonal residual analysis, moving
window anomaly detection, and EWMA control charts.

Zero-Hallucination: All calculations use deterministic Python
arithmetic. No LLM calls for numeric computations.

Example:
    >>> from greenlang.outlier_detector.temporal_detector import TemporalDetectorEngine
    >>> engine = TemporalDetectorEngine()
    >>> results = engine.detect_cusum([10, 11, 12, 50, 51, 52])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

from greenlang.outlier_detector.config import get_config
from greenlang.outlier_detector.models import (
    DetectionMethod,
    OutlierScore,
    SeverityLevel,
    TemporalMethod,
    TemporalResult,
)
from greenlang.outlier_detector.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: List[float], mean: Optional[float] = None) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _safe_mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _safe_median(values: List[float]) -> float:
    """Compute median of values."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _severity_from_score(score: float) -> SeverityLevel:
    """Map normalised score to severity level."""
    if score >= 0.95:
        return SeverityLevel.CRITICAL
    if score >= 0.80:
        return SeverityLevel.HIGH
    if score >= 0.60:
        return SeverityLevel.MEDIUM
    if score >= 0.40:
        return SeverityLevel.LOW
    return SeverityLevel.INFO


class TemporalDetectorEngine:
    """Time-series anomaly detection engine.

    Implements five temporal detection methods for identifying
    anomalies in time-ordered data, including change-point detection,
    trend breaks, seasonal residual analysis, moving window, and EWMA.

    Attributes:
        _config: Outlier detector configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = TemporalDetectorEngine()
        >>> results = engine.detect_ewma([10, 11, 12, 100, 11, 12])
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize TemporalDetectorEngine.

        Args:
            config: Optional OutlierDetectorConfig override.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        logger.info("TemporalDetectorEngine initialized")

    # ------------------------------------------------------------------
    # CUSUM Change-Point Detection
    # ------------------------------------------------------------------

    def detect_cusum(
        self,
        series: List[float],
        threshold: Optional[float] = None,
        drift: Optional[float] = None,
        column_name: str = "",
    ) -> List[TemporalResult]:
        """Detect change points using CUSUM (Cumulative Sum).

        Maintains upper (S_h) and lower (S_l) cumulative sums. When either
        exceeds the threshold, a change point is detected and the CUSUM
        is reset.

        Args:
            series: Time-ordered numeric values.
            threshold: CUSUM threshold for alarm (default: 4 * std).
            drift: Allowable drift (default: std / 2).
            column_name: Column name for provenance.

        Returns:
            List containing a single TemporalResult with change points.
        """
        start = time.time()
        n = len(series)
        if n < 5:
            return self._empty_temporal_result(
                series, TemporalMethod.CUSUM, column_name,
            )

        mean = _safe_mean(series)
        std = _safe_std(series, mean)
        h = threshold if threshold is not None else 4.0 * std
        k = drift if drift is not None else std / 2.0

        s_high = 0.0
        s_low = 0.0
        change_points: List[int] = []
        scores: List[OutlierScore] = []

        for i, v in enumerate(series):
            s_high = max(0.0, s_high + (v - mean) - k)
            s_low = max(0.0, s_low - (v - mean) - k)

            is_change = s_high > h or s_low > h
            if is_change:
                change_points.append(i)
                s_high = 0.0
                s_low = 0.0

            cusum_val = max(s_high, s_low)
            score = min(1.0, cusum_val / (h + 1e-10))

            provenance_hash = self._provenance.build_hash({
                "method": "cusum", "index": i, "value": v,
                "s_high": s_high, "s_low": s_low,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.TEMPORAL,
                score=score,
                is_outlier=is_change,
                threshold=h,
                severity=_severity_from_score(score),
                details={"cusum_high": s_high, "cusum_low": s_low,
                         "drift": k, "is_change_point": is_change},
                confidence=min(1.0, 0.5 + score * 0.5),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        result_hash = self._provenance.build_hash({
            "method": "cusum", "n": n, "changes": len(change_points),
        })

        return [TemporalResult(
            method=TemporalMethod.CUSUM,
            column_name=column_name,
            series_length=n,
            anomalies_found=len(change_points),
            change_points=change_points,
            scores=scores,
            baseline_mean=mean,
            baseline_std=std,
            confidence=min(1.0, 0.5 + len(change_points) / max(n, 1) * 5.0),
            provenance_hash=result_hash,
        )]

    # ------------------------------------------------------------------
    # Trend Break Detection
    # ------------------------------------------------------------------

    def detect_trend_breaks(
        self,
        series: List[float],
        window: Optional[int] = None,
        column_name: str = "",
    ) -> List[TemporalResult]:
        """Detect sudden trend changes using rolling slope comparison.

        Computes the rolling slope over a window and flags points where
        the slope change exceeds a threshold derived from the standard
        deviation of slopes.

        Args:
            series: Time-ordered numeric values.
            window: Rolling window size (default: max(5, len/10)).
            column_name: Column name for provenance.

        Returns:
            List containing a single TemporalResult.
        """
        start = time.time()
        n = len(series)
        if n < 10:
            return self._empty_temporal_result(
                series, TemporalMethod.TREND_BREAK, column_name,
            )

        w = window if window is not None else max(5, n // 10)
        w = min(w, n // 2)

        # Compute rolling slopes
        slopes: List[float] = []
        for i in range(w, n):
            segment = series[i - w:i]
            slope = self._compute_slope(segment)
            slopes.append(slope)

        if len(slopes) < 3:
            return self._empty_temporal_result(
                series, TemporalMethod.TREND_BREAK, column_name,
            )

        slope_mean = _safe_mean(slopes)
        slope_std = _safe_std(slopes, slope_mean)
        threshold_val = 2.5 * slope_std if slope_std > 0 else 1.0

        # Detect slope changes
        change_points: List[int] = []
        scores: List[OutlierScore] = []

        for i in range(n):
            if i < w:
                # Not enough data for slope
                provenance_hash = self._provenance.build_hash({
                    "method": "trend_break", "index": i, "value": series[i],
                })
                scores.append(OutlierScore(
                    record_index=i,
                    column_name=column_name,
                    value=series[i],
                    method=DetectionMethod.TEMPORAL,
                    score=0.0,
                    is_outlier=False,
                    threshold=threshold_val,
                    severity=SeverityLevel.INFO,
                    details={"reason": "warmup_period"},
                    confidence=0.0,
                    provenance_hash=provenance_hash,
                ))
                continue

            slope_idx = i - w
            if slope_idx >= len(slopes):
                slope_idx = len(slopes) - 1

            slope_change = abs(slopes[slope_idx] - slope_mean)
            is_break = slope_change > threshold_val
            if is_break:
                change_points.append(i)

            score = min(1.0, slope_change / (threshold_val * 2.0 + 1e-10))
            provenance_hash = self._provenance.build_hash({
                "method": "trend_break", "index": i, "value": series[i],
                "slope_change": slope_change,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=series[i],
                method=DetectionMethod.TEMPORAL,
                score=score,
                is_outlier=is_break,
                threshold=threshold_val,
                severity=_severity_from_score(score),
                details={"slope_change": slope_change,
                         "slope_mean": slope_mean, "slope_std": slope_std},
                confidence=min(1.0, 0.4 + score * 0.6),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        result_hash = self._provenance.build_hash({
            "method": "trend_break", "n": n, "breaks": len(change_points),
        })

        return [TemporalResult(
            method=TemporalMethod.TREND_BREAK,
            column_name=column_name,
            series_length=n,
            anomalies_found=len(change_points),
            change_points=change_points,
            scores=scores,
            baseline_mean=_safe_mean(series),
            baseline_std=_safe_std(series),
            confidence=min(1.0, 0.5 + len(change_points) / max(n, 1) * 5.0),
            provenance_hash=result_hash,
        )]

    # ------------------------------------------------------------------
    # Seasonal Residual Detection
    # ------------------------------------------------------------------

    def detect_seasonal_residuals(
        self,
        series: List[float],
        period: Optional[int] = None,
        column_name: str = "",
    ) -> List[TemporalResult]:
        """Detect outliers in seasonal residuals.

        Decomposes the series by subtracting a seasonal component
        (period-averaged), then detects outliers in the residuals
        using z-score.

        Args:
            series: Time-ordered numeric values.
            period: Seasonal period (default: 12).
            column_name: Column name for provenance.

        Returns:
            List containing a single TemporalResult.
        """
        start = time.time()
        n = len(series)
        p = period if period is not None else 12

        if n < p * 2:
            return self._empty_temporal_result(
                series, TemporalMethod.SEASONAL_RESIDUAL, column_name,
            )

        # Compute seasonal component (average by position in cycle)
        seasonal = [0.0] * p
        counts = [0] * p
        for i, v in enumerate(series):
            pos = i % p
            seasonal[pos] += v
            counts[pos] += 1

        for pos in range(p):
            if counts[pos] > 0:
                seasonal[pos] /= counts[pos]

        # Compute residuals
        residuals = [series[i] - seasonal[i % p] for i in range(n)]
        res_mean = _safe_mean(residuals)
        res_std = _safe_std(residuals, res_mean)
        threshold_val = self._config.zscore_threshold

        change_points: List[int] = []
        scores: List[OutlierScore] = []

        for i, r in enumerate(residuals):
            if res_std > 0:
                z = abs(r - res_mean) / res_std
                score = min(1.0, z / (threshold_val * 2.0))
            else:
                z = 0.0
                score = 0.0

            is_outlier = z > threshold_val
            if is_outlier:
                change_points.append(i)

            provenance_hash = self._provenance.build_hash({
                "method": "seasonal_residual", "index": i,
                "value": series[i], "residual": r,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=series[i],
                method=DetectionMethod.TEMPORAL,
                score=score,
                is_outlier=is_outlier,
                threshold=threshold_val,
                severity=_severity_from_score(score),
                details={"residual": r, "seasonal_value": seasonal[i % p],
                         "z_score": z, "period": p},
                confidence=min(1.0, 0.5 + score * 0.5),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        result_hash = self._provenance.build_hash({
            "method": "seasonal_residual", "n": n,
            "anomalies": len(change_points), "period": p,
        })

        return [TemporalResult(
            method=TemporalMethod.SEASONAL_RESIDUAL,
            column_name=column_name,
            series_length=n,
            anomalies_found=len(change_points),
            change_points=change_points,
            scores=scores,
            baseline_mean=res_mean,
            baseline_std=res_std,
            confidence=min(1.0, 0.6 + (n / (p * 4)) * 0.2),
            provenance_hash=result_hash,
        )]

    # ------------------------------------------------------------------
    # Moving Window Detection
    # ------------------------------------------------------------------

    def detect_moving_window(
        self,
        series: List[float],
        window: Optional[int] = None,
        threshold: Optional[float] = None,
        column_name: str = "",
    ) -> List[TemporalResult]:
        """Detect anomalies using a moving window comparison.

        For each point, computes the mean and std of the preceding
        window and flags the point if it deviates by more than
        threshold standard deviations.

        Args:
            series: Time-ordered numeric values.
            window: Window size (default: max(5, len/10)).
            threshold: Std dev multiplier (default from config zscore_threshold).
            column_name: Column name for provenance.

        Returns:
            List containing a single TemporalResult.
        """
        start = time.time()
        n = len(series)
        w = window if window is not None else max(5, n // 10)
        w = min(w, n // 2)
        t = threshold if threshold is not None else self._config.zscore_threshold

        if n < w + 1:
            return self._empty_temporal_result(
                series, TemporalMethod.MOVING_WINDOW, column_name,
            )

        change_points: List[int] = []
        scores: List[OutlierScore] = []

        for i in range(n):
            if i < w:
                provenance_hash = self._provenance.build_hash({
                    "method": "moving_window", "index": i, "value": series[i],
                })
                scores.append(OutlierScore(
                    record_index=i,
                    column_name=column_name,
                    value=series[i],
                    method=DetectionMethod.TEMPORAL,
                    score=0.0,
                    is_outlier=False,
                    threshold=t,
                    severity=SeverityLevel.INFO,
                    details={"reason": "warmup_period"},
                    confidence=0.0,
                    provenance_hash=provenance_hash,
                ))
                continue

            window_vals = series[i - w:i]
            win_mean = _safe_mean(window_vals)
            win_std = _safe_std(window_vals, win_mean)

            if win_std > 0:
                z = abs(series[i] - win_mean) / win_std
                score = min(1.0, z / (t * 2.0))
            else:
                z = 0.0
                score = 0.0

            is_outlier = z > t
            if is_outlier:
                change_points.append(i)

            provenance_hash = self._provenance.build_hash({
                "method": "moving_window", "index": i,
                "value": series[i], "z": z,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=series[i],
                method=DetectionMethod.TEMPORAL,
                score=score,
                is_outlier=is_outlier,
                threshold=t,
                severity=_severity_from_score(score),
                details={"window_mean": win_mean, "window_std": win_std,
                         "z_score": z, "window_size": w},
                confidence=min(1.0, 0.5 + score * 0.5),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        result_hash = self._provenance.build_hash({
            "method": "moving_window", "n": n, "anomalies": len(change_points),
        })

        return [TemporalResult(
            method=TemporalMethod.MOVING_WINDOW,
            column_name=column_name,
            series_length=n,
            anomalies_found=len(change_points),
            change_points=change_points,
            scores=scores,
            baseline_mean=_safe_mean(series),
            baseline_std=_safe_std(series),
            confidence=min(1.0, 0.5 + len(scores) / max(n, 1) * 0.5),
            provenance_hash=result_hash,
        )]

    # ------------------------------------------------------------------
    # EWMA Control Chart
    # ------------------------------------------------------------------

    def detect_ewma(
        self,
        series: List[float],
        alpha: float = 0.3,
        threshold: Optional[float] = None,
        column_name: str = "",
    ) -> List[TemporalResult]:
        """Detect anomalies using EWMA (Exponentially Weighted Moving Average).

        Maintains an EWMA and EWMA variance. Flags points where the
        current value deviates from the EWMA by more than threshold
        times the EWMA standard deviation.

        Args:
            series: Time-ordered numeric values.
            alpha: Smoothing parameter (0 < alpha <= 1, default 0.3).
            threshold: Deviation threshold in std devs (default from config).
            column_name: Column name for provenance.

        Returns:
            List containing a single TemporalResult.
        """
        start = time.time()
        n = len(series)
        t = threshold if threshold is not None else self._config.zscore_threshold

        if n < 5:
            return self._empty_temporal_result(
                series, TemporalMethod.EWMA, column_name,
            )

        # Clamp alpha
        alpha = max(0.01, min(1.0, alpha))
        overall_std = _safe_std(series)

        ewma = series[0]
        # EWMA variance: sigma^2 * (alpha / (2 - alpha))
        ewma_std = overall_std * math.sqrt(alpha / (2.0 - alpha)) if overall_std > 0 else 1.0

        change_points: List[int] = []
        scores: List[OutlierScore] = []

        for i, v in enumerate(series):
            if ewma_std > 0:
                deviation = abs(v - ewma) / ewma_std
                score = min(1.0, deviation / (t * 2.0))
            else:
                deviation = 0.0
                score = 0.0

            is_outlier = deviation > t
            if is_outlier:
                change_points.append(i)

            provenance_hash = self._provenance.build_hash({
                "method": "ewma", "index": i, "value": v,
                "ewma": ewma, "deviation": deviation,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.TEMPORAL,
                score=score,
                is_outlier=is_outlier,
                threshold=t,
                severity=_severity_from_score(score),
                details={"ewma": ewma, "ewma_std": ewma_std,
                         "deviation": deviation, "alpha": alpha},
                confidence=min(1.0, 0.5 + score * 0.5),
                provenance_hash=provenance_hash,
            ))

            # Update EWMA
            ewma = alpha * v + (1.0 - alpha) * ewma

        elapsed = time.time() - start
        result_hash = self._provenance.build_hash({
            "method": "ewma", "n": n, "anomalies": len(change_points),
        })

        return [TemporalResult(
            method=TemporalMethod.EWMA,
            column_name=column_name,
            series_length=n,
            anomalies_found=len(change_points),
            change_points=change_points,
            scores=scores,
            baseline_mean=_safe_mean(series),
            baseline_std=overall_std,
            confidence=min(1.0, 0.5 + len(scores) / max(n, 1) * 0.5),
            provenance_hash=result_hash,
        )]

    # ------------------------------------------------------------------
    # Change Point Detection (summary utility)
    # ------------------------------------------------------------------

    def detect_change_points(
        self,
        series: List[float],
        column_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Locate regime change points using multiple methods.

        Runs CUSUM and moving window detectors and returns the union
        of detected change points with metadata.

        Args:
            series: Time-ordered numeric values.
            column_name: Column name.

        Returns:
            List of dicts with keys: index, value, method, score.
        """
        change_points: List[Dict[str, Any]] = []

        cusum_results = self.detect_cusum(series, column_name=column_name)
        for result in cusum_results:
            for cp in result.change_points:
                change_points.append({
                    "index": cp,
                    "value": series[cp] if cp < len(series) else None,
                    "method": "cusum",
                    "score": result.scores[cp].score if cp < len(result.scores) else 0.0,
                })

        mw_results = self.detect_moving_window(series, column_name=column_name)
        for result in mw_results:
            for cp in result.change_points:
                change_points.append({
                    "index": cp,
                    "value": series[cp] if cp < len(series) else None,
                    "method": "moving_window",
                    "score": result.scores[cp].score if cp < len(result.scores) else 0.0,
                })

        # Deduplicate by index, keeping highest score
        seen: Dict[int, Dict[str, Any]] = {}
        for cp in change_points:
            idx = cp["index"]
            if idx not in seen or cp["score"] > seen[idx]["score"]:
                seen[idx] = cp

        return sorted(seen.values(), key=lambda x: x["index"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _empty_temporal_result(
        self,
        series: List[float],
        method: TemporalMethod,
        column_name: str,
    ) -> List[TemporalResult]:
        """Return empty result for insufficient data.

        Args:
            series: Input series.
            method: Temporal method.
            column_name: Column name.

        Returns:
            Single-element list with empty TemporalResult.
        """
        scores = []
        for i, v in enumerate(series):
            provenance_hash = self._provenance.build_hash({
                "method": method.value, "index": i, "value": v,
                "reason": "insufficient_data",
            })
            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.TEMPORAL,
                score=0.0,
                is_outlier=False,
                threshold=0.0,
                severity=SeverityLevel.INFO,
                details={"reason": "insufficient_data"},
                confidence=0.0,
                provenance_hash=provenance_hash,
            ))

        result_hash = self._provenance.build_hash({
            "method": method.value, "n": len(series),
            "reason": "insufficient_data",
        })

        return [TemporalResult(
            method=method,
            column_name=column_name,
            series_length=len(series),
            anomalies_found=0,
            change_points=[],
            scores=scores,
            baseline_mean=_safe_mean(series),
            baseline_std=_safe_std(series),
            confidence=0.0,
            provenance_hash=result_hash,
        )]

    @staticmethod
    def _compute_slope(segment: List[float]) -> float:
        """Compute linear slope of a segment using least squares.

        Args:
            segment: Numeric values (x = 0, 1, 2, ...).

        Returns:
            Slope value.
        """
        n = len(segment)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = sum(segment) / n

        num = 0.0
        den = 0.0
        for i, y in enumerate(segment):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx

        if den == 0:
            return 0.0
        return num / den


__all__ = [
    "TemporalDetectorEngine",
]
