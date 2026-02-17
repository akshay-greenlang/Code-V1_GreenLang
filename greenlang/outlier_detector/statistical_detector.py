# -*- coding: utf-8 -*-
"""
Statistical Outlier Detection Engine - AGENT-DATA-013

Pure-Python statistical outlier detection methods including IQR,
z-score, modified z-score (MAD-based), MAD, Grubbs test, Tukey
fences, and percentile-based detection. Also provides an ensemble
method that runs multiple detectors and combines their scores.

Zero-Hallucination: All calculations use deterministic Python
arithmetic. No LLM calls for numeric computations.

Example:
    >>> from greenlang.outlier_detector.statistical_detector import StatisticalDetectorEngine
    >>> engine = StatisticalDetectorEngine()
    >>> scores = engine.detect_iqr([1, 2, 3, 4, 100])
    >>> outliers = [s for s in scores if s.is_outlier]

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
    EnsembleMethod,
    EnsembleResult,
    OutlierScore,
    SeverityLevel,
)
from greenlang.outlier_detector.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


def _severity_from_score(score: float) -> SeverityLevel:
    """Map a normalised outlier score to a severity level.

    Args:
        score: Normalised outlier score (0.0-1.0).

    Returns:
        Severity classification.
    """
    if score >= 0.95:
        return SeverityLevel.CRITICAL
    if score >= 0.80:
        return SeverityLevel.HIGH
    if score >= 0.60:
        return SeverityLevel.MEDIUM
    if score >= 0.40:
        return SeverityLevel.LOW
    return SeverityLevel.INFO


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
        Standard deviation or 0.0.
    """
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _safe_mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _safe_median(values: List[float]) -> float:
    """Compute median of sorted values.

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


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile using linear interpolation.

    Args:
        values: List of numeric values.
        pct: Percentile (0.0-1.0).

    Returns:
        Percentile value.
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    k = pct * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


class StatisticalDetectorEngine:
    """Pure-Python statistical outlier detection engine.

    Implements seven univariate statistical detection methods plus an
    ensemble combiner. Each method returns a list of OutlierScore objects
    with normalised scores and provenance hashes.

    Attributes:
        _config: Outlier detector configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = StatisticalDetectorEngine()
        >>> scores = engine.detect_zscore([10, 12, 11, 13, 200])
        >>> assert any(s.is_outlier for s in scores)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize StatisticalDetectorEngine.

        Args:
            config: Optional OutlierDetectorConfig override.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        logger.info("StatisticalDetectorEngine initialized")

    # ------------------------------------------------------------------
    # IQR Detection
    # ------------------------------------------------------------------

    def detect_iqr(
        self,
        values: List[float],
        multiplier: Optional[float] = None,
        column_name: str = "",
    ) -> List[OutlierScore]:
        """Detect outliers using Interquartile Range fences.

        Computes Q1, Q3, IQR = Q3 - Q1, then flags points outside
        [Q1 - k*IQR, Q3 + k*IQR] where k is the multiplier.

        Args:
            values: List of numeric values.
            multiplier: IQR multiplier (default from config).
            column_name: Column name for provenance.

        Returns:
            List of OutlierScore for each value.
        """
        start = time.time()
        k = multiplier if multiplier is not None else self._config.iqr_multiplier
        if len(values) < 4:
            return self._no_detection_scores(
                values, DetectionMethod.IQR, column_name,
            )

        q1 = _percentile(values, 0.25)
        q3 = _percentile(values, 0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr

        scores = []
        for i, v in enumerate(values):
            is_outlier = v < lower or v > upper
            # Normalise score: distance from fence / fence range
            if iqr > 0:
                raw = max(0.0, max(lower - v, v - upper)) / (k * iqr)
                score = min(1.0, raw / 2.0)
            else:
                score = 0.0

            provenance_hash = self._provenance.build_hash({
                "method": "iqr", "index": i, "value": v,
                "q1": q1, "q3": q3, "iqr": iqr,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.IQR,
                score=score,
                is_outlier=is_outlier,
                threshold=k,
                severity=_severity_from_score(score),
                details={"q1": q1, "q3": q3, "iqr": iqr,
                         "lower_fence": lower, "upper_fence": upper},
                confidence=min(1.0, 0.5 + score * 0.5),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        logger.debug("IQR detection: %d values, %d outliers, %.3fs",
                      len(values), sum(1 for s in scores if s.is_outlier), elapsed)
        return scores

    # ------------------------------------------------------------------
    # Z-Score Detection
    # ------------------------------------------------------------------

    def detect_zscore(
        self,
        values: List[float],
        threshold: Optional[float] = None,
        column_name: str = "",
    ) -> List[OutlierScore]:
        """Detect outliers using standard z-score.

        Flags points where |z| > threshold.

        Args:
            values: List of numeric values.
            threshold: Z-score threshold (default from config).
            column_name: Column name for provenance.

        Returns:
            List of OutlierScore for each value.
        """
        t = threshold if threshold is not None else self._config.zscore_threshold
        if len(values) < 3:
            return self._no_detection_scores(
                values, DetectionMethod.ZSCORE, column_name,
            )

        mean = _safe_mean(values)
        std = _safe_std(values, mean)

        scores = []
        for i, v in enumerate(values):
            if std > 0:
                z = abs(v - mean) / std
                score = min(1.0, z / (t * 2.0))
            else:
                z = 0.0
                score = 0.0

            is_outlier = z > t
            provenance_hash = self._provenance.build_hash({
                "method": "zscore", "index": i, "value": v,
                "mean": mean, "std": std, "z": z,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.ZSCORE,
                score=score,
                is_outlier=is_outlier,
                threshold=t,
                severity=_severity_from_score(score),
                details={"mean": mean, "std": std, "z_score": z},
                confidence=min(1.0, 0.5 + score * 0.5),
                provenance_hash=provenance_hash,
            ))

        return scores

    # ------------------------------------------------------------------
    # Modified Z-Score (MAD-based)
    # ------------------------------------------------------------------

    def detect_modified_zscore(
        self,
        values: List[float],
        threshold: Optional[float] = None,
        column_name: str = "",
    ) -> List[OutlierScore]:
        """Detect outliers using MAD-based modified z-score.

        Modified z-score = 0.6745 * (x - median) / MAD.
        More robust to outliers than standard z-score.

        Args:
            values: List of numeric values.
            threshold: Modified z-score threshold (default from config).
            column_name: Column name for provenance.

        Returns:
            List of OutlierScore for each value.
        """
        t = threshold if threshold is not None else self._config.mad_threshold
        if len(values) < 3:
            return self._no_detection_scores(
                values, DetectionMethod.MODIFIED_ZSCORE, column_name,
            )

        median = _safe_median(values)
        deviations = [abs(v - median) for v in values]
        mad = _safe_median(deviations)

        scores = []
        for i, v in enumerate(values):
            if mad > 0:
                modified_z = 0.6745 * (v - median) / mad
                abs_mz = abs(modified_z)
                score = min(1.0, abs_mz / (t * 2.0))
            else:
                modified_z = 0.0
                abs_mz = 0.0
                score = 0.0

            is_outlier = abs_mz > t
            provenance_hash = self._provenance.build_hash({
                "method": "modified_zscore", "index": i, "value": v,
                "median": median, "mad": mad, "modified_z": modified_z,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.MODIFIED_ZSCORE,
                score=score,
                is_outlier=is_outlier,
                threshold=t,
                severity=_severity_from_score(score),
                details={"median": median, "mad": mad,
                         "modified_zscore": modified_z},
                confidence=min(1.0, 0.6 + score * 0.4),
                provenance_hash=provenance_hash,
            ))

        return scores

    # ------------------------------------------------------------------
    # MAD Detection
    # ------------------------------------------------------------------

    def detect_mad(
        self,
        values: List[float],
        threshold: Optional[float] = None,
        column_name: str = "",
    ) -> List[OutlierScore]:
        """Detect outliers using Median Absolute Deviation.

        Flags points where |x - median| > threshold * MAD.

        Args:
            values: List of numeric values.
            threshold: MAD multiplier threshold (default from config).
            column_name: Column name for provenance.

        Returns:
            List of OutlierScore for each value.
        """
        t = threshold if threshold is not None else self._config.mad_threshold
        if len(values) < 3:
            return self._no_detection_scores(
                values, DetectionMethod.MAD, column_name,
            )

        median = _safe_median(values)
        deviations = [abs(v - median) for v in values]
        mad = _safe_median(deviations)

        scores = []
        for i, v in enumerate(values):
            dev = abs(v - median)
            if mad > 0:
                ratio = dev / mad
                score = min(1.0, ratio / (t * 2.0))
            else:
                ratio = 0.0
                score = 0.0

            is_outlier = ratio > t if mad > 0 else False
            provenance_hash = self._provenance.build_hash({
                "method": "mad", "index": i, "value": v,
                "median": median, "mad": mad, "deviation": dev,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.MAD,
                score=score,
                is_outlier=is_outlier,
                threshold=t,
                severity=_severity_from_score(score),
                details={"median": median, "mad": mad,
                         "deviation": dev, "ratio": ratio},
                confidence=min(1.0, 0.6 + score * 0.4),
                provenance_hash=provenance_hash,
            ))

        return scores

    # ------------------------------------------------------------------
    # Grubbs Test
    # ------------------------------------------------------------------

    def detect_grubbs(
        self,
        values: List[float],
        alpha: Optional[float] = None,
        column_name: str = "",
    ) -> List[OutlierScore]:
        """Detect outliers using Grubbs test for the single most extreme value.

        Tests whether the maximum or minimum value is a significant outlier
        based on the Grubbs test statistic G = max|x_i - mean| / std.

        Uses a t-distribution approximation for the critical value.

        Args:
            values: List of numeric values.
            alpha: Significance level (default from config).
            column_name: Column name for provenance.

        Returns:
            List of OutlierScore for each value.
        """
        a = alpha if alpha is not None else self._config.grubbs_alpha
        n = len(values)
        if n < 3:
            return self._no_detection_scores(
                values, DetectionMethod.GRUBBS, column_name,
            )

        mean = _safe_mean(values)
        std = _safe_std(values, mean)

        # Grubbs critical value approximation using t-distribution
        # G_critical = ((n-1)/sqrt(n)) * sqrt(t^2 / (n-2+t^2))
        # Approximate t-critical for alpha/(2*n) with (n-2) df
        # Using a conservative approximation
        t_crit = self._approx_t_critical(a / (2 * n), n - 2)
        g_critical = ((n - 1) / math.sqrt(n)) * math.sqrt(
            t_crit ** 2 / (n - 2 + t_crit ** 2)
        )

        scores = []
        for i, v in enumerate(values):
            if std > 0:
                g = abs(v - mean) / std
                score = min(1.0, g / (g_critical * 2.0)) if g_critical > 0 else 0.0
            else:
                g = 0.0
                score = 0.0

            is_outlier = g > g_critical if std > 0 else False
            provenance_hash = self._provenance.build_hash({
                "method": "grubbs", "index": i, "value": v,
                "mean": mean, "std": std, "g": g, "g_critical": g_critical,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.GRUBBS,
                score=score,
                is_outlier=is_outlier,
                threshold=g_critical,
                severity=_severity_from_score(score),
                details={"mean": mean, "std": std, "g_statistic": g,
                         "g_critical": g_critical, "alpha": a},
                confidence=min(1.0, 0.5 + score * 0.5),
                provenance_hash=provenance_hash,
            ))

        return scores

    # ------------------------------------------------------------------
    # Tukey Fences
    # ------------------------------------------------------------------

    def detect_tukey(
        self,
        values: List[float],
        column_name: str = "",
    ) -> List[OutlierScore]:
        """Detect outliers using Tukey box-plot fences.

        Inner fences: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        Outer fences: [Q1 - 3.0*IQR, Q3 + 3.0*IQR]

        Points between inner and outer are mild outliers (score ~0.5).
        Points beyond outer are extreme outliers (score ~1.0).

        Args:
            values: List of numeric values.
            column_name: Column name for provenance.

        Returns:
            List of OutlierScore for each value.
        """
        if len(values) < 4:
            return self._no_detection_scores(
                values, DetectionMethod.TUKEY, column_name,
            )

        q1 = _percentile(values, 0.25)
        q3 = _percentile(values, 0.75)
        iqr = q3 - q1
        inner_lower = q1 - 1.5 * iqr
        inner_upper = q3 + 1.5 * iqr
        outer_lower = q1 - 3.0 * iqr
        outer_upper = q3 + 3.0 * iqr

        scores = []
        for i, v in enumerate(values):
            beyond_outer = v < outer_lower or v > outer_upper
            beyond_inner = v < inner_lower or v > inner_upper

            if beyond_outer:
                score = min(1.0, 0.75 + 0.25 * min(1.0,
                    max(0.0, max(outer_lower - v, v - outer_upper)) /
                    (1.5 * iqr + 1e-10)))
            elif beyond_inner:
                score = 0.4 + 0.35 * min(1.0,
                    max(0.0, max(inner_lower - v, v - inner_upper)) /
                    (1.5 * iqr + 1e-10))
            else:
                score = 0.0

            is_outlier = beyond_inner
            provenance_hash = self._provenance.build_hash({
                "method": "tukey", "index": i, "value": v,
                "q1": q1, "q3": q3, "iqr": iqr,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.TUKEY,
                score=score,
                is_outlier=is_outlier,
                threshold=1.5,
                severity=_severity_from_score(score),
                details={
                    "q1": q1, "q3": q3, "iqr": iqr,
                    "inner_lower": inner_lower, "inner_upper": inner_upper,
                    "outer_lower": outer_lower, "outer_upper": outer_upper,
                    "beyond_inner": beyond_inner, "beyond_outer": beyond_outer,
                },
                confidence=min(1.0, 0.6 + score * 0.4),
                provenance_hash=provenance_hash,
            ))

        return scores

    # ------------------------------------------------------------------
    # Percentile Detection
    # ------------------------------------------------------------------

    def detect_percentile(
        self,
        values: List[float],
        lower: float = 0.01,
        upper: float = 0.99,
        column_name: str = "",
    ) -> List[OutlierScore]:
        """Detect outliers using percentile-based bounds.

        Flags points below the lower percentile or above the upper
        percentile as outliers.

        Args:
            values: List of numeric values.
            lower: Lower percentile (0.0-1.0, default 0.01 = 1st).
            upper: Upper percentile (0.0-1.0, default 0.99 = 99th).
            column_name: Column name for provenance.

        Returns:
            List of OutlierScore for each value.
        """
        if len(values) < 3:
            return self._no_detection_scores(
                values, DetectionMethod.PERCENTILE, column_name,
            )

        lower_val = _percentile(values, lower)
        upper_val = _percentile(values, upper)
        span = upper_val - lower_val if upper_val > lower_val else 1.0

        scores = []
        for i, v in enumerate(values):
            is_outlier = v < lower_val or v > upper_val
            if v < lower_val:
                distance = lower_val - v
                score = min(1.0, distance / span)
            elif v > upper_val:
                distance = v - upper_val
                score = min(1.0, distance / span)
            else:
                score = 0.0

            provenance_hash = self._provenance.build_hash({
                "method": "percentile", "index": i, "value": v,
                "lower_pct": lower, "upper_pct": upper,
                "lower_val": lower_val, "upper_val": upper_val,
            })

            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=DetectionMethod.PERCENTILE,
                score=score,
                is_outlier=is_outlier,
                threshold=upper - lower,
                severity=_severity_from_score(score),
                details={
                    "lower_percentile": lower, "upper_percentile": upper,
                    "lower_value": lower_val, "upper_value": upper_val,
                },
                confidence=min(1.0, 0.5 + score * 0.5),
                provenance_hash=provenance_hash,
            ))

        return scores

    # ------------------------------------------------------------------
    # Ensemble Detection
    # ------------------------------------------------------------------

    def detect_ensemble(
        self,
        values: List[float],
        methods: Optional[List[DetectionMethod]] = None,
        weights: Optional[Dict[str, float]] = None,
        column_name: str = "",
    ) -> List[EnsembleResult]:
        """Run multiple detection methods and combine scores.

        Runs each specified method, normalises scores, and combines
        them using the configured ensemble method.

        Args:
            values: List of numeric values.
            methods: Detection methods to use (default: IQR, zscore, modified_zscore).
            weights: Per-method weights (default from DEFAULT_METHOD_WEIGHTS).
            column_name: Column name for provenance.

        Returns:
            List of EnsembleResult for each value.
        """
        start = time.time()
        from greenlang.outlier_detector.models import DEFAULT_METHOD_WEIGHTS

        if methods is None:
            methods = [
                DetectionMethod.IQR,
                DetectionMethod.ZSCORE,
                DetectionMethod.MODIFIED_ZSCORE,
            ]

        w = weights or DEFAULT_METHOD_WEIGHTS
        ensemble_method = EnsembleMethod(self._config.ensemble_method)
        min_consensus = self._config.min_consensus

        # Run each method
        method_results: Dict[str, List[OutlierScore]] = {}
        dispatcher = {
            DetectionMethod.IQR: self.detect_iqr,
            DetectionMethod.ZSCORE: self.detect_zscore,
            DetectionMethod.MODIFIED_ZSCORE: self.detect_modified_zscore,
            DetectionMethod.MAD: self.detect_mad,
            DetectionMethod.GRUBBS: self.detect_grubbs,
            DetectionMethod.TUKEY: self.detect_tukey,
            DetectionMethod.PERCENTILE: self.detect_percentile,
        }

        for method in methods:
            fn = dispatcher.get(method)
            if fn is not None:
                method_results[method.value] = fn(
                    values, column_name=column_name,
                )

        if not method_results:
            return []

        # Combine scores for each data point
        n = len(values)
        ensemble_results: List[EnsembleResult] = []

        for i in range(n):
            method_scores: Dict[str, float] = {}
            methods_flagged = 0

            for method_name, scores_list in method_results.items():
                if i < len(scores_list):
                    sc = scores_list[i]
                    method_scores[method_name] = sc.score
                    if sc.is_outlier:
                        methods_flagged += 1

            combined_score = self._combine_scores(
                method_scores, w, ensemble_method,
            )

            if ensemble_method == EnsembleMethod.MAJORITY_VOTE:
                is_outlier = methods_flagged >= min_consensus
            else:
                is_outlier = combined_score >= 0.5 and methods_flagged >= min_consensus

            provenance_hash = self._provenance.build_hash({
                "method": "ensemble", "index": i, "value": values[i],
                "method_scores": method_scores, "combined": combined_score,
            })

            ensemble_results.append(EnsembleResult(
                record_index=i,
                column_name=column_name,
                value=values[i],
                ensemble_score=combined_score,
                is_outlier=is_outlier,
                method_scores=method_scores,
                methods_flagged=methods_flagged,
                total_methods=len(method_results),
                ensemble_method=ensemble_method,
                severity=_severity_from_score(combined_score),
                confidence=min(1.0, 0.4 + combined_score * 0.3
                               + methods_flagged / max(len(method_results), 1) * 0.3),
                provenance_hash=provenance_hash,
            ))

        elapsed = time.time() - start
        logger.debug(
            "Ensemble detection: %d values, %d methods, %d outliers, %.3fs",
            n, len(method_results),
            sum(1 for r in ensemble_results if r.is_outlier), elapsed,
        )
        return ensemble_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _combine_scores(
        self,
        method_scores: Dict[str, float],
        weights: Dict[str, float],
        ensemble_method: EnsembleMethod,
    ) -> float:
        """Combine per-method scores using the ensemble method.

        Args:
            method_scores: Per-method normalised scores.
            weights: Per-method weights.
            ensemble_method: Combination method.

        Returns:
            Combined score (0.0-1.0).
        """
        if not method_scores:
            return 0.0

        if ensemble_method == EnsembleMethod.MAX_SCORE:
            return max(method_scores.values())

        if ensemble_method == EnsembleMethod.MEAN_SCORE:
            return _safe_mean(list(method_scores.values()))

        if ensemble_method == EnsembleMethod.MAJORITY_VOTE:
            votes = sum(1 for s in method_scores.values() if s >= 0.5)
            return votes / len(method_scores)

        # WEIGHTED_AVERAGE (default)
        total_weight = 0.0
        weighted_sum = 0.0
        for method_name, score in method_scores.items():
            w = weights.get(method_name, 1.0)
            weighted_sum += score * w
            total_weight += w

        if total_weight > 0:
            return min(1.0, weighted_sum / total_weight)
        return 0.0

    def _no_detection_scores(
        self,
        values: List[float],
        method: DetectionMethod,
        column_name: str,
    ) -> List[OutlierScore]:
        """Return zero-score results for insufficient data.

        Args:
            values: Input values.
            method: Detection method.
            column_name: Column name.

        Returns:
            List of OutlierScore with score=0 and is_outlier=False.
        """
        scores = []
        for i, v in enumerate(values):
            provenance_hash = self._provenance.build_hash({
                "method": method.value, "index": i, "value": v,
                "reason": "insufficient_data",
            })
            scores.append(OutlierScore(
                record_index=i,
                column_name=column_name,
                value=v,
                method=method,
                score=0.0,
                is_outlier=False,
                threshold=0.0,
                severity=SeverityLevel.INFO,
                details={"reason": "insufficient_data"},
                confidence=0.0,
                provenance_hash=provenance_hash,
            ))
        return scores

    @staticmethod
    def _approx_t_critical(alpha: float, df: int) -> float:
        """Approximate t-distribution critical value.

        Uses an approximation based on the normal distribution for
        large df, and hardcoded values for small df.

        Args:
            alpha: Significance level (one-tailed).
            df: Degrees of freedom.

        Returns:
            Approximate t-critical value.
        """
        if df <= 0:
            return 3.0
        if alpha <= 0 or alpha >= 0.5:
            return 3.0

        # Normal approximation z for alpha
        # Using Abramowitz and Stegun approximation
        p = alpha
        if p > 0.5:
            p = 1.0 - p
        t_val = math.sqrt(-2.0 * math.log(p))
        # Rational approximation
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t_val - (c0 + c1 * t_val + c2 * t_val ** 2) / (
            1.0 + d1 * t_val + d2 * t_val ** 2 + d3 * t_val ** 3
        )

        # Adjust for df (t is wider than normal for small df)
        if df < 30:
            z *= 1.0 + 1.0 / (4.0 * df)

        return z


__all__ = [
    "StatisticalDetectorEngine",
]
