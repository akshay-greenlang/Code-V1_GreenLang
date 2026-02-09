# -*- coding: utf-8 -*-
"""
Anomaly Detector Engine - AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)

Statistical anomaly detection with multiple methods: IQR (Interquartile
Range), Z-score, Modified Z-score (MAD-based), Grubbs test, and sudden
change/change-point detection. Profiles value distributions and classifies
anomaly severity.

Zero-Hallucination Guarantees:
    - All anomaly detection uses deterministic statistical formulae
    - IQR, Z-score, MAD, Grubbs use standard textbook definitions
    - No ML/LLM calls in the detection path
    - SHA-256 provenance on every detection mutation
    - Thread-safe in-memory storage

Anomaly Methods:
    - IQR: Values outside [Q1 - k*IQR, Q3 + k*IQR] (default k=1.5)
    - Z-score: |z| > threshold (default 3.0)
    - Modified Z-score: 0.6745 * (x - median) / MAD > threshold (default 3.5)
    - Grubbs: Tests single most extreme outlier against critical value
    - Sudden change: Detects change points via sliding window mean shift

Example:
    >>> from greenlang.data_quality_profiler.anomaly_detector import AnomalyDetector
    >>> detector = AnomalyDetector()
    >>> values = [10, 12, 11, 13, 100, 12, 11, 10, 14, 13]
    >>> anomalies = detector.detect_column_anomalies(values, "sensor_a")
    >>> print(len(anomalies))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "AnomalyDetector",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "ANM") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for an anomaly detection operation.

    Args:
        operation: Name of the operation.
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _try_float(value: Any) -> Optional[float]:
    """Attempt to convert a value to float.

    Args:
        value: Value to convert.

    Returns:
        Float or None if conversion fails.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    try:
        f = float(str(value).strip())
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_stdev(values: List[float]) -> float:
    """Compute sample standard deviation, 0.0 for < 2 values.

    Args:
        values: List of numeric values.

    Returns:
        Sample standard deviation or 0.0.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _safe_mean(values: List[float]) -> float:
    """Compute mean, 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean or 0.0.
    """
    if not values:
        return 0.0
    return statistics.mean(values)


def _safe_median(values: List[float]) -> float:
    """Compute median, 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Median or 0.0.
    """
    if not values:
        return 0.0
    return statistics.median(values)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_IQR = "iqr"
METHOD_ZSCORE = "zscore"
METHOD_MAD = "mad"
METHOD_GRUBBS = "grubbs"
METHOD_MODIFIED_ZSCORE = "modified_zscore"
METHOD_ALL = "all"

ALL_METHODS = frozenset({
    METHOD_IQR, METHOD_ZSCORE, METHOD_MAD,
    METHOD_GRUBBS, METHOD_MODIFIED_ZSCORE,
})

SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"

# Default thresholds
_DEFAULT_IQR_MULTIPLIER = 1.5
_DEFAULT_ZSCORE_THRESHOLD = 3.0
_DEFAULT_MAD_THRESHOLD = 3.5
_DEFAULT_CHANGE_WINDOW = 5

# Grubbs critical values (two-sided, alpha=0.05) for small samples
# Approximation: t_crit from t-distribution with (n-2) df
# For production, use scipy. For zero-dependency, use table approximation.
_GRUBBS_CRITICAL_TABLE: Dict[int, float] = {
    3: 1.1543, 4: 1.4812, 5: 1.7150, 6: 1.8871, 7: 2.0200,
    8: 2.1266, 9: 2.2150, 10: 2.2900, 11: 2.3547, 12: 2.4116,
    13: 2.4620, 14: 2.5073, 15: 2.5483, 16: 2.5857, 17: 2.6200,
    18: 2.6516, 19: 2.6809, 20: 2.7082, 25: 2.8217, 30: 2.9085,
    35: 2.9789, 40: 3.0361, 50: 3.1282, 60: 3.1997, 80: 3.3053,
    100: 3.3836,
}


def _grubbs_critical(n: int) -> float:
    """Look up or interpolate Grubbs critical value for sample size n.

    Args:
        n: Sample size.

    Returns:
        Grubbs critical value at alpha=0.05.
    """
    if n in _GRUBBS_CRITICAL_TABLE:
        return _GRUBBS_CRITICAL_TABLE[n]

    # Find bounding keys
    keys = sorted(_GRUBBS_CRITICAL_TABLE.keys())
    if n < keys[0]:
        return _GRUBBS_CRITICAL_TABLE[keys[0]]
    if n > keys[-1]:
        return _GRUBBS_CRITICAL_TABLE[keys[-1]]

    # Linear interpolation
    for i in range(len(keys) - 1):
        if keys[i] <= n <= keys[i + 1]:
            low_n = keys[i]
            high_n = keys[i + 1]
            low_v = _GRUBBS_CRITICAL_TABLE[low_n]
            high_v = _GRUBBS_CRITICAL_TABLE[high_n]
            frac = (n - low_n) / (high_n - low_n)
            return low_v + frac * (high_v - low_v)

    return _GRUBBS_CRITICAL_TABLE[keys[-1]]


# ---------------------------------------------------------------------------
# AnomalyDetector Engine
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Statistical anomaly detection engine with multiple methods.

    Detects outliers and anomalies using IQR, Z-score, Modified Z-score
    (MAD), Grubbs test, and sliding-window change-point detection.
    Profiles value distributions and classifies anomaly severity.

    Thread-safe: all mutations to internal storage are protected by
    a threading lock. SHA-256 provenance hashes on every detection.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for thread-safe storage access.
        _detections: In-memory storage of detection results.
        _stats: Aggregate detection statistics.

    Example:
        >>> detector = AnomalyDetector()
        >>> results = detector.detect_column_anomalies(
        ...     [10, 12, 11, 100, 13, 12], "col_a"
        ... )
        >>> assert len(results) > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AnomalyDetector.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``iqr_multiplier``: float (default 1.5)
                - ``zscore_threshold``: float (default 3.0)
                - ``mad_threshold``: float (default 3.5)
                - ``change_window_size``: int (default 5)
                - ``default_method``: str (default "iqr")
        """
        self._config = config or {}
        self._iqr_k: float = self._config.get("iqr_multiplier", _DEFAULT_IQR_MULTIPLIER)
        self._zscore_t: float = self._config.get("zscore_threshold", _DEFAULT_ZSCORE_THRESHOLD)
        self._mad_t: float = self._config.get("mad_threshold", _DEFAULT_MAD_THRESHOLD)
        self._change_window: int = self._config.get(
            "change_window_size", _DEFAULT_CHANGE_WINDOW
        )
        self._default_method: str = self._config.get("default_method", METHOD_IQR)
        self._lock = threading.Lock()
        self._detections: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Any] = {
            "detections_completed": 0,
            "total_anomalies_found": 0,
            "total_values_scanned": 0,
            "total_detection_time_ms": 0.0,
        }
        logger.info(
            "AnomalyDetector initialized: iqr_k=%.2f, zscore_t=%.2f, "
            "mad_t=%.2f, window=%d, default=%s",
            self._iqr_k, self._zscore_t, self._mad_t,
            self._change_window, self._default_method,
        )

    # ------------------------------------------------------------------
    # Public API - Full Dataset Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect anomalies across a dataset and return results.

        Args:
            data: List of row dictionaries.
            columns: Optional subset of columns to scan. If None,
                all numeric-capable columns are scanned.
            method: Detection method (iqr, zscore, mad, grubbs,
                modified_zscore, all). Defaults to configured method.

        Returns:
            Detection result dict with: detection_id, column_anomalies,
            total_anomalies, anomaly_rate, provenance_hash.

        Raises:
            ValueError: If data is empty.
        """
        start = time.monotonic()
        if not data:
            raise ValueError("Cannot detect anomalies in empty dataset")

        detection_id = _generate_id("ANM")
        detection_method = method or self._default_method
        all_keys = columns if columns else list(data[0].keys())

        column_anomalies: Dict[str, List[Dict[str, Any]]] = {}
        total_anomalies = 0
        total_values = 0

        for col in all_keys:
            raw_values = [row.get(col) for row in data]
            # Only process columns with numeric data
            numeric_vals = [_try_float(v) for v in raw_values]
            has_numeric = any(v is not None for v in numeric_vals)
            if not has_numeric:
                continue

            anomalies = self.detect_column_anomalies(
                raw_values, col, detection_method
            )
            column_anomalies[col] = anomalies
            total_anomalies += len(anomalies)
            total_values += len(raw_values)

        anomaly_rate = total_anomalies / total_values if total_values > 0 else 0.0

        # Issues
        issues = self.generate_anomaly_issues(column_anomalies)

        provenance_data = json.dumps({
            "detection_id": detection_id,
            "method": detection_method,
            "row_count": len(data),
            "total_anomalies": total_anomalies,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("detect", provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        result: Dict[str, Any] = {
            "detection_id": detection_id,
            "method": detection_method,
            "row_count": len(data),
            "columns_scanned": len(column_anomalies),
            "total_values_scanned": total_values,
            "total_anomalies": total_anomalies,
            "anomaly_rate": round(anomaly_rate, 4),
            "column_anomalies": column_anomalies,
            "issues": issues,
            "issue_count": len(issues),
            "provenance_hash": provenance_hash,
            "detection_time_ms": round(elapsed_ms, 2),
            "created_at": _utcnow().isoformat(),
        }

        with self._lock:
            self._detections[detection_id] = result
            self._stats["detections_completed"] += 1
            self._stats["total_anomalies_found"] += total_anomalies
            self._stats["total_values_scanned"] += total_values
            self._stats["total_detection_time_ms"] += elapsed_ms

        logger.info(
            "Anomaly detection: id=%s, method=%s, anomalies=%d/%d, time=%.1fms",
            detection_id, detection_method, total_anomalies,
            total_values, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Per-Column Detection
    # ------------------------------------------------------------------

    def detect_column_anomalies(
        self,
        values: List[Any],
        column_name: str,
        method: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in a single column.

        Args:
            values: List of values for this column.
            column_name: Column name.
            method: Detection method. Defaults to configured default.

        Returns:
            List of anomaly dicts with: index, value, score, method, severity.
        """
        detection_method = method or self._default_method

        # Extract numeric values with their indices
        indexed_nums: List[Tuple[int, float]] = []
        for i, v in enumerate(values):
            f = _try_float(v)
            if f is not None:
                indexed_nums.append((i, f))

        if len(indexed_nums) < 3:
            return []

        nums = [n for _, n in indexed_nums]
        anomalies: List[Dict[str, Any]] = []

        if detection_method == METHOD_ALL:
            # Run all methods and merge
            for m in (METHOD_IQR, METHOD_ZSCORE, METHOD_MAD, METHOD_GRUBBS):
                method_anomalies = self._run_method(m, indexed_nums, nums, column_name)
                anomalies.extend(method_anomalies)
            # Deduplicate by index, keep highest severity
            anomalies = self._deduplicate_anomalies(anomalies)
        else:
            anomalies = self._run_method(
                detection_method, indexed_nums, nums, column_name
            )

        return anomalies

    def _run_method(
        self,
        method: str,
        indexed_nums: List[Tuple[int, float]],
        nums: List[float],
        column_name: str,
    ) -> List[Dict[str, Any]]:
        """Run a specific detection method.

        Args:
            method: Method name.
            indexed_nums: List of (index, value) tuples.
            nums: List of numeric values.
            column_name: Column name.

        Returns:
            List of anomaly dicts.
        """
        if method == METHOD_IQR:
            return self.detect_iqr(indexed_nums, nums, column_name)
        elif method == METHOD_ZSCORE:
            return self.detect_zscore(indexed_nums, nums, column_name)
        elif method == METHOD_MAD:
            return self.detect_mad(indexed_nums, nums, column_name)
        elif method == METHOD_MODIFIED_ZSCORE:
            return self.detect_modified_zscore(indexed_nums, nums, column_name)
        elif method == METHOD_GRUBBS:
            return self.detect_grubbs(indexed_nums, nums, column_name)
        else:
            logger.warning("Unknown method '%s', falling back to IQR", method)
            return self.detect_iqr(indexed_nums, nums, column_name)

    # ------------------------------------------------------------------
    # IQR Method
    # ------------------------------------------------------------------

    def detect_iqr(
        self,
        indexed_nums: List[Tuple[int, float]],
        nums: List[float],
        column_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect outliers using the IQR (Interquartile Range) method.

        Outliers are values outside [Q1 - k*IQR, Q3 + k*IQR].

        Args:
            indexed_nums: List of (original_index, value) tuples.
            nums: List of numeric values (for quartile computation).
            column_name: Column name for reporting.

        Returns:
            List of anomaly dicts with index, value, score, bounds.
        """
        if len(nums) < 4:
            return []

        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = sorted_nums[q1_idx]
        q3 = sorted_nums[q3_idx]
        iqr = q3 - q1

        lower = q1 - self._iqr_k * iqr
        upper = q3 + self._iqr_k * iqr

        anomalies: List[Dict[str, Any]] = []
        for idx, val in indexed_nums:
            if val < lower or val > upper:
                # Score: how far outside the bounds (normalised by IQR)
                if iqr > 0:
                    if val < lower:
                        distance = (lower - val) / iqr
                    else:
                        distance = (val - upper) / iqr
                else:
                    distance = abs(val - q1) if q1 != 0 else abs(val)

                severity = self.compute_anomaly_severity(val, (lower, upper))
                anomalies.append({
                    "index": idx,
                    "value": val,
                    "score": round(distance, 4),
                    "method": METHOD_IQR,
                    "column": column_name,
                    "severity": severity,
                    "bounds": {"lower": round(lower, 4), "upper": round(upper, 4)},
                    "iqr": round(iqr, 4),
                })

        return anomalies

    # ------------------------------------------------------------------
    # Z-Score Method
    # ------------------------------------------------------------------

    def detect_zscore(
        self,
        indexed_nums: List[Tuple[int, float]],
        nums: List[float],
        column_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect outliers using the Z-score method.

        Outliers have |z-score| > threshold.

        Args:
            indexed_nums: List of (original_index, value) tuples.
            nums: Numeric values for mean/std computation.
            column_name: Column name.

        Returns:
            List of anomaly dicts with index, value, z_score.
        """
        if len(nums) < 3:
            return []

        mean_val = statistics.mean(nums)
        std_val = _safe_stdev(nums)
        if std_val == 0.0:
            return []

        anomalies: List[Dict[str, Any]] = []
        for idx, val in indexed_nums:
            z = (val - mean_val) / std_val
            if abs(z) > self._zscore_t:
                expected_lower = mean_val - self._zscore_t * std_val
                expected_upper = mean_val + self._zscore_t * std_val
                severity = self.compute_anomaly_severity(
                    val, (expected_lower, expected_upper)
                )
                anomalies.append({
                    "index": idx,
                    "value": val,
                    "score": round(abs(z), 4),
                    "z_score": round(z, 4),
                    "method": METHOD_ZSCORE,
                    "column": column_name,
                    "severity": severity,
                    "mean": round(mean_val, 4),
                    "stddev": round(std_val, 4),
                })

        return anomalies

    # ------------------------------------------------------------------
    # MAD (Median Absolute Deviation) Method
    # ------------------------------------------------------------------

    def detect_mad(
        self,
        indexed_nums: List[Tuple[int, float]],
        nums: List[float],
        column_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect outliers using Median Absolute Deviation.

        MAD = median(|x_i - median(x)|)
        modified_zscore = 0.6745 * (x - median) / MAD

        Args:
            indexed_nums: List of (original_index, value) tuples.
            nums: Numeric values.
            column_name: Column name.

        Returns:
            List of anomaly dicts.
        """
        if len(nums) < 3:
            return []

        median_val = statistics.median(nums)
        abs_devs = [abs(x - median_val) for x in nums]
        mad = statistics.median(abs_devs)

        if mad == 0.0:
            return []

        anomalies: List[Dict[str, Any]] = []
        for idx, val in indexed_nums:
            modified_z = 0.6745 * (val - median_val) / mad
            if abs(modified_z) > self._mad_t:
                severity = self._classify_anomaly_severity_by_score(abs(modified_z))
                anomalies.append({
                    "index": idx,
                    "value": val,
                    "score": round(abs(modified_z), 4),
                    "modified_zscore": round(modified_z, 4),
                    "method": METHOD_MAD,
                    "column": column_name,
                    "severity": severity,
                    "median": round(median_val, 4),
                    "mad": round(mad, 4),
                })

        return anomalies

    # ------------------------------------------------------------------
    # Modified Z-Score Method (alias/variant)
    # ------------------------------------------------------------------

    def detect_modified_zscore(
        self,
        indexed_nums: List[Tuple[int, float]],
        nums: List[float],
        column_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect outliers using modified z-score (median-based).

        Equivalent to MAD method with the 0.6745 consistency constant.
        Same formula: modified_z = 0.6745 * (x - median) / MAD

        Args:
            indexed_nums: List of (original_index, value) tuples.
            nums: Numeric values.
            column_name: Column name.

        Returns:
            List of anomaly dicts.
        """
        results = self.detect_mad(indexed_nums, nums, column_name)
        for r in results:
            r["method"] = METHOD_MODIFIED_ZSCORE
        return results

    # ------------------------------------------------------------------
    # Grubbs Test
    # ------------------------------------------------------------------

    def detect_grubbs(
        self,
        indexed_nums: List[Tuple[int, float]],
        nums: List[float],
        column_name: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect the single most extreme outlier using the Grubbs test.

        Grubbs statistic: G = max(|x_i - mean|) / std
        Compared to critical value from the Grubbs table.

        Args:
            indexed_nums: List of (original_index, value) tuples.
            nums: Numeric values.
            column_name: Column name.

        Returns:
            List with 0 or 1 anomaly dicts (Grubbs tests one at a time).
        """
        n = len(nums)
        if n < 3:
            return []

        mean_val = statistics.mean(nums)
        std_val = _safe_stdev(nums)
        if std_val == 0.0:
            return []

        # Find the most extreme value
        max_dev = 0.0
        max_idx = -1
        max_val = 0.0
        max_original_idx = 0

        for i, (orig_idx, val) in enumerate(indexed_nums):
            dev = abs(val - mean_val)
            if dev > max_dev:
                max_dev = dev
                max_idx = i
                max_val = val
                max_original_idx = orig_idx

        grubbs_stat = max_dev / std_val
        critical = _grubbs_critical(n)

        if grubbs_stat > critical:
            severity = self._classify_anomaly_severity_by_score(grubbs_stat)
            return [{
                "index": max_original_idx,
                "value": max_val,
                "score": round(grubbs_stat, 4),
                "grubbs_statistic": round(grubbs_stat, 4),
                "critical_value": round(critical, 4),
                "method": METHOD_GRUBBS,
                "column": column_name,
                "severity": severity,
                "mean": round(mean_val, 4),
                "stddev": round(std_val, 4),
                "sample_size": n,
            }]

        return []

    # ------------------------------------------------------------------
    # Distribution Profiling
    # ------------------------------------------------------------------

    def profile_distribution(self, values: List[Any]) -> Dict[str, Any]:
        """Profile the distribution of a list of values.

        Args:
            values: List of values (numeric conversion attempted).

        Returns:
            Dict with mean, median, std, skewness, kurtosis,
            normality_estimate, min, max, range, count.
        """
        nums: List[float] = []
        for v in values:
            f = _try_float(v)
            if f is not None:
                nums.append(f)

        if not nums:
            return {"count": 0, "numeric_count": 0}

        n = len(nums)
        mean_val = statistics.mean(nums)
        median_val = statistics.median(nums)
        std_val = _safe_stdev(nums)

        # Skewness
        skew = 0.0
        if n >= 3 and std_val > 0:
            m3 = sum(((x - mean_val) / std_val) ** 3 for x in nums)
            skew = (n / ((n - 1) * (n - 2))) * m3

        # Kurtosis (excess)
        kurt = 0.0
        if n >= 4 and std_val > 0:
            m4 = sum(((x - mean_val) / std_val) ** 4 for x in nums)
            term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
            term2 = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
            kurt = term1 * m4 - term2

        # Normality estimate: Jarque-Bera approximation
        # JB = (n/6) * (S^2 + (K^2)/4)
        # Small JB -> more normal
        jb = (n / 6.0) * (skew ** 2 + (kurt ** 2) / 4.0) if n >= 4 else 0.0
        # Convert to normality score (1.0 = perfectly normal)
        normality = max(0.0, 1.0 - min(jb / 100.0, 1.0))

        return {
            "count": len(values),
            "numeric_count": n,
            "mean": round(mean_val, 6),
            "median": round(median_val, 6),
            "std": round(std_val, 6),
            "min": round(min(nums), 6),
            "max": round(max(nums), 6),
            "range": round(max(nums) - min(nums), 6),
            "skewness": round(skew, 6),
            "kurtosis": round(kurt, 6),
            "normality_estimate": round(normality, 4),
            "jarque_bera": round(jb, 4),
        }

    # ------------------------------------------------------------------
    # Sudden Change / Change Point Detection
    # ------------------------------------------------------------------

    def detect_sudden_change(
        self,
        values: List[Any],
        window_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect sudden change points using sliding window mean shift.

        Compares the mean of a window before and after each point.
        A change is flagged when the ratio exceeds a threshold.

        Args:
            values: Ordered list of values (time series).
            window_size: Window size for comparison. Defaults to configured.

        Returns:
            List of change point dicts with: index, before_mean,
            after_mean, change_ratio, severity.
        """
        window = window_size if window_size is not None else self._change_window

        nums: List[float] = []
        for v in values:
            f = _try_float(v)
            nums.append(f if f is not None else 0.0)

        if len(nums) < 2 * window + 1:
            return []

        change_points: List[Dict[str, Any]] = []

        for i in range(window, len(nums) - window):
            before = nums[i - window:i]
            after = nums[i:i + window]

            mean_before = statistics.mean(before)
            mean_after = statistics.mean(after)

            # Change ratio
            baseline = max(abs(mean_before), 1e-10)
            change_ratio = abs(mean_after - mean_before) / baseline

            # Also check absolute std deviation of the entire local window
            local = nums[max(0, i - window):min(len(nums), i + window)]
            local_std = _safe_stdev(local)
            if local_std > 0:
                normalised_change = abs(mean_after - mean_before) / local_std
            else:
                normalised_change = change_ratio

            # Flag if change exceeds 2 standard deviations or 50% ratio
            if normalised_change > 2.0 or change_ratio > 0.5:
                severity = self._classify_anomaly_severity_by_score(normalised_change)
                change_points.append({
                    "index": i,
                    "value": nums[i],
                    "before_mean": round(mean_before, 4),
                    "after_mean": round(mean_after, 4),
                    "change_ratio": round(change_ratio, 4),
                    "normalised_change": round(normalised_change, 4),
                    "severity": severity,
                    "window_size": window,
                })

        return change_points

    # ------------------------------------------------------------------
    # Anomaly Severity
    # ------------------------------------------------------------------

    def compute_anomaly_severity(
        self,
        value: float,
        expected_range: Tuple[float, float],
    ) -> str:
        """Compute anomaly severity based on distance from expected range.

        Args:
            value: The anomalous value.
            expected_range: Tuple of (lower_bound, upper_bound).

        Returns:
            Severity string.
        """
        lower, upper = expected_range
        range_size = upper - lower if upper > lower else 1.0

        if value < lower:
            distance = (lower - value) / range_size
        elif value > upper:
            distance = (value - upper) / range_size
        else:
            return SEVERITY_INFO

        return self._classify_anomaly_severity_by_score(distance)

    def _classify_anomaly_severity_by_score(self, score: float) -> str:
        """Classify anomaly severity by a normalised score.

        Args:
            score: Normalised anomaly score (higher = more extreme).

        Returns:
            Severity string.
        """
        if score >= 5.0:
            return SEVERITY_CRITICAL
        if score >= 3.0:
            return SEVERITY_HIGH
        if score >= 2.0:
            return SEVERITY_MEDIUM
        if score > 0.0:
            return SEVERITY_LOW
        return SEVERITY_INFO

    # ------------------------------------------------------------------
    # Issue Generation
    # ------------------------------------------------------------------

    def generate_anomaly_issues(
        self,
        column_anomalies: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Generate anomaly quality issues from detection results.

        Args:
            column_anomalies: Dict mapping column -> list of anomaly dicts.

        Returns:
            List of issue dicts.
        """
        issues: List[Dict[str, Any]] = []

        for col_name, anomalies in column_anomalies.items():
            if not anomalies:
                continue

            # Count by severity
            severity_counts: Dict[str, int] = {}
            for a in anomalies:
                sev = a.get("severity", SEVERITY_LOW)
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            # Overall severity for the column
            if severity_counts.get(SEVERITY_CRITICAL, 0) > 0:
                col_severity = SEVERITY_CRITICAL
            elif severity_counts.get(SEVERITY_HIGH, 0) > 0:
                col_severity = SEVERITY_HIGH
            elif severity_counts.get(SEVERITY_MEDIUM, 0) > 0:
                col_severity = SEVERITY_MEDIUM
            else:
                col_severity = SEVERITY_LOW

            issues.append({
                "issue_id": _generate_id("ISS"),
                "type": "anomalies_detected",
                "severity": col_severity,
                "column": col_name,
                "message": (
                    f"Column '{col_name}' has {len(anomalies)} anomalies detected"
                ),
                "details": {
                    "anomaly_count": len(anomalies),
                    "severity_breakdown": severity_counts,
                    "methods_used": list(set(a.get("method", "") for a in anomalies)),
                    "sample_values": [a.get("value") for a in anomalies[:5]],
                },
                "created_at": _utcnow().isoformat(),
            })

        return issues

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deduplicate anomalies by index, keeping highest severity.

        Args:
            anomalies: List of anomaly dicts.

        Returns:
            Deduplicated list.
        """
        severity_order = {
            SEVERITY_CRITICAL: 4, SEVERITY_HIGH: 3,
            SEVERITY_MEDIUM: 2, SEVERITY_LOW: 1, SEVERITY_INFO: 0,
        }

        best: Dict[int, Dict[str, Any]] = {}
        for a in anomalies:
            idx = a.get("index", -1)
            if idx not in best:
                best[idx] = a
            else:
                current_sev = severity_order.get(
                    best[idx].get("severity", SEVERITY_INFO), 0
                )
                new_sev = severity_order.get(
                    a.get("severity", SEVERITY_INFO), 0
                )
                if new_sev > current_sev:
                    # Merge methods
                    methods = set()
                    m1 = best[idx].get("method", "")
                    m2 = a.get("method", "")
                    if m1:
                        methods.add(m1)
                    if m2:
                        methods.add(m2)
                    a["methods_detected"] = sorted(methods)
                    best[idx] = a
                else:
                    methods = set()
                    m1 = best[idx].get("method", "")
                    m2 = a.get("method", "")
                    if m1:
                        methods.add(m1)
                    if m2:
                        methods.add(m2)
                    best[idx]["methods_detected"] = sorted(methods)

        return sorted(best.values(), key=lambda x: x.get("index", 0))

    # ------------------------------------------------------------------
    # Storage and Retrieval
    # ------------------------------------------------------------------

    def get_detection(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored detection by ID.

        Args:
            detection_id: The detection identifier.

        Returns:
            Detection dict or None if not found.
        """
        with self._lock:
            return self._detections.get(detection_id)

    def list_detections(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored detections with pagination.

        Args:
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of detection dicts sorted by creation time descending.
        """
        with self._lock:
            all_detections = sorted(
                self._detections.values(),
                key=lambda d: d.get("created_at", ""),
                reverse=True,
            )
            return all_detections[offset:offset + limit]

    def delete_detection(self, detection_id: str) -> bool:
        """Delete a stored detection.

        Args:
            detection_id: The detection identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if detection_id in self._detections:
                del self._detections[detection_id]
                logger.info("Anomaly detection deleted: %s", detection_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate detection statistics.

        Returns:
            Dictionary with counters and totals for all anomaly
            detections performed by this engine instance.
        """
        with self._lock:
            completed = self._stats["detections_completed"]
            avg_time = (
                self._stats["total_detection_time_ms"] / completed
                if completed > 0 else 0.0
            )
            avg_anomalies = (
                self._stats["total_anomalies_found"] / completed
                if completed > 0 else 0.0
            )
            anomaly_rate = (
                self._stats["total_anomalies_found"] /
                self._stats["total_values_scanned"]
                if self._stats["total_values_scanned"] > 0 else 0.0
            )
            return {
                "detections_completed": completed,
                "total_anomalies_found": self._stats["total_anomalies_found"],
                "total_values_scanned": self._stats["total_values_scanned"],
                "overall_anomaly_rate": round(anomaly_rate, 6),
                "avg_anomalies_per_detection": round(avg_anomalies, 2),
                "total_detection_time_ms": round(
                    self._stats["total_detection_time_ms"], 2
                ),
                "avg_detection_time_ms": round(avg_time, 2),
                "stored_detections": len(self._detections),
                "timestamp": _utcnow().isoformat(),
            }
