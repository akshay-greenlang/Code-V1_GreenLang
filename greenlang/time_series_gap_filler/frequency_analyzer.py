# -*- coding: utf-8 -*-
"""
Frequency Analyzer Engine - AGENT-DATA-014

Pure-Python frequency analysis for time series data. Determines the
dominant observation frequency (sub-hourly through annual), detects
regularity and mixed-frequency patterns, computes interval statistics,
validates timestamps against expected frequencies, and estimates
dominant periodicity via simple autocorrelation.

Engine 2 of 7 in the Time Series Gap Filler pipeline:
    1. Gap Detector          - finds gaps in time series
    2. Frequency Analyzer    - determines observation frequency (THIS)
    3. Interpolation Engine  - fills gaps via interpolation
    4. Seasonal Decomposer   - seasonal-aware gap filling
    5. Cross Series Filler   - correlation-based gap filling
    6. Calendar Engine        - calendar-aware adjustments
    7. Strategy Selector     - picks best fill method per gap

Zero-Hallucination: All calculations use deterministic Python
arithmetic (datetime, math, statistics). No LLM calls for numeric
computations. No external libraries required.

Example:
    >>> from greenlang.time_series_gap_filler.frequency_analyzer import FrequencyAnalyzerEngine
    >>> engine = FrequencyAnalyzerEngine()
    >>> from datetime import datetime, timedelta
    >>> ts = [datetime(2025, 1, d) for d in range(1, 32)]
    >>> result = engine.analyze_frequency(ts)
    >>> assert result["frequency_level"] == "daily"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.time_series_gap_filler.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graceful imports for sibling modules (built in parallel)
# ---------------------------------------------------------------------------

try:
    from greenlang.time_series_gap_filler.models import (
        FrequencyLevel as _ModelFrequencyLevel,
    )
except ImportError:
    _ModelFrequencyLevel = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.provenance import (
        ProvenanceTracker,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.metrics import (
        observe_duration as _observe_duration,
        inc_errors as _inc_errors,
    )
except ImportError:
    _observe_duration = None  # type: ignore[assignment, misc]
    _inc_errors = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Frequency Level Enumeration
# ---------------------------------------------------------------------------


class FrequencyLevel(str, Enum):
    """Canonical time series observation frequency levels.

    SUB_HOURLY: Observations more frequent than once per hour (<3600s).
    HOURLY: Approximately once per hour (~3600s).
    DAILY: Approximately once per day (~86400s).
    WEEKLY: Approximately once per week (~604800s).
    BIWEEKLY: Approximately once per two weeks (~1209600s).
    MONTHLY: Approximately once per month (~2592000s / 30d).
    QUARTERLY: Approximately once per quarter (~7776000s / 90d).
    SEMI_ANNUAL: Approximately once per half year (~15552000s / 180d).
    ANNUAL: Approximately once per year (~31536000s / 365d).
    IRREGULAR: No consistent frequency detected.
    """

    SUB_HOURLY = "sub_hourly"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    IRREGULAR = "irregular"


# ---------------------------------------------------------------------------
# Frequency reference intervals (seconds)
# ---------------------------------------------------------------------------

#: Mapping of FrequencyLevel to canonical interval in seconds.
FREQUENCY_INTERVALS: Dict[FrequencyLevel, float] = {
    FrequencyLevel.SUB_HOURLY: 900.0,        # 15 minutes representative
    FrequencyLevel.HOURLY: 3600.0,            # 1 hour
    FrequencyLevel.DAILY: 86400.0,            # 24 hours
    FrequencyLevel.WEEKLY: 604800.0,          # 7 days
    FrequencyLevel.BIWEEKLY: 1209600.0,       # 14 days
    FrequencyLevel.MONTHLY: 2592000.0,        # 30 days
    FrequencyLevel.QUARTERLY: 7776000.0,      # 90 days
    FrequencyLevel.SEMI_ANNUAL: 15552000.0,   # 180 days
    FrequencyLevel.ANNUAL: 31536000.0,        # 365 days
}

#: Tolerance ranges (min_seconds, max_seconds) for each frequency level.
FREQUENCY_TOLERANCES: Dict[FrequencyLevel, Tuple[float, float]] = {
    FrequencyLevel.SUB_HOURLY: (0.0, 3000.0),
    FrequencyLevel.HOURLY: (3000.0, 7200.0),
    FrequencyLevel.DAILY: (43200.0, 172800.0),
    FrequencyLevel.WEEKLY: (432000.0, 864000.0),
    FrequencyLevel.BIWEEKLY: (864000.0, 1814400.0),
    FrequencyLevel.MONTHLY: (1814400.0, 5184000.0),
    FrequencyLevel.QUARTERLY: (5184000.0, 11664000.0),
    FrequencyLevel.SEMI_ANNUAL: (11664000.0, 23328000.0),
    FrequencyLevel.ANNUAL: (23328000.0, 63072000.0),
}


# ---------------------------------------------------------------------------
# Pure-Python math helpers
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


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _build_hash(data: Any) -> str:
    """Build a SHA-256 hash for arbitrary data.

    Args:
        data: Data to hash (dict, list, or other serialisable).

    Returns:
        Hex-encoded SHA-256 hash.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# FrequencyAnalyzerEngine
# ---------------------------------------------------------------------------


class FrequencyAnalyzerEngine:
    """Pure-Python frequency analysis engine for time series data.

    Analyses timestamp sequences to determine observation frequency,
    regularity, mixed-frequency patterns, dominant periodicity, and
    interval statistics. Supports both datetime objects and numeric
    epoch timestamps as input.

    Attributes:
        _config: Time Series Gap Filler configuration.
        _provenance: SHA-256 provenance tracker instance (or None).

    Example:
        >>> engine = FrequencyAnalyzerEngine()
        >>> from datetime import datetime
        >>> ts = [datetime(2025, 1, d) for d in range(1, 32)]
        >>> result = engine.analyze_frequency(ts)
        >>> assert result["frequency_level"] == "daily"
        >>> assert 0.0 <= result["regularity_score"] <= 1.0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize FrequencyAnalyzerEngine.

        Args:
            config: Optional TimeSeriesGapFillerConfig override.
                Falls back to singleton from ``get_config()``.
        """
        self._config = config or get_config()
        self._provenance = (
            ProvenanceTracker() if ProvenanceTracker is not None else None
        )
        logger.info("FrequencyAnalyzerEngine initialized")

    # ------------------------------------------------------------------
    # Public API - analyze_frequency
    # ------------------------------------------------------------------

    def analyze_frequency(
        self,
        timestamps: List[Union[datetime, float, int]],
    ) -> Dict[str, Any]:
        """Analyse observation frequency of a timestamp sequence.

        Computes all pairwise deltas between consecutive timestamps,
        finds the dominant interval (most common delta), maps it to
        a FrequencyLevel enum, and computes regularity and confidence
        scores.

        Args:
            timestamps: Ordered list of observation timestamps.
                Accepts datetime objects or numeric epoch seconds.

        Returns:
            Dictionary with keys:
                - frequency_level (str): Detected FrequencyLevel value.
                - dominant_interval_seconds (float): Most common interval.
                - regularity_score (float): 0-1 fraction matching dominant.
                - confidence (float): Confidence in the detection.
                - sample_size (int): Number of intervals analysed.
                - total_timestamps (int): Input timestamp count.
                - interval_statistics (dict): Full interval stats.
                - provenance_hash (str): SHA-256 audit trail hash.

        Raises:
            ValueError: If fewer timestamps than config.min_points.
        """
        start = time.time()
        min_points = self._config.min_data_points

        if len(timestamps) < 2:
            raise ValueError(
                f"At least 2 timestamps required, got {len(timestamps)}"
            )

        epochs = self._to_epochs(timestamps)
        deltas = self._compute_deltas(epochs)

        if not deltas:
            raise ValueError("No valid intervals computed from timestamps")

        # Find dominant interval
        dominant_seconds = self._find_dominant_interval(deltas)

        # Classify frequency
        frequency_level = self.classify_interval(dominant_seconds)

        # Compute regularity
        regularity_score = self._compute_regularity_score(
            deltas, dominant_seconds,
        )

        # Compute confidence based on sample size and regularity
        confidence = self._compute_confidence(
            sample_size=len(deltas),
            regularity=regularity_score,
            min_points=min_points,
        )

        # Interval statistics
        interval_stats = self.compute_interval_statistics(timestamps)

        # Provenance
        provenance_hash = self._record_provenance(
            operation="analyze_frequency",
            input_data={
                "timestamp_count": len(timestamps),
                "first_epoch": epochs[0] if epochs else 0,
                "last_epoch": epochs[-1] if epochs else 0,
            },
            output_data={
                "frequency_level": frequency_level.value,
                "dominant_interval_seconds": dominant_seconds,
                "regularity_score": regularity_score,
                "confidence": confidence,
            },
        )

        elapsed = time.time() - start
        self._observe_metric("analyze_frequency", elapsed)

        result: Dict[str, Any] = {
            "frequency_level": frequency_level.value,
            "dominant_interval_seconds": dominant_seconds,
            "regularity_score": round(regularity_score, 6),
            "confidence": round(confidence, 6),
            "sample_size": len(deltas),
            "total_timestamps": len(timestamps),
            "interval_statistics": interval_stats,
            "provenance_hash": provenance_hash,
        }

        logger.debug(
            "Frequency analysis: level=%s, interval=%.1fs, "
            "regularity=%.4f, confidence=%.4f, n=%d, %.3fs",
            frequency_level.value,
            dominant_seconds,
            regularity_score,
            confidence,
            len(deltas),
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Public API - detect_regularity
    # ------------------------------------------------------------------

    def detect_regularity(
        self,
        timestamps: List[Union[datetime, float, int]],
    ) -> float:
        """Detect regularity of inter-observation intervals.

        Uses the coefficient of variation (CV) of the intervals. A
        perfectly regular series has CV=0, yielding regularity=1.0.
        Completely irregular series approach regularity=0.0.

        Regularity = max(0, 1 - CV) where CV = std / mean.

        Args:
            timestamps: Ordered list of observation timestamps.

        Returns:
            Regularity score between 0.0 (irregular) and 1.0 (regular).
        """
        start = time.time()

        if len(timestamps) < 2:
            return 0.0

        epochs = self._to_epochs(timestamps)
        deltas = self._compute_deltas(epochs)

        if not deltas:
            return 0.0

        mean_delta = _safe_mean(deltas)
        std_delta = _safe_std(deltas, mean_delta)

        if mean_delta <= 0.0:
            return 0.0

        cv = std_delta / mean_delta
        regularity = max(0.0, min(1.0, 1.0 - cv))

        elapsed = time.time() - start
        self._observe_metric("detect_regularity", elapsed)

        logger.debug(
            "Regularity detection: cv=%.4f, regularity=%.4f, n=%d",
            cv, regularity, len(deltas),
        )
        return round(regularity, 6)

    # ------------------------------------------------------------------
    # Public API - detect_mixed_frequency
    # ------------------------------------------------------------------

    def detect_mixed_frequency(
        self,
        timestamps: List[Union[datetime, float, int]],
    ) -> Dict[str, Any]:
        """Detect mixed frequency patterns in a timestamp sequence.

        Classifies each inter-observation interval into a frequency
        level and reports the proportion of intervals belonging to
        each detected frequency.

        Args:
            timestamps: Ordered list of observation timestamps.

        Returns:
            Dictionary with keys:
                - is_mixed (bool): True if more than one frequency found.
                - frequencies_found (list[str]): Distinct frequency levels.
                - proportions (dict[str, float]): Level to proportion map.
                - dominant_frequency (str): Level with highest proportion.
                - total_intervals (int): Number of intervals analysed.
                - provenance_hash (str): SHA-256 audit trail hash.
        """
        start = time.time()

        if len(timestamps) < 2:
            return {
                "is_mixed": False,
                "frequencies_found": [],
                "proportions": {},
                "dominant_frequency": FrequencyLevel.IRREGULAR.value,
                "total_intervals": 0,
                "provenance_hash": _build_hash({"empty": True}),
            }

        epochs = self._to_epochs(timestamps)
        deltas = self._compute_deltas(epochs)

        if not deltas:
            return {
                "is_mixed": False,
                "frequencies_found": [],
                "proportions": {},
                "dominant_frequency": FrequencyLevel.IRREGULAR.value,
                "total_intervals": 0,
                "provenance_hash": _build_hash({"empty": True}),
            }

        # Classify each interval
        level_counts: Dict[str, int] = {}
        for delta in deltas:
            level = self.classify_interval(delta)
            level_val = level.value
            level_counts[level_val] = level_counts.get(level_val, 0) + 1

        total = len(deltas)
        proportions: Dict[str, float] = {
            level: round(count / total, 6)
            for level, count in sorted(
                level_counts.items(), key=lambda x: -x[1],
            )
        }

        frequencies_found = list(proportions.keys())
        dominant_frequency = frequencies_found[0] if frequencies_found else (
            FrequencyLevel.IRREGULAR.value
        )
        is_mixed = len(frequencies_found) > 1

        provenance_hash = self._record_provenance(
            operation="detect_mixed_frequency",
            input_data={"timestamp_count": len(timestamps)},
            output_data={
                "is_mixed": is_mixed,
                "frequencies_found": frequencies_found,
                "proportions": proportions,
            },
        )

        elapsed = time.time() - start
        self._observe_metric("detect_mixed_frequency", elapsed)

        logger.debug(
            "Mixed frequency detection: mixed=%s, found=%s, n=%d, %.3fs",
            is_mixed, frequencies_found, total, elapsed,
        )

        return {
            "is_mixed": is_mixed,
            "frequencies_found": frequencies_found,
            "proportions": proportions,
            "dominant_frequency": dominant_frequency,
            "total_intervals": total,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Public API - get_dominant_period
    # ------------------------------------------------------------------

    def get_dominant_period(
        self,
        values: List[float],
    ) -> int:
        """Detect the dominant period via simple autocorrelation.

        Computes the autocorrelation function for lags 1 through
        half the series length and returns the lag with the highest
        autocorrelation. This indicates the dominant periodic
        component in the value series.

        Args:
            values: Numeric value series (not timestamps).

        Returns:
            Dominant period length (in number of observations).
            Returns 0 if the series is too short or has no periodicity.
        """
        start = time.time()
        n = len(values)

        if n < 4:
            logger.debug("Series too short for period detection: n=%d", n)
            return 0

        mean = _safe_mean(values)
        # Denominator: variance * n
        denom = sum((v - mean) ** 2 for v in values)
        if denom <= 0.0:
            return 0

        max_lag = n // 2
        best_lag = 0
        best_acf = -1.0

        for lag in range(1, max_lag + 1):
            numerator = sum(
                (values[i] - mean) * (values[i + lag] - mean)
                for i in range(n - lag)
            )
            acf = numerator / denom
            if acf > best_acf:
                best_acf = acf
                best_lag = lag

        elapsed = time.time() - start
        self._observe_metric("get_dominant_period", elapsed)

        # Only return a period if autocorrelation is meaningfully positive
        if best_acf < 0.1:
            logger.debug(
                "No significant periodicity found: best_acf=%.4f at lag=%d",
                best_acf, best_lag,
            )
            return 0

        logger.debug(
            "Dominant period detected: lag=%d, acf=%.4f, n=%d, %.3fs",
            best_lag, best_acf, n, elapsed,
        )
        return best_lag

    # ------------------------------------------------------------------
    # Public API - classify_interval
    # ------------------------------------------------------------------

    def classify_interval(self, seconds: float) -> FrequencyLevel:
        """Map a time interval in seconds to the nearest FrequencyLevel.

        Uses the FREQUENCY_TOLERANCES ranges to find the best-matching
        level. If no tolerance range matches, picks the level with the
        smallest relative distance to the canonical interval.

        Args:
            seconds: Time interval in seconds.

        Returns:
            Closest matching FrequencyLevel.
        """
        if seconds <= 0.0:
            return FrequencyLevel.IRREGULAR

        # First try exact tolerance ranges
        for level, (lo, hi) in FREQUENCY_TOLERANCES.items():
            if lo <= seconds <= hi:
                return level

        # Fallback: pick level with smallest relative distance
        best_level = FrequencyLevel.IRREGULAR
        best_distance = float("inf")

        for level, canonical in FREQUENCY_INTERVALS.items():
            if canonical <= 0.0:
                continue
            # Relative distance from canonical
            distance = abs(seconds - canonical) / canonical
            if distance < best_distance:
                best_distance = distance
                best_level = level

        return best_level

    # ------------------------------------------------------------------
    # Public API - compute_interval_statistics
    # ------------------------------------------------------------------

    def compute_interval_statistics(
        self,
        timestamps: List[Union[datetime, float, int]],
    ) -> Dict[str, Any]:
        """Compute descriptive statistics on inter-observation intervals.

        Args:
            timestamps: Ordered list of observation timestamps.

        Returns:
            Dictionary with keys:
                - mean_interval (float): Mean interval in seconds.
                - median_interval (float): Median interval in seconds.
                - std_interval (float): Std deviation of intervals.
                - min_interval (float): Shortest interval in seconds.
                - max_interval (float): Longest interval in seconds.
                - cv (float): Coefficient of variation (std/mean).
                - count (int): Number of intervals.
                - provenance_hash (str): SHA-256 audit trail hash.
        """
        start = time.time()

        if len(timestamps) < 2:
            empty_result: Dict[str, Any] = {
                "mean_interval": 0.0,
                "median_interval": 0.0,
                "std_interval": 0.0,
                "min_interval": 0.0,
                "max_interval": 0.0,
                "cv": 0.0,
                "count": 0,
                "provenance_hash": _build_hash({"empty": True}),
            }
            return empty_result

        epochs = self._to_epochs(timestamps)
        deltas = self._compute_deltas(epochs)

        if not deltas:
            return {
                "mean_interval": 0.0,
                "median_interval": 0.0,
                "std_interval": 0.0,
                "min_interval": 0.0,
                "max_interval": 0.0,
                "cv": 0.0,
                "count": 0,
                "provenance_hash": _build_hash({"no_deltas": True}),
            }

        mean_val = _safe_mean(deltas)
        std_val = _safe_std(deltas, mean_val)
        median_val = _safe_median(deltas)
        min_val = min(deltas)
        max_val = max(deltas)
        cv = std_val / mean_val if mean_val > 0.0 else 0.0

        provenance_hash = self._record_provenance(
            operation="compute_interval_statistics",
            input_data={"timestamp_count": len(timestamps)},
            output_data={
                "mean": mean_val,
                "std": std_val,
                "median": median_val,
                "min": min_val,
                "max": max_val,
                "cv": cv,
            },
        )

        elapsed = time.time() - start
        self._observe_metric("compute_interval_statistics", elapsed)

        result: Dict[str, Any] = {
            "mean_interval": round(mean_val, 6),
            "median_interval": round(median_val, 6),
            "std_interval": round(std_val, 6),
            "min_interval": round(min_val, 6),
            "max_interval": round(max_val, 6),
            "cv": round(cv, 6),
            "count": len(deltas),
            "provenance_hash": provenance_hash,
        }

        logger.debug(
            "Interval statistics: mean=%.1f, median=%.1f, std=%.1f, "
            "cv=%.4f, n=%d",
            mean_val, median_val, std_val, cv, len(deltas),
        )
        return result

    # ------------------------------------------------------------------
    # Public API - validate_frequency
    # ------------------------------------------------------------------

    def validate_frequency(
        self,
        timestamps: List[Union[datetime, float, int]],
        expected_frequency: Union[str, FrequencyLevel],
    ) -> Dict[str, Any]:
        """Validate that timestamps match an expected frequency.

        Checks each inter-observation interval against the canonical
        interval for the expected frequency and computes the match
        percentage and deviation list.

        Args:
            timestamps: Ordered list of observation timestamps.
            expected_frequency: Expected FrequencyLevel (str or enum).

        Returns:
            Dictionary with keys:
                - is_valid (bool): True if match_pct >= 80%.
                - match_pct (float): Fraction of intervals matching.
                - expected_frequency (str): The expected level name.
                - expected_interval_seconds (float): Canonical interval.
                - deviations (list[dict]): Intervals that deviate.
                    Each dict: index, actual_seconds, expected_seconds,
                    deviation_pct.
                - total_intervals (int): Number of intervals checked.
                - provenance_hash (str): SHA-256 audit trail hash.
        """
        start = time.time()

        # Resolve expected frequency
        if isinstance(expected_frequency, str):
            try:
                freq_level = FrequencyLevel(expected_frequency)
            except ValueError:
                raise ValueError(
                    f"Unknown frequency level: {expected_frequency!r}. "
                    f"Valid levels: {[f.value for f in FrequencyLevel]}"
                )
        else:
            freq_level = expected_frequency

        canonical = FREQUENCY_INTERVALS.get(freq_level, 0.0)
        if canonical <= 0.0:
            return {
                "is_valid": False,
                "match_pct": 0.0,
                "expected_frequency": freq_level.value,
                "expected_interval_seconds": 0.0,
                "deviations": [],
                "total_intervals": 0,
                "provenance_hash": _build_hash({"no_canonical": True}),
            }

        if len(timestamps) < 2:
            return {
                "is_valid": False,
                "match_pct": 0.0,
                "expected_frequency": freq_level.value,
                "expected_interval_seconds": canonical,
                "deviations": [],
                "total_intervals": 0,
                "provenance_hash": _build_hash({"insufficient": True}),
            }

        epochs = self._to_epochs(timestamps)
        deltas = self._compute_deltas(epochs)

        if not deltas:
            return {
                "is_valid": False,
                "match_pct": 0.0,
                "expected_frequency": freq_level.value,
                "expected_interval_seconds": canonical,
                "deviations": [],
                "total_intervals": 0,
                "provenance_hash": _build_hash({"no_deltas": True}),
            }

        # Determine tolerance: 25% of canonical interval
        tolerance = canonical * 0.25

        matches = 0
        deviations: List[Dict[str, Any]] = []

        for i, delta in enumerate(deltas):
            if abs(delta - canonical) <= tolerance:
                matches += 1
            else:
                deviation_pct = (
                    abs(delta - canonical) / canonical * 100.0
                    if canonical > 0.0 else 0.0
                )
                deviations.append({
                    "index": i,
                    "actual_seconds": round(delta, 2),
                    "expected_seconds": round(canonical, 2),
                    "deviation_pct": round(deviation_pct, 2),
                })

        total = len(deltas)
        match_pct = matches / total if total > 0 else 0.0
        is_valid = match_pct >= 0.80

        provenance_hash = self._record_provenance(
            operation="validate_frequency",
            input_data={
                "timestamp_count": len(timestamps),
                "expected": freq_level.value,
            },
            output_data={
                "is_valid": is_valid,
                "match_pct": match_pct,
                "deviation_count": len(deviations),
            },
        )

        elapsed = time.time() - start
        self._observe_metric("validate_frequency", elapsed)

        logger.debug(
            "Frequency validation: expected=%s, match=%.2f%%, "
            "valid=%s, deviations=%d, n=%d, %.3fs",
            freq_level.value,
            match_pct * 100.0,
            is_valid,
            len(deviations),
            total,
            elapsed,
        )

        return {
            "is_valid": is_valid,
            "match_pct": round(match_pct, 6),
            "expected_frequency": freq_level.value,
            "expected_interval_seconds": canonical,
            "deviations": deviations,
            "total_intervals": total,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Private helpers - timestamp conversion
    # ------------------------------------------------------------------

    def _timestamp_to_epoch(
        self,
        ts: Union[datetime, float, int],
    ) -> float:
        """Convert a timestamp to epoch seconds.

        Supports datetime objects (timezone-aware or naive) and numeric
        epoch values (float or int). Naive datetimes are assumed UTC.

        Args:
            ts: Timestamp as datetime, float, or int.

        Returns:
            Epoch seconds as float.

        Raises:
            TypeError: If the timestamp type is unsupported.
        """
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                # Treat naive datetimes as UTC
                return ts.replace(tzinfo=timezone.utc).timestamp()
            return ts.timestamp()
        if isinstance(ts, (int, float)):
            return float(ts)
        raise TypeError(
            f"Unsupported timestamp type: {type(ts).__name__}. "
            "Expected datetime, float, or int."
        )

    def _to_epochs(
        self,
        timestamps: List[Union[datetime, float, int]],
    ) -> List[float]:
        """Convert a list of timestamps to sorted epoch seconds.

        Args:
            timestamps: List of timestamps to convert.

        Returns:
            Sorted list of epoch seconds.
        """
        epochs = [self._timestamp_to_epoch(ts) for ts in timestamps]
        epochs.sort()
        return epochs

    # ------------------------------------------------------------------
    # Private helpers - interval computation
    # ------------------------------------------------------------------

    def _compute_deltas(self, epochs: List[float]) -> List[float]:
        """Compute consecutive pairwise deltas between sorted epochs.

        Filters out zero or negative deltas (duplicates or unsorted
        remnants).

        Args:
            epochs: Sorted list of epoch seconds.

        Returns:
            List of positive pairwise deltas in seconds.
        """
        deltas: List[float] = []
        for i in range(1, len(epochs)):
            delta = epochs[i] - epochs[i - 1]
            if delta > 0.0:
                deltas.append(delta)
        return deltas

    def _find_dominant_interval(self, deltas: List[float]) -> float:
        """Find the most common (dominant) interval among deltas.

        Buckets intervals by rounding to a resolution appropriate to
        the order of magnitude of the median delta, then selects the
        bucket with the highest count.

        Args:
            deltas: List of positive interval deltas in seconds.

        Returns:
            Dominant interval in seconds.
        """
        if not deltas:
            return 0.0

        if len(deltas) == 1:
            return deltas[0]

        median = _safe_median(deltas)
        if median <= 0.0:
            return _safe_mean(deltas)

        # Choose bucket resolution: ~5% of the median
        resolution = max(1.0, median * 0.05)

        # Bucket each delta
        buckets: Dict[int, List[float]] = {}
        for d in deltas:
            bucket_key = int(round(d / resolution))
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(d)

        # Find the bucket with the most entries
        best_bucket_key = max(buckets, key=lambda k: len(buckets[k]))
        best_bucket_values = buckets[best_bucket_key]

        # Return the mean of the dominant bucket
        return _safe_mean(best_bucket_values)

    def _compute_regularity_score(
        self,
        deltas: List[float],
        dominant_interval: float,
    ) -> float:
        """Compute regularity score: fraction of intervals matching dominant.

        An interval matches if it is within 25% of the dominant interval.

        Args:
            deltas: List of interval deltas in seconds.
            dominant_interval: Dominant interval in seconds.

        Returns:
            Regularity score between 0.0 and 1.0.
        """
        if not deltas or dominant_interval <= 0.0:
            return 0.0

        tolerance = dominant_interval * 0.25
        matches = sum(
            1 for d in deltas if abs(d - dominant_interval) <= tolerance
        )
        return matches / len(deltas)

    def _compute_confidence(
        self,
        sample_size: int,
        regularity: float,
        min_points: int,
    ) -> float:
        """Compute confidence in frequency detection.

        Confidence combines sample size adequacy and regularity.
        More data points and higher regularity yield higher confidence.

        Formula:
            size_factor = min(1.0, sample_size / (min_points * 3))
            confidence = size_factor * 0.4 + regularity * 0.6

        Args:
            sample_size: Number of intervals analysed.
            regularity: Regularity score (0-1).
            min_points: Minimum required data points from config.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        target_size = max(1, min_points * 3)
        size_factor = min(1.0, sample_size / target_size)
        confidence = size_factor * 0.4 + regularity * 0.6
        return min(1.0, max(0.0, confidence))

    # ------------------------------------------------------------------
    # Private helpers - provenance and metrics
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        operation: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> str:
        """Record provenance and return a SHA-256 hash.

        When the provenance tracker is available, records a chain link.
        Always returns a deterministic hash of the combined input and
        output data.

        Args:
            operation: Operation name for provenance recording.
            input_data: Input data dictionary.
            output_data: Output data dictionary.

        Returns:
            SHA-256 hash string.
        """
        input_hash = _build_hash(input_data)
        output_hash = _build_hash(output_data)

        if self._provenance is not None:
            try:
                chain_hash = self._provenance.add_to_chain(
                    operation=operation,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    metadata={
                        "engine": "frequency_analyzer",
                        "agent": "AGENT-DATA-014",
                    },
                )
                return chain_hash
            except Exception as exc:
                logger.warning(
                    "Provenance recording failed for %s: %s",
                    operation, exc,
                )

        # Fallback: combined hash
        return _build_hash({
            "operation": operation,
            "input_hash": input_hash,
            "output_hash": output_hash,
        })

    def _observe_metric(self, operation: str, elapsed: float) -> None:
        """Record operation duration metric if metrics are available.

        Args:
            operation: Operation name label.
            elapsed: Duration in seconds.
        """
        if _observe_duration is not None:
            try:
                _observe_duration(operation, elapsed)
            except Exception as exc:
                logger.debug("Metric observation skipped: %s", exc)


__all__ = [
    "FrequencyAnalyzerEngine",
    "FrequencyLevel",
    "FREQUENCY_INTERVALS",
    "FREQUENCY_TOLERANCES",
]
