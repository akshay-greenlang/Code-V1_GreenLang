# -*- coding: utf-8 -*-
"""
Gap Detection Engine - AGENT-DATA-014 Time Series Gap Filler

Pure-Python gap detection engine for identifying, classifying, and
characterising gaps (missing values and missing timestamps) in time
series data.  Supports value-based gap detection (None/NaN), frequency-
based timestamp gap detection (hourly through annual), and gap
characterisation (short, medium, long, periodic, systematic, random).

Zero-Hallucination: All calculations use deterministic Python arithmetic.
No LLM calls for numeric computations.

Engine 1 of 7 in the Time Series Gap Filler pipeline.

Example:
    >>> from greenlang.time_series_gap_filler.gap_detector import GapDetectorEngine
    >>> engine = GapDetectorEngine()
    >>> result = engine.detect_gaps([1.0, None, None, 4.0, 5.0])
    >>> print(result.total_gaps, result.total_missing)
    1 2

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
Status: Production Ready
"""

from __future__ import annotations

import calendar
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.time_series_gap_filler.config import get_config
from greenlang.time_series_gap_filler.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.time_series_gap_filler.metrics import (
        inc_gaps_detected,
        observe_duration,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

    def inc_gaps_detected(count: int = 1) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    def observe_duration(operation: str, duration: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    logger.info(
        "time_series_gap_filler.metrics not available; "
        "gap detector metrics disabled"
    )


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class GapType(str, Enum):
    """Classification of a detected gap by its length.

    SHORT: Gap length <= config.short_gap_limit (default 3).
    MEDIUM: Gap length between short and long limits.
    LONG: Gap length > config.long_gap_limit (default 12).
    """

    SHORT = "short_gap"
    MEDIUM = "medium_gap"
    LONG = "long_gap"


class GapPattern(str, Enum):
    """Pattern classification of a gap based on its occurrence.

    PERIODIC: Gaps appear at regular intervals.
    SYSTEMATIC: Gaps correlate with specific positions (e.g. weekends).
    RANDOM: No discernible pattern.
    """

    PERIODIC = "periodic_gap"
    SYSTEMATIC = "systematic_gap"
    RANDOM = "random_gap"


class Frequency(str, Enum):
    """Expected frequency of a time series for gap detection.

    Defines the expected spacing between consecutive timestamps.
    """

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


# ---------------------------------------------------------------------------
# Frequency delta mapping
# ---------------------------------------------------------------------------

#: Mapping from Frequency to approximate timedelta for grid building.
#: Monthly, quarterly, and annual use day-based approximations because
#: calendar months vary in length; the grid builder handles these
#: specially via calendar arithmetic.
FREQUENCY_DELTAS: Dict[Frequency, timedelta] = {
    Frequency.HOURLY: timedelta(hours=1),
    Frequency.DAILY: timedelta(days=1),
    Frequency.WEEKLY: timedelta(days=7),
    Frequency.MONTHLY: timedelta(days=30),
    Frequency.QUARTERLY: timedelta(days=91),
    Frequency.ANNUAL: timedelta(days=365),
}


# ---------------------------------------------------------------------------
# Data models (dataclasses -- pure Python, no Pydantic dependency)
# ---------------------------------------------------------------------------


@dataclass
class GapRecord:
    """A single contiguous gap segment in the series.

    Attributes:
        start_index: Index of the first missing value in the gap.
        end_index: Index of the last missing value in the gap.
        length: Number of consecutive missing values.
        start_timestamp: Timestamp of the first missing point (if available).
        end_timestamp: Timestamp of the last missing point (if available).
    """

    start_index: int
    end_index: int
    length: int
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None


@dataclass
class GapCharacterization:
    """Characterisation of a single gap including type and pattern.

    Attributes:
        gap: The underlying gap record.
        gap_type: Length classification (short, medium, long).
        gap_pattern: Pattern classification (periodic, systematic, random).
        position_pct: Relative position of the gap in the series (0.0-1.0).
        details: Additional characterisation details.
    """

    gap: GapRecord
    gap_type: GapType
    gap_pattern: GapPattern
    position_pct: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GapDetectionResult:
    """Complete result of a gap detection operation.

    Attributes:
        gaps: List of detected gap segments.
        total_gaps: Number of distinct gap segments found.
        total_missing: Total number of missing values across all gaps.
        series_length: Length of the input series.
        gap_pct: Fraction of the series that is missing (0.0-1.0).
        characterizations: Optional list of gap characterisations.
        edge_gaps: Leading and trailing gap counts.
        provenance_hash: SHA-256 provenance hash for audit trail.
        processing_time_ms: Detection processing time in milliseconds.
    """

    gaps: List[GapRecord] = field(default_factory=list)
    total_gaps: int = 0
    total_missing: int = 0
    series_length: int = 0
    gap_pct: float = 0.0
    characterizations: List[GapCharacterization] = field(
        default_factory=list,
    )
    edge_gaps: Dict[str, int] = field(default_factory=dict)
    provenance_hash: str = ""
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """Determine whether a value represents a missing data point.

    Handles None, float('nan'), and the string ``'nan'``
    (case-insensitive).

    Args:
        value: The value to check.

    Returns:
        True if the value is considered missing.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() == "nan":
        return True
    return False


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


# ---------------------------------------------------------------------------
# GapDetectorEngine
# ---------------------------------------------------------------------------


class GapDetectorEngine:
    """Pure-Python engine for detecting and characterising gaps in time series.

    Detects gaps by scanning for missing values (None/NaN), by comparing
    timestamps against an expected frequency grid, or both.  Each gap
    segment is recorded with its start/end indices and length.  Gaps are
    then characterised by length (short / medium / long) and pattern
    (periodic / systematic / random).

    Attributes:
        _config: Time series gap filler configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = GapDetectorEngine()
        >>> result = engine.detect_gaps([1, 2, None, None, 5, 6, None, 8])
        >>> assert result.total_gaps == 2
        >>> assert result.total_missing == 3
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize GapDetectorEngine.

        Args:
            config: Optional TimeSeriesGapFillerConfig override.
                If not provided, the global singleton is used.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        logger.info("GapDetectorEngine initialized")

    # ------------------------------------------------------------------
    # Primary detection: value-based + optional timestamp-based
    # ------------------------------------------------------------------

    def detect_gaps(
        self,
        values: List[Any],
        timestamps: Optional[List[datetime]] = None,
        expected_frequency: Optional[str] = None,
    ) -> GapDetectionResult:
        """Detect gaps in a time series by missing values and timestamps.

        Scans *values* for None/NaN entries and groups consecutive
        missing positions into GapRecord segments.  When *timestamps*
        and *expected_frequency* are both provided, also detects
        missing timestamps that should exist according to the expected
        frequency grid.

        Args:
            values: List of data values (may contain None/NaN).
            timestamps: Optional list of timestamps aligned with values.
            expected_frequency: Optional frequency string for timestamp
                grid comparison (hourly, daily, weekly, monthly,
                quarterly, annual).

        Returns:
            GapDetectionResult with all detected gap segments.

        Raises:
            ValueError: If timestamps length does not match values length.
        """
        start = time.time()

        if timestamps is not None and len(timestamps) != len(values):
            raise ValueError(
                f"timestamps length ({len(timestamps)}) must match "
                f"values length ({len(values)})"
            )

        series_length = len(values)

        # Step 1: Detect value-based gaps (None/NaN)
        consecutive_segments = self.find_consecutive_gaps(values)

        gaps: List[GapRecord] = []
        for seg_start, seg_len in consecutive_segments:
            seg_end = seg_start + seg_len - 1
            gap = GapRecord(
                start_index=seg_start,
                end_index=seg_end,
                length=seg_len,
                start_timestamp=(
                    timestamps[seg_start] if timestamps else None
                ),
                end_timestamp=(
                    timestamps[seg_end] if timestamps else None
                ),
            )
            gaps.append(gap)

        # Step 2: If frequency provided, also detect timestamp gaps
        if (
            timestamps is not None
            and expected_frequency is not None
            and len(timestamps) >= 2
        ):
            freq_result = self.detect_gaps_by_frequency(
                timestamps, expected_frequency,
            )
            # Merge frequency gaps that are not already covered
            existing_indices = set()
            for g in gaps:
                for idx in range(g.start_index, g.end_index + 1):
                    existing_indices.add(idx)

            for freq_gap in freq_result.gaps:
                if freq_gap.start_index not in existing_indices:
                    gaps.append(freq_gap)

            gaps.sort(key=lambda g: g.start_index)

        total_missing = sum(g.length for g in gaps)
        gap_pct = total_missing / series_length if series_length > 0 else 0.0

        # Step 3: Characterise gaps
        characterizations: List[GapCharacterization] = []
        if gaps:
            characterizations = self.characterize_gaps(gaps, series_length)

        # Step 4: Detect edge gaps
        edge_gaps = self.detect_edge_gaps(values)

        # Step 5: Provenance
        input_hash = self._provenance.hash_record({
            "series_length": series_length,
            "total_gaps": len(gaps),
            "total_missing": total_missing,
        })
        output_hash = self._provenance.hash_record({
            "total_gaps": len(gaps),
            "total_missing": total_missing,
            "gap_pct": gap_pct,
        })
        provenance_hash = self._provenance.add_to_chain(
            operation="detect_gaps",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "series_length": series_length,
                "expected_frequency": expected_frequency,
                "has_timestamps": timestamps is not None,
            },
        )

        elapsed = time.time() - start
        processing_time_ms = elapsed * 1000.0

        # Metrics
        inc_gaps_detected(len(gaps))
        observe_duration("detect_gaps", elapsed)

        result = GapDetectionResult(
            gaps=gaps,
            total_gaps=len(gaps),
            total_missing=total_missing,
            series_length=series_length,
            gap_pct=gap_pct,
            characterizations=characterizations,
            edge_gaps=edge_gaps,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "Gap detection complete: series_length=%d, gaps=%d, "
            "missing=%d (%.1f%%), %.3fms",
            series_length, len(gaps), total_missing,
            gap_pct * 100, processing_time_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Frequency-based detection
    # ------------------------------------------------------------------

    def detect_gaps_by_frequency(
        self,
        timestamps: List[datetime],
        expected_frequency: str,
    ) -> GapDetectionResult:
        """Detect missing timestamps against an expected frequency grid.

        Builds an expected timestamp grid from the first to last
        observed timestamp and identifies grid positions with no
        corresponding observed timestamp.

        Supports: hourly, daily, weekly, monthly, quarterly, annual.

        Args:
            timestamps: Sorted list of observed timestamps.
            expected_frequency: Frequency string (e.g. ``"daily"``).

        Returns:
            GapDetectionResult with gaps at missing grid positions.

        Raises:
            ValueError: If expected_frequency is not recognised or
                fewer than two timestamps are provided.
        """
        start = time.time()

        if len(timestamps) < 2:
            raise ValueError(
                "At least two timestamps are required for "
                "frequency-based gap detection"
            )

        try:
            frequency = Frequency(expected_frequency.lower())
        except ValueError:
            raise ValueError(
                f"Unrecognised frequency: {expected_frequency!r}. "
                f"Supported: {[f.value for f in Frequency]}"
            )

        sorted_ts = sorted(timestamps)
        grid = self.build_frequency_grid(sorted_ts, frequency)
        observed_set = set(timestamps)

        # Build a list of "present" / "missing" flags on the grid
        missing_flags: List[bool] = [
            ts not in observed_set for ts in grid
        ]

        # Find consecutive missing segments on the grid
        segments: List[Tuple[int, int]] = []
        i = 0
        grid_len = len(grid)
        while i < grid_len:
            if missing_flags[i]:
                seg_start = i
                while i < grid_len and missing_flags[i]:
                    i += 1
                seg_len = i - seg_start
                segments.append((seg_start, seg_len))
            else:
                i += 1

        gaps: List[GapRecord] = []
        for seg_start, seg_len in segments:
            seg_end = seg_start + seg_len - 1
            gaps.append(GapRecord(
                start_index=seg_start,
                end_index=seg_end,
                length=seg_len,
                start_timestamp=grid[seg_start],
                end_timestamp=grid[seg_end],
            ))

        total_missing = sum(g.length for g in gaps)
        gap_pct = total_missing / grid_len if grid_len > 0 else 0.0

        # Provenance
        input_hash = self._provenance.hash_record({
            "n_timestamps": len(timestamps),
            "frequency": expected_frequency,
            "grid_length": grid_len,
        })
        output_hash = self._provenance.hash_record({
            "total_gaps": len(gaps),
            "total_missing": total_missing,
            "gap_pct": gap_pct,
        })
        provenance_hash = self._provenance.add_to_chain(
            operation="detect_gaps_by_frequency",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "frequency": expected_frequency,
                "grid_length": grid_len,
            },
        )

        elapsed = time.time() - start
        processing_time_ms = elapsed * 1000.0

        inc_gaps_detected(len(gaps))
        observe_duration("detect_gaps_by_frequency", elapsed)

        result = GapDetectionResult(
            gaps=gaps,
            total_gaps=len(gaps),
            total_missing=total_missing,
            series_length=grid_len,
            gap_pct=gap_pct,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "Frequency gap detection: freq=%s, grid=%d, gaps=%d, "
            "missing=%d (%.1f%%), %.3fms",
            expected_frequency, grid_len, len(gaps),
            total_missing, gap_pct * 100, processing_time_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Gap characterisation
    # ------------------------------------------------------------------

    def characterize_gaps(
        self,
        gaps: List[GapRecord],
        series_length: int,
    ) -> List[GapCharacterization]:
        """Classify and characterise each gap in the inventory.

        For each gap:
            - Assigns a length type: SHORT (<= short_gap_limit),
              LONG (> long_gap_limit), or MEDIUM (between the two).
            - Detects periodic gaps (regular spacing between starts).
            - Detects systematic gaps (correlated with specific
              modular positions, e.g. always at weekends).
            - Falls back to RANDOM if no pattern is found.

        Args:
            gaps: List of GapRecord objects to characterise.
            series_length: Total length of the original series.

        Returns:
            List of GapCharacterization, one per input gap.
        """
        short_limit = self._config.short_gap_limit
        long_limit = self._config.long_gap_limit

        # Pre-compute pattern flags across all gaps
        periodic = self._detect_periodic_pattern(gaps)
        systematic = self._detect_systematic_pattern(
            gaps, series_length,
        )

        characterizations: List[GapCharacterization] = []
        for gap in gaps:
            # Length classification
            if gap.length <= short_limit:
                gap_type = GapType.SHORT
            elif gap.length > long_limit:
                gap_type = GapType.LONG
            else:
                gap_type = GapType.MEDIUM

            # Pattern classification
            if periodic:
                gap_pattern = GapPattern.PERIODIC
            elif systematic:
                gap_pattern = GapPattern.SYSTEMATIC
            else:
                gap_pattern = GapPattern.RANDOM

            position_pct = (
                gap.start_index / series_length
                if series_length > 0
                else 0.0
            )

            details: Dict[str, Any] = {
                "length": gap.length,
                "gap_type": gap_type.value,
                "gap_pattern": gap_pattern.value,
                "start_index": gap.start_index,
                "end_index": gap.end_index,
                "short_limit": short_limit,
                "long_limit": long_limit,
            }

            characterizations.append(GapCharacterization(
                gap=gap,
                gap_type=gap_type,
                gap_pattern=gap_pattern,
                position_pct=position_pct,
                details=details,
            ))

        logger.debug(
            "Characterised %d gaps: short=%d, medium=%d, long=%d",
            len(characterizations),
            sum(1 for c in characterizations if c.gap_type == GapType.SHORT),
            sum(1 for c in characterizations if c.gap_type == GapType.MEDIUM),
            sum(1 for c in characterizations if c.gap_type == GapType.LONG),
        )
        return characterizations

    # ------------------------------------------------------------------
    # Gap statistics
    # ------------------------------------------------------------------

    def get_gap_statistics(
        self,
        detection_result: GapDetectionResult,
    ) -> Dict[str, Any]:
        """Compute summary statistics for a gap detection result.

        Args:
            detection_result: A GapDetectionResult from detect_gaps
                or detect_gaps_by_frequency.

        Returns:
            Dictionary containing:
                - total_gaps: Number of gap segments.
                - total_missing: Total missing values.
                - gap_pct: Fraction of series that is missing.
                - avg_gap_length: Average gap segment length.
                - max_gap_length: Length of the longest gap.
                - min_gap_length: Length of the shortest gap.
                - gap_type_distribution: Count per GapType.
                - consecutive_gap_segments: List of (start, length).
        """
        gaps = detection_result.gaps
        lengths = [g.length for g in gaps]

        avg_length = _safe_mean([float(l) for l in lengths])
        max_length = max(lengths) if lengths else 0
        min_length = min(lengths) if lengths else 0

        # Build type distribution from characterisations
        type_dist: Dict[str, int] = {
            GapType.SHORT.value: 0,
            GapType.MEDIUM.value: 0,
            GapType.LONG.value: 0,
        }
        for char in detection_result.characterizations:
            key = char.gap_type.value
            type_dist[key] = type_dist.get(key, 0) + 1

        # Build consecutive segment list
        consecutive_segments = [
            (g.start_index, g.length) for g in gaps
        ]

        stats: Dict[str, Any] = {
            "total_gaps": detection_result.total_gaps,
            "total_missing": detection_result.total_missing,
            "gap_pct": detection_result.gap_pct,
            "avg_gap_length": avg_length,
            "max_gap_length": max_length,
            "min_gap_length": min_length,
            "gap_type_distribution": type_dist,
            "consecutive_gap_segments": consecutive_segments,
        }

        logger.debug(
            "Gap statistics: gaps=%d, missing=%d, avg_len=%.1f, "
            "max_len=%d, min_len=%d",
            stats["total_gaps"], stats["total_missing"],
            avg_length, max_length, min_length,
        )
        return stats

    # ------------------------------------------------------------------
    # Edge gap detection
    # ------------------------------------------------------------------

    def detect_edge_gaps(
        self,
        values: List[Any],
    ) -> Dict[str, int]:
        """Detect leading and trailing gap counts in the series.

        A leading gap is a run of missing values starting at index 0.
        A trailing gap is a run of missing values ending at the last
        index.

        Args:
            values: List of data values (may contain None/NaN).

        Returns:
            Dictionary with ``leading_gap`` and ``trailing_gap`` counts.
        """
        n = len(values)
        leading = 0
        trailing = 0

        # Count leading missing values
        for i in range(n):
            if _is_missing(values[i]):
                leading += 1
            else:
                break

        # Count trailing missing values
        for i in range(n - 1, -1, -1):
            if _is_missing(values[i]):
                trailing += 1
            else:
                break

        # Edge case: entire series is missing
        if leading == n:
            trailing = n

        result = {
            "leading_gap": leading,
            "trailing_gap": trailing,
        }

        logger.debug(
            "Edge gaps: leading=%d, trailing=%d (series_length=%d)",
            leading, trailing, n,
        )
        return result

    # ------------------------------------------------------------------
    # Frequency grid builder
    # ------------------------------------------------------------------

    def build_frequency_grid(
        self,
        timestamps: List[datetime],
        frequency: Frequency,
    ) -> List[datetime]:
        """Build an expected timestamp grid from first to last timestamp.

        For sub-monthly frequencies (hourly, daily, weekly), uses
        fixed timedelta stepping.  For monthly, quarterly, and annual
        frequencies, uses calendar-aware month arithmetic to handle
        varying month lengths correctly.

        Args:
            timestamps: Sorted list of observed timestamps (at least 2).
            frequency: Expected frequency enum value.

        Returns:
            Sorted list of expected grid timestamps from min to max.

        Raises:
            ValueError: If fewer than two timestamps are provided.
        """
        if len(timestamps) < 2:
            raise ValueError(
                "At least two timestamps are required to build a "
                "frequency grid"
            )

        sorted_ts = sorted(timestamps)
        start = sorted_ts[0]
        end = sorted_ts[-1]
        grid: List[datetime] = []

        if frequency in (
            Frequency.HOURLY, Frequency.DAILY, Frequency.WEEKLY,
        ):
            delta = FREQUENCY_DELTAS[frequency]
            current = start
            while current <= end:
                grid.append(current)
                current = current + delta

        elif frequency == Frequency.MONTHLY:
            current = start
            while current <= end:
                grid.append(current)
                current = _add_months(current, 1)

        elif frequency == Frequency.QUARTERLY:
            current = start
            while current <= end:
                grid.append(current)
                current = _add_months(current, 3)

        elif frequency == Frequency.ANNUAL:
            current = start
            while current <= end:
                grid.append(current)
                current = _add_months(current, 12)

        logger.debug(
            "Built frequency grid: freq=%s, start=%s, end=%s, "
            "grid_size=%d",
            frequency.value, start.isoformat(), end.isoformat(),
            len(grid),
        )
        return grid

    # ------------------------------------------------------------------
    # Consecutive gap finder
    # ------------------------------------------------------------------

    def find_consecutive_gaps(
        self,
        values: List[Any],
    ) -> List[Tuple[int, int]]:
        """Find all runs of consecutive missing values.

        Scans the values list once (O(n)) and yields each contiguous
        segment of missing values as a (start_index, length) tuple.

        Args:
            values: List of data values (may contain None/NaN).

        Returns:
            List of (start_index, length) tuples for each gap segment.
        """
        segments: List[Tuple[int, int]] = []
        n = len(values)
        i = 0

        while i < n:
            if _is_missing(values[i]):
                seg_start = i
                while i < n and _is_missing(values[i]):
                    i += 1
                seg_len = i - seg_start
                segments.append((seg_start, seg_len))
            else:
                i += 1

        return segments

    # ------------------------------------------------------------------
    # Private pattern detection helpers
    # ------------------------------------------------------------------

    def _detect_periodic_pattern(
        self,
        gaps: List[GapRecord],
    ) -> bool:
        """Detect whether gaps occur at regular (periodic) intervals.

        Computes the intervals between consecutive gap start indices
        and checks whether they are approximately equal (coefficient
        of variation < 0.15).

        Args:
            gaps: List of GapRecord objects.

        Returns:
            True if gaps exhibit a periodic pattern.
        """
        if len(gaps) < 3:
            return False

        intervals: List[float] = []
        for i in range(1, len(gaps)):
            interval = gaps[i].start_index - gaps[i - 1].start_index
            intervals.append(float(interval))

        if not intervals:
            return False

        mean_interval = _safe_mean(intervals)
        if mean_interval <= 0:
            return False

        # Compute coefficient of variation
        variance = _safe_mean(
            [(x - mean_interval) ** 2 for x in intervals],
        )
        std = math.sqrt(variance) if variance > 0 else 0.0
        cv = std / mean_interval if mean_interval > 0 else 1.0

        # A CV below 0.15 indicates highly regular spacing
        is_periodic = cv < 0.15

        logger.debug(
            "Periodic pattern check: intervals=%d, mean=%.1f, "
            "cv=%.3f, periodic=%s",
            len(intervals), mean_interval, cv, is_periodic,
        )
        return is_periodic

    def _detect_systematic_pattern(
        self,
        gaps: List[GapRecord],
        series_length: int,
    ) -> bool:
        """Detect whether gaps correlate with specific modular positions.

        Checks if gap start indices cluster around specific positions
        modulo 7 (weekday pattern) or other common periods.  If more
        than 60%% of gaps share the same modular position, the pattern
        is classified as systematic.

        Args:
            gaps: List of GapRecord objects.
            series_length: Total length of the series.

        Returns:
            True if gaps exhibit a systematic positional pattern.
        """
        if len(gaps) < 3:
            return False

        # Check modulo-7 clustering (weekly/weekday pattern)
        for modulus in (7, 5, 12, 24):
            positions: Dict[int, int] = {}
            for gap in gaps:
                pos = gap.start_index % modulus
                positions[pos] = positions.get(pos, 0) + 1

            if positions:
                max_count = max(positions.values())
                concentration = max_count / len(gaps)
                if concentration > 0.6:
                    logger.debug(
                        "Systematic pattern: mod=%d, "
                        "concentration=%.2f",
                        modulus, concentration,
                    )
                    return True

        return False


# ---------------------------------------------------------------------------
# Calendar-aware month addition helper
# ---------------------------------------------------------------------------


def _add_months(dt: datetime, months: int) -> datetime:
    """Add a number of months to a datetime, clamping to valid day.

    Handles month-end edge cases (e.g. Jan 31 + 1 month = Feb 28/29).

    Args:
        dt: Source datetime.
        months: Number of months to add (positive integer).

    Returns:
        New datetime with the specified months added.
    """
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1

    # Clamp day to the last valid day of the target month
    max_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, max_day)

    return dt.replace(year=year, month=month, day=day)


# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = [
    "GapDetectorEngine",
    "GapDetectionResult",
    "GapRecord",
    "GapCharacterization",
    "GapType",
    "GapPattern",
    "Frequency",
    "FREQUENCY_DELTAS",
]
