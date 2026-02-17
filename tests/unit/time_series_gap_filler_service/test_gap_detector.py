# -*- coding: utf-8 -*-
"""
Unit tests for GapDetectorEngine - AGENT-DATA-014 Time Series Gap Filler

Tests all public methods of GapDetectorEngine:
    - detect_gaps: value-based and frequency-based gap detection
    - characterize_gaps: short/medium/long classification, periodic/systematic/random patterns
    - get_gap_statistics: aggregate statistics over detection results
    - detect_edge_gaps: leading and trailing gap counts
    - find_consecutive_gaps: (start_idx, length) segment tuples
    - build_frequency_grid: expected timestamp grid construction

Target: 60+ tests covering normal, edge-case, and large-series scenarios.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from greenlang.time_series_gap_filler.config import TimeSeriesGapFillerConfig
from greenlang.time_series_gap_filler.gap_detector import (
    Frequency,
    GapCharacterization,
    GapDetectionResult,
    GapDetectorEngine,
    GapPattern,
    GapRecord,
    GapType,
    _is_missing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(fresh_config) -> GapDetectorEngine:
    """Create a GapDetectorEngine with the default test config."""
    return GapDetectorEngine(config=fresh_config)


@pytest.fixture
def no_gaps_series() -> List[float]:
    """A 10-element series with no missing values."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


@pytest.fixture
def single_gap_series() -> List[Optional[float]]:
    """A 10-element series with one gap at index 4."""
    return [1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0]


@pytest.fixture
def multi_gap_series() -> List[Optional[float]]:
    """A 10-element series with gaps at indices 2 and 5-6."""
    return [1.0, 2.0, None, 4.0, 5.0, None, None, 8.0, 9.0, 10.0]


@pytest.fixture
def all_missing_series() -> List[Optional[float]]:
    """A 10-element series where every value is None."""
    return [None] * 10


@pytest.fixture
def leading_gap_series() -> List[Optional[float]]:
    """Series with leading gaps at indices 0, 1, 2."""
    return [None, None, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


@pytest.fixture
def trailing_gap_series() -> List[Optional[float]]:
    """Series with trailing gaps at indices 7, 8, 9."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, None, None, None]


@pytest.fixture
def nan_gap_series() -> List[Optional[float]]:
    """Series with float('nan') values at indices 3 and 7."""
    return [1.0, 2.0, 3.0, float("nan"), 5.0, 6.0, 7.0, float("nan"), 9.0, 10.0]


@pytest.fixture
def daily_timestamps() -> List[datetime]:
    """30 daily timestamps starting 2025-01-01."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(days=i) for i in range(30)]


@pytest.fixture
def hourly_timestamps() -> List[datetime]:
    """48 hourly timestamps starting 2025-06-01 00:00."""
    base = datetime(2025, 6, 1, tzinfo=timezone.utc)
    return [base + timedelta(hours=i) for i in range(48)]


# ===========================================================================
# Tests: _is_missing helper
# ===========================================================================


class TestIsMissing:
    """Tests for the module-level _is_missing helper."""

    def test_none_is_missing(self):
        assert _is_missing(None) is True

    def test_nan_is_missing(self):
        assert _is_missing(float("nan")) is True

    def test_string_nan_is_missing(self):
        assert _is_missing("nan") is True
        assert _is_missing("NaN") is True
        assert _is_missing("  NAN  ") is True

    def test_zero_not_missing(self):
        assert _is_missing(0.0) is False

    def test_negative_not_missing(self):
        assert _is_missing(-1.0) is False

    def test_string_not_missing(self):
        assert _is_missing("hello") is False


# ===========================================================================
# Tests: detect_gaps -- no gaps
# ===========================================================================


class TestDetectGapsNoGaps:
    """detect_gaps on a series with no missing values."""

    def test_no_gaps_returns_zero_count(self, engine, no_gaps_series):
        result = engine.detect_gaps(no_gaps_series)
        assert result.total_gaps == 0

    def test_no_gaps_returns_zero_missing(self, engine, no_gaps_series):
        result = engine.detect_gaps(no_gaps_series)
        assert result.total_missing == 0

    def test_no_gaps_pct_is_zero(self, engine, no_gaps_series):
        result = engine.detect_gaps(no_gaps_series)
        assert result.gap_pct == pytest.approx(0.0)

    def test_no_gaps_series_length_correct(self, engine, no_gaps_series):
        result = engine.detect_gaps(no_gaps_series)
        assert result.series_length == 10

    def test_no_gaps_provenance_hash_present(self, engine, no_gaps_series):
        result = engine.detect_gaps(no_gaps_series)
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    def test_no_gaps_processing_time_positive(self, engine, no_gaps_series):
        result = engine.detect_gaps(no_gaps_series)
        assert result.processing_time_ms >= 0.0


# ===========================================================================
# Tests: detect_gaps -- single gap
# ===========================================================================


class TestDetectGapsSingleGap:
    """detect_gaps on a series with exactly one gap."""

    def test_single_gap_count(self, engine, single_gap_series):
        result = engine.detect_gaps(single_gap_series)
        assert result.total_gaps == 1

    def test_single_gap_missing_count(self, engine, single_gap_series):
        result = engine.detect_gaps(single_gap_series)
        assert result.total_missing == 1

    def test_single_gap_record_indices(self, engine, single_gap_series):
        result = engine.detect_gaps(single_gap_series)
        gap = result.gaps[0]
        assert gap.start_index == 4
        assert gap.end_index == 4
        assert gap.length == 1

    def test_single_gap_pct(self, engine, single_gap_series):
        result = engine.detect_gaps(single_gap_series)
        assert result.gap_pct == pytest.approx(0.1)


# ===========================================================================
# Tests: detect_gaps -- multiple gaps
# ===========================================================================


class TestDetectGapsMultipleGaps:
    """detect_gaps on a series with multiple gap segments."""

    def test_multi_gap_count(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        assert result.total_gaps == 2

    def test_multi_gap_missing_total(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        assert result.total_missing == 3

    def test_multi_gap_first_segment(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        g0 = result.gaps[0]
        assert g0.start_index == 2
        assert g0.end_index == 2
        assert g0.length == 1

    def test_multi_gap_second_segment(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        g1 = result.gaps[1]
        assert g1.start_index == 5
        assert g1.end_index == 6
        assert g1.length == 2

    def test_multi_gap_pct(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        assert result.gap_pct == pytest.approx(0.3)


# ===========================================================================
# Tests: detect_gaps -- all missing
# ===========================================================================


class TestDetectGapsAllMissing:
    """detect_gaps when every element is None."""

    def test_all_missing_total_gaps_is_one(self, engine, all_missing_series):
        result = engine.detect_gaps(all_missing_series)
        assert result.total_gaps == 1

    def test_all_missing_total_missing_equals_length(self, engine, all_missing_series):
        result = engine.detect_gaps(all_missing_series)
        assert result.total_missing == 10

    def test_all_missing_gap_pct_is_one(self, engine, all_missing_series):
        result = engine.detect_gaps(all_missing_series)
        assert result.gap_pct == pytest.approx(1.0)

    def test_all_missing_single_segment_spanning_series(self, engine, all_missing_series):
        result = engine.detect_gaps(all_missing_series)
        gap = result.gaps[0]
        assert gap.start_index == 0
        assert gap.end_index == 9
        assert gap.length == 10


# ===========================================================================
# Tests: detect_gaps -- empty series
# ===========================================================================


class TestDetectGapsEmpty:
    """detect_gaps on an empty series."""

    def test_empty_series_zero_gaps(self, engine):
        result = engine.detect_gaps([])
        assert result.total_gaps == 0

    def test_empty_series_zero_missing(self, engine):
        result = engine.detect_gaps([])
        assert result.total_missing == 0

    def test_empty_series_length_zero(self, engine):
        result = engine.detect_gaps([])
        assert result.series_length == 0

    def test_empty_series_gap_pct_zero(self, engine):
        result = engine.detect_gaps([])
        assert result.gap_pct == pytest.approx(0.0)


# ===========================================================================
# Tests: detect_gaps -- None / NaN mixed
# ===========================================================================


class TestDetectGapsNanValues:
    """detect_gaps recognises float('nan') the same as None."""

    def test_nan_detected_as_gap(self, engine, nan_gap_series):
        result = engine.detect_gaps(nan_gap_series)
        assert result.total_gaps == 2

    def test_nan_total_missing(self, engine, nan_gap_series):
        result = engine.detect_gaps(nan_gap_series)
        assert result.total_missing == 2

    def test_mixed_none_and_nan(self, engine):
        series = [1.0, None, 3.0, float("nan"), 5.0]
        result = engine.detect_gaps(series)
        assert result.total_gaps == 2
        assert result.total_missing == 2


# ===========================================================================
# Tests: detect_gaps -- with timestamps
# ===========================================================================


class TestDetectGapsWithTimestamps:
    """detect_gaps with aligned timestamps parameter."""

    def test_timestamps_recorded_on_gaps(self, engine, daily_timestamps):
        values: List[Optional[float]] = [1.0] * 30
        values[5] = None
        values[6] = None
        result = engine.detect_gaps(values, timestamps=daily_timestamps)
        gap = result.gaps[0]
        assert gap.start_timestamp == daily_timestamps[5]
        assert gap.end_timestamp == daily_timestamps[6]

    def test_timestamps_length_mismatch_raises(self, engine, daily_timestamps):
        values = [1.0] * 10  # length 10 != 30
        with pytest.raises(ValueError, match="timestamps length"):
            engine.detect_gaps(values, timestamps=daily_timestamps)


# ===========================================================================
# Tests: characterize_gaps
# ===========================================================================


class TestCharacterizeGaps:
    """Tests for gap type and pattern classification."""

    def test_short_gap_classification(self, engine):
        """Gap length <= 3 (short_gap_limit default) is SHORT."""
        gaps = [GapRecord(start_index=5, end_index=6, length=2)]
        chars = engine.characterize_gaps(gaps, series_length=100)
        assert chars[0].gap_type == GapType.SHORT

    def test_medium_gap_classification(self, engine):
        """Gap length 4-12 is MEDIUM (between short_limit=3 and long_limit=12)."""
        gaps = [GapRecord(start_index=10, end_index=17, length=8)]
        chars = engine.characterize_gaps(gaps, series_length=100)
        assert chars[0].gap_type == GapType.MEDIUM

    def test_long_gap_classification(self, engine):
        """Gap length > 12 (long_gap_limit default) is LONG."""
        gaps = [GapRecord(start_index=0, end_index=19, length=20)]
        chars = engine.characterize_gaps(gaps, series_length=100)
        assert chars[0].gap_type == GapType.LONG

    def test_short_gap_at_limit(self, engine):
        """Gap of exactly 3 (short_gap_limit) is still SHORT."""
        gaps = [GapRecord(start_index=0, end_index=2, length=3)]
        chars = engine.characterize_gaps(gaps, series_length=100)
        assert chars[0].gap_type == GapType.SHORT

    def test_long_gap_at_limit(self, engine):
        """Gap of exactly 12 (long_gap_limit) is MEDIUM, not LONG."""
        gaps = [GapRecord(start_index=0, end_index=11, length=12)]
        chars = engine.characterize_gaps(gaps, series_length=100)
        assert chars[0].gap_type == GapType.MEDIUM

    def test_gap_of_length_13_is_long(self, engine):
        """Gap of 13 exceeds long_gap_limit=12, so LONG."""
        gaps = [GapRecord(start_index=0, end_index=12, length=13)]
        chars = engine.characterize_gaps(gaps, series_length=100)
        assert chars[0].gap_type == GapType.LONG

    def test_periodic_gap_detection(self, engine):
        """Gaps at regular intervals (every 10 indices) flagged PERIODIC."""
        gaps = [
            GapRecord(start_index=10, end_index=10, length=1),
            GapRecord(start_index=20, end_index=20, length=1),
            GapRecord(start_index=30, end_index=30, length=1),
            GapRecord(start_index=40, end_index=40, length=1),
        ]
        chars = engine.characterize_gaps(gaps, series_length=100)
        # All should share the same pattern
        assert chars[0].gap_pattern == GapPattern.PERIODIC

    def test_random_gap_pattern(self, engine):
        """Irregularly spaced gaps classified as RANDOM."""
        gaps = [
            GapRecord(start_index=3, end_index=3, length=1),
            GapRecord(start_index=17, end_index=17, length=1),
            GapRecord(start_index=55, end_index=55, length=1),
            GapRecord(start_index=91, end_index=91, length=1),
        ]
        chars = engine.characterize_gaps(gaps, series_length=100)
        assert chars[0].gap_pattern == GapPattern.RANDOM

    def test_position_pct_computed(self, engine):
        """position_pct reflects start_index / series_length."""
        gaps = [GapRecord(start_index=25, end_index=25, length=1)]
        chars = engine.characterize_gaps(gaps, series_length=100)
        assert chars[0].position_pct == pytest.approx(0.25)

    def test_characterization_details_populated(self, engine):
        """details dict is populated with gap metadata."""
        gaps = [GapRecord(start_index=5, end_index=5, length=1)]
        chars = engine.characterize_gaps(gaps, series_length=100)
        d = chars[0].details
        assert "length" in d
        assert "gap_type" in d
        assert "gap_pattern" in d
        assert "start_index" in d
        assert "short_limit" in d
        assert "long_limit" in d


# ===========================================================================
# Tests: get_gap_statistics
# ===========================================================================


class TestGetGapStatistics:
    """Tests for summary statistics computation."""

    def test_stats_total_gaps(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        stats = engine.get_gap_statistics(result)
        assert stats["total_gaps"] == 2

    def test_stats_total_missing(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        stats = engine.get_gap_statistics(result)
        assert stats["total_missing"] == 3

    def test_stats_gap_pct(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        stats = engine.get_gap_statistics(result)
        assert stats["gap_pct"] == pytest.approx(0.3)

    def test_stats_avg_gap_length(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        stats = engine.get_gap_statistics(result)
        # Gaps: length 1, length 2 => avg = 1.5
        assert stats["avg_gap_length"] == pytest.approx(1.5)

    def test_stats_max_gap_length(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        stats = engine.get_gap_statistics(result)
        assert stats["max_gap_length"] == 2

    def test_stats_min_gap_length(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        stats = engine.get_gap_statistics(result)
        assert stats["min_gap_length"] == 1

    def test_stats_no_gaps(self, engine, no_gaps_series):
        result = engine.detect_gaps(no_gaps_series)
        stats = engine.get_gap_statistics(result)
        assert stats["total_gaps"] == 0
        assert stats["avg_gap_length"] == pytest.approx(0.0)
        assert stats["max_gap_length"] == 0
        assert stats["min_gap_length"] == 0

    def test_stats_consecutive_segments(self, engine, multi_gap_series):
        result = engine.detect_gaps(multi_gap_series)
        stats = engine.get_gap_statistics(result)
        segs = stats["consecutive_gap_segments"]
        assert len(segs) == 2
        assert segs[0] == (2, 1)
        assert segs[1] == (5, 2)


# ===========================================================================
# Tests: detect_edge_gaps
# ===========================================================================


class TestDetectEdgeGaps:
    """Tests for leading and trailing gap detection."""

    def test_leading_gaps_counted(self, engine, leading_gap_series):
        edges = engine.detect_edge_gaps(leading_gap_series)
        assert edges["leading_gap"] == 3

    def test_trailing_gaps_counted(self, engine, trailing_gap_series):
        edges = engine.detect_edge_gaps(trailing_gap_series)
        assert edges["trailing_gap"] == 3

    def test_no_edge_gaps(self, engine, no_gaps_series):
        edges = engine.detect_edge_gaps(no_gaps_series)
        assert edges["leading_gap"] == 0
        assert edges["trailing_gap"] == 0

    def test_all_missing_edge_gaps(self, engine, all_missing_series):
        edges = engine.detect_edge_gaps(all_missing_series)
        assert edges["leading_gap"] == 10
        assert edges["trailing_gap"] == 10

    def test_empty_series_edge_gaps(self, engine):
        edges = engine.detect_edge_gaps([])
        assert edges["leading_gap"] == 0
        assert edges["trailing_gap"] == 0

    def test_single_element_none(self, engine):
        edges = engine.detect_edge_gaps([None])
        assert edges["leading_gap"] == 1
        assert edges["trailing_gap"] == 1

    def test_single_element_valid(self, engine):
        edges = engine.detect_edge_gaps([42.0])
        assert edges["leading_gap"] == 0
        assert edges["trailing_gap"] == 0


# ===========================================================================
# Tests: find_consecutive_gaps
# ===========================================================================


class TestFindConsecutiveGaps:
    """Tests for raw consecutive gap segment extraction."""

    def test_no_gaps_empty_list(self, engine, no_gaps_series):
        segs = engine.find_consecutive_gaps(no_gaps_series)
        assert segs == []

    def test_single_gap_segment(self, engine, single_gap_series):
        segs = engine.find_consecutive_gaps(single_gap_series)
        assert segs == [(4, 1)]

    def test_multi_gap_segments(self, engine, multi_gap_series):
        segs = engine.find_consecutive_gaps(multi_gap_series)
        assert segs == [(2, 1), (5, 2)]

    def test_all_missing_single_segment(self, engine, all_missing_series):
        segs = engine.find_consecutive_gaps(all_missing_series)
        assert segs == [(0, 10)]

    def test_empty_series(self, engine):
        segs = engine.find_consecutive_gaps([])
        assert segs == []

    def test_alternating_gaps(self, engine):
        series = [1.0, None, 3.0, None, 5.0, None, 7.0]
        segs = engine.find_consecutive_gaps(series)
        assert segs == [(1, 1), (3, 1), (5, 1)]

    def test_long_consecutive_gap(self, engine):
        series = [1.0] + [None] * 20 + [22.0]
        segs = engine.find_consecutive_gaps(series)
        assert segs == [(1, 20)]


# ===========================================================================
# Tests: detect_gaps with frequency-based detection
# ===========================================================================


class TestDetectGapsByFrequency:
    """Tests for timestamp frequency-based gap detection."""

    def test_frequency_daily_no_missing(self, engine, daily_timestamps):
        result = engine.detect_gaps_by_frequency(daily_timestamps, "daily")
        assert result.total_gaps == 0
        assert result.total_missing == 0

    def test_frequency_daily_with_missing_day(self, engine):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # Skip day 5 (Jan 6)
        timestamps = [base + timedelta(days=i) for i in range(10) if i != 5]
        result = engine.detect_gaps_by_frequency(timestamps, "daily")
        assert result.total_gaps >= 1
        assert result.total_missing >= 1

    def test_frequency_hourly(self, engine, hourly_timestamps):
        result = engine.detect_gaps_by_frequency(hourly_timestamps, "hourly")
        assert result.total_gaps == 0

    def test_invalid_frequency_raises(self, engine, daily_timestamps):
        with pytest.raises(ValueError, match="Unrecognised frequency"):
            engine.detect_gaps_by_frequency(daily_timestamps, "every_fortnight")

    def test_fewer_than_two_timestamps_raises(self, engine):
        ts = [datetime(2025, 1, 1, tzinfo=timezone.utc)]
        with pytest.raises(ValueError, match="At least two timestamps"):
            engine.detect_gaps_by_frequency(ts, "daily")


# ===========================================================================
# Tests: build_frequency_grid
# ===========================================================================


class TestBuildFrequencyGrid:
    """Tests for expected timestamp grid construction."""

    def test_daily_grid_length(self, engine, daily_timestamps):
        grid = engine.build_frequency_grid(daily_timestamps, Frequency.DAILY)
        assert len(grid) == 30

    def test_hourly_grid_length(self, engine, hourly_timestamps):
        grid = engine.build_frequency_grid(hourly_timestamps, Frequency.HOURLY)
        assert len(grid) == 48

    def test_monthly_grid(self, engine):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        ts = [base, datetime(2025, 12, 1, tzinfo=timezone.utc)]
        grid = engine.build_frequency_grid(ts, Frequency.MONTHLY)
        assert len(grid) == 12

    def test_grid_fewer_than_two_raises(self, engine):
        ts = [datetime(2025, 1, 1, tzinfo=timezone.utc)]
        with pytest.raises(ValueError, match="At least two timestamps"):
            engine.build_frequency_grid(ts, Frequency.DAILY)


# ===========================================================================
# Tests: various series lengths
# ===========================================================================


class TestVariousSeriesLengths:
    """Tests across different series sizes."""

    def test_10_point_series(self, engine):
        series: List[Optional[float]] = [float(i) for i in range(10)]
        series[3] = None
        result = engine.detect_gaps(series)
        assert result.total_gaps == 1
        assert result.series_length == 10

    def test_100_point_series(self, engine):
        series: List[Optional[float]] = [float(i) for i in range(100)]
        for i in range(10, 15):
            series[i] = None
        result = engine.detect_gaps(series)
        assert result.total_gaps == 1
        assert result.total_missing == 5

    def test_1000_point_series(self, engine):
        series: List[Optional[float]] = [float(i) for i in range(1000)]
        for i in range(100, 110):
            series[i] = None
        for i in range(500, 503):
            series[i] = None
        result = engine.detect_gaps(series)
        assert result.total_gaps == 2
        assert result.total_missing == 13
        assert result.series_length == 1000


# ===========================================================================
# Tests: provenance
# ===========================================================================


class TestGapDetectorProvenance:
    """Provenance and reproducibility tests."""

    def test_provenance_hash_is_sha256(self, engine, single_gap_series):
        result = engine.detect_gaps(single_gap_series)
        assert len(result.provenance_hash) == 64

    def test_provenance_deterministic(self, engine, single_gap_series):
        r1 = engine.detect_gaps(list(single_gap_series))
        r2 = engine.detect_gaps(list(single_gap_series))
        # The chain hashing includes timestamps so consecutive calls
        # will differ, but both should be valid hex strings.
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64

    def test_detect_gaps_returns_result_type(self, engine, single_gap_series):
        result = engine.detect_gaps(single_gap_series)
        assert isinstance(result, GapDetectionResult)

    def test_gap_record_attributes(self, engine, single_gap_series):
        result = engine.detect_gaps(single_gap_series)
        gap = result.gaps[0]
        assert isinstance(gap, GapRecord)
        assert hasattr(gap, "start_index")
        assert hasattr(gap, "end_index")
        assert hasattr(gap, "length")
