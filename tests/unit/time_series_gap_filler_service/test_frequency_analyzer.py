# -*- coding: utf-8 -*-
"""
Unit tests for FrequencyAnalyzerEngine - AGENT-DATA-014 Time Series Gap Filler

Tests all public methods of FrequencyAnalyzerEngine:
    - analyze_frequency: detect dominant observation frequency
    - detect_regularity: regularity score (0-1)
    - get_dominant_period: autocorrelation-based period detection
    - classify_interval: seconds to FrequencyLevel mapping
    - compute_interval_statistics: mean, median, std, min, max, cv
    - detect_mixed_frequency: multi-frequency detection
    - validate_frequency: validate against expected frequency

Target: 50+ tests covering all frequency levels, edge cases, and numeric
precision.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Union

import pytest

from greenlang.time_series_gap_filler.config import TimeSeriesGapFillerConfig
from greenlang.time_series_gap_filler.frequency_analyzer import (
    FREQUENCY_INTERVALS,
    FREQUENCY_TOLERANCES,
    FrequencyAnalyzerEngine,
    FrequencyLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(fresh_config) -> FrequencyAnalyzerEngine:
    """Create a FrequencyAnalyzerEngine with default test config."""
    return FrequencyAnalyzerEngine(config=fresh_config)


@pytest.fixture
def hourly_timestamps() -> List[datetime]:
    """48 perfectly regular hourly timestamps."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(hours=i) for i in range(48)]


@pytest.fixture
def daily_timestamps() -> List[datetime]:
    """60 perfectly regular daily timestamps."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(days=i) for i in range(60)]


@pytest.fixture
def weekly_timestamps() -> List[datetime]:
    """26 perfectly regular weekly timestamps (6 months)."""
    base = datetime(2025, 1, 6, tzinfo=timezone.utc)
    return [base + timedelta(weeks=i) for i in range(26)]


@pytest.fixture
def monthly_timestamps() -> List[datetime]:
    """24 monthly timestamps (2 years)."""
    timestamps = []
    for year in (2025, 2026):
        for month in range(1, 13):
            timestamps.append(datetime(year, month, 1, tzinfo=timezone.utc))
    return timestamps


@pytest.fixture
def quarterly_timestamps() -> List[datetime]:
    """12 quarterly timestamps (3 years)."""
    timestamps = []
    for year in (2023, 2024, 2025):
        for month in (1, 4, 7, 10):
            timestamps.append(datetime(year, month, 1, tzinfo=timezone.utc))
    return timestamps


@pytest.fixture
def annual_timestamps() -> List[datetime]:
    """10 annual timestamps."""
    return [datetime(2016 + i, 1, 1, tzinfo=timezone.utc) for i in range(10)]


@pytest.fixture
def irregular_timestamps() -> List[datetime]:
    """Timestamps with highly irregular spacing."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    offsets_days = [0, 1, 5, 6, 20, 21, 22, 50, 100, 101]
    return [base + timedelta(days=d) for d in offsets_days]


# ===========================================================================
# Tests: analyze_frequency -- hourly data
# ===========================================================================


class TestAnalyzeFrequencyHourly:
    """Hourly data should be detected as hourly."""

    def test_hourly_frequency_level(self, engine, hourly_timestamps):
        result = engine.analyze_frequency(hourly_timestamps)
        assert result["frequency_level"] == FrequencyLevel.HOURLY.value

    def test_hourly_dominant_interval(self, engine, hourly_timestamps):
        result = engine.analyze_frequency(hourly_timestamps)
        assert result["dominant_interval_seconds"] == pytest.approx(3600.0, rel=0.1)

    def test_hourly_regularity_high(self, engine, hourly_timestamps):
        result = engine.analyze_frequency(hourly_timestamps)
        assert result["regularity_score"] >= 0.95

    def test_hourly_sample_size(self, engine, hourly_timestamps):
        result = engine.analyze_frequency(hourly_timestamps)
        assert result["sample_size"] == 47  # 48 timestamps -> 47 intervals

    def test_hourly_provenance_present(self, engine, hourly_timestamps):
        result = engine.analyze_frequency(hourly_timestamps)
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Tests: analyze_frequency -- daily data
# ===========================================================================


class TestAnalyzeFrequencyDaily:
    """Daily data should be detected as daily."""

    def test_daily_frequency_level(self, engine, daily_timestamps):
        result = engine.analyze_frequency(daily_timestamps)
        assert result["frequency_level"] == FrequencyLevel.DAILY.value

    def test_daily_dominant_interval(self, engine, daily_timestamps):
        result = engine.analyze_frequency(daily_timestamps)
        assert result["dominant_interval_seconds"] == pytest.approx(86400.0, rel=0.1)

    def test_daily_regularity(self, engine, daily_timestamps):
        result = engine.analyze_frequency(daily_timestamps)
        assert result["regularity_score"] >= 0.95


# ===========================================================================
# Tests: analyze_frequency -- weekly data
# ===========================================================================


class TestAnalyzeFrequencyWeekly:
    """Weekly data should be detected as weekly."""

    def test_weekly_frequency_level(self, engine, weekly_timestamps):
        result = engine.analyze_frequency(weekly_timestamps)
        assert result["frequency_level"] == FrequencyLevel.WEEKLY.value

    def test_weekly_dominant_interval(self, engine, weekly_timestamps):
        result = engine.analyze_frequency(weekly_timestamps)
        expected = 7 * 86400.0
        assert result["dominant_interval_seconds"] == pytest.approx(expected, rel=0.1)


# ===========================================================================
# Tests: analyze_frequency -- monthly data
# ===========================================================================


class TestAnalyzeFrequencyMonthly:
    """Monthly data should be detected as monthly."""

    def test_monthly_frequency_level(self, engine, monthly_timestamps):
        result = engine.analyze_frequency(monthly_timestamps)
        assert result["frequency_level"] == FrequencyLevel.MONTHLY.value


# ===========================================================================
# Tests: analyze_frequency -- quarterly data
# ===========================================================================


class TestAnalyzeFrequencyQuarterly:
    """Quarterly data should be detected as quarterly."""

    def test_quarterly_frequency_level(self, engine, quarterly_timestamps):
        result = engine.analyze_frequency(quarterly_timestamps)
        assert result["frequency_level"] == FrequencyLevel.QUARTERLY.value


# ===========================================================================
# Tests: analyze_frequency -- annual data
# ===========================================================================


class TestAnalyzeFrequencyAnnual:
    """Annual data should be detected as annual."""

    def test_annual_frequency_level(self, engine, annual_timestamps):
        result = engine.analyze_frequency(annual_timestamps)
        assert result["frequency_level"] == FrequencyLevel.ANNUAL.value


# ===========================================================================
# Tests: analyze_frequency -- edge cases
# ===========================================================================


class TestAnalyzeFrequencyEdgeCases:
    """Edge cases for analyze_frequency."""

    def test_single_timestamp_raises(self, engine):
        ts = [datetime(2025, 1, 1, tzinfo=timezone.utc)]
        with pytest.raises(ValueError, match="At least 2 timestamps"):
            engine.analyze_frequency(ts)

    def test_empty_timestamps_raises(self, engine):
        with pytest.raises(ValueError):
            engine.analyze_frequency([])

    def test_two_timestamps_minimal(self, engine):
        ts = [
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
        ]
        result = engine.analyze_frequency(ts)
        assert result["frequency_level"] == FrequencyLevel.DAILY.value
        assert result["sample_size"] == 1

    def test_numeric_epoch_timestamps(self, engine):
        """Engine accepts numeric epoch seconds."""
        base = 1700000000.0
        epochs = [base + i * 3600.0 for i in range(24)]
        result = engine.analyze_frequency(epochs)
        assert result["frequency_level"] == FrequencyLevel.HOURLY.value


# ===========================================================================
# Tests: detect_regularity
# ===========================================================================


class TestDetectRegularity:
    """Tests for regularity score computation."""

    def test_perfectly_regular_score_one(self, engine, daily_timestamps):
        score = engine.detect_regularity(daily_timestamps)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_irregular_low_score(self, engine, irregular_timestamps):
        score = engine.detect_regularity(irregular_timestamps)
        assert score < 0.5

    def test_single_timestamp_returns_zero(self, engine):
        ts = [datetime(2025, 1, 1, tzinfo=timezone.utc)]
        score = engine.detect_regularity(ts)
        assert score == 0.0

    def test_empty_returns_zero(self, engine):
        score = engine.detect_regularity([])
        assert score == 0.0

    def test_regularity_bounded(self, engine, hourly_timestamps):
        score = engine.detect_regularity(hourly_timestamps)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# Tests: get_dominant_period
# ===========================================================================


class TestGetDominantPeriod:
    """Tests for autocorrelation-based dominant period detection."""

    def test_known_period_7(self, engine):
        """Weekly pattern (period=7) in value series."""
        values = [math.sin(2 * math.pi * i / 7) for i in range(56)]
        period = engine.get_dominant_period(values)
        assert period == 7

    def test_known_period_12(self, engine):
        """Monthly pattern (period=12) in value series."""
        values = [math.sin(2 * math.pi * i / 12) for i in range(120)]
        period = engine.get_dominant_period(values)
        assert period == 12

    def test_constant_series_returns_zero(self, engine):
        """Constant series has no periodicity."""
        values = [5.0] * 50
        period = engine.get_dominant_period(values)
        assert period == 0

    def test_short_series_returns_zero(self, engine):
        """Series shorter than 4 returns 0."""
        values = [1.0, 2.0, 3.0]
        period = engine.get_dominant_period(values)
        assert period == 0

    def test_random_noise_no_period(self, engine):
        """Random noise should return 0 or a non-meaningful period."""
        import random
        random.seed(42)
        values = [random.gauss(0, 1) for _ in range(200)]
        period = engine.get_dominant_period(values)
        # Could be 0 or some spurious lag; just check it does not crash
        assert isinstance(period, int)
        assert period >= 0


# ===========================================================================
# Tests: classify_interval
# ===========================================================================


class TestClassifyInterval:
    """Tests for interval seconds to FrequencyLevel mapping."""

    def test_sub_hourly(self, engine):
        assert engine.classify_interval(600.0) == FrequencyLevel.SUB_HOURLY

    def test_hourly(self, engine):
        assert engine.classify_interval(3600.0) == FrequencyLevel.HOURLY

    def test_daily(self, engine):
        assert engine.classify_interval(86400.0) == FrequencyLevel.DAILY

    def test_weekly(self, engine):
        assert engine.classify_interval(604800.0) == FrequencyLevel.WEEKLY

    def test_monthly(self, engine):
        assert engine.classify_interval(2592000.0) == FrequencyLevel.MONTHLY

    def test_quarterly(self, engine):
        assert engine.classify_interval(7776000.0) == FrequencyLevel.QUARTERLY

    def test_annual(self, engine):
        assert engine.classify_interval(31536000.0) == FrequencyLevel.ANNUAL

    def test_zero_returns_irregular(self, engine):
        assert engine.classify_interval(0.0) == FrequencyLevel.IRREGULAR

    def test_negative_returns_irregular(self, engine):
        assert engine.classify_interval(-100.0) == FrequencyLevel.IRREGULAR


# ===========================================================================
# Tests: compute_interval_statistics
# ===========================================================================


class TestComputeIntervalStatistics:
    """Tests for descriptive statistics on inter-observation intervals."""

    def test_daily_mean(self, engine, daily_timestamps):
        stats = engine.compute_interval_statistics(daily_timestamps)
        assert stats["mean_interval"] == pytest.approx(86400.0, rel=0.01)

    def test_daily_median(self, engine, daily_timestamps):
        stats = engine.compute_interval_statistics(daily_timestamps)
        assert stats["median_interval"] == pytest.approx(86400.0, rel=0.01)

    def test_daily_std_near_zero(self, engine, daily_timestamps):
        stats = engine.compute_interval_statistics(daily_timestamps)
        assert stats["std_interval"] < 1.0

    def test_count_equals_intervals(self, engine, daily_timestamps):
        stats = engine.compute_interval_statistics(daily_timestamps)
        assert stats["count"] == len(daily_timestamps) - 1

    def test_min_max_equal_for_regular(self, engine, daily_timestamps):
        stats = engine.compute_interval_statistics(daily_timestamps)
        assert stats["min_interval"] == pytest.approx(stats["max_interval"], rel=0.001)

    def test_cv_near_zero_for_regular(self, engine, daily_timestamps):
        stats = engine.compute_interval_statistics(daily_timestamps)
        assert stats["cv"] < 0.01

    def test_provenance_hash_present(self, engine, daily_timestamps):
        stats = engine.compute_interval_statistics(daily_timestamps)
        assert len(stats["provenance_hash"]) == 64

    def test_single_timestamp_empty_stats(self, engine):
        ts = [datetime(2025, 1, 1, tzinfo=timezone.utc)]
        stats = engine.compute_interval_statistics(ts)
        assert stats["count"] == 0
        assert stats["mean_interval"] == 0.0

    def test_empty_timestamps_empty_stats(self, engine):
        stats = engine.compute_interval_statistics([])
        assert stats["count"] == 0


# ===========================================================================
# Tests: detect_mixed_frequency
# ===========================================================================


class TestDetectMixedFrequency:
    """Tests for mixed frequency detection."""

    def test_uniform_daily_not_mixed(self, engine, daily_timestamps):
        result = engine.detect_mixed_frequency(daily_timestamps)
        assert result["is_mixed"] is False
        assert len(result["frequencies_found"]) == 1

    def test_mixed_daily_and_weekly(self, engine):
        """Series with some daily and some weekly gaps."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # 5 daily then jump by a week, then 5 daily again
        ts = [base + timedelta(days=i) for i in range(5)]
        ts.append(base + timedelta(days=12))
        ts.extend([base + timedelta(days=12 + i) for i in range(1, 5)])
        result = engine.detect_mixed_frequency(ts)
        assert result["is_mixed"] is True
        assert len(result["frequencies_found"]) >= 2

    def test_single_timestamp_not_mixed(self, engine):
        ts = [datetime(2025, 1, 1, tzinfo=timezone.utc)]
        result = engine.detect_mixed_frequency(ts)
        assert result["is_mixed"] is False
        assert result["total_intervals"] == 0

    def test_dominant_frequency_reported(self, engine, daily_timestamps):
        result = engine.detect_mixed_frequency(daily_timestamps)
        assert result["dominant_frequency"] == FrequencyLevel.DAILY.value

    def test_proportions_sum_to_one(self, engine, daily_timestamps):
        result = engine.detect_mixed_frequency(daily_timestamps)
        total = sum(result["proportions"].values())
        assert total == pytest.approx(1.0, abs=0.001)


# ===========================================================================
# Tests: validate_frequency
# ===========================================================================


class TestValidateFrequency:
    """Tests for frequency validation against expected level."""

    def test_daily_validates_as_daily(self, engine, daily_timestamps):
        result = engine.validate_frequency(daily_timestamps, "daily")
        assert result["is_valid"] is True
        assert result["match_pct"] >= 0.8

    def test_daily_fails_hourly_validation(self, engine, daily_timestamps):
        result = engine.validate_frequency(daily_timestamps, "hourly")
        assert result["is_valid"] is False

    def test_invalid_frequency_raises(self, engine, daily_timestamps):
        with pytest.raises(ValueError, match="Unknown frequency level"):
            engine.validate_frequency(daily_timestamps, "bicentennial")

    def test_insufficient_data(self, engine):
        ts = [datetime(2025, 1, 1, tzinfo=timezone.utc)]
        result = engine.validate_frequency(ts, "daily")
        assert result["is_valid"] is False
        assert result["total_intervals"] == 0
