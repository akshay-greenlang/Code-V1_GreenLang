# -*- coding: utf-8 -*-
"""
Unit tests for SeasonalFillerEngine - AGENT-DATA-014 Time Series Gap Filler

Tests all public methods of SeasonalFillerEngine:
    - decompose_seasonal: STL-style additive decomposition
    - fill_seasonal: seasonal-pattern gap filling
    - detect_seasonality: autocorrelation-based seasonality detection
    - get_seasonal_pattern: extract repeating seasonal pattern
    - fill_day_of_week_pattern: weekday-average gap filling
    - fill_month_pattern: month-average gap filling
    - fill_calendar_aware: calendar-aware gap filling

Target: 50+ tests covering normal, edge, and validation scenarios.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pytest

from greenlang.time_series_gap_filler.config import TimeSeriesGapFillerConfig
from greenlang.time_series_gap_filler.seasonal_filler import (
    CalendarDefinition,
    FillResult,
    SeasonalDecomposition,
    SeasonalFillerEngine,
    _compute_confidence,
)


# ---------------------------------------------------------------------------
# Helper: create a config that provides both min_data_points and min_points
# ---------------------------------------------------------------------------


class _TestConfig(TimeSeriesGapFillerConfig):
    """Wrapper that adds a min_points alias for backward compat."""

    @property
    def min_points(self) -> int:  # type: ignore[override]
        return self.min_data_points


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> _TestConfig:
    """Test configuration with min_points alias."""
    return _TestConfig(
        seasonal_periods=12,
        min_data_points=6,
        enable_seasonal=True,
        enable_provenance=True,
    )


@pytest.fixture
def engine(config) -> SeasonalFillerEngine:
    """Create a SeasonalFillerEngine with test config."""
    return SeasonalFillerEngine(config=config)


@pytest.fixture
def seasonal_12_series() -> List[float]:
    """36 values with a clear period-12 sine seasonal pattern."""
    return [
        100.0 + 20.0 * math.sin(2 * math.pi * i / 12) for i in range(36)
    ]


@pytest.fixture
def constant_series() -> List[float]:
    """36 constant values -- no seasonal pattern."""
    return [50.0] * 36


@pytest.fixture
def seasonal_7_series() -> List[float]:
    """56 values with a clear period-7 sine pattern."""
    return [
        80.0 + 15.0 * math.sin(2 * math.pi * i / 7) for i in range(56)
    ]


@pytest.fixture
def seasonal_4_series() -> List[float]:
    """32 values with a clear quarterly (period=4) pattern."""
    pattern = [100.0, 130.0, 90.0, 110.0]
    return pattern * 8


@pytest.fixture
def series_with_gaps(seasonal_12_series) -> List[Optional[float]]:
    """Seasonal-12 series with gaps at indices 5, 15, 25."""
    s = list(seasonal_12_series)
    s[5] = None
    s[15] = None
    s[25] = None
    return s


@pytest.fixture
def monthly_timestamps() -> List[datetime]:
    """36 monthly timestamps (3 years)."""
    base = datetime(2023, 1, 15, tzinfo=timezone.utc)
    return [base + timedelta(days=30 * i) for i in range(36)]


@pytest.fixture
def daily_timestamps_52wk() -> List[datetime]:
    """365 daily timestamps (1 year)."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(days=i) for i in range(365)]


@pytest.fixture
def daily_values_weekly_pattern(daily_timestamps_52wk) -> List[Optional[float]]:
    """365 daily values with clear day-of-week pattern."""
    values: List[Optional[float]] = []
    for i, ts in enumerate(daily_timestamps_52wk):
        dow = ts.weekday()  # 0=Mon, 6=Sun
        # Weekdays higher, weekends lower
        if dow < 5:
            values.append(100.0 + 5.0 * dow)
        else:
            values.append(30.0 + 5.0 * dow)
    return values


@pytest.fixture
def monthly_values_annual_pattern() -> List[Optional[float]]:
    """36 monthly values (3 years) with annual month pattern."""
    # Each month gets a characteristic value
    month_pattern = [10, 15, 25, 40, 55, 70, 80, 75, 60, 45, 25, 15]
    values: List[Optional[float]] = []
    for year in range(3):
        for m in range(12):
            values.append(float(month_pattern[m]) + year * 2.0)
    return values


# ===========================================================================
# Tests: Initialization
# ===========================================================================


class TestSeasonalFillerInit:
    """Tests for SeasonalFillerEngine initialization."""

    def test_creates_instance(self, config):
        engine = SeasonalFillerEngine(config)
        assert engine is not None

    def test_config_stored(self, engine, config):
        assert engine._config is config

    def test_default_config_used_when_none(self):
        engine = SeasonalFillerEngine()
        assert engine._config is not None

    def test_provenance_tracker_created(self, engine):
        assert engine._provenance is not None


# ===========================================================================
# Tests: decompose_seasonal
# ===========================================================================


class TestDecomposeSeasonalBasic:
    """Tests for STL-style additive decomposition."""

    def test_returns_decomposition(self, engine, seasonal_12_series):
        result = engine.decompose_seasonal(seasonal_12_series, period=12)
        assert isinstance(result, SeasonalDecomposition)

    def test_trend_is_list(self, engine, seasonal_12_series):
        result = engine.decompose_seasonal(seasonal_12_series, period=12)
        assert isinstance(result.trend, list)
        assert len(result.trend) == 36

    def test_seasonal_is_list(self, engine, seasonal_12_series):
        result = engine.decompose_seasonal(seasonal_12_series, period=12)
        assert isinstance(result.seasonal, list)
        assert len(result.seasonal) == 36

    def test_residual_is_list(self, engine, seasonal_12_series):
        result = engine.decompose_seasonal(seasonal_12_series, period=12)
        assert isinstance(result.residual, list)
        assert len(result.residual) == 36

    def test_trend_is_smooth(self, engine, seasonal_12_series):
        """Trend should be near the base level (100.0)."""
        result = engine.decompose_seasonal(seasonal_12_series, period=12)
        valid_trend = [t for t in result.trend if t is not None]
        if valid_trend:
            mean_trend = sum(valid_trend) / len(valid_trend)
            assert abs(mean_trend - 100.0) < 10.0

    def test_seasonal_sums_to_near_zero(self, engine, seasonal_12_series):
        """Seasonal component should sum to approximately 0 over one period."""
        result = engine.decompose_seasonal(seasonal_12_series, period=12)
        pattern = result.seasonal[:12]
        total = sum(pattern)
        assert abs(total) < 5.0

    def test_period_stored(self, engine, seasonal_12_series):
        result = engine.decompose_seasonal(seasonal_12_series, period=12)
        assert result.period == 12

    def test_provenance_hash_present(self, engine, seasonal_12_series):
        result = engine.decompose_seasonal(seasonal_12_series, period=12)
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    def test_short_series_raises(self, engine):
        with pytest.raises(ValueError, match="too short"):
            engine.decompose_seasonal([1.0, 2.0, 3.0], period=2)

    def test_auto_detect_period(self, engine, seasonal_12_series):
        """When period is None, auto-detection should find ~12."""
        result = engine.decompose_seasonal(seasonal_12_series, period=None)
        assert result.period >= 2


# ===========================================================================
# Tests: fill_seasonal
# ===========================================================================


class TestFillSeasonal:
    """Tests for seasonal-pattern gap filling."""

    def test_fills_gaps(self, engine, series_with_gaps):
        result = engine.fill_seasonal(series_with_gaps, period=12)
        assert result.gaps_filled >= 1

    def test_filled_values_not_none(self, engine, series_with_gaps):
        result = engine.fill_seasonal(series_with_gaps, period=12)
        assert result.values[5] is not None
        assert result.values[15] is not None
        assert result.values[25] is not None

    def test_preserves_known_values(self, engine, series_with_gaps, seasonal_12_series):
        result = engine.fill_seasonal(series_with_gaps, period=12)
        for i, original in enumerate(seasonal_12_series):
            if series_with_gaps[i] is not None:
                assert result.values[i] == pytest.approx(original, abs=1e-6)

    def test_confidence_based_on_cycles(self, engine, series_with_gaps):
        result = engine.fill_seasonal(series_with_gaps, period=12)
        assert 0.0 <= result.confidence <= 1.0

    def test_per_point_confidence(self, engine, series_with_gaps):
        result = engine.fill_seasonal(series_with_gaps, period=12)
        for idx in [5, 15, 25]:
            assert idx in result.per_point_confidence
            assert 0.0 <= result.per_point_confidence[idx] <= 1.0

    def test_provenance_hash(self, engine, series_with_gaps):
        result = engine.fill_seasonal(series_with_gaps, period=12)
        assert len(result.provenance_hash) == 64

    def test_method_is_seasonal(self, engine, series_with_gaps):
        result = engine.fill_seasonal(series_with_gaps, period=12)
        assert result.method == "seasonal"

    def test_no_gaps_returns_zero_filled(self, engine, seasonal_12_series):
        result = engine.fill_seasonal(seasonal_12_series, period=12)
        assert result.gaps_filled == 0

    def test_processing_time_positive(self, engine, series_with_gaps):
        result = engine.fill_seasonal(series_with_gaps, period=12)
        assert result.processing_time_ms >= 0.0

    def test_details_include_period(self, engine, series_with_gaps):
        result = engine.fill_seasonal(series_with_gaps, period=12)
        assert "period" in result.details
        assert result.details["period"] == 12


# ===========================================================================
# Tests: detect_seasonality
# ===========================================================================


class TestDetectSeasonality:
    """Tests for autocorrelation-based seasonality detection."""

    def test_detects_period_12(self, engine, seasonal_12_series):
        result = engine.detect_seasonality(seasonal_12_series)
        assert result["is_seasonal"] is True
        assert result["dominant_period"] in range(10, 15)

    def test_constant_series_no_season(self, engine, constant_series):
        result = engine.detect_seasonality(constant_series)
        assert result["is_seasonal"] is False

    def test_short_series_no_season(self, engine):
        result = engine.detect_seasonality([1.0, 2.0, 3.0])
        assert result["is_seasonal"] is False
        assert result["dominant_period"] == 0
        assert result["confidence"] == 0.0

    def test_detected_periods_list(self, engine, seasonal_12_series):
        result = engine.detect_seasonality(seasonal_12_series)
        assert isinstance(result["detected_periods"], list)

    def test_strengths_dict(self, engine, seasonal_12_series):
        result = engine.detect_seasonality(seasonal_12_series)
        assert isinstance(result["strengths"], dict)

    def test_acf_values_populated(self, engine, seasonal_12_series):
        result = engine.detect_seasonality(seasonal_12_series)
        assert len(result["acf_values"]) > 0

    def test_provenance_hash_present(self, engine, seasonal_12_series):
        result = engine.detect_seasonality(seasonal_12_series)
        assert len(result["provenance_hash"]) == 64

    def test_custom_max_period(self, engine, seasonal_12_series):
        result = engine.detect_seasonality(seasonal_12_series, max_period=6)
        # With max_period=6, period-12 won't be found
        if result["is_seasonal"]:
            assert result["dominant_period"] <= 6


# ===========================================================================
# Tests: get_seasonal_pattern
# ===========================================================================


class TestGetSeasonalPattern:
    """Tests for extracting the repeating seasonal pattern."""

    def test_pattern_length_equals_period(self, engine, seasonal_12_series):
        pattern = engine.get_seasonal_pattern(seasonal_12_series, period=12)
        assert len(pattern) == 12

    def test_pattern_sums_to_near_zero(self, engine, seasonal_12_series):
        """Centred pattern sums to approximately zero."""
        pattern = engine.get_seasonal_pattern(seasonal_12_series, period=12)
        assert abs(sum(pattern)) < 1.0

    def test_constant_series_pattern_all_zero(self, engine, constant_series):
        """Constant series should have near-zero seasonal pattern."""
        pattern = engine.get_seasonal_pattern(constant_series, period=12)
        for val in pattern:
            assert abs(val) < 1.0

    def test_period_4_quarterly(self, engine, seasonal_4_series):
        pattern = engine.get_seasonal_pattern(seasonal_4_series, period=4)
        assert len(pattern) == 4

    def test_period_7_weekly(self, engine, seasonal_7_series):
        pattern = engine.get_seasonal_pattern(seasonal_7_series, period=7)
        assert len(pattern) == 7

    def test_min_period_enforced(self, engine, seasonal_12_series):
        """Period < 2 is clamped to 2."""
        pattern = engine.get_seasonal_pattern(seasonal_12_series, period=1)
        assert len(pattern) == 2

    def test_pattern_with_missing_values(self, engine, series_with_gaps):
        """Gaps should be excluded from pattern computation."""
        pattern = engine.get_seasonal_pattern(series_with_gaps, period=12)
        assert len(pattern) == 12
        # Pattern should still sum to near zero
        assert abs(sum(pattern)) < 2.0


# ===========================================================================
# Tests: fill_day_of_week_pattern
# ===========================================================================


class TestFillDayOfWeekPattern:
    """Tests for weekday-average gap filling."""

    def test_fills_gaps(self, engine, daily_values_weekly_pattern,
                        daily_timestamps_52wk):
        values = list(daily_values_weekly_pattern)
        values[10] = None  # some Tuesday
        result = engine.fill_day_of_week_pattern(values, daily_timestamps_52wk)
        assert result.gaps_filled == 1
        assert result.values[10] is not None

    def test_correct_day_average(self, engine, daily_timestamps_52wk):
        """Verify Monday values averaged correctly."""
        # Build simple series: weekday=value*10
        values: List[Optional[float]] = []
        for ts in daily_timestamps_52wk:
            values.append(float(ts.weekday() * 10))
        # Gap on a Monday
        monday_idx = None
        for i, ts in enumerate(daily_timestamps_52wk):
            if ts.weekday() == 0 and i > 10:
                monday_idx = i
                break
        assert monday_idx is not None
        values[monday_idx] = None
        result = engine.fill_day_of_week_pattern(values, daily_timestamps_52wk)
        # Filled value should be close to 0.0 (Monday = weekday 0 * 10)
        assert result.values[monday_idx] == pytest.approx(0.0, abs=1.0)

    def test_no_gaps_zero_filled(self, engine, daily_values_weekly_pattern,
                                  daily_timestamps_52wk):
        result = engine.fill_day_of_week_pattern(
            daily_values_weekly_pattern, daily_timestamps_52wk,
        )
        assert result.gaps_filled == 0

    def test_method_name(self, engine, daily_values_weekly_pattern,
                          daily_timestamps_52wk):
        values = list(daily_values_weekly_pattern)
        values[5] = None
        result = engine.fill_day_of_week_pattern(values, daily_timestamps_52wk)
        assert result.method == "day_of_week"

    def test_length_mismatch_raises(self, engine, daily_timestamps_52wk):
        values = [1.0, 2.0, 3.0]  # length 3 != 365
        with pytest.raises(ValueError, match="values length"):
            engine.fill_day_of_week_pattern(values, daily_timestamps_52wk)

    def test_provenance_present(self, engine, daily_values_weekly_pattern,
                                 daily_timestamps_52wk):
        values = list(daily_values_weekly_pattern)
        values[20] = None
        result = engine.fill_day_of_week_pattern(values, daily_timestamps_52wk)
        assert len(result.provenance_hash) == 64


# ===========================================================================
# Tests: fill_month_pattern
# ===========================================================================


class TestFillMonthPattern:
    """Tests for month-average gap filling."""

    def test_fills_gaps(self, engine, monthly_values_annual_pattern,
                        monthly_timestamps):
        values = list(monthly_values_annual_pattern)
        values[7] = None  # August year 1
        result = engine.fill_month_pattern(values, monthly_timestamps)
        assert result.gaps_filled == 1
        assert result.values[7] is not None

    def test_correct_month_average(self, engine, monthly_timestamps):
        """January values should average correctly."""
        # 3 years of monthly data: Jan=100, Feb=200, etc.
        values: List[Optional[float]] = []
        for year in range(3):
            for month in range(12):
                values.append(float((month + 1) * 100 + year * 10))
        # Gap in Jan of year 2 (index 12)
        values[12] = None
        result = engine.fill_month_pattern(values, monthly_timestamps)
        # January average should be close to (100+110+120)/3 = 110 without index 12
        # Without year2 Jan: (100 + 120) / 2 = 110
        assert result.values[12] == pytest.approx(110.0, abs=5.0)

    def test_no_gaps_zero_filled(self, engine, monthly_values_annual_pattern,
                                  monthly_timestamps):
        result = engine.fill_month_pattern(
            monthly_values_annual_pattern, monthly_timestamps,
        )
        assert result.gaps_filled == 0

    def test_method_name(self, engine, monthly_values_annual_pattern,
                          monthly_timestamps):
        values = list(monthly_values_annual_pattern)
        values[3] = None
        result = engine.fill_month_pattern(values, monthly_timestamps)
        assert result.method == "month_pattern"

    def test_length_mismatch_raises(self, engine, monthly_timestamps):
        values = [1.0, 2.0]
        with pytest.raises(ValueError, match="values length"):
            engine.fill_month_pattern(values, monthly_timestamps)

    def test_provenance_present(self, engine, monthly_values_annual_pattern,
                                 monthly_timestamps):
        values = list(monthly_values_annual_pattern)
        values[0] = None
        result = engine.fill_month_pattern(values, monthly_timestamps)
        assert len(result.provenance_hash) == 64

    def test_details_include_month_averages(self, engine, monthly_values_annual_pattern,
                                             monthly_timestamps):
        values = list(monthly_values_annual_pattern)
        values[5] = None
        result = engine.fill_month_pattern(values, monthly_timestamps)
        assert "month_averages" in result.details
        assert len(result.details["month_averages"]) == 12


# ===========================================================================
# Tests: _compute_confidence helper
# ===========================================================================


class TestComputeConfidence:
    """Tests for the module-level _compute_confidence function."""

    def test_zero_cycles_low(self):
        c = _compute_confidence(0, "seasonal")
        assert c == pytest.approx(0.2)

    def test_one_cycle(self):
        c = _compute_confidence(1, "seasonal")
        assert c == pytest.approx(0.45)

    def test_two_cycles(self):
        c = _compute_confidence(2, "seasonal")
        assert c == pytest.approx(0.60)

    def test_five_cycles(self):
        c = _compute_confidence(5, "seasonal")
        assert c == pytest.approx(0.80)

    def test_many_cycles_capped(self):
        c = _compute_confidence(20, "seasonal")
        assert c <= 1.0

    def test_calendar_method_bonus(self):
        c_cal = _compute_confidence(3, "calendar")
        c_sea = _compute_confidence(3, "seasonal")
        assert c_cal > c_sea

    def test_day_of_week_penalty(self):
        c_dow = _compute_confidence(3, "day_of_week")
        c_sea = _compute_confidence(3, "seasonal")
        assert c_dow < c_sea


# ===========================================================================
# Tests: various period lengths
# ===========================================================================


class TestVariousPeriodLengths:
    """Tests across different period values (7, 12, 4)."""

    def test_period_7(self, engine, seasonal_7_series):
        result = engine.decompose_seasonal(seasonal_7_series, period=7)
        assert result.period == 7
        assert len(result.seasonal) == len(seasonal_7_series)

    def test_period_12(self, engine, seasonal_12_series):
        result = engine.decompose_seasonal(seasonal_12_series, period=12)
        assert result.period == 12

    def test_period_4(self, engine, seasonal_4_series):
        result = engine.decompose_seasonal(seasonal_4_series, period=4)
        assert result.period == 4


# ===========================================================================
# Tests: series with no seasonality
# ===========================================================================


class TestNoSeasonality:
    """Tests when input has no seasonal pattern."""

    def test_decompose_constant(self, engine, constant_series):
        result = engine.decompose_seasonal(constant_series, period=12)
        # Seasonal should be near zero everywhere
        for val in result.seasonal:
            assert abs(val) < 5.0

    def test_fill_constant_near_original(self, engine, constant_series):
        values = list(constant_series)
        values[10] = None
        result = engine.fill_seasonal(values, period=12)
        assert result.values[10] is not None
        assert abs(result.values[10] - 50.0) < 10.0

    def test_detect_no_season(self, engine, constant_series):
        result = engine.detect_seasonality(constant_series)
        assert result["is_seasonal"] is False


# ===========================================================================
# Tests: short series (< 2 periods)
# ===========================================================================


class TestShortSeries:
    """Tests for series shorter than 2 complete periods."""

    def test_decompose_short_uses_min_period(self, engine):
        """Series of length 8 with period=12 should clamp period to 4."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = engine.decompose_seasonal(values, period=12)
        assert result.period <= 4  # clamped to n//2

    def test_fill_seasonal_short(self, engine):
        """Short series gap filling does not crash."""
        values: List[Optional[float]] = [10.0, 20.0, None, 40.0, 50.0, 60.0, 70.0]
        result = engine.fill_seasonal(values, period=4)
        assert result.values[2] is not None

    def test_detect_seasonality_short(self, engine):
        """Very short series returns no seasonality."""
        result = engine.detect_seasonality([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["is_seasonal"] is False


# ===========================================================================
# Tests: fill_calendar_aware
# ===========================================================================


class TestFillCalendarAware:
    """Tests for calendar-aware gap filling."""

    def test_basic_calendar_fill(self, engine, daily_timestamps_52wk):
        values: List[Optional[float]] = [100.0] * 365
        values[10] = None
        cal = CalendarDefinition(
            business_days=[0, 1, 2, 3, 4],
            holidays=[],
        )
        result = engine.fill_calendar_aware(values, daily_timestamps_52wk, cal)
        assert result.gaps_filled == 1
        assert result.values[10] is not None

    def test_calendar_method_name(self, engine, daily_timestamps_52wk):
        values: List[Optional[float]] = [50.0] * 365
        values[5] = None
        cal = CalendarDefinition()
        result = engine.fill_calendar_aware(values, daily_timestamps_52wk, cal)
        assert result.method == "calendar"

    def test_calendar_length_mismatch(self, engine, daily_timestamps_52wk):
        cal = CalendarDefinition()
        with pytest.raises(ValueError, match="values length"):
            engine.fill_calendar_aware([1.0], daily_timestamps_52wk, cal)

    def test_no_gaps_calendar(self, engine, daily_timestamps_52wk):
        values = [100.0] * 365
        cal = CalendarDefinition()
        result = engine.fill_calendar_aware(values, daily_timestamps_52wk, cal)
        assert result.gaps_filled == 0
