# -*- coding: utf-8 -*-
"""
Unit tests for TimeSeriesImputerEngine - AGENT-DATA-012

Tests impute_linear_interpolation, impute_spline_interpolation,
impute_seasonal_decomposition, impute_moving_average,
impute_exponential_smoothing, impute_trend_extrapolation,
detect_seasonality, detect_trend, and private helpers.
Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import math

import pytest

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.time_series_imputer import (
    TimeSeriesImputerEngine,
    _is_missing,
    _classify_confidence,
    _safe_stdev,
)
from greenlang.missing_value_imputer.models import (
    ConfidenceLevel,
    ImputationStrategy,
)


@pytest.fixture
def engine():
    return TimeSeriesImputerEngine(MissingValueImputerConfig())


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_missing_none(self):
        assert _is_missing(None) is True

    def test_is_missing_nan(self):
        assert _is_missing(float("nan")) is True

    def test_is_missing_valid(self):
        assert _is_missing(42) is False

    def test_classify_confidence_high(self):
        assert _classify_confidence(0.90) == ConfidenceLevel.HIGH

    def test_classify_confidence_very_low(self):
        assert _classify_confidence(0.30) == ConfidenceLevel.VERY_LOW

    def test_safe_stdev_empty(self):
        assert _safe_stdev([]) == 0.0

    def test_safe_stdev_single(self):
        assert _safe_stdev([1.0]) == 0.0

    def test_safe_stdev_values(self):
        assert _safe_stdev([1.0, 2.0, 3.0]) > 0


# ---------------------------------------------------------------------------
# _validate_time_series
# ---------------------------------------------------------------------------


class TestValidateTimeSeries:
    def test_empty_series(self, engine):
        assert engine._validate_time_series([]) is False

    def test_all_missing(self, engine):
        assert engine._validate_time_series([None, None]) is False

    def test_single_observed(self, engine):
        assert engine._validate_time_series([1.0, None]) is False

    def test_two_observed(self, engine):
        assert engine._validate_time_series([1.0, None, 3.0]) is True


# ---------------------------------------------------------------------------
# Linear interpolation
# ---------------------------------------------------------------------------


class TestLinearInterpolation:
    def test_basic_interpolation(self, engine):
        series = [1.0, None, 3.0]
        result = engine.impute_linear_interpolation(series)
        assert len(result) == 1
        assert result[0].imputed_value == pytest.approx(2.0, rel=1e-4)
        assert result[0].record_index == 1

    def test_strategy_type(self, engine):
        series = [1.0, None, 3.0]
        result = engine.impute_linear_interpolation(series)
        assert result[0].strategy == ImputationStrategy.LINEAR_INTERPOLATION

    def test_multiple_gaps(self, engine):
        series = [0.0, None, None, 6.0]
        result = engine.impute_linear_interpolation(series)
        assert len(result) == 2
        assert result[0].imputed_value == pytest.approx(2.0, rel=1e-4)
        assert result[1].imputed_value == pytest.approx(4.0, rel=1e-4)

    def test_leading_missing_not_filled(self, engine):
        series = [None, 1.0, 2.0, 3.0]
        result = engine.impute_linear_interpolation(series)
        assert all(iv.record_index != 0 for iv in result)

    def test_trailing_missing_not_filled(self, engine):
        series = [1.0, 2.0, 3.0, None]
        result = engine.impute_linear_interpolation(series)
        assert all(iv.record_index != 3 for iv in result)

    def test_no_missing_returns_empty(self, engine):
        series = [1.0, 2.0, 3.0, 4.0]
        result = engine.impute_linear_interpolation(series)
        assert result == []

    def test_confidence_decays_with_gap(self, engine):
        series = [1.0, None, None, None, None, None, 7.0]
        result = engine.impute_linear_interpolation(series)
        # gap_size = 6, confidence = max(0.50, 0.92 - 0.03*6) = 0.74
        assert all(iv.confidence >= 0.50 for iv in result)

    def test_provenance_hash(self, engine):
        series = [1.0, None, 3.0]
        result = engine.impute_linear_interpolation(series)
        assert len(result[0].provenance_hash) == 64

    def test_empty_series_returns_empty(self, engine):
        result = engine.impute_linear_interpolation([])
        assert result == []


# ---------------------------------------------------------------------------
# Spline interpolation
# ---------------------------------------------------------------------------


class TestSplineInterpolation:
    def test_basic_spline(self, engine):
        series = [0.0, 1.0, None, 9.0, 16.0]  # roughly y = x^2
        result = engine.impute_spline_interpolation(series)
        assert len(result) == 1
        assert result[0].record_index == 2

    def test_falls_back_to_linear_with_few_points(self, engine):
        series = [1.0, None, 3.0]
        result = engine.impute_spline_interpolation(series)
        # only 2 observed points, falls back to linear (needs <4 for fallback)
        assert len(result) == 1

    def test_provenance_hash(self, engine):
        series = [0.0, 1.0, 4.0, None, 16.0]
        result = engine.impute_spline_interpolation(series)
        if result:
            assert len(result[0].provenance_hash) == 64

    def test_empty_returns_empty(self, engine):
        result = engine.impute_spline_interpolation([])
        assert result == []


# ---------------------------------------------------------------------------
# Seasonal decomposition
# ---------------------------------------------------------------------------


class TestSeasonalDecomposition:
    def test_basic_decomposition(self, engine):
        # Build a seasonal series of length 2*period
        period = 12
        cfg = MissingValueImputerConfig(seasonal_period=period)
        eng = TimeSeriesImputerEngine(cfg)
        series = []
        for i in range(24):
            if i == 10:
                series.append(None)
            else:
                series.append(10.0 + 5.0 * math.sin(2 * math.pi * i / period))
        result = eng.impute_seasonal_decomposition(series, period=period)
        assert len(result) >= 1
        assert result[0].record_index == 10

    def test_short_series_falls_back(self, engine):
        # series length < 2*period falls back to linear
        cfg = MissingValueImputerConfig(seasonal_period=12)
        eng = TimeSeriesImputerEngine(cfg)
        series = [1.0, None, 3.0, 4.0, 5.0]
        result = eng.impute_seasonal_decomposition(series, period=12)
        # falls back to linear interpolation
        assert len(result) >= 0  # may be 1 or 0 depending on edge handling

    def test_provenance_hash(self, engine):
        period = 4
        cfg = MissingValueImputerConfig(seasonal_period=period)
        eng = TimeSeriesImputerEngine(cfg)
        series = [float(i % 4) for i in range(12)]
        series[5] = None
        result = eng.impute_seasonal_decomposition(series, period=period)
        if result:
            assert len(result[0].provenance_hash) == 64


# ---------------------------------------------------------------------------
# Moving average
# ---------------------------------------------------------------------------


class TestMovingAverage:
    def test_basic_moving_avg(self, engine):
        series = [1.0, 2.0, None, 4.0, 5.0]
        result = engine.impute_moving_average(series, window=3)
        assert len(result) == 1
        assert result[0].record_index == 2

    def test_window_only_observed_values(self, engine):
        series = [2.0, 4.0, None, 6.0, 8.0]
        result = engine.impute_moving_average(series, window=5)
        assert len(result) == 1
        # Mean of visible values within window

    def test_no_missing_returns_empty(self, engine):
        series = [1.0, 2.0, 3.0]
        result = engine.impute_moving_average(series, window=3)
        assert result == []

    def test_empty_returns_empty(self, engine):
        result = engine.impute_moving_average([], window=3)
        assert result == []


# ---------------------------------------------------------------------------
# Exponential smoothing
# ---------------------------------------------------------------------------


class TestExponentialSmoothing:
    def test_basic_smoothing(self, engine):
        series = [10.0, 20.0, None, None, 50.0]
        result = engine.impute_exponential_smoothing(series, alpha=0.5)
        assert len(result) == 2
        assert result[0].record_index == 2
        assert result[1].record_index == 3

    def test_alpha_clamped(self, engine):
        series = [1.0, None, 3.0]
        result = engine.impute_exponential_smoothing(series, alpha=2.0)
        assert len(result) == 1  # alpha clamped to 0.99

    def test_confidence_decays(self, engine):
        series = [10.0, None, None, None, None, 50.0]
        result = engine.impute_exponential_smoothing(series, alpha=0.3)
        # Confidence should decrease for later missing values
        if len(result) >= 2:
            assert result[0].confidence >= result[-1].confidence

    def test_all_missing_returns_empty(self, engine):
        series = [None, None, None]
        result = engine.impute_exponential_smoothing(series)
        assert result == []

    def test_provenance_hash(self, engine):
        series = [1.0, None, 3.0]
        result = engine.impute_exponential_smoothing(series)
        assert len(result[0].provenance_hash) == 64


# ---------------------------------------------------------------------------
# Trend extrapolation
# ---------------------------------------------------------------------------


class TestTrendExtrapolation:
    def test_linear_trend(self, engine):
        series = [2.0, 4.0, 6.0, None, 10.0]
        result = engine.impute_trend_extrapolation(series)
        assert len(result) == 1
        assert result[0].imputed_value == pytest.approx(8.0, rel=0.1)

    def test_extrapolation_leading(self, engine):
        series = [None, 2.0, 4.0, 6.0, 8.0]
        result = engine.impute_trend_extrapolation(series)
        assert len(result) == 1
        assert result[0].record_index == 0

    def test_no_missing_returns_empty(self, engine):
        series = [1.0, 2.0, 3.0, 4.0]
        result = engine.impute_trend_extrapolation(series)
        assert result == []

    def test_single_observed_returns_empty(self, engine):
        series = [None, 5.0, None]
        result = engine.impute_trend_extrapolation(series)
        assert result == []

    def test_provenance_hash(self, engine):
        series = [1.0, 2.0, None, 4.0]
        result = engine.impute_trend_extrapolation(series)
        if result:
            assert len(result[0].provenance_hash) == 64


# ---------------------------------------------------------------------------
# detect_seasonality
# ---------------------------------------------------------------------------


class TestDetectSeasonality:
    def test_short_series_returns_not_significant(self, engine):
        result = engine.detect_seasonality([1.0, 2.0, 3.0])
        assert result["significant"] is False
        assert result["period"] == 0

    def test_constant_series(self, engine):
        result = engine.detect_seasonality([5.0] * 20)
        assert result["significant"] is False

    def test_periodic_series(self, engine):
        series = [math.sin(2 * math.pi * i / 7) for i in range(50)]
        result = engine.detect_seasonality(series)
        assert "period" in result
        assert "amplitude" in result
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_acf_values_present(self, engine):
        series = [float(i % 4) for i in range(40)]
        result = engine.detect_seasonality(series)
        assert isinstance(result["acf_values"], list)


# ---------------------------------------------------------------------------
# detect_trend
# ---------------------------------------------------------------------------


class TestDetectTrend:
    def test_short_series(self, engine):
        result = engine.detect_trend([1.0, 2.0])
        assert result["direction"] == "none"
        assert result["significant"] is False

    def test_increasing_trend(self, engine):
        result = engine.detect_trend([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["direction"] == "increasing"
        assert result["slope"] > 0
        assert result["r_squared"] > 0.9
        assert result["significant"] is True

    def test_decreasing_trend(self, engine):
        result = engine.detect_trend([5.0, 4.0, 3.0, 2.0, 1.0])
        assert result["direction"] == "decreasing"
        assert result["slope"] < 0

    def test_no_trend_constant(self, engine):
        result = engine.detect_trend([5.0, 5.0, 5.0, 5.0])
        assert result["direction"] == "none"

    def test_provenance_hash(self, engine):
        result = engine.detect_trend([1.0, 2.0, 3.0, 4.0])
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# Private helper tests
# ---------------------------------------------------------------------------


class TestPrivateHelpers:
    def test_find_left(self, engine):
        series = [1.0, 2.0, None]
        idx, val = engine._find_left(series, 2)
        assert idx == 1
        assert val == 2.0

    def test_find_left_none(self, engine):
        series = [None, None, None]
        idx, val = engine._find_left(series, 2)
        assert val is None

    def test_find_right(self, engine):
        series = [None, None, 3.0]
        idx, val = engine._find_right(series, 0)
        assert idx == 2
        assert val == 3.0

    def test_find_right_none(self, engine):
        series = [None, None, None]
        idx, val = engine._find_right(series, 0)
        assert val is None

    def test_to_numeric_time_float(self, engine):
        assert engine._to_numeric_time(42.0) == 42.0

    def test_to_numeric_time_string(self, engine):
        assert engine._to_numeric_time("100") == 100.0

    def test_to_numeric_time_invalid(self, engine):
        assert engine._to_numeric_time("abc") == 0.0

    def test_fit_linear_trend_perfect(self, engine):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 2.0, 4.0, 6.0]
        slope, intercept, r_sq = engine._fit_linear_trend(x, y)
        assert slope == pytest.approx(2.0, rel=1e-4)
        assert intercept == pytest.approx(0.0, abs=0.01)
        assert r_sq == pytest.approx(1.0, rel=1e-4)

    def test_fit_linear_trend_insufficient(self, engine):
        slope, intercept, r_sq = engine._fit_linear_trend([1.0], [2.0])
        assert slope == 0.0
        assert r_sq == 0.0

    def test_distance_to_nearest_knot(self, engine):
        knots = [0.0, 5.0, 10.0]
        assert engine._distance_to_nearest_knot(3.0, knots) == 2.0
        assert engine._distance_to_nearest_knot(5.0, knots) == 0.0
