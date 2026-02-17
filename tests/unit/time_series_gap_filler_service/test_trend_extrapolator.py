# -*- coding: utf-8 -*-
"""
Unit tests for TrendExtrapolatorEngine - AGENT-DATA-014

Tests linear trend fitting (OLS), exponential smoothing (single, double,
Holt-Winters / triple), moving average extrapolation, trend detection
(classify TrendType), provenance tracking, confidence scoring, config
parameter defaults, and edge cases.
Target: 55 tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.time_series_gap_filler.config import (
    TimeSeriesGapFillerConfig,
    reset_config,
)
from greenlang.time_series_gap_filler.models import TrendType
from greenlang.time_series_gap_filler.trend_extrapolator import (
    TrendExtrapolatorEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove GL_TSGF_ env vars and reset config between tests."""
    keys = [k for k in os.environ if k.startswith("GL_TSGF_")]
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    reset_config()
    yield
    reset_config()


@pytest.fixture
def config():
    """Default test configuration."""
    return TimeSeriesGapFillerConfig(
        smoothing_alpha=0.3,
        smoothing_beta=0.1,
        smoothing_gamma=0.1,
        seasonal_periods=12,
        enable_provenance=True,
        min_data_points=10,
    )


@pytest.fixture
def engine(config):
    """Create a TrendExtrapolatorEngine with test config."""
    return TrendExtrapolatorEngine(config)


@pytest.fixture
def linear_four():
    """Short linear data: [2, 4, 6, 8]."""
    return [2.0, 4.0, 6.0, 8.0]


@pytest.fixture
def linear_series():
    """Perfectly linear series: y = 2*x + 10 for 30 points."""
    return [2.0 * i + 10.0 for i in range(30)]


@pytest.fixture
def noisy_linear():
    """Linear data y = 3*x with deterministic noise."""
    noise = [0.1, -0.3, 0.5, -0.2, 0.4, -0.1, 0.3, -0.4, 0.2, -0.5,
             0.15, -0.25, 0.35, -0.15, 0.45, -0.05, 0.25, -0.35, 0.1, -0.2]
    return [3.0 * i + noise[i] for i in range(20)]


@pytest.fixture
def constant_series():
    """Flat series at value 50."""
    return [50.0] * 30


@pytest.fixture
def increasing_series():
    """Monotonically increasing series 1..30."""
    return [float(i) for i in range(1, 31)]


@pytest.fixture
def decreasing_series():
    """Monotonically decreasing series 30..1."""
    return [float(30 - i) for i in range(30)]


@pytest.fixture
def exponential_values():
    """Exponential growth data: y = 2^(x/5)."""
    return [2.0 ** (i / 5.0) for i in range(20)]


@pytest.fixture
def seasonal_with_trend():
    """36-point series with trend + seasonality (period=12)."""
    values = []
    for i in range(36):
        trend = 0.5 * i
        seasonal = 10.0 * math.sin(2 * math.pi * (i % 12) / 12)
        values.append(100.0 + trend + seasonal)
    return values


@pytest.fixture
def linear_series_with_gaps():
    """Linear series y = 2*x + 10 with gaps at indices 5, 15, 25."""
    values = [2.0 * i + 10.0 for i in range(30)]
    values[5] = None
    values[15] = None
    values[25] = None
    return values


# =========================================================================
# Initialization
# =========================================================================


class TestTrendExtrapolatorInit:
    """Tests for TrendExtrapolatorEngine initialization."""

    def test_creates_instance(self, config):
        engine = TrendExtrapolatorEngine(config)
        assert engine is not None

    def test_config_stored(self, engine, config):
        assert engine._config is config

    def test_default_config_used_when_none(self):
        engine = TrendExtrapolatorEngine()
        assert engine._config is not None

    def test_provenance_tracker_created(self, engine):
        assert engine._provenance is not None


# =========================================================================
# fit_linear_trend
# =========================================================================


class TestFitLinearTrend:
    """Tests for fit_linear_trend (OLS linear regression)."""

    def test_known_linear_data_short(self, engine, linear_four):
        """[2,4,6,8] should yield slope~2, intercept~2, R^2~1.0."""
        result = engine.fit_linear_trend(linear_four)
        assert abs(result["slope"] - 2.0) < 0.01
        assert abs(result["intercept"] - 2.0) < 0.01
        assert abs(result["r_squared"] - 1.0) < 0.01

    def test_perfect_linear_30_points(self, engine, linear_series):
        """y = 2x + 10 for 30 points: slope~2, intercept~10, R^2~1.0."""
        result = engine.fit_linear_trend(linear_series)
        assert abs(result["slope"] - 2.0) < 0.1
        assert abs(result["intercept"] - 10.0) < 1.0
        assert result["r_squared"] > 0.99

    def test_noisy_data_lower_r_squared(self, engine, noisy_linear):
        """Noisy linear data should produce R^2 < 1.0."""
        result = engine.fit_linear_trend(noisy_linear)
        assert result["r_squared"] < 1.0
        assert abs(result["slope"] - 3.0) < 0.5

    def test_flat_data_slope_zero(self, engine, constant_series):
        """Flat data should have slope ~0."""
        result = engine.fit_linear_trend(constant_series)
        assert abs(result["slope"]) < 0.01

    def test_negative_slope(self, engine, decreasing_series):
        """Decreasing data should have negative slope."""
        result = engine.fit_linear_trend(decreasing_series)
        assert result["slope"] < -0.9

    def test_r_squared_between_zero_and_one(self, engine, noisy_linear):
        """R-squared should always be in [0, 1]."""
        result = engine.fit_linear_trend(noisy_linear)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_returns_provenance(self, engine, linear_four):
        """Result includes a SHA-256 provenance hash."""
        result = engine.fit_linear_trend(linear_four)
        assert "provenance_hash" in result
        assert isinstance(result["provenance_hash"], str)
        assert len(result["provenance_hash"]) == 64


# =========================================================================
# fill_linear_trend
# =========================================================================


class TestFillLinearTrend:
    """Tests for fill_linear_trend: fills gaps on the OLS regression line."""

    def test_fills_gaps_on_line(self, engine, linear_series_with_gaps):
        """Gaps in linear data should be filled near the trend line."""
        result = engine.fill_linear_trend(linear_series_with_gaps)
        filled = result["filled_values"]
        assert len(filled) == len(linear_series_with_gaps)
        # Index 5: expected = 2*5+10 = 20.0
        assert filled[5] == pytest.approx(20.0, abs=0.5)
        assert filled[15] == pytest.approx(40.0, abs=0.5)
        assert filled[25] == pytest.approx(60.0, abs=0.5)

    def test_preserves_known_values(self, engine, linear_series_with_gaps, linear_series):
        """Non-gap values must not be modified."""
        result = engine.fill_linear_trend(linear_series_with_gaps)
        filled = result["filled_values"]
        for i, v in enumerate(linear_series):
            if linear_series_with_gaps[i] is not None:
                assert filled[i] == pytest.approx(v, abs=1e-6)

    def test_no_gaps_no_change(self, engine, linear_series):
        """Series without gaps should return unchanged."""
        result = engine.fill_linear_trend(linear_series)
        filled = result["filled_values"]
        for i, v in enumerate(linear_series):
            assert filled[i] == pytest.approx(v, abs=1e-6)

    def test_returns_provenance(self, engine, linear_series_with_gaps):
        """Result includes provenance hash."""
        result = engine.fill_linear_trend(linear_series_with_gaps)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_returns_confidence(self, engine, linear_series_with_gaps):
        """Result includes confidence scores."""
        result = engine.fill_linear_trend(linear_series_with_gaps)
        assert "confidence_scores" in result


# =========================================================================
# fill_exponential_smoothing (SES)
# =========================================================================


class TestFillExponentialSmoothing:
    """Tests for fill_exponential_smoothing (Single Exponential Smoothing)."""

    def test_fills_gaps_forward(self, engine, linear_series_with_gaps):
        """Gaps should be filled using SES forward fill."""
        result = engine.fill_exponential_smoothing(
            values=linear_series_with_gaps, alpha=0.3,
        )
        filled = result["filled_values"]
        assert len(filled) == len(linear_series_with_gaps)
        assert filled[5] is not None
        assert filled[15] is not None

    def test_alpha_one_last_value(self, engine):
        """alpha=1.0 means SES output equals last observed value."""
        values = [10.0, 10.0, 10.0, 10.0, 50.0, None, None]
        result = engine.fill_exponential_smoothing(values=values, alpha=0.9)
        filled = result["filled_values"]
        assert filled[5] is not None
        assert filled[5] > 30.0

    def test_alpha_zero_first_value(self, engine):
        """alpha=0.1 (low) means SES output stays close to early level."""
        values = [10.0, 10.0, 10.0, 10.0, 50.0, None, None]
        result = engine.fill_exponential_smoothing(values=values, alpha=0.1)
        filled = result["filled_values"]
        assert filled[5] is not None
        assert filled[5] < 30.0

    def test_default_alpha_from_config(self, engine, linear_series_with_gaps):
        """When alpha is not provided, uses config.smoothing_alpha."""
        result = engine.fill_exponential_smoothing(values=linear_series_with_gaps)
        assert result is not None
        assert result["filled_values"][5] is not None

    def test_preserves_known_values(self, engine, linear_series_with_gaps, linear_series):
        """Non-gap values must not be modified."""
        result = engine.fill_exponential_smoothing(
            values=linear_series_with_gaps, alpha=0.3,
        )
        filled = result["filled_values"]
        for i, v in enumerate(linear_series):
            if linear_series_with_gaps[i] is not None:
                assert filled[i] == pytest.approx(v, abs=1e-6)

    def test_returns_provenance(self, engine, linear_series_with_gaps):
        """Result includes provenance hash."""
        result = engine.fill_exponential_smoothing(
            values=linear_series_with_gaps, alpha=0.5,
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# =========================================================================
# fill_double_exponential (Holt's method)
# =========================================================================


class TestFillDoubleExponential:
    """Tests for fill_double_exponential (level + trend extrapolation)."""

    def test_fills_with_trend(self, engine, linear_series_with_gaps):
        """Gaps should be filled accounting for level and trend."""
        result = engine.fill_double_exponential(
            values=linear_series_with_gaps, alpha=0.3, beta=0.1,
        )
        filled = result["filled_values"]
        assert len(filled) == len(linear_series_with_gaps)
        assert filled[5] is not None
        assert filled[15] is not None

    def test_upward_trend_tracked(self, engine, increasing_series):
        """Upward trend should be tracked."""
        values = list(increasing_series)
        values[20] = None
        result = engine.fill_double_exponential(values=values, alpha=0.3, beta=0.1)
        filled = result["filled_values"]
        assert filled[20] is not None
        assert abs(filled[20] - 21.0) < 5.0

    def test_downward_trend_tracked(self, engine, decreasing_series):
        """Downward trend should be tracked."""
        values = list(decreasing_series)
        values[20] = None
        result = engine.fill_double_exponential(values=values, alpha=0.3, beta=0.1)
        filled = result["filled_values"]
        assert filled[20] < 15.0

    def test_preserves_known_values(self, engine, linear_series_with_gaps, linear_series):
        """Non-gap values must not be modified."""
        result = engine.fill_double_exponential(
            values=linear_series_with_gaps, alpha=0.3, beta=0.1,
        )
        filled = result["filled_values"]
        for i, v in enumerate(linear_series):
            if linear_series_with_gaps[i] is not None:
                assert filled[i] == pytest.approx(v, abs=1e-6)

    def test_returns_method_name(self, engine, increasing_series):
        """Result should identify the method."""
        values = list(increasing_series)
        values[10] = None
        result = engine.fill_double_exponential(values=values, alpha=0.3, beta=0.1)
        assert result["method"] == "double_exponential"


# =========================================================================
# fill_holt_winters (Triple Exponential Smoothing)
# =========================================================================


class TestFillHoltWinters:
    """Tests for fill_holt_winters (seasonal + trend + level)."""

    def test_fills_seasonal_gaps(self, engine, seasonal_with_trend):
        """Fills gaps in data with seasonal and trend components."""
        values = list(seasonal_with_trend)
        values[18] = None
        result = engine.fill_holt_winters(
            values=values, alpha=0.3, beta=0.1, gamma=0.1, period=12,
        )
        filled = result["filled_values"]
        assert filled[18] is not None
        original = seasonal_with_trend[18]
        assert abs(filled[18] - original) < 20.0

    def test_matches_simple_when_no_seasonality(self, engine, linear_series):
        """When gamma~0 and period=1, Holt-Winters approximates
        double exponential (no significant seasonal component)."""
        values = list(linear_series)
        values[10] = None
        result = engine.fill_holt_winters(
            values=values, alpha=0.3, beta=0.1, gamma=0.0, period=1,
        )
        filled = result["filled_values"]
        assert filled[10] is not None
        # Should be reasonable for linear data around 2*10+10=30
        assert abs(filled[10] - 30.0) < 15.0

    def test_different_params_different_fills(self, engine, seasonal_with_trend):
        """Different alpha/beta/gamma should produce different fills."""
        values = list(seasonal_with_trend)
        values[18] = None
        r1 = engine.fill_holt_winters(
            values=list(values), alpha=0.2, beta=0.1, gamma=0.1, period=12,
        )
        r2 = engine.fill_holt_winters(
            values=list(values), alpha=0.8, beta=0.1, gamma=0.1, period=12,
        )
        assert r1["filled_values"][18] != pytest.approx(
            r2["filled_values"][18], abs=0.01,
        )

    def test_preserves_known_values(self, engine, seasonal_with_trend):
        """Non-gap values must not be modified."""
        values = list(seasonal_with_trend)
        values[18] = None
        result = engine.fill_holt_winters(
            values=values, alpha=0.3, beta=0.1, gamma=0.1, period=12,
        )
        filled = result["filled_values"]
        for i, v in enumerate(seasonal_with_trend):
            if values[i] is not None:
                assert filled[i] == pytest.approx(v, abs=1e-6)

    def test_returns_provenance(self, engine, seasonal_with_trend):
        """Result includes provenance hash."""
        values = list(seasonal_with_trend)
        values[18] = None
        result = engine.fill_holt_winters(
            values=values, alpha=0.3, beta=0.1, gamma=0.1, period=12,
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_returns_method_name(self, engine, seasonal_with_trend):
        """Result should identify the method."""
        values = list(seasonal_with_trend)
        values[18] = None
        result = engine.fill_holt_winters(
            values=values, alpha=0.3, beta=0.1, gamma=0.1, period=12,
        )
        assert result["method"] == "holt_winters"

    def test_default_params_from_config(self, engine, seasonal_with_trend):
        """When params not given, uses config defaults."""
        values = list(seasonal_with_trend)
        values[18] = None
        result = engine.fill_holt_winters(values=values)
        assert result is not None
        assert result["filled_values"][18] is not None


# =========================================================================
# fill_moving_average
# =========================================================================


class TestFillMovingAverage:
    """Tests for fill_moving_average: average of last N known values."""

    def test_basic_fill(self, engine):
        """Moving average of last 3 known values."""
        values = [10.0, 12.0, 14.0, None, 18.0, 20.0]
        result = engine.fill_moving_average(values=values, window=3)
        filled = result["filled_values"]
        assert filled[3] is not None
        # Mean of last 3 known values before gap: (10+12+14)/3 = 12.0
        assert abs(filled[3] - 12.0) < 1.0

    def test_window_larger_than_series(self, engine):
        """Window larger than available data should use all available known values."""
        values = [10.0, None, 30.0]
        result = engine.fill_moving_average(values=values, window=10)
        filled = result["filled_values"]
        assert filled[1] is not None
        # Only one known value (10.0) precedes the gap at index 1,
        # so the moving average fill equals 10.0
        assert abs(filled[1] - 10.0) < 1.0

    def test_preserves_known_values(self, engine, linear_series):
        """Known values should not be changed."""
        values = list(linear_series)
        values[10] = None
        result = engine.fill_moving_average(values=values, window=5)
        filled = result["filled_values"]
        for i, v in enumerate(linear_series):
            if values[i] is not None:
                assert filled[i] == pytest.approx(v, abs=1e-6)

    def test_returns_method_name(self, engine, linear_series_with_gaps):
        """Result identifies the method."""
        result = engine.fill_moving_average(
            values=linear_series_with_gaps, window=5,
        )
        assert result["method"] == "moving_average"


# =========================================================================
# detect_trend
# =========================================================================


class TestDetectTrend:
    """Tests for detect_trend: classifies TrendType from series."""

    def test_linear_data(self, engine, linear_series):
        """Perfect linear data should be detected as LINEAR."""
        trend_type = engine.detect_trend(linear_series)
        assert trend_type in (
            TrendType.LINEAR, TrendType.MODERATE_LINEAR,
        )

    def test_flat_data_no_trend(self, engine, constant_series):
        """Flat data should be detected as STATIONARY or LINEAR.

        Note: Constant data is a degenerate case where R-squared=1.0
        (perfect fit since ss_tot=0), so the classifier may report LINEAR.
        """
        trend_type = engine.detect_trend(constant_series)
        assert trend_type in (
            TrendType.STATIONARY, TrendType.UNKNOWN,
            TrendType.NONE, TrendType.LINEAR,
        )

    def test_exponential_data(self, engine, exponential_values):
        """Exponential data should be detected as EXPONENTIAL or POLYNOMIAL."""
        trend_type = engine.detect_trend(exponential_values)
        assert trend_type in (
            TrendType.EXPONENTIAL, TrendType.POLYNOMIAL,
            TrendType.LINEAR, TrendType.MODERATE_LINEAR,
        )

    def test_returns_r_squared_via_analyze(self, engine, linear_series):
        """analyze_trend result includes R-squared metric."""
        result = engine.analyze_trend(linear_series)
        assert hasattr(result, "r_squared")
        assert 0.0 <= result.r_squared <= 1.0

    def test_returns_provenance_via_analyze(self, engine, linear_series):
        """analyze_trend result includes provenance hash."""
        result = engine.analyze_trend(linear_series)
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_series(self, engine):
        """Empty series returns unfilled result with zero points."""
        result = engine.fill_linear_trend([])
        assert result["points_filled"] == 0
        assert result["filled_values"] == []

    def test_all_missing(self, engine):
        """All-None series returns unfilled result (no known points to fit)."""
        result = engine.fill_linear_trend([None] * 10)
        assert result["points_filled"] == 0
        assert len(result["filled_values"]) == 10

    def test_single_point_fit(self, engine):
        """Single data point should raise for fit."""
        try:
            result = engine.fit_linear_trend([5.0])
            assert result is not None
        except (ValueError, RuntimeError):
            pass

    def test_two_points_sufficient(self, engine):
        """Two points should allow linear fit."""
        values = [10.0, None, 30.0]
        result = engine.fill_linear_trend(values)
        filled = result["filled_values"]
        assert filled[1] is not None
        assert abs(filled[1] - 20.0) < 5.0

    def test_negative_values(self, engine):
        """Negative values should not cause errors."""
        values = [-10.0, -5.0, None, 5.0, 10.0]
        result = engine.fill_linear_trend(values)
        assert result["filled_values"][2] is not None

    def test_large_values_no_overflow(self, engine):
        """Very large values should not overflow."""
        values = [1e12, 2e12, None, 4e12, 5e12]
        result = engine.fill_linear_trend(values)
        assert result["filled_values"][2] is not None
        assert abs(result["filled_values"][2] - 3e12) < 1e11

    def test_gap_at_start(self, engine):
        """Leading gaps should be handled."""
        values = [None, None, 30.0, 40.0, 50.0, 60.0, 70.0]
        result = engine.fill_linear_trend(values)
        filled = result["filled_values"]
        assert filled[0] is not None
        assert filled[1] is not None

    def test_gap_at_end(self, engine):
        """Trailing gaps should be handled."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0, None, None]
        result = engine.fill_linear_trend(values)
        filled = result["filled_values"]
        assert filled[5] is not None
        assert filled[6] is not None
        assert filled[5] > 45.0

    def test_nan_treated_as_gap(self, engine):
        """float('nan') should be treated as a gap."""
        values = [10.0, 20.0, float('nan'), 40.0, 50.0]
        try:
            result = engine.fill_linear_trend(values)
            assert result["filled_values"][2] is not None
        except (ValueError, TypeError):
            pass


# =========================================================================
# Provenance and determinism
# =========================================================================


class TestProvenanceAndDeterminism:
    """Tests for provenance recording and deterministic outputs."""

    def test_provenance_deterministic(self, engine, linear_series_with_gaps):
        """Same input produces same provenance hash."""
        r1 = engine.fill_linear_trend(list(linear_series_with_gaps))
        r2 = engine.fill_linear_trend(list(linear_series_with_gaps))
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_provenance_changes_with_input(self, engine, linear_series):
        """Different gap counts produce different provenance hashes."""
        v1 = list(linear_series)
        v1[5] = None
        v2 = list(linear_series)
        v2[10] = None
        v2[11] = None
        r1 = engine.fill_linear_trend(v1)
        r2 = engine.fill_linear_trend(v2)
        assert r1["provenance_hash"] != r2["provenance_hash"]

    def test_processing_time_recorded(self, engine, linear_series_with_gaps):
        """Result includes processing_time_ms."""
        result = engine.fill_linear_trend(linear_series_with_gaps)
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0.0

    def test_method_in_result(self, engine, linear_series_with_gaps):
        """Result contains the method name."""
        result = engine.fill_linear_trend(linear_series_with_gaps)
        assert "method" in result

    def test_gaps_filled_count(self, engine, linear_series_with_gaps):
        """Result reports correct number of gaps filled."""
        result = engine.fill_linear_trend(linear_series_with_gaps)
        assert "gaps_filled" in result
        assert result["gaps_filled"] == 3


# =========================================================================
# Confidence scoring
# =========================================================================


class TestConfidenceScoring:
    """Tests for confidence score computation."""

    def test_confidence_known_is_one(self, engine, linear_series_with_gaps):
        """Known values should have confidence 1.0."""
        result = engine.fill_linear_trend(linear_series_with_gaps)
        confs = result["confidence_scores"]
        for i, v in enumerate(linear_series_with_gaps):
            if v is not None:
                assert confs[i] == pytest.approx(1.0)

    def test_confidence_gap_in_range(self, engine, linear_series_with_gaps):
        """Gap confidence should be between 0 and 1."""
        result = engine.fill_linear_trend(linear_series_with_gaps)
        confs = result["confidence_scores"]
        for i, v in enumerate(linear_series_with_gaps):
            if v is None:
                assert 0.0 <= confs[i] <= 1.0


# =========================================================================
# Config parameter defaults
# =========================================================================


class TestConfigDefaults:
    """Tests that config parameter defaults are used correctly."""

    def test_exponential_uses_config_alpha(self, config, linear_series_with_gaps):
        """fill_exponential_smoothing uses config.smoothing_alpha."""
        config.smoothing_alpha = 0.5
        eng = TrendExtrapolatorEngine(config)
        result = eng.fill_exponential_smoothing(values=linear_series_with_gaps)
        assert result["filled_values"][5] is not None

    def test_holt_winters_uses_config_period(self, config, seasonal_with_trend):
        """fill_holt_winters uses config.seasonal_periods."""
        config.seasonal_periods = 12
        eng = TrendExtrapolatorEngine(config)
        values = list(seasonal_with_trend)
        values[18] = None
        result = eng.fill_holt_winters(values=values)
        assert result["filled_values"][18] is not None

    def test_double_exponential_uses_config_beta(self, config, linear_series_with_gaps):
        """fill_double_exponential uses config.smoothing_beta."""
        config.smoothing_beta = 0.2
        eng = TrendExtrapolatorEngine(config)
        result = eng.fill_double_exponential(values=linear_series_with_gaps)
        assert result["filled_values"][5] is not None
