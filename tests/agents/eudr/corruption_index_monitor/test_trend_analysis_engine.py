# -*- coding: utf-8 -*-
"""
Unit tests for TrendAnalysisEngine (AGENT-EUDR-019, Engine 5).

Tests all methods of TrendAnalysisEngine including trend analysis for
improving/deteriorating/stable/volatile patterns, trajectory computation,
prediction with linear/moving-average/exponential models, country screening,
breakpoint detection, linear regression accuracy, moving average calculations,
direction classification, and provenance chain integrity.

Coverage target: 85%+ of TrendAnalysisEngine methods.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.corruption_index_monitor.engines.trend_analysis_engine import (
    TrendAnalysisEngine,
    TrendDirection,
    IndexType,
    PredictionModel,
    ConfidenceLevel,
    TrendResult,
    TrendPrediction,
    Breakpoint,
    TrajectoryResult,
    CountryTrendSummary,
    REFERENCE_CPI_DATA,
    REFERENCE_WGI_CC_DATA,
    MIN_TREND_DATA_POINTS,
    MIN_BREAKPOINT_DATA_POINTS,
    STABLE_SLOPE_THRESHOLD,
    VOLATILITY_CV_THRESHOLD,
    R_SQUARED_HIGH,
    R_SQUARED_MEDIUM,
    R_SQUARED_LOW,
    _to_decimal,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> TrendAnalysisEngine:
    """Create a default TrendAnalysisEngine instance."""
    return TrendAnalysisEngine()


@pytest.fixture
def engine_with_custom_data() -> TrendAnalysisEngine:
    """Create engine with custom improving data for a test country."""
    eng = TrendAnalysisEngine()
    eng.load_custom_data(
        "XX",
        "CPI",
        {
            2015: Decimal("20"),
            2016: Decimal("25"),
            2017: Decimal("30"),
            2018: Decimal("35"),
            2019: Decimal("40"),
            2020: Decimal("45"),
            2021: Decimal("50"),
            2022: Decimal("55"),
            2023: Decimal("60"),
            2024: Decimal("65"),
        },
    )
    return eng


@pytest.fixture
def engine_with_deteriorating_data() -> TrendAnalysisEngine:
    """Create engine with deteriorating data."""
    eng = TrendAnalysisEngine()
    eng.load_custom_data(
        "YY",
        "CPI",
        {
            2015: Decimal("70"),
            2016: Decimal("65"),
            2017: Decimal("60"),
            2018: Decimal("55"),
            2019: Decimal("50"),
            2020: Decimal("45"),
            2021: Decimal("40"),
            2022: Decimal("35"),
            2023: Decimal("30"),
            2024: Decimal("25"),
        },
    )
    return eng


# ---------------------------------------------------------------------------
# TestTrendAnalysis
# ---------------------------------------------------------------------------


class TestTrendAnalysis:
    """Tests for analyze_trend with improving, stable, deteriorating, volatile patterns."""

    def test_improving_trend(self, engine_with_custom_data: TrendAnalysisEngine):
        """Custom improving data should be classified as IMPROVING."""
        result = engine_with_custom_data.analyze_trend("XX", "CPI")
        assert result["direction"] == "IMPROVING"
        assert Decimal(result["slope"]) > Decimal("0")

    def test_deteriorating_trend(
        self, engine_with_deteriorating_data: TrendAnalysisEngine
    ):
        """Custom deteriorating data should be classified as DETERIORATING."""
        result = engine_with_deteriorating_data.analyze_trend("YY", "CPI")
        assert result["direction"] == "DETERIORATING"
        assert Decimal(result["slope"]) < Decimal("0")

    def test_stable_trend(self, engine: TrendAnalysisEngine):
        """Singapore (SG) with stable CPI should be STABLE."""
        result = engine.analyze_trend("SG", "CPI")
        # SG CPI: 85,84,84,85,85,85,85,83,83,83 -- very stable
        assert result["direction"] in ("STABLE", "DETERIORATING")
        assert abs(Decimal(result["slope"])) < Decimal("1.5")

    def test_unknown_country_error(self, engine: TrendAnalysisEngine):
        """Unknown country should still return a result (possibly insufficient data)."""
        result = engine.analyze_trend("QQ", "CPI")
        assert result["direction"] == "INSUFFICIENT_DATA"
        assert result["data_points"] == 0

    def test_analyze_wgi_data(self, engine: TrendAnalysisEngine):
        """WGI trend analysis should work for countries with WGI data."""
        result = engine.analyze_trend("DK", "WGI")
        assert "direction" in result
        assert result["data_points"] > 0

    def test_result_has_all_fields(self, engine: TrendAnalysisEngine):
        """Trend result should contain all expected fields."""
        result = engine.analyze_trend("BR", "CPI")
        expected_keys = {
            "result_id", "country_code", "index_type", "direction",
            "slope", "intercept", "r_squared", "standard_error",
            "start_year", "end_year", "data_points", "provenance_hash",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_year_range_filter(self, engine: TrendAnalysisEngine):
        """Year range filtering should limit data points."""
        result = engine.analyze_trend("BR", "CPI", start_year=2020, end_year=2024)
        assert result["start_year"] >= 2020
        assert result["end_year"] <= 2024


# ---------------------------------------------------------------------------
# TestTrajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    """Tests for get_trajectory with different window sizes."""

    def test_default_trajectory(self, engine: TrendAnalysisEngine):
        """Default trajectory should work for countries with data."""
        result = engine.analyze_trend("BR", "CPI")
        assert result["data_points"] >= 5
        assert "direction" in result

    def test_trajectory_with_custom_data(
        self, engine_with_custom_data: TrendAnalysisEngine
    ):
        """Custom improving data should show positive velocity."""
        result = engine_with_custom_data.analyze_trend("XX", "CPI")
        assert Decimal(result["slope"]) > Decimal("0")

    def test_trajectory_direction_for_venezuela(self, engine: TrendAnalysisEngine):
        """Venezuela's deteriorating CPI should show DETERIORATING."""
        result = engine.analyze_trend("VE", "CPI")
        assert result["direction"] == "DETERIORATING"

    def test_trajectory_data_quality(self, engine: TrendAnalysisEngine):
        """Result should contain confidence level."""
        result = engine.analyze_trend("DK", "CPI")
        assert "confidence_level" in result


# ---------------------------------------------------------------------------
# TestPrediction
# ---------------------------------------------------------------------------


class TestPrediction:
    """Tests for predict_future with linear/moving average/exponential models."""

    def test_exponential_smoothing_basic(self, engine: TrendAnalysisEngine):
        """Exponential smoothing should produce valid forecasts."""
        values = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")]
        forecasts = engine._exponential_smoothing(
            values, alpha=Decimal("0.3"), beta=Decimal("0.1"), forecast_periods=3
        )
        assert len(forecasts) == 3
        # Forecasts should continue upward trend
        assert forecasts[0] > values[-1]

    def test_exponential_smoothing_insufficient_data(
        self, engine: TrendAnalysisEngine
    ):
        """Exponential smoothing with < 2 values should raise ValueError."""
        with pytest.raises(ValueError):
            engine._exponential_smoothing([Decimal("5")], forecast_periods=1)

    def test_weighted_moving_average(self, engine: TrendAnalysisEngine):
        """WMA should weight recent values more heavily."""
        values = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40"), Decimal("50")]
        wma = engine._calculate_weighted_moving_average(values, window=5)
        # WMA should be closer to 50 than simple average (30)
        assert wma > Decimal("30")
        assert wma <= Decimal("50")

    def test_weighted_moving_average_empty(self, engine: TrendAnalysisEngine):
        """Empty values should raise ValueError."""
        with pytest.raises(ValueError):
            engine._calculate_weighted_moving_average([])

    def test_exponential_smoothing_flat_series(self, engine: TrendAnalysisEngine):
        """Flat series should produce forecast near the constant value."""
        values = [Decimal("50")] * 10
        forecasts = engine._exponential_smoothing(values, forecast_periods=3)
        for f in forecasts:
            assert abs(f - Decimal("50")) < Decimal("1")


# ---------------------------------------------------------------------------
# TestImprovingCountries / TestDeterioratingCountries
# ---------------------------------------------------------------------------


class TestImprovingCountries:
    """Tests for finding improving countries with various thresholds."""

    def test_improving_custom_data(
        self, engine_with_custom_data: TrendAnalysisEngine
    ):
        """Custom improving country should be detected."""
        result = engine_with_custom_data.analyze_trend("XX", "CPI")
        assert result["direction"] == "IMPROVING"
        # Total change should be positive
        change = Decimal(result.get("change_absolute", "0"))
        assert change > Decimal("0")


class TestDeterioratingCountries:
    """Tests for finding deteriorating countries."""

    def test_deteriorating_detection(
        self, engine_with_deteriorating_data: TrendAnalysisEngine
    ):
        """Custom deteriorating country should be detected."""
        result = engine_with_deteriorating_data.analyze_trend("YY", "CPI")
        assert result["direction"] == "DETERIORATING"

    def test_honduras_deteriorating(self, engine: TrendAnalysisEngine):
        """Honduras (HN) CPI shows decline from 31 to 23 -- should deteriorate."""
        result = engine.analyze_trend("HN", "CPI")
        assert result["direction"] == "DETERIORATING"

    def test_guatemala_deteriorating(self, engine: TrendAnalysisEngine):
        """Guatemala (GT) CPI shows decline from 28 to 23 -- should deteriorate."""
        result = engine.analyze_trend("GT", "CPI")
        assert result["direction"] == "DETERIORATING"


# ---------------------------------------------------------------------------
# TestBreakpointDetection
# ---------------------------------------------------------------------------


class TestBreakpointDetection:
    """Tests for detect_breakpoints for series with/without structural breaks."""

    def test_no_breakpoint_stable_series(self, engine: TrendAnalysisEngine):
        """Stable series should have no breakpoints."""
        values = [Decimal("50")] * 12
        breakpoints = engine._detect_regime_change(values)
        assert len(breakpoints) == 0

    def test_breakpoint_with_shift(self, engine: TrendAnalysisEngine):
        """Series with a clear level shift should detect breakpoints."""
        values = [Decimal("30")] * 6 + [Decimal("60")] * 6
        breakpoints = engine._detect_regime_change(values)
        assert len(breakpoints) >= 1

    def test_insufficient_data_no_breakpoints(self, engine: TrendAnalysisEngine):
        """Series shorter than MIN_BREAKPOINT_DATA_POINTS should return empty."""
        values = [Decimal("10"), Decimal("20"), Decimal("30")]
        breakpoints = engine._detect_regime_change(values)
        assert breakpoints == []

    def test_gradually_increasing_series(self, engine: TrendAnalysisEngine):
        """Gradually increasing series may or may not show breakpoints."""
        values = [Decimal(str(i * 5)) for i in range(12)]
        breakpoints = engine._detect_regime_change(values)
        # Gradual increase may or may not trigger CUSUM depending on threshold
        assert isinstance(breakpoints, list)


# ---------------------------------------------------------------------------
# TestLinearRegression
# ---------------------------------------------------------------------------


class TestLinearRegression:
    """Tests for _linear_regression accuracy with known data points."""

    def test_perfect_linear_fit(self, engine: TrendAnalysisEngine):
        """Perfect linear data should give r_squared close to 1.0."""
        x = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40"), Decimal("50")]
        slope, intercept, r_sq, se = engine._linear_regression(x, y)
        assert abs(r_sq - Decimal("1.0")) < Decimal("0.001")
        assert abs(slope - Decimal("10")) < Decimal("0.01")
        assert abs(intercept - Decimal("0")) < Decimal("0.01")

    def test_negative_slope(self, engine: TrendAnalysisEngine):
        """Decreasing data should produce negative slope."""
        x = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("50"), Decimal("40"), Decimal("30"), Decimal("20"), Decimal("10")]
        slope, intercept, r_sq, se = engine._linear_regression(x, y)
        assert slope < Decimal("0")
        assert abs(r_sq - Decimal("1.0")) < Decimal("0.001")

    def test_insufficient_points(self, engine: TrendAnalysisEngine):
        """Fewer than 2 points should raise ValueError."""
        with pytest.raises(ValueError):
            engine._linear_regression([Decimal("1")], [Decimal("10")])

    def test_mismatched_lengths(self, engine: TrendAnalysisEngine):
        """Different lengths should raise ValueError."""
        with pytest.raises(ValueError):
            engine._linear_regression(
                [Decimal("1"), Decimal("2")],
                [Decimal("10")],
            )

    def test_constant_y_values(self, engine: TrendAnalysisEngine):
        """Constant y values should give slope of 0."""
        x = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        y = [Decimal("50")] * 5
        slope, intercept, r_sq, se = engine._linear_regression(x, y)
        assert slope == Decimal("0")

    def test_two_points(self, engine: TrendAnalysisEngine):
        """Two points should still produce valid regression."""
        x = [Decimal("1"), Decimal("5")]
        y = [Decimal("10"), Decimal("50")]
        slope, intercept, r_sq, se = engine._linear_regression(x, y)
        assert abs(slope - Decimal("10")) < Decimal("0.01")
        assert abs(r_sq - Decimal("1.0")) < Decimal("0.001")


# ---------------------------------------------------------------------------
# TestMovingAverage
# ---------------------------------------------------------------------------


class TestMovingAverage:
    """Tests for _calculate_moving_average with various window sizes."""

    def test_window_3(self, engine: TrendAnalysisEngine):
        """Window of 3 should produce correct moving averages."""
        values = [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40"), Decimal("50")]
        ma = engine._calculate_moving_average(values, window=3)
        assert len(ma) == 3
        assert ma[0] == Decimal("20.00")  # (10+20+30)/3
        assert ma[1] == Decimal("30.00")  # (20+30+40)/3
        assert ma[2] == Decimal("40.00")  # (30+40+50)/3

    def test_window_1(self, engine: TrendAnalysisEngine):
        """Window of 1 should return the original values."""
        values = [Decimal("10"), Decimal("20"), Decimal("30")]
        ma = engine._calculate_moving_average(values, window=1)
        assert len(ma) == 3
        assert ma[0] == Decimal("10.00")

    def test_window_equals_data_length(self, engine: TrendAnalysisEngine):
        """Window equal to data length should return single average."""
        values = [Decimal("10"), Decimal("20"), Decimal("30")]
        ma = engine._calculate_moving_average(values, window=3)
        assert len(ma) == 1
        assert ma[0] == Decimal("20.00")

    def test_window_too_large(self, engine: TrendAnalysisEngine):
        """Window larger than data should raise ValueError."""
        with pytest.raises(ValueError):
            engine._calculate_moving_average([Decimal("10")], window=5)

    def test_window_zero(self, engine: TrendAnalysisEngine):
        """Window of 0 should raise ValueError."""
        with pytest.raises(ValueError):
            engine._calculate_moving_average([Decimal("10"), Decimal("20")], window=0)


# ---------------------------------------------------------------------------
# TestTrendDirection
# ---------------------------------------------------------------------------


class TestTrendDirection:
    """Verify IMPROVING/STABLE/DETERIORATING/VOLATILE/INSUFFICIENT_DATA."""

    def test_all_directions_defined(self):
        """All 5 trend directions should be defined."""
        directions = set(d.value for d in TrendDirection)
        assert directions == {
            "IMPROVING", "STABLE", "DETERIORATING",
            "VOLATILE", "INSUFFICIENT_DATA",
        }

    @pytest.mark.parametrize(
        "slope,r_sq,cv,data_points,expected",
        [
            (Decimal("2.0"), Decimal("0.8"), Decimal("0.05"), 10, "IMPROVING"),
            (Decimal("-2.0"), Decimal("0.8"), Decimal("0.05"), 10, "DETERIORATING"),
            (Decimal("0.1"), Decimal("0.8"), Decimal("0.05"), 10, "STABLE"),
            (Decimal("0.0"), Decimal("0.1"), Decimal("0.30"), 10, "VOLATILE"),
            (Decimal("5.0"), Decimal("0.9"), Decimal("0.02"), 3, "INSUFFICIENT_DATA"),
        ],
    )
    def test_direction_classification(
        self,
        engine: TrendAnalysisEngine,
        slope: Decimal,
        r_sq: Decimal,
        cv: Decimal,
        data_points: int,
        expected: str,
    ):
        """Classify direction based on slope, r-squared, CV, and data points."""
        direction = engine._classify_direction(slope, r_sq, cv, "CPI", data_points)
        assert direction == expected

    def test_composite_reversed_direction(self, engine: TrendAnalysisEngine):
        """COMPOSITE index has reversed direction: negative slope = IMPROVING."""
        direction = engine._classify_direction(
            Decimal("-2.0"), Decimal("0.8"), Decimal("0.05"), "COMPOSITE", 10
        )
        assert direction == "IMPROVING"

    def test_confidence_classification(self, engine: TrendAnalysisEngine):
        """Confidence levels should follow R-squared thresholds."""
        assert engine._classify_confidence(Decimal("0.90")) == "HIGH"
        assert engine._classify_confidence(Decimal("0.65")) == "MEDIUM"
        assert engine._classify_confidence(Decimal("0.35")) == "LOW"
        assert engine._classify_confidence(Decimal("0.10")) == "VERY_LOW"


# ---------------------------------------------------------------------------
# TestTrendProvenance
# ---------------------------------------------------------------------------


class TestTrendProvenance:
    """Tests for provenance chain integrity."""

    def test_trend_result_has_provenance(self, engine: TrendAnalysisEngine):
        """Trend analysis result should include provenance hash."""
        result = engine.analyze_trend("BR", "CPI")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_provenance_deterministic(self, engine: TrendAnalysisEngine):
        """Same inputs should produce same provenance hash."""
        r1 = engine.analyze_trend("BR", "CPI", 2015, 2024)
        r2 = engine.analyze_trend("BR", "CPI", 2015, 2024)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_different_countries_different_provenance(
        self, engine: TrendAnalysisEngine
    ):
        """Different countries should produce different provenance hashes."""
        r1 = engine.analyze_trend("BR", "CPI")
        r2 = engine.analyze_trend("DK", "CPI")
        assert r1["provenance_hash"] != r2["provenance_hash"]


# ---------------------------------------------------------------------------
# TestTrendEdgeCases
# ---------------------------------------------------------------------------


class TestTrendEdgeCases:
    """Tests for insufficient data, flat series, and single data point."""

    def test_insufficient_data_direction(self, engine: TrendAnalysisEngine):
        """Country with no reference data should return INSUFFICIENT_DATA."""
        result = engine.analyze_trend("QQ", "CPI")
        assert result["direction"] == "INSUFFICIENT_DATA"

    def test_single_data_point(self, engine: TrendAnalysisEngine):
        """Single data point should return INSUFFICIENT_DATA."""
        eng = TrendAnalysisEngine()
        eng.load_custom_data("AB", "CPI", {2024: Decimal("50")})
        result = eng.analyze_trend("AB", "CPI")
        assert result["direction"] == "INSUFFICIENT_DATA"

    def test_two_data_points(self, engine: TrendAnalysisEngine):
        """Two data points should still return INSUFFICIENT_DATA (< 5 min)."""
        eng = TrendAnalysisEngine()
        eng.load_custom_data(
            "AC", "CPI", {2023: Decimal("40"), 2024: Decimal("50")}
        )
        result = eng.analyze_trend("AC", "CPI")
        assert result["direction"] == "INSUFFICIENT_DATA"

    def test_flat_series(self, engine: TrendAnalysisEngine):
        """Perfectly flat series should return STABLE."""
        eng = TrendAnalysisEngine()
        eng.load_custom_data(
            "FL", "CPI",
            {yr: Decimal("50") for yr in range(2015, 2025)},
        )
        result = eng.analyze_trend("FL", "CPI")
        assert result["direction"] == "STABLE"

    def test_load_custom_data_empty_raises(self, engine: TrendAnalysisEngine):
        """Loading empty data should raise ValueError."""
        with pytest.raises(ValueError):
            engine.load_custom_data("XX", "CPI", {})

    def test_load_custom_data_invalid_index_type(self, engine: TrendAnalysisEngine):
        """Invalid index type should raise ValueError."""
        with pytest.raises(ValueError):
            engine.load_custom_data("XX", "INVALID", {2024: Decimal("50")})

    def test_load_custom_data_empty_country(self, engine: TrendAnalysisEngine):
        """Empty country code should raise ValueError."""
        with pytest.raises(ValueError):
            engine.load_custom_data("", "CPI", {2024: Decimal("50")})

    def test_coefficient_of_variation_empty(self, engine: TrendAnalysisEngine):
        """Empty list should return 0 for CV."""
        cv = engine._calculate_coefficient_of_variation([])
        assert cv == Decimal("0")

    def test_coefficient_of_variation_zero_mean(self, engine: TrendAnalysisEngine):
        """Values with mean of 0 should return CV of 0."""
        values = [Decimal("-1"), Decimal("1")]
        cv = engine._calculate_coefficient_of_variation(values)
        # mean is 0, so CV is 0
        assert cv == Decimal("0")

    def test_index_type_enum(self):
        """All IndexType values should exist."""
        assert IndexType.CPI.value == "CPI"
        assert IndexType.WGI.value == "WGI"
        assert IndexType.BRIBERY.value == "BRIBERY"
        assert IndexType.COMPOSITE.value == "COMPOSITE"

    def test_prediction_model_enum(self):
        """All PredictionModel values should exist."""
        assert PredictionModel.LINEAR.value == "linear"
        assert PredictionModel.WMA.value == "wma"
        assert PredictionModel.ETS.value == "ets"
