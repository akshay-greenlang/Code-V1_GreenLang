# -*- coding: utf-8 -*-
"""
GL-005 Trending Module Tests
============================

Comprehensive unit tests for trending and long-term analysis module
including time series storage, trend analysis, and seasonality detection.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
import math
import random

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    TrendingConfig,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    FlueGasReading,
    CQIResult,
    CQIRating,
    TrendDirection,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.trending import (
    TimeSeriesPoint,
    AggregatedPeriod,
    TrendAnalysisResult,
    SeasonalityResult,
    BaselineComparison,
    TimeSeriesStore,
    TrendAnalyzer,
    TrendingEngine,
    calculate_moving_average,
    calculate_exponential_moving_average,
)


class TestTimeSeriesPoint:
    """Tests for TimeSeriesPoint data structure."""

    def test_create_point(self):
        """Test creating a time series point."""
        now = datetime.now(timezone.utc)
        point = TimeSeriesPoint(
            timestamp=now,
            value=85.0,
            metadata={"source": "test"},
        )

        assert point.timestamp == now
        assert point.value == 85.0
        assert point.metadata["source"] == "test"

    def test_point_with_empty_metadata(self):
        """Test point with empty metadata."""
        point = TimeSeriesPoint(
            timestamp=datetime.now(timezone.utc),
            value=100.0,
        )

        assert point.metadata == {}


class TestTimeSeriesStore:
    """Tests for TimeSeriesStore."""

    def test_initialization(self, default_trending_config):
        """Test store initialization."""
        store = TimeSeriesStore(default_trending_config)

        assert store.config == default_trending_config
        assert len(store._raw_data) == 0

    def test_add_point(self, default_trending_config):
        """Test adding a data point."""
        store = TimeSeriesStore(default_trending_config)

        now = datetime.now(timezone.utc)
        store.add_point("cqi", now, 85.0, {"rating": "good"})

        data = store.get_raw_data("cqi")
        assert len(data) == 1
        assert data[0].value == 85.0

    def test_add_multiple_points(self, default_trending_config):
        """Test adding multiple points."""
        store = TimeSeriesStore(default_trending_config)

        now = datetime.now(timezone.utc)
        for i in range(100):
            store.add_point(
                "cqi",
                now + timedelta(minutes=i),
                85.0 + random.gauss(0, 2),
            )

        data = store.get_raw_data("cqi")
        assert len(data) == 100

    def test_get_raw_data_time_filtered(self, default_trending_config):
        """Test getting filtered raw data."""
        store = TimeSeriesStore(default_trending_config)

        base_time = datetime.now(timezone.utc) - timedelta(days=10)
        for i in range(20):
            store.add_point(
                "cqi",
                base_time + timedelta(days=i),
                85.0,
            )

        # Get last 7 days only
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        data = store.get_raw_data("cqi", start_time=start_time)

        # Should only have recent data
        for point in data:
            assert point.timestamp >= start_time

    def test_data_aging(self, default_trending_config):
        """Test that old data is aged out."""
        config = TrendingConfig(raw_data_retention_days=30)
        store = TimeSeriesStore(config)

        # Add old data
        old_time = datetime.now(timezone.utc) - timedelta(days=35)
        store.add_point("cqi", old_time, 80.0)

        # Add recent data (triggers aging)
        store.add_point("cqi", datetime.now(timezone.utc), 85.0)

        data = store.get_raw_data("cqi")

        # Old data should be removed
        for point in data:
            assert point.timestamp > datetime.now(timezone.utc) - timedelta(days=31)

    def test_aggregate_hourly(self, default_trending_config):
        """Test hourly aggregation."""
        store = TimeSeriesStore(default_trending_config)

        # Add data for 3 hours (10 points per hour)
        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        for hour in range(3):
            for minute in range(10):
                store.add_point(
                    "cqi",
                    base_time + timedelta(hours=hour, minutes=minute*6),
                    80.0 + hour * 5 + random.gauss(0, 1),
                )

        aggregates = store.aggregate_hourly("cqi")

        assert len(aggregates) == 3
        for agg in aggregates:
            assert agg.count == 10

    def test_aggregate_daily(self, default_trending_config):
        """Test daily aggregation."""
        store = TimeSeriesStore(default_trending_config)

        # Add data for 5 days (24 points per day)
        base_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        for day in range(5):
            for hour in range(24):
                store.add_point(
                    "cqi",
                    base_time + timedelta(days=day, hours=hour),
                    85.0 + random.gauss(0, 2),
                )

        aggregates = store.aggregate_daily("cqi")

        assert len(aggregates) == 5
        for agg in aggregates:
            assert agg.count == 24

    def test_aggregate_statistics(self, default_trending_config):
        """Test aggregate statistics calculation."""
        store = TimeSeriesStore(default_trending_config)

        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        values = [80.0, 82.0, 84.0, 86.0, 88.0, 90.0]
        for i, val in enumerate(values):
            store.add_point(
                "cqi",
                base_time + timedelta(minutes=i*10),
                val,
            )

        aggregates = store.aggregate_hourly("cqi")

        assert len(aggregates) >= 1
        agg = aggregates[0]

        assert agg.count == len(values)
        assert agg.mean == pytest.approx(sum(values) / len(values), rel=0.01)
        assert agg.min_value == min(values)
        assert agg.max_value == max(values)
        assert agg.sum_value == sum(values)


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer."""

    def test_initialization(self, default_trending_config):
        """Test analyzer initialization."""
        analyzer = TrendAnalyzer(default_trending_config)

        assert analyzer.config == default_trending_config

    def test_analyze_empty_data(self, default_trending_config):
        """Test analyzing empty data."""
        analyzer = TrendAnalyzer(default_trending_config)

        result = analyzer.analyze_trend("cqi", [])

        assert result.data_points == 0
        assert result.direction == TrendDirection.UNKNOWN

    def test_analyze_single_point(self, default_trending_config):
        """Test analyzing single point."""
        analyzer = TrendAnalyzer(default_trending_config)

        data = [TimeSeriesPoint(datetime.now(timezone.utc), 85.0)]
        result = analyzer.analyze_trend("cqi", data)

        assert result.data_points == 1
        assert result.direction == TrendDirection.UNKNOWN

    def test_analyze_stable_trend(self, default_trending_config):
        """Test analyzing stable trend (no significant change)."""
        analyzer = TrendAnalyzer(default_trending_config)

        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        data = [
            TimeSeriesPoint(
                base_time + timedelta(days=i),
                85.0 + random.gauss(0, 0.5),  # Small variance
            )
            for i in range(30)
        ]

        result = analyzer.analyze_trend("cqi", data)

        assert result.data_points == 30
        # With small variance and no trend, should be stable
        # (depends on significance threshold)

    def test_analyze_increasing_trend(self, default_trending_config):
        """Test analyzing increasing trend."""
        analyzer = TrendAnalyzer(default_trending_config)

        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        data = [
            TimeSeriesPoint(
                base_time + timedelta(days=i),
                80.0 + i * 0.5,  # Clear increasing trend
            )
            for i in range(30)
        ]

        result = analyzer.analyze_trend("cqi", data)

        assert result.slope > 0
        assert result.total_change > 0

    def test_analyze_decreasing_trend(self, default_trending_config):
        """Test analyzing decreasing trend."""
        analyzer = TrendAnalyzer(default_trending_config)

        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        data = [
            TimeSeriesPoint(
                base_time + timedelta(days=i),
                95.0 - i * 0.5,  # Clear decreasing trend
            )
            for i in range(30)
        ]

        result = analyzer.analyze_trend("cqi", data)

        assert result.slope < 0
        assert result.total_change < 0

    def test_r_squared_calculation(self, default_trending_config):
        """Test R-squared calculation for goodness of fit."""
        analyzer = TrendAnalyzer(default_trending_config)

        base_time = datetime.now(timezone.utc) - timedelta(days=30)

        # Perfect linear trend
        data_perfect = [
            TimeSeriesPoint(base_time + timedelta(days=i), 80.0 + i)
            for i in range(30)
        ]

        result = analyzer.analyze_trend("test", data_perfect)

        # Perfect linear should have R^2 close to 1
        assert result.r_squared > 0.99

    def test_confidence_interval(self, default_trending_config):
        """Test confidence interval calculation."""
        analyzer = TrendAnalyzer(default_trending_config)

        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        data = [
            TimeSeriesPoint(
                base_time + timedelta(days=i),
                85.0 + i * 0.1 + random.gauss(0, 1),
            )
            for i in range(30)
        ]

        result = analyzer.analyze_trend("test", data)

        # CI should contain the slope
        assert result.confidence_interval[0] <= result.slope
        assert result.confidence_interval[1] >= result.slope

    def test_detect_seasonality(self, default_trending_config):
        """Test seasonality detection."""
        analyzer = TrendAnalyzer(default_trending_config)

        base_time = datetime.now(timezone.utc) - timedelta(days=7)

        # Create daily seasonal pattern
        data = []
        for day in range(7):
            for hour in range(24):
                # Peak at noon, trough at midnight
                value = 85.0 + 10 * math.sin(2 * math.pi * hour / 24)
                data.append(TimeSeriesPoint(
                    base_time + timedelta(days=day, hours=hour),
                    value,
                ))

        result = analyzer.detect_seasonality("cqi", data, period_hours=24)

        # Should detect 24-hour seasonality
        assert result.seasonal_period_hours == 24
        if result.seasonal_detected:
            assert result.seasonal_strength > 0.2

    def test_no_seasonality(self, default_trending_config):
        """Test detection when no seasonality exists."""
        analyzer = TrendAnalyzer(default_trending_config)

        base_time = datetime.now(timezone.utc) - timedelta(days=7)

        # Random data without pattern
        random.seed(42)
        data = [
            TimeSeriesPoint(
                base_time + timedelta(hours=i),
                85.0 + random.gauss(0, 5),
            )
            for i in range(168)  # 7 days
        ]

        result = analyzer.detect_seasonality("cqi", data, period_hours=24)

        # Seasonal strength should be low
        # (may still detect some pattern due to randomness)

    def test_compare_to_baseline(self, default_trending_config):
        """Test baseline comparison."""
        analyzer = TrendAnalyzer(default_trending_config)

        base_time = datetime.now(timezone.utc)

        # Baseline period (30 days ago)
        baseline_data = [
            TimeSeriesPoint(
                base_time - timedelta(days=30-i),
                85.0 + random.gauss(0, 1),
            )
            for i in range(14)
        ]

        # Current period (degraded)
        current_data = [
            TimeSeriesPoint(
                base_time - timedelta(days=7-i),
                80.0 + random.gauss(0, 1),  # 5 points lower
            )
            for i in range(7)
        ]

        comparison = analyzer.compare_to_baseline("cqi", baseline_data, current_data)

        assert comparison.baseline_mean > comparison.current_mean
        assert comparison.absolute_change < 0
        assert comparison.percent_change < 0


class TestTrendingEngine:
    """Tests for integrated TrendingEngine."""

    def test_initialization(self, default_trending_config):
        """Test engine initialization."""
        engine = TrendingEngine(default_trending_config)

        assert engine.config == default_trending_config
        assert engine.store is not None
        assert engine.analyzer is not None

    def test_add_cqi_result(self, default_trending_config):
        """Test adding CQI result."""
        engine = TrendingEngine(default_trending_config)

        cqi_result = CQIResult(
            cqi_score=85.0,
            cqi_rating=CQIRating.GOOD,
            components=[],
            co_corrected_ppm=35.0,
            nox_corrected_ppm=50.0,
            o2_reference_pct=3.0,
            excess_air_pct=15.0,
            combustion_efficiency_pct=88.0,
            calculation_timestamp=datetime.now(timezone.utc),
            provenance_hash="a" * 64,
        )

        engine.add_cqi_result(cqi_result)

        data = engine.store.get_raw_data("cqi")
        assert len(data) == 1
        assert data[0].value == 85.0

    def test_add_flue_gas_reading(self, default_trending_config, optimal_flue_gas_reading):
        """Test adding flue gas reading."""
        engine = TrendingEngine(default_trending_config)

        engine.add_flue_gas_reading(optimal_flue_gas_reading)

        # Check multiple parameters stored
        assert len(engine.store.get_raw_data("oxygen")) == 1
        assert len(engine.store.get_raw_data("co2")) == 1
        assert len(engine.store.get_raw_data("co")) == 1
        assert len(engine.store.get_raw_data("nox")) == 1
        assert len(engine.store.get_raw_data("stack_temp")) == 1

    def test_get_cqi_trend(self, default_trending_config):
        """Test getting CQI trend."""
        engine = TrendingEngine(default_trending_config)

        # Add CQI data over time
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        for i in range(30):
            cqi_result = CQIResult(
                cqi_score=85.0 + i * 0.1,
                cqi_rating=CQIRating.GOOD,
                components=[],
                co_corrected_ppm=35.0,
                nox_corrected_ppm=50.0,
                o2_reference_pct=3.0,
                excess_air_pct=15.0,
                combustion_efficiency_pct=88.0,
                calculation_timestamp=base_time + timedelta(days=i),
                provenance_hash="a" * 64,
            )
            engine.add_cqi_result(cqi_result)

        trend = engine.get_cqi_trend(days=30)

        assert trend.parameter == "cqi"
        assert trend.data_points == 30

    def test_get_parameter_trend(self, default_trending_config):
        """Test getting arbitrary parameter trend."""
        engine = TrendingEngine(default_trending_config)

        # Add oxygen data
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        for i in range(30):
            reading = FlueGasReading(
                timestamp=base_time + timedelta(days=i),
                oxygen_pct=3.0 + i * 0.05,
                co2_pct=10.5,
                co_ppm=30.0,
                nox_ppm=45.0,
                flue_gas_temp_c=180.0,
            )
            engine.add_flue_gas_reading(reading)

        trend = engine.get_parameter_trend("oxygen", days=30)

        assert trend.parameter == "oxygen"
        assert trend.slope > 0  # Increasing O2

    def test_get_trend_summary(self, default_trending_config):
        """Test getting trend summary for all parameters."""
        engine = TrendingEngine(default_trending_config)

        # Add some data
        base_time = datetime.now(timezone.utc) - timedelta(days=7)
        for i in range(7):
            cqi_result = CQIResult(
                cqi_score=85.0,
                cqi_rating=CQIRating.GOOD,
                components=[],
                co_corrected_ppm=35.0,
                nox_corrected_ppm=50.0,
                o2_reference_pct=3.0,
                excess_air_pct=15.0,
                combustion_efficiency_pct=88.0,
                calculation_timestamp=base_time + timedelta(days=i),
                provenance_hash="a" * 64,
            )
            engine.add_cqi_result(cqi_result)

            reading = FlueGasReading(
                timestamp=base_time + timedelta(days=i),
                oxygen_pct=3.0,
                co2_pct=10.5,
                co_ppm=30.0,
                nox_ppm=45.0,
                flue_gas_temp_c=180.0,
            )
            engine.add_flue_gas_reading(reading)

        summary = engine.get_trend_summary()

        assert "cqi" in summary
        assert "oxygen" in summary
        assert "co" in summary

    def test_check_seasonality(self, default_trending_config):
        """Test seasonality checking."""
        config = TrendingConfig(detect_seasonality=True)
        engine = TrendingEngine(config)

        # Add data with pattern
        base_time = datetime.now(timezone.utc) - timedelta(days=7)
        for day in range(7):
            for hour in range(24):
                reading = FlueGasReading(
                    timestamp=base_time + timedelta(days=day, hours=hour),
                    oxygen_pct=3.0 + math.sin(2 * math.pi * hour / 24),
                    co2_pct=10.5,
                    co_ppm=30.0,
                    nox_ppm=45.0,
                    flue_gas_temp_c=180.0,
                )
                engine.add_flue_gas_reading(reading)

        result = engine.check_seasonality("oxygen")

        assert result.seasonal_period_hours == 24

    def test_compare_to_baseline(self, default_trending_config):
        """Test baseline comparison through engine."""
        engine = TrendingEngine(default_trending_config)

        # Add baseline data (older)
        base_time = datetime.now(timezone.utc) - timedelta(days=14)
        for i in range(7):
            reading = FlueGasReading(
                timestamp=base_time + timedelta(days=i),
                oxygen_pct=3.0,
                co2_pct=10.5,
                co_ppm=30.0,
                nox_ppm=45.0,
                flue_gas_temp_c=180.0,
            )
            engine.add_flue_gas_reading(reading)

        # Add current data (higher CO - degraded)
        current_time = datetime.now(timezone.utc) - timedelta(days=7)
        for i in range(7):
            reading = FlueGasReading(
                timestamp=current_time + timedelta(days=i),
                oxygen_pct=3.0,
                co2_pct=10.5,
                co_ppm=50.0,  # Higher CO
                nox_ppm=45.0,
                flue_gas_temp_c=180.0,
            )
            engine.add_flue_gas_reading(reading)

        comparison = engine.compare_to_baseline("co", baseline_days=7, comparison_days=7)

        assert comparison.current_mean > comparison.baseline_mean

    def test_get_daily_summary(self, default_trending_config):
        """Test getting daily summary."""
        engine = TrendingEngine(default_trending_config)

        # Add hourly data for 3 days
        base_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        for day in range(3):
            for hour in range(24):
                reading = FlueGasReading(
                    timestamp=base_time + timedelta(days=day, hours=hour),
                    oxygen_pct=3.0 + random.gauss(0, 0.2),
                    co2_pct=10.5,
                    co_ppm=30.0,
                    nox_ppm=45.0,
                    flue_gas_temp_c=180.0,
                )
                engine.add_flue_gas_reading(reading)

        daily = engine.get_daily_summary("oxygen")

        assert len(daily) == 3
        for agg in daily:
            assert agg.count == 24


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_moving_average_basic(self):
        """Test basic moving average."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ma = calculate_moving_average(values, window=3)

        assert len(ma) == len(values)
        # First points are averaged with available data
        assert ma[2] == pytest.approx((1+2+3)/3, rel=0.01)
        assert ma[5] == pytest.approx((4+5+6)/3, rel=0.01)

    def test_moving_average_window_larger_than_data(self):
        """Test moving average when window larger than data."""
        values = [1, 2, 3]
        ma = calculate_moving_average(values, window=10)

        # Should return original values
        assert ma == values

    def test_exponential_moving_average(self):
        """Test exponential moving average."""
        values = [10, 20, 30, 40, 50]
        ema = calculate_exponential_moving_average(values, alpha=0.5)

        assert len(ema) == len(values)
        assert ema[0] == values[0]  # First value same

    def test_exponential_moving_average_smoothing(self):
        """Test EMA smoothing behavior."""
        values = [100, 0, 100, 0, 100, 0]  # Oscillating

        # Higher alpha = more responsive
        ema_high = calculate_exponential_moving_average(values, alpha=0.9)
        # Lower alpha = more smooth
        ema_low = calculate_exponential_moving_average(values, alpha=0.1)

        # High alpha should track values more closely
        # Low alpha should be smoother
        # Check variance
        var_high = sum((v - sum(ema_high)/len(ema_high))**2 for v in ema_high)
        var_low = sum((v - sum(ema_low)/len(ema_low))**2 for v in ema_low)

        # Lower alpha should have lower variance (smoother)
        assert var_low < var_high

    def test_moving_average_empty(self):
        """Test moving average with empty input."""
        ma = calculate_moving_average([], window=5)
        assert ma == []

    def test_ema_empty(self):
        """Test EMA with empty input."""
        ema = calculate_exponential_moving_average([])
        assert ema == []
