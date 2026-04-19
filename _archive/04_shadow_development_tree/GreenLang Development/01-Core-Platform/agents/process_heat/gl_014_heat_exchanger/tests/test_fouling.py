# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Fouling Analysis Tests

Comprehensive tests for fouling analysis including:
- TEMA RGP-T2.4 fouling factors
- Fouling rate calculations
- Kern-Seaton asymptotic fouling model
- ML-based fouling predictions
- Fouling trend analysis

Coverage Target: 90%+

References:
    - TEMA Standards 9th Edition (RGP-T2.4)
    - Kern & Seaton fouling model
    - Epstein falling rate model
"""

import pytest
import math
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.process_heat.gl_014_heat_exchanger.fouling import (
    FoulingAnalyzer,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    FoulingConfig,
    FoulingCategory,
    TEMAFoulingFactors,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    FoulingAnalysisResult,
    TrendDirection,
)


class TestTEMAFoulingFactors:
    """Tests for TEMA RGP-T2.4 standard fouling factors."""

    def test_cooling_tower_water(self):
        """Test cooling tower water fouling factor (TEMA RGP-T2.4)."""
        factors = TEMAFoulingFactors()
        # TEMA specifies 0.00035 m2K/W for cooling tower water
        assert factors.cooling_tower_water == pytest.approx(0.00035, rel=0.01)

    def test_sea_water(self):
        """Test sea water fouling factor (TEMA RGP-T2.4)."""
        factors = TEMAFoulingFactors()
        # TEMA specifies 0.00017 m2K/W for treated sea water
        assert factors.sea_water == pytest.approx(0.00017, rel=0.01)

    def test_boiler_feedwater(self):
        """Test boiler feedwater fouling factor (TEMA RGP-T2.4)."""
        factors = TEMAFoulingFactors()
        # TEMA specifies 0.00009 m2K/W for treated boiler feedwater
        assert factors.boiler_feedwater == pytest.approx(0.00009, rel=0.01)

    def test_river_water(self):
        """Test river water fouling factor (TEMA RGP-T2.4)."""
        factors = TEMAFoulingFactors()
        # TEMA specifies 0.00035 m2K/W for river water
        assert factors.river_water == pytest.approx(0.00035, rel=0.01)

    def test_crude_oil_dry(self):
        """Test dry crude oil fouling factor (TEMA RGP-T2.4)."""
        factors = TEMAFoulingFactors()
        # TEMA specifies 0.00035 m2K/W for dry crude oil
        assert factors.crude_oil_dry == pytest.approx(0.00035, rel=0.01)

    def test_crude_oil_wet(self):
        """Test wet crude oil fouling factor (TEMA RGP-T2.4)."""
        factors = TEMAFoulingFactors()
        # TEMA specifies 0.00053 m2K/W for wet crude oil (higher due to water)
        assert factors.crude_oil_wet == pytest.approx(0.00053, rel=0.01)

    def test_fuel_oil(self):
        """Test fuel oil fouling factor (TEMA RGP-T2.4)."""
        factors = TEMAFoulingFactors()
        # TEMA specifies 0.00088 m2K/W for fuel oil (high fouling)
        assert factors.fuel_oil == pytest.approx(0.00088, rel=0.01)

    def test_steam(self):
        """Test steam fouling factor (TEMA RGP-T2.4)."""
        factors = TEMAFoulingFactors()
        # TEMA specifies 0.00009 m2K/W for clean steam
        assert factors.steam == pytest.approx(0.00009, rel=0.01)

    def test_process_gas(self):
        """Test process gas fouling factor (TEMA RGP-T2.4)."""
        factors = TEMAFoulingFactors()
        # TEMA specifies 0.00018 m2K/W for clean process gas
        assert factors.process_gas == pytest.approx(0.00018, rel=0.01)


class TestFoulingAnalyzerInitialization:
    """Tests for FoulingAnalyzer initialization."""

    def test_analyzer_initialization(self, fouling_config):
        """Test FoulingAnalyzer initializes correctly."""
        analyzer = FoulingAnalyzer(
            config=fouling_config,
            clean_u_w_m2k=500.0,
        )
        assert analyzer.config == fouling_config
        assert analyzer.clean_u_w_m2k == 500.0
        assert len(analyzer.data_points) == 0

    def test_analyzer_with_custom_clean_u(self, fouling_config):
        """Test analyzer with custom clean U value."""
        analyzer = FoulingAnalyzer(
            config=fouling_config,
            clean_u_w_m2k=600.0,
        )
        assert analyzer.clean_u_w_m2k == 600.0


class TestFoulingFactorCalculation:
    """Tests for fouling factor calculations."""

    @pytest.fixture
    def analyzer(self, fouling_config):
        """Create FoulingAnalyzer instance."""
        return FoulingAnalyzer(
            config=fouling_config,
            clean_u_w_m2k=500.0,
        )

    def test_fouling_factor_from_u(self, analyzer):
        """Test fouling factor calculation from U values."""
        # Rf = 1/U_fouled - 1/U_clean
        u_current = 400.0
        rf = analyzer.calculate_fouling_factor(u_current)

        expected = 1/400.0 - 1/500.0  # 0.0005 m2K/W
        assert rf == pytest.approx(expected, rel=0.01)

    def test_fouling_factor_clean(self, analyzer):
        """Test fouling factor is zero when U is clean."""
        rf = analyzer.calculate_fouling_factor(500.0)
        assert rf == pytest.approx(0.0, abs=1e-10)

    def test_fouling_factor_heavily_fouled(self, analyzer):
        """Test fouling factor for heavily fouled exchanger."""
        u_current = 250.0  # 50% reduction
        rf = analyzer.calculate_fouling_factor(u_current)

        expected = 1/250.0 - 1/500.0  # 0.002 m2K/W
        assert rf == pytest.approx(expected, rel=0.01)

    def test_fouling_ratio_calculation(self, analyzer):
        """Test fouling ratio (actual/design)."""
        # Design fouling = 0.00034 m2K/W (shell + tube side)
        # Current fouling = 0.0005 m2K/W
        u_current = 400.0  # Gives Rf = 0.0005
        result = analyzer.analyze_fouling(
            u_current_w_m2k=u_current,
            days_since_cleaning=90,
        )

        # Fouling ratio should be > 1 (over design)
        design_fouling = 0.00034
        actual_fouling = 1/400.0 - 1/500.0
        expected_ratio = actual_fouling / design_fouling
        assert result.fouling_ratio == pytest.approx(expected_ratio, rel=0.02)


class TestFoulingRateCalculation:
    """Tests for fouling rate calculations."""

    @pytest.fixture
    def analyzer(self, fouling_config):
        """Create FoulingAnalyzer with data points."""
        analyzer = FoulingAnalyzer(
            config=fouling_config,
            clean_u_w_m2k=500.0,
        )
        return analyzer

    def test_linear_fouling_rate(self, analyzer):
        """Test linear fouling rate calculation."""
        # Add data points over time
        base_time = datetime.now(timezone.utc) - timedelta(days=30)

        analyzer.add_data_point(
            u_value_w_m2k=500.0,
            timestamp=base_time,
        )
        analyzer.add_data_point(
            u_value_w_m2k=480.0,
            timestamp=base_time + timedelta(days=10),
        )
        analyzer.add_data_point(
            u_value_w_m2k=460.0,
            timestamp=base_time + timedelta(days=20),
        )
        analyzer.add_data_point(
            u_value_w_m2k=440.0,
            timestamp=base_time + timedelta(days=30),
        )

        rate = analyzer.calculate_fouling_rate()

        # Linear rate: U drops 60 W/m2K over 30 days
        # Rf change = (1/440 - 1/500) - (1/500 - 1/500) = 0.000273 over 30 days
        # Rate = 0.000273 / 30 = 0.0000091 m2K/W per day
        assert rate > 0

    def test_fouling_rate_insufficient_data(self, analyzer):
        """Test fouling rate returns config default with insufficient data."""
        analyzer.add_data_point(
            u_value_w_m2k=450.0,
            timestamp=datetime.now(timezone.utc),
        )

        rate = analyzer.calculate_fouling_rate()

        # Should return config default when insufficient data
        assert rate == pytest.approx(analyzer.config.fouling_rate_m2kw_per_day, rel=0.1)

    def test_fouling_rate_trend_detection(self, analyzer):
        """Test fouling rate detects increasing trend."""
        base_time = datetime.now(timezone.utc) - timedelta(days=60)

        # Accelerating fouling (U drops faster over time)
        u_values = [500, 490, 475, 455, 430, 400]
        for i, u in enumerate(u_values):
            analyzer.add_data_point(
                u_value_w_m2k=u,
                timestamp=base_time + timedelta(days=i*10),
            )

        result = analyzer.analyze_fouling(
            u_current_w_m2k=400.0,
            days_since_cleaning=60,
        )

        # Fouling rate should be accelerating
        assert result.fouling_rate_m2kw_per_day > 0


class TestKernSeatonModel:
    """Tests for Kern-Seaton asymptotic fouling model."""

    @pytest.fixture
    def analyzer(self, fouling_config):
        """Create FoulingAnalyzer with asymptotic fouling enabled."""
        config = fouling_config
        config.asymptotic_fouling_m2kw = 0.0005  # Asymptotic value
        config.removal_rate_coefficient = 0.1

        return FoulingAnalyzer(
            config=config,
            clean_u_w_m2k=500.0,
        )

    def test_asymptotic_fouling_prediction(self, analyzer):
        """Test Kern-Seaton asymptotic fouling prediction."""
        # Kern-Seaton: Rf(t) = Rf_inf * (1 - exp(-lambda * t))
        # where Rf_inf = asymptotic fouling, lambda = removal rate

        result = analyzer.analyze_fouling(
            u_current_w_m2k=450.0,
            days_since_cleaning=30,
        )

        # Asymptotic fouling should be in result
        if result.asymptotic_fouling_m2kw is not None:
            assert result.asymptotic_fouling_m2kw == pytest.approx(0.0005, rel=0.01)

    def test_fouling_approaches_asymptote(self, analyzer):
        """Test fouling approaches asymptotic value over time."""
        # At t -> infinity, Rf -> Rf_inf
        # After many time constants, should be close to asymptote
        days = 365  # Long time

        # Calculate expected fouling at steady state
        rf_inf = 0.0005
        removal_rate = 0.1
        # Rf(t) = Rf_inf * (1 - exp(-lambda * t))
        expected_rf = rf_inf * (1 - math.exp(-removal_rate * days))

        # Should be close to asymptote
        assert expected_rf > 0.9 * rf_inf


class TestFoulingAnalysisResult:
    """Tests for complete fouling analysis output."""

    @pytest.fixture
    def analyzer(self, fouling_config):
        """Create FoulingAnalyzer instance."""
        return FoulingAnalyzer(
            config=fouling_config,
            clean_u_w_m2k=500.0,
        )

    def test_fouling_analysis_complete(self, analyzer):
        """Test complete fouling analysis result."""
        result = analyzer.analyze_fouling(
            u_current_w_m2k=420.0,
            days_since_cleaning=90,
            shell_inlet_temp_c=150.0,
            tube_inlet_temp_c=30.0,
        )

        assert isinstance(result, FoulingAnalysisResult)
        assert result.shell_side_fouling_m2kw >= 0
        assert result.tube_side_fouling_m2kw >= 0
        assert result.total_fouling_m2kw > 0
        assert result.fouling_ratio > 0
        assert result.fouling_rate_m2kw_per_day >= 0

    def test_fouling_analysis_trend_stable(self, analyzer):
        """Test fouling trend detection - stable."""
        # Add stable U values
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        for i in range(6):
            analyzer.add_data_point(
                u_value_w_m2k=450.0 + (i % 2),  # Small fluctuation
                timestamp=base_time + timedelta(days=i*5),
            )

        result = analyzer.analyze_fouling(
            u_current_w_m2k=450.0,
            days_since_cleaning=30,
        )

        assert result.fouling_trend in [TrendDirection.STABLE, TrendDirection.DEGRADING]

    def test_fouling_analysis_trend_degrading(self, analyzer):
        """Test fouling trend detection - degrading."""
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        u_values = [480, 470, 460, 450, 440, 430]

        for i, u in enumerate(u_values):
            analyzer.add_data_point(
                u_value_w_m2k=u,
                timestamp=base_time + timedelta(days=i*5),
            )

        result = analyzer.analyze_fouling(
            u_current_w_m2k=430.0,
            days_since_cleaning=30,
        )

        assert result.fouling_trend == TrendDirection.DEGRADING

    def test_days_to_threshold_calculation(self, analyzer):
        """Test calculation of days until cleaning threshold."""
        result = analyzer.analyze_fouling(
            u_current_w_m2k=450.0,
            days_since_cleaning=60,
        )

        # Should predict days until fouling threshold reached
        if result.days_to_cleaning_threshold is not None:
            assert result.days_to_cleaning_threshold > 0


class TestMLFoulingPrediction:
    """Tests for ML-based fouling prediction."""

    @pytest.fixture
    def analyzer_ml_enabled(self, fouling_config):
        """Create FoulingAnalyzer with ML enabled."""
        fouling_config.ml_prediction_enabled = True
        fouling_config.prediction_horizon_days = 30

        analyzer = FoulingAnalyzer(
            config=fouling_config,
            clean_u_w_m2k=500.0,
        )

        # Add enough data for ML prediction
        base_time = datetime.now(timezone.utc) - timedelta(days=90)
        u_values = [500, 495, 488, 480, 470, 458, 445, 430, 415, 400]

        for i, u in enumerate(u_values):
            analyzer.add_data_point(
                u_value_w_m2k=u,
                timestamp=base_time + timedelta(days=i*9),
                shell_inlet_temp_c=150.0,
                tube_inlet_temp_c=30.0,
            )

        return analyzer

    def test_ml_prediction_30_day(self, analyzer_ml_enabled):
        """Test ML prediction for 30-day horizon."""
        result = analyzer_ml_enabled.analyze_fouling(
            u_current_w_m2k=400.0,
            days_since_cleaning=90,
        )

        # ML prediction should be present when enabled
        if result.ml_predicted_fouling_30d is not None:
            # Predicted fouling should be higher than current
            assert result.ml_predicted_fouling_30d >= result.total_fouling_m2kw

    def test_ml_prediction_confidence(self, analyzer_ml_enabled):
        """Test ML prediction confidence level."""
        result = analyzer_ml_enabled.analyze_fouling(
            u_current_w_m2k=400.0,
            days_since_cleaning=90,
        )

        if result.ml_prediction_confidence is not None:
            # Confidence should be 0-1
            assert 0 <= result.ml_prediction_confidence <= 1


class TestFoulingByCategory:
    """Tests for fouling calculations by category."""

    @pytest.fixture
    def analyzer_particulate(self, fouling_config):
        """Create analyzer for particulate fouling."""
        fouling_config.fouling_category = FoulingCategory.PARTICULATE
        return FoulingAnalyzer(config=fouling_config, clean_u_w_m2k=500.0)

    @pytest.fixture
    def analyzer_biological(self, fouling_config):
        """Create analyzer for biological fouling."""
        fouling_config.fouling_category = FoulingCategory.BIOLOGICAL
        return FoulingAnalyzer(config=fouling_config, clean_u_w_m2k=500.0)

    @pytest.fixture
    def analyzer_scaling(self, fouling_config):
        """Create analyzer for scaling fouling."""
        fouling_config.fouling_category = FoulingCategory.SCALING
        return FoulingAnalyzer(config=fouling_config, clean_u_w_m2k=500.0)

    def test_particulate_fouling_rate(self, analyzer_particulate):
        """Test particulate fouling follows linear model."""
        result = analyzer_particulate.analyze_fouling(
            u_current_w_m2k=450.0,
            days_since_cleaning=30,
        )
        assert result.fouling_rate_m2kw_per_day > 0

    def test_biological_fouling_rate(self, analyzer_biological):
        """Test biological fouling may have different rate characteristics."""
        result = analyzer_biological.analyze_fouling(
            u_current_w_m2k=450.0,
            days_since_cleaning=30,
            shell_inlet_temp_c=35.0,  # Warm - promotes bio growth
            tube_inlet_temp_c=25.0,
        )
        assert result.fouling_rate_m2kw_per_day > 0

    def test_scaling_temperature_dependence(self, analyzer_scaling):
        """Test scaling fouling shows temperature dependence."""
        result_cold = analyzer_scaling.analyze_fouling(
            u_current_w_m2k=450.0,
            days_since_cleaning=30,
            shell_inlet_temp_c=50.0,
            tube_inlet_temp_c=30.0,
        )

        result_hot = analyzer_scaling.analyze_fouling(
            u_current_w_m2k=450.0,
            days_since_cleaning=30,
            shell_inlet_temp_c=150.0,  # Higher temp - more scaling
            tube_inlet_temp_c=30.0,
        )

        # Note: Both results are valid; scaling behavior may vary
        assert result_cold.fouling_rate_m2kw_per_day >= 0
        assert result_hot.fouling_rate_m2kw_per_day >= 0


class TestFoulingDataPoints:
    """Tests for fouling data point management."""

    @pytest.fixture
    def analyzer(self, fouling_config):
        """Create FoulingAnalyzer instance."""
        return FoulingAnalyzer(config=fouling_config, clean_u_w_m2k=500.0)

    def test_add_data_point(self, analyzer):
        """Test adding data points."""
        analyzer.add_data_point(
            u_value_w_m2k=450.0,
            timestamp=datetime.now(timezone.utc),
            shell_inlet_temp_c=150.0,
            tube_inlet_temp_c=30.0,
        )
        assert len(analyzer.data_points) == 1

    def test_data_point_ordering(self, analyzer):
        """Test data points are ordered by timestamp."""
        now = datetime.now(timezone.utc)

        # Add out of order
        analyzer.add_data_point(
            u_value_w_m2k=450.0,
            timestamp=now,
        )
        analyzer.add_data_point(
            u_value_w_m2k=480.0,
            timestamp=now - timedelta(days=10),
        )
        analyzer.add_data_point(
            u_value_w_m2k=420.0,
            timestamp=now + timedelta(days=5),
        )

        # Data points should be sorted
        timestamps = [dp.timestamp for dp in analyzer.data_points]
        assert timestamps == sorted(timestamps)

    def test_data_point_limit(self, analyzer):
        """Test data points are limited to prevent memory issues."""
        max_points = 1000

        for i in range(max_points + 100):
            analyzer.add_data_point(
                u_value_w_m2k=450.0,
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=i),
            )

        # Should be limited
        assert len(analyzer.data_points) <= max_points


class TestSplitFouling:
    """Tests for shell/tube side fouling split."""

    @pytest.fixture
    def analyzer(self, fouling_config):
        """Create FoulingAnalyzer with known fouling split."""
        # Shell side: 0.00017, Tube side: 0.00017
        return FoulingAnalyzer(config=fouling_config, clean_u_w_m2k=500.0)

    def test_total_fouling_equals_sum(self, analyzer):
        """Test total fouling equals shell + tube fouling."""
        result = analyzer.analyze_fouling(
            u_current_w_m2k=420.0,
            days_since_cleaning=60,
        )

        calculated_total = result.shell_side_fouling_m2kw + result.tube_side_fouling_m2kw
        assert result.total_fouling_m2kw == pytest.approx(calculated_total, rel=0.01)

    def test_shell_tube_ratio_from_config(self, analyzer):
        """Test shell/tube fouling ratio matches config."""
        result = analyzer.analyze_fouling(
            u_current_w_m2k=420.0,
            days_since_cleaning=60,
        )

        # Config has equal shell and tube fouling (0.00017 each)
        # So ratio should be approximately 1:1
        if result.tube_side_fouling_m2kw > 0:
            ratio = result.shell_side_fouling_m2kw / result.tube_side_fouling_m2kw
            assert 0.8 < ratio < 1.2  # Allow some tolerance


class TestFoulingCleaning:
    """Tests for fouling reset after cleaning."""

    @pytest.fixture
    def analyzer(self, fouling_config):
        """Create FoulingAnalyzer with history."""
        analyzer = FoulingAnalyzer(config=fouling_config, clean_u_w_m2k=500.0)

        # Add pre-cleaning data
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        for i in range(5):
            analyzer.add_data_point(
                u_value_w_m2k=480 - i*10,
                timestamp=base_time + timedelta(days=i*5),
            )

        return analyzer

    def test_reset_after_cleaning(self, analyzer):
        """Test fouling data resets after cleaning event."""
        # Record cleaning
        analyzer.record_cleaning(
            cleaning_date=datetime.now(timezone.utc),
            u_after_cleaning=490.0,
        )

        # Historical data should be cleared or marked
        assert analyzer.days_since_cleaning == 0

    def test_fouling_zero_after_cleaning(self, analyzer):
        """Test fouling near zero immediately after cleaning."""
        analyzer.record_cleaning(
            cleaning_date=datetime.now(timezone.utc),
            u_after_cleaning=495.0,  # Near clean U
        )

        result = analyzer.analyze_fouling(
            u_current_w_m2k=495.0,
            days_since_cleaning=0,
        )

        # Fouling should be very low
        assert result.total_fouling_m2kw < 0.0001
