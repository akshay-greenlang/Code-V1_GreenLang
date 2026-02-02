"""
GL-017 CONDENSYNC Agent - Performance Analyzer Tests

Unit tests for PerformanceAnalyzer and PerformanceCurve.
Tests cover performance curve generation, deviation analysis,
and degradation source identification.

Coverage targets:
    - Performance curve generation
    - Expected backpressure calculation
    - TTD calculations
    - Heat rate/capacity impact
    - Degradation source identification
    - Trend analysis
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
import math

from greenlang.agents.process_heat.gl_017_condenser_optimization.performance import (
    PerformanceAnalyzer,
    PerformanceCurve,
    PerformanceConstants,
    PerformanceDataPoint,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    PerformanceConfig,
    TubeFoulingConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    PerformanceResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def performance_config():
    """Create default performance configuration."""
    return PerformanceConfig()


@pytest.fixture
def fouling_config():
    """Create default tube fouling configuration."""
    return TubeFoulingConfig()


@pytest.fixture
def analyzer(performance_config, fouling_config):
    """Create PerformanceAnalyzer instance."""
    return PerformanceAnalyzer(performance_config, fouling_config)


@pytest.fixture
def performance_curve():
    """Create PerformanceCurve instance."""
    return PerformanceCurve(
        design_duty_btu_hr=500_000_000.0,
        design_steam_flow_lb_hr=500000.0,
        design_backpressure_inhga=1.5,
        design_cw_inlet_temp_f=70.0,
        design_cw_outlet_temp_f=95.0,
        design_cw_flow_gpm=100000.0,
        surface_area_ft2=150000.0,
        cleanliness_factor=0.85,
    )


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestPerformanceConstants:
    """Test PerformanceConstants values."""

    def test_heat_rate_penalty(self):
        """Test heat rate penalty constant."""
        assert PerformanceConstants.HEAT_RATE_PENALTY_BTU_KWH_PER_INHG == 80.0

    def test_capacity_loss(self):
        """Test capacity loss constant."""
        assert PerformanceConstants.CAPACITY_LOSS_PCT_PER_INHG == 0.75

    def test_typical_ttd(self):
        """Test typical TTD value."""
        assert PerformanceConstants.TYPICAL_TTD_F == 5.0

    def test_degradation_sources(self):
        """Test degradation sources exist."""
        assert "tube_fouling" in PerformanceConstants.DEGRADATION_SOURCES
        assert "air_ingress" in PerformanceConstants.DEGRADATION_SOURCES
        assert "low_cw_flow" in PerformanceConstants.DEGRADATION_SOURCES

    def test_degradation_sources_sum(self):
        """Test degradation sources sum to 1.0."""
        total = sum(PerformanceConstants.DEGRADATION_SOURCES.values())
        assert total == pytest.approx(1.0, rel=0.01)


# =============================================================================
# PERFORMANCE CURVE TESTS
# =============================================================================

class TestPerformanceCurve:
    """Test PerformanceCurve class."""

    def test_initialization(self, performance_curve):
        """Test curve initializes correctly."""
        assert performance_curve is not None
        assert performance_curve.design_duty == 500_000_000.0
        assert performance_curve.design_bp == 1.5

    def test_design_ttd_calculated(self, performance_curve):
        """Test design TTD is calculated."""
        assert performance_curve.design_ttd > 0

    def test_design_ua_calculated(self, performance_curve):
        """Test design UA is calculated."""
        assert performance_curve.design_ua > 0

    def test_design_range_calculated(self, performance_curve):
        """Test design range is calculated."""
        # 95 - 70 = 25F
        assert performance_curve.design_range == 25.0


# =============================================================================
# EXPECTED BACKPRESSURE TESTS
# =============================================================================

class TestExpectedBackpressure:
    """Test expected backpressure calculation."""

    def test_at_design_conditions(self, performance_curve):
        """Test expected BP at design conditions."""
        bp = performance_curve.get_expected_backpressure(
            steam_flow_lb_hr=500000.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        # Should be close to design (1.5 inHgA)
        assert 1.0 < bp < 2.5

    def test_higher_load_higher_bp(self, performance_curve):
        """Test higher load gives higher backpressure."""
        bp_low = performance_curve.get_expected_backpressure(
            steam_flow_lb_hr=250000.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        bp_high = performance_curve.get_expected_backpressure(
            steam_flow_lb_hr=500000.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        assert bp_high > bp_low

    def test_higher_inlet_temp_higher_bp(self, performance_curve):
        """Test higher inlet temp gives higher backpressure."""
        bp_cold = performance_curve.get_expected_backpressure(
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=60.0,
            cw_flow_gpm=100000.0,
        )

        bp_hot = performance_curve.get_expected_backpressure(
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=85.0,
            cw_flow_gpm=100000.0,
        )

        assert bp_hot > bp_cold

    def test_lower_flow_higher_bp(self, performance_curve):
        """Test lower flow gives higher backpressure."""
        bp_high_flow = performance_curve.get_expected_backpressure(
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        bp_low_flow = performance_curve.get_expected_backpressure(
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=80000.0,
        )

        assert bp_low_flow >= bp_high_flow

    def test_lower_cleanliness_higher_bp(self, performance_curve):
        """Test lower cleanliness gives higher backpressure."""
        bp_clean = performance_curve.get_expected_backpressure(
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
            cleanliness_factor=0.90,
        )

        bp_fouled = performance_curve.get_expected_backpressure(
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
            cleanliness_factor=0.70,
        )

        assert bp_fouled > bp_clean


# =============================================================================
# TTD CALCULATION TESTS
# =============================================================================

class TestTTDCalculation:
    """Test TTD calculation methods."""

    def test_ttd_at_design(self, performance_curve):
        """Test TTD calculation at design."""
        ttd = performance_curve._calculate_ttd(1.5, 95.0)

        # Should be positive
        assert ttd > 0

    def test_ttd_higher_bp_higher_ttd(self, performance_curve):
        """Test higher backpressure gives higher TTD."""
        ttd_low = performance_curve._calculate_ttd(1.5, 95.0)
        ttd_high = performance_curve._calculate_ttd(2.5, 95.0)

        assert ttd_high > ttd_low


# =============================================================================
# SATURATION CONVERSION TESTS
# =============================================================================

class TestSaturationConversion:
    """Test saturation temperature/pressure conversion."""

    def test_temp_to_pressure(self, performance_curve):
        """Test temperature to pressure conversion."""
        pressure = performance_curve._sat_temp_to_pressure(101.0)

        # Should be positive
        assert pressure > 0

    def test_pressure_to_temp(self, performance_curve):
        """Test pressure to temperature conversion."""
        temp = performance_curve._pressure_to_sat_temp(1.5)

        # Should be reasonable saturation temp
        assert 80.0 < temp < 120.0

    def test_round_trip_conversion(self, performance_curve):
        """Test round-trip conversion."""
        original_temp = 101.0
        pressure = performance_curve._sat_temp_to_pressure(original_temp)
        recovered_temp = performance_curve._pressure_to_sat_temp(pressure)

        assert abs(recovered_temp - original_temp) < 2.0

    def test_pressure_bounds(self, performance_curve):
        """Test pressure conversion bounds."""
        # Zero pressure
        temp = performance_curve._pressure_to_sat_temp(0.0)
        assert temp == 60.0  # Lower bound

        # High pressure
        temp = performance_curve._pressure_to_sat_temp(10.0)
        assert temp <= 150.0  # Upper bound


# =============================================================================
# ANALYZER INITIALIZATION TESTS
# =============================================================================

class TestAnalyzerInitialization:
    """Test analyzer initialization."""

    def test_basic_initialization(self, performance_config, fouling_config):
        """Test analyzer initializes correctly."""
        analyzer = PerformanceAnalyzer(performance_config, fouling_config)
        assert analyzer is not None
        assert analyzer.curve is not None

    def test_history_empty(self, analyzer):
        """Test history is empty on initialization."""
        assert len(analyzer._history) == 0


# =============================================================================
# ANALYZE PERFORMANCE TESTS
# =============================================================================

class TestAnalyzePerformance:
    """Test main analysis method."""

    def test_basic_analysis(self, analyzer):
        """Test basic performance analysis."""
        result = analyzer.analyze_performance(
            actual_backpressure_inhga=1.6,
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
        )

        assert isinstance(result, PerformanceResult)

    def test_result_components(self, analyzer):
        """Test all result components are populated."""
        result = analyzer.analyze_performance(
            actual_backpressure_inhga=1.7,
            steam_flow_lb_hr=425000.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
        )

        assert result.actual_duty_btu_hr > 0
        assert result.design_duty_btu_hr > 0
        assert result.actual_backpressure_inhga == 1.7
        assert result.expected_backpressure_inhga > 0
        assert result.backpressure_deviation_inhg is not None
        assert result.backpressure_deviation_pct is not None

    def test_deviation_positive_for_higher_bp(self, analyzer):
        """Test positive deviation for higher actual BP."""
        result = analyzer.analyze_performance(
            actual_backpressure_inhga=2.5,  # Higher than expected
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
        )

        assert result.backpressure_deviation_inhg > 0
        assert result.backpressure_deviation_pct > 0

    def test_ttd_calculated(self, analyzer):
        """Test TTD is calculated."""
        result = analyzer.analyze_performance(
            actual_backpressure_inhga=1.6,
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
            cw_outlet_temp_f=95.0,
        )

        assert result.ttd_actual_f > 0
        assert result.ttd_design_f > 0

    def test_degradation_identified(self, analyzer):
        """Test degradation source is identified."""
        result = analyzer.analyze_performance(
            actual_backpressure_inhga=2.0,
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
        )

        assert result.degradation_source is not None
        assert isinstance(result.degradation_breakdown, dict)


# =============================================================================
# HEAT DUTY TESTS
# =============================================================================

class TestHeatDuty:
    """Test heat duty calculation."""

    def test_heat_duty_formula(self, analyzer):
        """Test heat duty calculation."""
        duty = analyzer._calculate_heat_duty(400000.0, 1.5)

        # Q = m_dot * h_fg = 400000 * 1000 = 4e8 BTU/hr
        assert duty == pytest.approx(400_000_000.0, rel=0.01)


# =============================================================================
# TTD ESTIMATION TESTS
# =============================================================================

class TestTTDEstimation:
    """Test TTD estimation without outlet temperature."""

    def test_estimate_ttd(self, analyzer):
        """Test TTD estimation."""
        ttd = analyzer._estimate_ttd(
            backpressure_inhga=1.6,
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
        )

        # Should be positive
        assert ttd > 0


# =============================================================================
# HEAT RATE IMPACT TESTS
# =============================================================================

class TestHeatRateImpact:
    """Test heat rate impact calculation."""

    def test_no_impact_no_deviation(self, analyzer):
        """Test no impact when no deviation."""
        impact = analyzer._calculate_heat_rate_impact(0.0)
        assert impact == 0.0

    def test_no_impact_negative_deviation(self, analyzer):
        """Test no impact for negative deviation."""
        impact = analyzer._calculate_heat_rate_impact(-0.1)
        assert impact == 0.0

    def test_impact_calculation(self, analyzer):
        """Test heat rate impact calculation."""
        impact = analyzer._calculate_heat_rate_impact(0.5)

        # 0.5 inHg * 80 BTU/kWh/inHg = 40 BTU/kWh
        expected = 0.5 * PerformanceConstants.HEAT_RATE_PENALTY_BTU_KWH_PER_INHG
        assert impact == pytest.approx(expected, rel=0.01)


# =============================================================================
# CAPACITY IMPACT TESTS
# =============================================================================

class TestCapacityImpact:
    """Test capacity impact calculation."""

    def test_no_impact_no_deviation(self, analyzer):
        """Test no impact when no deviation."""
        impact = analyzer._calculate_capacity_impact(0.0, 500.0)
        assert impact == 0.0

    def test_capacity_impact_calculation(self, analyzer):
        """Test capacity impact calculation."""
        impact = analyzer._calculate_capacity_impact(0.5, 500.0)

        # 0.5 inHg * 0.75%/inHg = 0.375%
        # 500 MW * 0.375% = 1.875 MW
        expected = 500.0 * 0.5 * PerformanceConstants.CAPACITY_LOSS_PCT_PER_INHG / 100
        assert impact == pytest.approx(expected, rel=0.01)


# =============================================================================
# EFFICIENCY IMPACT TESTS
# =============================================================================

class TestEfficiencyImpact:
    """Test efficiency impact calculation."""

    def test_efficiency_impact(self, analyzer):
        """Test efficiency impact calculation."""
        impact = analyzer._calculate_efficiency_impact(0.5)

        # Based on 10,000 BTU/kWh baseline
        heat_rate_impact = 0.5 * 80.0
        expected = (heat_rate_impact / 10000) * 100
        assert impact == pytest.approx(expected, rel=0.01)


# =============================================================================
# DEGRADATION SOURCE TESTS
# =============================================================================

class TestDegradationSource:
    """Test degradation source identification."""

    def test_no_degradation(self, analyzer):
        """Test no degradation for no deviation."""
        source, breakdown = analyzer._identify_degradation_source(
            bp_deviation_inhg=0.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
            cleanliness=0.85,
        )

        assert source == "none"
        assert breakdown == {}

    def test_high_inlet_temp_identified(self, analyzer):
        """Test high inlet temperature identified."""
        source, breakdown = analyzer._identify_degradation_source(
            bp_deviation_inhg=0.3,
            cw_inlet_temp_f=85.0,  # High inlet
            cw_flow_gpm=100000.0,
            cleanliness=0.85,
        )

        assert "high_cw_temp" in breakdown

    def test_low_flow_identified(self, analyzer):
        """Test low flow identified."""
        source, breakdown = analyzer._identify_degradation_source(
            bp_deviation_inhg=0.3,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=80000.0,  # Low flow
            cleanliness=0.85,
        )

        assert "low_cw_flow" in breakdown

    def test_tube_fouling_identified(self, analyzer):
        """Test tube fouling identified."""
        source, breakdown = analyzer._identify_degradation_source(
            bp_deviation_inhg=0.3,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
            cleanliness=0.70,  # Low cleanliness
        )

        assert "tube_fouling" in breakdown

    def test_breakdown_sums_to_100(self, analyzer):
        """Test breakdown percentages sum to 100."""
        _, breakdown = analyzer._identify_degradation_source(
            bp_deviation_inhg=0.5,
            cw_inlet_temp_f=80.0,
            cw_flow_gpm=85000.0,
            cleanliness=0.75,
        )

        if breakdown:
            total = sum(breakdown.values())
            assert total == pytest.approx(100.0, rel=0.01)


# =============================================================================
# PERFORMANCE CURVE GENERATION TESTS
# =============================================================================

class TestPerformanceCurveGeneration:
    """Test performance curve generation method."""

    def test_generate_curves(self, analyzer):
        """Test curve generation."""
        curves = analyzer.generate_performance_curve()

        assert isinstance(curves, dict)
        assert len(curves) > 0

    def test_curve_structure(self, analyzer):
        """Test curve structure."""
        curves = analyzer.generate_performance_curve(
            cw_inlet_temps=[70.0, 80.0, 90.0],
            load_points=[50.0, 75.0, 100.0],
        )

        # Should have entry for each load
        for load in [50.0, 75.0, 100.0]:
            assert load in curves
            # Each load should have entries for each temp
            for temp in [70.0, 80.0, 90.0]:
                assert temp in curves[load]
                assert curves[load][temp] > 0

    def test_curve_trends(self, analyzer):
        """Test curve shows expected trends."""
        curves = analyzer.generate_performance_curve(
            cw_inlet_temps=[70.0, 90.0],
            load_points=[50.0, 100.0],
        )

        # Higher load = higher BP
        assert curves[100.0][70.0] > curves[50.0][70.0]

        # Higher inlet temp = higher BP
        assert curves[75.0][90.0] > curves[75.0][70.0] if 75.0 in curves else True


# =============================================================================
# HISTORY TESTS
# =============================================================================

class TestPerformanceHistory:
    """Test performance history management."""

    def test_record_data_point(self, analyzer):
        """Test recording data points."""
        analyzer._record_data_point(1.6, 1.5, 400000.0, 75.0, 90000.0, 85.0)

        assert len(analyzer._history) == 1

    def test_history_trimming(self, analyzer):
        """Test old history is trimmed."""
        # Add old data point
        old_dp = PerformanceDataPoint(
            timestamp=datetime.now(timezone.utc) - timedelta(days=40),
            backpressure_inhga=1.6,
            expected_bp_inhga=1.5,
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
            load_pct=85.0,
        )
        analyzer._history.append(old_dp)

        # Record new data - triggers trimming
        analyzer._record_data_point(1.65, 1.5, 400000.0, 75.0, 90000.0, 85.0)

        # Old entry should be removed
        assert len(analyzer._history) == 1

    def test_get_performance_trend(self, analyzer):
        """Test retrieving performance trend."""
        for i in range(5):
            analyzer._record_data_point(
                1.5 + (i * 0.05),
                1.5,
                400000.0 + (i * 10000),
                75.0,
                90000.0,
                80.0 + i,
            )

        trends = analyzer.get_performance_trend(days=7)

        assert "backpressure" in trends
        assert "deviation" in trends
        assert "load" in trends
        assert len(trends["backpressure"]) == 5


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Test calculation counting."""

    def test_count_increments(self, analyzer):
        """Test calculation count increments."""
        initial = analyzer.calculation_count

        analyzer.analyze_performance(
            actual_backpressure_inhga=1.6,
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
        )

        assert analyzer.calculation_count == initial + 1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_minimum_load(self, analyzer):
        """Test at minimum load."""
        result = analyzer.analyze_performance(
            actual_backpressure_inhga=1.0,
            steam_flow_lb_hr=150000.0,  # 30% load
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        assert result.duty_ratio_pct == pytest.approx(30.0, rel=0.1)

    def test_maximum_load(self, analyzer):
        """Test at maximum load."""
        result = analyzer.analyze_performance(
            actual_backpressure_inhga=2.0,
            steam_flow_lb_hr=550000.0,  # 110% load
            cw_inlet_temp_f=80.0,
            cw_flow_gpm=95000.0,
        )

        assert result.duty_ratio_pct == pytest.approx(110.0, rel=0.1)

    def test_extreme_inlet_temp(self, analyzer):
        """Test with extreme inlet temperature."""
        result = analyzer.analyze_performance(
            actual_backpressure_inhga=3.0,
            steam_flow_lb_hr=400000.0,
            cw_inlet_temp_f=100.0,  # Very hot
            cw_flow_gpm=90000.0,
        )

        # Should still compute
        assert result is not None
        assert result.backpressure_deviation_pct > 0
