"""
Unit tests for GL-009 THERMALIQ Agent Exergy Analysis

Tests exergy (2nd Law) efficiency calculations including Carnot efficiency,
exergy destruction, and improvement potential analysis.

Reference values based on thermodynamic principles and engineering standards.
"""

import pytest
import math
from typing import Dict, Any

from greenlang.agents.process_heat.gl_009_thermal_fluid.exergy import (
    ExergyAnalyzer,
    ExergyDestructionBreakdown,
    R_RANKINE_OFFSET,
    DEFAULT_REFERENCE_TEMP_F,
    calculate_exergy_efficiency,
    calculate_carnot_limit,
)
from greenlang.agents.process_heat.gl_009_thermal_fluid.schemas import (
    ThermalFluidType,
    ExergyAnalysis,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def exergy_analyzer():
    """Create exergy analyzer instance."""
    return ExergyAnalyzer()


@pytest.fixture
def exergy_analyzer_with_fluid():
    """Create exergy analyzer with fluid type."""
    return ExergyAnalyzer(fluid_type=ThermalFluidType.THERMINOL_66)


@pytest.fixture
def typical_system_params():
    """Create typical thermal fluid system parameters."""
    return {
        "hot_temp_f": 600.0,
        "cold_temp_f": 400.0,
        "heat_duty_btu_hr": 5_000_000.0,
        "heater_efficiency_pct": 85.0,
    }


# =============================================================================
# ANALYZER INITIALIZATION TESTS
# =============================================================================

class TestExergyAnalyzerInit:
    """Tests for ExergyAnalyzer initialization."""

    def test_default_initialization(self, exergy_analyzer):
        """Test analyzer initializes with defaults."""
        assert exergy_analyzer.reference_temp_f == DEFAULT_REFERENCE_TEMP_F
        assert exergy_analyzer.reference_temp_r == DEFAULT_REFERENCE_TEMP_F + R_RANKINE_OFFSET
        assert exergy_analyzer._calculation_count == 0

    def test_custom_reference_temperature(self):
        """Test custom reference temperature."""
        analyzer = ExergyAnalyzer(reference_temp_f=68.0)

        assert analyzer.reference_temp_f == 68.0
        assert analyzer.reference_temp_r == 68.0 + R_RANKINE_OFFSET

    def test_with_fluid_type(self, exergy_analyzer_with_fluid):
        """Test initialization with fluid type."""
        assert exergy_analyzer_with_fluid.fluid_type == ThermalFluidType.THERMINOL_66
        assert exergy_analyzer_with_fluid._property_db is not None


# =============================================================================
# CARNOT EFFICIENCY TESTS
# =============================================================================

class TestCarnotEfficiency:
    """Tests for Carnot efficiency calculations."""

    def test_carnot_efficiency_basic(self, exergy_analyzer):
        """Test basic Carnot efficiency calculation."""
        # eta_carnot = 1 - T_cold/T_hot (in absolute units)
        hot_temp_f = 600.0
        cold_temp_f = 77.0  # Reference temp

        carnot = exergy_analyzer.calculate_carnot_efficiency(hot_temp_f, cold_temp_f)

        # Manual calculation:
        # T_hot_R = 600 + 459.67 = 1059.67 R
        # T_cold_R = 77 + 459.67 = 536.67 R
        # eta = 1 - 536.67/1059.67 = 0.4936
        expected = 1.0 - (77 + R_RANKINE_OFFSET) / (600 + R_RANKINE_OFFSET)

        assert abs(carnot - expected) < 0.001

    def test_carnot_efficiency_with_default_reference(self, exergy_analyzer):
        """Test Carnot efficiency with default reference temperature."""
        carnot = exergy_analyzer.calculate_carnot_efficiency(600.0)

        assert 0.0 <= carnot <= 1.0
        # Should be approximately 49%
        assert 0.45 <= carnot <= 0.55

    @pytest.mark.parametrize("hot_temp_f,expected_carnot_range", [
        (200.0, (0.15, 0.25)),  # Low temp - low efficiency
        (400.0, (0.35, 0.45)),
        (600.0, (0.45, 0.55)),
        (750.0, (0.50, 0.60)),  # High temp - higher efficiency
    ])
    def test_carnot_efficiency_at_temperatures(
        self, exergy_analyzer, hot_temp_f, expected_carnot_range
    ):
        """Test Carnot efficiency at various temperatures."""
        carnot = exergy_analyzer.calculate_carnot_efficiency(hot_temp_f)

        assert expected_carnot_range[0] <= carnot <= expected_carnot_range[1]

    def test_carnot_increases_with_temperature(self, exergy_analyzer):
        """Test Carnot efficiency increases with temperature."""
        carnot_low = exergy_analyzer.calculate_carnot_efficiency(300.0)
        carnot_high = exergy_analyzer.calculate_carnot_efficiency(600.0)

        assert carnot_high > carnot_low

    def test_carnot_approaches_zero_at_reference(self, exergy_analyzer):
        """Test Carnot efficiency approaches zero at reference temp."""
        carnot = exergy_analyzer.calculate_carnot_efficiency(78.0)  # Just above reference

        assert carnot < 0.01

    def test_carnot_bounded_0_to_1(self, exergy_analyzer):
        """Test Carnot efficiency is bounded between 0 and 1."""
        for temp in [100, 200, 400, 600, 800, 1000]:
            carnot = exergy_analyzer.calculate_carnot_efficiency(float(temp))
            assert 0.0 <= carnot <= 1.0


# =============================================================================
# SYSTEM ANALYSIS TESTS
# =============================================================================

class TestSystemAnalysis:
    """Tests for full system exergy analysis."""

    def test_analyze_system_basic(self, exergy_analyzer, typical_system_params):
        """Test basic system analysis."""
        result = exergy_analyzer.analyze_system(**typical_system_params)

        assert isinstance(result, ExergyAnalysis)
        assert 0 <= result.exergy_efficiency_pct <= 100
        assert 0 <= result.first_law_efficiency_pct <= 100
        assert result.exergy_input_btu_hr > 0
        assert result.exergy_output_btu_hr >= 0
        assert result.exergy_destruction_btu_hr >= 0

    def test_analyze_system_returns_carnot(self, exergy_analyzer, typical_system_params):
        """Test system analysis returns Carnot efficiency."""
        result = exergy_analyzer.analyze_system(**typical_system_params)

        assert 0 <= result.carnot_efficiency_pct <= 100

    def test_exergy_efficiency_less_than_first_law(self, exergy_analyzer, typical_system_params):
        """Test exergy efficiency is less than first law efficiency."""
        result = exergy_analyzer.analyze_system(**typical_system_params)

        # Exergy (2nd law) efficiency is always <= 1st law
        assert result.exergy_efficiency_pct <= result.first_law_efficiency_pct

    def test_exergy_conservation(self, exergy_analyzer, typical_system_params):
        """Test exergy is conserved (output + destruction = input)."""
        result = exergy_analyzer.analyze_system(**typical_system_params)

        # Allow for rounding
        total = result.exergy_output_btu_hr + result.exergy_destruction_btu_hr
        # Total should be close to input (within reasonable tolerance)
        ratio = total / result.exergy_input_btu_hr if result.exergy_input_btu_hr > 0 else 0

        assert 0.8 <= ratio <= 1.2  # Within 20% due to calculation approximations

    def test_destruction_breakdown_percentages(self, exergy_analyzer, typical_system_params):
        """Test destruction breakdown percentages."""
        result = exergy_analyzer.analyze_system(**typical_system_params)

        total_destruction_pct = (
            result.heater_destruction_pct +
            result.piping_destruction_pct +
            result.mixing_destruction_pct
        )

        # Total should be reasonable
        assert 0 <= total_destruction_pct <= 100

    def test_log_mean_temp_ratio(self, exergy_analyzer, typical_system_params):
        """Test log mean temperature ratio is positive."""
        result = exergy_analyzer.analyze_system(**typical_system_params)

        assert result.log_mean_temp_ratio > 0
        # Should be greater than 1 for above-reference temps
        assert result.log_mean_temp_ratio > 1.0

    def test_reference_temperature_in_output(self, exergy_analyzer, typical_system_params):
        """Test reference temperature is in output."""
        result = exergy_analyzer.analyze_system(**typical_system_params)

        assert result.reference_temperature_f == DEFAULT_REFERENCE_TEMP_F

    def test_calculation_method_in_output(self, exergy_analyzer, typical_system_params):
        """Test calculation method is in output."""
        result = exergy_analyzer.analyze_system(**typical_system_params)

        assert result.calculation_method == "SECOND_LAW_AVAILABILITY"

    def test_hot_temp_must_exceed_reference(self, exergy_analyzer):
        """Test hot temperature must exceed reference."""
        with pytest.raises(ValueError) as exc_info:
            exergy_analyzer.analyze_system(
                hot_temp_f=70.0,  # Below reference
                cold_temp_f=50.0,
                heat_duty_btu_hr=1_000_000,
            )

        assert "must exceed reference" in str(exc_info.value)


# =============================================================================
# HEAT TRANSFER ANALYSIS TESTS
# =============================================================================

class TestHeatTransferAnalysis:
    """Tests for heat exchanger exergy analysis."""

    @pytest.fixture
    def hx_params(self):
        """Create heat exchanger parameters."""
        return {
            "hot_inlet_temp_f": 600.0,
            "hot_outlet_temp_f": 500.0,
            "cold_inlet_temp_f": 300.0,
            "cold_outlet_temp_f": 400.0,
            "heat_duty_btu_hr": 2_000_000.0,
        }

    def test_heat_transfer_analysis(self, exergy_analyzer, hx_params):
        """Test heat transfer exergy analysis."""
        result = exergy_analyzer.analyze_heat_transfer(**hx_params)

        assert "exergy_decrease_hot_btu_hr" in result
        assert "exergy_increase_cold_btu_hr" in result
        assert "exergy_destruction_btu_hr" in result
        assert "hx_exergy_efficiency_pct" in result

    def test_exergy_decrease_hot_positive(self, exergy_analyzer, hx_params):
        """Test hot side exergy decrease is positive."""
        result = exergy_analyzer.analyze_heat_transfer(**hx_params)

        assert result["exergy_decrease_hot_btu_hr"] > 0

    def test_exergy_increase_cold_positive(self, exergy_analyzer, hx_params):
        """Test cold side exergy increase is positive."""
        result = exergy_analyzer.analyze_heat_transfer(**hx_params)

        assert result["exergy_increase_cold_btu_hr"] > 0

    def test_exergy_destruction_non_negative(self, exergy_analyzer, hx_params):
        """Test exergy destruction is non-negative."""
        result = exergy_analyzer.analyze_heat_transfer(**hx_params)

        assert result["exergy_destruction_btu_hr"] >= 0

    def test_hx_efficiency_bounded(self, exergy_analyzer, hx_params):
        """Test HX exergy efficiency is bounded."""
        result = exergy_analyzer.analyze_heat_transfer(**hx_params)

        assert 0 <= result["hx_exergy_efficiency_pct"] <= 100


# =============================================================================
# STREAM EXERGY TESTS
# =============================================================================

class TestStreamExergy:
    """Tests for stream exergy calculations."""

    def test_stream_exergy_calculation(self, exergy_analyzer):
        """Test stream exergy calculation."""
        exergy = exergy_analyzer.calculate_stream_exergy(
            temperature_f=500.0,
            mass_flow_lb_hr=10000.0,
            specific_heat_btu_lb_f=0.55,
        )

        assert exergy > 0

    def test_stream_exergy_increases_with_temperature(self, exergy_analyzer):
        """Test stream exergy increases with temperature."""
        exergy_low = exergy_analyzer.calculate_stream_exergy(
            temperature_f=200.0,
            mass_flow_lb_hr=10000.0,
            specific_heat_btu_lb_f=0.55,
        )

        exergy_high = exergy_analyzer.calculate_stream_exergy(
            temperature_f=600.0,
            mass_flow_lb_hr=10000.0,
            specific_heat_btu_lb_f=0.55,
        )

        assert exergy_high > exergy_low

    def test_stream_exergy_increases_with_mass_flow(self, exergy_analyzer):
        """Test stream exergy increases with mass flow."""
        exergy_low = exergy_analyzer.calculate_stream_exergy(
            temperature_f=500.0,
            mass_flow_lb_hr=5000.0,
            specific_heat_btu_lb_f=0.55,
        )

        exergy_high = exergy_analyzer.calculate_stream_exergy(
            temperature_f=500.0,
            mass_flow_lb_hr=20000.0,
            specific_heat_btu_lb_f=0.55,
        )

        assert exergy_high > exergy_low

    def test_stream_exergy_at_reference_temp_zero(self, exergy_analyzer):
        """Test stream exergy is zero at reference temperature."""
        exergy = exergy_analyzer.calculate_stream_exergy(
            temperature_f=77.0,  # Reference temp
            mass_flow_lb_hr=10000.0,
            specific_heat_btu_lb_f=0.55,
        )

        # Should be very close to zero
        assert abs(exergy) < 1.0


# =============================================================================
# HEAT EXERGY TESTS
# =============================================================================

class TestHeatExergy:
    """Tests for heat exergy calculations."""

    def test_heat_exergy_calculation(self, exergy_analyzer):
        """Test heat exergy calculation."""
        exergy = exergy_analyzer.calculate_heat_exergy(
            heat_rate_btu_hr=1_000_000.0,
            temperature_f=600.0,
        )

        assert exergy > 0
        # Should be less than heat rate (exergy factor < 1)
        assert exergy < 1_000_000.0

    def test_heat_exergy_increases_with_temperature(self, exergy_analyzer):
        """Test heat exergy increases with temperature."""
        exergy_low = exergy_analyzer.calculate_heat_exergy(
            heat_rate_btu_hr=1_000_000.0,
            temperature_f=200.0,
        )

        exergy_high = exergy_analyzer.calculate_heat_exergy(
            heat_rate_btu_hr=1_000_000.0,
            temperature_f=600.0,
        )

        assert exergy_high > exergy_low

    def test_heat_exergy_proportional_to_heat_rate(self, exergy_analyzer):
        """Test heat exergy is proportional to heat rate."""
        exergy_1 = exergy_analyzer.calculate_heat_exergy(
            heat_rate_btu_hr=1_000_000.0,
            temperature_f=500.0,
        )

        exergy_2 = exergy_analyzer.calculate_heat_exergy(
            heat_rate_btu_hr=2_000_000.0,
            temperature_f=500.0,
        )

        # Should be exactly 2x
        assert abs(exergy_2 / exergy_1 - 2.0) < 0.001

    def test_heat_exergy_below_reference_is_zero(self, exergy_analyzer):
        """Test heat exergy below reference temperature is zero."""
        exergy = exergy_analyzer.calculate_heat_exergy(
            heat_rate_btu_hr=1_000_000.0,
            temperature_f=50.0,  # Below reference
        )

        assert exergy == 0.0


# =============================================================================
# RATIONAL EFFICIENCY TESTS
# =============================================================================

class TestRationalEfficiency:
    """Tests for rational (2nd Law) efficiency calculations."""

    def test_rational_efficiency_calculation(self, exergy_analyzer):
        """Test rational efficiency calculation."""
        efficiency = exergy_analyzer.calculate_rational_efficiency(
            actual_work_btu_hr=1_000_000.0,
            exergy_input_btu_hr=2_000_000.0,
        )

        assert efficiency == 0.5

    def test_rational_efficiency_bounded(self, exergy_analyzer):
        """Test rational efficiency is bounded 0-1."""
        # Normal case
        eff = exergy_analyzer.calculate_rational_efficiency(500_000, 1_000_000)
        assert 0 <= eff <= 1.0

        # Edge case: output > input (should cap at 1)
        eff = exergy_analyzer.calculate_rational_efficiency(2_000_000, 1_000_000)
        assert eff == 1.0

    def test_rational_efficiency_zero_input(self, exergy_analyzer):
        """Test rational efficiency with zero input."""
        efficiency = exergy_analyzer.calculate_rational_efficiency(
            actual_work_btu_hr=1_000_000.0,
            exergy_input_btu_hr=0.0,
        )

        assert efficiency == 0.0


# =============================================================================
# IMPROVEMENT POTENTIAL TESTS
# =============================================================================

class TestImprovementPotential:
    """Tests for improvement potential analysis."""

    @pytest.fixture
    def destruction_breakdown(self):
        """Create destruction breakdown fixture."""
        return ExergyDestructionBreakdown(
            heater_destruction_btu_hr=500_000.0,
            heater_destruction_pct=20.0,
            heat_exchanger_destruction_btu_hr=200_000.0,
            heat_exchanger_destruction_pct=12.0,
            piping_destruction_btu_hr=100_000.0,
            piping_destruction_pct=5.0,
            total_destruction_btu_hr=800_000.0,
        )

    def test_improvement_analysis_returns_recommendations(
        self, exergy_analyzer, destruction_breakdown
    ):
        """Test improvement analysis returns recommendations."""
        recommendations = exergy_analyzer.analyze_improvement_potential(
            current_exergy_efficiency_pct=35.0,
            destruction_breakdown=destruction_breakdown,
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_improvement_recommendations_have_required_fields(
        self, exergy_analyzer, destruction_breakdown
    ):
        """Test recommendations have required fields."""
        recommendations = exergy_analyzer.analyze_improvement_potential(
            current_exergy_efficiency_pct=35.0,
            destruction_breakdown=destruction_breakdown,
        )

        for rec in recommendations:
            assert "category" in rec
            assert "title" in rec
            assert "description" in rec
            assert "potential_reduction_btu_hr" in rec or "estimated_annual_savings_usd" in rec

    def test_improvement_sorted_by_savings(
        self, exergy_analyzer, destruction_breakdown
    ):
        """Test recommendations are sorted by savings."""
        recommendations = exergy_analyzer.analyze_improvement_potential(
            current_exergy_efficiency_pct=35.0,
            destruction_breakdown=destruction_breakdown,
        )

        if len(recommendations) > 1:
            savings = [r.get("estimated_annual_savings_usd", 0) for r in recommendations]
            assert savings == sorted(savings, reverse=True)

    def test_heater_improvement_triggered(self, exergy_analyzer):
        """Test heater improvement recommendation triggered."""
        breakdown = ExergyDestructionBreakdown(
            heater_destruction_btu_hr=1_000_000.0,
            heater_destruction_pct=25.0,  # Above 15% threshold
            total_destruction_btu_hr=1_500_000.0,
        )

        recommendations = exergy_analyzer.analyze_improvement_potential(
            current_exergy_efficiency_pct=35.0,
            destruction_breakdown=breakdown,
        )

        heater_recs = [r for r in recommendations if r["category"] == "heater"]
        assert len(heater_recs) > 0

    def test_piping_improvement_triggered(self, exergy_analyzer):
        """Test piping improvement recommendation triggered."""
        breakdown = ExergyDestructionBreakdown(
            piping_destruction_btu_hr=200_000.0,
            piping_destruction_pct=8.0,  # Above 3% threshold
            total_destruction_btu_hr=500_000.0,
        )

        recommendations = exergy_analyzer.analyze_improvement_potential(
            current_exergy_efficiency_pct=35.0,
            destruction_breakdown=breakdown,
        )

        piping_recs = [r for r in recommendations if r["category"] == "piping"]
        assert len(piping_recs) > 0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_exergy_efficiency_function(self):
        """Test calculate_exergy_efficiency convenience function."""
        efficiency = calculate_exergy_efficiency(
            hot_temp_f=600.0,
            cold_temp_f=400.0,
            heat_duty_btu_hr=5_000_000.0,
        )

        assert 0 <= efficiency <= 100

    def test_calculate_carnot_limit_function(self):
        """Test calculate_carnot_limit convenience function."""
        carnot = calculate_carnot_limit(hot_temp_f=600.0)

        assert 0 <= carnot <= 100
        # Should be about 49%
        assert 45 <= carnot <= 55

    def test_calculate_carnot_limit_with_custom_cold(self):
        """Test Carnot limit with custom cold temperature."""
        carnot = calculate_carnot_limit(hot_temp_f=600.0, cold_temp_f=200.0)

        # Higher cold temp = lower efficiency
        carnot_default = calculate_carnot_limit(hot_temp_f=600.0, cold_temp_f=77.0)
        assert carnot < carnot_default


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Tests for calculation counting."""

    def test_calculation_count_increments(self, exergy_analyzer, typical_system_params):
        """Test calculation count increments."""
        assert exergy_analyzer.calculation_count == 0

        exergy_analyzer.analyze_system(**typical_system_params)
        assert exergy_analyzer.calculation_count == 1

        exergy_analyzer.analyze_heat_transfer(
            hot_inlet_temp_f=600.0,
            hot_outlet_temp_f=500.0,
            cold_inlet_temp_f=300.0,
            cold_outlet_temp_f=400.0,
            heat_duty_btu_hr=1_000_000.0,
        )
        assert exergy_analyzer.calculation_count == 2


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for calculation determinism."""

    def test_same_input_same_output(self, typical_system_params):
        """Test same input produces identical output."""
        analyzer1 = ExergyAnalyzer()
        analyzer2 = ExergyAnalyzer()

        result1 = analyzer1.analyze_system(**typical_system_params)
        result2 = analyzer2.analyze_system(**typical_system_params)

        assert result1.exergy_efficiency_pct == result2.exergy_efficiency_pct
        assert result1.carnot_efficiency_pct == result2.carnot_efficiency_pct
        assert result1.exergy_destruction_btu_hr == result2.exergy_destruction_btu_hr

    def test_reproducible_calculations(self, exergy_analyzer, typical_system_params):
        """Test calculations are reproducible."""
        results = [
            exergy_analyzer.analyze_system(**typical_system_params)
            for _ in range(5)
        ]

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result.exergy_efficiency_pct == first.exergy_efficiency_pct
