"""
GL-018 FLUEFLOW - Efficiency Calculator Unit Tests

Comprehensive unit tests for EfficiencyCalculator with 95%+ coverage target.
Tests all methods, edge cases, error handling, and ASME PTC 4.1 compliance.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
    EfficiencyOutput,
    FUEL_LHV,
    calculate_stack_loss_siegert,
    calculate_efficiency_from_losses,
    calculate_available_heat,
)
from calculators.provenance import verify_provenance


@pytest.mark.unit
@pytest.mark.calculator
@pytest.mark.critical
class TestEfficiencyCalculator:
    """Comprehensive test suite for EfficiencyCalculator."""

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization(self):
        """Test EfficiencyCalculator initializes correctly."""
        calculator = EfficiencyCalculator()

        assert calculator.VERSION == "1.0.0"
        assert calculator.NAME == "EfficiencyCalculator"
        assert calculator._tracker is None

    # =========================================================================
    # HAPPY PATH TESTS - NATURAL GAS
    # =========================================================================

    def test_natural_gas_high_efficiency(self, efficiency_calculator, natural_gas_efficiency_input):
        """Test natural gas with high efficiency conditions."""
        result, provenance = efficiency_calculator.calculate(natural_gas_efficiency_input)

        assert isinstance(result, EfficiencyOutput)
        assert result.combustion_efficiency_pct >= 80.0
        assert result.combustion_efficiency_pct <= 95.0
        assert result.thermal_efficiency_pct == pytest.approx(85.0, rel=0.01)  # 8.5/10.0
        assert result.stack_loss_pct > 0
        assert result.stack_loss_pct < 15.0
        assert result.total_losses_pct > 0
        assert result.total_losses_pct < 20.0
        assert result.efficiency_rating in ["Excellent", "Good"]
        assert verify_provenance(provenance) is True

    def test_natural_gas_optimal_conditions(self, efficiency_calculator):
        """Test natural gas with optimal operating conditions."""
        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=3.0,
            CO2_pct_dry=12.5,
            CO_ppm=30.0,
            flue_gas_temp_c=160.0,  # Lower stack temp = higher efficiency
            ambient_temp_c=20.0,
            excess_air_pct=16.7,
            heat_input_mw=10.0,
            heat_output_mw=9.0,  # 90% thermal efficiency
            moisture_in_fuel_pct=0.0
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        assert result.combustion_efficiency_pct >= 88.0
        assert result.thermal_efficiency_pct == pytest.approx(90.0, rel=0.01)
        assert result.stack_loss_pct < 8.0
        assert result.efficiency_rating == "Excellent"

    # =========================================================================
    # HAPPY PATH TESTS - FUEL OIL
    # =========================================================================

    def test_fuel_oil_efficiency(self, efficiency_calculator, fuel_oil_efficiency_input):
        """Test fuel oil combustion efficiency."""
        result, provenance = efficiency_calculator.calculate(fuel_oil_efficiency_input)

        assert result.combustion_efficiency_pct >= 75.0
        assert result.combustion_efficiency_pct <= 90.0
        assert result.thermal_efficiency_pct == pytest.approx(83.33, rel=0.02)  # 7.5/9.0
        assert result.stack_loss_pct > result.radiation_loss_pct
        assert result.efficiency_rating in ["Excellent", "Good", "Fair"]
        assert verify_provenance(provenance) is True

    def test_fuel_oil_with_moisture(self, efficiency_calculator):
        """Test fuel oil with moisture content."""
        inputs = EfficiencyInput(
            fuel_type="Fuel Oil",
            fuel_flow_rate_kg_hr=800.0,
            O2_pct_dry=3.5,
            CO2_pct_dry=13.5,
            CO_ppm=80.0,
            flue_gas_temp_c=220.0,
            ambient_temp_c=25.0,
            excess_air_pct=20.0,
            heat_input_mw=9.0,
            heat_output_mw=7.2,
            moisture_in_fuel_pct=2.0  # Some moisture in fuel
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        # Moisture loss should be higher with moisture in fuel
        assert result.moisture_loss_pct > 0.5
        assert result.moisture_loss_pct < 3.0

    # =========================================================================
    # HAPPY PATH TESTS - COAL
    # =========================================================================

    def test_coal_efficiency(self, efficiency_calculator):
        """Test coal combustion efficiency."""
        inputs = EfficiencyInput(
            fuel_type="Coal",
            fuel_flow_rate_kg_hr=2000.0,
            O2_pct_dry=5.0,
            CO2_pct_dry=15.0,
            CO_ppm=200.0,
            flue_gas_temp_c=280.0,
            ambient_temp_c=25.0,
            excess_air_pct=31.25,
            heat_input_mw=15.0,
            heat_output_mw=12.0,
            moisture_in_fuel_pct=10.0  # Coal typically has moisture
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        assert result.combustion_efficiency_pct >= 70.0
        assert result.combustion_efficiency_pct <= 85.0
        assert result.thermal_efficiency_pct == pytest.approx(80.0, rel=0.01)
        assert result.moisture_loss_pct > 1.0  # Higher due to fuel moisture
        assert result.incomplete_combustion_loss_pct > 0  # Some CO present
        assert verify_provenance(provenance) is True

    # =========================================================================
    # STACK LOSS CALCULATION TESTS
    # =========================================================================

    @pytest.mark.parametrize("flue_temp,ambient_temp,CO2_pct,expected_loss", [
        (180.0, 25.0, 12.0, 6.7),  # Reference case
        (200.0, 25.0, 12.0, 7.58),  # Higher temp
        (160.0, 25.0, 12.0, 5.83),  # Lower temp
        (180.0, 25.0, 10.0, 8.06),  # Lower CO2
        (180.0, 25.0, 14.0, 5.76),  # Higher CO2
        (250.0, 30.0, 15.0, 7.60),  # Coal-like conditions
    ])
    def test_stack_loss_variations(self, efficiency_calculator, flue_temp, ambient_temp, CO2_pct, expected_loss):
        """Test stack loss calculation with various conditions."""
        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=3.5,
            CO2_pct_dry=CO2_pct,
            CO_ppm=50.0,
            flue_gas_temp_c=flue_temp,
            ambient_temp_c=ambient_temp,
            excess_air_pct=20.0,
            heat_input_mw=10.0,
            heat_output_mw=8.5
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        assert result.stack_loss_pct == pytest.approx(expected_loss, rel=0.05)

    # =========================================================================
    # INCOMPLETE COMBUSTION LOSS TESTS
    # =========================================================================

    @pytest.mark.parametrize("CO_ppm,expected_loss_range", [
        (0.0, (0.0, 0.1)),
        (50.0, (0.2, 0.3)),
        (100.0, (0.4, 0.6)),
        (200.0, (0.9, 1.1)),
        (400.0, (1.9, 2.1)),
        (800.0, (3.5, 4.5)),
    ])
    def test_incomplete_combustion_loss(self, efficiency_calculator, CO_ppm, expected_loss_range):
        """Test incomplete combustion loss calculation."""
        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=3.5,
            CO2_pct_dry=12.0,
            CO_ppm=CO_ppm,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_air_pct=20.0,
            heat_input_mw=10.0,
            heat_output_mw=8.5
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        assert result.incomplete_combustion_loss_pct >= expected_loss_range[0]
        assert result.incomplete_combustion_loss_pct <= expected_loss_range[1]

    # =========================================================================
    # EFFICIENCY RATING TESTS
    # =========================================================================

    @pytest.mark.parametrize("efficiency_pct,expected_rating", [
        (95.0, "Excellent"),
        (92.0, "Excellent"),
        (90.0, "Excellent"),
        (88.0, "Good"),
        (85.0, "Good"),
        (83.0, "Fair"),
        (80.0, "Fair"),
        (78.0, "Poor"),
        (75.0, "Poor"),
        (70.0, "Critical"),
    ])
    def test_efficiency_rating_classification(self, efficiency_calculator, efficiency_pct, expected_rating):
        """Test efficiency rating classification."""
        # Calculate required losses to achieve target efficiency
        total_losses = 100.0 - efficiency_pct

        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=3.5,
            CO2_pct_dry=12.0,
            CO_ppm=50.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_air_pct=20.0,
            heat_input_mw=10.0,
            heat_output_mw=efficiency_pct / 10.0,  # Scale to match efficiency
            radiation_loss_pct=total_losses / 5.0,  # Distribute losses
            unaccounted_loss_pct=total_losses / 10.0
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        assert result.efficiency_rating == expected_rating

    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================

    def test_high_stack_temperature(self, efficiency_calculator):
        """Test high stack temperature (low efficiency)."""
        inputs = EfficiencyInput(
            fuel_type="Coal",
            fuel_flow_rate_kg_hr=2000.0,
            O2_pct_dry=8.0,
            CO2_pct_dry=10.0,
            CO_ppm=300.0,
            flue_gas_temp_c=400.0,  # Very high stack temp
            ambient_temp_c=25.0,
            excess_air_pct=72.7,  # High excess air
            heat_input_mw=15.0,
            heat_output_mw=10.0
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        # High stack temp = high stack loss
        assert result.stack_loss_pct > 15.0
        assert result.combustion_efficiency_pct < 75.0
        assert result.efficiency_rating in ["Poor", "Critical"]

    def test_low_stack_temperature(self, efficiency_calculator):
        """Test low stack temperature (high efficiency potential)."""
        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=2.5,
            CO2_pct_dry=13.0,
            CO_ppm=40.0,
            flue_gas_temp_c=140.0,  # Low stack temp (good)
            ambient_temp_c=20.0,
            excess_air_pct=13.9,
            heat_input_mw=10.0,
            heat_output_mw=9.2
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        # Low stack temp = low stack loss
        assert result.stack_loss_pct < 7.0
        assert result.combustion_efficiency_pct >= 88.0
        assert result.efficiency_rating in ["Excellent", "Good"]

    def test_zero_excess_air(self, efficiency_calculator):
        """Test operation at stoichiometric (zero excess air)."""
        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=0.5,
            CO2_pct_dry=11.7,
            CO_ppm=300.0,  # Higher CO at low O2
            flue_gas_temp_c=200.0,
            ambient_temp_c=25.0,
            excess_air_pct=2.4,  # Very low excess air
            heat_input_mw=10.0,
            heat_output_mw=8.0
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        # Low excess air but higher incomplete combustion loss
        assert result.incomplete_combustion_loss_pct > 1.0

    def test_maximum_losses(self, efficiency_calculator):
        """Test worst case scenario (maximum losses)."""
        inputs = EfficiencyInput(
            fuel_type="Coal",
            fuel_flow_rate_kg_hr=3000.0,
            O2_pct_dry=10.0,
            CO2_pct_dry=8.0,
            CO_ppm=1000.0,  # Very high CO
            flue_gas_temp_c=500.0,  # Very high temperature
            ambient_temp_c=30.0,
            excess_air_pct=90.9,  # Excessive air
            heat_input_mw=20.0,
            heat_output_mw=12.0,
            moisture_in_fuel_pct=20.0,  # High moisture
            radiation_loss_pct=3.0,  # High radiation loss
            unaccounted_loss_pct=2.0
        )

        result, provenance = efficiency_calculator.calculate(inputs)

        assert result.total_losses_pct > 30.0
        assert result.combustion_efficiency_pct < 70.0
        assert result.efficiency_rating == "Critical"

    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================

    def test_invalid_fuel_flow_negative(self, efficiency_calculator):
        """Test negative fuel flow raises ValueError."""
        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=-100.0,  # Invalid
            O2_pct_dry=3.5,
            CO2_pct_dry=12.0,
            CO_ppm=50.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_air_pct=20.0,
            heat_input_mw=10.0,
            heat_output_mw=8.5
        )

        with pytest.raises(ValueError, match="Fuel flow rate must be positive"):
            efficiency_calculator.calculate(inputs)

    def test_invalid_heat_input_negative(self, efficiency_calculator):
        """Test negative heat input raises ValueError."""
        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=3.5,
            CO2_pct_dry=12.0,
            CO_ppm=50.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_air_pct=20.0,
            heat_input_mw=-10.0,  # Invalid
            heat_output_mw=8.5
        )

        with pytest.raises(ValueError, match="Heat input must be positive"):
            efficiency_calculator.calculate(inputs)

    def test_invalid_heat_output_negative(self, efficiency_calculator):
        """Test negative heat output raises ValueError."""
        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=3.5,
            CO2_pct_dry=12.0,
            CO_ppm=50.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_air_pct=20.0,
            heat_input_mw=10.0,
            heat_output_mw=-8.5  # Invalid
        )

        with pytest.raises(ValueError, match="Heat output cannot be negative"):
            efficiency_calculator.calculate(inputs)

    def test_invalid_heat_output_exceeds_input(self, efficiency_calculator):
        """Test heat output > heat input raises ValueError."""
        inputs = EfficiencyInput(
            fuel_type="Natural Gas",
            fuel_flow_rate_kg_hr=1000.0,
            O2_pct_dry=3.5,
            CO2_pct_dry=12.0,
            CO_ppm=50.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_air_pct=20.0,
            heat_input_mw=10.0,
            heat_output_mw=12.0  # Invalid (>input)
        )

        with pytest.raises(ValueError, match="Heat output cannot exceed heat input"):
            efficiency_calculator.calculate(inputs)

    # =========================================================================
    # PROVENANCE TESTS
    # =========================================================================

    def test_provenance_determinism(self, efficiency_calculator, natural_gas_efficiency_input):
        """Test provenance hash is deterministic."""
        result1, provenance1 = efficiency_calculator.calculate(natural_gas_efficiency_input)
        result2, provenance2 = efficiency_calculator.calculate(natural_gas_efficiency_input)

        assert provenance1.provenance_hash == provenance2.provenance_hash
        assert result1.combustion_efficiency_pct == result2.combustion_efficiency_pct

    def test_provenance_completeness(self, efficiency_calculator, natural_gas_efficiency_input):
        """Test provenance includes all calculation steps."""
        result, provenance = efficiency_calculator.calculate(natural_gas_efficiency_input)

        assert len(provenance.calculation_steps) >= 5
        assert verify_provenance(provenance) is True

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    def test_calculation_speed(self, efficiency_calculator, natural_gas_efficiency_input, benchmark):
        """Test calculation meets performance target (<5ms)."""
        result = benchmark(efficiency_calculator.calculate, natural_gas_efficiency_input)
        assert benchmark.stats.stats.mean < 0.005


# =============================================================================
# STANDALONE FUNCTION TESTS
# =============================================================================

@pytest.mark.unit
class TestStandaloneFunctions:
    """Test standalone utility functions."""

    def test_calculate_stack_loss_siegert(self):
        """Test Siegert formula for stack loss."""
        # Reference case: 180°C, 25°C ambient, 12% CO2
        loss = calculate_stack_loss_siegert(180.0, 25.0, 12.0)
        assert loss == pytest.approx(6.7, rel=0.02)

        # Higher temperature
        loss = calculate_stack_loss_siegert(250.0, 25.0, 12.0)
        assert loss == pytest.approx(9.75, rel=0.02)

    def test_calculate_efficiency_from_losses(self):
        """Test efficiency calculation from losses."""
        efficiency = calculate_efficiency_from_losses(
            stack_loss_pct=7.0,
            radiation_loss_pct=1.0,
            moisture_loss_pct=0.5,
            incomplete_combustion_loss_pct=0.25,
            unaccounted_loss_pct=0.5
        )

        assert efficiency == pytest.approx(90.75, rel=0.01)

    def test_calculate_available_heat(self):
        """Test available heat calculation."""
        available_heat = calculate_available_heat(
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            CO2_pct=12.0,
            radiation_loss_pct=1.0
        )

        assert available_heat == pytest.approx(92.3, rel=0.02)
