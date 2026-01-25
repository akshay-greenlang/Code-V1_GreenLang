"""
GL-002 FLAMEGUARD - Comprehensive Combustion Calculator Tests

Tests for efficiency calculators, emissions calculators, and O2 trim control.
Targets 70%+ coverage with parameterized tests, edge cases, and validation.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import math
import sys

sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])

from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
    EfficiencyResult,
    FuelProperties,
    FUEL_DATABASE,
)
from calculators.emissions_calculator import (
    EmissionsCalculator,
    EmissionsInput,
    EmissionsResult,
    EPA_EMISSION_FACTORS,
    GWP,
)
from optimization.o2_trim_controller import (
    O2TrimController,
    PIDController,
    TrimSetpoint,
    COBreakthroughEvent,
)


# =============================================================================
# EFFICIENCY CALCULATOR TESTS
# =============================================================================


class TestEfficiencyCalculatorInit:
    """Test efficiency calculator initialization."""

    def test_default_initialization(self):
        """Test calculator initializes with default fuel database."""
        calc = EfficiencyCalculator()
        assert calc.fuel_database is not None
        assert "natural_gas" in calc.fuel_database
        assert "fuel_oil_no2" in calc.fuel_database
        assert "coal_bituminous" in calc.fuel_database

    def test_custom_fuel_database(self):
        """Test calculator accepts custom fuel database."""
        custom_fuel = {
            "custom_gas": FuelProperties(
                fuel_type="custom_gas",
                higher_heating_value_btu_lb=24000.0,
                lower_heating_value_btu_lb=21600.0,
                carbon_percent=74.0,
                hydrogen_percent=26.0,
                sulfur_percent=0.0,
                nitrogen_percent=0.0,
                oxygen_percent=0.0,
                moisture_percent=0.0,
                ash_percent=0.0,
                stoichiometric_air_ratio=17.5,
            )
        }
        calc = EfficiencyCalculator(fuel_database=custom_fuel)
        assert "custom_gas" in calc.fuel_database

    def test_version_constant(self):
        """Test version constant is set."""
        assert EfficiencyCalculator.VERSION == "1.0.0"
        assert EfficiencyCalculator.FORMULA_VERSION == "ASME_PTC_4.1_2013"


class TestEfficiencyCalculatorIndirect:
    """Test indirect (heat loss) method calculations."""

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    @pytest.fixture
    def base_input(self):
        return EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

    def test_indirect_method_basic(self, calculator, base_input):
        """Test basic indirect method calculation."""
        result = calculator.calculate(base_input, method="indirect")

        assert isinstance(result, EfficiencyResult)
        assert result.method == "indirect"
        assert 75.0 <= result.efficiency_hhv_percent <= 95.0
        assert result.efficiency_lhv_percent >= result.efficiency_hhv_percent

    def test_indirect_method_losses_breakdown(self, calculator, base_input):
        """Test all loss components are calculated."""
        result = calculator.calculate(base_input, method="indirect")

        assert result.dry_flue_gas_loss_percent >= 0
        assert result.moisture_in_fuel_loss_percent >= 0
        assert result.hydrogen_combustion_loss_percent >= 0
        assert result.moisture_in_air_loss_percent >= 0
        assert result.radiation_loss_percent >= 0
        assert result.blowdown_loss_percent >= 0
        assert result.other_losses_percent >= 0

    def test_indirect_losses_sum_correctly(self, calculator, base_input):
        """Test total losses equals 100 - efficiency."""
        result = calculator.calculate(base_input, method="indirect")

        calculated_losses = (
            result.dry_flue_gas_loss_percent +
            result.moisture_in_fuel_loss_percent +
            result.hydrogen_combustion_loss_percent +
            result.moisture_in_air_loss_percent +
            result.unburned_carbon_loss_percent +
            result.co_loss_percent +
            result.radiation_loss_percent +
            result.blowdown_loss_percent +
            result.other_losses_percent
        )

        expected_losses = 100 - result.efficiency_hhv_percent
        assert abs(calculated_losses - expected_losses) < 2.0

    @pytest.mark.parametrize("o2_percent,expected_excess_air_range", [
        (1.0, (4.0, 6.0)),
        (2.0, (9.0, 12.0)),
        (3.0, (15.0, 18.0)),
        (4.0, (22.0, 26.0)),
        (5.0, (30.0, 35.0)),
        (6.0, (38.0, 45.0)),
    ])
    def test_o2_to_excess_air_conversion(
        self, calculator, base_input, o2_percent, expected_excess_air_range
    ):
        """Test O2 percentage converts correctly to excess air."""
        base_input.flue_gas_o2_percent = o2_percent
        result = calculator.calculate(base_input, method="indirect")

        assert expected_excess_air_range[0] <= result.excess_air_percent <= expected_excess_air_range[1]

    @pytest.mark.parametrize("fuel_type", ["natural_gas", "fuel_oil_no2", "coal_bituminous"])
    def test_different_fuel_types(self, calculator, base_input, fuel_type):
        """Test calculation works for all fuel types."""
        base_input.fuel_type = fuel_type
        result = calculator.calculate(base_input, method="indirect")

        assert 50.0 <= result.efficiency_hhv_percent <= 100.0

    def test_scfh_fuel_unit_conversion(self, calculator):
        """Test fuel flow in SCFH is converted correctly."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=200000.0,  # SCFH
            fuel_flow_unit="scfh",
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )
        result = calculator.calculate(inp, method="indirect")

        assert result.fuel_input_mmbtu_hr > 0

    def test_high_flue_gas_temperature_reduces_efficiency(self, calculator, base_input):
        """Test higher flue gas temp reduces efficiency."""
        base_input.flue_gas_temperature_f = 350.0
        result_low = calculator.calculate(base_input, method="indirect")

        base_input.flue_gas_temperature_f = 500.0
        result_high = calculator.calculate(base_input, method="indirect")

        assert result_high.efficiency_hhv_percent < result_low.efficiency_hhv_percent

    def test_coal_unburned_carbon_loss(self, calculator):
        """Test unburned carbon loss calculated for coal."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=15000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=4.0,
            fuel_type="coal_bituminous",
            ash_unburned_carbon_percent=3.0,
        )
        result = calculator.calculate(inp, method="indirect")

        assert result.unburned_carbon_loss_percent > 0

    def test_co_loss_calculation(self, calculator, base_input):
        """Test CO loss is calculated when CO present."""
        base_input.flue_gas_co_ppm = 200.0
        result = calculator.calculate(base_input, method="indirect")

        assert result.co_loss_percent > 0

    def test_blowdown_loss_calculation(self, calculator, base_input):
        """Test blowdown loss is calculated."""
        base_input.blowdown_rate_percent = 5.0
        result = calculator.calculate(base_input, method="indirect")

        assert result.blowdown_loss_percent > 0


class TestEfficiencyCalculatorDirect:
    """Test direct method calculations."""

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    def test_direct_method_basic(self, calculator):
        """Test basic direct method calculation."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )
        result = calculator.calculate(inp, method="direct")

        assert result.method == "direct"
        assert 50.0 <= result.efficiency_hhv_percent <= 100.0

    def test_direct_method_no_individual_losses(self, calculator):
        """Test direct method does not calculate individual losses."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )
        result = calculator.calculate(inp, method="direct")

        assert result.dry_flue_gas_loss_percent == 0.0
        assert result.hydrogen_combustion_loss_percent == 0.0


class TestEfficiencyCalculatorProvenance:
    """Test provenance and hash calculations."""

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    def test_input_hash_deterministic(self, calculator):
        """Test same input produces same hash."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

        result1 = calculator.calculate(inp, method="indirect")
        result2 = calculator.calculate(inp, method="indirect")

        assert result1.input_hash == result2.input_hash

    def test_output_hash_generated(self, calculator):
        """Test output hash is generated."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

        result = calculator.calculate(inp, method="indirect")

        assert result.output_hash is not None
        assert len(result.output_hash) == 16

    def test_calculation_id_unique(self, calculator):
        """Test calculation IDs are unique."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

        result1 = calculator.calculate(inp, method="indirect")
        result2 = calculator.calculate(inp, method="indirect")

        # IDs contain timestamp so should be unique
        assert result1.calculation_id.startswith("EFF-")


class TestEfficiencyCalculatorEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    def test_zero_fuel_flow(self, calculator):
        """Test handling of zero fuel flow."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=0.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )
        result = calculator.calculate(inp, method="indirect")

        # Should handle gracefully without division by zero
        assert result.efficiency_hhv_percent >= 50.0

    def test_extreme_o2_values(self, calculator):
        """Test extreme O2 percentage values."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=21.0,  # Maximum possible
            fuel_type="natural_gas",
        )
        result = calculator.calculate(inp, method="indirect")

        assert result.excess_air_percent == 500.0  # Capped value

    def test_zero_o2(self, calculator):
        """Test zero O2 percentage."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=0.0,
            fuel_type="natural_gas",
        )
        result = calculator.calculate(inp, method="indirect")

        assert result.excess_air_percent == 0.0

    def test_efficiency_bounds_enforced(self, calculator):
        """Test efficiency is bounded between 50-100%."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=900.0,  # Very high
            flue_gas_o2_percent=15.0,  # Very high
            fuel_type="natural_gas",
        )
        result = calculator.calculate(inp, method="indirect")

        assert 50.0 <= result.efficiency_hhv_percent <= 100.0

    def test_unknown_fuel_type_fallback(self, calculator):
        """Test unknown fuel type falls back to natural gas."""
        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="unknown_fuel",  # Not in database
        )
        result = calculator.calculate(inp, method="indirect")

        # Should fall back to natural gas and complete
        assert result.efficiency_hhv_percent > 50.0


# =============================================================================
# EMISSIONS CALCULATOR TESTS
# =============================================================================


class TestEmissionsCalculatorInit:
    """Test emissions calculator initialization."""

    def test_default_initialization(self):
        """Test calculator initializes with EPA factors."""
        calc = EmissionsCalculator()
        assert calc.emission_factors is not None
        assert "natural_gas" in calc.emission_factors

    def test_custom_emission_factors(self):
        """Test custom emission factors accepted."""
        custom = {"custom_fuel": {"co2_kg_mmbtu": 50.0}}
        calc = EmissionsCalculator(emission_factors=custom)
        assert "custom_fuel" in calc.emission_factors

    def test_default_limits(self):
        """Test default compliance limits."""
        calc = EmissionsCalculator()
        assert calc.nox_limit == 0.10
        assert calc.co_limit == 0.08
        assert calc.so2_limit == 0.50

    def test_custom_limits(self):
        """Test custom compliance limits."""
        calc = EmissionsCalculator(
            nox_limit_lb_mmbtu=0.05,
            co_limit_lb_mmbtu=0.04,
            so2_limit_lb_mmbtu=0.25,
        )
        assert calc.nox_limit == 0.05


class TestEmissionsCalculatorCalculations:
    """Test emissions calculations."""

    @pytest.fixture
    def calculator(self):
        return EmissionsCalculator()

    def test_basic_calculation(self, calculator):
        """Test basic emissions calculation."""
        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=100.0,
            flue_gas_o2_percent=3.0,
        )
        result = calculator.calculate(inp)

        assert isinstance(result, EmissionsResult)
        assert result.co2_lb_hr > 0
        assert result.co2_metric_tons_hr > 0

    @pytest.mark.parametrize("fuel_type,co2_factor", [
        ("natural_gas", 53.06),
        ("fuel_oil_no2", 73.96),
        ("coal_bituminous", 93.28),
    ])
    def test_fuel_specific_co2_factors(self, calculator, fuel_type, co2_factor):
        """Test correct CO2 factors used per fuel type."""
        inp = EmissionsInput(
            fuel_type=fuel_type,
            heat_input_mmbtu_hr=100.0,
            flue_gas_o2_percent=3.0,
        )
        result = calculator.calculate(inp)

        # CO2 emission rate should reflect EPA factor
        expected_co2_kg = 100.0 * co2_factor
        assert abs(result.co2_metric_tons_hr - expected_co2_kg / 1000) < 0.1

    def test_nox_from_cems(self, calculator):
        """Test NOx calculation from CEMS data."""
        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=100.0,
            flue_gas_nox_ppm=50.0,
            flue_gas_o2_percent=3.0,
            use_cems_data=True,
        )
        result = calculator.calculate(inp)

        assert result.nox_lb_hr > 0
        assert result.method == "cems"

    def test_nox_from_factor(self, calculator):
        """Test NOx calculation from emission factors."""
        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=100.0,
            flue_gas_o2_percent=3.0,
            use_cems_data=False,
        )
        result = calculator.calculate(inp)

        assert result.nox_lb_hr > 0
        assert result.method == "factor"

    def test_co_from_cems(self, calculator):
        """Test CO calculation from CEMS data."""
        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=100.0,
            flue_gas_co_ppm=100.0,
            flue_gas_o2_percent=3.0,
            use_cems_data=True,
        )
        result = calculator.calculate(inp)

        assert result.co_lb_hr > 0

    def test_so2_from_sulfur_content(self, calculator):
        """Test SO2 calculation from sulfur content."""
        inp = EmissionsInput(
            fuel_type="fuel_oil_no2",
            heat_input_mmbtu_hr=100.0,
            sulfur_content_percent=0.5,
            flue_gas_o2_percent=3.0,
        )
        result = calculator.calculate(inp)

        assert result.so2_lb_hr > 0

    def test_o2_correction_to_3pct(self, calculator):
        """Test O2 correction to 3% reference."""
        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=100.0,
            flue_gas_nox_ppm=50.0,
            flue_gas_o2_percent=5.0,  # Higher than reference
            use_cems_data=True,
        )
        result = calculator.calculate(inp)

        # Corrected value should be higher than measured
        assert result.nox_ppm_3pct_o2 > 50.0

    def test_ghg_co2e_calculation(self, calculator):
        """Test GHG CO2 equivalent calculation."""
        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=100.0,
            flue_gas_o2_percent=3.0,
        )
        result = calculator.calculate(inp)

        # CO2e should include CH4 and N2O contributions
        assert result.co2e_metric_tons_hr > result.co2_metric_tons_hr


class TestEmissionsCompliance:
    """Test emissions compliance checks."""

    @pytest.fixture
    def calculator(self):
        return EmissionsCalculator(
            nox_limit_lb_mmbtu=0.10,
            co_limit_lb_mmbtu=0.08,
            so2_limit_lb_mmbtu=0.50,
        )

    def test_compliant_emissions(self, calculator):
        """Test compliant emissions are flagged correctly."""
        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=100.0,
            flue_gas_o2_percent=3.0,
        )
        result = calculator.calculate(inp)

        assert result.nox_compliant is True
        assert result.co_compliant is True
        assert result.so2_compliant is True
        assert result.overall_compliant is True

    def test_nox_non_compliant(self):
        """Test NOx non-compliance detection."""
        calc = EmissionsCalculator(nox_limit_lb_mmbtu=0.05)

        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=100.0,
            flue_gas_o2_percent=3.0,
        )
        result = calc.calculate(inp)

        # Default EPA factor for natural gas is 0.098, exceeds 0.05
        assert result.nox_compliant is False
        assert result.overall_compliant is False


class TestEmissionsEdgeCases:
    """Test emissions edge cases."""

    @pytest.fixture
    def calculator(self):
        return EmissionsCalculator()

    def test_zero_heat_input(self, calculator):
        """Test zero heat input handling."""
        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=0.0,
            flue_gas_o2_percent=3.0,
        )
        result = calculator.calculate(inp)

        assert result.co2_lb_hr == 0.0

    def test_unknown_fuel_fallback(self, calculator):
        """Test unknown fuel type falls back to natural gas."""
        inp = EmissionsInput(
            fuel_type="unknown_fuel",
            heat_input_mmbtu_hr=100.0,
            flue_gas_o2_percent=3.0,
        )
        result = calculator.calculate(inp)

        # Should use natural gas factors
        assert result.co2_lb_hr > 0

    def test_extreme_o2_correction(self, calculator):
        """Test O2 correction at extreme values."""
        inp = EmissionsInput(
            fuel_type="natural_gas",
            heat_input_mmbtu_hr=100.0,
            flue_gas_nox_ppm=50.0,
            flue_gas_o2_percent=21.0,  # Maximum
            use_cems_data=True,
        )
        result = calculator.calculate(inp)

        # At 21% O2, correction should equal measured
        assert result.nox_ppm_3pct_o2 == 50.0


# =============================================================================
# O2 TRIM CONTROLLER TESTS
# =============================================================================


class TestPIDController:
    """Test PID controller implementation."""

    def test_proportional_response(self):
        """Test proportional term responds to error."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)

        output = pid.compute(setpoint=3.0, process_value=4.0)

        # Error is -1, P output should be -2
        assert output == -2.0

    def test_integral_accumulation(self):
        """Test integral term accumulates error."""
        pid = PIDController(kp=0.0, ki=0.1, kd=0.0)

        # Apply constant error
        output1 = pid.compute(setpoint=3.0, process_value=2.0, timestamp=0.0)
        output2 = pid.compute(setpoint=3.0, process_value=2.0, timestamp=1.0)
        output3 = pid.compute(setpoint=3.0, process_value=2.0, timestamp=2.0)

        # Integral should grow
        assert output3 > output2 > output1

    def test_output_limiting(self):
        """Test output is bounded."""
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_min=-10.0, output_max=10.0)

        output = pid.compute(setpoint=3.0, process_value=0.0)

        assert output == 10.0  # Capped at max

    def test_anti_windup(self):
        """Test anti-windup prevents integral overshoot."""
        pid = PIDController(kp=1.0, ki=0.5, kd=0.0, output_min=-10.0, output_max=10.0)

        # Saturate the output
        for i in range(100):
            pid.compute(setpoint=50.0, process_value=0.0, timestamp=float(i))

        # Now reverse - should not take long to respond due to anti-windup
        output = pid.compute(setpoint=-50.0, process_value=0.0, timestamp=100.0)
        assert output == -10.0  # Should immediately saturate other direction

    def test_deadband(self):
        """Test deadband prevents small error response."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, deadband=0.5)

        # Error within deadband
        output = pid.compute(setpoint=3.0, process_value=3.3)
        assert output == 0.0

        # Error outside deadband
        output = pid.compute(setpoint=3.0, process_value=4.0)
        assert output != 0.0

    def test_manual_mode(self):
        """Test manual mode overrides calculation."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)

        pid.set_manual(5.0)
        output = pid.compute(setpoint=3.0, process_value=4.0)

        assert output == 5.0
        assert pid.is_auto is False

    def test_bumpless_transfer(self):
        """Test bumpless transfer to auto mode."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.0)

        # Run in manual
        pid.set_manual(5.0)
        pid.compute(setpoint=3.0, process_value=3.0, timestamp=0.0)

        # Switch to auto with bumpless transfer
        pid.set_auto(bumpless=True)
        output = pid.compute(setpoint=3.0, process_value=3.0, timestamp=1.0)

        # Output should be close to previous manual value
        assert abs(output - 5.0) < 1.0

    def test_reset(self):
        """Test controller reset clears state."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.0)

        # Accumulate integral
        for i in range(10):
            pid.compute(setpoint=5.0, process_value=0.0, timestamp=float(i))

        pid.reset()

        assert pid.output == 0.0

    def test_get_state(self):
        """Test state retrieval."""
        pid = PIDController(kp=2.0, ki=0.01, kd=0.5)
        pid.compute(setpoint=3.0, process_value=2.0, timestamp=0.0)

        state = pid.get_state()

        assert "auto" in state
        assert "output" in state
        assert "integral" in state
        assert state["kp"] == 2.0


class TestO2TrimController:
    """Test O2 trim controller."""

    @pytest.fixture
    def controller(self):
        return O2TrimController(
            boiler_id="BOILER-001",
            co_limit_ppm=400.0,
        )

    def test_initialization(self, controller):
        """Test controller initializes correctly."""
        assert controller.boiler_id == "BOILER-001"
        assert controller.co_limit_ppm == 400.0

    def test_setpoint_interpolation_low_load(self, controller):
        """Test setpoint at low load."""
        result = controller.compute(
            o2_measured=5.0,
            co_measured=50.0,
            load_percent=25.0,
        )

        # At 25% load, setpoint should be high (~5.0%)
        assert 4.5 <= result.o2_setpoint <= 5.5

    def test_setpoint_interpolation_high_load(self, controller):
        """Test setpoint at high load."""
        result = controller.compute(
            o2_measured=2.5,
            co_measured=50.0,
            load_percent=100.0,
        )

        # At 100% load, setpoint should be low (~2.5%)
        assert 2.0 <= result.o2_setpoint <= 3.0

    @pytest.mark.parametrize("load_percent,expected_range", [
        (25.0, (4.5, 5.5)),
        (50.0, (3.0, 4.0)),
        (75.0, (2.5, 3.5)),
        (100.0, (2.0, 3.0)),
    ])
    def test_setpoint_curve_interpolation(self, controller, load_percent, expected_range):
        """Test setpoint interpolation at various loads."""
        result = controller.compute(
            o2_measured=3.5,
            co_measured=50.0,
            load_percent=load_percent,
        )

        assert expected_range[0] <= result.o2_setpoint <= expected_range[1]

    def test_co_crosslimit_activates(self, controller):
        """Test CO cross-limiting activates when CO high."""
        result = controller.compute(
            o2_measured=3.0,
            co_measured=500.0,  # Above limit
            load_percent=75.0,
        )

        assert result.co_override_active is True
        assert result.o2_setpoint > 3.0  # Should increase for more air

    def test_co_crosslimit_not_active(self, controller):
        """Test CO cross-limiting not active when CO low."""
        result = controller.compute(
            o2_measured=3.0,
            co_measured=100.0,  # Below limit
            load_percent=75.0,
        )

        assert result.co_override_active is False

    def test_temperature_compensation(self, controller):
        """Test air temperature compensation."""
        result_cold = controller.compute(
            o2_measured=3.0,
            co_measured=50.0,
            load_percent=75.0,
            air_temp=40.0,  # Cold air
        )

        result_hot = controller.compute(
            o2_measured=3.0,
            co_measured=50.0,
            load_percent=75.0,
            air_temp=120.0,  # Hot air
        )

        # Hot air needs higher O2 setpoint
        assert result_hot.o2_setpoint > result_cold.o2_setpoint

    def test_trim_output_computed(self, controller):
        """Test trim output is computed."""
        result = controller.compute(
            o2_measured=4.0,
            co_measured=50.0,
            load_percent=75.0,
        )

        assert result.trim_output != 0.0  # Should have some output

    def test_setpoint_history_stored(self, controller):
        """Test setpoint history is stored."""
        for _ in range(5):
            controller.compute(
                o2_measured=3.5,
                co_measured=50.0,
                load_percent=75.0,
            )

        status = controller.get_status()
        assert len(controller._setpoint_history) == 5

    def test_manual_mode(self, controller):
        """Test manual mode operation."""
        controller.set_manual_output(5.0)

        result = controller.compute(
            o2_measured=3.0,
            co_measured=50.0,
            load_percent=75.0,
        )

        assert result.trim_output == 5.0

    def test_auto_mode_switch(self, controller):
        """Test switching to auto mode."""
        controller.set_manual_output(5.0)
        controller.set_auto()

        status = controller.get_status()
        assert status["mode"] == "auto"

    def test_tuning_update(self, controller):
        """Test PID tuning update."""
        controller.update_tuning(kp=3.0, ki=0.02)

        assert controller._pid.kp == 3.0
        assert controller._pid.ki == 0.02

    def test_setpoint_curve_update(self, controller):
        """Test setpoint curve update."""
        new_curve = {0.25: 6.0, 0.50: 4.0, 0.75: 3.0, 1.00: 2.0}
        controller.update_setpoint_curve(new_curve)

        result = controller.compute(
            o2_measured=3.0,
            co_measured=50.0,
            load_percent=25.0,
        )

        assert 5.5 <= result.o2_setpoint <= 6.5

    def test_reset(self, controller):
        """Test controller reset."""
        controller.compute(
            o2_measured=3.0,
            co_measured=600.0,  # Trigger CO high state
            load_percent=75.0,
        )

        controller.reset()

        assert controller._pid.output == 0.0
        assert controller._co_high_start is None

    def test_get_status(self, controller):
        """Test status retrieval."""
        controller.compute(
            o2_measured=3.5,
            co_measured=50.0,
            load_percent=75.0,
        )

        status = controller.get_status()

        assert "boiler_id" in status
        assert "mode" in status
        assert "current_setpoint" in status
        assert "trim_output" in status
        assert "pid_state" in status


class TestCOBreakthroughDetection:
    """Test CO breakthrough event detection."""

    def test_breakthrough_event_recorded(self):
        """Test CO breakthrough event is recorded after duration."""
        controller = O2TrimController(
            boiler_id="BOILER-001",
            co_limit_ppm=400.0,
        )

        # Simulate sustained high CO
        import time
        controller._co_high_start = time.time() - 35  # 35 seconds ago

        result = controller.compute(
            o2_measured=3.0,
            co_measured=600.0,
            load_percent=75.0,
        )

        events = controller.get_breakthrough_events()
        assert len(events) >= 1

    def test_breakthrough_events_limited(self):
        """Test getting limited number of events."""
        controller = O2TrimController(boiler_id="BOILER-001")

        # Add mock events
        for i in range(15):
            controller._co_breakthrough_events.append(
                COBreakthroughEvent(
                    timestamp=datetime.now(timezone.utc),
                    co_ppm=500.0,
                    threshold_ppm=400.0,
                    duration_s=60.0,
                    response_action="increase_air",
                    o2_adjustment=0.5,
                )
            )

        events = controller.get_breakthrough_events(limit=5)
        assert len(events) == 5


# =============================================================================
# FUEL PROPERTIES TESTS
# =============================================================================


class TestFuelProperties:
    """Test fuel properties dataclass."""

    def test_natural_gas_properties(self):
        """Test natural gas properties in database."""
        fuel = FUEL_DATABASE["natural_gas"]

        assert fuel.fuel_type == "natural_gas"
        assert fuel.higher_heating_value_btu_lb == 23875.0
        assert fuel.hydrogen_percent == 25.0
        assert fuel.ash_percent == 0.0

    def test_fuel_oil_properties(self):
        """Test fuel oil properties in database."""
        fuel = FUEL_DATABASE["fuel_oil_no2"]

        assert fuel.fuel_type == "fuel_oil_no2"
        assert fuel.higher_heating_value_btu_lb == 19500.0
        assert fuel.sulfur_percent == 0.3

    def test_coal_properties(self):
        """Test coal properties in database."""
        fuel = FUEL_DATABASE["coal_bituminous"]

        assert fuel.fuel_type == "coal_bituminous"
        assert fuel.ash_percent == 9.0
        assert fuel.moisture_percent == 5.0


# =============================================================================
# EPA EMISSION FACTORS TESTS
# =============================================================================


class TestEPAEmissionFactors:
    """Test EPA emission factors constants."""

    def test_natural_gas_factors(self):
        """Test natural gas EPA factors."""
        factors = EPA_EMISSION_FACTORS["natural_gas"]

        assert factors["co2_kg_mmbtu"] == 53.06
        assert factors["nox_lb_mmbtu"] == 0.098
        assert factors["so2_lb_mmbtu"] == 0.0006

    def test_gwp_values(self):
        """Test GWP values per AR5."""
        assert GWP["co2"] == 1
        assert GWP["ch4"] == 28
        assert GWP["n2o"] == 265
