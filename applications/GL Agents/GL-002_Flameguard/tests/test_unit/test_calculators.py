"""
GL-002 FLAMEGUARD - Calculator Unit Tests

Tests for efficiency, emissions, and heat balance calculators.
"""

import pytest
from datetime import datetime, timezone

# Import calculators
import sys
sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])

from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
)
from calculators.emissions_calculator import EmissionsCalculator
from calculators.fuel_blending_calculator import FuelBlendingCalculator
from calculators.heat_balance_calculator import (
    HeatBalanceCalculator,
    HeatBalanceInput,
)


class TestEfficiencyCalculator:
    """Tests for ASME PTC 4.1 efficiency calculator."""

    def test_indirect_efficiency_calculation(self, sample_efficiency_input):
        """Test indirect method efficiency calculation."""
        calc = EfficiencyCalculator()

        inp = EfficiencyInput(
            fuel_flow_rate=25000.0,
            fuel_hhv=1050.0,
            steam_flow=150.0,
            steam_enthalpy=1200.0,
            feedwater_enthalpy=200.0,
            flue_gas_temp_f=350.0,
            ambient_temp_f=70.0,
            o2_percent=3.5,
            fuel_type="natural_gas",
        )

        result = calc.calculate(inp, method="indirect")

        assert result is not None
        assert 70.0 <= result.gross_efficiency_percent <= 95.0
        assert result.calculation_hash != ""
        assert result.method == "indirect"

    def test_direct_efficiency_calculation(self):
        """Test direct method efficiency calculation."""
        calc = EfficiencyCalculator()

        inp = EfficiencyInput(
            fuel_flow_rate=25000.0,
            fuel_hhv=1050.0,
            steam_flow=150.0,
            steam_enthalpy=1200.0,
            feedwater_enthalpy=200.0,
            flue_gas_temp_f=350.0,
            ambient_temp_f=70.0,
            o2_percent=3.5,
            fuel_type="natural_gas",
        )

        result = calc.calculate(inp, method="direct")

        assert result is not None
        assert result.method == "direct"

    def test_efficiency_losses_sum_correctly(self):
        """Test that efficiency losses sum correctly."""
        calc = EfficiencyCalculator()

        inp = EfficiencyInput(
            fuel_flow_rate=25000.0,
            fuel_hhv=1050.0,
            steam_flow=150.0,
            steam_enthalpy=1200.0,
            feedwater_enthalpy=200.0,
            flue_gas_temp_f=350.0,
            ambient_temp_f=70.0,
            o2_percent=3.5,
            fuel_type="natural_gas",
        )

        result = calc.calculate(inp, method="indirect")

        # Verify losses sum to approximately 100 - efficiency
        total_losses = (
            result.dry_flue_gas_loss +
            result.moisture_in_fuel_loss +
            result.moisture_in_air_loss +
            result.moisture_from_h2_loss +
            result.radiation_loss +
            result.unaccounted_loss
        )

        expected_losses = 100.0 - result.gross_efficiency_percent
        assert abs(total_losses - expected_losses) < 1.0  # Within 1%

    def test_efficiency_provenance_hash(self):
        """Test that provenance hash is generated."""
        calc = EfficiencyCalculator()

        inp = EfficiencyInput(
            fuel_flow_rate=25000.0,
            fuel_hhv=1050.0,
            steam_flow=150.0,
            steam_enthalpy=1200.0,
            feedwater_enthalpy=200.0,
            flue_gas_temp_f=350.0,
            ambient_temp_f=70.0,
            o2_percent=3.5,
            fuel_type="natural_gas",
        )

        result1 = calc.calculate(inp, method="indirect")
        result2 = calc.calculate(inp, method="indirect")

        # Same inputs should produce same hash
        assert result1.calculation_hash == result2.calculation_hash

    def test_efficiency_with_high_o2(self):
        """Test efficiency drops with high O2."""
        calc = EfficiencyCalculator()

        base_inp = EfficiencyInput(
            fuel_flow_rate=25000.0,
            fuel_hhv=1050.0,
            steam_flow=150.0,
            steam_enthalpy=1200.0,
            feedwater_enthalpy=200.0,
            flue_gas_temp_f=350.0,
            ambient_temp_f=70.0,
            o2_percent=3.0,
            fuel_type="natural_gas",
        )

        high_o2_inp = EfficiencyInput(
            fuel_flow_rate=25000.0,
            fuel_hhv=1050.0,
            steam_flow=150.0,
            steam_enthalpy=1200.0,
            feedwater_enthalpy=200.0,
            flue_gas_temp_f=350.0,
            ambient_temp_f=70.0,
            o2_percent=6.0,  # Higher O2
            fuel_type="natural_gas",
        )

        base_result = calc.calculate(base_inp)
        high_o2_result = calc.calculate(high_o2_inp)

        # Higher O2 should result in lower efficiency
        assert high_o2_result.gross_efficiency_percent < base_result.gross_efficiency_percent


class TestEmissionsCalculator:
    """Tests for EPA emissions calculator."""

    def test_natural_gas_emissions(self, sample_emissions_input):
        """Test natural gas emissions calculation."""
        calc = EmissionsCalculator()

        result = calc.calculate(
            fuel_type="natural_gas",
            fuel_flow_scfh=sample_emissions_input["fuel_flow_scfh"],
            fuel_hhv_btu_scf=sample_emissions_input["fuel_hhv_btu_scf"],
            o2_percent=sample_emissions_input["o2_percent"],
            nox_ppm=sample_emissions_input["nox_ppm"],
            co_ppm=sample_emissions_input["co_ppm"],
        )

        assert result is not None
        assert result.nox_lb_hr > 0
        assert result.co_lb_hr > 0
        assert result.co2_ton_hr > 0
        assert result.ghg_mtco2e_hr > 0

    def test_o2_correction(self):
        """Test O2 correction to 3% reference."""
        calc = EmissionsCalculator()

        # NOx at 5% O2
        result = calc.calculate(
            fuel_type="natural_gas",
            fuel_flow_scfh=25000.0,
            fuel_hhv_btu_scf=1050.0,
            o2_percent=5.0,
            nox_ppm=45.0,
            co_ppm=25.0,
        )

        # Corrected NOx should be higher than measured
        assert result.nox_ppm_corrected > 45.0

    def test_ghg_calculation(self):
        """Test GHG CO2e calculation."""
        calc = EmissionsCalculator()

        result = calc.calculate(
            fuel_type="natural_gas",
            fuel_flow_scfh=25000.0,
            fuel_hhv_btu_scf=1050.0,
            o2_percent=3.5,
        )

        # GHG should include CO2, CH4, N2O contributions
        assert result.ghg_mtco2e_hr > 0
        # CO2 dominates for natural gas
        assert result.ghg_mtco2e_hr >= result.co2_ton_hr * 0.9

    def test_emission_factors_source(self):
        """Test that emission factors source is tracked."""
        calc = EmissionsCalculator()

        result = calc.calculate(
            fuel_type="natural_gas",
            fuel_flow_scfh=25000.0,
            fuel_hhv_btu_scf=1050.0,
            o2_percent=3.5,
        )

        assert "EPA" in result.emission_factors_source


class TestHeatBalanceCalculator:
    """Tests for heat balance calculator."""

    def test_heat_balance(self):
        """Test basic heat balance calculation."""
        calc = HeatBalanceCalculator()

        inp = HeatBalanceInput(
            fuel_input_mmbtu_hr=185.0,
            steam_output_mmbtu_hr=152.0,
            blowdown_mmbtu_hr=3.7,
            stack_loss_mmbtu_hr=19.4,
            radiation_loss_mmbtu_hr=2.8,
        )

        result = calc.calculate(inp)

        assert result is not None
        assert result.total_heat_input_mmbtu_hr == 185.0
        assert result.balanced or abs(result.balance_error_percent) < 5.0

    def test_heat_balance_with_recovery(self):
        """Test heat balance with economizer/air preheater recovery."""
        calc = HeatBalanceCalculator()

        inp = HeatBalanceInput(
            fuel_input_mmbtu_hr=185.0,
            steam_output_mmbtu_hr=152.0,
            blowdown_mmbtu_hr=3.7,
            stack_loss_mmbtu_hr=15.0,  # Lower due to recovery
            radiation_loss_mmbtu_hr=2.8,
            air_preheat_recovery_mmbtu_hr=4.0,
            economizer_recovery_mmbtu_hr=3.0,
        )

        result = calc.calculate(inp)

        assert result.total_recovery_mmbtu_hr == 7.0


class TestFuelBlendingCalculator:
    """Tests for fuel blending calculator."""

    def test_two_fuel_blend(self):
        """Test blending two fuels."""
        calc = FuelBlendingCalculator()

        result = calc.optimize_blend(
            fuels=[
                {"name": "natural_gas", "hhv": 1050.0, "cost": 5.0},
                {"name": "biogas", "hhv": 600.0, "cost": 3.0},
            ],
            target_hhv=900.0,
            optimize_for="cost",
        )

        assert result is not None
        assert "natural_gas" in result.blend_fractions
        assert "biogas" in result.blend_fractions
        assert abs(sum(result.blend_fractions.values()) - 1.0) < 0.001

    def test_blend_constraints(self):
        """Test blend with constraints."""
        calc = FuelBlendingCalculator()

        result = calc.optimize_blend(
            fuels=[
                {"name": "natural_gas", "hhv": 1050.0, "cost": 5.0, "max_fraction": 0.8},
                {"name": "landfill_gas", "hhv": 500.0, "cost": 2.0},
            ],
            target_hhv=850.0,
            optimize_for="cost",
        )

        assert result.blend_fractions.get("natural_gas", 0) <= 0.8
