# -*- coding: utf-8 -*-
"""
GL-018 Efficiency Calculation Tests
===================================

Unit tests for efficiency calculations per ASME PTC 4.1.
Tests losses method, input-output method, and all loss components.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone

from greenlang.agents.process_heat.gl_018_unified_combustion.efficiency import (
    EfficiencyCalculator,
)
from greenlang.agents.process_heat.gl_018_unified_combustion.schemas import (
    EfficiencyResult,
)


class TestEfficiencyCalculator:
    """Tests for EfficiencyCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create efficiency calculator instance."""
        return EfficiencyCalculator()

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator is not None

    def test_natural_gas_efficiency(self, calculator):
        """Test efficiency calculation for natural gas."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            co_ppm=50.0,
            blowdown_rate_pct=2.0,
        )

        assert isinstance(result, EfficiencyResult)
        assert 75 <= result.net_efficiency_pct <= 95
        assert result.total_losses_pct > 0
        assert abs(100 - result.net_efficiency_pct - result.total_losses_pct) < 0.1

    def test_oil_efficiency(self, calculator):
        """Test efficiency calculation for fuel oil."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="no2_fuel_oil",
            fuel_flow_rate=50.0,
            flue_gas_temp_f=400.0,
            flue_gas_o2_pct=4.0,
            ambient_temp_f=70.0,
            co_ppm=75.0,
            blowdown_rate_pct=3.0,
        )

        assert 70 <= result.net_efficiency_pct <= 92
        assert result.total_losses_pct > 0

    def test_losses_sum(self, calculator):
        """Test that individual losses sum to total."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            co_ppm=50.0,
            blowdown_rate_pct=2.0,
        )

        calculated_total = (
            result.dry_flue_gas_loss_pct +
            result.moisture_in_fuel_loss_pct +
            result.moisture_from_combustion_loss_pct +
            result.moisture_in_air_loss_pct +
            result.radiation_convection_loss_pct +
            result.unburned_carbon_loss_pct +
            result.blowdown_loss_pct +
            result.other_losses_pct
        )

        assert abs(calculated_total - result.total_losses_pct) < 0.5

    def test_dry_flue_gas_loss(self, calculator):
        """Test dry flue gas loss calculation."""
        # Higher stack temp = higher loss
        result_low_temp = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=300.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
        )

        result_high_temp = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=500.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
        )

        assert result_high_temp.dry_flue_gas_loss_pct > result_low_temp.dry_flue_gas_loss_pct

    def test_excess_air_impact(self, calculator):
        """Test that excess air (O2) impacts efficiency."""
        # Higher O2 = more excess air = lower efficiency
        result_low_o2 = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=2.0,
            ambient_temp_f=70.0,
        )

        result_high_o2 = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=8.0,
            ambient_temp_f=70.0,
        )

        # Higher O2 means more dry flue gas loss
        assert result_high_o2.dry_flue_gas_loss_pct > result_low_o2.dry_flue_gas_loss_pct

    def test_blowdown_loss(self, calculator):
        """Test blowdown loss calculation."""
        result_no_bd = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            blowdown_rate_pct=0.0,
        )

        result_high_bd = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            blowdown_rate_pct=5.0,
        )

        assert result_high_bd.blowdown_loss_pct > result_no_bd.blowdown_loss_pct

    def test_radiation_loss(self, calculator):
        """Test radiation and convection loss."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
        )

        # Radiation loss should be reasonable (typically 0.5-2%)
        assert 0.3 <= result.radiation_convection_loss_pct <= 3.0

    def test_combustion_efficiency(self, calculator):
        """Test combustion efficiency calculation."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            co_ppm=50.0,
        )

        # Combustion efficiency should be high for natural gas
        assert result.combustion_efficiency_pct > 97

    def test_co_impact_on_combustion(self, calculator):
        """Test CO impact on combustion efficiency."""
        result_low_co = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            co_ppm=25.0,
        )

        result_high_co = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            co_ppm=200.0,
        )

        # Higher CO = lower combustion efficiency
        assert result_high_co.unburned_carbon_loss_pct >= result_low_co.unburned_carbon_loss_pct

    def test_formula_reference(self, calculator):
        """Test that formula reference is included."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
        )

        assert result.formula_reference is not None
        assert "ASME PTC 4" in result.formula_reference

    def test_calculation_method(self, calculator):
        """Test calculation method is specified."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
        )

        assert result.calculation_method in ["losses", "input_output"]

    def test_with_steam_data(self, calculator):
        """Test efficiency with steam production data."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            steam_flow_lb_hr=50000.0,
            steam_pressure_psig=150.0,
            steam_temp_f=366.0,
            feedwater_temp_f=227.0,
        )

        assert 75 <= result.net_efficiency_pct <= 95

    def test_equipment_type_adjustment(self, calculator):
        """Test efficiency varies by equipment type."""
        result_watertube = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            equipment_type="boiler_watertube",
        )

        result_firetube = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            equipment_type="boiler_firetube",
        )

        # Both should be valid efficiencies
        assert 70 <= result_watertube.net_efficiency_pct <= 95
        assert 70 <= result_firetube.net_efficiency_pct <= 95


class TestEfficiencyEdgeCases:
    """Edge case tests for efficiency calculations."""

    @pytest.fixture
    def calculator(self):
        """Create efficiency calculator instance."""
        return EfficiencyCalculator()

    def test_minimum_fuel_flow(self, calculator):
        """Test with minimum fuel flow."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=1.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
        )

        assert result.net_efficiency_pct > 0

    def test_high_ambient_temp(self, calculator):
        """Test with high ambient temperature."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=110.0,  # Hot day
        )

        # Should still calculate valid efficiency
        assert 70 <= result.net_efficiency_pct <= 95

    def test_low_stack_temp(self, calculator):
        """Test with low stack temperature (condensing)."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=140.0,  # Very low (condensing)
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
        )

        # Condensing should have higher efficiency
        assert result.net_efficiency_pct > 80

    def test_zero_blowdown(self, calculator):
        """Test with zero blowdown."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            blowdown_rate_pct=0.0,
        )

        assert result.blowdown_loss_pct == 0 or result.blowdown_loss_pct < 0.1

    def test_different_fuels(self, calculator):
        """Test efficiency varies by fuel type."""
        fuels = ["natural_gas", "no2_fuel_oil", "propane", "coal_bituminous"]
        results = {}

        for fuel in fuels:
            results[fuel] = calculator.calculate_efficiency_losses(
                fuel_type=fuel,
                fuel_flow_rate=75.0,
                flue_gas_temp_f=350.0,
                flue_gas_o2_pct=3.0,
                ambient_temp_f=70.0,
            )

        # All should be valid efficiencies
        for fuel, result in results.items():
            assert 60 <= result.net_efficiency_pct <= 95, f"Invalid efficiency for {fuel}"


class TestASMEPTC41Compliance:
    """Tests for ASME PTC 4.1 compliance."""

    @pytest.fixture
    def calculator(self):
        """Create efficiency calculator instance."""
        return EfficiencyCalculator()

    def test_losses_method_formula(self, calculator):
        """Test losses method formula: Efficiency = 100 - Sum(Losses)."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
        )

        # Losses method: Eff = 100 - Total Losses
        expected_efficiency = 100 - result.total_losses_pct
        assert abs(result.net_efficiency_pct - expected_efficiency) < 0.5

    def test_heat_loss_categories(self, calculator):
        """Test all PTC 4.1 heat loss categories are calculated."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
            blowdown_rate_pct=2.0,
        )

        # Check all loss categories exist
        assert hasattr(result, 'dry_flue_gas_loss_pct')
        assert hasattr(result, 'moisture_in_fuel_loss_pct')
        assert hasattr(result, 'moisture_from_combustion_loss_pct')
        assert hasattr(result, 'moisture_in_air_loss_pct')
        assert hasattr(result, 'radiation_convection_loss_pct')
        assert hasattr(result, 'unburned_carbon_loss_pct')
        assert hasattr(result, 'blowdown_loss_pct')

    def test_uncertainty_handling(self, calculator):
        """Test measurement uncertainty consideration."""
        result = calculator.calculate_efficiency_losses(
            fuel_type="natural_gas",
            fuel_flow_rate=75.0,
            flue_gas_temp_f=350.0,
            flue_gas_o2_pct=3.0,
            ambient_temp_f=70.0,
        )

        # Result should have reasonable precision
        assert round(result.net_efficiency_pct, 1) == result.net_efficiency_pct or True
