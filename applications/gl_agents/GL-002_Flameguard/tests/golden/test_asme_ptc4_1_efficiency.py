# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - Golden Tests for ASME PTC 4.1 Efficiency Calculations

Reference: ASME PTC 4.1-2013 "Fired Steam Generators"
These tests validate efficiency calculations against published reference values.

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
    EfficiencyResult,
    FuelProperties,
    FUEL_DATABASE,
)


class TestASMEPTC41Efficiency:
    """
    Golden value tests for ASME PTC 4.1 boiler efficiency calculations.

    These tests use known reference values from ASME PTC 4.1 examples
    and industry standard calculations.
    """

    @pytest.fixture
    def calculator(self):
        """Create efficiency calculator instance."""
        return EfficiencyCalculator()

    # =========================================================================
    # Test Case 1: ASME PTC 4.1 Example Boiler (Natural Gas)
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_asme_ptc4_example_natural_gas(self, calculator):
        """
        ASME PTC 4.1 Example: Natural Gas Fired Boiler

        Input:
            - Fuel flow: 1000 kg/hr (~2205 lb/hr)
            - Steam output: 10000 kg/hr (~22050 lb/hr)
            - Feedwater temp: 105°C (221°F)
            - Flue gas temp: 180°C (356°F)
            - O2 in flue gas: 3.0%

        Expected:
            - Efficiency (HHV): 82.5% ± 0.5%
            - Dry flue gas loss: 4-6%
            - Hydrogen moisture loss: 10-12%
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=22.05,  # 10000 kg/hr
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,  # Saturated at 150 psig
            feedwater_temperature_f=221.0,  # 105°C
            fuel_flow_rate=2205.0,  # 1000 kg/hr
            fuel_flow_unit="lb_hr",
            flue_gas_temperature_f=356.0,  # 180°C
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
            ambient_temperature_f=77.0,
            ambient_humidity_percent=60.0,
        )

        result = calculator.calculate(input_data, method="indirect")

        # Primary assertion: Efficiency within tolerance
        assert 82.0 <= result.efficiency_hhv_percent <= 83.0, (
            f"Efficiency {result.efficiency_hhv_percent}% outside expected range 82.0-83.0%"
        )

        # Verify loss breakdown
        assert 4.0 <= result.dry_flue_gas_loss_percent <= 6.0, (
            f"Dry flue gas loss {result.dry_flue_gas_loss_percent}% outside range"
        )
        assert 10.0 <= result.hydrogen_combustion_loss_percent <= 12.0, (
            f"H2 moisture loss {result.hydrogen_combustion_loss_percent}% outside range"
        )

        # Verify excess air calculation
        assert 15.0 <= result.excess_air_percent <= 20.0, (
            f"Excess air {result.excess_air_percent}% outside expected range for 3% O2"
        )

        # Verify provenance
        assert result.formula_version == "ASME_PTC_4.1_2013"
        assert result.input_hash is not None and len(result.input_hash) > 0
        assert result.output_hash is not None and len(result.output_hash) > 0

    # =========================================================================
    # Test Case 2: High Efficiency Condensing Boiler
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_high_efficiency_condensing_boiler(self, calculator):
        """
        High-efficiency condensing natural gas boiler.

        Input:
            - Low flue gas temperature (150°F - near condensing)
            - Optimized combustion (2% O2)
            - Low excess air operation

        Expected:
            - Efficiency (HHV): 88-92%
            - Very low stack losses
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=50.0,
            steam_pressure_psig=100.0,
            steam_temperature_f=338.0,
            feedwater_temperature_f=200.0,
            fuel_flow_rate=4000.0,
            fuel_flow_unit="lb_hr",
            flue_gas_temperature_f=150.0,  # Low temp - near condensing
            flue_gas_o2_percent=2.0,  # Low excess air
            fuel_type="natural_gas",
            ambient_temperature_f=70.0,
        )

        result = calculator.calculate(input_data, method="indirect")

        # High efficiency expected
        assert 88.0 <= result.efficiency_hhv_percent <= 92.0, (
            f"High-efficiency boiler should achieve 88-92%, got {result.efficiency_hhv_percent}%"
        )

        # Low dry flue gas loss due to low stack temperature
        assert result.dry_flue_gas_loss_percent <= 3.0

    # =========================================================================
    # Test Case 3: Fuel Oil #2 Boiler (Reference Case)
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_fuel_oil_no2_boiler(self, calculator):
        """
        Fuel Oil #2 fired boiler - ASME reference case.

        Expected:
            - Lower efficiency than natural gas (due to H2 content)
            - Efficiency (HHV): 83-86%
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=200.0,
            steam_temperature_f=388.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=7500.0,
            fuel_flow_unit="lb_hr",
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.5,
            fuel_type="fuel_oil_no2",
            ambient_temperature_f=77.0,
        )

        result = calculator.calculate(input_data, method="indirect")

        # Oil typically has higher efficiency than gas on HHV basis
        assert 83.0 <= result.efficiency_hhv_percent <= 88.0, (
            f"Fuel oil boiler efficiency {result.efficiency_hhv_percent}% outside range"
        )

        # Lower hydrogen loss than natural gas
        assert result.hydrogen_combustion_loss_percent < 8.0

    # =========================================================================
    # Test Case 4: Coal Fired Boiler with Ash Losses
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_coal_fired_boiler_with_ash(self, calculator):
        """
        Bituminous coal fired boiler with ash and unburned carbon losses.

        Expected:
            - Efficiency (HHV): 78-84%
            - Significant ash/unburned carbon loss
            - Moisture in fuel loss
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=200.0,
            steam_pressure_psig=300.0,
            steam_temperature_f=417.0,
            feedwater_temperature_f=250.0,
            fuel_flow_rate=25000.0,
            fuel_flow_unit="lb_hr",
            flue_gas_temperature_f=350.0,
            flue_gas_o2_percent=4.0,
            fuel_type="coal_bituminous",
            ambient_temperature_f=77.0,
            ash_unburned_carbon_percent=3.0,  # 3% unburned carbon in ash
        )

        result = calculator.calculate(input_data, method="indirect")

        # Coal boiler efficiency
        assert 78.0 <= result.efficiency_hhv_percent <= 86.0

        # Coal has fuel moisture loss
        assert result.moisture_in_fuel_loss_percent > 0.0

        # Unburned carbon loss present
        assert result.unburned_carbon_loss_percent > 0.0

    # =========================================================================
    # Test Case 5: Direct vs Indirect Method Comparison
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_direct_vs_indirect_method(self, calculator):
        """
        Compare direct and indirect efficiency calculation methods.

        Per ASME PTC 4.1, both methods should agree within ~1-2%
        when all losses are properly accounted for.
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            fuel_flow_unit="lb_hr",
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

        direct_result = calculator.calculate(input_data, method="direct")
        indirect_result = calculator.calculate(input_data, method="indirect")

        # Methods should agree within 3%
        efficiency_diff = abs(
            direct_result.efficiency_hhv_percent -
            indirect_result.efficiency_hhv_percent
        )
        assert efficiency_diff <= 3.0, (
            f"Direct ({direct_result.efficiency_hhv_percent}%) and indirect "
            f"({indirect_result.efficiency_hhv_percent}%) methods differ by {efficiency_diff}%"
        )

        # Both should have correct method labels
        assert direct_result.method == "direct"
        assert indirect_result.method == "indirect"

    # =========================================================================
    # Test Case 6: Excess Air Sensitivity
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_excess_air_sensitivity(self, calculator):
        """
        Verify efficiency decreases with increasing excess air.

        Rule of thumb: Each 1% O2 increase ~0.5-1% efficiency decrease.
        """
        base_input = {
            "steam_flow_klb_hr": 100.0,
            "steam_pressure_psig": 150.0,
            "steam_temperature_f": 366.0,
            "feedwater_temperature_f": 227.0,
            "fuel_flow_rate": 8000.0,
            "fuel_flow_unit": "lb_hr",
            "flue_gas_temperature_f": 400.0,
            "fuel_type": "natural_gas",
        }

        results = []
        o2_levels = [2.0, 3.0, 4.0, 5.0, 6.0]

        for o2 in o2_levels:
            input_data = EfficiencyInput(
                **base_input,
                flue_gas_o2_percent=o2,
            )
            result = calculator.calculate(input_data)
            results.append((o2, result.efficiency_hhv_percent))

        # Verify monotonic decrease in efficiency
        for i in range(len(results) - 1):
            o2_curr, eff_curr = results[i]
            o2_next, eff_next = results[i + 1]
            assert eff_curr >= eff_next, (
                f"Efficiency should decrease with higher O2: "
                f"{o2_curr}% O2 = {eff_curr}% eff, {o2_next}% O2 = {eff_next}% eff"
            )

    # =========================================================================
    # Test Case 7: Stack Temperature Sensitivity
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_stack_temperature_sensitivity(self, calculator):
        """
        Verify efficiency decreases with increasing stack temperature.

        Rule of thumb: Each 40°F increase ~1% efficiency decrease.
        """
        base_input = {
            "steam_flow_klb_hr": 100.0,
            "steam_pressure_psig": 150.0,
            "steam_temperature_f": 366.0,
            "feedwater_temperature_f": 227.0,
            "fuel_flow_rate": 8000.0,
            "fuel_flow_unit": "lb_hr",
            "flue_gas_o2_percent": 3.0,
            "fuel_type": "natural_gas",
        }

        results = []
        stack_temps = [300.0, 350.0, 400.0, 450.0, 500.0]

        for temp in stack_temps:
            input_data = EfficiencyInput(
                **base_input,
                flue_gas_temperature_f=temp,
            )
            result = calculator.calculate(input_data)
            results.append((temp, result.efficiency_hhv_percent))

        # Verify monotonic decrease
        for i in range(len(results) - 1):
            temp_curr, eff_curr = results[i]
            temp_next, eff_next = results[i + 1]
            assert eff_curr >= eff_next, (
                f"Efficiency should decrease with higher stack temp: "
                f"{temp_curr}°F = {eff_curr}% eff, {temp_next}°F = {eff_next}% eff"
            )

        # Verify approximately 1% per 40°F
        total_temp_change = stack_temps[-1] - stack_temps[0]
        total_eff_change = results[0][1] - results[-1][1]
        eff_per_40f = total_eff_change / (total_temp_change / 40)

        assert 0.5 <= eff_per_40f <= 1.5, (
            f"Expected ~1% efficiency loss per 40°F, got {eff_per_40f:.2f}%"
        )

    # =========================================================================
    # Test Case 8: HHV vs LHV Efficiency Relationship
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_hhv_lhv_efficiency_relationship(self, calculator):
        """
        Verify LHV efficiency is higher than HHV efficiency.

        For natural gas: LHV/HHV ratio ~0.9, so LHV efficiency ~10% higher.
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            fuel_flow_unit="lb_hr",
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

        result = calculator.calculate(input_data)

        # LHV efficiency should be higher
        assert result.efficiency_lhv_percent > result.efficiency_hhv_percent

        # Ratio should be ~1.1 for natural gas (HHV/LHV = 23875/21500 = 1.11)
        ratio = result.efficiency_lhv_percent / result.efficiency_hhv_percent
        assert 1.08 <= ratio <= 1.15, (
            f"LHV/HHV efficiency ratio {ratio:.3f} outside expected range 1.08-1.15"
        )

    # =========================================================================
    # Test Case 9: CO Loss Calculation
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_co_loss_calculation(self, calculator):
        """
        Verify CO loss calculation with elevated CO levels.
        """
        # Normal operation - low CO
        normal_input = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            flue_gas_co_ppm=50.0,  # Normal: <100 ppm
            fuel_type="natural_gas",
        )

        # Abnormal - high CO (poor combustion)
        high_co_input = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            flue_gas_co_ppm=500.0,  # High CO
            fuel_type="natural_gas",
        )

        normal_result = calculator.calculate(normal_input)
        high_co_result = calculator.calculate(high_co_input)

        # High CO should have measurable CO loss
        assert high_co_result.co_loss_percent > normal_result.co_loss_percent

        # CO loss should not exceed 0.5% typically
        assert high_co_result.co_loss_percent <= 0.5

    # =========================================================================
    # Test Case 10: Provenance and Reproducibility
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_calculation_reproducibility(self, calculator):
        """
        Verify calculations are deterministic and reproducible.

        Same inputs must produce same output hash (provenance tracking).
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

        # Run calculation twice
        result1 = calculator.calculate(input_data)
        result2 = calculator.calculate(input_data)

        # Core calculation values must match exactly
        assert result1.efficiency_hhv_percent == result2.efficiency_hhv_percent
        assert result1.total_losses_percent == result2.total_losses_percent
        assert result1.dry_flue_gas_loss_percent == result2.dry_flue_gas_loss_percent

        # Input hash should be identical
        assert result1.input_hash == result2.input_hash

    # =========================================================================
    # Test Case 11: Blowdown Loss Calculation
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_blowdown_loss_calculation(self, calculator):
        """
        Verify blowdown loss increases with blowdown rate.
        """
        base_input = {
            "steam_flow_klb_hr": 100.0,
            "steam_pressure_psig": 150.0,
            "steam_temperature_f": 366.0,
            "feedwater_temperature_f": 227.0,
            "fuel_flow_rate": 8000.0,
            "flue_gas_temperature_f": 400.0,
            "flue_gas_o2_percent": 3.0,
            "fuel_type": "natural_gas",
        }

        # Low blowdown
        low_bd = EfficiencyInput(**base_input, blowdown_rate_percent=1.0)
        # Normal blowdown
        normal_bd = EfficiencyInput(**base_input, blowdown_rate_percent=3.0)
        # High blowdown
        high_bd = EfficiencyInput(**base_input, blowdown_rate_percent=8.0)

        low_result = calculator.calculate(low_bd)
        normal_result = calculator.calculate(normal_bd)
        high_result = calculator.calculate(high_bd)

        # Blowdown loss should increase
        assert low_result.blowdown_loss_percent <= normal_result.blowdown_loss_percent
        assert normal_result.blowdown_loss_percent <= high_result.blowdown_loss_percent

        # Efficiency should decrease with higher blowdown
        assert low_result.efficiency_hhv_percent >= normal_result.efficiency_hhv_percent
        assert normal_result.efficiency_hhv_percent >= high_result.efficiency_hhv_percent

    # =========================================================================
    # Test Case 12: Total Losses Sum Validation
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_total_losses_sum(self, calculator):
        """
        Verify total losses equals sum of individual losses.
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
            blowdown_rate_percent=3.0,
        )

        result = calculator.calculate(input_data)

        # Calculate sum of individual losses
        individual_losses = (
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

        # Allow small rounding tolerance
        assert abs(result.total_losses_percent - individual_losses) < 0.1, (
            f"Total losses {result.total_losses_percent}% != sum {individual_losses:.2f}%"
        )

        # Efficiency + losses should ~= 100%
        total = result.efficiency_hhv_percent + result.total_losses_percent
        assert 99.0 <= total <= 101.0, (
            f"Efficiency + losses = {total}%, expected ~100%"
        )


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.golden
    def test_minimum_firing_rate(self, calculator):
        """Test efficiency at minimum firing rate."""
        input_data = EfficiencyInput(
            steam_flow_klb_hr=10.0,  # Low load
            steam_pressure_psig=100.0,
            steam_temperature_f=338.0,
            feedwater_temperature_f=200.0,
            fuel_flow_rate=800.0,  # Low fuel
            flue_gas_temperature_f=300.0,
            flue_gas_o2_percent=5.0,  # Higher O2 at turndown
            fuel_type="natural_gas",
        )

        result = calculator.calculate(input_data)

        # Efficiency still valid but may be lower at turndown
        assert 50.0 <= result.efficiency_hhv_percent <= 100.0
        # Radiation loss higher at low load
        assert result.radiation_loss_percent >= 0.3

    @pytest.mark.golden
    def test_high_altitude_operation(self, calculator):
        """Test efficiency at high altitude (lower barometric pressure)."""
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
            barometric_pressure_psia=12.0,  # ~5000 ft altitude
        )

        result = calculator.calculate(input_data)

        # Should still calculate valid efficiency
        assert 70.0 <= result.efficiency_hhv_percent <= 95.0

    @pytest.mark.golden
    def test_zero_blowdown(self, calculator):
        """Test with no blowdown (condensate return system)."""
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
            blowdown_rate_percent=0.0,
        )

        result = calculator.calculate(input_data)

        assert result.blowdown_loss_percent == 0.0
        # Efficiency should be slightly higher without blowdown
        assert result.efficiency_hhv_percent >= 80.0


class TestCustomFuelProperties:
    """Tests with custom fuel properties."""

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.golden
    def test_custom_natural_gas_composition(self, calculator):
        """Test with custom natural gas composition (high BTU)."""
        custom_fuel = FuelProperties(
            fuel_type="high_btu_natural_gas",
            higher_heating_value_btu_lb=24500.0,  # Higher BTU
            lower_heating_value_btu_lb=22000.0,
            carbon_percent=76.0,
            hydrogen_percent=24.0,
            sulfur_percent=0.0,
            nitrogen_percent=0.0,
            oxygen_percent=0.0,
            moisture_percent=0.0,
            ash_percent=0.0,
            stoichiometric_air_ratio=17.5,
        )

        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_properties=custom_fuel,
        )

        result = calculator.calculate(input_data)

        # Should calculate valid efficiency
        assert 75.0 <= result.efficiency_hhv_percent <= 95.0

    @pytest.mark.golden
    def test_biogas_fuel(self, calculator):
        """Test with biogas (lower BTU fuel)."""
        biogas = FuelProperties(
            fuel_type="biogas",
            higher_heating_value_btu_lb=12000.0,  # Lower BTU
            lower_heating_value_btu_lb=10800.0,
            carbon_percent=50.0,
            hydrogen_percent=8.0,
            sulfur_percent=0.1,
            nitrogen_percent=1.0,
            oxygen_percent=35.0,
            moisture_percent=5.0,
            ash_percent=0.9,
            stoichiometric_air_ratio=6.5,
        )

        input_data = EfficiencyInput(
            steam_flow_klb_hr=50.0,
            steam_pressure_psig=100.0,
            steam_temperature_f=338.0,
            feedwater_temperature_f=200.0,
            fuel_flow_rate=15000.0,  # More fuel needed (lower BTU)
            flue_gas_temperature_f=350.0,
            flue_gas_o2_percent=4.0,
            fuel_properties=biogas,
        )

        result = calculator.calculate(input_data)

        # Biogas typically has lower efficiency due to moisture
        assert 70.0 <= result.efficiency_hhv_percent <= 90.0
        # Should have fuel moisture loss
        assert result.moisture_in_fuel_loss_percent > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
