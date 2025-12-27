# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - Golden Tests for Stack Loss Calculations

Reference: ASME PTC 4.1-2013 Heat Loss Method
Tests verify individual loss components: dry gas, moisture, hydrogen, radiation.

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
    FuelProperties,
    FUEL_DATABASE,
)


class TestDryFlueGasLoss:
    """
    Tests for dry flue gas heat loss calculation.

    L1 = Cp * (Tfg - Ta) * AFR / HHV * 100

    Where:
    - Cp: Specific heat of flue gas (~0.24 BTU/lb-°F)
    - Tfg: Flue gas temperature
    - Ta: Ambient temperature
    - AFR: Air-fuel ratio
    - HHV: Higher heating value
    """

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.golden
    @pytest.mark.asme
    def test_dry_gas_loss_typical_range(self, calculator):
        """
        Typical dry flue gas loss for natural gas: 4-7%
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
            ambient_temperature_f=77.0,
        )

        result = calculator.calculate(input_data)

        # Dry flue gas loss should be in typical range
        assert 3.0 <= result.dry_flue_gas_loss_percent <= 8.0, (
            f"Dry flue gas loss {result.dry_flue_gas_loss_percent}% outside 3-8% range"
        )

    @pytest.mark.golden
    @pytest.mark.asme
    def test_dry_gas_loss_increases_with_stack_temp(self, calculator):
        """
        Dry gas loss should increase linearly with stack temperature.
        """
        base_input = {
            "steam_flow_klb_hr": 100.0,
            "steam_pressure_psig": 150.0,
            "steam_temperature_f": 366.0,
            "feedwater_temperature_f": 227.0,
            "fuel_flow_rate": 8000.0,
            "flue_gas_o2_percent": 3.0,
            "fuel_type": "natural_gas",
            "ambient_temperature_f": 77.0,
        }

        stack_temps = [300.0, 400.0, 500.0]
        losses = []

        for temp in stack_temps:
            input_data = EfficiencyInput(**base_input, flue_gas_temperature_f=temp)
            result = calculator.calculate(input_data)
            losses.append(result.dry_flue_gas_loss_percent)

        # Verify monotonic increase
        assert losses[0] < losses[1] < losses[2], (
            f"Dry gas loss should increase: {losses}"
        )

        # Verify approximately linear relationship
        delta1 = losses[1] - losses[0]  # 300->400°F
        delta2 = losses[2] - losses[1]  # 400->500°F
        assert abs(delta1 - delta2) < 0.5, "Loss increase should be linear"

    @pytest.mark.golden
    @pytest.mark.asme
    def test_dry_gas_loss_increases_with_excess_air(self, calculator):
        """
        Dry gas loss should increase with higher excess air (O2).
        """
        base_input = {
            "steam_flow_klb_hr": 100.0,
            "steam_pressure_psig": 150.0,
            "steam_temperature_f": 366.0,
            "feedwater_temperature_f": 227.0,
            "fuel_flow_rate": 8000.0,
            "flue_gas_temperature_f": 400.0,
            "fuel_type": "natural_gas",
            "ambient_temperature_f": 77.0,
        }

        o2_levels = [2.0, 4.0, 6.0]
        losses = []

        for o2 in o2_levels:
            input_data = EfficiencyInput(**base_input, flue_gas_o2_percent=o2)
            result = calculator.calculate(input_data)
            losses.append(result.dry_flue_gas_loss_percent)

        # More excess air = more flue gas mass = more heat loss
        assert losses[0] < losses[1] < losses[2], (
            f"Dry gas loss should increase with O2: {losses}"
        )


class TestMoistureLosses:
    """
    Tests for moisture-related heat losses.

    Includes:
    - L2: Moisture in fuel
    - L3: Moisture from hydrogen combustion (9 lb H2O per lb H2)
    - L4: Moisture in combustion air
    """

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.golden
    @pytest.mark.asme
    def test_hydrogen_moisture_loss_natural_gas(self, calculator):
        """
        Natural gas: 25% hydrogen -> significant moisture loss.

        9 lb H2O per lb H2 * 25% H2 = 2.25 lb H2O per lb fuel
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

        result = calculator.calculate(input_data)

        # Natural gas has high hydrogen content -> high moisture loss
        assert 10.0 <= result.hydrogen_combustion_loss_percent <= 13.0, (
            f"H2 moisture loss {result.hydrogen_combustion_loss_percent}% outside range"
        )

    @pytest.mark.golden
    @pytest.mark.asme
    def test_hydrogen_moisture_loss_fuel_oil(self, calculator):
        """
        Fuel oil: 12.5% hydrogen -> lower moisture loss than gas.
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=3.0,
            fuel_type="fuel_oil_no2",
        )

        result = calculator.calculate(input_data)

        # Fuel oil has less hydrogen
        assert 5.0 <= result.hydrogen_combustion_loss_percent <= 8.0

    @pytest.mark.golden
    @pytest.mark.asme
    def test_fuel_moisture_loss_coal(self, calculator):
        """
        Coal with 5% moisture should have measurable fuel moisture loss.
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=200.0,
            steam_pressure_psig=300.0,
            steam_temperature_f=417.0,
            feedwater_temperature_f=250.0,
            fuel_flow_rate=25000.0,
            flue_gas_temperature_f=350.0,
            flue_gas_o2_percent=4.0,
            fuel_type="coal_bituminous",
        )

        result = calculator.calculate(input_data)

        # Coal has 5% moisture -> measurable loss
        assert result.moisture_in_fuel_loss_percent > 0.0, (
            "Coal should have fuel moisture loss"
        )
        assert result.moisture_in_fuel_loss_percent < 3.0, (
            "Fuel moisture loss should be reasonable"
        )

    @pytest.mark.golden
    @pytest.mark.asme
    def test_air_moisture_loss(self, calculator):
        """
        Moisture in combustion air contributes small loss.
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

        # Low humidity
        low_humid = EfficiencyInput(**base_input, ambient_humidity_percent=30.0)
        # High humidity
        high_humid = EfficiencyInput(**base_input, ambient_humidity_percent=90.0)

        low_result = calculator.calculate(low_humid)
        high_result = calculator.calculate(high_humid)

        # Air moisture loss should be small (typically <0.5%)
        assert low_result.moisture_in_air_loss_percent < 0.5
        assert high_result.moisture_in_air_loss_percent < 1.0

        # Higher humidity = higher loss
        assert high_result.moisture_in_air_loss_percent >= low_result.moisture_in_air_loss_percent


class TestRadiationLoss:
    """
    Tests for radiation and convection heat loss.

    Uses ABMA correlation: L7 ~ 1.5 / (capacity)^0.15
    """

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.golden
    @pytest.mark.asme
    def test_radiation_loss_decreases_with_capacity(self, calculator):
        """
        Radiation loss % decreases with increasing boiler capacity.

        Small boilers: 2-3%
        Large boilers: 0.5-1%
        """
        base_input = {
            "steam_pressure_psig": 150.0,
            "steam_temperature_f": 366.0,
            "feedwater_temperature_f": 227.0,
            "flue_gas_temperature_f": 400.0,
            "flue_gas_o2_percent": 3.0,
            "fuel_type": "natural_gas",
        }

        # Small boiler (10 klb/hr)
        small = EfficiencyInput(
            steam_flow_klb_hr=10.0,
            fuel_flow_rate=800.0,
            **base_input,
        )

        # Large boiler (200 klb/hr)
        large = EfficiencyInput(
            steam_flow_klb_hr=200.0,
            fuel_flow_rate=16000.0,
            **base_input,
        )

        small_result = calculator.calculate(small)
        large_result = calculator.calculate(large)

        # Radiation % is higher for small boilers (surface area to volume ratio)
        assert small_result.radiation_loss_percent > large_result.radiation_loss_percent

        # Verify reasonable ranges
        assert 0.3 <= large_result.radiation_loss_percent <= 1.5
        assert 1.0 <= small_result.radiation_loss_percent <= 3.0

    @pytest.mark.golden
    @pytest.mark.asme
    def test_radiation_loss_minimum(self, calculator):
        """
        Radiation loss should never be below 0.3% (well-insulated).
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=500.0,  # Very large boiler
            steam_pressure_psig=400.0,
            steam_temperature_f=450.0,
            feedwater_temperature_f=300.0,
            fuel_flow_rate=50000.0,
            flue_gas_temperature_f=350.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
        )

        result = calculator.calculate(input_data)

        # Even large boilers have minimum radiation loss
        assert result.radiation_loss_percent >= 0.3

    @pytest.mark.golden
    @pytest.mark.asme
    def test_radiation_loss_maximum(self, calculator):
        """
        Radiation loss capped at 3% per ABMA guidelines.
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=5.0,  # Very small boiler
            steam_pressure_psig=50.0,
            steam_temperature_f=298.0,
            feedwater_temperature_f=150.0,
            fuel_flow_rate=500.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=4.0,
            fuel_type="natural_gas",
        )

        result = calculator.calculate(input_data)

        # Should not exceed 3%
        assert result.radiation_loss_percent <= 3.0


class TestUnburnedCarbonLoss:
    """
    Tests for unburned carbon loss (solid fuels only).
    """

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.golden
    @pytest.mark.asme
    def test_unburned_carbon_coal(self, calculator):
        """
        Coal with ash should have unburned carbon loss.

        L5 = ash% * unburned_C% * 14500 BTU/lb / HHV * 100
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=200.0,
            steam_pressure_psig=300.0,
            steam_temperature_f=417.0,
            feedwater_temperature_f=250.0,
            fuel_flow_rate=25000.0,
            flue_gas_temperature_f=350.0,
            flue_gas_o2_percent=4.0,
            fuel_type="coal_bituminous",
            ash_unburned_carbon_percent=3.0,  # 3% unburned C in ash
        )

        result = calculator.calculate(input_data)

        # Should have measurable unburned carbon loss
        assert result.unburned_carbon_loss_percent > 0.0
        assert result.unburned_carbon_loss_percent < 1.0  # Typically <1%

    @pytest.mark.golden
    @pytest.mark.asme
    def test_no_unburned_carbon_gas(self, calculator):
        """
        Natural gas has no ash -> no unburned carbon loss.
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

        result = calculator.calculate(input_data)

        # Gas has no ash -> no unburned carbon
        assert result.unburned_carbon_loss_percent == 0.0

    @pytest.mark.golden
    @pytest.mark.asme
    def test_high_unburned_carbon(self, calculator):
        """
        High unburned carbon indicates poor combustion.
        """
        input_data = EfficiencyInput(
            steam_flow_klb_hr=200.0,
            steam_pressure_psig=300.0,
            steam_temperature_f=417.0,
            feedwater_temperature_f=250.0,
            fuel_flow_rate=25000.0,
            flue_gas_temperature_f=350.0,
            flue_gas_o2_percent=4.0,
            fuel_type="coal_bituminous",
            ash_unburned_carbon_percent=10.0,  # Poor combustion
        )

        result = calculator.calculate(input_data)

        # Higher unburned carbon = higher loss
        assert result.unburned_carbon_loss_percent > 0.5


class TestTotalStackLoss:
    """
    Tests for total stack loss (sum of all losses).
    """

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.golden
    @pytest.mark.asme
    def test_total_loss_natural_gas_typical(self, calculator):
        """
        Total stack losses for natural gas: typically 15-20%.
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

        result = calculator.calculate(input_data)

        # Total losses should be 15-22%
        assert 15.0 <= result.total_losses_percent <= 22.0

    @pytest.mark.golden
    @pytest.mark.asme
    def test_loss_breakdown_sum(self, calculator):
        """
        Sum of individual losses should equal total losses.
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
            flue_gas_co_ppm=50.0,
        )

        result = calculator.calculate(input_data)

        # Calculate sum
        calculated_sum = (
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

        # Should match within rounding tolerance
        assert abs(result.total_losses_percent - calculated_sum) < 0.2

    @pytest.mark.golden
    @pytest.mark.asme
    def test_efficiency_plus_losses_equals_100(self, calculator):
        """
        Efficiency + Total Losses should equal ~100%.
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

        result = calculator.calculate(input_data)

        total = result.efficiency_hhv_percent + result.total_losses_percent

        # Should be ~100%
        assert 99.0 <= total <= 101.0, (
            f"Efficiency ({result.efficiency_hhv_percent}%) + "
            f"Losses ({result.total_losses_percent}%) = {total}%"
        )


class TestLossCalculationPrecision:
    """
    Tests for calculation precision and determinism.
    """

    @pytest.fixture
    def calculator(self):
        return EfficiencyCalculator()

    @pytest.mark.golden
    def test_reproducible_loss_calculations(self, calculator):
        """
        Same inputs should produce identical loss values.
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

        result1 = calculator.calculate(input_data)
        result2 = calculator.calculate(input_data)

        # All loss values should be identical
        assert result1.dry_flue_gas_loss_percent == result2.dry_flue_gas_loss_percent
        assert result1.hydrogen_combustion_loss_percent == result2.hydrogen_combustion_loss_percent
        assert result1.radiation_loss_percent == result2.radiation_loss_percent
        assert result1.total_losses_percent == result2.total_losses_percent

    @pytest.mark.golden
    def test_loss_rounding(self, calculator):
        """
        Loss values should be rounded to 2 decimal places.
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

        result = calculator.calculate(input_data)

        # Check rounding to 2 decimal places
        assert result.dry_flue_gas_loss_percent == round(result.dry_flue_gas_loss_percent, 2)
        assert result.hydrogen_combustion_loss_percent == round(result.hydrogen_combustion_loss_percent, 2)
        assert result.total_losses_percent == round(result.total_losses_percent, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
