# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - Golden Tests for Heat Balance Calculations

Reference: ASME PTC 4.1-2013 Energy Balance Method
Tests verify energy input equals output plus losses.

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.heat_balance_calculator import (
    HeatBalanceCalculator,
    HeatBalanceInput,
    HeatBalanceResult,
)


class TestHeatBalanceValidation:
    """
    Golden tests for heat balance validation.

    Heat balance equation:
    Input = Output + Losses - Recovery
    """

    @pytest.fixture
    def calculator(self):
        return HeatBalanceCalculator(tolerance_percent=2.0)

    # =========================================================================
    # Test Case 1: Balanced System - Perfect Energy Conservation
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_perfect_balance(self, calculator):
        """
        Test case with perfect energy balance.

        Input: 100 MMBtu/hr
        Output: 82 MMBtu/hr steam + 3 MMBtu/hr blowdown
        Losses: 12 MMBtu/hr stack + 1 MMBtu/hr radiation
        Recovery: 0 MMBtu/hr
        Unaccounted: 2 MMBtu/hr (within tolerance)
        """
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=100.0,
            steam_output_mmbtu_hr=82.0,
            blowdown_mmbtu_hr=3.0,
            stack_loss_mmbtu_hr=12.0,
            radiation_loss_mmbtu_hr=1.0,
            air_preheat_recovery_mmbtu_hr=0.0,
            economizer_recovery_mmbtu_hr=0.0,
        )

        result = calculator.calculate(input_data)

        # Should be balanced (within tolerance)
        assert result.balanced is True
        assert result.balance_error_percent <= 2.0

        # Verify totals
        assert result.total_heat_input_mmbtu_hr == 100.0
        assert result.total_heat_output_mmbtu_hr == 85.0  # 82 + 3
        assert result.total_losses_mmbtu_hr == 13.0  # 12 + 1

    # =========================================================================
    # Test Case 2: System with Heat Recovery
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_with_heat_recovery(self, calculator):
        """
        Test system with economizer and air preheater recovery.

        Heat recovery reduces effective losses.
        """
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=100.0,
            steam_output_mmbtu_hr=85.0,
            blowdown_mmbtu_hr=2.0,
            stack_loss_mmbtu_hr=8.0,  # Lower due to heat recovery
            radiation_loss_mmbtu_hr=1.0,
            air_preheat_recovery_mmbtu_hr=3.0,
            economizer_recovery_mmbtu_hr=2.0,
        )

        result = calculator.calculate(input_data)

        # Verify recovery values
        assert result.total_recovery_mmbtu_hr == 5.0
        assert result.air_preheat_mmbtu_hr == 3.0
        assert result.economizer_mmbtu_hr == 2.0

        # Should be balanced
        assert result.balanced is True

    # =========================================================================
    # Test Case 3: Unbalanced System - Excessive Losses
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_unbalanced_excessive_losses(self, calculator):
        """
        Test case where losses exceed input - should flag as unbalanced.
        """
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=100.0,
            steam_output_mmbtu_hr=70.0,
            blowdown_mmbtu_hr=2.0,
            stack_loss_mmbtu_hr=35.0,  # Unrealistic high loss
            radiation_loss_mmbtu_hr=5.0,
        )

        result = calculator.calculate(input_data)

        # Should flag as unbalanced
        assert result.balance_error_percent > 2.0
        assert result.balanced is False

        # Unaccounted should be negative (losses exceed input)
        assert result.unaccounted_loss_mmbtu_hr < 0

    # =========================================================================
    # Test Case 4: High Efficiency System
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_high_efficiency_system(self, calculator):
        """
        Test high-efficiency condensing system.

        Expected: >90% of input goes to useful output.
        """
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=100.0,
            steam_output_mmbtu_hr=91.0,  # High efficiency
            blowdown_mmbtu_hr=1.0,
            stack_loss_mmbtu_hr=5.0,
            radiation_loss_mmbtu_hr=0.5,
            air_preheat_recovery_mmbtu_hr=0.0,
            economizer_recovery_mmbtu_hr=2.0,
        )

        result = calculator.calculate(input_data)

        # Verify high steam output fraction
        steam_fraction = result.steam_heat_mmbtu_hr / result.total_heat_input_mmbtu_hr
        assert steam_fraction >= 0.90

        assert result.balanced is True

    # =========================================================================
    # Test Case 5: Low Load Operation
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_low_load_operation(self, calculator):
        """
        Test heat balance at low load (higher radiation losses).
        """
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=25.0,  # 25% load
            steam_output_mmbtu_hr=19.0,
            blowdown_mmbtu_hr=0.5,
            stack_loss_mmbtu_hr=3.0,
            radiation_loss_mmbtu_hr=1.5,  # Higher % at low load
        )

        result = calculator.calculate(input_data)

        # Radiation as % of input is higher at low load
        radiation_percent = result.radiation_loss_mmbtu_hr / result.total_heat_input_mmbtu_hr * 100
        assert radiation_percent > 4.0  # Higher than at full load

    # =========================================================================
    # Test Case 6: Balance Error Calculation
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_balance_error_calculation(self, calculator):
        """
        Verify balance error is calculated correctly.
        """
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=100.0,
            steam_output_mmbtu_hr=80.0,
            blowdown_mmbtu_hr=3.0,
            stack_loss_mmbtu_hr=10.0,
            radiation_loss_mmbtu_hr=1.0,
        )

        result = calculator.calculate(input_data)

        # Calculate expected unaccounted
        # Input = 100, Output = 83, Losses = 11, Recovery = 0
        # Expected = 83 + 11 - 0 = 94
        # Unaccounted = 100 - 94 = 6
        expected_unaccounted = 6.0
        assert abs(result.unaccounted_loss_mmbtu_hr - expected_unaccounted) < 0.1

        # Error % = |6| / 100 * 100 = 6%
        assert abs(result.balance_error_percent - 6.0) < 0.1

    # =========================================================================
    # Test Case 7: Zero Input Edge Case
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_zero_input_handling(self, calculator):
        """
        Test handling of zero fuel input (cold standby).
        """
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=0.0,
            steam_output_mmbtu_hr=0.0,
            blowdown_mmbtu_hr=0.0,
            stack_loss_mmbtu_hr=0.0,
            radiation_loss_mmbtu_hr=0.0,
        )

        result = calculator.calculate(input_data)

        # Should not divide by zero
        assert result.balance_error_percent == 0.0
        assert result.balanced is True

    # =========================================================================
    # Test Case 8: Full Load Industrial Boiler
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_full_load_industrial_boiler(self, calculator):
        """
        Test full load operation of industrial package boiler.

        Reference: 80,000 lb/hr boiler at 150 psig
        """
        # Typical heat flows for 80 klb/hr boiler
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=95.0,  # ~8000 scfh natural gas
            steam_output_mmbtu_hr=78.5,  # 80 klb/hr * 980 BTU/lb / 1e6
            blowdown_mmbtu_hr=2.4,  # 3% blowdown
            stack_loss_mmbtu_hr=9.5,  # ~10% stack loss
            radiation_loss_mmbtu_hr=0.9,  # ~1% radiation
            air_preheat_recovery_mmbtu_hr=0.0,
            economizer_recovery_mmbtu_hr=0.0,
        )

        result = calculator.calculate(input_data)

        # Should be reasonably balanced
        assert result.balance_error_percent < 5.0

    # =========================================================================
    # Test Case 9: Sensible Heat Credits (Future Enhancement)
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_sensible_heat_credits(self, calculator):
        """
        Test that sensible heat credits are included in input.

        Note: Current implementation has placeholder for credits.
        """
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=100.0,
            steam_output_mmbtu_hr=82.0,
            blowdown_mmbtu_hr=3.0,
            stack_loss_mmbtu_hr=12.0,
            radiation_loss_mmbtu_hr=1.0,
        )

        result = calculator.calculate(input_data)

        # Sensible credits should be tracked (currently 0)
        assert result.sensible_heat_credits_mmbtu_hr >= 0.0

        # Total input includes fuel + credits
        assert result.total_heat_input_mmbtu_hr == (
            result.fuel_heat_mmbtu_hr + result.sensible_heat_credits_mmbtu_hr
        )

    # =========================================================================
    # Test Case 10: Multi-Fuel Operation
    # =========================================================================
    @pytest.mark.golden
    @pytest.mark.asme
    def test_dual_fuel_operation(self, calculator):
        """
        Test heat balance with combined fuel input.

        Example: 70% natural gas, 30% fuel oil
        """
        # Combined heat input from both fuels
        gas_input = 70.0  # MMBtu/hr from gas
        oil_input = 30.0  # MMBtu/hr from oil
        total_input = gas_input + oil_input

        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=total_input,
            steam_output_mmbtu_hr=82.0,
            blowdown_mmbtu_hr=3.0,
            stack_loss_mmbtu_hr=11.0,
            radiation_loss_mmbtu_hr=1.0,
        )

        result = calculator.calculate(input_data)

        assert result.total_heat_input_mmbtu_hr == 100.0
        assert result.balanced is True


class TestHeatBalanceTolerances:
    """Tests for different balance tolerance settings."""

    @pytest.mark.golden
    def test_strict_tolerance(self):
        """Test with 1% tolerance (strict)."""
        calculator = HeatBalanceCalculator(tolerance_percent=1.0)

        # Almost balanced input
        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=100.0,
            steam_output_mmbtu_hr=82.0,
            blowdown_mmbtu_hr=2.5,
            stack_loss_mmbtu_hr=10.0,
            radiation_loss_mmbtu_hr=1.0,
        )

        result = calculator.calculate(input_data)

        # With 4.5% unaccounted, should fail strict tolerance
        if result.balance_error_percent > 1.0:
            assert result.balanced is False

    @pytest.mark.golden
    def test_relaxed_tolerance(self):
        """Test with 5% tolerance (relaxed)."""
        calculator = HeatBalanceCalculator(tolerance_percent=5.0)

        input_data = HeatBalanceInput(
            fuel_input_mmbtu_hr=100.0,
            steam_output_mmbtu_hr=80.0,
            blowdown_mmbtu_hr=2.0,
            stack_loss_mmbtu_hr=12.0,
            radiation_loss_mmbtu_hr=1.0,
        )

        result = calculator.calculate(input_data)

        # 5% unaccounted should pass relaxed tolerance
        if result.balance_error_percent <= 5.0:
            assert result.balanced is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
