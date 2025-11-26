"""Unit tests for First Law Efficiency Calculator.

Tests all methods of FirstLawEfficiencyCalculator with comprehensive coverage.
Validates business logic, error handling, edge cases, and calculation accuracy.

Target Coverage: 95%+
Test Count: 25+

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.first_law_efficiency import (
    FirstLawEfficiencyCalculator,
    FirstLawResult,
    EnergyInput,
    UsefulOutput,
    EnergyLoss,
    EnergySourceType,
    OutputType,
    LossType,
    calculate_first_law_efficiency
)


class TestFirstLawEfficiencyCalculator:
    """Test suite for FirstLawEfficiencyCalculator."""

    def test_initialization_default_params(self):
        """Test calculator initializes with default parameters."""
        calculator = FirstLawEfficiencyCalculator()

        assert calculator.balance_tolerance == 0.02  # 2%
        assert calculator.precision == 4
        assert calculator.VERSION == "1.0.0"
        assert len(calculator._calculation_steps) == 0

    def test_initialization_custom_params(self):
        """Test calculator initializes with custom parameters."""
        calculator = FirstLawEfficiencyCalculator(
            balance_tolerance_percent=5.0,
            precision=6
        )

        assert calculator.balance_tolerance == 0.05
        assert calculator.precision == 6

    def test_calculate_basic_efficiency(self):
        """Test basic efficiency calculation with valid inputs."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            energy_inputs={"natural_gas": 1000.0},
            useful_outputs={"steam": 850.0},
            losses={"flue_gas": 100.0, "radiation": 30.0, "other": 20.0}
        )

        assert isinstance(result, FirstLawResult)
        assert result.efficiency_percent == 85.0
        assert result.energy_input_kw == 1000.0
        assert result.useful_output_kw == 850.0
        assert result.total_losses_kw == 150.0

    def test_calculate_perfect_efficiency(self):
        """Test calculation with 100% efficiency (edge case)."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            energy_inputs={"electricity": 1000.0},
            useful_outputs={"process_heat": 1000.0},
            losses={}
        )

        assert result.efficiency_percent == 100.0
        assert result.total_losses_kw == 0.0

    def test_calculate_zero_output(self):
        """Test calculation with zero useful output."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            energy_inputs={"fuel": 1000.0},
            useful_outputs={"output": 0.0},
            losses={"total_loss": 1000.0}
        )

        assert result.efficiency_percent == 0.0
        assert result.useful_output_kw == 0.0

    def test_calculate_multiple_inputs_outputs(self):
        """Test calculation with multiple energy inputs and outputs."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            energy_inputs={
                "natural_gas": 800.0,
                "preheated_air": 100.0,
                "waste_heat": 100.0
            },
            useful_outputs={
                "steam": 700.0,
                "hot_water": 150.0
            },
            losses={
                "flue_gas": 80.0,
                "radiation": 40.0,
                "convection": 20.0,
                "other": 10.0
            }
        )

        assert result.energy_input_kw == 1000.0
        assert result.useful_output_kw == 850.0
        assert result.total_losses_kw == 150.0
        assert result.efficiency_percent == 85.0

    @pytest.mark.parametrize("fuel_type,fuel_input,steam_output,expected_efficiency", [
        ("natural_gas", 1000.0, 850.0, 85.0),
        ("coal", 1000.0, 800.0, 80.0),
        ("biomass", 1000.0, 750.0, 75.0),
        ("oil", 1000.0, 870.0, 87.0),
        ("waste_heat", 500.0, 475.0, 95.0),
    ])
    def test_efficiency_various_fuels(
        self, fuel_type, fuel_input, steam_output, expected_efficiency
    ):
        """Test efficiency calculation for various fuel types."""
        calculator = FirstLawEfficiencyCalculator()

        losses = fuel_input - steam_output

        result = calculator.calculate(
            energy_inputs={fuel_type: fuel_input},
            useful_outputs={"steam": steam_output},
            losses={"total": losses}
        )

        assert result.efficiency_percent == pytest.approx(expected_efficiency, rel=1e-6)

    def test_energy_balance_validation_pass(self):
        """Test energy balance validation passes with balanced inputs."""
        calculator = FirstLawEfficiencyCalculator(balance_tolerance_percent=2.0)

        result = calculator.calculate(
            energy_inputs={"fuel": 1000.0},
            useful_outputs={"steam": 850.0},
            losses={"flue_gas": 100.0, "radiation": 50.0},
            validate_balance=True
        )

        assert result.energy_balance.is_balanced is True
        assert result.energy_balance.balance_error_percent <= 2.0

    def test_energy_balance_validation_fail(self):
        """Test energy balance validation generates warning when imbalanced."""
        calculator = FirstLawEfficiencyCalculator(balance_tolerance_percent=1.0)

        # Create imbalanced case (input != output + losses)
        result = calculator.calculate(
            energy_inputs={"fuel": 1000.0},
            useful_outputs={"steam": 850.0},
            losses={"flue_gas": 100.0},  # Only 950 accounted for
            validate_balance=True
        )

        assert result.energy_balance.is_balanced is False
        assert len(result.warnings) > 0

    def test_loss_breakdown_percentages(self):
        """Test loss breakdown as percentage of input."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            energy_inputs={"fuel": 1000.0},
            useful_outputs={"steam": 850.0},
            losses={
                "flue_gas": 80.0,
                "radiation": 40.0,
                "convection": 20.0,
                "other": 10.0
            }
        )

        assert result.loss_percentage_breakdown["flue_gas"] == 8.0
        assert result.loss_percentage_breakdown["radiation"] == 4.0
        assert result.loss_percentage_breakdown["convection"] == 2.0
        assert result.loss_percentage_breakdown["other"] == 1.0

    def test_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic for same inputs."""
        calculator = FirstLawEfficiencyCalculator()

        inputs = {"fuel": 1000.0}
        outputs = {"steam": 850.0}
        losses = {"flue_gas": 100.0, "radiation": 50.0}

        result1 = calculator.calculate(inputs, outputs, losses)
        result2 = calculator.calculate(inputs, outputs, losses)

        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_hash_unique_for_different_inputs(self):
        """Test provenance hash changes when inputs change."""
        calculator = FirstLawEfficiencyCalculator()

        result1 = calculator.calculate(
            {"fuel": 1000.0}, {"steam": 850.0}, {"flue_gas": 150.0}
        )
        result2 = calculator.calculate(
            {"fuel": 1000.0}, {"steam": 800.0}, {"flue_gas": 200.0}
        )

        assert result1.provenance_hash != result2.provenance_hash

    def test_calculation_steps_audit_trail(self):
        """Test calculation steps are recorded for audit trail."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            {"fuel": 1000.0},
            {"steam": 850.0},
            {"flue_gas": 100.0, "radiation": 50.0}
        )

        assert len(result.calculation_steps) > 0

        # Verify step structure
        step = result.calculation_steps[0]
        assert hasattr(step, 'step_number')
        assert hasattr(step, 'description')
        assert hasattr(step, 'operation')
        assert hasattr(step, 'inputs')
        assert hasattr(step, 'output_value')

    def test_timestamp_generation(self):
        """Test calculation timestamp is generated correctly."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            {"fuel": 1000.0}, {"steam": 850.0}, {"flue_gas": 150.0}
        )

        # Verify timestamp format (ISO 8601 with Z)
        assert result.calculation_timestamp.endswith('Z')

        # Parse timestamp to verify it's valid
        timestamp = datetime.fromisoformat(result.calculation_timestamp.replace('Z', '+00:00'))
        assert isinstance(timestamp, datetime)

    def test_warning_efficiency_above_100(self):
        """Test warning generated when efficiency exceeds 100%."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            energy_inputs={"fuel": 1000.0},
            useful_outputs={"steam": 1100.0},  # More output than input
            losses={"flue_gas": 0.0}
        )

        assert result.efficiency_percent > 100
        assert len(result.warnings) > 0
        assert any("100%" in w for w in result.warnings)

    def test_warning_efficiency_below_20(self):
        """Test warning generated for unusually low efficiency."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            energy_inputs={"fuel": 1000.0},
            useful_outputs={"steam": 150.0},
            losses={"flue_gas": 850.0}
        )

        assert result.efficiency_percent < 20
        assert len(result.warnings) > 0
        assert any("20%" in w or "low" in w.lower() for w in result.warnings)

    def test_validate_inputs_empty_inputs_raises_error(self):
        """Test that empty energy inputs raises ValueError."""
        calculator = FirstLawEfficiencyCalculator()

        with pytest.raises(ValueError, match="At least one energy input is required"):
            calculator.calculate({}, {"steam": 850.0}, {"flue_gas": 150.0})

    def test_validate_inputs_empty_outputs_raises_error(self):
        """Test that empty useful outputs raises ValueError."""
        calculator = FirstLawEfficiencyCalculator()

        with pytest.raises(ValueError, match="At least one useful output is required"):
            calculator.calculate({"fuel": 1000.0}, {}, {"flue_gas": 150.0})

    def test_validate_inputs_negative_input_raises_error(self):
        """Test that negative energy input raises ValueError."""
        calculator = FirstLawEfficiencyCalculator()

        with pytest.raises(ValueError, match="Negative energy input not allowed"):
            calculator.calculate(
                {"fuel": -1000.0},
                {"steam": 850.0},
                {"flue_gas": 150.0}
            )

    def test_validate_inputs_negative_output_raises_error(self):
        """Test that negative useful output raises ValueError."""
        calculator = FirstLawEfficiencyCalculator()

        with pytest.raises(ValueError, match="Negative useful output not allowed"):
            calculator.calculate(
                {"fuel": 1000.0},
                {"steam": -850.0},
                {"flue_gas": 150.0}
            )

    def test_validate_inputs_negative_loss_raises_error(self):
        """Test that negative loss raises ValueError."""
        calculator = FirstLawEfficiencyCalculator()

        with pytest.raises(ValueError, match="Negative loss not allowed"):
            calculator.calculate(
                {"fuel": 1000.0},
                {"steam": 850.0},
                {"flue_gas": -150.0}
            )

    def test_validate_inputs_zero_total_input_raises_error(self):
        """Test that zero total energy input raises ValueError."""
        calculator = FirstLawEfficiencyCalculator()

        with pytest.raises(ValueError, match="Total energy input cannot be zero"):
            calculator.calculate(
                {"fuel": 0.0},
                {"steam": 850.0},
                {"flue_gas": 150.0}
            )

    def test_precision_rounding(self):
        """Test values are rounded to specified precision."""
        calculator = FirstLawEfficiencyCalculator(precision=2)

        result = calculator.calculate(
            {"fuel": 1000.123456},
            {"steam": 850.987654},
            {"flue_gas": 149.135802}
        )

        # Check all values are rounded to 2 decimal places
        assert result.efficiency_percent == 85.1
        assert result.energy_input_kw == 1000.12
        assert result.useful_output_kw == 850.99

    def test_calculate_from_objects_typed_inputs(self):
        """Test calculation using typed EnergyInput objects."""
        calculator = FirstLawEfficiencyCalculator()

        inputs = [
            EnergyInput(
                source_type=EnergySourceType.NATURAL_GAS,
                source_name="NG-001",
                energy_kw=1000.0,
                uncertainty_percent=2.0
            )
        ]

        outputs = [
            UsefulOutput(
                output_type=OutputType.STEAM,
                output_name="Steam-001",
                energy_kw=850.0,
                uncertainty_percent=2.0
            )
        ]

        losses = [
            EnergyLoss(
                loss_type=LossType.FLUE_GAS_SENSIBLE,
                loss_name="Flue Gas",
                energy_kw=100.0
            ),
            EnergyLoss(
                loss_type=LossType.RADIATION,
                loss_name="Radiation",
                energy_kw=50.0
            )
        ]

        result = calculator.calculate_from_objects(inputs, outputs, losses)

        assert result.efficiency_percent == 85.0
        assert result.energy_input_kw == 1000.0

    def test_calculate_direct_method(self):
        """Test direct method (input-output) efficiency calculation."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate_direct_method(
            fuel_energy_input_kw=1000.0,
            steam_output_kw=850.0,
            auxiliary_power_kw=20.0
        )

        # Net output = 850 - 20 = 830 kW
        # Efficiency = 830 / 1000 = 83%
        assert result.efficiency_percent == 83.0
        assert result.useful_output_kw == 830.0

    def test_calculate_indirect_method(self):
        """Test indirect method (heat loss) efficiency calculation."""
        calculator = FirstLawEfficiencyCalculator()

        losses = {
            "flue_gas": 80.0,
            "radiation": 40.0,
            "convection": 20.0,
            "other": 10.0
        }

        result = calculator.calculate_indirect_method(
            fuel_energy_input_kw=1000.0,
            losses=losses
        )

        # Total losses = 150 kW
        # Useful output = 1000 - 150 = 850 kW
        # Efficiency = 85%
        assert result.efficiency_percent == 85.0
        assert result.total_losses_kw == 150.0

    def test_result_to_dict_serialization(self):
        """Test FirstLawResult can be serialized to dictionary."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            {"fuel": 1000.0},
            {"steam": 850.0},
            {"flue_gas": 100.0, "radiation": 50.0}
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "efficiency_percent" in result_dict
        assert "energy_input_kw" in result_dict
        assert "provenance_hash" in result_dict
        assert result_dict["efficiency_percent"] == 85.0


class TestEnergyInputValidation:
    """Test EnergyInput validation."""

    def test_energy_input_valid_creation(self):
        """Test EnergyInput creation with valid values."""
        energy_input = EnergyInput(
            source_type=EnergySourceType.NATURAL_GAS,
            source_name="NG-001",
            energy_kw=1000.0,
            measurement_id="METER-001",
            uncertainty_percent=2.0
        )

        assert energy_input.energy_kw == 1000.0
        assert energy_input.uncertainty_percent == 2.0

    def test_energy_input_negative_energy_raises_error(self):
        """Test EnergyInput rejects negative energy."""
        with pytest.raises(ValueError, match="Energy input cannot be negative"):
            EnergyInput(
                source_type=EnergySourceType.NATURAL_GAS,
                source_name="NG-001",
                energy_kw=-1000.0
            )

    def test_energy_input_invalid_uncertainty_raises_error(self):
        """Test EnergyInput rejects invalid uncertainty."""
        with pytest.raises(ValueError, match="Uncertainty must be 0-100%"):
            EnergyInput(
                source_type=EnergySourceType.NATURAL_GAS,
                source_name="NG-001",
                energy_kw=1000.0,
                uncertainty_percent=150.0
            )


class TestConvenienceFunction:
    """Test module-level convenience function."""

    def test_convenience_function(self):
        """Test calculate_first_law_efficiency convenience function."""
        result = calculate_first_law_efficiency(
            energy_inputs={"fuel": 1000.0},
            useful_outputs={"steam": 850.0},
            losses={"flue_gas": 100.0, "radiation": 50.0},
            balance_tolerance_percent=2.0
        )

        assert isinstance(result, FirstLawResult)
        assert result.efficiency_percent == 85.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_values(self):
        """Test calculation with very small energy values."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            {"fuel": 0.001},
            {"steam": 0.0008},
            {"flue_gas": 0.0002}
        )

        assert result.efficiency_percent == 80.0

    def test_very_large_values(self):
        """Test calculation with very large energy values."""
        calculator = FirstLawEfficiencyCalculator()

        result = calculator.calculate(
            {"fuel": 1000000.0},
            {"steam": 850000.0},
            {"flue_gas": 150000.0}
        )

        assert result.efficiency_percent == 85.0

    def test_many_loss_categories(self):
        """Test calculation with many loss categories."""
        calculator = FirstLawEfficiencyCalculator()

        losses = {f"loss_{i}": 10.0 for i in range(15)}

        result = calculator.calculate(
            {"fuel": 1000.0},
            {"steam": 850.0},
            losses
        )

        assert result.total_losses_kw == 150.0
        assert len(result.loss_breakdown) == 15
