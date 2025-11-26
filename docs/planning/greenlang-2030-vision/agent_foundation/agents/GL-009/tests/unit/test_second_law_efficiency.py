"""Unit tests for Second Law (Exergy) Efficiency Calculator.

Tests exergy analysis calculations based on Second Law of Thermodynamics.

Target Coverage: 90%+
Test Count: 18+

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.second_law_efficiency import (
    SecondLawEfficiencyCalculator,
    SecondLawResult,
    ExergyStream,
    ReferenceEnvironment,
    StreamType,
    IrreversibilityType,
    calculate_second_law_efficiency
)


class TestSecondLawEfficiencyCalculator:
    """Test suite for SecondLawEfficiencyCalculator."""

    def test_initialization_default(self):
        """Test calculator initializes with default reference environment."""
        calculator = SecondLawEfficiencyCalculator()

        assert calculator.reference.temperature_k == 298.15
        assert calculator.reference.pressure_kpa == 101.325
        assert calculator.VERSION == "1.0.0"

    def test_initialization_custom_reference(self, sample_reference_environment):
        """Test calculator with custom reference environment."""
        custom_ref = ReferenceEnvironment(
            temperature_k=300.0,
            pressure_kpa=100.0
        )

        calculator = SecondLawEfficiencyCalculator(reference_environment=custom_ref)

        assert calculator.reference.temperature_k == 300.0
        assert calculator.reference.pressure_kpa == 100.0

    def test_calculate_basic_exergy_efficiency(self, sample_exergy_streams):
        """Test basic exergy efficiency calculation."""
        calculator = SecondLawEfficiencyCalculator()

        result = calculator.calculate(
            input_streams=sample_exergy_streams["inputs"],
            output_streams=sample_exergy_streams["outputs"]
        )

        assert isinstance(result, SecondLawResult)
        assert 0 <= result.exergy_efficiency_percent <= 100
        assert result.total_exergy_input_kw > 0
        assert result.total_exergy_output_kw >= 0

    def test_calculate_stream_exergy_basic(self):
        """Test exergy calculation for a single stream."""
        calculator = SecondLawEfficiencyCalculator()

        exergy = calculator.calculate_stream_exergy(
            mass_flow_kg_s=4.0,
            temperature_k=473.15,  # 200C
            pressure_kpa=1000.0,
            specific_enthalpy_kj_kg=2776.0,
            specific_entropy_kj_kg_k=6.587,
            chemical_exergy_kj_kg=0.0
        )

        assert exergy >= 0.0
        assert isinstance(exergy, float)

    def test_calculate_fuel_exergy_natural_gas(self):
        """Test fuel exergy calculation for natural gas."""
        calculator = SecondLawEfficiencyCalculator()

        fuel_exergy = calculator.calculate_fuel_exergy(
            fuel_hhv_kj_kg=50000.0,
            mass_flow_kg_s=0.5,
            fuel_type="natural_gas"
        )

        # Exergy ≈ 1.04 * HHV for natural gas
        expected = 1.04 * 50000.0 * 0.5
        assert fuel_exergy == pytest.approx(expected, rel=0.01)

    @pytest.mark.parametrize("fuel_type,phi_expected", [
        ("natural_gas", 1.04),
        ("methane", 1.04),
        ("coal", 1.06),
        ("oil", 1.065),
        ("biomass", 1.15),
        ("hydrogen", 0.985),
    ])
    def test_fuel_exergy_various_fuels(self, fuel_type, phi_expected):
        """Test fuel exergy for various fuel types."""
        calculator = SecondLawEfficiencyCalculator()

        hhv = 45000.0
        mass_flow = 1.0

        fuel_exergy = calculator.calculate_fuel_exergy(
            fuel_hhv_kj_kg=hhv,
            mass_flow_kg_s=mass_flow,
            fuel_type=fuel_type
        )

        expected = phi_expected * hhv * mass_flow
        assert fuel_exergy == pytest.approx(expected, rel=0.001)

    def test_calculate_heat_transfer_exergy_heating(self):
        """Test exergy of heat transfer for heating."""
        calculator = SecondLawEfficiencyCalculator()

        exergy = calculator.calculate_heat_transfer_exergy(
            heat_rate_kw=1000.0,
            temperature_k=473.15  # 200C, above T0
        )

        # Carnot factor = 1 - T0/T = 1 - 298.15/473.15 ≈ 0.37
        # Exergy ≈ 370 kW
        assert exergy > 0
        assert exergy < 1000.0  # Should be less than heat transfer

    def test_calculate_heat_transfer_exergy_cooling(self):
        """Test exergy of heat transfer for cooling."""
        calculator = SecondLawEfficiencyCalculator()

        exergy = calculator.calculate_heat_transfer_exergy(
            heat_rate_kw=500.0,
            temperature_k=273.15  # 0C, below T0
        )

        # Below reference temperature
        assert exergy > 0

    def test_calculate_combustion_irreversibility(self):
        """Test irreversibility calculation for combustion."""
        calculator = SecondLawEfficiencyCalculator()

        irreversibility = calculator.calculate_combustion_irreversibility(
            fuel_exergy_kw=25000.0,
            products_exergy_kw=20000.0,
            adiabatic_flame_temp_k=2200.0
        )

        assert irreversibility == 5000.0
        assert irreversibility >= 0

    def test_calculate_heat_transfer_irreversibility(self):
        """Test irreversibility from heat transfer."""
        calculator = SecondLawEfficiencyCalculator()

        irreversibility = calculator.calculate_heat_transfer_irreversibility(
            heat_rate_kw=1000.0,
            hot_temp_k=500.0,
            cold_temp_k=400.0
        )

        assert irreversibility > 0
        assert isinstance(irreversibility, float)

    def test_exergy_balance_validation(self, sample_exergy_streams):
        """Test exergy balance validation."""
        calculator = SecondLawEfficiencyCalculator()

        result = calculator.calculate(
            input_streams=sample_exergy_streams["inputs"],
            output_streams=sample_exergy_streams["outputs"]
        )

        # Exergy In = Exergy Out + Exergy Destroyed
        balance = (result.total_exergy_input_kw -
                  result.total_exergy_output_kw -
                  result.total_exergy_destruction_kw)

        assert abs(balance) < result.total_exergy_input_kw * 0.01  # 1% tolerance

    def test_irreversibility_breakdown(self):
        """Test irreversibility breakdown by type."""
        calculator = SecondLawEfficiencyCalculator()

        input_streams = [
            ExergyStream(
                stream_type=StreamType.FUEL,
                stream_name="fuel",
                mass_flow_kg_s=0.5,
                temperature_k=298.15,
                pressure_kpa=101.325,
                specific_enthalpy_kj_kg=50000.0,
                specific_entropy_kj_kg_k=0.0,
                chemical_exergy_kj_kg=51000.0
            )
        ]

        output_streams = [
            ExergyStream(
                stream_type=StreamType.STEAM,
                stream_name="steam",
                mass_flow_kg_s=4.0,
                temperature_k=473.15,
                pressure_kpa=1000.0,
                specific_enthalpy_kj_kg=2776.0,
                specific_entropy_kj_kg_k=6.587,
                is_input=False
            )
        ]

        irreversibilities = {
            "combustion": 5000.0,
            "heat_transfer": 2000.0,
            "friction": 500.0
        }

        result = calculator.calculate(
            input_streams=input_streams,
            output_streams=output_streams,
            irreversibilities=irreversibilities
        )

        assert len(result.irreversibility_breakdown) == 3
        assert result.total_exergy_destruction_kw == 7500.0

    def test_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic."""
        calculator = SecondLawEfficiencyCalculator()

        input_stream = ExergyStream(
            stream_type=StreamType.FUEL,
            stream_name="fuel",
            mass_flow_kg_s=0.5,
            temperature_k=298.15,
            pressure_kpa=101.325,
            specific_enthalpy_kj_kg=50000.0,
            specific_entropy_kj_kg_k=0.0
        )

        output_stream = ExergyStream(
            stream_type=StreamType.STEAM,
            stream_name="steam",
            mass_flow_kg_s=4.0,
            temperature_k=473.15,
            pressure_kpa=1000.0,
            specific_enthalpy_kj_kg=2776.0,
            specific_entropy_kj_kg_k=6.587,
            is_input=False
        )

        result1 = calculator.calculate([input_stream], [output_stream])
        result2 = calculator.calculate([input_stream], [output_stream])

        assert result1.provenance_hash == result2.provenance_hash

    def test_first_law_comparison(self, sample_exergy_streams):
        """Test First Law efficiency is calculated for comparison."""
        calculator = SecondLawEfficiencyCalculator()

        result = calculator.calculate(
            input_streams=sample_exergy_streams["inputs"],
            output_streams=sample_exergy_streams["outputs"]
        )

        assert result.first_law_efficiency_percent >= 0
        # Second Law efficiency should be lower than First Law
        assert result.exergy_efficiency_percent <= result.first_law_efficiency_percent

    def test_reference_environment_properties(self):
        """Test reference environment property access."""
        ref = ReferenceEnvironment(temperature_k=298.15)

        assert ref.temperature_c == 25.0
        assert ref.temperature_k == 298.15

    def test_result_serialization(self, sample_exergy_streams):
        """Test SecondLawResult can be serialized to dictionary."""
        calculator = SecondLawEfficiencyCalculator()

        result = calculator.calculate(
            input_streams=sample_exergy_streams["inputs"],
            output_streams=sample_exergy_streams["outputs"]
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "exergy_efficiency_percent" in result_dict
        assert "reference_environment" in result_dict
        assert "provenance_hash" in result_dict


class TestExergyStreamValidation:
    """Test ExergyStream validation."""

    def test_exergy_stream_valid_creation(self):
        """Test ExergyStream creation with valid values."""
        stream = ExergyStream(
            stream_type=StreamType.STEAM,
            stream_name="steam_001",
            mass_flow_kg_s=4.0,
            temperature_k=473.15,
            pressure_kpa=1000.0,
            specific_enthalpy_kj_kg=2776.0,
            specific_entropy_kj_kg_k=6.587
        )

        assert stream.mass_flow_kg_s == 4.0
        assert stream.is_input is True

    def test_exergy_stream_negative_mass_flow_raises_error(self):
        """Test ExergyStream rejects negative mass flow."""
        with pytest.raises(ValueError, match="Mass flow cannot be negative"):
            ExergyStream(
                stream_type=StreamType.STEAM,
                stream_name="steam",
                mass_flow_kg_s=-4.0,
                temperature_k=473.15,
                pressure_kpa=1000.0,
                specific_enthalpy_kj_kg=2776.0,
                specific_entropy_kj_kg_k=6.587
            )

    def test_exergy_stream_zero_temperature_raises_error(self):
        """Test ExergyStream rejects zero/negative temperature."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            ExergyStream(
                stream_type=StreamType.STEAM,
                stream_name="steam",
                mass_flow_kg_s=4.0,
                temperature_k=0.0,
                pressure_kpa=1000.0,
                specific_enthalpy_kj_kg=2776.0,
                specific_entropy_kj_kg_k=6.587
            )


class TestConvenienceFunction:
    """Test module-level convenience function."""

    def test_convenience_function(self, sample_exergy_streams):
        """Test calculate_second_law_efficiency convenience function."""
        result = calculate_second_law_efficiency(
            input_streams=sample_exergy_streams["inputs"],
            output_streams=sample_exergy_streams["outputs"],
            reference_temperature_k=298.15
        )

        assert isinstance(result, SecondLawResult)
        assert result.exergy_efficiency_percent >= 0
