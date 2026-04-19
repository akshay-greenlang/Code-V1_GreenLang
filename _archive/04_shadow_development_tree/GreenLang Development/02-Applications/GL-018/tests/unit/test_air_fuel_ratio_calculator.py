"""
GL-018 FLUEFLOW - Air-Fuel Ratio Calculator Unit Tests

Comprehensive unit tests for AirFuelRatioCalculator with 95%+ coverage target.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.air_fuel_ratio_calculator import (
    AirFuelRatioCalculator,
    AirFuelRatioInput,
    AirFuelRatioOutput,
    FUEL_COMPOSITIONS,
    calculate_theoretical_air_from_composition,
    calculate_lambda_from_O2,
)
from calculators.provenance import verify_provenance


@pytest.mark.unit
@pytest.mark.calculator
class TestAirFuelRatioCalculator:
    """Comprehensive test suite for AirFuelRatioCalculator."""

    def test_initialization(self):
        """Test AirFuelRatioCalculator initializes correctly."""
        calculator = AirFuelRatioCalculator()
        assert calculator.VERSION == "1.0.0"
        assert calculator.NAME == "AirFuelRatioCalculator"

    def test_natural_gas_optimal(self, air_fuel_ratio_calculator, natural_gas_air_fuel_input):
        """Test natural gas air-fuel ratio calculation."""
        result, provenance = air_fuel_ratio_calculator.calculate(natural_gas_air_fuel_input)

        assert isinstance(result, AirFuelRatioOutput)
        assert result.theoretical_air_kg_kg > 15.0
        assert result.theoretical_air_kg_kg < 20.0
        assert result.excess_air_pct == pytest.approx(20.0, rel=0.02)
        assert result.lambda_ratio == pytest.approx(1.2, rel=0.01)
        assert result.air_requirement_rating in ["Optimal", "Good"]
        assert verify_provenance(provenance) is True

    @pytest.mark.parametrize("O2_pct,expected_lambda", [
        (0.5, 1.024),
        (1.0, 1.050),
        (2.0, 1.105),
        (3.0, 1.167),
        (3.5, 1.200),
        (4.0, 1.235),
        (5.0, 1.312),
    ])
    def test_lambda_calculation(self, air_fuel_ratio_calculator, O2_pct, expected_lambda):
        """Test lambda calculation from O2 measurement."""
        inputs = AirFuelRatioInput(
            fuel_type="Natural Gas",
            O2_measured_pct=O2_pct
        )

        result, provenance = air_fuel_ratio_calculator.calculate(inputs)
        assert result.lambda_ratio == pytest.approx(expected_lambda, rel=0.01)

    @pytest.mark.parametrize("fuel_type,expected_air_range", [
        ("Natural Gas", (16.0, 18.0)),
        ("Fuel Oil", (13.5, 15.5)),
        ("Coal", (8.5, 10.5)),
        ("Diesel", (13.0, 15.0)),
        ("Propane", (15.0, 17.0)),
    ])
    def test_all_fuel_types(self, air_fuel_ratio_calculator, fuel_type, expected_air_range):
        """Test all fuel types."""
        inputs = AirFuelRatioInput(
            fuel_type=fuel_type,
            O2_measured_pct=3.5
        )

        result, provenance = air_fuel_ratio_calculator.calculate(inputs)
        assert result.theoretical_air_kg_kg >= expected_air_range[0]
        assert result.theoretical_air_kg_kg <= expected_air_range[1]

    @pytest.mark.parametrize("lambda_val,expected_rating", [
        (1.15, "Optimal"),
        (1.08, "Good"),
        (1.25, "Good"),
        (1.02, "Fair"),
        (1.40, "Fair"),
        (0.95, "Rich (Insufficient Air)"),
        (1.60, "Lean (Excessive Air)"),
    ])
    def test_air_requirement_rating(self, air_fuel_ratio_calculator, lambda_val, expected_rating):
        """Test air requirement rating classification."""
        # Calculate O2 that gives target lambda
        excess_air_pct = (lambda_val - 1.0) * 100.0
        O2_pct = (21.0 * excess_air_pct) / (100.0 + excess_air_pct)

        inputs = AirFuelRatioInput(
            fuel_type="Natural Gas",
            O2_measured_pct=O2_pct
        )

        result, provenance = air_fuel_ratio_calculator.calculate(inputs)
        assert result.air_requirement_rating == expected_rating

    def test_custom_fuel_composition(self, air_fuel_ratio_calculator):
        """Test custom fuel composition."""
        inputs = AirFuelRatioInput(
            fuel_type="Custom Fuel",
            O2_measured_pct=3.5,
            C_pct=80.0,
            H_pct=15.0,
            O_pct=3.0,
            S_pct=2.0,
            moisture_pct=5.0
        )

        result, provenance = air_fuel_ratio_calculator.calculate(inputs)
        assert result.theoretical_air_kg_kg > 0
        assert verify_provenance(provenance) is True

    def test_invalid_O2_raises_error(self, air_fuel_ratio_calculator):
        """Test invalid O2 measurement raises ValueError."""
        inputs = AirFuelRatioInput(
            fuel_type="Natural Gas",
            O2_measured_pct=-1.0  # Invalid
        )

        with pytest.raises(ValueError, match="O2 measurement out of range"):
            air_fuel_ratio_calculator.calculate(inputs)

    def test_provenance_determinism(self, air_fuel_ratio_calculator, natural_gas_air_fuel_input):
        """Test provenance determinism."""
        result1, provenance1 = air_fuel_ratio_calculator.calculate(natural_gas_air_fuel_input)
        result2, provenance2 = air_fuel_ratio_calculator.calculate(natural_gas_air_fuel_input)

        assert provenance1.provenance_hash == provenance2.provenance_hash


@pytest.mark.unit
class TestStandaloneFunctions:
    """Test standalone functions."""

    def test_calculate_theoretical_air_from_composition(self):
        """Test theoretical air calculation."""
        air = calculate_theoretical_air_from_composition(75.0, 25.0, 0.0, 0.0)
        assert air == pytest.approx(17.2, rel=0.02)

    def test_calculate_lambda_from_O2(self):
        """Test lambda calculation."""
        lambda_val = calculate_lambda_from_O2(3.5)
        assert lambda_val == pytest.approx(1.2, rel=0.01)
