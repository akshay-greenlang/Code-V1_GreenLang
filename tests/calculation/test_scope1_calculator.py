# -*- coding: utf-8 -*-
"""
Unit Tests for Scope 1 Calculator

Tests direct emission calculations (stationary combustion, mobile combustion,
fugitive emissions, process emissions).
"""

import pytest
from decimal import Decimal
from datetime import date
from greenlang.calculation.scope1_calculator import (
    Scope1Calculator,
    Scope1Result,
)
from greenlang.calculation.core_calculator import CalculationRequest


class TestScope1Calculator:
    """Test Scope 1 emissions calculator"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test calculator"""
        self.calculator = Scope1Calculator()

    def test_stationary_combustion_diesel(self):
        """Test stationary combustion - diesel boiler"""
        request = CalculationRequest(
            factor_id="diesel-stationary-us",
            activity_amount=1000,
            activity_unit="liters",
            region="US",
        )

        result = self.calculator.calculate(request)

        assert isinstance(result, Scope1Result)
        assert result.emissions_kg_co2e > 0
        assert result.scope == "scope1"
        assert result.category == "stationary_combustion"

    def test_mobile_combustion_gasoline(self):
        """Test mobile combustion - gasoline vehicle"""
        request = CalculationRequest(
            factor_id="gasoline-mobile-us",
            activity_amount=500,
            activity_unit="liters",
            region="US",
        )

        result = self.calculator.calculate(request)

        assert result.category == "mobile_combustion"
        assert result.emissions_kg_co2e > 0

    @pytest.mark.parametrize("fuel,quantity,unit,expected_min", [
        ("diesel", 100, "liters", 250),  # ~2.68 kg/L
        ("natural_gas", 1000, "cubic_meters", 1800),  # ~2.0 kg/m3
        ("coal", 1000, "kg", 3000),  # ~3.2 kg/kg
    ])
    def test_fuel_emission_calculations(self, fuel, quantity, unit, expected_min):
        """Test various fuel emission calculations"""
        request = CalculationRequest(
            factor_id=f"{fuel}-stationary-us",
            activity_amount=quantity,
            activity_unit=unit,
            region="US",
        )

        result = self.calculator.calculate(request)

        # Verify emissions are in expected range
        assert result.emissions_kg_co2e >= expected_min

    def test_deterministic_calculation(self):
        """Test calculation is deterministic"""
        request = CalculationRequest(
            factor_id="diesel-stationary-us",
            activity_amount=100,
            activity_unit="liters",
        )

        result1 = self.calculator.calculate(request)
        result2 = self.calculator.calculate(request)

        assert result1.emissions_kg_co2e == result2.emissions_kg_co2e
        assert result1.provenance_hash == result2.provenance_hash

    def test_zero_activity_warning(self):
        """Test warning when activity amount is zero"""
        request = CalculationRequest(
            factor_id="diesel-stationary-us",
            activity_amount=0,
            activity_unit="liters",
        )

        result = self.calculator.calculate(request)

        assert result.emissions_kg_co2e == Decimal("0")
        assert len(result.warnings) > 0

    def test_provenance_tracking(self):
        """Test provenance hash is generated and valid"""
        request = CalculationRequest(
            factor_id="diesel-stationary-us",
            activity_amount=100,
            activity_unit="liters",
        )

        result = self.calculator.calculate(request)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 length
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
