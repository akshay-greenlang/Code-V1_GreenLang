# -*- coding: utf-8 -*-
"""
Core Calculator Tests

Tests for EmissionCalculator - the heart of the calculation engine.
"""

import pytest
from decimal import Decimal
from datetime import date
from greenlang.agents.calculation.emissions.core_calculator import (
    EmissionCalculator,
    CalculationRequest,
    CalculationResult,
    CalculationStatus,
    EmissionFactorDatabase,
)


class TestEmissionCalculator:
    """Test EmissionCalculator core functionality"""

    def setup_method(self):
        """Setup test calculator"""
        self.calculator = EmissionCalculator()

    def test_calculate_diesel_gallons(self):
        """Test basic diesel calculation in gallons"""
        request = CalculationRequest(
            factor_id='diesel',
            activity_amount=100,
            activity_unit='gallons',
        )

        result = self.calculator.calculate(request)

        assert result.status == CalculationStatus.SUCCESS
        assert result.emissions_kg_co2e > 0
        assert result.provenance_hash is not None
        assert len(result.calculation_steps) > 0

        # Diesel is ~10.21 kg CO2e per gallon
        expected = Decimal('1021.0')  # 100 gallons × 10.21
        assert abs(result.emissions_kg_co2e - expected) < 10  # Within 10 kg

    def test_calculate_natural_gas_therms(self):
        """Test natural gas calculation in therms"""
        request = CalculationRequest(
            factor_id='natural_gas',
            activity_amount=500,
            activity_unit='therms',
        )

        result = self.calculator.calculate(request)

        assert result.status == CalculationStatus.SUCCESS
        assert result.emissions_kg_co2e > 0

        # Natural gas is ~5.3 kg CO2e per therm
        expected = Decimal('2650.0')  # 500 therms × 5.3
        assert abs(result.emissions_kg_co2e - expected) < 50

    def test_calculate_zero_activity(self):
        """Test calculation with zero activity amount"""
        request = CalculationRequest(
            factor_id='diesel',
            activity_amount=0,
            activity_unit='gallons',
        )

        result = self.calculator.calculate(request)

        assert result.status == CalculationStatus.WARNING
        assert result.emissions_kg_co2e == Decimal('0')
        assert 'zero' in result.warnings[0].lower()

    def test_calculate_negative_activity(self):
        """Test calculation with negative activity (should fail)"""
        with pytest.raises(ValueError, match="cannot be negative"):
            request = CalculationRequest(
                factor_id='diesel',
                activity_amount=-100,
                activity_unit='gallons',
            )
            self.calculator.calculate(request)

    def test_calculate_unknown_factor(self):
        """Test calculation with unknown emission factor"""
        request = CalculationRequest(
            factor_id='nonexistent_fuel_xyz',
            activity_amount=100,
            activity_unit='gallons',
        )

        result = self.calculator.calculate(request)

        assert result.status == CalculationStatus.FAILED
        assert len(result.errors) > 0
        assert 'not found' in result.errors[0].lower()

    def test_determinism(self):
        """Test that same input produces identical output (bit-perfect)"""
        request = CalculationRequest(
            factor_id='diesel',
            activity_amount=100,
            activity_unit='gallons',
        )

        # Run calculation twice
        result1 = self.calculator.calculate(request)
        result2 = self.calculator.calculate(request)

        # Results must be identical
        assert result1.emissions_kg_co2e == result2.emissions_kg_co2e
        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_integrity(self):
        """Test provenance hash verification"""
        request = CalculationRequest(
            factor_id='diesel',
            activity_amount=100,
            activity_unit='gallons',
        )

        result = self.calculator.calculate(request)

        # Verify provenance
        assert result.verify_provenance() is True

        # Tamper with result
        result.emissions_kg_co2e = Decimal('999999')

        # Verification should fail
        assert result.verify_provenance() is False

    def test_unit_conversion(self):
        """Test automatic unit conversion"""
        # Calculate diesel in liters
        request_liters = CalculationRequest(
            factor_id='diesel',
            activity_amount=100,
            activity_unit='liters',
        )

        result_liters = self.calculator.calculate(request_liters)

        # Calculate diesel in gallons (1 gallon = 3.78541 liters)
        request_gallons = CalculationRequest(
            factor_id='diesel',
            activity_amount=100 / 3.78541,  # ~26.42 gallons
            activity_unit='gallons',
        )

        result_gallons = self.calculator.calculate(request_gallons)

        # Results should be approximately equal
        assert abs(result_liters.emissions_kg_co2e - result_gallons.emissions_kg_co2e) < 1

    def test_calculation_performance(self):
        """Test calculation speed (<100ms target)"""
        request = CalculationRequest(
            factor_id='diesel',
            activity_amount=100,
            activity_unit='gallons',
        )

        result = self.calculator.calculate(request)

        # Should complete in < 100ms
        assert result.calculation_duration_ms < 100

    def test_factor_resolution(self):
        """Test emission factor resolution details"""
        request = CalculationRequest(
            factor_id='diesel',
            activity_amount=100,
            activity_unit='gallons',
        )

        result = self.calculator.calculate(request)

        # Factor resolution should exist
        assert result.factor_resolution is not None
        assert result.factor_resolution.factor_id == 'diesel'
        assert result.factor_resolution.source != ''
        assert result.factor_resolution.uri != ''
        assert result.factor_resolution.factor_value > 0


class TestEmissionFactorDatabase:
    """Test EmissionFactorDatabase"""

    def setup_method(self):
        """Setup test database"""
        self.db = EmissionFactorDatabase()

    def test_get_factor_diesel(self):
        """Test retrieving diesel emission factor"""
        factor = self.db.get_factor('diesel')

        assert factor is not None
        assert factor.factor_id == 'diesel'
        assert factor.factor_value > 0
        assert 'EPA' in factor.source or 'epa' in factor.source.lower()

    def test_get_factor_natural_gas(self):
        """Test retrieving natural gas emission factor"""
        factor = self.db.get_factor('natural_gas')

        assert factor is not None
        assert factor.factor_id == 'natural_gas'
        assert factor.factor_value > 0

    def test_get_factor_not_found(self):
        """Test handling of unknown factor"""
        with pytest.raises(ValueError, match="not found"):
            self.db.get_factor('nonexistent_fuel_xyz')

    def test_factor_has_uri(self):
        """Test that factors have source URIs"""
        factor = self.db.get_factor('diesel')

        assert factor.uri is not None
        assert len(factor.uri) > 0
        assert factor.uri.startswith('http')


class TestCalculationRequest:
    """Test CalculationRequest validation"""

    def test_create_valid_request(self):
        """Test creating valid request"""
        request = CalculationRequest(
            factor_id='diesel',
            activity_amount=100,
            activity_unit='gallons',
        )

        assert request.factor_id == 'diesel'
        assert request.activity_amount == Decimal('100')
        assert request.activity_unit == 'gallons'
        assert request.calculation_date == date.today()
        assert request.request_id is not None

    def test_decimal_conversion(self):
        """Test automatic conversion to Decimal"""
        request = CalculationRequest(
            factor_id='diesel',
            activity_amount=100.5,  # Float
            activity_unit='gallons',
        )

        assert isinstance(request.activity_amount, Decimal)
        assert request.activity_amount == Decimal('100.5')

    def test_invalid_activity_amount(self):
        """Test invalid activity amount"""
        with pytest.raises(ValueError):
            CalculationRequest(
                factor_id='diesel',
                activity_amount='invalid',  # Not a number
                activity_unit='gallons',
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
