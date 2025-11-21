# -*- coding: utf-8 -*-
"""
Unit Tests for Scope 2 Calculator

Tests indirect emissions from purchased electricity, steam, heating, and cooling.
"""

import pytest
from decimal import Decimal


class TestScope2Calculator:
    """Test Scope 2 emissions calculator"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test calculator"""
        try:
            from greenlang.calculation.scope2_calculator import Scope2Calculator
            self.calculator = Scope2Calculator()
        except ImportError:
            pytest.skip("Scope2Calculator not available")

    def test_location_based_electricity(self):
        """Test location-based electricity emissions"""
        request = {
            "electricity_kwh": 1000,
            "region": "US-CA",
            "calculation_method": "location_based"
        }

        result = self.calculator.calculate(request)

        assert result.scope == "scope2"
        assert result.emissions_kg_co2e > 0
        assert result.calculation_method == "location_based"

    def test_market_based_electricity(self):
        """Test market-based electricity emissions"""
        request = {
            "electricity_kwh": 1000,
            "region": "US-CA",
            "calculation_method": "market_based",
            "green_energy_pct": 30
        }

        result = self.calculator.calculate(request)

        assert result.calculation_method == "market_based"

    @pytest.mark.parametrize("region,expected_factor_range", [
        ("US-CA", (0.2, 0.3)),  # California - low carbon
        ("US-TX", (0.4, 0.6)),  # Texas - higher carbon
        ("US-WV", (0.7, 0.9)),  # West Virginia - coal heavy
    ])
    def test_regional_grid_factors(self, region, expected_factor_range):
        """Test regional grid emission factors"""
        request = {
            "electricity_kwh": 1000,
            "region": region,
            "calculation_method": "location_based"
        }

        result = self.calculator.calculate(request)

        # Verify emissions are in expected range
        emissions_per_kwh = result.emissions_kg_co2e / 1000
        assert expected_factor_range[0] <= emissions_per_kwh <= expected_factor_range[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
