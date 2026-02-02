# -*- coding: utf-8 -*-
"""
Unit Tests for Gas Decomposition

Tests CO2e decomposition into CO2, CH4, N2O components using GWP factors.
"""

import pytest
from decimal import Decimal


class TestMultiGasCalculator:
    """Test MultiGasCalculator"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup multi-gas calculator"""
        try:
            from greenlang.agents.calculation.emissions.gas_decomposition import MultiGasCalculator
            self.calculator = MultiGasCalculator()
        except ImportError:
            pytest.skip("MultiGasCalculator not available")

    def test_decompose_diesel_combustion(self):
        """Test decomposing diesel combustion emissions"""
        # Diesel combustion produces ~98% CO2, ~1% CH4, ~1% N2O
        total_co2e = Decimal("1000")  # kg CO2e

        breakdown = self.calculator.decompose(
            total_co2e=total_co2e,
            fuel_type="diesel",
            combustion_type="stationary"
        )

        assert breakdown.co2 > 0
        assert breakdown.ch4 > 0
        assert breakdown.n2o > 0

        # Total should equal input (within rounding)
        total = breakdown.co2 + (breakdown.ch4 * breakdown.gwp_ch4) + \
                (breakdown.n2o * breakdown.gwp_n2o)

        assert abs(total - total_co2e) < Decimal("1")

    def test_gwp_ar6_factors(self):
        """Test GWP factors match AR6 values"""
        try:
            from greenlang.agents.calculation.emissions.gas_decomposition import GWP_AR6_100YR
        except ImportError:
            pytest.skip("GWP factors not available")

        # AR6 100-year GWP values
        assert GWP_AR6_100YR["CO2"] == Decimal("1")
        assert GWP_AR6_100YR["CH4"] == Decimal("29.8")  # Fossil CH4
        assert GWP_AR6_100YR["N2O"] == Decimal("273")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
