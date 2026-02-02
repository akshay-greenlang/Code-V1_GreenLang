"""
Float Precision Tests

Validates numerical precision in emission calculations:
1. Rounding consistency
2. Decimal precision handling
3. No floating point errors
4. Regulatory-grade precision
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP


class TestFloatPrecision:
    """Tests float precision in Carbon Emissions Agent."""

    @pytest.fixture
    def carbon_agent(self):
        """Create Carbon Emissions Agent instance."""
        from backend.agents.gl_001_carbon_emissions.agent import CarbonEmissionsAgent
        return CarbonEmissionsAgent()

    @pytest.fixture
    def carbon_input_class(self):
        """Get Carbon Emissions Input class."""
        from backend.agents.gl_001_carbon_emissions.agent import CarbonEmissionsInput
        return CarbonEmissionsInput

    def test_6_decimal_precision(self, carbon_agent, carbon_input_class):
        """Test that results maintain 6 decimal places."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        input_data = carbon_input_class(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=123.456789,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result = carbon_agent.run(input_data)

        # Check result has appropriate precision
        # emissions = 123.456789 * 1.93 = 238.271602770...
        # Rounded to 6 decimals
        assert result.emissions_kgco2e == pytest.approx(238.271603, rel=1e-6)

    def test_no_floating_point_errors(self, carbon_agent, carbon_input_class):
        """Test classic floating point error cases."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        # Test case: 0.1 + 0.2 != 0.3 in floating point
        # We use 100 * EF and verify precision
        input_data = carbon_input_class(
            fuel_type=FuelType.DIESEL,
            quantity=100,
            unit="L",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result = carbon_agent.run(input_data)

        # 100 * 2.68 = 268.0 exactly
        assert result.emissions_kgco2e == 268.0

    def test_small_quantity_precision(self, carbon_agent, carbon_input_class):
        """Test precision with very small quantities."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        input_data = carbon_input_class(
            fuel_type=FuelType.DIESEL,
            quantity=0.001,
            unit="L",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result = carbon_agent.run(input_data)

        # 0.001 * 2.68 = 0.00268
        assert result.emissions_kgco2e == pytest.approx(0.00268, abs=1e-8)

    def test_large_quantity_precision(self, carbon_agent, carbon_input_class):
        """Test precision with very large quantities."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        input_data = carbon_input_class(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000000000,  # 1 billion
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result = carbon_agent.run(input_data)

        # 1,000,000,000 * 1.93 = 1,930,000,000
        assert result.emissions_kgco2e == pytest.approx(1930000000.0, rel=1e-10)

    @pytest.mark.parametrize("quantity,expected", [
        (0.1, 0.268),
        (0.2, 0.536),
        (0.3, 0.804),
        (0.7, 1.876),
        (1.1, 2.948),
        (1.9, 5.092),
        (3.3, 8.844),
    ])
    def test_decimal_quantities(self, carbon_agent, carbon_input_class, quantity, expected):
        """Test various decimal quantities for precision."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        input_data = carbon_input_class(
            fuel_type=FuelType.DIESEL,
            quantity=quantity,
            unit="L",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result = carbon_agent.run(input_data)

        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)


class TestCBAMPrecision:
    """Tests float precision in CBAM Compliance Agent."""

    @pytest.fixture
    def cbam_agent(self):
        """Create CBAM Compliance Agent instance."""
        from backend.agents.gl_002_cbam_compliance.agent import CBAMComplianceAgent
        return CBAMComplianceAgent()

    @pytest.fixture
    def cbam_input_class(self):
        """Get CBAM Input class."""
        from backend.agents.gl_002_cbam_compliance.agent import CBAMInput
        return CBAMInput

    def test_cbam_liability_precision(self, cbam_agent, cbam_input_class):
        """Test CBAM liability calculation precision."""
        input_data = cbam_input_class(
            cn_code="72081000",
            quantity_tonnes=123.456,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        # Direct: 123.456 * 2.10 = 259.2576
        # Indirect: 123.456 * 0.45 = 55.5552
        # Total: 314.8128
        # Liability: 314.8128 * 85 = 26759.088

        assert result.total_embedded_emissions_tco2e == pytest.approx(314.8128, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(26759.088, rel=1e-4)

    def test_cbam_rounding(self, cbam_agent, cbam_input_class):
        """Test CBAM results are properly rounded."""
        input_data = cbam_input_class(
            cn_code="72081000",
            quantity_tonnes=100,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        # Check liability is rounded to 2 decimal places
        assert result.cbam_liability_eur == round(result.cbam_liability_eur, 2)


class TestScope3Precision:
    """Tests float precision in Scope 3 Agent."""

    @pytest.fixture
    def scope3_agent(self):
        """Create Scope 3 Emissions Agent instance."""
        from backend.agents.gl_006_scope3_emissions.agent import Scope3EmissionsAgent
        return Scope3EmissionsAgent()

    @pytest.fixture
    def scope3_input_class(self):
        """Get Scope 3 Input class."""
        from backend.agents.gl_006_scope3_emissions.agent import Scope3Input
        return Scope3Input

    def test_spend_based_precision(self, scope3_agent, scope3_input_class):
        """Test spend-based calculation precision."""
        from backend.agents.gl_006_scope3_emissions.agent import (
            Scope3Category,
            SpendData,
        )

        input_data = scope3_input_class(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[
                SpendData(category="steel", spend_usd=123456.78),
            ],
        )

        result = scope3_agent.run(input_data)

        # 123456.78 * 0.85 = 104938.263
        expected = 123456.78 * 0.85

        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    def test_transport_precision(self, scope3_agent, scope3_input_class):
        """Test transport calculation precision."""
        from backend.agents.gl_006_scope3_emissions.agent import (
            Scope3Category,
            TransportData,
        )

        input_data = scope3_input_class(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[
                TransportData(mode="road_truck", distance_km=123.456, weight_tonnes=78.9),
            ],
        )

        result = scope3_agent.run(input_data)

        # tonne-km = 123.456 * 78.9 = 9740.6784
        # emissions = 9740.6784 * 0.089 = 866.9203776
        expected = 123.456 * 78.9 * 0.089

        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)


class TestRegulatoryPrecision:
    """Tests for regulatory-grade precision requirements."""

    def test_regulatory_decimal_places(self):
        """Test that results meet regulatory decimal place requirements."""
        from backend.agents.gl_001_carbon_emissions.agent import (
            CarbonEmissionsAgent,
            CarbonEmissionsInput,
            FuelType,
            Scope,
        )

        agent = CarbonEmissionsAgent()

        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result = agent.run(input_data)

        # Emissions should have at most 6 decimal places
        emissions_str = str(result.emissions_kgco2e)
        if "." in emissions_str:
            decimal_places = len(emissions_str.split(".")[1])
            assert decimal_places <= 6, \
                f"Emissions has {decimal_places} decimal places, max is 6"

    def test_no_scientific_notation_in_normal_range(self):
        """Test that normal range values don't use scientific notation."""
        from backend.agents.gl_001_carbon_emissions.agent import (
            CarbonEmissionsAgent,
            CarbonEmissionsInput,
            FuelType,
            Scope,
        )

        agent = CarbonEmissionsAgent()

        # Test range from 1 to 1 million
        for qty in [1, 100, 10000, 1000000]:
            input_data = CarbonEmissionsInput(
                fuel_type=FuelType.NATURAL_GAS,
                quantity=qty,
                unit="m3",
                region="US",
                scope=Scope.SCOPE_1,
            )

            result = agent.run(input_data)

            # Check string representation doesn't use scientific notation
            emissions_str = str(result.emissions_kgco2e)
            assert "e" not in emissions_str.lower(), \
                f"Scientific notation used for {qty}: {emissions_str}"

    def test_consistency_with_decimal_type(self):
        """Test that float results match Decimal calculations."""
        from backend.agents.gl_001_carbon_emissions.agent import (
            CarbonEmissionsAgent,
            CarbonEmissionsInput,
            FuelType,
            Scope,
        )

        agent = CarbonEmissionsAgent()

        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result = agent.run(input_data)

        # Calculate using Decimal for comparison
        qty_decimal = Decimal("1000")
        ef_decimal = Decimal("1.93")
        expected_decimal = qty_decimal * ef_decimal

        # Float result should match Decimal to within precision
        assert abs(Decimal(str(result.emissions_kgco2e)) - expected_decimal) < Decimal("0.0001")
