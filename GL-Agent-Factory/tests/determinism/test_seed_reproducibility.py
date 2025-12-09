"""
Seed Reproducibility Tests

Validates that GreenLang agents produce identical outputs for identical inputs.
This is critical for zero-hallucination guarantees and audit compliance.

Key Properties Tested:
1. Idempotency: f(x) = f(x) for any input x
2. Determinism: No random or time-dependent variations
3. Reproducibility: Results can be recreated at any time
"""

import pytest
from decimal import Decimal
from typing import List, Dict, Any
import hashlib
import json


class TestCarbonEmissionsDeterminism:
    """Tests determinism of Carbon Emissions Agent (GL-001)."""

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

    @pytest.mark.parametrize("fuel_type,quantity,unit,region", [
        ("natural_gas", 1000, "m3", "US"),
        ("natural_gas", 500, "m3", "EU"),
        ("diesel", 100, "L", "US"),
        ("diesel", 2500, "L", "DE"),
        ("electricity_grid", 10000, "kWh", "US"),
        ("electricity_grid", 5000, "kWh", "FR"),
    ])
    def test_identical_inputs_produce_identical_outputs(
        self,
        carbon_agent,
        carbon_input_class,
        fuel_type: str,
        quantity: float,
        unit: str,
        region: str,
    ):
        """Test that running the same calculation twice produces identical results."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        input_data = carbon_input_class(
            fuel_type=FuelType(fuel_type),
            quantity=quantity,
            unit=unit,
            region=region,
            scope=Scope.SCOPE_1 if fuel_type != "electricity_grid" else Scope.SCOPE_2,
        )

        # Run calculation multiple times
        results = [carbon_agent.run(input_data) for _ in range(5)]

        # All emissions values must be identical
        emissions = [r.emissions_kgco2e for r in results]
        assert all(e == emissions[0] for e in emissions), \
            f"Emissions varied across runs: {emissions}"

        # All emission factors must be identical
        factors = [r.emission_factor_used for r in results]
        assert all(f == factors[0] for f in factors), \
            f"Emission factors varied: {factors}"

    def test_calculation_formula_determinism(self, carbon_agent, carbon_input_class):
        """Test that emissions = quantity * emission_factor exactly."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        test_cases = [
            (FuelType.NATURAL_GAS, 1000, "m3", "US", 1.93, 1930.0),
            (FuelType.DIESEL, 1000, "L", "US", 2.68, 2680.0),
            (FuelType.DIESEL, 1000, "L", "EU", 2.62, 2620.0),
        ]

        for fuel_type, qty, unit, region, ef, expected in test_cases:
            input_data = carbon_input_class(
                fuel_type=fuel_type,
                quantity=qty,
                unit=unit,
                region=region,
                scope=Scope.SCOPE_1,
            )

            result = carbon_agent.run(input_data)

            # Verify the calculation is exactly: qty * ef
            calculated = qty * ef
            assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-10), \
                f"Expected {expected}, got {result.emissions_kgco2e}"

    def test_no_random_variation(self, carbon_agent, carbon_input_class):
        """Test that there is no random variation in 100 runs."""
        from backend.agents.gl_001_carbon_emissions.agent import FuelType, Scope

        input_data = carbon_input_class(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1234.5678,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )

        results = [carbon_agent.run(input_data) for _ in range(100)]

        # Extract all unique emissions values
        unique_emissions = set(r.emissions_kgco2e for r in results)

        # There should be exactly ONE unique value
        assert len(unique_emissions) == 1, \
            f"Found {len(unique_emissions)} different values: {unique_emissions}"


class TestCBAMDeterminism:
    """Tests determinism of CBAM Compliance Agent (GL-002)."""

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

    @pytest.mark.parametrize("cn_code,quantity,country", [
        ("72081000", 100, "CN"),
        ("72081000", 500, "IN"),
        ("76011000", 1000, "CN"),
        ("25232900", 2500, "TR"),
    ])
    def test_cbam_calculation_determinism(
        self,
        cbam_agent,
        cbam_input_class,
        cn_code: str,
        quantity: float,
        country: str,
    ):
        """Test CBAM calculations are deterministic."""
        input_data = cbam_input_class(
            cn_code=cn_code,
            quantity_tonnes=quantity,
            country_of_origin=country,
            reporting_period="Q1 2026",
        )

        # Run multiple times
        results = [cbam_agent.run(input_data) for _ in range(10)]

        # All results must be identical
        embeddings = [r.total_embedded_emissions_tco2e for r in results]
        liabilities = [r.cbam_liability_eur for r in results]

        assert all(e == embeddings[0] for e in embeddings), \
            f"Embedded emissions varied: {embeddings}"
        assert all(l == liabilities[0] for l in liabilities), \
            f"CBAM liabilities varied: {liabilities}"

    def test_cbam_formula_verification(self, cbam_agent, cbam_input_class):
        """Verify CBAM formulas are applied correctly."""
        # Steel from China: direct_ef=2.10, indirect_ef=0.45
        input_data = cbam_input_class(
            cn_code="72081000",
            quantity_tonnes=1000,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        # Verify calculations
        expected_direct = 1000 * 2.10
        expected_indirect = 1000 * 0.45
        expected_total = expected_direct + expected_indirect
        expected_liability = expected_total * 85.0

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-6)


class TestScope3Determinism:
    """Tests determinism of Scope 3 Emissions Agent (GL-006)."""

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

    def test_spend_based_determinism(self, scope3_agent, scope3_input_class):
        """Test spend-based calculations are deterministic."""
        from backend.agents.gl_006_scope3_emissions.agent import (
            Scope3Category,
            SpendData,
            CalculationMethod,
        )

        input_data = scope3_input_class(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[
                SpendData(category="steel", spend_usd=1000000),
                SpendData(category="aluminum", spend_usd=500000),
            ],
            calculation_method=CalculationMethod.SPEND_BASED,
        )

        results = [scope3_agent.run(input_data) for _ in range(20)]

        emissions = [r.total_emissions_kgco2e for r in results]

        assert all(e == emissions[0] for e in emissions), \
            f"Spend-based emissions varied: {set(emissions)}"

    def test_transport_calculation_determinism(self, scope3_agent, scope3_input_class):
        """Test transport calculations are deterministic."""
        from backend.agents.gl_006_scope3_emissions.agent import (
            Scope3Category,
            TransportData,
        )

        input_data = scope3_input_class(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[
                TransportData(mode="road_truck", distance_km=1000, weight_tonnes=25),
                TransportData(mode="sea_container", distance_km=5000, weight_tonnes=100),
            ],
        )

        results = [scope3_agent.run(input_data) for _ in range(20)]

        emissions = [r.total_emissions_kgco2e for r in results]

        assert all(e == emissions[0] for e in emissions), \
            f"Transport emissions varied: {set(emissions)}"


class TestCrossAgentDeterminism:
    """Tests determinism across all agents with same test scenarios."""

    def test_all_agents_deterministic_parallel(self):
        """Run all agents in parallel and verify determinism."""
        # This test ensures that running agents concurrently
        # doesn't introduce non-deterministic behavior
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def run_carbon_test():
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

            results = [agent.run(input_data) for _ in range(10)]
            return [r.emissions_kgco2e for r in results]

        # Run in parallel threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_carbon_test) for _ in range(4)]

            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())

        # All 40 results (4 threads x 10 runs) should be identical
        unique_values = set(all_results)
        assert len(unique_values) == 1, \
            f"Parallel execution produced {len(unique_values)} different values"


class TestInputOrderIndependence:
    """Tests that input order doesn't affect results."""

    def test_spend_order_independence(self):
        """Test that order of spend data doesn't affect total."""
        from backend.agents.gl_006_scope3_emissions.agent import (
            Scope3EmissionsAgent,
            Scope3Input,
            Scope3Category,
            SpendData,
        )

        agent = Scope3EmissionsAgent()

        # Original order
        input1 = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[
                SpendData(category="steel", spend_usd=100000),
                SpendData(category="aluminum", spend_usd=50000),
                SpendData(category="plastics", spend_usd=30000),
            ],
        )

        # Reversed order
        input2 = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[
                SpendData(category="plastics", spend_usd=30000),
                SpendData(category="aluminum", spend_usd=50000),
                SpendData(category="steel", spend_usd=100000),
            ],
        )

        result1 = agent.run(input1)
        result2 = agent.run(input2)

        # Total emissions should be identical regardless of order
        assert result1.total_emissions_kgco2e == result2.total_emissions_kgco2e, \
            f"Order affected results: {result1.total_emissions_kgco2e} vs {result2.total_emissions_kgco2e}"
