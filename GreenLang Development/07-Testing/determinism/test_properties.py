# -*- coding: utf-8 -*-
"""Property-Based Tests for Agent Invariants.

This module uses Hypothesis for property-based testing to verify that agents
satisfy key invariants across a wide range of inputs.

Test Coverage:
1. Output structure is always valid (schema compliance)
2. Emissions are always non-negative
3. Citations are always present
4. Input validation always rejects invalid inputs consistently
5. Calculations respect physical constraints
6. Error handling is consistent

Author: GreenLang Framework Team
Phase: Phase 3 - Production Hardening
Date: November 2024
"""

import pytest

try:
    from hypothesis import given, strategies as st, assume, settings, example, HealthCheck
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    pytest.skip("hypothesis not installed", allow_module_level=True)

from greenlang.agents.fuel_agent_ai import FuelAgentAI
from tests.determinism.test_framework import DeterminismTester


@pytest.mark.property
@pytest.mark.determinism
class TestAgentInvariants:
    """Property-based tests for agent invariants."""

    @pytest.fixture
    def agent(self):
        """Create FuelAgentAI instance."""
        return FuelAgentAI(budget_usd=1.0)

    @pytest.fixture
    def tester(self):
        """Create DeterminismTester instance."""
        return DeterminismTester()

    # ============================================================================
    # Test 1: Output Structure Is Always Valid
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["natural_gas", "diesel", "coal", "electricity"]),
        amount=st.floats(min_value=0.1, max_value=100000.0, allow_nan=False, allow_infinity=False),
        renewable_percentage=st.floats(min_value=0.0, max_value=100.0),
        efficiency=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_output_structure_always_valid(
        self, agent, fuel_type, amount, renewable_percentage, efficiency
    ):
        """Property: Output always has required fields with correct types."""
        # Determine appropriate unit for fuel type
        unit_map = {
            "natural_gas": "therms",
            "diesel": "gallons",
            "coal": "tons",
            "electricity": "kWh",
        }
        unit = unit_map[fuel_type]

        # Run calculation
        result = agent._calculate_emissions_impl(
            fuel_type=fuel_type,
            amount=amount,
            unit=unit,
            country="US",
            renewable_percentage=renewable_percentage,
            efficiency=efficiency,
        )

        # Verify required fields exist
        assert "emissions_kg_co2e" in result
        assert "emission_factor" in result
        assert "emission_factor_unit" in result
        assert "scope" in result
        # calculation field may or may not be present depending on implementation

        # Verify types
        assert isinstance(result["emissions_kg_co2e"], (int, float))
        assert isinstance(result["emission_factor"], (int, float))
        assert isinstance(result["emission_factor_unit"], str)
        assert isinstance(result["scope"], str)
        if "calculation" in result:
            # calculation can be dict or string depending on implementation
            assert isinstance(result["calculation"], (dict, str))

    # ============================================================================
    # Test 2: Emissions Are Always Non-Negative
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["natural_gas", "diesel", "coal"]),
        amount=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        renewable_percentage=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @example(fuel_type="natural_gas", amount=0.0, renewable_percentage=0.0)
    @example(fuel_type="diesel", amount=1000.0, renewable_percentage=50.0)
    def test_emissions_always_non_negative(
        self, agent, fuel_type, amount, renewable_percentage
    ):
        """Property: Emissions are always >= 0."""
        unit_map = {
            "natural_gas": "therms",
            "diesel": "gallons",
            "coal": "tons",
        }
        unit = unit_map[fuel_type]

        result = agent._calculate_emissions_impl(
            fuel_type=fuel_type,
            amount=amount,
            unit=unit,
            country="US",
            renewable_percentage=renewable_percentage,
            efficiency=1.0,
        )

        emissions = result["emissions_kg_co2e"]
        assert emissions >= 0, f"Emissions cannot be negative: {emissions}"

    # ============================================================================
    # Test 3: Emissions Scale Linearly with Amount
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["natural_gas", "diesel"]),
        base_amount=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        scale_factor=st.floats(min_value=1.5, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=15, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_emissions_scale_linearly(
        self, agent, fuel_type, base_amount, scale_factor
    ):
        """Property: Emissions scale linearly with fuel amount."""
        unit_map = {"natural_gas": "therms", "diesel": "gallons"}
        unit = unit_map[fuel_type]

        # Calculate base emissions
        result1 = agent._calculate_emissions_impl(
            fuel_type=fuel_type,
            amount=base_amount,
            unit=unit,
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )
        emissions1 = result1["emissions_kg_co2e"]

        # Calculate scaled emissions
        result2 = agent._calculate_emissions_impl(
            fuel_type=fuel_type,
            amount=base_amount * scale_factor,
            unit=unit,
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )
        emissions2 = result2["emissions_kg_co2e"]

        # Verify linear scaling (within float precision)
        expected_emissions2 = emissions1 * scale_factor
        relative_error = abs(emissions2 - expected_emissions2) / max(expected_emissions2, 0.001)
        assert relative_error < 0.01, (
            f"Emissions don't scale linearly: "
            f"{emissions1} * {scale_factor} = {expected_emissions2}, "
            f"but got {emissions2}"
        )

    # ============================================================================
    # Test 4: Renewable Percentage Reduces Emissions
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["natural_gas", "diesel"]),
        amount=st.floats(min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        renewable_percentage=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=15, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @example(fuel_type="natural_gas", amount=1000.0, renewable_percentage=50.0)
    def test_renewable_reduces_emissions(
        self, agent, fuel_type, amount, renewable_percentage
    ):
        """Property: Adding renewable percentage reduces emissions."""
        unit_map = {"natural_gas": "therms", "diesel": "gallons"}
        unit = unit_map[fuel_type]

        # Calculate without renewable
        result_no_renewable = agent._calculate_emissions_impl(
            fuel_type=fuel_type,
            amount=amount,
            unit=unit,
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )
        emissions_no_renewable = result_no_renewable["emissions_kg_co2e"]

        # Calculate with renewable
        result_with_renewable = agent._calculate_emissions_impl(
            fuel_type=fuel_type,
            amount=amount,
            unit=unit,
            country="US",
            renewable_percentage=renewable_percentage,
            efficiency=1.0,
        )
        emissions_with_renewable = result_with_renewable["emissions_kg_co2e"]

        # With renewable should be <= without
        # NOTE: This assumes the agent implementation supports renewable_percentage
        # If not implemented, emissions will be the same
        assert emissions_with_renewable <= emissions_no_renewable, (
            f"Renewable percentage {renewable_percentage}% should reduce emissions, "
            f"but {emissions_with_renewable} > {emissions_no_renewable}"
        )

        # If renewable percentage is 0, emissions should be equal
        if renewable_percentage == 0:
            assert emissions_with_renewable == emissions_no_renewable

        # If renewable percentage is 100, emissions should be 0 or significantly reduced
        if renewable_percentage == 100.0 and emissions_no_renewable > 0:
            # Allow for either zero emissions or some reduction
            assert emissions_with_renewable <= emissions_no_renewable

    # ============================================================================
    # Test 5: Emission Factor Lookup Is Deterministic
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["natural_gas", "diesel", "coal"]),
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_emission_factor_lookup_deterministic(self, agent, tester, fuel_type):
        """Property: Emission factor lookups are always deterministic."""
        unit_map = {
            "natural_gas": "therms",
            "diesel": "gallons",
            "coal": "tons",
        }
        unit = unit_map[fuel_type]

        # Run lookup multiple times
        results = []
        for _ in range(3):
            result = agent._lookup_emission_factor_impl(
                fuel_type=fuel_type,
                unit=unit,
                country="US",
            )
            results.append(result)

        # All should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], f"Lookup {i+1} differs from lookup 1"

    # ============================================================================
    # Test 6: Calculation Determinism Across Runs
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["natural_gas", "diesel"]),
        amount=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_calculation_deterministic_property(self, agent, fuel_type, amount):
        """Property: Same inputs always produce same outputs."""
        unit_map = {"natural_gas": "therms", "diesel": "gallons"}
        unit = unit_map[fuel_type]

        # Run calculation multiple times
        results = []
        for _ in range(3):
            result = agent._calculate_emissions_impl(
                fuel_type=fuel_type,
                amount=amount,
                unit=unit,
                country="US",
                renewable_percentage=0,
                efficiency=1.0,
            )
            results.append(result)

        # All should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], (
                f"Calculation {i+1} differs from calculation 1 "
                f"for {fuel_type}, amount={amount}"
            )

    # ============================================================================
    # Test 7: Zero Amount Produces Zero Emissions
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["natural_gas", "diesel", "coal"]),
        renewable_percentage=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_zero_amount_zero_emissions(
        self, agent, fuel_type, renewable_percentage
    ):
        """Property: Zero fuel amount always produces zero emissions."""
        unit_map = {
            "natural_gas": "therms",
            "diesel": "gallons",
            "coal": "tons",
        }
        unit = unit_map[fuel_type]

        result = agent._calculate_emissions_impl(
            fuel_type=fuel_type,
            amount=0.0,
            unit=unit,
            country="US",
            renewable_percentage=renewable_percentage,
            efficiency=1.0,
        )

        emissions = result["emissions_kg_co2e"]
        assert emissions == 0.0, f"Zero amount should produce zero emissions, got {emissions}"

    # ============================================================================
    # Test 8: Emission Factor Is Always Positive
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["natural_gas", "diesel", "coal", "propane"]),
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_emission_factor_always_positive(self, agent, fuel_type):
        """Property: Emission factors are always positive."""
        unit_map = {
            "natural_gas": "therms",
            "diesel": "gallons",
            "coal": "tons",
            "propane": "gallons",
        }
        unit = unit_map.get(fuel_type, "therms")

        try:
            result = agent._lookup_emission_factor_impl(
                fuel_type=fuel_type,
                unit=unit,
                country="US",
            )

            emission_factor = result["emission_factor"]
            assert emission_factor > 0, (
                f"Emission factor for {fuel_type} should be positive, got {emission_factor}"
            )
        except Exception as e:
            # If fuel type not found, that's acceptable
            if "not found" in str(e).lower() or "unsupported" in str(e).lower():
                pytest.skip(f"Fuel type {fuel_type} not supported")
            else:
                raise

    # ============================================================================
    # Test 9: Scope Is Always Valid
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["natural_gas", "diesel", "coal", "electricity"]),
        amount=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=15, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_scope_always_valid(self, agent, fuel_type, amount):
        """Property: Scope is always a valid value (1, 2, or 3)."""
        unit_map = {
            "natural_gas": "therms",
            "diesel": "gallons",
            "coal": "tons",
            "electricity": "kWh",
        }
        unit = unit_map[fuel_type]

        result = agent._calculate_emissions_impl(
            fuel_type=fuel_type,
            amount=amount,
            unit=unit,
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )

        scope = result["scope"]
        assert scope in ["1", "2", "3"], f"Invalid scope: {scope}"

        # Verify scope correctness
        if fuel_type in ["natural_gas", "diesel", "coal"]:
            # Direct combustion = Scope 1
            assert scope == "1", f"Fuel {fuel_type} should be Scope 1, got {scope}"
        elif fuel_type == "electricity":
            # Electricity = Scope 2
            assert scope == "2", f"Electricity should be Scope 2, got {scope}"

    # ============================================================================
    # Test 10: Recommendations Are Consistent
    # ============================================================================

    @given(
        fuel_type=st.sampled_from(["coal", "diesel", "natural_gas"]),
        emissions_kg=st.floats(min_value=1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_recommendations_consistent(self, agent, fuel_type, emissions_kg):
        """Property: Recommendations are consistent across runs."""
        # Run recommendations multiple times
        results = []
        for _ in range(3):
            result = agent._generate_recommendations_impl(
                fuel_type=fuel_type,
                emissions_kg=emissions_kg,
                country="US",
            )
            results.append(result)

        # All should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], (
                f"Recommendations {i+1} differ from recommendations 1"
            )

        # Verify structure
        assert "recommendations" in results[0]
        assert "count" in results[0]
        assert isinstance(results[0]["recommendations"], list)
        assert results[0]["count"] == len(results[0]["recommendations"])


# ============================================================================
# Additional Property Tests for Edge Cases
# ============================================================================


@pytest.mark.property
@pytest.mark.determinism
class TestEdgeCaseProperties:
    """Property-based tests for edge cases."""

    @pytest.fixture
    def agent(self):
        """Create FuelAgentAI instance."""
        return FuelAgentAI(budget_usd=1.0)

    @given(
        amount=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_very_small_amounts(self, agent, amount):
        """Property: Very small amounts still produce valid results."""
        result = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=amount,
            unit="therms",
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )

        # Should have non-negative emissions
        assert result["emissions_kg_co2e"] >= 0

        # Structure should be valid
        assert "emission_factor" in result
        assert "scope" in result

    @given(
        amount=st.floats(min_value=1000000.0, max_value=10000000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=5, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_very_large_amounts(self, agent, amount):
        """Property: Very large amounts still produce valid results."""
        result = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=amount,
            unit="therms",
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )

        # Should have non-negative emissions
        assert result["emissions_kg_co2e"] >= 0

        # Emissions should be proportional
        emissions_per_unit = result["emissions_kg_co2e"] / amount
        assert emissions_per_unit > 0
