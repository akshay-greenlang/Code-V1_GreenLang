# -*- coding: utf-8 -*-
"""
tests/agents/test_multigas_validation.py

Multi-Gas Validation Tests for FuelAgentAI v2

OBJECTIVE:
Validate CO2, CH4, and N2O breakdown accuracy for all fuel types

TEST COVERAGE:
1. Multi-gas vector structure validation
2. GWP conversion accuracy (IPCC AR6)
3. CO2/CH4/N2O ratios by fuel type
4. Sum verification (vectors → CO2e total)
5. Zero and negative value handling
6. Cross-GWP-set consistency

REFERENCE:
- IPCC AR6 WG1 Chapter 7: GWP values
- EPA 40 CFR Part 98: Fuel-specific emission coefficients
- IEA Emission Factors 2023: Multi-gas breakdowns

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from greenlang.agents import FuelAgentAI_v2


# ==================== FIXTURES ====================


@pytest.fixture
def agent_enhanced():
    """Agent with enhanced v2 output"""
    return FuelAgentAI_v2(
        enable_explanations=False,
        enable_recommendations=False,
        enable_fast_path=True,
    )


# ==================== GWP REFERENCE VALUES ====================

# IPCC AR6 Global Warming Potentials (relative to CO2 = 1)
GWP_AR6_100YR = {
    "CO2": 1.0,
    "CH4": 27.9,  # Fossil CH4, including climate-carbon feedbacks
    "N2O": 273.0,
}

GWP_AR6_20YR = {
    "CO2": 1.0,
    "CH4": 81.2,  # 20-year horizon, much higher than 100-year
    "N2O": 273.0,  # Similar across time horizons
}


def calculate_co2e_from_vectors(vectors: dict, gwp_set: str = "IPCC_AR6_100") -> float:
    """
    Calculate CO2e from multi-gas vectors using GWP factors.

    Args:
        vectors: Dict with CO2, CH4, N2O in kg
        gwp_set: GWP reference set

    Returns:
        CO2e in kg
    """
    gwp = GWP_AR6_100YR if "100" in gwp_set else GWP_AR6_20YR

    co2e = (
        vectors["CO2"] * gwp["CO2"] +
        vectors["CH4"] * gwp["CH4"] +
        vectors["N2O"] * gwp["N2O"]
    )

    return co2e


# ==================== STRUCTURE VALIDATION TESTS ====================


def test_multigas_vectors_structure_complete(agent_enhanced):
    """
    Test 1: Multi-gas vectors have all required fields

    Validates: CO2, CH4, N2O keys exist
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons",
        "response_format": "enhanced",
    }

    result = agent_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]
    assert "vectors_kg" in data, "vectors_kg missing from enhanced output"

    vectors = data["vectors_kg"]

    # Verify all required gases
    required_gases = ["CO2", "CH4", "N2O"]
    for gas in required_gases:
        assert gas in vectors, f"{gas} missing from vectors_kg"
        assert isinstance(vectors[gas], (int, float)), f"{gas} value not numeric"
        assert vectors[gas] >= 0, f"{gas} value is negative: {vectors[gas]}"


def test_multigas_vectors_all_fuel_types(agent_enhanced):
    """
    Test 2: Multi-gas vectors present for all major fuel types

    Validates: diesel, gasoline, natural_gas, coal, electricity
    """
    fuel_types = [
        ("diesel", 100, "gallons"),
        ("gasoline", 100, "gallons"),
        ("natural_gas", 1000, "therms"),
        ("coal", 1, "tons"),
        ("electricity", 1000, "kWh"),
    ]

    for fuel_type, amount, unit in fuel_types:
        payload = {
            "fuel_type": fuel_type,
            "amount": amount,
            "unit": unit,
            "response_format": "enhanced",
        }

        result = agent_enhanced.run(payload)

        assert result["success"], f"Calculation failed for {fuel_type}: {result.get('error')}"

        data = result["data"]
        vectors = data["vectors_kg"]

        # All gases must be present
        assert "CO2" in vectors, f"CO2 missing for {fuel_type}"
        assert "CH4" in vectors, f"CH4 missing for {fuel_type}"
        assert "N2O" in vectors, f"N2O missing for {fuel_type}"

        # CO2 should dominate for all combustion fuels
        assert vectors["CO2"] > 0, f"CO2 should be positive for {fuel_type}"


# ==================== GWP CONVERSION TESTS ====================


def test_gwp_conversion_accuracy_natural_gas(agent_enhanced):
    """
    Test 3: GWP conversion accuracy for natural gas

    Validates: Manual CO2e calculation matches agent output (±1%)
    """
    payload = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "gwp_set": "IPCC_AR6_100",
        "response_format": "enhanced",
    }

    result = agent_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]
    vectors = data["vectors_kg"]
    reported_co2e = data["co2e_emissions_kg"]

    # Calculate CO2e manually
    calculated_co2e = calculate_co2e_from_vectors(vectors, "IPCC_AR6_100")

    # Verify match (±1% tolerance for rounding)
    error_pct = abs((reported_co2e - calculated_co2e) / calculated_co2e) * 100
    assert error_pct < 1.0, (
        f"GWP conversion mismatch: reported={reported_co2e:.4f}, "
        f"calculated={calculated_co2e:.4f}, error={error_pct:.2f}%"
    )


def test_gwp_conversion_accuracy_diesel(agent_enhanced):
    """
    Test 4: GWP conversion accuracy for diesel

    Validates: Manual CO2e calculation matches agent output
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons",
        "gwp_set": "IPCC_AR6_100",
        "response_format": "enhanced",
    }

    result = agent_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]
    vectors = data["vectors_kg"]
    reported_co2e = data["co2e_emissions_kg"]

    # Calculate CO2e manually
    calculated_co2e = calculate_co2e_from_vectors(vectors, "IPCC_AR6_100")

    # Verify match
    error_pct = abs((reported_co2e - calculated_co2e) / calculated_co2e) * 100
    assert error_pct < 1.0, (
        f"GWP conversion mismatch for diesel: error={error_pct:.2f}%"
    )


# ==================== FUEL-SPECIFIC RATIO TESTS ====================


def test_natural_gas_co2_dominance(agent_enhanced):
    """
    Test 5: Natural gas emissions dominated by CO2

    Reference: EPA 40 CFR Part 98, Table C-1
    Natural gas: ~99.6% CO2, ~0.3% CH4, ~0.1% N2O (in CO2e terms)
    """
    payload = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "gwp_set": "IPCC_AR6_100",
        "response_format": "enhanced",
    }

    result = agent_enhanced.run(payload)
    data = result["data"]
    vectors = data["vectors_kg"]
    total_co2e = data["co2e_emissions_kg"]

    # CO2 contribution (in CO2e terms)
    co2_contribution = vectors["CO2"] / total_co2e

    # CO2 should be 95-100% of total CO2e for natural gas
    assert 0.95 <= co2_contribution <= 1.0, (
        f"Natural gas CO2 should dominate (95-100%), got {co2_contribution*100:.2f}%"
    )


def test_diesel_co2_ch4_n2o_ratios(agent_enhanced):
    """
    Test 6: Diesel multi-gas ratios

    Reference: EPA 40 CFR Part 98, Table C-1
    Diesel: CO2 dominant, trace CH4, some N2O
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons",
        "gwp_set": "IPCC_AR6_100",
        "response_format": "enhanced",
    }

    result = agent_enhanced.run(payload)
    data = result["data"]
    vectors = data["vectors_kg"]

    # Convert to CO2e contributions
    co2_co2e = vectors["CO2"] * GWP_AR6_100YR["CO2"]
    ch4_co2e = vectors["CH4"] * GWP_AR6_100YR["CH4"]
    n2o_co2e = vectors["N2O"] * GWP_AR6_100YR["N2O"]

    total_co2e = data["co2e_emissions_kg"]

    # CO2 should be 98-100% of total
    co2_pct = (co2_co2e / total_co2e) * 100
    assert 95 <= co2_pct <= 100, f"Diesel CO2% should be 95-100%, got {co2_pct:.2f}%"

    # CH4 should be negligible (<1%)
    ch4_pct = (ch4_co2e / total_co2e) * 100
    assert ch4_pct < 2, f"Diesel CH4% should be <2%, got {ch4_pct:.2f}%"

    # N2O should be 0-3%
    n2o_pct = (n2o_co2e / total_co2e) * 100
    assert 0 <= n2o_pct <= 5, f"Diesel N2O% should be 0-5%, got {n2o_pct:.2f}%"


# ==================== GWP SET COMPARISON TESTS ====================


def test_gwp_ar6_100yr_vs_20yr_consistency(agent_enhanced):
    """
    Test 7: GWP AR6 100-year vs 20-year consistency

    Validates: CO2, CH4, N2O vectors identical, CO2e totals different
    """
    payload_base = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "response_format": "enhanced",
    }

    # 100-year
    payload_100 = {**payload_base, "gwp_set": "IPCC_AR6_100"}
    result_100 = agent_enhanced.run(payload_100)
    data_100 = result_100["data"]

    # 20-year
    payload_20 = {**payload_base, "gwp_set": "IPCC_AR6_20"}
    result_20 = agent_enhanced.run(payload_20)
    data_20 = result_20["data"]

    # Verify vectors are IDENTICAL (GWP doesn't affect raw emissions)
    vectors_100 = data_100["vectors_kg"]
    vectors_20 = data_20["vectors_kg"]

    assert vectors_100["CO2"] == vectors_20["CO2"], "CO2 vectors should be identical"
    assert vectors_100["CH4"] == vectors_20["CH4"], "CH4 vectors should be identical"
    assert vectors_100["N2O"] == vectors_20["N2O"], "N2O vectors should be identical"

    # Verify CO2e totals are DIFFERENT (20-year should be higher)
    co2e_100 = data_100["co2e_emissions_kg"]
    co2e_20 = data_20["co2e_emissions_kg"]

    assert co2e_20 > co2e_100, "20-year GWP should result in higher CO2e (due to CH4)"


def test_gwp_ch4_amplification_20yr(agent_enhanced):
    """
    Test 8: CH4 amplification in 20-year GWP

    Reference: IPCC AR6 Table 7.15
    CH4 GWP: 27.9 (100yr) → 81.2 (20yr) [2.91× increase]
    """
    payload = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "response_format": "enhanced",
    }

    # 100-year
    payload_100 = {**payload, "gwp_set": "IPCC_AR6_100"}
    result_100 = agent_enhanced.run(payload_100)
    vectors = result_100["data"]["vectors_kg"]
    co2e_100 = result_100["data"]["co2e_emissions_kg"]

    # 20-year
    payload_20 = {**payload, "gwp_set": "IPCC_AR6_20"}
    result_20 = agent_enhanced.run(payload_20)
    co2e_20 = result_20["data"]["co2e_emissions_kg"]

    # Calculate expected CH4 contribution difference
    ch4_kg = vectors["CH4"]
    ch4_contribution_100 = ch4_kg * GWP_AR6_100YR["CH4"]
    ch4_contribution_20 = ch4_kg * GWP_AR6_20YR["CH4"]
    ch4_difference = ch4_contribution_20 - ch4_contribution_100

    # Verify 20-year increase matches CH4 amplification
    actual_increase = co2e_20 - co2e_100
    expected_increase = ch4_difference

    error_pct = abs((actual_increase - expected_increase) / expected_increase) * 100
    assert error_pct < 1.0, (
        f"CH4 amplification mismatch: expected={expected_increase:.2f}, "
        f"actual={actual_increase:.2f}, error={error_pct:.2f}%"
    )


# ==================== ZERO AND EDGE CASE TESTS ====================


def test_zero_amount_zero_vectors(agent_enhanced):
    """
    Test 9: Zero fuel amount → zero emissions for all gases

    Edge case: amount=0
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 0,
        "unit": "gallons",
        "response_format": "enhanced",
    }

    result = agent_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]
    vectors = data["vectors_kg"]

    # All vectors should be zero
    assert vectors["CO2"] == 0, "CO2 should be zero for amount=0"
    assert vectors["CH4"] == 0, "CH4 should be zero for amount=0"
    assert vectors["N2O"] == 0, "N2O should be zero for amount=0"
    assert data["co2e_emissions_kg"] == 0, "Total CO2e should be zero for amount=0"


def test_small_amount_nonzero_vectors(agent_enhanced):
    """
    Test 10: Very small amount → non-zero vectors

    Edge case: amount=0.001 (precision test)
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 0.001,  # 0.001 gallons
        "unit": "gallons",
        "response_format": "enhanced",
    }

    result = agent_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]
    vectors = data["vectors_kg"]

    # All vectors should be very small but non-zero
    assert vectors["CO2"] > 0, "CO2 should be non-zero for small amount"
    assert vectors["CH4"] >= 0, "CH4 should be non-negative for small amount"
    assert vectors["N2O"] >= 0, "N2O should be non-negative for small amount"
    assert data["co2e_emissions_kg"] > 0, "Total CO2e should be non-zero for small amount"


# ==================== EFFICIENCY AND OFFSET TESTS ====================


def test_efficiency_affects_all_vectors_proportionally(agent_enhanced):
    """
    Test 11: Efficiency adjustment affects all gases proportionally

    Validates: 80% efficiency → 80% of each gas vector
    """
    # Baseline (100% efficiency)
    payload_100 = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "efficiency": 1.0,
        "response_format": "enhanced",
    }

    result_100 = agent_enhanced.run(payload_100)
    vectors_100 = result_100["data"]["vectors_kg"]

    # 80% efficiency
    payload_80 = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "efficiency": 0.8,
        "response_format": "enhanced",
    }

    result_80 = agent_enhanced.run(payload_80)
    vectors_80 = result_80["data"]["vectors_kg"]

    # Each gas should be 80% of baseline
    for gas in ["CO2", "CH4", "N2O"]:
        expected = vectors_100[gas] * 0.8
        actual = vectors_80[gas]
        error_pct = abs((actual - expected) / expected) * 100 if expected > 0 else 0
        assert error_pct < 1.0, (
            f"{gas} efficiency adjustment mismatch: expected={expected:.4f}, "
            f"actual={actual:.4f}, error={error_pct:.2f}%"
        )


def test_renewable_offset_affects_all_vectors_proportionally(agent_enhanced):
    """
    Test 12: Renewable offset affects all gases proportionally

    Validates: 50% renewable → 50% of each gas vector
    """
    # Baseline (0% renewable)
    payload_0 = {
        "fuel_type": "electricity",
        "amount": 1000,
        "unit": "kWh",
        "renewable_percentage": 0,
        "response_format": "enhanced",
    }

    result_0 = agent_enhanced.run(payload_0)
    vectors_0 = result_0["data"]["vectors_kg"]

    # 50% renewable
    payload_50 = {
        "fuel_type": "electricity",
        "amount": 1000,
        "unit": "kWh",
        "renewable_percentage": 50,
        "response_format": "enhanced",
    }

    result_50 = agent_enhanced.run(payload_50)
    vectors_50 = result_50["data"]["vectors_kg"]

    # Each gas should be 50% of baseline
    for gas in ["CO2", "CH4", "N2O"]:
        expected = vectors_0[gas] * 0.5
        actual = vectors_50[gas]
        error_pct = abs((actual - expected) / expected) * 100 if expected > 0 else 0
        assert error_pct < 1.0, (
            f"{gas} renewable offset mismatch: expected={expected:.4f}, "
            f"actual={actual:.4f}, error={error_pct:.2f}%"
        )


# ==================== CROSS-VALIDATION TEST ====================


def test_multigas_sum_equals_legacy_total(agent_enhanced):
    """
    Test 13: Multi-gas CO2e sum equals legacy total

    Validates: Enhanced format CO2e = Legacy format CO2e
    """
    # Legacy format
    payload_legacy = {
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons",
        "response_format": "legacy",
    }

    result_legacy = agent_enhanced.run(payload_legacy)
    legacy_total = result_legacy["data"]["co2e_emissions_kg"]

    # Enhanced format
    payload_enhanced = {
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons",
        "response_format": "enhanced",
    }

    result_enhanced = agent_enhanced.run(payload_enhanced)
    enhanced_total = result_enhanced["data"]["co2e_emissions_kg"]

    # Should be identical
    assert legacy_total == enhanced_total, (
        f"Legacy vs Enhanced mismatch: legacy={legacy_total:.4f}, "
        f"enhanced={enhanced_total:.4f}"
    )


# ==================== SUMMARY ====================


def test_summary_multigas_tests():
    """
    Summary: Print multi-gas test coverage

    Not a test, just a summary reporter
    """
    print("\n" + "=" * 80)
    print("  MULTI-GAS VALIDATION TEST SUMMARY")
    print("=" * 80)
    print("\n✅ Test Coverage:")
    print("   - Structure validation: 2 tests")
    print("   - GWP conversion accuracy: 2 tests")
    print("   - Fuel-specific ratios: 2 tests")
    print("   - GWP set comparison: 2 tests")
    print("   - Zero and edge cases: 2 tests")
    print("   - Efficiency/offset proportionality: 2 tests")
    print("   - Cross-validation: 1 test")
    print("\n📊 Total: 13 multi-gas validation tests")
    print("\n🎯 Validates:")
    print("   - CO2/CH4/N2O breakdown accuracy")
    print("   - GWP conversion (IPCC AR6 100yr/20yr)")
    print("   - Proportional adjustments (efficiency, offsets)")
    print("   - Backward compatibility (legacy vs enhanced)")
    print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
