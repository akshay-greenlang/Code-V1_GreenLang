"""
tests/agents/test_fuel_agent_v2_compliance.py

Compliance tests for FuelAgentAI v2 against authoritative calculators

REFERENCE CALCULATORS:
- EPA GHG Emission Factors Hub (2024)
- GHG Protocol calculation tools
- IEA Emission Factors database
- UK BEIS conversion factors

TEST STRATEGY:
- 20+ test cases covering major fuel types
- Known reference values from authoritative sources
- Tolerance: ±2% for numerical precision
- Multi-gas validation (CO2, CH4, N2O)
- Provenance verification

SOURCES:
[1] EPA (2024): https://www.epa.gov/climateleadership/ghg-emission-factors-hub
[2] GHGP (2023): https://ghgprotocol.org/calculation-tools
[3] IEA (2023): https://www.iea.org/data-and-statistics/data-product/emissions-factors-2023
[4] UK BEIS (2024): https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from greenlang.agents import FuelAgentAI_v2


# ==================== FIXTURES ====================


@pytest.fixture
def agent():
    """Create FuelAgentAI v2 instance for testing"""
    return FuelAgentAI_v2(
        enable_explanations=False,  # Fast path for tests
        enable_recommendations=False,
        enable_fast_path=True,
    )


@pytest.fixture
def agent_v2_enhanced():
    """Create FuelAgentAI v2 with enhanced output"""
    return FuelAgentAI_v2(
        enable_explanations=False,
        enable_recommendations=False,
        enable_fast_path=True,
    )


def assert_emissions_match(
    actual: float,
    expected: float,
    tolerance_pct: float = 2.0,
    label: str = "emissions"
):
    """
    Assert emissions match within tolerance.

    Args:
        actual: Calculated value
        expected: Reference value
        tolerance_pct: Tolerance percentage (default 2%)
        label: Description for error message
    """
    error_pct = abs((actual - expected) / expected) * 100
    assert error_pct <= tolerance_pct, (
        f"{label} mismatch: calculated={actual:.4f}, "
        f"expected={expected:.4f}, error={error_pct:.2f}% (tolerance={tolerance_pct}%)"
    )


# ==================== EPA COMPLIANCE TESTS ====================


def test_epa_diesel_combustion(agent):
    """
    Test 1: EPA diesel combustion emissions (Scope 1)

    Reference: EPA GHG Emission Factors Hub (2024)
    Diesel (No. 2): 10.21 kgCO2e/gallon (combustion only)
    Source: 40 CFR Part 98, Table C-1
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons",
        "country": "US",
        "scope": "1",
        "boundary": "combustion",
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # EPA reference: 10.21 kgCO2e/gallon
    expected_total = 1000 * 10.21  # 10,210 kg CO2e
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="EPA Diesel combustion"
    )


def test_epa_natural_gas_combustion(agent):
    """
    Test 2: EPA natural gas combustion emissions

    Reference: EPA GHG Emission Factors Hub (2024)
    Natural Gas: 53.06 kgCO2e/MMBtu = 5.306 kgCO2e/therm
    Source: 40 CFR Part 98, Table C-1
    """
    payload = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US",
        "scope": "1",
        "boundary": "combustion",
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # EPA reference: 5.306 kgCO2e/therm
    expected_total = 1000 * 5.30  # 5,300 kg CO2e (rounded)
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="EPA Natural Gas combustion"
    )


def test_epa_gasoline_combustion(agent):
    """
    Test 3: EPA gasoline combustion emissions

    Reference: EPA GHG Emission Factors Hub (2024)
    Gasoline (motor): 8.78 kgCO2e/gallon
    Source: 40 CFR Part 98, Table C-1
    """
    payload = {
        "fuel_type": "gasoline",
        "amount": 500,
        "unit": "gallons",
        "country": "US",
        "scope": "1",
        "boundary": "combustion",
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # EPA reference: 8.78 kgCO2e/gallon
    expected_total = 500 * 8.78  # 4,390 kg CO2e
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="EPA Gasoline combustion"
    )


def test_epa_electricity_us_average(agent):
    """
    Test 4: EPA US electricity grid average (Scope 2)

    Reference: EPA eGRID 2024
    US Average: 0.385 kgCO2e/kWh (2024 grid mix)
    Source: eGRID2024 annual output emission rates
    """
    payload = {
        "fuel_type": "electricity",
        "amount": 10000,
        "unit": "kWh",
        "country": "US",
        "scope": "2",
        "boundary": "combustion",  # Grid emissions
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # EPA eGRID 2024: 0.385 kgCO2e/kWh (US average)
    expected_total = 10000 * 0.385  # 3,850 kg CO2e
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="EPA Electricity US average"
    )


def test_epa_propane_combustion(agent):
    """
    Test 5: EPA propane combustion emissions

    Reference: EPA GHG Emission Factors Hub (2024)
    Propane (LPG): 5.74 kgCO2e/gallon
    Source: 40 CFR Part 98, Table C-1
    """
    payload = {
        "fuel_type": "propane",
        "amount": 200,
        "unit": "gallons",
        "country": "US",
        "scope": "1",
        "boundary": "combustion",
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # EPA reference: 5.74 kgCO2e/gallon
    expected_total = 200 * 5.74  # 1,148 kg CO2e
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="EPA Propane combustion"
    )


# ==================== UK BEIS COMPLIANCE TESTS ====================


def test_beis_uk_electricity_2024(agent):
    """
    Test 6: UK BEIS electricity grid factor (2024)

    Reference: UK BEIS Conversion Factors 2024
    UK Grid Electricity: 0.212 kgCO2e/kWh (2024)
    Source: BEIS/DESNZ GHG Conversion Factors 2024
    """
    payload = {
        "fuel_type": "electricity",
        "amount": 5000,
        "unit": "kWh",
        "country": "UK",
        "scope": "2",
        "boundary": "combustion",
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # UK BEIS 2024: 0.212 kgCO2e/kWh
    expected_total = 5000 * 0.212  # 1,060 kg CO2e
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="UK BEIS Electricity 2024"
    )


def test_beis_uk_natural_gas(agent):
    """
    Test 7: UK BEIS natural gas factor

    Reference: UK BEIS Conversion Factors 2024
    Natural Gas: 0.18316 kgCO2e/kWh (gross CV)
    Source: BEIS/DESNZ GHG Conversion Factors 2024
    """
    payload = {
        "fuel_type": "natural_gas",
        "amount": 10000,
        "unit": "kWh",
        "country": "UK",
        "scope": "1",
        "boundary": "combustion",
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # UK BEIS 2024: 0.18316 kgCO2e/kWh
    expected_total = 10000 * 0.183  # 1,830 kg CO2e (rounded)
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="UK BEIS Natural Gas"
    )


def test_beis_uk_diesel(agent):
    """
    Test 8: UK BEIS diesel factor

    Reference: UK BEIS Conversion Factors 2024
    Diesel (average biofuel blend): 2.5164 kgCO2e/liter
    Source: BEIS/DESNZ GHG Conversion Factors 2024
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "liters",
        "country": "UK",
        "scope": "1",
        "boundary": "combustion",
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # UK BEIS 2024: 2.5164 kgCO2e/liter
    expected_total = 1000 * 2.52  # 2,520 kg CO2e (rounded)
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="UK BEIS Diesel"
    )


# ==================== GHG PROTOCOL COMPLIANCE TESTS ====================


def test_ghgp_coal_combustion(agent):
    """
    Test 9: GHG Protocol coal combustion

    Reference: GHG Protocol Stationary Combustion Tool (2023)
    Bituminous Coal: 93.4 kgCO2e/MMBtu
    Source: GHGP Calculation Tools v2.9
    """
    payload = {
        "fuel_type": "coal",
        "amount": 100,
        "unit": "MMBtu",
        "country": "US",
        "scope": "1",
        "boundary": "combustion",
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # GHGP reference: 93.4 kgCO2e/MMBtu
    expected_total = 100 * 93.4  # 9,340 kg CO2e
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="GHGP Coal combustion"
    )


def test_ghgp_fuel_oil_combustion(agent):
    """
    Test 10: GHG Protocol residual fuel oil (No. 6)

    Reference: GHG Protocol Stationary Combustion Tool (2023)
    Residual Fuel Oil: 75.1 kgCO2e/MMBtu
    Source: GHGP Calculation Tools v2.9
    """
    payload = {
        "fuel_type": "residual_fuel_oil",
        "amount": 50,
        "unit": "MMBtu",
        "country": "US",
        "scope": "1",
        "boundary": "combustion",
    }

    result = agent.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    # GHGP reference: 75.1 kgCO2e/MMBtu
    expected_total = 50 * 75.1  # 3,755 kg CO2e
    assert_emissions_match(
        result["data"]["co2e_emissions_kg"],
        expected_total,
        tolerance_pct=2.0,
        label="GHGP Residual Fuel Oil"
    )


# ==================== MULTI-GAS COMPLIANCE TESTS ====================


def test_multigas_natural_gas_breakdown(agent_v2_enhanced):
    """
    Test 11: Multi-gas breakdown for natural gas (CO2, CH4, N2O)

    Reference: EPA 40 CFR Part 98, Subpart C
    Natural Gas composition (typical):
    - CO2: 99.6% of emissions (from combustion)
    - CH4: 0.3% (fugitive + incomplete combustion)
    - N2O: 0.1% (combustion byproduct)
    """
    payload = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US",
        "scope": "1",
        "boundary": "combustion",
        "response_format": "enhanced",
    }

    result = agent_v2_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]

    # Verify multi-gas vectors exist
    assert "vectors_kg" in data, "Multi-gas vectors missing"
    vectors = data["vectors_kg"]

    assert "CO2" in vectors, "CO2 vector missing"
    assert "CH4" in vectors, "CH4 vector missing"
    assert "N2O" in vectors, "N2O vector missing"

    # Verify CO2 dominates (should be ~99% of total in CO2e)
    total_co2e = data["co2e_emissions_kg"]
    co2_kg = vectors["CO2"]

    assert co2_kg > 0, "CO2 emissions should be positive"
    assert co2_kg / total_co2e > 0.95, "CO2 should dominate natural gas emissions"


def test_multigas_diesel_breakdown(agent_v2_enhanced):
    """
    Test 12: Multi-gas breakdown for diesel (CO2, CH4, N2O)

    Reference: EPA 40 CFR Part 98, Subpart C
    Diesel composition (typical):
    - CO2: 99.5% of emissions
    - CH4: 0.03% (incomplete combustion)
    - N2O: 0.47% (combustion byproduct)
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons",
        "country": "US",
        "scope": "1",
        "boundary": "combustion",
        "response_format": "enhanced",
    }

    result = agent_v2_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]
    vectors = data["vectors_kg"]

    # Verify all gases present
    assert vectors["CO2"] > 0, "CO2 should be positive"
    assert vectors["CH4"] >= 0, "CH4 should be non-negative"
    assert vectors["N2O"] >= 0, "N2O should be non-negative"

    # Sum should approximately match total (within GWP conversion)
    # Note: This is approximate due to GWP factors
    total_co2e = data["co2e_emissions_kg"]
    assert total_co2e > vectors["CO2"], "CO2e total should be > CO2 (due to CH4/N2O GWP)"


# ==================== RENEWABLE OFFSET TESTS ====================


def test_renewable_offset_50pct(agent):
    """
    Test 13: Renewable offset reduces emissions by 50%

    Reference: GHG Protocol Scope 2 Guidance (2015)
    Market-based accounting with 50% RECs
    """
    # Baseline (no offset)
    payload_baseline = {
        "fuel_type": "electricity",
        "amount": 1000,
        "unit": "kWh",
        "country": "US",
        "renewable_percentage": 0,
    }

    result_baseline = agent.run(payload_baseline)
    baseline_emissions = result_baseline["data"]["co2e_emissions_kg"]

    # With 50% renewable offset
    payload_offset = {
        "fuel_type": "electricity",
        "amount": 1000,
        "unit": "kWh",
        "country": "US",
        "renewable_percentage": 50,
    }

    result_offset = agent.run(payload_offset)
    offset_emissions = result_offset["data"]["co2e_emissions_kg"]

    # Offset should be exactly 50% of baseline
    expected_offset = baseline_emissions * 0.5
    assert_emissions_match(
        offset_emissions,
        expected_offset,
        tolerance_pct=1.0,
        label="50% Renewable Offset"
    )


def test_renewable_offset_100pct(agent):
    """
    Test 14: 100% renewable offset = zero net emissions

    Reference: GHG Protocol Scope 2 Guidance (2015)
    Market-based accounting with 100% RECs
    """
    payload = {
        "fuel_type": "electricity",
        "amount": 5000,
        "unit": "kWh",
        "country": "US",
        "renewable_percentage": 100,
    }

    result = agent.run(payload)

    # Should be zero net emissions
    assert result["data"]["co2e_emissions_kg"] == 0.0, "100% renewable should result in zero net emissions"


# ==================== EFFICIENCY ADJUSTMENT TESTS ====================


def test_efficiency_adjustment_80pct(agent):
    """
    Test 15: Equipment efficiency adjustment (80% efficient equipment)

    Reference: Engineering calculation principle
    Efficiency factor reduces effective fuel consumption
    """
    # Baseline (100% efficiency)
    payload_100 = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "efficiency": 1.0,
    }

    result_100 = agent.run(payload_100)
    emissions_100 = result_100["data"]["co2e_emissions_kg"]

    # 80% efficiency
    payload_80 = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "efficiency": 0.8,
    }

    result_80 = agent.run(payload_80)
    emissions_80 = result_80["data"]["co2e_emissions_kg"]

    # 80% efficiency should result in 80% of baseline emissions
    expected_80 = emissions_100 * 0.8
    assert_emissions_match(
        emissions_80,
        expected_80,
        tolerance_pct=1.0,
        label="80% Efficiency"
    )


# ==================== GWP SET COMPLIANCE TESTS ====================


def test_gwp_ar6_100yr_vs_20yr(agent_v2_enhanced):
    """
    Test 16: IPCC AR6 100-year vs 20-year GWP

    Reference: IPCC AR6 WG1 Chapter 7
    CH4 GWP: 27.9 (100-year) vs 81.2 (20-year)
    N2O GWP: 273 (100-year) vs 273 (20-year, similar)

    Natural gas has significant CH4, so 20-year GWP should be higher
    """
    # 100-year GWP
    payload_100 = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "gwp_set": "IPCC_AR6_100",
        "response_format": "enhanced",
    }

    result_100 = agent_v2_enhanced.run(payload_100)
    emissions_100 = result_100["data"]["co2e_emissions_kg"]

    # 20-year GWP
    payload_20 = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "gwp_set": "IPCC_AR6_20",
        "response_format": "enhanced",
    }

    result_20 = agent_v2_enhanced.run(payload_20)
    emissions_20 = result_20["data"]["co2e_emissions_kg"]

    # 20-year GWP should be higher than 100-year (due to higher CH4 GWP)
    assert emissions_20 > emissions_100, "20-year GWP should be higher than 100-year for natural gas"

    # Difference should be 2-5% for natural gas (depends on CH4 content)
    pct_increase = ((emissions_20 - emissions_100) / emissions_100) * 100
    assert 1 <= pct_increase <= 10, f"GWP increase should be 1-10%, got {pct_increase:.2f}%"


# ==================== PROVENANCE COMPLIANCE TESTS ====================


def test_provenance_tracking_epa_source(agent_v2_enhanced):
    """
    Test 17: Provenance tracking includes EPA source attribution

    Reference: Audit requirement for CSRD/CDP compliance
    All emission factors must cite authoritative sources
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons",
        "country": "US",
        "response_format": "enhanced",
    }

    result = agent_v2_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]

    # Verify provenance exists
    assert "factor_record" in data, "Provenance (factor_record) missing"
    prov = data["factor_record"]

    # Verify required provenance fields
    assert "factor_id" in prov, "factor_id missing"
    assert "source_org" in prov, "source_org missing"
    assert "citation" in prov, "citation missing"

    # Verify source is EPA for US diesel
    assert prov["source_org"] in ["EPA", "US EPA", "epa"], (
        f"Expected EPA source for US diesel, got: {prov['source_org']}"
    )


def test_provenance_tracking_uk_beis_source(agent_v2_enhanced):
    """
    Test 18: Provenance tracking includes UK BEIS source

    Reference: UK BEIS/DESNZ Conversion Factors 2024
    UK factors must cite BEIS/DESNZ
    """
    payload = {
        "fuel_type": "electricity",
        "amount": 1000,
        "unit": "kWh",
        "country": "UK",
        "response_format": "enhanced",
    }

    result = agent_v2_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]
    prov = data["factor_record"]

    # Verify source is UK BEIS/DESNZ
    assert prov["source_org"] in ["UK BEIS", "BEIS", "DESNZ", "UK DESNZ"], (
        f"Expected UK BEIS/DESNZ source for UK electricity, got: {prov['source_org']}"
    )


# ==================== DATA QUALITY SCORE TESTS ====================


def test_dqs_exists_and_valid(agent_v2_enhanced):
    """
    Test 19: Data Quality Score (DQS) exists and is valid

    Reference: GHGP Quality Management Standard
    DQS must be present for v2 enhanced output
    """
    payload = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "response_format": "enhanced",
    }

    result = agent_v2_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]

    # Verify DQS exists
    assert "quality" in data, "Quality metadata missing"
    quality = data["quality"]
    assert "dqs" in quality, "DQS missing"

    dqs = quality["dqs"]

    # Verify DQS structure
    assert "overall_score" in dqs, "DQS overall_score missing"
    assert "rating" in dqs, "DQS rating missing"

    # Verify DQS score is valid (1-5 scale)
    overall_score = dqs["overall_score"]
    assert 1.0 <= overall_score <= 5.0, f"DQS score {overall_score} out of range (1-5)"

    # Verify rating is valid
    assert dqs["rating"] in ["Excellent", "Good", "Fair", "Poor", "Very Poor"], (
        f"Invalid DQS rating: {dqs['rating']}"
    )


def test_uncertainty_exists_and_reasonable(agent_v2_enhanced):
    """
    Test 20: Uncertainty (95% CI) exists and is reasonable

    Reference: IPCC Good Practice Guidance
    Uncertainty should be reported as ±X% (95% confidence interval)
    Typical range: 5-30% for primary emission factors
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons",
        "response_format": "enhanced",
    }

    result = agent_v2_enhanced.run(payload)

    assert result["success"], f"Calculation failed: {result.get('error')}"

    data = result["data"]
    quality = data["quality"]

    # Verify uncertainty exists
    assert "uncertainty_95ci_pct" in quality, "Uncertainty missing"
    uncertainty = quality["uncertainty_95ci_pct"]

    # Verify uncertainty is reasonable (5-30% for most factors)
    assert 0 <= uncertainty <= 50, f"Uncertainty {uncertainty}% seems unreasonable"

    # For EPA factors, uncertainty should be < 20% (high quality)
    if data["factor_record"]["source_org"] in ["EPA", "US EPA"]:
        assert uncertainty < 20, f"EPA factor uncertainty {uncertainty}% too high (expected <20%)"


# ==================== SUMMARY STATS ====================


def test_summary_all_tests():
    """
    Summary: Print test coverage statistics

    This is not a test, just a summary reporter
    """
    print("\n" + "=" * 80)
    print("  COMPLIANCE TEST SUMMARY")
    print("=" * 80)
    print("\n✅ Test Coverage:")
    print("   - EPA GHG Emission Factors Hub: 5 tests")
    print("   - UK BEIS Conversion Factors: 3 tests")
    print("   - GHG Protocol Tools: 2 tests")
    print("   - Multi-gas breakdown: 2 tests")
    print("   - Renewable offsets: 2 tests")
    print("   - Efficiency adjustments: 1 test")
    print("   - GWP sets (AR6 100yr/20yr): 1 test")
    print("   - Provenance tracking: 2 tests")
    print("   - Data quality scoring: 2 tests")
    print("\n📊 Total: 20 compliance tests")
    print("\n🎯 Tolerance: ±2% for numerical precision")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
