#!/usr/bin/env python
"""
Standalone test runner for GL-002 CBAM Compliance Agent tests.

This script runs tests without going through the parent package's __init__.py,
which avoids import issues with other agents.

Usage:
    python run_tests.py [pytest args]
"""

import sys
import os

# Add the current directory to path to enable direct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Mock the parent package to prevent it from loading other agents
class MockParentModule:
    pass

# Create mock parent packages
sys.modules['backend'] = MockParentModule()
sys.modules['backend.agents'] = MockParentModule()
sys.modules['backend.agents.gl_002_cbam_compliance'] = MockParentModule()

# Now import the agent module directly
from agent import (
    CBAMComplianceAgent,
    CBAMInput,
    CBAMOutput,
    CBAMProductCategory,
    CalculationMethod,
    EmissionType,
    CBAMDefaultFactor,
)

# Inject into mock module
sys.modules['backend.agents.gl_002_cbam_compliance'].CBAMComplianceAgent = CBAMComplianceAgent
sys.modules['backend.agents.gl_002_cbam_compliance'].CBAMInput = CBAMInput
sys.modules['backend.agents.gl_002_cbam_compliance'].CBAMOutput = CBAMOutput
sys.modules['backend.agents.gl_002_cbam_compliance'].CBAMProductCategory = CBAMProductCategory
sys.modules['backend.agents.gl_002_cbam_compliance'].CalculationMethod = CalculationMethod
sys.modules['backend.agents.gl_002_cbam_compliance'].EmissionType = EmissionType
sys.modules['backend.agents.gl_002_cbam_compliance'].CBAMDefaultFactor = CBAMDefaultFactor

def run_sample_tests():
    """Run sample tests to verify functionality."""
    import pytest
    from datetime import datetime

    print("=" * 70)
    print("GL-002 CBAM Compliance Agent - Test Suite")
    print("=" * 70)
    print()

    # Test Constants
    EF_STEEL_CN_DIRECT = 2.10
    EF_STEEL_CN_INDIRECT = 0.45
    EF_STEEL_CN_TOTAL = EF_STEEL_CN_DIRECT + EF_STEEL_CN_INDIRECT
    EU_ETS_PRICE = 85.0

    # Test 1: Agent Initialization
    print("[TEST 1] Agent Initialization...")
    agent = CBAMComplianceAgent()
    assert agent.AGENT_ID == "regulatory/cbam_compliance_v1"
    assert agent.VERSION == "1.0.0"
    print("  PASSED: Agent initialized correctly")

    # Test 2: Steel from China Calculation
    print("[TEST 2] Steel from China (1000t) Calculation...")
    input_data = CBAMInput(
        cn_code="7208.10.00",
        quantity_tonnes=1000.0,
        country_of_origin="CN",
        reporting_period="Q1 2026",
    )
    result = agent.run(input_data)

    expected_direct = 1000.0 * EF_STEEL_CN_DIRECT  # 2100
    expected_indirect = 1000.0 * EF_STEEL_CN_INDIRECT  # 450
    expected_total = expected_direct + expected_indirect  # 2550
    expected_liability = expected_total * EU_ETS_PRICE  # 216750

    assert abs(result.direct_emissions_tco2e - expected_direct) < 0.001
    assert abs(result.indirect_emissions_tco2e - expected_indirect) < 0.001
    assert abs(result.total_embedded_emissions_tco2e - expected_total) < 0.001
    assert abs(result.cbam_liability_eur - expected_liability) < 0.01
    print(f"  PASSED: Direct={result.direct_emissions_tco2e}, Indirect={result.indirect_emissions_tco2e}, Total={result.total_embedded_emissions_tco2e}, Liability=EUR {result.cbam_liability_eur}")

    # Test 3: Determinism
    print("[TEST 3] Determinism Test (5 runs)...")
    results = [agent.run(input_data).total_embedded_emissions_tco2e for _ in range(5)]
    assert all(r == results[0] for r in results)
    print(f"  PASSED: All 5 runs produced {results[0]} tCO2e")

    # Test 4: Provenance Hash
    print("[TEST 4] Provenance Hash Format...")
    assert len(result.provenance_hash) == 64
    assert all(c in "0123456789abcdef" for c in result.provenance_hash)
    print(f"  PASSED: Valid SHA-256 hash ({result.provenance_hash[:16]}...)")

    # Test 5: Product Category Classification
    print("[TEST 5] Product Category Classification...")
    assert result.product_category == "iron_steel"
    assert result.calculation_method == "country_default"
    print(f"  PASSED: Category={result.product_category}, Method={result.calculation_method}")

    # Test 6: Aluminium High Indirect Emissions
    print("[TEST 6] Aluminium from China (100t) - High Indirect Emissions...")
    al_input = CBAMInput(
        cn_code="7601.10.00",
        quantity_tonnes=100.0,
        country_of_origin="CN",
        reporting_period="Q2 2026",
    )
    al_result = agent.run(al_input)

    # Aluminium has high indirect due to electricity intensity
    assert al_result.indirect_emissions_tco2e > al_result.direct_emissions_tco2e
    print(f"  PASSED: Direct={al_result.direct_emissions_tco2e}, Indirect={al_result.indirect_emissions_tco2e} (indirect > direct)")

    # Test 7: Zero Quantity
    print("[TEST 7] Zero Quantity Handling...")
    zero_input = CBAMInput(
        cn_code="7208.10.00",
        quantity_tonnes=0.0,
        country_of_origin="CN",
        reporting_period="Q1 2026",
    )
    zero_result = agent.run(zero_input)
    assert zero_result.total_embedded_emissions_tco2e == 0.0
    assert zero_result.cbam_liability_eur == 0.0
    print("  PASSED: Zero quantity returns zero emissions")

    # Test 8: Out of Scope CN Code
    print("[TEST 8] Out of Scope CN Code Error...")
    try:
        invalid_input = CBAMInput(
            cn_code="99990000",
            quantity_tonnes=100.0,
            country_of_origin="DE",
            reporting_period="Q1 2026",
        )
        agent.run(invalid_input)
        print("  FAILED: Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        assert "not in CBAM scope" in str(e)
        print(f"  PASSED: Correctly raised ValueError")

    # Test 9: Actual Emissions Override
    print("[TEST 9] Actual Emissions Override...")
    actual_input = CBAMInput(
        cn_code="7208.10.00",
        quantity_tonnes=1000.0,
        country_of_origin="CN",
        actual_emissions=1.5,  # Lower than default
        reporting_period="Q1 2026",
    )
    actual_result = agent.run(actual_input)
    assert actual_result.calculation_method == "actual"
    assert actual_result.total_embedded_emissions_tco2e == 1500.0  # 1000 * 1.5
    print(f"  PASSED: Actual emissions used, Total={actual_result.total_embedded_emissions_tco2e} tCO2e")

    # Test 10: CN Code Normalization
    print("[TEST 10] CN Code Normalization...")
    formats = ["7208.10.00", "72081000", "7208 1000"]
    for fmt in formats:
        input_fmt = CBAMInput(
            cn_code=fmt,
            quantity_tonnes=100.0,
            country_of_origin="DE",
            reporting_period="Q1 2026",
        )
        result_fmt = agent.run(input_fmt)
        assert result_fmt.cn_code == "72081000"
    print("  PASSED: All CN code formats normalized correctly")

    print()
    print("=" * 70)
    print("All 10 sample tests PASSED!")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        # Run full pytest suite
        import pytest
        sys.exit(pytest.main(["-v", "--tb=short", "test_agent.py"] + sys.argv[2:]))
    else:
        # Run sample tests
        sys.exit(run_sample_tests())
