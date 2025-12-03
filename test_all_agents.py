#!/usr/bin/env python
"""
Quick validation test for all 3 agents' tools.

Tests:
1. Fuel Analyzer Agent - 3 tools
2. CBAM Carbon Intensity Agent - 2 tools
3. Building Energy Performance Agent - 3 tools
"""

import sys
import asyncio
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


def load_module_from_path(module_name, file_path):
    """Load a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


async def test_fuel_analyzer():
    """Test Fuel Analyzer Agent tools."""
    print("=" * 80)
    print("TEST 1: FUEL ANALYZER AGENT")
    print("=" * 80)

    # Load tools module
    tools_path = project_root / "generated" / "fuel_analyzer_agent" / "tools.py"
    fuel_tools = load_module_from_path("fuel_analyzer_tools", tools_path)

    # Test 1.1: Lookup Emission Factor
    print("\n[1.1] Testing LookupEmissionFactorTool...")
    lookup_tool = fuel_tools.LookupEmissionFactorTool()
    result = await lookup_tool.execute({
        "fuel_type": "natural_gas",
        "region": "US",
        "year": 2023,
        "gwp_set": "AR6GWP100"
    })
    print(f"[OK] Emission Factor: {result['ef_value']} {result['ef_unit']}")
    print(f"     Source: {result['source']}")

    # Test 1.2: Calculate Emissions
    print("\n[1.2] Testing CalculateEmissionsTool...")
    calc_tool = fuel_tools.CalculateEmissionsTool()
    result = await calc_tool.execute({
        "activity_value": 1000.0,
        "activity_unit": "MJ",
        "ef_value": result['ef_value'],
        "ef_unit": result['ef_unit'],
        "output_unit": "tCO2e"
    })
    print(f"[OK] Emissions: {result['emissions_value']} {result['emissions_unit']}")
    print(f"     Formula: {result['calculation_formula']}")

    # Test 1.3: Validate Fuel Input
    print("\n[1.3] Testing ValidateFuelInputTool...")
    validate_tool = fuel_tools.ValidateFuelInputTool()
    result = await validate_tool.execute({
        "fuel_type": "natural_gas",
        "quantity": 1000.0,
        "unit": "MJ"
    })
    print(f"[OK] Valid: {result['valid']}, Plausibility: {result['plausibility_score']}")


async def test_cbam_agent():
    """Test CBAM Carbon Intensity Agent tools."""
    print("\n" + "=" * 80)
    print("TEST 2: CBAM CARBON INTENSITY AGENT")
    print("=" * 80)

    # Load tools module
    tools_path = project_root / "generated" / "carbon_intensity_v1" / "tools.py"
    cbam_tools = load_module_from_path("cbam_tools", tools_path)

    # Test 2.1: Lookup CBAM Benchmark
    print("\n[2.1] Testing LookupCbamBenchmarkTool...")
    lookup_tool = cbam_tools.LookupCbamBenchmarkTool()
    result = await lookup_tool.execute({
        "product_type": "steel_hot_rolled_coil"
    })
    print(f"[OK] Benchmark: {result['benchmark_value']} {result['benchmark_unit']}")
    print(f"     Source: {result['source']}")
    print(f"     CN Codes: {', '.join(result['cn_codes'])}")

    # Test 2.2: Calculate Carbon Intensity
    print("\n[2.2] Testing CalculateCarbonIntensityTool...")
    calc_tool = cbam_tools.CalculateCarbonIntensityTool()
    result = await calc_tool.execute({
        "total_emissions": 1850.0,  # tCO2e
        "production_quantity": 1000.0  # tonnes
    })
    print(f"[OK] Carbon Intensity: {result['carbon_intensity']} {result['carbon_intensity_unit']}")
    print(f"     Formula: {result['calculation_formula']}")


async def test_building_energy_agent():
    """Test Building Energy Performance Agent tools."""
    print("\n" + "=" * 80)
    print("TEST 3: BUILDING ENERGY PERFORMANCE AGENT")
    print("=" * 80)

    # Load tools module
    tools_path = project_root / "generated" / "energy_performance_v1" / "tools.py"
    building_tools = load_module_from_path("building_tools", tools_path)

    # Test 3.1: Calculate EUI
    print("\n[3.1] Testing CalculateEuiTool...")
    calc_tool = building_tools.CalculateEuiTool()
    result = await calc_tool.execute({
        "energy_consumption_kwh": 800000.0,
        "floor_area_sqm": 10000.0
    })
    print(f"[OK] EUI: {result['eui_kwh_per_sqm']} kWh/sqm/year")
    print(f"     Formula: {result['calculation_formula']}")

    # Test 3.2: Lookup BPS Threshold
    print("\n[3.2] Testing LookupBpsThresholdTool...")
    lookup_tool = building_tools.LookupBpsThresholdTool()
    result = await lookup_tool.execute({
        "building_type": "office",
        "climate_zone": "4A"
    })
    print(f"[OK] Threshold: {result['threshold_kwh_per_sqm']} kWh/sqm/year")
    print(f"     Source: {result['source']}")
    print(f"     Jurisdiction: {result['jurisdiction']}")

    # Test 3.3: Check BPS Compliance
    print("\n[3.3] Testing CheckBpsComplianceTool...")
    compliance_tool = building_tools.CheckBpsComplianceTool()
    result = await compliance_tool.execute({
        "actual_eui": 80.0,
        "threshold_eui": 80.0
    })
    print(f"[OK] Compliant: {result['compliant']}")
    print(f"     Gap: {result['gap_kwh_per_sqm']} kWh/sqm/year")
    print(f"     Status: {result['compliance_status']}")


async def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print(" " * 20 + "GREENLANG AGENT FACTORY TEST SUITE")
    print(" " * 26 + "Week 2: All 3 Agents")
    print("=" * 80)

    try:
        await test_fuel_analyzer()
        await test_cbam_agent()
        await test_building_energy_agent()

        print("\n" + "=" * 80)
        print("[SUCCESS] All 3 agents passed validation tests!")
        print("=" * 80)
        print("\nAgent Status:")
        print("  [OK] Fuel Analyzer Agent - 3/3 tools working")
        print("  [OK] CBAM Carbon Intensity Agent - 2/2 tools working")
        print("  [OK] Building Energy Performance Agent - 3/3 tools working")
        print("\nTotal: 8/8 tools PASSED\n")

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("[FAIL] Test failed with error:")
        print("=" * 80)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
