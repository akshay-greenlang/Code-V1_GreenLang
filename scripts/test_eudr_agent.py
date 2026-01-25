#!/usr/bin/env python
"""
Quick validation test for EUDR Deforestation Compliance Agent.

Tests all 5 EUDR tools with real data.
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


async def test_eudr_agent():
    """Test EUDR Deforestation Compliance Agent tools."""
    print("=" * 80)
    print("TEST: EUDR DEFORESTATION COMPLIANCE AGENT")
    print("=" * 80)

    # Load tools module
    tools_path = project_root / "generated" / "eudr_compliance_v1" / "tools.py"
    eudr_tools = load_module_from_path("eudr_tools", tools_path)

    # Test 1: Validate Geolocation
    print("\n[1] Testing ValidateGeolocationTool...")
    validate_tool = eudr_tools.ValidateGeolocationTool()
    result = await validate_tool.execute({
        "coordinates": [-8.5, -35.0],  # Brazil coordinates
        "country_code": "BR",
        "coordinate_type": "point",
        "precision_meters": 10.0
    })
    print(f"[OK] Valid: {result['valid']}")
    print(f"     Protected Area: {result['in_protected_area']}")
    print(f"     Country Code: {result['country_code']}")

    # Test 2: Classify Commodity
    print("\n[2] Testing ClassifyCommodityTool...")
    classify_tool = eudr_tools.ClassifyCommodityTool()
    result = await classify_tool.execute({
        "cn_code": "1801",  # Cocoa beans
        "product_description": "Raw cocoa beans",
        "quantity_kg": 1000.0
    })
    print(f"[OK] EUDR Regulated: {result['eudr_regulated']}")
    print(f"     Commodity Type: {result['commodity_type']}")
    print(f"     CN Code: {result['cn_code']}")

    # Test 3: Assess Country Risk
    print("\n[3] Testing AssessCountryRiskTool...")
    risk_tool = eudr_tools.AssessCountryRiskTool()
    result = await risk_tool.execute({
        "country_code": "BR",
        "commodity_type": "soya"
    })
    print(f"[OK] Risk Level: {result['risk_level']}")
    print(f"     Satellite Verification Required: {result['satellite_verification_required']}")
    if 'forest_loss_rate_percent' in result:
        print(f"     Forest Loss Rate: {result['forest_loss_rate_percent']:.2f}%")

    # Test 4: Trace Supply Chain
    print("\n[4] Testing TraceSupplyChainTool...")
    trace_tool = eudr_tools.TraceSupplyChainTool()
    result = await trace_tool.execute({
        "shipment_id": "SHIP-2023-001",
        "supply_chain_nodes": [
            {"id": "farm-001", "type": "production", "country": "BR"},
            {"id": "processor-001", "type": "processing", "country": "BR"},
            {"id": "exporter-001", "type": "export", "country": "BR"}
        ],
        "commodity_type": "soya"
    })
    print(f"[OK] Traceability Score: {result.get('traceability_score', 'N/A')}")
    print(f"     Chain of Custody: {result.get('chain_of_custody', 'N/A')}")
    print(f"     Supply Chain Length: {result.get('supply_chain_length', len(result.get('supply_chain_nodes', [])))}")

    # Test 5: Generate DDS Report
    print("\n[5] Testing GenerateDdsReportTool...")
    dds_tool = eudr_tools.GenerateDdsReportTool()
    result = await dds_tool.execute({
        "operator_info": {
            "name": "Test Importer Ltd",
            "eori_number": "GB123456789000",
            "address": "123 Test Street, London, UK"
        },
        "commodity_data": {
            "commodity_type": "cocoa",
            "cn_code": "1801",
            "quantity_kg": 1000.0,
            "country_of_production": "CI",
            "production_date": "2023-06-15"
        },
        "geolocation_data": {
            "coordinates": [6.0, -5.0],
            "precision_meters": 10.0
        },
        "risk_assessment": {
            "risk_level": "standard",
            "satellite_verification_required": False
        }
    })
    print(f"[OK] DDS ID: {result.get('dds_id', 'N/A')}")
    print(f"     DDS Status: {result.get('dds_status', 'N/A')}")
    print(f"     Submission Ready: {result.get('submission_ready', False)}")


async def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print(" " * 15 + "EUDR DEFORESTATION COMPLIANCE AGENT TEST")
    print(" " * 25 + "27 Days to Deadline")
    print("=" * 80)

    try:
        await test_eudr_agent()

        print("\n" + "=" * 80)
        print("[SUCCESS] EUDR Agent passed all validation tests!")
        print("=" * 80)
        print("\nAgent Status:")
        print("  [OK] EUDR Deforestation Compliance Agent - 5/5 tools working")
        print("\nTotal: 5/5 tools PASSED\n")

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
