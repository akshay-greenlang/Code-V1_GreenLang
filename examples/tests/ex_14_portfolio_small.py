"""Example 14: Small portfolio calculation via SDK."""

import json
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.sdk import GreenLangSDK
except Exception:
    GreenLangSDK = None

@pytest.mark.example
def test_portfolio_small_via_sdk():
    """Process small portfolio of buildings."""
    if GreenLangSDK is None:
        pytest.skip("GreenLangSDK not importable")
    
    with open("examples/fixtures/portfolio_small.json", "r") as f:
        portfolio_data = json.load(f)
    
    sdk = GreenLangSDK()
    results = []
    
    for building in portfolio_data["buildings"]:
        result = sdk.calculate_building_emissions(building)
        if result and result.get("success"):
            results.append({
                "building_id": building["building_id"],
                "emissions": result["data"].get("emissions", {}).get("total_co2e_kg", 0)
            })
    
    assert len(results) == 3
    assert all(r["emissions"] > 0 for r in results)