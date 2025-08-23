"""Example 05: CarbonAgent aggregation invariants."""

import math
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.carbon_agent import CarbonAgent
except Exception:
    CarbonAgent = None

@pytest.mark.example
def test_aggregate_sums_to_total():
    """sum(by_fuel) ≈ total; percentages sum ~100%."""
    if CarbonAgent is None:
        pytest.skip("CarbonAgent not importable")
    
    payload = {
        "emissions": [
            {"fuel": "electricity", "co2e_emissions_kg": 1_065_000.0},
            {"fuel": "diesel", "co2e_emissions_kg": 26_800.0},
        ]
    }
    
    out = CarbonAgent().run(payload)
    assert out["success"] is True
    d = out["data"]
    
    total = d["total_co2e_kg"]
    s = sum(d["by_fuel"].values())
    assert math.isclose(total, s, rel_tol=1e-9, abs_tol=1e-6)
    
    pct = sum(d.get("by_fuel_percent", {}).values()) if d.get("by_fuel_percent") else 100.0
    assert math.isclose(pct, 100.0, rel_tol=1e-6, abs_tol=1e-6)