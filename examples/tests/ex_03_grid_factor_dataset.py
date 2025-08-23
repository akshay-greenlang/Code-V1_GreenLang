"""Example 03: GridFactorAgent factor retrieval with provenance."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.grid_factor_agent import GridFactorAgent
except Exception:
    GridFactorAgent = None

@pytest.mark.example
def test_gridfactor_country_fuel_unit():
    """Read factor + provenance for EU electricity."""
    if GridFactorAgent is None:
        pytest.skip("GridFactorAgent not importable")
    
    out = GridFactorAgent().run({
        "country": "EU",
        "fuel_type": "electricity",
        "unit": "kWh"
    })
    
    assert out["success"] is True
    d = out["data"]
    assert d["emission_factor"] > 0
    assert {"source", "version", "last_updated"} <= d.keys()