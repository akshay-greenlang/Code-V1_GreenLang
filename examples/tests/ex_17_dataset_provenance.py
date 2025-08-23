"""Example 17: Dataset provenance propagation."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.fuel_agent import FuelAgent
    from greenlang.agents.grid_factor_agent import GridFactorAgent
except Exception:
    FuelAgent = None
    GridFactorAgent = None

@pytest.mark.example
def test_dataset_provenance_propagation():
    """Provenance info propagates through pipeline."""
    if FuelAgent is None or GridFactorAgent is None:
        pytest.skip("Agents not importable")
    
    # Get factor with provenance
    factor_result = GridFactorAgent().run({
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh"
    })
    
    assert factor_result["success"] is True
    provenance = {
        "source": factor_result["data"].get("source"),
        "version": factor_result["data"].get("version"),
        "last_updated": factor_result["data"].get("last_updated")
    }
    
    # Calculate emissions
    fuel_result = FuelAgent().run({
        "fuel_type": "electricity",
        "consumption": {"value": 1000, "unit": "kWh"},
        "country": "US"
    })
    
    assert fuel_result["success"] is True
    
    # Check provenance is included
    for key in ["source", "version", "last_updated"]:
        assert key in fuel_result["data"]