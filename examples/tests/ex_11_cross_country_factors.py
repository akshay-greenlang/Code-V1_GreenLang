"""Example 11: Cross-country comparison (dataset-driven)."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.grid_factor_agent import GridFactorAgent
except Exception:
    GridFactorAgent = None

from examples.utils.dataset import load_emission_factor

@pytest.mark.example
def test_cross_country_comparison():
    """Compare emission factors across countries."""
    if GridFactorAgent is None:
        pytest.skip("GridFactorAgent not importable")
    
    countries = ["IN", "US", "EU"]
    factors = {}
    
    for country in countries:
        # From dataset
        dataset_factor = load_emission_factor(
            country=country,
            fuel="electricity",
            unit="kWh"
        )
        
        # From agent
        out = GridFactorAgent().run({
            "country": country,
            "fuel_type": "electricity",
            "unit": "kWh"
        })
        
        assert out["success"] is True
        agent_factor = out["data"]["emission_factor"]
        
        # Should match
        assert abs(dataset_factor - agent_factor) < 0.01
        factors[country] = agent_factor
    
    # Basic sanity check: factors should differ
    assert len(set(factors.values())) > 1