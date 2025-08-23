"""Example 01: Basic FuelAgent electricity calculation (dataset-driven)."""

import math
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.fuel_agent import FuelAgent
except Exception:
    FuelAgent = None

from examples.utils.dataset import load_emission_factor

@pytest.mark.example
def test_fuelagent_electricity_india_basic():
    """Dataset-driven electricity calc — no magic numbers."""
    if FuelAgent is None:
        pytest.skip("FuelAgent not importable")
    
    kwh = 1_500_000
    factor = load_emission_factor(country="IN", fuel="electricity", unit="kWh")
    
    out = FuelAgent().run({
        "fuel_type": "electricity",
        "consumption": {"value": kwh, "unit": "kWh"},
        "country": "IN"
    })
    
    assert out["success"] is True
    assert math.isclose(out["data"]["co2e_emissions_kg"], kwh * factor, rel_tol=1e-9)
    assert {"source", "version", "last_updated"} <= out["data"].keys()