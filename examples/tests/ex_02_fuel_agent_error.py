"""Example 02: FuelAgent error path (negative input)."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.fuel_agent import FuelAgent
except Exception:
    FuelAgent = None

@pytest.mark.example
def test_fuelagent_negative_input_error():
    """Negative consumption should fail gracefully."""
    if FuelAgent is None:
        pytest.skip("FuelAgent not importable")
    
    out = FuelAgent().run({
        "fuel_type": "electricity",
        "consumption": {"value": -1000, "unit": "kWh"},
        "country": "US"
    })
    
    assert out["success"] is False
    assert "error" in out or "message" in out