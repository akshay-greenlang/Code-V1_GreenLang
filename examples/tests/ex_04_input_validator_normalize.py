"""Example 04: InputValidatorAgent unit normalization."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.input_validator_agent import InputValidatorAgent
except Exception:
    InputValidatorAgent = None

@pytest.mark.example
def test_validator_normalizes_units():
    """Mixed units in → normalized out (sqm→sqft, MWh→kWh)."""
    if InputValidatorAgent is None:
        pytest.skip("InputValidatorAgent not importable")
    
    payload = {
        "metadata": {
            "building_type": "commercial_office",
            "area": 4645.152,
            "area_unit": "sqm",
            "location": {"country": "IN"}
        },
        "energy_consumption": {
            "electricity": {"value": 1_500, "unit": "MWh"}
        }
    }
    
    out = InputValidatorAgent().run(payload)
    assert out["success"] is True
    norm = out["data"]
    assert norm["metadata"]["area_unit"].lower() in ("sqft", "square_feet")
    assert norm["energy_consumption"]["electricity"]["unit"].lower() == "kwh"