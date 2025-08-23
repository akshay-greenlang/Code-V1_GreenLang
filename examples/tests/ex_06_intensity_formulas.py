"""Example 06: IntensityAgent formulas and zero guard."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.intensity_agent import IntensityAgent
except Exception:
    IntensityAgent = None

@pytest.mark.example
def test_intensity_calculation_with_zero_guard():
    """Intensity per sqft with zero area guard."""
    if IntensityAgent is None:
        pytest.skip("IntensityAgent not importable")
    
    # Normal case
    out = IntensityAgent().run({
        "total_co2e_kg": 1_091_800.0,
        "area": 50000,
        "area_unit": "sqft"
    })
    
    assert out["success"] is True
    assert out["data"]["co2e_per_sqft"] > 0
    
    # Zero area case
    out_zero = IntensityAgent().run({
        "total_co2e_kg": 1000.0,
        "area": 0,
        "area_unit": "sqft"
    })
    
    # Should handle gracefully
    assert out_zero["success"] is False or out_zero["data"]["co2e_per_sqft"] == 0