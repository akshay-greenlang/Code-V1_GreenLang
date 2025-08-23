"""Example 19: Property-based testing (additivity, scaling, unit round-trip)."""

import pytest
import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.carbon_agent import CarbonAgent
    from greenlang.agents.fuel_agent import FuelAgent
    from greenlang.utils.units import convert_unit
except Exception:
    CarbonAgent = None
    FuelAgent = None
    convert_unit = None

@pytest.mark.example
def test_property_additivity():
    """Property: emissions(A+B) = emissions(A) + emissions(B)."""
    if FuelAgent is None:
        pytest.skip("FuelAgent not importable")
    
    agent = FuelAgent()
    
    # Calculate separately
    result_a = agent.run({
        "fuel_type": "electricity",
        "consumption": {"value": 1000, "unit": "kWh"},
        "country": "US"
    })
    
    result_b = agent.run({
        "fuel_type": "electricity",
        "consumption": {"value": 500, "unit": "kWh"},
        "country": "US"
    })
    
    # Calculate combined
    result_combined = agent.run({
        "fuel_type": "electricity",
        "consumption": {"value": 1500, "unit": "kWh"},
        "country": "US"
    })
    
    # Additivity property
    emissions_a = result_a["data"]["co2e_emissions_kg"]
    emissions_b = result_b["data"]["co2e_emissions_kg"]
    emissions_combined = result_combined["data"]["co2e_emissions_kg"]
    
    assert math.isclose(emissions_a + emissions_b, emissions_combined, rel_tol=1e-9)

@pytest.mark.example
def test_property_scaling():
    """Property: emissions(k*X) = k * emissions(X)."""
    if FuelAgent is None:
        pytest.skip("FuelAgent not importable")
    
    agent = FuelAgent()
    base_value = 100
    scale_factor = 7
    
    # Base calculation
    result_base = agent.run({
        "fuel_type": "natural_gas",
        "consumption": {"value": base_value, "unit": "therms"},
        "country": "US"
    })
    
    # Scaled calculation
    result_scaled = agent.run({
        "fuel_type": "natural_gas",
        "consumption": {"value": base_value * scale_factor, "unit": "therms"},
        "country": "US"
    })
    
    # Scaling property
    emissions_base = result_base["data"]["co2e_emissions_kg"]
    emissions_scaled = result_scaled["data"]["co2e_emissions_kg"]
    
    assert math.isclose(emissions_base * scale_factor, emissions_scaled, rel_tol=1e-9)

@pytest.mark.example
def test_property_unit_round_trip():
    """Property: convert(convert(X, A→B), B→A) = X."""
    if convert_unit is None:
        pytest.skip("Unit converter not importable")
    
    # Test various unit pairs
    test_cases = [
        (1000, "kWh", "MWh"),
        (50000, "sqft", "sqm"),
        (100, "liters", "gallons"),
    ]
    
    for value, unit_a, unit_b in test_cases:
        # Convert A → B
        converted = convert_unit(value, unit_a, unit_b)
        
        # Convert B → A
        round_trip = convert_unit(converted, unit_b, unit_a)
        
        # Should get back original value (within floating point tolerance)
        assert math.isclose(value, round_trip, rel_tol=1e-9), \
            f"Round trip failed for {value} {unit_a} → {unit_b} → {unit_a}"