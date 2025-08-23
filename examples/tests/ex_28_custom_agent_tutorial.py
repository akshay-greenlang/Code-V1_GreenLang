"""Example 28: Test for custom agent tutorial (30 lines)."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from examples.tutorials.custom_agent_30_lines import WaterUsageAgent
except Exception:
    WaterUsageAgent = None

@pytest.mark.example
def test_custom_water_agent():
    """Test the custom water usage agent."""
    if WaterUsageAgent is None:
        pytest.skip("WaterUsageAgent tutorial not available")
    
    agent = WaterUsageAgent()
    
    # Test basic calculation
    result = agent.run({
        "water_consumption": {"value": 10000, "unit": "liters"}
    })
    
    assert result["success"] is True
    assert result["data"]["water_consumption_liters"] == 10000
    assert result["data"]["co2e_emissions_kg"] == 3.0  # 10000 * 0.0003
    
    # Test empty input
    result_empty = agent.run({})
    assert result_empty["success"] is True
    assert result_empty["data"]["co2e_emissions_kg"] == 0

@pytest.mark.example
def test_custom_agent_properties():
    """Test custom agent has required properties."""
    if WaterUsageAgent is None:
        pytest.skip("WaterUsageAgent tutorial not available")
    
    agent = WaterUsageAgent()
    
    # Check agent properties
    assert agent.agent_id == "water_usage"
    assert agent.name == "Water Usage Emissions Calculator"
    assert agent.version == "1.0.0"
    assert hasattr(agent, "run")