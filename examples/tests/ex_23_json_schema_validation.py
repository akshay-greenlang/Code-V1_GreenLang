"""Example 23: JSON Schema validation for outputs."""

import pytest
import json
import jsonschema
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.agents.fuel_agent import FuelAgent
    from greenlang.agents.report_agent import ReportAgent
except Exception:
    FuelAgent = None
    ReportAgent = None

# Define schemas for validation
FUEL_AGENT_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["success", "data"],
    "properties": {
        "success": {"type": "boolean"},
        "data": {
            "type": "object",
            "required": ["co2e_emissions_kg", "fuel_type", "consumption_value"],
            "properties": {
                "co2e_emissions_kg": {"type": "number", "minimum": 0},
                "fuel_type": {"type": "string"},
                "consumption_value": {"type": "number", "minimum": 0},
                "consumption_unit": {"type": "string"},
                "emission_factor": {"type": "number", "minimum": 0},
                "source": {"type": "string"},
                "version": {"type": "string"},
                "last_updated": {"type": "string"}
            }
        },
        "error": {"type": "string"}
    }
}

REPORT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["total_emissions_kg", "emissions_by_fuel", "intensity"],
    "properties": {
        "total_emissions_kg": {"type": "number", "minimum": 0},
        "emissions_by_fuel": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0}
        },
        "intensity": {
            "type": "object",
            "properties": {
                "co2e_per_sqft": {"type": "number", "minimum": 0}
            }
        },
        "benchmark": {
            "type": "object",
            "properties": {
                "rating": {"type": "string"},
                "threshold": {"type": "number"}
            }
        }
    }
}

@pytest.mark.example
def test_fuel_agent_output_schema():
    """Validate FuelAgent output against JSON schema."""
    if FuelAgent is None:
        pytest.skip("FuelAgent not importable")
    
    agent = FuelAgent()
    result = agent.run({
        "fuel_type": "electricity",
        "consumption": {"value": 1000, "unit": "kWh"},
        "country": "US"
    })
    
    # Validate against schema
    try:
        jsonschema.validate(instance=result, schema=FUEL_AGENT_OUTPUT_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        pytest.fail(f"Output does not match schema: {e.message}")

@pytest.mark.example
def test_report_json_schema():
    """Validate report JSON output against schema."""
    if ReportAgent is None:
        pytest.skip("ReportAgent not importable")
    
    agent = ReportAgent()
    result = agent.run({
        "emissions": {
            "total_co2e_kg": 1091800.0,
            "by_fuel": {"electricity": 1065000.0, "diesel": 26800.0}
        },
        "intensity": {"co2e_per_sqft": 21.84},
        "benchmark": {"rating": "good", "threshold": 25.0},
        "format": "json"
    })
    
    if result["success"]:
        report_data = result["data"]["report"]
        if isinstance(report_data, str):
            report_data = json.loads(report_data)
        
        # Validate against schema
        try:
            jsonschema.validate(instance=report_data, schema=REPORT_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            pytest.fail(f"Report does not match schema: {e.message}")

@pytest.mark.example
def test_schema_driven_property_testing():
    """Use schema to generate test cases."""
    if FuelAgent is None:
        pytest.skip("FuelAgent not importable")
    
    # Generate test cases from schema constraints
    test_cases = [
        {"fuel_type": "electricity", "value": 0, "unit": "kWh"},     # Minimum
        {"fuel_type": "natural_gas", "value": 1e6, "unit": "therms"}, # Large
        {"fuel_type": "diesel", "value": 0.001, "unit": "liters"},   # Small
    ]
    
    agent = FuelAgent()
    for case in test_cases:
        result = agent.run({
            "fuel_type": case["fuel_type"],
            "consumption": {"value": case["value"], "unit": case["unit"]},
            "country": "US"
        })
        
        # All should validate against schema
        try:
            jsonschema.validate(instance=result, schema=FUEL_AGENT_OUTPUT_SCHEMA)
        except jsonschema.exceptions.ValidationError:
            pytest.fail(f"Failed schema validation for case: {case}")