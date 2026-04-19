# -*- coding: utf-8 -*-
"""Tests for base agent contract compliance."""

import pytest
from typing import Dict, Any
from greenlang.agents.base_agent import BaseAgent
from greenlang.agents.fuel_agent import FuelAgent
from greenlang.agents.grid_factor_agent import GridFactorAgent
from greenlang.agents.carbon_agent import CarbonAgent
from greenlang.agents.intensity_agent import IntensityAgent
from greenlang.agents.benchmark_agent import BenchmarkAgent
from greenlang.agents.input_validator_agent import InputValidatorAgent
from greenlang.agents.building_profile_agent import BuildingProfileAgent
from greenlang.agents.recommendation_agent import RecommendationAgent
from greenlang.agents.report_agent import ReportAgent


class TestBaseAgentContract:
    """Test that all agents follow the base agent contract."""
    
    @pytest.fixture
    def all_agents(self):
        """Fixture providing all agent classes."""
        return [
            FuelAgent,
            GridFactorAgent,
            CarbonAgent,
            IntensityAgent,
            BenchmarkAgent,
            InputValidatorAgent,
            BuildingProfileAgent,
            RecommendationAgent,
            ReportAgent
        ]
    
    def test_all_agents_inherit_base(self, all_agents):
        """Test that all agents inherit from BaseAgent."""
        for agent_class in all_agents:
            assert issubclass(agent_class, BaseAgent), \
                f"{agent_class.__name__} doesn't inherit from BaseAgent"
    
    def test_all_agents_have_run_method(self, all_agents):
        """Test that all agents have a run method."""
        for agent_class in all_agents:
            agent = agent_class()
            assert hasattr(agent, 'run'), f"{agent_class.__name__} missing run method"
            assert callable(agent.run), f"{agent_class.__name__}.run is not callable"
    
    def test_run_returns_dict(self, all_agents):
        """Test that run method returns a dictionary."""
        test_inputs = {
            FuelAgent: {
                "fuel_type": "electricity",
                "consumption": {"value": 1000, "unit": "kWh"},
                "country": "IN"
            },
            GridFactorAgent: {
                "country": "IN",
                "fuel_type": "electricity",
                "unit": "kWh"
            },
            CarbonAgent: {
                "fuel_emissions": [
                    {"fuel_type": "electricity", "co2e_emissions_kg": 1000}
                ]
            },
            IntensityAgent: {
                "total_emissions_kg": 1000,
                "total_area_sqft": 100,
                "period": "annual"
            },
            BenchmarkAgent: {
                "emission_intensity_per_sqft": 10,
                "building_type": "office",
                "country": "IN"
            },
            InputValidatorAgent: {
                "building_type": "office",
                "country": "IN",
                "total_area": {"value": 1000, "unit": "sqft"}
            },
            BuildingProfileAgent: {
                "building_type": "office",
                "country": "IN",
                "total_area_sqft": 1000
            },
            RecommendationAgent: {
                "country": "IN",
                "building_type": "office",
                "performance_rating": "C",
                "building_age": 10
            },
            ReportAgent: {
                "report_data": {"total_emissions": 1000},
                "format": "json"
            }
        }
        
        for agent_class in all_agents:
            agent = agent_class()
            test_input = test_inputs.get(agent_class, {})
            
            result = agent.run(test_input)
            assert isinstance(result, dict), \
                f"{agent_class.__name__}.run doesn't return dict"
    
    def test_success_field_in_response(self, all_agents):
        """Test that response contains success field."""
        for agent_class in all_agents:
            agent = agent_class()
            
            # Test with empty input to trigger error
            result = agent.run({})
            
            assert "success" in result, \
                f"{agent_class.__name__} response missing 'success' field"
            assert isinstance(result["success"], bool), \
                f"{agent_class.__name__} 'success' field is not boolean"
    
    def test_error_structure_on_failure(self):
        """Test that errors follow consistent structure."""
        agent = FuelAgent()
        
        # Trigger an error with invalid input
        result = agent.run({
            "fuel_type": "invalid",
            "consumption": {"value": -100, "unit": "kWh"},
            "country": "XX"
        })
        
        if not result["success"]:
            assert "error" in result, "Failed response missing 'error' field"
            error = result["error"]
            
            # Check error structure
            assert "type" in error, "Error missing 'type' field"
            assert "message" in error, "Error missing 'message' field"
            
            # Type should be a string
            assert isinstance(error["type"], str), "Error type is not string"
            assert isinstance(error["message"], str), "Error message is not string"
            
            # Optional context field
            if "context" in error:
                assert isinstance(error["context"], dict), "Error context is not dict"
    
    def test_data_field_on_success(self):
        """Test that successful responses have data field."""
        agent = FuelAgent()
        
        result = agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": 1000, "unit": "kWh"},
            "country": "IN"
        })
        
        if result["success"]:
            assert "data" in result, "Successful response missing 'data' field"
            assert isinstance(result["data"], dict), "Data field is not dict"
    
    def test_agent_id_in_responses(self, all_agents):
        """Test that responses include agent identification."""
        for agent_class in all_agents:
            agent = agent_class()
            
            # Some agents might include agent_id
            if hasattr(agent, 'agent_id'):
                result = agent.run({})
                
                # Check if agent_id is in response or error
                if "agent_id" in result:
                    assert result["agent_id"] == agent.agent_id
                elif not result["success"] and "error" in result:
                    if "agent_id" in result["error"]:
                        assert result["error"]["agent_id"] == agent.agent_id
    
    def test_input_validation(self):
        """Test that agents validate input properly."""
        agent = FuelAgent()
        
        # Test with None input
        result = agent.run(None)
        assert not result["success"] or result["data"] == {}
        
        # Test with wrong type
        result = agent.run("not a dict")
        assert not result["success"] or result["data"] == {}
        
        # Test with missing required fields
        result = agent.run({"fuel_type": "electricity"})  # Missing consumption
        assert not result["success"]
    
    def test_error_types_consistent(self):
        """Test that error types follow naming convention."""
        common_error_types = [
            "ValidationError",
            "DataNotFoundError",
            "CalculationError",
            "ConfigurationError",
            "ValueError",
            "TypeError",
            "KeyError",
            "UnitError",
            "MissingFieldError"
        ]
        
        agent = FuelAgent()
        
        # Trigger various errors
        test_cases = [
            {"fuel_type": "invalid"},  # Should trigger validation/data error
            {"fuel_type": "electricity", "consumption": {"value": -100, "unit": "kWh"}},  # Validation
            {}  # Missing fields
        ]
        
        for test_input in test_cases:
            result = agent.run(test_input)
            if not result["success"] and "error" in result:
                error_type = result["error"]["type"]
                # Should be one of the common types or end with "Error"
                assert error_type.endswith("Error") or error_type in common_error_types, \
                    f"Non-standard error type: {error_type}"
    
    def test_numeric_outputs_are_finite(self):
        """Test that numeric outputs are finite (not NaN or Inf)."""
        import math
        
        agent = IntensityAgent()
        
        result = agent.run({
            "total_emissions_kg": 1000,
            "total_area_sqft": 100,
            "period": "annual"
        })
        
        if result["success"]:
            data = result["data"]
            
            def check_finite(obj, path=""):
                if isinstance(obj, (int, float)):
                    assert math.isfinite(obj), f"Non-finite value at {path}: {obj}"
                elif isinstance(obj, dict):
                    for key, value in obj.items():
                        check_finite(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_finite(item, f"{path}[{i}]")
            
            check_finite(data)
    
    def test_thread_safety(self, all_agents):
        """Test that agents can be used safely in multiple threads."""
        import threading
        import time
        
        def run_agent(agent_class, results, index):
            """Run agent and store result."""
            agent = agent_class()
            test_input = {
                "fuel_type": "electricity",
                "consumption": {"value": 1000, "unit": "kWh"},
                "country": "IN"
            } if agent_class == FuelAgent else {}
            
            result = agent.run(test_input)
            results[index] = result
        
        for agent_class in all_agents[:3]:  # Test a few agents
            results = {}
            threads = []
            
            # Create multiple threads
            for i in range(5):
                thread = threading.Thread(
                    target=run_agent,
                    args=(agent_class, results, i)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join(timeout=5)
            
            # All threads should complete
            assert len(results) == 5, f"Not all threads completed for {agent_class.__name__}"