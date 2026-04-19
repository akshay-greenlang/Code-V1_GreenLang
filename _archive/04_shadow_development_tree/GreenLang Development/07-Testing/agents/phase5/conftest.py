# -*- coding: utf-8 -*-
"""
Pytest fixtures for Phase 5 Critical Path Compliance Tests

Provides test data and utilities for validating deterministic agent behavior.
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime
import hashlib
import json


@pytest.fixture
def sample_fuel_consumption_natural_gas():
    """Sample natural gas consumption data."""
    return {
        "fuel_type": "natural_gas",
        "amount": 1000.0,
        "unit": "therms",
        "country": "US",
        "year": 2025
    }


@pytest.fixture
def sample_fuel_consumption_electricity():
    """Sample electricity consumption data."""
    return {
        "fuel_type": "electricity",
        "amount": 5000.0,
        "unit": "kWh",
        "country": "US",
        "year": 2025
    }


@pytest.fixture
def sample_fuel_consumption_diesel():
    """Sample diesel consumption data."""
    return {
        "fuel_type": "diesel",
        "amount": 250.0,
        "unit": "gallons",
        "country": "US",
        "year": 2025
    }


@pytest.fixture
def sample_grid_factor_request():
    """Sample grid factor request."""
    return {
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh",
        "year": 2025
    }


@pytest.fixture
def sample_boiler_thermal_output():
    """Sample boiler with thermal output input."""
    return {
        "boiler_type": "condensing",
        "fuel_type": "natural_gas",
        "thermal_output": {
            "value": 100.0,
            "unit": "MMBtu"
        },
        "efficiency": 0.95,
        "country": "US",
        "year": 2025
    }


@pytest.fixture
def sample_boiler_fuel_consumption():
    """Sample boiler with fuel consumption input."""
    return {
        "boiler_type": "standard",
        "fuel_type": "natural_gas",
        "fuel_consumption": {
            "value": 1000.0,
            "unit": "therms"
        },
        "efficiency": 0.85,
        "country": "US",
        "year": 2025
    }


@pytest.fixture
def sample_carbon_aggregation():
    """Sample carbon aggregation data."""
    return {
        "emissions": [
            {
                "fuel_type": "natural_gas",
                "co2e_emissions_kg": 5310.0,
                "source": "boiler"
            },
            {
                "fuel_type": "electricity",
                "co2e_emissions_kg": 1925.0,
                "source": "lighting"
            },
            {
                "fuel_type": "diesel",
                "co2e_emissions_kg": 2552.5,
                "source": "backup_generator"
            }
        ],
        "building_area": 10000,
        "occupancy": 100
    }


@pytest.fixture
def determinism_test_inputs():
    """Multiple input variations for determinism testing."""
    return [
        # Natural gas variations
        {
            "fuel_type": "natural_gas",
            "amount": 100.0,
            "unit": "therms",
            "country": "US",
            "year": 2025
        },
        {
            "fuel_type": "natural_gas",
            "amount": 500.0,
            "unit": "therms",
            "country": "US",
            "year": 2025
        },
        {
            "fuel_type": "natural_gas",
            "amount": 1000.0,
            "unit": "therms",
            "country": "US",
            "year": 2025
        },
        # Electricity variations
        {
            "fuel_type": "electricity",
            "amount": 1000.0,
            "unit": "kWh",
            "country": "US",
            "year": 2025
        },
        {
            "fuel_type": "electricity",
            "amount": 5000.0,
            "unit": "kWh",
            "country": "US",
            "year": 2025
        },
        # Diesel variations
        {
            "fuel_type": "diesel",
            "amount": 100.0,
            "unit": "gallons",
            "country": "US",
            "year": 2025
        },
    ]


@pytest.fixture
def hash_result():
    """Helper function to create deterministic hash of agent result."""
    def _hash(result: Dict[str, Any]) -> str:
        """Create SHA256 hash of agent result for comparison.

        Args:
            result: Agent result dictionary

        Returns:
            str: SHA256 hash of serialized result
        """
        if not result.get("success"):
            return "ERROR"

        # Extract only data portion for hashing
        data = result.get("data", {})

        # Sort keys for deterministic serialization
        serialized = json.dumps(data, sort_keys=True, default=str)

        # Create hash
        return hashlib.sha256(serialized.encode()).hexdigest()

    return _hash


@pytest.fixture
def assert_deterministic_result():
    """Helper to assert two results are byte-for-byte identical."""
    def _assert(result1: Dict[str, Any], result2: Dict[str, Any]):
        """Assert two agent results are identical.

        Args:
            result1: First agent result
            result2: Second agent result
        """
        assert result1["success"] == result2["success"], "Success flags differ"

        if result1["success"]:
            data1 = result1["data"]
            data2 = result2["data"]

            # Check all numeric values match exactly
            for key in data1:
                if isinstance(data1[key], (int, float)):
                    assert data1[key] == data2[key], f"Value mismatch for {key}: {data1[key]} != {data2[key]}"

            # Check string values match
            for key in data1:
                if isinstance(data1[key], str):
                    assert data1[key] == data2[key], f"String mismatch for {key}"

    return _assert


@pytest.fixture
def assert_no_llm_dependencies():
    """Helper to check agent has no LLM dependencies."""
    def _assert(agent_module):
        """Check agent module has no ChatSession or LLM imports.

        Args:
            agent_module: The agent module to check
        """
        import inspect
        source = inspect.getsource(agent_module)

        # Check for banned imports
        banned_patterns = [
            "from greenlang.intelligence.chatsession import ChatSession",
            "from greenlang.intelligence.rag",
            "temperature=",
            "openai",
            "anthropic",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY"
        ]

        for pattern in banned_patterns:
            assert pattern not in source, f"Found banned pattern: {pattern}"

    return _assert


@pytest.fixture
def assert_complete_audit_trail():
    """Helper to verify complete audit trail in result."""
    def _assert(result: Dict[str, Any]):
        """Check result has complete audit trail.

        Args:
            result: Agent result to validate
        """
        assert result["success"], "Result must be successful"

        # Check metadata exists
        assert "metadata" in result, "Missing metadata"
        metadata = result["metadata"]

        # Check required metadata fields
        assert "agent_id" in metadata, "Missing agent_id"
        assert "calculation" in metadata or "source" in metadata, "Missing calculation details"

        # Check data has required fields
        data = result["data"]
        assert "co2e_emissions_kg" in data or "emission_factor" in data or "total_co2e_kg" in data, "Missing emissions data"

    return _assert


@pytest.fixture
def performance_benchmark():
    """Helper to benchmark agent performance."""
    def _benchmark(agent_func, payload: Dict[str, Any], target_ms: float = 10.0):
        """Benchmark agent execution time.

        Args:
            agent_func: Agent run function to benchmark
            payload: Input payload
            target_ms: Target execution time in milliseconds

        Returns:
            tuple: (result, execution_time_ms)
        """
        import time

        start = time.perf_counter()
        result = agent_func(payload)
        end = time.perf_counter()

        execution_time_ms = (end - start) * 1000

        return result, execution_time_ms

    return _benchmark


@pytest.fixture
def cross_country_test_data():
    """Test data with multiple countries for grid factor testing."""
    return [
        {"country": "US", "fuel_type": "electricity", "unit": "kWh", "year": 2025},
        {"country": "UK", "fuel_type": "electricity", "unit": "kWh", "year": 2025},
        {"country": "DE", "fuel_type": "electricity", "unit": "kWh", "year": 2025},
        {"country": "IN", "fuel_type": "electricity", "unit": "kWh", "year": 2025},
        {"country": "CN", "fuel_type": "electricity", "unit": "kWh", "year": 2025},
    ]
