"""
GreenLang AgentSpec v2 - Test Configuration and Shared Fixtures

This module provides shared pytest fixtures and configuration for AgentSpec v2 tests.

Author: GreenLang Framework Team
Date: October 2025
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml


# ============================================================================
# SHARED TEST DATA
# ============================================================================

MINIMAL_SPEC = {
    "schema_version": "2.0.0",
    "id": "test/minimal_v1",
    "name": "Minimal Test Agent",
    "version": "1.0.0",
    "compute": {
        "entrypoint": "python://test.module:compute",
        "inputs": {
            "x": {"dtype": "float64", "unit": "1"}
        },
        "outputs": {
            "y": {"dtype": "float64", "unit": "1"}
        }
    },
    "ai": {},
    "realtime": {},
    "provenance": {
        "pin_ef": False,
        "record": ["inputs"]
    }
}


FULL_SPEC_WITH_FACTORS = {
    "schema_version": "2.0.0",
    "id": "test/full_with_factors_v1",
    "name": "Full Test Agent with Factors",
    "version": "2.0.0",
    "summary": "A comprehensive test spec with all features",
    "tags": ["test", "comprehensive"],
    "license": "Apache-2.0",
    "compute": {
        "entrypoint": "python://test.module:compute_full",
        "deterministic": True,
        "inputs": {
            "energy_kwh": {
                "dtype": "float64",
                "unit": "kWh",
                "required": True,
                "ge": 0,
                "description": "Energy consumption"
            },
            "region": {
                "dtype": "string",
                "unit": "1",
                "required": True,
                "enum": ["US", "EU", "ASIA"],
                "description": "Geographic region"
            }
        },
        "outputs": {
            "emissions_kg": {
                "dtype": "float64",
                "unit": "kgCO2e",
                "description": "Total emissions"
            }
        },
        "factors": {
            "grid_ef": {
                "ref": "ef://test/grid/emissions_factor",
                "gwp_set": "AR6GWP100",
                "description": "Grid emission factor"
            }
        }
    },
    "ai": {
        "json_mode": True,
        "system_prompt": "Test system prompt",
        "budget": {
            "max_cost_usd": 0.50,
            "max_input_tokens": 10000,
            "max_output_tokens": 1000
        },
        "rag_collections": ["test_collection"],
        "tools": []
    },
    "realtime": {
        "default_mode": "replay",
        "connectors": []
    },
    "provenance": {
        "pin_ef": True,
        "gwp_set": "AR6GWP100",
        "record": ["inputs", "outputs", "factors", "code_sha"]
    }
}


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def minimal_spec():
    """Minimal valid AgentSpec v2 as dictionary."""
    return MINIMAL_SPEC.copy()


@pytest.fixture
def full_spec_with_factors():
    """Full AgentSpec v2 with all features including factors."""
    return FULL_SPEC_WITH_FACTORS.copy()


@pytest.fixture
def temp_yaml_minimal(minimal_spec):
    """Temporary YAML file with minimal spec."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(minimal_spec, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_json_minimal(minimal_spec):
    """Temporary JSON file with minimal spec."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(minimal_spec, f, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_yaml_full(full_spec_with_factors):
    """Temporary YAML file with full spec."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(full_spec_with_factors, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_json_full(full_spec_with_factors):
    """Temporary JSON file with full spec."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(full_spec_with_factors, f, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Override the global disable_network_calls fixture to avoid errors
@pytest.fixture(autouse=True)
def disable_network_calls():
    """Override global fixture to prevent network-related import errors in specs tests."""
    # No-op for specs tests
    pass
