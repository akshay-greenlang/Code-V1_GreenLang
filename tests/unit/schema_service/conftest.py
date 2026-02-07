# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Schema Service Unit Tests (AGENT-FOUND-002)
================================================================

Provides shared fixtures for testing the schema service metrics,
setup facade, and related components.

All tests are self-contained with no external dependencies.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Schema Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_schema() -> Dict[str, Any]:
    """A simple JSON Schema dict for testing."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:greenlang:test:emissions",
        "title": "Emissions Record",
        "description": "Test schema for emissions data",
        "type": "object",
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "fuel_type": {"type": "string", "enum": ["diesel", "natural_gas", "coal"]},
            "quantity": {"type": "number", "minimum": 0},
            "unit": {"type": "string"},
            "co2e_kg": {"type": "number", "minimum": 0},
            "scope": {"type": "integer", "enum": [1, 2, 3]},
        },
        "required": ["source_id", "fuel_type", "quantity", "unit", "co2e_kg"],
    }


@pytest.fixture
def sample_payload_valid() -> Dict[str, Any]:
    """Valid payload matching sample_schema."""
    return {
        "source_id": "FAC-001",
        "fuel_type": "diesel",
        "quantity": 1000.0,
        "unit": "liters",
        "co2e_kg": 2680.0,
        "scope": 1,
    }


@pytest.fixture
def sample_payload_invalid() -> Dict[str, Any]:
    """Invalid payload with type errors and missing fields."""
    return {
        "source_id": "",  # minLength violation
        "fuel_type": "unknown_fuel",  # enum violation
        "quantity": -50,  # minimum violation
        # missing "unit" (required)
        # missing "co2e_kg" (required)
        "scope": "not_an_integer",  # type error
    }


@pytest.fixture
def sample_inline_schema() -> Dict[str, Any]:
    """Inline schema with GreenLang extensions ($unit, $rules)."""
    return {
        "type": "object",
        "properties": {
            "energy_consumed": {
                "type": "number",
                "minimum": 0,
                "$unit": "kWh",
            },
            "co2_emissions": {
                "type": "number",
                "minimum": 0,
                "$unit": "kgCO2e",
            },
            "reporting_period": {
                "type": "string",
                "format": "date",
            },
        },
        "required": ["energy_consumed", "co2_emissions"],
        "$rules": [
            {
                "name": "emissions_proportional",
                "description": "CO2 emissions should be proportional to energy consumed",
                "expression": "co2_emissions <= energy_consumed * 2.0",
            }
        ],
    }


@pytest.fixture
def mock_registry():
    """Mock schema registry."""
    registry = MagicMock()
    registry.resolve.return_value = MagicMock(
        content={"type": "object", "properties": {}},
        schema_id="mock/schema",
        version="1.0.0",
    )
    registry.list_schemas.return_value = [
        {"schema_id": "emissions/activity", "version": "1.3.0"},
        {"schema_id": "emissions/facility", "version": "2.0.0"},
    ]
    return registry


@pytest.fixture
def mock_prometheus(monkeypatch):
    """Monkey-patched prometheus_client (or mock if not available)."""
    mock_counter = MagicMock()
    mock_counter.labels.return_value = mock_counter
    mock_histogram = MagicMock()
    mock_histogram.labels.return_value = mock_histogram
    mock_gauge = MagicMock()
    mock_gauge.labels.return_value = mock_gauge

    mock_prom = MagicMock()
    mock_prom.Counter.return_value = mock_counter
    mock_prom.Histogram.return_value = mock_histogram
    mock_prom.Gauge.return_value = mock_gauge
    mock_prom.generate_latest.return_value = b"# HELP test_metric\n# TYPE test_metric counter\n"

    return mock_prom


@pytest.fixture
def schema_service_config() -> Dict[str, Any]:
    """Default configuration for schema service."""
    return {
        "service_name": "test-schema-service",
        "environment": "test",
        "log_level": "DEBUG",
        "metrics_enabled": True,
        "cache_enabled": True,
        "cache_ttl_seconds": 300,
        "max_payload_bytes": 1_048_576,
        "max_batch_items": 1000,
        "default_profile": "standard",
        "enable_api": True,
    }


@pytest.fixture
def multiple_payloads_mixed(sample_payload_valid) -> List[Dict[str, Any]]:
    """Multiple payloads, some valid and some invalid, for batch tests."""
    return [
        sample_payload_valid,
        {
            "source_id": "FAC-002",
            "fuel_type": "natural_gas",
            "quantity": 500.0,
            "unit": "m3",
            "co2e_kg": 965.0,
        },
        {
            # Invalid: missing required fields
            "fuel_type": "coal",
            "quantity": -10,
        },
        {
            "source_id": "FAC-004",
            "fuel_type": "diesel",
            "quantity": 250.0,
            "unit": "liters",
            "co2e_kg": 670.0,
            "scope": 2,
        },
    ]
