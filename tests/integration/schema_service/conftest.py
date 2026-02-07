# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Schema Service Integration Tests (AGENT-FOUND-002)
======================================================================

Provides shared fixtures for integration testing the schema service,
including the foundation agent, API endpoints, and end-to-end SDK flows.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Async helper for Windows compatibility
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine synchronously. Windows-compatible."""
    if sys.platform == "win32":
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    else:
        return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def run_async():
    """Provide _run_async helper as a fixture."""
    return _run_async


# ---------------------------------------------------------------------------
# Schema Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def emissions_schema() -> Dict[str, Any]:
    """Emissions activity schema for integration tests."""
    return {
        "type": "object",
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "fuel_type": {"type": "string"},
            "quantity": {"type": "number", "minimum": 0},
            "unit": {"type": "string"},
            "co2e_kg": {"type": "number", "minimum": 0},
            "scope": {"type": "integer", "enum": [1, 2, 3]},
            "reporting_period": {"type": "string", "format": "date"},
        },
        "required": ["source_id", "fuel_type", "quantity", "unit", "co2e_kg"],
    }


@pytest.fixture
def inline_schema_with_extensions() -> Dict[str, Any]:
    """Inline schema with GreenLang $unit and $rules extensions."""
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
        },
        "required": ["energy_consumed", "co2_emissions"],
        "$rules": [
            {
                "name": "emissions_proportional",
                "expression": "co2_emissions <= energy_consumed * 2.0",
            }
        ],
    }


@pytest.fixture
def valid_emissions_payload() -> Dict[str, Any]:
    """Valid emissions payload."""
    return {
        "source_id": "FAC-001",
        "fuel_type": "diesel",
        "quantity": 1000.0,
        "unit": "liters",
        "co2e_kg": 2680.0,
        "scope": 1,
    }


@pytest.fixture
def invalid_emissions_payload() -> Dict[str, Any]:
    """Invalid emissions payload (missing required, type errors)."""
    return {
        "fuel_type": "diesel",
        "quantity": -50,
        # missing: source_id, unit, co2e_kg
        "scope": "not_a_number",
    }


@pytest.fixture
def batch_payloads(valid_emissions_payload, invalid_emissions_payload) -> List[Dict[str, Any]]:
    """Mixed batch of valid and invalid payloads."""
    return [
        valid_emissions_payload,
        {
            "source_id": "FAC-002",
            "fuel_type": "natural_gas",
            "quantity": 500.0,
            "unit": "m3",
            "co2e_kg": 965.0,
        },
        invalid_emissions_payload,
        {
            "source_id": "FAC-003",
            "fuel_type": "coal",
            "quantity": 200.0,
            "unit": "tonnes",
            "co2e_kg": 690.0,
            "scope": 2,
        },
    ]


@pytest.fixture
def compute_hash():
    """Utility to compute SHA-256 hash of data."""

    def _hash(data: Any) -> str:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    return _hash


# ---------------------------------------------------------------------------
# Override parent conftest fixtures that may interfere
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents (not needed for schema service tests)."""
    yield


import socket as _socket_module

# Save references to the REAL socket functions at module load time,
# BEFORE the session-scoped NetworkBlocker patches them.
_ORIGINAL_SOCKET = _socket_module.socket
_ORIGINAL_CREATE_CONNECTION = _socket_module.create_connection
_ORIGINAL_SOCKETPAIR = getattr(_socket_module, "socketpair", None)


@pytest.fixture
def restore_sockets():
    """
    Temporarily restore real sockets for FastAPI TestClient tests.

    The parent integration conftest blocks all sockets via a session-scoped
    autouse fixture. TestClient needs sockets internally (even though it
    communicates in-process). This fixture restores socket access for the
    duration of the test.
    """
    # Save the current (blocked) values
    blocked_socket = _socket_module.socket
    blocked_create_connection = _socket_module.create_connection
    blocked_socketpair = getattr(_socket_module, "socketpair", None)

    # Restore the real socket functions saved at module load time
    _socket_module.socket = _ORIGINAL_SOCKET
    _socket_module.create_connection = _ORIGINAL_CREATE_CONNECTION
    if _ORIGINAL_SOCKETPAIR is not None:
        _socket_module.socketpair = _ORIGINAL_SOCKETPAIR

    yield

    # Restore the blocked state
    _socket_module.socket = blocked_socket
    _socket_module.create_connection = blocked_create_connection
    if blocked_socketpair is not None:
        _socket_module.socketpair = blocked_socketpair
