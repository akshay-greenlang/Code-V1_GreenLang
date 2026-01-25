"""
Integration Test Configuration and Fixtures

This module provides shared fixtures and configuration for the
integration test suite.

Run with: pytest tests/integration/ -v
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import AsyncGenerator, Dict, Generator, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

try:
    from httpx import AsyncClient
except ImportError:
    AsyncClient = None


# =============================================================================
# Pytest Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Application Fixtures
# =============================================================================


@pytest.fixture
def app() -> FastAPI:
    """
    Create FastAPI application for testing.

    Returns:
        Configured FastAPI application instance
    """
    from fastapi import FastAPI

    app = FastAPI(
        title="GL-Agent-Factory Test",
        version="1.0.0",
    )

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "gl-agent-factory", "version": "1.0.0"}

    @app.get("/ready")
    def readiness_check():
        return {"ready": True, "checks": {"database": "ok", "redis": "ok"}}

    return app


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """
    Create synchronous test client.

    Args:
        app: FastAPI application

    Yields:
        TestClient instance for making test requests
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator:
    """
    Create async test client for async endpoint testing.

    Args:
        app: FastAPI application

    Yields:
        AsyncClient instance for making async test requests
    """
    if AsyncClient is None:
        pytest.skip("httpx not installed")
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        yield ac


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_carbon_input() -> Dict[str, Any]:
    """Create sample carbon emissions input."""
    return {
        "fuel_type": "natural_gas",
        "quantity": 1000.0,
        "unit": "m3",
        "region": "US",
        "scope": 1,
        "calculation_method": "location",
    }


@pytest.fixture
def sample_cbam_input() -> Dict[str, Any]:
    """Create sample CBAM input for steel import."""
    return {
        "cn_code": "72081000",
        "quantity_tonnes": 500.0,
        "country_of_origin": "CN",
        "installation_id": "CN-STEEL-FACTORY-001",
        "reporting_period": "Q1 2026",
    }


@pytest.fixture
def sample_scope3_input() -> Dict[str, Any]:
    """Create sample Scope 3 input."""
    return {
        "category": "cat_1_purchased_goods",
        "reporting_year": 2024,
        "spend_data": [
            {"category": "steel", "spend_usd": 500000, "supplier_name": "Steel Corp"},
            {"category": "aluminum", "spend_usd": 200000, "supplier_name": "Aluminum Inc"},
        ],
        "calculation_method": "spend_based",
        "revenue_usd": 10000000,
        "employees": 500,
    }


@pytest.fixture
def sample_agent_create_request() -> Dict[str, Any]:
    """Create sample agent creation request data."""
    return {
        "agent_id": "test/sample_agent_v1",
        "name": "Sample Test Agent",
        "version": "1.0.0",
        "description": "Sample agent for integration testing",
        "category": "test",
        "tags": ["test", "integration", "sample"],
        "entrypoint": "python://tests.sample_agent:SampleAgent",
        "deterministic": True,
        "inputs": {"field1": "string", "field2": "number"},
        "outputs": {"result": "number", "provenance_hash": "string"},
        "regulatory_frameworks": ["TEST"],
    }


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Create authorization headers for API requests."""
    return {
        "Authorization": "Bearer test-jwt-token",
        "X-Tenant-ID": "test-tenant-1",
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_database():
    """Create mock database session."""
    return MagicMock()


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    return mock


@pytest.fixture
def mock_execution_service():
    """Create mock execution service."""
    mock = AsyncMock()
    mock.execute_agent.return_value = {
        "execution_id": str(uuid4()),
        "status": "COMPLETED",
        "result": {"emissions_kgco2e": 1930.0},
        "provenance_hash": "a" * 64,
    }
    return mock


# =============================================================================
# Utility Functions
# =============================================================================


def assert_valid_provenance_hash(hash_value: str) -> None:
    """Assert that a provenance hash is valid SHA-256."""
    assert hash_value is not None, "Provenance hash should not be None"
    assert len(hash_value) == 64, f"SHA-256 hash should be 64 chars, got {len(hash_value)}"
    assert all(c in "0123456789abcdef" for c in hash_value.lower()), "Invalid hex characters"


def assert_recent_timestamp(dt: datetime, max_age_seconds: int = 60) -> None:
    """Assert that a datetime is recent (within max_age_seconds)."""
    now = datetime.utcnow()
    delta = abs((now - dt).total_seconds())
    assert delta < max_age_seconds, f"Timestamp too old: {delta}s > {max_age_seconds}s"


def assert_api_error_response(response_json: dict, expected_code: str) -> None:
    """Assert that an API error response has the expected structure."""
    assert "error" in response_json, "Response should contain 'error' key"
    assert "code" in response_json["error"], "Error should have 'code'"
    assert response_json["error"]["code"] == expected_code
