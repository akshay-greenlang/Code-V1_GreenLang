"""
Test Configuration and Fixtures

This module provides shared fixtures and configuration for the
Agent Registry test suite.

Fixtures:
- service: In-memory AgentRegistryService
- app: FastAPI test application
- client: TestClient for API testing
- sample_agents: Pre-populated agent data

Run with: pytest backend/registry/tests/ -v
"""

import pytest
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from backend.registry.api import router
from backend.registry.service import AgentRegistryService
from backend.registry.models import AgentRecord, AgentVersion, AgentStatus


# =============================================================================
# Pytest Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Service Fixtures
# =============================================================================


@pytest.fixture
def service() -> AgentRegistryService:
    """
    Create in-memory AgentRegistryService for testing.

    Returns:
        AgentRegistryService instance with no database connection
    """
    return AgentRegistryService(session=None)


@pytest.fixture
async def populated_service() -> AgentRegistryService:
    """
    Create service with pre-populated test data.

    Returns:
        AgentRegistryService with sample agents
    """
    svc = AgentRegistryService(session=None)

    # Create sample agents
    await svc.create_agent(
        name="carbon-calculator",
        version="1.0.0",
        description="Calculate carbon emissions for various activities",
        category="emissions",
        author="greenlang",
        tags=["carbon", "ghg", "emissions"],
        regulatory_frameworks=["GHG Protocol"],
    )

    await svc.create_agent(
        name="cbam-compliance-checker",
        version="2.0.0",
        description="Check CBAM compliance for EU border carbon adjustments",
        category="regulatory",
        author="greenlang",
        tags=["cbam", "eu", "compliance", "border"],
        regulatory_frameworks=["CBAM"],
    )

    await svc.create_agent(
        name="csrd-reporter",
        version="1.5.0",
        description="Generate CSRD sustainability reports",
        category="regulatory",
        author="compliance-team",
        tags=["csrd", "sustainability", "reporting"],
        regulatory_frameworks=["CSRD", "ESRS"],
    )

    return svc


# =============================================================================
# FastAPI Fixtures
# =============================================================================


@pytest.fixture
def app() -> FastAPI:
    """
    Create FastAPI application for testing.

    Returns:
        FastAPI application instance
    """
    test_app = FastAPI(
        title="Agent Registry Test",
        version="1.0.0",
    )
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """
    Create synchronous test client.

    Args:
        app: FastAPI application

    Yields:
        TestClient instance
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    Create async test client.

    Args:
        app: FastAPI application

    Yields:
        AsyncClient instance
    """
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_data() -> dict:
    """
    Sample data for creating an agent.

    Returns:
        Dictionary with agent creation data
    """
    return {
        "name": "sample-agent",
        "version": "1.0.0",
        "description": "Sample agent for testing",
        "category": "test",
        "author": "test-user",
        "pack_yaml": {
            "name": "sample-agent",
            "version": "1.0.0",
            "entrypoint": "agent.py",
        },
        "tags": ["test", "sample"],
        "regulatory_frameworks": ["TEST"],
    }


@pytest.fixture
def sample_version_data() -> dict:
    """
    Sample data for creating a version.

    Returns:
        Dictionary with version creation data
    """
    return {
        "version": "2.0.0",
        "changelog": "Major update with breaking changes",
        "breaking_changes": True,
        "release_notes": "This version includes significant improvements.",
    }


@pytest.fixture
def sample_publish_data() -> dict:
    """
    Sample data for publishing an agent.

    Returns:
        Dictionary with publish request data
    """
    return {
        "version": "1.0.0",
        "release_notes": "Initial public release",
        "certifications": ["CBAM", "CSRD"],
    }


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def sample_agent() -> AgentRecord:
    """
    Create sample AgentRecord instance.

    Returns:
        AgentRecord instance
    """
    return AgentRecord(
        name="fixture-agent",
        version="1.0.0",
        description="Agent created from fixture",
        category="test",
        author="fixture-author",
        tags=["fixture", "test"],
    )


@pytest.fixture
def sample_version(sample_agent: AgentRecord) -> AgentVersion:
    """
    Create sample AgentVersion instance.

    Args:
        sample_agent: Parent agent

    Returns:
        AgentVersion instance
    """
    return AgentVersion(
        agent_id=sample_agent.id,
        version="1.0.0",
        changelog="Initial version",
        breaking_changes=False,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def assert_agent_response(response: dict, expected: dict) -> None:
    """
    Assert agent response matches expected values.

    Args:
        response: API response dictionary
        expected: Expected values dictionary
    """
    for key, value in expected.items():
        assert response.get(key) == value, f"Mismatch for {key}: {response.get(key)} != {value}"


def assert_datetime_recent(dt: datetime, seconds: int = 60) -> None:
    """
    Assert datetime is recent (within specified seconds).

    Args:
        dt: Datetime to check
        seconds: Maximum age in seconds
    """
    now = datetime.utcnow()
    delta = abs((now - dt).total_seconds())
    assert delta < seconds, f"Datetime not recent: {delta} seconds old"
