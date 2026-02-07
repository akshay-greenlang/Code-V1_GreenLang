"""
Integration test fixtures for security_operations module.

Provides test database, API clients, and service instances for integration testing.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient


# -----------------------------------------------------------------------------
# Database Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_database():
    """Create test database connection."""
    # Mock database for integration tests
    db = AsyncMock()
    db.execute = AsyncMock(return_value=None)
    db.fetch_one = AsyncMock(return_value=None)
    db.fetch_all = AsyncMock(return_value=[])

    yield db

    # Cleanup
    await db.execute("TRUNCATE incidents, alerts, threat_models, waf_rules CASCADE")


# -----------------------------------------------------------------------------
# API Client Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def integration_app() -> FastAPI:
    """Create FastAPI app with all security operations routers."""
    from fastapi import FastAPI

    app = FastAPI(title="Security Operations Integration Tests")

    # Import and include all routers
    # Note: In actual implementation, import from real modules
    # from greenlang.infrastructure.incident_response.api.incident_routes import router as incident_router
    # app.include_router(incident_router, prefix="/api/v1/incidents")

    return app


@pytest.fixture
def api_client(integration_app) -> TestClient:
    """Create synchronous API test client."""
    return TestClient(integration_app)


@pytest_asyncio.fixture
async def async_api_client(integration_app) -> AsyncGenerator[AsyncClient, None]:
    """Create async API test client."""
    async with AsyncClient(app=integration_app, base_url="http://test") as client:
        yield client


# -----------------------------------------------------------------------------
# Authentication Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def admin_auth_headers() -> Dict[str, str]:
    """Create admin authentication headers."""
    return {
        "Authorization": "Bearer test-admin-integration-token",
        "X-User-Id": "integration-admin",
        "X-User-Roles": "admin,security-admin,incident-responder,waf-admin",
    }


@pytest.fixture
def analyst_auth_headers() -> Dict[str, str]:
    """Create analyst authentication headers."""
    return {
        "Authorization": "Bearer test-analyst-integration-token",
        "X-User-Id": "integration-analyst",
        "X-User-Roles": "security-analyst,viewer",
    }


# -----------------------------------------------------------------------------
# Service Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_notification_service():
    """Create mock notification service."""
    service = AsyncMock()
    service.send_slack.return_value = {"ok": True}
    service.send_pagerduty.return_value = {"ok": True}
    service.send_email.return_value = {"ok": True}
    return service


@pytest.fixture
def mock_metrics_service():
    """Create mock metrics service."""
    service = MagicMock()
    service.increment = MagicMock()
    service.gauge = MagicMock()
    service.histogram = MagicMock()
    return service


# -----------------------------------------------------------------------------
# Test Data Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def integration_alert_data() -> Dict[str, Any]:
    """Create alert data for integration tests."""
    return {
        "alert_id": str(uuid4()),
        "title": "Integration Test Alert",
        "description": "Alert created for integration testing",
        "severity": "high",
        "source": "prometheus",
        "timestamp": datetime.utcnow().isoformat(),
        "labels": {
            "alertname": "IntegrationTest",
            "instance": "test-node",
            "job": "integration-tests",
        },
    }


@pytest.fixture
def integration_incident_data(integration_alert_data) -> Dict[str, Any]:
    """Create incident data for integration tests."""
    return {
        "title": "Integration Test Incident",
        "description": "Incident created for integration testing",
        "incident_type": "infrastructure",
        "escalation_level": "P2",
        "alerts": [integration_alert_data],
        "tags": ["integration-test", "automated"],
    }


@pytest.fixture
def integration_threat_model_data() -> Dict[str, Any]:
    """Create threat model data for integration tests."""
    return {
        "name": "Integration Test Threat Model",
        "description": "Threat model for integration testing",
        "scope": "Integration test scope",
        "components": [
            {
                "name": "test-api",
                "component_type": "service",
                "description": "Test API service",
                "data_classification": "confidential",
            }
        ],
        "data_flows": [],
        "trust_boundaries": [],
    }


@pytest.fixture
def integration_waf_rule_data() -> Dict[str, Any]:
    """Create WAF rule data for integration tests."""
    return {
        "name": "Integration Test Rate Limit",
        "description": "Rate limit rule for integration testing",
        "rule_type": "rate_limit",
        "action": "block",
        "priority": "medium",
        "parameters": {
            "limit": 100,
            "window_seconds": 60,
        },
        "enabled": True,
    }
