# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for API Integration Tests.

This module provides shared fixtures for testing the GreenLang API layer,
including mock HTTP clients, authentication fixtures, and API response helpers.
"""

import pytest
import asyncio
import json
import hashlib
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path

# Add project paths
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = "http://localhost:8000/api/v1"
JWT_SECRET = "test-secret-key-for-testing-only"
JWT_ALGORITHM = "HS256"


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture
def jwt_secret():
    """Provide JWT secret for tests."""
    return JWT_SECRET


@pytest.fixture
def generate_jwt_token(jwt_secret):
    """Generate JWT tokens for testing."""
    def _generate(
        user_id: str = "user-001",
        tenant_id: str = "tenant-001",
        roles: List[str] = None,
        expires_in_hours: int = 1,
        custom_claims: Dict[str, Any] = None,
    ) -> str:
        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "roles": roles or ["user"],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=expires_in_hours),
        }
        if custom_claims:
            payload.update(custom_claims)
        return jwt.encode(payload, jwt_secret, algorithm=JWT_ALGORITHM)

    return _generate


@pytest.fixture
def valid_auth_headers(generate_jwt_token):
    """Generate valid authentication headers."""
    token = generate_jwt_token()
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(generate_jwt_token):
    """Generate admin authentication headers."""
    token = generate_jwt_token(roles=["admin", "user"])
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def expired_auth_headers(jwt_secret):
    """Generate expired authentication headers."""
    payload = {
        "sub": "user-001",
        "tenant_id": "tenant-001",
        "roles": ["user"],
        "iat": datetime.utcnow() - timedelta(hours=2),
        "exp": datetime.utcnow() - timedelta(hours=1),
    }
    token = jwt.encode(payload, jwt_secret, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api_key_headers():
    """Generate API key authentication headers."""
    return {
        "X-API-Key": "gl-api-key-test-12345",
        "X-Tenant-ID": "tenant-001",
    }


# =============================================================================
# Mock HTTP Client
# =============================================================================

@pytest.fixture
def mock_http_client():
    """Create mock HTTP client for API testing."""
    class MockHTTPClient:
        def __init__(self):
            self.requests = []
            self.responses = {}
            self._default_response = {"status": "ok"}

        def set_response(self, method: str, path: str, response: Dict[str, Any], status_code: int = 200):
            """Set mock response for a specific endpoint."""
            key = f"{method.upper()}:{path}"
            self.responses[key] = (response, status_code)

        async def request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
            """Make a mock HTTP request."""
            request_record = {
                "method": method.upper(),
                "path": path,
                "headers": kwargs.get("headers", {}),
                "json": kwargs.get("json"),
                "params": kwargs.get("params"),
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.requests.append(request_record)

            key = f"{method.upper()}:{path}"
            if key in self.responses:
                response, status_code = self.responses[key]
                return {"data": response, "status_code": status_code}

            return {"data": self._default_response, "status_code": 200}

        async def get(self, path: str, **kwargs) -> Dict[str, Any]:
            return await self.request("GET", path, **kwargs)

        async def post(self, path: str, **kwargs) -> Dict[str, Any]:
            return await self.request("POST", path, **kwargs)

        async def put(self, path: str, **kwargs) -> Dict[str, Any]:
            return await self.request("PUT", path, **kwargs)

        async def delete(self, path: str, **kwargs) -> Dict[str, Any]:
            return await self.request("DELETE", path, **kwargs)

        async def patch(self, path: str, **kwargs) -> Dict[str, Any]:
            return await self.request("PATCH", path, **kwargs)

        def get_requests(self, method: str = None, path: str = None) -> List[Dict]:
            """Get recorded requests, optionally filtered."""
            filtered = self.requests
            if method:
                filtered = [r for r in filtered if r["method"] == method.upper()]
            if path:
                filtered = [r for r in filtered if r["path"] == path]
            return filtered

        def clear(self):
            """Clear recorded requests."""
            self.requests = []

    return MockHTTPClient()


# =============================================================================
# Agent and Pipeline Fixtures
# =============================================================================

@pytest.fixture
def sample_agent_definition():
    """Sample agent definition for lifecycle tests."""
    return {
        "name": "fuel-emissions-calculator",
        "version": "1.0.0",
        "description": "Calculates emissions from fuel consumption",
        "type": "calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "fuel_type": {"type": "string", "enum": ["diesel", "natural_gas", "coal"]},
                "quantity": {"type": "number", "minimum": 0},
                "unit": {"type": "string", "enum": ["L", "MJ", "kg"]},
            },
            "required": ["fuel_type", "quantity", "unit"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "emissions_kgco2e": {"type": "number"},
                "provenance_hash": {"type": "string"},
            },
        },
        "tags": ["emissions", "fuel", "scope1"],
    }


@pytest.fixture
def sample_pipeline_definition():
    """Sample pipeline definition for execution tests."""
    return {
        "name": "carbon-to-cbam-pipeline",
        "version": "1.0.0",
        "description": "Full carbon calculation to CBAM reporting pipeline",
        "agents": [
            {
                "id": "data-parser",
                "name": "data-parser-agent",
                "order": 1,
            },
            {
                "id": "fuel-calculator",
                "name": "fuel-emissions-calculator",
                "order": 2,
                "depends_on": ["data-parser"],
            },
            {
                "id": "cbam-reporter",
                "name": "cbam-reporting-agent",
                "order": 3,
                "depends_on": ["fuel-calculator"],
            },
        ],
        "error_handling": {
            "on_failure": "halt",
            "retry_count": 3,
            "retry_delay_ms": 100,
        },
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch data for batch processing tests."""
    return [
        {"fuel_type": "diesel", "quantity": 1000, "unit": "L", "facility_id": "FAC-001"},
        {"fuel_type": "natural_gas", "quantity": 5000, "unit": "MJ", "facility_id": "FAC-002"},
        {"fuel_type": "diesel", "quantity": 2500, "unit": "L", "facility_id": "FAC-001"},
        {"fuel_type": "coal", "quantity": 800, "unit": "kg", "facility_id": "FAC-003"},
        {"fuel_type": "natural_gas", "quantity": 3000, "unit": "MJ", "facility_id": "FAC-002"},
    ]


# =============================================================================
# Rate Limiting Fixtures
# =============================================================================

@pytest.fixture
def rate_limiter():
    """Create rate limiter for testing."""
    class RateLimiter:
        def __init__(self, limit: int = 100, window_seconds: int = 60):
            self.limit = limit
            self.window_seconds = window_seconds
            self.requests = {}

        def check(self, key: str) -> tuple:
            """Check if request is allowed. Returns (allowed, remaining, reset_time)."""
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=self.window_seconds)

            if key not in self.requests:
                self.requests[key] = []

            # Clean old requests
            self.requests[key] = [
                ts for ts in self.requests[key]
                if ts > window_start
            ]

            current_count = len(self.requests[key])
            remaining = max(0, self.limit - current_count)

            if current_count >= self.limit:
                reset_time = self.requests[key][0] + timedelta(seconds=self.window_seconds)
                return False, 0, reset_time

            self.requests[key].append(now)
            return True, remaining - 1, None

        def reset(self, key: str = None):
            """Reset rate limiter."""
            if key:
                self.requests.pop(key, None)
            else:
                self.requests = {}

    return RateLimiter(limit=100, window_seconds=60)


# =============================================================================
# Webhook Fixtures
# =============================================================================

@pytest.fixture
def webhook_registry():
    """Create webhook registry for testing."""
    class WebhookRegistry:
        def __init__(self):
            self.webhooks = {}
            self.deliveries = []

        def register(self, webhook_id: str, url: str, events: List[str], secret: str = None):
            """Register a webhook."""
            self.webhooks[webhook_id] = {
                "id": webhook_id,
                "url": url,
                "events": events,
                "secret": secret,
                "created_at": datetime.utcnow().isoformat(),
                "active": True,
            }
            return self.webhooks[webhook_id]

        def unregister(self, webhook_id: str):
            """Unregister a webhook."""
            if webhook_id in self.webhooks:
                del self.webhooks[webhook_id]
                return True
            return False

        async def trigger(self, event: str, payload: Dict[str, Any]):
            """Trigger webhooks for an event."""
            results = []
            for webhook_id, webhook in self.webhooks.items():
                if event in webhook["events"] and webhook["active"]:
                    delivery = {
                        "id": str(uuid.uuid4()),
                        "webhook_id": webhook_id,
                        "event": event,
                        "payload": payload,
                        "url": webhook["url"],
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "delivered",
                    }
                    # Add signature if secret is set
                    if webhook["secret"]:
                        delivery["signature"] = self._sign_payload(payload, webhook["secret"])
                    self.deliveries.append(delivery)
                    results.append(delivery)
            return results

        def _sign_payload(self, payload: Dict[str, Any], secret: str) -> str:
            """Sign webhook payload."""
            payload_bytes = json.dumps(payload, sort_keys=True).encode()
            return hashlib.sha256(payload_bytes + secret.encode()).hexdigest()

        def get_deliveries(self, webhook_id: str = None) -> List[Dict]:
            """Get webhook deliveries."""
            if webhook_id:
                return [d for d in self.deliveries if d["webhook_id"] == webhook_id]
            return self.deliveries

    return WebhookRegistry()


# =============================================================================
# Response Assertion Helpers
# =============================================================================

@pytest.fixture
def assert_api_response():
    """Provide API response assertion helpers."""
    class APIResponseAssertions:
        @staticmethod
        def assert_success(response: Dict, expected_status: int = 200):
            """Assert successful API response."""
            assert response["status_code"] == expected_status
            assert "data" in response

        @staticmethod
        def assert_error(response: Dict, expected_status: int, error_type: str = None):
            """Assert error API response."""
            assert response["status_code"] == expected_status
            if error_type:
                assert response["data"].get("error", {}).get("type") == error_type

        @staticmethod
        def assert_paginated(response: Dict, expected_page_size: int = None):
            """Assert paginated API response."""
            data = response["data"]
            assert "items" in data
            assert "pagination" in data
            assert "total" in data["pagination"]
            assert "page" in data["pagination"]
            assert "page_size" in data["pagination"]
            if expected_page_size:
                assert len(data["items"]) <= expected_page_size

        @staticmethod
        def assert_has_provenance(response: Dict):
            """Assert response has provenance information."""
            data = response["data"]
            assert "provenance_hash" in data or "provenance" in data

    return APIResponseAssertions()


# =============================================================================
# Test Data Generators
# =============================================================================

@pytest.fixture
def generate_agent_id():
    """Generate unique agent IDs."""
    counter = 0

    def _generate(prefix: str = "agent"):
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter:05d}"

    return _generate


@pytest.fixture
def generate_execution_id():
    """Generate unique execution IDs."""
    def _generate():
        return str(uuid.uuid4())

    return _generate
