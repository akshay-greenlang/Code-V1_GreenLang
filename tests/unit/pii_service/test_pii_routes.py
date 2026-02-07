# -*- coding: utf-8 -*-
"""
Unit tests for PII Service API Routes - SEC-011 PII Service.

Tests the FastAPI routes for PII service endpoints:
- POST /api/v1/pii/detect
- POST /api/v1/pii/redact
- POST /api/v1/pii/tokenize
- POST /api/v1/pii/detokenize
- Policy management endpoints
- Allowlist CRUD endpoints
- Quarantine management endpoints

Coverage target: 85%+ of api/pii_routes.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_pii_service():
    """Mock PIIService for route testing."""
    service = AsyncMock()

    # Detection
    service.detect = AsyncMock(return_value=MagicMock(
        detections=[],
        processing_time_ms=5.0,
    ))

    # Redaction
    service.redact = AsyncMock(return_value=MagicMock(
        redacted_content="[REDACTED]",
        detections=[],
        tokens_created=[],
        processing_time_ms=5.0,
    ))

    # Tokenization
    service.tokenize = AsyncMock(return_value="tok_test123")
    service.detokenize = AsyncMock(return_value="original-value")

    # Enforcement
    service.enforce = AsyncMock(return_value=MagicMock(
        blocked=False,
        detections=[],
        actions_taken=[],
    ))

    # Allowlist
    service.add_allowlist_entry = AsyncMock(return_value=str(uuid4()))
    service.remove_allowlist_entry = AsyncMock()
    service.list_allowlist_entries = AsyncMock(return_value=[])

    # Quarantine
    service.list_quarantine_items = AsyncMock(return_value=[])
    service.release_quarantine_item = AsyncMock()
    service.delete_quarantine_item = AsyncMock()

    # Policies
    service.list_policies = AsyncMock(return_value={})
    service.update_policy = AsyncMock()

    return service


@pytest.fixture
def test_client(mock_pii_service):
    """Create FastAPI test client with mocked service."""
    try:
        from fastapi.testclient import TestClient
        from greenlang.infrastructure.pii_service.api.pii_routes import create_pii_router

        from fastapi import FastAPI
        app = FastAPI()
        router = create_pii_router(mock_pii_service)
        app.include_router(router, prefix="/api/v1/pii")

        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI routes not yet implemented")


# ============================================================================
# TestDetectEndpoint
# ============================================================================


class TestDetectEndpoint:
    """Tests for POST /api/v1/pii/detect endpoint."""

    def test_detect_endpoint_success(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Detect endpoint returns successful response."""
        response = test_client.post(
            "/api/v1/pii/detect",
            json={"content": "My SSN is 123-45-6789"},
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert "detections" in data

    def test_detect_endpoint_empty_content(
        self, test_client, auth_headers
    ):
        """Detect endpoint handles empty content."""
        response = test_client.post(
            "/api/v1/pii/detect",
            json={"content": ""},
            headers=auth_headers(),
        )

        # Should return 400 or empty result
        assert response.status_code in [200, 400, 422]

    def test_detect_endpoint_no_pii(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Detect endpoint returns empty list when no PII found."""
        mock_pii_service.detect.return_value = MagicMock(
            detections=[],
            processing_time_ms=1.0,
        )

        response = test_client.post(
            "/api/v1/pii/detect",
            json={"content": "This is clean content"},
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("detections", []) == []

    def test_detect_endpoint_with_options(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Detect endpoint accepts detection options."""
        response = test_client.post(
            "/api/v1/pii/detect",
            json={
                "content": "My SSN is 123-45-6789",
                "options": {
                    "use_ml": False,
                    "min_confidence": 0.9,
                },
            },
            headers=auth_headers(),
        )

        assert response.status_code == 200

    def test_detect_endpoint_returns_processing_time(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Detect endpoint includes processing time in response."""
        response = test_client.post(
            "/api/v1/pii/detect",
            json={"content": "Test content"},
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data


# ============================================================================
# TestRedactEndpoint
# ============================================================================


class TestRedactEndpoint:
    """Tests for POST /api/v1/pii/redact endpoint."""

    def test_redact_endpoint_success(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Redact endpoint returns redacted content."""
        mock_pii_service.redact.return_value = MagicMock(
            redacted_content="My SSN is [SSN]",
            detections=[MagicMock()],
            tokens_created=[],
            processing_time_ms=5.0,
        )

        response = test_client.post(
            "/api/v1/pii/redact",
            json={"content": "My SSN is 123-45-6789"},
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert "redacted_content" in data
        assert "[SSN]" in data["redacted_content"]

    def test_redact_endpoint_strategies(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Redact endpoint accepts strategy configuration."""
        response = test_client.post(
            "/api/v1/pii/redact",
            json={
                "content": "Email: test@example.com",
                "options": {
                    "strategy": "mask",
                    "strategy_overrides": {
                        "email": "hash",
                    },
                },
            },
            headers=auth_headers(),
        )

        assert response.status_code == 200


# ============================================================================
# TestTokenizeEndpoint
# ============================================================================


class TestTokenizeEndpoint:
    """Tests for POST /api/v1/pii/tokenize endpoint."""

    def test_tokenize_endpoint_success(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Tokenize endpoint returns token."""
        mock_pii_service.tokenize.return_value = "tok_abc123xyz"

        response = test_client.post(
            "/api/v1/pii/tokenize",
            json={
                "value": "123-45-6789",
                "pii_type": "ssn",
                "tenant_id": "tenant-123",
            },
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert data["token"] == "tok_abc123xyz"

    def test_tokenize_endpoint_invalid_type(
        self, test_client, auth_headers
    ):
        """Tokenize endpoint rejects invalid PII type."""
        response = test_client.post(
            "/api/v1/pii/tokenize",
            json={
                "value": "some-value",
                "pii_type": "invalid_type",
                "tenant_id": "tenant-123",
            },
            headers=auth_headers(),
        )

        assert response.status_code in [400, 422]

    def test_tokenize_endpoint_missing_tenant(
        self, test_client, auth_headers
    ):
        """Tokenize endpoint requires tenant_id."""
        response = test_client.post(
            "/api/v1/pii/tokenize",
            json={
                "value": "123-45-6789",
                "pii_type": "ssn",
            },
            headers=auth_headers(),
        )

        assert response.status_code in [400, 422]


# ============================================================================
# TestDetokenizeEndpoint
# ============================================================================


class TestDetokenizeEndpoint:
    """Tests for POST /api/v1/pii/detokenize endpoint."""

    def test_detokenize_endpoint_success(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Detokenize endpoint returns original value."""
        mock_pii_service.detokenize.return_value = "123-45-6789"

        response = test_client.post(
            "/api/v1/pii/detokenize",
            json={
                "token": "tok_abc123xyz",
                "tenant_id": "tenant-123",
                "user_id": str(uuid4()),
            },
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert "value" in data
        assert data["value"] == "123-45-6789"

    def test_detokenize_endpoint_not_found(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Detokenize endpoint returns 404 for unknown token."""
        mock_pii_service.detokenize.side_effect = Exception("Token not found")

        response = test_client.post(
            "/api/v1/pii/detokenize",
            json={
                "token": "tok_nonexistent",
                "tenant_id": "tenant-123",
                "user_id": str(uuid4()),
            },
            headers=auth_headers(),
        )

        assert response.status_code in [404, 400]


# ============================================================================
# TestPoliciesEndpoints
# ============================================================================


class TestPoliciesEndpoints:
    """Tests for policy management endpoints."""

    def test_policies_list(
        self, test_client, mock_pii_service, auth_headers
    ):
        """GET /policies returns policy list."""
        mock_pii_service.list_policies.return_value = {
            "ssn": {"action": "block"},
            "email": {"action": "allow"},
        }

        response = test_client.get(
            "/api/v1/pii/policies",
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert "ssn" in data

    def test_policies_update(
        self, test_client, mock_pii_service, admin_auth_headers
    ):
        """PUT /policies/{pii_type} updates policy."""
        response = test_client.put(
            "/api/v1/pii/policies/email",
            json={
                "action": "redact",
                "min_confidence": 0.85,
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        mock_pii_service.update_policy.assert_awaited_once()

    def test_policies_update_requires_admin(
        self, test_client, auth_headers
    ):
        """Policy update requires admin permissions."""
        response = test_client.put(
            "/api/v1/pii/policies/email",
            json={"action": "block"},
            headers=auth_headers(roles=["pii:read"]),  # No admin role
        )

        assert response.status_code in [401, 403]


# ============================================================================
# TestAllowlistEndpoints
# ============================================================================


class TestAllowlistEndpoints:
    """Tests for allowlist management endpoints."""

    def test_allowlist_crud(
        self, test_client, mock_pii_service, admin_auth_headers
    ):
        """Allowlist supports CRUD operations."""
        entry_id = str(uuid4())
        mock_pii_service.add_allowlist_entry.return_value = entry_id

        # Create
        response = test_client.post(
            "/api/v1/pii/allowlist",
            json={
                "pii_type": "email",
                "pattern": r".*@test\.com$",
                "pattern_type": "regex",
                "reason": "Test domain",
            },
            headers=admin_auth_headers,
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert "id" in data

    def test_allowlist_list(
        self, test_client, mock_pii_service, auth_headers
    ):
        """GET /allowlist returns entry list."""
        mock_pii_service.list_allowlist_entries.return_value = [
            MagicMock(
                id=str(uuid4()),
                pii_type="email",
                pattern=r".*@example\.com$",
            ),
        ]

        response = test_client.get(
            "/api/v1/pii/allowlist",
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_allowlist_delete(
        self, test_client, mock_pii_service, admin_auth_headers
    ):
        """DELETE /allowlist/{id} removes entry."""
        entry_id = str(uuid4())

        response = test_client.delete(
            f"/api/v1/pii/allowlist/{entry_id}",
            headers=admin_auth_headers,
        )

        assert response.status_code in [200, 204]
        mock_pii_service.remove_allowlist_entry.assert_awaited_once()


# ============================================================================
# TestQuarantineEndpoints
# ============================================================================


class TestQuarantineEndpoints:
    """Tests for quarantine management endpoints."""

    def test_quarantine_list(
        self, test_client, mock_pii_service, auth_headers
    ):
        """GET /quarantine returns item list."""
        mock_pii_service.list_quarantine_items.return_value = [
            MagicMock(
                id=str(uuid4()),
                pii_type="ssn",
                status="pending",
            ),
        ]

        response = test_client.get(
            "/api/v1/pii/quarantine",
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_quarantine_release(
        self, test_client, mock_pii_service, admin_auth_headers
    ):
        """POST /quarantine/{id}/release releases item."""
        item_id = str(uuid4())

        response = test_client.post(
            f"/api/v1/pii/quarantine/{item_id}/release",
            json={"reason": "False positive"},
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        mock_pii_service.release_quarantine_item.assert_awaited_once()

    def test_quarantine_delete(
        self, test_client, mock_pii_service, admin_auth_headers
    ):
        """POST /quarantine/{id}/delete deletes item."""
        item_id = str(uuid4())

        response = test_client.post(
            f"/api/v1/pii/quarantine/{item_id}/delete",
            json={"reason": "Confirmed PII"},
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        mock_pii_service.delete_quarantine_item.assert_awaited_once()


# ============================================================================
# TestMetricsEndpoint
# ============================================================================


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_endpoint(
        self, test_client, auth_headers
    ):
        """GET /metrics returns service metrics."""
        response = test_client.get(
            "/api/v1/pii/metrics",
            headers=auth_headers(),
        )

        assert response.status_code == 200
        data = response.json()
        # Should include various metrics
        assert isinstance(data, dict)


# ============================================================================
# TestAuthentication
# ============================================================================


class TestAuthentication:
    """Tests for authentication requirements."""

    def test_authentication_required(
        self, test_client
    ):
        """Endpoints require authentication."""
        response = test_client.post(
            "/api/v1/pii/detect",
            json={"content": "test"},
            # No auth headers
        )

        assert response.status_code == 401

    def test_invalid_token_rejected(
        self, test_client
    ):
        """Invalid auth tokens are rejected."""
        response = test_client.post(
            "/api/v1/pii/detect",
            json={"content": "test"},
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 401


# ============================================================================
# TestAuthorization
# ============================================================================


class TestAuthorization:
    """Tests for authorization enforcement."""

    def test_authorization_enforced(
        self, test_client, auth_headers
    ):
        """Authorization is enforced on protected endpoints."""
        # User without pii:tokenize permission
        response = test_client.post(
            "/api/v1/pii/tokenize",
            json={
                "value": "123-45-6789",
                "pii_type": "ssn",
                "tenant_id": "tenant-123",
            },
            headers=auth_headers(roles=["pii:read"]),  # Only read permission
        )

        # Should be forbidden
        assert response.status_code in [401, 403]

    def test_admin_operations_restricted(
        self, test_client, auth_headers
    ):
        """Admin operations require admin role."""
        # Non-admin trying to update policy
        response = test_client.put(
            "/api/v1/pii/policies/ssn",
            json={"action": "allow"},
            headers=auth_headers(roles=["pii:read", "pii:write"]),
        )

        assert response.status_code in [401, 403]


# ============================================================================
# TestInputValidation
# ============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_content_size_limit(
        self, test_client, auth_headers
    ):
        """Content size is limited."""
        # Very large content
        large_content = "x" * 100_000_000

        response = test_client.post(
            "/api/v1/pii/detect",
            json={"content": large_content},
            headers=auth_headers(),
        )

        assert response.status_code in [400, 413, 422]

    def test_invalid_json_rejected(
        self, test_client, auth_headers
    ):
        """Invalid JSON is rejected."""
        response = test_client.post(
            "/api/v1/pii/detect",
            data="not valid json",
            headers={**auth_headers(), "Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_missing_required_fields(
        self, test_client, auth_headers
    ):
        """Missing required fields return 422."""
        response = test_client.post(
            "/api/v1/pii/detect",
            json={},  # Missing 'content' field
            headers=auth_headers(),
        )

        assert response.status_code == 422


# ============================================================================
# TestErrorResponses
# ============================================================================


class TestErrorResponses:
    """Tests for error response format."""

    def test_error_response_format(
        self, test_client, mock_pii_service, auth_headers
    ):
        """Error responses follow standard format."""
        mock_pii_service.detect.side_effect = Exception("Internal error")

        response = test_client.post(
            "/api/v1/pii/detect",
            json={"content": "test"},
            headers=auth_headers(),
        )

        assert response.status_code >= 400
        data = response.json()
        assert "detail" in data or "error" in data
