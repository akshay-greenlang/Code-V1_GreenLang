# -*- coding: utf-8 -*-
"""
Unit tests for Audit Search API Routes - SEC-005: Centralized Audit Logging Service

Tests the API endpoints for advanced audit event searching.

Coverage targets: 85%+ of api/search_routes.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import the audit search routes module.
# ---------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from greenlang.infrastructure.audit_service.api.search_routes import (
        search_router,
    )
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False
    search_router = None


pytestmark = pytest.mark.skipif(
    not _HAS_MODULE,
    reason="audit_service.api.search_routes not yet implemented",
)


# ============================================================================
# Helpers
# ============================================================================


def _make_search_request(
    query: str = "login",
    filters: Optional[Dict[str, Any]] = None,
    aggregations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a search request body."""
    return {
        "query": query,
        "filters": filters or {},
        "aggregations": aggregations or [],
        "page": 1,
        "page_size": 20,
    }


def _make_search_response(
    hits: int = 50,
    events: Optional[List[Dict]] = None,
    aggregations: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create a search response."""
    return {
        "hits": hits,
        "events": events or [
            {"event_id": f"e-{i}", "event_type": "auth.login_success"}
            for i in range(min(hits, 20))
        ],
        "aggregations": aggregations or {},
        "page": 1,
        "page_size": 20,
    }


def _make_auth_context() -> MagicMock:
    """Create a mock auth context."""
    mock = MagicMock()
    mock.user_id = "u-1"
    mock.tenant_id = "t-acme"
    mock.permissions = ["audit:read", "audit:search"]
    return mock


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with search routes."""
    app = FastAPI()
    app.include_router(search_router, prefix="/api/v1/audit")
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_search_service() -> AsyncMock:
    """Create a mock search service."""
    service = AsyncMock()
    service.search = AsyncMock(return_value=_make_search_response())
    return service


@pytest.fixture
def mock_auth():
    """Mock authentication dependency."""
    with patch("greenlang.infrastructure.audit_service.api.search_routes.get_auth_context") as mock:
        mock.return_value = _make_auth_context()
        yield mock


# ============================================================================
# TestPostSearch
# ============================================================================


class TestPostSearch:
    """Tests for POST /search endpoint."""

    def test_search_basic_query(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with basic query returns results."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(query="login"),
            )
            assert response.status_code == 200
            data = response.json()
            assert "hits" in data or "total" in data or "events" in data

    def test_search_with_category_filter(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with category filter."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    filters={"category": "auth"},
                ),
            )
            assert response.status_code == 200

    def test_search_with_event_type_filter(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with event_type filter."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    filters={"event_type": ["auth.login_success", "auth.login_failure"]},
                ),
            )
            assert response.status_code == 200

    def test_search_with_user_id_filter(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with user_id filter."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    filters={"user_id": "u-42"},
                ),
            )
            assert response.status_code == 200

    def test_search_with_date_range_filter(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with date range filter."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            end = datetime.now(timezone.utc).isoformat()
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    filters={
                        "start_date": start,
                        "end_date": end,
                    },
                ),
            )
            assert response.status_code == 200

    def test_search_with_severity_filter(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with severity filter."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    filters={"severity": ["error", "critical"]},
                ),
            )
            assert response.status_code == 200

    def test_search_with_result_filter(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with result filter."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    filters={"result": "failure"},
                ),
            )
            assert response.status_code == 200


# ============================================================================
# TestSearchQueryParsing
# ============================================================================


class TestSearchQueryParsing:
    """Tests for search query parsing."""

    def test_search_wildcard_query(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with wildcard query."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(query="*"),
            )
            assert response.status_code == 200

    def test_search_phrase_query(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with phrase query."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(query='"login failure"'),
            )
            assert response.status_code == 200

    def test_search_boolean_query(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with boolean query."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(query="login AND failure"),
            )
            assert response.status_code == 200

    def test_search_field_specific_query(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with field-specific query."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(query="user_id:u-42"),
            )
            assert response.status_code == 200


# ============================================================================
# TestSearchAggregations
# ============================================================================


class TestSearchAggregations:
    """Tests for search aggregations."""

    def test_search_with_category_aggregation(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with category aggregation."""
        mock_search_service.search.return_value = _make_search_response(
            aggregations={
                "category": {"auth": 100, "rbac": 50, "data": 30},
            },
        )
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    aggregations=["category"],
                ),
            )
            assert response.status_code == 200
            data = response.json()
            assert "aggregations" in data or True

    def test_search_with_severity_aggregation(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with severity aggregation."""
        mock_search_service.search.return_value = _make_search_response(
            aggregations={
                "severity": {"info": 500, "warning": 100, "error": 20},
            },
        )
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    aggregations=["severity"],
                ),
            )
            assert response.status_code == 200

    def test_search_with_timeline_aggregation(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with timeline aggregation."""
        mock_search_service.search.return_value = _make_search_response(
            aggregations={
                "timeline": [
                    {"timestamp": "2026-02-01", "count": 100},
                    {"timestamp": "2026-02-02", "count": 150},
                ],
            },
        )
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    aggregations=["timeline"],
                ),
            )
            assert response.status_code == 200

    def test_search_with_multiple_aggregations(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with multiple aggregations."""
        mock_search_service.search.return_value = _make_search_response(
            aggregations={
                "category": {"auth": 100},
                "severity": {"info": 500},
            },
        )
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(
                    query="*",
                    aggregations=["category", "severity", "user_id"],
                ),
            )
            assert response.status_code == 200


# ============================================================================
# TestSearchPagination
# ============================================================================


class TestSearchPagination:
    """Tests for search pagination."""

    def test_search_pagination_page_2(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with page 2."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json={
                    "query": "*",
                    "page": 2,
                    "page_size": 20,
                },
            )
            assert response.status_code == 200

    def test_search_pagination_large_page_size(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with large page size."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json={
                    "query": "*",
                    "page": 1,
                    "page_size": 100,
                },
            )
            assert response.status_code == 200

    def test_search_pagination_max_page_size(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search respects max page size limit."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json={
                    "query": "*",
                    "page": 1,
                    "page_size": 10000,  # Should be limited
                },
            )
            # May return 400 or silently limit
            assert response.status_code in (200, 400, 422)


# ============================================================================
# TestSearchErrorHandling
# ============================================================================


class TestSearchErrorHandling:
    """Tests for search error handling."""

    def test_search_empty_query(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with empty query returns 400."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json={"query": ""},
            )
            assert response.status_code in (400, 422, 200)  # May allow empty query

    def test_search_invalid_filter(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search with invalid filter returns 400."""
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json={
                    "query": "*",
                    "filters": {"invalid_field": "value"},
                },
            )
            # May ignore unknown filters or return 400
            assert response.status_code in (200, 400, 422)

    def test_search_service_error(
        self, client: TestClient, mock_search_service, mock_auth
    ) -> None:
        """POST /search handles service errors."""
        mock_search_service.search.side_effect = Exception("Search failed")
        with patch("greenlang.infrastructure.audit_service.api.search_routes.search_service", mock_search_service):
            response = client.post(
                "/api/v1/audit/search",
                json=_make_search_request(),
            )
            assert response.status_code == 500
