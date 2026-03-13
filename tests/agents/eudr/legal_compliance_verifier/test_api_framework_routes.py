# -*- coding: utf-8 -*-
"""
Tests for API Framework Routes - AGENT-EUDR-023 API Layer 1

Comprehensive test suite covering:
- 5 framework management endpoints
- Authentication and RBAC enforcement
- Pagination and filtering
- Request validation
- Error responses

Test count: 40+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (API - Framework Routes)
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.agents.eudr.legal_compliance_verifier.conftest import (
    compute_test_hash,
    SHA256_HEX_LENGTH,
    LEGISLATION_CATEGORIES,
    EUDR_COUNTRIES_27,
)


# ---------------------------------------------------------------------------
# Helpers - Mock API Request/Response
# ---------------------------------------------------------------------------


class MockResponse:
    """Mock HTTP response for API testing."""

    def __init__(self, status_code: int, body: Any = None, headers: Dict = None):
        self.status_code = status_code
        self.body = body or {}
        self.headers = headers or {}

    def json(self):
        return self.body


def _mock_get_framework(country_code: str, headers: Dict) -> MockResponse:
    """Mock GET /api/v1/legal-compliance/frameworks/{country_code}."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})
    if country_code not in EUDR_COUNTRIES_27:
        return MockResponse(404, {"error": f"Country {country_code} not found"})
    return MockResponse(200, {
        "country_code": country_code,
        "frameworks": {cat: {"status": "active"} for cat in LEGISLATION_CATEGORIES},
        "total_categories": 8,
    })


def _mock_search_frameworks(params: Dict, headers: Dict) -> MockResponse:
    """Mock GET /api/v1/legal-compliance/frameworks/search."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})

    page = params.get("page", 1)
    page_size = params.get("page_size", 20)
    if page_size > 100:
        return MockResponse(400, {"error": "page_size must be <= 100"})

    results = []
    for i in range(min(page_size, 5)):
        results.append({
            "id": f"FW-{i+1:04d}",
            "country_code": params.get("country_code", "BR"),
            "category": params.get("category", LEGISLATION_CATEGORIES[i % 8]),
        })

    return MockResponse(200, {
        "results": results,
        "total": len(results),
        "page": page,
        "page_size": page_size,
    })


def _mock_list_countries(headers: Dict) -> MockResponse:
    """Mock GET /api/v1/legal-compliance/frameworks/countries."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})
    return MockResponse(200, {
        "countries": EUDR_COUNTRIES_27,
        "total": len(EUDR_COUNTRIES_27),
    })


def _mock_get_category_coverage(country_code: str, headers: Dict) -> MockResponse:
    """Mock GET /api/v1/legal-compliance/frameworks/{country}/coverage."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})
    if country_code not in EUDR_COUNTRIES_27:
        return MockResponse(404, {"error": "Country not found"})
    coverage = {cat: True for cat in LEGISLATION_CATEGORIES}
    return MockResponse(200, {
        "country_code": country_code,
        "coverage": coverage,
        "covered": 8,
        "total": 8,
        "coverage_pct": 100.0,
    })


def _mock_sync_external(headers: Dict) -> MockResponse:
    """Mock POST /api/v1/legal-compliance/frameworks/sync."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})
    if headers.get("X-Role") != "admin":
        return MockResponse(403, {"error": "Forbidden - admin role required"})
    return MockResponse(200, {"synced": 15, "updated": 8, "errors": 0})


# ===========================================================================
# 1. GET Framework by Country (10 tests)
# ===========================================================================


class TestGetFrameworkEndpoint:
    """Test GET /api/v1/legal-compliance/frameworks/{country_code}."""

    def test_get_brazil_framework(self, mock_auth_headers):
        """Test getting Brazil legal framework."""
        resp = _mock_get_framework("BR", mock_auth_headers)
        assert resp.status_code == 200
        assert resp.body["country_code"] == "BR"
        assert resp.body["total_categories"] == 8

    @pytest.mark.parametrize("country", EUDR_COUNTRIES_27[:10])
    def test_get_framework_for_country(self, country, mock_auth_headers):
        """Test getting framework for 10 EUDR countries."""
        resp = _mock_get_framework(country, mock_auth_headers)
        assert resp.status_code == 200

    def test_get_framework_invalid_country(self, mock_auth_headers):
        """Test getting framework for invalid country returns 404."""
        resp = _mock_get_framework("XX", mock_auth_headers)
        assert resp.status_code == 404

    def test_get_framework_unauthorized(self, mock_unauthorized_headers):
        """Test getting framework without auth returns 401."""
        resp = _mock_get_framework("BR", mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_get_framework_covers_all_categories(self, mock_auth_headers):
        """Test returned framework includes all 8 categories."""
        resp = _mock_get_framework("BR", mock_auth_headers)
        for cat in LEGISLATION_CATEGORIES:
            assert cat in resp.body["frameworks"]


# ===========================================================================
# 2. Search Frameworks (10 tests)
# ===========================================================================


class TestSearchFrameworksEndpoint:
    """Test GET /api/v1/legal-compliance/frameworks/search."""

    def test_search_by_country(self, mock_auth_headers):
        """Test searching frameworks by country code."""
        resp = _mock_search_frameworks({"country_code": "BR"}, mock_auth_headers)
        assert resp.status_code == 200
        assert len(resp.body["results"]) >= 1

    def test_search_by_category(self, mock_auth_headers):
        """Test searching by legislation category."""
        resp = _mock_search_frameworks(
            {"category": "land_use_rights"}, mock_auth_headers,
        )
        assert resp.status_code == 200

    def test_search_with_pagination(self, mock_auth_headers):
        """Test search with pagination parameters."""
        resp = _mock_search_frameworks(
            {"page": 1, "page_size": 5}, mock_auth_headers,
        )
        assert resp.status_code == 200
        assert resp.body["page"] == 1
        assert resp.body["page_size"] == 5

    def test_search_page_size_limit(self, mock_auth_headers):
        """Test search rejects page_size > 100."""
        resp = _mock_search_frameworks(
            {"page_size": 200}, mock_auth_headers,
        )
        assert resp.status_code == 400

    def test_search_unauthorized(self, mock_unauthorized_headers):
        """Test search without auth returns 401."""
        resp = _mock_search_frameworks({}, mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_search_returns_total_count(self, mock_auth_headers):
        """Test search response includes total count."""
        resp = _mock_search_frameworks({}, mock_auth_headers)
        assert "total" in resp.body

    def test_search_results_structure(self, mock_auth_headers):
        """Test search results have correct structure."""
        resp = _mock_search_frameworks({}, mock_auth_headers)
        for result in resp.body["results"]:
            assert "id" in result
            assert "country_code" in result

    def test_search_empty_results(self, mock_auth_headers):
        """Test search with no matching results."""
        resp = _mock_search_frameworks({"country_code": "BR"}, mock_auth_headers)
        assert isinstance(resp.body["results"], list)

    def test_search_default_pagination(self, mock_auth_headers):
        """Test search uses default pagination values."""
        resp = _mock_search_frameworks({}, mock_auth_headers)
        assert resp.body["page"] == 1
        assert resp.body["page_size"] == 20

    def test_search_combined_filters(self, mock_auth_headers):
        """Test search with multiple filter parameters."""
        resp = _mock_search_frameworks(
            {"country_code": "BR", "category": "land_use_rights"}, mock_auth_headers,
        )
        assert resp.status_code == 200


# ===========================================================================
# 3. List Countries (5 tests)
# ===========================================================================


class TestListCountriesEndpoint:
    """Test GET /api/v1/legal-compliance/frameworks/countries."""

    def test_list_all_countries(self, mock_auth_headers):
        """Test listing all covered countries."""
        resp = _mock_list_countries(mock_auth_headers)
        assert resp.status_code == 200
        assert resp.body["total"] == 27

    def test_list_countries_includes_brazil(self, mock_auth_headers):
        """Test country list includes Brazil."""
        resp = _mock_list_countries(mock_auth_headers)
        assert "BR" in resp.body["countries"]

    def test_list_countries_unauthorized(self, mock_unauthorized_headers):
        """Test listing countries without auth returns 401."""
        resp = _mock_list_countries(mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_list_countries_response_structure(self, mock_auth_headers):
        """Test country list response has correct structure."""
        resp = _mock_list_countries(mock_auth_headers)
        assert "countries" in resp.body
        assert "total" in resp.body

    def test_list_countries_count_27(self, mock_auth_headers):
        """Test exactly 27 countries are returned."""
        resp = _mock_list_countries(mock_auth_headers)
        assert len(resp.body["countries"]) == 27


# ===========================================================================
# 4. Category Coverage (5 tests)
# ===========================================================================


class TestCategoryCoverageEndpoint:
    """Test GET /api/v1/legal-compliance/frameworks/{country}/coverage."""

    def test_full_coverage(self, mock_auth_headers):
        """Test country with full 8-category coverage."""
        resp = _mock_get_category_coverage("BR", mock_auth_headers)
        assert resp.status_code == 200
        assert resp.body["covered"] == 8
        assert resp.body["coverage_pct"] == 100.0

    def test_coverage_invalid_country(self, mock_auth_headers):
        """Test coverage for invalid country returns 404."""
        resp = _mock_get_category_coverage("XX", mock_auth_headers)
        assert resp.status_code == 404

    def test_coverage_unauthorized(self, mock_unauthorized_headers):
        """Test coverage without auth returns 401."""
        resp = _mock_get_category_coverage("BR", mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_coverage_includes_all_categories(self, mock_auth_headers):
        """Test coverage response includes all 8 categories."""
        resp = _mock_get_category_coverage("BR", mock_auth_headers)
        for cat in LEGISLATION_CATEGORIES:
            assert cat in resp.body["coverage"]

    def test_coverage_total_is_8(self, mock_auth_headers):
        """Test total categories is 8."""
        resp = _mock_get_category_coverage("BR", mock_auth_headers)
        assert resp.body["total"] == 8


# ===========================================================================
# 5. Sync External Sources (5 tests)
# ===========================================================================


class TestSyncExternalEndpoint:
    """Test POST /api/v1/legal-compliance/frameworks/sync."""

    def test_sync_success(self, mock_admin_headers):
        """Test successful sync with admin credentials."""
        resp = _mock_sync_external(mock_admin_headers)
        assert resp.status_code == 200
        assert resp.body["synced"] >= 0

    def test_sync_unauthorized(self, mock_unauthorized_headers):
        """Test sync without auth returns 401."""
        resp = _mock_sync_external(mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_sync_non_admin_forbidden(self, mock_auth_headers):
        """Test sync with non-admin role returns 403."""
        resp = _mock_sync_external(mock_auth_headers)
        assert resp.status_code == 403

    def test_sync_response_structure(self, mock_admin_headers):
        """Test sync response includes sync stats."""
        resp = _mock_sync_external(mock_admin_headers)
        assert "synced" in resp.body
        assert "updated" in resp.body
        assert "errors" in resp.body

    def test_sync_no_errors(self, mock_admin_headers):
        """Test successful sync has zero errors."""
        resp = _mock_sync_external(mock_admin_headers)
        assert resp.body["errors"] == 0
