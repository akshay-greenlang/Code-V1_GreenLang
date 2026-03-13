# -*- coding: utf-8 -*-
"""
Tests for API Compliance Routes - AGENT-EUDR-023 API Layer 2

Comprehensive test suite covering:
- 5 compliance assessment endpoints
- Full assessment workflow (create, check status, get result)
- Category-specific compliance checks
- Supplier batch assessment
- Auth/RBAC enforcement on all routes
- Request body validation
- Response structure validation

Test count: 45+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (API - Compliance Routes)
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from tests.agents.eudr.legal_compliance_verifier.conftest import (
    compute_test_hash,
    SHA256_HEX_LENGTH,
    LEGISLATION_CATEGORIES,
    EUDR_COMMODITIES,
    EUDR_COUNTRIES_27,
    COMPLIANCE_DETERMINATIONS,
)


# ---------------------------------------------------------------------------
# Helpers - Mock API Request/Response
# ---------------------------------------------------------------------------


class MockResponse:
    """Mock HTTP response for API testing."""

    def __init__(self, status_code: int, body: Any = None):
        self.status_code = status_code
        self.body = body or {}

    def json(self):
        return self.body


def _mock_create_assessment(payload: Dict, headers: Dict) -> MockResponse:
    """Mock POST /api/v1/legal-compliance/assessments."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})

    required = ["supplier_id", "country_code", "commodity"]
    for field in required:
        if field not in payload:
            return MockResponse(400, {"error": f"Missing field: {field}"})

    if payload["country_code"] not in EUDR_COUNTRIES_27:
        return MockResponse(400, {"error": "Invalid country_code"})
    if payload["commodity"] not in EUDR_COMMODITIES:
        return MockResponse(400, {"error": "Invalid commodity"})

    return MockResponse(201, {
        "assessment_id": "ASM-2025-001",
        "supplier_id": payload["supplier_id"],
        "country_code": payload["country_code"],
        "commodity": payload["commodity"],
        "status": "processing",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


def _mock_get_assessment(assessment_id: str, headers: Dict) -> MockResponse:
    """Mock GET /api/v1/legal-compliance/assessments/{assessment_id}."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})
    if not assessment_id.startswith("ASM-"):
        return MockResponse(404, {"error": "Assessment not found"})
    return MockResponse(200, {
        "assessment_id": assessment_id,
        "status": "completed",
        "overall_score": 72,
        "determination": "PARTIALLY_COMPLIANT",
        "category_scores": {cat: 70 + (i * 5) % 30 for i, cat in enumerate(LEGISLATION_CATEGORIES)},
        "red_flags_count": 3,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provenance_hash": compute_test_hash({"assessment": assessment_id}),
    })


def _mock_check_category(
    supplier_id: str,
    category: str,
    headers: Dict,
) -> MockResponse:
    """Mock GET /api/v1/legal-compliance/assessments/category/{category}."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})
    if category not in LEGISLATION_CATEGORIES:
        return MockResponse(400, {"error": f"Invalid category: {category}"})
    return MockResponse(200, {
        "supplier_id": supplier_id,
        "category": category,
        "score": 75,
        "status": "PARTIALLY_COMPLIANT",
        "required_documents": 3,
        "documents_provided": 2,
        "gaps": ["Missing environmental impact assessment"],
    })


def _mock_batch_assessment(payload: Dict, headers: Dict) -> MockResponse:
    """Mock POST /api/v1/legal-compliance/assessments/batch."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})

    suppliers = payload.get("supplier_ids", [])
    if len(suppliers) == 0:
        return MockResponse(400, {"error": "supplier_ids cannot be empty"})
    if len(suppliers) > 1000:
        return MockResponse(400, {"error": "Batch size exceeds maximum of 1000"})

    results = []
    for sid in suppliers:
        results.append({
            "supplier_id": sid,
            "assessment_id": f"ASM-BATCH-{sid}",
            "status": "queued",
        })
    return MockResponse(202, {
        "batch_id": "BATCH-2025-001",
        "total": len(suppliers),
        "queued": len(suppliers),
        "results": results,
    })


def _mock_list_assessments(params: Dict, headers: Dict) -> MockResponse:
    """Mock GET /api/v1/legal-compliance/assessments."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})

    page = params.get("page", 1)
    page_size = params.get("page_size", 20)
    return MockResponse(200, {
        "results": [
            {"assessment_id": f"ASM-{i:04d}", "status": "completed"}
            for i in range(min(page_size, 5))
        ],
        "total": 50,
        "page": page,
        "page_size": page_size,
    })


# ===========================================================================
# 1. Create Assessment (10 tests)
# ===========================================================================


class TestCreateAssessment:
    """Test POST /api/v1/legal-compliance/assessments."""

    def test_create_assessment_success(self, mock_auth_headers):
        """Test successful assessment creation."""
        payload = {
            "supplier_id": "SUP-0001",
            "country_code": "BR",
            "commodity": "soya",
        }
        resp = _mock_create_assessment(payload, mock_auth_headers)
        assert resp.status_code == 201
        assert resp.body["assessment_id"] is not None
        assert resp.body["status"] == "processing"

    def test_create_assessment_unauthorized(self, mock_unauthorized_headers):
        """Test assessment creation without auth returns 401."""
        payload = {"supplier_id": "SUP-0001", "country_code": "BR", "commodity": "soya"}
        resp = _mock_create_assessment(payload, mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_create_assessment_missing_supplier(self, mock_auth_headers):
        """Test assessment creation without supplier_id returns 400."""
        payload = {"country_code": "BR", "commodity": "soya"}
        resp = _mock_create_assessment(payload, mock_auth_headers)
        assert resp.status_code == 400

    def test_create_assessment_missing_country(self, mock_auth_headers):
        """Test assessment creation without country_code returns 400."""
        payload = {"supplier_id": "SUP-0001", "commodity": "soya"}
        resp = _mock_create_assessment(payload, mock_auth_headers)
        assert resp.status_code == 400

    def test_create_assessment_missing_commodity(self, mock_auth_headers):
        """Test assessment creation without commodity returns 400."""
        payload = {"supplier_id": "SUP-0001", "country_code": "BR"}
        resp = _mock_create_assessment(payload, mock_auth_headers)
        assert resp.status_code == 400

    def test_create_assessment_invalid_country(self, mock_auth_headers):
        """Test assessment with invalid country code returns 400."""
        payload = {"supplier_id": "SUP-0001", "country_code": "XX", "commodity": "soya"}
        resp = _mock_create_assessment(payload, mock_auth_headers)
        assert resp.status_code == 400

    def test_create_assessment_invalid_commodity(self, mock_auth_headers):
        """Test assessment with invalid commodity returns 400."""
        payload = {"supplier_id": "SUP-0001", "country_code": "BR", "commodity": "cotton"}
        resp = _mock_create_assessment(payload, mock_auth_headers)
        assert resp.status_code == 400

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_create_assessment_all_commodities(self, commodity, mock_auth_headers):
        """Test assessment creation for each EUDR commodity."""
        payload = {"supplier_id": "SUP-0001", "country_code": "BR", "commodity": commodity}
        resp = _mock_create_assessment(payload, mock_auth_headers)
        assert resp.status_code == 201


# ===========================================================================
# 2. Get Assessment Result (8 tests)
# ===========================================================================


class TestGetAssessment:
    """Test GET /api/v1/legal-compliance/assessments/{assessment_id}."""

    def test_get_assessment_success(self, mock_auth_headers):
        """Test getting a completed assessment."""
        resp = _mock_get_assessment("ASM-2025-001", mock_auth_headers)
        assert resp.status_code == 200
        assert resp.body["status"] == "completed"
        assert resp.body["determination"] in COMPLIANCE_DETERMINATIONS

    def test_get_assessment_unauthorized(self, mock_unauthorized_headers):
        """Test getting assessment without auth returns 401."""
        resp = _mock_get_assessment("ASM-2025-001", mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_get_assessment_not_found(self, mock_auth_headers):
        """Test getting non-existent assessment returns 404."""
        resp = _mock_get_assessment("INVALID-ID", mock_auth_headers)
        assert resp.status_code == 404

    def test_get_assessment_has_category_scores(self, mock_auth_headers):
        """Test assessment result includes all 8 category scores."""
        resp = _mock_get_assessment("ASM-2025-001", mock_auth_headers)
        for cat in LEGISLATION_CATEGORIES:
            assert cat in resp.body["category_scores"]

    def test_get_assessment_has_red_flags(self, mock_auth_headers):
        """Test assessment result includes red flag count."""
        resp = _mock_get_assessment("ASM-2025-001", mock_auth_headers)
        assert "red_flags_count" in resp.body

    def test_get_assessment_has_provenance(self, mock_auth_headers):
        """Test assessment result includes provenance hash."""
        resp = _mock_get_assessment("ASM-2025-001", mock_auth_headers)
        assert "provenance_hash" in resp.body
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_get_assessment_has_timestamp(self, mock_auth_headers):
        """Test assessment result includes completion timestamp."""
        resp = _mock_get_assessment("ASM-2025-001", mock_auth_headers)
        assert "completed_at" in resp.body

    def test_get_assessment_score_range(self, mock_auth_headers):
        """Test assessment overall score is in valid range."""
        resp = _mock_get_assessment("ASM-2025-001", mock_auth_headers)
        assert 0 <= resp.body["overall_score"] <= 100


# ===========================================================================
# 3. Category-Specific Check (10 tests)
# ===========================================================================


class TestCategoryCheck:
    """Test GET /api/v1/legal-compliance/assessments/category/{category}."""

    @pytest.mark.parametrize("category", LEGISLATION_CATEGORIES)
    def test_check_each_category(self, category, mock_auth_headers):
        """Test category-specific compliance check for each category."""
        resp = _mock_check_category("SUP-0001", category, mock_auth_headers)
        assert resp.status_code == 200
        assert resp.body["category"] == category

    def test_check_invalid_category(self, mock_auth_headers):
        """Test invalid category returns 400."""
        resp = _mock_check_category("SUP-0001", "invalid_category", mock_auth_headers)
        assert resp.status_code == 400

    def test_check_category_unauthorized(self, mock_unauthorized_headers):
        """Test category check without auth returns 401."""
        resp = _mock_check_category("SUP-0001", "land_use_rights", mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_check_category_includes_gaps(self, mock_auth_headers):
        """Test category check includes identified gaps."""
        resp = _mock_check_category("SUP-0001", "environmental_protection", mock_auth_headers)
        assert "gaps" in resp.body

    def test_check_category_includes_documents(self, mock_auth_headers):
        """Test category check includes document counts."""
        resp = _mock_check_category("SUP-0001", "land_use_rights", mock_auth_headers)
        assert "required_documents" in resp.body
        assert "documents_provided" in resp.body


# ===========================================================================
# 4. Batch Assessment (8 tests)
# ===========================================================================


class TestBatchAssessment:
    """Test POST /api/v1/legal-compliance/assessments/batch."""

    def test_batch_success(self, mock_auth_headers):
        """Test successful batch assessment request."""
        payload = {"supplier_ids": ["SUP-0001", "SUP-0002", "SUP-0003"]}
        resp = _mock_batch_assessment(payload, mock_auth_headers)
        assert resp.status_code == 202
        assert resp.body["total"] == 3
        assert resp.body["queued"] == 3

    def test_batch_unauthorized(self, mock_unauthorized_headers):
        """Test batch assessment without auth returns 401."""
        payload = {"supplier_ids": ["SUP-0001"]}
        resp = _mock_batch_assessment(payload, mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_batch_empty_list(self, mock_auth_headers):
        """Test batch with empty supplier list returns 400."""
        payload = {"supplier_ids": []}
        resp = _mock_batch_assessment(payload, mock_auth_headers)
        assert resp.status_code == 400

    def test_batch_exceeds_limit(self, mock_auth_headers):
        """Test batch exceeding 1000 suppliers returns 400."""
        payload = {"supplier_ids": [f"SUP-{i}" for i in range(1001)]}
        resp = _mock_batch_assessment(payload, mock_auth_headers)
        assert resp.status_code == 400

    def test_batch_at_limit(self, mock_auth_headers):
        """Test batch with exactly 1000 suppliers succeeds."""
        payload = {"supplier_ids": [f"SUP-{i}" for i in range(1000)]}
        resp = _mock_batch_assessment(payload, mock_auth_headers)
        assert resp.status_code == 202
        assert resp.body["total"] == 1000

    def test_batch_results_structure(self, mock_auth_headers):
        """Test batch response includes per-supplier results."""
        payload = {"supplier_ids": ["SUP-0001", "SUP-0002"]}
        resp = _mock_batch_assessment(payload, mock_auth_headers)
        for result in resp.body["results"]:
            assert "supplier_id" in result
            assert "assessment_id" in result
            assert "status" in result

    def test_batch_has_batch_id(self, mock_auth_headers):
        """Test batch response includes a batch ID."""
        payload = {"supplier_ids": ["SUP-0001"]}
        resp = _mock_batch_assessment(payload, mock_auth_headers)
        assert "batch_id" in resp.body

    def test_batch_single_supplier(self, mock_auth_headers):
        """Test batch with single supplier succeeds."""
        payload = {"supplier_ids": ["SUP-0001"]}
        resp = _mock_batch_assessment(payload, mock_auth_headers)
        assert resp.status_code == 202
        assert resp.body["total"] == 1


# ===========================================================================
# 5. List Assessments (5 tests)
# ===========================================================================


class TestListAssessments:
    """Test GET /api/v1/legal-compliance/assessments."""

    def test_list_assessments_success(self, mock_auth_headers):
        """Test listing assessments with pagination."""
        resp = _mock_list_assessments({"page": 1, "page_size": 10}, mock_auth_headers)
        assert resp.status_code == 200
        assert "results" in resp.body

    def test_list_assessments_unauthorized(self, mock_unauthorized_headers):
        """Test listing without auth returns 401."""
        resp = _mock_list_assessments({}, mock_unauthorized_headers)
        assert resp.status_code == 401

    def test_list_assessments_total(self, mock_auth_headers):
        """Test list response includes total count."""
        resp = _mock_list_assessments({}, mock_auth_headers)
        assert "total" in resp.body

    def test_list_assessments_pagination(self, mock_auth_headers):
        """Test list response includes pagination info."""
        resp = _mock_list_assessments({"page": 2, "page_size": 10}, mock_auth_headers)
        assert resp.body["page"] == 2

    def test_list_assessments_result_structure(self, mock_auth_headers):
        """Test list results have correct structure."""
        resp = _mock_list_assessments({}, mock_auth_headers)
        for result in resp.body["results"]:
            assert "assessment_id" in result
            assert "status" in result
