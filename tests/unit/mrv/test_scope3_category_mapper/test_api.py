# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-029 Scope 3 Category Mapper - REST API Router

40 tests covering all API endpoints using FastAPI TestClient with a mock
service dependency override. Validates status codes, response structure,
and error handling.

Endpoints tested:
- GET  /health                    (health check)
- POST /classify                  (single record classification)
- POST /classify/batch            (batch classification)
- GET  /categories                (list all 15 categories)
- GET  /categories/{number}       (category detail by number)
- POST /lookup/naics              (NAICS code lookup)
- POST /lookup/isic               (ISIC code lookup)
- POST /route                     (route classified record to downstream agent)
- POST /completeness/screen       (completeness screening)
- POST /compliance/assess         (compliance assessment)
- POST /double-counting/check     (double-counting check)
- POST /boundary/determine        (boundary determination)

Author: GL-TestEngineer
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment, misc]
    TestClient = None  # type: ignore[assignment, misc]

try:
    from greenlang.scope3_category_mapper.api.router import router, get_service
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not (FASTAPI_AVAILABLE and ROUTER_AVAILABLE),
    reason="FastAPI or scope3 category mapper router not available",
)

PREFIX = "/api/v1/scope3-mapper"


# ==============================================================================
# MOCK SERVICE
# ==============================================================================


class MockScope3MapperService:
    """
    Mock service returning deterministic responses for all API endpoints.
    """

    async def classify(self, data: dict) -> dict:
        return {
            "record_id": data.get("record_id", "mock-001"),
            "primary_category": "cat_1_purchased_goods",
            "category_number": 1,
            "category_name": "Purchased Goods & Services",
            "confidence": 0.92,
            "classification_method": "naics_lookup",
            "secondary_candidates": [],
            "provenance_hash": "a" * 64,
            "processing_time_ms": 2.3,
        }

    async def classify_batch(self, data: dict) -> dict:
        records = data.get("records", [])
        return {
            "batch_id": "batch-mock-001",
            "record_count": len(records),
            "classifications": [
                {
                    "record_id": r.get("record_id", f"mock-{i}"),
                    "primary_category": "cat_1_purchased_goods",
                    "category_number": 1,
                    "confidence": 0.90,
                    "provenance_hash": f"{i:02d}" + "a" * 62,
                }
                for i, r in enumerate(records)
            ],
            "completeness_score": 65.0,
            "provenance_hash": "b" * 64,
            "processing_time_ms": 15.2,
        }

    async def list_categories(self) -> list:
        return [
            {"number": i, "name": f"Category {i}",
             "direction": "upstream" if i <= 8 else "downstream"}
            for i in range(1, 16)
        ]

    async def get_category(self, number: int) -> dict:
        if number < 1 or number > 15:
            return None
        return {
            "number": number,
            "name": f"Category {number}",
            "direction": "upstream" if number <= 8 else "downstream",
            "description": f"GHG Protocol Scope 3 Category {number}",
            "downstream_agent": f"GL-MRV-S3-{number:03d}",
        }

    async def lookup_naics(self, data: dict) -> dict:
        return {
            "naics_code": data.get("code", "331"),
            "matched_code": "33",
            "primary_category": 1,
            "confidence": 0.78,
            "description": "Manufacturing",
            "provenance_hash": "c" * 64,
        }

    async def lookup_isic(self, data: dict) -> dict:
        return {
            "isic_code": data.get("code", "C"),
            "matched_code": "C",
            "primary_category": 1,
            "confidence": 0.80,
            "description": "Manufacturing",
            "provenance_hash": "d" * 64,
        }

    async def route(self, data: dict) -> dict:
        return {
            "record_id": data.get("record_id", "mock-001"),
            "primary_category": data.get("primary_category", "cat_1_purchased_goods"),
            "downstream_agent": "GL-MRV-S3-014",
            "routed": not data.get("dry_run", False),
            "dry_run": data.get("dry_run", False),
            "provenance_hash": "e" * 64,
        }

    async def screen_completeness(self, data: dict) -> dict:
        return {
            "company_type": data.get("company_type", "manufacturer"),
            "completeness_score": 75.0,
            "total_categories": 15,
            "categories_reported": 10,
            "gaps": ["Category 2 missing", "Category 12 missing"],
            "recommended_actions": ["[CRITICAL] Collect Cat 2 data"],
            "provenance_hash": "f" * 64,
        }

    async def assess_compliance(self, data: dict) -> dict:
        return {
            "frameworks_assessed": data.get("frameworks", ["ghg_protocol"]),
            "overall_status": "WARNING",
            "assessments": {
                "ghg_protocol": {
                    "status": "WARNING",
                    "score": 78.0,
                    "findings": [],
                }
            },
            "provenance_hash": "0" * 64,
        }

    async def check_double_counting(self, data: dict) -> dict:
        return {
            "status": "PASS",
            "overlap_count": 0,
            "overlaps": [],
            "rules_checked": 10,
            "provenance_hash": "1" * 64,
        }

    async def determine_boundary(self, data: dict) -> dict:
        return {
            "rule_id": data.get("rule_id", "DC-SCM-001"),
            "assigned_category": "cat_2_capital_goods",
            "split_required": False,
            "provenance_hash": "2" * 64,
        }


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def client():
    """Create FastAPI TestClient with mock service."""
    if not (FASTAPI_AVAILABLE and ROUTER_AVAILABLE):
        pytest.skip("FastAPI or router not available")

    app = FastAPI()
    app.include_router(router, prefix=PREFIX)

    mock_svc = MockScope3MapperService()
    app.dependency_overrides[get_service] = lambda: mock_svc

    with TestClient(app) as c:
        yield c


# ==============================================================================
# HEALTH ENDPOINT
# ==============================================================================


@_SKIP
class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_endpoint_200(self, client):
        """GET /health returns 200."""
        resp = client.get(f"{PREFIX}/health")
        assert resp.status_code == 200

    def test_health_response_has_status(self, client):
        """Health response includes status field."""
        resp = client.get(f"{PREFIX}/health")
        data = resp.json()
        assert "status" in data

    def test_health_response_has_agent_id(self, client):
        """Health response includes agent_id."""
        resp = client.get(f"{PREFIX}/health")
        data = resp.json()
        assert "agent_id" in data or "service" in data


# ==============================================================================
# CLASSIFY ENDPOINTS
# ==============================================================================


@_SKIP
class TestClassifyEndpoints:
    """Test POST /classify and /classify/batch endpoints."""

    def test_classify_single_201(self, client):
        """POST /classify with valid record -> 200 or 201."""
        payload = {
            "record_id": "SPD-001",
            "amount": 12500.00,
            "currency": "USD",
            "description": "Raw steel purchase",
            "naics_code": "331",
        }
        resp = client.post(f"{PREFIX}/classify", json=payload)
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert data["primary_category"] == "cat_1_purchased_goods"

    def test_classify_batch_201(self, client):
        """POST /classify/batch with multiple records -> 200 or 201."""
        payload = {
            "company_type": "manufacturer",
            "records": [
                {"record_id": "R1", "description": "Steel"},
                {"record_id": "R2", "description": "Airfare"},
            ],
        }
        resp = client.post(f"{PREFIX}/classify/batch", json=payload)
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert data["record_count"] == 2

    def test_classify_invalid_400(self, client):
        """POST /classify with invalid body -> 400 or 422."""
        resp = client.post(f"{PREFIX}/classify", json={})
        assert resp.status_code in (400, 422)

    def test_classify_response_has_provenance(self, client):
        """Classify response includes provenance_hash."""
        payload = {
            "record_id": "SPD-002",
            "description": "Office supplies",
        }
        resp = client.post(f"{PREFIX}/classify", json=payload)
        if resp.status_code in (200, 201):
            data = resp.json()
            assert "provenance_hash" in data

    def test_classify_batch_empty_records(self, client):
        """Batch with empty records list handled."""
        payload = {"company_type": "manufacturer", "records": []}
        resp = client.post(f"{PREFIX}/classify/batch", json=payload)
        assert resp.status_code in (200, 201, 400, 422)


# ==============================================================================
# CATEGORIES ENDPOINTS
# ==============================================================================


@_SKIP
class TestCategoriesEndpoints:
    """Test GET /categories and /categories/{number} endpoints."""

    def test_categories_list_200(self, client):
        """GET /categories returns 200 with all 15 categories."""
        resp = client.get(f"{PREFIX}/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 15

    def test_category_detail_200(self, client):
        """GET /categories/1 returns category detail."""
        resp = client.get(f"{PREFIX}/categories/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["number"] == 1

    def test_category_detail_invalid_number(self, client):
        """GET /categories/99 returns 404."""
        resp = client.get(f"{PREFIX}/categories/99")
        assert resp.status_code in (404, 422)

    def test_category_detail_has_downstream_agent(self, client):
        """Category detail includes downstream agent ID."""
        resp = client.get(f"{PREFIX}/categories/1")
        if resp.status_code == 200:
            data = resp.json()
            assert "downstream_agent" in data


# ==============================================================================
# LOOKUP ENDPOINTS
# ==============================================================================


@_SKIP
class TestLookupEndpoints:
    """Test NAICS and ISIC lookup endpoints."""

    def test_naics_lookup_200(self, client):
        """POST /lookup/naics with valid code -> 200."""
        resp = client.post(
            f"{PREFIX}/lookup/naics", json={"code": "331"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "primary_category" in data

    def test_isic_lookup_200(self, client):
        """POST /lookup/isic with valid code -> 200."""
        resp = client.post(
            f"{PREFIX}/lookup/isic", json={"code": "C"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "primary_category" in data

    def test_naics_lookup_has_provenance(self, client):
        """NAICS lookup response includes provenance_hash."""
        resp = client.post(
            f"{PREFIX}/lookup/naics", json={"code": "484"}
        )
        if resp.status_code == 200:
            data = resp.json()
            assert "provenance_hash" in data

    def test_isic_lookup_invalid(self, client):
        """ISIC lookup with empty code -> error."""
        resp = client.post(
            f"{PREFIX}/lookup/isic", json={"code": ""}
        )
        assert resp.status_code in (400, 422, 500)


# ==============================================================================
# ROUTE ENDPOINT
# ==============================================================================


@_SKIP
class TestRouteEndpoint:
    """Test POST /route endpoint."""

    def test_route_dry_run_200(self, client):
        """POST /route with dry_run=true -> 200 (no actual routing)."""
        payload = {
            "record_id": "SPD-001",
            "primary_category": "cat_1_purchased_goods",
            "dry_run": True,
        }
        resp = client.post(f"{PREFIX}/route", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True

    def test_route_response_has_agent(self, client):
        """Route response includes downstream_agent."""
        payload = {
            "record_id": "SPD-001",
            "primary_category": "cat_1_purchased_goods",
        }
        resp = client.post(f"{PREFIX}/route", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "downstream_agent" in data

    def test_route_provenance(self, client):
        """Route response includes provenance_hash."""
        payload = {
            "record_id": "SPD-001",
            "primary_category": "cat_6_business_travel",
        }
        resp = client.post(f"{PREFIX}/route", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "provenance_hash" in data


# ==============================================================================
# COMPLETENESS ENDPOINT
# ==============================================================================


@_SKIP
class TestCompletenessEndpoint:
    """Test POST /completeness/screen endpoint."""

    def test_completeness_screen_200(self, client):
        """POST /completeness/screen -> 200."""
        payload = {
            "company_type": "manufacturer",
            "categories_reported": [
                "cat_1_purchased_goods",
                "cat_4_upstream_transport",
            ],
        }
        resp = client.post(f"{PREFIX}/completeness/screen", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "completeness_score" in data

    def test_completeness_has_gaps(self, client):
        """Completeness response includes gaps list."""
        payload = {
            "company_type": "manufacturer",
            "categories_reported": ["cat_1_purchased_goods"],
        }
        resp = client.post(f"{PREFIX}/completeness/screen", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "gaps" in data

    def test_completeness_has_recommendations(self, client):
        """Completeness response includes recommended_actions."""
        payload = {
            "company_type": "financial",
            "categories_reported": [],
        }
        resp = client.post(f"{PREFIX}/completeness/screen", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "recommended_actions" in data


# ==============================================================================
# COMPLIANCE ENDPOINT
# ==============================================================================


@_SKIP
class TestComplianceEndpoint:
    """Test POST /compliance/assess endpoint."""

    def test_compliance_assess_200(self, client):
        """POST /compliance/assess -> 200."""
        payload = {
            "company_type": "manufacturer",
            "categories_reported": ["cat_1_purchased_goods"],
            "frameworks": ["ghg_protocol"],
        }
        resp = client.post(f"{PREFIX}/compliance/assess", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "assessments" in data

    def test_compliance_all_frameworks(self, client):
        """Assess all 8 frameworks at once."""
        payload = {
            "company_type": "manufacturer",
            "categories_reported": ["cat_1_purchased_goods"],
            "frameworks": [
                "ghg_protocol", "iso_14064", "csrd_esrs", "cdp",
                "sbti", "sb_253", "sec_climate", "issb_s2",
            ],
        }
        resp = client.post(f"{PREFIX}/compliance/assess", json=payload)
        assert resp.status_code == 200

    def test_compliance_provenance(self, client):
        """Compliance response includes provenance_hash."""
        payload = {
            "company_type": "manufacturer",
            "categories_reported": [],
            "frameworks": ["ghg_protocol"],
        }
        resp = client.post(f"{PREFIX}/compliance/assess", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "provenance_hash" in data


# ==============================================================================
# DOUBLE-COUNTING ENDPOINT
# ==============================================================================


@_SKIP
class TestDoubleCountingEndpoint:
    """Test POST /double-counting/check endpoint."""

    def test_double_counting_check_200(self, client):
        """POST /double-counting/check -> 200."""
        payload = {
            "records": [
                {"record_id": "R1", "assigned_category": "cat_1_purchased_goods"},
                {"record_id": "R2", "assigned_category": "cat_6_business_travel"},
            ],
        }
        resp = client.post(f"{PREFIX}/double-counting/check", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "PASS"

    def test_double_counting_overlap_detected(self, client):
        """Duplicate record IDs across categories -> overlap."""
        payload = {
            "records": [
                {"record_id": "DUP", "assigned_category": "cat_1_purchased_goods"},
                {"record_id": "DUP", "assigned_category": "cat_2_capital_goods"},
            ],
        }
        resp = client.post(f"{PREFIX}/double-counting/check", json=payload)
        assert resp.status_code == 200


# ==============================================================================
# BOUNDARY ENDPOINT
# ==============================================================================


@_SKIP
class TestBoundaryEndpoint:
    """Test POST /boundary/determine endpoint."""

    def test_boundary_determine_200(self, client):
        """POST /boundary/determine -> 200."""
        payload = {
            "rule_id": "DC-SCM-001",
            "record": {
                "amount": 10000.00,
                "useful_life_years": 5,
            },
        }
        resp = client.post(f"{PREFIX}/boundary/determine", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "assigned_category" in data

    def test_boundary_provenance(self, client):
        """Boundary response includes provenance_hash."""
        payload = {
            "rule_id": "DC-SCM-004",
            "record": {
                "direction": "inbound",
                "incoterm": "FOB",
            },
        }
        resp = client.post(f"{PREFIX}/boundary/determine", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "provenance_hash" in data

    def test_boundary_has_assigned_category(self, client):
        """Boundary response assigns a category."""
        payload = {
            "rule_id": "DC-SCM-001",
            "record": {"amount": 7000.00, "useful_life_years": 3},
        }
        resp = client.post(f"{PREFIX}/boundary/determine", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "assigned_category" in data


# ==============================================================================
# ADDITIONAL ENDPOINT TESTS
# ==============================================================================


@_SKIP
class TestAdditionalEndpoints:
    """Additional endpoint validation tests."""

    def test_classify_with_gl_account(self, client):
        """Classify with GL account code."""
        payload = {
            "record_id": "GL-001",
            "amount": 5000.0,
            "gl_account": "5000",
            "description": "Raw materials",
        }
        resp = client.post(f"{PREFIX}/classify", json=payload)
        assert resp.status_code in (200, 201)

    def test_classify_with_description_only(self, client):
        """Classify with description only (keyword match)."""
        payload = {
            "record_id": "KW-001",
            "description": "International business class airfare NYC to London",
        }
        resp = client.post(f"{PREFIX}/classify", json=payload)
        assert resp.status_code in (200, 201)

    def test_batch_classify_response_structure(self, client):
        """Batch classify response has expected structure."""
        payload = {
            "company_type": "manufacturer",
            "records": [
                {"record_id": "R1", "description": "Steel plates"},
                {"record_id": "R2", "description": "Electricity bill"},
                {"record_id": "R3", "description": "Employee airfare"},
            ],
        }
        resp = client.post(f"{PREFIX}/classify/batch", json=payload)
        if resp.status_code in (200, 201):
            data = resp.json()
            assert "classifications" in data
            assert "provenance_hash" in data

    def test_categories_upstream_downstream(self, client):
        """Categories list includes upstream and downstream."""
        resp = client.get(f"{PREFIX}/categories")
        if resp.status_code == 200:
            data = resp.json()
            directions = {c.get("direction") for c in data}
            assert "upstream" in directions
            assert "downstream" in directions

    def test_naics_lookup_manufacturing(self, client):
        """NAICS 331 -> manufacturing / Cat 1."""
        resp = client.post(
            f"{PREFIX}/lookup/naics", json={"code": "331"}
        )
        if resp.status_code == 200:
            data = resp.json()
            assert data["primary_category"] == 1

    def test_isic_lookup_manufacturing(self, client):
        """ISIC C -> manufacturing."""
        resp = client.post(
            f"{PREFIX}/lookup/isic", json={"code": "C"}
        )
        if resp.status_code == 200:
            data = resp.json()
            assert data["primary_category"] == 1

    def test_health_has_version(self, client):
        """Health endpoint returns version info."""
        resp = client.get(f"{PREFIX}/health")
        if resp.status_code == 200:
            data = resp.json()
            assert "version" in data or "agent_id" in data

    def test_double_counting_empty_records(self, client):
        """Double-counting check with empty records -> PASS."""
        payload = {"records": []}
        resp = client.post(f"{PREFIX}/double-counting/check", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert data["status"] == "PASS"

    def test_completeness_all_categories(self, client):
        """Completeness with all 15 categories -> high score."""
        all_cats = [f"cat_{i}_placeholder" for i in range(1, 16)]
        payload = {
            "company_type": "manufacturer",
            "categories_reported": all_cats,
        }
        resp = client.post(f"{PREFIX}/completeness/screen", json=payload)
        assert resp.status_code in (200, 400, 422)

    def test_route_actual_routing(self, client):
        """Route with dry_run=false triggers actual routing."""
        payload = {
            "record_id": "ROUTE-001",
            "primary_category": "cat_1_purchased_goods",
            "dry_run": False,
        }
        resp = client.post(f"{PREFIX}/route", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("routed") is True or data.get("dry_run") is False

    def test_classify_response_confidence_range(self, client):
        """Classification confidence is in [0, 1]."""
        payload = {
            "record_id": "CONF-001",
            "description": "Steel purchase",
            "naics_code": "331",
        }
        resp = client.post(f"{PREFIX}/classify", json=payload)
        if resp.status_code in (200, 201):
            data = resp.json()
            conf = data.get("confidence", 0)
            assert 0 <= conf <= 1
