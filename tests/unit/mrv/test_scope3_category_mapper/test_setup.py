# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-029 Scope 3 Category Mapper - Service Setup (setup.py)

30 tests covering:
- Service instantiation and singleton pattern
- Router creation and endpoint registration
- Request/response model validation
- Health endpoint structure
- Engine access through service facade

Author: GL-TestEngineer
Date: March 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.scope3_category_mapper.setup import (
        Scope3CategoryMapperService,
        get_service,
        get_router,
        ClassifyRequest,
        ClassifyResponse,
        BatchClassifyRequest,
        BatchClassifyResponse,
        RouteRequest,
        RouteResponse,
        CompletenessScreenRequest,
        CompletenessScreenResponse,
        ComplianceAssessRequest,
        ComplianceAssessResponse,
        DoubleCountingCheckRequest,
        DoubleCountingCheckResponse,
        BoundaryDetermineRequest,
        BoundaryDetermineResponse,
        CategoryListResponse,
        CategoryDetailResponse,
        NAICSLookupRequest,
        NAICSLookupResponse,
        ISICLookupRequest,
        ISICLookupResponse,
        HealthResponse,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not SETUP_AVAILABLE,
    reason="Scope3CategoryMapperService not available",
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def service():
    """Create a fresh Scope3CategoryMapperService instance."""
    return Scope3CategoryMapperService()


# ==============================================================================
# SERVICE TESTS
# ==============================================================================


@_SKIP
class TestServiceInstantiation:
    """Test service creation and singleton behavior."""

    def test_get_service_returns_instance(self):
        """get_service returns a Scope3CategoryMapperService."""
        svc = get_service()
        assert isinstance(svc, Scope3CategoryMapperService)

    def test_get_service_singleton(self):
        """get_service returns the same instance on repeated calls."""
        svc1 = get_service()
        svc2 = get_service()
        assert svc1 is svc2

    def test_service_has_classify(self, service):
        """Service exposes a classify method."""
        assert hasattr(service, "classify")
        assert callable(service.classify)

    def test_service_has_classify_batch(self, service):
        """Service exposes a classify_batch method."""
        assert hasattr(service, "classify_batch")
        assert callable(service.classify_batch)

    def test_service_has_screen_completeness(self, service):
        """Service exposes completeness screening."""
        assert hasattr(service, "screen_completeness")
        assert callable(service.screen_completeness)

    def test_service_has_assess_compliance(self, service):
        """Service exposes compliance assessment."""
        assert hasattr(service, "assess_compliance")
        assert callable(service.assess_compliance)

    def test_service_has_check_double_counting(self, service):
        """Service exposes double-counting check."""
        assert hasattr(service, "check_double_counting")
        assert callable(service.check_double_counting)

    def test_service_has_determine_boundary(self, service):
        """Service exposes boundary determination."""
        assert hasattr(service, "determine_boundary")
        assert callable(service.determine_boundary)

    def test_service_has_lookup_naics(self, service):
        """Service exposes NAICS lookup."""
        assert hasattr(service, "lookup_naics")
        assert callable(service.lookup_naics)

    def test_service_has_lookup_isic(self, service):
        """Service exposes ISIC lookup."""
        assert hasattr(service, "lookup_isic")
        assert callable(service.lookup_isic)


# ==============================================================================
# ROUTER TESTS
# ==============================================================================


@_SKIP
class TestRouter:
    """Test router creation and endpoint registration."""

    def test_get_router_returns_api_router(self):
        """get_router returns a FastAPI APIRouter."""
        router = get_router()
        assert router is not None
        # Check it has routes attribute (APIRouter)
        assert hasattr(router, "routes")

    def test_router_has_classify_endpoint(self):
        """Router has POST /classify endpoint."""
        router = get_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("classify" in p for p in paths)

    def test_router_has_batch_endpoint(self):
        """Router has POST /classify/batch endpoint."""
        router = get_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("batch" in p for p in paths)

    def test_router_has_route_endpoint(self):
        """Router has POST /route endpoint."""
        router = get_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("route" in p for p in paths)

    def test_router_has_health_endpoint(self):
        """Router has GET /health endpoint."""
        router = get_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("health" in p for p in paths)

    def test_router_has_categories_endpoint(self):
        """Router has GET /categories endpoint."""
        router = get_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("categories" in p or "category" in p for p in paths)

    def test_router_has_completeness_endpoint(self):
        """Router has POST /completeness endpoint."""
        router = get_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("completeness" in p for p in paths)

    def test_router_has_compliance_endpoint(self):
        """Router has POST /compliance endpoint."""
        router = get_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("compliance" in p for p in paths)

    def test_router_has_double_counting_endpoint(self):
        """Router has POST /double-counting endpoint."""
        router = get_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("double" in p or "dc" in p for p in paths)

    def test_router_has_boundary_endpoint(self):
        """Router has POST /boundary endpoint."""
        router = get_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("boundary" in p for p in paths)


# ==============================================================================
# REQUEST/RESPONSE MODEL TESTS
# ==============================================================================


@_SKIP
class TestModels:
    """Test request and response Pydantic models."""

    def test_health_response_model(self):
        """HealthResponse model instantiation."""
        resp = HealthResponse(
            status="healthy",
            agent_id="GL-MRV-X-040",
            version="1.0.0",
            uptime_seconds=120.5,
        )
        assert resp.status == "healthy"

    def test_classify_request_model(self):
        """ClassifyRequest model instantiation."""
        req = ClassifyRequest(
            record_id="SPD-001",
            amount=Decimal("5000.00"),
            currency="USD",
            description="Steel purchase",
            naics_code="331",
        )
        assert req.record_id == "SPD-001"

    def test_classify_response_model(self):
        """ClassifyResponse model instantiation."""
        resp = ClassifyResponse(
            record_id="SPD-001",
            primary_category="cat_1_purchased_goods",
            category_number=1,
            category_name="Purchased Goods & Services",
            confidence=Decimal("0.92"),
            classification_method="naics_lookup",
            provenance_hash="a" * 64,
            processing_time_ms=2.5,
        )
        assert resp.primary_category == "cat_1_purchased_goods"
        assert resp.confidence == Decimal("0.92")

    def test_batch_classify_request(self):
        """BatchClassifyRequest with multiple records."""
        req = BatchClassifyRequest(
            company_type="manufacturer",
            records=[
                {"record_id": "R1", "description": "Steel"},
                {"record_id": "R2", "description": "Airfare"},
            ],
        )
        assert len(req.records) == 2

    def test_completeness_screen_request(self):
        """CompletenessScreenRequest model."""
        req = CompletenessScreenRequest(
            company_type="manufacturer",
            categories_reported=["cat_1_purchased_goods", "cat_4_upstream_transport"],
        )
        assert req.company_type == "manufacturer"

    def test_compliance_assess_request(self):
        """ComplianceAssessRequest model."""
        req = ComplianceAssessRequest(
            company_type="manufacturer",
            categories_reported=["cat_1_purchased_goods"],
            frameworks=["ghg_protocol", "sbti"],
        )
        assert len(req.frameworks) == 2

    def test_naics_lookup_request(self):
        """NAICSLookupRequest model."""
        req = NAICSLookupRequest(code="331")
        assert req.code == "331"

    def test_isic_lookup_request(self):
        """ISICLookupRequest model."""
        req = ISICLookupRequest(code="C")
        assert req.code == "C"

    def test_route_request(self):
        """RouteRequest model."""
        req = RouteRequest(
            record_id="SPD-001",
            primary_category="cat_1_purchased_goods",
            dry_run=True,
        )
        assert req.dry_run is True

    def test_double_counting_check_request(self):
        """DoubleCountingCheckRequest model."""
        req = DoubleCountingCheckRequest(
            records=[
                {"record_id": "R1", "assigned_category": "cat_1_purchased_goods"},
                {"record_id": "R2", "assigned_category": "cat_2_capital_goods"},
            ],
        )
        assert len(req.records) == 2
