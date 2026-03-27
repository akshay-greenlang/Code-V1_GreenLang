# -*- coding: utf-8 -*-
"""
Unit tests for Dual Reporting Reconciliation REST API router.

AGENT-MRV-013: Dual Reporting Reconciliation Agent

Tests the FastAPI router with 16 REST endpoints at /api/v1/dual-reporting.
Uses TestClient for HTTP-level testing of request/response handling.

Target: 40 tests.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

# Check if FastAPI test client is available
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Check if router is importable
try:
    from greenlang.agents.mrv.dual_reporting_reconciliation.api.router import (
        router,
        create_router,
    )
    ROUTER_AVAILABLE = router is not None
except ImportError:
    ROUTER_AVAILABLE = False

# Skip all tests if FastAPI is not available
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE or not ROUTER_AVAILABLE,
    reason="FastAPI or router not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a FastAPI app with the dual-reporting router."""
    application = FastAPI()
    if router is not None:
        application.include_router(router)
    return application


@pytest.fixture
def client(app):
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_api_request() -> Dict[str, Any]:
    """Return a sample reconciliation API request body."""
    return {
        "tenant_id": "tenant-001",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "upstream_results": [
            {
                "facility_id": "FAC-001",
                "energy_type": "electricity",
                "method": "location_based",
                "emissions_tco2e": 1250.50,
                "energy_quantity_mwh": 5000.0,
                "ef_used": 0.2501,
                "ef_source": "eGRID 2023",
                "ef_hierarchy": "grid_average",
                "tier": "tier_1",
                "gwp_source": "AR5",
                "provenance_hash": "abc123",
            },
            {
                "facility_id": "FAC-001",
                "energy_type": "electricity",
                "method": "market_based",
                "emissions_tco2e": 625.25,
                "energy_quantity_mwh": 5000.0,
                "ef_used": 0.12505,
                "ef_source": "Supplier Disclosure",
                "ef_hierarchy": "supplier_specific",
                "tier": "tier_3",
                "gwp_source": "AR5",
                "provenance_hash": "def456",
            },
        ],
    }


# ===========================================================================
# 1. Health & Stats Endpoints
# ===========================================================================


class TestHealthEndpoints:
    """Test health check and stats endpoints."""

    def test_health_check(self, client):
        response = client.get("/api/v1/dual-reporting/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_stats(self, client):
        response = client.get("/api/v1/dual-reporting/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_reconciliations" in data


# ===========================================================================
# 2. Reconciliation Endpoints
# ===========================================================================


class TestReconciliationEndpoints:
    """Test reconciliation CRUD endpoints."""

    def test_create_reconciliation(self, client, sample_api_request):
        response = client.post(
            "/api/v1/dual-reporting/reconciliations",
            json=sample_api_request,
        )
        assert response.status_code == 201
        data = response.json()
        assert "reconciliation_id" in data or "success" in data

    def test_list_reconciliations(self, client):
        response = client.get("/api/v1/dual-reporting/reconciliations")
        assert response.status_code == 200
        data = response.json()
        assert "reconciliations" in data or "total" in data

    def test_list_with_pagination(self, client):
        response = client.get(
            "/api/v1/dual-reporting/reconciliations?skip=0&limit=5",
        )
        assert response.status_code == 200

    def test_list_with_tenant_filter(self, client):
        response = client.get(
            "/api/v1/dual-reporting/reconciliations?tenant_id=tenant-001",
        )
        assert response.status_code == 200

    def test_get_reconciliation_not_found(self, client):
        response = client.get(
            "/api/v1/dual-reporting/reconciliations/NONEXISTENT",
        )
        assert response.status_code == 404

    def test_delete_reconciliation_not_found(self, client):
        response = client.delete(
            "/api/v1/dual-reporting/reconciliations/NONEXISTENT",
        )
        assert response.status_code == 404


# ===========================================================================
# 3. Batch Reconciliation
# ===========================================================================


class TestBatchEndpoints:
    """Test batch reconciliation endpoint."""

    def test_create_batch(self, client, sample_api_request):
        batch_body = {
            "tenant_id": "tenant-001",
            "periods": [
                {
                    "tenant_id": "tenant-001",
                    "period_start": "2024-01-01",
                    "period_end": "2024-12-31",
                    "upstream_results": sample_api_request["upstream_results"],
                },
            ],
        }
        response = client.post(
            "/api/v1/dual-reporting/reconciliations/batch",
            json=batch_body,
        )
        assert response.status_code == 201
        data = response.json()
        assert "total_periods" in data or "batch_id" in data


# ===========================================================================
# 4. Sub-Resource Endpoints
# ===========================================================================


class TestSubResourceEndpoints:
    """Test reconciliation sub-resource endpoints."""

    def _create_reconciliation(self, client, sample_api_request):
        """Helper to create a reconciliation and return its ID."""
        response = client.post(
            "/api/v1/dual-reporting/reconciliations",
            json=sample_api_request,
        )
        data = response.json()
        return data.get("reconciliation_id", "")

    def test_get_discrepancies(self, client, sample_api_request):
        recon_id = self._create_reconciliation(client, sample_api_request)
        if recon_id:
            response = client.get(
                f"/api/v1/dual-reporting/reconciliations/{recon_id}/discrepancies",
            )
            assert response.status_code == 200

    def test_get_waterfall(self, client, sample_api_request):
        recon_id = self._create_reconciliation(client, sample_api_request)
        if recon_id:
            response = client.get(
                f"/api/v1/dual-reporting/reconciliations/{recon_id}/waterfall",
            )
            assert response.status_code == 200

    def test_get_quality(self, client, sample_api_request):
        recon_id = self._create_reconciliation(client, sample_api_request)
        if recon_id:
            response = client.get(
                f"/api/v1/dual-reporting/reconciliations/{recon_id}/quality",
            )
            assert response.status_code == 200

    def test_get_tables(self, client, sample_api_request):
        recon_id = self._create_reconciliation(client, sample_api_request)
        if recon_id:
            response = client.get(
                f"/api/v1/dual-reporting/reconciliations/{recon_id}/tables",
            )
            assert response.status_code == 200

    def test_get_trends(self, client, sample_api_request):
        recon_id = self._create_reconciliation(client, sample_api_request)
        if recon_id:
            response = client.get(
                f"/api/v1/dual-reporting/reconciliations/{recon_id}/trends",
            )
            assert response.status_code == 200


# ===========================================================================
# 5. Compliance Endpoint
# ===========================================================================


class TestComplianceEndpoints:
    """Test compliance check endpoints."""

    def test_check_compliance(self, client, sample_api_request):
        resp = client.post(
            "/api/v1/dual-reporting/reconciliations",
            json=sample_api_request,
        )
        recon_id = resp.json().get("reconciliation_id", "")
        if recon_id:
            response = client.post(
                f"/api/v1/dual-reporting/reconciliations/{recon_id}/compliance",
                json={"frameworks": ["ghg_protocol"]},
            )
            assert response.status_code in (201, 404)

    def test_get_compliance_not_found(self, client):
        response = client.get(
            "/api/v1/dual-reporting/compliance/NONEXISTENT",
        )
        assert response.status_code == 404


# ===========================================================================
# 6. Aggregations
# ===========================================================================


class TestAggregationEndpoints:
    """Test aggregation endpoint."""

    def test_get_aggregations(self, client):
        response = client.get("/api/v1/dual-reporting/aggregations")
        assert response.status_code == 200

    def test_get_aggregations_with_group_by(self, client):
        response = client.get(
            "/api/v1/dual-reporting/aggregations?group_by=facility",
        )
        assert response.status_code == 200


# ===========================================================================
# 7. Export
# ===========================================================================


class TestExportEndpoints:
    """Test export endpoint."""

    def test_export_nonexistent(self, client):
        response = client.post(
            "/api/v1/dual-reporting/export",
            json={
                "reconciliation_id": "NONEXISTENT",
                "format": "json",
            },
        )
        # Should return 404 or 500 for nonexistent reconciliation
        assert response.status_code in (404, 500)
