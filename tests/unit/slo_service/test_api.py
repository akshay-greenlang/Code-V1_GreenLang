# -*- coding: utf-8 -*-
"""
Unit tests for SLO Service REST API (OBS-005)

Tests all API endpoints using FastAPI TestClient with mocked service
layer.  Covers CRUD, budgets, burn rates, compliance, rule generation,
and error handling.

Coverage target: 85%+ of api/router.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE, reason="FastAPI not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_slo_service(sample_slo, sample_slo_list, sample_error_budget):
    """Create a mock SLOService for API testing."""
    svc = MagicMock()
    svc.manager = MagicMock()
    svc.manager.list_all.return_value = sample_slo_list
    svc.manager.get.return_value = sample_slo
    svc.manager.create.return_value = sample_slo
    svc.manager.update.return_value = sample_slo
    svc.manager.delete.return_value = True
    svc.manager.get_history.return_value = [sample_slo.to_dict()]
    svc.manager.load_from_yaml.return_value = sample_slo_list

    svc.get_budget = MagicMock(return_value=sample_error_budget)
    svc.get_budget_history = MagicMock(return_value=[sample_error_budget])
    svc.get_all_budgets = MagicMock(return_value=[sample_error_budget])
    svc.get_burn_rates = MagicMock(return_value={"fast": 0.0, "medium": 0.0, "slow": 0.0})

    from greenlang.infrastructure.slo_service.models import SLOReport
    mock_report = SLOReport(report_type="weekly", total_slos=3, slos_met=3)
    svc.generate_compliance_report = MagicMock(return_value=mock_report)

    svc.generate_recording_rules = MagicMock(return_value="/tmp/rules.yaml")
    svc.generate_alert_rules = MagicMock(return_value="/tmp/alerts.yaml")
    svc.generate_dashboards = MagicMock(return_value=["/tmp/overview.json"])
    svc.evaluate_all = AsyncMock(return_value=[{"slo_id": "test", "status": "evaluated"}])

    return svc


@pytest.fixture
def test_app(mock_slo_service):
    """Create a FastAPI app with the SLO router and mock service."""
    from greenlang.infrastructure.slo_service.api.router import slo_router

    app = FastAPI()
    app.state.slo_service = mock_slo_service
    if slo_router is not None:
        app.include_router(slo_router)
    return app


@pytest.fixture
def client(test_app):
    """Create a TestClient for the FastAPI app."""
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSLOAPIListAndCreate:
    """Tests for SLO listing and creation endpoints."""

    def test_list_slos(self, client):
        """GET /api/v1/slos returns list of SLOs."""
        resp = client.get("/api/v1/slos")
        assert resp.status_code == 200
        data = resp.json()
        assert "slos" in data
        assert "count" in data
        assert data["count"] == 3

    def test_create_slo(self, client):
        """POST /api/v1/slos creates a new SLO."""
        body = {
            "slo_id": "new-slo",
            "name": "New SLO",
            "service": "test-svc",
            "target": 99.9,
            "sli": {
                "name": "test_sli",
                "sli_type": "availability",
                "good_query": "good",
                "total_query": "total",
            },
        }
        resp = client.post("/api/v1/slos", json=body)
        assert resp.status_code == 201

    def test_create_slo_validation_error(self, client, mock_slo_service):
        """POST with duplicate SLO returns 409."""
        mock_slo_service.manager.create.side_effect = ValueError("already exists")
        body = {
            "slo_id": "dup",
            "name": "Dup",
            "service": "svc",
            "sli": {
                "name": "sli",
                "sli_type": "availability",
                "good_query": "g",
                "total_query": "t",
            },
        }
        resp = client.post("/api/v1/slos", json=body)
        assert resp.status_code == 409


class TestSLOAPICRUD:
    """Tests for single-SLO CRUD endpoints."""

    def test_get_slo_by_id(self, client):
        """GET /api/v1/slos/{id} returns SLO details."""
        resp = client.get("/api/v1/slos/api-availability-99-9")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slo_id"] == "api-availability-99-9"

    def test_get_nonexistent_slo_404(self, client, mock_slo_service):
        """GET non-existent SLO returns 404."""
        mock_slo_service.manager.get.return_value = None
        resp = client.get("/api/v1/slos/nonexistent")
        assert resp.status_code == 404

    def test_update_slo(self, client):
        """PATCH /api/v1/slos/{id} updates an SLO."""
        resp = client.patch(
            "/api/v1/slos/api-availability-99-9",
            json={"description": "Updated"},
        )
        assert resp.status_code == 200

    def test_update_nonexistent_slo_404(self, client, mock_slo_service):
        """PATCH non-existent SLO returns 404."""
        mock_slo_service.manager.update.side_effect = KeyError("not found")
        resp = client.patch(
            "/api/v1/slos/nonexistent",
            json={"description": "x"},
        )
        assert resp.status_code == 404

    def test_delete_slo(self, client):
        """DELETE /api/v1/slos/{id} soft-deletes an SLO."""
        resp = client.delete("/api/v1/slos/api-availability-99-9")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_get_slo_history(self, client):
        """GET /api/v1/slos/{id}/history returns version history."""
        resp = client.get("/api/v1/slos/api-availability-99-9/history")
        assert resp.status_code == 200
        assert "history" in resp.json()


class TestSLOAPIBudget:
    """Tests for budget endpoints."""

    def test_get_error_budget(self, client):
        """GET /api/v1/slos/{id}/budget returns budget."""
        resp = client.get("/api/v1/slos/api-availability-99-9/budget")
        assert resp.status_code == 200

    def test_get_budget_history(self, client):
        """GET /api/v1/slos/{id}/budget/history returns budget snapshots."""
        resp = client.get("/api/v1/slos/api-availability-99-9/budget/history")
        assert resp.status_code == 200
        assert "history" in resp.json()

    def test_get_burn_rate(self, client):
        """GET /api/v1/slos/{id}/burn-rate returns burn rates."""
        resp = client.get("/api/v1/slos/api-availability-99-9/burn-rate")
        assert resp.status_code == 200
        assert "burn_rates" in resp.json()

    def test_all_budgets(self, client):
        """GET /api/v1/slos/budgets returns all budgets."""
        resp = client.get("/api/v1/slos/budgets")
        assert resp.status_code == 200
        assert "budgets" in resp.json()


class TestSLOAPICompliance:
    """Tests for compliance and generation endpoints."""

    def test_compliance_report(self, client):
        """GET /api/v1/slos/compliance returns report."""
        resp = client.get("/api/v1/slos/compliance")
        assert resp.status_code == 200

    def test_generate_recording_rules(self, client):
        """POST /api/v1/slos/recording-rules generates rules."""
        resp = client.post("/api/v1/slos/recording-rules")
        assert resp.status_code == 200
        assert resp.json()["status"] == "generated"

    def test_generate_alert_rules(self, client):
        """POST /api/v1/slos/alert-rules generates alert rules."""
        resp = client.post("/api/v1/slos/alert-rules")
        assert resp.status_code == 200

    def test_generate_dashboards(self, client):
        """POST /api/v1/slos/dashboards generates dashboards."""
        resp = client.post("/api/v1/slos/dashboards")
        assert resp.status_code == 200

    def test_evaluate_slos(self, client):
        """POST /api/v1/slos/evaluate triggers evaluation."""
        resp = client.post("/api/v1/slos/evaluate")
        assert resp.status_code == 200
        assert "evaluated" in resp.json()


class TestSLOAPIImportExport:
    """Tests for import/export endpoints."""

    def test_import_slos(self, client):
        """POST /api/v1/slos/import imports SLOs."""
        resp = client.post(
            "/api/v1/slos/import",
            json={"yaml_path": "/tmp/slos.yaml"},
        )
        assert resp.status_code == 200

    def test_export_slos(self, client):
        """GET /api/v1/slos/export returns all SLOs."""
        resp = client.get("/api/v1/slos/export")
        assert resp.status_code == 200
        assert "slos" in resp.json()


class TestSLOAPIOverview:
    """Tests for overview and health endpoints."""

    def test_slo_overview(self, client):
        """GET /api/v1/slos/overview returns summary."""
        resp = client.get("/api/v1/slos/overview")
        assert resp.status_code == 200
        assert "total_slos" in resp.json()

    def test_health_check(self, client):
        """GET /api/v1/slos/health returns healthy status."""
        resp = client.get("/api/v1/slos/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestSLOAPIErrors:
    """Tests for error response format."""

    def test_error_response_format(self, client, mock_slo_service):
        """Error responses include detail field."""
        mock_slo_service.manager.get.return_value = None
        resp = client.get("/api/v1/slos/nonexistent")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_service_not_configured_503(self):
        """503 when SLO service is not attached."""
        app = FastAPI()
        from greenlang.infrastructure.slo_service.api.router import slo_router
        if slo_router is not None:
            app.include_router(slo_router)
        bare_client = TestClient(app)
        resp = bare_client.get("/api/v1/slos")
        assert resp.status_code == 503
