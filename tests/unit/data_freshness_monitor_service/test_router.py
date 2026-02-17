# -*- coding: utf-8 -*-
"""
Unit tests for Data Freshness Monitor REST API Router - AGENT-DATA-016

Tests all 20 endpoints via FastAPI TestClient, verifying status codes,
response structure, error paths (404, 503), query parameters, and
request body handling.

Target: 50+ tests.

Author: GreenLang Platform Team
Date: February 2026
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.data_freshness_monitor.api.router import router
from greenlang.data_freshness_monitor.setup import DataFreshnessMonitorService


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def service() -> DataFreshnessMonitorService:
    """Return a started DataFreshnessMonitorService."""
    svc = DataFreshnessMonitorService()
    svc.startup()
    return svc


@pytest.fixture()
def app(service: DataFreshnessMonitorService) -> FastAPI:
    """Create a FastAPI app with the service attached and router included."""
    application = FastAPI()
    application.state.data_freshness_monitor_service = service
    application.include_router(router)
    return application


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    """Return a TestClient bound to the configured app."""
    return TestClient(app)


@pytest.fixture()
def unconfigured_client() -> TestClient:
    """Return a TestClient with no service attached (for 503 tests)."""
    application = FastAPI()
    application.include_router(router)
    return TestClient(application)


@pytest.fixture()
def dataset_id(client: TestClient) -> str:
    """Register a dataset and return its ID."""
    resp = client.post(
        "/api/v1/freshness/datasets",
        json={"name": "router-test-ds", "source": "erp", "owner": "qa"},
    )
    assert resp.status_code == 201
    return resp.json()["data"]["dataset_id"]


@pytest.fixture()
def sla_id(client: TestClient, dataset_id: str) -> str:
    """Create an SLA and return its ID."""
    resp = client.post(
        "/api/v1/freshness/sla",
        json={
            "dataset_id": dataset_id,
            "name": "router-sla",
            "warning_hours": 12.0,
            "critical_hours": 48.0,
        },
    )
    assert resp.status_code == 201
    return resp.json()["data"]["sla_id"]


@pytest.fixture()
def breach_id(
    client: TestClient,
    dataset_id: str,
    sla_id: str,
    service: DataFreshnessMonitorService,
) -> str:
    """Create a breach by running a stale check and return its ID."""
    ds = service.get_dataset(dataset_id)
    ds["last_refreshed_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=100)
    ).isoformat()
    resp = client.post(
        "/api/v1/freshness/check",
        json={"dataset_id": dataset_id},
    )
    assert resp.status_code == 200
    breach = resp.json()["data"].get("sla_breach")
    assert breach is not None
    return breach["breach_id"]


# ===================================================================
# POST /datasets  (Endpoint 1)
# ===================================================================


class TestPostDatasets:
    """Tests for POST /api/v1/freshness/datasets."""

    def test_register_dataset_201(self, client: TestClient):
        """POST /datasets returns 201 with valid body."""
        resp = client.post(
            "/api/v1/freshness/datasets",
            json={"name": "new-ds"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "created"
        assert "dataset_id" in body["data"]

    def test_register_dataset_full_body(self, client: TestClient):
        """POST /datasets accepts all optional fields."""
        resp = client.post(
            "/api/v1/freshness/datasets",
            json={
                "name": "full-ds",
                "source": "api",
                "owner": "team-a",
                "refresh_cadence": "hourly",
                "priority": 2,
                "tags": ["scope1"],
                "metadata": {"k": "v"},
            },
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["source"] == "api"
        assert data["priority"] == 2

    def test_register_dataset_missing_name_422(self, client: TestClient):
        """POST /datasets without required 'name' returns 422."""
        resp = client.post(
            "/api/v1/freshness/datasets",
            json={},
        )
        assert resp.status_code == 422

    def test_register_dataset_503_no_service(self, unconfigured_client: TestClient):
        """POST /datasets returns 503 when service not configured."""
        resp = unconfigured_client.post(
            "/api/v1/freshness/datasets",
            json={"name": "fail"},
        )
        assert resp.status_code == 503


# ===================================================================
# GET /datasets  (Endpoint 2)
# ===================================================================


class TestGetDatasets:
    """Tests for GET /api/v1/freshness/datasets."""

    def test_list_datasets_200(self, client: TestClient):
        """GET /datasets returns 200."""
        resp = client.get("/api/v1/freshness/datasets")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_datasets_after_register(
        self, client: TestClient, dataset_id: str,
    ):
        """GET /datasets includes registered datasets."""
        resp = client.get("/api/v1/freshness/datasets")
        assert resp.json()["data"]["total"] >= 1

    def test_list_datasets_status_filter(
        self, client: TestClient, dataset_id: str,
    ):
        """GET /datasets?status=active filters correctly."""
        resp = client.get("/api/v1/freshness/datasets?status=active")
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] >= 1

    def test_list_datasets_source_filter(
        self, client: TestClient, dataset_id: str,
    ):
        """GET /datasets?source=erp filters by source."""
        resp = client.get("/api/v1/freshness/datasets?source=erp")
        assert resp.status_code == 200

    def test_list_datasets_pagination(self, client: TestClient):
        """GET /datasets respects limit and offset params."""
        resp = client.get(
            "/api/v1/freshness/datasets?limit=5&offset=0",
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["limit"] == 5

    def test_list_datasets_503(self, unconfigured_client: TestClient):
        """GET /datasets returns 503 when service not configured."""
        resp = unconfigured_client.get("/api/v1/freshness/datasets")
        assert resp.status_code == 503


# ===================================================================
# GET /datasets/{id}  (Endpoint 3)
# ===================================================================


class TestGetDatasetById:
    """Tests for GET /api/v1/freshness/datasets/{dataset_id}."""

    def test_get_dataset_200(
        self, client: TestClient, dataset_id: str,
    ):
        """GET /datasets/{id} returns 200 for existing dataset."""
        resp = client.get(f"/api/v1/freshness/datasets/{dataset_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["dataset_id"] == dataset_id

    def test_get_dataset_404(self, client: TestClient):
        """GET /datasets/{id} returns 404 for missing dataset."""
        resp = client.get("/api/v1/freshness/datasets/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# PUT /datasets/{id}  (Endpoint 4)
# ===================================================================


class TestPutDataset:
    """Tests for PUT /api/v1/freshness/datasets/{dataset_id}."""

    def test_update_dataset_200(
        self, client: TestClient, dataset_id: str,
    ):
        """PUT /datasets/{id} returns 200 with valid body."""
        resp = client.put(
            f"/api/v1/freshness/datasets/{dataset_id}",
            json={"name": "updated-name"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "updated-name"

    def test_update_dataset_404(self, client: TestClient):
        """PUT /datasets/{id} returns 404 for missing dataset."""
        resp = client.put(
            "/api/v1/freshness/datasets/nonexistent",
            json={"name": "fail"},
        )
        assert resp.status_code == 404


# ===================================================================
# DELETE /datasets/{id}  (Endpoint 5)
# ===================================================================


class TestDeleteDataset:
    """Tests for DELETE /api/v1/freshness/datasets/{dataset_id}."""

    def test_delete_dataset_200(
        self, client: TestClient, dataset_id: str,
    ):
        """DELETE /datasets/{id} returns 200 for existing dataset."""
        resp = client.delete(f"/api/v1/freshness/datasets/{dataset_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "deleted"

    def test_delete_dataset_404(self, client: TestClient):
        """DELETE /datasets/{id} returns 404 for missing dataset."""
        resp = client.delete("/api/v1/freshness/datasets/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# POST /sla  (Endpoint 6)
# ===================================================================


class TestPostSLA:
    """Tests for POST /api/v1/freshness/sla."""

    def test_create_sla_201(self, client: TestClient):
        """POST /sla returns 201."""
        resp = client.post(
            "/api/v1/freshness/sla",
            json={"name": "test-sla"},
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "created"
        assert "sla_id" in resp.json()["data"]

    def test_create_sla_full_body(self, client: TestClient, dataset_id: str):
        """POST /sla accepts all optional fields."""
        resp = client.post(
            "/api/v1/freshness/sla",
            json={
                "dataset_id": dataset_id,
                "name": "full-sla",
                "warning_hours": 6.0,
                "critical_hours": 24.0,
                "severity": "critical",
                "escalation_policy": {"notify": "ops"},
                "metadata": {"env": "prod"},
            },
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["warning_hours"] == 6.0

    def test_create_sla_503(self, unconfigured_client: TestClient):
        """POST /sla returns 503 when service not configured."""
        resp = unconfigured_client.post(
            "/api/v1/freshness/sla",
            json={"name": "fail"},
        )
        assert resp.status_code == 503


# ===================================================================
# GET /sla  (Endpoint 7)
# ===================================================================


class TestGetSLAs:
    """Tests for GET /api/v1/freshness/sla."""

    def test_list_slas_200(self, client: TestClient):
        """GET /sla returns 200."""
        resp = client.get("/api/v1/freshness/sla")
        assert resp.status_code == 200

    def test_list_slas_dataset_filter(
        self, client: TestClient, dataset_id: str, sla_id: str,
    ):
        """GET /sla?dataset_id=... filters by dataset."""
        resp = client.get(
            f"/api/v1/freshness/sla?dataset_id={dataset_id}",
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] >= 1


# ===================================================================
# GET /sla/{id}  (Endpoint 8)
# ===================================================================


class TestGetSLAById:
    """Tests for GET /api/v1/freshness/sla/{sla_id}."""

    def test_get_sla_200(self, client: TestClient, sla_id: str):
        """GET /sla/{id} returns 200 for existing SLA."""
        resp = client.get(f"/api/v1/freshness/sla/{sla_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["sla_id"] == sla_id

    def test_get_sla_404(self, client: TestClient):
        """GET /sla/{id} returns 404 for missing SLA."""
        resp = client.get("/api/v1/freshness/sla/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# PUT /sla/{id}  (Endpoint 9)
# ===================================================================


class TestPutSLA:
    """Tests for PUT /api/v1/freshness/sla/{sla_id}."""

    def test_update_sla_200(self, client: TestClient, sla_id: str):
        """PUT /sla/{id} returns 200 with valid body."""
        resp = client.put(
            f"/api/v1/freshness/sla/{sla_id}",
            json={"name": "updated-sla"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "updated-sla"

    def test_update_sla_404(self, client: TestClient):
        """PUT /sla/{id} returns 404 for missing SLA."""
        resp = client.put(
            "/api/v1/freshness/sla/nonexistent",
            json={"name": "fail"},
        )
        assert resp.status_code == 404


# ===================================================================
# POST /check  (Endpoint 10)
# ===================================================================


class TestPostCheck:
    """Tests for POST /api/v1/freshness/check."""

    def test_run_check_200(self, client: TestClient, dataset_id: str):
        """POST /check returns 200 for valid dataset."""
        resp = client.post(
            "/api/v1/freshness/check",
            json={"dataset_id": dataset_id},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "check_id" in data
        assert data["dataset_id"] == dataset_id

    def test_run_check_with_timestamp(
        self, client: TestClient, dataset_id: str,
    ):
        """POST /check accepts last_refreshed_at."""
        ts = datetime.now(timezone.utc).isoformat()
        resp = client.post(
            "/api/v1/freshness/check",
            json={
                "dataset_id": dataset_id,
                "last_refreshed_at": ts,
            },
        )
        assert resp.status_code == 200

    def test_run_check_404_missing_dataset(self, client: TestClient):
        """POST /check returns 404 for unknown dataset."""
        resp = client.post(
            "/api/v1/freshness/check",
            json={"dataset_id": "nonexistent"},
        )
        assert resp.status_code == 404

    def test_run_check_missing_body_422(self, client: TestClient):
        """POST /check without body returns 422."""
        resp = client.post(
            "/api/v1/freshness/check",
            json={},
        )
        assert resp.status_code == 422

    def test_run_check_503(self, unconfigured_client: TestClient):
        """POST /check returns 503 when service not configured."""
        resp = unconfigured_client.post(
            "/api/v1/freshness/check",
            json={"dataset_id": "any"},
        )
        assert resp.status_code == 503


# ===================================================================
# POST /check/batch  (Endpoint 11)
# ===================================================================


class TestPostCheckBatch:
    """Tests for POST /api/v1/freshness/check/batch."""

    def test_batch_check_200(self, client: TestClient, dataset_id: str):
        """POST /check/batch returns 200."""
        resp = client.post(
            "/api/v1/freshness/check/batch",
            json={"dataset_ids": [dataset_id]},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total_checked"] == 1

    def test_batch_check_all_datasets(self, client: TestClient, dataset_id: str):
        """POST /check/batch with null IDs checks all datasets."""
        resp = client.post(
            "/api/v1/freshness/check/batch",
            json={},
        )
        assert resp.status_code == 200

    def test_batch_check_503(self, unconfigured_client: TestClient):
        """POST /check/batch returns 503 when service not configured."""
        resp = unconfigured_client.post(
            "/api/v1/freshness/check/batch",
            json={},
        )
        assert resp.status_code == 503


# ===================================================================
# GET /checks  (Endpoint 12)
# ===================================================================


class TestGetChecks:
    """Tests for GET /api/v1/freshness/checks."""

    def test_list_checks_200(self, client: TestClient):
        """GET /checks returns 200."""
        resp = client.get("/api/v1/freshness/checks")
        assert resp.status_code == 200

    def test_list_checks_with_filter(
        self, client: TestClient, dataset_id: str,
    ):
        """GET /checks?dataset_id=... filters by dataset."""
        client.post(
            "/api/v1/freshness/check",
            json={"dataset_id": dataset_id},
        )
        resp = client.get(
            f"/api/v1/freshness/checks?dataset_id={dataset_id}",
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] >= 1


# ===================================================================
# GET /breaches  (Endpoint 13)
# ===================================================================


class TestGetBreaches:
    """Tests for GET /api/v1/freshness/breaches."""

    def test_list_breaches_200(self, client: TestClient):
        """GET /breaches returns 200."""
        resp = client.get("/api/v1/freshness/breaches")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_breaches_severity_filter(self, client: TestClient):
        """GET /breaches?severity=critical filters correctly."""
        resp = client.get("/api/v1/freshness/breaches?severity=critical")
        assert resp.status_code == 200

    def test_list_breaches_status_filter(self, client: TestClient):
        """GET /breaches?status=detected filters correctly."""
        resp = client.get("/api/v1/freshness/breaches?status=detected")
        assert resp.status_code == 200

    def test_list_breaches_503(self, unconfigured_client: TestClient):
        """GET /breaches returns 503 when service not configured."""
        resp = unconfigured_client.get("/api/v1/freshness/breaches")
        assert resp.status_code == 503


# ===================================================================
# GET /breaches/{id}  (Endpoint 14)
# ===================================================================


class TestGetBreachById:
    """Tests for GET /api/v1/freshness/breaches/{breach_id}."""

    def test_get_breach_200(self, client: TestClient, breach_id: str):
        """GET /breaches/{id} returns 200 for existing breach."""
        resp = client.get(f"/api/v1/freshness/breaches/{breach_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["breach_id"] == breach_id

    def test_get_breach_404(self, client: TestClient):
        """GET /breaches/{id} returns 404 for missing breach."""
        resp = client.get("/api/v1/freshness/breaches/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# PUT /breaches/{id}  (Endpoint 15)
# ===================================================================


class TestPutBreach:
    """Tests for PUT /api/v1/freshness/breaches/{breach_id}."""

    def test_update_breach_200(self, client: TestClient, breach_id: str):
        """PUT /breaches/{id} returns 200 for existing breach."""
        resp = client.put(
            f"/api/v1/freshness/breaches/{breach_id}",
            json={"status": "acknowledged"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "acknowledged"

    def test_update_breach_resolved(self, client: TestClient, breach_id: str):
        """PUT /breaches/{id} with 'resolved' sets resolved_at."""
        resp = client.put(
            f"/api/v1/freshness/breaches/{breach_id}",
            json={
                "status": "resolved",
                "resolution_notes": "Fixed",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["resolved_at"] is not None

    def test_update_breach_404(self, client: TestClient):
        """PUT /breaches/{id} returns 404 for missing breach."""
        resp = client.put(
            "/api/v1/freshness/breaches/nonexistent",
            json={"status": "resolved"},
        )
        assert resp.status_code == 404


# ===================================================================
# GET /alerts  (Endpoint 16)
# ===================================================================


class TestGetAlerts:
    """Tests for GET /api/v1/freshness/alerts."""

    def test_list_alerts_200(self, client: TestClient):
        """GET /alerts returns 200."""
        resp = client.get("/api/v1/freshness/alerts")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_alerts_severity_filter(self, client: TestClient):
        """GET /alerts?severity=warning filters by severity."""
        resp = client.get("/api/v1/freshness/alerts?severity=warning")
        assert resp.status_code == 200

    def test_list_alerts_status_filter(self, client: TestClient):
        """GET /alerts?status=open filters by status."""
        resp = client.get("/api/v1/freshness/alerts?status=open")
        assert resp.status_code == 200

    def test_list_alerts_503(self, unconfigured_client: TestClient):
        """GET /alerts returns 503 when service not configured."""
        resp = unconfigured_client.get("/api/v1/freshness/alerts")
        assert resp.status_code == 503


# ===================================================================
# GET /predictions  (Endpoint 17)
# ===================================================================


class TestGetPredictions:
    """Tests for GET /api/v1/freshness/predictions."""

    def test_predictions_200(self, client: TestClient):
        """GET /predictions returns 200."""
        resp = client.get("/api/v1/freshness/predictions")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predictions_dataset_filter(
        self, client: TestClient, dataset_id: str,
    ):
        """GET /predictions?dataset_id=... filters by dataset."""
        resp = client.get(
            f"/api/v1/freshness/predictions?dataset_id={dataset_id}",
        )
        assert resp.status_code == 200

    def test_predictions_503(self, unconfigured_client: TestClient):
        """GET /predictions returns 503 when service not configured."""
        resp = unconfigured_client.get("/api/v1/freshness/predictions")
        assert resp.status_code == 503


# ===================================================================
# POST /pipeline  (Endpoint 18)
# ===================================================================


class TestPostPipeline:
    """Tests for POST /api/v1/freshness/pipeline."""

    def test_pipeline_200(self, client: TestClient):
        """POST /pipeline returns 200."""
        resp = client.post(
            "/api/v1/freshness/pipeline",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "completed"
        assert "pipeline_id" in data

    def test_pipeline_with_ids(self, client: TestClient, dataset_id: str):
        """POST /pipeline with dataset_ids monitors specific datasets."""
        resp = client.post(
            "/api/v1/freshness/pipeline",
            json={"dataset_ids": [dataset_id]},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["dataset_ids"] == [dataset_id]

    def test_pipeline_options(self, client: TestClient):
        """POST /pipeline accepts run_predictions and generate_alerts."""
        resp = client.post(
            "/api/v1/freshness/pipeline",
            json={
                "run_predictions": False,
                "generate_alerts": False,
            },
        )
        assert resp.status_code == 200

    def test_pipeline_503(self, unconfigured_client: TestClient):
        """POST /pipeline returns 503 when service not configured."""
        resp = unconfigured_client.post(
            "/api/v1/freshness/pipeline",
            json={},
        )
        assert resp.status_code == 503


# ===================================================================
# GET /health  (Endpoint 19)
# ===================================================================


class TestGetHealth:
    """Tests for GET /api/v1/freshness/health."""

    def test_health_200(self, client: TestClient):
        """GET /health returns 200."""
        resp = client.get("/api/v1/freshness/health")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["service"] == "data_freshness_monitor"

    def test_health_engines_present(self, client: TestClient):
        """GET /health response includes engines dict."""
        resp = client.get("/api/v1/freshness/health")
        assert "engines" in resp.json()["data"]

    def test_health_503(self, unconfigured_client: TestClient):
        """GET /health returns 503 when service not configured."""
        resp = unconfigured_client.get("/api/v1/freshness/health")
        assert resp.status_code == 503


# ===================================================================
# GET /stats  (Endpoint 20)
# ===================================================================


class TestGetStats:
    """Tests for GET /api/v1/freshness/stats."""

    def test_stats_200(self, client: TestClient):
        """GET /stats returns 200."""
        resp = client.get("/api/v1/freshness/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "total_datasets" in data
        assert "timestamp" in data

    def test_stats_reflects_operations(
        self, client: TestClient, dataset_id: str,
    ):
        """GET /stats reflects registered datasets."""
        resp = client.get("/api/v1/freshness/stats")
        data = resp.json()["data"]
        assert data["total_datasets"] >= 1

    def test_stats_503(self, unconfigured_client: TestClient):
        """GET /stats returns 503 when service not configured."""
        resp = unconfigured_client.get("/api/v1/freshness/stats")
        assert resp.status_code == 503


# ===================================================================
# Cross-cutting 503 tests (service not configured)
# ===================================================================


class TestServiceNotConfigured503:
    """Verify all endpoints return 503 when service is missing."""

    @pytest.fixture()
    def uc(self) -> TestClient:
        """Unconfigured client for 503 tests."""
        application = FastAPI()
        application.include_router(router)
        return TestClient(application)

    def test_get_dataset_by_id_503(self, uc: TestClient):
        """GET /datasets/{id} returns 503 when unconfigured."""
        resp = uc.get("/api/v1/freshness/datasets/any")
        assert resp.status_code == 503

    def test_put_dataset_503(self, uc: TestClient):
        """PUT /datasets/{id} returns 503 when unconfigured."""
        resp = uc.put(
            "/api/v1/freshness/datasets/any",
            json={"name": "x"},
        )
        assert resp.status_code == 503

    def test_delete_dataset_503(self, uc: TestClient):
        """DELETE /datasets/{id} returns 503 when unconfigured."""
        resp = uc.delete("/api/v1/freshness/datasets/any")
        assert resp.status_code == 503

    def test_get_sla_list_503(self, uc: TestClient):
        """GET /sla returns 503 when unconfigured."""
        resp = uc.get("/api/v1/freshness/sla")
        assert resp.status_code == 503

    def test_get_sla_by_id_503(self, uc: TestClient):
        """GET /sla/{id} returns 503 when unconfigured."""
        resp = uc.get("/api/v1/freshness/sla/any")
        assert resp.status_code == 503

    def test_put_sla_503(self, uc: TestClient):
        """PUT /sla/{id} returns 503 when unconfigured."""
        resp = uc.put(
            "/api/v1/freshness/sla/any",
            json={"name": "x"},
        )
        assert resp.status_code == 503

    def test_get_checks_503(self, uc: TestClient):
        """GET /checks returns 503 when unconfigured."""
        resp = uc.get("/api/v1/freshness/checks")
        assert resp.status_code == 503

    def test_get_breach_by_id_503(self, uc: TestClient):
        """GET /breaches/{id} returns 503 when unconfigured."""
        resp = uc.get("/api/v1/freshness/breaches/any")
        assert resp.status_code == 503

    def test_put_breach_503(self, uc: TestClient):
        """PUT /breaches/{id} returns 503 when unconfigured."""
        resp = uc.put(
            "/api/v1/freshness/breaches/any",
            json={"status": "resolved"},
        )
        assert resp.status_code == 503
