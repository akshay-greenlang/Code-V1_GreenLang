# -*- coding: utf-8 -*-
"""
API endpoint integration tests for AGENT-DATA-016 Data Freshness Monitor.

Tests all 20 REST API endpoints via FastAPI TestClient, validating the
full request-response cycle through the router -> service -> engines.

10+ tests covering:
- Dataset CRUD endpoints
- SLA CRUD endpoints
- Freshness check endpoints (single and batch)
- Breach and alert listing
- Pipeline execution via API
- Health and stats endpoints

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from datetime import datetime, timedelta, timezone

import pytest


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===================================================================
# Health & Stats Endpoints
# ===================================================================


class TestHealthAndStatsAPI:
    """Test the /health and /stats endpoints."""

    def test_health_endpoint_returns_200(self, test_client):
        """GET /api/v1/freshness/health returns 200 with healthy status."""
        resp = test_client.get("/api/v1/freshness/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["data"]["service"] == "data_freshness_monitor"

    def test_stats_endpoint_returns_200(self, test_client):
        """GET /api/v1/freshness/stats returns 200 with aggregate counters."""
        resp = test_client.get("/api/v1/freshness/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "total_datasets" in data["data"]
        assert "total_checks" in data["data"]
        assert "timestamp" in data["data"]


# ===================================================================
# Dataset CRUD Endpoints
# ===================================================================


class TestDatasetCRUDAPI:
    """Test dataset registration, listing, retrieval, update, and deletion."""

    def test_register_dataset_201(self, test_client):
        """POST /api/v1/freshness/datasets creates a new dataset."""
        resp = test_client.post(
            "/api/v1/freshness/datasets",
            json={
                "name": "API Test Dataset",
                "source": "TestAPI",
                "owner": "api-team",
                "refresh_cadence": "daily",
                "priority": 3,
                "tags": ["test", "api"],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "created"
        assert data["data"]["name"] == "API Test Dataset"
        assert data["data"]["dataset_id"]

    def test_list_datasets_200(self, test_client):
        """GET /api/v1/freshness/datasets returns registered datasets."""
        # Register one dataset first
        test_client.post(
            "/api/v1/freshness/datasets",
            json={"name": "List Test DS", "source": "test"},
        )

        resp = test_client.get("/api/v1/freshness/datasets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["data"]["total"] >= 1

    def test_get_dataset_200(self, test_client):
        """GET /api/v1/freshness/datasets/{id} returns a specific dataset."""
        create_resp = test_client.post(
            "/api/v1/freshness/datasets",
            json={"name": "Get Test DS", "source": "test"},
        )
        dataset_id = create_resp.json()["data"]["dataset_id"]

        resp = test_client.get(f"/api/v1/freshness/datasets/{dataset_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["dataset_id"] == dataset_id

    def test_get_dataset_404(self, test_client):
        """GET /api/v1/freshness/datasets/{id} returns 404 for unknown ID."""
        resp = test_client.get("/api/v1/freshness/datasets/nonexistent-id")
        assert resp.status_code == 404

    def test_update_dataset_200(self, test_client):
        """PUT /api/v1/freshness/datasets/{id} updates dataset fields."""
        create_resp = test_client.post(
            "/api/v1/freshness/datasets",
            json={"name": "Update Test DS", "source": "original"},
        )
        dataset_id = create_resp.json()["data"]["dataset_id"]

        resp = test_client.put(
            f"/api/v1/freshness/datasets/{dataset_id}",
            json={"name": "Updated Name", "source": "updated-source"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "Updated Name"
        assert resp.json()["data"]["source"] == "updated-source"

    def test_delete_dataset_200(self, test_client):
        """DELETE /api/v1/freshness/datasets/{id} removes a dataset."""
        create_resp = test_client.post(
            "/api/v1/freshness/datasets",
            json={"name": "Delete Test DS", "source": "test"},
        )
        dataset_id = create_resp.json()["data"]["dataset_id"]

        resp = test_client.delete(
            f"/api/v1/freshness/datasets/{dataset_id}",
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "deleted"

        # Verify it's gone
        get_resp = test_client.get(
            f"/api/v1/freshness/datasets/{dataset_id}",
        )
        assert get_resp.status_code == 404


# ===================================================================
# SLA CRUD Endpoints
# ===================================================================


class TestSLACRUDAPI:
    """Test SLA creation, listing, retrieval, and update via API."""

    def test_create_sla_201(self, test_client):
        """POST /api/v1/freshness/sla creates a new SLA definition."""
        resp = test_client.post(
            "/api/v1/freshness/sla",
            json={
                "name": "API Test SLA",
                "warning_hours": 12.0,
                "critical_hours": 48.0,
                "severity": "high",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "created"
        assert data["data"]["sla_id"]
        assert data["data"]["warning_hours"] == 12.0

    def test_list_slas_200(self, test_client):
        """GET /api/v1/freshness/sla returns SLA definitions."""
        test_client.post(
            "/api/v1/freshness/sla",
            json={"name": "List SLA", "warning_hours": 24.0, "critical_hours": 72.0},
        )

        resp = test_client.get("/api/v1/freshness/sla")
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] >= 1

    def test_get_sla_404(self, test_client):
        """GET /api/v1/freshness/sla/{id} returns 404 for unknown SLA."""
        resp = test_client.get("/api/v1/freshness/sla/nonexistent-sla-id")
        assert resp.status_code == 404

    def test_update_sla_200(self, test_client):
        """PUT /api/v1/freshness/sla/{id} updates SLA thresholds."""
        create_resp = test_client.post(
            "/api/v1/freshness/sla",
            json={"name": "Updatable SLA", "warning_hours": 24.0, "critical_hours": 72.0},
        )
        sla_id = create_resp.json()["data"]["sla_id"]

        resp = test_client.put(
            f"/api/v1/freshness/sla/{sla_id}",
            json={"warning_hours": 6.0, "critical_hours": 24.0},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["warning_hours"] == 6.0
        assert resp.json()["data"]["critical_hours"] == 24.0


# ===================================================================
# Freshness Check Endpoints
# ===================================================================


class TestFreshnessCheckAPI:
    """Test the check and batch-check endpoints."""

    def test_run_check_via_api(self, test_client):
        """POST /api/v1/freshness/check runs a single freshness check."""
        # Register a dataset first
        create_resp = test_client.post(
            "/api/v1/freshness/datasets",
            json={"name": "Check API DS", "source": "test"},
        )
        dataset_id = create_resp.json()["data"]["dataset_id"]

        fresh_ts = (_utcnow() - timedelta(minutes=10)).isoformat()
        resp = test_client.post(
            "/api/v1/freshness/check",
            json={
                "dataset_id": dataset_id,
                "last_refreshed_at": fresh_ts,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["freshness_level"] == "excellent"
        assert data["sla_status"] == "compliant"

    def test_run_check_404_unknown_dataset(self, test_client):
        """POST /api/v1/freshness/check returns 404 for unknown dataset."""
        resp = test_client.post(
            "/api/v1/freshness/check",
            json={"dataset_id": "unknown-id-12345"},
        )
        assert resp.status_code == 404

    def test_run_batch_check_via_api(self, test_client):
        """POST /api/v1/freshness/check/batch runs batch freshness checks."""
        # Register two datasets
        ids = []
        for i in range(2):
            r = test_client.post(
                "/api/v1/freshness/datasets",
                json={"name": f"Batch API DS {i}", "source": "test"},
            )
            ids.append(r.json()["data"]["dataset_id"])

        resp = test_client.post(
            "/api/v1/freshness/check/batch",
            json={"dataset_ids": ids},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total_checked"] == 2

    def test_list_checks_via_api(self, test_client):
        """GET /api/v1/freshness/checks returns check results."""
        # Register and check a dataset
        create_resp = test_client.post(
            "/api/v1/freshness/datasets",
            json={"name": "List Checks DS", "source": "test"},
        )
        dataset_id = create_resp.json()["data"]["dataset_id"]
        test_client.post(
            "/api/v1/freshness/check",
            json={"dataset_id": dataset_id},
        )

        resp = test_client.get("/api/v1/freshness/checks")
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] >= 1


# ===================================================================
# Breach and Alert Endpoints
# ===================================================================


class TestBreachAndAlertAPI:
    """Test breach listing, retrieval, update, and alert listing via API."""

    def _create_breach(self, test_client):
        """Helper to create a dataset with a critical breach."""
        create_resp = test_client.post(
            "/api/v1/freshness/datasets",
            json={"name": "Breach API DS", "source": "test"},
        )
        dataset_id = create_resp.json()["data"]["dataset_id"]

        stale_ts = (_utcnow() - timedelta(hours=100)).isoformat()
        check_resp = test_client.post(
            "/api/v1/freshness/check",
            json={
                "dataset_id": dataset_id,
                "last_refreshed_at": stale_ts,
            },
        )
        return check_resp.json()["data"]

    def test_list_breaches_via_api(self, test_client):
        """GET /api/v1/freshness/breaches returns breach records."""
        self._create_breach(test_client)

        resp = test_client.get("/api/v1/freshness/breaches")
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] >= 1

    def test_get_breach_via_api(self, test_client):
        """GET /api/v1/freshness/breaches/{id} returns breach details."""
        check_data = self._create_breach(test_client)
        breach_id = check_data["sla_breach"]["breach_id"]

        resp = test_client.get(f"/api/v1/freshness/breaches/{breach_id}")
        assert resp.status_code == 200
        assert resp.json()["data"]["breach_id"] == breach_id

    def test_update_breach_via_api(self, test_client):
        """PUT /api/v1/freshness/breaches/{id} updates breach status."""
        check_data = self._create_breach(test_client)
        breach_id = check_data["sla_breach"]["breach_id"]

        resp = test_client.put(
            f"/api/v1/freshness/breaches/{breach_id}",
            json={
                "status": "resolved",
                "resolution_notes": "Manual refresh completed",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "resolved"

    def test_list_alerts_via_api(self, test_client):
        """GET /api/v1/freshness/alerts returns alert records."""
        resp = test_client.get("/api/v1/freshness/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "alerts" in data["data"]


# ===================================================================
# Pipeline and Predictions Endpoints
# ===================================================================


class TestPipelineAndPredictionsAPI:
    """Test pipeline execution and prediction retrieval via API."""

    def test_run_pipeline_via_api(self, test_client):
        """POST /api/v1/freshness/pipeline runs the full monitoring pipeline."""
        # Register a dataset
        create_resp = test_client.post(
            "/api/v1/freshness/datasets",
            json={"name": "Pipeline API DS", "source": "test"},
        )
        dataset_id = create_resp.json()["data"]["dataset_id"]

        resp = test_client.post(
            "/api/v1/freshness/pipeline",
            json={
                "dataset_ids": [dataset_id],
                "run_predictions": False,
                "generate_alerts": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "completed"
        assert data["pipeline_id"]
        assert data["batch_result"]["total_checked"] == 1

    def test_predictions_endpoint_200(self, test_client):
        """GET /api/v1/freshness/predictions returns predictions list."""
        resp = test_client.get("/api/v1/freshness/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "predictions" in data["data"]


# ===================================================================
# Full Register -> Check -> Breach -> Alert Flow via API
# ===================================================================


class TestFullFlowViaAPI:
    """Test the complete register -> check -> breach -> alert flow via API."""

    def test_end_to_end_api_flow(self, test_client):
        """Register dataset -> create SLA -> check (stale) -> list breaches
        -> update breach -> list alerts -> run pipeline -> get stats."""
        # 1. Register dataset
        ds_resp = test_client.post(
            "/api/v1/freshness/datasets",
            json={
                "name": "E2E Flow DS",
                "source": "E2E",
                "owner": "test",
                "priority": 1,
            },
        )
        assert ds_resp.status_code == 201
        dataset_id = ds_resp.json()["data"]["dataset_id"]

        # 2. Create SLA
        sla_resp = test_client.post(
            "/api/v1/freshness/sla",
            json={
                "dataset_id": dataset_id,
                "name": "E2E SLA",
                "warning_hours": 2.0,
                "critical_hours": 6.0,
            },
        )
        assert sla_resp.status_code == 201

        # 3. Check with 8-hour-old data (critical breach)
        stale_ts = (_utcnow() - timedelta(hours=8)).isoformat()
        check_resp = test_client.post(
            "/api/v1/freshness/check",
            json={
                "dataset_id": dataset_id,
                "last_refreshed_at": stale_ts,
            },
        )
        assert check_resp.status_code == 200
        check_data = check_resp.json()["data"]
        assert check_data["sla_status"] == "critical"
        breach_id = check_data["sla_breach"]["breach_id"]

        # 4. List breaches
        breaches_resp = test_client.get("/api/v1/freshness/breaches")
        assert breaches_resp.status_code == 200
        assert breaches_resp.json()["data"]["total"] >= 1

        # 5. Update breach to resolved
        update_resp = test_client.put(
            f"/api/v1/freshness/breaches/{breach_id}",
            json={"status": "resolved", "resolution_notes": "E2E resolved"},
        )
        assert update_resp.status_code == 200

        # 6. Run pipeline
        pipeline_resp = test_client.post(
            "/api/v1/freshness/pipeline",
            json={
                "dataset_ids": [dataset_id],
                "run_predictions": False,
                "generate_alerts": True,
            },
        )
        assert pipeline_resp.status_code == 200

        # 7. Verify stats
        stats_resp = test_client.get("/api/v1/freshness/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()["data"]
        assert stats["total_datasets"] >= 1
        assert stats["total_sla_definitions"] >= 1
        assert stats["total_checks"] >= 1
        assert stats["total_breaches"] >= 1
        assert stats["total_pipelines"] >= 1
