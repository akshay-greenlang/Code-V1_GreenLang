# -*- coding: utf-8 -*-
"""
API-level integration tests for AGENT-DATA-015 Cross-Source Reconciliation.

Tests the FastAPI router endpoints using TestClient:
- POST /sources -> POST /match -> POST /compare -> POST /resolve
- GET /golden-records
- GET /health returns ok
- GET /stats returns counts
- Job CRUD operations via API
- Error handling for missing resources

All tests use the real router and a real CrossSourceReconciliationService
instance (no mocks). The TestClient is synchronous and connects to an
in-memory FastAPI app.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from typing import Any, Dict, List

import pytest


# =========================================================================
# Test class: Health and Statistics Endpoints
# =========================================================================


class TestApiHealthStats:
    """Test health check and statistics API endpoints."""

    def test_health_endpoint_returns_ok(self, test_client):
        """GET /health returns status ok with service info."""
        response = test_client.get("/api/v1/reconciliation/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["data"]["status"] == "healthy"
        assert body["data"]["service"] == "cross_source_reconciliation"
        assert "engines" in body["data"]
        assert "timestamp" in body["data"]

    def test_stats_endpoint_returns_counts(self, test_client):
        """GET /stats returns aggregate counters."""
        response = test_client.get("/api/v1/reconciliation/stats")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        data = body["data"]
        assert "total_jobs" in data
        assert "total_sources" in data
        assert "total_matches" in data
        assert "total_discrepancies" in data
        assert "total_golden_records" in data
        assert "provenance_entries" in data


# =========================================================================
# Test class: Source Registration API
# =========================================================================


class TestApiSourceRegistration:
    """Test source registration and listing API endpoints."""

    def test_register_source_via_api(self, test_client):
        """POST /sources registers a new data source."""
        response = test_client.post(
            "/api/v1/reconciliation/sources",
            json={
                "name": "ERP System",
                "source_type": "erp",
                "priority": 1,
                "credibility_score": 0.95,
                "refresh_cadence": "daily",
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "created"
        assert body["data"]["name"] == "ERP System"
        assert body["data"]["source_id"] != ""

    def test_list_sources_via_api(self, test_client):
        """GET /sources returns registered sources."""
        # Register two sources first
        test_client.post(
            "/api/v1/reconciliation/sources",
            json={"name": "Source A", "source_type": "erp"},
        )
        test_client.post(
            "/api/v1/reconciliation/sources",
            json={"name": "Source B", "source_type": "utility"},
        )

        response = test_client.get("/api/v1/reconciliation/sources")

        assert response.status_code == 200
        body = response.json()
        assert body["data"]["total"] >= 2

    def test_get_source_by_id(self, test_client):
        """GET /sources/{source_id} returns a specific source."""
        create_resp = test_client.post(
            "/api/v1/reconciliation/sources",
            json={"name": "Test Source", "source_type": "meter"},
        )
        source_id = create_resp.json()["data"]["source_id"]

        response = test_client.get(
            f"/api/v1/reconciliation/sources/{source_id}",
        )

        assert response.status_code == 200
        body = response.json()
        assert body["data"]["source_id"] == source_id
        assert body["data"]["name"] == "Test Source"

    def test_get_nonexistent_source_returns_404(self, test_client):
        """GET /sources/{fake_id} returns 404."""
        response = test_client.get(
            "/api/v1/reconciliation/sources/nonexistent-id",
        )
        assert response.status_code == 404


# =========================================================================
# Test class: Full API Pipeline Flow
# =========================================================================


class TestApiPipelineFlow:
    """Test the full API pipeline: sources -> match -> compare -> resolve -> golden."""

    def test_full_pipeline_via_api(self, test_client):
        """POST /pipeline executes end-to-end reconciliation."""
        records_a = [
            {
                "entity_id": "FAC-001",
                "period": "2025-Q1",
                "emissions_total": 1000.0,
                "energy_kwh": 40000.0,
            },
            {
                "entity_id": "FAC-002",
                "period": "2025-Q1",
                "emissions_total": 500.0,
                "energy_kwh": 20000.0,
            },
        ]
        records_b = [
            {
                "entity_id": "FAC-001",
                "period": "2025-Q1",
                "emissions_total": 1050.0,
                "energy_kwh": 40500.0,
            },
            {
                "entity_id": "FAC-002",
                "period": "2025-Q1",
                "emissions_total": 520.0,
                "energy_kwh": 20300.0,
            },
        ]

        response = test_client.post(
            "/api/v1/reconciliation/pipeline",
            json={
                "records_a": records_a,
                "records_b": records_b,
                "match_keys": ["entity_id", "period"],
                "resolution_strategy": "priority_wins",
                "generate_golden_records": True,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["data"]["status"] == "completed"
        assert body["data"]["golden_record_count"] >= 1

    def test_match_then_compare_via_api(self, test_client):
        """POST /match then POST /compare individually."""
        records_a = [
            {"entity_id": "E1", "period": "Q1", "value": 100.0},
        ]
        records_b = [
            {"entity_id": "E1", "period": "Q1", "value": 110.0},
        ]

        # Step 1: Match
        match_resp = test_client.post(
            "/api/v1/reconciliation/match",
            json={
                "records_a": records_a,
                "records_b": records_b,
                "match_keys": ["entity_id", "period"],
            },
        )

        assert match_resp.status_code == 200
        match_data = match_resp.json()["data"]
        assert match_data["total_matched"] >= 1

        # Step 2: Compare using inline records
        compare_resp = test_client.post(
            "/api/v1/reconciliation/compare",
            json={
                "record_a": records_a[0],
                "record_b": records_b[0],
                "tolerance_pct": 5.0,
            },
        )

        assert compare_resp.status_code == 200
        comp_data = compare_resp.json()["data"]
        assert comp_data["total_fields"] >= 1

    def test_golden_records_endpoint(self, test_client):
        """GET /golden-records returns assembled golden records."""
        # First run pipeline to create golden records
        test_client.post(
            "/api/v1/reconciliation/pipeline",
            json={
                "records_a": [
                    {"entity_id": "E1", "period": "Q1", "val": 100.0},
                ],
                "records_b": [
                    {"entity_id": "E1", "period": "Q1", "val": 105.0},
                ],
                "generate_golden_records": True,
            },
        )

        response = test_client.get("/api/v1/reconciliation/golden-records")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["total"] >= 1


# =========================================================================
# Test class: Job Management API
# =========================================================================


class TestApiJobManagement:
    """Test job CRUD operations via API endpoints."""

    def test_create_and_get_job(self, test_client):
        """POST /jobs creates a job; GET /jobs/{id} retrieves it."""
        create_resp = test_client.post(
            "/api/v1/reconciliation/jobs",
            json={
                "name": "Test Reconciliation Job",
                "source_ids": ["src_1", "src_2"],
                "strategy": "priority_wins",
            },
        )

        assert create_resp.status_code == 201
        job_data = create_resp.json()["data"]
        job_id = job_data["job_id"]

        get_resp = test_client.get(f"/api/v1/reconciliation/jobs/{job_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["data"]["job_id"] == job_id

    def test_delete_job_via_api(self, test_client):
        """DELETE /jobs/{id} cancels and removes a job."""
        create_resp = test_client.post(
            "/api/v1/reconciliation/jobs",
            json={
                "name": "Job to Delete",
                "source_ids": [],
                "strategy": "auto",
            },
        )
        job_id = create_resp.json()["data"]["job_id"]

        delete_resp = test_client.delete(
            f"/api/v1/reconciliation/jobs/{job_id}",
        )
        assert delete_resp.status_code == 200
        assert delete_resp.json()["data"]["status"] == "cancelled"

    def test_delete_nonexistent_job_returns_404(self, test_client):
        """DELETE /jobs/{fake_id} returns 404."""
        response = test_client.delete(
            "/api/v1/reconciliation/jobs/nonexistent-job-id",
        )
        assert response.status_code == 404
