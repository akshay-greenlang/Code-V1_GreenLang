# -*- coding: utf-8 -*-
"""
Unit Tests for Cross-Source Reconciliation REST API Router (AGENT-DATA-015)
============================================================================

Comprehensive test suite for ``greenlang.cross_source_reconciliation.api.router``
covering all 20 endpoints using ``starlette.testclient.TestClient``.

Target: 50+ tests covering all endpoints, happy paths, 404 cases, and 503 cases.

The test uses a MagicMock service wired into
``app.state.cross_source_reconciliation_service`` so that the router's
``_get_service(request)`` helper returns the mock rather than raising
503 Service Unavailable.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from greenlang.cross_source_reconciliation.api.router import FASTAPI_AVAILABLE

# Skip the entire module if FastAPI is not installed
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not available; skipping router tests",
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def mock_service():
    """Create a MagicMock mimicking CrossSourceReconciliationService.

    Every method called by the router is stubbed to return the shape
    of data the router expects.
    """
    svc = MagicMock()

    # ---- Job CRUD (endpoints 1-4) ----
    svc.create_job.return_value = {
        "job_id": "job-001",
        "name": "reconciliation-job-001",
        "source_ids": ["src-a", "src-b"],
        "strategy": "auto",
        "status": "pending",
        "match_count": 0,
        "discrepancy_count": 0,
        "golden_record_count": 0,
        "created_at": "2026-02-01T00:00:00+00:00",
        "started_at": None,
        "completed_at": None,
        "provenance_hash": "a" * 64,
    }
    svc.list_jobs.return_value = {
        "jobs": [{"job_id": "job-001", "status": "pending"}],
        "count": 1,
        "total": 1,
        "limit": 50,
        "offset": 0,
    }
    svc.get_job.return_value = {
        "job_id": "job-001",
        "name": "reconciliation-job-001",
        "status": "pending",
    }
    svc.delete_job.return_value = {"job_id": "job-001", "status": "cancelled"}

    # ---- Source CRUD (endpoints 5-8) ----
    svc.register_source.return_value = {
        "source_id": "src-001",
        "name": "SAP ERP",
        "source_type": "erp",
        "priority": 5,
        "credibility_score": 0.8,
        "refresh_cadence": "monthly",
        "record_count": 0,
        "status": "active",
        "created_at": "2026-02-01T00:00:00+00:00",
        "updated_at": "2026-02-01T00:00:00+00:00",
        "provenance_hash": "b" * 64,
    }
    svc.list_sources.return_value = {
        "sources": [{"source_id": "src-001", "name": "SAP ERP"}],
        "count": 1,
        "total": 1,
        "limit": 50,
        "offset": 0,
    }
    svc.get_source.return_value = {
        "source_id": "src-001",
        "name": "SAP ERP",
        "source_type": "erp",
    }
    svc.update_source.return_value = {
        "source_id": "src-001",
        "name": "SAP ERP Updated",
        "priority": 3,
        "provenance_hash": "c" * 64,
    }

    # ---- Matching (endpoints 9-11) ----
    svc.match_records.return_value = {
        "match_id": "match-001",
        "source_ids": ["src-a", "src-b"],
        "strategy": "composite",
        "threshold": 0.85,
        "matched_pairs": [
            {"record_a": {"entity_id": "f1"}, "record_b": {"entity_id": "f1"}, "confidence": 1.0},
        ],
        "total_matched": 1,
        "total_unmatched_a": 0,
        "total_unmatched_b": 0,
        "avg_confidence": 1.0,
        "processing_time_ms": 2.5,
        "provenance_hash": "d" * 64,
    }
    svc.list_matches.return_value = {
        "matches": [{"match_id": "match-001"}],
        "count": 1,
        "total": 1,
        "limit": 50,
        "offset": 0,
    }
    svc.get_match.return_value = {
        "match_id": "match-001",
        "total_matched": 1,
    }

    # ---- Comparison (endpoint 12) ----
    svc.compare_records.return_value = {
        "comparison_id": "comp-001",
        "match_id": "",
        "fields_compared": [
            {"field": "electricity_kwh", "result": "match", "value_a": 100, "value_b": 100},
        ],
        "total_fields": 1,
        "matching_fields": 1,
        "mismatching_fields": 0,
        "missing_fields": 0,
        "match_rate": 1.0,
        "processing_time_ms": 1.0,
        "provenance_hash": "e" * 64,
    }

    # ---- Discrepancies (endpoints 13-14) ----
    svc.list_discrepancies.return_value = {
        "discrepancies": [
            {"discrepancy_id": "disc-001", "field": "val", "severity": "high", "status": "open"},
        ],
        "count": 1,
        "total": 1,
        "limit": 50,
        "offset": 0,
    }
    svc.get_discrepancy.return_value = {
        "discrepancy_id": "disc-001",
        "field": "val",
        "severity": "high",
        "status": "open",
    }

    # ---- Resolution (endpoint 15) ----
    svc.resolve_discrepancies.return_value = {
        "resolution_id": "res-001",
        "strategy": "priority_wins",
        "resolutions": [
            {"resolution_id": "res-sub-001", "discrepancy_id": "disc-001",
             "field": "val", "resolved_value": 100, "winning_source": "source_a"},
        ],
        "total_resolved": 1,
        "processing_time_ms": 1.0,
        "provenance_hash": "f" * 64,
    }

    # ---- Golden Records (endpoints 16-17) ----
    svc.get_golden_records.return_value = {
        "golden_records": [
            {"record_id": "golden-001", "entity_id": "f1", "overall_confidence": 0.95},
        ],
        "count": 1,
        "total": 1,
        "limit": 50,
        "offset": 0,
    }
    svc.get_golden_record.return_value = {
        "record_id": "golden-001",
        "entity_id": "f1",
        "overall_confidence": 0.95,
        "status": "active",
    }

    # ---- Pipeline (endpoint 18) ----
    svc.run_pipeline.return_value = {
        "pipeline_id": "pipe-001",
        "source_ids": [],
        "match_result": {"total_matched": 2},
        "comparison_count": 2,
        "discrepancy_result": {"total_discrepancies": 1},
        "resolution_result": {"total_resolved": 1},
        "golden_records": [{"record_id": "golden-001"}],
        "golden_record_count": 1,
        "status": "completed",
        "total_processing_time_ms": 15.0,
        "provenance_hash": "0" * 64,
    }

    # ---- Health + Stats (endpoints 19-20) ----
    svc.get_health.return_value = {
        "status": "healthy",
        "service": "cross_source_reconciliation",
        "engines": {"source_registry": True, "matching_engine": True},
        "stores": {"jobs": 0, "sources": 0},
        "timestamp": "2026-02-01T00:00:00+00:00",
    }
    svc.get_statistics.return_value = {
        "total_jobs": 5,
        "total_sources": 3,
        "total_matches": 10,
        "total_comparisons": 8,
        "total_discrepancies": 4,
        "total_resolutions": 3,
        "total_golden_records": 2,
        "total_pipelines": 1,
        "provenance_entries": 30,
        "timestamp": "2026-02-01T00:00:00+00:00",
    }

    return svc


@pytest.fixture
def client(mock_service):
    """Create a FastAPI TestClient with the mock service wired in."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from greenlang.cross_source_reconciliation.api.router import router

    app = FastAPI()
    app.include_router(router)
    # The router's _get_service reads this exact attribute name
    app.state.cross_source_reconciliation_service = mock_service

    return TestClient(app)


@pytest.fixture
def client_no_service():
    """TestClient WITHOUT the service (for 503 tests)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from greenlang.cross_source_reconciliation.api.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ===================================================================
# 1. POST /api/v1/reconciliation/jobs (Create job) -- endpoint 1
# ===================================================================


class TestCreateJob:
    """Tests for POST /jobs."""

    def test_create_job_returns_201(self, client):
        """POST /jobs returns 201 with job data."""
        resp = client.post(
            "/api/v1/reconciliation/jobs",
            json={
                "name": "test-reconciliation",
                "source_ids": ["src-a", "src-b"],
                "strategy": "auto",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "created"
        assert "data" in body
        assert body["data"]["job_id"] == "job-001"

    def test_create_job_default_body(self, client):
        """POST /jobs with minimal body uses defaults."""
        resp = client.post(
            "/api/v1/reconciliation/jobs",
            json={},
        )
        assert resp.status_code == 201

    def test_create_job_calls_service(self, client, mock_service):
        """POST /jobs delegates to service.create_job."""
        client.post(
            "/api/v1/reconciliation/jobs",
            json={"name": "test"},
        )
        mock_service.create_job.assert_called_once()

    def test_create_job_503_without_service(self, client_no_service):
        """POST /jobs returns 503 when service is not configured."""
        resp = client_no_service.post(
            "/api/v1/reconciliation/jobs",
            json={"name": "test"},
        )
        assert resp.status_code == 503


# ===================================================================
# 2. GET /api/v1/reconciliation/jobs (List jobs) -- endpoint 2
# ===================================================================


class TestListJobs:
    """Tests for GET /jobs."""

    def test_list_jobs_returns_200(self, client):
        """GET /jobs returns 200 with a jobs list."""
        resp = client.get("/api/v1/reconciliation/jobs")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert "jobs" in body["data"]
        assert isinstance(body["data"]["jobs"], list)

    def test_list_jobs_has_count(self, client):
        """GET /jobs response includes count."""
        resp = client.get("/api/v1/reconciliation/jobs")
        body = resp.json()
        assert body["data"]["count"] >= 1

    def test_list_jobs_with_status_filter(self, client, mock_service):
        """GET /jobs with status query param passes filter to service."""
        resp = client.get("/api/v1/reconciliation/jobs?status=pending")
        assert resp.status_code == 200
        mock_service.list_jobs.assert_called_once()

    def test_list_jobs_503_without_service(self, client_no_service):
        """GET /jobs returns 503 when service is not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/jobs")
        assert resp.status_code == 503


# ===================================================================
# 3. GET /api/v1/reconciliation/jobs/{id} (Get job) -- endpoint 3
# ===================================================================


class TestGetJob:
    """Tests for GET /jobs/{job_id}."""

    def test_get_job_returns_200(self, client):
        """GET /jobs/{id} returns 200 when the service finds the job."""
        resp = client.get("/api/v1/reconciliation/jobs/job-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["job_id"] == "job-001"

    def test_get_job_not_found(self, client, mock_service):
        """GET /jobs/{id} returns 404 for unknown ID."""
        mock_service.get_job.return_value = None
        resp = client.get("/api/v1/reconciliation/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_get_job_503_without_service(self, client_no_service):
        """GET /jobs/{id} returns 503 when service is not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/jobs/job-001")
        assert resp.status_code == 503


# ===================================================================
# 4. DELETE /api/v1/reconciliation/jobs/{id} (Delete job) -- endpoint 4
# ===================================================================


class TestDeleteJob:
    """Tests for DELETE /jobs/{job_id}."""

    def test_delete_job_returns_200(self, client):
        """DELETE /jobs/{id} returns 200."""
        resp = client.delete("/api/v1/reconciliation/jobs/job-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["status"] == "cancelled"

    def test_delete_job_not_found(self, client, mock_service):
        """DELETE /jobs/{id} returns 404 when service raises ValueError."""
        mock_service.delete_job.side_effect = ValueError("Job not found")
        resp = client.delete("/api/v1/reconciliation/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_job_503_without_service(self, client_no_service):
        """DELETE /jobs/{id} returns 503 when service is not configured."""
        resp = client_no_service.delete("/api/v1/reconciliation/jobs/job-001")
        assert resp.status_code == 503


# ===================================================================
# 5. POST /api/v1/reconciliation/sources (Register source) -- endpoint 5
# ===================================================================


class TestRegisterSource:
    """Tests for POST /sources."""

    def test_register_source_returns_201(self, client):
        """POST /sources returns 201 with source data."""
        resp = client.post(
            "/api/v1/reconciliation/sources",
            json={
                "name": "SAP ERP",
                "source_type": "erp",
                "priority": 3,
                "credibility_score": 0.9,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "created"
        assert body["data"]["source_id"] == "src-001"

    def test_register_source_calls_service(self, client, mock_service):
        """POST /sources delegates to service.register_source."""
        client.post(
            "/api/v1/reconciliation/sources",
            json={"name": "Test Source"},
        )
        mock_service.register_source.assert_called_once()

    def test_register_source_503_without_service(self, client_no_service):
        """POST /sources returns 503 when service is not configured."""
        resp = client_no_service.post(
            "/api/v1/reconciliation/sources",
            json={"name": "Test"},
        )
        assert resp.status_code == 503


# ===================================================================
# 6. GET /api/v1/reconciliation/sources (List sources) -- endpoint 6
# ===================================================================


class TestListSources:
    """Tests for GET /sources."""

    def test_list_sources_returns_200(self, client):
        """GET /sources returns 200 with a sources list."""
        resp = client.get("/api/v1/reconciliation/sources")
        assert resp.status_code == 200
        body = resp.json()
        assert "sources" in body["data"]

    def test_list_sources_has_count(self, client):
        """GET /sources response includes count."""
        resp = client.get("/api/v1/reconciliation/sources")
        body = resp.json()
        assert body["data"]["count"] >= 1

    def test_list_sources_pagination(self, client, mock_service):
        """GET /sources with limit and offset passes them to service."""
        resp = client.get("/api/v1/reconciliation/sources?limit=10&offset=5")
        assert resp.status_code == 200
        mock_service.list_sources.assert_called_once()

    def test_list_sources_503_without_service(self, client_no_service):
        """GET /sources returns 503 when service is not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/sources")
        assert resp.status_code == 503


# ===================================================================
# 7. GET /api/v1/reconciliation/sources/{id} (Get source) -- endpoint 7
# ===================================================================


class TestGetSource:
    """Tests for GET /sources/{source_id}."""

    def test_get_source_returns_200(self, client):
        """GET /sources/{id} returns 200 when source found."""
        resp = client.get("/api/v1/reconciliation/sources/src-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["source_id"] == "src-001"

    def test_get_source_not_found(self, client, mock_service):
        """GET /sources/{id} returns 404 for unknown ID."""
        mock_service.get_source.return_value = None
        resp = client.get("/api/v1/reconciliation/sources/nonexistent-id")
        assert resp.status_code == 404

    def test_get_source_503_without_service(self, client_no_service):
        """GET /sources/{id} returns 503 when service is not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/sources/src-001")
        assert resp.status_code == 503


# ===================================================================
# 8. PUT /api/v1/reconciliation/sources/{id} (Update source) -- endpoint 8
# ===================================================================


class TestUpdateSource:
    """Tests for PUT /sources/{source_id}."""

    def test_update_source_returns_200(self, client):
        """PUT /sources/{id} returns 200 with updated data."""
        resp = client.put(
            "/api/v1/reconciliation/sources/src-001",
            json={"name": "Updated Name", "priority": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_update_source_not_found(self, client, mock_service):
        """PUT /sources/{id} returns 404 when service raises ValueError."""
        mock_service.update_source.side_effect = ValueError("Source not found")
        resp = client.put(
            "/api/v1/reconciliation/sources/nonexistent-id",
            json={"name": "X"},
        )
        assert resp.status_code == 404

    def test_update_source_503_without_service(self, client_no_service):
        """PUT /sources/{id} returns 503 when service is not configured."""
        resp = client_no_service.put(
            "/api/v1/reconciliation/sources/src-001",
            json={"name": "X"},
        )
        assert resp.status_code == 503


# ===================================================================
# 9. POST /api/v1/reconciliation/match (Match records) -- endpoint 9
# ===================================================================


class TestMatchRecords:
    """Tests for POST /match."""

    def test_match_records_returns_200(self, client):
        """POST /match returns 200 with match results."""
        resp = client.post(
            "/api/v1/reconciliation/match",
            json={
                "source_ids": ["src-a", "src-b"],
                "records_a": [{"entity_id": "f1", "period": "Q1"}],
                "records_b": [{"entity_id": "f1", "period": "Q1"}],
                "match_keys": ["entity_id", "period"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["match_id"] == "match-001"

    def test_match_records_default_body(self, client):
        """POST /match with minimal body uses defaults."""
        resp = client.post(
            "/api/v1/reconciliation/match",
            json={},
        )
        assert resp.status_code == 200

    def test_match_records_calls_service(self, client, mock_service):
        """POST /match delegates to service.match_records."""
        client.post(
            "/api/v1/reconciliation/match",
            json={"records_a": [], "records_b": []},
        )
        mock_service.match_records.assert_called_once()

    def test_match_records_503_without_service(self, client_no_service):
        """POST /match returns 503 when service is not configured."""
        resp = client_no_service.post(
            "/api/v1/reconciliation/match",
            json={},
        )
        assert resp.status_code == 503


# ===================================================================
# 10. GET /api/v1/reconciliation/matches (List matches) -- endpoint 10
# ===================================================================


class TestListMatches:
    """Tests for GET /matches."""

    def test_list_matches_returns_200(self, client):
        """GET /matches returns 200 with matches list."""
        resp = client.get("/api/v1/reconciliation/matches")
        assert resp.status_code == 200
        body = resp.json()
        assert "matches" in body["data"]

    def test_list_matches_has_count(self, client):
        """GET /matches response includes count."""
        resp = client.get("/api/v1/reconciliation/matches")
        body = resp.json()
        assert body["data"]["count"] >= 1

    def test_list_matches_503_without_service(self, client_no_service):
        """GET /matches returns 503 when service is not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/matches")
        assert resp.status_code == 503


# ===================================================================
# 11. GET /api/v1/reconciliation/matches/{id} (Get match) -- endpoint 11
# ===================================================================


class TestGetMatch:
    """Tests for GET /matches/{match_id}."""

    def test_get_match_returns_200(self, client):
        """GET /matches/{id} returns 200 when match found."""
        resp = client.get("/api/v1/reconciliation/matches/match-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["match_id"] == "match-001"

    def test_get_match_not_found(self, client, mock_service):
        """GET /matches/{id} returns 404 for unknown ID."""
        mock_service.get_match.return_value = None
        resp = client.get("/api/v1/reconciliation/matches/nonexistent-id")
        assert resp.status_code == 404

    def test_get_match_503_without_service(self, client_no_service):
        """GET /matches/{id} returns 503 when service is not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/matches/match-001")
        assert resp.status_code == 503


# ===================================================================
# 12. POST /api/v1/reconciliation/compare (Compare records) -- endpoint 12
# ===================================================================


class TestCompareRecords:
    """Tests for POST /compare."""

    def test_compare_returns_200(self, client):
        """POST /compare returns 200 with comparison results."""
        resp = client.post(
            "/api/v1/reconciliation/compare",
            json={
                "record_a": {"val": 100},
                "record_b": {"val": 105},
                "fields": ["val"],
                "tolerance_pct": 5.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["comparison_id"] == "comp-001"

    def test_compare_default_body(self, client):
        """POST /compare with minimal body uses defaults."""
        resp = client.post(
            "/api/v1/reconciliation/compare",
            json={},
        )
        assert resp.status_code == 200

    def test_compare_calls_service(self, client, mock_service):
        """POST /compare delegates to service.compare_records."""
        client.post(
            "/api/v1/reconciliation/compare",
            json={"record_a": {}, "record_b": {}},
        )
        mock_service.compare_records.assert_called_once()

    def test_compare_503_without_service(self, client_no_service):
        """POST /compare returns 503 when service is not configured."""
        resp = client_no_service.post(
            "/api/v1/reconciliation/compare",
            json={},
        )
        assert resp.status_code == 503


# ===================================================================
# 13. GET /api/v1/reconciliation/discrepancies (List discrepancies) -- endpoint 13
# ===================================================================


class TestListDiscrepancies:
    """Tests for GET /discrepancies."""

    def test_list_discrepancies_returns_200(self, client):
        """GET /discrepancies returns 200 with discrepancies list."""
        resp = client.get("/api/v1/reconciliation/discrepancies")
        assert resp.status_code == 200
        body = resp.json()
        assert "discrepancies" in body["data"]

    def test_list_discrepancies_with_severity_filter(self, client, mock_service):
        """GET /discrepancies with severity filter passes it to service."""
        resp = client.get("/api/v1/reconciliation/discrepancies?severity=high")
        assert resp.status_code == 200
        mock_service.list_discrepancies.assert_called_once()

    def test_list_discrepancies_with_status_filter(self, client, mock_service):
        """GET /discrepancies with status filter passes it to service."""
        resp = client.get("/api/v1/reconciliation/discrepancies?status=open")
        assert resp.status_code == 200
        mock_service.list_discrepancies.assert_called_once()

    def test_list_discrepancies_503_without_service(self, client_no_service):
        """GET /discrepancies returns 503 when service is not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/discrepancies")
        assert resp.status_code == 503


# ===================================================================
# 14. GET /api/v1/reconciliation/discrepancies/{id} -- endpoint 14
# ===================================================================


class TestGetDiscrepancy:
    """Tests for GET /discrepancies/{discrepancy_id}."""

    def test_get_discrepancy_returns_200(self, client):
        """GET /discrepancies/{id} returns 200 when found."""
        resp = client.get("/api/v1/reconciliation/discrepancies/disc-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["discrepancy_id"] == "disc-001"

    def test_get_discrepancy_not_found(self, client, mock_service):
        """GET /discrepancies/{id} returns 404 for unknown ID."""
        mock_service.get_discrepancy.return_value = None
        resp = client.get("/api/v1/reconciliation/discrepancies/nonexistent-id")
        assert resp.status_code == 404

    def test_get_discrepancy_503_without_service(self, client_no_service):
        """GET /discrepancies/{id} returns 503 when service not configured."""
        resp = client_no_service.get(
            "/api/v1/reconciliation/discrepancies/disc-001",
        )
        assert resp.status_code == 503


# ===================================================================
# 15. POST /api/v1/reconciliation/resolve (Resolve) -- endpoint 15
# ===================================================================


class TestResolveDiscrepancies:
    """Tests for POST /resolve."""

    def test_resolve_returns_200(self, client):
        """POST /resolve returns 200 with resolution results."""
        resp = client.post(
            "/api/v1/reconciliation/resolve",
            json={
                "discrepancy_ids": ["disc-001"],
                "strategy": "priority_wins",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["total_resolved"] == 1

    def test_resolve_default_body(self, client):
        """POST /resolve with minimal body uses defaults."""
        resp = client.post(
            "/api/v1/reconciliation/resolve",
            json={},
        )
        assert resp.status_code == 200

    def test_resolve_calls_service(self, client, mock_service):
        """POST /resolve delegates to service.resolve_discrepancies."""
        client.post(
            "/api/v1/reconciliation/resolve",
            json={"discrepancy_ids": ["d1"]},
        )
        mock_service.resolve_discrepancies.assert_called_once()

    def test_resolve_503_without_service(self, client_no_service):
        """POST /resolve returns 503 when service is not configured."""
        resp = client_no_service.post(
            "/api/v1/reconciliation/resolve",
            json={},
        )
        assert resp.status_code == 503


# ===================================================================
# 16. GET /api/v1/reconciliation/golden-records (List golden) -- endpoint 16
# ===================================================================


class TestListGoldenRecords:
    """Tests for GET /golden-records."""

    def test_list_golden_records_returns_200(self, client):
        """GET /golden-records returns 200 with golden records list."""
        resp = client.get("/api/v1/reconciliation/golden-records")
        assert resp.status_code == 200
        body = resp.json()
        assert "golden_records" in body["data"]

    def test_list_golden_records_has_count(self, client):
        """GET /golden-records response includes count."""
        resp = client.get("/api/v1/reconciliation/golden-records")
        body = resp.json()
        assert body["data"]["count"] >= 1

    def test_list_golden_records_pagination(self, client, mock_service):
        """GET /golden-records with pagination passes params to service."""
        resp = client.get(
            "/api/v1/reconciliation/golden-records?limit=10&offset=5",
        )
        assert resp.status_code == 200
        mock_service.get_golden_records.assert_called_once()

    def test_list_golden_records_503_without_service(self, client_no_service):
        """GET /golden-records returns 503 when service not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/golden-records")
        assert resp.status_code == 503


# ===================================================================
# 17. GET /api/v1/reconciliation/golden-records/{id} -- endpoint 17
# ===================================================================


class TestGetGoldenRecord:
    """Tests for GET /golden-records/{record_id}."""

    def test_get_golden_record_returns_200(self, client):
        """GET /golden-records/{id} returns 200 when found."""
        resp = client.get("/api/v1/reconciliation/golden-records/golden-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["record_id"] == "golden-001"

    def test_get_golden_record_not_found(self, client, mock_service):
        """GET /golden-records/{id} returns 404 for unknown ID."""
        mock_service.get_golden_record.return_value = None
        resp = client.get(
            "/api/v1/reconciliation/golden-records/nonexistent-id",
        )
        assert resp.status_code == 404

    def test_get_golden_record_503_without_service(self, client_no_service):
        """GET /golden-records/{id} returns 503 when service not configured."""
        resp = client_no_service.get(
            "/api/v1/reconciliation/golden-records/golden-001",
        )
        assert resp.status_code == 503


# ===================================================================
# 18. POST /api/v1/reconciliation/pipeline (Run pipeline) -- endpoint 18
# ===================================================================


class TestRunPipeline:
    """Tests for POST /pipeline."""

    def test_run_pipeline_returns_200(self, client):
        """POST /pipeline returns 200 with pipeline results."""
        resp = client.post(
            "/api/v1/reconciliation/pipeline",
            json={
                "records_a": [{"entity_id": "f1", "period": "Q1", "val": 100}],
                "records_b": [{"entity_id": "f1", "period": "Q1", "val": 105}],
                "match_keys": ["entity_id", "period"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["pipeline_id"] == "pipe-001"
        assert body["data"]["status"] == "completed"

    def test_run_pipeline_default_body(self, client):
        """POST /pipeline with minimal body uses defaults."""
        resp = client.post(
            "/api/v1/reconciliation/pipeline",
            json={},
        )
        assert resp.status_code == 200

    def test_run_pipeline_calls_service(self, client, mock_service):
        """POST /pipeline delegates to service.run_pipeline."""
        client.post(
            "/api/v1/reconciliation/pipeline",
            json={"records_a": [], "records_b": []},
        )
        mock_service.run_pipeline.assert_called_once()

    def test_run_pipeline_503_without_service(self, client_no_service):
        """POST /pipeline returns 503 when service is not configured."""
        resp = client_no_service.post(
            "/api/v1/reconciliation/pipeline",
            json={},
        )
        assert resp.status_code == 503


# ===================================================================
# 19. GET /api/v1/reconciliation/health (Health check) -- endpoint 19
# ===================================================================


class TestHealthCheck:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """GET /health returns 200 with health data."""
        resp = client.get("/api/v1/reconciliation/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["status"] == "healthy"

    def test_health_includes_service_name(self, client):
        """GET /health response includes correct service name."""
        resp = client.get("/api/v1/reconciliation/health")
        body = resp.json()
        assert body["data"]["service"] == "cross_source_reconciliation"

    def test_health_calls_service(self, client, mock_service):
        """GET /health delegates to service.get_health."""
        client.get("/api/v1/reconciliation/health")
        mock_service.get_health.assert_called_once()

    def test_health_503_without_service(self, client_no_service):
        """GET /health returns 503 when service is not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/health")
        assert resp.status_code == 503


# ===================================================================
# 20. GET /api/v1/reconciliation/stats (Statistics) -- endpoint 20
# ===================================================================


class TestStatistics:
    """Tests for GET /stats."""

    def test_stats_returns_200(self, client):
        """GET /stats returns 200 with statistics data."""
        resp = client.get("/api/v1/reconciliation/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert body["data"]["total_jobs"] == 5

    def test_stats_includes_all_counters(self, client):
        """GET /stats response includes all expected counters."""
        resp = client.get("/api/v1/reconciliation/stats")
        body = resp.json()
        data = body["data"]
        for key in (
            "total_jobs", "total_sources", "total_matches",
            "total_comparisons", "total_discrepancies",
            "total_resolutions", "total_golden_records",
            "total_pipelines",
        ):
            assert key in data

    def test_stats_calls_service(self, client, mock_service):
        """GET /stats delegates to service.get_statistics."""
        client.get("/api/v1/reconciliation/stats")
        mock_service.get_statistics.assert_called_once()

    def test_stats_503_without_service(self, client_no_service):
        """GET /stats returns 503 when service is not configured."""
        resp = client_no_service.get("/api/v1/reconciliation/stats")
        assert resp.status_code == 503
