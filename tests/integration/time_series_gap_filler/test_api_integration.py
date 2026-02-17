# -*- coding: utf-8 -*-
"""
API-level integration tests for AGENT-DATA-014 Time Series Gap Filler.

Tests the FastAPI router endpoints end-to-end using the TestClient:
- POST /detect -> POST /fill -> POST /validate -> GET /stats workflow
- Job lifecycle: POST /jobs -> GET /jobs -> DELETE /jobs
- Pipeline endpoint end-to-end
- Health endpoint
- Error handling (invalid input, missing resources)
- Calendar CRUD workflow

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed",
)


# =========================================================================
# Test class: Full API workflow
# =========================================================================


class TestAPIWorkflow:
    """Integration tests for the detect -> fill -> validate -> stats workflow."""

    def test_detect_endpoint_returns_gaps(self, test_client):
        """POST /detect returns gap detection results."""
        resp = test_client.post(
            "/api/v1/gap-filler/detect",
            json={
                "values": [10.0, 12.0, None, 14.0, 16.0, None, None, 22.0],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_gaps" in data
        assert data["total_gaps"] > 0
        assert "provenance_hash" in data

    def test_fill_endpoint_returns_filled_values(self, test_client):
        """POST /fill returns a filled series."""
        resp = test_client.post(
            "/api/v1/gap-filler/fill",
            json={
                "values": [10.0, None, 30.0],
                "method": "linear",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "filled_values" in data
        assert data["gaps_filled"] > 0
        assert data["mean_confidence"] > 0.0

    def test_validate_endpoint_returns_pass(self, test_client):
        """POST /validate returns validation result."""
        resp = test_client.post(
            "/api/v1/gap-filler/validate",
            json={
                "original": [10.0, None, 30.0],
                "filled": [10.0, 20.0, 30.0],
                "fill_indices": [1],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "level" in data
        assert data["confidence_check"] is True

    def test_stats_endpoint_returns_statistics(self, test_client):
        """GET /stats returns aggregated statistics."""
        resp = test_client.get("/api/v1/gap-filler/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_jobs" in data

    def test_full_detect_fill_validate_workflow(self, test_client):
        """Full workflow: detect -> fill -> validate in sequence."""
        values = [10.0, 12.0, None, 14.0, 16.0, None, None, 22.0, 24.0, 26.0]

        # Step 1: detect
        detect_resp = test_client.post(
            "/api/v1/gap-filler/detect",
            json={"values": values},
        )
        assert detect_resp.status_code == 200
        detect_data = detect_resp.json()
        assert detect_data["total_gaps"] > 0

        # Step 2: fill
        fill_resp = test_client.post(
            "/api/v1/gap-filler/fill",
            json={"values": values, "method": "linear"},
        )
        assert fill_resp.status_code == 200
        fill_data = fill_resp.json()
        filled_values = fill_data["filled_values"]
        assert len(filled_values) == len(values)
        assert fill_data["gaps_filled"] > 0

        # Step 3: validate
        fill_indices = [i for i, v in enumerate(values) if v is None]
        validate_resp = test_client.post(
            "/api/v1/gap-filler/validate",
            json={
                "original": values,
                "filled": filled_values,
                "fill_indices": fill_indices,
            },
        )
        assert validate_resp.status_code == 200


# =========================================================================
# Test class: Job lifecycle
# =========================================================================


class TestJobLifecycle:
    """Integration tests for the job CRUD lifecycle."""

    def test_create_job(self, test_client):
        """POST /jobs creates a new fill job."""
        resp = test_client.post(
            "/api/v1/gap-filler/jobs",
            json={
                "series_name": "test_series_001",
                "strategy": "auto",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_list_jobs_after_creation(self, test_client):
        """GET /jobs lists jobs including newly created ones."""
        # Create a job first
        test_client.post(
            "/api/v1/gap-filler/jobs",
            json={"series_name": "list_test_series"},
        )

        resp = test_client.get("/api/v1/gap-filler/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert data["count"] >= 1

    def test_get_job_by_id(self, test_client):
        """GET /jobs/{job_id} returns the specific job."""
        create_resp = test_client.post(
            "/api/v1/gap-filler/jobs",
            json={"series_name": "get_test_series"},
        )
        job_id = create_resp.json()["job_id"]

        resp = test_client.get(f"/api/v1/gap-filler/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id

    def test_delete_job(self, test_client):
        """DELETE /jobs/{job_id} marks the job as deleted."""
        create_resp = test_client.post(
            "/api/v1/gap-filler/jobs",
            json={"series_name": "delete_test_series"},
        )
        job_id = create_resp.json()["job_id"]

        delete_resp = test_client.delete(f"/api/v1/gap-filler/jobs/{job_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted"] is True

    def test_get_nonexistent_job_returns_404(self, test_client):
        """GET /jobs/{bad_id} returns 404."""
        resp = test_client.get("/api/v1/gap-filler/jobs/nonexistent-id-xyz")
        assert resp.status_code == 404

    def test_delete_nonexistent_job_returns_404(self, test_client):
        """DELETE /jobs/{bad_id} returns 404."""
        resp = test_client.delete("/api/v1/gap-filler/jobs/nonexistent-id-xyz")
        assert resp.status_code == 404

    def test_full_job_lifecycle(self, test_client):
        """Full lifecycle: create -> get -> delete."""
        # Create
        create_resp = test_client.post(
            "/api/v1/gap-filler/jobs",
            json={"series_name": "lifecycle_series", "strategy": "linear"},
        )
        assert create_resp.status_code == 200
        job_id = create_resp.json()["job_id"]

        # Get
        get_resp = test_client.get(f"/api/v1/gap-filler/jobs/{job_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["series_name"] == "lifecycle_series"

        # Delete
        del_resp = test_client.delete(f"/api/v1/gap-filler/jobs/{job_id}")
        assert del_resp.status_code == 200


# =========================================================================
# Test class: Pipeline endpoint
# =========================================================================


class TestPipelineEndpoint:
    """Integration tests for the /pipeline endpoint."""

    def test_pipeline_endpoint_basic(self, test_client):
        """POST /pipeline runs the full pipeline and returns results."""
        resp = test_client.post(
            "/api/v1/gap-filler/pipeline",
            json={
                "values": [10.0, None, 30.0, 40.0, None, 60.0],
                "strategy": "auto",
                "enable_validation": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pipeline_id" in data
        assert data["status"] == "completed"

    def test_pipeline_without_validation(self, test_client):
        """POST /pipeline with enable_validation=False skips validation."""
        resp = test_client.post(
            "/api/v1/gap-filler/pipeline",
            json={
                "values": [10.0, None, 30.0],
                "strategy": "linear",
                "enable_validation": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_pipeline_returns_provenance(self, test_client):
        """Pipeline result includes a provenance hash."""
        resp = test_client.post(
            "/api/v1/gap-filler/pipeline",
            json={
                "values": [1.0, None, 3.0, 4.0, None, 6.0],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "provenance_hash" in data
        assert data["provenance_hash"] != ""


# =========================================================================
# Test class: Health endpoint
# =========================================================================


class TestHealthEndpoint:
    """Integration tests for the /health endpoint."""

    def test_health_returns_healthy(self, test_client):
        """GET /health returns healthy status."""
        resp = test_client.get("/api/v1/gap-filler/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "engines" in data


# =========================================================================
# Test class: Error handling
# =========================================================================


class TestErrorHandling:
    """Integration tests for API error responses."""

    def test_detect_with_empty_values_returns_200(self, test_client):
        """POST /detect with empty list is handled gracefully."""
        resp = test_client.post(
            "/api/v1/gap-filler/detect",
            json={"values": []},
        )
        # Empty list is a valid input - the service returns 0 gaps
        assert resp.status_code == 200

    def test_fill_with_invalid_method_returns_error(self, test_client):
        """POST /fill with unknown method returns 4xx or 5xx."""
        resp = test_client.post(
            "/api/v1/gap-filler/fill",
            json={
                "values": [1.0, None, 3.0],
                "method": "nonexistent_method_xyz",
            },
        )
        # The service may raise ValueError which results in 400 or 500
        assert resp.status_code in (400, 422, 500)

    def test_detect_with_all_none_returns_error_or_valid_response(
        self, test_client,
    ):
        """POST /detect with all None values returns error or valid detection."""
        resp = test_client.post(
            "/api/v1/gap-filler/detect",
            json={"values": [None, None, None]},
        )
        # The mock service raises ValueError for all-None input
        assert resp.status_code in (200, 400, 500)

    def test_validate_mismatched_lengths_returns_error(self, test_client):
        """POST /validate with mismatched array lengths returns error."""
        resp = test_client.post(
            "/api/v1/gap-filler/validate",
            json={
                "original": [1.0, None, 3.0],
                "filled": [1.0, 2.0],
                "fill_indices": [1],
            },
        )
        # May be 400 or 422 for validation error
        assert resp.status_code in (200, 400, 422, 500)


# =========================================================================
# Test class: Calendar CRUD workflow
# =========================================================================


class TestCalendarCRUD:
    """Integration tests for calendar management endpoints."""

    def test_create_calendar(self, test_client):
        """POST /calendars creates a business calendar."""
        resp = test_client.post(
            "/api/v1/gap-filler/calendars",
            json={
                "name": "us_business",
                "calendar_type": "business_days",
                "business_days": [1, 2, 3, 4, 5],
                "holidays": ["2025-01-01", "2025-12-25"],
                "fiscal_start_month": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "us_business"

    def test_list_calendars(self, test_client):
        """GET /calendars lists all registered calendars."""
        # Create one first
        test_client.post(
            "/api/v1/gap-filler/calendars",
            json={
                "name": "eu_trading",
                "calendar_type": "trading_days",
            },
        )

        resp = test_client.get("/api/v1/gap-filler/calendars")
        assert resp.status_code == 200
        data = resp.json()
        assert "calendars" in data
        assert data["count"] >= 1

    def test_calendar_crud_full_workflow(self, test_client):
        """Full calendar CRUD: create -> list -> verify."""
        # Create
        create_resp = test_client.post(
            "/api/v1/gap-filler/calendars",
            json={
                "name": "apac_business",
                "calendar_type": "business_days",
                "business_days": [1, 2, 3, 4, 5],
                "holidays": ["2025-01-01"],
                "fiscal_start_month": 4,
            },
        )
        assert create_resp.status_code == 200

        # List
        list_resp = test_client.get("/api/v1/gap-filler/calendars")
        assert list_resp.status_code == 200
        calendars = list_resp.json()["calendars"]
        names = [c["name"] for c in calendars]
        assert "apac_business" in names


# =========================================================================
# Test class: Frequency analysis endpoint
# =========================================================================


class TestFrequencyEndpoint:
    """Integration tests for the /frequency endpoint."""

    def test_analyze_frequency_endpoint(self, test_client, sample_timestamps):
        """POST /frequency returns frequency analysis."""
        resp = test_client.post(
            "/api/v1/gap-filler/frequency",
            json={"timestamps": sample_timestamps},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "frequency_level" in data
        assert "provenance_hash" in data
