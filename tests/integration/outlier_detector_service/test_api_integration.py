# -*- coding: utf-8 -*-
"""
Integration tests for Outlier Detection REST API endpoints - AGENT-DATA-013

Tests all 16+ API endpoints via FastAPI TestClient, validating HTTP status
codes, response shapes, error handling, pagination, and cross-endpoint
data flow.

20 test cases covering:
- test_create_job_endpoint
- test_create_job_with_columns
- test_list_jobs_endpoint
- test_list_jobs_with_pagination
- test_get_job_details_endpoint
- test_get_job_not_found
- test_delete_job_endpoint
- test_delete_job_not_found
- test_detect_endpoint
- test_detect_batch_endpoint
- test_detect_empty_records_400
- test_classify_endpoint
- test_treat_endpoint
- test_create_threshold_endpoint
- test_list_thresholds_endpoint
- test_submit_feedback_endpoint
- test_run_pipeline_endpoint
- test_pipeline_with_methods
- test_pipeline_empty_records_400
- test_health_endpoint
- test_stats_endpoint
- test_stats_reflect_operations
- test_auth_required_on_protected_endpoint
- test_full_api_workflow
- test_pipeline_then_stats_then_health

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection Agent (GL-DATA-X-016)
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# Job Management API Tests
# ===================================================================


class TestJobEndpoints:
    """Tests for job CRUD API endpoints."""

    def test_create_job_endpoint(self, test_client):
        """POST /api/v1/outlier/jobs creates an outlier detection job.

        The service create_job returns status 'pending' by default.
        The conftest route handler adds 'dataset_ids' to the response
        for test convenience.
        """
        resp = test_client.post(
            "/api/v1/outlier/jobs",
            json={"dataset_ids": ["ds-001", "ds-002"]},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        uuid.UUID(body["job_id"])
        assert body["status"] == "pending"
        assert body["dataset_ids"] == ["ds-001", "ds-002"]
        assert "created_at" in body

    def test_create_job_with_columns(self, test_client):
        """POST /api/v1/outlier/jobs with columns parameter."""
        resp = test_client.post(
            "/api/v1/outlier/jobs",
            json={
                "dataset_ids": ["ds-001"],
                "columns": ["emissions", "temperature"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "pending"

    def test_list_jobs_endpoint(self, test_client):
        """GET /api/v1/outlier/jobs lists all created jobs."""
        # Create multiple jobs
        for i in range(3):
            test_client.post(
                "/api/v1/outlier/jobs",
                json={"dataset_ids": [f"ds-{i}"]},
            )

        resp = test_client.get("/api/v1/outlier/jobs")
        assert resp.status_code == 200

        body = resp.json()
        assert "jobs" in body
        assert body["count"] >= 3
        assert body["total"] >= 3

    def test_list_jobs_with_pagination(self, test_client):
        """GET /api/v1/outlier/jobs?limit=2&offset=0 returns paginated results."""
        for i in range(5):
            test_client.post(
                "/api/v1/outlier/jobs",
                json={"dataset_ids": [f"ds-{i}"]},
            )

        resp = test_client.get("/api/v1/outlier/jobs?limit=2&offset=0")
        assert resp.status_code == 200

        body = resp.json()
        assert body["count"] == 2
        assert body["total"] >= 5

    def test_get_job_details_endpoint(self, test_client):
        """GET /api/v1/outlier/jobs/{job_id} returns the specific job."""
        create_resp = test_client.post(
            "/api/v1/outlier/jobs",
            json={"dataset_ids": ["ds-001"]},
        )
        job_id = create_resp.json()["job_id"]

        resp = test_client.get(f"/api/v1/outlier/jobs/{job_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "pending"

    def test_get_job_not_found(self, test_client):
        """GET /api/v1/outlier/jobs/{nonexistent} returns 404."""
        resp = test_client.get(f"/api/v1/outlier/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_delete_job_endpoint(self, test_client):
        """DELETE /api/v1/outlier/jobs/{job_id} deletes (cancels) the job.

        The service.delete_job() sets status to 'cancelled' and returns True.
        The conftest route handler then fetches the updated job and returns it.
        """
        create_resp = test_client.post(
            "/api/v1/outlier/jobs",
            json={"dataset_ids": ["ds-001"]},
        )
        job_id = create_resp.json()["job_id"]

        resp = test_client.delete(f"/api/v1/outlier/jobs/{job_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "cancelled"

    def test_delete_job_not_found(self, test_client):
        """DELETE /api/v1/outlier/jobs/{nonexistent} returns 404."""
        resp = test_client.delete(f"/api/v1/outlier/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404


# ===================================================================
# Detection API Tests
# ===================================================================


class TestDetectEndpoints:
    """Tests for outlier detection API endpoints."""

    def test_detect_endpoint(self, test_client):
        """POST /api/v1/outlier/detect runs single-column detection.

        Validates:
        - HTTP 200 response
        - total_points matches input length
        - outliers_found >= 1 (500.0 is outlier)
        - provenance_hash is 64-char hex
        """
        records = [
            {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
            {"val": 500.0}, {"val": 9.0}, {"val": 13.0},
            {"val": 11.5}, {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
        ]

        resp = test_client.post(
            "/api/v1/outlier/detect",
            json={"records": records, "column": "val"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_points"] == 10
        assert body["column_name"] == "val"
        assert body["outliers_found"] >= 1
        assert len(body["provenance_hash"]) == 64
        assert len(body["scores"]) == 10

    def test_detect_batch_endpoint(self, test_client):
        """POST /api/v1/outlier/detect/batch runs multi-column detection.

        Validates:
        - HTTP 200 response
        - total_columns reflects columns analyzed
        - Valid provenance hash
        """
        records = [
            {"emissions": 10.0, "temperature": 22.0},
            {"emissions": 12.0, "temperature": 23.0},
            {"emissions": 500.0, "temperature": 22.5},
            {"emissions": 9.0, "temperature": -50.0},
            {"emissions": 13.0, "temperature": 21.0},
        ]

        resp = test_client.post(
            "/api/v1/outlier/detect/batch",
            json={
                "records": records,
                "columns": ["emissions", "temperature"],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_columns"] >= 1
        assert len(body["provenance_hash"]) == 64

    def test_detect_empty_records_400(self, test_client):
        """POST /api/v1/outlier/detect with empty records returns 400."""
        resp = test_client.post(
            "/api/v1/outlier/detect",
            json={"records": [], "column": "val"},
        )
        assert resp.status_code == 400


# ===================================================================
# Classification API Tests
# ===================================================================


class TestClassifyEndpoint:
    """Tests for the classify API endpoint."""

    def test_classify_endpoint(self, test_client):
        """POST /api/v1/outlier/classify classifies detected outliers.

        First detects outliers, then classifies them into categories.
        """
        records = [
            {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
            {"val": 500.0}, {"val": 9.0}, {"val": 13.0},
            {"val": 11.5}, {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
        ]

        # First detect
        detect_resp = test_client.post(
            "/api/v1/outlier/detect",
            json={"records": records, "column": "val"},
        )
        assert detect_resp.status_code == 200
        detections = detect_resp.json()["scores"]

        # Then classify
        resp = test_client.post(
            "/api/v1/outlier/classify",
            json={"records": records, "detections": detections},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "classifications" in body
        assert len(body["provenance_hash"]) == 64


# ===================================================================
# Treatment API Tests
# ===================================================================


class TestTreatEndpoint:
    """Tests for the treatment API endpoint."""

    def test_treat_endpoint(self, test_client):
        """POST /api/v1/outlier/treat applies treatment to outliers.

        First detects outliers, then applies flag treatment.
        """
        records = [
            {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
            {"val": 500.0}, {"val": 9.0}, {"val": 13.0},
            {"val": 11.5}, {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
        ]

        # First detect
        detect_resp = test_client.post(
            "/api/v1/outlier/detect",
            json={"records": records, "column": "val"},
        )
        detections = detect_resp.json()["scores"]

        # Apply treatment
        resp = test_client.post(
            "/api/v1/outlier/treat",
            json={
                "records": records,
                "detections": detections,
                "strategy": "flag",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "treatments" in body
        assert len(body["provenance_hash"]) == 64


# ===================================================================
# Threshold API Tests
# ===================================================================


class TestThresholdEndpoints:
    """Tests for threshold CRUD endpoints."""

    def test_create_threshold_endpoint(self, test_client):
        """POST /api/v1/outlier/thresholds creates a domain threshold."""
        resp = test_client.post(
            "/api/v1/outlier/thresholds",
            json={
                "column_name": "emissions",
                "lower_bound": 0.0,
                "upper_bound": 100.0,
                "source": "domain",
                "description": "Max expected emissions",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["column_name"] == "emissions"
        assert body["upper_bound"] == 100.0

    def test_list_thresholds_endpoint(self, test_client):
        """GET /api/v1/outlier/thresholds lists all thresholds."""
        # Create a threshold first
        test_client.post(
            "/api/v1/outlier/thresholds",
            json={
                "column_name": "temperature",
                "lower_bound": -40.0,
                "upper_bound": 60.0,
            },
        )

        resp = test_client.get("/api/v1/outlier/thresholds")
        assert resp.status_code == 200

        body = resp.json()
        assert "thresholds" in body
        assert body["count"] >= 1


# ===================================================================
# Feedback API Tests
# ===================================================================


class TestFeedbackEndpoint:
    """Tests for the feedback API endpoint."""

    def test_submit_feedback_endpoint(self, test_client):
        """POST /api/v1/outlier/feedback submits human-in-the-loop feedback.

        The conftest route maps 'comment' from the request body to 'notes'
        when calling service.submit_feedback().
        """
        # First detect to get a detection_id
        records = [
            {"val": 10.0}, {"val": 12.0}, {"val": 500.0},
            {"val": 9.0}, {"val": 13.0},
        ]
        detect_resp = test_client.post(
            "/api/v1/outlier/detect",
            json={"records": records, "column": "val"},
        )
        detection_id = detect_resp.json()["detection_id"]

        resp = test_client.post(
            "/api/v1/outlier/feedback",
            json={
                "detection_id": detection_id,
                "feedback_type": "false_positive",
                "comment": "This is a valid reading",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["detection_id"] == detection_id
        assert body["feedback_type"] == "false_positive"


# ===================================================================
# Pipeline API Tests
# ===================================================================


class TestPipelineEndpoint:
    """Tests for the full pipeline API endpoint."""

    def test_run_pipeline_endpoint(self, test_client):
        """POST /api/v1/outlier/pipeline runs the full detection pipeline.

        Validates:
        - HTTP 200 response
        - All expected fields in response
        - Provenance hash present
        """
        records = [
            {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
            {"val": 500.0}, {"val": 9.0}, {"val": 13.0},
            {"val": 11.5}, {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
        ]

        resp = test_client.post(
            "/api/v1/outlier/pipeline",
            json={"records": records, "columns": ["val"]},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("completed", "failed")
        assert body["total_records"] == 10
        assert "pipeline_id" in body
        uuid.UUID(body["pipeline_id"])
        assert len(body["provenance_hash"]) == 64

    def test_pipeline_with_methods(self, test_client):
        """POST /api/v1/outlier/pipeline with custom methods."""
        records = [
            {"val": 10.0}, {"val": 12.0}, {"val": 500.0},
        ]

        resp = test_client.post(
            "/api/v1/outlier/pipeline",
            json={
                "records": records,
                "columns": ["val"],
                "methods": ["iqr", "zscore"],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 3

    def test_pipeline_empty_records_400(self, test_client):
        """POST /api/v1/outlier/pipeline with empty records returns 400."""
        resp = test_client.post(
            "/api/v1/outlier/pipeline",
            json={"records": []},
        )
        assert resp.status_code == 400


# ===================================================================
# Health & Stats API Tests
# ===================================================================


class TestHealthAndStatsEndpoints:
    """Tests for health check and statistics endpoints."""

    def test_health_endpoint(self, test_client):
        """GET /api/v1/outlier/health returns service health status.

        Validates:
        - HTTP 200 response
        - status is 'healthy' (service._started = True from mock_app)
        - All expected health fields are present
        """
        resp = test_client.get("/api/v1/outlier/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "outlier-detector"
        assert body["started"] is True
        assert "jobs" in body
        assert "detections" in body
        assert "provenance_entries" in body

    def test_stats_endpoint(self, test_client):
        """GET /api/v1/outlier/stats returns aggregate statistics.

        StatsResponse fields: total_jobs, total_records_processed,
        total_outliers_detected, total_feedback, total_thresholds,
        provenance_entries.
        """
        resp = test_client.get("/api/v1/outlier/stats")

        assert resp.status_code == 200
        body = resp.json()
        assert "total_jobs" in body
        assert "total_records_processed" in body
        assert "total_outliers_detected" in body
        assert "total_feedback" in body
        assert "total_thresholds" in body
        assert "provenance_entries" in body

    def test_stats_reflect_operations(self, test_client):
        """GET /api/v1/outlier/stats reflects operations performed.

        Runs a pipeline, then checks that total_records_processed is updated.
        """
        records = [
            {"val": 10.0}, {"val": 12.0}, {"val": 500.0},
            {"val": 9.0}, {"val": 13.0},
        ]
        test_client.post(
            "/api/v1/outlier/pipeline",
            json={"records": records, "columns": ["val"]},
        )

        resp = test_client.get("/api/v1/outlier/stats")
        assert resp.status_code == 200

        body = resp.json()
        assert body["total_records_processed"] >= 5


# ===================================================================
# Authentication Tests
# ===================================================================


class TestAuthEndpoints:
    """Tests for authentication requirements on protected endpoints."""

    def test_auth_required_on_protected_endpoint(self, test_client):
        """Verify that the protected endpoint requires authentication.

        Tests the sentinel /api/v1/outlier/protected endpoint to confirm
        that requests without a valid Bearer token are rejected with 401.
        """
        # Request without auth header -> 401
        resp = test_client.get("/api/v1/outlier/protected")
        assert resp.status_code == 401
        assert "Not authenticated" in resp.json()["detail"]

        # Request with invalid auth scheme -> 401
        resp = test_client.get(
            "/api/v1/outlier/protected",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401

        # Request with valid Bearer token -> 200
        resp = test_client.get(
            "/api/v1/outlier/protected",
            headers={"Authorization": "Bearer valid-test-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ===================================================================
# Cross-Endpoint Data Flow Tests
# ===================================================================


class TestCrossEndpointFlow:
    """Test data flow across multiple endpoints in sequence."""

    def test_full_api_workflow(self, test_client):
        """Test the complete API workflow from job creation to treatment.

        Executes the following sequence:
        1. Create a job
        2. Detect outliers
        3. Classify outliers
        4. Apply treatment
        5. Submit feedback
        6. Create threshold
        7. Check stats reflect all operations
        """
        # Step 1: Create job
        job_resp = test_client.post(
            "/api/v1/outlier/jobs",
            json={"dataset_ids": ["ds-api-flow"]},
        )
        assert job_resp.status_code == 200
        job_id = job_resp.json()["job_id"]

        # Step 2: Detect
        records = [
            {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
            {"val": 500.0}, {"val": 9.0}, {"val": 13.0},
            {"val": 11.5}, {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
        ]
        detect_resp = test_client.post(
            "/api/v1/outlier/detect",
            json={"records": records, "column": "val"},
        )
        assert detect_resp.status_code == 200
        detection_id = detect_resp.json()["detection_id"]
        detections = detect_resp.json()["scores"]

        # Step 3: Classify
        classify_resp = test_client.post(
            "/api/v1/outlier/classify",
            json={"records": records, "detections": detections},
        )
        assert classify_resp.status_code == 200

        # Step 4: Treat
        treat_resp = test_client.post(
            "/api/v1/outlier/treat",
            json={
                "records": records,
                "detections": detections,
                "strategy": "flag",
            },
        )
        assert treat_resp.status_code == 200

        # Step 5: Feedback
        feedback_resp = test_client.post(
            "/api/v1/outlier/feedback",
            json={
                "detection_id": detection_id,
                "feedback_type": "confirmed_outlier",
                "comment": "Validated by domain expert",
            },
        )
        assert feedback_resp.status_code == 200

        # Step 6: Create threshold
        threshold_resp = test_client.post(
            "/api/v1/outlier/thresholds",
            json={
                "column_name": "val",
                "lower_bound": 0.0,
                "upper_bound": 100.0,
            },
        )
        assert threshold_resp.status_code == 200

        # Step 7: Verify stats
        stats_resp = test_client.get("/api/v1/outlier/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_records_processed"] >= 10
        assert stats["total_feedback"] >= 1
        assert stats["total_thresholds"] >= 1

        # Verify the job is still retrievable
        get_job_resp = test_client.get(f"/api/v1/outlier/jobs/{job_id}")
        assert get_job_resp.status_code == 200

    def test_pipeline_then_stats_then_health(self, test_client):
        """Test that pipeline results flow through to stats and health.

        Runs a pipeline, then verifies both stats and health endpoints
        reflect the pipeline execution.
        """
        records = [
            {"val": 10.0}, {"val": 12.0}, {"val": 500.0},
            {"val": 9.0}, {"val": 13.0},
        ]

        # Run pipeline
        pipeline_resp = test_client.post(
            "/api/v1/outlier/pipeline",
            json={"records": records, "columns": ["val"]},
        )
        assert pipeline_resp.status_code == 200

        # Check stats updated
        stats_resp = test_client.get("/api/v1/outlier/stats")
        stats = stats_resp.json()
        assert stats["total_records_processed"] >= 5

        # Check health
        health_resp = test_client.get("/api/v1/outlier/health")
        health = health_resp.json()
        assert health["status"] == "healthy"
        assert health["detections"] >= 0
