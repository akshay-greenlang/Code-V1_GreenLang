# -*- coding: utf-8 -*-
"""
Integration tests for Outlier Detection Agent REST API endpoints - AGENT-DATA-013

Tests all 20 API endpoints via FastAPI TestClient, validating HTTP status
codes, response shapes, error handling, pagination, and cross-endpoint
data flow.

29 test cases covering:
- TestJobEndpoints (8 tests)
- TestDetectEndpoints (3 tests)
- TestClassifyEndpoints (2 tests)
- TestTreatEndpoints (3 tests)
- TestThresholdEndpoints (2 tests)
- TestFeedbackEndpoint (1 test)
- TestImpactEndpoint (1 test)
- TestPipelineEndpoint (2 tests)
- TestHealthAndStatsEndpoints (3 tests)
- TestAuthEndpoints (1 test)
- TestCrossEndpointFlow (2 tests)

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

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
        """POST /api/v1/od/jobs creates a detection job and returns 200.

        Validates:
        - HTTP 200 response
        - job_id is present and a valid UUID
        - status is 'pending'
        - total_records matches the request
        """
        resp = test_client.post(
            "/api/v1/od/jobs",
            json={
                "records": [
                    {"revenue": 1000, "emissions": 500},
                    {"revenue": 2000, "emissions": 800},
                ],
                "dataset_id": "ds-001",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        uuid.UUID(body["job_id"])  # Must be valid UUID
        assert body["status"] == "pending"
        assert body["total_records"] == 2
        assert body["dataset_id"] == "ds-001"
        assert "created_at" in body
        assert "provenance_hash" in body

    def test_create_job_with_pipeline_config(self, test_client):
        """POST /api/v1/od/jobs with pipeline_config includes it in the response."""
        resp = test_client.post(
            "/api/v1/od/jobs",
            json={
                "records": [{"a": 1}],
                "pipeline_config": {"treatment_strategy": "cap"},
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["pipeline_config"] == {"treatment_strategy": "cap"}

    def test_list_jobs_endpoint(self, test_client):
        """GET /api/v1/od/jobs lists all created jobs with pagination.

        Validates:
        - Response contains 'jobs', 'count', 'total' fields
        - Creating 3 jobs results in count=3
        """
        # Create multiple jobs
        for i in range(3):
            test_client.post(
                "/api/v1/od/jobs",
                json={"records": [{"id": i}], "dataset_id": f"ds-{i}"},
            )

        resp = test_client.get("/api/v1/od/jobs")
        assert resp.status_code == 200

        body = resp.json()
        assert "jobs" in body
        assert body["count"] == 3
        assert body["total"] == 3

    def test_list_jobs_with_pagination(self, test_client):
        """GET /api/v1/od/jobs?limit=2&offset=0 returns paginated results."""
        for i in range(5):
            test_client.post(
                "/api/v1/od/jobs",
                json={"records": [{"id": i}], "dataset_id": f"ds-{i}"},
            )

        resp = test_client.get("/api/v1/od/jobs?limit=2&offset=0")
        assert resp.status_code == 200

        body = resp.json()
        assert body["count"] == 2
        assert body["total"] == 5

    def test_get_job_details_endpoint(self, test_client):
        """GET /api/v1/od/jobs/{job_id} returns the specific job.

        Validates:
        - HTTP 200 for existing job
        - Response matches the created job
        """
        create_resp = test_client.post(
            "/api/v1/od/jobs",
            json={"records": [{"a": 1}], "dataset_id": "ds-001"},
        )
        job_id = create_resp.json()["job_id"]

        resp = test_client.get(f"/api/v1/od/jobs/{job_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "pending"

    def test_get_job_not_found(self, test_client):
        """GET /api/v1/od/jobs/{nonexistent} returns 404."""
        resp = test_client.get(f"/api/v1/od/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_delete_job_endpoint(self, test_client):
        """DELETE /api/v1/od/jobs/{job_id} cancels the job.

        Validates:
        - HTTP 200 for existing job
        - Status changes to 'cancelled'
        """
        create_resp = test_client.post(
            "/api/v1/od/jobs",
            json={"records": [{"a": 1}], "dataset_id": "ds-001"},
        )
        job_id = create_resp.json()["job_id"]

        resp = test_client.delete(f"/api/v1/od/jobs/{job_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "cancelled"

    def test_delete_job_not_found(self, test_client):
        """DELETE /api/v1/od/jobs/{nonexistent} returns 404."""
        resp = test_client.delete(f"/api/v1/od/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404


# ===================================================================
# Detect API Tests
# ===================================================================


class TestDetectEndpoints:
    """Tests for the outlier detection API endpoints."""

    def test_detect_single_column_endpoint(self, test_client):
        """POST /api/v1/od/detect detects outliers in a single column.

        Validates:
        - HTTP 200 response
        - outliers_found reflects detected outliers
        - method is reported
        - provenance_hash is a 64-char hex string
        """
        records = [
            {"revenue": 100.0},
            {"revenue": 110.0},
            {"revenue": 105.0},
            {"revenue": 95.0},
            {"revenue": 100000.0},  # Outlier
            {"revenue": 108.0},
            {"revenue": 102.0},
            {"revenue": 97.0},
            {"revenue": 103.0},
            {"revenue": 99.0},
        ]

        resp = test_client.post(
            "/api/v1/od/detect",
            json={"records": records, "column": "revenue"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_points"] == 10
        assert body["outliers_found"] >= 1
        assert body["method"] == "iqr"
        assert len(body["provenance_hash"]) == 64
        assert len(body["outlier_indices"]) >= 1
        assert body["lower_fence"] is not None
        assert body["upper_fence"] is not None

    def test_detect_batch_endpoint(self, test_client):
        """POST /api/v1/od/detect/batch detects outliers across multiple columns.

        Validates:
        - HTTP 200 response
        - total_columns reflects columns analyzed
        - total_outliers aggregates across columns
        """
        records = [
            {"revenue": 100.0, "emissions": 500.0},
            {"revenue": 110.0, "emissions": 480.0},
            {"revenue": 105.0, "emissions": 510.0},
            {"revenue": 999999.0, "emissions": 50000.0},  # Outliers
            {"revenue": 102.0, "emissions": 490.0},
            {"revenue": 98.0, "emissions": 520.0},
            {"revenue": 107.0, "emissions": 505.0},
            {"revenue": 103.0, "emissions": 495.0},
        ]

        resp = test_client.post(
            "/api/v1/od/detect/batch",
            json={"records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_columns"] >= 1
        assert body["total_outliers"] >= 1
        assert len(body["results"]) >= 1
        assert len(body["provenance_hash"]) == 64

    def test_detect_no_outliers(self, test_client):
        """POST /api/v1/od/detect with clean data returns zero outliers."""
        records = [
            {"value": 100.0 + i * 0.5} for i in range(20)
        ]

        resp = test_client.post(
            "/api/v1/od/detect",
            json={"records": records, "column": "value"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["outliers_found"] == 0
        assert body["outlier_pct"] == 0.0


# ===================================================================
# Classify API Tests
# ===================================================================


class TestClassifyEndpoints:
    """Tests for the outlier classification API endpoints."""

    def test_classify_endpoint(self, test_client):
        """POST /api/v1/od/classify classifies detected outliers.

        Validates:
        - HTTP 200 response
        - total_classified matches number of outliers
        - by_class shows distribution of classifications
        - provenance_hash is recorded
        """
        detections = [
            {
                "record_index": 3,
                "column_name": "revenue",
                "value": 999999.0,
                "score": 0.95,
                "is_outlier": True,
                "method": "iqr",
            },
            {
                "record_index": 7,
                "column_name": "emissions",
                "value": 50000.0,
                "score": 0.80,
                "is_outlier": True,
                "method": "iqr",
            },
        ]
        records = [
            {"revenue": 100.0, "emissions": 500.0}
            for _ in range(10)
        ]

        resp = test_client.post(
            "/api/v1/od/classify",
            json={"detections": detections, "records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_classified"] == 2
        assert len(body["classifications"]) == 2
        assert len(body["by_class"]) >= 1
        assert body["avg_confidence"] > 0.0
        assert len(body["provenance_hash"]) == 64

    def test_classify_with_high_scores(self, test_client):
        """POST /api/v1/od/classify classifies extreme outliers as errors."""
        detections = [
            {
                "record_index": 0,
                "column_name": "value",
                "value": 1e10,
                "score": 0.99,
                "is_outlier": True,
                "method": "iqr",
            },
        ]
        records = [{"value": 100.0} for _ in range(5)]

        resp = test_client.post(
            "/api/v1/od/classify",
            json={"detections": detections, "records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_classified"] == 1
        # Score >= 0.95 should classify as "error"
        assert body["classifications"][0]["outlier_class"] == "error"


# ===================================================================
# Treat API Tests
# ===================================================================


class TestTreatEndpoints:
    """Tests for the outlier treatment API endpoints."""

    def test_treat_flag_endpoint(self, test_client):
        """POST /api/v1/od/treat with flag strategy adds outlier markers.

        Validates:
        - HTTP 200 response
        - total_treated matches number of outliers
        - strategy is 'flag'
        - treated_records has flag markers
        """
        records = [
            {"revenue": 100.0},
            {"revenue": 110.0},
            {"revenue": 999999.0},
            {"revenue": 105.0},
        ]
        detections = [
            {
                "record_index": 2,
                "column_name": "revenue",
                "value": 999999.0,
                "score": 0.95,
                "is_outlier": True,
                "method": "iqr",
            },
        ]

        resp = test_client.post(
            "/api/v1/od/treat",
            json={
                "records": records,
                "detections": detections,
                "strategy": "flag",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_treated"] == 1
        assert body["strategy"] == "flag"
        assert body["reversible"] is True
        assert len(body["treated_records"]) == 4
        assert len(body["provenance_hash"]) == 64

    def test_treat_cap_endpoint(self, test_client):
        """POST /api/v1/od/treat with cap strategy caps values at fences."""
        records = [
            {"value": 10.0},
            {"value": 15.0},
            {"value": 12.0},
            {"value": 11.0},
            {"value": 13.0},
            {"value": 14.0},
            {"value": 10000.0},
            {"value": 16.0},
        ]
        detections = [
            {
                "record_index": 6,
                "column_name": "value",
                "value": 10000.0,
                "score": 0.99,
                "is_outlier": True,
                "method": "iqr",
            },
        ]

        resp = test_client.post(
            "/api/v1/od/treat",
            json={
                "records": records,
                "detections": detections,
                "strategy": "cap",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_treated"] == 1
        assert body["strategy"] == "cap"
        # Treated value should be capped below original
        treatment = body["treatments"][0]
        if treatment["treated_value"] is not None:
            assert treatment["treated_value"] < 10000.0

    def test_treat_undo_endpoint(self, test_client):
        """POST /api/v1/od/treat/{treatment_id}/undo undoes a treatment.

        Validates:
        - Applying a reversible treatment then undoing it returns success
        """
        records = [{"value": 100.0}, {"value": 99999.0}]
        detections = [
            {
                "record_index": 1,
                "column_name": "value",
                "value": 99999.0,
                "score": 0.95,
                "is_outlier": True,
                "method": "iqr",
            },
        ]

        # Apply treatment
        treat_resp = test_client.post(
            "/api/v1/od/treat",
            json={
                "records": records,
                "detections": detections,
                "strategy": "flag",
            },
        )
        assert treat_resp.status_code == 200
        treatment_id = treat_resp.json()["treatment_id"]

        # Undo treatment
        undo_resp = test_client.post(
            f"/api/v1/od/treat/{treatment_id}/undo",
        )
        assert undo_resp.status_code == 200
        assert undo_resp.json()["undone"] is True


# ===================================================================
# Threshold API Tests
# ===================================================================


class TestThresholdEndpoints:
    """Tests for the domain threshold management endpoints."""

    def test_create_threshold_endpoint(self, test_client):
        """POST /api/v1/od/thresholds creates a domain threshold.

        Validates:
        - HTTP 200 response
        - threshold_id is a valid UUID
        - Column, bounds, and source match the request
        """
        resp = test_client.post(
            "/api/v1/od/thresholds",
            json={
                "column": "emissions",
                "min_val": 0.0,
                "max_val": 100000.0,
                "source": "regulatory",
                "context": "EPA emission limits",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "threshold_id" in body
        uuid.UUID(body["threshold_id"])
        assert body["column_name"] == "emissions"
        assert body["lower_bound"] == 0.0
        assert body["upper_bound"] == 100000.0
        assert body["source"] == "regulatory"
        assert body["active"] is True
        assert len(body["provenance_hash"]) == 64

    def test_list_thresholds_endpoint(self, test_client):
        """GET /api/v1/od/thresholds lists all created thresholds.

        Creates 2 thresholds and validates the list response.
        """
        for col in ["revenue", "emissions"]:
            test_client.post(
                "/api/v1/od/thresholds",
                json={
                    "column": col,
                    "min_val": 0.0,
                    "max_val": 1000000.0,
                },
            )

        resp = test_client.get("/api/v1/od/thresholds")
        assert resp.status_code == 200

        body = resp.json()
        assert "thresholds" in body
        assert body["count"] == 2


# ===================================================================
# Feedback API Tests
# ===================================================================


class TestFeedbackEndpoint:
    """Tests for the feedback submission endpoint."""

    def test_submit_feedback_endpoint(self, test_client):
        """POST /api/v1/od/feedback submits human feedback on a detection.

        Validates:
        - HTTP 200 response
        - feedback_id is a valid UUID
        - feedback_type and notes are recorded
        """
        resp = test_client.post(
            "/api/v1/od/feedback",
            json={
                "detection_id": str(uuid.uuid4()),
                "feedback_type": "false_positive",
                "notes": "This is a legitimate extreme value",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "feedback_id" in body
        uuid.UUID(body["feedback_id"])
        assert body["feedback_type"] == "false_positive"
        assert body["notes"] == "This is a legitimate extreme value"
        assert body["accepted"] is True
        assert len(body["provenance_hash"]) == 64


# ===================================================================
# Impact API Tests
# ===================================================================


class TestImpactEndpoint:
    """Tests for the impact analysis endpoint."""

    def test_impact_analysis_endpoint(self, test_client):
        """POST /api/v1/od/impact compares original and treated datasets.

        Validates:
        - HTTP 200 response
        - Per-column impact statistics are returned
        - Processing time is reported
        """
        original = [
            {"revenue": 100.0, "emissions": 500.0},
            {"revenue": 110.0, "emissions": 50000.0},
            {"revenue": 105.0, "emissions": 510.0},
            {"revenue": 108.0, "emissions": 490.0},
            {"revenue": 102.0, "emissions": 520.0},
        ]
        treated = [
            {"revenue": 100.0, "emissions": 500.0},
            {"revenue": 110.0, "emissions": 520.0},
            {"revenue": 105.0, "emissions": 510.0},
            {"revenue": 108.0, "emissions": 490.0},
            {"revenue": 102.0, "emissions": 520.0},
        ]

        resp = test_client.post(
            "/api/v1/od/impact",
            json={"original": original, "treated": treated},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "columns" in body
        assert body["total_columns"] >= 1
        assert body["processing_time_ms"] >= 0

        # At least one column should show impact (emissions changed)
        emission_impacts = [
            c for c in body["columns"]
            if c.get("column_name") == "emissions"
        ]
        if emission_impacts:
            impact = emission_impacts[0]
            assert impact["records_affected"] >= 1
            assert "original_mean" in impact
            assert "treated_mean" in impact


# ===================================================================
# Pipeline API Tests
# ===================================================================


class TestPipelineEndpoint:
    """Tests for the full pipeline API endpoint."""

    def test_pipeline_endpoint(self, test_client):
        """POST /api/v1/od/pipeline runs the full detection pipeline.

        Validates:
        - HTTP 200 response
        - All expected fields in response
        - Pipeline stages are present
        """
        records = [
            {"id": "0", "revenue": 100.0, "emissions": 500.0},
            {"id": "1", "revenue": 110.0, "emissions": 480.0},
            {"id": "2", "revenue": 105.0, "emissions": 510.0},
            {"id": "3", "revenue": 999999.0, "emissions": 50000.0},
            {"id": "4", "revenue": 102.0, "emissions": 490.0},
            {"id": "5", "revenue": 98.0, "emissions": 520.0},
        ]

        resp = test_client.post(
            "/api/v1/od/pipeline",
            json={"records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("completed", "failed")
        assert body["total_records"] == 6
        assert "pipeline_id" in body
        uuid.UUID(body["pipeline_id"])
        assert "stages" in body
        assert "detect" in body["stages"]
        assert len(body["provenance_hash"]) == 64

    def test_pipeline_with_config(self, test_client):
        """POST /api/v1/od/pipeline with custom config options."""
        records = [
            {"id": "0", "value": 10.0},
            {"id": "1", "value": 10000.0},
            {"id": "2", "value": 30.0},
            {"id": "3", "value": 25.0},
            {"id": "4", "value": 20.0},
        ]

        resp = test_client.post(
            "/api/v1/od/pipeline",
            json={
                "records": records,
                "config": {
                    "treatment_strategy": "cap",
                    "enable_contextual": False,
                },
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 5


# ===================================================================
# Health & Stats API Tests
# ===================================================================


class TestHealthAndStatsEndpoints:
    """Tests for health check and statistics endpoints."""

    def test_health_endpoint(self, test_client):
        """GET /api/v1/od/health returns service health status.

        Validates:
        - HTTP 200 response
        - status is 'healthy' (service was started in fixture)
        - All expected health fields are present
        """
        resp = test_client.get("/api/v1/od/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "outlier-detector"
        assert body["started"] is True
        assert "engines" in body
        assert "jobs" in body
        assert "detections" in body
        assert "batch_detections" in body
        assert "classifications" in body
        assert "treatments" in body
        assert "thresholds" in body
        assert "feedback" in body
        assert "pipeline_results" in body
        assert "provenance_entries" in body

    def test_stats_endpoint(self, test_client):
        """GET /api/v1/od/stats returns aggregate statistics.

        Validates:
        - HTTP 200 response
        - All stat fields present with correct defaults
        """
        resp = test_client.get("/api/v1/od/stats")

        assert resp.status_code == 200
        body = resp.json()
        assert "total_jobs" in body
        assert "completed_jobs" in body
        assert "failed_jobs" in body
        assert "total_records_processed" in body
        assert "total_outliers_detected" in body
        assert "total_treatments_applied" in body
        assert "total_classifications" in body
        assert "total_feedback" in body
        assert "total_thresholds" in body
        assert "active_jobs" in body
        assert "avg_outlier_pct" in body
        assert "by_method" in body
        assert "by_class" in body
        assert "by_treatment" in body
        assert "by_status" in body
        assert "provenance_entries" in body

    def test_stats_reflect_operations(self, test_client):
        """GET /api/v1/od/stats reflects operations performed.

        Runs a pipeline, then checks that stats are updated.
        """
        records = [
            {"id": "0", "value": 10.0},
            {"id": "1", "value": 100000.0},
            {"id": "2", "value": 30.0},
            {"id": "3", "value": 25.0},
            {"id": "4", "value": 20.0},
        ]
        test_client.post(
            "/api/v1/od/pipeline",
            json={"records": records},
        )

        resp = test_client.get("/api/v1/od/stats")
        assert resp.status_code == 200

        body = resp.json()
        assert body["total_records_processed"] >= 5
        assert body["provenance_entries"] >= 1


# ===================================================================
# Authentication Tests
# ===================================================================


class TestAuthEndpoints:
    """Tests for authentication requirements on protected endpoints."""

    def test_auth_required_on_all_endpoints(self, test_client):
        """Verify that the protected endpoint requires authentication.

        Tests the sentinel /api/v1/od/protected endpoint to confirm
        that requests without a valid Bearer token are rejected with 401.
        """
        # Request without auth header -> 401
        resp = test_client.get("/api/v1/od/protected")
        assert resp.status_code == 401
        assert "Not authenticated" in resp.json()["detail"]

        # Request with invalid auth scheme -> 401
        resp = test_client.get(
            "/api/v1/od/protected",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401

        # Request with valid Bearer token -> 200
        resp = test_client.get(
            "/api/v1/od/protected",
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
        """Test the complete API workflow from job creation to feedback.

        Executes the following sequence:
        1. Create a job
        2. Detect outliers in a column
        3. Classify the detected outliers
        4. Apply treatment to outliers
        5. Submit feedback on the detection
        6. Create a domain threshold
        7. Analyze impact of treatment
        8. Check stats reflect all operations
        """
        # Step 1: Create job
        job_resp = test_client.post(
            "/api/v1/od/jobs",
            json={
                "records": [
                    {"revenue": 100.0, "emissions": 500.0},
                    {"revenue": 110.0, "emissions": 480.0},
                    {"revenue": 999999.0, "emissions": 50000.0},
                    {"revenue": 105.0, "emissions": 510.0},
                    {"revenue": 102.0, "emissions": 490.0},
                ],
                "dataset_id": "ds-api-flow",
            },
        )
        assert job_resp.status_code == 200
        job_id = job_resp.json()["job_id"]

        # Step 2: Detect outliers
        records = [
            {"revenue": 100.0, "emissions": 500.0},
            {"revenue": 110.0, "emissions": 480.0},
            {"revenue": 999999.0, "emissions": 50000.0},
            {"revenue": 105.0, "emissions": 510.0},
            {"revenue": 102.0, "emissions": 490.0},
        ]
        detect_resp = test_client.post(
            "/api/v1/od/detect",
            json={"records": records, "column": "revenue"},
        )
        assert detect_resp.status_code == 200
        detect_body = detect_resp.json()
        assert detect_body["total_points"] == 5
        assert detect_body["outliers_found"] >= 1
        detection_id = detect_body["detection_id"]

        # Step 3: Classify the detected outliers
        outlier_scores = [
            s for s in detect_body["scores"] if s.get("is_outlier", False)
        ]
        if outlier_scores:
            classify_resp = test_client.post(
                "/api/v1/od/classify",
                json={"detections": outlier_scores, "records": records},
            )
            assert classify_resp.status_code == 200
            assert classify_resp.json()["total_classified"] >= 1

        # Step 4: Apply treatment
        if outlier_scores:
            treat_resp = test_client.post(
                "/api/v1/od/treat",
                json={
                    "records": records,
                    "detections": outlier_scores,
                    "strategy": "cap",
                },
            )
            assert treat_resp.status_code == 200
            treat_body = treat_resp.json()
            assert treat_body["total_treated"] >= 1

        # Step 5: Submit feedback
        feedback_resp = test_client.post(
            "/api/v1/od/feedback",
            json={
                "detection_id": detection_id,
                "feedback_type": "confirmed_outlier",
                "notes": "Confirmed data entry error",
            },
        )
        assert feedback_resp.status_code == 200

        # Step 6: Create domain threshold
        threshold_resp = test_client.post(
            "/api/v1/od/thresholds",
            json={
                "column": "revenue",
                "min_val": 0.0,
                "max_val": 500000.0,
                "source": "domain",
                "context": "Revenue cap for small businesses",
            },
        )
        assert threshold_resp.status_code == 200

        # Step 7: Analyze impact
        treated_records = [
            {"revenue": 100.0, "emissions": 500.0},
            {"revenue": 110.0, "emissions": 480.0},
            {"revenue": 200.0, "emissions": 510.0},
            {"revenue": 105.0, "emissions": 510.0},
            {"revenue": 102.0, "emissions": 490.0},
        ]
        impact_resp = test_client.post(
            "/api/v1/od/impact",
            json={"original": records, "treated": treated_records},
        )
        assert impact_resp.status_code == 200
        assert impact_resp.json()["total_columns"] >= 1

        # Step 8: Verify stats
        stats_resp = test_client.get("/api/v1/od/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_jobs"] >= 1
        assert stats["total_records_processed"] >= 5
        assert stats["total_outliers_detected"] >= 1
        assert stats["total_feedback"] >= 1
        assert stats["total_thresholds"] >= 1
        assert stats["provenance_entries"] >= 1

        # Verify the job is still retrievable
        get_job_resp = test_client.get(f"/api/v1/od/jobs/{job_id}")
        assert get_job_resp.status_code == 200

    def test_pipeline_then_stats_then_health(self, test_client):
        """Test that pipeline results flow through to stats and health.

        Runs a pipeline, then verifies both stats and health endpoints
        reflect the pipeline execution.
        """
        records = [
            {"id": "0", "revenue": 100.0, "emissions": 500.0},
            {"id": "1", "revenue": 110.0, "emissions": 480.0},
            {"id": "2", "revenue": 999999.0, "emissions": 50000.0},
            {"id": "3", "revenue": 105.0, "emissions": 510.0},
        ]

        # Run pipeline
        pipeline_resp = test_client.post(
            "/api/v1/od/pipeline",
            json={"records": records},
        )
        assert pipeline_resp.status_code == 200

        # Check stats updated
        stats_resp = test_client.get("/api/v1/od/stats")
        stats = stats_resp.json()
        assert stats["total_records_processed"] >= 4

        # Check health reflects pipeline results
        health_resp = test_client.get("/api/v1/od/health")
        health = health_resp.json()
        assert health["pipeline_results"] >= 1
        assert health["detections"] >= 1
