# -*- coding: utf-8 -*-
"""
Unit tests for Outlier Detection REST API Router - AGENT-DATA-013

Tests all 20 endpoints under /api/v1/outlier including job CRUD,
detect (single + batch), detections listing, classify, treat (+ undo),
thresholds, feedback, impact analysis, pipeline, health, and stats.
Also tests error handling (400, 404, 503).
Target: 40+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.outlier_detector.api.router import FASTAPI_AVAILABLE

# Skip the entire module if FastAPI is not installed
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not available; skipping router tests",
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_service():
    """Create a MagicMock that mimics OutlierDetectorService."""
    from greenlang.outlier_detector.setup import (
        BatchDetectionResponse,
        ClassificationResponse,
        DetectionResponse,
        FeedbackResponse,
        PipelineResponse,
        StatsResponse,
        ThresholdResponse,
        TreatmentResponse,
    )

    svc = MagicMock()

    # Job CRUD
    svc.create_job.return_value = {
        "job_id": "job-001",
        "status": "pending",
        "total_records": 10,
        "provenance_hash": "a" * 64,
    }
    svc.list_jobs.return_value = [
        {"job_id": "job-001", "status": "pending"},
    ]
    svc.get_job.return_value = {
        "job_id": "job-001",
        "status": "pending",
    }
    svc.delete_job.return_value = True

    # Detection
    svc.detect_outliers.return_value = DetectionResponse(
        column_name="val",
        method="iqr",
        total_points=10,
        outliers_found=1,
        outlier_pct=0.1,
        provenance_hash="b" * 64,
    )
    svc.detect_batch.return_value = BatchDetectionResponse(
        total_columns=2,
        total_outliers=3,
        provenance_hash="c" * 64,
    )
    svc.get_detections.return_value = [
        DetectionResponse(
            detection_id="det-001",
            column_name="val",
            provenance_hash="d" * 64,
        ),
    ]
    svc.get_detection.return_value = DetectionResponse(
        detection_id="det-001",
        column_name="val",
        provenance_hash="d" * 64,
    )

    # Classification
    svc.classify_outliers.return_value = ClassificationResponse(
        total_classified=1,
        provenance_hash="e" * 64,
    )
    svc.get_classification.return_value = ClassificationResponse(
        classification_id="cls-001",
        provenance_hash="e" * 64,
    )

    # Treatment
    svc.apply_treatment.return_value = TreatmentResponse(
        strategy="flag",
        total_treated=1,
        provenance_hash="f" * 64,
    )
    svc.get_treatment.return_value = TreatmentResponse(
        treatment_id="treat-001",
        provenance_hash="f" * 64,
    )
    svc.undo_treatment.return_value = True

    # Thresholds
    svc.create_threshold.return_value = ThresholdResponse(
        column_name="val",
        lower_bound=0.0,
        upper_bound=100.0,
        provenance_hash="g" * 64,
    )
    svc.list_thresholds.return_value = [
        ThresholdResponse(
            column_name="val",
            provenance_hash="g" * 64,
        ),
    ]

    # Feedback
    svc.submit_feedback.return_value = FeedbackResponse(
        detection_id="det-001",
        feedback_type="confirmed_outlier",
        provenance_hash="h" * 64,
    )

    # Impact
    svc.analyze_impact.return_value = {
        "columns": [{"column_name": "val", "records_affected": 1}],
        "total_columns": 1,
    }

    # Pipeline
    svc.run_pipeline.return_value = PipelineResponse(
        status="completed",
        total_records=10,
        provenance_hash="i" * 64,
    )

    # Health + Stats
    svc.health_check.return_value = {
        "status": "healthy",
        "service": "outlier-detector",
    }
    svc.get_statistics.return_value = StatsResponse(total_jobs=5)

    return svc


@pytest.fixture
def client(mock_service):
    """Create a FastAPI TestClient with the mock service wired in."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from greenlang.outlier_detector.api.router import router

    app = FastAPI()
    app.include_router(router)
    app.state.outlier_detector_service = mock_service

    return TestClient(app)


@pytest.fixture
def client_no_service():
    """Create a FastAPI TestClient WITHOUT the service configured (for 503 tests)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from greenlang.outlier_detector.api.router import router

    app = FastAPI()
    app.include_router(router)
    # Deliberately do NOT set app.state.outlier_detector_service

    return TestClient(app)


# =========================================================================
# 1. Create job - POST /jobs
# =========================================================================


class TestCreateJob:
    """Tests for POST /api/v1/outlier/jobs."""

    def test_create_job_success(self, client):
        resp = client.post("/api/v1/outlier/jobs", json={
            "records": [{"val": 10.0}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data

    def test_create_job_calls_service(self, client, mock_service):
        client.post("/api/v1/outlier/jobs", json={
            "records": [{"val": 10.0}],
        })
        mock_service.create_job.assert_called_once()

    def test_create_job_passes_dataset_id(self, client, mock_service):
        client.post("/api/v1/outlier/jobs", json={
            "dataset_id": "ds-42",
            "records": [{"val": 10.0}],
        })
        call_kwargs = mock_service.create_job.call_args
        assert call_kwargs[1]["request"]["dataset_id"] == "ds-42"

    def test_create_job_passes_pipeline_config(self, client, mock_service):
        client.post("/api/v1/outlier/jobs", json={
            "records": [{"val": 10.0}],
            "pipeline_config": {"methods": ["iqr"]},
        })
        call_kwargs = mock_service.create_job.call_args
        assert call_kwargs[1]["request"]["pipeline_config"] == {"methods": ["iqr"]}


# =========================================================================
# 2. List jobs - GET /jobs
# =========================================================================


class TestListJobs:
    """Tests for GET /api/v1/outlier/jobs."""

    def test_list_jobs_success(self, client):
        resp = client.get("/api/v1/outlier/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert "count" in data

    def test_list_jobs_with_status_filter(self, client, mock_service):
        resp = client.get("/api/v1/outlier/jobs?status=completed")
        assert resp.status_code == 200
        mock_service.list_jobs.assert_called_once_with(
            status="completed", limit=50, offset=0,
        )

    def test_list_jobs_with_pagination(self, client, mock_service):
        resp = client.get("/api/v1/outlier/jobs?limit=10&offset=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 10
        assert data["offset"] == 5


# =========================================================================
# 3. Get job - GET /jobs/{job_id}
# =========================================================================


class TestGetJob:
    """Tests for GET /api/v1/outlier/jobs/{job_id}."""

    def test_get_job_success(self, client):
        resp = client.get("/api/v1/outlier/jobs/job-001")
        assert resp.status_code == 200

    def test_get_job_not_found(self, client, mock_service):
        mock_service.get_job.return_value = None
        resp = client.get("/api/v1/outlier/jobs/nonexistent")
        assert resp.status_code == 404


# =========================================================================
# 4. Delete job - DELETE /jobs/{job_id}
# =========================================================================


class TestDeleteJob:
    """Tests for DELETE /api/v1/outlier/jobs/{job_id}."""

    def test_delete_job_success(self, client):
        resp = client.delete("/api/v1/outlier/jobs/job-001")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert resp.json()["job_id"] == "job-001"

    def test_delete_job_not_found(self, client, mock_service):
        mock_service.delete_job.return_value = False
        resp = client.delete("/api/v1/outlier/jobs/nonexistent")
        assert resp.status_code == 404


# =========================================================================
# 5. Detect outliers - POST /detect
# =========================================================================


class TestDetectOutliers:
    """Tests for POST /api/v1/outlier/detect."""

    def test_detect_success(self, client):
        resp = client.post("/api/v1/outlier/detect", json={
            "records": [{"val": 10.0}, {"val": 500.0}],
            "column": "val",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "column_name" in data

    def test_detect_calls_service(self, client, mock_service):
        client.post("/api/v1/outlier/detect", json={
            "records": [{"val": 10.0}],
            "column": "val",
        })
        mock_service.detect_outliers.assert_called_once()

    def test_detect_with_methods_and_options(self, client, mock_service):
        resp = client.post("/api/v1/outlier/detect", json={
            "records": [{"val": 10.0}],
            "column": "val",
            "methods": ["iqr", "zscore"],
            "options": {"threshold": 2.0},
        })
        assert resp.status_code == 200
        call_kwargs = mock_service.detect_outliers.call_args[1]
        assert call_kwargs["methods"] == ["iqr", "zscore"]
        assert call_kwargs["options"] == {"threshold": 2.0}


# =========================================================================
# 6. Batch detect - POST /detect/batch
# =========================================================================


class TestBatchDetect:
    """Tests for POST /api/v1/outlier/detect/batch."""

    def test_batch_detect_success(self, client):
        resp = client.post("/api/v1/outlier/detect/batch", json={
            "records": [{"val": 10.0, "temp": 22.0}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "total_columns" in data

    def test_batch_detect_with_columns(self, client, mock_service):
        resp = client.post("/api/v1/outlier/detect/batch", json={
            "records": [{"val": 10.0, "temp": 22.0}],
            "columns": ["val"],
        })
        assert resp.status_code == 200
        call_kwargs = mock_service.detect_batch.call_args[1]
        assert call_kwargs["columns"] == ["val"]


# =========================================================================
# 7. List detections - GET /detections
# =========================================================================


class TestListDetections:
    """Tests for GET /api/v1/outlier/detections."""

    def test_list_detections_success(self, client):
        resp = client.get("/api/v1/outlier/detections")
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert "total" in data

    def test_list_detections_pagination(self, client):
        resp = client.get("/api/v1/outlier/detections?limit=5&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 5
        assert data["offset"] == 0


# =========================================================================
# 8. Get detection - GET /detections/{detection_id}
# =========================================================================


class TestGetDetection:
    """Tests for GET /api/v1/outlier/detections/{detection_id}."""

    def test_get_detection_success(self, client):
        resp = client.get("/api/v1/outlier/detections/det-001")
        assert resp.status_code == 200

    def test_get_detection_not_found(self, client, mock_service):
        mock_service.get_detection.return_value = None
        resp = client.get("/api/v1/outlier/detections/nonexistent")
        assert resp.status_code == 404


# =========================================================================
# 9. Classify outliers - POST /classify
# =========================================================================


class TestClassifyOutliers:
    """Tests for POST /api/v1/outlier/classify."""

    def test_classify_success(self, client):
        resp = client.post("/api/v1/outlier/classify", json={
            "detections": [{"record_index": 0, "score": 0.9, "is_outlier": True}],
            "records": [{"val": 500.0}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "total_classified" in data


# =========================================================================
# 10. Get classification - GET /classify/{id}
# =========================================================================


class TestGetClassification:
    """Tests for GET /api/v1/outlier/classify/{classification_id}."""

    def test_get_classification_success(self, client):
        resp = client.get("/api/v1/outlier/classify/cls-001")
        assert resp.status_code == 200

    def test_get_classification_not_found(self, client, mock_service):
        mock_service.get_classification.return_value = None
        resp = client.get("/api/v1/outlier/classify/nonexistent")
        assert resp.status_code == 404


# =========================================================================
# 11. Apply treatment - POST /treat
# =========================================================================


class TestApplyTreatment:
    """Tests for POST /api/v1/outlier/treat."""

    def test_treat_success(self, client):
        resp = client.post("/api/v1/outlier/treat", json={
            "records": [{"val": 10.0}, {"val": 500.0}],
            "detections": [{"record_index": 1, "score": 0.9, "is_outlier": True}],
            "strategy": "flag",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "strategy" in data

    def test_treat_default_strategy_is_flag(self, client, mock_service):
        resp = client.post("/api/v1/outlier/treat", json={
            "records": [{"val": 500.0}],
            "detections": [{"record_index": 0}],
        })
        assert resp.status_code == 200
        call_kwargs = mock_service.apply_treatment.call_args[1]
        assert call_kwargs["strategy"] == "flag"


# =========================================================================
# 12. Get treatment - GET /treat/{treatment_id}
# =========================================================================


class TestGetTreatment:
    """Tests for GET /api/v1/outlier/treat/{treatment_id}."""

    def test_get_treatment_success(self, client):
        resp = client.get("/api/v1/outlier/treat/treat-001")
        assert resp.status_code == 200

    def test_get_treatment_not_found(self, client, mock_service):
        mock_service.get_treatment.return_value = None
        resp = client.get("/api/v1/outlier/treat/nonexistent")
        assert resp.status_code == 404


# =========================================================================
# 13. Undo treatment - POST /treat/{treatment_id}/undo
# =========================================================================


class TestUndoTreatment:
    """Tests for POST /api/v1/outlier/treat/{treatment_id}/undo."""

    def test_undo_success(self, client):
        resp = client.post("/api/v1/outlier/treat/treat-001/undo")
        assert resp.status_code == 200
        data = resp.json()
        assert data["undone"] is True
        assert data["treatment_id"] == "treat-001"

    def test_undo_not_found(self, client, mock_service):
        mock_service.undo_treatment.return_value = False
        resp = client.post("/api/v1/outlier/treat/nonexistent/undo")
        assert resp.status_code == 404


# =========================================================================
# 14. Create threshold - POST /thresholds
# =========================================================================


class TestCreateThreshold:
    """Tests for POST /api/v1/outlier/thresholds."""

    def test_create_threshold_success(self, client):
        resp = client.post("/api/v1/outlier/thresholds", json={
            "column": "emissions",
            "min_val": 0.0,
            "max_val": 100.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "column_name" in data

    def test_create_threshold_with_source_and_context(self, client, mock_service):
        resp = client.post("/api/v1/outlier/thresholds", json={
            "column": "emissions",
            "min_val": 0.0,
            "max_val": 100.0,
            "source": "regulatory",
            "context": "EU ETS limits",
        })
        assert resp.status_code == 200
        call_kwargs = mock_service.create_threshold.call_args[1]
        assert call_kwargs["source"] == "regulatory"
        assert call_kwargs["context"] == "EU ETS limits"


# =========================================================================
# 15. List thresholds - GET /thresholds
# =========================================================================


class TestListThresholds:
    """Tests for GET /api/v1/outlier/thresholds."""

    def test_list_thresholds_success(self, client):
        resp = client.get("/api/v1/outlier/thresholds")
        assert resp.status_code == 200
        data = resp.json()
        assert "thresholds" in data
        assert "count" in data


# =========================================================================
# 16. Submit feedback - POST /feedback
# =========================================================================


class TestSubmitFeedback:
    """Tests for POST /api/v1/outlier/feedback."""

    def test_submit_feedback_success(self, client):
        resp = client.post("/api/v1/outlier/feedback", json={
            "detection_id": "det-001",
            "feedback_type": "confirmed_outlier",
            "notes": "Valid outlier",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "detection_id" in data

    def test_submit_feedback_forwards_params(self, client, mock_service):
        client.post("/api/v1/outlier/feedback", json={
            "detection_id": "det-001",
            "feedback_type": "false_positive",
            "notes": "Not an outlier",
        })
        call_kwargs = mock_service.submit_feedback.call_args[1]
        assert call_kwargs["feedback_type"] == "false_positive"
        assert call_kwargs["notes"] == "Not an outlier"


# =========================================================================
# 17. Analyze impact - POST /impact
# =========================================================================


class TestAnalyzeImpact:
    """Tests for POST /api/v1/outlier/impact."""

    def test_impact_success(self, client):
        resp = client.post("/api/v1/outlier/impact", json={
            "original": [{"val": 10.0}, {"val": 500.0}],
            "treated": [{"val": 10.0}, {"val": 50.0}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "columns" in data

    def test_impact_value_error_returns_400(self, client, mock_service):
        mock_service.analyze_impact.side_effect = ValueError("length mismatch")
        resp = client.post("/api/v1/outlier/impact", json={
            "original": [{"val": 10.0}],
            "treated": [],
        })
        assert resp.status_code == 400


# =========================================================================
# 18. Run pipeline - POST /pipeline
# =========================================================================


class TestRunPipeline:
    """Tests for POST /api/v1/outlier/pipeline."""

    def test_pipeline_success(self, client):
        resp = client.post("/api/v1/outlier/pipeline", json={
            "records": [{"val": 10.0}, {"val": 500.0}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_pipeline_with_config(self, client, mock_service):
        resp = client.post("/api/v1/outlier/pipeline", json={
            "records": [{"val": 10.0}],
            "config": {"methods": ["iqr", "zscore"]},
        })
        assert resp.status_code == 200
        call_kwargs = mock_service.run_pipeline.call_args[1]
        assert call_kwargs["config"] == {"methods": ["iqr", "zscore"]}


# =========================================================================
# 19. Health check - GET /health
# =========================================================================


class TestHealthCheck:
    """Tests for GET /api/v1/outlier/health."""

    def test_health_check_success(self, client):
        resp = client.get("/api/v1/outlier/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


# =========================================================================
# 20. Statistics - GET /stats
# =========================================================================


class TestStats:
    """Tests for GET /api/v1/outlier/stats."""

    def test_stats_success(self, client):
        resp = client.get("/api/v1/outlier/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_jobs" in data


# =========================================================================
# Error handling: 503 when service not configured
# =========================================================================


class TestServiceNotConfigured:
    """Tests for 503 errors when service is not attached to app.state."""

    def test_health_503(self, client_no_service):
        resp = client_no_service.get("/api/v1/outlier/health")
        assert resp.status_code == 503

    def test_detect_503(self, client_no_service):
        resp = client_no_service.post("/api/v1/outlier/detect", json={
            "records": [{"val": 10.0}],
            "column": "val",
        })
        assert resp.status_code == 503

    def test_jobs_list_503(self, client_no_service):
        resp = client_no_service.get("/api/v1/outlier/jobs")
        assert resp.status_code == 503

    def test_stats_503(self, client_no_service):
        resp = client_no_service.get("/api/v1/outlier/stats")
        assert resp.status_code == 503

    def test_create_job_503(self, client_no_service):
        resp = client_no_service.post("/api/v1/outlier/jobs", json={
            "records": [{"val": 10.0}],
        })
        assert resp.status_code == 503

    def test_pipeline_503(self, client_no_service):
        resp = client_no_service.post("/api/v1/outlier/pipeline", json={
            "records": [{"val": 10.0}],
        })
        assert resp.status_code == 503

    def test_thresholds_503(self, client_no_service):
        resp = client_no_service.get("/api/v1/outlier/thresholds")
        assert resp.status_code == 503

    def test_feedback_503(self, client_no_service):
        resp = client_no_service.post("/api/v1/outlier/feedback", json={
            "detection_id": "det-001",
        })
        assert resp.status_code == 503


# =========================================================================
# Error handling: 400 when service raises ValueError
# =========================================================================


class TestValueErrors:
    """Tests for 400 errors when service raises ValueError."""

    def test_detect_400(self, client, mock_service):
        mock_service.detect_outliers.side_effect = ValueError("empty records")
        resp = client.post("/api/v1/outlier/detect", json={
            "records": [],
            "column": "val",
        })
        assert resp.status_code == 400

    def test_classify_400(self, client, mock_service):
        mock_service.classify_outliers.side_effect = ValueError("empty detections")
        resp = client.post("/api/v1/outlier/classify", json={
            "detections": [],
            "records": [],
        })
        assert resp.status_code == 400

    def test_treat_400(self, client, mock_service):
        mock_service.apply_treatment.side_effect = ValueError("empty records")
        resp = client.post("/api/v1/outlier/treat", json={
            "records": [],
            "detections": [],
            "strategy": "flag",
        })
        assert resp.status_code == 400

    def test_threshold_400(self, client, mock_service):
        mock_service.create_threshold.side_effect = ValueError("empty column")
        resp = client.post("/api/v1/outlier/thresholds", json={
            "column": "",
        })
        assert resp.status_code == 400

    def test_pipeline_400(self, client, mock_service):
        mock_service.run_pipeline.side_effect = ValueError("empty records")
        resp = client.post("/api/v1/outlier/pipeline", json={
            "records": [],
        })
        assert resp.status_code == 400

    def test_batch_detect_400(self, client, mock_service):
        mock_service.detect_batch.side_effect = ValueError("empty records")
        resp = client.post("/api/v1/outlier/detect/batch", json={
            "records": [],
        })
        assert resp.status_code == 400

    def test_create_job_400(self, client, mock_service):
        mock_service.create_job.side_effect = ValueError("bad request")
        resp = client.post("/api/v1/outlier/jobs", json={
            "records": [],
        })
        assert resp.status_code == 400
