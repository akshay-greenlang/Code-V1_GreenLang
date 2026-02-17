# -*- coding: utf-8 -*-
"""
Unit Tests for Time Series Gap Filler REST API Router (AGENT-DATA-014)
========================================================================

Comprehensive test suite for ``greenlang.time_series_gap_filler.api.router``
covering all 20 endpoints using ``starlette.testclient.TestClient``.

Target: 40+ tests covering all endpoints, happy paths, and 404 cases.

The test uses a MagicMock service wired into
``app.state.time_series_gap_filler_service`` so that the router's
``_get_service(request)`` helper returns the mock rather than raising
503 Service Unavailable.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from greenlang.time_series_gap_filler.api.router import FASTAPI_AVAILABLE

# Skip the entire module if FastAPI is not installed
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not available; skipping router tests",
)


# ===================================================================
# Helpers
# ===================================================================


def _daily_timestamps(n: int = 10) -> List[str]:
    """Return ISO-8601 daily timestamps."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [(base + timedelta(days=i)).isoformat() for i in range(n)]


def _make_model_dump_mock(data: Dict[str, Any]) -> MagicMock:
    """Create a MagicMock that has a model_dump method returning *data*."""
    m = MagicMock()
    m.model_dump.return_value = data
    # Also allow attribute access on common fields for router logic
    for k, v in data.items():
        setattr(m, k, v)
    return m


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def mock_service():
    """Create a MagicMock mimicking TimeSeriesGapFillerService.

    Every method called by the router is stubbed to return the shape
    of data the router expects.  Endpoints that call
    ``result.model_dump(mode="json")`` receive a MagicMock with a
    ``model_dump`` method.
    """
    svc = MagicMock()

    # ---- Job CRUD (endpoints 1-4) ----
    svc.create_job.return_value = {
        "job_id": "job-001",
        "series_id": "",
        "status": "pending",
        "strategy": "auto",
        "created_at": "2026-02-01T00:00:00+00:00",
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
        "status": "pending",
    }
    # delete_job raises ValueError for unknown IDs in the router
    svc.delete_job.return_value = {"deleted": True, "job_id": "job-001"}

    # ---- Detect (endpoints 5-6) ----
    svc.detect_gaps.return_value = _make_model_dump_mock({
        "detection_id": "det-001",
        "series_name": "test",
        "total_points": 5,
        "total_gaps": 1,
        "total_missing": 1,
        "gap_pct": 0.2,
        "gap_count": 1,
        "gaps": [],
        "processing_time_ms": 1.0,
        "provenance_hash": "a" * 64,
    })
    svc.detect_gaps_batch.return_value = _make_model_dump_mock({
        "batch_id": "batch-001",
        "results": [
            {"detection_id": "det-001", "total_gaps": 1},
            {"detection_id": "det-002", "total_gaps": 0},
        ],
        "count": 2,
    })

    # ---- List / Get detections (endpoints 7-8) ----
    svc.list_detections.return_value = {
        "detections": [{"detection_id": "det-001"}],
        "count": 1,
        "total": 1,
        "limit": 50,
        "offset": 0,
    }
    svc.get_detection.return_value = _make_model_dump_mock({
        "detection_id": "det-001",
        "total_gaps": 1,
        "provenance_hash": "a" * 64,
    })

    # ---- Frequency (endpoints 9-10) ----
    svc.analyze_frequency.return_value = _make_model_dump_mock({
        "analysis_id": "freq-001",
        "detected_frequency": "daily",
        "frequency_seconds": 86400.0,
        "regularity_score": 0.95,
        "confidence": 0.9,
        "num_points": 20,
        "provenance_hash": "b" * 64,
    })
    svc.get_frequency_analysis.return_value = {
        "analysis_id": "freq-001",
        "detected_frequency": "daily",
    }

    # ---- Fill (endpoints 11-12) ----
    svc.fill_gaps.return_value = _make_model_dump_mock({
        "fill_id": "fill-001",
        "series_name": "test",
        "strategy": "linear",
        "total_filled": 2,
        "total_gaps": 2,
        "fill_rate": 1.0,
        "filled_values": [
            {"index": 1, "value": 2.0, "confidence": 0.9},
        ],
        "avg_confidence": 0.9,
        "min_confidence": 0.8,
        "processing_time_ms": 1.0,
        "provenance_hash": "c" * 64,
    })
    svc.get_fill.return_value = {
        "fill_id": "fill-001",
        "strategy": "linear",
        "filled_values": [1.0, 2.0, 3.0],
    }

    # ---- Validate (endpoints 13-14) ----
    _val_mock = _make_model_dump_mock({
        "validation_id": "val-001",
        "passed": True,
        "total_checks": 3,
        "passed_checks": 3,
        "failed_checks": 0,
        "checks": [],
        "overall_confidence": 0.95,
        "provenance_hash": "d" * 64,
    })
    _val_mock.passed = True
    svc.validate_fills.return_value = [_val_mock]
    svc.get_validation.return_value = {
        "validation_id": "val-001",
        "passed": True,
    }

    # ---- Correlations (endpoints 15-16) ----
    _corr_mock = _make_model_dump_mock({
        "correlation_id": "corr-001",
        "coefficient": 0.85,
        "suitable_for_fill": True,
        "provenance_hash": "e" * 64,
    })
    _corr_mock.suitable_for_fill = True
    svc.compute_correlations.return_value = [_corr_mock]
    svc.list_correlations.return_value = {
        "correlations": [{"correlation_id": "corr-001"}],
        "count": 1,
        "total": 1,
        "limit": 50,
        "offset": 0,
    }

    # ---- Calendars (endpoints 17-18) ----
    svc.create_calendar.return_value = _make_model_dump_mock({
        "calendar_id": "cal-001",
        "name": "test_calendar",
        "calendar_type": "business",
        "created_at": "2026-02-01T00:00:00+00:00",
    })
    svc.list_calendars.return_value = {
        "calendars": [{"calendar_id": "cal-001", "name": "test_calendar"}],
        "count": 1,
        "total": 1,
        "limit": 50,
        "offset": 0,
    }

    # ---- Health + Stats (endpoints 19-20) ----
    svc.health_check.return_value = {
        "status": "healthy",
        "service": "time-series-gap-filler",
        "started": True,
    }
    svc.get_statistics.return_value = _make_model_dump_mock({
        "total_jobs": 5,
        "completed_jobs": 3,
        "failed_jobs": 0,
        "cancelled_jobs": 0,
        "total_gaps_detected": 10,
        "total_gaps_filled": 8,
        "total_validations": 2,
        "total_frequency_analyses": 1,
        "total_correlations": 0,
        "total_calendars": 1,
        "active_jobs": 0,
    })

    return svc


@pytest.fixture
def client(mock_service):
    """Create a FastAPI TestClient with the mock service wired in."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from greenlang.time_series_gap_filler.api.router import router

    app = FastAPI()
    app.include_router(router)
    # The router's _get_service reads this exact attribute name
    app.state.time_series_gap_filler_service = mock_service

    return TestClient(app)


@pytest.fixture
def client_no_service():
    """TestClient WITHOUT the service (for 503 tests)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from greenlang.time_series_gap_filler.api.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ===================================================================
# 1. POST /api/v1/gap-filler/jobs (Create job) -- endpoint 1
# ===================================================================


class TestCreateJob:
    """Tests for POST /jobs."""

    def test_create_job_returns_success(self, client):
        """POST /jobs returns 200 with a job_id."""
        resp = client.post(
            "/api/v1/gap-filler/jobs",
            json={
                "series_id": "my_ts",
                "series": [1.0, 2.0, 3.0],
                "timestamps": _daily_timestamps(3),
                "strategy": "auto",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body

    def test_create_job_default_strategy(self, client):
        """POST /jobs with minimal body uses default strategy."""
        resp = client.post(
            "/api/v1/gap-filler/jobs",
            json={"series_id": "ts_001"},
        )
        assert resp.status_code == 200

    def test_create_job_calls_service(self, client, mock_service):
        """POST /jobs delegates to service.create_job."""
        client.post(
            "/api/v1/gap-filler/jobs",
            json={"series_id": "ts_002"},
        )
        mock_service.create_job.assert_called_once()

    def test_create_job_503_without_service(self, client_no_service):
        """POST /jobs returns 503 when service is not configured."""
        resp = client_no_service.post(
            "/api/v1/gap-filler/jobs",
            json={"series_id": "ts_003"},
        )
        assert resp.status_code == 503


# ===================================================================
# 2. GET /api/v1/gap-filler/jobs (List jobs) -- endpoint 2
# ===================================================================


class TestListJobs:
    """Tests for GET /jobs."""

    def test_list_jobs_returns_200(self, client):
        """GET /jobs returns 200 with a jobs list."""
        resp = client.get("/api/v1/gap-filler/jobs")
        assert resp.status_code == 200
        body = resp.json()
        assert "jobs" in body
        assert isinstance(body["jobs"], list)

    def test_list_jobs_has_count(self, client):
        """GET /jobs response includes count."""
        resp = client.get("/api/v1/gap-filler/jobs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1


# ===================================================================
# 3. GET /api/v1/gap-filler/jobs/{id} (Get job) -- endpoint 3
# ===================================================================


class TestGetJob:
    """Tests for GET /jobs/{job_id}."""

    def test_get_job_returns_200(self, client):
        """GET /jobs/{id} returns 200 when the service finds the job."""
        resp = client.get("/api/v1/gap-filler/jobs/job-001")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-001"

    def test_get_job_not_found(self, client, mock_service):
        """GET /jobs/{id} returns 404 for unknown ID."""
        mock_service.get_job.return_value = None
        resp = client.get("/api/v1/gap-filler/jobs/nonexistent-id")
        assert resp.status_code == 404


# ===================================================================
# 4. DELETE /api/v1/gap-filler/jobs/{id} (Delete job) -- endpoint 4
# ===================================================================


class TestDeleteJob:
    """Tests for DELETE /jobs/{job_id}."""

    def test_delete_job_returns_200(self, client):
        """DELETE /jobs/{id} returns 200."""
        resp = client.delete("/api/v1/gap-filler/jobs/job-001")
        assert resp.status_code == 200

    def test_delete_job_not_found(self, client, mock_service):
        """DELETE /jobs/{id} returns 404 when service raises ValueError."""
        mock_service.delete_job.side_effect = ValueError("Job not found")
        resp = client.delete("/api/v1/gap-filler/jobs/nonexistent-id")
        assert resp.status_code == 404


# ===================================================================
# 5. POST /api/v1/gap-filler/detect (Detect gaps) -- endpoint 5
# ===================================================================


class TestDetectGaps:
    """Tests for POST /detect."""

    def test_detect_returns_200(self, client):
        """POST /detect returns 200."""
        resp = client.post(
            "/api/v1/gap-filler/detect",
            json={
                "series": [1.0, None, 3.0, 4.0, 5.0],
                "timestamps": _daily_timestamps(5),
            },
        )
        assert resp.status_code == 200

    def test_detect_with_gaps(self, client):
        """POST /detect identifies gaps in the values."""
        resp = client.post(
            "/api/v1/gap-filler/detect",
            json={
                "series": [1.0, None, 3.0, 4.0, 5.0],
                "timestamps": _daily_timestamps(5),
            },
        )
        body = resp.json()
        assert body.get("total_missing", 0) >= 1 or body.get("gap_count", 0) >= 1

    def test_detect_no_gaps(self, client, mock_service):
        """POST /detect on a complete series returns zero gaps."""
        mock_service.detect_gaps.return_value = _make_model_dump_mock({
            "detection_id": "det-002",
            "total_gaps": 0,
            "total_missing": 0,
            "gap_pct": 0.0,
            "gaps": [],
            "provenance_hash": "a" * 64,
        })
        resp = client.post(
            "/api/v1/gap-filler/detect",
            json={
                "series": [1.0, 2.0, 3.0, 4.0, 5.0],
                "timestamps": _daily_timestamps(5),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("total_missing", 0) == 0


# ===================================================================
# 6. POST /api/v1/gap-filler/detect/batch (Batch detect) -- endpoint 6
# ===================================================================


class TestDetectBatch:
    """Tests for POST /detect/batch."""

    def test_batch_detect_returns_200(self, client):
        """POST /detect/batch returns 200."""
        resp = client.post(
            "/api/v1/gap-filler/detect/batch",
            json=[
                {
                    "series": [1.0, None, 3.0],
                    "timestamps": _daily_timestamps(3),
                    "name": "s1",
                },
                {
                    "series": [4.0, 5.0, 6.0],
                    "timestamps": _daily_timestamps(3),
                    "name": "s2",
                },
            ],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2


# ===================================================================
# 7. GET /api/v1/gap-filler/detections (List detections) -- endpoint 7
# ===================================================================


class TestListDetections:
    """Tests for GET /detections."""

    def test_list_detections_returns_200(self, client):
        """GET /detections returns 200."""
        resp = client.get("/api/v1/gap-filler/detections")
        assert resp.status_code == 200
        body = resp.json()
        assert "detections" in body

    def test_list_detections_has_count(self, client):
        """GET /detections returns count."""
        resp = client.get("/api/v1/gap-filler/detections")
        body = resp.json()
        assert body["count"] >= 1


# ===================================================================
# 8. GET /api/v1/gap-filler/detections/{id} -- endpoint 8
# ===================================================================


class TestGetDetection:
    """Tests for GET /detections/{detection_id}."""

    def test_get_detection_not_found(self, client, mock_service):
        """GET /detections/{id} returns 404 for unknown ID."""
        mock_service.get_detection.return_value = None
        resp = client.get("/api/v1/gap-filler/detections/nonexistent-id")
        assert resp.status_code == 404

    def test_get_detection_returns_200(self, client):
        """GET /detections/{id} returns 200 when found."""
        resp = client.get("/api/v1/gap-filler/detections/det-001")
        assert resp.status_code == 200


# ===================================================================
# 9. POST /api/v1/gap-filler/frequency (Analyze frequency) -- endpoint 9
# ===================================================================


class TestAnalyzeFrequency:
    """Tests for POST /frequency."""

    def test_frequency_returns_200(self, client):
        """POST /frequency returns 200."""
        resp = client.post(
            "/api/v1/gap-filler/frequency",
            json={"timestamps": _daily_timestamps(20)},
        )
        assert resp.status_code == 200


# ===================================================================
# 10. GET /api/v1/gap-filler/frequency/{id} -- endpoint 10
# ===================================================================


class TestGetFrequency:
    """Tests for GET /frequency/{analysis_id}."""

    def test_get_frequency_not_found(self, client, mock_service):
        """GET /frequency/{id} returns 404 for unknown ID."""
        mock_service.get_frequency_analysis.return_value = None
        resp = client.get("/api/v1/gap-filler/frequency/nonexistent-id")
        assert resp.status_code == 404

    def test_get_frequency_returns_200(self, client):
        """GET /frequency/{id} returns 200 when found."""
        resp = client.get("/api/v1/gap-filler/frequency/freq-001")
        assert resp.status_code == 200


# ===================================================================
# 11. POST /api/v1/gap-filler/fill (Fill gaps) -- endpoint 11
# ===================================================================


class TestFillGaps:
    """Tests for POST /fill."""

    def test_fill_returns_200(self, client):
        """POST /fill returns 200."""
        resp = client.post(
            "/api/v1/gap-filler/fill",
            json={
                "series": [1.0, None, 3.0, 4.0, 5.0],
                "timestamps": _daily_timestamps(5),
                "strategy": "linear",
            },
        )
        assert resp.status_code == 200

    def test_fill_returns_filled_values(self, client):
        """POST /fill returns filled_values list."""
        resp = client.post(
            "/api/v1/gap-filler/fill",
            json={
                "series": [1.0, None, 3.0],
                "timestamps": _daily_timestamps(3),
            },
        )
        body = resp.json()
        assert "filled_values" in body


# ===================================================================
# 12. GET /api/v1/gap-filler/fills/{id} -- endpoint 12
# ===================================================================


class TestGetFill:
    """Tests for GET /fills/{fill_id}."""

    def test_get_fill_not_found(self, client, mock_service):
        """GET /fills/{id} returns 404 for unknown ID."""
        mock_service.get_fill.return_value = None
        resp = client.get("/api/v1/gap-filler/fills/nonexistent-id")
        assert resp.status_code == 404

    def test_get_fill_returns_200(self, client):
        """GET /fills/{id} returns 200 when found."""
        resp = client.get("/api/v1/gap-filler/fills/fill-001")
        assert resp.status_code == 200


# ===================================================================
# 13. POST /api/v1/gap-filler/validate (Validate fills) -- endpoint 13
# ===================================================================


class TestValidateFills:
    """Tests for POST /validate."""

    def test_validate_returns_200(self, client):
        """POST /validate returns 200."""
        resp = client.post(
            "/api/v1/gap-filler/validate",
            json={
                "fills": [{"index": 1, "value": 2.0, "confidence": 0.9}],
                "original_series": [1.0, None, 3.0],
            },
        )
        assert resp.status_code == 200

    def test_validate_returns_all_passed(self, client):
        """POST /validate response includes all_passed."""
        resp = client.post(
            "/api/v1/gap-filler/validate",
            json={
                "fills": [{"index": 1, "value": 2.0, "confidence": 0.9}],
                "original_series": [1.0, None, 3.0],
            },
        )
        body = resp.json()
        assert "all_passed" in body


# ===================================================================
# 14. GET /api/v1/gap-filler/validations/{id} -- endpoint 14
# ===================================================================


class TestGetValidation:
    """Tests for GET /validations/{validation_id}."""

    def test_get_validation_not_found(self, client, mock_service):
        """GET /validations/{id} returns 404 for unknown ID."""
        mock_service.get_validation.return_value = None
        resp = client.get("/api/v1/gap-filler/validations/nonexistent-id")
        assert resp.status_code == 404

    def test_get_validation_returns_200(self, client):
        """GET /validations/{id} returns 200 when found."""
        resp = client.get("/api/v1/gap-filler/validations/val-001")
        assert resp.status_code == 200


# ===================================================================
# 15. POST /api/v1/gap-filler/correlations -- endpoint 15
# ===================================================================


class TestComputeCorrelations:
    """Tests for POST /correlations."""

    def test_correlations_returns_200(self, client):
        """POST /correlations returns 200."""
        resp = client.post(
            "/api/v1/gap-filler/correlations",
            json={
                "target": [1.0, 2.0, 3.0, 4.0, 5.0],
                "references": [
                    [1.1, 2.1, 3.1, 4.1, 5.1],
                ],
                "method": "pearson",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "correlations" in body

    def test_correlations_suitable_count(self, client):
        """POST /correlations response includes suitable_count."""
        resp = client.post(
            "/api/v1/gap-filler/correlations",
            json={
                "target": [1.0, 2.0, 3.0],
                "references": [[1.1, 2.1, 3.1]],
            },
        )
        body = resp.json()
        assert "suitable_count" in body


# ===================================================================
# 16. GET /api/v1/gap-filler/correlations -- endpoint 16
# ===================================================================


class TestListCorrelations:
    """Tests for GET /correlations."""

    def test_list_correlations_returns_200(self, client):
        """GET /correlations returns 200."""
        resp = client.get("/api/v1/gap-filler/correlations")
        assert resp.status_code == 200
        body = resp.json()
        assert "correlations" in body


# ===================================================================
# 17. POST /api/v1/gap-filler/calendars -- endpoint 17
# ===================================================================


class TestCreateCalendar:
    """Tests for POST /calendars."""

    def test_create_calendar_returns_200(self, client):
        """POST /calendars returns 200."""
        resp = client.post(
            "/api/v1/gap-filler/calendars",
            json={"name": "test_calendar"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "calendar_id" in body or "name" in body

    def test_create_calendar_calls_service(self, client, mock_service):
        """POST /calendars delegates to service.create_calendar."""
        client.post(
            "/api/v1/gap-filler/calendars",
            json={"name": "us_business", "calendar_type": "business"},
        )
        mock_service.create_calendar.assert_called_once()


# ===================================================================
# 18. GET /api/v1/gap-filler/calendars -- endpoint 18
# ===================================================================


class TestListCalendars:
    """Tests for GET /calendars."""

    def test_list_calendars_returns_200(self, client):
        """GET /calendars returns 200."""
        resp = client.get("/api/v1/gap-filler/calendars")
        assert resp.status_code == 200
        body = resp.json()
        assert "calendars" in body
        assert isinstance(body["calendars"], list)


# ===================================================================
# 19. GET /api/v1/gap-filler/health -- endpoint 19
# ===================================================================


class TestHealthCheck:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """GET /health returns 200."""
        resp = client.get("/api/v1/gap-filler/health")
        assert resp.status_code == 200

    def test_health_has_status_key(self, client):
        """GET /health response includes a 'status' key."""
        resp = client.get("/api/v1/gap-filler/health")
        body = resp.json()
        assert "status" in body


# ===================================================================
# 20. GET /api/v1/gap-filler/stats -- endpoint 20
# ===================================================================


class TestGetStats:
    """Tests for GET /stats."""

    def test_stats_returns_200(self, client):
        """GET /stats returns 200."""
        resp = client.get("/api/v1/gap-filler/stats")
        assert resp.status_code == 200

    def test_stats_has_counters(self, client):
        """GET /stats response includes counter keys."""
        resp = client.get("/api/v1/gap-filler/stats")
        body = resp.json()
        assert isinstance(body, dict)
        assert "total_jobs" in body


# ===================================================================
# 503 Service Unavailable Tests
# ===================================================================


class TestServiceUnavailable:
    """Verify all endpoints return 503 when the service is not configured."""

    def test_health_503(self, client_no_service):
        resp = client_no_service.get("/api/v1/gap-filler/health")
        assert resp.status_code == 503

    def test_stats_503(self, client_no_service):
        resp = client_no_service.get("/api/v1/gap-filler/stats")
        assert resp.status_code == 503

    def test_jobs_503(self, client_no_service):
        resp = client_no_service.get("/api/v1/gap-filler/jobs")
        assert resp.status_code == 503

    def test_detections_503(self, client_no_service):
        resp = client_no_service.get("/api/v1/gap-filler/detections")
        assert resp.status_code == 503

    def test_calendars_503(self, client_no_service):
        resp = client_no_service.get("/api/v1/gap-filler/calendars")
        assert resp.status_code == 503


# ===================================================================
# Cross-Endpoint Integration Tests
# ===================================================================


class TestEndToEndWorkflow:
    """Integration tests exercising multiple endpoints."""

    def test_create_and_list_jobs(self, client, mock_service):
        """Create a job, then list and verify count."""
        client.post(
            "/api/v1/gap-filler/jobs",
            json={"series_id": "ts_0"},
        )
        resp = client.get("/api/v1/gap-filler/jobs")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_create_and_delete_job(self, client, mock_service):
        """Create then delete a job, verify 404 on re-get."""
        create_resp = client.post(
            "/api/v1/gap-filler/jobs",
            json={"series_id": "to_delete"},
        )
        job_id = create_resp.json()["job_id"]

        del_resp = client.delete(f"/api/v1/gap-filler/jobs/{job_id}")
        assert del_resp.status_code == 200

        # After deletion, get_job returns None
        mock_service.get_job.return_value = None
        get_resp = client.get(f"/api/v1/gap-filler/jobs/{job_id}")
        assert get_resp.status_code == 404

    def test_detect_fill_validate_workflow(self, client):
        """Detect gaps, fill them, then validate."""
        # Detect
        detect_resp = client.post(
            "/api/v1/gap-filler/detect",
            json={
                "series": [1.0, None, 3.0, None, 5.0],
                "timestamps": _daily_timestamps(5),
            },
        )
        assert detect_resp.status_code == 200

        # Fill
        fill_resp = client.post(
            "/api/v1/gap-filler/fill",
            json={
                "series": [1.0, None, 3.0, None, 5.0],
                "timestamps": _daily_timestamps(5),
                "strategy": "linear",
            },
        )
        assert fill_resp.status_code == 200

        # Validate
        validate_resp = client.post(
            "/api/v1/gap-filler/validate",
            json={
                "fills": [{"index": 1, "value": 2.0, "confidence": 0.9}],
                "original_series": [1.0, None, 3.0, None, 5.0],
            },
        )
        assert validate_resp.status_code == 200

    def test_create_and_list_calendars(self, client):
        """Create a calendar then list to verify it appears."""
        client.post(
            "/api/v1/gap-filler/calendars",
            json={"name": "us_business"},
        )
        resp = client.get("/api/v1/gap-filler/calendars")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_stats_after_operations(self, client):
        """Stats are returned after performing operations."""
        client.post(
            "/api/v1/gap-filler/detect",
            json={
                "series": [1.0, None, 3.0],
                "timestamps": _daily_timestamps(3),
            },
        )
        client.post(
            "/api/v1/gap-filler/fill",
            json={
                "series": [1.0, None, 3.0],
                "timestamps": _daily_timestamps(3),
            },
        )
        resp = client.get("/api/v1/gap-filler/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_jobs"] >= 0

    def test_frequency_analysis_workflow(self, client):
        """Analyze frequency then retrieve the result."""
        # Analyze
        freq_resp = client.post(
            "/api/v1/gap-filler/frequency",
            json={"timestamps": _daily_timestamps(20)},
        )
        assert freq_resp.status_code == 200

        # Retrieve
        get_resp = client.get("/api/v1/gap-filler/frequency/freq-001")
        assert get_resp.status_code == 200

    def test_correlation_workflow(self, client):
        """Compute correlations and list them."""
        # Compute
        corr_resp = client.post(
            "/api/v1/gap-filler/correlations",
            json={
                "target": [1.0, 2.0, 3.0],
                "references": [[1.1, 2.1, 3.1]],
            },
        )
        assert corr_resp.status_code == 200
        assert corr_resp.json()["suitable_count"] >= 1

        # List
        list_resp = client.get("/api/v1/gap-filler/correlations")
        assert list_resp.status_code == 200
