# -*- coding: utf-8 -*-
"""
Unit tests for Missing Value Imputer REST API Router - AGENT-DATA-012

Tests all 20 FastAPI endpoints under /api/v1/imputer, _get_service helper,
FASTAPI_AVAILABLE flag, request body models, error handling (400/404/503),
and edge cases.
Target: 40+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.missing_value_imputer.api.router import (
    FASTAPI_AVAILABLE,
    router,
)

# Skip all tests if FastAPI is not available
pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not available",
)


# ---------------------------------------------------------------------------
# Test client fixture
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from greenlang.missing_value_imputer.setup import (
        MissingValueImputerService,
    )


@pytest.fixture
def service():
    """Create a MissingValueImputerService for testing."""
    svc = MissingValueImputerService()
    svc.startup()
    return svc


@pytest.fixture
def app(service):
    """Create a FastAPI app with the imputer service configured."""
    application = FastAPI()
    application.state.missing_value_imputer_service = service
    application.include_router(router)
    return application


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def unconfigured_client():
    """Create a test client without the imputer service configured."""
    application = FastAPI()
    application.include_router(router)
    return TestClient(application)


# ---------------------------------------------------------------------------
# FASTAPI_AVAILABLE / router tests
# ---------------------------------------------------------------------------


class TestRouterAvailability:
    def test_fastapi_available(self):
        assert FASTAPI_AVAILABLE is True

    def test_router_not_none(self):
        assert router is not None

    def test_router_prefix(self):
        assert router.prefix == "/api/v1/imputer"


# ---------------------------------------------------------------------------
# _get_service (503 error)
# ---------------------------------------------------------------------------


class TestGetService:
    def test_service_not_configured_returns_503(self, unconfigured_client):
        resp = unconfigured_client.get("/api/v1/imputer/health")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# 1. POST /jobs - Create imputation job
# ---------------------------------------------------------------------------


class TestCreateJob:
    def test_create_job(self, client):
        resp = client.post("/api/v1/imputer/jobs", json={
            "dataset_id": "ds1",
            "records": [{"a": 1.0}, {"a": None}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["total_records"] == 2

    def test_create_job_empty_records(self, client):
        resp = client.post("/api/v1/imputer/jobs", json={
            "records": [],
        })
        # Empty records is allowed for job creation (no validation)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. GET /jobs - List jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    def test_list_jobs_empty(self, client):
        resp = client.get("/api/v1/imputer/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["count"] == 0

    def test_list_jobs_after_create(self, client):
        client.post("/api/v1/imputer/jobs", json={
            "records": [{"a": 1}],
        })
        resp = client.get("/api/v1/imputer/jobs")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_list_jobs_with_status_filter(self, client):
        client.post("/api/v1/imputer/jobs", json={"records": [{"a": 1}]})
        resp = client.get("/api/v1/imputer/jobs?status=pending")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

        resp = client.get("/api/v1/imputer/jobs?status=completed")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ---------------------------------------------------------------------------
# 3. GET /jobs/{job_id} - Get job details
# ---------------------------------------------------------------------------


class TestGetJob:
    def test_get_job(self, client):
        resp = client.post("/api/v1/imputer/jobs", json={
            "records": [{"a": 1}],
        })
        job_id = resp.json()["job_id"]
        resp = client.get(f"/api/v1/imputer/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_id

    def test_get_job_not_found(self, client):
        resp = client.get("/api/v1/imputer/jobs/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 4. DELETE /jobs/{job_id} - Delete job
# ---------------------------------------------------------------------------


class TestDeleteJob:
    def test_delete_job(self, client):
        resp = client.post("/api/v1/imputer/jobs", json={
            "records": [{"a": 1}],
        })
        job_id = resp.json()["job_id"]
        resp = client.delete(f"/api/v1/imputer/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_delete_job_not_found(self, client):
        resp = client.delete("/api/v1/imputer/jobs/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. POST /analyze - Analyze missingness
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_analyze(self, client):
        resp = client.post("/api/v1/imputer/analyze", json={
            "records": [{"x": 1.0}, {"x": None}, {"x": 3.0}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "analysis_id" in data
        assert data["total_records"] == 3

    def test_analyze_with_columns(self, client):
        resp = client.post("/api/v1/imputer/analyze", json={
            "records": [{"x": 1.0, "y": 2.0}, {"x": None, "y": None}],
            "columns": ["x"],
        })
        assert resp.status_code == 200

    def test_analyze_empty_records_400(self, client):
        resp = client.post("/api/v1/imputer/analyze", json={
            "records": [],
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 6. GET /analyze/{analysis_id} - Get analysis result
# ---------------------------------------------------------------------------


class TestGetAnalysis:
    def test_get_analysis(self, client):
        resp = client.post("/api/v1/imputer/analyze", json={
            "records": [{"x": 1.0}, {"x": None}],
        })
        analysis_id = resp.json()["analysis_id"]
        resp = client.get(f"/api/v1/imputer/analyze/{analysis_id}")
        assert resp.status_code == 200

    def test_get_analysis_not_found(self, client):
        resp = client.get("/api/v1/imputer/analyze/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 7. POST /impute - Impute values
# ---------------------------------------------------------------------------


class TestImpute:
    def test_impute(self, client):
        resp = client.post("/api/v1/imputer/impute", json={
            "records": [{"x": 1.0}, {"x": None}, {"x": 3.0}],
            "column": "x",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["column_name"] == "x"
        assert data["values_imputed"] > 0

    def test_impute_with_strategy(self, client):
        resp = client.post("/api/v1/imputer/impute", json={
            "records": [{"x": 1.0}, {"x": None}, {"x": 3.0}],
            "column": "x",
            "strategy": "median",
        })
        assert resp.status_code == 200

    def test_impute_empty_records_400(self, client):
        resp = client.post("/api/v1/imputer/impute", json={
            "records": [],
            "column": "x",
        })
        assert resp.status_code == 400

    def test_impute_missing_column_400(self, client):
        resp = client.post("/api/v1/imputer/impute", json={
            "records": [{"x": 1.0}],
            "column": "nonexistent",
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 8. POST /impute/batch - Batch impute
# ---------------------------------------------------------------------------


class TestImputeBatch:
    def test_batch_impute(self, client):
        resp = client.post("/api/v1/imputer/impute/batch", json={
            "records": [{"x": 1.0, "y": None}, {"x": None, "y": 2.0}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "total_values_imputed" in data

    def test_batch_impute_empty_400(self, client):
        resp = client.post("/api/v1/imputer/impute/batch", json={
            "records": [],
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 9. GET /results/{result_id} - Get imputation result
# ---------------------------------------------------------------------------


class TestGetResults:
    def test_get_results(self, client):
        resp = client.post("/api/v1/imputer/impute", json={
            "records": [{"x": 1.0}, {"x": None}],
            "column": "x",
        })
        result_id = resp.json()["result_id"]
        resp = client.get(f"/api/v1/imputer/results/{result_id}")
        assert resp.status_code == 200

    def test_get_results_not_found(self, client):
        resp = client.get("/api/v1/imputer/results/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 10. POST /validate - Validate imputation
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate(self, client):
        resp = client.post("/api/v1/imputer/validate", json={
            "original_records": [{"x": 1.0}, {"x": None}, {"x": 3.0}],
            "imputed_records": [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_passed" in data

    def test_validate_empty_original_400(self, client):
        resp = client.post("/api/v1/imputer/validate", json={
            "original_records": [],
            "imputed_records": [{"x": 1.0}],
        })
        assert resp.status_code == 400

    def test_validate_empty_imputed_400(self, client):
        resp = client.post("/api/v1/imputer/validate", json={
            "original_records": [{"x": 1.0}],
            "imputed_records": [],
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 11. GET /validate/{validation_id} - Get validation result
# ---------------------------------------------------------------------------


class TestGetValidation:
    def test_get_validation(self, client):
        resp = client.post("/api/v1/imputer/validate", json={
            "original_records": [{"x": 1.0}, {"x": None}],
            "imputed_records": [{"x": 1.0}, {"x": 2.0}],
        })
        validation_id = resp.json()["validation_id"]
        resp = client.get(f"/api/v1/imputer/validate/{validation_id}")
        assert resp.status_code == 200

    def test_get_validation_not_found(self, client):
        resp = client.get("/api/v1/imputer/validate/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 12. POST /rules - Create rule
# ---------------------------------------------------------------------------


class TestCreateRule:
    def test_create_rule(self, client):
        resp = client.post("/api/v1/imputer/rules", json={
            "name": "test_rule",
            "target_column": "val",
            "impute_value": 1.0,
            "priority": "high",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test_rule"
        assert data["is_active"] is True


# ---------------------------------------------------------------------------
# 13. GET /rules - List rules
# ---------------------------------------------------------------------------


class TestListRules:
    def test_list_rules_empty(self, client):
        resp = client.get("/api/v1/imputer/rules")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_list_rules_after_create(self, client):
        client.post("/api/v1/imputer/rules", json={
            "name": "r1", "target_column": "c1",
        })
        resp = client.get("/api/v1/imputer/rules")
        assert resp.json()["count"] == 1


# ---------------------------------------------------------------------------
# 14. PUT /rules/{rule_id} - Update rule
# ---------------------------------------------------------------------------


class TestUpdateRule:
    def test_update_rule(self, client):
        resp = client.post("/api/v1/imputer/rules", json={
            "name": "r1", "target_column": "c1",
        })
        rule_id = resp.json()["rule_id"]
        resp = client.put(f"/api/v1/imputer/rules/{rule_id}", json={
            "name": "updated_r1",
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "updated_r1"

    def test_update_rule_not_found(self, client):
        resp = client.put("/api/v1/imputer/rules/nonexistent", json={
            "name": "test",
        })
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 15. DELETE /rules/{rule_id} - Delete rule
# ---------------------------------------------------------------------------


class TestDeleteRule:
    def test_delete_rule(self, client):
        resp = client.post("/api/v1/imputer/rules", json={
            "name": "r1", "target_column": "c1",
        })
        rule_id = resp.json()["rule_id"]
        resp = client.delete(f"/api/v1/imputer/rules/{rule_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_rule_not_found(self, client):
        resp = client.delete("/api/v1/imputer/rules/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 16. POST /templates - Create template
# ---------------------------------------------------------------------------


class TestCreateTemplate:
    def test_create_template(self, client):
        resp = client.post("/api/v1/imputer/templates", json={
            "name": "test_template",
            "description": "A test template",
            "strategies": {"val": "median"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test_template"
        assert data["is_active"] is True


# ---------------------------------------------------------------------------
# 17. GET /templates - List templates
# ---------------------------------------------------------------------------


class TestListTemplates:
    def test_list_templates_empty(self, client):
        resp = client.get("/api/v1/imputer/templates")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_list_templates_after_create(self, client):
        client.post("/api/v1/imputer/templates", json={
            "name": "t1",
        })
        resp = client.get("/api/v1/imputer/templates")
        assert resp.json()["count"] == 1


# ---------------------------------------------------------------------------
# 18. POST /pipeline - Run pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_run_pipeline(self, client):
        resp = client.post("/api/v1/imputer/pipeline", json={
            "records": [{"x": 1.0}, {"x": None}, {"x": 3.0}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["total_records"] == 3

    def test_run_pipeline_empty_records_400(self, client):
        resp = client.post("/api/v1/imputer/pipeline", json={
            "records": [],
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 19. GET /health - Health check
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health(self, client):
        resp = client.get("/api/v1/imputer/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "missing-value-imputer"


# ---------------------------------------------------------------------------
# 20. GET /stats - Statistics
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats(self, client):
        resp = client.get("/api/v1/imputer/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_jobs" in data
        assert "total_values_imputed" in data
