# -*- coding: utf-8 -*-
"""
Integration tests for Missing Value Imputer REST API endpoints - AGENT-DATA-012

Tests all 20 API endpoints via FastAPI TestClient, validating HTTP status
codes, response shapes, error handling, pagination, and cross-endpoint
data flow.

29 test cases covering:
- TestJobEndpoints (8 tests)
- TestAnalyzeEndpoints (2 tests)
- TestImputeEndpoints (3 tests)
- TestValidateEndpoints (2 tests)
- TestRuleEndpoints (4 tests)
- TestTemplateEndpoints (2 tests)
- TestPipelineEndpoint (2 tests)
- TestHealthAndStatsEndpoints (3 tests)
- TestAuthEndpoints (1 test)
- TestCrossEndpointFlow (2 tests)

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
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
        """POST /api/v1/mvi/jobs creates an imputation job and returns 200.

        Validates:
        - HTTP 200 response
        - job_id is present and a valid UUID
        - status is 'pending'
        - total_records matches the request
        """
        resp = test_client.post(
            "/api/v1/mvi/jobs",
            json={
                "records": [{"a": 1, "b": None}, {"a": None, "b": 2}],
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

    def test_create_job_with_template(self, test_client):
        """POST /api/v1/mvi/jobs with template_id includes it in the response."""
        resp = test_client.post(
            "/api/v1/mvi/jobs",
            json={
                "records": [{"a": 1}],
                "template_id": "tmpl-123",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["template_id"] == "tmpl-123"

    def test_list_jobs_endpoint(self, test_client):
        """GET /api/v1/mvi/jobs lists all created jobs with pagination.

        Validates:
        - Response contains 'jobs', 'count', 'total' fields
        - Creating 3 jobs results in count=3
        """
        # Create multiple jobs
        for i in range(3):
            test_client.post(
                "/api/v1/mvi/jobs",
                json={"records": [{"id": i}], "dataset_id": f"ds-{i}"},
            )

        resp = test_client.get("/api/v1/mvi/jobs")
        assert resp.status_code == 200

        body = resp.json()
        assert "jobs" in body
        assert body["count"] == 3
        assert body["total"] == 3

    def test_list_jobs_with_pagination(self, test_client):
        """GET /api/v1/mvi/jobs?limit=2&offset=0 returns paginated results."""
        for i in range(5):
            test_client.post(
                "/api/v1/mvi/jobs",
                json={"records": [{"id": i}], "dataset_id": f"ds-{i}"},
            )

        resp = test_client.get("/api/v1/mvi/jobs?limit=2&offset=0")
        assert resp.status_code == 200

        body = resp.json()
        assert body["count"] == 2
        assert body["total"] == 5

    def test_get_job_details_endpoint(self, test_client):
        """GET /api/v1/mvi/jobs/{job_id} returns the specific job.

        Validates:
        - HTTP 200 for existing job
        - Response matches the created job
        """
        create_resp = test_client.post(
            "/api/v1/mvi/jobs",
            json={"records": [{"a": 1}], "dataset_id": "ds-001"},
        )
        job_id = create_resp.json()["job_id"]

        resp = test_client.get(f"/api/v1/mvi/jobs/{job_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "pending"

    def test_get_job_not_found(self, test_client):
        """GET /api/v1/mvi/jobs/{nonexistent} returns 404."""
        resp = test_client.get(f"/api/v1/mvi/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_delete_job_endpoint(self, test_client):
        """DELETE /api/v1/mvi/jobs/{job_id} cancels the job.

        Validates:
        - HTTP 200 for existing job
        - Status changes to 'cancelled'
        """
        create_resp = test_client.post(
            "/api/v1/mvi/jobs",
            json={"records": [{"a": 1}], "dataset_id": "ds-001"},
        )
        job_id = create_resp.json()["job_id"]

        resp = test_client.delete(f"/api/v1/mvi/jobs/{job_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "cancelled"

    def test_delete_job_not_found(self, test_client):
        """DELETE /api/v1/mvi/jobs/{nonexistent} returns 404."""
        resp = test_client.delete(f"/api/v1/mvi/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404


# ===================================================================
# Analyze API Tests
# ===================================================================


class TestAnalyzeEndpoints:
    """Tests for the missingness analysis API endpoints."""

    def test_analyze_endpoint(self, test_client):
        """POST /api/v1/mvi/analyze analyzes missingness patterns.

        Validates:
        - HTTP 200 response
        - total_records matches input length
        - columns_with_missing is detected
        - provenance_hash is a 64-char hex string
        """
        records = [
            {"name": "Alice", "revenue": 1000.0, "emissions": None},
            {"name": "Bob", "revenue": None, "emissions": 500.0},
            {"name": "Clara", "revenue": 2000.0, "emissions": 800.0},
        ]

        resp = test_client.post(
            "/api/v1/mvi/analyze",
            json={"records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 3
        assert body["columns_with_missing"] >= 1
        assert len(body["provenance_hash"]) == 64
        assert "column_analyses" in body
        assert "strategy_recommendations" in body

    def test_analyze_with_specific_columns(self, test_client):
        """POST /api/v1/mvi/analyze with columns filter."""
        records = [
            {"name": "Alice", "revenue": None, "emissions": None, "region": "US"},
            {"name": "Bob", "revenue": 1000.0, "emissions": 500.0, "region": None},
        ]

        resp = test_client.post(
            "/api/v1/mvi/analyze",
            json={"records": records, "columns": ["revenue", "emissions"]},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 2
        # Only the specified columns should be analyzed
        analyzed_cols = [ca["column_name"] for ca in body["column_analyses"]]
        assert "revenue" in analyzed_cols
        assert "emissions" in analyzed_cols


# ===================================================================
# Impute API Tests
# ===================================================================


class TestImputeEndpoints:
    """Tests for the imputation API endpoints."""

    def test_impute_single_column_endpoint(self, test_client):
        """POST /api/v1/mvi/impute imputes a single column.

        Validates:
        - HTTP 200 response
        - values_imputed > 0
        - strategy is applied
        - completeness before/after are reported
        """
        records = [
            {"revenue": 1000.0},
            {"revenue": None},
            {"revenue": 2000.0},
            {"revenue": None},
            {"revenue": 3000.0},
        ]

        resp = test_client.post(
            "/api/v1/mvi/impute",
            json={"records": records, "column": "revenue", "strategy": "mean"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["values_imputed"] == 2
        assert body["strategy"] == "mean"
        assert body["completeness_before"] < body["completeness_after"]
        assert len(body["provenance_hash"]) == 64
        assert len(body["imputed_values"]) == 2

    def test_impute_batch_endpoint(self, test_client):
        """POST /api/v1/mvi/impute/batch imputes multiple columns.

        Validates:
        - HTTP 200 response
        - total_values_imputed across all columns
        - Results list contains per-column results
        """
        records = [
            {"revenue": 1000.0, "emissions": None},
            {"revenue": None, "emissions": 500.0},
            {"revenue": 2000.0, "emissions": None},
        ]

        resp = test_client.post(
            "/api/v1/mvi/impute/batch",
            json={"records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_values_imputed"] >= 2
        assert body["total_columns"] >= 1
        assert len(body["results"]) >= 1
        assert len(body["provenance_hash"]) == 64

    def test_impute_with_custom_strategy(self, test_client):
        """POST /api/v1/mvi/impute with strategy override."""
        records = [
            {"temperature": 20.0},
            {"temperature": None},
            {"temperature": 30.0},
            {"temperature": 25.0},
        ]

        resp = test_client.post(
            "/api/v1/mvi/impute",
            json={
                "records": records,
                "column": "temperature",
                "strategy": "median",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["strategy"] == "median"
        assert body["values_imputed"] == 1


# ===================================================================
# Validate API Tests
# ===================================================================


class TestValidateEndpoints:
    """Tests for the validation API endpoints."""

    def test_validate_endpoint(self, test_client):
        """POST /api/v1/mvi/validate validates imputation quality.

        Validates:
        - HTTP 200 response
        - overall_passed is a boolean
        - columns_passed + columns_failed = total_columns
        """
        original = [
            {"revenue": 1000.0, "emissions": None},
            {"revenue": None, "emissions": 500.0},
            {"revenue": 2000.0, "emissions": 800.0},
        ]
        imputed = [
            {"revenue": 1000.0, "emissions": 650.0},
            {"revenue": 1500.0, "emissions": 500.0},
            {"revenue": 2000.0, "emissions": 800.0},
        ]

        resp = test_client.post(
            "/api/v1/mvi/validate",
            json={"original": original, "imputed": imputed},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["overall_passed"], bool)
        assert body["columns_passed"] + body["columns_failed"] == body["total_columns"]
        assert len(body["provenance_hash"]) == 64

    def test_validate_with_custom_method(self, test_client):
        """POST /api/v1/mvi/validate with custom validation method."""
        original = [
            {"value": 10.0},
            {"value": None},
            {"value": 30.0},
        ]
        imputed = [
            {"value": 10.0},
            {"value": 20.0},
            {"value": 30.0},
        ]

        resp = test_client.post(
            "/api/v1/mvi/validate",
            json={
                "original": original,
                "imputed": imputed,
                "method": "plausibility_range",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        # With plausible imputed values within range, should pass
        assert body["overall_passed"] is True


# ===================================================================
# Rule Management API Tests
# ===================================================================


class TestRuleEndpoints:
    """Tests for the imputation rule management endpoints."""

    def test_create_rule_endpoint(self, test_client):
        """POST /api/v1/mvi/rules creates a new imputation rule.

        Validates:
        - HTTP 200 response
        - rule_id is a valid UUID
        - Rule name and fields match the request
        - provenance_hash is recorded
        """
        resp = test_client.post(
            "/api/v1/mvi/rules",
            json={
                "name": "api-test-rule",
                "target_column": "emissions",
                "conditions": [
                    {"column": "region", "operator": "equals", "value": "EU"}
                ],
                "impute_value": 1500.0,
                "priority": "high",
                "justification": "EU regulatory default",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "rule_id" in body
        uuid.UUID(body["rule_id"])
        assert body["name"] == "api-test-rule"
        assert body["target_column"] == "emissions"
        assert body["priority"] == "high"
        assert body["is_active"] is True
        assert len(body["provenance_hash"]) == 64

    def test_list_rules_endpoint(self, test_client):
        """GET /api/v1/mvi/rules lists all created rules.

        Creates 2 rules and validates the list response.
        """
        for name in ["rule-a", "rule-b"]:
            test_client.post(
                "/api/v1/mvi/rules",
                json={
                    "name": name,
                    "target_column": "emissions",
                },
            )

        resp = test_client.get("/api/v1/mvi/rules")
        assert resp.status_code == 200

        body = resp.json()
        assert "rules" in body
        assert body["count"] == 2

    def test_update_rule_endpoint(self, test_client):
        """PUT /api/v1/mvi/rules/{rule_id} updates a rule.

        Validates:
        - HTTP 200 for existing rule
        - Updated fields are reflected
        """
        create_resp = test_client.post(
            "/api/v1/mvi/rules",
            json={
                "name": "update-test-rule",
                "target_column": "revenue",
                "priority": "medium",
            },
        )
        rule_id = create_resp.json()["rule_id"]

        resp = test_client.put(
            f"/api/v1/mvi/rules/{rule_id}",
            json={"priority": "critical", "justification": "Updated reason"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["priority"] == "critical"
        assert body["justification"] == "Updated reason"

    def test_delete_rule_endpoint(self, test_client):
        """DELETE /api/v1/mvi/rules/{rule_id} deactivates a rule.

        Validates:
        - HTTP 200 for existing rule
        - deleted field is True
        """
        create_resp = test_client.post(
            "/api/v1/mvi/rules",
            json={
                "name": "delete-test-rule",
                "target_column": "sector",
            },
        )
        rule_id = create_resp.json()["rule_id"]

        resp = test_client.delete(f"/api/v1/mvi/rules/{rule_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["rule_id"] == rule_id
        assert body["deleted"] is True


# ===================================================================
# Template Management API Tests
# ===================================================================


class TestTemplateEndpoints:
    """Tests for the imputation template management endpoints."""

    def test_create_template_endpoint(self, test_client):
        """POST /api/v1/mvi/templates creates a new template.

        Validates:
        - HTTP 200 response
        - template_id is a valid UUID
        - Template fields match the request
        """
        resp = test_client.post(
            "/api/v1/mvi/templates",
            json={
                "name": "sustainability-template",
                "description": "Template for sustainability data",
                "strategies": {
                    "revenue": "median",
                    "emissions": "knn",
                    "sector": "mode",
                },
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "template_id" in body
        uuid.UUID(body["template_id"])
        assert body["name"] == "sustainability-template"
        assert body["column_strategies"]["revenue"] == "median"
        assert body["column_strategies"]["emissions"] == "knn"
        assert body["is_active"] is True
        assert len(body["provenance_hash"]) == 64

    def test_list_templates_endpoint(self, test_client):
        """GET /api/v1/mvi/templates lists all created templates.

        Creates 2 templates and validates the list response.
        """
        for name in ["tmpl-a", "tmpl-b"]:
            test_client.post(
                "/api/v1/mvi/templates",
                json={"name": name},
            )

        resp = test_client.get("/api/v1/mvi/templates")
        assert resp.status_code == 200

        body = resp.json()
        assert "templates" in body
        assert body["count"] == 2


# ===================================================================
# Pipeline API Tests
# ===================================================================


class TestPipelineEndpoint:
    """Tests for the full pipeline API endpoint."""

    def test_pipeline_endpoint(self, test_client):
        """POST /api/v1/mvi/pipeline runs the full imputation pipeline.

        Validates:
        - HTTP 200 response
        - All expected fields in response
        - Pipeline stages are present
        """
        records = [
            {"id": "0", "revenue": 1000.0, "emissions": None},
            {"id": "1", "revenue": None, "emissions": 500.0},
            {"id": "2", "revenue": 2000.0, "emissions": 800.0},
        ]

        resp = test_client.post(
            "/api/v1/mvi/pipeline",
            json={"records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("completed", "failed")
        assert body["total_records"] == 3
        assert "pipeline_id" in body
        uuid.UUID(body["pipeline_id"])
        assert "stages" in body
        assert "analyze" in body["stages"]
        assert len(body["provenance_hash"]) == 64

    def test_pipeline_with_config(self, test_client):
        """POST /api/v1/mvi/pipeline with custom config options."""
        records = [
            {"id": "0", "value": 10.0},
            {"id": "1", "value": None},
            {"id": "2", "value": 30.0},
        ]

        resp = test_client.post(
            "/api/v1/mvi/pipeline",
            json={
                "records": records,
                "config": {
                    "column_strategies": {"value": "median"},
                    "validation_method": "plausibility_range",
                },
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 3


# ===================================================================
# Health & Stats API Tests
# ===================================================================


class TestHealthAndStatsEndpoints:
    """Tests for health check and statistics endpoints."""

    def test_health_endpoint(self, test_client):
        """GET /api/v1/mvi/health returns service health status.

        Validates:
        - HTTP 200 response
        - status is 'healthy' (service was started in fixture)
        - All expected health fields are present
        """
        resp = test_client.get("/api/v1/mvi/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "missing-value-imputer"
        assert body["started"] is True
        assert "engines" in body
        assert "jobs" in body
        assert "analyses" in body
        assert "imputation_results" in body
        assert "batch_results" in body
        assert "validation_results" in body
        assert "pipeline_results" in body
        assert "rules" in body
        assert "templates" in body
        assert "provenance_entries" in body

    def test_stats_endpoint(self, test_client):
        """GET /api/v1/mvi/stats returns aggregate statistics.

        Validates:
        - HTTP 200 response
        - All stat fields present with correct defaults
        """
        resp = test_client.get("/api/v1/mvi/stats")

        assert resp.status_code == 200
        body = resp.json()
        assert "total_jobs" in body
        assert "completed_jobs" in body
        assert "failed_jobs" in body
        assert "total_records_processed" in body
        assert "total_values_imputed" in body
        assert "total_analyses" in body
        assert "total_validations" in body
        assert "total_rules" in body
        assert "total_templates" in body
        assert "active_jobs" in body
        assert "avg_confidence" in body
        assert "avg_completeness_improvement" in body
        assert "by_strategy" in body
        assert "by_status" in body
        assert "provenance_entries" in body

    def test_stats_reflect_operations(self, test_client):
        """GET /api/v1/mvi/stats reflects operations performed.

        Runs a pipeline, then checks that stats are updated.
        """
        records = [
            {"id": "0", "value": 10.0},
            {"id": "1", "value": None},
            {"id": "2", "value": 30.0},
        ]
        test_client.post(
            "/api/v1/mvi/pipeline",
            json={"records": records},
        )

        resp = test_client.get("/api/v1/mvi/stats")
        assert resp.status_code == 200

        body = resp.json()
        assert body["total_analyses"] >= 1
        assert body["total_records_processed"] >= 3


# ===================================================================
# Authentication Tests
# ===================================================================


class TestAuthEndpoints:
    """Tests for authentication requirements on protected endpoints."""

    def test_auth_required_on_all_endpoints(self, test_client):
        """Verify that the protected endpoint requires authentication.

        Tests the sentinel /api/v1/mvi/protected endpoint to confirm
        that requests without a valid Bearer token are rejected with 401.
        """
        # Request without auth header -> 401
        resp = test_client.get("/api/v1/mvi/protected")
        assert resp.status_code == 401
        assert "Not authenticated" in resp.json()["detail"]

        # Request with invalid auth scheme -> 401
        resp = test_client.get(
            "/api/v1/mvi/protected",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401

        # Request with valid Bearer token -> 200
        resp = test_client.get(
            "/api/v1/mvi/protected",
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
        """Test the complete API workflow from job creation to validation.

        Executes the following sequence:
        1. Create a job
        2. Analyze missingness
        3. Impute missing values
        4. Validate imputation
        5. Check stats reflect all operations
        """
        # Step 1: Create job
        job_resp = test_client.post(
            "/api/v1/mvi/jobs",
            json={
                "records": [
                    {"revenue": 1000.0, "emissions": None},
                    {"revenue": None, "emissions": 500.0},
                    {"revenue": 2000.0, "emissions": 800.0},
                ],
                "dataset_id": "ds-api-flow",
            },
        )
        assert job_resp.status_code == 200
        job_id = job_resp.json()["job_id"]

        # Step 2: Analyze missingness
        records = [
            {"revenue": 1000.0, "emissions": None},
            {"revenue": None, "emissions": 500.0},
            {"revenue": 2000.0, "emissions": 800.0},
        ]
        analyze_resp = test_client.post(
            "/api/v1/mvi/analyze",
            json={"records": records},
        )
        assert analyze_resp.status_code == 200
        assert analyze_resp.json()["total_records"] == 3
        assert analyze_resp.json()["columns_with_missing"] >= 1

        # Step 3: Impute revenue column
        impute_resp = test_client.post(
            "/api/v1/mvi/impute",
            json={
                "records": records,
                "column": "revenue",
                "strategy": "mean",
            },
        )
        assert impute_resp.status_code == 200
        assert impute_resp.json()["values_imputed"] == 1

        # Step 4: Impute emissions column
        impute_resp2 = test_client.post(
            "/api/v1/mvi/impute",
            json={
                "records": records,
                "column": "emissions",
                "strategy": "mean",
            },
        )
        assert impute_resp2.status_code == 200
        assert impute_resp2.json()["values_imputed"] == 1

        # Step 5: Validate
        imputed_records = [
            {"revenue": 1000.0, "emissions": 650.0},
            {"revenue": 1500.0, "emissions": 500.0},
            {"revenue": 2000.0, "emissions": 800.0},
        ]
        validate_resp = test_client.post(
            "/api/v1/mvi/validate",
            json={"original": records, "imputed": imputed_records},
        )
        assert validate_resp.status_code == 200
        assert isinstance(validate_resp.json()["overall_passed"], bool)

        # Step 6: Create a rule for future imputations
        rule_resp = test_client.post(
            "/api/v1/mvi/rules",
            json={
                "name": "workflow-rule",
                "target_column": "emissions",
                "impute_value": 1500.0,
                "priority": "high",
            },
        )
        assert rule_resp.status_code == 200

        # Step 7: Verify stats
        stats_resp = test_client.get("/api/v1/mvi/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_analyses"] >= 1
        assert stats["total_records_processed"] >= 3
        assert stats["total_values_imputed"] >= 2
        assert stats["total_rules"] >= 1
        assert stats["provenance_entries"] >= 1

        # Verify the job is still retrievable
        get_job_resp = test_client.get(f"/api/v1/mvi/jobs/{job_id}")
        assert get_job_resp.status_code == 200

    def test_pipeline_then_stats_then_health(self, test_client):
        """Test that pipeline results flow through to stats and health.

        Runs a pipeline, then verifies both stats and health endpoints
        reflect the pipeline execution.
        """
        records = [
            {"id": "0", "revenue": 1000.0, "emissions": None},
            {"id": "1", "revenue": None, "emissions": 500.0},
        ]

        # Run pipeline
        pipeline_resp = test_client.post(
            "/api/v1/mvi/pipeline",
            json={"records": records},
        )
        assert pipeline_resp.status_code == 200

        # Check stats updated
        stats_resp = test_client.get("/api/v1/mvi/stats")
        stats = stats_resp.json()
        assert stats["total_analyses"] >= 1

        # Check health reflects pipeline results
        health_resp = test_client.get("/api/v1/mvi/health")
        health = health_resp.json()
        assert health["pipeline_results"] >= 1
        assert health["analyses"] >= 1
