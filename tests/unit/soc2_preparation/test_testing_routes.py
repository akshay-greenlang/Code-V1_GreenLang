# -*- coding: utf-8 -*-
"""
Unit tests for SOC 2 Control Testing API Routes - SEC-009 Phase 11

Tests the control testing API endpoints:
- GET /tests
- POST /tests/run
- GET /tests/runs
- GET /tests/runs/{run_id}
- GET /tests/{test_id}/result
- GET /tests/report

Coverage targets: 85%+ of testing_routes.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.infrastructure.soc2_preparation.api.testing_routes import (
    router,
    TestCase,
    TestCaseListResponse,
    TestRunRequest,
    TestRunResponse,
    TestResult,
    TestRun,
    TestRunReport,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def app() -> FastAPI:
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/soc2")
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


# ============================================================================
# Test TestCase Model
# ============================================================================


class TestTestCaseModel:
    """Tests for TestCase Pydantic model."""

    def test_test_case_required_fields(self) -> None:
        """TestCase requires test_id, criterion_id, description."""
        test = TestCase(
            test_id="CC6.1.1",
            criterion_id="CC6.1",
            test_type="automated",
            description="Verify MFA enforcement",
        )
        assert test.test_id == "CC6.1.1"
        assert test.criterion_id == "CC6.1"
        assert test.test_type == "automated"

    def test_test_case_defaults(self) -> None:
        """TestCase has correct defaults."""
        test = TestCase(
            test_id="CC6.1.1",
            criterion_id="CC6.1",
            test_type="automated",
            description="Test description",
        )
        assert test.frequency == "quarterly"
        assert test.owner == ""
        assert test.enabled is True
        assert test.procedure == ""

    def test_test_case_with_all_fields(self) -> None:
        """TestCase with all optional fields."""
        test = TestCase(
            test_id="CC6.1.2",
            criterion_id="CC6.1",
            test_type="semi_automated",
            description="Password policy check",
            procedure="Query auth config",
            expected_result="Policy meets requirements",
            frequency="weekly",
            owner="security-team",
            enabled=True,
            last_executed=datetime.now(timezone.utc),
            last_status="passed",
        )
        assert test.frequency == "weekly"
        assert test.owner == "security-team"


# ============================================================================
# Test TestRunRequest Model
# ============================================================================


class TestTestRunRequestModel:
    """Tests for TestRunRequest Pydantic model."""

    def test_run_request_defaults(self) -> None:
        """TestRunRequest has correct defaults."""
        request = TestRunRequest()
        assert request.criteria is None
        assert request.test_type is None
        assert request.parallel is False
        assert request.run_name is None

    def test_run_request_with_criteria(self) -> None:
        """TestRunRequest with criteria filter."""
        request = TestRunRequest(
            criteria=["CC6", "CC7"],
            parallel=True,
        )
        assert request.criteria == ["CC6", "CC7"]
        assert request.parallel is True


# ============================================================================
# Test TestResult Model
# ============================================================================


class TestTestResultModel:
    """Tests for TestResult Pydantic model."""

    def test_result_required_fields(self) -> None:
        """TestResult requires core fields."""
        result = TestResult(
            result_id=uuid.uuid4(),
            test_id="CC6.1.1",
            test_run_id=uuid.uuid4(),
            status="passed",
            started_at=datetime.now(timezone.utc),
        )
        assert result.status == "passed"
        assert result.actual_result == ""
        assert result.error_message == ""

    def test_result_with_evidence(self) -> None:
        """TestResult with evidence count."""
        result = TestResult(
            result_id=uuid.uuid4(),
            test_id="CC6.1.1",
            test_run_id=uuid.uuid4(),
            status="passed",
            evidence_count=5,
            started_at=datetime.now(timezone.utc),
        )
        assert result.evidence_count == 5


# ============================================================================
# Test GET /tests Endpoint
# ============================================================================


class TestListTestsEndpoint:
    """Tests for GET /tests endpoint."""

    def test_list_tests_success(self, client: TestClient) -> None:
        """GET /tests returns test list."""
        response = client.get("/api/v1/soc2/tests")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "tests" in data
        assert isinstance(data["tests"], list)

    def test_list_tests_includes_counts(self, client: TestClient) -> None:
        """GET /tests includes count breakdowns."""
        response = client.get("/api/v1/soc2/tests")
        assert response.status_code == 200
        data = response.json()
        assert "by_criterion" in data
        assert "by_type" in data

    def test_list_tests_filter_by_criterion(self, client: TestClient) -> None:
        """GET /tests with criterion filter."""
        response = client.get("/api/v1/soc2/tests?criterion=CC6")
        assert response.status_code == 200
        data = response.json()
        for test in data["tests"]:
            assert test["criterion_id"].startswith("CC6")

    def test_list_tests_filter_by_type(self, client: TestClient) -> None:
        """GET /tests with type filter."""
        response = client.get("/api/v1/soc2/tests?test_type=automated")
        assert response.status_code == 200
        data = response.json()
        for test in data["tests"]:
            assert test["test_type"] == "automated"

    def test_list_tests_enabled_only(self, client: TestClient) -> None:
        """GET /tests with enabled_only filter."""
        response = client.get("/api/v1/soc2/tests?enabled_only=true")
        assert response.status_code == 200


# ============================================================================
# Test POST /tests/run Endpoint
# ============================================================================


class TestRunTestsEndpoint:
    """Tests for POST /tests/run endpoint."""

    def test_run_tests_success(self, client: TestClient) -> None:
        """POST /tests/run starts test execution."""
        response = client.post(
            "/api/v1/soc2/tests/run",
            json={},
        )
        assert response.status_code == 202
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "running"

    def test_run_tests_with_criteria(self, client: TestClient) -> None:
        """POST /tests/run with criteria filter."""
        response = client.post(
            "/api/v1/soc2/tests/run",
            json={"criteria": ["CC6", "CC7"]},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["test_count"] > 0

    def test_run_tests_parallel(self, client: TestClient) -> None:
        """POST /tests/run with parallel execution."""
        response = client.post(
            "/api/v1/soc2/tests/run",
            json={"parallel": True},
        )
        assert response.status_code == 202
        data = response.json()
        # Parallel should have shorter estimated duration
        assert data["estimated_duration_seconds"] > 0

    def test_run_tests_includes_estimate(self, client: TestClient) -> None:
        """POST /tests/run includes duration estimate."""
        response = client.post(
            "/api/v1/soc2/tests/run",
            json={"criteria": ["CC6"]},
        )
        assert response.status_code == 202
        data = response.json()
        assert "estimated_duration_seconds" in data


# ============================================================================
# Test GET /tests/runs Endpoint
# ============================================================================


class TestListRunsEndpoint:
    """Tests for GET /tests/runs endpoint."""

    def test_list_runs_success(self, client: TestClient) -> None:
        """GET /tests/runs returns run list."""
        response = client.get("/api/v1/soc2/tests/runs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_runs_pagination(self, client: TestClient) -> None:
        """GET /tests/runs with pagination."""
        response = client.get("/api/v1/soc2/tests/runs?limit=10&offset=0")
        assert response.status_code == 200

    def test_list_runs_item_structure(self, client: TestClient) -> None:
        """GET /tests/runs items have required fields."""
        response = client.get("/api/v1/soc2/tests/runs")
        assert response.status_code == 200
        data = response.json()
        for run in data:
            assert "run_id" in run
            assert "status" in run
            assert "total_tests" in run
            assert "pass_rate" in run


# ============================================================================
# Test GET /tests/runs/{run_id} Endpoint
# ============================================================================


class TestGetRunEndpoint:
    """Tests for GET /tests/runs/{run_id} endpoint."""

    def test_get_run_success(self, client: TestClient) -> None:
        """GET /tests/runs/{id} returns run details."""
        run_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/soc2/tests/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id
        assert "total_tests" in data
        assert "passed_count" in data


# ============================================================================
# Test GET /tests/{test_id}/result Endpoint
# ============================================================================


class TestGetResultEndpoint:
    """Tests for GET /tests/{test_id}/result endpoint."""

    def test_get_result_success(self, client: TestClient) -> None:
        """GET /tests/{id}/result returns test result."""
        response = client.get("/api/v1/soc2/tests/CC6.1.1/result")
        assert response.status_code == 200
        data = response.json()
        assert data["test_id"] == "CC6.1.1"
        assert "status" in data
        assert "duration_ms" in data

    def test_get_result_with_run_id(self, client: TestClient) -> None:
        """GET /tests/{id}/result with specific run."""
        run_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/soc2/tests/CC6.1.1/result?run_id={run_id}")
        assert response.status_code == 200


# ============================================================================
# Test GET /tests/report Endpoint
# ============================================================================


class TestGetReportEndpoint:
    """Tests for GET /tests/report endpoint."""

    def test_get_report_success(self, client: TestClient) -> None:
        """GET /tests/report returns test report."""
        response = client.get("/api/v1/soc2/tests/report")
        assert response.status_code == 200
        data = response.json()
        assert "run" in data
        assert "results" in data
        assert "findings" in data
        assert "recommendations" in data

    def test_get_report_json_format(self, client: TestClient) -> None:
        """GET /tests/report with JSON format."""
        response = client.get("/api/v1/soc2/tests/report?format=json")
        assert response.status_code == 200
        data = response.json()
        assert data["report_format"] == "json"

    def test_get_report_markdown_format(self, client: TestClient) -> None:
        """GET /tests/report with markdown format."""
        response = client.get("/api/v1/soc2/tests/report?format=markdown")
        assert response.status_code == 200

    def test_get_report_includes_findings(self, client: TestClient) -> None:
        """GET /tests/report includes findings."""
        response = client.get("/api/v1/soc2/tests/report")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["findings"], list)

    def test_get_report_includes_recommendations(self, client: TestClient) -> None:
        """GET /tests/report includes recommendations."""
        response = client.get("/api/v1/soc2/tests/report")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["recommendations"], list)
