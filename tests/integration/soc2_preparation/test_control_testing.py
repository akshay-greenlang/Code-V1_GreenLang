# -*- coding: utf-8 -*-
"""
Integration tests for SOC 2 Control Testing - SEC-009 Phase 11

End-to-end tests for control testing workflow:
1. Register test cases
2. Execute test suite
3. Track results
4. Generate reports

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def testing_app() -> FastAPI:
    """Create FastAPI app for control testing tests."""
    from greenlang.infrastructure.soc2_preparation.api import soc2_router

    app = FastAPI(title="Control Testing Integration Test")
    app.include_router(soc2_router)
    return app


@pytest.fixture
def testing_client(testing_app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(testing_app)


# ============================================================================
# Test Suite Execution Tests
# ============================================================================


class TestControlTestingIntegration:
    """Integration tests for control testing workflow."""

    def test_full_test_execution_workflow(self, testing_client: TestClient) -> None:
        """Test complete control testing workflow."""
        # Step 1: List available tests
        response = testing_client.get("/api/v1/soc2/tests")
        assert response.status_code == 200
        tests_data = response.json()
        assert tests_data["total"] > 0

        # Step 2: Execute test suite
        response = testing_client.post(
            "/api/v1/soc2/tests/run",
            json={"criteria": ["CC6"]},
        )
        assert response.status_code == 202
        run_data = response.json()
        run_id = run_data["run_id"]
        assert run_data["status"] == "running"

        # Step 3: Get test run details
        response = testing_client.get(f"/api/v1/soc2/tests/runs/{run_id}")
        assert response.status_code == 200
        run_details = response.json()
        assert run_details["run_id"] == run_id

        # Step 4: Get individual test result
        response = testing_client.get("/api/v1/soc2/tests/CC6.1.1/result")
        assert response.status_code == 200
        result_data = response.json()
        assert "status" in result_data

        # Step 5: Generate report
        response = testing_client.get("/api/v1/soc2/tests/report")
        assert response.status_code == 200
        report_data = response.json()
        assert "run" in report_data
        assert "results" in report_data

    def test_parallel_test_execution(self, testing_client: TestClient) -> None:
        """Test parallel test execution mode."""
        response = testing_client.post(
            "/api/v1/soc2/tests/run",
            json={"parallel": True, "criteria": ["CC6", "CC7"]},
        )
        assert response.status_code == 202
        data = response.json()
        # Parallel should have lower estimated duration
        assert data["estimated_duration_seconds"] > 0

    def test_test_filtering_workflow(self, testing_client: TestClient) -> None:
        """Test filtering tests by various criteria."""
        # Filter by criterion
        response = testing_client.get("/api/v1/soc2/tests?criterion=CC6")
        assert response.status_code == 200
        data = response.json()
        for test in data["tests"]:
            assert test["criterion_id"].startswith("CC6")

        # Filter by type
        response = testing_client.get("/api/v1/soc2/tests?test_type=automated")
        assert response.status_code == 200
        data = response.json()
        for test in data["tests"]:
            assert test["test_type"] == "automated"

    def test_test_report_formats(self, testing_client: TestClient) -> None:
        """Test report generation in different formats."""
        formats = ["json", "markdown"]
        for fmt in formats:
            response = testing_client.get(f"/api/v1/soc2/tests/report?format={fmt}")
            assert response.status_code == 200
            data = response.json()
            assert data["report_format"] == fmt


# ============================================================================
# Test Run History Tests
# ============================================================================


class TestTestRunHistoryIntegration:
    """Integration tests for test run history."""

    def test_list_historical_runs(self, testing_client: TestClient) -> None:
        """Test listing historical test runs."""
        response = testing_client.get("/api/v1/soc2/tests/runs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        for run in data:
            assert "run_id" in run
            assert "status" in run
            assert "pass_rate" in run

    def test_run_pagination(self, testing_client: TestClient) -> None:
        """Test run listing pagination."""
        response = testing_client.get("/api/v1/soc2/tests/runs?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5
