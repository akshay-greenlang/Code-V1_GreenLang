# -*- coding: utf-8 -*-
"""
Load tests for SOC 2 Concurrent Assessments - SEC-009 Phase 11

Performance tests for concurrent assessment operations:
- Multiple simultaneous assessment runs
- Concurrent score calculations
- Parallel gap analysis
- Dashboard aggregation under load

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def assessment_app() -> FastAPI:
    """Create FastAPI app for assessment load testing."""
    from greenlang.infrastructure.soc2_preparation.api import soc2_router

    app = FastAPI(title="Assessment Load Test")
    app.include_router(soc2_router)
    return app


@pytest.fixture
def assessment_client(assessment_app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(assessment_app)


# ============================================================================
# Concurrent Assessment Tests
# ============================================================================


class TestConcurrentAssessments:
    """Load tests for concurrent assessment operations."""

    def test_concurrent_assessment_runs(
        self, assessment_client: TestClient
    ) -> None:
        """Test multiple concurrent assessment runs."""
        num_assessments = 10
        category_sets = [
            ["security"],
            ["availability"],
            ["security", "availability"],
            ["confidentiality"],
            ["security", "privacy"],
        ]

        def run_assessment(categories):
            start = time.time()
            response = assessment_client.post(
                "/api/v1/soc2/assessment/run",
                json={"categories": categories},
            )
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
                "assessment_id": response.json().get("assessment_id"),
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(run_assessment, category_sets[i % len(category_sets)])
                for i in range(num_assessments)
            ]
            results = [f.result() for f in futures]

        # All should be accepted (202)
        accepted_count = sum(1 for r in results if r["status"] == 202)
        assert accepted_count == num_assessments

        # Each should have unique ID
        assessment_ids = [r["assessment_id"] for r in results]
        assert len(set(assessment_ids)) == num_assessments

        # Average duration should be reasonable
        avg_duration = sum(r["duration"] for r in results) / num_assessments
        assert avg_duration < 2.0, f"Average duration {avg_duration:.3f}s exceeds 2s"

    def test_concurrent_score_requests(
        self, assessment_client: TestClient
    ) -> None:
        """Test concurrent readiness score requests."""
        num_requests = 100
        results = []

        def get_score():
            start = time.time()
            response = assessment_client.get("/api/v1/soc2/assessment/score")
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
            }

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_score) for _ in range(num_requests)]
            results = [f.result() for f in futures]

        # All should succeed
        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count == num_requests

        # Calculate P95 latency
        durations = sorted([r["duration"] for r in results])
        p95_index = int(0.95 * len(durations))
        p95_latency = durations[p95_index]
        assert p95_latency < 0.5, f"P95 latency {p95_latency:.3f}s exceeds 500ms"

    def test_concurrent_gap_requests(self, assessment_client: TestClient) -> None:
        """Test concurrent gap analysis requests."""
        num_requests = 50
        severity_filters = ["critical", "high", "medium", "low", None]

        def get_gaps(severity):
            start = time.time()
            url = "/api/v1/soc2/assessment/gaps"
            if severity:
                url += f"?severity={severity}"
            response = assessment_client.get(url)
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(get_gaps, severity_filters[i % len(severity_filters)])
                for i in range(num_requests)
            ]
            results = [f.result() for f in futures]

        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count == num_requests


# ============================================================================
# Dashboard Load Tests
# ============================================================================


class TestDashboardLoad:
    """Load tests for dashboard endpoints."""

    def test_concurrent_dashboard_requests(
        self, assessment_client: TestClient
    ) -> None:
        """Test concurrent dashboard summary requests."""
        num_requests = 100
        results = []

        def get_dashboard():
            start = time.time()
            response = assessment_client.get("/api/v1/soc2/dashboard/summary")
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
            }

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_dashboard) for _ in range(num_requests)]
            results = [f.result() for f in futures]

        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count == num_requests

        # Dashboard should be fast - P95 under 300ms
        durations = sorted([r["duration"] for r in results])
        p95_index = int(0.95 * len(durations))
        p95_latency = durations[p95_index]
        assert p95_latency < 0.3, f"P95 latency {p95_latency:.3f}s exceeds 300ms"

    def test_concurrent_metrics_requests(
        self, assessment_client: TestClient
    ) -> None:
        """Test concurrent metrics requests."""
        num_requests = 50
        results = []

        def get_metrics():
            start = time.time()
            response = assessment_client.get("/api/v1/soc2/dashboard/metrics")
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_metrics) for _ in range(num_requests)]
            results = [f.result() for f in futures]

        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count == num_requests

    def test_mixed_dashboard_endpoints(
        self, assessment_client: TestClient
    ) -> None:
        """Test mixed dashboard endpoint requests."""
        num_requests = 60
        endpoints = [
            "/api/v1/soc2/dashboard/summary",
            "/api/v1/soc2/dashboard/timeline",
            "/api/v1/soc2/dashboard/metrics",
        ]

        def get_endpoint(endpoint):
            start = time.time()
            response = assessment_client.get(endpoint)
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
                "endpoint": endpoint,
            }

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(get_endpoint, endpoints[i % len(endpoints)])
                for i in range(num_requests)
            ]
            results = [f.result() for f in futures]

        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count == num_requests


# ============================================================================
# Stress Tests
# ============================================================================


class TestAssessmentStress:
    """Stress tests for assessment system."""

    def test_sustained_assessment_load(
        self, assessment_client: TestClient
    ) -> None:
        """Test sustained load on assessment endpoints."""
        duration_seconds = 5
        requests_per_second = 30
        total_requests = duration_seconds * requests_per_second

        endpoints = [
            "/api/v1/soc2/assessment",
            "/api/v1/soc2/assessment/score",
            "/api/v1/soc2/assessment/gaps",
        ]

        start_time = time.time()
        results = []

        def make_timed_request(request_num):
            target_time = start_time + (request_num / requests_per_second)
            current = time.time()
            if current < target_time:
                time.sleep(target_time - current)

            endpoint = endpoints[request_num % len(endpoints)]
            request_start = time.time()
            response = assessment_client.get(endpoint)
            duration = time.time() - request_start
            return {
                "status": response.status_code,
                "duration": duration,
                "endpoint": endpoint,
            }

        with ThreadPoolExecutor(max_workers=requests_per_second) as executor:
            futures = [
                executor.submit(make_timed_request, i)
                for i in range(total_requests)
            ]
            results = [f.result() for f in futures]

        # At least 95% should succeed
        success_count = sum(1 for r in results if r["status"] == 200)
        success_rate = success_count / total_requests
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"

        # P99 latency should be under 2 seconds
        durations = sorted([r["duration"] for r in results])
        p99_index = int(0.99 * len(durations))
        p99_latency = durations[p99_index]
        assert p99_latency < 2.0, f"P99 latency {p99_latency:.3f}s exceeds 2s"

    def test_burst_assessment_requests(
        self, assessment_client: TestClient
    ) -> None:
        """Test handling of burst traffic."""
        burst_size = 50
        results = []

        def burst_request():
            start = time.time()
            response = assessment_client.get("/api/v1/soc2/assessment")
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
            }

        # Send all requests at once
        with ThreadPoolExecutor(max_workers=burst_size) as executor:
            futures = [executor.submit(burst_request) for _ in range(burst_size)]
            results = [f.result() for f in futures]

        # All should succeed even under burst
        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count == burst_size
