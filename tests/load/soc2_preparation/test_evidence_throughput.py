# -*- coding: utf-8 -*-
"""
Load tests for SOC 2 Evidence Throughput - SEC-009 Phase 11

Performance tests for evidence collection and packaging:
- Concurrent evidence collection requests
- Large evidence package creation
- Parallel source adapter queries
- Download request handling

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
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
def load_app() -> FastAPI:
    """Create FastAPI app for load testing."""
    from greenlang.infrastructure.soc2_preparation.api import soc2_router

    app = FastAPI(title="SOC 2 Load Test")
    app.include_router(soc2_router)
    return app


@pytest.fixture
def load_client(load_app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(load_app)


# ============================================================================
# Evidence Collection Throughput Tests
# ============================================================================


class TestEvidenceCollectionThroughput:
    """Load tests for evidence collection."""

    def test_concurrent_evidence_list_requests(
        self, load_client: TestClient
    ) -> None:
        """Test concurrent evidence listing requests."""
        num_requests = 50
        results = []

        def make_request():
            start = time.time()
            response = load_client.get("/api/v1/soc2/evidence")
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [f.result() for f in futures]

        # All requests should succeed
        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count == num_requests

        # Calculate average response time
        avg_duration = sum(r["duration"] for r in results) / num_requests
        # Average should be under 1 second
        assert avg_duration < 1.0, f"Average response time {avg_duration:.3f}s exceeds 1s"

    def test_concurrent_evidence_collection_requests(
        self, load_client: TestClient
    ) -> None:
        """Test concurrent evidence collection triggers."""
        num_requests = 20
        criteria_sets = [
            ["CC6.1"],
            ["CC6.2"],
            ["CC7.1"],
            ["CC7.2"],
        ]

        def make_collection_request(criteria):
            start = time.time()
            response = load_client.post(
                "/api/v1/soc2/evidence/collect",
                json={
                    "criteria": criteria,
                    "period_start": "2026-01-01T00:00:00Z",
                    "period_end": "2026-02-01T00:00:00Z",
                },
            )
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_collection_request, criteria_sets[i % len(criteria_sets)])
                for i in range(num_requests)
            ]
            results = [f.result() for f in futures]

        # All requests should be accepted (202)
        accepted_count = sum(1 for r in results if r["status"] == 202)
        assert accepted_count == num_requests

    def test_evidence_package_creation_throughput(
        self, load_client: TestClient
    ) -> None:
        """Test evidence package creation performance."""
        num_packages = 10
        results = []

        def create_package(package_num):
            start = time.time()
            response = load_client.post(
                "/api/v1/soc2/evidence/package",
                json={
                    "package_name": f"Load Test Package {package_num}",
                    "criteria": ["CC6", "CC7", "CC8"],
                    "period_start": "2026-01-01T00:00:00Z",
                    "period_end": "2026-03-31T00:00:00Z",
                },
            )
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
                "package_id": response.json().get("package_id"),
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(create_package, i) for i in range(num_packages)
            ]
            results = [f.result() for f in futures]

        # All should be accepted
        accepted_count = sum(1 for r in results if r["status"] == 202)
        assert accepted_count == num_packages

        # Each package should have unique ID
        package_ids = [r["package_id"] for r in results]
        assert len(set(package_ids)) == num_packages

    def test_concurrent_evidence_downloads(self, load_client: TestClient) -> None:
        """Test concurrent evidence download requests."""
        num_downloads = 30
        evidence_ids = [str(uuid.uuid4()) for _ in range(10)]

        def download_evidence(evidence_id):
            start = time.time()
            response = load_client.get(
                f"/api/v1/soc2/portal/download/{evidence_id}"
            )
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(download_evidence, evidence_ids[i % len(evidence_ids)])
                for i in range(num_downloads)
            ]
            results = [f.result() for f in futures]

        # All should succeed
        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count == num_downloads


# ============================================================================
# Source Adapter Throughput Tests
# ============================================================================


class TestSourceAdapterThroughput:
    """Load tests for evidence source adapters."""

    def test_list_sources_throughput(self, load_client: TestClient) -> None:
        """Test sources listing under load."""
        num_requests = 100
        results = []

        def list_sources():
            start = time.time()
            response = load_client.get("/api/v1/soc2/evidence/sources")
            duration = time.time() - start
            return {
                "status": response.status_code,
                "duration": duration,
            }

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(list_sources) for _ in range(num_requests)]
            results = [f.result() for f in futures]

        success_count = sum(1 for r in results if r["status"] == 200)
        assert success_count == num_requests

        # P95 latency should be under 500ms
        durations = sorted([r["duration"] for r in results])
        p95_index = int(0.95 * len(durations))
        p95_latency = durations[p95_index]
        assert p95_latency < 0.5, f"P95 latency {p95_latency:.3f}s exceeds 500ms"


# ============================================================================
# Stress Tests
# ============================================================================


class TestEvidenceStress:
    """Stress tests for evidence system."""

    def test_sustained_evidence_listing(self, load_client: TestClient) -> None:
        """Test sustained load on evidence listing."""
        duration_seconds = 5
        requests_per_second = 20
        total_requests = duration_seconds * requests_per_second
        results = []

        start_time = time.time()

        def make_timed_request(request_num):
            # Stagger requests across the test duration
            target_time = start_time + (request_num / requests_per_second)
            current = time.time()
            if current < target_time:
                time.sleep(target_time - current)

            request_start = time.time()
            response = load_client.get("/api/v1/soc2/evidence")
            duration = time.time() - request_start
            return {
                "status": response.status_code,
                "duration": duration,
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
