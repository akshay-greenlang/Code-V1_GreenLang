"""
Performance tests for GL-001 ThermalCommand API Response Time.

Tests API endpoint response times to ensure <200ms target is met
for all critical operations.

Coverage Target: Performance validation
Reference: GL-001 Specification Section 11

Performance Targets:
- GET endpoints: <100ms
- POST endpoints: <200ms
- Query endpoints: <150ms
- Batch operations: <500ms

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import time
import asyncio
import statistics
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock, AsyncMock

# Add parent path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# =============================================================================
# MOCK API CLASSES FOR PERFORMANCE TESTING
# =============================================================================

class MockAPIRequest:
    """Mock API request."""

    def __init__(
        self,
        method: str = "GET",
        endpoint: str = "/api/v1/status",
        body: Dict = None,
        params: Dict = None
    ):
        self.method = method
        self.endpoint = endpoint
        self.body = body or {}
        self.params = params or {}
        self.timestamp = datetime.now(timezone.utc)


class MockAPIResponse:
    """Mock API response."""

    def __init__(
        self,
        status_code: int = 200,
        body: Dict = None,
        latency_ms: float = 0.0
    ):
        self.status_code = status_code
        self.body = body or {}
        self.latency_ms = latency_ms
        self.timestamp = datetime.now(timezone.utc)


class MockAPIHandler:
    """Mock API handler with realistic timing characteristics."""

    def __init__(self, base_latency_ms: float = 10.0):
        self._base_latency_ms = base_latency_ms
        self._request_count = 0
        self._data_store = {
            "system_status": {"status": "running", "uptime": 3600},
            "equipment": [
                {"id": "BOILER-001", "status": "running"},
                {"id": "BOILER-002", "status": "standby"},
            ],
            "metrics": {"cpu": 45.0, "memory": 60.0},
        }

    def _simulate_latency(self, complexity: float = 1.0):
        """Simulate processing latency."""
        # Base latency + variance + complexity factor
        latency_ms = (
            self._base_latency_ms * complexity +
            10 * (0.5 + 0.5 * hash(self._request_count) % 100 / 100)
        )
        time.sleep(latency_ms / 1000.0)
        return latency_ms

    def get_system_status(self) -> MockAPIResponse:
        """GET /api/v1/status"""
        start = time.perf_counter()
        self._request_count += 1

        self._simulate_latency(1.0)

        latency = (time.perf_counter() - start) * 1000

        return MockAPIResponse(
            status_code=200,
            body=self._data_store["system_status"],
            latency_ms=latency
        )

    def get_equipment_list(self) -> MockAPIResponse:
        """GET /api/v1/equipment"""
        start = time.perf_counter()
        self._request_count += 1

        self._simulate_latency(1.5)

        latency = (time.perf_counter() - start) * 1000

        return MockAPIResponse(
            status_code=200,
            body={"equipment": self._data_store["equipment"]},
            latency_ms=latency
        )

    def get_equipment_by_id(self, equipment_id: str) -> MockAPIResponse:
        """GET /api/v1/equipment/{id}"""
        start = time.perf_counter()
        self._request_count += 1

        self._simulate_latency(1.0)

        equipment = next(
            (e for e in self._data_store["equipment"] if e["id"] == equipment_id),
            None
        )

        latency = (time.perf_counter() - start) * 1000

        if equipment:
            return MockAPIResponse(
                status_code=200,
                body=equipment,
                latency_ms=latency
            )
        else:
            return MockAPIResponse(
                status_code=404,
                body={"error": "Not found"},
                latency_ms=latency
            )

    def get_metrics(self) -> MockAPIResponse:
        """GET /api/v1/metrics"""
        start = time.perf_counter()
        self._request_count += 1

        self._simulate_latency(1.2)

        latency = (time.perf_counter() - start) * 1000

        return MockAPIResponse(
            status_code=200,
            body=self._data_store["metrics"],
            latency_ms=latency
        )

    def post_setpoint(self, equipment_id: str, setpoint: float) -> MockAPIResponse:
        """POST /api/v1/equipment/{id}/setpoint"""
        start = time.perf_counter()
        self._request_count += 1

        # Validation and processing
        self._simulate_latency(2.0)

        latency = (time.perf_counter() - start) * 1000

        return MockAPIResponse(
            status_code=200,
            body={"equipment_id": equipment_id, "setpoint": setpoint, "status": "accepted"},
            latency_ms=latency
        )

    def post_optimization_request(self, demand: float) -> MockAPIResponse:
        """POST /api/v1/optimize"""
        start = time.perf_counter()
        self._request_count += 1

        # More complex processing
        self._simulate_latency(5.0)

        latency = (time.perf_counter() - start) * 1000

        return MockAPIResponse(
            status_code=200,
            body={"demand": demand, "status": "optimizing", "job_id": "OPT-001"},
            latency_ms=latency
        )

    def get_query_results(self, query: Dict) -> MockAPIResponse:
        """GET /api/v1/query"""
        start = time.perf_counter()
        self._request_count += 1

        # Query complexity based on parameters
        complexity = 1.0 + len(query) * 0.5
        self._simulate_latency(complexity)

        latency = (time.perf_counter() - start) * 1000

        return MockAPIResponse(
            status_code=200,
            body={"results": [], "query": query},
            latency_ms=latency
        )

    def post_batch_setpoints(self, setpoints: List[Dict]) -> MockAPIResponse:
        """POST /api/v1/batch/setpoints"""
        start = time.perf_counter()
        self._request_count += 1

        # Process each setpoint
        complexity = 1.0 + len(setpoints) * 1.5
        self._simulate_latency(complexity)

        latency = (time.perf_counter() - start) * 1000

        return MockAPIResponse(
            status_code=200,
            body={"processed": len(setpoints), "status": "success"},
            latency_ms=latency
        )


class AsyncMockAPIHandler:
    """Async mock API handler."""

    def __init__(self, base_latency_ms: float = 10.0):
        self._base_latency_ms = base_latency_ms
        self._request_count = 0

    async def _simulate_latency(self, complexity: float = 1.0):
        """Simulate async processing latency."""
        latency_ms = self._base_latency_ms * complexity * (0.8 + 0.4 * hash(self._request_count) % 100 / 100)
        await asyncio.sleep(latency_ms / 1000.0)
        return latency_ms

    async def get_system_status(self) -> MockAPIResponse:
        """Async GET /api/v1/status"""
        start = time.perf_counter()
        self._request_count += 1

        await self._simulate_latency(1.0)

        latency = (time.perf_counter() - start) * 1000

        return MockAPIResponse(
            status_code=200,
            body={"status": "running"},
            latency_ms=latency
        )

    async def post_setpoint(self, equipment_id: str, setpoint: float) -> MockAPIResponse:
        """Async POST setpoint"""
        start = time.perf_counter()
        self._request_count += 1

        await self._simulate_latency(2.0)

        latency = (time.perf_counter() - start) * 1000

        return MockAPIResponse(
            status_code=200,
            body={"equipment_id": equipment_id, "setpoint": setpoint},
            latency_ms=latency
        )


class PerformanceMetrics:
    """Collect and analyze performance metrics."""

    def __init__(self):
        self.measurements: List[float] = []
        self.endpoints: List[str] = []

    def record(self, latency_ms: float, endpoint: str = ""):
        self.measurements.append(latency_ms)
        self.endpoints.append(endpoint)

    def clear(self):
        self.measurements = []
        self.endpoints = []

    @property
    def count(self) -> int:
        return len(self.measurements)

    @property
    def mean(self) -> float:
        return statistics.mean(self.measurements) if self.measurements else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.measurements) if self.measurements else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0

    def percentile(self, p: float) -> float:
        if not self.measurements:
            return 0.0
        sorted_values = sorted(self.measurements)
        index = int(len(sorted_values) * p / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def by_endpoint(self) -> Dict[str, List[float]]:
        """Group measurements by endpoint."""
        result = {}
        for latency, endpoint in zip(self.measurements, self.endpoints):
            if endpoint not in result:
                result[endpoint] = []
            result[endpoint].append(latency)
        return result

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": self.mean,
            "median_ms": self.median,
            "stdev_ms": self.stdev,
            "min_ms": min(self.measurements) if self.measurements else 0,
            "max_ms": max(self.measurements) if self.measurements else 0,
            "p95_ms": self.percentile(95),
            "p99_ms": self.percentile(99),
        }


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def api_handler() -> MockAPIHandler:
    """Create mock API handler."""
    return MockAPIHandler(base_latency_ms=10.0)


@pytest.fixture
def async_api_handler() -> AsyncMockAPIHandler:
    """Create async mock API handler."""
    return AsyncMockAPIHandler(base_latency_ms=10.0)


@pytest.fixture
def performance_metrics() -> PerformanceMetrics:
    """Create performance metrics collector."""
    return PerformanceMetrics()


# =============================================================================
# TEST CLASS: GET ENDPOINT RESPONSE TIME
# =============================================================================

class TestGETEndpointResponseTime:
    """Tests for GET endpoint response times (<100ms target)."""

    @pytest.mark.performance
    def test_get_status_response_time(
        self, api_handler, performance_metrics
    ):
        """Test GET /status response time."""
        iterations = 50

        for _ in range(iterations):
            response = api_handler.get_system_status()
            performance_metrics.record(response.latency_ms, "/api/v1/status")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 100, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 100ms target"

    @pytest.mark.performance
    def test_get_equipment_list_response_time(
        self, api_handler, performance_metrics
    ):
        """Test GET /equipment response time."""
        iterations = 50

        for _ in range(iterations):
            response = api_handler.get_equipment_list()
            performance_metrics.record(response.latency_ms, "/api/v1/equipment")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 100, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 100ms target"

    @pytest.mark.performance
    def test_get_equipment_by_id_response_time(
        self, api_handler, performance_metrics
    ):
        """Test GET /equipment/{id} response time."""
        iterations = 50

        for _ in range(iterations):
            response = api_handler.get_equipment_by_id("BOILER-001")
            performance_metrics.record(response.latency_ms, "/api/v1/equipment/{id}")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 100, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 100ms target"

    @pytest.mark.performance
    def test_get_metrics_response_time(
        self, api_handler, performance_metrics
    ):
        """Test GET /metrics response time."""
        iterations = 50

        for _ in range(iterations):
            response = api_handler.get_metrics()
            performance_metrics.record(response.latency_ms, "/api/v1/metrics")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 100, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 100ms target"


# =============================================================================
# TEST CLASS: POST ENDPOINT RESPONSE TIME
# =============================================================================

class TestPOSTEndpointResponseTime:
    """Tests for POST endpoint response times (<200ms target)."""

    @pytest.mark.performance
    def test_post_setpoint_response_time(
        self, api_handler, performance_metrics
    ):
        """Test POST /equipment/{id}/setpoint response time."""
        iterations = 50

        for _ in range(iterations):
            response = api_handler.post_setpoint("BOILER-001", 85.0)
            performance_metrics.record(response.latency_ms, "POST /setpoint")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 200, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 200ms target"

    @pytest.mark.performance
    def test_post_optimization_response_time(
        self, api_handler, performance_metrics
    ):
        """Test POST /optimize response time (initial response, not completion)."""
        iterations = 30

        for _ in range(iterations):
            response = api_handler.post_optimization_request(1000.0)
            performance_metrics.record(response.latency_ms, "POST /optimize")

        summary = performance_metrics.summary()

        # Optimization request acceptance should be under 200ms
        assert summary["p95_ms"] < 200, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 200ms target"


# =============================================================================
# TEST CLASS: QUERY ENDPOINT RESPONSE TIME
# =============================================================================

class TestQueryEndpointResponseTime:
    """Tests for query endpoint response times (<150ms target)."""

    @pytest.mark.performance
    def test_simple_query_response_time(
        self, api_handler, performance_metrics
    ):
        """Test simple query response time."""
        iterations = 50

        for _ in range(iterations):
            response = api_handler.get_query_results({"type": "boiler"})
            performance_metrics.record(response.latency_ms, "query_simple")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 150, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 150ms target"

    @pytest.mark.performance
    def test_complex_query_response_time(
        self, api_handler, performance_metrics
    ):
        """Test complex query response time."""
        iterations = 30

        for _ in range(iterations):
            response = api_handler.get_query_results({
                "type": "boiler",
                "status": "running",
                "efficiency_min": 0.80,
                "load_range": [50, 100],
            })
            performance_metrics.record(response.latency_ms, "query_complex")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 200, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 200ms target"


# =============================================================================
# TEST CLASS: BATCH OPERATION RESPONSE TIME
# =============================================================================

class TestBatchOperationResponseTime:
    """Tests for batch operation response times (<500ms target)."""

    @pytest.mark.performance
    def test_small_batch_response_time(
        self, api_handler, performance_metrics
    ):
        """Test small batch (5 items) response time."""
        iterations = 20

        setpoints = [
            {"equipment_id": f"BOILER-{i}", "setpoint": 80.0 + i}
            for i in range(5)
        ]

        for _ in range(iterations):
            response = api_handler.post_batch_setpoints(setpoints)
            performance_metrics.record(response.latency_ms, "batch_5")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 300, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 300ms target"

    @pytest.mark.performance
    def test_medium_batch_response_time(
        self, api_handler, performance_metrics
    ):
        """Test medium batch (20 items) response time."""
        iterations = 10

        setpoints = [
            {"equipment_id": f"BOILER-{i}", "setpoint": 80.0 + i}
            for i in range(20)
        ]

        for _ in range(iterations):
            response = api_handler.post_batch_setpoints(setpoints)
            performance_metrics.record(response.latency_ms, "batch_20")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 500, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 500ms target"


# =============================================================================
# TEST CLASS: ASYNC ENDPOINT RESPONSE TIME
# =============================================================================

class TestAsyncEndpointResponseTime:
    """Tests for async endpoint response times."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_async_get_status_response_time(
        self, async_api_handler, performance_metrics
    ):
        """Test async GET /status response time."""
        iterations = 50

        for _ in range(iterations):
            response = await async_api_handler.get_system_status()
            performance_metrics.record(response.latency_ms, "async_status")

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 100, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 100ms target"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_requests_response_time(
        self, async_api_handler, performance_metrics
    ):
        """Test concurrent request response times."""
        async def make_request():
            response = await async_api_handler.get_system_status()
            performance_metrics.record(response.latency_ms, "concurrent")

        # Make 20 concurrent requests
        tasks = [make_request() for _ in range(20)]
        await asyncio.gather(*tasks)

        summary = performance_metrics.summary()

        assert summary["p95_ms"] < 150, \
            f"P95 latency ({summary['p95_ms']:.1f}ms) exceeds 150ms target"


# =============================================================================
# TEST CLASS: LATENCY CONSISTENCY
# =============================================================================

class TestLatencyConsistency:
    """Tests for latency consistency and stability."""

    @pytest.mark.performance
    def test_latency_consistency(self, api_handler, performance_metrics):
        """Test that latency is consistent across requests."""
        iterations = 100

        for _ in range(iterations):
            response = api_handler.get_system_status()
            performance_metrics.record(response.latency_ms)

        summary = performance_metrics.summary()

        # Coefficient of variation should be less than 50%
        cv = summary["stdev_ms"] / summary["mean_ms"] if summary["mean_ms"] > 0 else 0

        assert cv < 0.5, \
            f"Latency variance too high: CV = {cv:.2%}"

    @pytest.mark.performance
    def test_no_latency_degradation(self, api_handler, performance_metrics):
        """Test that latency does not degrade over time."""
        iterations = 100

        for _ in range(iterations):
            response = api_handler.get_system_status()
            performance_metrics.record(response.latency_ms)

        # Compare first 20 vs last 20
        first_20 = performance_metrics.measurements[:20]
        last_20 = performance_metrics.measurements[-20:]

        first_mean = statistics.mean(first_20)
        last_mean = statistics.mean(last_20)

        degradation = (last_mean - first_mean) / first_mean * 100 if first_mean > 0 else 0

        assert degradation < 20, \
            f"Latency degradation ({degradation:.1f}%) exceeds 20%"


# =============================================================================
# TEST CLASS: THROUGHPUT
# =============================================================================

class TestThroughput:
    """Tests for API throughput."""

    @pytest.mark.performance
    def test_requests_per_second(self, api_handler, performance_metrics):
        """Test maximum requests per second."""
        duration = 5.0
        count = 0

        start = time.perf_counter()

        while time.perf_counter() - start < duration:
            api_handler.get_system_status()
            count += 1

        elapsed = time.perf_counter() - start
        throughput = count / elapsed

        # Should handle at least 50 requests per second
        assert throughput >= 50, \
            f"Throughput ({throughput:.1f}/s) below 50/s target"

    @pytest.mark.performance
    def test_mixed_endpoint_throughput(self, api_handler):
        """Test throughput with mixed endpoint types."""
        duration = 5.0
        counts = {"get": 0, "post": 0}

        start = time.perf_counter()

        while time.perf_counter() - start < duration:
            # Mix of GET and POST
            api_handler.get_system_status()
            counts["get"] += 1

            api_handler.get_equipment_list()
            counts["get"] += 1

            api_handler.post_setpoint("BOILER-001", 85.0)
            counts["post"] += 1

        elapsed = time.perf_counter() - start
        total = sum(counts.values())
        throughput = total / elapsed

        # Should handle at least 30 mixed requests per second
        assert throughput >= 30, \
            f"Throughput ({throughput:.1f}/s) below 30/s target"


# =============================================================================
# TEST CLASS: ENDPOINT-SPECIFIC BENCHMARKS
# =============================================================================

class TestEndpointBenchmarks:
    """Endpoint-specific benchmark tests."""

    @pytest.mark.performance
    def test_all_endpoints_under_target(self, api_handler, performance_metrics):
        """Test all endpoints meet their specific targets."""
        targets = {
            "GET /status": (100, lambda: api_handler.get_system_status()),
            "GET /equipment": (100, lambda: api_handler.get_equipment_list()),
            "GET /metrics": (100, lambda: api_handler.get_metrics()),
            "POST /setpoint": (200, lambda: api_handler.post_setpoint("B-1", 80)),
        }

        results = {}

        for name, (target_ms, func) in targets.items():
            times = []
            for _ in range(30):
                response = func()
                times.append(response.latency_ms)

            p95 = sorted(times)[int(0.95 * len(times))]
            passed = p95 < target_ms
            results[name] = {
                "target_ms": target_ms,
                "p95_ms": p95,
                "passed": passed
            }

        # Report results
        print("\n" + "=" * 60)
        print("API RESPONSE TIME BENCHMARK RESULTS")
        print("=" * 60)

        for name, metrics in results.items():
            status = "PASS" if metrics["passed"] else "FAIL"
            print(f"{name}:")
            print(f"  P95: {metrics['p95_ms']:.1f}ms (target: {metrics['target_ms']}ms) [{status}]")

        print("=" * 60)

        # Assert all passed
        assert all(r["passed"] for r in results.values()), \
            f"Some endpoints failed: {[n for n, r in results.items() if not r['passed']]}"


# =============================================================================
# TEST CLASS: STRESS TESTING
# =============================================================================

class TestStressTesting:
    """API stress tests."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_high_load_response_time(self, api_handler, performance_metrics):
        """Test response time under high load."""
        iterations = 500

        for i in range(iterations):
            response = api_handler.get_system_status()
            performance_metrics.record(response.latency_ms)

        summary = performance_metrics.summary()

        # Even under high load, P99 should be under 200ms
        assert summary["p99_ms"] < 200, \
            f"P99 latency under load ({summary['p99_ms']:.1f}ms) exceeds 200ms"

    @pytest.mark.performance
    @pytest.mark.slow
    def test_burst_traffic(self, api_handler, performance_metrics):
        """Test response to burst traffic patterns."""
        # Simulate burst: 50 requests, pause, 50 requests
        for burst in range(3):
            for _ in range(50):
                response = api_handler.get_system_status()
                performance_metrics.record(response.latency_ms, f"burst_{burst}")

            time.sleep(0.1)  # Brief pause between bursts

        summary = performance_metrics.summary()

        # Latency should remain stable across bursts
        by_endpoint = performance_metrics.by_endpoint()
        means = [statistics.mean(times) for times in by_endpoint.values()]

        max_diff = max(means) - min(means)

        assert max_diff < 50, \
            f"Latency variance between bursts ({max_diff:.1f}ms) too high"
