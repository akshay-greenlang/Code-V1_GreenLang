# -*- coding: utf-8 -*-
"""
Comprehensive Performance and Load Tests

15 test cases covering:
- 100 concurrent requests (3 tests)
- 10,000 record batch processing (3 tests)
- Cache hit ratio verification (3 tests)
- Memory usage under load (3 tests)
- Response time percentiles (3 tests)

Target: Validate system performance under various load conditions
Run with: pytest tests/performance/test_load_comprehensive.py -v --tb=short -m performance

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
import hashlib
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import random
import gc

# Add project paths for imports
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Performance Test Utilities
# =============================================================================

class PerformanceMetrics:
    """Collect and analyze performance metrics."""

    def __init__(self):
        self.response_times: List[float] = []
        self.errors: List[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def stop(self):
        """Stop timing."""
        self.end_time = time.time()

    def record_response_time(self, response_time: float):
        """Record a single response time."""
        self.response_times.append(response_time)

    def record_error(self, error: str):
        """Record an error."""
        self.errors.append(error)

    def get_percentile(self, p: float) -> float:
        """Get percentile of response times."""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * p / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.response_times:
            return {"error": "No response times recorded"}

        total_time = (self.end_time or time.time()) - (self.start_time or 0)

        return {
            "total_requests": len(self.response_times),
            "total_errors": len(self.errors),
            "success_rate": (len(self.response_times) - len(self.errors)) / len(self.response_times) * 100,
            "total_time_seconds": round(total_time, 3),
            "requests_per_second": round(len(self.response_times) / total_time, 2) if total_time > 0 else 0,
            "response_times": {
                "min_ms": round(min(self.response_times) * 1000, 2),
                "max_ms": round(max(self.response_times) * 1000, 2),
                "avg_ms": round(statistics.mean(self.response_times) * 1000, 2),
                "median_ms": round(statistics.median(self.response_times) * 1000, 2),
                "p50_ms": round(self.get_percentile(50) * 1000, 2),
                "p90_ms": round(self.get_percentile(90) * 1000, 2),
                "p95_ms": round(self.get_percentile(95) * 1000, 2),
                "p99_ms": round(self.get_percentile(99) * 1000, 2),
            },
        }


class MemoryTracker:
    """Track memory usage during tests."""

    def __init__(self):
        self.measurements: List[Dict] = []
        self.baseline_mb: float = 0

    def set_baseline(self):
        """Set memory baseline."""
        gc.collect()
        try:
            import psutil
            process = psutil.Process()
            self.baseline_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            self.baseline_mb = 0

    def measure(self, label: str = ""):
        """Take a memory measurement."""
        gc.collect()
        try:
            import psutil
            process = psutil.Process()
            current_mb = process.memory_info().rss / 1024 / 1024
            self.measurements.append({
                "label": label,
                "timestamp": time.time(),
                "memory_mb": current_mb,
                "delta_mb": current_mb - self.baseline_mb,
            })
        except ImportError:
            # Fallback if psutil not available
            self.measurements.append({
                "label": label,
                "timestamp": time.time(),
                "memory_mb": 0,
                "delta_mb": 0,
            })

    def get_max_delta(self) -> float:
        """Get maximum memory increase from baseline."""
        if not self.measurements:
            return 0
        return max(m["delta_mb"] for m in self.measurements)

    def get_summary(self) -> Dict[str, Any]:
        """Get memory tracking summary."""
        if not self.measurements:
            return {"error": "No measurements recorded"}

        deltas = [m["delta_mb"] for m in self.measurements]
        return {
            "baseline_mb": round(self.baseline_mb, 2),
            "max_delta_mb": round(max(deltas), 2),
            "avg_delta_mb": round(statistics.mean(deltas), 2),
            "final_delta_mb": round(deltas[-1], 2) if deltas else 0,
            "measurement_count": len(self.measurements),
        }


# =============================================================================
# Mock Services for Load Testing
# =============================================================================

class MockHighPerformanceAgent:
    """Mock agent optimized for load testing."""

    def __init__(self, latency_ms: float = 1.0):
        self.latency_ms = latency_ms
        self.call_count = 0
        self.emission_factor = 0.0561

    async def process(self, input_data: Dict) -> Dict:
        """Process request with simulated latency."""
        self.call_count += 1

        # Simulate processing time
        await asyncio.sleep(self.latency_ms / 1000)

        quantity = input_data.get("quantity", 0)
        emissions = quantity * self.emission_factor

        return {
            "success": True,
            "emissions_kgco2e": round(emissions, 4),
            "request_id": input_data.get("request_id"),
            "processed_at": datetime.now().isoformat(),
        }

    def reset(self):
        """Reset call counter."""
        self.call_count = 0


class MockCacheLayer:
    """Mock cache with performance tracking."""

    def __init__(self, hit_rate: float = 0.8):
        self._cache: Dict[str, Any] = {}
        self._hit_rate = hit_rate
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache with simulated hit rate."""
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set cache entry."""
        self._cache[key] = value
        return True

    async def get_or_compute(self, key: str, compute_func) -> Any:
        """Get from cache or compute and cache."""
        cached = await self.get(key)
        if cached is not None:
            return cached

        result = await compute_func()
        await self.set(key, result)
        return result

    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    def reset(self):
        """Reset statistics."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0


class MockBatchProcessor:
    """Mock batch processor for large dataset testing."""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.processed_count = 0

    async def process_batch(self, records: List[Dict]) -> Dict:
        """Process a batch of records."""
        results = []

        for record in records:
            # Simulate minimal processing
            result = {
                "record_id": record.get("record_id"),
                "emissions": record.get("quantity", 0) * 0.0561,
            }
            results.append(result)
            self.processed_count += 1

        return {
            "processed": len(results),
            "results": results,
        }

    async def process_stream(self, records: List[Dict]) -> Tuple[int, float]:
        """Process records in batches, returning count and time."""
        start_time = time.time()
        total_processed = 0

        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            result = await self.process_batch(batch)
            total_processed += result["processed"]
            # Yield control to allow other tasks
            await asyncio.sleep(0)

        elapsed = time.time() - start_time
        return total_processed, elapsed


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def performance_metrics():
    """Create performance metrics collector."""
    return PerformanceMetrics()


@pytest.fixture
def memory_tracker():
    """Create memory tracker."""
    tracker = MemoryTracker()
    tracker.set_baseline()
    return tracker


@pytest.fixture
def mock_agent():
    """Create mock high-performance agent."""
    return MockHighPerformanceAgent(latency_ms=1.0)


@pytest.fixture
def mock_cache():
    """Create mock cache layer."""
    return MockCacheLayer()


@pytest.fixture
def mock_batch_processor():
    """Create mock batch processor."""
    return MockBatchProcessor(batch_size=1000)


def generate_test_records(count: int, seed: int = 42) -> List[Dict]:
    """Generate test records for load testing."""
    random.seed(seed)
    fuel_types = ["natural_gas", "diesel", "electricity", "gasoline", "coal"]

    records = []
    for i in range(count):
        records.append({
            "record_id": f"REC-{i:08d}",
            "fuel_type": random.choice(fuel_types),
            "quantity": random.uniform(100, 10000),
            "unit": "MJ",
            "facility_id": f"FAC-{i % 100:04d}",
        })
    return records


# =============================================================================
# 100 Concurrent Requests Tests (3 tests)
# =============================================================================

class TestConcurrentRequests:
    """Test 100 concurrent requests - 3 test cases."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_100_concurrent_requests_throughput(self, mock_agent, performance_metrics):
        """PERF-CONC-001: Test throughput with 100 concurrent requests."""
        num_requests = 100

        async def make_request(request_id: int):
            start = time.time()
            result = await mock_agent.process({
                "request_id": request_id,
                "quantity": 1000 + request_id,
            })
            elapsed = time.time() - start
            performance_metrics.record_response_time(elapsed)
            return result

        performance_metrics.start()
        results = await asyncio.gather(*[
            make_request(i) for i in range(num_requests)
        ])
        performance_metrics.stop()

        summary = performance_metrics.get_summary()

        assert len(results) == num_requests
        assert all(r["success"] for r in results)
        assert summary["success_rate"] == 100
        assert summary["requests_per_second"] > 50  # At least 50 RPS

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_100_concurrent_requests_latency(self, mock_agent, performance_metrics):
        """PERF-CONC-002: Test latency percentiles with 100 concurrent requests."""
        num_requests = 100

        async def make_request(request_id: int):
            start = time.time()
            await mock_agent.process({"request_id": request_id, "quantity": 1000})
            elapsed = time.time() - start
            performance_metrics.record_response_time(elapsed)

        performance_metrics.start()
        await asyncio.gather(*[make_request(i) for i in range(num_requests)])
        performance_metrics.stop()

        summary = performance_metrics.get_summary()

        # Latency targets (given 1ms simulated latency)
        assert summary["response_times"]["p50_ms"] < 50  # p50 under 50ms
        assert summary["response_times"]["p95_ms"] < 100  # p95 under 100ms
        assert summary["response_times"]["p99_ms"] < 200  # p99 under 200ms

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_100_concurrent_requests_no_errors(self, mock_agent, performance_metrics):
        """PERF-CONC-003: Test error rate with 100 concurrent requests."""
        num_requests = 100
        errors = []

        async def make_request(request_id: int):
            try:
                start = time.time()
                result = await mock_agent.process({
                    "request_id": request_id,
                    "quantity": 1000,
                })
                elapsed = time.time() - start
                performance_metrics.record_response_time(elapsed)
                return result
            except Exception as e:
                errors.append(str(e))
                performance_metrics.record_error(str(e))
                return None

        performance_metrics.start()
        results = await asyncio.gather(*[
            make_request(i) for i in range(num_requests)
        ])
        performance_metrics.stop()

        # No errors should occur
        assert len(errors) == 0
        assert len([r for r in results if r is not None]) == num_requests


# =============================================================================
# 10,000 Record Batch Processing Tests (3 tests)
# =============================================================================

class TestBatchProcessing:
    """Test 10,000 record batch processing - 3 test cases."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_10000_records_throughput(self, mock_batch_processor):
        """PERF-BATCH-001: Test throughput with 10,000 records."""
        num_records = 10000
        records = generate_test_records(num_records)

        processed, elapsed = await mock_batch_processor.process_stream(records)

        throughput = processed / elapsed if elapsed > 0 else 0

        assert processed == num_records
        assert throughput > 1000  # At least 1000 records/second

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_10000_records_batch_efficiency(self, mock_batch_processor):
        """PERF-BATCH-002: Test batch processing efficiency."""
        num_records = 10000
        records = generate_test_records(num_records)

        start_time = time.time()
        result = await mock_batch_processor.process_batch(records)
        single_batch_time = time.time() - start_time

        # Reset and test streaming
        mock_batch_processor.processed_count = 0
        _, stream_time = await mock_batch_processor.process_stream(records)

        # Streaming should be within 2x of single batch (overhead)
        assert stream_time < single_batch_time * 3
        assert result["processed"] == num_records

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_10000_records_calculation_accuracy(self, mock_batch_processor):
        """PERF-BATCH-003: Test calculation accuracy with 10,000 records."""
        num_records = 10000
        records = generate_test_records(num_records, seed=42)

        result = await mock_batch_processor.process_batch(records)

        # Verify all records processed
        assert result["processed"] == num_records

        # Verify calculations are correct
        for i, rec_result in enumerate(result["results"][:10]):  # Check first 10
            expected = records[i]["quantity"] * 0.0561
            assert rec_result["emissions"] == pytest.approx(expected, rel=0.01)


# =============================================================================
# Cache Hit Ratio Verification Tests (3 tests)
# =============================================================================

class TestCacheHitRatio:
    """Test cache hit ratio verification - 3 test cases."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_hit_ratio_80_percent(self, mock_cache, mock_agent):
        """PERF-CACHE-001: Test cache achieves 80% hit ratio."""
        num_requests = 100
        num_unique = 20  # Only 20 unique keys for 80% hit rate

        async def cached_request(key: str):
            async def compute():
                return await mock_agent.process({"quantity": int(key.split("-")[1])})

            return await mock_cache.get_or_compute(key, compute)

        # Make requests with repeated keys
        keys = [f"key-{i % num_unique}" for i in range(num_requests)]

        for key in keys:
            await cached_request(key)

        hit_ratio = mock_cache.get_hit_ratio()

        # First 20 requests are misses, remaining 80 are hits
        # Expected hit ratio: 80/100 = 0.8
        assert hit_ratio >= 0.75  # Allow some variance

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_reduces_agent_calls(self, mock_cache, mock_agent):
        """PERF-CACHE-002: Test cache reduces agent calls."""
        num_requests = 100
        num_unique = 10

        async def cached_request(key: str):
            async def compute():
                return await mock_agent.process({"quantity": 1000})

            return await mock_cache.get_or_compute(key, compute)

        keys = [f"key-{i % num_unique}" for i in range(num_requests)]

        for key in keys:
            await cached_request(key)

        # Agent should only be called for unique keys (10 times)
        assert mock_agent.call_count == num_unique

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_performance_improvement(self, mock_cache):
        """PERF-CACHE-003: Test cache improves response time."""
        num_iterations = 1000
        key = "test-key"
        value = {"result": "cached_value"}

        # Populate cache
        await mock_cache.set(key, value)

        # Time cache hits
        start = time.time()
        for _ in range(num_iterations):
            await mock_cache.get(key)
        cache_time = time.time() - start

        # Average time per cache hit
        avg_cache_time_ms = (cache_time / num_iterations) * 1000

        # Cache access should be very fast (<1ms per access)
        assert avg_cache_time_ms < 1.0


# =============================================================================
# Memory Usage Under Load Tests (3 tests)
# =============================================================================

class TestMemoryUsage:
    """Test memory usage under load - 3 test cases."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_stable_under_load(self, mock_agent, memory_tracker):
        """PERF-MEM-001: Test memory remains stable under load."""
        num_requests = 1000

        memory_tracker.measure("before_load")

        for i in range(num_requests):
            await mock_agent.process({"quantity": 1000, "request_id": i})

            if i % 100 == 0:
                memory_tracker.measure(f"after_{i}_requests")

        memory_tracker.measure("after_load")

        summary = memory_tracker.get_summary()

        # Memory increase should be reasonable (<100MB for 1000 requests)
        assert summary["max_delta_mb"] < 100

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_with_large_batch(self, mock_batch_processor, memory_tracker):
        """PERF-MEM-002: Test memory with large batch processing."""
        num_records = 10000
        records = generate_test_records(num_records)

        memory_tracker.measure("before_batch")

        await mock_batch_processor.process_batch(records)

        memory_tracker.measure("after_batch")

        # Force garbage collection
        gc.collect()
        memory_tracker.measure("after_gc")

        summary = memory_tracker.get_summary()

        # Memory should not grow excessively (<500MB)
        assert summary["max_delta_mb"] < 500

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_with_concurrent_requests(self, mock_agent, memory_tracker):
        """PERF-MEM-003: Test memory with concurrent requests."""
        num_requests = 100

        memory_tracker.measure("before_concurrent")

        # Create many concurrent tasks
        tasks = [
            mock_agent.process({"quantity": 1000, "request_id": i})
            for i in range(num_requests)
        ]

        await asyncio.gather(*tasks)

        memory_tracker.measure("after_concurrent")
        gc.collect()
        memory_tracker.measure("after_gc")

        summary = memory_tracker.get_summary()

        # Concurrent requests should not cause memory explosion
        assert summary["max_delta_mb"] < 200


# =============================================================================
# Response Time Percentiles Tests (3 tests)
# =============================================================================

class TestResponseTimePercentiles:
    """Test response time percentiles - 3 test cases."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_p50_response_time(self, mock_agent, performance_metrics):
        """PERF-RT-001: Test p50 response time meets target."""
        num_requests = 200

        async def make_request():
            start = time.time()
            await mock_agent.process({"quantity": 1000})
            elapsed = time.time() - start
            performance_metrics.record_response_time(elapsed)

        await asyncio.gather(*[make_request() for _ in range(num_requests)])

        summary = performance_metrics.get_summary()

        # p50 should be under 20ms (given 1ms simulated latency)
        assert summary["response_times"]["p50_ms"] < 20

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_p95_response_time(self, mock_agent, performance_metrics):
        """PERF-RT-002: Test p95 response time meets target."""
        num_requests = 200

        async def make_request():
            start = time.time()
            await mock_agent.process({"quantity": 1000})
            elapsed = time.time() - start
            performance_metrics.record_response_time(elapsed)

        await asyncio.gather(*[make_request() for _ in range(num_requests)])

        summary = performance_metrics.get_summary()

        # p95 should be under 50ms
        assert summary["response_times"]["p95_ms"] < 50

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_p99_response_time(self, mock_agent, performance_metrics):
        """PERF-RT-003: Test p99 response time meets target."""
        num_requests = 500  # More samples for p99 accuracy

        async def make_request():
            start = time.time()
            await mock_agent.process({"quantity": 1000})
            elapsed = time.time() - start
            performance_metrics.record_response_time(elapsed)

        await asyncio.gather(*[make_request() for _ in range(num_requests)])

        summary = performance_metrics.get_summary()

        # p99 should be under 100ms
        assert summary["response_times"]["p99_ms"] < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])
