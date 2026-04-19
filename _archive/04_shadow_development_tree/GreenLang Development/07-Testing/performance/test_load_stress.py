# -*- coding: utf-8 -*-
"""
Performance and Load Testing Suite

Tests system performance under various load conditions:
- Load testing (1K, 5K, 10K concurrent agents)
- Stress testing (find breaking point)
- Endurance testing (24-hour sustained load)
- Spike testing (sudden traffic surge)

Tools: Locust, pytest-benchmark, asyncio
Target: <500ms p95 latency, 10K concurrent agents
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import statistics

from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history

from greenlang_core.agents import BaseAgent
from greenlang_core.pipeline import AgentPipeline
from greenlang_core.api import AgentFactoryAPI


# Test Configuration
LOAD_TEST_USERS = [100, 500, 1000, 5000, 10000]
STRESS_TEST_INCREMENT = 100
ENDURANCE_TEST_DURATION_HOURS = 24
TARGET_P95_LATENCY_MS = 500
TARGET_P99_LATENCY_MS = 1000


# Fixtures
@pytest.fixture
def performance_agent():
    """Create agent optimized for performance testing."""
    agent = Mock(spec=BaseAgent)
    agent.name = "performance_test_agent"

    async def fast_process(*args, **kwargs):
        # Simulate realistic processing time
        await asyncio.sleep(0.01)  # 10ms
        return {"status": "success", "result": 42}

    agent.process = fast_process
    return agent


@pytest.fixture
def api_client():
    """Create API client for load testing."""
    from httpx import AsyncClient
    return AsyncClient(base_url="http://localhost:8000")


# Load Testing
class TestLoadPerformance:
    """Load testing suite - measure performance under expected load."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_concurrent", [100, 500, 1000, 5000])
    async def test_concurrent_agent_execution(self, performance_agent, num_concurrent):
        """Test system handles concurrent agent executions."""
        latencies = []

        async def execute_agent():
            start = time.time()
            result = await performance_agent.process({"test": "data"})
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
            return result

        # Execute concurrent requests
        start_time = time.time()
        tasks = [execute_agent() for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        total_duration = time.time() - start_time

        # Calculate metrics
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        throughput = num_concurrent / total_duration

        print(f"\n=== Load Test Results ({num_concurrent} concurrent) ===")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Throughput: {throughput:.2f} req/s")
        print(f"Latency P50: {p50:.2f}ms")
        print(f"Latency P95: {p95:.2f}ms")
        print(f"Latency P99: {p99:.2f}ms")

        # Assertions
        assert len(results) == num_concurrent
        assert all(r['status'] == 'success' for r in results)

        # Performance targets
        if num_concurrent <= 1000:
            assert p95 < TARGET_P95_LATENCY_MS, f"P95 latency {p95}ms exceeds target {TARGET_P95_LATENCY_MS}ms"
            assert p99 < TARGET_P99_LATENCY_MS, f"P99 latency {p99}ms exceeds target {TARGET_P99_LATENCY_MS}ms"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_sustained_load_throughput(self, performance_agent):
        """Test sustained throughput over extended period (5 minutes)."""
        duration_seconds = 300  # 5 minutes
        target_rps = 100  # 100 requests per second

        total_requests = 0
        latencies = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            batch_start = time.time()

            # Execute batch of requests
            tasks = [performance_agent.process({"test": "data"}) for _ in range(target_rps)]

            batch_results = await asyncio.gather(*tasks)
            batch_latency = (time.time() - batch_start) * 1000

            total_requests += len(batch_results)
            latencies.append(batch_latency)

            # Sleep to maintain target RPS
            sleep_time = 1.0 - (time.time() - batch_start)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        total_duration = time.time() - start_time
        actual_rps = total_requests / total_duration
        avg_latency = statistics.mean(latencies)

        print(f"\n=== Sustained Load Test Results ===")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Total Requests: {total_requests}")
        print(f"Target RPS: {target_rps}")
        print(f"Actual RPS: {actual_rps:.2f}")
        print(f"Average Latency: {avg_latency:.2f}ms")

        # Should maintain target throughput
        assert actual_rps >= target_rps * 0.95  # Within 5% of target

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_pipeline_throughput(self):
        """Test pipeline processing throughput."""
        from greenlang_core.pipeline import AgentPipeline, PipelineConfig

        config = PipelineConfig(
            name="performance_test_pipeline",
            max_parallel_agents=10
        )

        pipeline = AgentPipeline(config)

        # Add multiple agents
        for i in range(5):
            agent = Mock(spec=BaseAgent)
            agent.name = f"agent_{i}"
            agent.process = AsyncMock(return_value={"status": "success"})
            pipeline.add_agent(agent)

        # Process batch
        num_records = 1000
        records = [{"id": i, "data": f"record_{i}"} for i in range(num_records)]

        start_time = time.time()
        results = []
        for record in records:
            result = await pipeline.execute(record)
            results.append(result)

        duration = time.time() - start_time
        throughput = num_records / duration

        print(f"\n=== Pipeline Throughput Test ===")
        print(f"Records Processed: {num_records}")
        print(f"Duration: {duration:.2f}s")
        print(f"Throughput: {throughput:.2f} records/s")

        # Target: 100+ records/second
        assert throughput >= 100


# Stress Testing
class TestStressPerformance:
    """Stress testing suite - find system breaking point."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_find_breaking_point(self, performance_agent):
        """Gradually increase load until system breaks."""
        max_successful_load = 0
        current_load = 100

        while current_load <= 50000:
            print(f"\nTesting with {current_load} concurrent requests...")

            try:
                # Execute with current load
                tasks = [performance_agent.process({"test": "data"}) for _ in range(current_load)]

                start_time = time.time()
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=60.0  # 60 second timeout
                )
                duration = time.time() - start_time

                # Check for failures
                failures = sum(1 for r in results if isinstance(r, Exception))
                success_rate = (len(results) - failures) / len(results)

                print(f"Duration: {duration:.2f}s")
                print(f"Success Rate: {success_rate:.1%}")

                # If success rate drops below 95%, we've found the breaking point
                if success_rate < 0.95:
                    print(f"\n=== Breaking Point Found ===")
                    print(f"System breaks at ~{current_load} concurrent requests")
                    print(f"Last successful load: {max_successful_load}")
                    break

                max_successful_load = current_load
                current_load += STRESS_TEST_INCREMENT

            except asyncio.TimeoutError:
                print(f"Timeout at {current_load} concurrent requests")
                print(f"Breaking point: ~{max_successful_load}")
                break

        # System should handle at least 5000 concurrent requests
        assert max_successful_load >= 5000

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_under_stress(self, performance_agent):
        """Test memory usage doesn't grow unbounded under stress."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute large number of requests in batches
        num_batches = 10
        batch_size = 1000

        memory_samples = []

        for batch in range(num_batches):
            tasks = [performance_agent.process({"test": "data"}) for _ in range(batch_size)]
            await asyncio.gather(*tasks)

            # Record memory after each batch
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)

            print(f"Batch {batch + 1}: Memory = {current_memory:.2f} MB")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"\n=== Memory Stress Test ===")
        print(f"Initial Memory: {initial_memory:.2f} MB")
        print(f"Final Memory: {final_memory:.2f} MB")
        print(f"Increase: {memory_increase:.2f} MB")

        # Memory should not grow unbounded (< 500MB increase)
        assert memory_increase < 500

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cpu_under_stress(self, performance_agent):
        """Test CPU usage under high load."""
        import psutil

        # Monitor CPU during high load
        cpu_samples = []

        async def monitor_cpu():
            for _ in range(30):  # Monitor for 30 seconds
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_samples.append(cpu_percent)

        async def generate_load():
            tasks = [performance_agent.process({"test": "data"}) for _ in range(5000)]
            await asyncio.gather(*tasks)

        # Run load and monitoring concurrently
        await asyncio.gather(monitor_cpu(), generate_load())

        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)

        print(f"\n=== CPU Stress Test ===")
        print(f"Average CPU: {avg_cpu:.1f}%")
        print(f"Max CPU: {max_cpu:.1f}%")

        # CPU should be utilized but not maxed out constantly
        assert avg_cpu < 90  # Average should be below 90%


# Spike Testing
class TestSpikePerformance:
    """Spike testing suite - sudden traffic surge."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_sudden_spike(self, performance_agent):
        """Test system handles sudden spike in traffic."""
        # Start with baseline load
        baseline_load = 100

        print("Establishing baseline load...")
        baseline_tasks = [performance_agent.process({"test": "data"}) for _ in range(baseline_load)]
        await asyncio.gather(*baseline_tasks)

        # Sudden spike to 10x load
        spike_load = 1000

        print(f"Generating spike to {spike_load} requests...")
        start_time = time.time()
        spike_tasks = [performance_agent.process({"test": "data"}) for _ in range(spike_load)]
        results = await asyncio.gather(*spike_tasks, return_exceptions=True)
        spike_duration = time.time() - start_time

        failures = sum(1 for r in results if isinstance(r, Exception))
        success_rate = (len(results) - failures) / len(results)

        print(f"\n=== Spike Test Results ===")
        print(f"Spike Load: {spike_load}")
        print(f"Duration: {spike_duration:.2f}s")
        print(f"Success Rate: {success_rate:.1%}")

        # Should handle spike with >95% success rate
        assert success_rate >= 0.95

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_spike_recovery(self, performance_agent):
        """Test system recovers after traffic spike."""
        # Generate spike
        spike_tasks = [performance_agent.process({"test": "data"}) for _ in range(2000)]
        await asyncio.gather(*spike_tasks)

        # Wait for recovery
        await asyncio.sleep(5)

        # Test normal load after spike
        normal_load = 100
        start_time = time.time()
        normal_tasks = [performance_agent.process({"test": "data"}) for _ in range(normal_load)]
        results = await asyncio.gather(*normal_tasks)
        recovery_duration = time.time() - start_time

        recovery_latency = (recovery_duration / normal_load) * 1000  # ms per request

        print(f"\n=== Spike Recovery Test ===")
        print(f"Recovery Latency: {recovery_latency:.2f}ms per request")

        # Should return to normal latency (<100ms per request)
        assert recovery_latency < 100


# Endurance Testing
class TestEndurancePerformance:
    """Endurance testing suite - sustained load over extended period."""

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_24_hour_endurance(self, performance_agent):
        """Test system stability over 24 hours (use with caution)."""
        # NOTE: This is a very long test - only run in dedicated test environment

        duration_hours = 24
        duration_seconds = duration_hours * 3600

        target_rps = 50  # Moderate sustained load
        total_requests = 0
        failures = 0
        start_time = time.time()

        print(f"Starting {duration_hours}-hour endurance test...")

        while time.time() - start_time < duration_seconds:
            batch_start = time.time()

            # Execute batch
            tasks = [performance_agent.process({"test": "data"}) for _ in range(target_rps)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            total_requests += len(results)
            failures += sum(1 for r in results if isinstance(r, Exception))

            # Sleep to maintain target RPS
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)

            # Print progress every hour
            elapsed_hours = (time.time() - start_time) / 3600
            if int(elapsed_hours) > int((batch_start - start_time) / 3600):
                print(f"Hour {int(elapsed_hours)}: {total_requests} requests, {failures} failures")

        total_duration = time.time() - start_time
        success_rate = (total_requests - failures) / total_requests
        actual_rps = total_requests / total_duration

        print(f"\n=== Endurance Test Results ===")
        print(f"Duration: {total_duration / 3600:.1f} hours")
        print(f"Total Requests: {total_requests}")
        print(f"Failures: {failures}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average RPS: {actual_rps:.2f}")

        # Should maintain >99% success rate over 24 hours
        assert success_rate >= 0.99

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, performance_agent):
        """Test for memory leaks over extended execution."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Run for 1 hour (shorter endurance test)
        duration_seconds = 3600
        sample_interval = 300  # Sample every 5 minutes

        memory_samples = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            # Execute batch of requests
            tasks = [performance_agent.process({"test": "data"}) for _ in range(100)]
            await asyncio.gather(*tasks)

            # Sample memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)

            await asyncio.sleep(sample_interval)

        # Analyze memory trend
        if len(memory_samples) > 1:
            # Calculate linear regression to detect trend
            x = list(range(len(memory_samples)))
            y = memory_samples

            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi ** 2 for xi in x)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

            print(f"\n=== Memory Leak Detection ===")
            print(f"Memory samples: {memory_samples}")
            print(f"Memory growth rate: {slope:.2f} MB per sample")

            # Memory should not grow significantly (< 1MB per sample)
            assert slope < 1.0


# Locust Load Test Scenarios
class AgentFactoryUser(HttpUser):
    """Locust user for load testing Agent Factory API."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    @task(3)
    def execute_agent(self):
        """Execute agent (most common operation)."""
        self.client.post("/api/v1/agents/execute", json={
            "agent_name": "emission_calculator",
            "input_data": {
                "fuel_type": "diesel",
                "quantity": 1000,
                "region": "US"
            }
        })

    @task(2)
    def list_agents(self):
        """List available agents."""
        self.client.get("/api/v1/agents")

    @task(1)
    def get_agent_status(self):
        """Get agent execution status."""
        self.client.get("/api/v1/agents/emission_calculator/status")

    @task(1)
    def get_metrics(self):
        """Get system metrics."""
        self.client.get("/api/v1/metrics")


class DatabaseUser(HttpUser):
    """Locust user for database-heavy operations."""

    wait_time = between(0.5, 2)

    @task(5)
    def query_emissions(self):
        """Query emissions data."""
        self.client.get("/api/v1/emissions?limit=100")

    @task(3)
    def create_record(self):
        """Create new emission record."""
        self.client.post("/api/v1/emissions", json={
            "shipment_id": "TEST-001",
            "emissions_kg": 2680.0
        })

    @task(2)
    def update_record(self):
        """Update emission record."""
        self.client.put("/api/v1/emissions/TEST-001", json={
            "emissions_kg": 2700.0
        })


def run_locust_test(user_class, num_users, spawn_rate, duration):
    """Run Locust load test programmatically."""
    env = Environment(user_classes=[user_class])

    # Start test
    env.create_local_runner()
    env.runner.start(num_users, spawn_rate=spawn_rate)

    # Run for duration
    import gevent
    gevent.spawn_later(duration, lambda: env.runner.quit())

    # Wait for test to complete
    env.runner.greenlet.join()

    # Get statistics
    stats = env.stats

    print("\n=== Locust Test Results ===")
    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Failures: {stats.total.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th Percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"99th Percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"Requests/s: {stats.total.total_rps:.2f}")

    return stats


# Pytest integration for Locust tests
@pytest.mark.performance
@pytest.mark.locust
def test_api_load_with_locust():
    """Run Locust load test against API."""
    stats = run_locust_test(
        user_class=AgentFactoryUser,
        num_users=100,
        spawn_rate=10,
        duration=60  # 1 minute
    )

    # Assertions
    success_rate = (stats.total.num_requests - stats.total.num_failures) / stats.total.num_requests
    assert success_rate >= 0.95

    p95_latency = stats.total.get_response_time_percentile(0.95)
    assert p95_latency < TARGET_P95_LATENCY_MS


@pytest.mark.performance
@pytest.mark.locust
def test_database_load_with_locust():
    """Run Locust load test for database operations."""
    stats = run_locust_test(
        user_class=DatabaseUser,
        num_users=50,
        spawn_rate=5,
        duration=60
    )

    success_rate = (stats.total.num_requests - stats.total.num_failures) / stats.total.num_requests
    assert success_rate >= 0.95
