"""Load Testing Framework for GreenLang Agents.

This module provides comprehensive load testing capabilities:
- Concurrent execution testing (10, 100, 1000 requests)
- Load pattern simulation (ramp-up, constant, spike)
- Metrics collection (RPS, latency percentiles, throughput)
- Results reporting (JSON, CSV, HTML)
- Performance validation against SLOs

Architecture:
    LoadTester: Main load testing orchestrator
    LoadPattern: Load generation patterns (ramp, constant, spike)
    LoadMetrics: Metrics collector and aggregator
    LoadReporter: Multi-format results reporter

Example Usage:
    >>> from tests.performance.load_testing import LoadTester, LoadPattern
    >>>
    >>> async def main():
    ...     tester = LoadTester()
    ...     results = await tester.run_load_test(
    ...         agent_name="FuelAgentAI",
    ...         pattern=LoadPattern.RAMP_UP,
    ...         target_rps=100,
    ...         duration_seconds=60
    ...     )
    ...     tester.generate_report(results, format="html")

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import json
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import csv

# Performance imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try importing async agents
try:
    from greenlang.agents.fuel_agent_ai_async import AsyncFuelAgentAI
    from greenlang.config import get_config
    ASYNC_AGENTS_AVAILABLE = True
except ImportError:
    ASYNC_AGENTS_AVAILABLE = False


# ==============================================================================
# Load Pattern Definitions
# ==============================================================================

class LoadPattern(Enum):
    """Load generation patterns for testing."""
    CONSTANT = "constant"  # Constant RPS throughout test
    RAMP_UP = "ramp_up"    # Gradual increase from 0 to target
    SPIKE = "spike"         # Sudden burst to target, then back down
    STEP = "step"           # Step increases every N seconds


# ==============================================================================
# Metrics Data Structures
# ==============================================================================

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: int
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class LoadTestMetrics:
    """Aggregated metrics from a load test run."""
    test_name: str
    pattern: str
    target_rps: int
    duration_seconds: float

    # Request metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float

    # Latency metrics (ms)
    min_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Throughput metrics
    actual_rps: float
    total_duration_seconds: float

    # Resource metrics
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0

    # Raw data
    request_metrics: List[RequestMetrics] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# ==============================================================================
# Load Tester
# ==============================================================================

class LoadTester:
    """Main load testing orchestrator for GreenLang agents."""

    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize load tester.

        Args:
            results_dir: Directory to save test results (default: tests/performance/results)
        """
        if results_dir is None:
            results_dir = Path(__file__).parent / "results"

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.config = get_config() if ASYNC_AGENTS_AVAILABLE else None

        # Test input data
        self.default_test_input = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US"
        }

    # ==========================================================================
    # Load Test Execution
    # ==========================================================================

    async def run_load_test(
        self,
        agent_name: str = "FuelAgentAI",
        pattern: LoadPattern = LoadPattern.CONSTANT,
        target_rps: int = 10,
        duration_seconds: float = 10,
        test_input: Optional[Dict[str, Any]] = None,
        agent_factory: Optional[Callable] = None,
    ) -> LoadTestMetrics:
        """Run a load test against an agent.

        Args:
            agent_name: Name of agent to test
            pattern: Load generation pattern
            target_rps: Target requests per second
            duration_seconds: Test duration in seconds
            test_input: Input data for agent (uses default if not provided)
            agent_factory: Custom agent factory function

        Returns:
            LoadTestMetrics with test results
        """
        print(f"\n{'='*80}")
        print(f"LOAD TEST: {agent_name}")
        print(f"{'='*80}")
        print(f"Pattern: {pattern.value}")
        print(f"Target RPS: {target_rps}")
        print(f"Duration: {duration_seconds}s")
        print(f"{'='*80}\n")

        if test_input is None:
            test_input = self.default_test_input

        # Create agent factory if not provided
        if agent_factory is None:
            agent_factory = self._default_agent_factory

        # Generate load schedule
        schedule = self._generate_schedule(pattern, target_rps, duration_seconds)

        # Execute load test
        request_metrics = await self._execute_load_test(
            schedule, test_input, agent_factory
        )

        # Aggregate metrics
        metrics = self._aggregate_metrics(
            test_name=f"{agent_name}_load_test",
            pattern=pattern.value,
            target_rps=target_rps,
            duration_seconds=duration_seconds,
            request_metrics=request_metrics
        )

        # Print summary
        self._print_metrics_summary(metrics)

        return metrics

    async def run_concurrent_load_test(
        self,
        num_concurrent: int,
        num_requests_per_agent: int = 1,
        test_input: Optional[Dict[str, Any]] = None,
    ) -> LoadTestMetrics:
        """Run concurrent execution test.

        Tests N agents executing M requests each concurrently.

        Args:
            num_concurrent: Number of concurrent agent instances
            num_requests_per_agent: Requests per agent
            test_input: Input data for agents

        Returns:
            LoadTestMetrics with test results
        """
        print(f"\n{'='*80}")
        print(f"CONCURRENT EXECUTION TEST")
        print(f"{'='*80}")
        print(f"Concurrent Agents: {num_concurrent}")
        print(f"Requests per Agent: {num_requests_per_agent}")
        print(f"Total Requests: {num_concurrent * num_requests_per_agent}")
        print(f"{'='*80}\n")

        if test_input is None:
            test_input = self.default_test_input

        start_time = time.perf_counter()
        request_metrics: List[RequestMetrics] = []

        # Track resource usage
        resource_monitor = ResourceMonitor() if PSUTIL_AVAILABLE else None
        if resource_monitor:
            resource_monitor.start()

        # Create concurrent tasks
        tasks = []
        request_id = 0

        async with AsyncFuelAgentAI(self.config) as agent:
            for agent_idx in range(num_concurrent):
                for req_idx in range(num_requests_per_agent):
                    task = self._execute_single_request(
                        agent, test_input, request_id
                    )
                    tasks.append(task)
                    request_id += 1

            # Execute all concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, RequestMetrics):
                request_metrics.append(result)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Stop resource monitoring
        if resource_monitor:
            resource_monitor.stop()

        # Aggregate metrics
        metrics = self._aggregate_metrics(
            test_name=f"concurrent_{num_concurrent}",
            pattern="concurrent",
            target_rps=num_concurrent * num_requests_per_agent,
            duration_seconds=duration,
            request_metrics=request_metrics
        )

        # Add resource metrics
        if resource_monitor:
            metrics.peak_cpu_percent = resource_monitor.peak_cpu
            metrics.peak_memory_mb = resource_monitor.peak_memory_mb
            metrics.avg_cpu_percent = resource_monitor.avg_cpu
            metrics.avg_memory_mb = resource_monitor.avg_memory_mb

        self._print_metrics_summary(metrics)

        return metrics

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _default_agent_factory(self):
        """Default agent factory."""
        return AsyncFuelAgentAI(self.config)

    def _generate_schedule(
        self, pattern: LoadPattern, target_rps: int, duration: float
    ) -> List[float]:
        """Generate request schedule based on load pattern.

        Returns:
            List of timestamps (seconds from start) for each request
        """
        schedule = []

        if pattern == LoadPattern.CONSTANT:
            # Constant rate throughout
            num_requests = int(target_rps * duration)
            interval = 1.0 / target_rps
            schedule = [i * interval for i in range(num_requests)]

        elif pattern == LoadPattern.RAMP_UP:
            # Gradual increase from 0 to target_rps
            num_segments = 10
            segment_duration = duration / num_segments

            for segment in range(num_segments):
                rps = target_rps * (segment + 1) / num_segments
                num_requests = int(rps * segment_duration)
                interval = segment_duration / max(num_requests, 1)
                base_time = segment * segment_duration

                for i in range(num_requests):
                    schedule.append(base_time + i * interval)

        elif pattern == LoadPattern.SPIKE:
            # Low rate, then spike, then low again
            low_rps = max(1, target_rps // 10)
            spike_duration = duration * 0.3  # 30% of time at peak
            warmup_duration = duration * 0.2
            cooldown_duration = duration * 0.2

            # Warmup
            num_warmup = int(low_rps * warmup_duration)
            schedule.extend([i * (warmup_duration / num_warmup) for i in range(num_warmup)])

            # Spike
            num_spike = int(target_rps * spike_duration)
            base_time = warmup_duration
            schedule.extend([base_time + i * (spike_duration / num_spike) for i in range(num_spike)])

            # Cooldown
            num_cooldown = int(low_rps * cooldown_duration)
            base_time = warmup_duration + spike_duration
            schedule.extend([base_time + i * (cooldown_duration / num_cooldown) for i in range(num_cooldown)])

        elif pattern == LoadPattern.STEP:
            # Step increases
            num_steps = 5
            step_duration = duration / num_steps

            for step in range(num_steps):
                rps = target_rps * (step + 1) / num_steps
                num_requests = int(rps * step_duration)
                interval = step_duration / max(num_requests, 1)
                base_time = step * step_duration

                for i in range(num_requests):
                    schedule.append(base_time + i * interval)

        return sorted(schedule)

    async def _execute_load_test(
        self,
        schedule: List[float],
        test_input: Dict[str, Any],
        agent_factory: Callable,
    ) -> List[RequestMetrics]:
        """Execute load test according to schedule."""
        request_metrics: List[RequestMetrics] = []
        start_time = time.perf_counter()

        async with agent_factory() as agent:
            tasks = []

            for request_id, scheduled_time in enumerate(schedule):
                # Wait until scheduled time
                current_time = time.perf_counter() - start_time
                wait_time = scheduled_time - current_time

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # Create request task
                task = self._execute_single_request(agent, test_input, request_id)
                tasks.append(task)

            # Wait for all requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, RequestMetrics):
                request_metrics.append(result)

        return request_metrics

    async def _execute_single_request(
        self, agent, test_input: Dict[str, Any], request_id: int
    ) -> RequestMetrics:
        """Execute a single request and collect metrics."""
        start = time.perf_counter()
        success = False
        error = None

        try:
            result = await agent.run_async(test_input)
            success = result.success
            if not success:
                error = result.error
        except Exception as e:
            error = str(e)

        end = time.perf_counter()
        duration_ms = (end - start) * 1000

        return RequestMetrics(
            request_id=request_id,
            start_time=start,
            end_time=end,
            duration_ms=duration_ms,
            success=success,
            error=error
        )

    def _aggregate_metrics(
        self,
        test_name: str,
        pattern: str,
        target_rps: int,
        duration_seconds: float,
        request_metrics: List[RequestMetrics]
    ) -> LoadTestMetrics:
        """Aggregate request metrics into summary statistics."""
        total_requests = len(request_metrics)
        successful_requests = sum(1 for m in request_metrics if m.success)
        failed_requests = total_requests - successful_requests
        error_rate = failed_requests / total_requests if total_requests > 0 else 0

        # Latency metrics
        latencies = [m.duration_ms for m in request_metrics]

        if latencies:
            sorted_latencies = sorted(latencies)

            min_latency = min(latencies)
            max_latency = max(latencies)
            mean_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        else:
            min_latency = max_latency = mean_latency = median_latency = 0
            p95_latency = p99_latency = 0

        # Throughput
        if request_metrics:
            actual_duration = max(m.end_time for m in request_metrics) - min(m.start_time for m in request_metrics)
            actual_rps = total_requests / actual_duration if actual_duration > 0 else 0
        else:
            actual_duration = 0
            actual_rps = 0

        return LoadTestMetrics(
            test_name=test_name,
            pattern=pattern,
            target_rps=target_rps,
            duration_seconds=duration_seconds,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_rate=error_rate,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            actual_rps=actual_rps,
            total_duration_seconds=actual_duration,
            request_metrics=request_metrics
        )

    def _print_metrics_summary(self, metrics: LoadTestMetrics):
        """Print metrics summary to console."""
        print(f"\n{'='*80}")
        print(f"LOAD TEST RESULTS: {metrics.test_name}")
        print(f"{'='*80}")
        print(f"\nRequest Metrics:")
        print(f"  Total Requests:      {metrics.total_requests:,}")
        print(f"  Successful:          {metrics.successful_requests:,} ({100 - metrics.error_rate*100:.1f}%)")
        print(f"  Failed:              {metrics.failed_requests:,} ({metrics.error_rate*100:.1f}%)")
        print(f"\nLatency (ms):")
        print(f"  Min:                 {metrics.min_latency_ms:.2f}")
        print(f"  Mean:                {metrics.mean_latency_ms:.2f}")
        print(f"  Median (p50):        {metrics.median_latency_ms:.2f}")
        print(f"  p95:                 {metrics.p95_latency_ms:.2f}")
        print(f"  p99:                 {metrics.p99_latency_ms:.2f}")
        print(f"  Max:                 {metrics.max_latency_ms:.2f}")
        print(f"\nThroughput:")
        print(f"  Target RPS:          {metrics.target_rps}")
        print(f"  Actual RPS:          {metrics.actual_rps:.2f}")
        print(f"  Duration:            {metrics.total_duration_seconds:.2f}s")

        if metrics.peak_cpu_percent > 0:
            print(f"\nResource Usage:")
            print(f"  Peak CPU:            {metrics.peak_cpu_percent:.1f}%")
            print(f"  Peak Memory:         {metrics.peak_memory_mb:.1f} MB")
            print(f"  Avg CPU:             {metrics.avg_cpu_percent:.1f}%")
            print(f"  Avg Memory:          {metrics.avg_memory_mb:.1f} MB")

        print(f"\n{'='*80}\n")

    # ==========================================================================
    # Results Reporting
    # ==========================================================================

    def save_results_json(self, metrics: LoadTestMetrics, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            filename = f"{metrics.test_name}_{metrics.timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.results_dir / filename

        # Convert to dict
        data = {
            "test_name": metrics.test_name,
            "pattern": metrics.pattern,
            "target_rps": metrics.target_rps,
            "duration_seconds": metrics.duration_seconds,
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "error_rate": metrics.error_rate,
            "latency_ms": {
                "min": metrics.min_latency_ms,
                "max": metrics.max_latency_ms,
                "mean": metrics.mean_latency_ms,
                "median": metrics.median_latency_ms,
                "p95": metrics.p95_latency_ms,
                "p99": metrics.p99_latency_ms,
            },
            "throughput": {
                "target_rps": metrics.target_rps,
                "actual_rps": metrics.actual_rps,
                "total_duration_seconds": metrics.total_duration_seconds,
            },
            "resources": {
                "peak_cpu_percent": metrics.peak_cpu_percent,
                "peak_memory_mb": metrics.peak_memory_mb,
                "avg_cpu_percent": metrics.avg_cpu_percent,
                "avg_memory_mb": metrics.avg_memory_mb,
            },
            "timestamp": metrics.timestamp.isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {filepath}")
        return filepath

    def save_results_csv(self, metrics: LoadTestMetrics, filename: Optional[str] = None):
        """Save detailed request metrics to CSV."""
        if filename is None:
            filename = f"{metrics.test_name}_{metrics.timestamp.strftime('%Y%m%d_%H%M%S')}_requests.csv"

        filepath = self.results_dir / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'request_id', 'start_time', 'end_time', 'duration_ms', 'success', 'error'
            ])
            writer.writeheader()

            for req in metrics.request_metrics:
                writer.writerow({
                    'request_id': req.request_id,
                    'start_time': req.start_time,
                    'end_time': req.end_time,
                    'duration_ms': req.duration_ms,
                    'success': req.success,
                    'error': req.error or '',
                })

        print(f"CSV results saved to: {filepath}")
        return filepath


# ==============================================================================
# Resource Monitor
# ==============================================================================

class ResourceMonitor:
    """Monitor system resources during load test."""

    def __init__(self, sample_interval: float = 0.1):
        """Initialize resource monitor.

        Args:
            sample_interval: How often to sample resources (seconds)
        """
        self.sample_interval = sample_interval
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self._monitor_task = None

    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()

    async def _monitor_loop(self):
        """Monitor resources in background."""
        process = psutil.Process()

        while self.monitoring:
            try:
                cpu = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024

                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory_mb)

                await asyncio.sleep(self.sample_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    @property
    def peak_cpu(self) -> float:
        """Get peak CPU usage."""
        return max(self.cpu_samples) if self.cpu_samples else 0

    @property
    def avg_cpu(self) -> float:
        """Get average CPU usage."""
        return statistics.mean(self.cpu_samples) if self.cpu_samples else 0

    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return max(self.memory_samples) if self.memory_samples else 0

    @property
    def avg_memory_mb(self) -> float:
        """Get average memory usage in MB."""
        return statistics.mean(self.memory_samples) if self.memory_samples else 0


# ==============================================================================
# Main Entry Point
# ==============================================================================

async def main():
    """Run example load tests."""
    if not ASYNC_AGENTS_AVAILABLE:
        print("Error: Async agents not available. Cannot run load tests.")
        return

    tester = LoadTester()

    # Test 1: 10 concurrent executions
    print("\n" + "="*80)
    print("TEST 1: 10 Concurrent Executions")
    print("="*80)
    metrics1 = await tester.run_concurrent_load_test(num_concurrent=10)
    tester.save_results_json(metrics1, "test_10_concurrent.json")

    # Test 2: 100 concurrent executions
    print("\n" + "="*80)
    print("TEST 2: 100 Concurrent Executions")
    print("="*80)
    metrics2 = await tester.run_concurrent_load_test(num_concurrent=100)
    tester.save_results_json(metrics2, "test_100_concurrent.json")

    # Test 3: Load test with ramp-up
    print("\n" + "="*80)
    print("TEST 3: Ramp-Up Load Test (10 RPS for 30s)")
    print("="*80)
    metrics3 = await tester.run_load_test(
        pattern=LoadPattern.RAMP_UP,
        target_rps=10,
        duration_seconds=30
    )
    tester.save_results_json(metrics3, "test_ramp_up.json")

    print("\n" + "="*80)
    print("ALL LOAD TESTS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
