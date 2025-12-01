# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Performance Tests - Shared Pytest Fixtures.

This module provides comprehensive performance test fixtures including:
- Timing utilities and decorators
- Load generators for stress testing
- Resource monitors (CPU, memory)
- Mock connectors with configurable latency
- Performance test configuration

Author: GL-TestEngineer
Version: 1.0.0
"""

import asyncio
import gc
import os
import random
import statistics
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

# Add parent directories to path for imports
TEST_DIR = Path(__file__).parent
AGENT_DIR = TEST_DIR.parent.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR / "calculators"))


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers for performance tests."""
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests with timing")
    config.addinivalue_line("markers", "stress: Stress tests")
    config.addinivalue_line("markers", "scalability: Scalability tests")
    config.addinivalue_line("markers", "latency: Latency measurement tests")
    config.addinivalue_line("markers", "throughput: Throughput tests")
    config.addinivalue_line("markers", "memory: Memory usage tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "asyncio: Async tests")


# =============================================================================
# PERFORMANCE TARGET CONSTANTS
# =============================================================================

@dataclass
class PerformanceTargets:
    """Performance benchmark targets for GL-012 STEAMQUAL."""

    # Calculation benchmarks (milliseconds)
    single_steam_quality_calc_ms: float = 1.0
    batch_throughput_per_sec: int = 10000
    memory_per_calc_bytes: int = 1024  # 1KB
    desuperheater_injection_calc_ms: float = 0.5
    energy_balance_validation_ms: float = 0.5
    pid_calculation_ms: float = 0.1
    valve_position_calc_ms: float = 0.2
    pressure_drop_calc_ms: float = 0.3

    # Orchestration performance
    full_orchestration_cycle_ms: float = 100.0
    concurrent_orchestrations: int = 10
    cache_hit_rate_min: float = 0.80
    memory_growth_max_mb: float = 50.0

    # Integration latency (milliseconds)
    meter_read_latency_ms: float = 50.0
    valve_command_latency_ms: float = 100.0
    scada_tag_read_ms: float = 50.0
    batch_tag_read_100_ms: float = 200.0
    subscription_delivery_ms: float = 10.0

    # Scalability
    max_steam_headers: int = 100
    memory_scaling_factor: float = 1.5  # Linear + 50% overhead
    cpu_scaling_exponent: float = 0.8  # Sub-linear

    # Stress testing
    sustained_ops_per_sec: int = 1000
    sustained_duration_sec: int = 60  # Reduced for testing (10 min in prod)
    burst_multiplier: int = 10
    burst_duration_sec: int = 30
    recovery_time_sec: float = 5.0


@pytest.fixture
def performance_targets() -> PerformanceTargets:
    """Provide performance targets for tests."""
    return PerformanceTargets()


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class PerformanceTimer:
    """
    High-precision timer for performance measurements.

    Uses time.perf_counter_ns() for nanosecond precision.
    Supports context manager and explicit start/stop.
    """

    def __init__(self):
        self.start_time_ns: Optional[int] = None
        self.end_time_ns: Optional[int] = None
        self._samples: List[int] = []

    def __enter__(self) -> "PerformanceTimer":
        self.start_time_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args) -> None:
        self.end_time_ns = time.perf_counter_ns()

    def start(self) -> None:
        """Start timing."""
        self.start_time_ns = time.perf_counter_ns()

    def stop(self) -> int:
        """Stop timing and return elapsed nanoseconds."""
        self.end_time_ns = time.perf_counter_ns()
        elapsed = self.elapsed_ns
        self._samples.append(elapsed)
        return elapsed

    def lap(self) -> int:
        """Record lap time without stopping."""
        lap_time = time.perf_counter_ns()
        if self.start_time_ns:
            elapsed = lap_time - self.start_time_ns
            self._samples.append(elapsed)
            self.start_time_ns = lap_time
            return elapsed
        return 0

    @property
    def elapsed_ns(self) -> int:
        """Get elapsed time in nanoseconds."""
        if self.start_time_ns and self.end_time_ns:
            return self.end_time_ns - self.start_time_ns
        return 0

    @property
    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds."""
        return self.elapsed_ns / 1000

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics for all samples."""
        if not self._samples:
            return {}

        samples_ms = [s / 1_000_000 for s in self._samples]
        return {
            "count": len(samples_ms),
            "mean_ms": statistics.mean(samples_ms),
            "median_ms": statistics.median(samples_ms),
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
            "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0,
            "p95_ms": sorted(samples_ms)[int(len(samples_ms) * 0.95)] if len(samples_ms) >= 20 else max(samples_ms),
            "p99_ms": sorted(samples_ms)[int(len(samples_ms) * 0.99)] if len(samples_ms) >= 100 else max(samples_ms),
        }


@pytest.fixture
def performance_timer() -> PerformanceTimer:
    """Provide performance timer fixture."""
    return PerformanceTimer()


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        timer = PerformanceTimer()
        timer.start()
        result = func(*args, **kwargs)
        elapsed = timer.stop()
        print(f"{func.__name__} took {timer.elapsed_ms:.4f}ms")
        return result
    return wrapper


def async_timed(func: Callable) -> Callable:
    """Decorator to time async function execution."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        timer = PerformanceTimer()
        timer.start()
        result = await func(*args, **kwargs)
        elapsed = timer.stop()
        print(f"{func.__name__} took {timer.elapsed_ms:.4f}ms")
        return result
    return wrapper


@contextmanager
def benchmark_context(name: str, iterations: int = 1) -> Generator[PerformanceTimer, None, None]:
    """Context manager for benchmarking with summary output."""
    timer = PerformanceTimer()
    yield timer
    stats = timer.get_statistics()
    if stats:
        print(f"\n{name} ({iterations} iterations):")
        print(f"  Mean: {stats['mean_ms']:.4f}ms")
        print(f"  Min/Max: {stats['min_ms']:.4f}ms / {stats['max_ms']:.4f}ms")
        if 'p95_ms' in stats:
            print(f"  P95/P99: {stats['p95_ms']:.4f}ms / {stats.get('p99_ms', 'N/A')}ms")


# =============================================================================
# LOAD GENERATORS
# =============================================================================

@dataclass
class LoadProfile:
    """Configuration for load generation."""
    base_ops_per_sec: int = 100
    duration_sec: float = 10.0
    ramp_up_sec: float = 1.0
    ramp_down_sec: float = 1.0
    burst_factor: float = 1.0
    burst_duration_sec: float = 0.0
    num_workers: int = 4


class LoadGenerator:
    """
    Generate controlled load for performance testing.

    Supports steady load, ramp-up/down, and burst patterns.
    """

    def __init__(self, profile: LoadProfile):
        self.profile = profile
        self._stop_event = threading.Event()
        self._results: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def generate_sync(
        self,
        operation: Callable[[], Any],
        callback: Optional[Callable[[int, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Generate synchronous load.

        Args:
            operation: Operation to execute
            callback: Optional callback(ops_completed, elapsed_sec)

        Returns:
            Load test results with throughput and latencies
        """
        self._stop_event.clear()
        self._results = []

        start_time = time.perf_counter()
        ops_completed = 0
        latencies_ms: List[float] = []

        with ThreadPoolExecutor(max_workers=self.profile.num_workers) as executor:
            while not self._stop_event.is_set():
                elapsed = time.perf_counter() - start_time

                # Check if duration exceeded
                if elapsed >= self.profile.duration_sec:
                    break

                # Calculate target ops/sec based on ramp and burst
                target_ops_per_sec = self._calculate_target_ops(elapsed)

                # Submit operations to maintain target rate
                interval = 1.0 / target_ops_per_sec if target_ops_per_sec > 0 else 0.01

                op_start = time.perf_counter()
                try:
                    future = executor.submit(operation)
                    future.result(timeout=1.0)
                    latency_ms = (time.perf_counter() - op_start) * 1000
                    latencies_ms.append(latency_ms)
                    ops_completed += 1
                except Exception as e:
                    with self._lock:
                        self._results.append({"error": str(e)})

                if callback:
                    callback(ops_completed, elapsed)

                # Rate limiting
                sleep_time = interval - (time.perf_counter() - op_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        duration = time.perf_counter() - start_time
        return self._compile_results(ops_completed, duration, latencies_ms)

    async def generate_async(
        self,
        operation: Callable[[], Any],
        callback: Optional[Callable[[int, float], None]] = None
    ) -> Dict[str, Any]:
        """Generate asynchronous load."""
        self._stop_event.clear()
        start_time = time.perf_counter()
        ops_completed = 0
        latencies_ms: List[float] = []

        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - start_time

            if elapsed >= self.profile.duration_sec:
                break

            target_ops_per_sec = self._calculate_target_ops(elapsed)
            interval = 1.0 / target_ops_per_sec if target_ops_per_sec > 0 else 0.01

            op_start = time.perf_counter()
            try:
                if asyncio.iscoroutinefunction(operation):
                    await operation()
                else:
                    operation()
                latency_ms = (time.perf_counter() - op_start) * 1000
                latencies_ms.append(latency_ms)
                ops_completed += 1
            except Exception as e:
                pass

            if callback:
                callback(ops_completed, elapsed)

            sleep_time = interval - (time.perf_counter() - op_start)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        duration = time.perf_counter() - start_time
        return self._compile_results(ops_completed, duration, latencies_ms)

    def _calculate_target_ops(self, elapsed: float) -> float:
        """Calculate target operations per second based on profile."""
        profile = self.profile

        # Ramp-up phase
        if elapsed < profile.ramp_up_sec:
            ramp_factor = elapsed / profile.ramp_up_sec
            return profile.base_ops_per_sec * ramp_factor

        # Ramp-down phase
        if elapsed > profile.duration_sec - profile.ramp_down_sec:
            remaining = profile.duration_sec - elapsed
            ramp_factor = remaining / profile.ramp_down_sec
            return profile.base_ops_per_sec * max(0.1, ramp_factor)

        # Burst phase
        if profile.burst_duration_sec > 0:
            burst_start = profile.ramp_up_sec + (profile.duration_sec - profile.ramp_up_sec - profile.ramp_down_sec - profile.burst_duration_sec) / 2
            if burst_start <= elapsed < burst_start + profile.burst_duration_sec:
                return profile.base_ops_per_sec * profile.burst_factor

        return profile.base_ops_per_sec

    def _compile_results(
        self,
        ops_completed: int,
        duration: float,
        latencies_ms: List[float]
    ) -> Dict[str, Any]:
        """Compile load test results."""
        results = {
            "ops_completed": ops_completed,
            "duration_sec": duration,
            "throughput_ops_sec": ops_completed / duration if duration > 0 else 0,
        }

        if latencies_ms:
            sorted_latencies = sorted(latencies_ms)
            results["latency_mean_ms"] = statistics.mean(latencies_ms)
            results["latency_median_ms"] = statistics.median(latencies_ms)
            results["latency_min_ms"] = min(latencies_ms)
            results["latency_max_ms"] = max(latencies_ms)
            results["latency_p95_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            results["latency_p99_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) >= 100 else max(latencies_ms)
            if len(latencies_ms) > 1:
                results["latency_stdev_ms"] = statistics.stdev(latencies_ms)

        return results

    def stop(self) -> None:
        """Stop load generation."""
        self._stop_event.set()


@pytest.fixture
def load_generator() -> Callable[[LoadProfile], LoadGenerator]:
    """Factory fixture for creating load generators."""
    def _create_generator(profile: LoadProfile) -> LoadGenerator:
        return LoadGenerator(profile)
    return _create_generator


# =============================================================================
# RESOURCE MONITORS
# =============================================================================

class MemoryMonitor:
    """
    Monitor memory usage during performance tests.

    Uses tracemalloc for detailed memory tracking.
    """

    def __init__(self):
        self._snapshots: List[Tuple[float, int]] = []
        self._start_memory: int = 0
        self._peak_memory: int = 0

    def start(self) -> None:
        """Start memory monitoring."""
        gc.collect()
        tracemalloc.start()
        self._start_memory = tracemalloc.get_traced_memory()[0]
        self._snapshots = [(0, self._start_memory)]

    def sample(self) -> int:
        """Take memory sample."""
        current, peak = tracemalloc.get_traced_memory()
        elapsed = len(self._snapshots)
        self._snapshots.append((elapsed, current))
        self._peak_memory = max(self._peak_memory, peak)
        return current

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return results."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        growth = current - self._start_memory
        return {
            "start_bytes": self._start_memory,
            "end_bytes": current,
            "peak_bytes": peak,
            "growth_bytes": growth,
            "growth_mb": growth / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
            "samples": self._snapshots,
        }

    def get_growth_mb(self) -> float:
        """Get memory growth in MB."""
        current = tracemalloc.get_traced_memory()[0]
        return (current - self._start_memory) / (1024 * 1024)


@pytest.fixture
def memory_monitor() -> MemoryMonitor:
    """Provide memory monitor fixture."""
    return MemoryMonitor()


class CPUMonitor:
    """Monitor CPU usage during performance tests."""

    def __init__(self):
        self._samples: List[Tuple[float, float]] = []
        self._start_time: float = 0
        self._start_cpu_time: float = 0

    def start(self) -> None:
        """Start CPU monitoring."""
        self._start_time = time.perf_counter()
        self._start_cpu_time = time.process_time()
        self._samples = []

    def sample(self) -> float:
        """Take CPU utilization sample."""
        elapsed = time.perf_counter() - self._start_time
        cpu_time = time.process_time() - self._start_cpu_time
        utilization = (cpu_time / elapsed * 100) if elapsed > 0 else 0
        self._samples.append((elapsed, utilization))
        return utilization

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return results."""
        elapsed = time.perf_counter() - self._start_time
        cpu_time = time.process_time() - self._start_cpu_time

        utilizations = [s[1] for s in self._samples] if self._samples else [0]
        return {
            "elapsed_sec": elapsed,
            "cpu_time_sec": cpu_time,
            "avg_utilization_pct": statistics.mean(utilizations),
            "max_utilization_pct": max(utilizations),
            "samples": self._samples,
        }


@pytest.fixture
def cpu_monitor() -> CPUMonitor:
    """Provide CPU monitor fixture."""
    return CPUMonitor()


# =============================================================================
# MOCK CONNECTORS WITH CONFIGURABLE LATENCY
# =============================================================================

class MockLatencyMeter:
    """Mock steam meter with configurable read latency."""

    def __init__(
        self,
        base_latency_ms: float = 10.0,
        latency_variance_ms: float = 2.0
    ):
        self.base_latency_ms = base_latency_ms
        self.latency_variance_ms = latency_variance_ms
        self._connected = False
        self._read_count = 0

    async def connect(self) -> bool:
        await asyncio.sleep(0.01)  # Connection overhead
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        return True

    async def read_pressure(self) -> float:
        """Read pressure with simulated latency."""
        await self._simulate_latency()
        self._read_count += 1
        return 10.0 + random.uniform(-0.1, 0.1)

    async def read_temperature(self) -> float:
        """Read temperature with simulated latency."""
        await self._simulate_latency()
        self._read_count += 1
        return 250.0 + random.uniform(-1.0, 1.0)

    async def read_flow_rate(self) -> float:
        """Read flow rate with simulated latency."""
        await self._simulate_latency()
        self._read_count += 1
        return 50.0 + random.uniform(-2.0, 2.0)

    async def read_all(self) -> Dict[str, float]:
        """Read all values in batch."""
        await self._simulate_latency()
        self._read_count += 1
        return {
            "pressure_bar": 10.0 + random.uniform(-0.1, 0.1),
            "temperature_c": 250.0 + random.uniform(-1.0, 1.0),
            "flow_rate_kg_s": 50.0 + random.uniform(-2.0, 2.0),
            "dryness_fraction": 0.98 + random.uniform(-0.01, 0.01),
        }

    async def _simulate_latency(self) -> None:
        """Simulate I/O latency."""
        latency = self.base_latency_ms + random.uniform(
            -self.latency_variance_ms,
            self.latency_variance_ms
        )
        await asyncio.sleep(max(0, latency) / 1000)


class MockLatencyValve:
    """Mock control valve with configurable command latency."""

    def __init__(
        self,
        base_latency_ms: float = 20.0,
        latency_variance_ms: float = 5.0
    ):
        self.base_latency_ms = base_latency_ms
        self.latency_variance_ms = latency_variance_ms
        self._position = 50.0
        self._command_count = 0

    async def connect(self) -> bool:
        await asyncio.sleep(0.02)
        return True

    async def disconnect(self) -> bool:
        return True

    async def get_position(self) -> float:
        """Get current valve position."""
        await self._simulate_latency()
        return self._position

    async def set_position(self, position: float) -> bool:
        """Set valve position with simulated latency."""
        await self._simulate_latency()
        self._position = max(0.0, min(100.0, position))
        self._command_count += 1
        return True

    async def _simulate_latency(self) -> None:
        """Simulate command latency."""
        latency = self.base_latency_ms + random.uniform(
            -self.latency_variance_ms,
            self.latency_variance_ms
        )
        await asyncio.sleep(max(0, latency) / 1000)


class MockLatencySCADA:
    """Mock SCADA connector with configurable latency."""

    def __init__(
        self,
        base_latency_ms: float = 15.0,
        per_tag_latency_ms: float = 1.0
    ):
        self.base_latency_ms = base_latency_ms
        self.per_tag_latency_ms = per_tag_latency_ms
        self._subscriptions: Dict[str, Callable] = {}

    async def connect(self) -> bool:
        await asyncio.sleep(0.05)
        return True

    async def disconnect(self) -> bool:
        return True

    async def read_tag(self, tag: str) -> Any:
        """Read single tag."""
        await asyncio.sleep(self.base_latency_ms / 1000)
        return self._generate_tag_value(tag)

    async def read_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Read multiple tags in batch."""
        latency_ms = self.base_latency_ms + len(tags) * self.per_tag_latency_ms
        await asyncio.sleep(latency_ms / 1000)
        return {tag: self._generate_tag_value(tag) for tag in tags}

    async def write_tag(self, tag: str, value: Any) -> bool:
        """Write tag value."""
        await asyncio.sleep(self.base_latency_ms / 1000)
        return True

    async def subscribe(self, tag: str, callback: Callable) -> bool:
        """Subscribe to tag changes."""
        self._subscriptions[tag] = callback
        return True

    async def trigger_subscription(self, tag: str, value: Any) -> None:
        """Trigger subscription callback (for testing)."""
        if tag in self._subscriptions:
            await asyncio.sleep(0.005)  # 5ms delivery latency
            self._subscriptions[tag](value)

    def _generate_tag_value(self, tag: str) -> Any:
        """Generate mock tag value."""
        if "pressure" in tag.lower():
            return 10.0 + random.uniform(-0.2, 0.2)
        elif "temperature" in tag.lower():
            return 250.0 + random.uniform(-2.0, 2.0)
        elif "flow" in tag.lower():
            return 50.0 + random.uniform(-3.0, 3.0)
        else:
            return random.random() * 100


@pytest.fixture
def mock_meter() -> MockLatencyMeter:
    """Provide mock meter with realistic latency."""
    return MockLatencyMeter(base_latency_ms=10.0, latency_variance_ms=2.0)


@pytest.fixture
def mock_valve() -> MockLatencyValve:
    """Provide mock valve with realistic latency."""
    return MockLatencyValve(base_latency_ms=20.0, latency_variance_ms=5.0)


@pytest.fixture
def mock_scada() -> MockLatencySCADA:
    """Provide mock SCADA with realistic latency."""
    return MockLatencySCADA(base_latency_ms=15.0, per_tag_latency_ms=1.0)


# =============================================================================
# EVENT LOOP FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def generate_steam_quality_inputs(count: int) -> List[Dict[str, float]]:
    """Generate random steam quality input data."""
    inputs = []
    for _ in range(count):
        inputs.append({
            "pressure_mpa": random.uniform(0.1, 20.0),
            "temperature_c": random.uniform(100.0, 400.0),
            "pressure_stability": random.uniform(0.8, 1.0),
            "temperature_stability": random.uniform(0.8, 1.0),
        })
    return inputs


def generate_desuperheater_inputs(count: int) -> List[Dict[str, float]]:
    """Generate random desuperheater input data."""
    inputs = []
    for _ in range(count):
        inputs.append({
            "steam_flow_kg_s": random.uniform(5.0, 100.0),
            "inlet_temperature_c": random.uniform(250.0, 450.0),
            "inlet_pressure_mpa": random.uniform(1.0, 10.0),
            "target_temperature_c": random.uniform(200.0, 350.0),
            "water_temperature_c": random.uniform(20.0, 100.0),
        })
    return inputs


def generate_pressure_control_inputs(count: int) -> List[Dict[str, float]]:
    """Generate random pressure control input data."""
    inputs = []
    for _ in range(count):
        inputs.append({
            "setpoint_mpa": random.uniform(0.5, 10.0),
            "actual_mpa": random.uniform(0.4, 10.5),
            "flow_rate_kg_s": random.uniform(1.0, 50.0),
            "fluid_density_kg_m3": random.uniform(2.0, 50.0),
        })
    return inputs


@pytest.fixture
def steam_quality_test_data() -> List[Dict[str, float]]:
    """Generate test data for steam quality calculations."""
    return generate_steam_quality_inputs(1000)


@pytest.fixture
def desuperheater_test_data() -> List[Dict[str, float]]:
    """Generate test data for desuperheater calculations."""
    return generate_desuperheater_inputs(1000)


@pytest.fixture
def pressure_control_test_data() -> List[Dict[str, float]]:
    """Generate test data for pressure control calculations."""
    return generate_pressure_control_inputs(1000)


# =============================================================================
# THREAD POOL FIXTURES
# =============================================================================

@pytest.fixture
def thread_pool_4() -> ThreadPoolExecutor:
    """Provide 4-worker thread pool."""
    executor = ThreadPoolExecutor(max_workers=4)
    yield executor
    executor.shutdown(wait=True)


@pytest.fixture
def thread_pool_16() -> ThreadPoolExecutor:
    """Provide 16-worker thread pool."""
    executor = ThreadPoolExecutor(max_workers=16)
    yield executor
    executor.shutdown(wait=True)


@pytest.fixture
def thread_pool_50() -> ThreadPoolExecutor:
    """Provide 50-worker thread pool."""
    executor = ThreadPoolExecutor(max_workers=50)
    yield executor
    executor.shutdown(wait=True)


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

def assert_latency_target(
    actual_ms: float,
    target_ms: float,
    operation: str,
    tolerance_pct: float = 10.0
) -> None:
    """Assert latency meets target within tolerance."""
    max_allowed = target_ms * (1 + tolerance_pct / 100)
    assert actual_ms <= max_allowed, (
        f"{operation} latency {actual_ms:.2f}ms exceeds target "
        f"{target_ms:.2f}ms (max allowed: {max_allowed:.2f}ms)"
    )


def assert_throughput_target(
    actual_ops_sec: float,
    target_ops_sec: float,
    operation: str,
    tolerance_pct: float = 10.0
) -> None:
    """Assert throughput meets target within tolerance."""
    min_allowed = target_ops_sec * (1 - tolerance_pct / 100)
    assert actual_ops_sec >= min_allowed, (
        f"{operation} throughput {actual_ops_sec:.0f}/sec below target "
        f"{target_ops_sec:.0f}/sec (min allowed: {min_allowed:.0f}/sec)"
    )


def assert_memory_target(
    growth_mb: float,
    max_growth_mb: float,
    operation: str
) -> None:
    """Assert memory growth within limits."""
    assert growth_mb <= max_growth_mb, (
        f"{operation} memory growth {growth_mb:.2f}MB exceeds "
        f"limit {max_growth_mb:.2f}MB"
    )


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_between_tests():
    """Clean up between tests."""
    gc.collect()
    yield
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def cleanup_session():
    """Clean up after test session."""
    yield
    gc.collect()
