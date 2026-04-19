"""
Performance Test Fixtures for GL-001 ThermalCommand.

Additional fixtures specific to performance tests.
These supplement the global fixtures in tests/conftest.py.

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import time
import statistics
from typing import List, Dict, Any
import tracemalloc


# =============================================================================
# Performance Targets
# =============================================================================

OPTIMIZATION_CYCLE_TARGET_S = 5.0
API_RESPONSE_TARGET_S = 0.200
CONTROL_CYCLE_TARGET_S = 0.100
DATA_THROUGHPUT_TARGET_PPS = 10000  # Points per second


# =============================================================================
# Benchmark Configuration
# =============================================================================

@pytest.fixture
def benchmark_config():
    """Provide benchmark configuration."""
    return {
        "warmup_iterations": 5,
        "measurement_iterations": 50,
        "target_optimization_time_s": OPTIMIZATION_CYCLE_TARGET_S,
        "target_api_response_time_s": API_RESPONSE_TARGET_S,
        "target_control_cycle_time_s": CONTROL_CYCLE_TARGET_S,
    }


# =============================================================================
# Latency Measurement Fixtures
# =============================================================================

@pytest.fixture
def latency_collector():
    """Provide latency measurement collector."""
    class LatencyCollector:
        def __init__(self):
            self.measurements: List[float] = []
            self.labels: List[str] = []

        def record(self, latency_ms: float, label: str = ""):
            self.measurements.append(latency_ms)
            self.labels.append(label)

        def get_percentile(self, percentile: float) -> float:
            if not self.measurements:
                return 0.0
            sorted_values = sorted(self.measurements)
            index = int(len(sorted_values) * percentile / 100)
            return sorted_values[min(index, len(sorted_values) - 1)]

        def get_mean(self) -> float:
            if not self.measurements:
                return 0.0
            return statistics.mean(self.measurements)

        def get_stdev(self) -> float:
            if len(self.measurements) < 2:
                return 0.0
            return statistics.stdev(self.measurements)

        def summary(self) -> Dict[str, float]:
            if not self.measurements:
                return {}
            return {
                "count": len(self.measurements),
                "mean_ms": self.get_mean(),
                "stdev_ms": self.get_stdev(),
                "min_ms": min(self.measurements),
                "max_ms": max(self.measurements),
                "p50_ms": self.get_percentile(50),
                "p95_ms": self.get_percentile(95),
                "p99_ms": self.get_percentile(99),
            }

        def clear(self):
            self.measurements = []
            self.labels = []

    return LatencyCollector()


# =============================================================================
# Timer Fixture
# =============================================================================

@pytest.fixture
def timer():
    """Provide a timer context manager."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed_ms = 0.0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed_ms = (self.end_time - self.start_time) * 1000

    return Timer


# =============================================================================
# Memory Tracking Fixture
# =============================================================================

@pytest.fixture
def memory_tracker():
    """Provide memory usage tracking."""
    class MemoryTracker:
        def __init__(self):
            self.start_size = 0
            self.peak_size = 0
            self.end_size = 0

        def __enter__(self):
            tracemalloc.start()
            self.start_size, _ = tracemalloc.get_traced_memory()
            return self

        def __exit__(self, *args):
            self.end_size, self.peak_size = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        @property
        def memory_used_mb(self) -> float:
            return (self.peak_size - self.start_size) / (1024 * 1024)

        @property
        def memory_increase_mb(self) -> float:
            return (self.end_size - self.start_size) / (1024 * 1024)

    return MemoryTracker


# =============================================================================
# Throughput Measurement Fixture
# =============================================================================

@pytest.fixture
def throughput_measurer():
    """Provide throughput measurement."""
    class ThroughputMeasurer:
        def __init__(self):
            self.count = 0
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()
            self.count = 0

        def increment(self, amount: int = 1):
            self.count += amount

        def stop(self):
            self.end_time = time.perf_counter()

        @property
        def duration_seconds(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time

        @property
        def throughput_per_second(self) -> float:
            if self.duration_seconds == 0:
                return 0.0
            return self.count / self.duration_seconds

    return ThroughputMeasurer


# =============================================================================
# Load Generation Fixtures
# =============================================================================

@pytest.fixture
def load_profiles():
    """Provide various load profile scenarios."""
    return {
        "constant": {
            "description": "Constant load",
            "pattern": [100.0] * 100,
        },
        "ramp_up": {
            "description": "Ramping up load",
            "pattern": [i * 2.0 for i in range(100)],
        },
        "step": {
            "description": "Step changes",
            "pattern": [50.0] * 30 + [100.0] * 40 + [50.0] * 30,
        },
        "sinusoidal": {
            "description": "Sinusoidal variation",
            "pattern": [50.0 + 30.0 * __import__('math').sin(i * 0.1) for i in range(100)],
        },
    }


@pytest.fixture
def stress_test_config():
    """Provide stress test configuration."""
    return {
        "num_iterations": 1000,
        "num_concurrent_threads": 10,
        "duration_seconds": 60,
        "max_memory_mb": 512,
        "max_latency_ms": 5000,
    }


# =============================================================================
# Benchmark Reporting Fixture
# =============================================================================

@pytest.fixture
def benchmark_reporter():
    """Provide benchmark reporting utility."""
    class BenchmarkReporter:
        def __init__(self):
            self.results: Dict[str, Dict] = {}

        def add_result(
            self,
            name: str,
            mean_ms: float,
            p95_ms: float,
            target_ms: float,
            passed: bool
        ):
            self.results[name] = {
                "mean_ms": mean_ms,
                "p95_ms": p95_ms,
                "target_ms": target_ms,
                "passed": passed,
            }

        def print_report(self):
            print("\n" + "=" * 70)
            print("PERFORMANCE BENCHMARK RESULTS")
            print("=" * 70)

            for name, result in self.results.items():
                status = "PASS" if result["passed"] else "FAIL"
                print(f"\n{name}:")
                print(f"  Mean:   {result['mean_ms']:.2f}ms")
                print(f"  P95:    {result['p95_ms']:.2f}ms")
                print(f"  Target: {result['target_ms']:.2f}ms")
                print(f"  Status: [{status}]")

            print("\n" + "=" * 70)

            passed = sum(1 for r in self.results.values() if r["passed"])
            total = len(self.results)
            print(f"Summary: {passed}/{total} tests passed")
            print("=" * 70)

        def all_passed(self) -> bool:
            return all(r["passed"] for r in self.results.values())

    return BenchmarkReporter


# =============================================================================
# Problem Size Fixtures
# =============================================================================

@pytest.fixture
def problem_sizes():
    """Provide various problem sizes for scaling tests."""
    return {
        "small": {
            "num_equipment": 3,
            "num_tags": 50,
            "num_policies": 10,
        },
        "medium": {
            "num_equipment": 10,
            "num_tags": 200,
            "num_policies": 50,
        },
        "large": {
            "num_equipment": 25,
            "num_tags": 500,
            "num_policies": 100,
        },
        "extra_large": {
            "num_equipment": 50,
            "num_tags": 1000,
            "num_policies": 200,
        },
    }
