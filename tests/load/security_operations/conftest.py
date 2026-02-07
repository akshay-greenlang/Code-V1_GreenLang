"""
Load test fixtures for security_operations module.

Provides data generators and performance measurement utilities.
"""

import pytest
import time
from datetime import datetime
from typing import Dict, Any, List, Generator
from uuid import uuid4
from contextlib import contextmanager


# -----------------------------------------------------------------------------
# Performance Measurement Utilities
# -----------------------------------------------------------------------------

@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start = time.perf_counter()
    result = {"elapsed_ms": 0}
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000


class PerformanceMetrics:
    """Track performance metrics during load tests."""

    def __init__(self):
        self.latencies: List[float] = []
        self.errors: int = 0
        self.successes: int = 0

    def record_latency(self, latency_ms: float):
        """Record a latency measurement."""
        self.latencies.append(latency_ms)
        self.successes += 1

    def record_error(self):
        """Record an error."""
        self.errors += 1

    @property
    def throughput(self) -> float:
        """Calculate throughput (operations per second)."""
        if not self.latencies:
            return 0.0
        total_time_s = sum(self.latencies) / 1000
        return len(self.latencies) / total_time_s if total_time_s > 0 else 0.0

    @property
    def p50(self) -> float:
        """Calculate P50 latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        return sorted_latencies[len(sorted_latencies) // 2]

    @property
    def p95(self) -> float:
        """Calculate P95 latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99(self) -> float:
        """Calculate P99 latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.successes + self.errors
        return self.errors / total if total > 0 else 0.0


@pytest.fixture
def performance_metrics():
    """Create performance metrics tracker."""
    return PerformanceMetrics()


# -----------------------------------------------------------------------------
# Data Generators
# -----------------------------------------------------------------------------

@pytest.fixture
def alert_generator():
    """Generator for creating test alerts."""
    def generate(count: int) -> Generator[Dict[str, Any], None, None]:
        for i in range(count):
            yield {
                "alert_id": str(uuid4()),
                "title": f"Load Test Alert {i}",
                "description": f"Alert {i} for load testing",
                "severity": ["critical", "high", "warning", "info"][i % 4],
                "source": ["prometheus", "loki", "guardduty"][i % 3],
                "timestamp": datetime.utcnow().isoformat(),
                "labels": {
                    "alertname": f"LoadTestAlert{i}",
                    "instance": f"node-{i % 10}",
                },
            }
    return generate


@pytest.fixture
def traffic_sample_generator():
    """Generator for creating traffic samples."""
    def generate(count: int) -> Generator[Dict[str, Any], None, None]:
        for i in range(count):
            yield {
                "sample_id": str(uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "source_ip": f"192.168.{i % 256}.{i % 256}",
                "destination_ip": "10.0.0.50",
                "source_port": 50000 + (i % 15000),
                "destination_port": 443,
                "method": ["GET", "POST", "PUT", "DELETE"][i % 4],
                "path": f"/api/v1/resource/{i % 1000}",
                "response_code": [200, 201, 400, 401, 500][i % 5],
                "response_time_ms": 20 + (i % 100),
            }
    return generate


@pytest.fixture
def threat_generator():
    """Generator for creating test threats."""
    def generate(count: int) -> Generator[Dict[str, Any], None, None]:
        categories = ["spoofing", "tampering", "repudiation", "info_disclosure", "dos", "elevation"]
        for i in range(count):
            yield {
                "threat_id": str(uuid4()),
                "title": f"Load Test Threat {i}",
                "category": categories[i % 6],
                "likelihood": (i % 100) / 100,
                "impact": (50 + (i % 50)) / 100,
            }
    return generate


# -----------------------------------------------------------------------------
# Load Test Configuration
# -----------------------------------------------------------------------------

@pytest.fixture
def load_test_config() -> Dict[str, Any]:
    """Load test configuration."""
    return {
        "small_load": {
            "alerts": 100,
            "traffic_samples": 1000,
            "threats": 50,
            "concurrent_users": 10,
        },
        "medium_load": {
            "alerts": 1000,
            "traffic_samples": 10000,
            "threats": 500,
            "concurrent_users": 50,
        },
        "large_load": {
            "alerts": 10000,
            "traffic_samples": 100000,
            "threats": 5000,
            "concurrent_users": 100,
        },
    }


# -----------------------------------------------------------------------------
# Performance Targets
# -----------------------------------------------------------------------------

@pytest.fixture
def performance_targets() -> Dict[str, Any]:
    """Performance targets for load tests."""
    return {
        "incident_detection": {
            "throughput_min": 1000,  # alerts/second
            "p95_latency_max_ms": 100,
            "error_rate_max": 0.01,
        },
        "threat_analysis": {
            "throughput_min": 100,  # components/second
            "p95_latency_max_ms": 500,
            "error_rate_max": 0.01,
        },
        "waf_rule_evaluation": {
            "throughput_min": 10000,  # requests/second
            "p95_latency_max_ms": 5,
            "error_rate_max": 0.001,
        },
        "traffic_analysis": {
            "throughput_min": 5000,  # samples/second
            "p95_latency_max_ms": 50,
            "error_rate_max": 0.01,
        },
    }
