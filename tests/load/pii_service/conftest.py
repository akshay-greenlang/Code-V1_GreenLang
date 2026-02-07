# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for PII Service load tests.

Provides fixtures for load testing:
- Bulk data generators
- Performance measurement utilities
- Concurrent request helpers

Author: GreenLang Test Engineering Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import pytest


# ============================================================================
# Performance Measurement
# ============================================================================


class PerformanceMetrics:
    """Collect and calculate performance metrics."""

    def __init__(self):
        self.latencies: List[float] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Start timing."""
        self.start_time = time.perf_counter()

    def stop(self):
        """Stop timing."""
        self.end_time = time.perf_counter()

    def record_latency(self, latency_ms: float):
        """Record a single operation latency."""
        self.latencies.append(latency_ms)

    @property
    def total_duration_seconds(self) -> float:
        """Total test duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def throughput(self) -> float:
        """Operations per second."""
        if self.total_duration_seconds > 0:
            return len(self.latencies) / self.total_duration_seconds
        return 0.0

    @property
    def p50(self) -> float:
        """50th percentile latency in ms."""
        if self.latencies:
            return statistics.median(self.latencies)
        return 0.0

    @property
    def p95(self) -> float:
        """95th percentile latency in ms."""
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            idx = int(len(sorted_latencies) * 0.95)
            return sorted_latencies[idx] if idx < len(sorted_latencies) else sorted_latencies[-1]
        return 0.0

    @property
    def p99(self) -> float:
        """99th percentile latency in ms."""
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            idx = int(len(sorted_latencies) * 0.99)
            return sorted_latencies[idx] if idx < len(sorted_latencies) else sorted_latencies[-1]
        return 0.0

    @property
    def avg(self) -> float:
        """Average latency in ms."""
        if self.latencies:
            return statistics.mean(self.latencies)
        return 0.0

    @property
    def min(self) -> float:
        """Minimum latency in ms."""
        if self.latencies:
            return min(self.latencies)
        return 0.0

    @property
    def max(self) -> float:
        """Maximum latency in ms."""
        if self.latencies:
            return max(self.latencies)
        return 0.0

    def report(self) -> Dict[str, float]:
        """Generate metrics report."""
        return {
            "total_operations": len(self.latencies),
            "total_duration_seconds": self.total_duration_seconds,
            "throughput_ops_per_sec": self.throughput,
            "latency_avg_ms": self.avg,
            "latency_min_ms": self.min,
            "latency_max_ms": self.max,
            "latency_p50_ms": self.p50,
            "latency_p95_ms": self.p95,
            "latency_p99_ms": self.p99,
        }


@pytest.fixture
def performance_metrics():
    """Create performance metrics collector."""
    return PerformanceMetrics()


# ============================================================================
# Data Generators
# ============================================================================


@pytest.fixture
def generate_pii_content():
    """Generate content with random PII."""
    pii_templates = [
        "Email: {email}, Phone: {phone}",
        "Customer {name} has SSN {ssn}",
        "Credit card ending in {card_last4}: {card}",
        "Contact: {name} at {email}, {phone}",
        "User profile - Name: {name}, DOB: {dob}, Address: {address}",
    ]

    def _generate(count: int = 1) -> List[str]:
        results = []
        for i in range(count):
            template = random.choice(pii_templates)
            content = template.format(
                email=f"user{i}@company{random.randint(1, 100)}.com",
                phone=f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                name=f"User {i} Name",
                ssn=f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
                card=f"{random.randint(4000, 4999)}{''.join(str(random.randint(0, 9)) for _ in range(12))}",
                card_last4=f"{random.randint(1000, 9999)}",
                dob=f"19{random.randint(50, 99)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                address=f"{random.randint(100, 9999)} Main St, City {random.randint(1, 100)}",
            )
            results.append(content)
        return results

    return _generate


@pytest.fixture
def generate_bulk_tokens():
    """Generate bulk token requests."""
    pii_types = ["ssn", "email", "phone", "credit_card"]

    def _generate(count: int = 1000) -> List[Dict[str, Any]]:
        return [
            {
                "value": f"value-{i}-{uuid4().hex[:8]}",
                "pii_type": random.choice(pii_types),
                "tenant_id": f"tenant-{i % 10}",
            }
            for i in range(count)
        ]

    return _generate


@pytest.fixture
def generate_stream_messages():
    """Generate simulated stream messages."""

    def _generate(count: int = 10000) -> List[Dict[str, Any]]:
        return [
            {
                "topic": "raw-events",
                "value": {
                    "event_id": str(uuid4()),
                    "user_email": f"user{i}@company.com",
                    "data": f"Some data {i}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "headers": {
                    "X-Tenant-ID": f"tenant-{i % 10}",
                },
            }
            for i in range(count)
        ]

    return _generate


# ============================================================================
# Concurrent Execution Helpers
# ============================================================================


@pytest.fixture
def run_concurrent():
    """Run async operations concurrently with metrics."""

    async def _run(
        operation: Callable,
        items: List[Any],
        max_concurrent: int = 100,
        metrics: Optional[PerformanceMetrics] = None,
    ) -> List[Any]:
        """Execute operations concurrently.

        Args:
            operation: Async function to call for each item.
            items: Items to process.
            max_concurrent: Maximum concurrent operations.
            metrics: Optional metrics collector.

        Returns:
            List of results.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def limited_operation(item):
            async with semaphore:
                start = time.perf_counter()
                try:
                    result = await operation(item)
                    if metrics:
                        latency_ms = (time.perf_counter() - start) * 1000
                        metrics.record_latency(latency_ms)
                    return result
                except Exception as e:
                    return {"error": str(e)}

        if metrics:
            metrics.start()

        results = await asyncio.gather(*[limited_operation(item) for item in items])

        if metrics:
            metrics.stop()

        return results

    return _run


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest markers for load tests."""
    config.addinivalue_line(
        "markers", "load: mark test as a load test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (may take minutes)"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to load tests."""
    for item in items:
        if "load" in item.nodeid:
            item.add_marker(pytest.mark.load)
