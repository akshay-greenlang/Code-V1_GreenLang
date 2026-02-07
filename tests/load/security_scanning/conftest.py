# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for Security Scanning load tests.

Provides fixtures for:
    - Large finding datasets
    - Performance measurement utilities
    - Memory monitoring
"""

from __future__ import annotations

import gc
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List

import pytest


# ============================================================================
# Data Generation Fixtures
# ============================================================================


@pytest.fixture
def large_findings_set() -> List[Dict[str, Any]]:
    """Generate a large set of findings for load testing."""
    findings = []
    for i in range(10000):
        findings.append({
            "id": str(uuid.uuid4()),
            "cve_id": f"CVE-2024-{i % 1000:04d}" if i % 3 == 0 else None,
            "title": f"Finding {i}",
            "severity": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"][i % 5],
            "scanner": ["bandit", "trivy", "gitleaks", "tfsec"][i % 4],
            "file_path": f"src/module{i % 100}/file{i % 10}.py",
            "line_number": (i % 1000) + 1,
            "rule_id": f"RULE-{i % 50:03d}",
            "description": f"Security issue description for finding {i}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    return findings


@pytest.fixture
def medium_codebase_structure() -> Dict[str, int]:
    """Create a simulated medium-sized codebase structure."""
    return {
        "total_files": 500,
        "python_files": 200,
        "javascript_files": 150,
        "terraform_files": 50,
        "docker_files": 20,
        "yaml_files": 80,
        "total_lines": 150000,
    }


@pytest.fixture
def huge_findings_set() -> Generator[Dict[str, Any], None, None]:
    """Generator for huge finding sets (memory efficient)."""
    for i in range(100000):
        yield {
            "id": str(uuid.uuid4()),
            "cve_id": f"CVE-2024-{i % 1000:04d}" if i % 3 == 0 else None,
            "severity": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"][i % 5],
            "scanner": ["bandit", "trivy", "gitleaks", "tfsec"][i % 4],
            "file_path": f"src/module{i % 100}/file{i % 10}.py",
            "line_number": (i % 1000) + 1,
            "rule_id": f"RULE-{i % 50:03d}",
        }


# ============================================================================
# Performance Measurement Fixtures
# ============================================================================


@pytest.fixture
def performance_timer():
    """Provide a performance timer context manager."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time

    return Timer


@pytest.fixture
def memory_tracker():
    """Provide memory tracking utilities."""
    import sys

    class MemoryTracker:
        def __init__(self):
            self.initial_objects = 0
            self.final_objects = 0

        def start(self):
            gc.collect()
            self.initial_objects = len(gc.get_objects())

        def stop(self):
            gc.collect()
            self.final_objects = len(gc.get_objects())

        @property
        def object_growth(self) -> int:
            return self.final_objects - self.initial_objects

        @staticmethod
        def get_size(obj: Any) -> int:
            return sys.getsizeof(obj)

    return MemoryTracker()


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "load: mark test as a load/performance test",
    )
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as a benchmark test",
    )
    config.addinivalue_line(
        "markers",
        "memory: mark test as a memory efficiency test",
    )
