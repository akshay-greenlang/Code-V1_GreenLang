# -*- coding: utf-8 -*-
"""
Performance Test Configuration for GL-006 HEATRECLAIM.

This module provides shared fixtures and configuration for performance testing.
"""

import pytest
import sys
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def pytest_configure(config):
    """Configure pytest markers for performance tests."""
    config.addinivalue_line(
        "markers", "performance: mark test as a performance benchmark"
    )


@pytest.fixture(scope="session")
def performance_thresholds():
    """Define performance thresholds for GL-006 tests."""
    return {
        "pinch_analysis_ms": 100.0,  # <100ms for 20 streams
        "exergy_calculation_ms": 50.0,  # <50ms per stream
        "hen_optimization_ms": 500.0,  # <500ms for 10 exchangers
        "roi_calculation_ms": 10.0,  # <10ms
        "full_pipeline_ms": 2000.0,  # <2s for standard problems
        "hash_calculation_ms": 1.0,  # <1ms
        "min_throughput_per_sec": 1000,  # Minimum throughput
    }


@pytest.fixture(scope="session")
def large_dataset_config():
    """Configuration for large dataset scalability tests."""
    return {
        "max_streams": 100,
        "max_exchangers": 50,
        "max_temperature_intervals": 200,
        "test_iterations": 1000,
    }
