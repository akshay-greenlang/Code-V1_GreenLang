# -*- coding: utf-8 -*-
"""
Pytest Configuration for GreenLang Agent Testing
Provides fixtures, hooks, and configuration for comprehensive testing.
"""

import pytest
import asyncio
import logging
import sys
import os
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from datetime import datetime
import numpy as np
from greenlang.determinism import deterministic_random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Pytest Configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "security: Security and vulnerability tests"
    )
    config.addinivalue_line(
        "markers", "compliance: Regulatory compliance tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take >1 second"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: Tests requiring GPU acceleration"
    )
    config.addinivalue_line(
        "markers", "requires_llm: Tests requiring LLM API access"
    )


# Test Session Fixtures
@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration for entire session."""
    return {
        "coverage_target": 0.90,  # 90% coverage target
        "performance_targets": {
            "agent_creation_ms": 100,
            "message_passing_ms": 10,
            "memory_retrieval_ms": 50,
            "llm_call_avg_ms": 2000,
            "llm_call_p99_ms": 5000
        },
        "quality_targets": {
            "functional": 0.9,
            "performance": 0.85,
            "reliability": 0.95,
            "security": 0.9
        },
        "test_data_seed": 42,
        "async_timeout": 30,
        "max_memory_mb": 4096,
        "enable_mocking": True,
        "enable_determinism": True,
        "enable_provenance": True
    }


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory) -> Path:
    """Create temporary directory for test outputs."""
    return tmp_path_factory.mktemp("test_output")


# Module-level Fixtures
@pytest.fixture(scope="module")
def logger():
    """Provide logger for tests."""
    return logging.getLogger("test")


# Function-level Fixtures
@pytest.fixture
def mock_agent_config() -> Dict[str, Any]:
    """Provide mock agent configuration."""
    return {
        "name": f"test_agent_{np.deterministic_random().randint(1000, 9999)}",
        "version": "1.0.0",
        "environment": "test",
        "debug": True,
        "max_memory_mb": 1024,
        "timeout_s": 30,
        "retry_count": 3
    }


@pytest.fixture
def mock_llm_provider():
    """Provide mock LLM provider."""
    from testing.agent_test_framework import DeterministicLLMProvider
    return DeterministicLLMProvider(seed=42)


@pytest.fixture
def mock_vector_store():
    """Provide mock vector store."""
    from testing.agent_test_framework import AgentTestFixtures
    return AgentTestFixtures.create_vector_store_mock()


@pytest.fixture
def mock_rag_system():
    """Provide mock RAG system."""
    from testing.agent_test_framework import AgentTestFixtures
    return AgentTestFixtures.create_rag_system_mock()


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    from testing.agent_test_framework import TestDataGenerator
    return TestDataGenerator(seed=42)


@pytest.fixture
def quality_validator():
    """Provide quality validator."""
    from testing.quality_validators import ComprehensiveQualityValidator
    return ComprehensiveQualityValidator(target_score=0.8)


@pytest.fixture
def performance_runner():
    """Provide performance test runner."""
    from testing.agent_test_framework import PerformanceTestRunner
    return PerformanceTestRunner()


# Async Support
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Performance Monitoring
@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    start_time = time.perf_counter()
    start_memory = 0

    try:
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass

    yield

    duration = time.perf_counter() - start_time

    if start_memory > 0:
        try:
            import psutil
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = end_memory - start_memory
            logging.info(f"Test duration: {duration:.3f}s, Memory increase: {memory_increase:.2f}MB")
        except ImportError:
            logging.info(f"Test duration: {duration:.3f}s")
    else:
        logging.info(f"Test duration: {duration:.3f}s")


# Test Data Fixtures
@pytest.fixture
def sample_carbon_data(test_data_generator):
    """Provide sample carbon emissions data."""
    return test_data_generator.generate_carbon_data(count=10)


@pytest.fixture
def sample_agent_configs(test_data_generator):
    """Provide sample agent configurations."""
    return test_data_generator.generate_agent_configs(count=5)


@pytest.fixture
def sample_messages(test_data_generator):
    """Provide sample agent messages."""
    return test_data_generator.generate_test_messages(count=20)


@pytest.fixture
def sample_memory_entries(test_data_generator):
    """Provide sample memory entries."""
    return test_data_generator.generate_memory_entries(count=100)


# Cleanup Fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Cleanup code here if needed
    import gc
    gc.collect()


# Parameterized Test Data
def pytest_generate_tests(metafunc):
    """Generate parameterized test data."""

    # Parameterize agent states
    if "agent_state" in metafunc.fixturenames:
        from testing.agent_test_framework import AgentState
        metafunc.parametrize("agent_state", list(AgentState))

    # Parameterize quality dimensions
    if "quality_dimension" in metafunc.fixturenames:
        from testing.quality_validators import QualityDimension
        metafunc.parametrize("quality_dimension", list(QualityDimension))

    # Parameterize performance targets
    if "performance_target" in metafunc.fixturenames:
        metafunc.parametrize("performance_target", [
            ("agent_creation", 100),  # ms
            ("message_passing", 10),  # ms
            ("memory_retrieval", 50),  # ms
            ("llm_call_avg", 2000),  # ms
            ("llm_call_p99", 5000)  # ms
        ])


# Test Reporting Hooks
def pytest_runtest_makereport(item, call):
    """Customize test report generation."""
    if call.when == "call":
        # Add custom metrics to report if available
        if hasattr(item, "performance_metrics"):
            item.user_properties.append(("performance", item.performance_metrics))

        if hasattr(item, "quality_scores"):
            item.user_properties.append(("quality", item.quality_scores))


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test location
    for item in items:
        # Add unit marker for unit tests
        if "unit_tests" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker for integration tests
        elif "integration_tests" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add e2e marker for e2e tests
        elif "e2e_tests" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add performance marker for performance tests
        elif "performance_tests" in str(item.fspath):
            item.add_marker(pytest.mark.performance)

        # Mark slow tests
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# HTML Report Customization
def pytest_html_report_title(report):
    """Customize HTML report title."""
    report.title = "GreenLang Agent Foundation Test Report"


def pytest_configure_node(node):
    """Configure test node for distributed testing."""
    # Configuration for pytest-xdist if used
    node.workerinput["shared_data"] = {
        "test_start_time": DeterministicClock.now().isoformat()
    }


# Coverage Configuration
def pytest_coverage_html_report_title(config, report):
    """Customize coverage report title."""
    return "GreenLang Agent Foundation Coverage Report"


# Test Result Storage
@pytest.fixture(scope="session", autouse=True)
def test_results_collector():
    """Collect test results for final report."""
    results = {
        "start_time": DeterministicClock.now().isoformat(),
        "tests": [],
        "performance_metrics": [],
        "quality_scores": [],
        "coverage": {}
    }

    yield results

    # Save results to file
    results["end_time"] = DeterministicClock.now().isoformat()

    output_file = Path("test_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logging.info(f"Test results saved to {output_file}")


# Benchmark Fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "warmup_iterations": 10,
        "test_iterations": 100,
        "timeout_seconds": 60
    }


# Mock External Services
@pytest.fixture
def mock_external_services(monkeypatch):
    """Mock external service calls."""
    # Mock HTTP requests
    import requests

    def mock_get(*args, **kwargs):
        return type('MockResponse', (), {
            'status_code': 200,
            'json': lambda: {"status": "success"},
            'text': '{"status": "success"}'
        })()

    def mock_post(*args, **kwargs):
        return type('MockResponse', (), {
            'status_code': 201,
            'json': lambda: {"id": "created"},
            'text': '{"id": "created"}'
        })()

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)


# Database Fixtures
@pytest.fixture
def mock_database():
    """Provide mock database connection."""
    from testing.agent_test_framework import AgentTestFixtures
    return AgentTestFixtures.create_mock_database()


# Cache Fixtures
@pytest.fixture
def mock_cache():
    """Provide mock cache."""
    from testing.agent_test_framework import AgentTestFixtures
    return AgentTestFixtures.create_mock_cache()


# Emission Factor Fixtures
@pytest.fixture
def mock_emission_factors():
    """Provide mock emission factors."""
    from testing.agent_test_framework import AgentTestFixtures
    return AgentTestFixtures.create_emission_factors_mock()


# CBAM Test Data
@pytest.fixture
def cbam_test_data():
    """Provide CBAM test data."""
    from testing.agent_test_framework import AgentTestFixtures
    return AgentTestFixtures.create_cbam_test_data()


if __name__ == "__main__":
    print("Pytest configuration loaded successfully")
    print("Available fixtures:", [
        "test_config",
        "mock_agent_config",
        "mock_llm_provider",
        "mock_vector_store",
        "mock_rag_system",
        "test_data_generator",
        "quality_validator",
        "performance_runner"
    ])