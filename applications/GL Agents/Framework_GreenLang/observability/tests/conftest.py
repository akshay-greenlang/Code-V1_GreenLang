"""
GreenLang Observability - Test Configuration
=============================================

Pytest configuration and shared fixtures for observability tests.
"""

import pytest
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def reset_tracing():
    """Reset tracing manager between tests."""
    from observability.tracing import TracingManager

    # Store original instance
    original = TracingManager._instance

    yield

    # Reset to original
    TracingManager._instance = original
    TracingManager._current_span = None
    TracingManager._current_context = None


@pytest.fixture
def reset_metrics():
    """Reset metrics registry between tests."""
    from observability.metrics import MetricsRegistry

    # Store original instance
    original = MetricsRegistry._default_registry

    yield

    # Reset to original
    MetricsRegistry._default_registry = original


@pytest.fixture
def reset_logging():
    """Reset logging context between tests."""
    from observability.logging import clear_log_context, _correlation_id

    # Clear context
    clear_log_context()

    yield

    # Clear again after test
    clear_log_context()


@pytest.fixture
def health_manager():
    """Create a fresh health check manager for testing."""
    from observability.health import HealthCheckManager

    manager = HealthCheckManager(
        service_name="test-service",
        service_version="1.0.0",
    )

    return manager


@pytest.fixture
def tracing_manager():
    """Create a fresh tracing manager for testing."""
    from observability.tracing import TracingManager, ExporterType

    manager = TracingManager(
        service_name="test-service",
        exporter_type=ExporterType.CONSOLE,
    )

    return manager


@pytest.fixture
def metrics_registry():
    """Create a fresh metrics registry for testing."""
    from observability.metrics import MetricsRegistry

    registry = MetricsRegistry(namespace="test")

    return registry


@pytest.fixture
def structured_logger():
    """Create a fresh structured logger for testing."""
    from observability.logging import StructuredLogger, LogConfig, LogLevel

    config = LogConfig(
        level=LogLevel.DEBUG,
        format="json",
    )

    logger = StructuredLogger("test-logger", config)

    return logger
