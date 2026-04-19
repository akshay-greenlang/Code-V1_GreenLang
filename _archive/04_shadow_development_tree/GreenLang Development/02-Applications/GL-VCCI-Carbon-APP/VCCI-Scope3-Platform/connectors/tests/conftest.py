# -*- coding: utf-8 -*-
"""
Shared Fixtures for Integration and Performance Tests
GL-VCCI Scope 3 Platform

Provides shared pytest fixtures for ERP connector integration and performance testing,
including sandbox connections, database fixtures, and performance monitoring utilities.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import os
import time
import pytest
import redis
import psutil
from typing import Dict, Any, Generator, Optional
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Import connector clients
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sap.client import SAPODataClient
from sap.config import SAPConnectorConfig, SAPEnvironment
from oracle.client import OracleRESTClient
from oracle.config import OracleConnectorConfig, OracleEnvironment

# Performance monitoring
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Track performance metrics during tests."""

    requests_made: int = 0
    total_records: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    errors: int = 0
    api_latencies: list = field(default_factory=list)
    memory_samples: list = field(default_factory=list)

    def record_request(self, latency_ms: float, records_count: int = 0):
        """Record a single request."""
        self.requests_made += 1
        self.total_records += records_count
        self.api_latencies.append(latency_ms)

    def record_error(self):
        """Record an error."""
        self.errors += 1

    def record_memory(self):
        """Record current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)

    def finalize(self):
        """Finalize metrics."""
        self.end_time = time.time()

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def throughput_per_second(self) -> float:
        """Calculate records per second."""
        duration = self.duration_seconds
        return self.total_records / duration if duration > 0 else 0

    @property
    def throughput_per_hour(self) -> float:
        """Calculate records per hour."""
        return self.throughput_per_second * 3600

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        return sum(self.api_latencies) / len(self.api_latencies) if self.api_latencies else 0

    @property
    def p95_latency_ms(self) -> float:
        """Calculate 95th percentile latency."""
        if not self.api_latencies:
            return 0
        sorted_latencies = sorted(self.api_latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def p99_latency_ms(self) -> float:
        """Calculate 99th percentile latency."""
        if not self.api_latencies:
            return 0
        sorted_latencies = sorted(self.api_latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]

    @property
    def max_memory_mb(self) -> float:
        """Get maximum memory usage."""
        return max(self.memory_samples) if self.memory_samples else 0


# ==================== Sandbox Availability Markers ====================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring sandbox environment"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and throughput tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests (>60 seconds)"
    )
    config.addinivalue_line(
        "markers", "sap_sandbox: Requires SAP sandbox"
    )
    config.addinivalue_line(
        "markers", "oracle_sandbox: Requires Oracle sandbox"
    )
    config.addinivalue_line(
        "markers", "workday_sandbox: Requires Workday sandbox"
    )


def is_sap_sandbox_available() -> bool:
    """Check if SAP sandbox is available."""
    return all([
        os.getenv("SAP_SANDBOX_URL"),
        os.getenv("SAP_SANDBOX_CLIENT_ID"),
        os.getenv("SAP_SANDBOX_CLIENT_SECRET"),
        os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
    ])


def is_oracle_sandbox_available() -> bool:
    """Check if Oracle sandbox is available."""
    return all([
        os.getenv("ORACLE_SANDBOX_URL"),
        os.getenv("ORACLE_SANDBOX_CLIENT_ID"),
        os.getenv("ORACLE_SANDBOX_CLIENT_SECRET"),
        os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
    ])


def is_workday_sandbox_available() -> bool:
    """Check if Workday sandbox is available."""
    return all([
        os.getenv("WORKDAY_SANDBOX_URL"),
        os.getenv("WORKDAY_SANDBOX_CLIENT_ID"),
        os.getenv("WORKDAY_SANDBOX_CLIENT_SECRET"),
        os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"
    ])


# ==================== SAP Fixtures ====================

@pytest.fixture(scope="session")
def sap_sandbox_config() -> Optional[SAPConnectorConfig]:
    """
    Create SAP connector config for sandbox environment.

    Returns:
        SAPConnectorConfig if sandbox is available, None otherwise
    """
    if not is_sap_sandbox_available():
        pytest.skip("SAP sandbox not available")

    return SAPConnectorConfig(
        environment=SAPEnvironment.SANDBOX,
        base_url=os.getenv("SAP_SANDBOX_URL"),
        oauth={
            "client_id": os.getenv("SAP_SANDBOX_CLIENT_ID"),
            "client_secret": os.getenv("SAP_SANDBOX_CLIENT_SECRET"),
            "token_url": os.getenv("SAP_SANDBOX_TOKEN_URL"),
            "scope": os.getenv("SAP_SANDBOX_OAUTH_SCOPE", "API_BUSINESS_PARTNER")
        }
    )


@pytest.fixture(scope="function")
def sap_client(sap_sandbox_config) -> Generator[SAPODataClient, None, None]:
    """
    Create SAP OData client for testing.

    Yields:
        SAPODataClient instance
    """
    client = SAPODataClient(sap_sandbox_config)
    yield client
    client.close()


@pytest.fixture(scope="function")
def mock_sap_client() -> Generator[SAPODataClient, None, None]:
    """
    Create mock SAP client for CI/CD testing.

    Yields:
        Mocked SAPODataClient instance
    """
    with patch("sap.client.requests.Session") as mock_session:
        # Configure mock responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "value": [
                {"PurchaseOrder": "PO001", "Supplier": "S001", "TotalAmount": 1000.00}
            ]
        }
        mock_session.return_value.request.return_value = mock_response

        # Create client with test config
        config = SAPConnectorConfig(
            environment=SAPEnvironment.SANDBOX,
            base_url="https://mock.sap.com",
            oauth={
                "client_id": "test_client",
                "client_secret": "test_secret",
                "token_url": "https://mock.sap.com/oauth/token"
            }
        )

        client = SAPODataClient(config)
        yield client
        client.close()


# ==================== Oracle Fixtures ====================

@pytest.fixture(scope="session")
def oracle_sandbox_config() -> Optional[OracleConnectorConfig]:
    """
    Create Oracle connector config for sandbox environment.

    Returns:
        OracleConnectorConfig if sandbox is available, None otherwise
    """
    if not is_oracle_sandbox_available():
        pytest.skip("Oracle sandbox not available")

    return OracleConnectorConfig(
        environment=OracleEnvironment.SANDBOX,
        base_url=os.getenv("ORACLE_SANDBOX_URL"),
        oauth={
            "client_id": os.getenv("ORACLE_SANDBOX_CLIENT_ID"),
            "client_secret": os.getenv("ORACLE_SANDBOX_CLIENT_SECRET"),
            "token_url": os.getenv("ORACLE_SANDBOX_TOKEN_URL")
        }
    )


@pytest.fixture(scope="function")
def oracle_client(oracle_sandbox_config) -> Generator[OracleRESTClient, None, None]:
    """
    Create Oracle REST client for testing.

    Yields:
        OracleRESTClient instance
    """
    client = OracleRESTClient(oracle_sandbox_config)
    yield client
    client.close()


@pytest.fixture(scope="function")
def mock_oracle_client() -> Generator[OracleRESTClient, None, None]:
    """
    Create mock Oracle client for CI/CD testing.

    Yields:
        Mocked OracleRESTClient instance
    """
    with patch("oracle.client.requests.Session") as mock_session:
        # Configure mock responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {"OrderNumber": "PO001", "Supplier": "S001", "TotalAmount": 1000.00}
            ],
            "count": 1,
            "hasMore": False,
            "links": []
        }
        mock_session.return_value.request.return_value = mock_response

        # Create client with test config
        config = OracleConnectorConfig(
            environment=OracleEnvironment.SANDBOX,
            base_url="https://mock.oracle.com",
            oauth={
                "client_id": "test_client",
                "client_secret": "test_secret",
                "token_url": "https://mock.oracle.com/oauth/token"
            }
        )

        client = OracleRESTClient(config)
        yield client
        client.close()


# ==================== Workday Fixtures ====================

@pytest.fixture(scope="function")
def mock_workday_client():
    """
    Create mock Workday client for CI/CD testing.

    Yields:
        Mocked Workday client instance
    """
    # Mock Workday RaaS client
    mock_client = MagicMock()
    mock_client.extract_expense_reports.return_value = [
        {
            "expense_id": "EXP001",
            "employee_id": "E001",
            "total_amount": 500.00,
            "expense_date": "2024-01-15"
        }
    ]
    yield mock_client


# ==================== Database Fixtures ====================

@pytest.fixture(scope="function")
def test_db_connection():
    """
    Create test database connection.

    Yields:
        Database connection for test data cleanup
    """
    # For now, return mock
    yield MagicMock()


# ==================== Redis Fixtures ====================

@pytest.fixture(scope="function")
def redis_client():
    """
    Create Redis client for cache testing.

    Yields:
        Redis client instance
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis.from_url(redis_url)
        client.ping()
        yield client
        # Cleanup test keys
        test_keys = client.keys("test:*")
        if test_keys:
            client.delete(*test_keys)
    except redis.ConnectionError:
        pytest.skip("Redis not available")


# ==================== Performance Monitoring Fixtures ====================

@pytest.fixture(scope="function")
def performance_metrics() -> Generator[PerformanceMetrics, None, None]:
    """
    Create performance metrics tracker.

    Yields:
        PerformanceMetrics instance
    """
    metrics = PerformanceMetrics()
    yield metrics
    metrics.finalize()


@pytest.fixture(scope="function")
def memory_monitor():
    """
    Monitor memory usage during test.

    Yields:
        Function to get current memory usage
    """
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        current = process.memory_info().rss / 1024 / 1024
        return {
            "current_mb": current,
            "initial_mb": initial_memory,
            "delta_mb": current - initial_memory
        }

    yield get_memory_usage
