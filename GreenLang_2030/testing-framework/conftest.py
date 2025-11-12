"""
Global pytest configuration and fixtures for GreenLang testing.

Provides shared fixtures, test configuration, and utilities for all test suites.
"""

import os
import sys
import json
import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, Generator, AsyncGenerator
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, MagicMock, AsyncMock
import logging

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from faker import Faker
import redis
from minio import Minio
import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang_core import AgentConfig, Pipeline
from greenlang_core.database import Base, get_db
from greenlang_core.cache import RedisCache
from greenlang_core.storage import MinioStorage
from greenlang_core.monitoring import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ============================================================================
# Session Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Load test configuration."""
    config = {
        "environment": "test",
        "database_url": os.getenv("GREENLANG_DB_URL", "postgresql://test:test@localhost:5432/greenlang_test"),
        "redis_url": os.getenv("GREENLANG_REDIS_URL", "redis://localhost:6379/0"),
        "minio_url": os.getenv("GREENLANG_MINIO_URL", "localhost:9000"),
        "minio_access_key": os.getenv("GREENLANG_MINIO_ACCESS", "minioadmin"),
        "minio_secret_key": os.getenv("GREENLANG_MINIO_SECRET", "minioadmin"),
        "random_seed": 42,
        "deterministic_mode": True
    }
    return config


@pytest.fixture(scope="session")
def faker():
    """Create Faker instance with fixed seed for deterministic data."""
    fake = Faker()
    Faker.seed(42)
    return fake


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def database_engine(test_config):
    """Create database engine for testing."""
    engine = create_engine(
        test_config["database_url"],
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    yield engine

    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(database_engine) -> Generator[Session, None, None]:
    """Create database session for tests."""
    SessionLocal = sessionmaker(bind=database_engine, autocommit=False, autoflush=False)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def clean_db(db_session):
    """Clean database before each test."""
    # Truncate all tables except migrations
    for table in reversed(Base.metadata.sorted_tables):
        if table.name != "alembic_version":
            db_session.execute(table.delete())
    db_session.commit()

    yield db_session


# ============================================================================
# Cache Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def redis_client(test_config):
    """Create Redis client for testing."""
    client = redis.Redis.from_url(
        test_config["redis_url"],
        decode_responses=True
    )

    yield client

    # Cleanup
    client.flushdb()
    client.close()


@pytest.fixture(scope="function")
def cache(redis_client) -> RedisCache:
    """Create cache instance for tests."""
    cache = RedisCache(redis_client)

    # Clear cache before test
    redis_client.flushdb()

    yield cache

    # Clear cache after test
    redis_client.flushdb()


# ============================================================================
# Storage Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def minio_client(test_config):
    """Create MinIO client for testing."""
    client = Minio(
        test_config["minio_url"],
        access_key=test_config["minio_access_key"],
        secret_key=test_config["minio_secret_key"],
        secure=False
    )

    # Create test bucket
    bucket_name = "greenlang-test"
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    yield client

    # Cleanup - remove all objects and bucket
    objects = client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        client.remove_object(bucket_name, obj.object_name)


@pytest.fixture(scope="function")
def storage(minio_client) -> MinioStorage:
    """Create storage instance for tests."""
    storage = MinioStorage(minio_client, bucket="greenlang-test")
    yield storage


# ============================================================================
# Agent Fixtures
# ============================================================================

@pytest.fixture
def agent_config() -> AgentConfig:
    """Create default agent configuration."""
    return AgentConfig(
        name="test_agent",
        version="1.0.0",
        environment="test",
        timeout_seconds=30,
        retry_count=3,
        cache_enabled=True,
        cache_ttl_seconds=3600,
        metrics_enabled=True,
        deterministic_mode=True,
        random_seed=42
    )


@pytest.fixture
def mock_emission_factors() -> Dict[tuple, float]:
    """Create mock emission factors database."""
    return {
        ("diesel", "US", "stationary_combustion"): 2.68,
        ("natural_gas", "US", "stationary_combustion"): 1.93,
        ("coal", "US", "stationary_combustion"): 3.45,
        ("electricity", "US", "grid"): 0.45,
        ("gasoline", "US", "mobile_combustion"): 2.35,
        ("jet_fuel", "US", "aviation"): 3.16,
        ("marine_fuel", "GLOBAL", "shipping"): 3.11,
        ("cement", "GLOBAL", "production"): 0.83,
        ("steel", "GLOBAL", "production"): 2.32,
        ("aluminum", "GLOBAL", "production"): 11.89
    }


@pytest.fixture
def mock_erp_client():
    """Create mock ERP client."""
    client = Mock()
    client.fetch_data = AsyncMock(return_value={
        "orders": [
            {"id": 1, "total": 1000.00, "status": "completed"},
            {"id": 2, "total": 2500.00, "status": "pending"}
        ],
        "inventory": [
            {"sku": "PROD-001", "quantity": 100, "location": "warehouse-1"},
            {"sku": "PROD-002", "quantity": 250, "location": "warehouse-2"}
        ]
    })
    return client


# ============================================================================
# Pipeline Fixtures
# ============================================================================

@pytest.fixture
def sample_pipeline(agent_config) -> Pipeline:
    """Create sample pipeline for testing."""
    pipeline = Pipeline(name="test_pipeline", config=agent_config)

    # Add sample agents
    from tests.fixtures.sample_agents import (
        DataIngestionAgent,
        ValidationAgent,
        CalculationAgent,
        ReportingAgent
    )

    pipeline.add_agent(DataIngestionAgent(agent_config))
    pipeline.add_agent(ValidationAgent(agent_config))
    pipeline.add_agent(CalculationAgent(agent_config))
    pipeline.add_agent(ReportingAgent(agent_config))

    return pipeline


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_shipment_data(faker) -> Dict[str, Any]:
    """Generate sample shipment data for CBAM testing."""
    return {
        "shipment_id": faker.uuid4(),
        "product_category": "cement",
        "weight_tonnes": Decimal("45.67"),
        "origin_country": "CN",
        "import_date": "2025-01-15",
        "supplier_name": faker.company(),
        "hs_code": "2523.10",
        "declared_emissions": Decimal("37.91"),
        "documents": ["invoice.pdf", "certificate.pdf"]
    }


@pytest.fixture
def sample_fuel_consumption_data() -> Dict[str, Any]:
    """Generate sample fuel consumption data."""
    return {
        "facility_id": "FAC-001",
        "reporting_period": "2025-Q1",
        "fuel_data": [
            {
                "fuel_type": "natural_gas",
                "quantity": Decimal("1000.0"),
                "unit": "m3",
                "combustion_type": "stationary"
            },
            {
                "fuel_type": "diesel",
                "quantity": Decimal("500.0"),
                "unit": "liters",
                "combustion_type": "mobile"
            }
        ],
        "location": "US",
        "facility_type": "manufacturing"
    }


@pytest.fixture
def sample_scope3_data() -> Dict[str, Any]:
    """Generate sample Scope 3 emissions data."""
    return {
        "company_id": "COMP-001",
        "reporting_year": 2025,
        "categories": {
            "purchased_goods": Decimal("1234.56"),
            "capital_goods": Decimal("567.89"),
            "fuel_energy": Decimal("234.56"),
            "transportation": Decimal("456.78"),
            "waste": Decimal("123.45"),
            "business_travel": Decimal("89.12"),
            "employee_commuting": Decimal("67.89"),
            "leased_assets": Decimal("345.67")
        },
        "verification_status": "pending",
        "data_quality_score": 0.85
    }


# ============================================================================
# HTTP/API Fixtures
# ============================================================================

@pytest.fixture
async def aiohttp_client():
    """Create aiohttp client for API testing."""
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture
def mock_api_responses():
    """Mock external API responses."""
    return {
        "emission_factors": {
            "status": 200,
            "data": {
                "factors": [
                    {"fuel": "diesel", "factor": 2.68, "unit": "kg CO2e/L"},
                    {"fuel": "natural_gas", "factor": 1.93, "unit": "kg CO2e/m3"}
                ]
            }
        },
        "weather_data": {
            "status": 200,
            "data": {
                "temperature": 22.5,
                "humidity": 65,
                "pressure": 1013.25
            }
        }
    }


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_benchmark():
    """Create performance benchmarking context."""
    class PerformanceBenchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.metrics = {}

        def start(self):
            self.start_time = datetime.now(timezone.utc)

        def stop(self):
            self.end_time = datetime.now(timezone.utc)
            self.metrics["duration_ms"] = (
                (self.end_time - self.start_time).total_seconds() * 1000
            )

        def assert_performance(self, max_duration_ms: float):
            assert self.metrics["duration_ms"] < max_duration_ms, \
                f"Performance target failed: {self.metrics['duration_ms']}ms > {max_duration_ms}ms"

    return PerformanceBenchmark()


# ============================================================================
# Determinism Testing Fixtures
# ============================================================================

@pytest.fixture
def determinism_validator():
    """Create determinism validation helper."""
    class DeterminismValidator:
        def __init__(self):
            self.runs = []

        def add_run(self, result: Any):
            self.runs.append(result)

        def validate(self, min_runs: int = 3) -> bool:
            """Validate that all runs produce identical results."""
            if len(self.runs) < min_runs:
                raise ValueError(f"Need at least {min_runs} runs, got {len(self.runs)}")

            # Compare all runs to first run
            first_run = json.dumps(self.runs[0], sort_keys=True, default=str)

            for i, run in enumerate(self.runs[1:], 1):
                current_run = json.dumps(run, sort_keys=True, default=str)
                if current_run != first_run:
                    logger.error(f"Determinism failed: Run {i+1} differs from Run 1")
                    return False

            return True

    return DeterminismValidator()


# ============================================================================
# Metrics/Monitoring Fixtures
# ============================================================================

@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Create metrics collector for testing."""
    collector = MetricsCollector(namespace="greenlang_test")
    yield collector
    collector.reset()


# ============================================================================
# Security Testing Fixtures
# ============================================================================

@pytest.fixture
def security_scanner():
    """Create security scanner for vulnerability testing."""
    class SecurityScanner:
        def scan_sql_injection(self, query: str) -> bool:
            """Check for SQL injection vulnerabilities."""
            dangerous_patterns = [
                "'; DROP TABLE",
                "1=1",
                "OR '1'='1'",
                "UNION SELECT",
                "--",
                "/*",
                "xp_cmdshell"
            ]
            return not any(pattern in query.upper() for pattern in dangerous_patterns)

        def scan_xss(self, content: str) -> bool:
            """Check for XSS vulnerabilities."""
            dangerous_patterns = [
                "<script>",
                "javascript:",
                "onerror=",
                "onload=",
                "<iframe"
            ]
            return not any(pattern in content.lower() for pattern in dangerous_patterns)

    return SecurityScanner()


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_db: marks tests that require database"
    )
    config.addinivalue_line(
        "markers", "requires_redis: marks tests that require Redis"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add markers based on test path
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset any singleton patterns used in the codebase
    yield
    # Cleanup code here