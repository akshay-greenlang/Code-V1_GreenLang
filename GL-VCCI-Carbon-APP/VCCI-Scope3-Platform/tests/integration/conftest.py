# -*- coding: utf-8 -*-
"""
===============================================================================
GL-VCCI Scope 3 Platform - Integration Test Fixtures
===============================================================================

Comprehensive test infrastructure for E2E integration testing.
Provides fixtures for database, cache, external services, and test data.

Test Coverage:
- 30+ E2E scenarios
- Multi-agent workflows
- ERP integrations
- Error handling
- Performance testing
- Data flow validation

Version: 2.0.0
Created: 2025-11-09
===============================================================================
"""

import asyncio
import json
import os
import pytest
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

import pandas as pd
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock

# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_db_engine():
    """Create test database engine."""
    # Use in-memory SQLite for testing
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        pool_pre_ping=True
    )

    # Create all tables
    from database import Base  # Import your models
    try:
        Base.metadata.create_all(engine)
    except:
        pass  # Tables might not exist in Base

    yield engine

    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_db_engine):
    """Provide a clean database session for each test."""
    SessionLocal = sessionmaker(bind=test_db_engine)
    session = SessionLocal()

    yield session

    session.rollback()
    session.close()


@pytest.fixture(scope="function")
def test_db_connection(db_session):
    """Provide database connection with transaction."""
    connection = db_session.connection()

    yield connection

    connection.close()


# ============================================================================
# Cache Fixtures (Redis Mock)
# ============================================================================

@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis cache for testing."""
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    mock_cache.exists.return_value = False
    mock_cache.keys.return_value = []
    mock_cache.flushdb.return_value = True

    return mock_cache


@pytest.fixture(scope="function")
def redis_cache(mock_redis):
    """Provide Redis cache interface."""
    return mock_redis


# ============================================================================
# Test Data Factories
# ============================================================================

class SupplierFactory:
    """Factory for creating test supplier data."""

    @staticmethod
    def create_supplier(
        supplier_id: Optional[str] = None,
        name: Optional[str] = None,
        category: int = 1,
        spend: float = 100000.0,
        tier: int = 1
    ) -> Dict[str, Any]:
        """Create a test supplier."""
        return {
            "supplier_id": supplier_id or str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            "name": name or f"Test Supplier {deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}",
            "category": category,
            "spend_amount": spend,
            "spend_currency": "USD",
            "tier": tier,
            "country": "United States",
            "industry": "Manufacturing",
            "naics_code": "334111",
            "created_at": DeterministicClock.utcnow().isoformat(),
            "updated_at": DeterministicClock.utcnow().isoformat()
        }

    @staticmethod
    def create_batch(count: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Create multiple test suppliers."""
        return [
            SupplierFactory.create_supplier(**kwargs)
            for _ in range(count)
        ]


class EmissionDataFactory:
    """Factory for creating test emission data."""

    @staticmethod
    def create_emission_factor(
        category: int = 1,
        subcategory: Optional[str] = None,
        factor: float = 0.5,
        unit: str = "kg CO2e/USD"
    ) -> Dict[str, Any]:
        """Create test emission factor."""
        return {
            "category": category,
            "subcategory": subcategory or f"Category {category}",
            "emission_factor": factor,
            "unit": unit,
            "source": "EPA",
            "year": 2024,
            "geography": "US",
            "quality_tier": "Tier 1"
        }

    @staticmethod
    def create_calculation_result(
        supplier_id: str,
        category: int = 1,
        emissions: float = 50.0
    ) -> Dict[str, Any]:
        """Create test calculation result."""
        return {
            "supplier_id": supplier_id,
            "category": category,
            "total_emissions": emissions,
            "emissions_unit": "kg CO2e",
            "uncertainty": 0.15,
            "tier": 1,
            "calculated_at": DeterministicClock.utcnow().isoformat(),
            "method": "spend-based"
        }


class FileDataFactory:
    """Factory for creating test data files."""

    @staticmethod
    def create_csv_file(
        suppliers: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """Create CSV file from supplier data."""
        df = pd.DataFrame(suppliers)

        with tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.csv'
        ) as f:
            df.to_csv(f, index=False)
            return f.name

    @staticmethod
    def create_excel_file(
        suppliers: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """Create Excel file from supplier data."""
        df = pd.DataFrame(suppliers)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.xlsx'
        ) as f:
            df.to_excel(f.name, index=False)
            return f.name

    @staticmethod
    def create_json_file(
        data: Any,
        filename: Optional[str] = None
    ) -> str:
        """Create JSON file from data."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.json'
        ) as f:
            json.dump(data, f, indent=2)
            return f.name


@pytest.fixture(scope="function")
def supplier_factory():
    """Provide supplier factory."""
    return SupplierFactory()


@pytest.fixture(scope="function")
def emission_data_factory():
    """Provide emission data factory."""
    return EmissionDataFactory()


@pytest.fixture(scope="function")
def file_data_factory():
    """Provide file data factory."""
    return FileDataFactory()


# ============================================================================
# Sample Test Data
# ============================================================================

@pytest.fixture(scope="function")
def sample_suppliers(supplier_factory):
    """Provide sample supplier data."""
    return supplier_factory.create_batch(count=10)


@pytest.fixture(scope="function")
def sample_large_suppliers(supplier_factory):
    """Provide large batch of suppliers for performance testing."""
    return supplier_factory.create_batch(count=1000)


@pytest.fixture(scope="function")
def sample_emission_factors(emission_data_factory):
    """Provide sample emission factors."""
    return [
        emission_data_factory.create_emission_factor(category=i)
        for i in range(1, 16)
    ]


# ============================================================================
# External Service Mocks
# ============================================================================

@pytest.fixture(scope="function")
def mock_sap_connector():
    """Mock SAP connector."""
    mock = AsyncMock()
    mock.connect.return_value = True
    mock.disconnect.return_value = True
    mock.fetch_suppliers.return_value = {
        "suppliers": SupplierFactory.create_batch(5),
        "total": 5,
        "status": "success"
    }
    return mock


@pytest.fixture(scope="function")
def mock_oracle_connector():
    """Mock Oracle connector."""
    mock = AsyncMock()
    mock.connect.return_value = True
    mock.disconnect.return_value = True
    mock.fetch_suppliers.return_value = {
        "suppliers": SupplierFactory.create_batch(5),
        "total": 5,
        "status": "success"
    }
    return mock


@pytest.fixture(scope="function")
def mock_workday_connector():
    """Mock Workday connector."""
    mock = AsyncMock()
    mock.connect.return_value = True
    mock.disconnect.return_value = True
    mock.fetch_suppliers.return_value = {
        "suppliers": SupplierFactory.create_batch(5),
        "total": 5,
        "status": "success"
    }
    return mock


@pytest.fixture(scope="function")
def mock_llm_provider():
    """Mock LLM provider (if used)."""
    mock = AsyncMock()
    mock.complete.return_value = {
        "response": "Mocked LLM response",
        "tokens_used": 100,
        "model": "mock-model"
    }
    return mock


# ============================================================================
# Agent Mocks
# ============================================================================

@pytest.fixture(scope="function")
def mock_intake_agent():
    """Mock Intake Agent."""
    mock = AsyncMock()
    mock.process.return_value = {
        "status": "success",
        "suppliers_processed": 10,
        "validation_errors": []
    }
    return mock


@pytest.fixture(scope="function")
def mock_calculator_agent():
    """Mock Calculator Agent."""
    mock = AsyncMock()
    mock.calculate.return_value = {
        "status": "success",
        "calculations": [
            EmissionDataFactory.create_calculation_result(
                supplier_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                category=1,
                emissions=50.0
            )
            for _ in range(10)
        ]
    }
    return mock


@pytest.fixture(scope="function")
def mock_hotspot_agent():
    """Mock Hotspot Agent."""
    mock = AsyncMock()
    mock.analyze.return_value = {
        "status": "success",
        "hotspots": [
            {
                "supplier_id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                "emissions": 1000.0,
                "rank": 1,
                "percentage": 15.0
            }
        ]
    }
    return mock


@pytest.fixture(scope="function")
def mock_engagement_agent():
    """Mock Engagement Agent."""
    mock = AsyncMock()
    mock.engage.return_value = {
        "status": "success",
        "campaigns_created": 5,
        "suppliers_contacted": 25
    }
    return mock


@pytest.fixture(scope="function")
def mock_reporting_agent():
    """Mock Reporting Agent."""
    mock = AsyncMock()
    mock.generate_report.return_value = {
        "status": "success",
        "report_id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "format": "pdf",
        "file_path": "/tmp/report.pdf"
    }
    return mock


# ============================================================================
# Circuit Breaker Mock
# ============================================================================

@pytest.fixture(scope="function")
def mock_circuit_breaker():
    """Mock circuit breaker."""
    mock = MagicMock()
    mock.call.return_value = {"status": "success"}
    mock.state = "closed"
    mock.failure_count = 0
    return mock


# ============================================================================
# Performance Monitoring
# ============================================================================

@pytest.fixture(scope="function")
def performance_monitor():
    """Performance monitoring fixture."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.start_time = None

        def start(self, metric_name: str):
            """Start timing a metric."""
            self.start_time = DeterministicClock.utcnow()

        def stop(self, metric_name: str):
            """Stop timing and record metric."""
            if self.start_time:
                duration = (DeterministicClock.utcnow() - self.start_time).total_seconds()
                self.metrics[metric_name] = duration
                self.start_time = None

        def get_metrics(self) -> Dict[str, float]:
            """Get all recorded metrics."""
            return self.metrics

    return PerformanceMonitor()


# ============================================================================
# Authentication Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_auth_token():
    """Mock JWT authentication token."""
    return {
        "access_token": "mock_access_token_" + deterministic_uuid(__name__, str(DeterministicClock.now())).hex,
        "refresh_token": "mock_refresh_token_" + deterministic_uuid(__name__, str(DeterministicClock.now())).hex,
        "token_type": "Bearer",
        "expires_in": 3600,
        "user_id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "tenant_id": str(deterministic_uuid(__name__, str(DeterministicClock.now())))
    }


@pytest.fixture(scope="function")
def mock_user():
    """Mock authenticated user."""
    return {
        "user_id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "username": "test_user",
        "email": "test@example.com",
        "tenant_id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "roles": ["user", "analyst"],
        "permissions": ["read", "write", "calculate"]
    }


# ============================================================================
# Multi-tenant Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_tenant():
    """Mock tenant data."""
    return {
        "tenant_id": str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        "name": "Test Tenant",
        "plan": "enterprise",
        "features": ["calculations", "reporting", "integrations"],
        "limits": {
            "max_suppliers": 10000,
            "max_calculations_per_month": 100000
        }
    }


# ============================================================================
# Event Loop Fixture for Async Tests
# ============================================================================

@pytest.fixture(scope="function")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Cleanup Helpers
# ============================================================================

@pytest.fixture(scope="function", autouse=True)
def cleanup_temp_files():
    """Cleanup temporary files after each test."""
    temp_files = []

    yield temp_files

    # Cleanup
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "database": {
            "url": "sqlite:///:memory:",
            "pool_size": 5
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "api": {
            "base_url": "http://localhost:8000",
            "timeout": 30
        },
        "performance": {
            "max_response_time": 5.0,
            "max_calculation_time": 10.0
        }
    }


# ============================================================================
# Markers
# ============================================================================

# Add custom markers
pytest.mark.e2e_happy_path = pytest.mark.e2e_happy_path
pytest.mark.e2e_error = pytest.mark.e2e_error
pytest.mark.e2e_performance = pytest.mark.e2e_performance
pytest.mark.e2e_dataflow = pytest.mark.e2e_dataflow
