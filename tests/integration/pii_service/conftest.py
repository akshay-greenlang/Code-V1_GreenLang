# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for PII Service integration tests.

Provides fixtures that connect to real test infrastructure:
- PostgreSQL test database
- Redis test cluster
- Test encryption service

Author: GreenLang Test Engineering Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional
from uuid import uuid4

import pytest
import pytest_asyncio


# ============================================================================
# Environment Configuration
# ============================================================================


def get_test_db_url() -> str:
    """Get test database URL from environment."""
    return os.environ.get(
        "GL_TEST_DATABASE_URL",
        "postgresql://test:test@localhost:5432/greenlang_test"
    )


def get_test_redis_url() -> str:
    """Get test Redis URL from environment."""
    return os.environ.get(
        "GL_TEST_REDIS_URL",
        "redis://localhost:6379/1"
    )


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def db_pool():
    """Create async database pool for integration tests."""
    try:
        import psycopg_pool
        from psycopg.rows import dict_row

        pool = psycopg_pool.AsyncConnectionPool(
            get_test_db_url(),
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
        )

        await pool.open()
        yield pool
        await pool.close()
    except ImportError:
        pytest.skip("psycopg not available for integration tests")


@pytest_asyncio.fixture
async def clean_db(db_pool):
    """Clean database tables before each test."""
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            # Clean PII service tables
            tables = [
                "pii_service.token_vault",
                "pii_service.allowlist",
                "pii_service.quarantine",
                "pii_service.remediation_log",
                "pii_service.audit_log",
            ]
            for table in tables:
                try:
                    await cur.execute(f"TRUNCATE {table} CASCADE")
                except Exception:
                    pass  # Table may not exist
        await conn.commit()

    yield

    # Cleanup after test
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            for table in tables:
                try:
                    await cur.execute(f"TRUNCATE {table} CASCADE")
                except Exception:
                    pass
        await conn.commit()


# ============================================================================
# Redis Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def redis_client():
    """Create Redis client for integration tests."""
    try:
        import redis.asyncio as redis

        client = redis.from_url(get_test_redis_url())
        yield client
        await client.close()
    except ImportError:
        pytest.skip("redis not available for integration tests")


@pytest_asyncio.fixture
async def clean_redis(redis_client):
    """Clean Redis keys before each test."""
    # Clean PII service keys
    keys = await redis_client.keys("gl:pii:*")
    if keys:
        await redis_client.delete(*keys)

    yield

    # Cleanup after test
    keys = await redis_client.keys("gl:pii:*")
    if keys:
        await redis_client.delete(*keys)


# ============================================================================
# Service Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def pii_service(db_pool, redis_client, clean_db, clean_redis):
    """Create PIIService with real infrastructure."""
    try:
        from greenlang.infrastructure.pii_service.service import PIIService
        from greenlang.infrastructure.pii_service.config import (
            PIIServiceConfig,
            PersistenceBackend,
        )

        config = PIIServiceConfig.for_environment("test")
        config.vault.persistence_backend = PersistenceBackend.POSTGRESQL

        service = PIIService(
            config=config,
            db_pool=db_pool,
            redis_client=redis_client,
        )

        yield service
    except ImportError:
        pytest.skip("PIIService not available")


@pytest_asyncio.fixture
async def secure_vault(db_pool, redis_client, clean_db, clean_redis):
    """Create SecureTokenVault with real infrastructure."""
    try:
        from greenlang.infrastructure.pii_service.secure_vault import SecureTokenVault
        from greenlang.infrastructure.pii_service.config import (
            VaultConfig,
            PersistenceBackend,
        )

        config = VaultConfig(
            persistence_backend=PersistenceBackend.POSTGRESQL,
            token_ttl_days=1,  # Short TTL for tests
            max_tokens_per_tenant=1000,
        )

        vault = SecureTokenVault(
            config=config,
            db_pool=db_pool,
            redis_client=redis_client,
        )

        yield vault
    except ImportError:
        pytest.skip("SecureTokenVault not available")


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def test_tenant_id():
    """Generate unique tenant ID for test isolation."""
    return f"test-tenant-{uuid4().hex[:8]}"


@pytest.fixture
def test_user_id():
    """Generate unique user ID for tests."""
    return str(uuid4())


@pytest.fixture
def sample_pii_data():
    """Sample PII data for testing."""
    return {
        "ssn": "123-45-6789",
        "email": "john.doe@company.com",
        "phone": "+1-555-123-4567",
        "credit_card": "4111111111111111",
        "name": "John Doe",
    }


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest markers for integration tests."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "requires_redis: mark test as requiring Redis"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to integration tests."""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
