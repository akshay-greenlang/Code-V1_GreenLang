# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for Secrets Service unit tests.

Provides common fixtures for:
- Mock VaultClient
- Mock Redis client
- SecretsServiceConfig
- Test data generators
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# ============================================================================
# VaultClient Fixtures
# ============================================================================


@pytest.fixture
def mock_vault_secret():
    """Factory for creating mock VaultSecret objects."""
    def _create(
        data: Optional[Dict[str, Any]] = None,
        version: int = 1,
        created_time: Optional[str] = None,
    ):
        mock = MagicMock()
        mock.data = data or {"key": "value"}
        mock.metadata = {
            "version": version,
            "created_time": created_time or datetime.now(timezone.utc).isoformat(),
            "deletion_time": "",
            "destroyed": False,
        }
        mock.lease_id = None
        mock.lease_duration = 0
        mock.renewable = False
        mock.get = lambda k, d=None: mock.data.get(k, d)
        return mock
    return _create


@pytest.fixture
def mock_vault_client(mock_vault_secret):
    """Create a fully mocked VaultClient."""
    client = AsyncMock()

    # Basic operations
    client.get_secret = AsyncMock(return_value=mock_vault_secret())
    client.put_secret = AsyncMock(return_value={"version": 1})
    client.delete_secret = AsyncMock()

    # Credential operations
    db_creds = MagicMock()
    db_creds.username = "db_user_test"
    db_creds.password = "db_pass_test"
    db_creds.host = "localhost"
    db_creds.port = 5432
    db_creds.database = "greenlang"
    db_creds.ssl_mode = "require"
    db_creds.lease_id = "database/creds/test/abc123"
    db_creds.lease_duration = 3600
    db_creds.connection_string = "postgresql://db_user_test:db_pass_test@localhost:5432/greenlang"
    client.get_database_credentials = AsyncMock(return_value=db_creds)

    # Certificate operations
    cert = MagicMock()
    cert.certificate = "-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----"
    cert.private_key = "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----"
    cert.ca_chain = ["-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----"]
    cert.serial_number = "12:34:56:78:90"
    cert.expiration = datetime.now(timezone.utc) + timedelta(days=30)
    cert.is_expired = False
    client.generate_certificate = AsyncMock(return_value=cert)

    # Lease operations
    client.renew_lease = AsyncMock(return_value={"lease_id": "renewed", "lease_duration": 3600})
    client.revoke_lease = AsyncMock()

    # Health operations
    client.health_check = AsyncMock(return_value={
        "initialized": True,
        "sealed": False,
        "standby": False,
        "version": "1.15.0",
    })
    client.is_healthy = AsyncMock(return_value=True)

    # Connection operations
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.authenticate = AsyncMock()

    # Internal request method
    client._request = AsyncMock(return_value={"data": {"keys": ["secret1", "secret2"]}})

    return client


# ============================================================================
# Redis Client Fixtures
# ============================================================================


@pytest.fixture
def mock_redis_client():
    """Create a mocked Redis client with cache behavior."""
    cache: Dict[str, str] = {}

    client = AsyncMock()

    async def mock_get(key: str):
        return cache.get(key)

    async def mock_set(key: str, value: str, ex: Optional[int] = None):
        cache[key] = value

    async def mock_delete(*keys: str):
        for key in keys:
            cache.pop(key, None)

    async def mock_keys(pattern: str):
        import fnmatch
        return [k for k in cache.keys() if fnmatch.fnmatch(k, pattern)]

    client.get = mock_get
    client.set = mock_set
    client.delete = mock_delete
    client.keys = mock_keys
    client.publish = AsyncMock()
    client.close = AsyncMock()

    # Store reference to cache for testing
    client._test_cache = cache

    return client


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def secrets_service_config():
    """Create SecretsServiceConfig for testing."""
    try:
        from greenlang.infrastructure.secrets_service.config import SecretsServiceConfig
        return SecretsServiceConfig(
            vault_addr="https://vault.test:8200",
            auth_method="token",
            vault_token="test-token",
            cache_enabled=True,
            cache_ttl_seconds=300,
            redis_cache_enabled=True,
            memory_cache_enabled=True,
            tenant_path_prefix="secret/data/tenants",
            platform_path_prefix="secret/data/greenlang",
            rotation_enabled=True,
            audit_enabled=True,
            metrics_enabled=True,
        )
    except ImportError:
        # Return a simple object with the same attributes
        class ConfigStub:
            pass

        config = ConfigStub()
        config.vault_addr = "https://vault.test:8200"
        config.auth_method = "token"
        config.vault_token = "test-token"
        config.cache_enabled = True
        config.cache_ttl_seconds = 300
        config.redis_cache_enabled = True
        config.memory_cache_enabled = True
        config.tenant_path_prefix = "secret/data/tenants"
        config.platform_path_prefix = "secret/data/greenlang"
        config.rotation_enabled = True
        config.audit_enabled = True
        config.metrics_enabled = True
        return config


# ============================================================================
# Test Data Generators
# ============================================================================


@pytest.fixture
def generate_secret_path():
    """Generate unique secret paths for test isolation."""
    def _generate(prefix: str = "test") -> str:
        return f"{prefix}/{uuid.uuid4().hex[:8]}"
    return _generate


@pytest.fixture
def generate_tenant_id():
    """Generate unique tenant IDs for test isolation."""
    def _generate(prefix: str = "t") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8]}"
    return _generate


@pytest.fixture
def sample_secret_data():
    """Sample secret data for testing."""
    return {
        "username": "admin",
        "password": "secret123!@#",
        "host": "db.greenlang.svc",
        "port": 5432,
        "database": "greenlang",
    }


@pytest.fixture
def sample_api_key_data():
    """Sample API key data for testing."""
    return {
        "api_key": f"sk_test_{uuid.uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
    }


# ============================================================================
# Authentication Fixtures
# ============================================================================


@pytest.fixture
def auth_headers():
    """Generate authentication headers for API tests."""
    def _generate(
        user_id: str = "user-1",
        tenant_id: str = "t-acme",
        roles: Optional[list] = None,
    ) -> Dict[str, str]:
        roles = roles or ["secrets:read", "secrets:write"]
        return {
            "Authorization": "Bearer test-jwt-token",
            "X-Tenant-ID": tenant_id,
            "X-User-ID": user_id,
            "X-Roles": ",".join(roles),
        }
    return _generate


@pytest.fixture
def admin_auth_headers(auth_headers):
    """Generate admin authentication headers."""
    return auth_headers(
        user_id="admin-1",
        tenant_id="t-platform",
        roles=["secrets:admin", "rotation:admin", "secrets:read", "secrets:write"],
    )


# ============================================================================
# Mock Exception Classes
# ============================================================================


@pytest.fixture
def vault_exceptions():
    """Provide Vault exception classes for testing."""
    try:
        from greenlang.execution.infrastructure.secrets import (
            VaultError,
            VaultSecretNotFoundError,
            VaultPermissionError,
            VaultConnectionError,
        )
        return {
            "VaultError": VaultError,
            "VaultSecretNotFoundError": VaultSecretNotFoundError,
            "VaultPermissionError": VaultPermissionError,
            "VaultConnectionError": VaultConnectionError,
        }
    except ImportError:
        class VaultError(Exception):
            pass

        class VaultSecretNotFoundError(VaultError):
            pass

        class VaultPermissionError(VaultError):
            pass

        class VaultConnectionError(VaultError):
            pass

        return {
            "VaultError": VaultError,
            "VaultSecretNotFoundError": VaultSecretNotFoundError,
            "VaultPermissionError": VaultPermissionError,
            "VaultConnectionError": VaultConnectionError,
        }


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as a load test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )


@pytest.fixture(autouse=True)
def reset_context():
    """Reset any context variables between tests."""
    yield
    # Cleanup after each test
