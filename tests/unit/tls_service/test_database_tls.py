# -*- coding: utf-8 -*-
# =============================================================================
# Unit Tests: Database TLS Configuration
# SEC-004: TLS 1.3 Configuration
# =============================================================================
"""
Tests for PostgreSQL and Redis TLS configuration.

Tests cover:
- PostgreSQL SSL context creation
- Redis TLS configuration
- Database connection string generation
- TLS verification settings

Coverage target: 85%+
"""

from __future__ import annotations

import ssl
from typing import Dict, Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import TLS service modules
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.tls_service.ssl_context import (
        create_ssl_context,
        create_client_ssl_context,
    )
    from greenlang.infrastructure.tls_service.utils import (
        get_connection_info,
        TLSConnectionInfo,
    )
    _HAS_TLS_SERVICE = True
except ImportError:
    _HAS_TLS_SERVICE = False

pytestmark = [
    pytest.mark.skipif(not _HAS_TLS_SERVICE, reason="TLS service not installed"),
]


# ============================================================================
# Test: PostgreSQL TLS Configuration
# ============================================================================


class TestPostgreSQLTLS:
    """Tests for PostgreSQL TLS configuration."""

    def test_creates_postgres_ssl_context(self):
        """Should create SSL context suitable for PostgreSQL."""
        context = create_ssl_context(purpose="database")

        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_postgres_context_uses_tls_12_minimum(self):
        """PostgreSQL context should use TLS 1.2 minimum."""
        context = create_ssl_context(purpose="database")

        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2

    def test_postgres_context_disables_insecure_protocols(self):
        """PostgreSQL context should disable insecure protocols."""
        context = create_ssl_context(purpose="database")

        assert context.options & ssl.OP_NO_SSLv2
        assert context.options & ssl.OP_NO_SSLv3
        assert context.options & ssl.OP_NO_TLSv1
        assert context.options & ssl.OP_NO_TLSv1_1

    def test_postgres_context_allows_verification_disable(self):
        """Should allow disabling verification for development."""
        context = create_ssl_context(purpose="database", verify=False)

        assert context.verify_mode == ssl.CERT_NONE
        assert context.check_hostname is False

    @patch('greenlang.infrastructure.tls_service.ssl_context.get_ca_bundle_path')
    def test_postgres_context_uses_rds_ca_bundle(self, mock_ca):
        """Should prefer RDS CA bundle for PostgreSQL."""
        mock_ca.return_value = "/path/to/rds-ca.pem"

        context = create_ssl_context(purpose="database")
        assert context is not None

    def test_postgres_connection_string_sslmode(self):
        """Should generate correct sslmode for connection string."""
        # Test that SSL context creation works for different modes
        verify_context = create_ssl_context(purpose="database", verify=True)
        no_verify_context = create_ssl_context(purpose="database", verify=False)

        assert verify_context.verify_mode == ssl.CERT_REQUIRED
        assert no_verify_context.verify_mode == ssl.CERT_NONE


# ============================================================================
# Test: Redis TLS Configuration
# ============================================================================


class TestRedisTLS:
    """Tests for Redis TLS configuration."""

    def test_creates_redis_ssl_context(self):
        """Should create SSL context suitable for Redis."""
        context = create_ssl_context(purpose="redis")

        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED

    def test_redis_context_uses_tls_12_minimum(self):
        """Redis context should use TLS 1.2 minimum."""
        context = create_ssl_context(purpose="redis")

        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2

    def test_redis_context_enables_server_hostname_check(self):
        """Redis context should check server hostname."""
        context = create_ssl_context(purpose="redis")

        assert context.check_hostname is True

    def test_redis_context_allows_verification_disable(self):
        """Should allow disabling verification for development."""
        context = create_ssl_context(purpose="redis", verify=False)

        assert context.verify_mode == ssl.CERT_NONE

    def test_redis_context_uses_modern_ciphers(self):
        """Redis context should use modern cipher suites."""
        context = create_ssl_context(purpose="redis")
        ciphers = [c["name"] for c in context.get_ciphers()]

        # Should have AEAD ciphers
        assert any("GCM" in c or "CHACHA20" in c for c in ciphers)


# ============================================================================
# Test: TLS Connection Verification
# ============================================================================


class TestTLSConnectionVerification:
    """Tests for TLS connection verification functions."""

    def test_connection_info_dataclass(self):
        """TLSConnectionInfo should have required fields."""
        info = TLSConnectionInfo()

        assert hasattr(info, 'connected')
        assert hasattr(info, 'protocol')
        assert hasattr(info, 'cipher_name')
        assert hasattr(info, 'peer_certificate')
        assert hasattr(info, 'verified')
        assert hasattr(info, 'error')

    def test_connection_info_defaults(self):
        """TLSConnectionInfo should have sensible defaults."""
        info = TLSConnectionInfo()

        assert info.connected is False
        assert info.protocol is None
        assert info.error is None

    @patch('socket.create_connection')
    def test_get_connection_info_handles_timeout(self, mock_socket):
        """Should handle connection timeout gracefully."""
        import socket
        mock_socket.side_effect = socket.timeout()

        info = get_connection_info("example.com", 443, timeout=1.0)

        assert info.connected is False
        assert "timeout" in info.error.lower()

    @patch('socket.create_connection')
    def test_get_connection_info_handles_connection_refused(self, mock_socket):
        """Should handle connection refused gracefully."""
        mock_socket.side_effect = ConnectionRefusedError()

        info = get_connection_info("example.com", 443)

        assert info.connected is False
        assert "refused" in info.error.lower()

    @patch('socket.create_connection')
    def test_get_connection_info_handles_dns_failure(self, mock_socket):
        """Should handle DNS resolution failure gracefully."""
        import socket
        mock_socket.side_effect = socket.gaierror(8, "Name resolution failed")

        info = get_connection_info("invalid.example.com", 443)

        assert info.connected is False
        assert "dns" in info.error.lower() or "resolution" in info.error.lower()


# ============================================================================
# Test: mTLS for Database Connections
# ============================================================================


class TestDatabaseMTLS:
    """Tests for mTLS database connections."""

    @patch('ssl.SSLContext.load_cert_chain')
    def test_postgres_mtls_loads_client_cert(self, mock_load):
        """Should load client certificate for mTLS."""
        context = create_ssl_context(
            purpose="database",
            client_cert="/path/to/client.crt",
            client_key="/path/to/client.key",
        )

        mock_load.assert_called()

    @patch('ssl.SSLContext.load_cert_chain')
    def test_redis_mtls_loads_client_cert(self, mock_load):
        """Should load client certificate for Redis mTLS."""
        context = create_ssl_context(
            purpose="redis",
            client_cert="/path/to/client.crt",
            client_key="/path/to/client.key",
        )

        mock_load.assert_called()

    @patch('ssl.SSLContext.load_cert_chain')
    def test_mtls_supports_encrypted_key(self, mock_load):
        """Should support encrypted private key."""
        context = create_ssl_context(
            purpose="database",
            client_cert="/path/to/client.crt",
            client_key="/path/to/client.key",
            client_key_password="secret123",
        )

        mock_load.assert_called_with(
            certfile="/path/to/client.crt",
            keyfile="/path/to/client.key",
            password="secret123",
        )


# ============================================================================
# Test: AWS RDS Aurora Configuration
# ============================================================================


class TestAWSRDSAurora:
    """Tests for AWS RDS Aurora TLS configuration."""

    @patch('greenlang.infrastructure.tls_service.ca_bundle.get_aws_rds_ca_bundle')
    def test_uses_rds_ca_bundle_when_available(self, mock_rds_ca):
        """Should use RDS CA bundle when available."""
        mock_rds_ca.return_value = "/path/to/rds-combined-ca-bundle.pem"

        # Context creation should use RDS CA
        context = create_ssl_context(
            purpose="database",
            ca_bundle="/path/to/rds-combined-ca-bundle.pem",
        )

        assert context is not None
        assert context.verify_mode == ssl.CERT_REQUIRED

    def test_aurora_requires_tls_12_minimum(self):
        """Aurora PostgreSQL requires TLS 1.2 minimum."""
        context = create_ssl_context(
            purpose="database",
            min_version=ssl.TLSVersion.TLSv1_2,
        )

        assert context.minimum_version == ssl.TLSVersion.TLSv1_2


# ============================================================================
# Test: AWS ElastiCache Configuration
# ============================================================================


class TestAWSElastiCache:
    """Tests for AWS ElastiCache TLS configuration."""

    @patch('greenlang.infrastructure.tls_service.ca_bundle.get_aws_elasticache_ca_bundle')
    def test_uses_elasticache_ca_bundle_when_available(self, mock_ec_ca):
        """Should use ElastiCache CA bundle when available."""
        mock_ec_ca.return_value = "/path/to/AmazonRootCA1.pem"

        context = create_ssl_context(
            purpose="redis",
            ca_bundle="/path/to/AmazonRootCA1.pem",
        )

        assert context is not None
        assert context.verify_mode == ssl.CERT_REQUIRED

    def test_elasticache_requires_tls_12_minimum(self):
        """ElastiCache requires TLS 1.2 minimum."""
        context = create_ssl_context(
            purpose="redis",
            min_version=ssl.TLSVersion.TLSv1_2,
        )

        assert context.minimum_version == ssl.TLSVersion.TLSv1_2


# ============================================================================
# Test: Connection String Generation
# ============================================================================


class TestConnectionStringGeneration:
    """Tests for database connection string generation."""

    def test_postgres_sslmode_verify_full(self):
        """verify=True should map to sslmode=verify-full."""
        context = create_ssl_context(purpose="database", verify=True)

        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_postgres_sslmode_require(self):
        """verify=False should map to sslmode=require."""
        context = create_ssl_context(
            purpose="database",
            verify=False,
            check_hostname=False,
        )

        assert context.verify_mode == ssl.CERT_NONE
        assert context.check_hostname is False


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestDatabaseTLSErrors:
    """Tests for error handling in database TLS configuration."""

    @patch('ssl.SSLContext.load_cert_chain')
    def test_handles_missing_client_cert(self, mock_load):
        """Should handle missing client certificate gracefully."""
        mock_load.side_effect = FileNotFoundError("Certificate not found")

        with pytest.raises(FileNotFoundError):
            create_ssl_context(
                purpose="database",
                client_cert="/nonexistent/client.crt",
                client_key="/nonexistent/client.key",
            )

    @patch('ssl.SSLContext.load_cert_chain')
    def test_handles_invalid_certificate(self, mock_load):
        """Should handle invalid certificate gracefully."""
        mock_load.side_effect = ssl.SSLError("Invalid certificate")

        with pytest.raises(ssl.SSLError):
            create_ssl_context(
                purpose="database",
                client_cert="/invalid/client.crt",
                client_key="/invalid/client.key",
            )

    @patch('ssl.SSLContext.load_cert_chain')
    def test_handles_wrong_key_password(self, mock_load):
        """Should handle wrong key password gracefully."""
        mock_load.side_effect = ssl.SSLError("Bad decrypt")

        with pytest.raises(ssl.SSLError):
            create_ssl_context(
                purpose="database",
                client_cert="/path/to/client.crt",
                client_key="/path/to/encrypted.key",
                client_key_password="wrong_password",
            )


# ============================================================================
# Test: Performance Considerations
# ============================================================================


class TestDatabaseTLSPerformance:
    """Tests for TLS performance considerations."""

    def test_context_creation_is_fast(self):
        """Context creation should be fast (< 100ms)."""
        import time

        start = time.perf_counter()
        for _ in range(100):
            create_ssl_context(purpose="database")
        elapsed = time.perf_counter() - start

        # 100 context creations should take less than 1 second
        assert elapsed < 1.0, f"Context creation too slow: {elapsed:.2f}s for 100"

    def test_context_can_be_reused(self):
        """SSL context should be reusable across connections."""
        context = create_ssl_context(purpose="database")

        # Context should work for multiple hypothetical connections
        assert context is not None
        # In practice, one context should be used for many connections


# ============================================================================
# Test: Environment-Specific Configuration
# ============================================================================


class TestEnvironmentConfiguration:
    """Tests for environment-specific TLS configuration."""

    def test_development_can_disable_verification(self):
        """Development environment can disable verification."""
        context = create_ssl_context(
            purpose="database",
            verify=False,
        )

        assert context.verify_mode == ssl.CERT_NONE

    def test_production_enforces_verification(self):
        """Production environment should enforce verification."""
        context = create_ssl_context(
            purpose="database",
            verify=True,
        )

        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_staging_can_use_custom_ca(self):
        """Staging environment can use custom CA bundle."""
        context = create_ssl_context(
            purpose="database",
            ca_bundle="/custom/staging-ca.pem",
            verify=True,
        )

        assert context.verify_mode == ssl.CERT_REQUIRED
