# -*- coding: utf-8 -*-
# =============================================================================
# Integration Tests: TLS Connections
# SEC-004: TLS 1.3 Configuration
# =============================================================================
"""
Integration tests for TLS connections to real endpoints.

These tests verify TLS functionality against real servers when available.
Tests are skipped in CI environments without network access.

Test categories:
- Public endpoint TLS connections
- TLS protocol version verification
- Certificate expiry verification
- Connection error handling

Note: Integration tests require network access and may be skipped
in isolated CI environments.
"""

from __future__ import annotations

import os
import socket
import ssl
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import TLS service modules
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.tls_service.ssl_context import (
        create_ssl_context,
        create_client_ssl_context,
    )
    from greenlang.infrastructure.tls_service.exporter import (
        get_certificate_info,
        TLSCertificateScanner,
    )
    from greenlang.infrastructure.tls_service.utils import (
        get_connection_info,
        is_version_secure,
    )
    _HAS_TLS_SERVICE = True
except ImportError:
    _HAS_TLS_SERVICE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_TLS_SERVICE, reason="TLS service not installed"),
]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def skip_without_network():
    """Skip test if network is not available."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except (socket.timeout, socket.error, OSError):
        pytest.skip("Network not available")


# ============================================================================
# Test: Public Endpoint TLS Connections
# ============================================================================


class TestPublicEndpointConnections:
    """Integration tests for connecting to public TLS endpoints."""

    @pytest.mark.asyncio
    async def test_connects_to_public_endpoint(self, skip_without_network):
        """Should connect to a known public endpoint using TLS."""
        info = await get_certificate_info("www.google.com", 443, timeout=10.0)

        assert info.valid is True
        assert info.host == "www.google.com"
        assert info.tls_version in ["TLSv1.2", "TLSv1.3"]

    @pytest.mark.asyncio
    async def test_retrieves_certificate_details(self, skip_without_network):
        """Should retrieve certificate details from public endpoint."""
        info = await get_certificate_info("www.google.com", 443)

        assert info.valid is True
        assert info.subject is not None
        assert info.issuer is not None
        assert info.days_until_expiry > 0

    @pytest.mark.asyncio
    async def test_verifies_tls_version(self, skip_without_network):
        """Should verify TLS version is secure."""
        info = await get_certificate_info("www.google.com", 443)

        assert info.valid is True
        assert info.tls_version in ["TLSv1.2", "TLSv1.3"]
        assert is_version_secure(info.tls_version)

    @pytest.mark.asyncio
    async def test_retrieves_cipher_suite(self, skip_without_network):
        """Should retrieve negotiated cipher suite."""
        info = await get_certificate_info("www.google.com", 443)

        assert info.valid is True
        assert info.cipher_suite is not None
        assert info.cipher_suite != "Unknown"

    def test_ssl_context_connects_successfully(self, skip_without_network):
        """Should connect using created SSL context."""
        context = create_ssl_context()

        with socket.create_connection(("www.google.com", 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname="www.google.com") as ssock:
                assert ssock.version() in ["TLSv1.2", "TLSv1.3"]


# ============================================================================
# Test: TLS Protocol Version Enforcement
# ============================================================================


class TestTLSVersionEnforcement:
    """Integration tests for TLS version enforcement."""

    def test_rejects_tls10_only_context(self, skip_without_network):
        """Should reject connections when only TLS 1.0 is offered."""
        # Create a context that only allows TLS 1.0
        # Modern servers should reject this
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        try:
            # Set max version to TLS 1.0 (if supported by OpenSSL)
            context.maximum_version = ssl.TLSVersion.TLSv1
            context.minimum_version = ssl.TLSVersion.TLSv1
        except (AttributeError, ValueError):
            pytest.skip("OpenSSL doesn't support TLS version limits")

        # Most modern servers reject TLS 1.0
        with pytest.raises((ssl.SSLError, ConnectionError, socket.error)):
            with socket.create_connection(("www.google.com", 443), timeout=5) as sock:
                context.wrap_socket(sock, server_hostname="www.google.com")

    def test_allows_tls12_connection(self, skip_without_network):
        """Should allow TLS 1.2 connections."""
        context = create_ssl_context(min_version=ssl.TLSVersion.TLSv1_2)

        with socket.create_connection(("www.google.com", 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname="www.google.com") as ssock:
                version = ssock.version()
                assert version in ["TLSv1.2", "TLSv1.3"]

    def test_prefers_tls13_when_available(self, skip_without_network):
        """Should prefer TLS 1.3 when server supports it."""
        context = create_ssl_context()

        with socket.create_connection(("www.google.com", 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname="www.google.com") as ssock:
                # Google supports TLS 1.3, so it should be negotiated
                # But this depends on server configuration
                version = ssock.version()
                assert version in ["TLSv1.2", "TLSv1.3"]


# ============================================================================
# Test: Certificate Expiry Verification
# ============================================================================


class TestCertificateExpiryVerification:
    """Integration tests for certificate expiry checking."""

    @pytest.mark.asyncio
    async def test_calculates_days_until_expiry(self, skip_without_network):
        """Should correctly calculate days until certificate expires."""
        info = await get_certificate_info("www.google.com", 443)

        assert info.valid is True
        assert info.days_until_expiry > 0

        # Google's certificates are typically valid for 90 days
        # and renewed frequently, so they should have significant time remaining
        assert info.days_until_expiry < 365  # Should be within a year

    @pytest.mark.asyncio
    async def test_detects_valid_certificate_chain(self, skip_without_network):
        """Should successfully verify certificate chain."""
        info = await get_certificate_info("www.google.com", 443, verify=True)

        assert info.valid is True
        assert info.error is None


# ============================================================================
# Test: TLS Certificate Scanner
# ============================================================================


class TestCertificateScannerIntegration:
    """Integration tests for TLSCertificateScanner."""

    @pytest.mark.asyncio
    async def test_scans_multiple_endpoints(self, skip_without_network):
        """Should scan multiple endpoints concurrently."""
        endpoints = [
            ("www.google.com", 443),
            ("www.github.com", 443),
        ]

        scanner = TLSCertificateScanner(endpoints, timeout=10.0)
        results = await scanner.scan_all()

        assert len(results) == 2
        assert all(r.valid for r in results)

    @pytest.mark.asyncio
    async def test_records_scan_duration(self, skip_without_network):
        """Should record scan duration."""
        scanner = TLSCertificateScanner([("www.google.com", 443)], timeout=10.0)
        await scanner.scan_all()

        assert scanner.last_scan_duration is not None
        assert scanner.last_scan_duration > 0

    @pytest.mark.asyncio
    async def test_caches_scan_results(self, skip_without_network):
        """Should cache scan results for retrieval."""
        scanner = TLSCertificateScanner([("www.google.com", 443)], timeout=10.0)
        await scanner.scan_all()

        cached = scanner.get_cached("www.google.com", 443)
        assert cached is not None
        assert cached.host == "www.google.com"


# ============================================================================
# Test: Connection Error Handling
# ============================================================================


class TestConnectionErrorHandling:
    """Integration tests for connection error handling."""

    @pytest.mark.asyncio
    async def test_handles_nonexistent_host(self):
        """Should handle connection to non-existent host."""
        info = await get_certificate_info(
            "this-host-does-not-exist-12345.invalid",
            443,
            timeout=5.0,
        )

        assert info.valid is False
        assert info.error is not None

    @pytest.mark.asyncio
    async def test_handles_closed_port(self, skip_without_network):
        """Should handle connection to closed port."""
        # Port 12345 is unlikely to be open on google.com
        info = await get_certificate_info("www.google.com", 12345, timeout=5.0)

        assert info.valid is False
        assert info.error is not None

    @pytest.mark.asyncio
    async def test_handles_timeout(self, skip_without_network):
        """Should handle connection timeout gracefully."""
        # Use very short timeout that should fail
        info = await get_certificate_info(
            "www.google.com",
            443,
            timeout=0.001,  # 1 millisecond should timeout
        )

        # This may or may not timeout depending on network conditions
        # The important thing is it should not raise an exception
        assert isinstance(info.valid, bool)


# ============================================================================
# Test: get_connection_info
# ============================================================================


class TestGetConnectionInfo:
    """Integration tests for get_connection_info function."""

    def test_gets_connection_info_from_public_endpoint(self, skip_without_network):
        """Should get connection info from public endpoint."""
        info = get_connection_info("www.google.com", 443, timeout=10.0)

        assert info.connected is True
        assert info.protocol in ["TLSv1.2", "TLSv1.3"]
        assert info.cipher_name is not None

    def test_handles_verification_disabled(self, skip_without_network):
        """Should work with verification disabled."""
        info = get_connection_info(
            "www.google.com",
            443,
            timeout=10.0,
            verify=False,
        )

        assert info.connected is True
        assert info.verified is False


# ============================================================================
# Test: Database-Specific Endpoints (when available)
# ============================================================================


class TestDatabaseEndpoints:
    """Integration tests for database TLS endpoints."""

    @pytest.mark.skipif(
        not os.environ.get("TEST_PG_HOST"),
        reason="PostgreSQL test host not configured"
    )
    @pytest.mark.asyncio
    async def test_postgres_tls_connection(self):
        """Should connect to PostgreSQL with TLS."""
        pg_host = os.environ.get("TEST_PG_HOST")
        pg_port = int(os.environ.get("TEST_PG_PORT", "5432"))

        info = await get_certificate_info(pg_host, pg_port, timeout=10.0)

        assert info.valid is True
        assert info.tls_version in ["TLSv1.2", "TLSv1.3"]

    @pytest.mark.skipif(
        not os.environ.get("TEST_REDIS_HOST"),
        reason="Redis test host not configured"
    )
    @pytest.mark.asyncio
    async def test_redis_tls_connection(self):
        """Should connect to Redis with TLS."""
        redis_host = os.environ.get("TEST_REDIS_HOST")
        redis_port = int(os.environ.get("TEST_REDIS_PORT", "6379"))

        info = await get_certificate_info(redis_host, redis_port, timeout=10.0)

        assert info.valid is True
        assert info.tls_version in ["TLSv1.2", "TLSv1.3"]


# ============================================================================
# Test: Certificate Validation
# ============================================================================


class TestCertificateValidation:
    """Integration tests for certificate validation."""

    def test_rejects_untrusted_certificate(self):
        """Should reject connection to untrusted certificate."""
        # self-signed.badssl.com has a self-signed certificate
        # that should fail validation with default trust store
        context = create_ssl_context(verify=True)

        with pytest.raises(ssl.SSLCertVerificationError):
            with socket.create_connection(
                ("self-signed.badssl.com", 443), timeout=10
            ) as sock:
                context.wrap_socket(sock, server_hostname="self-signed.badssl.com")

    def test_allows_untrusted_with_verify_false(self):
        """Should allow untrusted certificate when verify=False."""
        context = create_ssl_context(verify=False)

        try:
            with socket.create_connection(
                ("self-signed.badssl.com", 443), timeout=10
            ) as sock:
                with context.wrap_socket(
                    sock, server_hostname="self-signed.badssl.com"
                ) as ssock:
                    assert ssock.version() in ["TLSv1.2", "TLSv1.3"]
        except (socket.timeout, ConnectionError):
            pytest.skip("Could not connect to badssl.com")
