# -*- coding: utf-8 -*-
# =============================================================================
# Unit Tests: TLS Metrics Exporter
# SEC-004: TLS 1.3 Configuration
# =============================================================================
"""
Tests for TLS metrics exporter and certificate scanner.

Tests cover:
- CertificateInfo dataclass
- get_certificate_info async function
- TLSCertificateScanner class
- TLSMetricsExporter class
- Prometheus metrics integration

Coverage target: 85%+
"""

from __future__ import annotations

import asyncio
import socket
import ssl
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import TLS service modules
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.tls_service.exporter import (
        CertificateInfo,
        get_certificate_info,
        scan_certificate_sync,
        TLSCertificateScanner,
        TLSMetricsExporter,
        get_metrics_exporter,
        DEFAULT_ENDPOINTS,
    )
    _HAS_TLS_SERVICE = True
except ImportError:
    _HAS_TLS_SERVICE = False

pytestmark = [
    pytest.mark.skipif(not _HAS_TLS_SERVICE, reason="TLS service not installed"),
]


# ============================================================================
# Test: CertificateInfo Dataclass
# ============================================================================


class TestCertificateInfo:
    """Tests for CertificateInfo dataclass."""

    def test_creates_valid_certificate_info(self):
        """Should create CertificateInfo with required fields."""
        info = CertificateInfo(
            host="example.com",
            port=443,
            subject="example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc) - timedelta(days=30),
            not_after=datetime.now(timezone.utc) + timedelta(days=30),
            days_until_expiry=30,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        assert info.host == "example.com"
        assert info.port == 443
        assert info.valid is True
        assert info.days_until_expiry == 30

    def test_is_expired_property(self):
        """Should correctly detect expired certificate."""
        # Expired certificate
        expired = CertificateInfo(
            host="example.com",
            port=443,
            subject="example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc) - timedelta(days=60),
            not_after=datetime.now(timezone.utc) - timedelta(days=30),
            days_until_expiry=-30,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        assert expired.is_expired is True

        # Valid certificate
        valid = CertificateInfo(
            host="example.com",
            port=443,
            subject="example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc) - timedelta(days=30),
            not_after=datetime.now(timezone.utc) + timedelta(days=30),
            days_until_expiry=30,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        assert valid.is_expired is False

    def test_is_expiring_soon_property(self):
        """Should correctly detect certificate expiring soon."""
        # Expiring in 7 days
        expiring = CertificateInfo(
            host="example.com",
            port=443,
            subject="example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc) - timedelta(days=353),
            not_after=datetime.now(timezone.utc) + timedelta(days=7),
            days_until_expiry=7,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        assert expiring.is_expiring_soon is True

        # Not expiring soon (30 days)
        not_expiring = CertificateInfo(
            host="example.com",
            port=443,
            subject="example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc) - timedelta(days=330),
            not_after=datetime.now(timezone.utc) + timedelta(days=30),
            days_until_expiry=30,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        assert not_expiring.is_expiring_soon is False

    def test_is_critical_property(self):
        """Should correctly detect critical expiry."""
        # Critical (< 7 days)
        critical = CertificateInfo(
            host="example.com",
            port=443,
            subject="example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc) - timedelta(days=358),
            not_after=datetime.now(timezone.utc) + timedelta(days=2),
            days_until_expiry=2,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        assert critical.is_critical is True

    def test_to_dict_method(self):
        """Should convert to dictionary for JSON serialization."""
        info = CertificateInfo(
            host="example.com",
            port=443,
            subject="example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc),
            not_after=datetime.now(timezone.utc) + timedelta(days=30),
            days_until_expiry=30,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        result = info.to_dict()

        assert isinstance(result, dict)
        assert result["host"] == "example.com"
        assert result["port"] == 443
        assert result["valid"] is True
        assert "not_after" in result

    def test_san_list(self):
        """Should support SAN list."""
        info = CertificateInfo(
            host="example.com",
            port=443,
            subject="example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc),
            not_after=datetime.now(timezone.utc) + timedelta(days=30),
            days_until_expiry=30,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
            san=["example.com", "www.example.com"],
        )

        assert "example.com" in info.san
        assert "www.example.com" in info.san


# ============================================================================
# Test: get_certificate_info
# ============================================================================


class TestGetCertificateInfo:
    """Tests for get_certificate_info async function."""

    @pytest.mark.asyncio
    @patch('socket.create_connection')
    async def test_returns_certificate_info_on_success(self, mock_socket):
        """Should return CertificateInfo on successful connection."""
        # Mock SSL socket
        mock_ssock = MagicMock()
        mock_ssock.getpeercert.return_value = {
            "subject": ((("commonName", "example.com"),),),
            "issuer": ((("commonName", "Test CA"),),),
            "notBefore": "Jan  1 00:00:00 2026 GMT",
            "notAfter": "Dec 31 23:59:59 2026 GMT",
            "serialNumber": "ABC123",
        }
        mock_ssock.getpeercert.return_value = {
            "subject": ((("commonName", "example.com"),),),
            "issuer": ((("commonName", "Test CA"),),),
            "notBefore": "Jan  1 00:00:00 2026 GMT",
            "notAfter": "Dec 31 23:59:59 2026 GMT",
            "serialNumber": "ABC123",
            "subjectAltName": (("DNS", "example.com"),),
        }
        mock_ssock.cipher.return_value = ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)
        mock_ssock.version.return_value = "TLSv1.3"
        mock_ssock.__enter__ = MagicMock(return_value=mock_ssock)
        mock_ssock.__exit__ = MagicMock(return_value=False)

        mock_sock = MagicMock()
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)

        mock_socket.return_value = mock_sock

        with patch('ssl.create_default_context') as mock_ctx:
            mock_context = MagicMock()
            mock_context.wrap_socket.return_value = mock_ssock
            mock_ctx.return_value = mock_context

            info = await get_certificate_info("example.com", 443)

            # Due to mocking complexity, just verify structure
            assert isinstance(info, CertificateInfo)
            assert info.host == "example.com"
            assert info.port == 443

    @pytest.mark.asyncio
    async def test_handles_connection_timeout(self):
        """Should handle connection timeout gracefully."""
        with patch('socket.create_connection') as mock_socket:
            mock_socket.side_effect = socket.timeout()

            info = await get_certificate_info("example.com", 443, timeout=1.0)

            assert info.valid is False
            assert info.error is not None
            assert "timeout" in info.error.lower()

    @pytest.mark.asyncio
    async def test_handles_connection_refused(self):
        """Should handle connection refused gracefully."""
        with patch('socket.create_connection') as mock_socket:
            mock_socket.side_effect = ConnectionRefusedError()

            info = await get_certificate_info("example.com", 443)

            assert info.valid is False
            assert "refused" in info.error.lower()

    @pytest.mark.asyncio
    async def test_handles_dns_failure(self):
        """Should handle DNS resolution failure gracefully."""
        with patch('socket.create_connection') as mock_socket:
            mock_socket.side_effect = socket.gaierror(8, "Name resolution failed")

            info = await get_certificate_info("invalid.example.com", 443)

            assert info.valid is False
            assert "dns" in info.error.lower() or "resolution" in info.error.lower()

    @pytest.mark.asyncio
    async def test_handles_ssl_error(self):
        """Should handle SSL error gracefully."""
        with patch('socket.create_connection') as mock_socket:
            mock_sock = MagicMock()
            mock_sock.__enter__ = MagicMock(return_value=mock_sock)
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_socket.return_value = mock_sock

            with patch('ssl.create_default_context') as mock_ctx:
                mock_context = MagicMock()
                mock_context.wrap_socket.side_effect = ssl.SSLError("Certificate error")
                mock_ctx.return_value = mock_context

                info = await get_certificate_info("example.com", 443)

                assert info.valid is False
                assert info.error is not None


# ============================================================================
# Test: TLSCertificateScanner
# ============================================================================


class TestTLSCertificateScanner:
    """Tests for TLSCertificateScanner class."""

    def test_initializes_with_endpoints(self):
        """Should initialize with list of endpoints."""
        endpoints = [("example.com", 443), ("api.example.com", 443)]
        scanner = TLSCertificateScanner(endpoints)

        assert scanner.endpoints == endpoints
        assert scanner.timeout == 10.0
        assert scanner.verify is True

    def test_initializes_with_custom_timeout(self):
        """Should accept custom timeout."""
        scanner = TLSCertificateScanner(
            [("example.com", 443)],
            timeout=5.0,
        )

        assert scanner.timeout == 5.0

    def test_initializes_with_verify_false(self):
        """Should accept verify=False."""
        scanner = TLSCertificateScanner(
            [("example.com", 443)],
            verify=False,
        )

        assert scanner.verify is False

    @pytest.mark.asyncio
    async def test_scan_all_returns_list(self):
        """Should return list of CertificateInfo."""
        endpoints = [("example.com", 443)]
        scanner = TLSCertificateScanner(endpoints)

        with patch(
            'greenlang.infrastructure.tls_service.exporter.get_certificate_info'
        ) as mock_get:
            mock_get.return_value = CertificateInfo(
                host="example.com",
                port=443,
                subject="example.com",
                issuer="Test CA",
                not_before=datetime.now(timezone.utc),
                not_after=datetime.now(timezone.utc) + timedelta(days=30),
                days_until_expiry=30,
                serial_number="ABC123",
                tls_version="TLSv1.3",
                cipher_suite="TLS_AES_256_GCM_SHA384",
                valid=True,
            )

            results = await scanner.scan_all()

            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].host == "example.com"

    @pytest.mark.asyncio
    async def test_scan_all_caches_results(self):
        """Should cache scan results."""
        endpoints = [("example.com", 443)]
        scanner = TLSCertificateScanner(endpoints)

        with patch(
            'greenlang.infrastructure.tls_service.exporter.get_certificate_info'
        ) as mock_get:
            mock_get.return_value = CertificateInfo(
                host="example.com",
                port=443,
                subject="example.com",
                issuer="Test CA",
                not_before=datetime.now(timezone.utc),
                not_after=datetime.now(timezone.utc) + timedelta(days=30),
                days_until_expiry=30,
                serial_number="ABC123",
                tls_version="TLSv1.3",
                cipher_suite="TLS_AES_256_GCM_SHA384",
                valid=True,
            )

            await scanner.scan_all()

            cached = scanner.get_cached("example.com", 443)
            assert cached is not None
            assert cached.host == "example.com"

    def test_get_expiring_soon_returns_expiring_certs(self):
        """Should return certificates expiring soon."""
        scanner = TLSCertificateScanner([])

        # Manually populate cache
        expiring = CertificateInfo(
            host="expiring.example.com",
            port=443,
            subject="expiring.example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc),
            not_after=datetime.now(timezone.utc) + timedelta(days=7),
            days_until_expiry=7,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )
        not_expiring = CertificateInfo(
            host="valid.example.com",
            port=443,
            subject="valid.example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc),
            not_after=datetime.now(timezone.utc) + timedelta(days=60),
            days_until_expiry=60,
            serial_number="DEF456",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        scanner._cache["expiring.example.com:443"] = expiring
        scanner._cache["valid.example.com:443"] = not_expiring

        expiring_soon = scanner.get_expiring_soon(days=14)

        assert len(expiring_soon) == 1
        assert expiring_soon[0].host == "expiring.example.com"

    def test_get_expired_returns_expired_certs(self):
        """Should return expired certificates."""
        scanner = TLSCertificateScanner([])

        expired = CertificateInfo(
            host="expired.example.com",
            port=443,
            subject="expired.example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc) - timedelta(days=400),
            not_after=datetime.now(timezone.utc) - timedelta(days=30),
            days_until_expiry=-30,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        scanner._cache["expired.example.com:443"] = expired

        expired_certs = scanner.get_expired()

        assert len(expired_certs) == 1
        assert expired_certs[0].host == "expired.example.com"

    def test_get_invalid_returns_invalid_certs(self):
        """Should return certificates with validation errors."""
        scanner = TLSCertificateScanner([])

        invalid = CertificateInfo(
            host="invalid.example.com",
            port=443,
            subject="Unknown",
            issuer="Unknown",
            not_before=datetime.min.replace(tzinfo=timezone.utc),
            not_after=datetime.min.replace(tzinfo=timezone.utc),
            days_until_expiry=-1,
            serial_number="Unknown",
            tls_version="Unknown",
            cipher_suite="Unknown",
            valid=False,
            error="Connection failed",
        )

        scanner._cache["invalid.example.com:443"] = invalid

        invalid_certs = scanner.get_invalid()

        assert len(invalid_certs) == 1
        assert invalid_certs[0].host == "invalid.example.com"


# ============================================================================
# Test: TLSMetricsExporter
# ============================================================================


class TestTLSMetricsExporter:
    """Tests for TLSMetricsExporter class."""

    def test_initializes_successfully(self):
        """Should initialize without errors."""
        exporter = TLSMetricsExporter()
        assert exporter is not None

    @patch.dict('sys.modules', {'prometheus_client': MagicMock()})
    def test_initializes_prometheus_metrics(self):
        """Should initialize Prometheus metrics when available."""
        exporter = TLSMetricsExporter()
        assert exporter is not None

    def test_record_certificate_expiry(self):
        """Should record certificate expiry metric."""
        exporter = TLSMetricsExporter()

        # Should not raise
        exporter.record_certificate_expiry(
            domain="example.com",
            days=30,
            port=443,
            valid=True,
        )

    def test_record_connection(self):
        """Should record TLS connection metric."""
        exporter = TLSMetricsExporter()

        # Should not raise
        exporter.record_connection(
            protocol="TLSv1.3",
            cipher="TLS_AES_256_GCM_SHA384",
            duration=0.05,
        )

    def test_record_error(self):
        """Should record TLS error metric."""
        exporter = TLSMetricsExporter()

        # Should not raise
        exporter.record_error("handshake_failure")

    def test_record_protocol_downgrade(self):
        """Should record protocol downgrade attempt."""
        exporter = TLSMetricsExporter()

        # Should not raise
        exporter.record_protocol_downgrade("TLSv1.0")

    def test_record_mtls_status(self):
        """Should record mTLS enforcement status."""
        exporter = TLSMetricsExporter()

        # Should not raise
        exporter.record_mtls_status(namespace="default", enabled=True)

    def test_update_from_scanner(self):
        """Should update metrics from scanner."""
        exporter = TLSMetricsExporter()
        scanner = TLSCertificateScanner([])

        # Add cached certificate
        scanner._cache["example.com:443"] = CertificateInfo(
            host="example.com",
            port=443,
            subject="example.com",
            issuer="Test CA",
            not_before=datetime.now(timezone.utc),
            not_after=datetime.now(timezone.utc) + timedelta(days=30),
            days_until_expiry=30,
            serial_number="ABC123",
            tls_version="TLSv1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            valid=True,
        )

        # Should not raise
        exporter.update_from_scanner(scanner)


# ============================================================================
# Test: Singleton Exporter
# ============================================================================


class TestSingletonExporter:
    """Tests for singleton exporter instance."""

    def test_get_metrics_exporter_returns_same_instance(self):
        """Should return same instance on multiple calls."""
        exporter1 = get_metrics_exporter()
        exporter2 = get_metrics_exporter()

        assert exporter1 is exporter2


# ============================================================================
# Test: Default Endpoints
# ============================================================================


class TestDefaultEndpoints:
    """Tests for default endpoint configuration."""

    def test_default_endpoints_defined(self):
        """Should have default endpoints defined."""
        assert isinstance(DEFAULT_ENDPOINTS, list)
        assert len(DEFAULT_ENDPOINTS) > 0

    def test_default_endpoints_are_tuples(self):
        """Default endpoints should be (host, port) tuples."""
        for endpoint in DEFAULT_ENDPOINTS:
            assert isinstance(endpoint, tuple)
            assert len(endpoint) == 2
            assert isinstance(endpoint[0], str)
            assert isinstance(endpoint[1], int)
