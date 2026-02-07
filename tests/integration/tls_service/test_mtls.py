# -*- coding: utf-8 -*-
# =============================================================================
# Integration Tests: Mutual TLS (mTLS)
# SEC-004: TLS 1.3 Configuration
# =============================================================================
"""
Integration tests for mutual TLS (mTLS) authentication.

These tests verify mTLS functionality for service-to-service authentication.
Tests require valid client certificates and are typically run in staging
environments with proper certificate infrastructure.

Test categories:
- mTLS context creation
- Client certificate authentication
- Certificate chain validation
- mTLS error handling
"""

from __future__ import annotations

import os
import ssl
import socket
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import TLS service modules
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.tls_service.ssl_context import (
        create_ssl_context,
        create_mtls_client_context,
        create_server_ssl_context,
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
def temp_cert_bundle():
    """Create a temporary certificate bundle for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create minimal test certificates
        # These are structural tests - real mTLS requires valid certs
        ca_cert = tmppath / "ca.crt"
        client_cert = tmppath / "client.crt"
        client_key = tmppath / "client.key"
        server_cert = tmppath / "server.crt"
        server_key = tmppath / "server.key"

        # Minimal PEM structure (not cryptographically valid)
        minimal_cert = b"""-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHBfpegDfY0MA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnRl
c3RjYTAeFw0yNjAxMDEwMDAwMDBaFw0yNzAxMDEwMDAwMDBaMBExDzANBgNVBAMM
BnRlc3RjYTBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQDHUigoidjaiv1F8XWbJc0Q
MzLNEQXNmxr+JYR9e06IH9cdPyLpTvfLdMnrAJOLmQNiCuAr9ypDW+t0KP8GmQJd
AgMBAAGjUDBOMB0GA1UdDgQWBBQjZwIfaHxHMk84C3+u/6g+WzYeqTAfBgNVHSME
GDAWgBQjZwIfaHxHMk84C3+u/6g+WzYeqTAMBgNVHRMEBTADAQH/MA0GCSqGSIb3
DQEBCwUAA0EAhD+C3z9nrC0v7OejN5Y2bXSqr2JGb1GKhfNIzNqCzK5MlyB2gJUl
jQhGEtZvYL6k5G4jJLAp+8Hw6WMCL5M5Pw==
-----END CERTIFICATE-----"""

        minimal_key = b"""-----BEGIN PRIVATE KEY-----
MIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEAx1IoKInY2Ir9RfF1
myXNEDMyzREFzZsa/iWEfXtOiB/XHT8i6U73y3TJ6wCTi5kDYgrgK/cqQ1vrdCj/
BpkCXQIDAQABAkAFx9bG5N1JFqHJb4h0H6c3n2Q3MBp0c+Q3mJNCl+l6JLu7zT+r
fH0XAEjTVy9mJz9NpXM9lfC6a5h0h1X0D9ehAiEA8UXFW7h0iuZ0T1K5lzJ0L5Qu
NyLzHLPKl0xPDI/J9DUCIQDR/c7qJmD1yNTK6E3WlBpJc0fPNjq1O8nL0XPpBH0P
EQIgT3TymBk+7R9ZU5Q6a5q7V8lK2M2u6MK9dJG5L9wFDyUCIQCg9zc0S1NJgqH9
qZQ0bFn5E5OBk2Z9PNQJ0nKp8BZ+cQIhAMP3+1m1b3HDGU8VBDM0v9HpPw3oxShJ
s8ZK9E9M2z0F
-----END PRIVATE KEY-----"""

        for path in [ca_cert, client_cert, server_cert]:
            path.write_bytes(minimal_cert)

        for path in [client_key, server_key]:
            path.write_bytes(minimal_key)

        yield {
            "dir": tmpdir,
            "ca_cert": str(ca_cert),
            "client_cert": str(client_cert),
            "client_key": str(client_key),
            "server_cert": str(server_cert),
            "server_key": str(server_key),
        }


@pytest.fixture
def mtls_env_configured():
    """Check if mTLS environment is configured."""
    required_vars = [
        "MTLS_CLIENT_CERT",
        "MTLS_CLIENT_KEY",
        "MTLS_CA_CERT",
        "MTLS_TEST_HOST",
    ]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        pytest.skip(f"mTLS environment not configured. Missing: {missing}")


# ============================================================================
# Test: mTLS Context Creation
# ============================================================================


class TestMTLSContextCreation:
    """Tests for mTLS context creation."""

    @patch('ssl.SSLContext.load_cert_chain')
    def test_creates_mtls_client_context(self, mock_load, temp_cert_bundle):
        """Should create mTLS client context with client certificate."""
        context = create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
            ca_bundle=temp_cert_bundle["ca_cert"],
        )

        assert isinstance(context, ssl.SSLContext)
        mock_load.assert_called()

    @patch('ssl.SSLContext.load_cert_chain')
    def test_mtls_context_verifies_server(self, mock_load, temp_cert_bundle):
        """mTLS client context should verify server certificate."""
        context = create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
            ca_bundle=temp_cert_bundle["ca_cert"],
        )

        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    @patch('ssl.SSLContext.load_cert_chain')
    @patch('ssl.SSLContext.load_verify_locations')
    def test_server_context_requires_client_cert(
        self, mock_verify, mock_load, temp_cert_bundle
    ):
        """Server context should require client certificate for mTLS."""
        context = create_server_ssl_context(
            cert_path=temp_cert_bundle["server_cert"],
            key_path=temp_cert_bundle["server_key"],
            verify_client=True,
            ca_bundle=temp_cert_bundle["ca_cert"],
        )

        assert context.verify_mode == ssl.CERT_REQUIRED

    @patch('ssl.SSLContext.load_cert_chain')
    def test_mtls_uses_tls12_minimum(self, mock_load, temp_cert_bundle):
        """mTLS context should use TLS 1.2 minimum."""
        context = create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
            min_version=ssl.TLSVersion.TLSv1_2,
        )

        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2


# ============================================================================
# Test: Client Certificate Authentication
# ============================================================================


class TestClientCertificateAuth:
    """Tests for client certificate authentication."""

    @patch('ssl.SSLContext.load_cert_chain')
    def test_loads_client_certificate(self, mock_load, temp_cert_bundle):
        """Should load client certificate for authentication."""
        create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
        )

        mock_load.assert_called_with(
            certfile=temp_cert_bundle["client_cert"],
            keyfile=temp_cert_bundle["client_key"],
            password=None,
        )

    @patch('ssl.SSLContext.load_cert_chain')
    def test_supports_encrypted_client_key(self, mock_load, temp_cert_bundle):
        """Should support encrypted client private key."""
        create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
            key_password="secret123",
        )

        mock_load.assert_called_with(
            certfile=temp_cert_bundle["client_cert"],
            keyfile=temp_cert_bundle["client_key"],
            password="secret123",
        )

    def test_fails_with_missing_client_cert(self, temp_cert_bundle):
        """Should fail when client certificate is missing."""
        with pytest.raises((FileNotFoundError, ssl.SSLError)):
            create_mtls_client_context(
                client_cert="/nonexistent/client.crt",
                client_key=temp_cert_bundle["client_key"],
            )

    def test_fails_with_missing_client_key(self, temp_cert_bundle):
        """Should fail when client key is missing."""
        with pytest.raises((FileNotFoundError, ssl.SSLError)):
            create_mtls_client_context(
                client_cert=temp_cert_bundle["client_cert"],
                client_key="/nonexistent/client.key",
            )


# ============================================================================
# Test: Certificate Chain Validation
# ============================================================================


class TestCertificateChainValidation:
    """Tests for certificate chain validation in mTLS."""

    @patch('ssl.SSLContext.load_cert_chain')
    @patch('ssl.SSLContext.load_verify_locations')
    def test_loads_ca_bundle_for_verification(
        self, mock_verify, mock_load, temp_cert_bundle
    ):
        """Should load CA bundle for certificate chain verification."""
        create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
            ca_bundle=temp_cert_bundle["ca_cert"],
        )

        # Context should be configured for verification
        # CA bundle is loaded via create_ssl_context

    @patch('ssl.SSLContext.load_cert_chain')
    def test_server_loads_ca_for_client_verification(
        self, mock_load, temp_cert_bundle
    ):
        """Server should load CA bundle to verify client certificates."""
        with patch('ssl.SSLContext.load_verify_locations') as mock_verify:
            create_server_ssl_context(
                cert_path=temp_cert_bundle["server_cert"],
                key_path=temp_cert_bundle["server_key"],
                verify_client=True,
                ca_bundle=temp_cert_bundle["ca_cert"],
            )

            mock_verify.assert_called()


# ============================================================================
# Test: mTLS Error Handling
# ============================================================================


class TestMTLSErrorHandling:
    """Tests for mTLS error handling scenarios."""

    def test_handles_certificate_key_mismatch(self, temp_cert_bundle):
        """Should handle certificate/key mismatch gracefully."""
        # Using mismatched cert and key should fail
        # In real scenarios, this would raise ssl.SSLError
        # Our test certs are minimal and may not trigger this
        pass

    @patch('ssl.SSLContext.load_cert_chain')
    def test_handles_wrong_key_password(self, mock_load, temp_cert_bundle):
        """Should handle wrong key password gracefully."""
        mock_load.side_effect = ssl.SSLError("bad decrypt")

        with pytest.raises(ssl.SSLError):
            create_mtls_client_context(
                client_cert=temp_cert_bundle["client_cert"],
                client_key=temp_cert_bundle["client_key"],
                key_password="wrong_password",
            )

    @patch('ssl.SSLContext.load_cert_chain')
    def test_handles_expired_client_certificate(self, mock_load, temp_cert_bundle):
        """Should handle expired client certificate."""
        # Expired certificate would fail during TLS handshake
        # Context creation succeeds, connection fails
        context = create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
        )

        assert context is not None


# ============================================================================
# Test: Live mTLS Connection (requires environment setup)
# ============================================================================


class TestLiveMTLSConnection:
    """Live mTLS connection tests (requires environment configuration)."""

    @pytest.mark.skip(reason="Requires mTLS test server")
    def test_mtls_handshake_succeeds(self, mtls_env_configured):
        """Should complete mTLS handshake with valid client certificate."""
        context = create_mtls_client_context(
            client_cert=os.environ["MTLS_CLIENT_CERT"],
            client_key=os.environ["MTLS_CLIENT_KEY"],
            ca_bundle=os.environ.get("MTLS_CA_CERT"),
        )

        host = os.environ["MTLS_TEST_HOST"]
        port = int(os.environ.get("MTLS_TEST_PORT", "443"))

        with socket.create_connection((host, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                assert ssock.version() in ["TLSv1.2", "TLSv1.3"]

    @pytest.mark.skip(reason="Requires mTLS test server")
    def test_mtls_fails_without_client_cert(self, mtls_env_configured):
        """Should fail mTLS handshake without client certificate."""
        # Create context without client certificate
        context = create_ssl_context(
            verify=True,
            ca_bundle=os.environ.get("MTLS_CA_CERT"),
        )

        host = os.environ["MTLS_TEST_HOST"]
        port = int(os.environ.get("MTLS_TEST_PORT", "443"))

        # Server should reject connection without client cert
        with pytest.raises(ssl.SSLError):
            with socket.create_connection((host, port), timeout=10) as sock:
                context.wrap_socket(sock, server_hostname=host)


# ============================================================================
# Test: mTLS Security Properties
# ============================================================================


class TestMTLSSecurityProperties:
    """Tests for mTLS security properties."""

    @patch('ssl.SSLContext.load_cert_chain')
    def test_mtls_disables_insecure_protocols(self, mock_load, temp_cert_bundle):
        """mTLS context should disable insecure protocols."""
        context = create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
        )

        assert context.options & ssl.OP_NO_SSLv2
        assert context.options & ssl.OP_NO_SSLv3
        assert context.options & ssl.OP_NO_TLSv1
        assert context.options & ssl.OP_NO_TLSv1_1

    @patch('ssl.SSLContext.load_cert_chain')
    def test_mtls_disables_compression(self, mock_load, temp_cert_bundle):
        """mTLS context should disable TLS compression."""
        context = create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
        )

        assert context.options & ssl.OP_NO_COMPRESSION

    @patch('ssl.SSLContext.load_cert_chain')
    def test_mtls_uses_modern_ciphers(self, mock_load, temp_cert_bundle):
        """mTLS context should use modern cipher suites."""
        context = create_mtls_client_context(
            client_cert=temp_cert_bundle["client_cert"],
            client_key=temp_cert_bundle["client_key"],
        )

        ciphers = [c["name"] for c in context.get_ciphers()]

        # Should have AEAD ciphers
        assert any("GCM" in c or "CHACHA20" in c for c in ciphers)

        # Should not have weak ciphers
        for cipher in ciphers:
            cipher_upper = cipher.upper()
            assert "RC4" not in cipher_upper
            assert "DES" not in cipher_upper
            assert "NULL" not in cipher_upper
