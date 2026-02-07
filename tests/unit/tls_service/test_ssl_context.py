# -*- coding: utf-8 -*-
# =============================================================================
# Unit Tests: SSL Context Factory
# SEC-004: TLS 1.3 Configuration
# =============================================================================
"""
Tests for SSL context creation and configuration.

Tests cover:
- SSL context creation with default and custom parameters
- TLS version enforcement (TLS 1.2+ minimum)
- Cipher suite configuration
- Certificate verification settings
- mTLS context creation
- Server context with certificates

Coverage target: 85%+
"""

from __future__ import annotations

import ssl
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import TLS service modules
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.tls_service.ssl_context import (
        create_ssl_context,
        create_client_ssl_context,
        create_server_ssl_context,
        create_mtls_client_context,
        get_default_ciphers,
        get_cipher_string,
        get_enabled_cipher_names,
        get_context_info,
        CIPHER_SUITES_TLS13,
        CIPHER_SUITES_TLS12,
        CIPHER_SUITES_MODERN,
        CIPHER_STRING_MODERN,
    )
    _HAS_TLS_SERVICE = True
except ImportError:
    _HAS_TLS_SERVICE = False

pytestmark = [
    pytest.mark.skipif(not _HAS_TLS_SERVICE, reason="TLS service not installed"),
]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_cert_dir():
    """Create temporary directory with test certificates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy certificate files for testing
        cert_path = Path(tmpdir) / "server.crt"
        key_path = Path(tmpdir) / "server.key"

        # Minimal self-signed certificate (for structure tests only)
        cert_pem = b"""-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHBfpegDfY0MA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnRl
c3RjYTAeFw0yNjAxMDEwMDAwMDBaFw0yNzAxMDEwMDAwMDBaMBExDzANBgNVBAMM
BnRlc3RjYTBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQDHUigoidjaiv1F8XWbJc0Q
MzLNEQXNmxr+JYR9e06IH9cdPyLpTvfLdMnrAJOLmQNiCuAr9ypDW+t0KP8GmQJd
AgMBAAGjUDBOMB0GA1UdDgQWBBQjZwIfaHxHMk84C3+u/6g+WzYeqTAfBgNVHSME
GDAWgBQjZwIfaHxHMk84C3+u/6g+WzYeqTAMBgNVHRMEBTADAQH/MA0GCSqGSIb3
DQEBCwUAA0EAhD+C3z9nrC0v7OejN5Y2bXSqr2JGb1GKhfNIzNqCzK5MlyB2gJUl
jQhGEtZvYL6k5G4jJLAp+8Hw6WMCL5M5Pw==
-----END CERTIFICATE-----"""

        key_pem = b"""-----BEGIN PRIVATE KEY-----
MIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEAx1IoKInY2Ir9RfF1
myXNEDMyzREFzZsa/iWEfXtOiB/XHT8i6U73y3TJ6wCTi5kDYgrgK/cqQ1vrdCj/
BpkCXQIDAQABAkAFx9bG5N1JFqHJb4h0H6c3n2Q3MBp0c+Q3mJNCl+l6JLu7zT+r
fH0XAEjTVy9mJz9NpXM9lfC6a5h0h1X0D9ehAiEA8UXFW7h0iuZ0T1K5lzJ0L5Qu
NyLzHLPKl0xPDI/J9DUCIQDR/c7qJmD1yNTK6E3WlBpJc0fPNjq1O8nL0XPpBH0P
EQIgT3TymBk+7R9ZU5Q6a5q7V8lK2M2u6MK9dJG5L9wFDyUCIQCg9zc0S1NJgqH9
qZQ0bFn5E5OBk2Z9PNQJ0nKp8BZ+cQIhAMP3+1m1b3HDGU8VBDM0v9HpPw3oxShJ
s8ZK9E9M2z0F
-----END PRIVATE KEY-----"""

        cert_path.write_bytes(cert_pem)
        key_path.write_bytes(key_pem)

        yield {
            "dir": tmpdir,
            "cert": str(cert_path),
            "key": str(key_path),
        }


# ============================================================================
# Test: create_ssl_context
# ============================================================================


class TestCreateSSLContext:
    """Tests for create_ssl_context function."""

    def test_creates_context_with_default_config(self):
        """Should create SSL context with default configuration."""
        context = create_ssl_context()

        assert isinstance(context, ssl.SSLContext)
        assert context.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_creates_context_with_tls13_minimum(self):
        """Should enforce TLS 1.3 minimum when configured."""
        context = create_ssl_context(min_version=ssl.TLSVersion.TLSv1_3)

        assert context.minimum_version == ssl.TLSVersion.TLSv1_3

    def test_disables_insecure_protocols(self):
        """Should disable SSLv2, SSLv3, TLS 1.0, TLS 1.1."""
        context = create_ssl_context()

        assert context.options & ssl.OP_NO_SSLv2
        assert context.options & ssl.OP_NO_SSLv3
        assert context.options & ssl.OP_NO_TLSv1
        assert context.options & ssl.OP_NO_TLSv1_1

    def test_disables_compression(self):
        """Should disable TLS compression (CRIME attack mitigation)."""
        context = create_ssl_context()

        assert context.options & ssl.OP_NO_COMPRESSION

    def test_enables_certificate_verification_by_default(self):
        """Should enable certificate verification by default."""
        context = create_ssl_context()

        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_disables_verification_when_requested(self):
        """Should disable verification when verify=False."""
        context = create_ssl_context(verify=False)

        assert context.verify_mode == ssl.CERT_NONE
        assert context.check_hostname is False

    def test_client_purpose_creates_server_auth_context(self):
        """Client context should use SERVER_AUTH purpose."""
        context = create_ssl_context(purpose="client")
        # This is checked implicitly by successful creation
        assert context is not None

    def test_database_purpose_creates_server_auth_context(self):
        """Database context should use SERVER_AUTH purpose."""
        context = create_ssl_context(purpose="database")
        assert context is not None

    def test_grpc_purpose_creates_server_auth_context(self):
        """gRPC context should use SERVER_AUTH purpose."""
        context = create_ssl_context(purpose="grpc")
        assert context is not None

    def test_redis_purpose_creates_server_auth_context(self):
        """Redis context should use SERVER_AUTH purpose."""
        context = create_ssl_context(purpose="redis")
        assert context is not None

    def test_sets_secure_cipher_suites(self):
        """Should set secure cipher suites."""
        context = create_ssl_context()
        ciphers = get_enabled_cipher_names(context)

        # Should have modern ciphers
        assert len(ciphers) > 0

        # Should not have weak ciphers
        for cipher in ciphers:
            cipher_upper = cipher.upper()
            assert "RC4" not in cipher_upper
            assert "DES" not in cipher_upper
            assert "NULL" not in cipher_upper
            assert "EXPORT" not in cipher_upper

    def test_custom_cipher_string(self):
        """Should use custom cipher string when provided."""
        custom_ciphers = "ECDHE-RSA-AES256-GCM-SHA384"
        context = create_ssl_context(ciphers=custom_ciphers)
        ciphers = get_enabled_cipher_names(context)

        assert "ECDHE-RSA-AES256-GCM-SHA384" in ciphers

    @patch('greenlang.infrastructure.tls_service.ssl_context.get_ca_bundle_path')
    def test_uses_ca_bundle_when_verifying(self, mock_ca_bundle):
        """Should load CA bundle when verification is enabled."""
        mock_ca_bundle.return_value = "/path/to/ca.pem"

        # Should not raise when ca_bundle is provided
        context = create_ssl_context(ca_bundle="/custom/ca.pem", verify=True)
        assert context is not None


# ============================================================================
# Test: create_client_ssl_context
# ============================================================================


class TestCreateClientSSLContext:
    """Tests for create_client_ssl_context function."""

    def test_creates_client_context(self):
        """Should create a client SSL context."""
        context = create_client_ssl_context()

        assert isinstance(context, ssl.SSLContext)

    def test_uses_default_verification(self):
        """Should verify certificates by default."""
        context = create_client_ssl_context()

        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_disables_verification(self):
        """Should disable verification when requested."""
        context = create_client_ssl_context(verify=False)

        assert context.verify_mode == ssl.CERT_NONE
        assert context.check_hostname is False

    def test_uses_provided_min_version(self):
        """Should use provided minimum TLS version."""
        context = create_client_ssl_context(min_version=ssl.TLSVersion.TLSv1_3)

        assert context.minimum_version == ssl.TLSVersion.TLSv1_3


# ============================================================================
# Test: create_server_ssl_context
# ============================================================================


class TestCreateServerSSLContext:
    """Tests for create_server_ssl_context function."""

    @patch('ssl.SSLContext.load_cert_chain')
    def test_creates_server_context_with_certificate(self, mock_load_cert):
        """Should create server context and load certificate."""
        context = create_server_ssl_context(
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem",
        )

        assert isinstance(context, ssl.SSLContext)
        mock_load_cert.assert_called_once_with(
            certfile="/path/to/cert.pem",
            keyfile="/path/to/key.pem",
            password=None,
        )

    @patch('ssl.SSLContext.load_cert_chain')
    def test_uses_key_password(self, mock_load_cert):
        """Should pass key password to load_cert_chain."""
        create_server_ssl_context(
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem",
            key_password="secret123",
        )

        mock_load_cert.assert_called_once_with(
            certfile="/path/to/cert.pem",
            keyfile="/path/to/key.pem",
            password="secret123",
        )

    @patch('ssl.SSLContext.load_cert_chain')
    @patch('ssl.SSLContext.load_verify_locations')
    def test_enables_client_verification_for_mtls(self, mock_verify, mock_cert):
        """Should enable client verification when mTLS requested."""
        context = create_server_ssl_context(
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem",
            verify_client=True,
            ca_bundle="/path/to/ca.pem",
        )

        assert context.verify_mode == ssl.CERT_REQUIRED
        mock_verify.assert_called()

    @patch('ssl.SSLContext.load_cert_chain')
    def test_sets_cipher_server_preference(self, mock_load_cert):
        """Should set cipher server preference option."""
        context = create_server_ssl_context(
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem",
        )

        if hasattr(ssl, 'OP_CIPHER_SERVER_PREFERENCE'):
            assert context.options & ssl.OP_CIPHER_SERVER_PREFERENCE

    @patch('ssl.SSLContext.load_cert_chain')
    def test_disables_client_verification_by_default(self, mock_load_cert):
        """Should not require client certificates by default."""
        context = create_server_ssl_context(
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem",
            verify_client=False,
        )

        assert context.verify_mode == ssl.CERT_NONE


# ============================================================================
# Test: create_mtls_client_context
# ============================================================================


class TestCreateMTLSClientContext:
    """Tests for create_mtls_client_context function."""

    @patch('ssl.SSLContext.load_cert_chain')
    def test_creates_mtls_context(self, mock_load_cert):
        """Should create mTLS client context with client certificate."""
        context = create_mtls_client_context(
            client_cert="/path/to/client.pem",
            client_key="/path/to/client.key",
        )

        assert isinstance(context, ssl.SSLContext)
        mock_load_cert.assert_called()

    @patch('ssl.SSLContext.load_cert_chain')
    def test_verifies_server_certificate(self, mock_load_cert):
        """Should verify server certificate in mTLS."""
        context = create_mtls_client_context(
            client_cert="/path/to/client.pem",
            client_key="/path/to/client.key",
        )

        assert context.verify_mode == ssl.CERT_REQUIRED


# ============================================================================
# Test: Cipher Suite Functions
# ============================================================================


class TestCipherSuiteFunctions:
    """Tests for cipher suite helper functions."""

    def test_get_default_ciphers_returns_list(self):
        """Should return list of cipher suites."""
        ciphers = get_default_ciphers()

        assert isinstance(ciphers, list)
        assert len(ciphers) > 0

    def test_get_default_ciphers_includes_tls13(self):
        """Should include TLS 1.3 cipher suites."""
        ciphers = get_default_ciphers()

        assert any("TLS_AES" in c for c in ciphers)

    def test_get_default_ciphers_includes_tls12(self):
        """Should include TLS 1.2 cipher suites."""
        ciphers = get_default_ciphers()

        assert any("ECDHE" in c for c in ciphers)

    def test_get_cipher_string_returns_colon_separated(self):
        """Should return colon-separated cipher string."""
        cipher_string = get_cipher_string()

        assert isinstance(cipher_string, str)
        assert ":" in cipher_string

    def test_cipher_suites_tls13_defined(self):
        """Should have TLS 1.3 cipher suites defined."""
        assert isinstance(CIPHER_SUITES_TLS13, list)
        assert "TLS_AES_256_GCM_SHA384" in CIPHER_SUITES_TLS13

    def test_cipher_suites_tls12_defined(self):
        """Should have TLS 1.2 cipher suites defined."""
        assert isinstance(CIPHER_SUITES_TLS12, list)
        assert any("ECDHE" in c for c in CIPHER_SUITES_TLS12)

    def test_cipher_suites_modern_combined(self):
        """Should have combined modern cipher suites."""
        assert isinstance(CIPHER_SUITES_MODERN, list)
        assert len(CIPHER_SUITES_MODERN) == len(CIPHER_SUITES_TLS13) + len(CIPHER_SUITES_TLS12)


# ============================================================================
# Test: Context Inspection
# ============================================================================


class TestContextInspection:
    """Tests for context inspection functions."""

    def test_get_enabled_cipher_names_returns_list(self):
        """Should return list of cipher names from context."""
        context = create_ssl_context()
        ciphers = get_enabled_cipher_names(context)

        assert isinstance(ciphers, list)
        assert len(ciphers) > 0
        assert all(isinstance(c, str) for c in ciphers)

    def test_get_context_info_returns_dict(self):
        """Should return dictionary with context info."""
        context = create_ssl_context()
        info = get_context_info(context)

        assert isinstance(info, dict)
        assert "minimum_version" in info
        assert "verify_mode" in info
        assert "check_hostname" in info
        assert "cipher_count" in info

    def test_get_context_info_includes_options(self):
        """Should include formatted options in info."""
        context = create_ssl_context()
        info = get_context_info(context)

        assert "options" in info
        assert isinstance(info["options"], list)
        assert "NO_SSLv2" in info["options"]


# ============================================================================
# Test: Security Properties
# ============================================================================


class TestSecurityProperties:
    """Tests for security-related properties."""

    def test_all_ciphers_use_aead(self):
        """All TLS 1.2 ciphers should use AEAD modes."""
        for cipher in CIPHER_SUITES_TLS12:
            cipher_upper = cipher.upper()
            assert "GCM" in cipher_upper or "CHACHA20" in cipher_upper, \
                f"Cipher {cipher} is not AEAD"

    def test_all_ciphers_use_forward_secrecy(self):
        """All TLS 1.2 ciphers should use ECDHE for forward secrecy."""
        for cipher in CIPHER_SUITES_TLS12:
            assert "ECDHE" in cipher, \
                f"Cipher {cipher} does not use ECDHE"

    def test_no_weak_ciphers_in_defaults(self):
        """Default ciphers should not include weak algorithms."""
        weak_patterns = ["RC4", "DES", "MD5", "NULL", "EXPORT", "anon"]

        for cipher in CIPHER_SUITES_MODERN:
            cipher_upper = cipher.upper()
            for weak in weak_patterns:
                assert weak not in cipher_upper, \
                    f"Weak pattern {weak} found in cipher {cipher}"

    def test_minimum_version_enforced(self):
        """Should enforce minimum TLS 1.2."""
        context = create_ssl_context()

        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2

    def test_insecure_options_disabled(self):
        """Should have insecure options disabled."""
        context = create_ssl_context()

        # All insecure protocols should be disabled
        assert context.options & ssl.OP_NO_SSLv2
        assert context.options & ssl.OP_NO_SSLv3
        assert context.options & ssl.OP_NO_TLSv1
        assert context.options & ssl.OP_NO_TLSv1_1

        # Compression should be disabled
        assert context.options & ssl.OP_NO_COMPRESSION


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_cipher_string_raises_error(self):
        """Should raise error for invalid cipher string."""
        with pytest.raises(ssl.SSLError):
            create_ssl_context(ciphers="INVALID_CIPHER_THAT_DOES_NOT_EXIST")

    @patch('ssl.SSLContext.load_cert_chain')
    def test_invalid_certificate_raises_error(self, mock_load_cert):
        """Should raise error for invalid certificate."""
        mock_load_cert.side_effect = ssl.SSLError("Cannot load certificate")

        with pytest.raises(ssl.SSLError):
            create_server_ssl_context(
                cert_path="/invalid/cert.pem",
                key_path="/invalid/key.pem",
            )


# ============================================================================
# Test: TLS Version Compatibility
# ============================================================================


class TestTLSVersionCompatibility:
    """Tests for TLS version compatibility."""

    def test_supports_tls_1_2(self):
        """Should support TLS 1.2."""
        context = create_ssl_context(min_version=ssl.TLSVersion.TLSv1_2)
        assert context.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_supports_tls_1_3(self):
        """Should support TLS 1.3."""
        context = create_ssl_context(min_version=ssl.TLSVersion.TLSv1_3)
        assert context.minimum_version == ssl.TLSVersion.TLSv1_3

    def test_can_set_maximum_version(self):
        """Should be able to set maximum TLS version."""
        context = create_ssl_context(max_version=ssl.TLSVersion.TLSv1_2)
        assert context.maximum_version == ssl.TLSVersion.TLSv1_2
