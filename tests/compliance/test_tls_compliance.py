# -*- coding: utf-8 -*-
# =============================================================================
# Compliance Tests: TLS 1.3 Configuration (SEC-004)
# =============================================================================
"""
Compliance tests for TLS configuration requirements.

Verifies that the TLS implementation meets regulatory and security requirements
for SOC 2, ISO 27001, PCI-DSS, and other security standards.

Regulatory References:
    - SOC 2 Type II: CC6.1 (Encryption in Transit)
    - ISO 27001:2022: A.8.24 (Use of Cryptography), A.14.1.2 (Secure Transport)
    - PCI-DSS v4.0: Requirement 4.1 (Strong Cryptography for Transit)
    - NIST SP 800-52 Rev2: Guidelines for TLS Implementations
    - FIPS 140-2/3: Approved Algorithms

Pass/Fail Criteria:
    All tests must pass for compliance certification.
    Any failure indicates a compliance gap that must be remediated.
"""

from __future__ import annotations

import os
import ssl
import socket
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Set
from unittest.mock import patch, MagicMock

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
        CIPHER_SUITES_TLS13,
        CIPHER_SUITES_TLS12,
        CIPHER_SUITES_MODERN,
        get_enabled_cipher_names,
    )
    from greenlang.infrastructure.tls_service.utils import (
        is_version_secure,
        is_cipher_secure,
    )
    _HAS_TLS_SERVICE = True
except ImportError:
    _HAS_TLS_SERVICE = False

pytestmark = [
    pytest.mark.compliance,
    pytest.mark.security,
    pytest.mark.skipif(not _HAS_TLS_SERVICE, reason="TLS service not installed"),
]


# ============================================================================
# Compliance Test Markers
# ============================================================================

SOC2_CC6_1 = pytest.mark.soc2_cc6_1
ISO27001_A8_24 = pytest.mark.iso27001_a8_24
ISO27001_A14_1_2 = pytest.mark.iso27001_a14_1_2
PCI_DSS_4_1 = pytest.mark.pci_dss_4_1
NIST_SP800_52 = pytest.mark.nist_sp800_52
FIPS_140 = pytest.mark.fips_140


# ============================================================================
# Constants: Compliance Requirements
# ============================================================================

# Minimum TLS version per compliance requirements
MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_2

# Deprecated protocols that must be disabled
DEPRECATED_PROTOCOLS = ["SSLv2", "SSLv3", "TLSv1.0", "TLSv1.1"]

# Weak cipher patterns that must be excluded
WEAK_CIPHER_PATTERNS = [
    "RC4",
    "DES",
    "3DES",
    "MD5",
    "NULL",
    "EXPORT",
    "anon",
    "CBC",  # CBC is considered weak due to padding oracle attacks
]

# Required cipher characteristics
REQUIRED_CIPHER_CHARACTERISTICS = [
    "AEAD",  # Must use authenticated encryption
    "FS",    # Must provide forward secrecy
]


# ============================================================================
# Test: Protocol Version Compliance
# ============================================================================


@SOC2_CC6_1
@ISO27001_A14_1_2
@PCI_DSS_4_1
@NIST_SP800_52
class TestProtocolVersionCompliance:
    """
    Compliance tests for TLS protocol version requirements.

    Requirement: Only TLS 1.2 and TLS 1.3 are permitted.
    Reference: PCI-DSS v4.0 Requirement 4.1, NIST SP 800-52 Rev2
    """

    def test_minimum_tls_version_is_1_2(self):
        """
        Test that minimum TLS version is 1.2.

        COMPLIANCE: PCI-DSS 4.1, SOC 2 CC6.1
        REQUIREMENT: TLS 1.2 minimum for all connections.
        """
        context = create_ssl_context()

        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2, \
            "Minimum TLS version must be 1.2 or higher"

    def test_sslv2_is_disabled(self):
        """
        Test that SSLv2 is disabled.

        COMPLIANCE: PCI-DSS 4.1
        REQUIREMENT: SSLv2 must be disabled (deprecated since 2011).
        """
        context = create_ssl_context()

        assert context.options & ssl.OP_NO_SSLv2, \
            "SSLv2 must be explicitly disabled"

    def test_sslv3_is_disabled(self):
        """
        Test that SSLv3 is disabled.

        COMPLIANCE: PCI-DSS 4.1, RFC 7568
        REQUIREMENT: SSLv3 must be disabled (POODLE vulnerability).
        """
        context = create_ssl_context()

        assert context.options & ssl.OP_NO_SSLv3, \
            "SSLv3 must be explicitly disabled"

    def test_tls_1_0_is_disabled(self):
        """
        Test that TLS 1.0 is disabled.

        COMPLIANCE: PCI-DSS 4.1 (effective June 2018)
        REQUIREMENT: TLS 1.0 must be disabled.
        """
        context = create_ssl_context()

        assert context.options & ssl.OP_NO_TLSv1, \
            "TLS 1.0 must be explicitly disabled"

    def test_tls_1_1_is_disabled(self):
        """
        Test that TLS 1.1 is disabled.

        COMPLIANCE: PCI-DSS 4.1 (recommended), NIST SP 800-52 Rev2
        REQUIREMENT: TLS 1.1 should be disabled.
        """
        context = create_ssl_context()

        assert context.options & ssl.OP_NO_TLSv1_1, \
            "TLS 1.1 must be explicitly disabled"

    def test_tls_1_2_is_supported(self):
        """
        Test that TLS 1.2 is supported.

        COMPLIANCE: PCI-DSS 4.1
        REQUIREMENT: TLS 1.2 must be supported.
        """
        context = create_ssl_context(min_version=ssl.TLSVersion.TLSv1_2)

        assert context.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_tls_1_3_is_supported(self):
        """
        Test that TLS 1.3 is supported.

        COMPLIANCE: Best practice, NIST SP 800-52 Rev2
        REQUIREMENT: TLS 1.3 should be supported.
        """
        context = create_ssl_context(min_version=ssl.TLSVersion.TLSv1_3)

        assert context.minimum_version == ssl.TLSVersion.TLSv1_3


# ============================================================================
# Test: Cipher Suite Compliance
# ============================================================================


@SOC2_CC6_1
@ISO27001_A8_24
@PCI_DSS_4_1
@NIST_SP800_52
@FIPS_140
class TestCipherSuiteCompliance:
    """
    Compliance tests for cipher suite requirements.

    Requirement: Only strong, AEAD cipher suites with forward secrecy.
    Reference: NIST SP 800-52 Rev2 Section 3.3
    """

    def test_all_ciphers_use_aead(self):
        """
        Test that all enabled ciphers use AEAD mode.

        COMPLIANCE: NIST SP 800-52 Rev2
        REQUIREMENT: AEAD modes (GCM, CCM, ChaCha20-Poly1305) required.
        """
        for cipher in CIPHER_SUITES_TLS12:
            cipher_upper = cipher.upper()
            assert any(aead in cipher_upper for aead in ["GCM", "CHACHA20", "CCM"]), \
                f"Cipher {cipher} does not use AEAD mode"

    def test_all_ciphers_use_forward_secrecy(self):
        """
        Test that all enabled ciphers provide forward secrecy.

        COMPLIANCE: PCI-DSS 4.1, NIST SP 800-52 Rev2
        REQUIREMENT: Forward secrecy (ECDHE, DHE) required.
        """
        for cipher in CIPHER_SUITES_TLS12:
            assert "ECDHE" in cipher or "DHE" in cipher, \
                f"Cipher {cipher} does not provide forward secrecy"

    def test_no_rc4_ciphers(self):
        """
        Test that RC4 ciphers are not enabled.

        COMPLIANCE: RFC 7465, PCI-DSS 4.1
        REQUIREMENT: RC4 is prohibited (multiple vulnerabilities).
        """
        for cipher in CIPHER_SUITES_MODERN:
            assert "RC4" not in cipher.upper(), \
                f"RC4 cipher found: {cipher}"

    def test_no_des_ciphers(self):
        """
        Test that DES/3DES ciphers are not enabled.

        COMPLIANCE: NIST SP 800-131A Rev2
        REQUIREMENT: DES/3DES are deprecated.
        """
        for cipher in CIPHER_SUITES_MODERN:
            cipher_upper = cipher.upper()
            assert "DES" not in cipher_upper, \
                f"DES cipher found: {cipher}"

    def test_no_null_ciphers(self):
        """
        Test that NULL ciphers are not enabled.

        COMPLIANCE: All standards
        REQUIREMENT: NULL ciphers provide no encryption.
        """
        for cipher in CIPHER_SUITES_MODERN:
            assert "NULL" not in cipher.upper(), \
                f"NULL cipher found: {cipher}"

    def test_no_export_ciphers(self):
        """
        Test that EXPORT ciphers are not enabled.

        COMPLIANCE: PCI-DSS 4.1
        REQUIREMENT: EXPORT ciphers are weak (FREAK, Logjam).
        """
        for cipher in CIPHER_SUITES_MODERN:
            assert "EXPORT" not in cipher.upper(), \
                f"EXPORT cipher found: {cipher}"

    def test_no_anonymous_ciphers(self):
        """
        Test that anonymous ciphers are not enabled.

        COMPLIANCE: All standards
        REQUIREMENT: Anonymous ciphers have no authentication.
        """
        for cipher in CIPHER_SUITES_MODERN:
            assert "anon" not in cipher.lower(), \
                f"Anonymous cipher found: {cipher}"

    def test_minimum_key_strength_128_bits(self):
        """
        Test that cipher key strength is at least 128 bits.

        COMPLIANCE: NIST SP 800-131A
        REQUIREMENT: Minimum 128-bit symmetric keys.
        """
        for cipher in CIPHER_SUITES_TLS12:
            # AES-128 and AES-256 are acceptable
            assert "AES128" in cipher or "AES256" in cipher or "CHACHA20" in cipher, \
                f"Cipher {cipher} may not meet 128-bit minimum"

    def test_context_ciphers_are_secure(self):
        """
        Test that all ciphers in created context are secure.

        COMPLIANCE: All standards
        REQUIREMENT: Only approved ciphers should be enabled.
        """
        context = create_ssl_context()
        ciphers = get_enabled_cipher_names(context)

        for cipher in ciphers:
            # Skip TLS 1.3 ciphers which have different naming
            if cipher.startswith("TLS_"):
                continue

            assert is_cipher_secure(cipher), \
                f"Insecure cipher enabled: {cipher}"


# ============================================================================
# Test: Certificate Verification Compliance
# ============================================================================


@SOC2_CC6_1
@ISO27001_A14_1_2
class TestCertificateVerificationCompliance:
    """
    Compliance tests for certificate verification requirements.

    Requirement: Certificates must be verified for all connections.
    Reference: SOC 2 CC6.1, ISO 27001 A.14.1.2
    """

    def test_certificate_verification_enabled_by_default(self):
        """
        Test that certificate verification is enabled by default.

        COMPLIANCE: SOC 2 CC6.1
        REQUIREMENT: Certificate verification must be default behavior.
        """
        context = create_ssl_context()

        assert context.verify_mode == ssl.CERT_REQUIRED, \
            "Certificate verification must be enabled by default"

    def test_hostname_checking_enabled_by_default(self):
        """
        Test that hostname checking is enabled by default.

        COMPLIANCE: RFC 6125
        REQUIREMENT: Hostname must match certificate.
        """
        context = create_ssl_context()

        assert context.check_hostname is True, \
            "Hostname checking must be enabled by default"

    def test_client_context_verifies_server(self):
        """
        Test that client context verifies server certificate.

        COMPLIANCE: ISO 27001 A.14.1.2
        REQUIREMENT: Server certificates must be verified.
        """
        context = create_client_ssl_context()

        assert context.verify_mode == ssl.CERT_REQUIRED


# ============================================================================
# Test: TLS Security Options Compliance
# ============================================================================


@NIST_SP800_52
class TestTLSSecurityOptionsCompliance:
    """
    Compliance tests for TLS security options.

    Requirement: Secure TLS options must be enabled.
    Reference: NIST SP 800-52 Rev2
    """

    def test_compression_disabled(self):
        """
        Test that TLS compression is disabled.

        COMPLIANCE: NIST SP 800-52 Rev2
        REQUIREMENT: Compression must be disabled (CRIME attack).
        """
        context = create_ssl_context()

        assert context.options & ssl.OP_NO_COMPRESSION, \
            "TLS compression must be disabled (CRIME vulnerability)"

    def test_single_dh_use_enabled(self):
        """
        Test that fresh DH parameters are used per connection.

        COMPLIANCE: Best practice
        REQUIREMENT: Fresh DH parameters per connection.
        """
        context = create_ssl_context()

        assert context.options & ssl.OP_SINGLE_DH_USE, \
            "Single DH use must be enabled"

    def test_single_ecdh_use_enabled(self):
        """
        Test that fresh ECDH parameters are used per connection.

        COMPLIANCE: Best practice
        REQUIREMENT: Fresh ECDH parameters per connection.
        """
        context = create_ssl_context()

        assert context.options & ssl.OP_SINGLE_ECDH_USE, \
            "Single ECDH use must be enabled"


# ============================================================================
# Test: mTLS Compliance
# ============================================================================


@SOC2_CC6_1
@ISO27001_A8_24
class TestMTLSCompliance:
    """
    Compliance tests for mutual TLS requirements.

    Requirement: Service-to-service authentication via mTLS.
    Reference: SOC 2 CC6.1, ISO 27001 A.8.24
    """

    @patch('ssl.SSLContext.load_cert_chain')
    def test_mtls_client_context_loads_certificate(self, mock_load):
        """
        Test that mTLS client context loads client certificate.

        COMPLIANCE: SOC 2 CC6.1
        REQUIREMENT: Client must present certificate for mTLS.
        """
        create_mtls_client_context(
            client_cert="/path/to/client.crt",
            client_key="/path/to/client.key",
        )

        mock_load.assert_called()

    @patch('ssl.SSLContext.load_cert_chain')
    @patch('ssl.SSLContext.load_verify_locations')
    def test_mtls_server_requires_client_cert(self, mock_verify, mock_load):
        """
        Test that mTLS server requires client certificate.

        COMPLIANCE: SOC 2 CC6.1
        REQUIREMENT: Server must require client certificate.
        """
        context = create_server_ssl_context(
            cert_path="/path/to/server.crt",
            key_path="/path/to/server.key",
            verify_client=True,
            ca_bundle="/path/to/ca.crt",
        )

        assert context.verify_mode == ssl.CERT_REQUIRED


# ============================================================================
# Test: Key Exchange Compliance
# ============================================================================


@FIPS_140
@NIST_SP800_52
class TestKeyExchangeCompliance:
    """
    Compliance tests for key exchange requirements.

    Requirement: Only approved key exchange algorithms.
    Reference: NIST SP 800-52 Rev2, FIPS 140-3
    """

    def test_ecdhe_key_exchange_supported(self):
        """
        Test that ECDHE key exchange is supported.

        COMPLIANCE: NIST SP 800-52 Rev2
        REQUIREMENT: ECDHE is preferred for forward secrecy.
        """
        ecdhe_ciphers = [c for c in CIPHER_SUITES_TLS12 if "ECDHE" in c]
        assert len(ecdhe_ciphers) > 0, \
            "ECDHE key exchange must be supported"

    def test_rsa_key_exchange_not_used(self):
        """
        Test that static RSA key exchange is not used.

        COMPLIANCE: NIST SP 800-52 Rev2
        REQUIREMENT: Static RSA does not provide forward secrecy.
        """
        for cipher in CIPHER_SUITES_TLS12:
            # Ensure RSA is only used with ECDHE/DHE
            if "RSA" in cipher:
                assert "ECDHE" in cipher or "DHE" in cipher, \
                    f"Static RSA key exchange found: {cipher}"


# ============================================================================
# Test: Logging and Audit Compliance
# ============================================================================


@SOC2_CC6_1
@ISO27001_A8_24
class TestLoggingCompliance:
    """
    Compliance tests for TLS logging requirements.

    Requirement: TLS events must be logged for audit.
    Reference: SOC 2 CC6.1
    """

    def test_no_keys_in_logs(self, caplog):
        """
        Test that encryption keys are not logged.

        COMPLIANCE: SOC 2 CC6.1, ISO 27001 A.8.24.1
        REQUIREMENT: Key material must never appear in logs.
        """
        # Create a context and verify no sensitive data in logs
        with caplog.at_level(logging.DEBUG):
            context = create_ssl_context()

        for record in caplog.records:
            msg = record.message.lower()
            # Check for common key material patterns
            assert "private key" not in msg
            assert "secret" not in msg


# ============================================================================
# Test: Performance Compliance
# ============================================================================


class TestPerformanceCompliance:
    """
    Compliance tests for TLS performance requirements.

    Requirement: TLS operations must not significantly impact performance.
    """

    def test_context_creation_under_100ms(self):
        """
        Test that context creation is fast.

        REQUIREMENT: Context creation should be under 100ms.
        """
        import time

        start = time.perf_counter()
        context = create_ssl_context()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, \
            f"Context creation took {elapsed_ms:.2f}ms, expected <100ms"


# ============================================================================
# Test: Configuration Compliance
# ============================================================================


class TestConfigurationCompliance:
    """
    Compliance tests for TLS configuration.

    Requirement: TLS configuration must be consistent across environments.
    """

    def test_default_configuration_is_secure(self):
        """
        Test that default configuration meets security requirements.

        REQUIREMENT: Default configuration must be production-ready.
        """
        context = create_ssl_context()

        # Protocol version
        assert context.minimum_version >= ssl.TLSVersion.TLSv1_2

        # Verification
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

        # Options
        assert context.options & ssl.OP_NO_SSLv2
        assert context.options & ssl.OP_NO_SSLv3
        assert context.options & ssl.OP_NO_TLSv1
        assert context.options & ssl.OP_NO_TLSv1_1
        assert context.options & ssl.OP_NO_COMPRESSION

    def test_tls13_only_configuration(self):
        """
        Test that TLS 1.3 only configuration works.

        REQUIREMENT: Should support TLS 1.3 only mode.
        """
        context = create_ssl_context(min_version=ssl.TLSVersion.TLSv1_3)

        assert context.minimum_version == ssl.TLSVersion.TLSv1_3
