# -*- coding: utf-8 -*-
# =============================================================================
# GreenLang TLS Service - Utility Functions
# SEC-004: TLS 1.3 Configuration for All Services
# =============================================================================
"""
TLS utility functions for GreenLang applications.

Provides helper functions for certificate operations, TLS version detection,
cipher suite parsing, and connection diagnostics.

Follows the GreenLang zero-hallucination principle: all operations are
deterministic using the standard library ``ssl`` module and ``cryptography``.

Functions:
    - get_tls_version_string: Convert SSL version constant to string.
    - parse_cipher_string: Parse cipher suite string to list.
    - format_cipher_string: Format cipher list to string.
    - get_certificate_info: Extract certificate details.
    - check_certificate_expiry: Check if certificate is expiring.
    - get_connection_info: Get TLS connection details.
    - verify_hostname: Verify certificate hostname match.

Example:
    >>> from greenlang.infrastructure.tls_service.utils import (
    ...     get_tls_version_string,
    ...     get_certificate_info,
    ... )
    >>> version_str = get_tls_version_string(ssl.TLSVersion.TLSv1_3)
    >>> cert_info = get_certificate_info("/path/to/cert.pem")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import socket
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TLS Version Utilities
# ---------------------------------------------------------------------------


def get_tls_version_string(version: Union[ssl.TLSVersion, int, str]) -> str:
    """
    Convert TLS version to human-readable string.

    Args:
        version: SSL/TLS version (TLSVersion enum, int constant, or string).

    Returns:
        Human-readable version string.
    """
    version_map = {
        ssl.TLSVersion.SSLv3: "SSLv3",
        ssl.TLSVersion.TLSv1: "TLSv1.0",
        ssl.TLSVersion.TLSv1_1: "TLSv1.1",
        ssl.TLSVersion.TLSv1_2: "TLSv1.2",
        ssl.TLSVersion.TLSv1_3: "TLSv1.3",
    }

    if isinstance(version, ssl.TLSVersion):
        return version_map.get(version, str(version))
    elif isinstance(version, int):
        # Handle raw protocol version numbers
        int_map = {
            0x0300: "SSLv3",
            0x0301: "TLSv1.0",
            0x0302: "TLSv1.1",
            0x0303: "TLSv1.2",
            0x0304: "TLSv1.3",
        }
        return int_map.get(version, f"Unknown(0x{version:04x})")
    elif isinstance(version, str):
        return version
    return str(version)


def get_tls_version_enum(version_str: str) -> ssl.TLSVersion:
    """
    Convert version string to TLSVersion enum.

    Args:
        version_str: Version string (e.g., "TLSv1.3", "1.3").

    Returns:
        TLSVersion enum value.

    Raises:
        ValueError: If version string is not recognized.
    """
    version_str = version_str.upper().replace("V", "v").replace(".", "_")

    if "1_3" in version_str or version_str == "1.3":
        return ssl.TLSVersion.TLSv1_3
    elif "1_2" in version_str or version_str == "1.2":
        return ssl.TLSVersion.TLSv1_2
    elif "1_1" in version_str or version_str == "1.1":
        return ssl.TLSVersion.TLSv1_1
    elif "1_0" in version_str or version_str == "1.0" or version_str == "TLSv1":
        return ssl.TLSVersion.TLSv1
    else:
        raise ValueError(f"Unknown TLS version: {version_str}")


def is_version_secure(version: Union[ssl.TLSVersion, str]) -> bool:
    """
    Check if TLS version meets security requirements (TLS 1.2+).

    Args:
        version: TLS version to check.

    Returns:
        True if version is TLS 1.2 or higher.
    """
    if isinstance(version, str):
        try:
            version = get_tls_version_enum(version)
        except ValueError:
            return False

    secure_versions = {ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3}
    return version in secure_versions


# ---------------------------------------------------------------------------
# Cipher Suite Utilities
# ---------------------------------------------------------------------------


def parse_cipher_string(cipher_string: str) -> List[str]:
    """
    Parse colon-separated cipher string to list.

    Args:
        cipher_string: Colon-separated cipher suites.

    Returns:
        List of cipher suite names.
    """
    if not cipher_string:
        return []
    return [c.strip() for c in cipher_string.split(":") if c.strip()]


def format_cipher_string(ciphers: List[str]) -> str:
    """
    Format cipher list to colon-separated string.

    Args:
        ciphers: List of cipher suite names.

    Returns:
        Colon-separated cipher string.
    """
    return ":".join(ciphers)


def is_cipher_secure(cipher_name: str) -> bool:
    """
    Check if a cipher suite is considered secure.

    Secure ciphers:
    - Use AEAD (GCM, CHACHA20-POLY1305)
    - Use ECDHE or DHE for forward secrecy
    - Use AES-128 or AES-256

    Args:
        cipher_name: Cipher suite name.

    Returns:
        True if cipher is considered secure.
    """
    cipher_upper = cipher_name.upper()

    # TLS 1.3 ciphers are all secure by design
    if cipher_upper.startswith("TLS_"):
        return True

    # Check for AEAD
    if not any(aead in cipher_upper for aead in ["GCM", "CHACHA20", "CCM"]):
        return False

    # Check for forward secrecy
    if not any(fs in cipher_upper for fs in ["ECDHE", "DHE"]):
        return False

    # Check key exchange/authentication
    if any(weak in cipher_upper for weak in ["NULL", "ANON", "EXPORT", "RC4", "MD5", "DES"]):
        return False

    return True


def filter_secure_ciphers(ciphers: List[str]) -> List[str]:
    """
    Filter cipher list to only secure ciphers.

    Args:
        ciphers: List of cipher suite names.

    Returns:
        List of secure cipher suites.
    """
    return [c for c in ciphers if is_cipher_secure(c)]


def get_context_ciphers(context: ssl.SSLContext) -> List[str]:
    """
    Get list of enabled ciphers from SSL context.

    Args:
        context: SSL context.

    Returns:
        List of cipher names.
    """
    return [c["name"] for c in context.get_ciphers()]


# ---------------------------------------------------------------------------
# Certificate Utilities
# ---------------------------------------------------------------------------


@dataclass
class CertificateInfo:
    """Certificate information extracted from X.509 certificate."""

    subject: Dict[str, str] = field(default_factory=dict)
    issuer: Dict[str, str] = field(default_factory=dict)
    serial_number: Optional[str] = None
    not_before: Optional[datetime] = None
    not_after: Optional[datetime] = None
    fingerprint_sha256: Optional[str] = None
    san: List[str] = field(default_factory=list)
    is_ca: bool = False
    version: int = 3
    key_size: Optional[int] = None
    signature_algorithm: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if certificate is expired."""
        if self.not_after is None:
            return False
        now = datetime.now(timezone.utc)
        return now > self.not_after.replace(tzinfo=timezone.utc)

    @property
    def days_until_expiry(self) -> Optional[int]:
        """Get days until certificate expires."""
        if self.not_after is None:
            return None
        now = datetime.now(timezone.utc)
        delta = self.not_after.replace(tzinfo=timezone.utc) - now
        return delta.days

    @property
    def common_name(self) -> Optional[str]:
        """Get common name from subject."""
        return self.subject.get("commonName") or self.subject.get("CN")


def get_certificate_info(
    cert_path: Optional[str] = None,
    cert_data: Optional[bytes] = None,
) -> CertificateInfo:
    """
    Extract information from X.509 certificate.

    Args:
        cert_path: Path to PEM-encoded certificate file.
        cert_data: Raw certificate data (PEM or DER).

    Returns:
        CertificateInfo with extracted details.

    Raises:
        ValueError: If neither cert_path nor cert_data provided.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        raise ImportError("cryptography library required for certificate inspection")

    if cert_path:
        cert_data = Path(cert_path).read_bytes()
    elif cert_data is None:
        raise ValueError("Either cert_path or cert_data must be provided")

    # Try PEM first, then DER
    try:
        cert = x509.load_pem_x509_certificate(cert_data, default_backend())
    except Exception:
        cert = x509.load_der_x509_certificate(cert_data, default_backend())

    # Extract subject and issuer
    def name_to_dict(name: x509.Name) -> Dict[str, str]:
        result = {}
        for attr in name:
            oid_name = attr.oid._name
            if oid_name:
                result[oid_name] = attr.value
        return result

    info = CertificateInfo(
        subject=name_to_dict(cert.subject),
        issuer=name_to_dict(cert.issuer),
        serial_number=format(cert.serial_number, "x"),
        not_before=cert.not_valid_before_utc,
        not_after=cert.not_valid_after_utc,
        fingerprint_sha256=cert.fingerprint(hashes.SHA256()).hex(),
        version=cert.version.value + 1,  # x509.Version is 0-indexed
    )

    # Extract SAN
    try:
        san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        info.san = [str(name) for name in san_ext.value]
    except x509.ExtensionNotFound:
        pass

    # Check if CA
    try:
        bc_ext = cert.extensions.get_extension_for_class(x509.BasicConstraints)
        info.is_ca = bc_ext.value.ca
    except x509.ExtensionNotFound:
        pass

    # Get key size
    try:
        public_key = cert.public_key()
        info.key_size = public_key.key_size
    except Exception:
        pass

    # Get signature algorithm
    info.signature_algorithm = cert.signature_algorithm_oid._name

    return info


def check_certificate_expiry(
    cert_path: Optional[str] = None,
    cert_data: Optional[bytes] = None,
    warn_days: int = 30,
) -> Tuple[bool, int, str]:
    """
    Check certificate expiration status.

    Args:
        cert_path: Path to certificate file.
        cert_data: Raw certificate data.
        warn_days: Days before expiry to trigger warning.

    Returns:
        Tuple of (is_valid, days_remaining, status_message).
    """
    try:
        info = get_certificate_info(cert_path=cert_path, cert_data=cert_data)
    except Exception as e:
        return False, -1, f"Failed to parse certificate: {e}"

    days = info.days_until_expiry
    if days is None:
        return False, -1, "Certificate has no expiration date"

    if info.is_expired:
        return False, days, f"Certificate expired {abs(days)} days ago"
    elif days <= warn_days:
        return True, days, f"Certificate expires in {days} days (warning threshold: {warn_days})"
    else:
        return True, days, f"Certificate valid for {days} days"


# ---------------------------------------------------------------------------
# Connection Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class TLSConnectionInfo:
    """TLS connection diagnostic information."""

    connected: bool = False
    protocol: Optional[str] = None
    cipher_name: Optional[str] = None
    cipher_bits: Optional[int] = None
    cipher_version: Optional[str] = None
    peer_certificate: Optional[CertificateInfo] = None
    server_hostname: Optional[str] = None
    alpn_protocol: Optional[str] = None
    verified: bool = False
    error: Optional[str] = None


def get_connection_info(
    host: str,
    port: int = 443,
    timeout: float = 10.0,
    verify: bool = True,
    ca_bundle: Optional[str] = None,
) -> TLSConnectionInfo:
    """
    Get TLS connection information for a server.

    Args:
        host: Server hostname.
        port: Server port.
        timeout: Connection timeout in seconds.
        verify: Whether to verify server certificate.
        ca_bundle: Path to CA bundle file.

    Returns:
        TLSConnectionInfo with connection details.
    """
    info = TLSConnectionInfo(server_hostname=host)

    try:
        # Create context
        context = ssl.create_default_context()
        if ca_bundle:
            context.load_verify_locations(cafile=ca_bundle)

        if not verify:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        # Connect and wrap socket
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                info.connected = True
                info.protocol = ssock.version()
                info.verified = verify

                # Cipher info
                cipher = ssock.cipher()
                if cipher:
                    info.cipher_name = cipher[0]
                    info.cipher_version = cipher[1]
                    info.cipher_bits = cipher[2]

                # ALPN
                info.alpn_protocol = ssock.selected_alpn_protocol()

                # Peer certificate
                peer_cert_der = ssock.getpeercert(binary_form=True)
                if peer_cert_der:
                    try:
                        info.peer_certificate = get_certificate_info(
                            cert_data=peer_cert_der
                        )
                    except Exception as e:
                        logger.debug(f"Failed to parse peer certificate: {e}")

    except ssl.SSLCertVerificationError as e:
        info.error = f"Certificate verification failed: {e}"
    except ssl.SSLError as e:
        info.error = f"SSL error: {e}"
    except socket.timeout:
        info.error = "Connection timeout"
    except socket.gaierror as e:
        info.error = f"DNS resolution failed: {e}"
    except ConnectionRefusedError:
        info.error = "Connection refused"
    except Exception as e:
        info.error = f"Connection failed: {e}"

    return info


def verify_hostname(
    cert_info: CertificateInfo,
    hostname: str,
) -> Tuple[bool, str]:
    """
    Verify that certificate matches hostname.

    Args:
        cert_info: Certificate information.
        hostname: Hostname to verify.

    Returns:
        Tuple of (matches, reason).
    """
    hostname = hostname.lower()

    # Check SAN first
    for san in cert_info.san:
        san_lower = san.lower()
        if san_lower.startswith("dns:"):
            san_lower = san_lower[4:]

        if _hostname_matches(san_lower, hostname):
            return True, f"Matches SAN: {san}"

    # Check CN
    cn = cert_info.common_name
    if cn and _hostname_matches(cn.lower(), hostname):
        return True, f"Matches CN: {cn}"

    return False, f"No match found for {hostname}"


def _hostname_matches(pattern: str, hostname: str) -> bool:
    """Check if hostname matches pattern (supports wildcards)."""
    if pattern == hostname:
        return True

    # Wildcard matching
    if pattern.startswith("*."):
        # *.example.com matches foo.example.com but not foo.bar.example.com
        pattern_suffix = pattern[2:]
        if hostname.endswith("." + pattern_suffix):
            prefix = hostname[: -(len(pattern_suffix) + 1)]
            return "." not in prefix

    return False


# ---------------------------------------------------------------------------
# Hash Utilities
# ---------------------------------------------------------------------------


def hash_certificate(cert_data: bytes, algorithm: str = "sha256") -> str:
    """
    Calculate hash of certificate.

    Args:
        cert_data: Raw certificate data.
        algorithm: Hash algorithm (sha256, sha384, sha512).

    Returns:
        Hexadecimal hash string.
    """
    if algorithm == "sha256":
        return hashlib.sha256(cert_data).hexdigest()
    elif algorithm == "sha384":
        return hashlib.sha384(cert_data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(cert_data).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def get_certificate_fingerprint(
    cert_path: Optional[str] = None,
    cert_data: Optional[bytes] = None,
    algorithm: str = "sha256",
) -> str:
    """
    Get certificate fingerprint.

    Args:
        cert_path: Path to certificate file.
        cert_data: Raw certificate data.
        algorithm: Hash algorithm.

    Returns:
        Fingerprint as hex string with colons.
    """
    if cert_path:
        cert_data = Path(cert_path).read_bytes()
    elif cert_data is None:
        raise ValueError("Either cert_path or cert_data must be provided")

    # Get DER form for consistent hashing
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.serialization import Encoding

        try:
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
        except Exception:
            cert = x509.load_der_x509_certificate(cert_data, default_backend())

        der_data = cert.public_bytes(Encoding.DER)
    except ImportError:
        # Fallback: hash raw data
        der_data = cert_data

    hex_hash = hash_certificate(der_data, algorithm)

    # Format with colons
    return ":".join(hex_hash[i:i+2].upper() for i in range(0, len(hex_hash), 2))


__all__ = [
    # TLS Version
    "get_tls_version_string",
    "get_tls_version_enum",
    "is_version_secure",
    # Cipher Suites
    "parse_cipher_string",
    "format_cipher_string",
    "is_cipher_secure",
    "filter_secure_ciphers",
    "get_context_ciphers",
    # Certificates
    "CertificateInfo",
    "get_certificate_info",
    "check_certificate_expiry",
    # Connection
    "TLSConnectionInfo",
    "get_connection_info",
    "verify_hostname",
    # Hash
    "hash_certificate",
    "get_certificate_fingerprint",
]
