# -*- coding: utf-8 -*-
# =============================================================================
# GreenLang TLS Service - SSL Context Factory
# SEC-004: TLS 1.3 Configuration for All Services
# =============================================================================
"""
SSL context factory for creating properly configured SSL contexts.

Provides factory functions for creating SSL contexts with GreenLang security
requirements: TLS 1.2 minimum, strong cipher suites, certificate verification,
and proper hostname checking.

Follows the GreenLang zero-hallucination principle: all SSL operations are
deterministic using the standard library ``ssl`` module.

Functions:
    - create_ssl_context: General-purpose SSL context factory.
    - create_client_ssl_context: Client connection SSL context.
    - create_server_ssl_context: Server SSL context with certificate.
    - create_mtls_client_context: Mutual TLS client context.
    - get_default_ciphers: Get default secure cipher list.

Example:
    >>> from greenlang.infrastructure.tls_service.ssl_context import (
    ...     create_ssl_context,
    ...     create_client_ssl_context,
    ...     create_server_ssl_context,
    ... )
    >>> ctx = create_client_ssl_context()
    >>> server_ctx = create_server_ssl_context(
    ...     cert_path="/path/to/cert.pem",
    ...     key_path="/path/to/key.pem",
    ... )

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import ssl
from typing import Any, Dict, List, Literal, Optional

from greenlang.infrastructure.tls_service.ca_bundle import get_ca_bundle_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants: Secure Cipher Suites
# ---------------------------------------------------------------------------

# TLS 1.3 cipher suites (fixed, cannot be configured)
CIPHER_SUITES_TLS13: List[str] = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256",
]

# TLS 1.2 cipher suites (ECDHE + AEAD only)
CIPHER_SUITES_TLS12: List[str] = [
    "ECDHE-ECDSA-AES256-GCM-SHA384",
    "ECDHE-RSA-AES256-GCM-SHA384",
    "ECDHE-ECDSA-CHACHA20-POLY1305",
    "ECDHE-RSA-CHACHA20-POLY1305",
    "ECDHE-ECDSA-AES128-GCM-SHA256",
    "ECDHE-RSA-AES128-GCM-SHA256",
]

# Combined modern cipher suite list
CIPHER_SUITES_MODERN: List[str] = CIPHER_SUITES_TLS13 + CIPHER_SUITES_TLS12

# OpenSSL cipher string format
CIPHER_STRING_MODERN: str = ":".join(CIPHER_SUITES_TLS12)

# Context purposes
Purpose = Literal["client", "server", "http", "database", "grpc", "redis"]


# ---------------------------------------------------------------------------
# SSL Context Factory
# ---------------------------------------------------------------------------


def create_ssl_context(
    purpose: Purpose = "client",
    min_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2,
    max_version: Optional[ssl.TLSVersion] = None,
    verify: bool = True,
    check_hostname: bool = True,
    ca_bundle: Optional[str] = None,
    client_cert: Optional[str] = None,
    client_key: Optional[str] = None,
    client_key_password: Optional[str] = None,
    ciphers: Optional[str] = None,
    hostname: Optional[str] = None,
) -> ssl.SSLContext:
    """
    Create a configured SSL context.

    Args:
        purpose: Intended use of the context (affects verification mode).
        min_version: Minimum TLS version (default TLS 1.2).
        max_version: Maximum TLS version (None = no limit).
        verify: Whether to verify peer certificates.
        check_hostname: Whether to verify hostname matches certificate.
        ca_bundle: Path to CA bundle file for verification.
        client_cert: Path to client certificate for mTLS.
        client_key: Path to client private key for mTLS.
        client_key_password: Password for encrypted client key.
        ciphers: OpenSSL cipher string (None = use secure defaults).
        hostname: Expected hostname for verification.

    Returns:
        Configured SSL context.

    Raises:
        ssl.SSLError: If context configuration fails.
    """
    # Determine SSL purpose
    if purpose in ("client", "http", "database", "grpc", "redis"):
        ssl_purpose = ssl.Purpose.SERVER_AUTH
    else:
        ssl_purpose = ssl.Purpose.CLIENT_AUTH

    # Create base context
    if verify and ca_bundle is None:
        ca_bundle = get_ca_bundle_path()

    context = ssl.create_default_context(
        purpose=ssl_purpose,
        cafile=ca_bundle if verify else None,
    )

    # Set TLS version bounds
    context.minimum_version = min_version
    if max_version:
        context.maximum_version = max_version

    # Disable insecure protocols explicitly
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1

    # Additional security options
    context.options |= ssl.OP_NO_COMPRESSION  # Prevent CRIME attack
    context.options |= ssl.OP_SINGLE_DH_USE  # Fresh DH keys
    context.options |= ssl.OP_SINGLE_ECDH_USE  # Fresh ECDH keys

    # Set cipher suites (TLS 1.2 only; TLS 1.3 ciphers are fixed)
    if ciphers:
        context.set_ciphers(ciphers)
    else:
        context.set_ciphers(CIPHER_STRING_MODERN)

    # Certificate verification
    if verify:
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = check_hostname
    else:
        context.verify_mode = ssl.CERT_NONE
        context.check_hostname = False

    # Load client certificate for mTLS
    if client_cert and client_key:
        context.load_cert_chain(
            certfile=client_cert,
            keyfile=client_key,
            password=client_key_password,
        )
    elif client_cert:
        # Certificate and key in same file
        context.load_cert_chain(
            certfile=client_cert,
            password=client_key_password,
        )

    logger.debug(
        "Created SSL context",
        extra={
            "purpose": purpose,
            "min_version": min_version.name,
            "verify": verify,
            "check_hostname": check_hostname,
            "has_client_cert": client_cert is not None,
        }
    )

    return context


def create_client_ssl_context(
    verify: bool = True,
    ca_bundle: Optional[str] = None,
    min_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2,
) -> ssl.SSLContext:
    """
    Create an SSL context for client connections.

    Simplified factory for common client connection use case.

    Args:
        verify: Whether to verify server certificate.
        ca_bundle: Path to CA bundle file.
        min_version: Minimum TLS version.

    Returns:
        Configured SSL context.
    """
    return create_ssl_context(
        purpose="client",
        verify=verify,
        ca_bundle=ca_bundle,
        min_version=min_version,
    )


def create_server_ssl_context(
    cert_path: str,
    key_path: str,
    key_password: Optional[str] = None,
    verify_client: bool = False,
    ca_bundle: Optional[str] = None,
    min_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2,
    ciphers: Optional[str] = None,
) -> ssl.SSLContext:
    """
    Create an SSL context for server connections.

    Args:
        cert_path: Path to server certificate.
        key_path: Path to server private key.
        key_password: Password for encrypted key.
        verify_client: Whether to verify client certificates (mTLS).
        ca_bundle: Path to CA bundle for client verification.
        min_version: Minimum TLS version.
        ciphers: OpenSSL cipher string.

    Returns:
        Configured SSL context.

    Raises:
        ssl.SSLError: If certificate loading fails.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    # Set TLS version bounds
    context.minimum_version = min_version

    # Disable insecure protocols
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1

    # Security options
    context.options |= ssl.OP_NO_COMPRESSION
    context.options |= ssl.OP_SINGLE_DH_USE
    context.options |= ssl.OP_SINGLE_ECDH_USE
    context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE  # Server chooses cipher

    # Set cipher suites
    if ciphers:
        context.set_ciphers(ciphers)
    else:
        context.set_ciphers(CIPHER_STRING_MODERN)

    # Load server certificate
    context.load_cert_chain(
        certfile=cert_path,
        keyfile=key_path,
        password=key_password,
    )

    # Client verification (mTLS)
    if verify_client:
        if ca_bundle is None:
            ca_bundle = get_ca_bundle_path()
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(cafile=ca_bundle)
    else:
        context.verify_mode = ssl.CERT_NONE

    logger.debug(
        "Created server SSL context",
        extra={
            "min_version": min_version.name,
            "verify_client": verify_client,
        }
    )

    return context


def create_mtls_client_context(
    client_cert: str,
    client_key: str,
    ca_bundle: Optional[str] = None,
    key_password: Optional[str] = None,
    min_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2,
) -> ssl.SSLContext:
    """
    Create an SSL context for mutual TLS (mTLS) client connections.

    Args:
        client_cert: Path to client certificate.
        client_key: Path to client private key.
        ca_bundle: Path to CA bundle for server verification.
        key_password: Password for encrypted client key.
        min_version: Minimum TLS version.

    Returns:
        Configured SSL context for mTLS.
    """
    return create_ssl_context(
        purpose="client",
        verify=True,
        ca_bundle=ca_bundle,
        client_cert=client_cert,
        client_key=client_key,
        client_key_password=key_password,
        min_version=min_version,
    )


# ---------------------------------------------------------------------------
# Cipher Suite Helpers
# ---------------------------------------------------------------------------


def get_default_ciphers() -> List[str]:
    """
    Get the default secure cipher list.

    Returns:
        List of cipher suite names.
    """
    return CIPHER_SUITES_MODERN.copy()


def get_cipher_string() -> str:
    """
    Get OpenSSL cipher string for TLS 1.2.

    Returns:
        Colon-separated cipher string.
    """
    return CIPHER_STRING_MODERN


def get_enabled_ciphers(context: ssl.SSLContext) -> List[Dict[str, Any]]:
    """
    Get list of enabled ciphers in a context.

    Args:
        context: SSL context to inspect.

    Returns:
        List of cipher info dictionaries.
    """
    return list(context.get_ciphers())


def get_enabled_cipher_names(context: ssl.SSLContext) -> List[str]:
    """
    Get list of enabled cipher names in a context.

    Args:
        context: SSL context to inspect.

    Returns:
        List of cipher names.
    """
    return [c["name"] for c in context.get_ciphers()]


# ---------------------------------------------------------------------------
# Context Inspection
# ---------------------------------------------------------------------------


def get_context_info(context: ssl.SSLContext) -> Dict[str, Any]:
    """
    Get information about an SSL context configuration.

    Args:
        context: SSL context to inspect.

    Returns:
        Dictionary with context configuration details.
    """
    return {
        "protocol": context.protocol.name if hasattr(context.protocol, "name") else str(context.protocol),
        "minimum_version": context.minimum_version.name,
        "maximum_version": context.maximum_version.name if context.maximum_version else None,
        "verify_mode": context.verify_mode.name,
        "check_hostname": context.check_hostname,
        "cipher_count": len(context.get_ciphers()),
        "ciphers": get_enabled_cipher_names(context)[:5],  # First 5
        "options": _format_options(context.options),
    }


def _format_options(options: int) -> List[str]:
    """Format SSL options bitmask to list of names."""
    option_names = []

    option_map = {
        ssl.OP_NO_SSLv2: "NO_SSLv2",
        ssl.OP_NO_SSLv3: "NO_SSLv3",
        ssl.OP_NO_TLSv1: "NO_TLSv1",
        ssl.OP_NO_TLSv1_1: "NO_TLSv1_1",
        ssl.OP_NO_COMPRESSION: "NO_COMPRESSION",
        ssl.OP_SINGLE_DH_USE: "SINGLE_DH_USE",
        ssl.OP_SINGLE_ECDH_USE: "SINGLE_ECDH_USE",
    }

    if hasattr(ssl, "OP_CIPHER_SERVER_PREFERENCE"):
        option_map[ssl.OP_CIPHER_SERVER_PREFERENCE] = "CIPHER_SERVER_PREFERENCE"

    for opt_value, opt_name in option_map.items():
        if options & opt_value:
            option_names.append(opt_name)

    return option_names


__all__ = [
    # Constants
    "CIPHER_SUITES_TLS13",
    "CIPHER_SUITES_TLS12",
    "CIPHER_SUITES_MODERN",
    "CIPHER_STRING_MODERN",
    # Context factories
    "create_ssl_context",
    "create_client_ssl_context",
    "create_server_ssl_context",
    "create_mtls_client_context",
    # Cipher helpers
    "get_default_ciphers",
    "get_cipher_string",
    "get_enabled_ciphers",
    "get_enabled_cipher_names",
    # Inspection
    "get_context_info",
]
