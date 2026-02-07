# -*- coding: utf-8 -*-
# =============================================================================
# GreenLang TLS Service
# SEC-004: TLS 1.3 Configuration for All Services
# =============================================================================
"""
Centralized TLS configuration module for GreenLang applications.

This module provides a unified interface for configuring TLS/SSL connections
across all GreenLang services, including databases (PostgreSQL, Redis),
HTTP clients, gRPC, and internal service communication.

Key Features:
    - SSL context factories for various connection types
    - CA bundle management for AWS RDS, ElastiCache, and custom CAs
    - TLS metrics and monitoring via Prometheus
    - Database-specific TLS helpers (PostgreSQL, Redis)
    - HTTP client factories (httpx, aiohttp)
    - Certificate validation and expiry checking

Security Requirements (SEC-004):
    - Minimum TLS 1.2 for all connections
    - TLS 1.3 preferred where supported
    - Strong cipher suites (ECDHE + AEAD)
    - Certificate verification enabled by default
    - Hostname verification for production

Sub-modules:
    ssl_context    - SSL context factory functions
    ca_bundle      - CA bundle management and validation
    database_tls   - PostgreSQL TLS configuration
    redis_tls      - Redis/ElastiCache TLS configuration
    http_tls       - HTTP client TLS configuration
    tls_metrics    - Prometheus metrics for TLS
    utils          - Certificate utilities and diagnostics

Quick Start:
    >>> from greenlang.infrastructure.tls_service import (
    ...     TLSConfig,
    ...     create_client_ssl_context,
    ...     PostgresTLSConfig,
    ...     get_postgres_ssl_context,
    ... )
    >>> # Create general SSL context
    >>> ctx = create_client_ssl_context()
    >>>
    >>> # Create PostgreSQL SSL context
    >>> pg_config = PostgresTLSConfig(sslmode="verify-full")
    >>> pg_ctx = get_postgres_ssl_context(pg_config)

Database Connection Example:
    >>> from greenlang.infrastructure.tls_service import (
    ...     PostgresTLSConfig,
    ...     get_postgres_connection_params,
    ...     build_postgres_dsn,
    ... )
    >>> config = PostgresTLSConfig()
    >>> dsn = build_postgres_dsn(
    ...     host="db.example.com",
    ...     database="myapp",
    ...     user="appuser",
    ...     password="secret",
    ...     config=config,
    ... )

Redis Connection Example:
    >>> from greenlang.infrastructure.tls_service import (
    ...     RedisTLSConfig,
    ...     get_redis_connection_kwargs,
    ...     create_redis_tls_url,
    ... )
    >>> config = RedisTLSConfig()
    >>> kwargs = get_redis_connection_kwargs(config)
    >>> url = create_redis_tls_url("redis.example.com", password="secret")

HTTP Client Example:
    >>> from greenlang.infrastructure.tls_service import create_httpx_client
    >>> async with create_httpx_client() as client:
    ...     response = await client.get("https://api.example.com")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version Info
# ---------------------------------------------------------------------------

__version__ = "1.0.0"
__author__ = "GreenLang Framework Team"


# ---------------------------------------------------------------------------
# TLS Version Enum
# ---------------------------------------------------------------------------


class TLSVersion(str, Enum):
    """Supported TLS versions.

    GreenLang requires TLS 1.2 minimum, with TLS 1.3 preferred.
    """

    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Cipher Suite Constants
# ---------------------------------------------------------------------------

# TLS 1.3 cipher suites (fixed by the protocol, cannot be configured)
CIPHER_SUITES_TLS13: List[str] = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_GCM_SHA256",
]

# TLS 1.2 cipher suites (ECDHE + AEAD only for forward secrecy)
CIPHER_SUITES_TLS12: List[str] = [
    "ECDHE-ECDSA-AES256-GCM-SHA384",
    "ECDHE-RSA-AES256-GCM-SHA384",
    "ECDHE-ECDSA-CHACHA20-POLY1305",
    "ECDHE-RSA-CHACHA20-POLY1305",
    "ECDHE-ECDSA-AES128-GCM-SHA256",
    "ECDHE-RSA-AES128-GCM-SHA256",
]

# Combined modern cipher suite list (TLS 1.3 + TLS 1.2)
CIPHER_SUITES_MODERN: List[str] = CIPHER_SUITES_TLS13 + CIPHER_SUITES_TLS12

# OpenSSL cipher string format (for TLS 1.2 configuration)
CIPHER_STRING_MODERN: str = ":".join(CIPHER_SUITES_TLS12)


# ---------------------------------------------------------------------------
# Global TLS Configuration
# ---------------------------------------------------------------------------


@dataclass
class TLSConfig:
    """Global TLS configuration for GreenLang applications.

    Attributes:
        min_version: Minimum TLS version (TLS 1.2 or TLS 1.3).
        verify_certificates: Whether to verify server certificates.
        check_hostname: Whether to verify hostname matches certificate.
        cipher_suites: List of allowed cipher suites.
        ca_bundle_path: Path to CA bundle for verification.
        client_cert_path: Path to client certificate for mTLS.
        client_key_path: Path to client private key for mTLS.
        hsts_enabled: Enable HSTS headers for HTTP responses.
        hsts_max_age: HSTS max-age in seconds (default 1 year).
        hsts_include_subdomains: Include subdomains in HSTS.
        hsts_preload: Enable HSTS preload.

    Example:
        >>> config = TLSConfig(
        ...     min_version=TLSVersion.TLS_1_3,
        ...     verify_certificates=True,
        ...     hsts_enabled=True,
        ... )
    """

    min_version: TLSVersion = TLSVersion.TLS_1_2
    verify_certificates: bool = True
    check_hostname: bool = True
    cipher_suites: List[str] = field(default_factory=lambda: CIPHER_SUITES_MODERN.copy())
    ca_bundle_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False

    @property
    def cipher_string(self) -> str:
        """Get cipher suites as colon-separated string for OpenSSL."""
        # Filter to TLS 1.2 ciphers only (TLS 1.3 uses fixed ciphers)
        tls12_ciphers = [c for c in self.cipher_suites if not c.startswith("TLS_")]
        return ":".join(tls12_ciphers)

    @property
    def hsts_header(self) -> str:
        """Get HSTS header value for HTTP responses.

        Returns:
            HSTS header value or empty string if disabled.
        """
        if not self.hsts_enabled:
            return ""

        parts = [f"max-age={self.hsts_max_age}"]
        if self.hsts_include_subdomains:
            parts.append("includeSubDomains")
        if self.hsts_preload:
            parts.append("preload")

        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Global Configuration Management
# ---------------------------------------------------------------------------

_default_config: Optional[TLSConfig] = None


def get_default_config() -> TLSConfig:
    """Get the default TLS configuration.

    Returns:
        Global TLSConfig instance.
    """
    global _default_config
    if _default_config is None:
        _default_config = TLSConfig()
    return _default_config


def set_default_config(config: TLSConfig) -> None:
    """Set the default TLS configuration.

    Args:
        config: TLSConfig to use as default.
    """
    global _default_config
    _default_config = config
    logger.info(
        "Updated default TLS config",
        extra={"min_version": config.min_version.value},
    )


# ---------------------------------------------------------------------------
# Public API Imports
# ---------------------------------------------------------------------------

# SSL Context Factory
from greenlang.infrastructure.tls_service.ssl_context import (  # noqa: E402
    create_ssl_context,
    create_client_ssl_context,
    create_server_ssl_context,
    create_mtls_client_context,
    get_default_ciphers,
    get_cipher_string,
    get_enabled_ciphers,
    get_enabled_cipher_names,
    get_context_info,
)

# CA Bundle Management
from greenlang.infrastructure.tls_service.ca_bundle import (  # noqa: E402
    get_ca_bundle_path,
    get_aws_rds_ca_bundle,
    get_aws_elasticache_ca_bundle,
    get_system_ca_bundle,
    refresh_ca_bundle,
    refresh_ca_bundle_sync,
    validate_ca_bundle,
    get_ca_bundle_info,
)

# PostgreSQL TLS
from greenlang.infrastructure.tls_service.database_tls import (  # noqa: E402
    PostgresTLSConfig,
    PostgresSSLMode,
    get_postgres_ssl_context,
    get_postgres_connection_params,
    get_postgres_dsn_suffix,
    build_postgres_dsn,
    verify_postgres_tls_connection,
    verify_postgres_tls_connection_sync,
)

# Redis TLS
from greenlang.infrastructure.tls_service.redis_tls import (  # noqa: E402
    RedisTLSConfig,
    RedisCertReqs,
    get_redis_ssl_context,
    get_redis_connection_kwargs,
    get_redis_connection_kwargs_with_context,
    create_redis_tls_url,
    parse_redis_url,
    verify_redis_tls_connection,
    verify_redis_tls_connection_sync,
    get_elasticache_config,
)

# HTTP TLS
from greenlang.infrastructure.tls_service.http_tls import (  # noqa: E402
    HTTPTLSConfig,
    get_http_ssl_context,
    create_httpx_client,
    create_httpx_client_sync,
    create_aiohttp_connector,
    create_aiohttp_session,
    fetch_json,
    post_json,
)

# TLS Metrics
from greenlang.infrastructure.tls_service.tls_metrics import (  # noqa: E402
    TLSMetrics,
    get_tls_metrics,
)

# Utilities
from greenlang.infrastructure.tls_service.utils import (  # noqa: E402
    get_tls_version_string,
    get_tls_version_enum,
    is_version_secure,
    parse_cipher_string,
    format_cipher_string,
    is_cipher_secure,
    filter_secure_ciphers,
    get_context_ciphers,
    CertificateInfo,
    get_certificate_info,
    check_certificate_expiry,
    TLSConnectionInfo,
    get_connection_info,
    verify_hostname,
    hash_certificate,
    get_certificate_fingerprint,
)

# Exporter (Certificate Scanner and Metrics)
from greenlang.infrastructure.tls_service.exporter import (  # noqa: E402
    CertificateInfo as ExporterCertificateInfo,
    TLSCertificateScanner,
    TLSMetricsExporter,
    get_certificate_info as get_endpoint_certificate_info,
    scan_certificate_sync,
    get_metrics_exporter,
    DEFAULT_ENDPOINTS,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Enums and Constants
    "TLSVersion",
    "CIPHER_SUITES_TLS13",
    "CIPHER_SUITES_TLS12",
    "CIPHER_SUITES_MODERN",
    "CIPHER_STRING_MODERN",
    # Configuration
    "TLSConfig",
    "get_default_config",
    "set_default_config",
    # SSL Context
    "create_ssl_context",
    "create_client_ssl_context",
    "create_server_ssl_context",
    "create_mtls_client_context",
    "get_default_ciphers",
    "get_cipher_string",
    "get_enabled_ciphers",
    "get_enabled_cipher_names",
    "get_context_info",
    # CA Bundle
    "get_ca_bundle_path",
    "get_aws_rds_ca_bundle",
    "get_aws_elasticache_ca_bundle",
    "get_system_ca_bundle",
    "refresh_ca_bundle",
    "refresh_ca_bundle_sync",
    "validate_ca_bundle",
    "get_ca_bundle_info",
    # PostgreSQL
    "PostgresTLSConfig",
    "PostgresSSLMode",
    "get_postgres_ssl_context",
    "get_postgres_connection_params",
    "get_postgres_dsn_suffix",
    "build_postgres_dsn",
    "verify_postgres_tls_connection",
    "verify_postgres_tls_connection_sync",
    # Redis
    "RedisTLSConfig",
    "RedisCertReqs",
    "get_redis_ssl_context",
    "get_redis_connection_kwargs",
    "get_redis_connection_kwargs_with_context",
    "create_redis_tls_url",
    "parse_redis_url",
    "verify_redis_tls_connection",
    "verify_redis_tls_connection_sync",
    "get_elasticache_config",
    # HTTP
    "HTTPTLSConfig",
    "get_http_ssl_context",
    "create_httpx_client",
    "create_httpx_client_sync",
    "create_aiohttp_connector",
    "create_aiohttp_session",
    "fetch_json",
    "post_json",
    # Metrics
    "TLSMetrics",
    "get_tls_metrics",
    # Utilities
    "get_tls_version_string",
    "get_tls_version_enum",
    "is_version_secure",
    "parse_cipher_string",
    "format_cipher_string",
    "is_cipher_secure",
    "filter_secure_ciphers",
    "get_context_ciphers",
    "CertificateInfo",
    "get_certificate_info",
    "check_certificate_expiry",
    "TLSConnectionInfo",
    "get_connection_info",
    "verify_hostname",
    "hash_certificate",
    "get_certificate_fingerprint",
    # Exporter
    "TLSCertificateScanner",
    "TLSMetricsExporter",
    "get_endpoint_certificate_info",
    "scan_certificate_sync",
    "get_metrics_exporter",
    "DEFAULT_ENDPOINTS",
]
