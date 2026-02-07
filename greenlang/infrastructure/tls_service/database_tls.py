# -*- coding: utf-8 -*-
# =============================================================================
# GreenLang TLS Service - Database TLS Configuration
# SEC-004: TLS 1.3 Configuration for All Services
# =============================================================================
"""
Database TLS configuration for PostgreSQL connections.

Provides SSL context factories and connection parameter helpers for
PostgreSQL (RDS Aurora) connections with TLS.

Follows the GreenLang zero-hallucination principle: all TLS operations are
deterministic using the standard library ``ssl`` module.

Classes:
    - PostgresTLSConfig: Configuration dataclass for PostgreSQL TLS.

Functions:
    - get_postgres_ssl_context: Create SSL context for PostgreSQL.
    - get_postgres_connection_params: Get psycopg connection parameters.
    - get_postgres_dsn_suffix: Get DSN query string for SSL.
    - verify_postgres_tls_connection: Verify TLS connection to PostgreSQL.

Example:
    >>> from greenlang.infrastructure.tls_service.database_tls import (
    ...     PostgresTLSConfig,
    ...     get_postgres_ssl_context,
    ...     get_postgres_connection_params,
    ... )
    >>> config = PostgresTLSConfig(sslmode="verify-full")
    >>> ssl_ctx = get_postgres_ssl_context(config)
    >>> conn_params = get_postgres_connection_params(config)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import ssl
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from greenlang.infrastructure.tls_service.ca_bundle import (
    AWS_RDS_CA_BUNDLE_PATHS,
    get_ca_bundle_path,
)
from greenlang.infrastructure.tls_service.ssl_context import (
    CIPHER_STRING_MODERN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PostgreSQL SSL Modes
# ---------------------------------------------------------------------------

# PostgreSQL sslmode options (from least to most secure)
PostgresSSLMode = Literal[
    "disable",      # No SSL
    "allow",        # Try non-SSL first, then SSL
    "prefer",       # Try SSL first, then non-SSL
    "require",      # Require SSL, no certificate verification
    "verify-ca",    # Require SSL, verify CA only
    "verify-full",  # Require SSL, verify CA and hostname
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _find_ca_bundle(paths: List[str]) -> Optional[str]:
    """Find the first existing CA bundle from a list of paths."""
    for path in paths:
        expanded = os.path.expanduser(os.path.expandvars(path))
        if Path(expanded).exists():
            return expanded
    return None


@dataclass
class PostgresTLSConfig:
    """PostgreSQL TLS connection configuration.

    Attributes:
        sslmode: PostgreSQL SSL mode (disable, allow, prefer, require,
            verify-ca, verify-full). Default is "verify-full" for production.
        sslrootcert: Path to CA certificate bundle for verification.
            Defaults to AWS RDS CA bundle if available.
        sslcert: Path to client certificate for mTLS authentication.
        sslkey: Path to client private key for mTLS authentication.
        sslpassword: Password for encrypted client key.
        min_tls_version: Minimum TLS version ("TLSv1.2" or "TLSv1.3").
        check_hostname: Whether to verify hostname matches certificate.
            Only used with "verify-full" mode.
        ciphers: OpenSSL cipher string. Defaults to secure modern ciphers.

    Example:
        >>> config = PostgresTLSConfig(
        ...     sslmode="verify-full",
        ...     sslrootcert="/etc/ssl/certs/rds-combined-ca-bundle.pem",
        ... )
    """

    sslmode: PostgresSSLMode = "verify-full"
    sslrootcert: Optional[str] = None
    sslcert: Optional[str] = None
    sslkey: Optional[str] = None
    sslpassword: Optional[str] = None
    min_tls_version: str = "TLSv1.2"
    check_hostname: bool = True
    ciphers: Optional[str] = None

    def __post_init__(self) -> None:
        """Auto-detect CA bundle if not specified."""
        if self.sslrootcert is None and self.sslmode in ("verify-ca", "verify-full"):
            # Try RDS-specific bundle first
            self.sslrootcert = _find_ca_bundle(AWS_RDS_CA_BUNDLE_PATHS)
            if self.sslrootcert is None:
                # Fall back to general CA bundle
                try:
                    self.sslrootcert = get_ca_bundle_path(
                        prefer_aws=True, service_type="rds"
                    )
                except FileNotFoundError:
                    logger.warning("No CA bundle found for PostgreSQL TLS")

    @classmethod
    def from_env(cls, prefix: str = "POSTGRES") -> "PostgresTLSConfig":
        """Create configuration from environment variables.

        Environment variables:
            {PREFIX}_SSLMODE: SSL mode
            {PREFIX}_SSLROOTCERT: CA bundle path
            {PREFIX}_SSLCERT: Client certificate path
            {PREFIX}_SSLKEY: Client key path
            {PREFIX}_MIN_TLS_VERSION: Minimum TLS version

        Args:
            prefix: Environment variable prefix.

        Returns:
            PostgresTLSConfig from environment.
        """
        return cls(
            sslmode=os.environ.get(f"{prefix}_SSLMODE", "verify-full"),  # type: ignore
            sslrootcert=os.environ.get(f"{prefix}_SSLROOTCERT"),
            sslcert=os.environ.get(f"{prefix}_SSLCERT"),
            sslkey=os.environ.get(f"{prefix}_SSLKEY"),
            min_tls_version=os.environ.get(f"{prefix}_MIN_TLS_VERSION", "TLSv1.2"),
        )

    @property
    def requires_verification(self) -> bool:
        """Check if this mode requires certificate verification."""
        return self.sslmode in ("verify-ca", "verify-full")

    @property
    def requires_ssl(self) -> bool:
        """Check if this mode requires SSL."""
        return self.sslmode not in ("disable", "allow")


# ---------------------------------------------------------------------------
# SSL Context Factory
# ---------------------------------------------------------------------------


def get_postgres_ssl_context(
    config: Optional[PostgresTLSConfig] = None,
    verify: bool = True,
) -> ssl.SSLContext:
    """
    Create an SSL context for PostgreSQL connections.

    This context is suitable for use with psycopg's ssl_context parameter.

    Args:
        config: PostgreSQL TLS configuration.
        verify: Whether to verify server certificate.
            Overrides config.sslmode if False.

    Returns:
        Configured SSL context.

    Example:
        >>> config = PostgresTLSConfig(sslmode="verify-full")
        >>> ctx = get_postgres_ssl_context(config)
        >>> # Use with psycopg
        >>> conn = await psycopg.AsyncConnection.connect(
        ...     conninfo, ssl_context=ctx
        ... )
    """
    if config is None:
        config = PostgresTLSConfig()

    # Override verify based on sslmode if not explicitly False
    if verify and not config.requires_verification:
        verify = False

    # Create context for client connections
    context = ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        cafile=config.sslrootcert if verify else None,
    )

    # Set minimum TLS version
    if config.min_tls_version == "TLSv1.3":
        context.minimum_version = ssl.TLSVersion.TLSv1_3
    else:
        context.minimum_version = ssl.TLSVersion.TLSv1_2

    # Disable insecure options
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1

    # Set cipher suites
    if config.ciphers:
        context.set_ciphers(config.ciphers)
    else:
        context.set_ciphers(CIPHER_STRING_MODERN)

    # Certificate verification
    if verify:
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = config.check_hostname and config.sslmode == "verify-full"
    else:
        context.verify_mode = ssl.CERT_NONE
        context.check_hostname = False

    # Client certificate (mTLS) if provided
    if config.sslcert and config.sslkey:
        context.load_cert_chain(
            certfile=config.sslcert,
            keyfile=config.sslkey,
            password=config.sslpassword,
        )
    elif config.sslcert:
        # Cert and key in same file
        context.load_cert_chain(
            certfile=config.sslcert,
            password=config.sslpassword,
        )

    logger.debug(
        "Created PostgreSQL SSL context",
        extra={
            "sslmode": config.sslmode,
            "min_version": config.min_tls_version,
            "verify": verify,
            "ca_bundle": config.sslrootcert,
            "has_client_cert": config.sslcert is not None,
        }
    )

    return context


# ---------------------------------------------------------------------------
# Connection Parameters
# ---------------------------------------------------------------------------


def get_postgres_connection_params(
    config: Optional[PostgresTLSConfig] = None,
) -> Dict[str, Any]:
    """
    Get PostgreSQL connection parameters for SSL.

    For use with psycopg connection strings or parameter dicts.
    These parameters are passed to libpq.

    Args:
        config: PostgreSQL TLS configuration.

    Returns:
        Dictionary of SSL-related connection parameters.

    Example:
        >>> config = PostgresTLSConfig(sslmode="verify-full")
        >>> params = get_postgres_connection_params(config)
        >>> # Use with psycopg
        >>> conn = await psycopg.AsyncConnection.connect(**params)
    """
    if config is None:
        config = PostgresTLSConfig()

    params: Dict[str, Any] = {
        "sslmode": config.sslmode,
    }

    if config.sslrootcert:
        params["sslrootcert"] = config.sslrootcert

    if config.sslcert:
        params["sslcert"] = config.sslcert

    if config.sslkey:
        params["sslkey"] = config.sslkey

    return params


def get_postgres_dsn_suffix(
    config: Optional[PostgresTLSConfig] = None,
) -> str:
    """
    Get DSN suffix for PostgreSQL connection string.

    Returns a query string that can be appended to a DSN.

    Args:
        config: PostgreSQL TLS configuration.

    Returns:
        Query string (e.g., "?sslmode=verify-full&sslrootcert=/path/to/ca.pem").

    Example:
        >>> config = PostgresTLSConfig(sslmode="verify-full")
        >>> suffix = get_postgres_dsn_suffix(config)
        >>> dsn = f"postgresql://user:pass@host:5432/db{suffix}"
    """
    params = get_postgres_connection_params(config)
    if not params:
        return ""

    parts = [f"{k}={v}" for k, v in params.items() if v is not None]
    return "?" + "&".join(parts) if parts else ""


def build_postgres_dsn(
    host: str,
    port: int = 5432,
    database: str = "postgres",
    user: str = "postgres",
    password: Optional[str] = None,
    config: Optional[PostgresTLSConfig] = None,
) -> str:
    """
    Build a complete PostgreSQL DSN with TLS parameters.

    Args:
        host: PostgreSQL hostname.
        port: PostgreSQL port.
        database: Database name.
        user: Username.
        password: Password.
        config: TLS configuration.

    Returns:
        Complete DSN string.

    Example:
        >>> dsn = build_postgres_dsn(
        ...     host="db.example.com",
        ...     database="myapp",
        ...     user="appuser",
        ...     password="secret",
        ... )
    """
    if password:
        auth = f"{user}:{password}"
    else:
        auth = user

    base_dsn = f"postgresql://{auth}@{host}:{port}/{database}"
    return base_dsn + get_postgres_dsn_suffix(config)


# ---------------------------------------------------------------------------
# Connection Verification
# ---------------------------------------------------------------------------


async def verify_postgres_tls_connection(
    host: str,
    port: int = 5432,
    config: Optional[PostgresTLSConfig] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Verify TLS connection to PostgreSQL server.

    Performs a TLS handshake without authenticating to the database.
    Useful for testing TLS configuration before application starts.

    Args:
        host: PostgreSQL hostname.
        port: PostgreSQL port.
        config: TLS configuration.
        timeout: Connection timeout in seconds.

    Returns:
        Dictionary with connection info:
        - connected: Whether connection succeeded
        - protocol: TLS protocol version
        - cipher: Cipher suite info
        - peer_cert: Peer certificate details
        - verified: Whether certificate was verified
        - error: Error message if connection failed

    Example:
        >>> result = await verify_postgres_tls_connection("db.example.com")
        >>> if result["connected"]:
        ...     print(f"Connected with {result['protocol']}")
    """
    if config is None:
        config = PostgresTLSConfig()

    result: Dict[str, Any] = {
        "connected": False,
        "protocol": None,
        "cipher": None,
        "peer_cert": None,
        "verified": False,
        "error": None,
    }

    try:
        context = get_postgres_ssl_context(config)

        # PostgreSQL uses STARTTLS, but for verification we can
        # attempt a raw TLS connection to see if the port accepts TLS.
        # Note: This won't work with PostgreSQL's actual protocol,
        # but can verify basic TLS configuration.

        with socket.create_connection((host, port), timeout=timeout) as sock:
            # Send PostgreSQL SSLRequest message
            # Length (8 bytes) + SSL code (80877103)
            ssl_request = b'\x00\x00\x00\x08\x04\xd2\x16/'
            sock.sendall(ssl_request)

            # Read response (1 byte: 'S' for SSL OK, 'N' for no SSL)
            response = sock.recv(1)

            if response == b'S':
                # Server accepts SSL, upgrade connection
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    result["connected"] = True
                    result["protocol"] = ssock.version()
                    result["cipher"] = ssock.cipher()
                    result["peer_cert"] = ssock.getpeercert()
                    result["verified"] = config.sslmode in ("verify-ca", "verify-full")
            elif response == b'N':
                result["error"] = "Server does not support SSL"
            else:
                result["error"] = f"Unexpected response: {response!r}"

    except ssl.SSLCertVerificationError as e:
        result["error"] = f"Certificate verification failed: {e}"
    except ssl.SSLError as e:
        result["error"] = f"SSL error: {e}"
    except socket.timeout:
        result["error"] = "Connection timeout"
    except socket.gaierror as e:
        result["error"] = f"DNS resolution failed: {e}"
    except ConnectionRefusedError:
        result["error"] = "Connection refused"
    except Exception as e:
        result["error"] = f"Connection failed: {e}"

    return result


def verify_postgres_tls_connection_sync(
    host: str,
    port: int = 5432,
    config: Optional[PostgresTLSConfig] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Synchronous version of verify_postgres_tls_connection.

    Args:
        host: PostgreSQL hostname.
        port: PostgreSQL port.
        config: TLS configuration.
        timeout: Connection timeout.

    Returns:
        Dictionary with connection info.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        verify_postgres_tls_connection(host, port, config, timeout)
    )


__all__ = [
    # Types
    "PostgresSSLMode",
    # Configuration
    "PostgresTLSConfig",
    # SSL Context
    "get_postgres_ssl_context",
    # Connection Parameters
    "get_postgres_connection_params",
    "get_postgres_dsn_suffix",
    "build_postgres_dsn",
    # Verification
    "verify_postgres_tls_connection",
    "verify_postgres_tls_connection_sync",
]
