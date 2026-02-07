# -*- coding: utf-8 -*-
# =============================================================================
# GreenLang TLS Service - Redis TLS Configuration
# SEC-004: TLS 1.3 Configuration for All Services
# =============================================================================
"""
Redis TLS configuration for ElastiCache connections.

Provides SSL context factories and connection helpers for Redis/ElastiCache
connections with TLS.

Follows the GreenLang zero-hallucination principle: all TLS operations are
deterministic using the standard library ``ssl`` module.

Classes:
    - RedisTLSConfig: Configuration dataclass for Redis TLS.

Functions:
    - get_redis_ssl_context: Create SSL context for Redis.
    - get_redis_connection_kwargs: Get redis-py connection kwargs.
    - create_redis_tls_url: Create rediss:// URL.
    - verify_redis_tls_connection: Verify TLS connection to Redis.

Example:
    >>> from greenlang.infrastructure.tls_service.redis_tls import (
    ...     RedisTLSConfig,
    ...     get_redis_ssl_context,
    ...     get_redis_connection_kwargs,
    ... )
    >>> config = RedisTLSConfig()
    >>> ssl_ctx = get_redis_ssl_context(config)
    >>> kwargs = get_redis_connection_kwargs(config)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import socket
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from greenlang.infrastructure.tls_service.ca_bundle import (
    AWS_ELASTICACHE_CA_PATHS,
    get_ca_bundle_path,
)
from greenlang.infrastructure.tls_service.ssl_context import (
    CIPHER_STRING_MODERN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redis SSL Certificate Requirements
# ---------------------------------------------------------------------------

RedisCertReqs = Literal["none", "optional", "required"]


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
class RedisTLSConfig:
    """Redis/ElastiCache TLS connection configuration.

    Attributes:
        ssl: Whether to use SSL/TLS. Default True for secure connections.
        ssl_cert_reqs: Certificate verification requirement.
            - "required": Verify server certificate (default for production)
            - "optional": Verify if certificate provided
            - "none": No verification (not recommended)
        ssl_ca_certs: Path to CA certificate bundle for verification.
            Defaults to AWS ElastiCache CA if available.
        ssl_certfile: Path to client certificate for mTLS.
        ssl_keyfile: Path to client private key for mTLS.
        ssl_password: Password for encrypted client key.
        ssl_check_hostname: Whether to verify hostname matches certificate.
        min_tls_version: Minimum TLS version ("TLSv1.2" or "TLSv1.3").
        ciphers: OpenSSL cipher string. Defaults to secure modern ciphers.

    Example:
        >>> config = RedisTLSConfig(
        ...     ssl_cert_reqs="required",
        ...     ssl_ca_certs="/etc/ssl/certs/AmazonRootCA1.pem",
        ... )
    """

    ssl: bool = True
    ssl_cert_reqs: RedisCertReqs = "required"
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_password: Optional[str] = None
    ssl_check_hostname: bool = True
    min_tls_version: str = "TLSv1.2"
    ciphers: Optional[str] = None

    def __post_init__(self) -> None:
        """Auto-detect CA bundle if not specified."""
        if self.ssl_ca_certs is None and self.ssl_cert_reqs != "none":
            # Try ElastiCache-specific bundle first
            self.ssl_ca_certs = _find_ca_bundle(AWS_ELASTICACHE_CA_PATHS)
            if self.ssl_ca_certs is None:
                # Fall back to general CA bundle
                try:
                    self.ssl_ca_certs = get_ca_bundle_path(
                        prefer_aws=True, service_type="elasticache"
                    )
                except FileNotFoundError:
                    logger.warning("No CA bundle found for Redis TLS")

    @classmethod
    def from_env(cls, prefix: str = "REDIS") -> "RedisTLSConfig":
        """Create configuration from environment variables.

        Environment variables:
            {PREFIX}_SSL: Enable SSL ("true"/"false")
            {PREFIX}_SSL_CERT_REQS: Certificate requirement
            {PREFIX}_SSL_CA_CERTS: CA bundle path
            {PREFIX}_SSL_CERTFILE: Client certificate path
            {PREFIX}_SSL_KEYFILE: Client key path
            {PREFIX}_MIN_TLS_VERSION: Minimum TLS version

        Args:
            prefix: Environment variable prefix.

        Returns:
            RedisTLSConfig from environment.
        """
        ssl_str = os.environ.get(f"{prefix}_SSL", "true").lower()
        ssl_enabled = ssl_str in ("true", "1", "yes")

        return cls(
            ssl=ssl_enabled,
            ssl_cert_reqs=os.environ.get(f"{prefix}_SSL_CERT_REQS", "required"),  # type: ignore
            ssl_ca_certs=os.environ.get(f"{prefix}_SSL_CA_CERTS"),
            ssl_certfile=os.environ.get(f"{prefix}_SSL_CERTFILE"),
            ssl_keyfile=os.environ.get(f"{prefix}_SSL_KEYFILE"),
            min_tls_version=os.environ.get(f"{prefix}_MIN_TLS_VERSION", "TLSv1.2"),
        )

    @property
    def requires_verification(self) -> bool:
        """Check if this config requires certificate verification."""
        return self.ssl_cert_reqs == "required"


# ---------------------------------------------------------------------------
# SSL Context Factory
# ---------------------------------------------------------------------------


def get_redis_ssl_context(
    config: Optional[RedisTLSConfig] = None,
    verify: bool = True,
) -> ssl.SSLContext:
    """
    Create an SSL context for Redis connections.

    This context is suitable for use with redis-py's ssl_context parameter.

    Args:
        config: Redis TLS configuration.
        verify: Whether to verify server certificate.
            Overrides config.ssl_cert_reqs if False.

    Returns:
        Configured SSL context.

    Example:
        >>> config = RedisTLSConfig(ssl_cert_reqs="required")
        >>> ctx = get_redis_ssl_context(config)
        >>> # Use with redis-py
        >>> client = redis.Redis(host="redis.example.com", ssl_context=ctx)
    """
    if config is None:
        config = RedisTLSConfig()

    # Override verify based on ssl_cert_reqs if not explicitly False
    if verify and config.ssl_cert_reqs == "none":
        verify = False

    # Create context for client connections
    context = ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        cafile=config.ssl_ca_certs if verify else None,
    )

    # Set minimum TLS version
    if config.min_tls_version == "TLSv1.3":
        context.minimum_version = ssl.TLSVersion.TLSv1_3
    else:
        context.minimum_version = ssl.TLSVersion.TLSv1_2

    # Disable insecure protocols
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
    if verify and config.ssl_cert_reqs == "required":
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = config.ssl_check_hostname
    elif config.ssl_cert_reqs == "optional":
        context.verify_mode = ssl.CERT_OPTIONAL
        context.check_hostname = False
    else:
        context.verify_mode = ssl.CERT_NONE
        context.check_hostname = False

    # Client certificate (mTLS) if provided
    if config.ssl_certfile and config.ssl_keyfile:
        context.load_cert_chain(
            certfile=config.ssl_certfile,
            keyfile=config.ssl_keyfile,
            password=config.ssl_password,
        )
    elif config.ssl_certfile:
        # Cert and key in same file
        context.load_cert_chain(
            certfile=config.ssl_certfile,
            password=config.ssl_password,
        )

    logger.debug(
        "Created Redis SSL context",
        extra={
            "ssl_cert_reqs": config.ssl_cert_reqs,
            "min_version": config.min_tls_version,
            "verify": verify,
            "ca_bundle": config.ssl_ca_certs,
            "has_client_cert": config.ssl_certfile is not None,
        }
    )

    return context


# ---------------------------------------------------------------------------
# Connection Parameters
# ---------------------------------------------------------------------------


def get_redis_connection_kwargs(
    config: Optional[RedisTLSConfig] = None,
) -> Dict[str, Any]:
    """
    Get Redis connection kwargs for SSL.

    For use with redis-py client initialization. Returns kwargs that
    can be unpacked into redis.Redis() or redis.StrictRedis().

    Args:
        config: Redis TLS configuration.

    Returns:
        Dictionary of SSL-related connection kwargs.

    Example:
        >>> config = RedisTLSConfig()
        >>> kwargs = get_redis_connection_kwargs(config)
        >>> client = redis.Redis(host="redis.example.com", **kwargs)
    """
    if config is None:
        config = RedisTLSConfig()

    if not config.ssl:
        return {}

    kwargs: Dict[str, Any] = {
        "ssl": True,
        "ssl_cert_reqs": config.ssl_cert_reqs,
    }

    if config.ssl_ca_certs:
        kwargs["ssl_ca_certs"] = config.ssl_ca_certs

    if config.ssl_certfile:
        kwargs["ssl_certfile"] = config.ssl_certfile

    if config.ssl_keyfile:
        kwargs["ssl_keyfile"] = config.ssl_keyfile

    kwargs["ssl_check_hostname"] = config.ssl_check_hostname

    return kwargs


def get_redis_connection_kwargs_with_context(
    config: Optional[RedisTLSConfig] = None,
) -> Dict[str, Any]:
    """
    Get Redis connection kwargs with pre-built SSL context.

    Uses ssl_context parameter instead of individual SSL options.
    This is the recommended approach for newer redis-py versions.

    Args:
        config: Redis TLS configuration.

    Returns:
        Dictionary with ssl_context.

    Example:
        >>> config = RedisTLSConfig()
        >>> kwargs = get_redis_connection_kwargs_with_context(config)
        >>> client = redis.Redis(host="redis.example.com", **kwargs)
    """
    if config is None:
        config = RedisTLSConfig()

    if not config.ssl:
        return {}

    return {
        "ssl": True,
        "ssl_context": get_redis_ssl_context(config),
    }


# ---------------------------------------------------------------------------
# URL Builder
# ---------------------------------------------------------------------------


def create_redis_tls_url(
    host: str,
    port: int = 6379,
    password: Optional[str] = None,
    db: int = 0,
    use_tls: bool = True,
    username: Optional[str] = None,
) -> str:
    """
    Create Redis URL with TLS scheme.

    Args:
        host: Redis hostname.
        port: Redis port.
        password: Redis password.
        db: Redis database number.
        use_tls: Whether to use TLS (rediss:// vs redis://).
        username: Redis username (for ACL auth).

    Returns:
        Redis connection URL.

    Example:
        >>> url = create_redis_tls_url(
        ...     host="redis.example.com",
        ...     password="secret",
        ... )
        >>> # Returns: rediss://:secret@redis.example.com:6379/0
    """
    scheme = "rediss" if use_tls else "redis"

    if username and password:
        auth = f"{username}:{password}@"
    elif password:
        auth = f":{password}@"
    else:
        auth = ""

    return f"{scheme}://{auth}{host}:{port}/{db}"


def parse_redis_url(url: str) -> Dict[str, Any]:
    """
    Parse Redis URL into components.

    Args:
        url: Redis URL (redis:// or rediss://).

    Returns:
        Dictionary with parsed components:
        - scheme: "redis" or "rediss"
        - host: Hostname
        - port: Port number
        - password: Password (if present)
        - username: Username (if present)
        - db: Database number
        - use_tls: Whether URL uses TLS

    Example:
        >>> parsed = parse_redis_url("rediss://:secret@redis.example.com:6379/0")
        >>> parsed["use_tls"]
        True
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)

    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 6379,
        "password": parsed.password,
        "username": parsed.username,
        "db": int(parsed.path.lstrip("/") or 0),
        "use_tls": parsed.scheme == "rediss",
    }


# ---------------------------------------------------------------------------
# Connection Verification
# ---------------------------------------------------------------------------


async def verify_redis_tls_connection(
    host: str,
    port: int = 6379,
    config: Optional[RedisTLSConfig] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Verify TLS connection to Redis server.

    Performs a TLS handshake without authenticating to Redis.
    Useful for testing TLS configuration before application starts.

    Args:
        host: Redis hostname.
        port: Redis port.
        config: TLS configuration.
        timeout: Connection timeout in seconds.

    Returns:
        Dictionary with connection info:
        - connected: Whether connection succeeded
        - protocol: TLS protocol version
        - cipher: Cipher suite info
        - peer_cert: Peer certificate details
        - error: Error message if connection failed

    Example:
        >>> result = await verify_redis_tls_connection("redis.example.com")
        >>> if result["connected"]:
        ...     print(f"Connected with {result['protocol']}")
    """
    if config is None:
        config = RedisTLSConfig()

    result: Dict[str, Any] = {
        "connected": False,
        "protocol": None,
        "cipher": None,
        "peer_cert": None,
        "error": None,
    }

    try:
        context = get_redis_ssl_context(config)

        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                result["connected"] = True
                result["protocol"] = ssock.version()
                result["cipher"] = ssock.cipher()
                result["peer_cert"] = ssock.getpeercert()

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


def verify_redis_tls_connection_sync(
    host: str,
    port: int = 6379,
    config: Optional[RedisTLSConfig] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Synchronous version of verify_redis_tls_connection.

    Args:
        host: Redis hostname.
        port: Redis port.
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
        verify_redis_tls_connection(host, port, config, timeout)
    )


# ---------------------------------------------------------------------------
# ElastiCache-specific Helpers
# ---------------------------------------------------------------------------


def get_elasticache_config(
    cluster_mode: bool = False,
    transit_encryption: bool = True,
) -> RedisTLSConfig:
    """
    Get recommended TLS configuration for AWS ElastiCache.

    Args:
        cluster_mode: Whether cluster mode is enabled.
        transit_encryption: Whether transit encryption is enabled.

    Returns:
        RedisTLSConfig for ElastiCache.
    """
    return RedisTLSConfig(
        ssl=transit_encryption,
        ssl_cert_reqs="required" if transit_encryption else "none",
        ssl_check_hostname=True,
        min_tls_version="TLSv1.2",
    )


__all__ = [
    # Types
    "RedisCertReqs",
    # Configuration
    "RedisTLSConfig",
    # SSL Context
    "get_redis_ssl_context",
    # Connection Parameters
    "get_redis_connection_kwargs",
    "get_redis_connection_kwargs_with_context",
    # URL
    "create_redis_tls_url",
    "parse_redis_url",
    # Verification
    "verify_redis_tls_connection",
    "verify_redis_tls_connection_sync",
    # ElastiCache
    "get_elasticache_config",
]
