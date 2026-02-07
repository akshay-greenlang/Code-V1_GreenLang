# -*- coding: utf-8 -*-
# =============================================================================
# GreenLang TLS Service - HTTP TLS Configuration
# SEC-004: TLS 1.3 Configuration for All Services
# =============================================================================
"""
HTTP client TLS configuration for GreenLang applications.

Provides factory functions for creating HTTP clients (httpx, aiohttp)
with proper TLS configuration.

Follows the GreenLang zero-hallucination principle: all TLS operations are
deterministic using the standard library ``ssl`` module.

Functions:
    - create_httpx_client: Create httpx.AsyncClient with TLS.
    - create_httpx_client_sync: Create httpx.Client with TLS.
    - create_aiohttp_connector: Create aiohttp TCPConnector with TLS.
    - get_http_ssl_context: Get SSL context for HTTP clients.

Example:
    >>> from greenlang.infrastructure.tls_service.http_tls import (
    ...     create_httpx_client,
    ...     create_aiohttp_connector,
    ... )
    >>> async with create_httpx_client() as client:
    ...     response = await client.get("https://api.example.com")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import ssl
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

from greenlang.infrastructure.tls_service.ssl_context import (
    create_ssl_context,
    CIPHER_STRING_MODERN,
)
from greenlang.infrastructure.tls_service.ca_bundle import get_ca_bundle_path

if TYPE_CHECKING:
    import httpx
    import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HTTPTLSConfig:
    """HTTP TLS configuration.

    Attributes:
        verify: Whether to verify server certificates. Default True.
        ca_bundle: Path to CA bundle for verification.
        client_cert: Path to client certificate for mTLS.
        client_key: Path to client key for mTLS.
        client_key_password: Password for encrypted client key.
        min_tls_version: Minimum TLS version ("TLSv1.2" or "TLSv1.3").
        timeout: Default timeout in seconds.
        follow_redirects: Whether to follow HTTP redirects.
        http2: Whether to enable HTTP/2.

    Example:
        >>> config = HTTPTLSConfig(
        ...     verify=True,
        ...     min_tls_version="TLSv1.3",
        ...     http2=True,
        ... )
    """

    verify: bool = True
    ca_bundle: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    client_key_password: Optional[str] = None
    min_tls_version: str = "TLSv1.2"
    timeout: float = 30.0
    follow_redirects: bool = True
    http2: bool = False


# ---------------------------------------------------------------------------
# SSL Context
# ---------------------------------------------------------------------------


def get_http_ssl_context(
    config: Optional[HTTPTLSConfig] = None,
) -> ssl.SSLContext:
    """
    Get SSL context for HTTP clients.

    Args:
        config: HTTP TLS configuration.

    Returns:
        Configured SSL context.

    Example:
        >>> ctx = get_http_ssl_context()
        >>> # Use with requests library
        >>> import requests
        >>> response = requests.get("https://api.example.com", verify=ctx)
    """
    if config is None:
        config = HTTPTLSConfig()

    min_version = ssl.TLSVersion.TLSv1_3 if config.min_tls_version == "TLSv1.3" else ssl.TLSVersion.TLSv1_2

    return create_ssl_context(
        purpose="http",
        min_version=min_version,
        verify=config.verify,
        ca_bundle=config.ca_bundle,
        client_cert=config.client_cert,
        client_key=config.client_key,
        client_key_password=config.client_key_password,
    )


# ---------------------------------------------------------------------------
# httpx Client Factory
# ---------------------------------------------------------------------------


def create_httpx_client(
    config: Optional[HTTPTLSConfig] = None,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> "httpx.AsyncClient":
    """
    Create httpx.AsyncClient with TLS configuration.

    The client should be used as a context manager or closed explicitly.

    Args:
        config: HTTP TLS configuration.
        base_url: Base URL for requests.
        headers: Default headers for requests.
        **kwargs: Additional kwargs passed to httpx.AsyncClient.

    Returns:
        Configured httpx.AsyncClient.

    Example:
        >>> async with create_httpx_client() as client:
        ...     response = await client.get("https://api.example.com")
        ...     data = response.json()

    Note:
        Requires httpx to be installed: pip install httpx
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for HTTP client. Install with: pip install httpx")

    if config is None:
        config = HTTPTLSConfig()

    # Build verify parameter
    if config.verify:
        if config.ca_bundle:
            verify = config.ca_bundle
        else:
            try:
                verify = get_ca_bundle_path()
            except FileNotFoundError:
                verify = True  # Use httpx default
    else:
        verify = False

    # Build cert parameter for mTLS
    cert = None
    if config.client_cert:
        if config.client_key:
            cert = (config.client_cert, config.client_key)
        else:
            cert = config.client_cert

    # Build timeout
    timeout = httpx.Timeout(config.timeout)

    client_kwargs: Dict[str, Any] = {
        "verify": verify,
        "timeout": timeout,
        "follow_redirects": config.follow_redirects,
        "http2": config.http2,
        **kwargs,
    }

    if cert:
        client_kwargs["cert"] = cert

    if base_url:
        client_kwargs["base_url"] = base_url

    if headers:
        client_kwargs["headers"] = headers

    logger.debug(
        "Created httpx AsyncClient",
        extra={
            "verify": bool(verify),
            "http2": config.http2,
            "has_client_cert": cert is not None,
        }
    )

    return httpx.AsyncClient(**client_kwargs)


def create_httpx_client_sync(
    config: Optional[HTTPTLSConfig] = None,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> "httpx.Client":
    """
    Create synchronous httpx.Client with TLS configuration.

    The client should be used as a context manager or closed explicitly.

    Args:
        config: HTTP TLS configuration.
        base_url: Base URL for requests.
        headers: Default headers for requests.
        **kwargs: Additional kwargs passed to httpx.Client.

    Returns:
        Configured httpx.Client.

    Example:
        >>> with create_httpx_client_sync() as client:
        ...     response = client.get("https://api.example.com")
        ...     data = response.json()
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for HTTP client. Install with: pip install httpx")

    if config is None:
        config = HTTPTLSConfig()

    # Build verify parameter
    if config.verify:
        if config.ca_bundle:
            verify = config.ca_bundle
        else:
            try:
                verify = get_ca_bundle_path()
            except FileNotFoundError:
                verify = True
    else:
        verify = False

    # Build cert parameter for mTLS
    cert = None
    if config.client_cert:
        if config.client_key:
            cert = (config.client_cert, config.client_key)
        else:
            cert = config.client_cert

    # Build timeout
    timeout = httpx.Timeout(config.timeout)

    client_kwargs: Dict[str, Any] = {
        "verify": verify,
        "timeout": timeout,
        "follow_redirects": config.follow_redirects,
        "http2": config.http2,
        **kwargs,
    }

    if cert:
        client_kwargs["cert"] = cert

    if base_url:
        client_kwargs["base_url"] = base_url

    if headers:
        client_kwargs["headers"] = headers

    return httpx.Client(**client_kwargs)


# ---------------------------------------------------------------------------
# aiohttp Connector Factory
# ---------------------------------------------------------------------------


def create_aiohttp_connector(
    config: Optional[HTTPTLSConfig] = None,
    limit: int = 100,
    limit_per_host: int = 10,
    enable_cleanup_closed: bool = True,
    **kwargs: Any,
) -> "aiohttp.TCPConnector":
    """
    Create aiohttp TCPConnector with TLS configuration.

    Args:
        config: HTTP TLS configuration.
        limit: Total connection limit.
        limit_per_host: Per-host connection limit.
        enable_cleanup_closed: Enable cleanup of closed connections.
        **kwargs: Additional kwargs passed to TCPConnector.

    Returns:
        Configured aiohttp.TCPConnector.

    Example:
        >>> connector = create_aiohttp_connector()
        >>> async with aiohttp.ClientSession(connector=connector) as session:
        ...     async with session.get("https://api.example.com") as response:
        ...         data = await response.json()

    Note:
        Requires aiohttp to be installed: pip install aiohttp
    """
    try:
        import aiohttp
    except ImportError:
        raise ImportError("aiohttp is required for aiohttp connector. Install with: pip install aiohttp")

    if config is None:
        config = HTTPTLSConfig()

    # Build SSL context
    ssl_context: Any
    if config.verify:
        ssl_context = get_http_ssl_context(config)
    else:
        ssl_context = False

    connector_kwargs: Dict[str, Any] = {
        "ssl": ssl_context,
        "limit": limit,
        "limit_per_host": limit_per_host,
        "enable_cleanup_closed": enable_cleanup_closed,
        **kwargs,
    }

    logger.debug(
        "Created aiohttp TCPConnector",
        extra={
            "verify": config.verify,
            "limit": limit,
            "limit_per_host": limit_per_host,
        }
    )

    return aiohttp.TCPConnector(**connector_kwargs)


def create_aiohttp_session(
    config: Optional[HTTPTLSConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    connector_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> "aiohttp.ClientSession":
    """
    Create aiohttp ClientSession with TLS configuration.

    Args:
        config: HTTP TLS configuration.
        headers: Default headers for requests.
        connector_kwargs: kwargs for TCPConnector.
        **kwargs: Additional kwargs passed to ClientSession.

    Returns:
        Configured aiohttp.ClientSession.

    Example:
        >>> async with create_aiohttp_session() as session:
        ...     async with session.get("https://api.example.com") as response:
        ...         data = await response.json()
    """
    try:
        import aiohttp
    except ImportError:
        raise ImportError("aiohttp is required. Install with: pip install aiohttp")

    if config is None:
        config = HTTPTLSConfig()

    connector = create_aiohttp_connector(config, **(connector_kwargs or {}))

    session_kwargs: Dict[str, Any] = {
        "connector": connector,
        **kwargs,
    }

    if headers:
        session_kwargs["headers"] = headers

    timeout = aiohttp.ClientTimeout(total=config.timeout)
    session_kwargs["timeout"] = timeout

    return aiohttp.ClientSession(**session_kwargs)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


async def fetch_json(
    url: str,
    config: Optional[HTTPTLSConfig] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Convenience function to fetch JSON from URL with TLS.

    Args:
        url: URL to fetch.
        config: HTTP TLS configuration.
        headers: Request headers.

    Returns:
        Parsed JSON response.

    Example:
        >>> data = await fetch_json("https://api.example.com/data")
    """
    async with create_httpx_client(config, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def post_json(
    url: str,
    data: Any,
    config: Optional[HTTPTLSConfig] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Convenience function to POST JSON with TLS.

    Args:
        url: URL to post to.
        data: Data to send as JSON.
        config: HTTP TLS configuration.
        headers: Request headers.

    Returns:
        Parsed JSON response.

    Example:
        >>> result = await post_json(
        ...     "https://api.example.com/data",
        ...     {"key": "value"},
        ... )
    """
    async with create_httpx_client(config, headers=headers) as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()


__all__ = [
    # Configuration
    "HTTPTLSConfig",
    # SSL Context
    "get_http_ssl_context",
    # httpx
    "create_httpx_client",
    "create_httpx_client_sync",
    # aiohttp
    "create_aiohttp_connector",
    "create_aiohttp_session",
    # Convenience
    "fetch_json",
    "post_json",
]
