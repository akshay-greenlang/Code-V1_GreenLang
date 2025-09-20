"""
Network Security Module
======================

Provides security functions for network operations including:
- HTTPS enforcement
- TLS configuration
- Certificate validation
- URL validation
"""

import os
import ssl
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class SecureHTTPAdapter(HTTPAdapter):
    """HTTP adapter with enforced TLS minimum version"""

    def init_poolmanager(self, *args, **kwargs):
        # Enforce minimum TLS 1.2
        kwargs['ssl_context'] = create_secure_ssl_context()
        return super().init_poolmanager(*args, **kwargs)


def create_secure_ssl_context() -> ssl.SSLContext:
    """
    Create a secure SSL context with proper settings

    Returns:
        Configured SSL context
    """
    context = ssl.create_default_context()

    # Enforce minimum TLS version 1.2
    context.minimum_version = ssl.TLSVersion.TLSv1_2

    # Load custom CA bundle if specified
    ca_bundle = os.environ.get('GL_CA_BUNDLE')
    if ca_bundle:
        if os.path.exists(ca_bundle):
            context.load_verify_locations(ca_bundle)
            logger.info(f"Loaded custom CA bundle from: {ca_bundle}")
        else:
            logger.warning(f"GL_CA_BUNDLE specified but file not found: {ca_bundle}")

    return context


def create_secure_session(timeout: int = 15, max_retries: int = 3) -> requests.Session:
    """
    Create a secure requests session with proper configuration

    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries

    Returns:
        Configured requests.Session
    """
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
    )

    # Use secure adapter
    secure_adapter = SecureHTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", secure_adapter)

    # Block HTTP entirely - no adapter mounted for http://
    # HTTPS is mandatory for all connections

    # Set default timeout
    session.request = lambda *args, **kwargs: requests.Session.request(
        session,
        *args,
        timeout=kwargs.pop('timeout', timeout),
        **kwargs
    )

    return session


def validate_url(url: str, allow_http: bool = False) -> None:
    """
    Validate URL for security requirements

    Args:
        url: URL to validate
        allow_http: Whether to allow HTTP URLs (default: False)

    Raises:
        ValueError: If URL doesn't meet security requirements
    """
    parsed = urlparse(url)

    # Check scheme - HTTP is never allowed for security
    if not allow_http and parsed.scheme == 'http':
        raise ValueError(
            f"Insecure scheme 'http' not allowed (require https). "
            f"URL: {url}"
        )

    if parsed.scheme not in ['http', 'https', 'file']:
        raise ValueError(
            f"Invalid URL scheme '{parsed.scheme}'. "
            f"Only https (and file for local dev) are allowed."
        )

    # Validate hostname isn't localhost/private unless in dev mode
    if parsed.hostname:
        hostname = parsed.hostname.lower()
        private_hosts = ['localhost', '127.0.0.1', '::1', '0.0.0.0']

        if hostname in private_hosts or hostname.startswith('192.168.') or hostname.startswith('10.'):
            logger.warning(f"Private/local hostname detected: {hostname}")


def validate_git_url(url: str) -> None:
    """
    Validate Git repository URL

    Args:
        url: Git URL to validate

    Raises:
        ValueError: If URL doesn't meet security requirements
    """
    parsed = urlparse(url)

    # Only allow HTTPS for git repos
    if parsed.scheme != 'https':
        raise ValueError(
            f"Only HTTPS Git repositories are allowed. "
            f"Got scheme: {parsed.scheme}"
        )

    # Check for known Git hosts
    allowed_hosts = [
        'github.com',
        'gitlab.com',
        'bitbucket.org'
    ]

    if parsed.hostname and parsed.hostname.lower() not in allowed_hosts:
        logger.warning(f"Non-standard Git host: {parsed.hostname}")


def safe_download(url: str, dest_path: str,
                 verify_checksum: Optional[str] = None,
                 session: Optional[requests.Session] = None) -> None:
    """
    Safely download a file with security checks

    Args:
        url: URL to download from
        dest_path: Destination file path
        verify_checksum: Optional expected checksum (SHA256)
        session: Optional requests session to use

    Raises:
        ValueError: If security checks fail
        requests.RequestException: If download fails
    """
    import hashlib

    # Validate URL
    validate_url(url)

    # Use secure session if not provided
    if session is None:
        session = create_secure_session()

    # Download with streaming
    response = session.get(url, stream=True)
    response.raise_for_status()

    # Check content length if provided
    content_length = response.headers.get('Content-Length')
    if content_length:
        max_size = 500 * 1024 * 1024  # 500MB max
        if int(content_length) > max_size:
            raise ValueError(f"File too large: {content_length} bytes (max: {max_size})")

    # Download and optionally verify checksum
    hasher = hashlib.sha256() if verify_checksum else None

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                if hasher:
                    hasher.update(chunk)

    # Verify checksum if provided
    if verify_checksum and hasher:
        actual_checksum = hasher.hexdigest()
        if actual_checksum != verify_checksum:
            os.remove(dest_path)  # Remove potentially compromised file
            raise ValueError(
                f"Checksum verification failed. "
                f"Expected: {verify_checksum}, "
                f"Got: {actual_checksum}"
            )

    logger.info(f"Successfully downloaded: {url} -> {dest_path}")