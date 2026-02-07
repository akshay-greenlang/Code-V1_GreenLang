# -*- coding: utf-8 -*-
# =============================================================================
# GreenLang TLS Service - CA Bundle Management
# SEC-004: TLS 1.3 Configuration for All Services
# =============================================================================
"""
CA bundle management for GreenLang TLS connections.

Provides functions to locate and manage Certificate Authority bundles for
verifying TLS connections to PostgreSQL (RDS), Redis (ElastiCache), and
other services.

Follows the GreenLang zero-hallucination principle: all path resolution
is deterministic using known CA bundle locations and fallback to certifi.

Functions:
    - get_ca_bundle_path: Get the best available CA bundle path.
    - get_aws_rds_ca_bundle: Get AWS RDS-specific CA bundle.
    - get_aws_elasticache_ca_bundle: Get AWS ElastiCache CA bundle.
    - get_system_ca_bundle: Get system CA bundle path.
    - refresh_ca_bundle: Download fresh CA bundles from AWS.
    - validate_ca_bundle: Verify CA bundle is valid and not expired.

Example:
    >>> from greenlang.infrastructure.tls_service.ca_bundle import (
    ...     get_ca_bundle_path,
    ...     get_aws_rds_ca_bundle,
    ... )
    >>> ca_path = get_ca_bundle_path()
    >>> rds_ca = get_aws_rds_ca_bundle()

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import os
import ssl
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants: Known CA Bundle Locations
# ---------------------------------------------------------------------------

# AWS RDS CA bundle locations (ordered by preference)
AWS_RDS_CA_BUNDLE_PATHS: List[str] = [
    "/etc/ssl/certs/rds-combined-ca-bundle.pem",
    "/usr/local/share/ca-certificates/rds-combined-ca-bundle.pem",
    "/opt/greenlang/certs/rds-combined-ca-bundle.pem",
    "certs/rds-combined-ca-bundle.pem",
    "./certs/rds-combined-ca-bundle.pem",
]

# AWS ElastiCache CA bundle locations
AWS_ELASTICACHE_CA_PATHS: List[str] = [
    "/etc/ssl/certs/AmazonRootCA1.pem",
    "/usr/local/share/ca-certificates/AmazonRootCA1.pem",
    "/opt/greenlang/certs/AmazonRootCA1.pem",
    "certs/AmazonRootCA1.pem",
    "./certs/AmazonRootCA1.pem",
]

# System CA bundle locations (platform-specific)
SYSTEM_CA_BUNDLE_PATHS: List[str] = [
    # Linux/Ubuntu
    "/etc/ssl/certs/ca-certificates.crt",
    # CentOS/RHEL/Fedora
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    # Alpine
    "/etc/ssl/certs/ca-certificates.crt",
    # macOS
    "/etc/ssl/cert.pem",
    # Windows (common locations)
    "C:\\Windows\\System32\\curl-ca-bundle.crt",
]

# AWS CA bundle download URLs
AWS_RDS_CA_BUNDLE_URL = (
    "https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem"
)
AWS_ELASTICACHE_CA_URL = (
    "https://www.amazontrust.com/repository/AmazonRootCA1.pem"
)


# ---------------------------------------------------------------------------
# CA Bundle Path Resolution
# ---------------------------------------------------------------------------


def _find_first_existing(paths: List[str]) -> Optional[str]:
    """Find the first existing file from a list of paths.

    Args:
        paths: List of file paths to check.

    Returns:
        First existing path or None.
    """
    for path in paths:
        expanded = os.path.expanduser(os.path.expandvars(path))
        if Path(expanded).exists():
            return expanded
    return None


def get_ca_bundle_path(
    prefer_aws: bool = True,
    service_type: Optional[str] = None,
) -> str:
    """
    Get the best available CA bundle path.

    Resolution order:
    1. Environment variable (SSL_CERT_FILE or REQUESTS_CA_BUNDLE)
    2. AWS-specific bundle if prefer_aws=True and service_type specified
    3. System CA bundle
    4. Certifi bundle (fallback)

    Args:
        prefer_aws: Prefer AWS-specific CA bundles for AWS services.
        service_type: Service type hint ("rds", "elasticache", "s3").

    Returns:
        Path to CA bundle file.

    Raises:
        FileNotFoundError: If no CA bundle can be found.
    """
    # Check environment variables first
    env_ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if env_ca and Path(env_ca).exists():
        logger.debug(f"Using CA bundle from environment: {env_ca}")
        return env_ca

    # AWS-specific bundles
    if prefer_aws and service_type:
        if service_type.lower() in ("rds", "postgresql", "postgres", "aurora"):
            aws_path = get_aws_rds_ca_bundle()
            if aws_path:
                return aws_path
        elif service_type.lower() in ("elasticache", "redis"):
            aws_path = get_aws_elasticache_ca_bundle()
            if aws_path:
                return aws_path

    # System CA bundle
    system_path = get_system_ca_bundle()
    if system_path:
        return system_path

    # Fallback to certifi
    try:
        import certifi
        certifi_path = certifi.where()
        logger.debug(f"Using certifi CA bundle: {certifi_path}")
        return certifi_path
    except ImportError:
        pass

    # Last resort: create default context and extract
    try:
        ctx = ssl.create_default_context()
        # On Windows, this may use the system store directly
        default_verify = ssl.get_default_verify_paths()
        if default_verify.cafile:
            return default_verify.cafile
    except Exception as e:
        logger.warning(f"Could not determine default CA path: {e}")

    raise FileNotFoundError(
        "No CA bundle found. Install certifi or set SSL_CERT_FILE environment variable."
    )


def get_aws_rds_ca_bundle() -> Optional[str]:
    """
    Get AWS RDS CA bundle path.

    Returns:
        Path to RDS CA bundle or None if not found.
    """
    path = _find_first_existing(AWS_RDS_CA_BUNDLE_PATHS)
    if path:
        logger.debug(f"Using AWS RDS CA bundle: {path}")
    return path


def get_aws_elasticache_ca_bundle() -> Optional[str]:
    """
    Get AWS ElastiCache CA bundle path.

    Returns:
        Path to ElastiCache CA bundle or None if not found.
    """
    path = _find_first_existing(AWS_ELASTICACHE_CA_PATHS)
    if path:
        logger.debug(f"Using AWS ElastiCache CA bundle: {path}")
    return path


def get_system_ca_bundle() -> Optional[str]:
    """
    Get system CA bundle path.

    Returns:
        Path to system CA bundle or None if not found.
    """
    path = _find_first_existing(SYSTEM_CA_BUNDLE_PATHS)
    if path:
        logger.debug(f"Using system CA bundle: {path}")
    return path


# ---------------------------------------------------------------------------
# CA Bundle Refresh
# ---------------------------------------------------------------------------


async def refresh_ca_bundle(
    bundle_type: str = "rds",
    target_dir: Optional[str] = None,
) -> str:
    """
    Download fresh CA bundle from AWS.

    Args:
        bundle_type: Type of bundle ("rds" or "elasticache").
        target_dir: Target directory for bundle. Uses temp if None.

    Returns:
        Path to downloaded CA bundle.

    Raises:
        ValueError: If bundle_type is not recognized.
        RuntimeError: If download fails.
    """
    import httpx

    # Select URL based on bundle type
    if bundle_type.lower() in ("rds", "postgresql", "postgres"):
        url = AWS_RDS_CA_BUNDLE_URL
        filename = "rds-combined-ca-bundle.pem"
    elif bundle_type.lower() in ("elasticache", "redis"):
        url = AWS_ELASTICACHE_CA_URL
        filename = "AmazonRootCA1.pem"
    else:
        raise ValueError(f"Unknown bundle type: {bundle_type}")

    # Determine target path
    if target_dir:
        target_path = Path(target_dir) / filename
    else:
        target_path = Path(tempfile.gettempdir()) / filename

    logger.info(f"Downloading CA bundle from {url}")

    try:
        async with httpx.AsyncClient(verify=True) as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()

            # Write to file
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(response.content)

            # Validate the bundle
            is_valid, _ = validate_ca_bundle(str(target_path))
            if not is_valid:
                raise RuntimeError(f"Downloaded bundle failed validation: {target_path}")

            logger.info(f"CA bundle saved to {target_path}")
            return str(target_path)

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to download CA bundle: {e}") from e


def refresh_ca_bundle_sync(
    bundle_type: str = "rds",
    target_dir: Optional[str] = None,
) -> str:
    """
    Synchronous version of refresh_ca_bundle.

    Args:
        bundle_type: Type of bundle ("rds" or "elasticache").
        target_dir: Target directory for bundle.

    Returns:
        Path to downloaded CA bundle.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(refresh_ca_bundle(bundle_type, target_dir))


# ---------------------------------------------------------------------------
# CA Bundle Validation
# ---------------------------------------------------------------------------


def validate_ca_bundle(ca_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a CA bundle file.

    Checks:
    - File exists and is readable
    - Contains valid PEM-encoded certificates
    - Certificates are not expired
    - Can be loaded into an SSL context

    Args:
        ca_path: Path to CA bundle file.

    Returns:
        Tuple of (is_valid, details_dict).
    """
    details: Dict[str, Any] = {
        "path": ca_path,
        "exists": False,
        "readable": False,
        "valid_pem": False,
        "cert_count": 0,
        "expired_count": 0,
        "sha256": None,
        "errors": [],
    }

    path = Path(ca_path)

    # Check existence
    if not path.exists():
        details["errors"].append("File does not exist")
        return False, details

    details["exists"] = True

    # Check readability
    try:
        content = path.read_bytes()
        details["readable"] = True
        details["size_bytes"] = len(content)
        details["sha256"] = hashlib.sha256(content).hexdigest()
    except PermissionError:
        details["errors"].append("Permission denied")
        return False, details
    except Exception as e:
        details["errors"].append(f"Read error: {e}")
        return False, details

    # Check PEM format
    content_str = content.decode("utf-8", errors="ignore")
    if "-----BEGIN CERTIFICATE-----" not in content_str:
        details["errors"].append("No PEM certificates found")
        return False, details

    # Count certificates
    cert_count = content_str.count("-----BEGIN CERTIFICATE-----")
    details["cert_count"] = cert_count
    details["valid_pem"] = True

    # Try loading into SSL context
    try:
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=ca_path)
    except ssl.SSLError as e:
        details["errors"].append(f"SSL load error: {e}")
        return False, details
    except Exception as e:
        details["errors"].append(f"Context error: {e}")
        return False, details

    # Check certificate expiration (requires cryptography library)
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend

        # Split PEM content and check each certificate
        pem_certs = content_str.split("-----END CERTIFICATE-----")
        now = datetime.now(timezone.utc)

        for pem in pem_certs:
            if "-----BEGIN CERTIFICATE-----" in pem:
                pem_data = pem + "-----END CERTIFICATE-----"
                try:
                    cert = x509.load_pem_x509_certificate(
                        pem_data.encode(), default_backend()
                    )
                    if cert.not_valid_after_utc < now:
                        details["expired_count"] += 1
                except Exception:
                    pass  # Skip individual cert errors

    except ImportError:
        logger.debug("cryptography not installed; skipping expiration check")

    if details["expired_count"] > 0:
        details["errors"].append(
            f"{details['expired_count']} certificate(s) expired"
        )
        # Warning but not failure - bundle may still work

    return True, details


def get_ca_bundle_info(ca_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a CA bundle.

    Args:
        ca_path: Path to CA bundle file.

    Returns:
        Dictionary with bundle information.
    """
    is_valid, details = validate_ca_bundle(ca_path)
    details["is_valid"] = is_valid
    return details


__all__ = [
    # Constants
    "AWS_RDS_CA_BUNDLE_PATHS",
    "AWS_ELASTICACHE_CA_PATHS",
    "SYSTEM_CA_BUNDLE_PATHS",
    # Functions
    "get_ca_bundle_path",
    "get_aws_rds_ca_bundle",
    "get_aws_elasticache_ca_bundle",
    "get_system_ca_bundle",
    "refresh_ca_bundle",
    "refresh_ca_bundle_sync",
    "validate_ca_bundle",
    "get_ca_bundle_info",
]
