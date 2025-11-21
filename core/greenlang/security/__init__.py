# -*- coding: utf-8 -*-
"""
GreenLang Security Module
=========================

Provides security features for GreenLang including:
- Network security (HTTPS enforcement, TLS configuration)
- Path security (traversal protection, safe extraction)
- Signature verification (pack integrity and authentication)
"""

from .network import (
    create_secure_session,
    validate_url,
    validate_git_url,
    safe_download,
    SecureHTTPAdapter,
    create_secure_ssl_context,
)

from .paths import (
    validate_safe_path,
    safe_extract_tar,
    safe_extract_zip,
    safe_extract_archive,
    validate_pack_structure,
    safe_create_directory,
)

from .signatures import PackVerifier, SignatureVerificationError, verify_pack_integrity

__all__ = [
    # Network
    "create_secure_session",
    "validate_url",
    "validate_git_url",
    "safe_download",
    "SecureHTTPAdapter",
    "create_secure_ssl_context",
    # Paths
    "validate_safe_path",
    "safe_extract_tar",
    "safe_extract_zip",
    "safe_extract_archive",
    "validate_pack_structure",
    "safe_create_directory",
    # Signatures
    "PackVerifier",
    "SignatureVerificationError",
    "verify_pack_integrity",
]
