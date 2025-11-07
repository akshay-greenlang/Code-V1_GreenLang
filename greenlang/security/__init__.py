"""
GreenLang Security Module
=========================

Comprehensive security features for GreenLang including:
- Network security (HTTPS enforcement, TLS configuration)
- Path security (traversal protection, safe extraction)
- Signature verification (pack integrity and authentication)
- Audit logging (security-sensitive operations)
- Input validation (XSS, SQL injection, path traversal, command injection)
- Security configuration (headers, rate limiting, CORS, API keys)

Phase 3 Security Hardening - Production Ready
"""

# Network security
from .network import (
    create_secure_session,
    validate_url,
    validate_git_url,
    safe_download,
    SecureHTTPAdapter,
    create_secure_ssl_context,
)

# Path security
from .paths import (
    validate_safe_path,
    safe_extract_tar,
    safe_extract_zip,
    safe_extract_archive,
    validate_pack_structure,
    safe_create_directory,
)

# Signature verification
from .signatures import PackVerifier, SignatureVerificationError, verify_pack_integrity

# Audit logging
from .audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    get_audit_logger,
    configure_audit_logger,
)

# Input validation
from .validators import (
    ValidationError,
    SQLInjectionValidator,
    XSSValidator,
    PathTraversalValidator,
    CommandInjectionValidator,
    URLValidator,
    validate_api_key,
    validate_email,
    validate_username,
    validate_json_data,
)

# Security configuration
from .config import (
    SecurityConfig,
    SecurityLevel,
    SecurityHeaders,
    RateLimitConfig,
    CORSConfig,
    APIKeyConfig,
    AuthenticationConfig,
    EncryptionConfig,
    AuditConfig,
    get_security_config,
    configure_security,
)

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
    # Audit logging
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "get_audit_logger",
    "configure_audit_logger",
    # Validation
    "ValidationError",
    "SQLInjectionValidator",
    "XSSValidator",
    "PathTraversalValidator",
    "CommandInjectionValidator",
    "URLValidator",
    "validate_api_key",
    "validate_email",
    "validate_username",
    "validate_json_data",
    # Configuration
    "SecurityConfig",
    "SecurityLevel",
    "SecurityHeaders",
    "RateLimitConfig",
    "CORSConfig",
    "APIKeyConfig",
    "AuthenticationConfig",
    "EncryptionConfig",
    "AuditConfig",
    "get_security_config",
    "configure_security",
]
