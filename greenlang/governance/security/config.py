# -*- coding: utf-8 -*-
"""
GreenLang Security Configuration
=================================

Security configuration management for GreenLang.
Provides centralized security settings for:
- Security headers
- Rate limiting
- CORS configuration
- API key management
- Authentication settings

Author: GreenLang Security Team
Phase: 3 - Security Hardening
"""

from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator


class SecurityLevel(str, Enum):
    """Security levels for different environments."""

    LOW = "low"  # Development/testing
    MEDIUM = "medium"  # Staging
    HIGH = "high"  # Production
    MAXIMUM = "maximum"  # High-security production


class SecurityHeaders(BaseModel):
    """Security headers configuration."""

    # Content Security Policy
    content_security_policy: str = Field(
        default="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        description="Content Security Policy directive",
    )

    # Strict Transport Security
    strict_transport_security: str = Field(
        default="max-age=31536000; includeSubDomains; preload",
        description="HSTS header value",
    )

    # X-Frame-Options
    x_frame_options: str = Field(
        default="DENY",
        description="X-Frame-Options header (DENY, SAMEORIGIN, or ALLOW-FROM)",
    )

    # X-Content-Type-Options
    x_content_type_options: str = Field(
        default="nosniff",
        description="X-Content-Type-Options header",
    )

    # X-XSS-Protection
    x_xss_protection: str = Field(
        default="1; mode=block",
        description="X-XSS-Protection header",
    )

    # Referrer-Policy
    referrer_policy: str = Field(
        default="strict-origin-when-cross-origin",
        description="Referrer-Policy header",
    )

    # Permissions-Policy (formerly Feature-Policy)
    permissions_policy: str = Field(
        default="geolocation=(), microphone=(), camera=()",
        description="Permissions-Policy header",
    )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary of header name: value."""
        return {
            "Content-Security-Policy": self.content_security_policy,
            "Strict-Transport-Security": self.strict_transport_security,
            "X-Frame-Options": self.x_frame_options,
            "X-Content-Type-Options": self.x_content_type_options,
            "X-XSS-Protection": self.x_xss_protection,
            "Referrer-Policy": self.referrer_policy,
            "Permissions-Policy": self.permissions_policy,
        }


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable rate limiting",
    )

    requests_per_minute: int = Field(
        default=60,
        description="Maximum requests per minute per client",
    )

    requests_per_hour: int = Field(
        default=1000,
        description="Maximum requests per hour per client",
    )

    burst_size: int = Field(
        default=10,
        description="Burst size for token bucket algorithm",
    )

    block_duration_seconds: int = Field(
        default=300,
        description="Block duration when limit exceeded (seconds)",
    )

    whitelist_ips: List[str] = Field(
        default_factory=list,
        description="IP addresses exempt from rate limiting",
    )

    @field_validator("requests_per_minute")
    @classmethod
    def validate_rpm(cls, v: int) -> int:
        if v < 1 or v > 10000:
            raise ValueError("requests_per_minute must be between 1 and 10000")
        return v


class CORSConfig(BaseModel):
    """CORS (Cross-Origin Resource Sharing) configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable CORS",
    )

    allowed_origins: List[str] = Field(
        default_factory=lambda: ["https://greenlang.io"],
        description="Allowed origins for CORS",
    )

    allowed_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods",
    )

    allowed_headers: List[str] = Field(
        default_factory=lambda: [
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "X-Request-ID",
        ],
        description="Allowed request headers",
    )

    exposed_headers: List[str] = Field(
        default_factory=lambda: ["X-Request-ID", "X-RateLimit-Remaining"],
        description="Headers exposed to browser",
    )

    allow_credentials: bool = Field(
        default=True,
        description="Allow credentials (cookies, authorization headers)",
    )

    max_age: int = Field(
        default=3600,
        description="Max age for preflight cache (seconds)",
    )

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, v: List[str]) -> List[str]:
        if "*" in v and len(v) > 1:
            raise ValueError("Cannot use wildcard with other origins")
        return v


class APIKeyConfig(BaseModel):
    """API key management configuration."""

    enabled: bool = Field(
        default=True,
        description="Require API keys for authentication",
    )

    header_name: str = Field(
        default="X-API-Key",
        description="HTTP header name for API key",
    )

    query_param_name: Optional[str] = Field(
        default=None,
        description="Query parameter name for API key (use with caution)",
    )

    key_prefix: str = Field(
        default="gl_",
        description="Prefix for generated API keys",
    )

    key_length: int = Field(
        default=32,
        description="Length of generated API keys (characters)",
    )

    rotation_days: int = Field(
        default=90,
        description="Recommended key rotation period (days)",
    )

    max_keys_per_user: int = Field(
        default=5,
        description="Maximum API keys per user",
    )

    @field_validator("key_length")
    @classmethod
    def validate_key_length(cls, v: int) -> int:
        if v < 16 or v > 128:
            raise ValueError("key_length must be between 16 and 128")
        return v


class AuthenticationConfig(BaseModel):
    """Authentication configuration."""

    # Password requirements
    password_min_length: int = Field(
        default=12,
        description="Minimum password length",
    )

    password_require_uppercase: bool = Field(
        default=True,
        description="Require uppercase letters in password",
    )

    password_require_lowercase: bool = Field(
        default=True,
        description="Require lowercase letters in password",
    )

    password_require_numbers: bool = Field(
        default=True,
        description="Require numbers in password",
    )

    password_require_special: bool = Field(
        default=True,
        description="Require special characters in password",
    )

    # Session management
    session_timeout_minutes: int = Field(
        default=30,
        description="Session timeout (minutes)",
    )

    max_login_attempts: int = Field(
        default=5,
        description="Maximum failed login attempts before lockout",
    )

    lockout_duration_minutes: int = Field(
        default=15,
        description="Account lockout duration (minutes)",
    )

    # Multi-factor authentication
    mfa_enabled: bool = Field(
        default=False,
        description="Enable multi-factor authentication",
    )

    mfa_required_for_admin: bool = Field(
        default=True,
        description="Require MFA for admin accounts",
    )


class EncryptionConfig(BaseModel):
    """Encryption configuration."""

    # Data at rest
    encrypt_data_at_rest: bool = Field(
        default=True,
        description="Encrypt sensitive data at rest",
    )

    encryption_algorithm: str = Field(
        default="AES-256-GCM",
        description="Encryption algorithm for data at rest",
    )

    # Data in transit
    tls_version: str = Field(
        default="1.3",
        description="Minimum TLS version",
    )

    tls_ciphers: List[str] = Field(
        default_factory=lambda: [
            "TLS_AES_128_GCM_SHA256",
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
        ],
        description="Allowed TLS cipher suites",
    )

    # Key management
    key_rotation_days: int = Field(
        default=90,
        description="Key rotation period (days)",
    )


class AuditConfig(BaseModel):
    """Audit logging configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable audit logging",
    )

    log_authentication: bool = Field(
        default=True,
        description="Log authentication events",
    )

    log_authorization: bool = Field(
        default=True,
        description="Log authorization decisions",
    )

    log_data_access: bool = Field(
        default=True,
        description="Log data access events",
    )

    log_config_changes: bool = Field(
        default=True,
        description="Log configuration changes",
    )

    log_agent_execution: bool = Field(
        default=True,
        description="Log agent execution",
    )

    retention_days: int = Field(
        default=365,
        description="Audit log retention period (days)",
    )

    siem_integration: bool = Field(
        default=False,
        description="Enable SIEM integration",
    )

    siem_endpoint: Optional[str] = Field(
        default=None,
        description="SIEM endpoint URL",
    )


class SecurityConfig(BaseModel):
    """
    Comprehensive security configuration for GreenLang.

    This configuration integrates with the existing ConfigManager
    and provides centralized security settings.
    """

    # Security level
    level: SecurityLevel = Field(
        default=SecurityLevel.HIGH,
        description="Security level for this environment",
    )

    # Component configurations
    headers: SecurityHeaders = Field(
        default_factory=SecurityHeaders,
        description="Security headers configuration",
    )

    rate_limiting: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Rate limiting configuration",
    )

    cors: CORSConfig = Field(
        default_factory=CORSConfig,
        description="CORS configuration",
    )

    api_keys: APIKeyConfig = Field(
        default_factory=APIKeyConfig,
        description="API key management configuration",
    )

    authentication: AuthenticationConfig = Field(
        default_factory=AuthenticationConfig,
        description="Authentication configuration",
    )

    encryption: EncryptionConfig = Field(
        default_factory=EncryptionConfig,
        description="Encryption configuration",
    )

    audit: AuditConfig = Field(
        default_factory=AuditConfig,
        description="Audit logging configuration",
    )

    # Security features
    enable_security_headers: bool = Field(
        default=True,
        description="Enable security headers in HTTP responses",
    )

    enable_input_validation: bool = Field(
        default=True,
        description="Enable input validation",
    )

    enable_output_encoding: bool = Field(
        default=True,
        description="Enable output encoding",
    )

    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging",
    )

    # Threat protection
    enable_xss_protection: bool = Field(
        default=True,
        description="Enable XSS protection",
    )

    enable_csrf_protection: bool = Field(
        default=True,
        description="Enable CSRF protection",
    )

    enable_sql_injection_protection: bool = Field(
        default=True,
        description="Enable SQL injection protection",
    )

    enable_path_traversal_protection: bool = Field(
        default=True,
        description="Enable path traversal protection",
    )

    # Allowed hosts (for SSRF prevention)
    allowed_external_hosts: List[str] = Field(
        default_factory=lambda: [
            "api.greenlang.io",
            "data.greenlang.io",
        ],
        description="Allowed external hosts for API calls",
    )

    # Blocked patterns
    blocked_file_extensions: List[str] = Field(
        default_factory=lambda: [
            ".exe",
            ".dll",
            ".bat",
            ".sh",
            ".ps1",
            ".cmd",
        ],
        description="Blocked file extensions for uploads",
    )

    @classmethod
    def create_for_environment(cls, environment: str) -> "SecurityConfig":
        """
        Create security configuration for specific environment.

        Args:
            environment: Environment name (development, staging, production)

        Returns:
            SecurityConfig instance
        """
        if environment == "development":
            return cls(
                level=SecurityLevel.LOW,
                rate_limiting=RateLimitConfig(enabled=False),
                audit=AuditConfig(enabled=False),
            )
        elif environment == "staging":
            return cls(level=SecurityLevel.MEDIUM)
        elif environment == "production":
            return cls(level=SecurityLevel.HIGH)
        else:
            # Default to high security
            return cls(level=SecurityLevel.HIGH)

    def is_production_ready(self) -> bool:
        """Check if configuration meets production security requirements."""
        return (
            self.level in {SecurityLevel.HIGH, SecurityLevel.MAXIMUM}
            and self.enable_security_headers
            and self.enable_audit_logging
            and self.rate_limiting.enabled
            and self.authentication.password_min_length >= 12
            and self.encryption.encrypt_data_at_rest
        )


# Global security configuration instance
_global_security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """Get global security configuration instance."""
    global _global_security_config
    if _global_security_config is None:
        _global_security_config = SecurityConfig()
    return _global_security_config


def configure_security(config: SecurityConfig) -> SecurityConfig:
    """Set global security configuration."""
    global _global_security_config
    _global_security_config = config
    return _global_security_config
